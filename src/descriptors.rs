// SPDX-License-Identifier: BSD-2-Clause
//! The descriptor set a draw binds, written from the stores.
//!
//! [`crate::pipelines`] decides what a family's set *is* -- the bindings, their types and the two
//! layouts. This fills one in: the storage buffers from [`crate::blocks`], and the sampled images
//! and samplers for the textures a drawable names.
//!
//! # The two samplers
//!
//! Filter comes off the wire. `TextureRef::filter` is per *binding* rather than per texture, and the
//! ABI says why: the glyph atlas is linear always, and the icon atlas is linear only when the icons
//! are scaled -- so one texture can want both filters in one frame depending on which drawable is
//! sampling it. A sampler per texture could not express that; a sampler per filter can.
//!
//! There are therefore exactly two, made once. Wrap is not a parameter at all: every texture here
//! wants `CLAMP_TO_EDGE`, because an atlas is a shared sheet and a repeating address mode would walk
//! into the neighboring sprite. A pattern's own body wraps its coordinate by hand instead, which is
//! what `shaders::PREAMBLE`'s `wrap` is for.
//!
//! # When a set dies
//!
//! With the frame that bound it, not with the drawable. A set names a buffer and an image view and
//! keeps neither alive, and the stores free on a frame *completing* rather than on a retire --
//! [`crate::residency`] and [`crate::textures`] both. So [`Sets::reset`] is on that same clock, and
//! calling it early is the one way to use a set after free.

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::TextureFilter;
use tessella_vk::{DescriptorPool, Gpu, Sampler};

use crate::blocks::{Blocks, Which};
use crate::images::Images;
use crate::pipelines::{Binding, Kind, Layout};

/// Why a set could not be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A binding named a block buffer the layer does not have.
    NoBlocks {
        /// Which layer.
        which: Which,
    },
    /// A binding named a texture with no image.
    NoTexture {
        /// Which slot of the drawable's texture list.
        slot: usize,
    },
    /// The set's bindings and the textures given do not agree about how many there are.
    ///
    /// A family declares its image count in its own table, so a drawable naming a different number
    /// is a producer and this consumer disagreeing -- and writing what arrived would leave a
    /// binding unwritten, which is a descriptor the shader reads and nothing filled.
    WrongTextureCount {
        /// How many the set has bindings for.
        wanted: usize,
        /// How many arrived.
        got: usize,
    },
    /// The device refused.
    Device(tessella_vk::Error),
}

impl From<tessella_vk::Error> for Error {
    fn from(why: tessella_vk::Error) -> Self {
        Self::Device(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoBlocks { which } => write!(
                f,
                "view {:?} layer {} has no block buffer",
                which.view, which.layer
            ),
            Self::NoTexture { slot } => write!(f, "texture slot {slot} has no image"),
            Self::WrongTextureCount { wanted, got } => {
                write!(f, "the set has {wanted} texture bindings and {got} arrived")
            }
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// A texture a drawable binds, and how it wants it sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    /// The texture's image view.
    pub view: vk::ImageView,
    /// Which of the two samplers.
    pub filter: TextureFilter,
}

/// The two samplers, and the pool the sets come from.
///
/// `'d` is the device's lifetime: these cannot outlive the device, which the compiler checks.
pub struct Sets<'d> {
    pool: DescriptorPool<'d>,
    linear: Sampler<'d>,
    nearest: Sampler<'d>,
    held: BTreeMap<Which, vk::DescriptorSet>,
}

impl core::fmt::Debug for Sets<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sets")
            .field("sets", &self.held.len())
            .finish_non_exhaustive()
    }
}

impl<'d> Sets<'d> {
    /// Makes the samplers and a pool for `sets` sets of `bindings`.
    ///
    /// The pool is sized from the widest set rather than per family, because a pool is reset whole
    /// and sizing it per family would mean a pool per family.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the device refuses a sampler or the pool.
    pub fn new(gpu: Gpu<'d>, sets: u32, bindings: &[Binding]) -> Result<Self, Error> {
        let sizes: Vec<vk::DescriptorPoolSize> = crate::pipelines::pool_sizes(bindings)
            .into_iter()
            .map(|(kind, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(kind.descriptor_type())
                    .descriptor_count(count * sets.max(1))
            })
            .collect();
        Ok(Self {
            pool: gpu.descriptor_pool(sets, &sizes)?,
            linear: gpu.sampler(vk::Filter::LINEAR)?,
            nearest: gpu.sampler(vk::Filter::NEAREST)?,
            held: BTreeMap::new(),
        })
    }

    /// How many sets are allocated.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.held.len()
    }

    /// The set for a layer, once it has one.
    #[must_use]
    pub fn get(&self, which: Which) -> Option<vk::DescriptorSet> {
        self.held.get(&which).copied()
    }

    /// The sampler for a filter.
    #[must_use]
    pub fn sampler(&self, filter: TextureFilter) -> vk::Sampler {
        match filter {
            TextureFilter::Linear => self.linear.raw(),
            TextureFilter::Nearest => self.nearest.raw(),
        }
    }

    /// Allocates and writes the set for one layer.
    ///
    /// Every binding is written, in the order [`crate::pipelines::bindings`] fixes: the block buffer
    /// for each storage binding, then the image and the sampler for each texture. A binding left
    /// unwritten is a descriptor the shader reads and nothing filled, so a missing resource is an
    /// error rather than a skipped write.
    ///
    /// # Errors
    ///
    /// [`Error::NoBlocks`] or [`Error::NoTexture`] for a resource the stores do not have,
    /// [`Error::WrongTextureCount`] when the drawable names a different number of textures than the
    /// family declares, and [`Error::Device`] when the pool refuses.
    pub fn write(
        &mut self,
        layout: &Layout<'_>,
        bindings: &[Binding],
        which: Which,
        blocks: &Blocks<'_>,
        textures: &[Bound],
    ) -> Result<vk::DescriptorSet, Error> {
        let images = bindings
            .iter()
            .filter(|b| b.kind == Kind::SampledImage)
            .count();
        if images != textures.len() {
            return Err(Error::WrongTextureCount {
                wanted: images,
                got: textures.len(),
            });
        }
        let buffer = blocks.buffer(which).ok_or(Error::NoBlocks { which })?;
        for (slot, bound) in textures.iter().enumerate() {
            if bound.view == vk::ImageView::null() {
                return Err(Error::NoTexture { slot });
            }
        }

        let set = self.pool.allocate(layout.set())?;

        // The infos outlive the writes that point at them, which is why they are collected first.
        let buffers: Vec<vk::DescriptorBufferInfo> = bindings
            .iter()
            .filter(|b| b.kind == Kind::StorageBuffer)
            .map(|_| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let pictures: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|bound| {
                vk::DescriptorImageInfo::default()
                    .image_view(bound.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let samplers: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|bound| vk::DescriptorImageInfo::default().sampler(self.sampler(bound.filter)))
            .collect();

        // Indexed through `get` rather than `[]`. The count check above means every one of these
        // is present, so the error arms are unreachable -- and written as conversions that can fail
        // anyway, because the alternative is a `pub fn` that panics on a protocol input and a panic
        // documented is still a panic. Removing the check turns this into an error rather than an
        // index out of bounds.
        let mut writes = Vec::with_capacity(bindings.len());
        let mut block = 0usize;
        let mut image = 0usize;
        let mut sampler = 0usize;
        for binding in bindings {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding.binding)
                .descriptor_type(binding.kind.descriptor_type());
            writes.push(match binding.kind {
                Kind::StorageBuffer => {
                    let info = buffers.get(block).ok_or(Error::NoBlocks { which })?;
                    block += 1;
                    write.buffer_info(core::slice::from_ref(info))
                }
                Kind::SampledImage => {
                    let info = pictures
                        .get(image)
                        .ok_or(Error::NoTexture { slot: image })?;
                    image += 1;
                    write.image_info(core::slice::from_ref(info))
                }
                Kind::Sampler => {
                    let info = samplers
                        .get(sampler)
                        .ok_or(Error::NoTexture { slot: sampler })?;
                    sampler += 1;
                    write.image_info(core::slice::from_ref(info))
                }
            });
        }
        self.pool.write(&writes);
        self.held.insert(which, set);
        Ok(set)
    }

    /// Frees every set at once.
    ///
    /// # Correctness, which this cannot check
    ///
    /// Every frame that bound one must have completed -- see this module's own notes on when a set
    /// dies, and `DescriptorPool::reset`.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] if the pool will not reset.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.pool.reset()?;
        self.held.clear();
        Ok(())
    }
}

/// The image views a drawable's textures are, from the texture store.
///
/// Split out because a drawable names textures by [`tessella_capture_abi::envelope::TextureId`] and
/// a set wants views, and the lookup can fail -- which is a producer naming a texture it never
/// uploaded, not a device problem.
///
/// # Errors
///
/// [`Error::NoTexture`] naming the first slot whose texture the store does not hold.
pub fn bound_from(
    images: &Images<'_>,
    refs: &[(tessella_capture_abi::envelope::TextureId, TextureFilter)],
) -> Result<Vec<Bound>, Error> {
    refs.iter()
        .enumerate()
        .map(|(slot, (texture, filter))| {
            images
                .view(*texture)
                .map(|view| Bound {
                    view,
                    filter: *filter,
                })
                .ok_or(Error::NoTexture { slot })
        })
        .collect()
}
