// SPDX-License-Identifier: BSD-2-Clause
//! The descriptor set a draw binds, written from the stores.
//!
//! [`crate::pipelines`] decides what a family's set *is* -- the bindings, their types and the two
//! layouts. This fills one in: the storage buffers from [`crate::blocks`], and the sampled images
//! and samplers for the textures a drawable names.
//!
//! # A binding per buffer, not a buffer per set
//!
//! Every family declares at least two blocks and they arrive in *different buffers*: a fill layer's
//! drawables at slot 2, its tile properties at 4, its evaluated properties at 5. So each storage
//! binding is resolved on its own, through the slot [`crate::pipelines::bindings`] carries for it.
//!
//! This bound every storage binding to one buffer once, because [`crate::blocks`] held one per
//! layer. Nothing failed: the set was complete, the pipeline was valid and the shader read a color
//! out of a matrix's bytes.
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
use tessella_capture_abi::envelope::{TextureFilter, TextureRef};
use tessella_vk::{DescriptorPool, Gpu, Sampler};

use crate::blocks::{Blocks, Which};
use crate::images::Images;
use crate::pipelines::{Binding, Kind, Layout};

/// Why a set could not be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A binding named a block buffer the layer does not have at that slot.
    NoBlocks {
        /// Which layer.
        which: Which,
        /// Which of its slots.
        slot: u32,
    },
    /// A storage binding whose block has no slot agreed.
    ///
    /// [`crate::slots`] resolves every block any family or surface here declares, which
    /// `tests/block_slots.rs` checks one by one -- so this is a caller passing bindings it built
    /// itself, and the alternative to reporting it is binding that descriptor to another block's
    /// buffer.
    UnknownSlot {
        /// Which binding of the set.
        binding: u32,
    },
    /// A binding named a texture with no image.
    NoTexture {
        /// Which position of the set's texture bindings.
        slot: usize,
    },
    /// A texture binding carrying no slot.
    ///
    /// Every family's textures come from the generated table and every surface's from
    /// [`crate::surface::Surface::texture_slots`], so a binding built by
    /// [`crate::pipelines::bindings`] always has one. This is a caller passing bindings it built
    /// itself, and the alternative to reporting it is placing an image by its position again.
    NoTextureSlot {
        /// Which binding of the set.
        binding: u32,
    },
    /// Two `TextureRef`s claiming one slot.
    ///
    /// The second would overwrite the first, and which of the two images the shader then reads
    /// would be whichever the producer happened to send last.
    DuplicateTextureSlot {
        /// The slot named twice.
        slot: u32,
    },
    /// A `TextureRef` naming a slot the set has no binding for.
    UndeclaredTextureSlot {
        /// The slot nothing declares.
        slot: u32,
    },
    /// A texture binding no `TextureRef` named.
    ///
    /// A descriptor the shader reads and nothing filled -- the same fault
    /// [`Self::WrongTextureCount`] catches by counting, found instead by the slot that stayed
    /// empty, which says *which* one.
    UnfilledTextureSlot {
        /// The slot nothing claimed.
        slot: u32,
    },
    /// A `TextureRef` whose filter discriminant this build does not know.
    BadFilter {
        /// The slot it named.
        slot: u32,
        /// The discriminant that did not decode.
        raw: u32,
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
            Self::NoBlocks { which, slot } => write!(
                f,
                "view {:?} layer {} has no block buffer at slot {slot}",
                which.view, which.layer
            ),
            Self::UnknownSlot { binding } => {
                write!(f, "binding {binding} is a block with no slot agreed")
            }
            Self::NoTexture { slot } => write!(f, "texture binding {slot} has no image"),
            Self::NoTextureSlot { binding } => {
                write!(f, "binding {binding} is a texture with no slot agreed")
            }
            Self::DuplicateTextureSlot { slot } => {
                write!(f, "two textures claim slot {slot}")
            }
            Self::UndeclaredTextureSlot { slot } => {
                write!(f, "no binding of this set takes a texture at slot {slot}")
            }
            Self::UnfilledTextureSlot { slot } => {
                write!(f, "nothing named the texture at slot {slot}")
            }
            Self::BadFilter { slot, raw } => write!(
                f,
                "slot {slot}'s filter is {raw}, which is not one this build knows"
            ),
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
    /// `textures` is consumed by position, one entry per `SampledImage` binding in binding order,
    /// which is what [`bound_from`] returns. The slot is read there rather than here: this walks
    /// the bindings once and a lookup per binding would ask the same question again.
    ///
    /// # Errors
    ///
    /// [`Error::NoBlocks`] or [`Error::NoTexture`] for a resource the stores do not have,
    /// [`Error::UnknownSlot`] for a storage binding carrying no slot, [`Error::WrongTextureCount`]
    /// when the drawable names a different number of textures than the family declares, and
    /// [`Error::Device`] when the pool refuses.
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
        for (slot, bound) in textures.iter().enumerate() {
            if bound.view == vk::ImageView::null() {
                return Err(Error::NoTexture { slot });
            }
        }

        // Each storage binding's own buffer, resolved before anything is allocated: a set half
        // written is a set that has to be freed, and the pool frees whole.
        //
        // Bound at offset zero with the whole range, which is why no
        // `minStorageBufferOffsetAlignment` appears here or in `blocks`: one buffer per slot means
        // there is no offset to align.
        let mut buffers: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(bindings.len());
        for binding in bindings.iter().filter(|b| b.kind == Kind::StorageBuffer) {
            let slot = binding.slot.ok_or(Error::UnknownSlot {
                binding: binding.binding,
            })?;
            let buffer = blocks
                .buffer(which, slot)
                .ok_or(Error::NoBlocks { which, slot })?;
            buffers.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }

        let set = self.pool.allocate(layout.set())?;

        // The infos outlive the writes that point at them, which is why they are collected first.
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
                    let info = buffers.get(block).ok_or(Error::UnknownSlot {
                        binding: binding.binding,
                    })?;
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

/// The image views a drawable's textures are, placed where the set's bindings take them.
///
/// Split out because a drawable names textures by [`tessella_capture_abi::envelope::TextureId`] and
/// a set wants views, and the lookup can fail -- which is a producer naming a texture it never
/// uploaded, not a device problem.
///
/// # Placed by slot, not by arrival
///
/// The returned list is in the order [`Sets::write`] consumes it: one entry per `SampledImage`
/// binding, in binding order. Which `TextureRef` lands at which is decided by its `slot` against
/// the binding's, so the order the run arrives in does not matter.
///
/// It used to. This built the list in the order the refs arrived, which was right only because
/// `encode_raster` happens to emit them in slot order -- #95 measured it by negating the slots in
/// a fixture, so the two refs claimed each other's, and the frame came out unchanged on both
/// attachment formats. A field on the wire that decides which image a shader samples was being
/// thrown away.
///
/// # Errors
///
/// [`Error::NoTextureSlot`] for a texture binding carrying no slot,
/// [`Error::DuplicateTextureSlot`] for two refs claiming one, [`Error::UndeclaredTextureSlot`] for
/// a ref naming a slot the set has no binding for, [`Error::UnfilledTextureSlot`] for a binding no
/// ref named, [`Error::BadFilter`] for a filter discriminant this build does not know, and
/// [`Error::NoTexture`] for a texture the store does not hold.
pub fn bound_from(
    images: &Images<'_>,
    bindings: &[Binding],
    refs: &[TextureRef],
) -> Result<Vec<Bound>, Error> {
    // The slots this set takes a texture at, in binding order. The sampler bindings carry the same
    // slots and are left out: one `TextureRef` fills both, and `Sets::write` reads this list once
    // per kind.
    let wanted: Vec<(u32, u32)> = bindings
        .iter()
        .filter(|b| b.kind == Kind::SampledImage)
        .map(|b| {
            b.slot
                .map(|slot| (b.binding, slot))
                .ok_or(Error::NoTextureSlot { binding: b.binding })
        })
        .collect::<Result<_, _>>()?;

    let mut found: Vec<Option<Bound>> = vec![None; wanted.len()];
    for bound in refs {
        let filter = bound.filter().ok_or(Error::BadFilter {
            slot: bound.slot,
            raw: bound.filter,
        })?;
        let at = wanted
            .iter()
            .position(|(_, slot)| *slot == bound.slot)
            .ok_or(Error::UndeclaredTextureSlot { slot: bound.slot })?;
        if found[at].is_some() {
            return Err(Error::DuplicateTextureSlot { slot: bound.slot });
        }
        let view = images
            .view(bound.texture)
            .ok_or(Error::NoTexture { slot: at })?;
        found[at] = Some(Bound { view, filter });
    }

    found
        .into_iter()
        .zip(&wanted)
        .map(|(bound, (_, slot))| bound.ok_or(Error::UnfilledTextureSlot { slot: *slot }))
        .collect()
}
