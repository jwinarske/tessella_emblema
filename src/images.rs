// SPDX-License-Identifier: BSD-2-Clause
//! A texture's image on the device, and the staged copy that fills it.
//!
//! The device half of [`crate::textures`], which owns the decisions that need no device: which
//! `VkFormat` a texture is, and where its regions sit in one staging buffer. This is the image those
//! regions are copied into.
//!
//! # Why this one needs a command buffer
//!
//! The first part of this crate that does. A buffer is filled by mapping it and writing, which is
//! what [`crate::store`] and [`crate::blocks`] do. An optimally tiled image cannot be mapped at all
//! -- its layout is the driver's secret -- so pixels reach it only through
//! `vkCmdCopyBufferToImage`, which has to be recorded and submitted.
//!
//! So this module records and never submits. Beginning the command buffer, submitting it and knowing
//! when it completed belong to whoever owns the queue, which in the arrangement this exists for is
//! emblema. `tessella_vk::Recorder` is the handle it hands over.
//!
//! # A new image is cleared
//!
//! `vkCreateImage` promises nothing about what its memory holds, and an update that names rects
//! writes only those rects -- so the rest of a new image is whatever the allocation arrived holding,
//! and it is still sampled. Measured for buffers across three drivers: zeros on RADV and V3D, and on
//! the `VeriSilicon` `GC7000UL` a previously exited process's bytes. `declare` clears, for the same
//! reason `blocks::declare` writes its shadow's zeros.
//!
//! # A different shape is a different image
//!
//! Size or format changed is not damage, it is another image: the old one has to go and a new one be
//! made. [`crate::textures::Needs::Recreate`] is where that is decided, and the old allocation
//! outlives the decision because a frame recorded earlier may still be sampling it.

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::{Extent, Rect16, TextureId};
use tessella_capture_abi::generated::mbgl_enums::{TextureChannelDataType, TexturePixelType};
use tessella_vk::{Buffer, Gpu, Image, ImageView, Memory, Recorder};

use crate::device;
use crate::textures::{self, Placed};

/// Why a texture could not be put on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The pixel and channel type pair has no `VkFormat` here.
    ///
    /// `Depth` and `Luminance`, which mbgl's Vulkan backend gives no format -- see
    /// [`device::texture_format`]. Refused rather than substituted: a guessed format samples, and
    /// samples something nobody chose.
    NoFormat {
        /// What was asked for.
        pixel: TexturePixelType,
        /// And with which channels.
        channel: TextureChannelDataType,
    },
    /// A texture already exists at a different size or format.
    ///
    /// Not an error the producer can cause by writing: it is the caller asking for an image that
    /// disagrees with one already made. [`crate::textures::Textures::updated`] answers `Recreate`
    /// for that case, and the old image should have been forgotten first.
    Reshaped {
        /// The size it already has.
        size: Extent,
        /// And the format.
        format: vk::Format,
    },
    /// A region's pixels were not resolvable, or were short.
    ///
    /// The payload and the rects have to describe the same thing.
    /// `tessella_consume::upload::rows` is the first half of that check; this is a backend finding
    /// the bytes it was promised are not there.
    Short {
        /// Which region, by its position in the list.
        rect: usize,
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
            Self::NoFormat { pixel, channel } => {
                write!(f, "{pixel:?} with {channel:?} channels has no format here")
            }
            Self::Reshaped { size, format } => write!(
                f,
                "this texture is already {}x{} {format:?}, which is a different image",
                size.width, size.height
            ),
            Self::Short { rect } => write!(f, "region {rect} has fewer pixels than it claims"),
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// One texture's image, its view, and the staging buffer that fills it.
struct Held<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    /// Kept so the allocation outlives the image, and read for the footprint.
    memory: Memory<'d>,
    size: Extent,
    format: vk::Format,
    texel: u64,
    /// Kept rather than created per upload, and the reason is the submission.
    ///
    /// A recorded copy names a buffer it does not keep alive, so a staging buffer created inside an
    /// upload would be destroyed when the call returned -- before the queue ever read it. Holding it
    /// with the texture makes its life at least as long as the image's.
    staging: Option<(Buffer<'d>, Memory<'d>, u64)>,
}

/// Every texture the device holds.
///
/// `'d` is the device's lifetime, borrowed through [`Gpu`]: these cannot outlive the device their
/// images are on, which the compiler checks.
#[derive(Default)]
pub struct Images<'d> {
    held: BTreeMap<TextureId, Held<'d>>,
}

impl core::fmt::Debug for Images<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Images")
            .field("textures", &self.held.len())
            .finish_non_exhaustive()
    }
}

impl<'d> Images<'d> {
    /// Nothing held.
    #[must_use]
    pub fn new() -> Self {
        Self {
            held: BTreeMap::new(),
        }
    }

    /// How many textures have an image.
    #[must_use]
    pub fn textures(&self) -> usize {
        self.held.len()
    }

    /// The view a descriptor binds, once the texture has one.
    #[must_use]
    pub fn view(&self, texture: TextureId) -> Option<vk::ImageView> {
        Some(self.held.get(&texture)?.view.raw())
    }

    /// The image itself, for a caller recording its own barrier or copy.
    ///
    /// Lent rather than handed over, and typed rather than raw: the borrow is bounded by `&self`, so
    /// a `Recorder` cannot be given an image that outlives the store holding it. That is the whole
    /// guarantee `tessella_vk` is built on, and returning a bare `vk::Image` here would have thrown
    /// it away at the one place a caller actually records.
    ///
    /// The caller records the transition to `SHADER_READ_ONLY_OPTIMAL` through this, once per
    /// texture per frame rather than once per region.
    #[must_use]
    pub fn image(&self, texture: TextureId) -> Option<&Image<'d>> {
        Some(&self.held.get(&texture)?.image)
    }

    /// The image and the view a pass renders into, for a texture that is a render target.
    ///
    /// Both, because `target::Host` needs both: the image for the layout transitions and the view
    /// for the rendering scope. [`Self::view`] gives the raw handle a descriptor wants, which is a
    /// different question -- a pass needs the wrapper, because that is what carries the aspects its
    /// attachments are named by.
    #[must_use]
    pub fn rendered(&self, texture: TextureId) -> Option<(&Image<'d>, &ImageView<'d>)> {
        let held = self.held.get(&texture)?;
        Some((&held.image, &held.view))
    }

    /// The format a texture was made with.
    #[must_use]
    pub fn format(&self, texture: TextureId) -> Option<vk::Format> {
        Some(self.held.get(&texture)?.format)
    }

    /// Device bytes every image and staging buffer comes to.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.held
            .values()
            .map(|held| {
                held.memory.size()
                    + held
                        .staging
                        .as_ref()
                        .map_or(0, |(_, memory, _)| memory.size())
            })
            .sum()
    }

    /// Makes this texture's image if it has none, and answers whether anything was made.
    ///
    /// Records a transition out of `UNDEFINED` and a clear, so a new image is zeros everywhere the
    /// producer has not written -- see this module's own notes on why that is not assumed. The image
    /// is left in `TRANSFER_DST_OPTIMAL`, ready for [`Self::upload`].
    ///
    /// Idempotent for a texture whose shape has not changed, which is the ordinary case: an atlas
    /// keeps its size for as long as the style does.
    ///
    /// # Errors
    ///
    /// [`Error::NoFormat`] for a pair with no format, [`Error::Reshaped`] when the texture exists at
    /// another shape -- forget it first -- and [`Error::Device`] when the device refuses.
    pub fn declare(
        &mut self,
        gpu: Gpu<'d>,
        record: Recorder<'_>,
        texture: TextureId,
        size: Extent,
        pixel: TexturePixelType,
        channel: TextureChannelDataType,
    ) -> Result<bool, Error> {
        let format =
            device::texture_format(pixel, channel).ok_or(Error::NoFormat { pixel, channel })?;
        if let Some(held) = self.held.get(&texture) {
            if held.size == size && held.format == format {
                return Ok(false);
            }
            return Err(Error::Reshaped {
                size: held.size,
                format: held.format,
            });
        }

        let image = gpu.image(
            size.width,
            size.height,
            format,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )?;
        let requirements = [image.requirements()];
        let memory = gpu.allocate(
            requirements[0].size,
            &requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // Bound before the view is made, which `Memory::bind_image` states and nothing can check.
        memory.bind_image(&image, 0)?;
        let view = gpu.view(&image, format)?;

        record.transition(
            &image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        record.clear(&image);

        self.held.insert(
            texture,
            Held {
                image,
                view,
                memory,
                size,
                format,
                texel: textures::texel(pixel, channel),
                staging: None,
            },
        );
        Ok(true)
    }

    /// Makes the image an offscreen *view* draws into, rather than one the producer uploads.
    ///
    /// DR-25's render target. `ViewTarget` names it in `TextureUpdate`'s id space and says it is
    /// "never the subject of one -- nothing uploads pixels to a render target", so it is held here
    /// beside the uploaded textures for the reason that id space is shared: a drawable in the
    /// parent view binds it by [`crate::descriptors::bound_from`] like any other, and that lookup
    /// is by id.
    ///
    /// # What differs from [`Self::declare`]
    ///
    /// The usage, and nothing else about the image. A target needs `COLOR_ATTACHMENT` because a
    /// pass renders into it, and does not need `TRANSFER_DST` because no copy ever fills it.
    ///
    /// And it records nothing. `declare` transitions to `TRANSFER_DST_OPTIMAL` and clears, because
    /// an uploaded texture is sampled wherever its rects did not reach; a target is written whole
    /// by the pass that owns it, and that pass states the layout it is taking the image *from* --
    /// `target::Host::layout`, where `UNDEFINED` is the legal way to say the contents may go. So
    /// this needs no command buffer, which is why it does not take one.
    ///
    /// # Errors
    ///
    /// [`Error::NoFormat`] for a pair with no format, [`Error::Reshaped`] when the texture exists at
    /// another shape -- forget it first, which a parent resizing requires -- and [`Error::Device`]
    /// when the device refuses.
    pub fn declare_target(
        &mut self,
        gpu: Gpu<'d>,
        texture: TextureId,
        size: Extent,
        pixel: TexturePixelType,
        channel: TextureChannelDataType,
    ) -> Result<bool, Error> {
        let format =
            device::texture_format(pixel, channel).ok_or(Error::NoFormat { pixel, channel })?;
        if let Some(held) = self.held.get(&texture) {
            if held.size == size && held.format == format {
                return Ok(false);
            }
            return Err(Error::Reshaped {
                size: held.size,
                format: held.format,
            });
        }

        let image = gpu.image(
            size.width,
            size.height,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )?;
        let requirements = [image.requirements()];
        let memory = gpu.allocate(
            requirements[0].size,
            &requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        memory.bind_image(&image, 0)?;
        let view = gpu.view(&image, format)?;

        self.held.insert(
            texture,
            Held {
                image,
                view,
                memory,
                size,
                format,
                texel: textures::texel(pixel, channel),
                staging: None,
            },
        );
        Ok(true)
    }

    /// Stages a texture's regions and records the copy into its image.
    ///
    /// `pixels` resolves one region to its bytes, row by row: given the region's index and row, it
    /// answers that row's pixels. That shape rather than one flat slice because the two payload forms
    /// differ exactly in how rows are reached -- `tessella_consume::upload::rows` gives the offset and
    /// stride, and a backend walking it is where those two numbers are used.
    ///
    /// The image must be in `TRANSFER_DST_OPTIMAL`, which [`Self::declare`] leaves it in and which a
    /// caller repainting an existing texture records itself. It is left in that layout: the caller
    /// transitions to `SHADER_READ_ONLY_OPTIMAL` when it has finished copying, because a frame
    /// writing several regions of one atlas should pay for one barrier and not one per region.
    ///
    /// # Correctness, which this cannot check
    ///
    /// The staging buffer is reused across uploads. A second upload overwrites what the first staged,
    /// so the submission carrying the first must have completed -- the same obligation
    /// [`crate::store::Store::free`] carries, and the same reason: only the caller knows when its
    /// queue is done.
    ///
    /// # Errors
    ///
    /// [`Error::Short`] when a row's bytes are missing or the wrong length, and [`Error::Device`]
    /// when the device refuses. A texture with no image answers `Ok(false)`.
    pub fn upload<'bytes>(
        &mut self,
        gpu: Gpu<'d>,
        record: Recorder<'_>,
        texture: TextureId,
        rects: &[Rect16],
        pixels: &dyn Fn(usize, u16) -> Option<&'bytes [u8]>,
    ) -> Result<bool, Error> {
        let Some(held) = self.held.get_mut(&texture) else {
            return Ok(false);
        };
        let plan = textures::staging_for(rects, held.size, held.texel);
        if plan.placed.is_empty() {
            return Ok(true);
        }

        // Grown rather than reallocated per upload, and never shrunk: an atlas's damage varies
        // frame to frame and a buffer that shrank would be reallocated the next time it grew back.
        let wanted = plan.total;
        let fits = held
            .staging
            .as_ref()
            .is_some_and(|(_, memory, _)| memory.size() >= wanted);
        if !fits {
            let buffer = gpu.buffer(wanted, vk::BufferUsageFlags::TRANSFER_SRC)?;
            let requirements = [buffer.requirements()];
            let memory = gpu.allocate(
                wanted.max(requirements[0].size),
                &requirements,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            memory.bind(&buffer, 0)?;
            held.staging = Some((buffer, memory, wanted));
        }
        let Some((buffer, memory, _)) = held.staging.as_ref() else {
            return Ok(false);
        };

        // Written before anything is recorded, so a short region costs no command.
        {
            let mut mapping = memory.map()?;
            for (index, placed) in plan.placed.iter().enumerate() {
                let row_bytes = u64::from(placed.rect.w) * held.texel;
                for row in 0..placed.rect.h {
                    let Some(bytes) = pixels(index, row) else {
                        return Err(Error::Short { rect: index });
                    };
                    if bytes.len() as u64 != row_bytes {
                        return Err(Error::Short { rect: index });
                    }
                    mapping.write(placed.at + u64::from(row) * row_bytes, bytes)?;
                }
            }
        }

        let regions: Vec<vk::BufferImageCopy> = plan.placed.iter().map(copy_of).collect();
        record.copy_to_image(buffer, &held.image, &regions);
        Ok(true)
    }

    /// Forgets a texture's image, freeing it.
    ///
    /// # Correctness, which this cannot check
    ///
    /// Every frame that sampled it must have completed, as for [`crate::store::Store::free`].
    pub fn forget(&mut self, texture: TextureId) {
        self.held.remove(&texture);
    }

    /// Forgets every texture, for a frame going away with its device still alive.
    pub fn clear(&mut self) {
        self.held.clear();
    }
}

/// The copy one placed region becomes.
///
/// `buffer_image_height` is left at zero, which means "the region's own height" -- the staged rows
/// are tight, so there is no gap between them to describe.
fn copy_of(placed: &Placed) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .buffer_offset(placed.at)
        .buffer_row_length(placed.row_length)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D {
            x: i32::from(placed.rect.x),
            y: i32::from(placed.rect.y),
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: u32::from(placed.rect.w),
            height: u32::from(placed.rect.h),
            depth: 1,
        })
}
