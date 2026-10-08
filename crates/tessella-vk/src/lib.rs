// SPDX-License-Identifier: BSD-2-Clause
//! The thin safe layer over the Vulkan calls `tessella_emblema`'s map pass makes.
//!
//! `ash` is an unsafe FFI binding: `vkCreateBuffer`, `vkBindBufferMemory` and `vkMapMemory` are all
//! `unsafe fn`. A crate that forbids `unsafe` cannot call Vulkan at all, so the map pass and
//! `tessella_emblema`'s `#![forbid(unsafe_code)]` could not both stand. This is where that `unsafe`
//! went, so the forbid could stay where the map pass is written.
//!
//! # What makes it safe rather than merely wrapped
//!
//! Two things, and the first is the one that matters.
//!
//! **Every object borrows the device it was made from.** A [`Buffer`] holds `&'d ash::Device`, so it
//! cannot outlive the device, and `Drop` cannot call into a destroyed one. That is checked by the
//! compiler rather than promised in a doc comment, which is the usual arrangement in `ash` code: the
//! common pattern is to clone `ash::Device` into each object, and a clone does not own the device, so
//! destroying the device first leaves every `Drop` calling through dangling function pointers.
//!
//! The cost is that the lifetime propagates — a store holding these is `Store<'d>` — and that is the
//! honest price of the guarantee.
//!
//! **Writes are bounds-checked.** [`Mapping::write`] takes an offset and a slice and refuses what
//! would run past the allocation, which is the one place the map pass would otherwise reach for
//! `copy_nonoverlapping`.
//!
//! # What "thin" is meant to exclude
//!
//! It is not a renderer and not a HAL. It owns no device, no queue, no command pool, and chooses
//! nothing: no suballocator, no staging policy, no frame graph, no pipeline cache. Those are
//! decisions with measurements behind them and they belong to the pass, which is the thing that can
//! measure them.
//!
//! It also deliberately does not grow to cover Vulkan. It covers what the map pass needs, and gains a
//! type when a slice of that pass needs one — so what is here is always something with a caller.

use ash::vk;

/// Why a Vulkan call did not do what was asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A call failed, with the call named so a message says which.
    Call {
        /// Which entry point.
        call: &'static str,
        /// What it answered.
        result: vk::Result,
    },
    /// No memory type satisfies both the buffers and the properties asked for.
    NoMemoryType {
        /// The property flags that were required.
        wanted: vk::MemoryPropertyFlags,
    },
    /// A write would run past the end of the allocation.
    ///
    /// Carried rather than panicking because the offsets come from a layout computed elsewhere, and a
    /// caller that got one wrong wants to know which write and by how much.
    OutOfBounds {
        /// Where the write would start.
        offset: u64,
        /// How many bytes it would write.
        length: u64,
        /// How large the mapped allocation is.
        size: u64,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Call { call, result } => write!(f, "{call}: {result:?}"),
            Self::NoMemoryType { wanted } => {
                write!(f, "no memory type with {wanted:?}")
            }
            Self::OutOfBounds {
                offset,
                length,
                size,
            } => write!(
                f,
                "a write of {length} bytes at {offset} runs past an allocation of {size}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// What binding a buffer requires of the memory behind it.
///
/// `ash`'s own `VkMemoryRequirements` repeated rather than re-exported, so a caller computing a layout
/// does not need `ash` in its own dependency list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Requirements {
    /// Bytes the driver wants for this buffer, which may exceed what was asked for.
    pub size: u64,
    /// Alignment the offset must satisfy.
    pub alignment: u64,
    /// Bit per memory type index that may back it.
    pub types: u32,
}

/// A device somebody else owns, and what its memory looks like.
///
/// Borrowed, never owned: in the arrangement this exists for, emblema owns the `VkDevice` and the map
/// pass is a guest on it. The borrow is what makes everything made from it safe.
#[derive(Clone, Copy)]
pub struct Gpu<'d> {
    device: &'d ash::Device,
    memory: &'d vk::PhysicalDeviceMemoryProperties,
}

impl core::fmt::Debug for Gpu<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gpu")
            .field("memory_types", &self.memory.memory_type_count)
            .finish_non_exhaustive()
    }
}

impl<'d> Gpu<'d> {
    /// Borrows a device and the memory properties of the physical device behind it.
    ///
    /// The properties are borrowed too rather than copied, because `VkPhysicalDeviceMemoryProperties`
    /// is a little over four kilobytes and a `Gpu` is copied freely.
    #[must_use]
    pub fn new(device: &'d ash::Device, memory: &'d vk::PhysicalDeviceMemoryProperties) -> Self {
        Self { device, memory }
    }

    /// The device, for the parts of the pass this crate does not cover yet.
    ///
    /// An escape hatch, and the thing to notice growing: a call site that reaches through this is one
    /// this crate should have a method for. It is `unsafe` to *use*, not to obtain, so it is a plain
    /// accessor.
    #[must_use]
    pub fn raw(self) -> &'d ash::Device {
        self.device
    }

    /// Creates a buffer of `length` bytes, or at least one.
    ///
    /// A zero length becomes one byte, because `vkCreateBuffer` refuses zero and a caller whose
    /// layout has an empty slot would otherwise have to special-case it — which is where the slot
    /// gets dropped and every later index shifts onto the wrong bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateBuffer` fails.
    pub fn buffer(self, length: u64, usage: vk::BufferUsageFlags) -> Result<Buffer<'d>, Error> {
        // SAFETY: the create info is fully initialized, and the device outlives the returned buffer
        // by the lifetime on `Gpu`.
        let raw = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(length.max(1))
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreateBuffer",
            result,
        })?;
        Ok(Buffer {
            device: self.device,
            raw,
        })
    }

    /// Creates a two-dimensional image, destroyed when the returned value is dropped.
    ///
    /// Optimally tiled and `EXCLUSIVE`, one mip level and one array layer -- which is every texture
    /// the producer sends. Optimal rather than linear because the image is sampled and a tiler's
    /// whole advantage is in the swizzle; a linear sampled image is legal and slow.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateImage` fails, which is what an unsupported format or an extent
    /// past `maxImageDimension2D` produces.
    pub fn image(
        self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Image<'d>, Error> {
        // SAFETY: the create info is fully initialized, and the device outlives the returned image by
        // the lifetime on `Gpu`.
        let raw = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        // A zero extent is rejected by `vkCreateImage`, and a texture of no pixels is
                        // not something to allocate for -- but the caller should not have to know
                        // which call refuses it, so this is the one place it is raised.
                        width: width.max(1),
                        height: height.max(1),
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreateImage",
            result,
        })?;
        Ok(Image {
            device: self.device,
            raw,
        })
    }

    /// Creates a color-aspect view of a whole image, destroyed when dropped.
    ///
    /// The image must already be bound to memory: `vkCreateImageView` of an unbound image is
    /// undefined, and this cannot check it -- a `Memory::bind_image` has no handle to leave behind.
    /// Binding before viewing is the order [`crate::Memory::bind_image`] documents.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateImageView` fails.
    pub fn view(self, image: &Image<'_>, format: vk::Format) -> Result<ImageView<'d>, Error> {
        // SAFETY: the create info is fully initialized and names an image made by this device; the
        // device outlives the returned view by the lifetime on `Gpu`.
        let raw = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.raw)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(WHOLE_COLOR),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreateImageView",
            result,
        })?;
        Ok(ImageView {
            device: self.device,
            raw,
        })
    }

    /// Creates a shader module from SPIR-V words, destroyed when dropped.
    ///
    /// Takes `u32` words rather than bytes, which is what `vkCreateShaderModule` wants and what
    /// naga produces -- a byte slice would need an alignment cast, and that is the one place a
    /// wrapper like this would have to reach for `unsafe` on the caller's behalf.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateShaderModule` fails. Note what it does *not* catch: on the
    /// `VeriSilicon` `GC7000UL` a module whose SPIR-V the compiler cannot digest is accepted here
    /// and segfaults later, in `vkCreateGraphicsPipelines`.
    pub fn shader(self, words: &[u32]) -> Result<ShaderModule<'d>, Error> {
        // SAFETY: the create info is fully initialized and borrows the words only for the call; the
        // device outlives the returned module by the lifetime on `Gpu`.
        let raw = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        }
        .map_err(|result| Error::Call {
            call: "vkCreateShaderModule",
            result,
        })?;
        Ok(ShaderModule {
            device: self.device,
            raw,
        })
    }

    /// Creates a descriptor set layout from bindings the caller describes.
    ///
    /// The bindings are the caller's, because which descriptors a family declares is
    /// `tessella_emblema::pipelines::bindings` and this crate chooses nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateDescriptorSetLayout` fails, which is what asking for more
    /// descriptors of a kind than the device's per-stage limit allows produces.
    pub fn set_layout(
        self,
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ) -> Result<DescriptorSetLayout<'d>, Error> {
        // SAFETY: the create info is fully initialized and borrows the bindings only for the call.
        let raw = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreateDescriptorSetLayout",
            result,
        })?;
        Ok(DescriptorSetLayout {
            device: self.device,
            raw,
        })
    }

    /// Creates a pipeline layout over one descriptor set and no push constants.
    ///
    /// One set because that is what the modules declare -- group zero and nothing else -- and no
    /// push constants because nothing in the map pass uses them: a drawable's state is a block in
    /// its layer's consolidated buffer, indexed by the order entry.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreatePipelineLayout` fails.
    pub fn pipeline_layout(
        self,
        set: &DescriptorSetLayout<'_>,
    ) -> Result<PipelineLayout<'d>, Error> {
        let sets = [set.raw];
        // SAFETY: the create info is fully initialized and names a layout made by this device.
        let raw = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreatePipelineLayout",
            result,
        })?;
        Ok(PipelineLayout {
            device: self.device,
            raw,
        })
    }

    /// Creates one graphics pipeline, destroyed when the returned value is dropped.
    ///
    /// Takes the whole `VkGraphicsPipelineCreateInfo` rather than assembling it: every field of it
    /// is a decision with a reason, and those belong to the pass.
    /// `tessella_emblema::pipelines::build` is where they are made and written down.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateGraphicsPipelines` fails. Note what that does *not* cover: on the
    /// `VeriSilicon` `GC7000UL` this call segfaults on SPIR-V the driver cannot digest -- see
    /// jwinarske/vivante-spirv-crash -- which no `Result` can report.
    pub fn graphics_pipeline(
        self,
        create: &vk::GraphicsPipelineCreateInfo<'_>,
    ) -> Result<Pipeline<'d>, Error> {
        let infos = [*create];
        // SAFETY: the create info is the caller's, fully initialized, and borrowed only for the
        // call; the device outlives the returned pipeline by the lifetime on `Gpu`.
        let raw = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &infos, None)
        }
        .map_err(|(_, result)| Error::Call {
            call: "vkCreateGraphicsPipelines",
            result,
        })?;
        Ok(Pipeline {
            device: self.device,
            raw: raw[0],
        })
    }

    /// Creates a depth-stencil view of a whole image, destroyed when dropped.
    ///
    /// Both aspects, for the reason [`WHOLE_DEPTH_STENCIL`] gives. As [`Self::view`], the image must
    /// already be bound to memory.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkCreateImageView` fails.
    pub fn depth_view(self, image: &Image<'_>, format: vk::Format) -> Result<ImageView<'d>, Error> {
        // SAFETY: as `view`; the create info is fully initialized and names an image of this device.
        let raw = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.raw)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(WHOLE_DEPTH_STENCIL),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkCreateImageView",
            result,
        })?;
        Ok(ImageView {
            device: self.device,
            raw,
        })
    }

    /// Allocates memory able to back every one of `requirements`, with the properties given.
    ///
    /// The returned allocation is `size` bytes, which the caller chooses: it is laying the buffers out
    /// within it and this does not know where. The alignment to satisfy is the largest of the
    /// requirements', and the type must be allowed by all of them.
    ///
    /// # Errors
    ///
    /// [`Error::NoMemoryType`] when no type is allowed by every requirement and has the properties,
    /// and [`Error::Call`] if `vkAllocateMemory` fails.
    pub fn allocate(
        self,
        size: u64,
        requirements: &[Requirements],
        wanted: vk::MemoryPropertyFlags,
    ) -> Result<Memory<'d>, Error> {
        let allowed = requirements
            .iter()
            .fold(u32::MAX, |all, need| all & need.types);
        let index = (0..self.memory.memory_type_count)
            .find(|index| {
                allowed & (1 << index) != 0
                    && self.memory.memory_types[*index as usize]
                        .property_flags
                        .contains(wanted)
            })
            .ok_or(Error::NoMemoryType { wanted })?;

        // SAFETY: the size is non-zero and the type index came from this device's own properties.
        let raw = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size.max(1))
                    .memory_type_index(index),
                None,
            )
        }
        .map_err(|result| Error::Call {
            call: "vkAllocateMemory",
            result,
        })?;
        Ok(Memory {
            device: self.device,
            raw,
            size: size.max(1),
        })
    }
}

/// Every depth and stencil level of an image.
///
/// Both aspects in one view, which is what a packed depth-stencil format wants: the rendering scope
/// names the same view as its depth attachment and as its stencil attachment, because they are one
/// image. A view of a single aspect could be one or the other and not both.
const WHOLE_DEPTH_STENCIL: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::from_raw(
        vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw(),
    ),
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Every color level and layer of an image, which is all any texture here has.
///
/// One mip level and one array layer, matching what [`Gpu::image`] creates. Named once because a
/// barrier, a view and a copy all have to agree about it, and three spellings of the same range is
/// three chances to disagree.
const WHOLE_COLOR: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// An image, destroyed when dropped.
pub struct Image<'d> {
    device: &'d ash::Device,
    raw: vk::Image,
}

impl core::fmt::Debug for Image<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Image").field(&self.raw).finish()
    }
}

impl Image<'_> {
    /// The handle, for recording a copy or a barrier.
    ///
    /// Copying it out does not extend its life, as for [`Buffer::raw`].
    #[must_use]
    pub fn raw(&self) -> vk::Image {
        self.raw
    }

    /// What binding this image requires.
    ///
    /// An image's requirements are its own, not its extent times its texel size: a driver pads rows
    /// and planes to suit its swizzle, and on V3D the answer is routinely larger than the arithmetic.
    #[must_use]
    pub fn requirements(&self) -> Requirements {
        // SAFETY: the image was made by this device and is alive.
        let need = unsafe { self.device.get_image_memory_requirements(self.raw) };
        Requirements {
            size: need.size,
            alignment: need.alignment,
            types: need.memory_type_bits,
        }
    }
}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle was made by this device, which outlives this by the struct's lifetime,
        // and is destroyed exactly once.
        unsafe { self.device.destroy_image(self.raw, None) };
    }
}

/// A view of an image, destroyed when dropped.
pub struct ImageView<'d> {
    device: &'d ash::Device,
    raw: vk::ImageView,
}

impl core::fmt::Debug for ImageView<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ImageView").field(&self.raw).finish()
    }
}

impl ImageView<'_> {
    /// The handle, for a descriptor write.
    #[must_use]
    pub fn raw(&self) -> vk::ImageView {
        self.raw
    }
}

impl Drop for ImageView<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and destroyed exactly once. A descriptor
        // still referring to it is the pass's obligation, as for `Buffer::raw`.
        unsafe { self.device.destroy_image_view(self.raw, None) };
    }
}

/// A buffer, destroyed when dropped.
pub struct Buffer<'d> {
    device: &'d ash::Device,
    raw: vk::Buffer,
}

impl core::fmt::Debug for Buffer<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Buffer").field(&self.raw).finish()
    }
}

impl Buffer<'_> {
    /// The handle, for recording a draw.
    ///
    /// Copying it out does not extend its life: the `Buffer` still owns it and still destroys it, so a
    /// recorded command referring to it must be submitted and completed first. That is the pass's
    /// obligation and no borrow here can express it.
    #[must_use]
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// What binding this buffer requires.
    #[must_use]
    pub fn requirements(&self) -> Requirements {
        // SAFETY: the buffer was made by this device and is alive.
        let need = unsafe { self.device.get_buffer_memory_requirements(self.raw) };
        Requirements {
            size: need.size,
            alignment: need.alignment,
            types: need.memory_type_bits,
        }
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle was made by this device, which outlives this by the struct's lifetime,
        // and is destroyed exactly once.
        unsafe { self.device.destroy_buffer(self.raw, None) };
    }
}

/// An allocation, freed when dropped.
pub struct Memory<'d> {
    device: &'d ash::Device,
    raw: vk::DeviceMemory,
    size: u64,
}

impl core::fmt::Debug for Memory<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Memory")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl Memory<'_> {
    /// How many bytes were allocated.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Binds `image` to this allocation at `offset`.
    ///
    /// Must happen before a view of it is made or a copy into it recorded. Nothing here can check
    /// that order -- an unbound image has the same handle as a bound one -- so it is stated.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkBindImageMemory` fails, which is what an offset not satisfying the
    /// image's alignment produces.
    pub fn bind_image(&self, image: &Image<'_>, offset: u64) -> Result<(), Error> {
        // SAFETY: both handles belong to this device, and an `Image` is bound once because this is
        // the only thing that binds one.
        unsafe { self.device.bind_image_memory(image.raw, self.raw, offset) }.map_err(|result| {
            Error::Call {
                call: "vkBindImageMemory",
                result,
            }
        })
    }

    /// Binds `buffer` to this allocation at `offset`.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkBindBufferMemory` fails, which is what an offset that does not satisfy
    /// the buffer's alignment produces.
    pub fn bind(&self, buffer: &Buffer<'_>, offset: u64) -> Result<(), Error> {
        // SAFETY: both handles belong to this device, and nothing else has been bound to this
        // allocation at this offset -- a second bind of one buffer is what Vulkan forbids, and a
        // `Buffer` is bound once because `Memory::bind` is the only thing that binds it.
        unsafe { self.device.bind_buffer_memory(buffer.raw, self.raw, offset) }.map_err(|result| {
            Error::Call {
                call: "vkBindBufferMemory",
                result,
            }
        })
    }

    /// Maps the whole allocation for writing, unmapped when the returned value is dropped.
    ///
    /// # Errors
    ///
    /// [`Error::Call`] if `vkMapMemory` fails, which includes asking of memory that is not
    /// host-visible.
    pub fn map(&self) -> Result<Mapping<'_>, Error> {
        // SAFETY: the allocation belongs to this device and is not already mapped -- `Mapping` holds
        // a borrow of it for as long as it lives, so a second `map` cannot overlap the first.
        let base = unsafe {
            self.device
                .map_memory(self.raw, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|result| Error::Call {
            call: "vkMapMemory",
            result,
        })?
        .cast::<u8>();
        Ok(Mapping { memory: self, base })
    }
}

impl Drop for Memory<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and freed exactly once. Any `Mapping`
        // borrowed this and has therefore already been dropped, which unmapped it.
        unsafe { self.device.free_memory(self.raw, None) };
    }
}

/// A mapped allocation, unmapped when dropped.
///
/// Borrows its [`Memory`], so the allocation cannot be freed while it is mapped and a second mapping
/// cannot overlap this one.
pub struct Mapping<'m> {
    memory: &'m Memory<'m>,
    base: *mut u8,
}

impl core::fmt::Debug for Mapping<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mapping")
            .field("size", &self.memory.size)
            .finish_non_exhaustive()
    }
}

impl Mapping<'_> {
    /// Writes `bytes` at `offset`.
    ///
    /// The bounds check is the point of this type: it is the one place the map pass would otherwise
    /// reach for `copy_nonoverlapping` with an offset it computed, and an offset computed wrongly
    /// writes over another buffer's bytes or past the allocation entirely.
    ///
    /// Coherent memory needs no flush, and this does not offer one — see the crate note on what
    /// "thin" excludes. Asking for a non-coherent type and expecting this to flush would be a
    /// corruption that appears only on the parts where it matters, so there is nothing here to
    /// mislead a caller into thinking it was handled.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfBounds`] when the write would not fit.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let length = bytes.len() as u64;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.memory.size)
        {
            return Err(Error::OutOfBounds {
                offset,
                length,
                size: self.memory.size,
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: the mapping covers the whole allocation and the bounds check above puts
        // `offset + length` inside it, so the destination range is mapped and writable. The source is
        // a slice and cannot overlap the mapping, which is device memory.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.base.add(offset as usize),
                bytes.len(),
            );
        }
        Ok(())
    }

    /// Reads `into.len()` bytes from `offset`.
    ///
    /// Here because a write that cannot be read back is a write nothing can check: the bench that
    /// proves a geometry reached the device does it by reading the bytes out again. Bounds-checked
    /// for the same reason as [`Self::write`].
    ///
    /// # Errors
    ///
    /// [`Error::OutOfBounds`] when the read would run past the allocation.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), Error> {
        let length = into.len() as u64;
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.memory.size)
        {
            return Err(Error::OutOfBounds {
                offset,
                length,
                size: self.memory.size,
            });
        }
        if into.is_empty() {
            return Ok(());
        }
        // SAFETY: the mapping covers the whole allocation and the check above puts `offset + length`
        // inside it, so the source range is mapped and readable. The destination is a slice and
        // cannot overlap device memory.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.base.add(offset as usize),
                into.as_mut_ptr(),
                into.len(),
            );
        }
        Ok(())
    }
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        // SAFETY: mapped by `Memory::map`, which is the only thing that constructs this, and unmapped
        // exactly once because the borrow kept anything else from mapping it meanwhile.
        unsafe { self.memory.device.unmap_memory(self.memory.raw) };
    }
}

/// Commands recorded into a command buffer somebody else owns.
///
/// Borrowed, never owned — the same arrangement as [`Gpu`], and for the same reason. This crate owns
/// no command pool, so a `Recorder` is a handle the pass was given plus the device to record through.
/// Beginning the buffer, ending it, submitting it and knowing when it completed all stay with
/// whoever owns the pool.
///
/// # What recording does and does not promise
///
/// Each method appends one command. None of them submits, and none of them waits. A recorded command
/// names handles it does not keep alive, so every object it refers to has to outlive the submission —
/// which is the obligation [`Buffer::raw`] states and which no borrow here can express, because the
/// lifetime that matters is the queue's and not the compiler's.
#[derive(Clone, Copy)]
pub struct Recorder<'c> {
    device: &'c ash::Device,
    raw: vk::CommandBuffer,
}

impl core::fmt::Debug for Recorder<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Recorder").field(&self.raw).finish()
    }
}

impl<'c> Recorder<'c> {
    /// Records into a command buffer the caller owns and has already begun.
    #[must_use]
    pub fn new(device: &'c ash::Device, raw: vk::CommandBuffer) -> Self {
        Self { device, raw }
    }

    /// Moves a whole image from one layout to another.
    ///
    /// The access masks and stages are derived from the pair rather than taken as arguments, because
    /// there are only three transitions a texture upload makes and each has one right answer:
    ///
    /// | from | to | why |
    /// | --- | --- | --- |
    /// | `UNDEFINED` | `TRANSFER_DST_OPTIMAL` | a new image, before its first copy |
    /// | `SHADER_READ_ONLY_OPTIMAL` | `TRANSFER_DST_OPTIMAL` | an existing image, before a repaint |
    /// | `TRANSFER_DST_OPTIMAL` | `SHADER_READ_ONLY_OPTIMAL` | after the copy, before it is sampled |
    ///
    /// Taking them as arguments would move the decision to every call site, and a barrier with the
    /// wrong source stage is the defect that does not reproduce: it is a race, so it draws correctly
    /// until the driver schedules the copy and the sample close enough together.
    ///
    /// An unrecognized pair gets `ALL_COMMANDS` both sides with both access masks, which is correct
    /// and slow — the conservative answer rather than a silently narrow one.
    pub fn transition(&self, image: &Image<'_>, from: vk::ImageLayout, to: vk::ImageLayout) {
        use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
        let (src_stage, src_access, dst_stage, dst_access) = match (from, to) {
            (L::UNDEFINED, L::TRANSFER_DST_OPTIMAL) => {
                (S::TOP_OF_PIPE, A::empty(), S::TRANSFER, A::TRANSFER_WRITE)
            }
            (L::SHADER_READ_ONLY_OPTIMAL, L::TRANSFER_DST_OPTIMAL) => (
                S::FRAGMENT_SHADER,
                A::SHADER_READ,
                S::TRANSFER,
                A::TRANSFER_WRITE,
            ),
            (L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL) => (
                S::TRANSFER,
                A::TRANSFER_WRITE,
                S::FRAGMENT_SHADER,
                A::SHADER_READ,
            ),
            _ => (
                S::ALL_COMMANDS,
                A::MEMORY_READ | A::MEMORY_WRITE,
                S::ALL_COMMANDS,
                A::MEMORY_READ | A::MEMORY_WRITE,
            ),
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.raw)
            .subresource_range(WHOLE_COLOR);
        // SAFETY: the command buffer is in the recording state -- `Recorder::new` says the caller has
        // begun it -- and the barrier is fully initialized and names an image of this device.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.raw,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Opens a rendering scope over one color attachment and one depth-stencil attachment.
    ///
    /// The dynamic-rendering replacement for a render pass and a framebuffer: the views are named
    /// here, per frame, so nothing per-image is cached anywhere. Both attachments must already be in
    /// the layouts named below, which the caller transitions them into.
    ///
    /// The color attachment loads rather than clears when `clear` is `None`, which is how a host
    /// image that already holds something is drawn over. The depth-stencil attachment always clears:
    /// it is the pass's own, nothing outside the frame reads it, and a frame that inherited the last
    /// one's depth would hide geometry behind a surface that is no longer there.
    ///
    /// Must be closed with [`Self::end_rendering`] before the command buffer ends.
    pub fn begin_rendering(
        &self,
        color: &ImageView<'_>,
        depth_stencil: &ImageView<'_>,
        width: u32,
        height: u32,
        clear: Option<[f32; 4]>,
    ) {
        let mut color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(color.raw)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .store_op(vk::AttachmentStoreOp::STORE);
        color_attachment = match clear {
            Some(rgba) => color_attachment
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: rgba },
                }),
            None => color_attachment.load_op(vk::AttachmentLoadOp::LOAD),
        };
        let colors = [color_attachment];

        // Depth one and stencil zero, which is what the tile masks and the extrusion prepass both
        // expect to start from.
        let cleared = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(depth_stencil.raw)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            // Nothing outside the frame reads it, so there is no reason to write it back -- which on
            // a tiler is the saving that matters most about saying so.
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(cleared);

        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(1)
            .color_attachments(&colors)
            .depth_attachment(&depth)
            .stencil_attachment(&depth);

        // SAFETY: the command buffer is recording, the views belong to this device, and the info is
        // fully initialized and borrowed only for the call.
        unsafe { self.device.cmd_begin_rendering(self.raw, &info) };
    }

    /// Closes the rendering scope.
    pub fn end_rendering(&self) {
        // SAFETY: the command buffer is recording and inside a scope `begin_rendering` opened.
        unsafe { self.device.cmd_end_rendering(self.raw) };
    }

    /// Sets the viewport and scissor to the whole of a target.
    ///
    /// Both are dynamic state, which is what lets one pipeline serve a ring of any size --
    /// `tessella_emblema::pipelines` says why. The viewport's y runs down, matching the clip space
    /// naga emits.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a u32 to f32 cast is exact below 2^24, and a viewport is bounded by \
                  maxViewportDimensions -- measured 4096 on V3D, 8192 on the GC7000UL and 16384 on \
                  RADV, all three orders of magnitude inside it. A target larger than 16,777,216 \
                  pixels on a side is not a thing this pass can be handed."
    )]
    pub fn viewport(&self, width: u32, height: u32) {
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        }];
        // SAFETY: the command buffer is recording; both slices are read only for the call.
        unsafe {
            self.device.cmd_set_viewport(self.raw, 0, &viewports);
            self.device.cmd_set_scissor(self.raw, 0, &scissors);
        }
    }

    /// Clears a whole image in `TRANSFER_DST_OPTIMAL` to zero.
    ///
    /// For a newly created image, before anything is copied into it. `vkCreateImage` says nothing
    /// about what the memory holds and `vkAllocateMemory` says nothing either -- measured on three
    /// drivers for buffers, where the `VeriSilicon` `GC7000UL` returned a previously exited process's
    /// bytes. An image written only where the producer reported damage leaves the rest of itself at
    /// whatever the allocation arrived holding, and that part is still sampled.
    pub fn clear(&self, image: &Image<'_>) {
        let zero = vk::ClearColorValue { float32: [0.0; 4] };
        // SAFETY: the command buffer is recording, the image belongs to this device and is in the
        // layout named, and the range is one this crate's images all have.
        unsafe {
            self.device.cmd_clear_color_image(
                self.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &zero,
                &[WHOLE_COLOR],
            );
        }
    }

    /// Copies regions of a buffer into an image already in `TRANSFER_DST_OPTIMAL`.
    ///
    /// The regions are the caller's, because deciding them is the layout arithmetic that
    /// `tessella_emblema::textures::staging` owns and this crate chooses nothing. An empty list
    /// records nothing rather than a command with no regions, which some drivers reject.
    pub fn copy_to_image(
        &self,
        buffer: &Buffer<'_>,
        image: &Image<'_>,
        regions: &[vk::BufferImageCopy],
    ) {
        if regions.is_empty() {
            return;
        }
        // SAFETY: the command buffer is recording, both handles belong to this device, and the
        // regions are a slice this call only reads.
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.raw,
                buffer.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );
        }
    }

    /// Copies a whole image in `TRANSFER_SRC_OPTIMAL` back into a buffer.
    ///
    /// For reading a texture back, which is how a bench checks that the pixels arrived — the same job
    /// `Store::read_vertex_bytes` does for geometry, and the only way to look at an image without
    /// drawing it.
    pub fn copy_to_buffer(&self, image: &Image<'_>, buffer: &Buffer<'_>, width: u32, height: u32) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(width)
            .buffer_image_height(height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        // SAFETY: as `copy_to_image`; the image is in the layout named and belongs to this device.
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.raw,
                &[region],
            );
        }
    }
}

/// A shader module, destroyed when dropped.
pub struct ShaderModule<'d> {
    device: &'d ash::Device,
    raw: vk::ShaderModule,
}

impl core::fmt::Debug for ShaderModule<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ShaderModule").field(&self.raw).finish()
    }
}

impl ShaderModule<'_> {
    /// The handle, for a pipeline's stage.
    #[must_use]
    pub fn raw(&self) -> vk::ShaderModule {
        self.raw
    }
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and destroyed exactly once. A module may
        // be destroyed as soon as the pipelines built from it exist, so this needs no further order.
        unsafe { self.device.destroy_shader_module(self.raw, None) };
    }
}

/// A descriptor set layout, destroyed when dropped.
pub struct DescriptorSetLayout<'d> {
    device: &'d ash::Device,
    raw: vk::DescriptorSetLayout,
}

impl core::fmt::Debug for DescriptorSetLayout<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DescriptorSetLayout")
            .field(&self.raw)
            .finish()
    }
}

impl DescriptorSetLayout<'_> {
    /// The handle, for allocating a set or making a pipeline layout.
    #[must_use]
    pub fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }
}

impl Drop for DescriptorSetLayout<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and destroyed exactly once.
        unsafe {
            self.device.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}

/// A pipeline layout, destroyed when dropped.
pub struct PipelineLayout<'d> {
    device: &'d ash::Device,
    raw: vk::PipelineLayout,
}

impl core::fmt::Debug for PipelineLayout<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("PipelineLayout").field(&self.raw).finish()
    }
}

impl PipelineLayout<'_> {
    /// The handle, for creating a pipeline or binding descriptors.
    #[must_use]
    pub fn raw(&self) -> vk::PipelineLayout {
        self.raw
    }
}

impl Drop for PipelineLayout<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and destroyed exactly once. A pipeline
        // built with it may outlive it, which Vulkan allows.
        unsafe { self.device.destroy_pipeline_layout(self.raw, None) };
    }
}

/// A graphics pipeline, destroyed when dropped.
pub struct Pipeline<'d> {
    device: &'d ash::Device,
    raw: vk::Pipeline,
}

impl core::fmt::Debug for Pipeline<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Pipeline").field(&self.raw).finish()
    }
}

impl Pipeline<'_> {
    /// The handle, for binding it in a recorded draw.
    ///
    /// Copying it out does not extend its life, as for [`Buffer::raw`]: a recorded bind must be
    /// submitted and completed before this is dropped.
    #[must_use]
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        // SAFETY: made by this device, which outlives this, and destroyed exactly once.
        unsafe { self.device.destroy_pipeline(self.raw, None) };
    }
}
