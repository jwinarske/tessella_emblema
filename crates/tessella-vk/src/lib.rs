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
