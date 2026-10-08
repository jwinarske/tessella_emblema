// SPDX-License-Identifier: BSD-2-Clause
//! A layer's consolidated block buffer on the device, and the writes that bring it level.
//!
//! The second part of #60's frame half, after [`crate::store`]. [`crate::uniforms`] is the host side
//! of this: it shadows the buffer, remembers which slots moved and turns them into contiguous ranges.
//! This is the buffer those ranges are written into.
//!
//! # One buffer per view and layer
//!
//! Which is the shape the README argues for and what `UboUpdate` is keyed by -- a view and a layer
//! index. One buffer per layer rather than one per drawable is the whole reason the host side has a
//! shadow: a frame touches a handful of slots in a buffer holding hundreds, and §11.7 asks for the
//! sub-ranges rather than a whole-buffer rewrite, which is bandwidth a tiler does not have spare.
//!
//! A buffer bound at offset zero needs no `minStorageBufferOffsetAlignment`, which is why that limit
//! does not appear here. A later change that packed several layers into one allocation with dynamic
//! offsets would need it, and would be the place to ask for it.
//!
//! # A different shape is a different buffer
//!
//! Slots and block size are fixed when the buffer is made. A layer arriving with either changed is
//! not damage to this buffer, it is another buffer -- the same rule [`crate::textures`] states for an
//! image whose size or format changed. Writing the new shape into the old bytes would put a drawable's
//! block at another's offset, which draws.

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::ViewId;
use tessella_vk::{Buffer, Gpu, Memory};

use crate::uniforms::{Consolidated, Rejected};

/// Which layer of which view a buffer belongs to.
///
/// The pair `UboUpdate` carries. A feature id is unique only within a source layer and a block slot
/// is unique only within one of these, which is why the key is both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Which {
    /// The view whose frame holds it.
    pub view: ViewId,
    /// The layer's index in the style, as the producer numbers it.
    pub layer: i32,
}

/// Why a block buffer could not be made or brought level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The write was not one block, or named a slot the buffer does not have.
    Write(Rejected),
    /// A layer already has a buffer of a different shape.
    ///
    /// Not an error the producer can cause by writing: it is the caller asking for a buffer that
    /// disagrees with one already made, which means the layer's layout changed and the old buffer
    /// should have been dropped first.
    Reshaped {
        /// Slots the existing buffer has.
        slots: usize,
        /// Bytes per block it has.
        block: usize,
    },
    /// The layer has no slots, or its blocks are no bytes.
    ///
    /// Refused rather than made as a one-byte buffer, which is what the wrapper's `max(1)` would
    /// otherwise produce. A layer with no slots has nothing to draw, and a block of zero bytes makes
    /// every slot's offset zero -- so the shadow would take a write for any slot at all and put it
    /// over slot zero, which is a layout disagreement answering `Ok`.
    Degenerate {
        /// Slots asked for.
        slots: usize,
        /// Bytes per block asked for.
        block: usize,
    },
    /// The slots and block size multiply to more than an allocation can be.
    ///
    /// `Consolidated::new` sizes its shadow with an unchecked `slots * block`, so this has to be
    /// refused a step earlier: `usize` is thirty-two bits on some targets this builds for, where a
    /// product that wraps makes a shadow far smaller than the slot count claims and every write past
    /// the wrap lands on an earlier slot.
    Overflows {
        /// Slots asked for.
        slots: usize,
        /// Bytes per block asked for.
        block: usize,
    },
    /// The device refused.
    Device(tessella_vk::Error),
}

impl From<tessella_vk::Error> for Error {
    fn from(why: tessella_vk::Error) -> Self {
        Self::Device(why)
    }
}

impl From<Rejected> for Error {
    fn from(why: Rejected) -> Self {
        Self::Write(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Write(why) => write!(f, "{why:?}"),
            Self::Degenerate { slots, block } => write!(
                f,
                "a layer of {slots} slots of {block} bytes has nothing to draw and no slot arithmetic"
            ),
            Self::Overflows { slots, block } => write!(
                f,
                "{slots} slots of {block} bytes is more than an allocation can be"
            ),
            Self::Reshaped { slots, block } => write!(
                f,
                "this layer already has {slots} slots of {block} bytes, which is a different buffer"
            ),
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// How many bytes a layer of this shape needs, or why it cannot have any.
///
/// The device-free half of [`Blocks::declare`], split out for the reason [`crate::store::layout`] is:
/// the rest of this module can only be exercised in a bench, so a decision left inside it would be a
/// decision nothing in CI checks.
///
/// There is no offset arithmetic here to go with it, and that is the design rather than an omission --
/// each layer's buffer is bound at offset zero, so a flushed range *is* its byte offset. The
/// arithmetic appears the moment several layers share an allocation, and that is when this grows a
/// companion.
///
/// # Errors
///
/// [`Error::Degenerate`] for no slots or zero-byte blocks, and [`Error::Overflows`] when the product
/// will not fit a `usize` on this target.
pub fn sizing(slots: usize, block: usize) -> Result<u64, Error> {
    if slots == 0 || block == 0 {
        return Err(Error::Degenerate { slots, block });
    }
    let bytes = slots
        .checked_mul(block)
        .ok_or(Error::Overflows { slots, block })?;
    u64::try_from(bytes).map_err(|_| Error::Overflows { slots, block })
}

/// One layer's shadow and the buffer it is written into.
struct Held<'d> {
    shadow: Consolidated,
    buffer: Buffer<'d>,
    memory: Memory<'d>,
}

/// The block buffers the drawn layers read, one per view and layer.
///
/// `'d` is the device's lifetime, borrowed through [`Gpu`]: these cannot outlive the device their
/// buffers are on, which the compiler checks rather than a comment asking.
#[derive(Default)]
pub struct Blocks<'d> {
    held: BTreeMap<Which, Held<'d>>,
}

impl core::fmt::Debug for Blocks<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Blocks")
            .field("layers", &self.held.len())
            .finish_non_exhaustive()
    }
}

impl<'d> Blocks<'d> {
    /// No layers yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            held: BTreeMap::new(),
        }
    }

    /// How many layers have a buffer.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.held.len()
    }

    /// The buffer a layer's drawables read, once it has one.
    #[must_use]
    pub fn buffer(&self, which: Which) -> Option<vk::Buffer> {
        Some(self.held.get(&which)?.buffer.raw())
    }

    /// Device bytes every layer's buffer comes to.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.held.values().map(|held| held.memory.size()).sum()
    }

    /// Makes this layer's buffer if it has none, and answers whether anything was made.
    ///
    /// Idempotent for a layer whose shape has not changed, which is the ordinary case: a layer keeps
    /// its slot count for as long as the style does.
    ///
    /// # Errors
    ///
    /// [`Error::Reshaped`] when the layer has a buffer of another shape -- drop it first, because the
    /// old bytes are laid out for the old slot count. [`Error::Degenerate`] for a layer with no slots
    /// or zero-byte blocks. [`Error::Device`] when the device refuses.
    pub fn declare(
        &mut self,
        gpu: Gpu<'d>,
        which: Which,
        slots: usize,
        block: usize,
    ) -> Result<bool, Error> {
        if let Some(held) = self.held.get(&which) {
            let (have_slots, have_block) = (held.shadow.slots(), held.shadow.block());
            if have_slots == slots && have_block == block {
                return Ok(false);
            }
            return Err(Error::Reshaped {
                slots: have_slots,
                block: have_block,
            });
        }

        let bytes = sizing(slots, block)?;
        let shadow = Consolidated::new(slots, block);
        debug_assert_eq!(shadow.bytes().len() as u64, bytes);
        let buffer = gpu.buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let requirements = [buffer.requirements()];
        let memory = gpu.allocate(
            bytes.max(requirements[0].size),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        memory.bind(&buffer, 0)?;
        // The shadow's zeros have to be put on the device, not just assumed there.
        //
        // `vkAllocateMemory` says nothing about what the memory holds, and a slot nothing has written
        // is never in a flushed range -- so without this it reads back as whatever the allocation came
        // with. Measured: on RADV and V3D that is zeros and this write changes nothing, and on the
        // VeriSilicon GC7000UL an unwritten slot came back holding 0x01. Three drivers, two answers,
        // and the spec allows both, so the one that looked right was luck.
        //
        // A drawable whose block has not been written yet is reached by anything that names it before
        // the producer fills it, which is the first frame of every layer. Garbage there is a feature
        // drawn at a garbage width in a garbage color rather than an invisible one.
        memory.map()?.write(0, shadow.bytes())?;
        self.held.insert(
            which,
            Held {
                shadow,
                buffer,
                memory,
            },
        );
        Ok(true)
    }

    /// Writes one slot into a layer's shadow, leaving the device until a flush.
    ///
    /// A layer with no buffer is `Ok(false)`: the producer may send a block for a layer this frame has
    /// not declared, and dropping it is what a consumer that does not draw that layer should do.
    ///
    /// # Errors
    ///
    /// [`Error::Write`] when the data is not one block or the slot is past the end.
    pub fn write(&mut self, which: Which, slot: u32, data: &[u8]) -> Result<bool, Error> {
        let Some(held) = self.held.get_mut(&which) else {
            return Ok(false);
        };
        held.shadow.write(slot, data)?;
        Ok(true)
    }

    /// Whether a layer has writes the device has not seen.
    #[must_use]
    pub fn is_dirty(&self, which: Which) -> bool {
        self.held
            .get(&which)
            .is_some_and(|held| held.shadow.is_dirty())
    }

    /// Brings one layer's buffer level with its shadow, answering how many ranges were written.
    ///
    /// `max_gap` is how many clean bytes a merge will cross, which [`crate::uniforms`] explains is the
    /// caller's to choose rather than a rule.
    ///
    /// # Why the mapping is taken first
    ///
    /// `Consolidated::flush` *consumes* the dirty set, and there is no way to put it back. Mapping
    /// before flushing means the one call that can realistically fail has failed before anything is
    /// forgotten: a map that fails leaves the shadow dirty and the next flush tries again. The writes
    /// after it are bounds-checked against a buffer sized from the same shadow, so they cannot fail
    /// unless this module has a bug -- and if one ever does, the error says so while the shadow has
    /// already been cleared, which is why that case is called out here rather than left to be found.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the allocation will not map. A layer with no buffer answers `Ok(0)`.
    pub fn flush(&mut self, which: Which, max_gap: usize) -> Result<usize, Error> {
        let Some(held) = self.held.get_mut(&which) else {
            return Ok(0);
        };
        if !held.shadow.is_dirty() {
            return Ok(0);
        }
        let mut mapping = held.memory.map()?;
        let ranges = held.shadow.flush(max_gap);
        let bytes = held.shadow.bytes();
        let written = ranges.len();
        for range in ranges {
            mapping.write(range.start as u64, &bytes[range])?;
        }
        Ok(written)
    }

    /// Forgets a layer's buffer, freeing it.
    ///
    /// # Correctness, which this cannot check
    ///
    /// Every frame that read it must have completed, as for [`crate::store::Store::free`]. The same
    /// reasoning applies and the same module holds the judgment.
    pub fn forget(&mut self, which: Which) {
        self.held.remove(&which);
    }

    /// Forgets every layer, for a frame going away with its device still alive.
    pub fn clear(&mut self) {
        self.held.clear();
    }

    /// Reads back what a layer's buffer holds, for checking a hand-off or a test.
    ///
    /// As [`crate::store::Store::read_vertex_bytes`]: the slow way to look at a buffer, and the only
    /// way without drawing it.
    ///
    /// Answers `false` for a layer with no buffer rather than leaving `into` as it found it: a read
    /// that quietly filled nothing would let a check pass against whatever the caller's buffer held,
    /// which for a zeroed one is indistinguishable from a buffer of zeros.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the allocation will not map or the read runs past it.
    pub fn read_bytes(&self, which: Which, offset: u64, into: &mut [u8]) -> Result<bool, Error> {
        let Some(held) = self.held.get(&which) else {
            return Ok(false);
        };
        let mapping = held.memory.map()?;
        mapping.read(offset, into)?;
        Ok(true)
    }
}
