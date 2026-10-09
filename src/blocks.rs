// SPDX-License-Identifier: BSD-2-Clause
//! A layer's consolidated block buffer on the device, and the writes that bring it level.
//!
//! The second part of #60's frame half, after [`crate::store`]. [`crate::uniforms`] is the host side
//! of this: it shadows a buffer, remembers which entries moved and turns them into contiguous
//! ranges. This is the buffer those ranges are written into.
//!
//! # One buffer per view, layer and slot
//!
//! `UboUpdate` carries `(view, layer_index, slot, bytes)`, and all four parts are the key. The first
//! two are the layer; `slot` is **which of that layer's buffers** -- a fill layer's drawable array
//! arrives at slot 2, its tile properties at 4 and its evaluated properties at 5, and the family's
//! three bindings read those three buffers. [`crate::slots`] is what turns a declared block into the
//! slot its bytes come in at.
//!
//! Keyed by the layer alone this held one buffer where a family needs several, and every one of its
//! storage bindings pointed at that one: the second block's fields would be read out of the first's
//! bytes, which draws.
//!
//! One buffer per layer rather than one per drawable is the whole reason the host side has a
//! shadow: a frame touches a handful of entries in a buffer holding hundreds, and §11.7 asks for the
//! sub-ranges rather than a whole-buffer rewrite, which is bandwidth a tiler does not have spare.
//!
//! A buffer bound at offset zero needs no `minStorageBufferOffsetAlignment`, which is why that limit
//! does not appear here. A later change that packed several layers into one allocation with dynamic
//! offsets would need it, and would be the place to ask for it.
//!
//! # Two numbers called `slot`, and only one of them is
//!
//! The wire has `UboUpdate::slot` for the buffer and `uboIndex` for an entry within it, and this
//! module keeps those words: `slot` is always the buffer and `index` is always the entry. They were
//! both called `slot` here and in [`crate::uniforms`] once, which put `write(which, slot, slot, ..)`
//! in reach of a caller.
//!
//! # A different shape is a different buffer
//!
//! The entry count and the block size are fixed when the buffer is made. A slot arriving with either
//! changed is not damage to this buffer, it is another buffer -- the same rule [`crate::textures`]
//! states for an image whose size or format changed. Writing the new shape into the old bytes would
//! put a drawable's block at another's offset, which draws.

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::ViewId;
use tessella_vk::{Buffer, Gpu, Memory};

use crate::uniforms::{Consolidated, Rejected};

/// Which layer of which view a buffer belongs to.
///
/// The pair `UboUpdate` carries, and not the whole of a buffer's identity: a layer has one buffer
/// per slot, so the methods here take a `slot` beside this. It is the key on its own for a
/// *descriptor set*, which is per layer rather than per buffer -- which is why it stays a type of
/// its own rather than growing a third field.
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
    /// The write was not one block, or named an entry the buffer does not have.
    Write(Rejected),
    /// A slot already has a buffer of a different shape.
    ///
    /// Not an error the producer can cause by writing: it is the caller asking for a buffer that
    /// disagrees with one already made, which means the layer's layout changed and the old buffer
    /// should have been dropped first.
    Reshaped {
        /// Entries the existing buffer has.
        entries: usize,
        /// Bytes per block it has.
        block: usize,
    },
    /// The buffer has no entries, or its blocks are no bytes.
    ///
    /// Refused rather than made as a one-byte buffer, which is what the wrapper's `max(1)` would
    /// otherwise produce. A buffer with no entries has nothing to draw, and a block of zero bytes
    /// makes every entry's offset zero -- so the shadow would take a write for any index at all and
    /// put it over entry zero, which is a layout disagreement answering `Ok`.
    Degenerate {
        /// Entries asked for.
        entries: usize,
        /// Bytes per block asked for.
        block: usize,
    },
    /// The entry count and block size multiply to more than an allocation can be.
    ///
    /// `Consolidated::new` sizes its shadow with an unchecked `entries * block`, so this has to be
    /// refused a step earlier: `usize` is thirty-two bits on some targets this builds for, where a
    /// product that wraps makes a shadow far smaller than the entry count claims and every write
    /// past the wrap lands on an earlier entry.
    Overflows {
        /// Entries asked for.
        entries: usize,
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
            Self::Degenerate { entries, block } => write!(
                f,
                "a buffer of {entries} entries of {block} bytes has nothing to draw and no entry arithmetic"
            ),
            Self::Overflows { entries, block } => write!(
                f,
                "{entries} entries of {block} bytes is more than an allocation can be"
            ),
            Self::Reshaped { entries, block } => write!(
                f,
                "this slot already has {entries} entries of {block} bytes, which is a different buffer"
            ),
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// How many bytes a buffer of this shape needs, or why it cannot have any.
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
/// [`Error::Degenerate`] for no entries or zero-byte blocks, and [`Error::Overflows`] when the
/// product will not fit a `usize` on this target.
pub fn sizing(entries: usize, block: usize) -> Result<u64, Error> {
    if entries == 0 || block == 0 {
        return Err(Error::Degenerate { entries, block });
    }
    let bytes = entries
        .checked_mul(block)
        .ok_or(Error::Overflows { entries, block })?;
    u64::try_from(bytes).map_err(|_| Error::Overflows { entries, block })
}

/// One slot's shadow and the buffer it is written into.
struct Held<'d> {
    shadow: Consolidated,
    buffer: Buffer<'d>,
    memory: Memory<'d>,
}

/// The block buffers the drawn layers read, one per view, layer and slot.
///
/// `'d` is the device's lifetime, borrowed through [`Gpu`]: these cannot outlive the device their
/// buffers are on, which the compiler checks rather than a comment asking.
#[derive(Default)]
pub struct Blocks<'d> {
    held: BTreeMap<(Which, u32), Held<'d>>,
}

impl core::fmt::Debug for Blocks<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Blocks")
            .field("buffers", &self.held.len())
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

    /// How many layers have at least one buffer.
    #[must_use]
    pub fn layers(&self) -> usize {
        // The map is ordered by `(Which, slot)`, so a layer's buffers are adjacent and `dedup`
        // leaves one of each run.
        let mut layers: Vec<Which> = self.held.keys().map(|(which, _)| *which).collect();
        layers.dedup();
        layers.len()
    }

    /// How many buffers are held, across every layer.
    #[must_use]
    pub fn buffers(&self) -> usize {
        self.held.len()
    }

    /// The buffer a layer's bindings read at one slot, once it has one.
    #[must_use]
    pub fn buffer(&self, which: Which, slot: u32) -> Option<vk::Buffer> {
        Some(self.held.get(&(which, slot))?.buffer.raw())
    }

    /// Device bytes every buffer comes to.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.held.values().map(|held| held.memory.size()).sum()
    }

    /// Makes this slot's buffer if it has none, and answers whether anything was made.
    ///
    /// Idempotent for a slot whose shape has not changed, which is the ordinary case: a layer keeps
    /// its entry count for as long as the style does.
    ///
    /// # Errors
    ///
    /// [`Error::Reshaped`] when the slot has a buffer of another shape -- drop it first, because the
    /// old bytes are laid out for the old entry count. [`Error::Degenerate`] for a buffer with no
    /// entries or zero-byte blocks. [`Error::Device`] when the device refuses.
    pub fn declare(
        &mut self,
        gpu: Gpu<'d>,
        which: Which,
        slot: u32,
        entries: usize,
        block: usize,
    ) -> Result<bool, Error> {
        if let Some(held) = self.held.get(&(which, slot)) {
            let (have_entries, have_block) = (held.shadow.entries(), held.shadow.block());
            if have_entries == entries && have_block == block {
                return Ok(false);
            }
            return Err(Error::Reshaped {
                entries: have_entries,
                block: have_block,
            });
        }

        let bytes = sizing(entries, block)?;
        let shadow = Consolidated::new(entries, block);
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
        // `vkAllocateMemory` says nothing about what the memory holds, and an entry nothing has
        // written is never in a flushed range -- so without this it reads back as whatever the allocation came
        // with. Measured: on RADV and V3D that is zeros and this write changes nothing, and on the
        // VeriSilicon GC7000UL an unwritten entry came back holding 0x01. Three drivers, two answers,
        // and the spec allows both, so the one that looked right was luck.
        //
        // A drawable whose block has not been written yet is reached by anything that names it before
        // the producer fills it, which is the first frame of every layer. Garbage there is a feature
        // drawn at a garbage width in a garbage color rather than an invisible one.
        memory.map()?.write(0, shadow.bytes())?;
        self.held.insert(
            (which, slot),
            Held {
                shadow,
                buffer,
                memory,
            },
        );
        Ok(true)
    }

    /// Writes one entry of one buffer into its shadow, leaving the device until a flush.
    ///
    /// A slot with no buffer is `Ok(false)`: the producer may send a block for a layer or a slot this
    /// frame has not declared, and dropping it is what a consumer that does not draw that layer
    /// should do.
    ///
    /// # Errors
    ///
    /// [`Error::Write`] when the data is not one block or the entry is past the end.
    pub fn write(
        &mut self,
        which: Which,
        slot: u32,
        index: u32,
        data: &[u8],
    ) -> Result<bool, Error> {
        let Some(held) = self.held.get_mut(&(which, slot)) else {
            return Ok(false);
        };
        held.shadow.write(index, data)?;
        Ok(true)
    }

    /// Whether a buffer has writes the device has not seen.
    #[must_use]
    pub fn is_dirty(&self, which: Which, slot: u32) -> bool {
        self.held
            .get(&(which, slot))
            .is_some_and(|held| held.shadow.is_dirty())
    }

    /// Brings one buffer level with its shadow, answering how many ranges were written.
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
    /// [`Error::Device`] when the allocation will not map. A slot with no buffer answers `Ok(0)`.
    pub fn flush(&mut self, which: Which, slot: u32, max_gap: usize) -> Result<usize, Error> {
        let Some(held) = self.held.get_mut(&(which, slot)) else {
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

    /// Forgets every buffer a layer has, freeing them.
    ///
    /// The whole layer rather than one slot, because that is how a layer goes away: a style change
    /// retires it with all of its blocks, and a slot dropped on its own would leave a binding with
    /// nothing to point at while the rest of the set still resolved.
    ///
    /// # Correctness, which this cannot check
    ///
    /// Every frame that read it must have completed, as for [`crate::store::Store::free`]. The same
    /// reasoning applies and the same module holds the judgment.
    pub fn forget(&mut self, which: Which) {
        self.held.retain(|(had, _), _| *had != which);
    }

    /// Forgets every layer, for a frame going away with its device still alive.
    pub fn clear(&mut self) {
        self.held.clear();
    }

    /// Reads back what one buffer holds, for checking a hand-off or a test.
    ///
    /// As [`crate::store::Store::read_vertex_bytes`]: the slow way to look at a buffer, and the only
    /// way without drawing it.
    ///
    /// Answers `false` for a slot with no buffer rather than leaving `into` as it found it: a read
    /// that quietly filled nothing would let a check pass against whatever the caller's buffer held,
    /// which for a zeroed one is indistinguishable from a buffer of zeros.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the allocation will not map or the read runs past it.
    pub fn read_bytes(
        &self,
        which: Which,
        slot: u32,
        offset: u64,
        into: &mut [u8],
    ) -> Result<bool, Error> {
        let Some(held) = self.held.get(&(which, slot)) else {
            return Ok(false);
        };
        let mapping = held.memory.map()?;
        mapping.read(offset, into)?;
        Ok(true)
    }
}
