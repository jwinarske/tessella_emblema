//! A layer's consolidated buffer, and the smallest set of writes that brings the device level.
//!
//! §11.7 asks a consumer for sub-range buffer updates from the dirty ranges, not whole-buffer
//! rewrites. A frame touches a handful of entries in a buffer holding hundreds, and rewriting all of
//! it is bandwidth a tiler does not have spare.
//!
//! So the buffer is shadowed here. Entries are written into the shadow, the entries touched since
//! the last flush are remembered, and a flush turns those into contiguous byte ranges for the
//! backend to copy. Nothing here touches a device: the shadow is the thing that makes a sub-range write
//! expressible at all, since a device write needs its source contiguous and the producer's updates
//! arrive scattered.
//!
//! # An entry, not a slot
//!
//! The wire has two numbers and `UboUpdate` spells them `slot` and `uboIndex`: the first is which of
//! a layer's buffers, the second is which entry within it. This module is inside one buffer, so
//! everything here is an *index* and the word slot does not appear. Both were called `slot` once,
//! which put `blocks::write(which, slot, slot, ..)` in reach of a caller.
//!
//! # Why merging has a threshold rather than a rule
//!
//! Two dirty entries either side of a clean one can go as two writes or as one covering all three.
//! One write moves more bytes; two pay the per-write overhead twice. Neither is right in general,
//! so the gap a merge will cross is the caller's to set and this does not guess: a backend
//! recording `vkCmdCopyBuffer` regions and one calling `vkCmdUpdateBuffer` have different answers,
//! and so does the same backend on a different part.

use std::collections::BTreeSet;
use std::ops::Range;

/// Why a write could not be taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// The data is not one block.
    ///
    /// A buffer's blocks are one size, fixed when it was made. A write of another length is the
    /// producer and this consumer disagreeing about the layout, and writing it anyway puts the right
    /// bytes at the wrong offsets for every entry after it -- which draws, and draws wrong.
    WrongLength {
        /// What the buffer's blocks are.
        expected: usize,
        /// What arrived.
        got: usize,
    },
    /// The data is not the whole buffer.
    ///
    /// [`Consolidated::replace`] takes what an `UboUpdate` carries, which is the buffer entire. A
    /// shorter one is not a partial write to apply: the record describes the whole buffer, so a
    /// length that disagrees means the producer and this consumer disagree about how many entries
    /// the layer has -- and that is a *different buffer*, which `blocks::declare` refuses for the
    /// same reason.
    NotTheBuffer {
        /// How many bytes this buffer holds.
        expected: usize,
        /// What arrived.
        got: usize,
    },
    /// The index is past what this buffer can hold.
    ///
    /// Refused rather than grown. The buffer's size comes from the view's own declaration, so an
    /// index past it is a stale order naming a drawable this layer no longer has, and growing to
    /// fit would hide that behind an allocation.
    NoSuchIndex {
        /// How many entries the buffer has.
        entries: usize,
        /// Which was asked for.
        got: u32,
    },
}

/// One layer's consolidated buffer, shadowed.
#[derive(Debug, Clone)]
pub struct Consolidated {
    bytes: Vec<u8>,
    block: usize,
    dirty: BTreeSet<u32>,
}

impl Consolidated {
    /// A buffer of `entries` blocks of `block` bytes, zeroed.
    ///
    /// Zeroed rather than uninitialized because an entry nothing has written yet is read by anything
    /// that names it before the producer fills it, and zeros are at least a defined picture.
    #[must_use]
    pub fn new(entries: usize, block: usize) -> Self {
        Self {
            bytes: vec![0; entries * block],
            block,
            dirty: BTreeSet::new(),
        }
    }

    /// Bytes per entry.
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// How many entries the buffer holds.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.bytes.len().checked_div(self.block).unwrap_or(0)
    }

    /// The shadow, for a backend copying a flushed range out of it.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Writes one entry, marking it dirty.
    ///
    /// Latest wins: an entry written twice before a flush carries the second write's bytes and is
    /// still one dirty entry, which is the producer's own contract for a `UboUpdate`.
    ///
    /// # Errors
    ///
    /// [`Rejected`] when the data is not one block, or the index is past the buffer. Nothing is
    /// written in either case.
    pub fn write(&mut self, index: u32, data: &[u8]) -> Result<(), Rejected> {
        if data.len() != self.block {
            return Err(Rejected::WrongLength {
                expected: self.block,
                got: data.len(),
            });
        }
        // Checked, because `usize` is thirty-two bits on some targets this builds for and an index
        // near `u32::MAX` times a block size then wraps to a small offset -- which passes the
        // bounds test below and writes over entry zero. The arithmetic is the bounds check here,
        // not a step before it.
        let at = (index as usize)
            .checked_mul(self.block)
            .and_then(|at| (at.checked_add(self.block)? <= self.bytes.len()).then_some(at));
        let Some(at) = at else {
            return Err(Rejected::NoSuchIndex {
                entries: self.entries(),
                got: index,
            });
        };
        self.bytes[at..at + self.block].copy_from_slice(data);
        self.dirty.insert(index);
        Ok(())
    }

    /// Takes a whole buffer, marking the entries whose bytes changed.
    ///
    /// What an `UboUpdate` actually delivers. The producer writes a layer's buffer entire -- one
    /// record carrying `pack_drawable_buffer`'s whole array -- and [`Self::write`] takes one entry,
    /// so before this there was no call that could apply what arrived. A twelve-drawable fill layer
    /// turns up as 1152 bytes and `write` rejects every length but 96.
    ///
    /// # The dirty set comes from the comparison
    ///
    /// §11.7 asks a consumer for "sub-range buffer updates from UBO dirty ranges" and the wire
    /// carries no ranges, so deriving them is this side's job. That is what the shadow is for: the
    /// bytes that arrived are compared against the bytes already there, and only the entries that
    /// differ are marked. A still map re-sending an identical buffer flushes nothing.
    ///
    /// Comparing rather than trusting also means a producer that re-sends a buffer it did not
    /// change costs a comparison and no bandwidth, which is the common case on a parked frame.
    ///
    /// # Errors
    ///
    /// [`Rejected::NotTheBuffer`] when the data is not exactly this buffer's size. Nothing is
    /// written in that case.
    pub fn replace(&mut self, data: &[u8]) -> Result<usize, Rejected> {
        if data.len() != self.bytes.len() {
            return Err(Rejected::NotTheBuffer {
                expected: self.bytes.len(),
                got: data.len(),
            });
        }
        let mut changed = 0;
        for (index, (held, arrived)) in self
            .bytes
            .chunks_mut(self.block)
            .zip(data.chunks(self.block))
            .enumerate()
        {
            if held == arrived {
                continue;
            }
            held.copy_from_slice(arrived);
            // The index is bounded by the entry count, which was a `usize` when the buffer was
            // made -- so a buffer with more entries than `u32` can name cannot exist to reach here.
            // Skipped rather than panicked over, because the alternative is a panic in a path fed
            // by a protocol record.
            if let Ok(index) = u32::try_from(index) {
                self.dirty.insert(index);
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// Whether anything has been written since the last flush.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// How many entries are dirty, for a consumer watching its own upload traffic.
    #[must_use]
    pub fn dirty_entries(&self) -> usize {
        self.dirty.len()
    }

    /// The byte ranges to copy, merging runs separated by `max_gap` bytes or fewer.
    ///
    /// Clears the dirty set: what this returns is what the backend owes the device, and returning
    /// it twice would have it copy the same bytes again. Ranges are in ascending order and do not
    /// overlap, so a backend can hand them straight to a region list.
    ///
    /// `max_gap` of zero merges only entries that touch. A gap large enough to span the buffer
    /// collapses everything to one range, which is the whole-buffer rewrite this exists to avoid --
    /// so it is a number worth choosing rather than defaulting.
    pub fn flush(&mut self, max_gap: usize) -> Vec<Range<usize>> {
        let mut out: Vec<Range<usize>> = Vec::with_capacity(self.dirty.len());
        for index in &self.dirty {
            let at = *index as usize * self.block;
            let range = at..at + self.block;
            match out.last_mut() {
                // The indexes are ascending, so only the open run can ever be extended.
                Some(open) if range.start - open.end <= max_gap => open.end = range.end,
                _ => out.push(range),
            }
        }
        self.dirty.clear();
        out
    }
}
