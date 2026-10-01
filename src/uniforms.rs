//! A layer's consolidated buffer, and the smallest set of writes that brings the device level.
//!
//! §11.7 asks a consumer for sub-range buffer updates from the dirty ranges, not whole-buffer
//! rewrites. A frame touches a handful of slots in a buffer holding hundreds, and rewriting all of
//! it is bandwidth a tiler does not have spare.
//!
//! So the buffer is shadowed here. Slots are written into the shadow, the slots touched since the
//! last flush are remembered, and a flush turns those into contiguous byte ranges for the backend
//! to copy. Nothing here touches a device: the shadow is the thing that makes a sub-range write
//! expressible at all, since a device write needs its source contiguous and the producer's updates
//! arrive scattered.
//!
//! # Why merging has a threshold rather than a rule
//!
//! Two dirty slots either side of a clean one can go as two writes or as one covering all three.
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
    /// A layer's blocks are one size, fixed when the buffer was made. A write of another length is
    /// the producer and this consumer disagreeing about the layout, and writing it anyway puts the
    /// right bytes at the wrong offsets for every slot after it -- which draws, and draws wrong.
    WrongLength {
        /// What the buffer's blocks are.
        expected: usize,
        /// What arrived.
        got: usize,
    },
    /// The slot is past what this buffer can hold.
    ///
    /// Refused rather than grown. The buffer's size comes from the view's own declaration, so a
    /// slot past it is a stale order naming a drawable this layer no longer has, and growing to
    /// fit would hide that behind an allocation.
    NoSuchSlot {
        /// How many slots the buffer has.
        slots: usize,
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
    /// A buffer of `slots` blocks of `block` bytes, zeroed.
    ///
    /// Zeroed rather than uninitialized because a slot nothing has written yet is read by anything
    /// that names it before the producer fills it, and zeros are at least a defined picture.
    #[must_use]
    pub fn new(slots: usize, block: usize) -> Self {
        Self {
            bytes: vec![0; slots * block],
            block,
            dirty: BTreeSet::new(),
        }
    }

    /// Bytes per slot.
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// How many slots the buffer holds.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.bytes.len().checked_div(self.block).unwrap_or(0)
    }

    /// The shadow, for a backend copying a flushed range out of it.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Writes one slot, marking it dirty.
    ///
    /// Latest wins: a slot written twice before a flush carries the second write's bytes and is
    /// still one dirty slot, which is the producer's own contract for a `UboUpdate`.
    ///
    /// # Errors
    ///
    /// [`Rejected`] when the data is not one block, or the slot is past the buffer. Nothing is
    /// written in either case.
    pub fn write(&mut self, slot: u32, data: &[u8]) -> Result<(), Rejected> {
        if data.len() != self.block {
            return Err(Rejected::WrongLength {
                expected: self.block,
                got: data.len(),
            });
        }
        // Checked, because `usize` is thirty-two bits on some targets this builds for and a slot
        // near `u32::MAX` times a block size then wraps to a small offset -- which passes the
        // bounds test below and writes over slot zero. The arithmetic is the bounds check here,
        // not a step before it.
        let at = (slot as usize)
            .checked_mul(self.block)
            .and_then(|at| (at.checked_add(self.block)? <= self.bytes.len()).then_some(at));
        let Some(at) = at else {
            return Err(Rejected::NoSuchSlot {
                slots: self.slots(),
                got: slot,
            });
        };
        self.bytes[at..at + self.block].copy_from_slice(data);
        self.dirty.insert(slot);
        Ok(())
    }

    /// Whether anything has been written since the last flush.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// How many slots are dirty, for a consumer watching its own upload traffic.
    #[must_use]
    pub fn dirty_slots(&self) -> usize {
        self.dirty.len()
    }

    /// The byte ranges to copy, merging runs separated by `max_gap` bytes or fewer.
    ///
    /// Clears the dirty set: what this returns is what the backend owes the device, and returning
    /// it twice would have it copy the same bytes again. Ranges are in ascending order and do not
    /// overlap, so a backend can hand them straight to a region list.
    ///
    /// `max_gap` of zero merges only slots that touch. A gap large enough to span the buffer
    /// collapses everything to one range, which is the whole-buffer rewrite this exists to avoid --
    /// so it is a number worth choosing rather than defaulting.
    pub fn flush(&mut self, max_gap: usize) -> Vec<Range<usize>> {
        let mut out: Vec<Range<usize>> = Vec::with_capacity(self.dirty.len());
        for slot in &self.dirty {
            let at = *slot as usize * self.block;
            let range = at..at + self.block;
            match out.last_mut() {
                // The slots are ascending, so only the open run can ever be extended.
                Some(open) if range.start - open.end <= max_gap => open.end = range.end,
                _ => out.push(range),
            }
        }
        self.dirty.clear();
        out
    }
}
