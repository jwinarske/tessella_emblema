// SPDX-License-Identifier: BSD-2-Clause
//! Which stencil reference each tile's clip mask gets.
//!
//! `StencilTiles` carries a matrix and a tile per mask and **no reference value**, deliberately:
//!
//! > mbgl assigns them from a running counter it resets on overflow, which is bookkeeping about a
//! > stencil buffer the producer does not own. The consumer assigns its own and keys them by tile.
//!
//! So this is that counter. It is transcribed from `PaintParameters` in
//! `src/mbgl/renderer/paint_parameters.cpp` rather than invented, because the numbers have to agree
//! with nothing else -- they are the consumer's own -- but the *rules* around them are the ones a
//! stencil buffer imposes and mbgl has already met.
//!
//! # The rules, and what each one is for
//!
//! **Zero is never assigned.** The rendering scope clears the stencil to zero, so a mask drawn with
//! reference zero is indistinguishable from the cleared buffer and a content draw testing `EQUAL`
//! against it passes everywhere -- which is a layer with no clipping at all, drawn over its
//! neighbors. mbgl starts its counter at one for the same reason.
//!
//! **A repeated tile keeps its reference.** One pass's list can name a tile twice, and the second
//! must not consume a new number: the mask is already in the buffer under the first.
//!
//! **Overflow clears rather than wraps.** The stencil is eight bits, so there are 255 usable
//! values. mbgl's rule is to check before assigning a whole pass -- `nextStencilID + count >
//! maxStencilValue` -- and reset the counter *and clear the buffer* if it would not fit. Wrapping
//! instead would hand a new tile the reference an old mask is still drawn under, which clips the
//! new tile to the old one's shape: geometry cut along an edge that is not there.
//!
//! # Where this does more than mbgl
//!
//! A pass of more than 254 tiles cannot be satisfied, because resetting to one still leaves fewer
//! references than tiles. mbgl does not check for that -- its reset happens and the counter then
//! runs past 255 into values a `uint8` stencil truncates. [`Pass::unreferenced`] reports it instead,
//! so a caller can decide, rather than drawing masks whose references silently collide.

use std::collections::{BTreeMap, BTreeSet};

use tessella_capture_abi::envelope::TileId;

/// The largest reference an eight-bit stencil buffer can hold.
///
/// mbgl's `maxStencilValue`. Not a device limit -- `VkPhysicalDeviceLimits` has no stencil-value
/// field, because the width is the format's: every depth-stencil format this crate selects carries
/// `S8_UINT`, and eight bits is 255 usable values above the reserved zero.
pub const MAX: u32 = 255;

/// The first reference assigned, and the one the counter resets to.
///
/// One, not zero: zero is what the buffer is cleared to.
pub const FIRST: u32 = 1;

/// What one pass of masks needs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pass {
    /// Whether the stencil buffer must be cleared before these masks are drawn.
    ///
    /// True when the counter had to reset. A caller that ignores it draws new masks under
    /// references that old masks still occupy, and the content tested against them is clipped to
    /// another tile's shape.
    pub clear_first: bool,
    /// One reference per tile, in the order the tiles were given.
    ///
    /// Shorter than the tile list only when [`Self::unreferenced`] is non-zero.
    pub references: Vec<u32>,
    /// Tiles that could not be given a reference.
    ///
    /// Non-zero only for a pass of more than `MAX - FIRST` distinct tiles, which is more than a
    /// view has ever had. Reported rather than wrapped, because a wrapped reference collides with a
    /// live mask.
    pub unreferenced: usize,
}

/// The running counter, and what it has assigned.
#[derive(Debug, Clone, Default)]
pub struct References {
    next: u32,
    assigned: BTreeMap<TileId, u32>,
}

impl References {
    /// A fresh counter, at [`FIRST`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            next: FIRST,
            assigned: BTreeMap::new(),
        }
    }

    /// The reference a tile holds, if this pass gave it one.
    #[must_use]
    pub fn of(&self, tile: TileId) -> Option<u32> {
        self.assigned.get(&tile).copied()
    }

    /// How many references this pass handed out.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assigned.len()
    }

    /// Whether nothing is assigned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assigned.is_empty()
    }

    /// Assigns a reference to each tile of one pass.
    ///
    /// The previous pass's assignments are forgotten, which is mbgl's own order: it clears its map
    /// at the start of every pass and keeps the counter, so references keep climbing across passes
    /// until one would not fit.
    ///
    /// Repeats within the list share a reference and consume one number.
    pub fn assign(&mut self, tiles: &[TileId]) -> Pass {
        self.assigned.clear();

        // Distinct tiles, because that is what consumes numbers. Counted before anything is handed
        // out, which is what lets the overflow be decided once for the whole pass rather than
        // discovered partway through it -- mbgl checks `nextStencilID + count` the same way.
        let distinct = {
            let mut seen = BTreeSet::new();
            tiles.iter().filter(|tile| seen.insert(**tile)).count()
        };

        let room = (MAX - FIRST + 1) as usize;
        let clear_first = self.next as usize + distinct > MAX as usize + 1;
        if clear_first {
            self.next = FIRST;
        }

        let mut references = Vec::with_capacity(tiles.len());
        let mut unreferenced = 0usize;
        for tile in tiles {
            if let Some(had) = self.assigned.get(tile) {
                references.push(*had);
                continue;
            }
            if self.next > MAX {
                // Only reachable for a pass of more than `room` distinct tiles, where resetting
                // does not help. Nothing is wrapped: the tile goes unreferenced and the caller is
                // told how many did.
                unreferenced += 1;
                continue;
            }
            let reference = self.next;
            self.next += 1;
            self.assigned.insert(*tile, reference);
            references.push(reference);
        }
        debug_assert!(unreferenced == 0 || distinct > room);

        Pass {
            clear_first,
            references,
            unreferenced,
        }
    }
}
