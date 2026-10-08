// SPDX-License-Identifier: BSD-2-Clause
//! Which stencil reference each tile's clip mask gets.
//!
//! # What this is for
//!
//! `StencilTiles` carries no reference value, so the consumer assigns them -- and every rule about
//! how is a rule a stencil buffer imposes, with a failure that draws rather than fails.
//!
//! # What would be caught
//!
//! **Reference zero.** The rendering scope clears the stencil to zero, so a mask drawn under zero
//! is indistinguishable from the cleared buffer: a content draw testing `EQUAL` against it passes
//! everywhere, which is a layer with no clipping drawn over its neighbors.
//!
//! **A wrapped reference.** Eight bits is 255 usable values. Wrapping hands a new tile the number
//! an old mask is still drawn under, and the new tile is then clipped to the old one's shape --
//! geometry cut along an edge that is not there, in a frame nothing else distinguishes.
//!
//! **A reset without a clear.** Resetting the counter without clearing the buffer is the same
//! failure by the other route, which is why `clear_first` exists and why it is a field the caller
//! has to read rather than something inferred.

use tessella_capture_abi::envelope::TileId;
use tessella_emblema::masks::{self, References};

/// A tile at a canonical address, with no overscale and no wrap.
fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId {
        z,
        x,
        y,
        overscaled_z: z,
        wrap: 0,
    }
}

/// References start at one, because zero is what the buffer is cleared to.
#[test]
fn references_start_above_zero() {
    assert_eq!(masks::FIRST, 1, "zero is the cleared value, not a mask");
    let mut refs = References::new();
    let pass = refs.assign(&[tile(14, 0, 0), tile(14, 1, 0)]);
    assert_eq!(pass.references, [1, 2]);
    assert!(
        !pass.references.contains(&0),
        "a mask under zero is clipped by nothing"
    );
    assert!(!pass.clear_first, "a fresh counter needs no clear");
    assert_eq!(pass.unreferenced, 0);
}

/// A tile named twice in one pass keeps one reference and consumes one number.
///
/// The mask is already in the buffer under the first; drawing it again under a second reference
/// would leave the first occupied by a mask nothing tests against, and spend a number from a budget
/// of 255.
#[test]
fn a_repeated_tile_keeps_its_reference() {
    let mut refs = References::new();
    let repeated = tile(14, 5, 5);
    let pass = refs.assign(&[repeated, tile(14, 6, 5), repeated]);
    assert_eq!(pass.references, [1, 2, 1], "the third is the first again");
    assert_eq!(refs.len(), 2, "two tiles, two numbers");
}

/// The counter keeps climbing across passes.
///
/// mbgl's own order: it clears its tile map every pass and keeps the counter, so a tile drawn in
/// two passes gets two references and the second pass does not reuse the first's numbers -- which
/// it must not, because the masks from the first are still in the buffer.
#[test]
fn the_counter_climbs_across_passes() {
    let mut refs = References::new();
    let first = refs.assign(&[tile(14, 0, 0), tile(14, 1, 0)]);
    assert_eq!(first.references, [1, 2]);

    let second = refs.assign(&[tile(14, 2, 0)]);
    assert_eq!(
        second.references,
        [3],
        "the second pass must not reuse a live reference"
    );
    assert!(!second.clear_first);

    // And the previous pass's assignments are forgotten, which is what lets a tile be renumbered.
    assert_eq!(refs.of(tile(14, 0, 0)), None);
    assert_eq!(refs.of(tile(14, 2, 0)), Some(3));
}

/// Overflow resets to one and says the buffer must be cleared.
///
/// Decided for the whole pass before anything is handed out, as mbgl does -- a pass that discovered
/// the overflow partway through would have drawn some of its masks under the old numbering already.
#[test]
fn overflow_resets_and_demands_a_clear() {
    let mut refs = References::new();
    // Fill the budget: 255 distinct tiles take references 1 through 255.
    let full: Vec<TileId> = (0..255u32).map(|x| tile(14, x, 0)).collect();
    let first = refs.assign(&full);
    assert!(!first.clear_first);
    assert_eq!(first.references.first(), Some(&masks::FIRST));
    assert_eq!(first.references.last(), Some(&masks::MAX));
    assert_eq!(first.unreferenced, 0);

    // One more tile cannot fit, so the counter resets and the buffer has to be cleared.
    let next = refs.assign(&[tile(14, 999, 0)]);
    assert!(
        next.clear_first,
        "a reset without a clear leaves live masks under the reused numbers"
    );
    assert_eq!(next.references, [masks::FIRST]);
}

/// No reference ever exceeds what eight bits hold.
///
/// Asserted as a property over a long run of passes rather than against a table: a wrapped
/// reference is the failure that clips a tile to another's shape, and it would appear only after
/// enough passes to exhaust the budget.
#[test]
fn no_reference_ever_exceeds_the_maximum() {
    let mut refs = References::new();
    for pass in 0..40u32 {
        // Twenty tiles a pass, so the budget is crossed several times over.
        let tiles: Vec<TileId> = (0..20u32).map(|x| tile(14, pass * 100 + x, 0)).collect();
        let assigned = refs.assign(&tiles);
        for reference in &assigned.references {
            assert!(
                *reference >= masks::FIRST && *reference <= masks::MAX,
                "pass {pass} assigned {reference}, outside {}..={}",
                masks::FIRST,
                masks::MAX
            );
        }
        assert_eq!(assigned.unreferenced, 0, "pass {pass}");
    }
}

/// A pass larger than the whole budget reports what it could not number.
///
/// Where this does more than mbgl, which resets and then runs its counter past 255 into values a
/// `uint8` stencil truncates. Nothing is wrapped here; the caller is told instead.
#[test]
fn a_pass_larger_than_the_budget_reports_it() {
    let mut refs = References::new();
    let room = (masks::MAX - masks::FIRST + 1) as usize;
    let too_many: Vec<TileId> = (0..room as u32 + 7).map(|x| tile(14, x, 0)).collect();
    let pass = refs.assign(&too_many);

    assert_eq!(
        pass.unreferenced, 7,
        "seven tiles past a budget of {room} must be reported"
    );
    assert_eq!(pass.references.len(), room);
    for reference in &pass.references {
        assert!(*reference >= masks::FIRST && *reference <= masks::MAX);
    }
}

/// Tiles that differ only in their world copy are different tiles.
///
/// A map wrapped past the antimeridian draws the same canonical tile twice, in two places. They
/// need two masks, and a key that ignored `wrap` would give the second the first's reference --
/// clipping the wrapped copy to where the original is.
#[test]
fn a_wrapped_copy_is_its_own_tile() {
    let mut refs = References::new();
    let here = tile(14, 3, 3);
    let wrapped = TileId { wrap: 1, ..here };
    let pass = refs.assign(&[here, wrapped]);
    assert_eq!(
        pass.references,
        [1, 2],
        "a world copy needs its own mask and its own reference"
    );
}
