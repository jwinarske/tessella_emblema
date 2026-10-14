// SPDX-License-Identifier: BSD-2-Clause
//! Segments as draw parameters.

use tessella_capture_abi::envelope::Segment;
use tessella_emblema::draws::{Draw, Unusable, indexed};

fn segment(
    vertex_offset: u32,
    index_offset: u32,
    vertex_length: u32,
    index_length: u32,
) -> Segment {
    Segment {
        vertex_offset,
        index_offset,
        vertex_length,
        index_length,
    }
}

#[test]
fn one_segment_is_one_draw_with_the_slot_as_its_instance() {
    let got = indexed(&[segment(0, 0, 4, 6)], 7, 1).expect("draws");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].index_count, 6);
    assert_eq!(got[0].instance_count, 1);
    assert_eq!(got[0].first_index, 0);
    assert_eq!(got[0].vertex_offset, 0);
    assert_eq!(
        got[0].first_instance, 7,
        "the body reads this as `ubo_index`"
    );
}

#[test]
fn segments_become_draws_in_order_carrying_their_own_bases() {
    let run = [
        segment(0, 0, 4, 6),
        segment(4, 6, 8, 12),
        segment(12, 18, 4, 6),
    ];
    let got = indexed(&run, 2, 1).expect("draws");
    let firsts: Vec<u32> = got.iter().map(|draw| draw.first_index).collect();
    let bases: Vec<i32> = got.iter().map(|draw| draw.vertex_offset).collect();
    let counts: Vec<u32> = got.iter().map(|draw| draw.index_count).collect();
    assert_eq!(firsts, vec![0, 6, 18]);
    assert_eq!(bases, vec![0, 4, 12]);
    assert_eq!(counts, vec![6, 12, 6]);
}

#[test]
fn a_segment_with_no_indices_draws_nothing_and_does_not_shift_the_rest() {
    let run = [
        segment(0, 0, 4, 6),
        segment(4, 6, 0, 0),
        segment(4, 6, 4, 6),
    ];
    let got = indexed(&run, 0, 1).expect("draws");
    assert_eq!(got.len(), 2, "the empty one contributes no draw");
    assert_eq!(got[1].first_index, 6, "and the next keeps its own offset");
}

#[test]
fn no_instances_is_no_draws() {
    let got = indexed(&[segment(0, 0, 4, 6)], 3, 0).expect("not an error");
    assert_eq!(got, [] as [Draw; 0]);
}

/// An instanced draw repeats every segment, and its slot has to be zero.
///
/// Both halves of one rule. `firstInstance` addresses the instance-rate attribute fetch as well as
/// the builtin a body reads as `ubo_index`, so a slot there and more than one instance cannot both
/// be meant -- `benches/instance_fetch.rs` measures it on a device. This asked for `slot` one with
/// sixty-four instances and got draws, which is the shape the instanced wall family would have
/// recorded: the instance stream read from element one, and sixty-three of the sixty-four bodies
/// reading another drawable's uniforms.
#[test]
fn an_instanced_family_repeats_every_segment() {
    let run = [segment(0, 0, 4, 6), segment(4, 6, 4, 6)];
    let got = indexed(&run, 0, 64).expect("draws");
    assert_eq!(got.len(), 2);
    for draw in &got {
        assert_eq!(draw.instance_count, 64);
        assert_eq!(draw.first_instance, 0, "the slot cannot ride firstInstance");
    }

    assert_eq!(
        indexed(&run, 1, 64),
        Err(Unusable::SlotWithInstances {
            slot: 1,
            instances: 64
        }),
        "a slot and more than one instance cannot share firstInstance"
    );
}

/// One instance keeps the slot, which is every family drawn today.
///
/// The boundary, stated on its own: the refusal is about more than one instance and not about
/// instancing as an idea, so the eighteen families that draw once go on carrying their slot.
#[test]
fn one_instance_still_carries_the_slot() {
    let got = indexed(&[segment(0, 0, 4, 6)], 7, 1).expect("draws");
    assert_eq!(got[0].first_instance, 7);

    // Two is already too many, which is where the boundary is rather than at some larger count.
    assert_eq!(
        indexed(&[segment(0, 0, 4, 6)], 7, 2),
        Err(Unusable::SlotWithInstances {
            slot: 7,
            instances: 2
        })
    );
}

/// A draw of no instances is not a disagreement, whatever the slot says.
///
/// Nothing is drawn, so the two numbers have nothing to be inconsistent about -- and refusing it
/// would make a drawable whose geometry was clipped away an error rather than an empty draw.
#[test]
fn no_instances_with_a_slot_is_still_no_draws() {
    assert_eq!(indexed(&[segment(0, 0, 4, 6)], 7, 0), Ok(Vec::new()));
}

/// The signed field is what the call takes, so an offset past its range is refused rather than
/// wrapped into a draw that reads somewhere else in the buffer.
#[test]
fn a_vertex_offset_past_the_signed_range_is_refused() {
    let run = [segment(0, 0, 4, 6), segment(u32::MAX, 6, 4, 6)];
    assert_eq!(
        indexed(&run, 0, 1),
        Err(Unusable::VertexOffset {
            segment: 1,
            offset: u32::MAX
        })
    );
}

/// Reported even when that segment would have drawn nothing, so a run is judged on its contents
/// rather than on which of its segments happened to be empty.
#[test]
fn a_bad_offset_on_an_empty_segment_is_still_refused() {
    let run = [segment(u32::MAX, 0, 0, 0)];
    assert_eq!(
        indexed(&run, 0, 1),
        Err(Unusable::VertexOffset {
            segment: 0,
            offset: u32::MAX
        })
    );
}

#[test]
fn the_largest_usable_offset_is_accepted() {
    let at = u32::try_from(i32::MAX).expect("fits");
    let got = indexed(&[segment(at, 0, 4, 6)], 0, 1).expect("draws");
    assert_eq!(got[0].vertex_offset, i32::MAX);
}

#[test]
fn an_empty_run_is_no_draws() {
    assert_eq!(indexed(&[], 0, 1).expect("not an error"), [] as [Draw; 0]);
}
