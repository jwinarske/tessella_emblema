// SPDX-License-Identifier: BSD-2-Clause
//! Where a geometry's buffers sit inside its one allocation.
//!
//! # What this is for
//!
//! `store::layout` is the arithmetic half of putting a geometry on a device, split out so it can be
//! tested without one. The device half can only be run in a bench, because CI has no GPU -- so if the
//! offsets were decided there, they would be decided where nothing checks them.
//!
//! # What would be caught
//!
//! Two buffers overlapping. That is two bindings reading each other's bytes, and the result draws:
//! a vertex stream reinterpreted as another is geometry in the wrong place rather than a blank frame,
//! so nothing about it announces itself. The alignment cases are the same failure one step earlier --
//! `vkBindBufferMemory` refuses an offset that does not satisfy the buffer's alignment, which at least
//! fails loudly, but only on the driver whose alignment is the larger one.

use tessella_capture_abi::envelope::SlabRef;
use tessella_emblema::buffers::{self, Needs, Reads};
use tessella_emblema::store::{self, At};
use tessella_emblema::vertices::Plan;

/// A reference to `length` bytes at `offset` in slab `slab`.
const fn at(slab: u32, offset: u32, length: u32) -> SlabRef {
    SlabRef {
        slab,
        offset,
        length,
    }
}

/// The needs of a geometry whose vertex buffers are the lengths given, with no indices.
fn vertices_of(lengths: &[u32]) -> Needs {
    Needs {
        vertices: lengths
            .iter()
            .enumerate()
            .map(|(index, length)| at(index as u32, 0, *length))
            .collect(),
        reads: lengths
            .iter()
            .enumerate()
            .map(|(index, _)| Reads {
                slot: index as u32,
                buffer: index,
            })
            .collect(),
        indexes: None,
    }
}

/// Buffers are placed end to end, each aligned up.
#[test]
fn buffers_are_placed_end_to_end() {
    let needs = vertices_of(&[100, 40, 8]);
    let placed = store::layout(&needs, 16);

    assert_eq!(
        placed.vertices,
        [
            At {
                offset: 0,
                length: 100
            },
            At {
                offset: 112,
                length: 40
            },
            At {
                offset: 160,
                length: 8
            },
        ],
        "100 rounds to 112 and 40 to 48, so the third starts at 160"
    );
    assert_eq!(placed.indexes, None);
    assert_eq!(placed.total, 176, "8 rounds up to 16 past 160");
}

/// No two buffers overlap, at any alignment.
///
/// The assertion that matters, and it is checked as a property rather than against a table: an
/// overlap is two bindings reading each other's bytes, which draws geometry in the wrong place and
/// announces nothing. The lengths are deliberately awkward -- a prime, a power of two, a zero -- so a
/// rounding error cannot hide in a tidy number.
#[test]
fn no_two_buffers_overlap() {
    let needs = Needs {
        vertices: vec![at(0, 0, 97), at(1, 0, 64), at(2, 0, 0), at(3, 0, 1)],
        reads: (0..4)
            .map(|index| Reads {
                slot: index as u32,
                buffer: index,
            })
            .collect(),
        indexes: Some(at(4, 0, 33)),
    };

    for alignment in [1, 4, 16, 64, 256, 1024] {
        let placed = store::layout(&needs, alignment);
        let mut spans: Vec<(u64, u64)> = placed
            .vertices
            .iter()
            .chain(placed.indexes.iter())
            .map(|At { offset, length }| (*offset, *offset + *length))
            .collect();
        spans.sort_unstable();

        for pair in spans.windows(2) {
            let (_, first_end) = pair[0];
            let (second_start, _) = pair[1];
            assert!(
                first_end <= second_start,
                "alignment {alignment}: a buffer ending at {first_end} overlaps one starting at \
                 {second_start}"
            );
        }
        let last = spans.last().copied().unwrap_or((0, 0));
        assert!(
            last.1 <= placed.total,
            "alignment {alignment}: the last buffer ends at {} past a total of {}",
            last.1,
            placed.total
        );
    }
}

/// Every offset satisfies the alignment it was laid out for.
///
/// `vkBindBufferMemory` refuses one that does not, so this is the failure that at least shows itself
/// -- but only on a driver whose alignment is the larger one, which is why it is asserted here rather
/// than left to the bench.
#[test]
fn every_offset_is_aligned() {
    let needs = Needs {
        vertices: vec![at(0, 0, 13), at(1, 0, 1), at(2, 0, 4095)],
        reads: (0..3)
            .map(|index| Reads {
                slot: index as u32,
                buffer: index,
            })
            .collect(),
        indexes: Some(at(3, 0, 7)),
    };
    for alignment in [1, 2, 16, 64, 256] {
        let placed = store::layout(&needs, alignment);
        for At { offset, .. } in placed.vertices.iter().chain(placed.indexes.iter()) {
            assert_eq!(
                offset % alignment,
                0,
                "alignment {alignment}: offset {offset} is not a multiple of it"
            );
        }
        assert_eq!(
            placed.total % alignment,
            0,
            "alignment {alignment}: the total is not a multiple of it, so a second geometry packed \
             after this one would start misaligned"
        );
    }
}

/// A zero-length buffer keeps its slot.
///
/// `Needs::reads` indexes `vertices` positionally, so a layout that dropped an empty one would move
/// every later binding onto the wrong buffer -- the same failure as an overlap, reached by a different
/// route. Its offset is still aligned, so the buffer after it does not depend on it being empty.
#[test]
fn a_zero_length_buffer_keeps_its_slot() {
    let needs = vertices_of(&[32, 0, 32]);
    let placed = store::layout(&needs, 16);

    assert_eq!(placed.vertices.len(), 3, "the empty slot was dropped");
    assert_eq!(placed.vertices[1].length, 0);
    assert_eq!(
        placed.vertices[1].offset, 32,
        "the empty buffer sits where the first one ended"
    );
    assert_eq!(
        placed.vertices[2].offset, 32,
        "and the third starts there too, because nothing was written in between"
    );
    assert_eq!(placed.total, 64);
}

/// An alignment of zero is treated as one rather than dividing by it.
///
/// Not a real requirement -- Vulkan's alignments are powers of two and at least one -- but the value
/// arrives from `max()` over a possibly-empty list of requirements, so zero is reachable by a geometry
/// with no buffers at all.
#[test]
fn an_alignment_of_zero_does_not_divide_by_it() {
    let needs = vertices_of(&[8, 8]);
    let placed = store::layout(&needs, 0);
    assert_eq!(placed.vertices[0].offset, 0);
    assert_eq!(placed.vertices[1].offset, 8);
    assert_eq!(placed.total, 16);
}

/// A geometry with nothing in it lays out to nothing.
#[test]
fn an_empty_geometry_needs_no_bytes() {
    let placed = store::layout(&Needs::default(), 256);
    assert_eq!(placed.vertices, [] as [At; 0]);
    assert_eq!(placed.indexes, None);
    assert_eq!(placed.total, 0);
}

/// The index buffer is placed after the vertices and is its own slot.
///
/// Separate even when it points into the same slab, because it binds through `vkCmdBindIndexBuffer`
/// rather than as a vertex binding -- `buffers::needs` says so and this is the layout agreeing.
#[test]
fn the_index_buffer_is_placed_separately() {
    let shared = at(7, 0, 48);
    let needs = Needs {
        vertices: vec![shared],
        reads: vec![Reads { slot: 0, buffer: 0 }],
        // The *same* reference as the vertex buffer, which is a thing the producer sends.
        indexes: Some(shared),
    };
    let placed = store::layout(&needs, 16);

    let indexes = placed.indexes.expect("an index buffer");
    assert_eq!(placed.vertices[0].offset, 0);
    assert_eq!(
        indexes.offset, 48,
        "one reference used twice is still two buffers and two places in the allocation"
    );
    assert_eq!(placed.total, 96);
}

/// A layout over the real dedup, so the two halves agree about how many buffers there are.
///
/// `buffers::needs` is what decides that three descriptors over one interleaved vertex are one
/// buffer; this checks the layout places that one rather than three. Built through `needs` rather than
/// by hand, because a `Needs` written by hand cannot disagree with the dedup and this is the agreement
/// under test.
#[test]
fn a_deduplicated_plan_lays_out_one_buffer() {
    let interleaved = at(3, 128, 240);
    // One interleaved vertex of three floats, read as three attributes at three offsets -- which is
    // what `encode_raster` and its two siblings send.
    let bound = |slot: u32, offset: u32| tessella_emblema::vertices::Bound {
        slot,
        format: ash::vk::Format::R32_SFLOAT,
        stride: 12,
        offset,
        vertex_offset: 0,
        source: interleaved,
        rate: ash::vk::VertexInputRate::VERTEX,
    };
    let plan = Plan {
        bound: vec![bound(0, 0), bound(1, 4), bound(2, 8)],
        ..Plan::default()
    };
    let needs = buffers::needs(&plan, at(0, 0, 0));
    assert_eq!(
        needs.vertices.len(),
        1,
        "three descriptors over one interleaved buffer are one buffer"
    );
    assert_eq!(needs.reads.len(), 3, "and three bindings reading it");

    let placed = store::layout(&needs, 64);
    assert_eq!(placed.vertices.len(), 1);
    assert_eq!(placed.total, 256, "240 rounded up to 64");
    assert_eq!(
        buffers::vertex_bytes(&needs),
        240,
        "the dedup is what makes this 240 rather than 720"
    );
}
