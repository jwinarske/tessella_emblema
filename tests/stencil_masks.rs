// SPDX-License-Identifier: BSD-2-Clause
//! Why the stencil compare and write masks are dynamic state.
//!
//! # What this is for
//!
//! `tessella_consume::stencil::partition` hands each tile an `Assignment` whose masks are **not**
//! `0xFF`, and are not equal to each other. The pipelines bake neither, and these are the facts
//! that make that necessary -- asserted here because the bench cannot show them.
//!
//! # What the bench cannot show
//!
//! `benches/clip_masks.rs` draws one tile's mask and tests against it. Its cover is one layer group,
//! so every mask the partition hands out occupies the low bits of the byte and nothing occupies the
//! rest -- which means `0xFF` and the real mask select the same bits *in that scene*, and a draw
//! using either behaves identically. The bench proves the masks are wired (removing the compare-mask
//! call clips everything away); it cannot prove they are *right*.
//!
//! What makes them right is the partition's own contract, and that is checkable without a device.
//!
//! # What would be caught
//!
//! A pipeline baking `0xFF`, which is what both of these did before. For a descendant tile the write
//! mask is wider than the read mask -- it clears the ancestor's field as well as setting its own --
//! so a single baked value cannot be both, and a byte-wide one is neither.

use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use tessella_capture_abi::envelope::TileId;
use tessella_consume::stencil;
use tessella_emblema::pipelines;

fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId {
        z,
        x,
        y,
        overscaled_z: z,
        wrap: 0,
    }
}

/// A parent and its four children, in one layer group.
fn two_zooms() -> stencil::Partition {
    let mut tiles = BTreeSet::new();
    tiles.insert(tile(14, 5, 5));
    for (x, y) in [(10, 10), (11, 10), (10, 11), (11, 11)] {
        tiles.insert(tile(15, x, y));
    }
    let mut groups = BTreeMap::new();
    groups.insert(0i32, tiles.clone());
    stencil::partition(&tiles, &groups, stencil::ALL_BITS)
}

/// A descendant's write mask is wider than its read mask.
///
/// The mechanism the partition exists for: a child's mask clears the parent's field as well as
/// setting its own, so the parent's content stops drawing where the child covers the ground. The
/// read mask is its own field alone, because the parent's bit is not the child's to test.
///
/// So one baked mask cannot serve both, which is the whole reason they are dynamic.
#[test]
fn a_descendants_write_mask_is_wider_than_its_read_mask() {
    let partition = two_zooms();
    assert!(
        partition.partitioned,
        "five tiles over two zooms should fit"
    );

    let child = partition
        .tiles
        .get(&tile(15, 10, 10))
        .copied()
        .expect("the child is in the cover");
    assert!(
        child.write_mask > child.read_mask,
        "a child writes its own field and the parent's: read {:#04x}, write {:#04x}",
        child.read_mask,
        child.write_mask
    );
    assert_ne!(
        child.read_mask, child.write_mask,
        "one baked mask cannot be both"
    );

    let parent = partition
        .tiles
        .get(&tile(14, 5, 5))
        .copied()
        .expect("the parent is in the cover");
    assert!(
        parent.read_mask < child.read_mask,
        "the two zooms hold different fields: parent {:#04x}, child {:#04x}",
        parent.read_mask,
        child.read_mask
    );
    // And the child's write covers the parent's field, which is how it clears it.
    assert_eq!(
        child.write_mask & parent.read_mask,
        parent.read_mask,
        "a child that did not cover the parent's field could not clear it"
    );
}

/// No mask the partition hands out is the whole byte.
///
/// Which is what both pipelines baked. A content draw comparing `0xFF` reads bits the partition did
/// not give it, and a mask writing `0xFF` stamps over every field but its own -- including any a
/// caller held back, which `Partition`'s own doc warns about.
#[test]
fn no_mask_is_the_whole_byte() {
    for (tile, assignment) in &two_zooms().tiles {
        assert_ne!(
            assignment.read_mask, 0xFF,
            "z{} {},{} reads the whole byte",
            tile.z, tile.x, tile.y
        );
        assert_ne!(
            assignment.write_mask, 0xFF,
            "z{} {},{} writes the whole byte",
            tile.z, tile.x, tile.y
        );
        assert_ne!(assignment.value, 0, "zero is the cleared value, not a mask");
    }
}

/// Neither pipeline bakes a mask.
///
/// The baked values are zero, and zero is the safe direction: a dynamic mask that was never set
/// then compares no bits and the draw is clipped away, or writes no bits and the mask is missing.
/// `0xFF` baked would compare every bit and draw everywhere -- a layer with no clipping, which is
/// the failure that looks like working.
#[test]
fn the_pipelines_bake_no_mask() {
    use tessella_emblema::device::Attachment;

    for attachment in [Attachment::StencilOnly, Attachment::DepthStencil] {
        let content = pipelines::depth_stencil(attachment);
        for face in [content.front, content.back] {
            assert_eq!(face.compare_mask, 0, "a baked compare mask");
            assert_eq!(face.write_mask, 0, "content never writes the stencil");
        }
    }

    let mask = pipelines::depth_stencil_write();
    for face in [mask.front, mask.back] {
        assert_eq!(face.write_mask, 0, "a baked write mask");
        // The compare op is `ALWAYS`, so the compare mask is never read and is left at zero too.
        assert_eq!(face.compare_op, vk::CompareOp::ALWAYS);
    }
}

/// The content pipeline sets its compare mask per draw; the mask pipeline sets its write mask.
///
/// And neither sets the other's: a content draw that could set a write mask is a content draw that
/// could erase the tile mask it is testing against, which is the failure `depth_stencil`'s zero
/// write mask exists to make unreachable.
#[test]
fn each_pipeline_declares_only_the_masks_it_sets() {
    let content = pipelines::CONTENT_DYNAMIC;
    assert!(content.contains(&vk::DynamicState::STENCIL_COMPARE_MASK));
    assert!(
        !content.contains(&vk::DynamicState::STENCIL_WRITE_MASK),
        "a content draw must not be able to write the stencil"
    );
    assert!(
        !content.contains(&vk::DynamicState::STENCIL_REFERENCE),
        "a content draw's reference is set, but as part of the draw rather than declared here"
    );

    let mask = pipelines::MASK_DYNAMIC;
    assert!(mask.contains(&vk::DynamicState::STENCIL_WRITE_MASK));
    assert!(mask.contains(&vk::DynamicState::STENCIL_REFERENCE));

    // Both set the viewport and scissor, which is what lets one pipeline serve a ring of any size.
    for state in [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR] {
        assert!(content.contains(&state), "content misses {state:?}");
        assert!(mask.contains(&state), "the mask misses {state:?}");
    }
}
