// SPDX-License-Identifier: BSD-2-Clause
//! Which stencil reference and compare mask a drawable draws with.
//!
//! # What this is for
//!
//! `record::content` sets these per drawable, and both failures are silent. A reference from the
//! wrong tile clips a drawable to another tile's shape -- geometry cut along an edge that is not
//! there. A compare mask that is wrong in the *other* direction clips nothing, which is a layer
//! drawn over its neighbors.
//!
//! The lookup is the whole of that decision, and it needs no device: the stores do, and this does
//! not, which is why `clipping` takes the joiner and the partition rather than the whole `Scene`.
//!
//! # What would be caught
//!
//! Unclipped meaning anything other than a compare mask of zero. `Partition`'s own words are "a
//! tile absent here has no mask; its geometry is left unclipped", and the way this expresses that is
//! `(0, 0)`: the stencil test is a baked `EQUAL`, so `(0 & 0) == (stencil & 0)` passes for every
//! texel whatever the buffer holds. A `0xFF` there would clip the drawable to wherever the buffer
//! happened to be zero.

use std::collections::{BTreeMap, BTreeSet};

use tessella_capture_abi::RenderPass;
use tessella_capture_abi::envelope::{
    DrawFlags, GeometryAdd, GeometryId, Span, TileId, ViewId, ViewUse,
};
use tessella_consume::join::{Announcement, Joiner};
use tessella_consume::stencil::{self, Partition};
use tessella_emblema::record;

const VIEW: ViewId = ViewId(1);

fn tile(z: u8, x: u32, y: u32) -> TileId {
    TileId {
        z,
        x,
        y,
        overscaled_z: z,
        wrap: 0,
    }
}

/// A geometry announced with nothing in it, which is all this lookup reads.
fn announced(geometry: GeometryId) -> Announcement {
    Announcement {
        add: GeometryAdd {
            geometry,
            permutation_key: 0,
            indexes: tessella_capture_abi::envelope::SlabRef::default(),
            vertex_count: 0,
            attrs: Span::default(),
            instance_attrs: Span::default(),
            segments: Span::default(),
            texture_refs: Span::default(),
            builtin_shader: 0,
            vertex_type: 0,
            reason: 0,
            topology: 0,
            _pad: [0; 1],
        },
        announced_at: 0,
        attrs: Vec::new(),
        instance_attrs: Vec::new(),
        segments: Vec::new(),
        texture_refs: Vec::new(),
    }
}

/// A use of that geometry by `VIEW`, optionally carrying a tile.
fn used(geometry: GeometryId, at: Option<TileId>) -> ViewUse {
    ViewUse {
        geometry,
        view: VIEW,
        layer_index: 0,
        sub_layer_index: 0,
        // When `has_tile` is zero the field is "meaningful only when `has_tile` is set", so it can
        // hold anything -- and a placeholder the partition *does* assign is what makes ignoring the
        // flag observable. A zeroed one would give the same answer as reading it properly, and the
        // test would pass either way.
        tile: at.unwrap_or(tile(15, 10, 10)),
        render_pass: RenderPass::NONE,
        draw_flags: DrawFlags::NONE,
        has_tile: u8::from(at.is_some()),
        _pad: [0; 5],
    }
}

/// A parent and its four children in one group, which is where the masks differ.
fn two_zooms() -> Partition {
    let mut tiles = BTreeSet::new();
    tiles.insert(tile(14, 5, 5));
    for (x, y) in [(10, 10), (11, 10), (10, 11), (11, 11)] {
        tiles.insert(tile(15, x, y));
    }
    let mut groups = BTreeMap::new();
    groups.insert(0i32, tiles.clone());
    stencil::partition(&tiles, &groups, stencil::ALL_BITS)
}

fn joined(geometry: GeometryId, at: Option<TileId>) -> Joiner {
    let mut joiner = Joiner::new();
    let _ = joiner.announce(announced(geometry));
    joiner.used(used(geometry, at));
    joiner
}

/// A drawable in a tile the partition assigned draws with that tile's value and read mask.
#[test]
fn a_masked_drawable_takes_its_tiles_assignment() {
    let partition = two_zooms();
    let child = tile(15, 10, 10);
    let assignment = partition.tiles.get(&child).copied().expect("assigned");
    let joiner = joined(GeometryId(1), Some(child));

    let (reference, compare) = record::clipping(&joiner, &partition, VIEW, GeometryId(1));
    assert_eq!(reference, u32::from(assignment.value));
    assert_eq!(
        compare,
        u32::from(assignment.read_mask),
        "the compare mask is the tile's own zoom's field"
    );
    assert_ne!(compare, 0, "a masked drawable is clipped");
    assert_ne!(compare, 0xFF, "and not by every bit in the byte");
}

/// Two tiles of one zoom give two references and the same mask.
///
/// Which is what makes a batch's drawables distinguishable: they share a field and differ in the
/// value within it, so the reference is what has to be set per drawable.
#[test]
fn two_tiles_differ_in_reference_and_not_in_mask() {
    let partition = two_zooms();
    let first = joined(GeometryId(1), Some(tile(15, 10, 10)));
    let second = joined(GeometryId(2), Some(tile(15, 11, 11)));

    let (one, mask_one) = record::clipping(&first, &partition, VIEW, GeometryId(1));
    let (two, mask_two) = record::clipping(&second, &partition, VIEW, GeometryId(2));

    assert_ne!(one, two, "two tiles need two references");
    assert_eq!(mask_one, mask_two, "one zoom is one field");
}

/// A use carrying no tile is unclipped, which is a compare mask of zero.
#[test]
fn a_use_without_a_tile_is_unclipped() {
    let partition = two_zooms();
    let joiner = joined(GeometryId(1), None);
    // The placeholder in that use is a tile the partition assigns, so a lookup that ignored
    // `has_tile` would come back with its assignment rather than nothing.
    assert!(partition.tiles.contains_key(&tile(15, 10, 10)));
    assert_eq!(
        record::clipping(&joiner, &partition, VIEW, GeometryId(1)),
        (0, 0),
        "no tile means no mask, and no mask is a compare mask of zero"
    );
}

/// A tile the partition did not assign is unclipped too.
///
/// `Partition`'s own words: "a tile absent here has no mask; its geometry is left unclipped". It
/// happens for a tile outside the layer group the partition was built from.
#[test]
fn a_tile_outside_the_partition_is_unclipped() {
    let partition = two_zooms();
    let elsewhere = tile(14, 999, 999);
    assert!(
        !partition.tiles.contains_key(&elsewhere),
        "the fixture must not assign this one"
    );
    let joiner = joined(GeometryId(1), Some(elsewhere));
    assert_eq!(
        record::clipping(&joiner, &partition, VIEW, GeometryId(1)),
        (0, 0)
    );
}

/// A geometry the joiner has no use for in this view is unclipped.
///
/// The third way to reach `(0, 0)`. A geometry can be announced and used by another view, or used
/// and not announced -- `Joiner::drawable` wants both halves and this view's.
#[test]
fn a_geometry_this_view_does_not_use_is_unclipped() {
    let partition = two_zooms();
    let joiner = joined(GeometryId(1), Some(tile(15, 10, 10)));

    // Announced and used, but by view 1 -- asked about view 2.
    assert_eq!(
        record::clipping(&joiner, &partition, ViewId(2), GeometryId(1)),
        (0, 0),
        "another view's use is not this view's"
    );
    // And a geometry nothing announced.
    assert_eq!(
        record::clipping(&joiner, &partition, VIEW, GeometryId(404)),
        (0, 0)
    );
}
