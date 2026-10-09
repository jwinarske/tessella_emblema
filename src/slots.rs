// SPDX-License-Identifier: BSD-2-Clause
//! Which buffer a block is, as the producer names it.
//!
//! `UboUpdate` carries `(view, layer_index, slot, bytes)`, and `slot` is the **buffer's identity**
//! rather than an index inside one. The ABI says so where it reserves one:
//!
//! > `UboUpdate` already carries `(layer, slot, bytes)` for an arbitrary slot, so [...] slot is the
//! > only thing that has to be agreed, and it is `ID_GLOBE_BEND_UBO`.
//!
//! So a layer does not have *a* block buffer. It has one per slot: a fill layer's drawable array at
//! slot 2, its tile properties at 4 and its evaluated properties at 5, each arriving whole. This
//! module is what turns a family's declared block into the slot its bytes come in at, which is what
//! [`crate::descriptors`] needs to point a binding at the right buffer.
//!
//! # Derived, not tabulated
//!
//! `ubo_slots::SLOTS` is generated from mbgl's own chain of enums and pairs every slot name with its
//! value, and a block's slot is named after the block: `FillDrawableUBO` arrives at
//! `idFillDrawableUBO`. So the ordinary case is a lookup and not a table written here, which is the
//! point -- a table would be a second copy of mbgl's numbering to keep in step.
//!
//! What the lookup misses is the *variants*. mbgl gives a pattern fill its own struct and binds it at
//! the plain one's slot, which is what `ubo_layouts::UNIONS` records: `FillPatternDrawableUBO` is a
//! member of `FillDrawableUnionUBO`, and that union's slot is `idFillDrawableUBO`. Both halves are
//! generated, so the variants are derived too -- the union's name without `Union` is the slot's.
//!
//! That leaves three blocks this crate declares that mbgl does not, and one variant with no union.
//! Those four are the only pairings written by hand here.

use tessella_capture_abi::generated::ubo_layouts::{UNIONS, UboLayout};
use tessella_capture_abi::generated::ubo_slots::SLOTS;

use crate::surface::GLOBE_CAMERA_SLOT;

/// The slot a block's bytes arrive in, or `None` for a block with no slot agreed.
///
/// `None` is not reachable for a block any family or surface in this crate declares, which
/// `tests/block_slots.rs` checks family by family. It is an `Option` rather than a panic because the
/// argument is a layout and a caller could pass one from anywhere.
#[must_use]
pub fn of(layout: &UboLayout) -> Option<u32> {
    if let Some(found) = named(layout.name) {
        return Some(found);
    }
    // A variant: its own name is not a slot, but the union it belongs to names one.
    for union in &UNIONS {
        if union.members.contains(&layout.name)
            && let Some((head, tail)) = without_union(union.name)
            && let Some(found) = SLOTS
                .iter()
                .find(|(name, _)| joins(name.strip_prefix("id").unwrap_or(name), head, tail))
        {
            return Some(found.1);
        }
    }
    hand_agreed(layout.name)
}

/// The slot mbgl calls `id<name>`.
fn named(block: &str) -> Option<u32> {
    SLOTS
        .iter()
        .find(|(name, _)| name.strip_prefix("id") == Some(block))
        .map(|(_, slot)| *slot)
}

/// A union's name split either side of `Union`, which is the block name mbgl binds it at.
///
/// `FillDrawableUnionUBO` gives `("FillDrawable", "UBO")`, and the slot is `idFillDrawableUBO`.
/// Returned as two pieces rather than one joined string so this allocates nothing: it runs once per
/// descriptor set layout built.
fn without_union(union: &str) -> Option<(&str, &str)> {
    let at = union.find("Union")?;
    Some((&union[..at], &union[at + "Union".len()..]))
}

/// Whether `candidate` is exactly `head` followed by `tail`.
///
/// Equality in two pieces, which is all the length test is there for -- joining them first would
/// allocate a string per union per set layout built. No slot mbgl names today is a prefix-and-suffix
/// match without being an exact one, so this is exactness kept rather than a case being excluded.
fn joins(candidate: &str, head: &str, tail: &str) -> bool {
    candidate.len() == head.len() + tail.len()
        && candidate.starts_with(head)
        && candidate.ends_with(tail)
}

/// The four pairings neither mbgl's slots nor its unions give.
///
/// Three are blocks mbgl has no equivalent for, and each is agreed somewhere that is not this crate:
/// the bend's slot and the terrain block's slot are both in the ABI, beside the structs they belong
/// to, and the globe camera's is in [`crate::surface`] because that block never travels.
///
/// The fourth is `BackgroundPatternPropsUBO`. mbgl declares no properties *union* for a background,
/// so there is nothing generated to derive it from -- but the producer writes a plain background's
/// properties and a patterned one's at the same slot, and says why where it does:
///
/// > A background with a pattern writes a different block at the same slot: sixty-four bytes of
/// > corners, display sizes and the crossfade where a plain one writes thirty-two of color and
/// > opacity. The two are told apart by their size, which is why this slot is not a union the way a
/// > fill's is.
fn hand_agreed(block: &str) -> Option<u32> {
    match block {
        "BackgroundPatternPropsUBO" => {
            Some(tessella_capture_abi::generated::ubo_slots::ID_BACKGROUND_PROPS_UBO)
        }
        "GlobeBendUBO" => Some(tessella_capture_abi::globe_ubo::ID_GLOBE_BEND_UBO),
        "TerrainDrawableUBO" => Some(tessella_capture_abi::terrain_ubo::ID_TERRAIN_DRAWABLE_UBO),
        "GlobeCameraUBO" => Some(GLOBE_CAMERA_SLOT),
        _ => None,
    }
}

/// The stride the entries of a slot's buffer sit at.
///
/// The union's where the block is a member of one, and the block's own otherwise. A layer's drawable
/// buffer is an array of the *union* of its drawable blocks, which the producer says where it packs
/// one:
///
/// > A plain fill writes an 80-byte `FillDrawableUBO` into a 96-byte slot, because the pattern
/// > variants are larger and set the stride for everyone. Packing at 80 would put every entry after
/// > the first at the wrong offset -- a layer whose tiles are drawn with each other's matrices,
/// > which is plausible-looking output no size check would catch.
///
/// This is the reading half of that. `ubo_layouts::UNIONS` is generated from the same headers and
/// carries both the membership and the stride, so neither is written here.
///
/// Eight of the fifty blocks answer something other than their own stride: `BackgroundDrawableUBO`
/// (64 against 96), `FillDrawableUBO`, `FillOutlineDrawableUBO` and
/// `FillOutlineTriangulatedDrawableUBO` (80 against 96), `LineDrawableUBO`,
/// `LineGradientDrawableUBO` and `LinePatternDrawableUBO` (96 against 128), and
/// `LineSDFTilePropsUBO` (16 against 64). Five of them are declared by a family this crate draws:
/// `background`, `fill`, `fill_outline`, `line` and `line_pattern`.
#[must_use]
pub fn stride(layout: &UboLayout) -> u32 {
    UNIONS
        .iter()
        .find(|union| union.members.contains(&layout.name))
        .map_or(layout.stride, |union| union.stride)
}
