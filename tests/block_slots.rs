// SPDX-License-Identifier: BSD-2-Clause
//! Which buffer each family's blocks arrive in.
//!
//! # What this is for
//!
//! `UboUpdate::slot` is the buffer's identity, not an index inside one: a fill layer's drawables
//! arrive at slot 2, its tile properties at 4 and its evaluated properties at 5. So a family's
//! bindings read *different buffers*, and [`tessella_emblema::slots::of`] is what says which.
//!
//! # What would be caught
//!
//! A block with no slot, which is a binding [`tessella_emblema::descriptors`] cannot point
//! anywhere. And two of a family's blocks resolving to one slot, which is the same defect one step
//! further on -- both bindings would read the same bytes, and the second block's fields would be
//! read out of the first's.
//!
//! The derivation itself is checked where it can be: every block mbgl names directly must resolve
//! to exactly the slot mbgl names it at, and every variant to its union's. Those two leave only the
//! four hand-agreed pairings unchecked, which is the point of deriving the rest.

use tessella_capture_abi::generated::ubo_layouts::{UNIONS, UboLayout};
use tessella_capture_abi::generated::ubo_slots::SLOTS;
use tessella_emblema::families::ALL;
use tessella_emblema::slots;

/// Every block of every family and surface has a slot.
#[test]
fn every_declared_block_has_a_slot() {
    for family in ALL {
        for surface in family.surfaces {
            for block in family.blocks.iter().chain(surface.blocks()) {
                assert!(
                    slots::of(block).is_some(),
                    "{} on {surface:?} declares {} and no slot carries it",
                    family.name,
                    block.name
                );
            }
        }
    }
}

/// A family's blocks are in distinct buffers.
///
/// The defect this whole module exists for. Two blocks at one slot is one buffer read as two
/// structs: the second binding gets the first's bytes, and a shader reading a color out of a matrix
/// draws rather than fails.
#[test]
fn a_familys_blocks_are_in_distinct_buffers() {
    for family in ALL {
        for surface in family.surfaces {
            let mut seen: Vec<(u32, &str)> = Vec::new();
            for block in family.blocks.iter().chain(surface.blocks()) {
                let slot = slots::of(block).expect("a slot");
                if let Some((_, other)) = seen.iter().find(|(had, _)| *had == slot) {
                    panic!(
                        "{} on {surface:?} reads {} and {other} from slot {slot}",
                        family.name, block.name
                    );
                }
                seen.push((slot, block.name));
            }
        }
    }
}

/// A block mbgl names resolves to the slot mbgl names it at.
///
/// Over the generated table rather than over this crate's families, so it covers the blocks no
/// family here declares yet as well.
#[test]
fn a_block_mbgl_names_resolves_to_mbgls_slot() {
    let mut checked = 0;
    for (name, slot) in &SLOTS {
        let Some(block) = name.strip_prefix("id") else {
            continue;
        };
        // The slot names that are not blocks: the counts, the start ids, the textures and the
        // vertex attributes. A block is what some `UboLayout` is called.
        let Some(layout) = layout_named(block) else {
            continue;
        };
        assert_eq!(
            slots::of(layout),
            Some(*slot),
            "{block} arrives at {slot} and resolved elsewhere"
        );
        checked += 1;
    }
    assert!(
        checked > 20,
        "only {checked} blocks were checked, so the lookup found almost nothing"
    );
}

/// A variant resolves to its union's slot.
///
/// mbgl gives a pattern fill its own struct and binds it where the plain one binds, which is what a
/// union records. A variant resolving to a slot of its own would be a buffer the producer never
/// writes.
#[test]
fn a_variant_resolves_to_its_unions_slot() {
    let mut checked = 0;
    for union in &UNIONS {
        let base = union
            .members
            .iter()
            .filter_map(|member| layout_named(member))
            .find_map(slots::of)
            .unwrap_or_else(|| panic!("no member of {} resolved", union.name));
        for member in union.members {
            let Some(layout) = layout_named(member) else {
                continue;
            };
            assert_eq!(
                slots::of(layout),
                Some(base),
                "{member} is a {} and resolved away from it",
                union.name
            );
            checked += 1;
        }
    }
    assert!(checked > 10, "only {checked} variants were checked");
}

/// The three blocks this crate declares that mbgl does not are past mbgl's own slots.
///
/// Each is agreed elsewhere -- two in the ABI and one in `surface` -- and what is checkable here is
/// that none of them lands on a slot mbgl already uses. A collision would be a block the producer
/// overwrites with another.
#[test]
fn the_blocks_mbgl_has_no_slot_for_sit_past_its_slots() {
    let mbgls: Vec<u32> = SLOTS
        .iter()
        .filter(|(name, _)| name.starts_with("id") && layout_named(&name[2..]).is_some())
        .map(|(_, slot)| *slot)
        .collect();
    let highest = mbgls.iter().copied().max().expect("mbgl has slots");

    for block in [
        &tessella_emblema::surface::GLOBE_BEND_UBO,
        &tessella_emblema::surface::TERRAIN_DRAWABLE_UBO,
        &tessella_emblema::surface::GLOBE_CAMERA_UBO,
    ] {
        let slot = slots::of(block).expect("a slot");
        assert!(
            slot > highest,
            "{} sits at {slot}, which is not past mbgl's {highest}",
            block.name
        );
    }
}

/// Every block the generated layouts know, by the name mbgl spells it.
fn layout_named(name: &str) -> Option<&'static UboLayout> {
    tessella_capture_abi::generated::ubo_layouts::LAYOUTS
        .iter()
        .find(|layout| layout.name == name)
}
