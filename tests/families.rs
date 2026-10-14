// SPDX-License-Identifier: BSD-2-Clause
//! The family table against the shader enum it is keyed by.
//!
//! What these pin is coverage rather than behavior: which of mbgl's shaders this crate draws, and
//! that every one it claims assembles. A family added to the table without being removed from
//! `UNDRAWN` fails here, and so does the reverse, so the two cannot both be wrong in the same
//! direction.

use std::collections::BTreeSet;

use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_emblema::families::{ALL, UNDRAWN, family, for_wire};

#[test]
fn every_family_is_keyed_by_a_distinct_shader() {
    let mut seen = BTreeSet::new();
    for entry in ALL {
        assert!(
            seen.insert(entry.shader),
            "{:?} appears twice in the table",
            entry.shader
        );
    }
    assert_eq!(seen.len(), ALL.len());
}

#[test]
fn every_family_has_a_distinct_name() {
    let mut seen = BTreeSet::new();
    for entry in ALL {
        assert!(seen.insert(entry.name), "{} appears twice", entry.name);
    }
}

/// The table and the undrawn list partition the enum. Neither may grow without the other
/// shrinking.
#[test]
fn drawn_and_undrawn_cover_the_enum_exactly() {
    let drawn: BTreeSet<i32> = ALL.iter().map(|entry| entry.shader).collect();
    let undrawn: BTreeSet<i32> = UNDRAWN.iter().map(|(shader, _)| *shader as i32).collect();
    assert!(
        drawn.is_disjoint(&undrawn),
        "a shader is both drawn and undrawn: {:?}",
        drawn.intersection(&undrawn).collect::<Vec<_>>()
    );

    // Walk the enum by discriminant rather than listing it: a variant added upstream then belongs
    // to neither set and is named here instead of passing unnoticed.
    let mut missing = Vec::new();
    for repr in 0..256i32 {
        let Some(shader) = BuiltIn::from_repr(repr) else {
            continue;
        };
        if !drawn.contains(&repr) && !undrawn.contains(&repr) {
            missing.push(shader);
        }
    }
    assert!(
        missing.is_empty(),
        "these shaders are in neither table: {missing:?}"
    );
}

#[test]
fn every_undrawn_shader_says_why() {
    for (shader, why) in UNDRAWN {
        assert!(
            why.len() > 20,
            "{shader:?} is listed undrawn with no reason: {why:?}"
        );
        assert!(
            family(*shader).is_none(),
            "{shader:?} is listed undrawn and resolves anyway"
        );
    }
}

#[test]
fn the_wire_resolves_a_drawn_shader_and_skips_the_rest() {
    for entry in ALL {
        let got = for_wire(entry.shader).expect("a drawn shader resolves");
        assert_eq!(got.shader, entry.shader);
        assert_eq!(got.name, entry.name);
    }
    for (shader, _) in UNDRAWN {
        assert!(
            for_wire(*shader as i32).is_none(),
            "{shader:?} is undrawn and must not resolve"
        );
    }
    // A discriminant that is not a shader at all is skipped the same way, because an order may
    // name a family a build does not have.
    assert!(for_wire(-1).is_none());
    assert!(for_wire(9999).is_none());
}

/// Every family claims at least one surface, and the modules themselves are compiled by
/// `tests/shaders.rs`.
///
/// Not checked here by calling `module`: that only concatenates text and returns `Ok` for a body
/// naming a binding nothing declared, so an assembly check that stopped there would pass a table
/// entry missing a block. `every_family_on_every_surface_compiles` takes all sixty-six pairs
/// through naga, which is what actually catches it.
#[test]
fn every_family_claims_a_surface() {
    for entry in ALL {
        assert!(
            !entry.surfaces.is_empty(),
            "{} claims no surface",
            entry.name
        );
    }
}

/// A family's blocks are what its body names, so an empty list is always wrong.
#[test]
fn every_family_declares_at_least_one_block() {
    for entry in ALL {
        assert!(
            !entry.blocks.is_empty(),
            "{} declares no uniform block",
            entry.name
        );
    }
}

/// The families that hand `place` a height are the ones that leave the surface.
///
/// Both halves of a building, twice over: the roof and the walls standing under it, plain and
/// patterned. Nothing else here has a third component to give, and a family that gained one
/// without needing it would be asking a surface for a term it has no use for -- which on the
/// direct bend does not exist at all, so these four are also the only families that skip that
/// surface.
#[test]
fn only_the_extrusions_have_height() {
    let raised: Vec<&str> = ALL
        .iter()
        .filter(|entry| entry.height)
        .map(|entry| entry.name)
        .collect();
    assert_eq!(
        raised,
        vec![
            "fill_extrusion",
            "fill_extrusion_instanced",
            "fill_extrusion_pattern",
            "fill_extrusion_pattern_instanced",
        ]
    );
}

/// And the only families drawn per instance are the walls, plain and patterned.
///
/// Pinned because an empty `instance_attributes` is what the other twenty rely on: it decides
/// whether `ubo_index` rides `firstInstance` or a push constant, whether the module declares a
/// second descriptor group, and whether a geometry's buffer is created with a second usage. A
/// family that gained the field by accident would move all three at once.
#[test]
fn only_the_walls_are_drawn_per_instance() {
    let instanced: Vec<&str> = ALL
        .iter()
        .filter(|entry| !entry.instance_attributes.is_empty())
        .map(|entry| entry.name)
        .collect();
    assert_eq!(
        instanced,
        vec![
            "fill_extrusion_instanced",
            "fill_extrusion_pattern_instanced"
        ]
    );

    // And every one of their instance attributes that names a shared buffer names the same one,
    // which is what makes each module's single group-one binding right. Two for the plain walls
    // and two for the patterned ones: the base, the height and the two rectangles are the
    // drawable's own buffers, not the shared outline.
    for name in instanced {
        let walls = ALL
            .iter()
            .find(|entry| entry.name == name)
            .expect("it was just found");
        let shared: Vec<&str> = walls
            .instance_attributes
            .iter()
            .filter_map(|attribute| attribute.shared_buffer)
            .collect();
        assert_eq!(shared, vec!["idFillExtrusionInstanced"; 2], "{name}");
    }
}

/// A family is found by the id the wire carries, and the table holds one mbgl has no name for.
///
/// `Family::shader` was a `BuiltIn` and is the `i32` the wire carries. For the twenty-three families
/// mbgl
/// declares that is the same set of values -- every one of them is an mbgl shader and this checks
/// it -- and what it buys is the one that is not: mbgl has no terrain, so the ground's id is
/// tessella's own `BUILTIN_TERRAIN_SHADER`, which is 128 and past the enum's thirty-six.
/// `BuiltIn::from_repr` answers `None` for it, so a lookup that decoded first could never find it
/// however the table was written.
///
/// Both directions: the ground is found by the wire's own id and is the only family mbgl has no
/// shader for, and every other family is found by its id *and* by its own `BuiltIn`. An entry that
/// quietly took an mbgl shader's id would be found twice and draw as that family.
#[test]
fn a_family_is_found_by_the_id_the_wire_carries() {
    use tessella_capture_abi::terrain_ubo::BUILTIN_TERRAIN_SHADER;

    let mut not_mbgls = Vec::new();
    for entry in ALL {
        let found = for_wire(entry.shader).unwrap_or_else(|| {
            panic!(
                "{} is in the table and {} does not find it",
                entry.name, "for_wire"
            )
        });
        assert_eq!(found.name, entry.name, "two families share an id");
        let Some(decoded) = BuiltIn::from_repr(entry.shader) else {
            not_mbgls.push(entry.name);
            continue;
        };
        assert_eq!(
            family(decoded).map(|found| found.name),
            Some(entry.name),
            "{} is not found by its own `BuiltIn`",
            entry.name
        );
    }

    assert!(
        BuiltIn::from_repr(BUILTIN_TERRAIN_SHADER).is_none(),
        "the ground's id is an mbgl shader after all, which would make it a `BuiltIn` family"
    );
    assert_eq!(
        for_wire(BUILTIN_TERRAIN_SHADER).map(|found| found.name),
        Some("terrain"),
        "the ground is in the table and the wire's own id does not find it"
    );
    assert_eq!(
        not_mbgls,
        vec!["terrain"],
        "the ground is the one family mbgl has no shader for"
    );
}
