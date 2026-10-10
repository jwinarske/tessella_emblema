//! Every texture binding carries the slot its `TextureRef` names it by.
//!
//! `descriptors::bound_from` places an image by matching its `TextureRef::slot` against the
//! binding's, so a binding with no slot cannot be filled and a wrong one is filled from the wrong
//! atlas. The slots come from two places -- a family's from the generated table, a surface's
//! hand-agreed beside its names -- and this is where the second is held to the first's standard.
//!
//! `tests/block_slots.rs` is the same check for the blocks.

use tessella_capture_abi::generated::texture_slots;
use tessella_emblema::families;
use tessella_emblema::pipelines::{self, Kind};
use tessella_emblema::surface::{Surface, TERRAIN_ELEVATION_SLOT};

/// A surface's names and slots are parallel arrays, so their lengths must agree.
///
/// The one thing a parallel array needs and the compiler does not give: adding a texture to a
/// surface without adding its slot would leave `bindings` short a pair, which is a set whose later
/// bindings are all one place out.
#[test]
fn a_surface_names_as_many_textures_as_it_slots() {
    for surface in [
        Surface::Plane,
        Surface::Globe,
        Surface::GlobeAnchored,
        Surface::Terrain,
    ] {
        assert_eq!(
            surface.textures().len(),
            surface.texture_slots().len(),
            "{surface:?} names {:?} and slots {:?}",
            surface.textures(),
            surface.texture_slots()
        );
    }
}

/// Terrain's elevation slot is the one the producer sends.
///
/// Hand-agreed, because the producer holds it as a private constant and publishes no table for a
/// surface's textures. Nothing can derive it, so what is checked is that it is not one of mbgl's
/// own -- which is the property its note on the producer's side claims: "past every slot mbgl's own
/// families use, so a terrain variant's second texture cannot land on one the flat variant already
/// reads."
#[test]
fn the_elevation_slot_is_past_every_family_slot() {
    let mut highest = None;
    for family in families::ALL {
        for texture in family.textures {
            assert_ne!(
                texture.binding, TERRAIN_ELEVATION_SLOT,
                "{} reads slot {} and so would its terrain variant's elevation",
                family.name, texture.binding
            );
            highest =
                Some(highest.map_or(texture.binding, |so_far: u32| so_far.max(texture.binding)));
        }
    }
    let highest = highest.expect("some family samples something");
    assert!(
        TERRAIN_ELEVATION_SLOT > highest,
        "the elevation is slot {TERRAIN_ELEVATION_SLOT} and some family already reads {highest}"
    );
}

/// Every texture binding of every set carries a slot, and the pairs agree.
///
/// A set's image and sampler bindings come two per texture and one `TextureRef` fills both, so the
/// two must name the same slot -- otherwise an image and the sampler reading it would be placed
/// from different refs.
#[test]
fn every_texture_binding_carries_its_slot() {
    for family in families::ALL {
        for surface in family.surfaces {
            let bindings = pipelines::bindings(family, *surface);
            let textures: Vec<_> = bindings
                .iter()
                .filter(|b| b.kind == Kind::SampledImage || b.kind == Kind::Sampler)
                .collect();
            let expected: Vec<u32> = family
                .textures
                .iter()
                .map(|texture| texture.binding)
                .chain(surface.texture_slots().iter().copied())
                .collect();
            assert_eq!(
                textures.len(),
                expected.len() * 2,
                "{} on {surface:?}",
                family.name
            );
            for (pair, slot) in textures.chunks(2).zip(&expected) {
                assert_eq!(pair[0].kind, Kind::SampledImage);
                assert_eq!(pair[1].kind, Kind::Sampler);
                assert_eq!(
                    pair[0].slot,
                    Some(*slot),
                    "{} on {surface:?}: the image at binding {}",
                    family.name,
                    pair[0].binding
                );
                assert_eq!(
                    pair[1].slot,
                    Some(*slot),
                    "{} on {surface:?}: the sampler at binding {} reads a different slot than its \
                     image",
                    family.name,
                    pair[1].binding
                );
            }
        }
    }
}

/// A family's slots are the generated table's, not this crate's idea of them.
///
/// The producer builds a run from the same table -- `texture_slots::textures(shader)` zipped with
/// the textures bound -- so this is the two halves reading one source rather than agreeing twice.
#[test]
fn a_familys_slots_are_the_generated_tables() {
    for family in families::ALL {
        let table = texture_slots::textures(family.shader);
        assert_eq!(
            family.textures.len(),
            table.len(),
            "{} declares {} textures and the table has {}",
            family.name,
            family.textures.len(),
            table.len()
        );
        for (mine, theirs) in family.textures.iter().zip(table) {
            assert_eq!(mine, theirs, "{}", family.name);
        }
    }
}
