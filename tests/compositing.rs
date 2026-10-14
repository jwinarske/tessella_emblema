// SPDX-License-Identifier: BSD-2-Clause
//! How each family composites, against mbgl's own `setColorMode` calls.
//!
//! Nothing on the wire says this. `grep -i color_mode crates/tessella-capture-abi/src` is empty,
//! and it is empty correctly: mbgl does not take a color mode from a style either, it writes one
//! per render layer in `src/mbgl/renderer/layers/`. So the consumer owns the fact, and owning it
//! means transcribing twenty-one entries by hand -- and deciding a twenty-second, the ground,
//! which has no render layer to read because mbgl has no terrain.
//!
//! Which is why the first test here is a second copy of the table. That is a weak check on its
//! own -- it catches an edit rather than a mistranscription -- and it is paired with one that is
//! not a copy: twenty of the twenty-two are derivable from whether the family draws onto the map
//! or into a target of its own, and that derivation agrees with the transcription. The two that are
//! left are the two the derivation names.
//!
//! What no test here can reach is whether the twenty-one mbgl rows were read off correctly in the
//! first place. The pixels are what settles that, and `benches/first_pixel.rs` draws the background's
//! two arms over an opaque clear: the same four bytes at an opacity of one, two different ones
//! faded.

use std::collections::BTreeSet;

use tessella_capture_abi::RenderPass;
use tessella_capture_abi::terrain_ubo::BUILTIN_TERRAIN_SHADER;
use tessella_emblema::families::{ALL, Composite, Family};
use tessella_emblema::pipelines::Blend;
use tessella_emblema::surface::Surface;

/// What mbgl's render layers call `setColorMode` with, read off `src/mbgl/renderer/layers/`.
///
/// One entry per family in [`ALL`], by name. The line numbers are from the oracle checkout at
/// `ad5e73527f3a`, and are a reading aid rather than part of the check -- a name is what this
/// agrees with the table on.
///
/// One row cites no layer, because mbgl has none to cite: the ground is tessella's own family. Its
/// mode is the sibling consumer's, `tessella_fluorite`'s `terrain.mat`, which declares
/// `blending: transparent` -- alpha over a premultiplied source, which is what `Blend::Alpha` is.
///
/// Four of mbgl's calls have no family here and so no row: the collision box's and the collision
/// circle's, which `families::UNDRAWN` lists, and the location indicator's two, which is a layer
/// with no builtin shader at all.
const MBGL: &[(&str, Composite)] = &[
    // render_background_layer.cpp:217, `drawPasses == RenderPass::Translucent`.
    ("background", Composite::ByPass),
    ("background_pattern", Composite::ByPass),
    // render_fill_layer.cpp:391 for the fill, 403 for both outlines, 480 for the pattern and 492
    // for its outline. The triangulated outline reuses the outline's own builder, so its mode is
    // the one set at 403.
    ("fill", Composite::Fixed(Blend::Alpha)),
    ("fill_outline", Composite::Fixed(Blend::Alpha)),
    ("fill_outline_triangulated", Composite::Fixed(Blend::Alpha)),
    ("fill_pattern", Composite::Fixed(Blend::Alpha)),
    ("fill_outline_pattern", Composite::Fixed(Blend::Alpha)),
    // render_fill_extrusion_layer.cpp:304 and :425, the second being the instanced path. The
    // patterned families take the same two lines: `hasPattern` chooses the shader at :159 and
    // :185 and nothing else, so one color builder serves both and its mode is alpha either way.
    ("fill_extrusion", Composite::Fixed(Blend::Alpha)),
    ("fill_extrusion_instanced", Composite::Fixed(Blend::Alpha)),
    ("fill_extrusion_pattern", Composite::Fixed(Blend::Alpha)),
    (
        "fill_extrusion_pattern_instanced",
        Composite::Fixed(Blend::Alpha),
    ),
    // render_line_layer.cpp:276, on the passes with the opaque bit masked out.
    ("line", Composite::ByPass),
    ("line_pattern", Composite::ByPass),
    // render_circle_layer.cpp:272.
    ("circle", Composite::Fixed(Blend::Alpha)),
    // render_raster_layer.cpp:154.
    ("raster", Composite::Fixed(Blend::Alpha)),
    // render_hillshade_layer.cpp:279 for the prepare pass and :372 for the draw.
    ("hillshade_prepare", Composite::Fixed(Blend::Unblended)),
    ("hillshade", Composite::Fixed(Blend::Alpha)),
    // render_color_relief_layer.cpp:379.
    ("color_relief", Composite::Fixed(Blend::Alpha)),
    // render_heatmap_layer.cpp:303 for the kernels and :368 for the texture that reads them.
    ("heatmap", Composite::Fixed(Blend::Additive)),
    ("heatmap_texture", Composite::Fixed(Blend::Alpha)),
    // render_symbol_layer.cpp:786, on the translucent bit.
    ("symbol_icon", Composite::ByPass),
    ("symbol_sdf", Composite::ByPass),
    ("symbol_text_and_icon", Composite::ByPass),
    // No render layer: the ground is tessella's. `terrain.mat`'s `blending: transparent`.
    ("terrain", Composite::Fixed(Blend::Alpha)),
];

/// Every family's mode is the one mbgl's layer sets, and every row names a family.
///
/// Both directions, so the two lists cannot drift apart in either: a family added to the table
/// without a row fails, and a row for a family that no longer exists fails too.
#[test]
fn every_family_composites_the_way_its_render_layer_does() {
    let rows: BTreeSet<&str> = MBGL.iter().map(|(name, _)| *name).collect();
    let families: BTreeSet<&str> = ALL.iter().map(|family| family.name).collect();
    assert_eq!(
        rows, families,
        "the rows here and the families in the table are not the same set"
    );
    assert_eq!(rows.len(), MBGL.len(), "a family is named twice here");

    for (name, mode) in MBGL {
        let family = ALL
            .iter()
            .find(|family| family.name == *name)
            .unwrap_or_else(|| panic!("no family named {name}"));
        assert_eq!(
            family.composite, *mode,
            "{name} composites {:?} and mbgl's layer sets {mode:?}",
            family.composite
        );
    }
}

/// Whether a family draws onto the map, which is what every surface but a bare plane means --
/// except for the family that *is* the map's surface.
///
/// Two families are drawn on a plane alone because they render into a target of their own, at
/// their own size, which is then sampled by a second family. That is the `surfaces: FLAT` the
/// table gives them.
///
/// The ground is a plane alone for the opposite reason, which is the third case this cannot
/// derive. It draws onto the map -- it is the backmost thing in it -- and it has one placement
/// because a ground is only ever drawn in a Mercator frame: `Map::surface` takes the globe where a
/// style asks for both, so the producer sends no ground at all on a sphere. Named by its shader
/// rather than by its name, because the id is the thing mbgl has no counterpart for.
fn onto_the_map(family: &Family) -> bool {
    family.surfaces != [Surface::Plane] || family.shader == BUILTIN_TERRAIN_SHADER
}

/// A family that draws onto the map alpha-blends, always or by pass. Nothing else can be right.
///
/// This is the half of the transcription that is derivable, and it covers twenty of the
/// twenty-two.
/// A draw onto the map lands on pixels some other layer has already written -- that is what a
/// painter order is -- so replacing them loses the layer underneath and adding to them brightens
/// it. Either is a mode that still draws, which is the whole reason to pin this at all.
///
/// The exceptions are exactly the two that draw into their own target, and the test asserts that
/// they are exactly those two rather than allowing any family to be one. `hillshade_prepare`
/// writes a field into a cleared image, where there is nothing underneath to lose, and `heatmap`
/// accumulates its kernels into one, where the accumulation *is* the output.
#[test]
fn only_a_family_with_its_own_target_does_not_alpha_blend() {
    let mut offscreen = BTreeSet::new();
    for family in ALL {
        let blends = matches!(
            family.composite,
            Composite::Fixed(Blend::Alpha) | Composite::ByPass
        );
        if onto_the_map(family) {
            assert!(
                blends,
                "{} draws onto the map and composites {:?}",
                family.name, family.composite
            );
        } else {
            assert!(
                !blends,
                "{} draws into its own target and alpha-blends, which composites over nothing",
                family.name
            );
            offscreen.insert(family.name);
        }
    }
    assert_eq!(
        offscreen,
        BTreeSet::from(["heatmap", "hillshade_prepare"]),
        "the families drawing into their own target are not the two this test knows"
    );
}

/// `ByPass` reads the translucent bit, and reads it as a bit.
///
/// `Opaque | Translucent` is the input the three spellings in mbgl disagree on -- the background's
/// equality gives `unblended` for it and this gives `Alpha` -- and the reason that is the same
/// pixel is in [`Composite::ByPass`]'s own doc. What matters here is that it is tested as a bit
/// and not compared: `pass == TRANSLUCENT` passes every other case in this test.
#[test]
fn the_pass_dependent_arm_is_chosen_by_the_translucent_bit() {
    let background = ALL
        .iter()
        .find(|family| family.name == "background")
        .expect("a background family");
    assert_eq!(background.composite, Composite::ByPass, "the fixture moved");

    assert_eq!(
        background.blend(RenderPass::TRANSLUCENT.bits()),
        Blend::Alpha
    );
    assert_eq!(
        background.blend((RenderPass::OPAQUE | RenderPass::TRANSLUCENT).bits()),
        Blend::Alpha,
        "the translucent bit decides whichever others are set with it"
    );
    assert_eq!(
        background.blend(RenderPass::OPAQUE.bits()),
        Blend::Unblended
    );
    assert_eq!(
        background.blend(RenderPass::PASS3D.bits()),
        Blend::Unblended,
        "a pass that is neither is not blended on the strength of not being opaque"
    );
    assert_eq!(
        background.blend(RenderPass::NONE.bits()),
        Blend::Unblended,
        "a drawable mbgl would not have drawn at all"
    );
}

/// A fixed mode is the same in every pass, which is what makes it fixed.
///
/// Every pass byte, including the ones no order entry carries: a mode that varied with a bit it
/// has no business reading would be a second pipeline for one family and one layer, and the cache
/// would hand it out by [`tessella_emblema::pipelines::Key`] without anything noticing.
#[test]
fn a_fixed_mode_does_not_vary_with_the_pass() {
    for family in ALL {
        let Composite::Fixed(fixed) = family.composite else {
            continue;
        };
        for pass in 0..=u8::MAX {
            assert_eq!(
                family.blend(pass),
                fixed,
                "{} composites {:?} in pass {pass:#04x} and {fixed:?} elsewhere",
                family.name,
                family.blend(pass)
            );
        }
    }
}
