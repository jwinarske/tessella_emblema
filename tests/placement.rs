// SPDX-License-Identifier: BSD-2-Clause
//! Which surface a drawable is placed on, from the camera and the drawable's own flags.
//!
//! Two decisions, taken in two places because they rest on different things. [`Surface::of`] reads
//! the wire -- the view's `ProjectionMode`, the drawable's `ON_TERRAIN` bit, and whether its layer
//! was sent a bend block -- and names the surface the *producer* asked for. `Family::surface` then
//! narrows that to one this crate has a module for, because only sixty-five of the eighty
//! (family, surface) pairs exist.
//!
//! Neither decision is observable in a pixel on its own: a surface shows up as *where* geometry
//! lands, and `benches/first_pixel.rs` is where that is read, on a device, with the case's
//! placement reached through these two calls rather than named. What is here is the rule itself,
//! against the producer's own statements of it -- `Map::surface` for which of a globe and a raised
//! ground wins, and `DrawFlags::ON_TERRAIN` for what a family with no raised module draws instead.

use tessella_capture_abi::ProjectionMode;
use tessella_capture_abi::envelope::DrawFlags;
use tessella_emblema::families::{ALL, Family};
use tessella_emblema::surface::Surface;

/// A family by name, for the cases that are about one in particular.
fn named(name: &str) -> &'static Family {
    ALL.iter()
        .find(|family| family.name == name)
        .unwrap_or_else(|| panic!("{name} is not a family here"))
}

/// Every flags byte a drawable can carry, which is what the one bit has to be read out of.
fn every_flags() -> impl Iterator<Item = u8> {
    0..=u8::MAX
}

/// The three inputs name the four surfaces, each by the one thing that distinguishes it.
#[test]
fn the_wire_names_the_four_surfaces() {
    let raised = DrawFlags::ON_TERRAIN.bits();
    assert_eq!(
        Surface::of(ProjectionMode::Mercator, 0, false),
        Surface::Plane
    );
    assert_eq!(
        Surface::of(ProjectionMode::Mercator, raised, false),
        Surface::Terrain
    );
    assert_eq!(
        Surface::of(ProjectionMode::Globe, 0, false),
        Surface::Globe,
        "a globe with no bend block is the direct bend"
    );
    assert_eq!(
        Surface::of(ProjectionMode::Globe, 0, true),
        Surface::GlobeAnchored,
        "the block the producer writes per tile is the whole of the anchored bend's selection"
    );
    // And the bend block says nothing on a plane, where there is no bend to anchor: a layer whose
    // block is still held from the frame before the projection changed must not move its geometry.
    assert_eq!(
        Surface::of(ProjectionMode::Mercator, 0, true),
        Surface::Plane
    );
    assert_eq!(
        Surface::of(ProjectionMode::Mercator, raised, true),
        Surface::Terrain
    );
}

/// A globe wins over a raised ground, which is the producer's own resolution.
///
/// The two inputs are independent -- a projection is what a caller picks and a raised ground
/// follows from the style -- so they can both say yes, and `Map::surface` takes the globe
/// "because this build has no terrain on one". A consumer taking the other one would place
/// geometry through the terrain's matrix, which on a globe frame does not reach clip space: the
/// bend is the vertex stage's and `proj_matrix` stops at normalized Mercator.
///
/// Over every byte, not only the raised one, so the rule is the projection's and not a special
/// case for a value.
#[test]
fn a_globe_wins_over_a_raised_ground() {
    for flags in every_flags() {
        for bend in [false, true] {
            let surface = Surface::of(ProjectionMode::Globe, flags, bend);
            assert_ne!(
                surface,
                Surface::Terrain,
                "flags {flags:#010b} raised a globe frame"
            );
            assert_eq!(
                surface,
                if bend {
                    Surface::GlobeAnchored
                } else {
                    Surface::Globe
                },
                "flags {flags:#010b} changed which bend a globe draws"
            );
        }
    }
}

/// On a plane, one bit of the flags decides and the other seven do not.
///
/// `draw_flags` is the whole byte -- `batch::Key` carries it whole because every bit of it is
/// pipeline state -- and exactly one of those bits is a surface. A rule testing the byte rather
/// than the bit raises every drawable that is depth-tested, which is almost all of them.
#[test]
fn only_the_terrain_bit_raises_a_drawable() {
    for flags in every_flags() {
        let raised = flags & DrawFlags::ON_TERRAIN.bits() != 0;
        assert_eq!(
            Surface::of(ProjectionMode::Mercator, flags, false),
            if raised {
                Surface::Terrain
            } else {
                Surface::Plane
            },
            "flags {flags:#010b}"
        );
    }
}

/// Whatever the wire asks for, a family is drawn on a surface it declares a module for.
///
/// The producer marks a drawable raised wherever a ground covers its tile, whatever family draws
/// it, and sends a bend block for every layer whose geometry is a tile's. So a family is asked for
/// pairs that are not among the sixty-five, and what it must never do is be handed one: a module
/// is assembled per pair, and a pair with no module is a pipeline built against bindings nothing
/// wrote.
#[test]
fn every_family_is_placed_on_a_surface_it_declares() {
    for family in ALL {
        for projection in [ProjectionMode::Mercator, ProjectionMode::Globe] {
            for flags in every_flags() {
                for bend in [false, true] {
                    let asked = Surface::of(projection, flags, bend);
                    let drawn = family.surface(asked);
                    assert!(
                        family.surfaces.contains(&drawn),
                        "{} asked for {asked:?} and is drawn on {drawn:?}, which it has no module \
                         for",
                        family.name
                    );
                }
            }
        }
    }
}

/// A family that has the surface asked for is drawn on it, and not on the one below it.
///
/// The other half of the test above, and the one that keeps it from being satisfied by a narrowing
/// that always answers `Plane` -- which would pass every assertion there and draw a globe flat.
#[test]
fn a_declared_surface_is_never_narrowed() {
    for family in ALL {
        for surface in family.surfaces {
            assert_eq!(
                family.surface(*surface),
                *surface,
                "{} declares {surface:?} and is drawn elsewhere",
                family.name
            );
        }
    }
}

/// The fallback chain is finite, ends at a plane, and a plane is a surface every family has.
///
/// `Family::surface` walks it until the family declares what it reaches, so a chain that cycled
/// would hang and one ending anywhere else would end somewhere a family may not declare. The
/// second half is what makes the walk's last step safe, and it is a property of the table rather
/// than of the chain, so it is asserted rather than assumed.
#[test]
fn the_fallback_chain_ends_at_a_plane() {
    for start in Surface::ALL {
        let mut at = start;
        // Four surfaces, so a chain longer than four has revisited one.
        for _ in 0..Surface::ALL.len() {
            match at.fallback() {
                Some(next) => {
                    assert_ne!(next, at, "{at:?} falls back to itself");
                    at = next;
                }
                None => break,
            }
        }
        assert_eq!(
            at,
            Surface::Plane,
            "the chain from {start:?} ends at {at:?}, which not every family declares"
        );
        assert_eq!(Surface::Plane.fallback(), None, "a plane is the last step");
    }

    for family in ALL {
        assert!(
            family.surfaces.contains(&Surface::Plane),
            "{} has no plane to fall back to",
            family.name
        );
    }
}

/// What each family with a missing module draws instead, named one by one.
///
/// The three above say the narrowing is *consistent*; this says what it costs, because every one
/// of these is a picture rather than an error:
///
/// * `color_relief` has no anchored bend, so a globe draws it through the direct one. That is the
///   same sphere by the arithmetic every zoom below the crossover uses.
/// * The two offscreen families and the one that composites their target are drawn on a plane
///   whatever the camera does, which is not a loss: a target of one's own has no camera.
/// * A background is drawn on a plane where its drawable is raised, because it covers the viewport
///   rather than a tile and is never raised in the first place.
///
/// The extrusions used to be the fourth and are not any more: they had no raised module, so a
/// terrain style drew its buildings on the Mercator plane while the ground under them lifted. That
/// was a gap rather than a decision, and the pair exists now -- so what this names is what is left,
/// and every one of them is a decision.
#[test]
fn a_missing_module_falls_where_the_producer_says() {
    let raised = Surface::of(
        ProjectionMode::Mercator,
        DrawFlags::ON_TERRAIN.bits(),
        false,
    );
    // And the extrusions, from the other side: a raised drawable of one is drawn raised.
    for extrusion in ["fill_extrusion", "fill_extrusion_instanced"] {
        assert_eq!(
            named(extrusion).surface(raised),
            Surface::Terrain,
            "{extrusion} stands on the plane under raised ground"
        );
    }

    let anchored = Surface::of(ProjectionMode::Globe, 0, true);
    assert_eq!(named("color_relief").surface(anchored), Surface::Globe);

    let bent = Surface::of(ProjectionMode::Globe, 0, false);
    for offscreen in ["heatmap", "hillshade_prepare"] {
        assert_eq!(named(offscreen).surface(bent), Surface::Plane);
        assert_eq!(named(offscreen).surface(raised), Surface::Plane);
        assert_eq!(named(offscreen).surface(anchored), Surface::Plane);
    }
    // A background is never raised -- it covers the viewport rather than a tile -- but it does
    // declare the direct bend, so a chain stepping from a terrain through the bend instead of
    // straight to the plane would draw one on a sphere in a Mercator frame.
    assert_eq!(named("background").surface(raised), Surface::Plane);
    assert_eq!(named("heatmap_texture").surface(raised), Surface::Plane);
    assert_eq!(
        named("heatmap_texture").surface(bent),
        Surface::Globe,
        "the composite of the target does follow the camera"
    );
}
