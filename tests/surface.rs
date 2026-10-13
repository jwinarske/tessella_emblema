//! Does the surface's own block read the bytes the producer wrote, and does the placement read all
//! of it?
//!
//! The family blocks come from the ABI's generated tables and `tests/preamble.rs` checks all fifty
//! against those. A surface's blocks do not: mbgl has no globe and no terrain, so these three are
//! tessella's own, declared here as tables and checked here against the producer's own structs.
//! Nothing generates them, which is exactly why they are checked.

use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::ubo_layouts::{LAYOUTS, UboFieldKind};
use tessella_capture_abi::globe_ubo::GlobeBendUbo;
use tessella_capture_abi::terrain_ubo::TerrainDrawableUbo;
use tessella_emblema::families::family;
use tessella_emblema::preamble::{declare, offsets, type_name};
use tessella_emblema::slots;
use tessella_emblema::surface::{GLOBE_BEND_UBO, GLOBE_CAMERA_UBO, Surface, TERRAIN_DRAWABLE_UBO};

/// A fill on `surface`, which is the family that has every surface and the fewest blocks.
///
/// Through the family rather than from the tables, because the body reads its color and opacity
/// through the `paint_*` functions `Family::module` declares and they are the family's to name.
fn fill(surface: Surface) -> String {
    family(BuiltIn::FillShader)
        .expect("a fill is drawn")
        .module(surface, &[])
        .expect("a fill assembles on every surface")
}

/// The bend table is the producer's struct, field for field.
///
/// Seven rows of four floats and nothing else, so there is no padding to get wrong -- which is a
/// reason to check rather than a reason not to: the table is hand-written, and a row transposed or
/// an offset typed as 46 instead of 64 reads every later coefficient from the one before it. That
/// draws: the terms are all small floats and the result is a tile bent slightly wrong.
#[test]
fn the_bend_table_is_the_producers_struct() {
    let placed = offsets(&GLOBE_BEND_UBO).expect("declarable");
    let declared = [
        ("anchor", core::mem::offset_of!(GlobeBendUbo, anchor)),
        ("d_u", core::mem::offset_of!(GlobeBendUbo, d_u)),
        ("d_v", core::mem::offset_of!(GlobeBendUbo, d_v)),
        ("d_uu", core::mem::offset_of!(GlobeBendUbo, d_uu)),
        ("d_vv", core::mem::offset_of!(GlobeBendUbo, d_vv)),
        ("d_uv", core::mem::offset_of!(GlobeBendUbo, d_uv)),
        ("d_h", core::mem::offset_of!(GlobeBendUbo, d_h)),
    ];
    assert_eq!(placed.len(), declared.len(), "the table lost a row");
    for ((name, at), (want_name, want_at)) in placed.iter().zip(declared) {
        assert_eq!(*name, want_name, "the rows are in a different order");
        assert_eq!(
            *at as usize, want_at,
            "{name} would be read from {at}, written at {want_at}"
        );
    }
    assert_eq!(GLOBE_BEND_UBO.stride, GlobeBendUbo::STRIDE);
    assert_eq!(
        GLOBE_BEND_UBO.size as usize,
        core::mem::size_of::<GlobeBendUbo>()
    );
}

/// And so is the raise table.
///
/// This one has padding to get wrong: a `mat4x4<f32>` and four `vec4`s, where WGSL's own offsets
/// happen to agree with the struct's. "Happen to" is why the check is here.
#[test]
fn the_raise_table_is_the_producers_struct() {
    let placed = offsets(&TERRAIN_DRAWABLE_UBO).expect("declarable");
    let declared = [
        ("matrix", core::mem::offset_of!(TerrainDrawableUbo, matrix)),
        ("unpack", core::mem::offset_of!(TerrainDrawableUbo, unpack)),
        ("color", core::mem::offset_of!(TerrainDrawableUbo, color)),
        ("params", core::mem::offset_of!(TerrainDrawableUbo, params)),
        ("skirt", core::mem::offset_of!(TerrainDrawableUbo, skirt)),
    ];
    assert_eq!(placed.len(), declared.len(), "the table lost a field");
    for ((name, at), (want_name, want_at)) in placed.iter().zip(declared) {
        assert_eq!(*name, want_name, "the fields are in a different order");
        assert_eq!(
            *at as usize, want_at,
            "{name} would be read from {at}, written at {want_at}"
        );
    }
    assert_eq!(TERRAIN_DRAWABLE_UBO.stride, TerrainDrawableUbo::STRIDE);
    assert_eq!(
        TERRAIN_DRAWABLE_UBO.size as usize,
        core::mem::size_of::<TerrainDrawableUbo>()
    );
}

/// The globe matrix block holds as many elements as the camera sends.
///
/// The one surface block with no producer struct behind it: the matrix arrives on `CameraUpdate`
/// in `f64` and the consumer narrows it, so what is checked is the count. Taken from the gap
/// between that field and the next rather than from a literal, because a matrix that grew a row
/// would move the field after it.
#[test]
fn the_globe_matrix_block_is_one_camera_matrix() {
    use tessella_capture_abi::envelope::CameraUpdate;

    let span = core::mem::offset_of!(CameraUpdate, center_zoom0)
        - core::mem::offset_of!(CameraUpdate, globe_matrix);
    let elements = span / core::mem::size_of::<f64>();
    assert_eq!(
        elements, 16,
        "the camera's globe matrix is not four by four"
    );
    assert_eq!(
        GLOBE_CAMERA_UBO.size as usize,
        elements * core::mem::size_of::<f32>(),
        "the block does not hold the matrix the camera sends"
    );
    assert_eq!(GLOBE_CAMERA_UBO.fields.len(), 1);
    assert_eq!(GLOBE_CAMERA_UBO.fields[0].kind, UboFieldKind::Mat4);
}

/// No surface block's WGSL type name collides with a family block's.
///
/// Two structs of the same name in one module is a parse error, which would be caught -- but only
/// for the pair that happened to be assembled. Checked against every family block instead.
#[test]
fn no_surface_block_collides_with_a_family_block() {
    let family: Vec<String> = LAYOUTS.iter().map(|l| type_name(l.name)).collect();
    for surface in Surface::ALL {
        for block in surface.blocks() {
            let name = type_name(block.name);
            assert!(
                !family.contains(&name),
                "{name} is also a family block's type name"
            );
        }
    }
    // And the three do not collide with each other.
    let mut mine: Vec<String> = Surface::ALL
        .iter()
        .flat_map(|surface| surface.blocks().iter().map(|b| type_name(b.name)))
        .collect();
    let before = mine.len();
    mine.sort();
    mine.dedup();
    assert_eq!(
        mine.len(),
        before,
        "two surfaces declare the same type name"
    );
}

/// Every surface block declares, which is what the padding check would refuse.
#[test]
fn every_surface_block_declares() {
    for surface in Surface::ALL {
        for block in surface.blocks() {
            let source = declare(block, slots::stride(block))
                .unwrap_or_else(|why| panic!("{} cannot be declared: {why:?}", block.name));
            for field in block.fields {
                assert!(
                    source.contains(field.name),
                    "{}::{} is missing from the declaration",
                    block.name,
                    field.name
                );
            }
        }
    }
}

/// A surface's blocks bind after the family's, and its textures after those.
///
/// The order is the contract between this and the descriptor set the renderer will write. A family
/// with no surface block then has no gap in its set, and a surface's binding number does not
/// depend on which family it is attached to beyond the count.
#[test]
fn a_surface_binds_after_the_family() {
    let plane = fill(Surface::Plane);
    assert!(plane.contains("@binding(0) var<storage, read> fill_drawable_ubo"));
    assert!(plane.contains("@binding(1) var<storage, read> fill_evaluated_props_ubo"));
    assert!(
        !plane.contains("@binding(2)"),
        "a plane adds nothing:\n{plane}"
    );

    let raised = fill(Surface::Terrain);
    assert!(raised.contains("@binding(2) var<storage, read> terrain_drawable_ubo"));
    assert!(raised.contains("@binding(3) var elevation: texture_2d<f32>;"));
    assert!(raised.contains("@binding(4) var elevation_sampler: sampler;"));

    let bent = fill(Surface::Globe);
    assert!(bent.contains("@binding(2) var<storage, read> globe_camera_ubo"));
    assert!(!bent.contains("texture_2d"), "a bend samples nothing");
}

/// Every coefficient the producer sends is read by the bend that receives it.
///
/// A row declared and never read is a term of the expansion dropped, and dropping one does not
/// look like a bug: the tile is still bent, slightly wrong, in a way that reads as the quadratic
/// being a quadratic. Field-qualified rather than by name alone, because `matrix` and `color` are
/// a family's field names too.
#[test]
fn every_bend_coefficient_is_read() {
    let source = fill(Surface::GlobeAnchored);
    for row in GLOBE_BEND_UBO.fields {
        assert!(
            source.contains(&format!("bend.{}", row.name)),
            "the bend never reads {}",
            row.name
        );
    }
}

/// The raise reads the three fields that describe the DEM, and not the two the ground owns.
///
/// `matrix` and `color` travel in the same block because the block is one shape -- the producer
/// says so -- and a layer standing on the ground takes its matrix from its own family block and
/// paints itself. Named here so that a later reader finds a statement rather than an omission.
#[test]
fn the_raise_reads_the_dem_and_not_the_grounds_own_fields() {
    let source = fill(Surface::Terrain);
    for read in ["terrain.unpack", "terrain.params", "terrain.skirt"] {
        assert!(source.contains(read), "the raise never reads {read}");
    }
    for unread in ["terrain.matrix", "terrain.color"] {
        assert!(
            !source.contains(unread),
            "{unread} is the ground's, and this reads it"
        );
    }
}

/// The height a family carries reaches the surface that can lift it.
///
/// A fill passes zero and a fill extrusion will pass meters. A surface that drops the component
/// draws every building flat on the ground, which is a picture that looks deliberate.
#[test]
fn a_surface_that_can_lift_a_height_reads_one() {
    for surface in [Surface::Plane, Surface::GlobeAnchored, Surface::Terrain] {
        assert!(
            surface.placement().contains("position.z")
                || surface.placement().contains("transform(columns, position)"),
            "{surface:?} drops the height it was given"
        );
    }
    // And the direct bend does not, which is the one case that is a decision rather than a slip:
    // lifting along the sphere's normal needs the coefficient only the anchored bend carries.
    assert!(
        !Surface::Globe.placement().contains("position.z"),
        "the direct bend grew a height term; it has no coefficient for one"
    );
}

/// The direct bend keeps the three decisions the projection's own chain makes.
///
/// The chain is `tessella_tile::globe`'s and the arithmetic cannot be checked here -- a vertex
/// stage is not something this suite can run. What it can check is that the three steps which are
/// decisions rather than algebra survive an edit:
///
/// * the clamp, which lands a tile edge running off the top of the world on the pole instead of
///   letting the exponent diverge;
/// * the negated `y`, which is the convention everything downstream of `sphere_point` assumes and
///   which `clip_matrix` carries the compensating flip for -- so a shader that drops it draws the
///   southern hemisphere in the north;
/// * the depth offset added to clip `z` after the bend, which is how coincident layers separate on
///   a surface that has no per-drawable projection to bake one into.
///
/// Each of those is silent when wrong: the first diverges only at a pole, the second draws a
/// complete and inverted planet, the third draws a fill over its own outline at random.
#[test]
fn the_direct_bend_keeps_the_projections_decisions() {
    let placement = Surface::Globe.placement();
    for decision in ["clamp(merc.y, 0.0, 1.0)", "-sin(lat)", "clip.z += merc.z;"] {
        assert!(
            placement.contains(decision),
            "the bend no longer does `{decision}`:\n{placement}"
        );
    }
}

/// The bend is measured from the tile's center, and from the center this consumer agrees on.
///
/// The one number in a placement that the stream does not carry. The producer expands about the
/// tile's center and sends the coefficients; which coordinate that center is has to be the same on
/// both sides, and a shader measuring from the tile's corner instead reads every coefficient at
/// four thousand times its intended argument. So the convention is pinned here rather than left to
/// whoever next edits the expansion.
#[test]
fn the_bend_is_measured_from_the_tile_center() {
    let placement = Surface::GlobeAnchored.placement();
    assert!(
        placement.contains("const TILE_CENTER: f32 = 4096.0;"),
        "half of the 8192 coordinate range every tile is produced at"
    );
    assert!(
        placement.contains("position.xy - vec2<f32>(TILE_CENTER, TILE_CENTER)"),
        "the expansion's argument is not an offset from the center:\n{placement}"
    );
}

/// Every surface names its placement `place`, with the one signature a body is written against.
#[test]
fn every_surface_places_through_one_signature() {
    for surface in Surface::ALL {
        assert!(
            surface.placement().contains(
                "fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32>"
            ),
            "{surface:?} does not declare the signature bodies call"
        );
    }
}

/// A plane's suffix is empty and no two surfaces share one.
#[test]
fn the_suffixes_name_the_surfaces_apart() {
    assert_eq!(Surface::Plane.suffix(), "");
    let mut suffixes: Vec<&str> = Surface::ALL.iter().map(|s| s.suffix()).collect();
    let before = suffixes.len();
    suffixes.sort_unstable();
    suffixes.dedup();
    assert_eq!(suffixes.len(), before);
}

/// Every placement hands its clip position to `vulkan_depth`, and so does the clipping mask.
///
/// The producer's matrices are mbgl's, and mbgl's projection is OpenGL's: `mat4::perspective` sets
/// `out[10] = (far + near) / (near - far)` with `out[11] = -1`, which puts clip `z` in `-w..w`.
/// Vulkan's volume is `0..w`, so a placement that returns the matrix's own `z` throws away the near
/// half of every frustum -- and a drawable whose whole `z` is a sublayer nudge is in that half.
/// The producer's viewport quad is one: `DrawableEntry::for_viewport` has no camera in it at all,
/// so its clip `z` is `-depth_offset` and nothing else.
///
/// Measured rather than argued: the captured Berlin frame draws 538,201 of its 786,432 pixels as
/// the clear with any one placement left unmapped, and none of them with all of them mapped. The
/// mask is the same rule for a different reason -- a mask outside the volume is not drawn, and what
/// that loses is not the mask but every content draw it was meant to let through.
#[test]
fn every_placement_ends_in_the_vulkan_clip_volume() {
    assert!(
        tessella_emblema::shaders::PRELUDE.contains("fn vulkan_depth(clip: vec4<f32>)"),
        "the preamble does not define the remap every placement calls"
    );
    assert!(
        tessella_emblema::shaders::PRELUDE
            .contains("return vec4<f32>(clip.xy, (clip.z + clip.w) * 0.5, clip.w);"),
        "the remap is not the half-and-shift that takes -w..w to 0..w"
    );

    for surface in Surface::ALL {
        let placement = surface.placement();
        let place = placement
            .split("fn place(")
            .nth(1)
            .and_then(|rest| rest.split("\nfn ").next())
            .unwrap_or_else(|| panic!("{} has no placement", surface.suffix()));
        assert!(
            place.contains("vulkan_depth("),
            "{} returns a clip position it has not remapped",
            surface.suffix()
        );
    }

    // The direct bend's displacement is the one that must *not* remap again: it is the difference
    // of two placements, and the map is linear, so the difference is already in Vulkan's volume.
    let bent = Surface::Globe.placement();
    let displace = bent
        .split("fn displace(")
        .nth(1)
        .and_then(|rest| rest.split("\nfn ").next())
        .expect("a displacement");
    assert!(
        !displace.contains("vulkan_depth("),
        "the direct bend's displacement remaps a difference of two remapped places"
    );

    assert!(
        tessella_emblema::masks::BODY
            .contains("return vec4<f32>(clip.xy, (clip.z + clip.w) * 0.5, clip.w);"),
        "a clipping mask is drawn through the producer's matrix and must be remapped too"
    );
}
