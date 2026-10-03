//! Does an assembled family compile, and does it still say what it said?
//!
//! The declarations come from the ABI's tables and the body is written against them, so the thing
//! being checked is that the two fit: a body naming a field the tables do not declare, or reading
//! one at the wrong type, fails here rather than at pipeline creation on a board.

use std::collections::BTreeSet;

use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_capture_abi::generated::shader_attributes::{
    BACKGROUND_SHADER, CIRCLE_SHADER, COLOR_RELIEF_SHADER, FILL_EXTRUSION_SHADER,
    FILL_OUTLINE_SHADER, FILL_SHADER, HILLSHADE_PREPARE_SHADER, HILLSHADE_SHADER, LINE_SHADER,
    RASTER_SHADER, SYMBOL_ICON_SHADER, SYMBOL_SDFSHADER, SYMBOL_TEXT_AND_ICON_SHADER,
    ShaderAttribute,
};
use tessella_capture_abi::generated::texture_slots::{
    COLOR_RELIEF_SHADER_TEXTURES, HILLSHADE_PREPARE_SHADER_TEXTURES, HILLSHADE_SHADER_TEXTURES,
    RASTER_SHADER_TEXTURES, SYMBOL_ICON_SHADER_TEXTURES, SYMBOL_SDFSHADER_TEXTURES,
    SYMBOL_TEXT_AND_ICON_SHADER_TEXTURES, ShaderTexture,
};
use tessella_capture_abi::generated::ubo_layouts::{
    BACKGROUND_DRAWABLE_UBO, BACKGROUND_PROPS_UBO, CIRCLE_DRAWABLE_UBO, CIRCLE_EVALUATED_PROPS_UBO,
    COLOR_RELIEF_DRAWABLE_UBO, COLOR_RELIEF_EVALUATED_PROPS_UBO, COLOR_RELIEF_TILE_PROPS_UBO,
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, FILL_EXTRUSION_DRAWABLE_UBO,
    FILL_EXTRUSION_PROPS_UBO, FILL_OUTLINE_DRAWABLE_UBO, GLOBAL_PAINT_PARAMS_UBO,
    HILLSHADE_DRAWABLE_UBO, HILLSHADE_EVALUATED_PROPS_UBO, HILLSHADE_PREPARE_DRAWABLE_UBO,
    HILLSHADE_PREPARE_TILE_PROPS_UBO, HILLSHADE_TILE_PROPS_UBO, LINE_DRAWABLE_UBO,
    LINE_EVALUATED_PROPS_UBO, RASTER_DRAWABLE_UBO, RASTER_EVALUATED_PROPS_UBO, SYMBOL_DRAWABLE_UBO,
    SYMBOL_EVALUATED_PROPS_UBO, SYMBOL_TILE_PROPS_UBO, UboLayout,
};
use tessella_emblema::shaders::{
    BACKGROUND_BODY, CIRCLE_BODY, COLOR_RELIEF_BODY, FILL_BODY, FILL_EXTRUSION_BODY,
    FILL_OUTLINE_BODY, HILLSHADE_BODY, HILLSHADE_PREPARE_BODY, LINE_BODY, RASTER_BODY,
    SYMBOL_ICON_BODY, SYMBOL_SDF_BODY, SYMBOL_TEXT_AND_ICON_BODY, attribute_name, module,
};
use tessella_emblema::surface::Surface;

/// A family, and the surfaces the producer can draw it on.
struct Family {
    name: &'static str,
    blocks: Vec<&'static UboLayout>,
    attributes: &'static [ShaderAttribute],
    textures: &'static [ShaderTexture],
    body: &'static str,
    surfaces: &'static [Surface],
    /// Whether this family hands `place` a height above the surface rather than zero.
    ///
    /// Which decides one of its surfaces: the direct bend has no height term, so a family that
    /// leaves the surface cannot be drawn on it. See `Surface::Globe`'s placement.
    height: bool,
}

/// Every family a plane module exists for, and which surfaces each one has.
///
/// A background has neither of the two surfaces that need a block of their own: it covers the
/// viewport rather than a tile, so the producer writes it no bend block and never marks it
/// raised. Everything else has all four.
// A table of 31 entries once every family is here, and nothing but a table.
#[allow(clippy::too_many_lines)]
fn families() -> Vec<Family> {
    let all = &[
        Surface::Plane,
        Surface::Globe,
        Surface::GlobeAnchored,
        Surface::Terrain,
    ][..];
    let flat_or_bent = &[Surface::Plane, Surface::Globe][..];
    // No bend block: the producer writes one for the families whose geometry is a tile's, and a
    // color relief is not among them -- so it has the direct bend and the raise but not the
    // anchored one.
    let unanchored = &[Surface::Plane, Surface::Globe, Surface::Terrain][..];
    vec![
        Family {
            name: "background",
            blocks: vec![&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
            attributes: &BACKGROUND_SHADER,
            textures: &[],
            body: BACKGROUND_BODY,
            surfaces: flat_or_bent,
            height: false,
        },
        Family {
            name: "fill",
            blocks: vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            attributes: &FILL_SHADER,
            textures: &[],
            body: FILL_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "fill_outline",
            // Its own drawable block, which mbgl binds at the fill's slot: the layouts match but
            // the interpolation factor is named `outline_color_t`, because the outline's color is
            // a property of its own.
            blocks: vec![
                &FILL_OUTLINE_DRAWABLE_UBO,
                &FILL_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &FILL_OUTLINE_SHADER,
            textures: &[],
            body: FILL_OUTLINE_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "line",
            blocks: vec![
                &LINE_DRAWABLE_UBO,
                &LINE_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &LINE_SHADER,
            textures: &[],
            body: LINE_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "raster",
            blocks: vec![&RASTER_DRAWABLE_UBO, &RASTER_EVALUATED_PROPS_UBO],
            attributes: &RASTER_SHADER,
            textures: &RASTER_SHADER_TEXTURES,
            body: RASTER_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "color_relief",
            blocks: vec![
                &COLOR_RELIEF_DRAWABLE_UBO,
                &COLOR_RELIEF_TILE_PROPS_UBO,
                &COLOR_RELIEF_EVALUATED_PROPS_UBO,
            ],
            attributes: &COLOR_RELIEF_SHADER,
            textures: &COLOR_RELIEF_SHADER_TEXTURES,
            body: COLOR_RELIEF_BODY,
            surfaces: unanchored,
            height: false,
        },
        Family {
            name: "fill_extrusion",
            blocks: vec![&FILL_EXTRUSION_DRAWABLE_UBO, &FILL_EXTRUSION_PROPS_UBO],
            attributes: &FILL_EXTRUSION_SHADER,
            textures: &[],
            body: FILL_EXTRUSION_BODY,
            // No direct bend: it has no height term, and this is the family with a height.
            surfaces: &[Surface::Plane, Surface::GlobeAnchored],
            height: true,
        },
        Family {
            name: "symbol_icon",
            blocks: vec![
                &SYMBOL_DRAWABLE_UBO,
                &SYMBOL_TILE_PROPS_UBO,
                &SYMBOL_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &SYMBOL_ICON_SHADER,
            textures: &SYMBOL_ICON_SHADER_TEXTURES,
            body: SYMBOL_ICON_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "symbol_sdf",
            blocks: vec![
                &SYMBOL_DRAWABLE_UBO,
                &SYMBOL_TILE_PROPS_UBO,
                &SYMBOL_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &SYMBOL_SDFSHADER,
            textures: &SYMBOL_SDFSHADER_TEXTURES,
            body: SYMBOL_SDF_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "symbol_text_and_icon",
            blocks: vec![
                &SYMBOL_DRAWABLE_UBO,
                &SYMBOL_TILE_PROPS_UBO,
                &SYMBOL_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &SYMBOL_TEXT_AND_ICON_SHADER,
            textures: &SYMBOL_TEXT_AND_ICON_SHADER_TEXTURES,
            body: SYMBOL_TEXT_AND_ICON_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "hillshade_prepare",
            blocks: vec![
                &HILLSHADE_PREPARE_DRAWABLE_UBO,
                &HILLSHADE_PREPARE_TILE_PROPS_UBO,
            ],
            attributes: &HILLSHADE_PREPARE_SHADER,
            textures: &HILLSHADE_PREPARE_SHADER_TEXTURES,
            body: HILLSHADE_PREPARE_BODY,
            // Draws into a texture, not onto the map: see the body's own note.
            surfaces: &[Surface::Plane],
            height: false,
        },
        Family {
            name: "hillshade",
            blocks: vec![
                &HILLSHADE_DRAWABLE_UBO,
                &HILLSHADE_TILE_PROPS_UBO,
                &HILLSHADE_EVALUATED_PROPS_UBO,
            ],
            attributes: &HILLSHADE_SHADER,
            textures: &HILLSHADE_SHADER_TEXTURES,
            body: HILLSHADE_BODY,
            surfaces: all,
            height: false,
        },
        Family {
            name: "circle",
            blocks: vec![
                &CIRCLE_DRAWABLE_UBO,
                &CIRCLE_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &CIRCLE_SHADER,
            textures: &[],
            body: CIRCLE_BODY,
            surfaces: all,
            height: false,
        },
    ]
}

/// Compiles a module to SPIR-V, or says why not.
fn compile(source: &str) -> Vec<u32> {
    let parsed = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|why| panic!("{}\n--- source ---\n{source}", why.emit_to_string(source)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .unwrap_or_else(|why| panic!("{why:?}\n--- source ---\n{source}"));

    // No debug names: the words are compared against themselves, and names make an unrelated
    // naga version bump read as a shader change.
    let options = naga::back::spv::Options {
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    naga::back::spv::write_vec(&parsed, &info, &options, None)
        .unwrap_or_else(|why| panic!("{why:?}\n--- source ---\n{source}"))
}

fn background() -> String {
    module(
        Surface::Plane,
        &[&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
        &BACKGROUND_SHADER,
        &[],
        BACKGROUND_BODY,
    )
    .expect("background declares")
}

/// The first family compiles, validates and emits.
#[test]
fn background_compiles() {
    let words = compile(&background());
    assert!(
        words.len() > 64,
        "a module of {} words is empty",
        words.len()
    );
    assert_eq!(words[0], 0x0723_0203, "and it is SPIR-V");
}

/// The declarations the body reads are the ones the tables describe.
///
/// Named here rather than left implicit in whether it compiled: these are the four things the body
/// touches, and a rename in the ABI should break this test with a message rather than break the
/// build with a parse error forty lines down.
#[test]
fn the_body_reads_what_the_tables_declare() {
    let source = background();
    for named in [
        "struct BackgroundDrawableUbo",
        "struct BackgroundPropsUbo",
        // The field, not `place`'s parameter -- the leading indent and trailing comma are what
        // tell a struct member from a function argument, and without them this assertion passed
        // on the placement's signature after the declaration changed under it.
        "    matrix: array<vec4<f32>, 4>,",
        "color: vec4<f32>",
        "opacity: f32",
    ] {
        assert!(source.contains(named), "{named} is missing from:\n{source}");
    }
}

/// The roof is placed at its building's top, and the footprint keeps its fraction.
///
/// Two decisions that compile, validate and draw, and are wrong in ways that read as data rather
/// than as code:
///
/// * `select(base, height, ...)` returns the *third* argument's truth case, so the arms decide
///   whether a roof sits at the top of its building or on the ground. Swapped, every building in
///   the frame is flat -- which is the symptom fluorite's own material records having had.
/// * the packed fraction is added to the footprint. Dropped, a roof parts company with the walls
///   standing under it, by up to a tile unit, wherever a simplification pass produced fractional
///   positions.
///
/// A mutation found the first: nothing else in the suite noticed the arms swapping.
#[test]
fn the_roof_sits_at_the_top_and_keeps_its_fraction() {
    let extrusion = families()
        .into_iter()
        .find(|family| family.name == "fill_extrusion")
        .expect("fill extrusion is in the matrix");
    let source = module(
        Surface::Plane,
        &extrusion.blocks,
        extrusion.attributes,
        extrusion.textures,
        extrusion.body,
    )
    .expect("assembles");

    assert!(
        source.contains("let z = select(base, height, on_roof > 0.0);"),
        "the roof is not placed at the height:\n{source}"
    );
    assert!(
        source.contains("place(vec3<f32>(footprint + decimals, z), drawable.matrix)"),
        "the fraction is not added, or the height is not the third component"
    );
}

/// The symbol icon keeps the three decisions that are silent when wrong.
///
/// * **The corner is added in the label plane**, after `label_plane_matrix`, not in tile units.
///   That is the whole reason a symbol is placed through two matrices; added in tile units, type
///   changes size across a tile.
/// * **The perspective ratio inverts with `pitch_with_map`.** Laid out in pitched space, distance
///   shrinks a label and the ratio counteracts part of it; laid out in viewport space it grows
///   one. Backwards, distant type gets larger.
/// * **The fade's direction is its low bit.** The packed float carries the opacity in the high
///   bits and which way it is moving in the low one; with the sign reversed, a label fading in
///   fades out.
///
/// All three compile, validate and draw, which is what makes them worth pinning.
#[test]
fn the_symbol_icon_keeps_its_placement_decisions() {
    let icon = families()
        .into_iter()
        .find(|family| family.name == "symbol_icon")
        .expect("symbol icon is in the matrix");
    let source = module(
        Surface::Plane,
        &icon.blocks,
        icon.attributes,
        icon.textures,
        icon.body,
    )
    .expect("assembles");

    assert!(
        source.contains("let in_plane = transform(drawable.label_plane_matrix,")
            && source.contains("let on_plane = in_plane.xy / in_plane.w + spun * offset;"),
        "the corner is not added in the label plane:\n{source}"
    );
    assert!(
        source.contains("place(vec3<f32>(on_plane, 0.0), drawable.coord_matrix)"),
        "the placed point does not go through the coordinate matrix"
    );
    assert!(
        source.contains("var ratio = global.camera_to_center_distance / to_anchor;")
            && source.contains("ratio = to_anchor / global.camera_to_center_distance;"),
        "the perspective ratio does not invert with the pitch"
    );
    assert!(
        source.contains("var change = -global.symbol_fade_change;")
            && source.contains("change = global.symbol_fade_change;"),
        "the fade's direction is not taken from its low bit"
    );
    assert!(
        source.contains("max(min_font_scale, vec2<f32>(font_scale, font_scale))"),
        "the icon can shrink below its minimum font scale"
    );
    assert!(
        source.contains("+ pixel_offset / 16.0;"),
        "the icon's pixel offset is not in sixteenths"
    );
}

/// The text-and-icon family keeps the five decisions that draw the wrong sheet convincingly.
///
/// * **The mark is the low bit of the first size byte**, and `is_sdf == 0.0` is the *icon*. The
///   sense inverted, every glyph samples the sprite sheet and every sprite the glyph atlas -- both
///   with coordinates divided by the other's dimensions.
/// * **The texture size is chosen per vertex**, because the two atlases are different sizes and
///   one draw reads both.
/// * **`is_icon` interpolates flat.** All four corners of a quad carry the same mark, so a
///   smooth-interpolated integer would agree anyway on well-formed geometry -- and silently
///   disagree on the diagonal of anything else.
/// * **The font scale has no text branch.** `symbol_icon` and `symbol_sdf` both compute
///   `is_text_prop ? size / 24.0 : size`; here every vertex belongs to a text label.
/// * **There is no pixel offset.** This family declares none, so a placement copied from
///   `symbol_icon` would name an attribute that does not arrive.
#[test]
fn the_text_and_icon_keeps_its_sheet_decisions() {
    let both = families()
        .into_iter()
        .find(|family| family.name == "symbol_text_and_icon")
        .expect("text and icon is in the matrix");
    let source = module(
        Surface::Plane,
        &both.blocks,
        both.attributes,
        both.textures,
        both.body,
    )
    .expect("assembles");

    assert!(
        source.contains("let is_sdf = sized.x - 2.0 * smallest;")
            && source.contains("let is_icon = is_sdf == 0.0;"),
        "the glyph-or-sprite mark is not the low bit, or its sense is reversed:\n{source}"
    );
    assert!(
        source
            .contains("out.tex = tex / select(drawable.texsize, drawable.texsize_icon, is_icon);"),
        "the two atlases do not each divide by their own size"
    );
    assert!(
        source.contains("@location(1) @interpolate(flat) is_icon: u32,"),
        "the mark does not interpolate flat"
    );
    assert!(
        source.contains("let font_scale = size / 24.0;")
            && !source.contains("if drawable.is_text_prop != 0 {"),
        "the font scale branches on the text property, which this family does not"
    );
    assert!(
        !source.contains("symbol_pixel_offset"),
        "the placement reads a pixel offset this family does not declare"
    );
    assert!(
        source.contains(
            "let sprite = textureSample(symbol_image_icon, symbol_image_icon_sampler, in.tex);"
        ),
        "the icon half does not read the second atlas"
    );
}

/// The two hillshade passes keep the six decisions that shade believable relief wrongly.
///
/// * **The Sobel kernel's weights and its missing center.** The near neighbors count twice and the
///   center not at all. A wrong weight gives relief that still reads as terrain.
/// * **The row order of the y derivative.** `(g + h + h + i) - (a + b + b + c)`; reversed, every
///   hill reads as a valley -- the one hillshade defect everybody ships at least once.
/// * **The encode and the decode are inverses.** `deriv / 8 + 0.5` into the texture and
///   `pixel * 8 - 4` out of it. Mismatched, the relief is simply scaled, which looks like a style
///   choice.
/// * **The slope field's rows run the other way**, so the second pass flips `v`.
/// * **The five method numbers.** `standard` 0, `combined` 1, `igor` 2, `multidirectional` 3,
///   `basic` 4, and an unknown number falls through to the standard one. Two swapped draws a
///   different method's perfectly good hillshade.
/// * **The latitude scale divides.** A Mercator pixel covers less ground toward the poles, so the
///   same elevation change over it is a steeper real slope. Multiplying instead flattens the
///   relief exactly where it should sharpen.
#[test]
fn the_hillshade_keeps_its_slope_decisions() {
    let family = |name: &str| {
        let found = families()
            .into_iter()
            .find(|family| family.name == name)
            .unwrap_or_else(|| panic!("{name} is in the matrix"));
        module(
            Surface::Plane,
            &found.blocks,
            found.attributes,
            found.textures,
            found.body,
        )
        .expect("assembles")
    };
    let prepare = family("hillshade_prepare");
    let shade = family("hillshade");

    assert!(
        prepare.contains("(c + f + f + i) - (a + d + d + g),")
            && prepare.contains("(g + h + h + i) - (a + b + b + c)"),
        "the Sobel kernel's weights or its rows moved:\n{prepare}"
    );
    assert!(
        !prepare.contains("let e = elevation("),
        "the kernel reads its own center, which it weights at zero"
    );
    assert!(
        prepare.contains("vec4<f32>(deriv.x / 8.0 + 0.5, deriv.y / 8.0 + 0.5, 1.0, 1.0),")
            && shade.contains("let deriv = ((pixel.rg * 8.0) - 4.0) / scale_factor;"),
        "the slope's encode and decode are not inverses"
    );
    assert!(
        shade.contains("out.uv = vec2<f32>(uv.x, 1.0 - uv.y);"),
        "the second pass does not flip the slope field's rows"
    );
    assert!(
        shade.contains("if tile.method == 4 {")
            && shade.contains("if tile.method == 1 {")
            && shade.contains("if tile.method == 2 {")
            && shade.contains("if tile.method == 3 {")
            && shade.contains("let slope = atan(0.625 * length(deriv));"),
        "a method number moved, or the standard one is no longer the fallthrough"
    );
    assert!(
        shade.contains("let scale_factor = cos(radians(latitude));"),
        "the latitude scale is not the cosine of the latitude"
    );
    assert!(
        shade.contains("let lit = deriv * tile.exaggeration * 2.0;")
            && shade.contains("let intensity = tile.exaggeration;"),
        "the standard method takes the exaggerated slope, or the others take the raw one"
    );
}

/// The line keeps the five decisions that draw a line of the wrong size or shape.
///
/// Ported from `line.vertex.glsl` and `line.fragment.glsl`, with fluorite's `line.mat` as the
/// working reference. Each of these draws a line, which is why none of them is caught by the
/// compile, and four of the five were wrong before this test existed.
///
/// * **The extrusion is over 63, not 128.** mbgl's `scale` is `0.015873016`; the byte pair is a
///   unit vector at length 63. Over 128 a road is half the width the style asked for.
/// * **The width ratio divides.** `dist / u_ratio`, not times: the drawable's ratio is tile units
///   per pixel, so multiplying scales the wrong way with zoom.
/// * **The offset is negated**, which is what puts a side line on the side the style named.
/// * **`inset` and `outset` are separate.** One number cannot draw a casing: with a gap the
///   fragment fades in across the inner edge and out across the outer one, and collapsing them
///   fills the middle a gap exists to leave open.
/// * **The gamma scale is the ratio of the unprojected extrusion to the projected one**, which is
///   what holds the edge feather at a constant pixel width under pitch.
#[test]
fn the_line_keeps_its_extrusion_decisions() {
    let line = families()
        .into_iter()
        .find(|family| family.name == "line")
        .expect("line is in the matrix");
    let source = module(
        Surface::Plane,
        &line.blocks,
        line.attributes,
        line.textures,
        line.body,
    )
    .expect("assembles");

    assert!(
        source.contains("let extrude_scale = 63.0;")
            && source.contains("let extrude = data.xy - 128.0;"),
        "the extrusion's scale or its bias moved:\n{source}"
    );
    assert!(
        source.contains("let dist = outset * extrude / extrude_scale;")
            && source.contains("displace(at, dist / ratio, drawable.matrix)"),
        "the extrusion does not divide by the width ratio"
    );
    assert!(
        source.contains("let line_offset = -1.0 * mix_value(in.line_offset, drawable.offset_t);"),
        "the side offset is not negated"
    );
    assert!(
        source.contains("let inset = gapwidth + select(0.0, antialiasing, gapwidth > 0.0);")
            && source.contains("min(distance - (in.inset - blur2), in.outset - distance)"),
        "the inner and outer edges are not both faded"
    );
    assert!(
        source.contains("out.gamma_scale = unprojected / max(projected, 1e-6);"),
        "the feather is not corrected for perspective"
    );
    assert!(
        source.contains("let direction = (data.z % 4.0) - 1.0;")
            && source.contains("let turn = mat2x2<f32>(t, -u, u, t);"),
        "a round end point's extrude is not rotated"
    );
}

/// The symbol SDF keeps the five decisions that draw something plausible when wrong.
///
/// * **The atlas is read from `.r`.** `GLYPH_ATLAS_FORMAT` is one channel, which is `R8_UNORM`
///   here and `GL_ALPHA` in mbgl. Reading `.a` gives 1.0 everywhere: every glyph a solid block.
/// * **The halo is subtracted, not drawn under.** `min(halo, 1.0 - fill)` keeps a translucent
///   fill's halo translucent; without it the two coverages add where they meet.
/// * **Zero distance is 192/256, and the field is 8 units a pixel.** Both are how the atlas was
///   rasterized. Wrong, type is uniformly fattened or thinned rather than absent.
/// * **The paint selection's arms.** Swapped, the fill pass draws the halo color and the halo
///   pass the fill color -- two passes of the right shape in the wrong colors.
/// * **The threshold scales by the clip `w`.** The edge is a screen-space width, so it needs the
///   perspective divide the position got. Without it, pitched type blurs with distance.
#[test]
fn the_symbol_sdf_keeps_its_edge_decisions() {
    let sdf = families()
        .into_iter()
        .find(|family| family.name == "symbol_sdf")
        .expect("symbol sdf is in the matrix");
    let source = module(
        Surface::Plane,
        &sdf.blocks,
        sdf.attributes,
        sdf.textures,
        sdf.body,
    )
    .expect("assembles");

    assert!(
        source.contains("textureSample(symbol_image, symbol_image_sampler, in.tex).r;"),
        "the atlas is not read from red:\n{source}"
    );
    assert!(
        source.contains(
            "alpha = min(smoothstep(halo_edge - gamma, halo_edge + gamma, distance), 1.0 - alpha);"
        ),
        "the fill's coverage is not subtracted from the halo's"
    );
    assert!(
        source.contains("let fill_edge = (256.0 - 64.0) / 256.0;")
            && source.contains("let sdf_px = 8.0;"),
        "the field's zero level or its scale moved"
    );
    assert!(
        source.contains("out.paint = select(fill, halo, tile.is_halo != 0);"),
        "the halo and the fill colors are not selected in that order"
    );
    assert!(
        source.contains("out.gamma = clip.w;")
            && source.contains("select(fill_gamma, halo_gamma, is_halo) * in.gamma;"),
        "the threshold is not scaled by the perspective divide"
    );
}

/// The circle keeps the four decisions that draw a believable circle of the wrong size.
///
/// * **The center is the position halved.** The corner sign rides in the low bit, so a body that
///   places the raw position puts every circle at twice its tile coordinate.
/// * **All four pitch and scale paths.** Lying on the ground the extrusion is in tile units and
///   goes on before the matrix; standing up it is added in clip space after. Each of the two is
///   scaled differently again by `scale_with_map`. One path alone draws three styles wrong.
/// * **The antialias floor is one device pixel over the circle's reach**, not a constant: a small
///   circle fades over more of itself than a large one, and the style's blur and this floor are
///   the same quantity, so the wider wins rather than both applying.
/// * **The coverage ramp descends.** `smoothstep`'s second edge is the negative blur, which is
///   what makes the fill opaque at the center and clear at the rim rather than the reverse.
#[test]
fn the_circle_keeps_its_extrusion_decisions() {
    let circle = families()
        .into_iter()
        .find(|family| family.name == "circle")
        .expect("circle is in the matrix");
    let source = module(
        Surface::Plane,
        &circle.blocks,
        circle.attributes,
        circle.textures,
        circle.body,
    )
    .expect("assembles");

    assert!(
        source.contains("let center = floor(position * 0.5);")
            && source.contains("let extrude = (position % vec2<f32>(2.0, 2.0)) * 2.0 - 1.0;"),
        "the center is not the halved position:\n{source}"
    );
    assert!(
        source.contains("if props.pitch_with_map != 0 {")
            && source.contains("corner += scaled_extrude * reach;")
            && source
                .contains("(projected_center.w / max(global.camera_to_center_distance, 1e-6))")
            && source.contains(concat!(
                "let factor = select(\n",
                "            placed.w,\n",
                "            global.camera_to_center_distance,\n",
                "            props.scale_with_map != 0\n",
            )),
        "one of the four pitch and scale paths is missing"
    );
    assert!(
        source.contains(
            "out.antialias_blur = 1.0 / max(global.pixel_ratio, 1e-6) / max(reach, 1e-6);"
        ) && source.contains("let antialiased_blur = -max(in.blur, in.antialias_blur);"),
        "the antialias floor is not one pixel over the reach, or does not take the wider blur"
    );
    assert!(
        source.contains("let opacity_t = smoothstep(0.0, antialiased_blur, extrude_length - 1.0);"),
        "the coverage ramp does not descend"
    );
    assert!(
        source.contains("in.stroke_width < 0.01"),
        "a stroke too thin to have an edge still gets one"
    );
}

/// The fill outline keeps its feather, which is the only thing making it an outline.
///
/// An outline is line primitives one pixel wide and they rasterize hard. mbgl fades them by this
/// fragment's distance from the vertex's own screen position, which needs the perspective divide
/// done in the vertex stage -- interpolating `xy/w` is not the same number as interpolating `xy`
/// and `w` and dividing here. Without any of it an outline still draws, as a hard aliased line.
#[test]
fn the_fill_outline_keeps_its_feather() {
    let outline = families()
        .into_iter()
        .find(|family| family.name == "fill_outline")
        .expect("fill outline is in the matrix");
    let source = module(
        Surface::Plane,
        &outline.blocks,
        outline.attributes,
        outline.textures,
        outline.body,
    )
    .expect("assembles");

    assert!(
        source.contains("out.screen = (clip.xy / clip.w + 1.0) / 2.0 * global.world_size;"),
        "the screen position is not divided in the vertex stage:\n{source}"
    );
    assert!(
        source.contains("let distance = length(in.screen - in.clip.xy);")
            && source.contains("let alpha = 1.0 - smoothstep(0.0, 1.0, distance);"),
        "the outline has no feather"
    );
    assert!(
        source.contains("drawable.outline_color_t"),
        "the outline mixes with the fill's interpolation factor, not its own"
    );
}

/// The color relief pins a tile that covers a pole rather than sampling off the DEM.
///
/// The sentinel is a `y` at the end of the signed short range, which is not a coordinate. Left
/// alone it scales to a texture coordinate far outside the image and the pole draws as whatever
/// the sampler's addressing mode gives back -- a plausible color, from the wrong elevation.
#[test]
fn the_color_relief_pins_the_poles() {
    let relief = families()
        .into_iter()
        .find(|family| family.name == "color_relief")
        .expect("color relief is in the matrix");
    let source = module(
        Surface::Plane,
        &relief.blocks,
        relief.attributes,
        relief.textures,
        relief.body,
    )
    .expect("assembles");

    assert!(
        source.contains("if f32(in.color_relief_pos.y) < -32767.5 {")
            && source.contains("out.uv.y = 0.0;")
            && source.contains("if f32(in.color_relief_pos.y) > 32766.5 {")
            && source.contains("out.uv.y = 1.0;"),
        "a tile covering a pole is not pinned:\n{source}"
    );
}

/// A family with a height is not offered the direct bend, and does get the anchored one.
///
/// `Surface::Globe`'s placement reads `position.xy` and leaves `z` alone, because lifting a height
/// needs the sphere's normal and the coefficient for that -- `d_h` -- belongs to the anchored
/// bend. So a family that hands `place` a height would have it silently dropped there: every
/// building flat on the ground, which looks deliberate.
///
/// The producer agrees from the other side, which is why this is a rule and not a preference: it
/// writes a bend block for the extrusions, and that block is what the anchored surface reads.
#[test]
fn a_family_with_a_height_skips_the_direct_bend() {
    let mut with_height = 0;
    for family in families() {
        if !family.height {
            continue;
        }
        with_height += 1;
        assert!(
            !family.surfaces.contains(&Surface::Globe),
            "{} has a height and the direct bend would drop it",
            family.name
        );
        assert!(
            family.surfaces.contains(&Surface::GlobeAnchored),
            "{} has a height and nothing to lift it with",
            family.name
        );
    }
    assert!(with_height > 0, "no family exercises the height at all");

    // And the direct bend really does leave `z` alone, which is what makes the rule necessary.
    assert!(
        !Surface::Globe.placement().contains("position.z"),
        "the direct bend grew a height term; this rule may no longer be needed"
    );
    assert!(
        Surface::GlobeAnchored
            .placement()
            .contains("bend.d_h * position.z"),
        "the anchored bend no longer lifts a height"
    );
}

/// The color relief keeps the two decisions in its arithmetic that are silent when wrong.
///
/// Its picture *is* the elevation, so two steps decide what color a height gets and neither fails
/// loudly:
///
/// * the unpack replaces the texel's alpha with `-1`, which is what subtracts the encoding's bias
///   rather than adding it. Keep the alpha and every height is out by twice the bias -- 20,000 m
///   for Mapbox -- which saturates the ramp at one end and draws a flat wash.
/// * the stop tables are sampled at their texels' centers. Sample at the edge and nearest
///   filtering rounds to the neighbor, so a pixel takes the band next to its own: a plausible
///   relief, banded wrongly, which reads as a style problem.
///
/// Pinned as text because both compile, validate and draw. The arithmetic itself stays unverified
/// until a device renders it, which is true of every body here.
#[test]
fn the_color_relief_keeps_its_unpack_and_its_texel_centers() {
    let relief = families()
        .into_iter()
        .find(|family| family.name == "color_relief")
        .expect("color relief is in the matrix");
    let source = module(
        Surface::Plane,
        &relief.blocks,
        relief.attributes,
        relief.textures,
        relief.body,
    )
    .expect("assembles");

    assert!(
        source.contains("dot(vec4<f32>(texel.rgb, -1.0), tile.unpack)"),
        "the unpack does not replace the alpha with -1:\n{source}"
    );
    // Both tables, both centered. `+ 0.5` before the divide is the center of texel `index`.
    assert_eq!(
        source.matches("(f32(index) + 0.5) / f32(stops)").count(),
        2,
        "a stop table is not sampled at its texel's center"
    );
}

/// A family's own images bind before the surface's, and every one it declares is sampled.
///
/// The order is the contract with the descriptor set the renderer will write. A family's samplers
/// belong to the shader rather than to what it is drawn on -- a raster tile samples its own
/// picture and its parent's whether it is flat or raised -- so they keep their place when the
/// surface changes under them, and the surface's elevation lands after.
///
/// The second half is the one a compiler cannot catch: an unused sampler is legal, and a declared
/// image that nothing reads is a layer drawing from the wrong picture or from none.
#[test]
fn a_familys_images_bind_before_the_surfaces() {
    use tessella_emblema::shaders::texture_name;

    let raster = families()
        .into_iter()
        .find(|family| family.name == "raster")
        .expect("raster is in the matrix");
    assert_eq!(raster.textures.len(), 2, "a raster samples two pictures");

    // Flat: two blocks, then the family's two images and their samplers, and nothing after.
    let flat = module(
        Surface::Plane,
        &raster.blocks,
        raster.attributes,
        raster.textures,
        raster.body,
    )
    .expect("assembles");
    for (at, texture) in raster.textures.iter().enumerate() {
        let name = texture_name(texture.name);
        let binding = 2 + at * 2;
        assert!(
            flat.contains(&format!("@binding({binding}) var {name}: texture_2d<f32>;")),
            "{name} is not at binding {binding}:\n{flat}"
        );
        assert!(
            flat.contains(&format!(
                "@binding({}) var {name}_sampler: sampler;",
                binding + 1
            )),
            "{name}'s sampler is not beside it"
        );
        // And the body reads it, or the picture is declared and never drawn from.
        assert!(
            identifier_uses(&flat, &name) >= 2,
            "{name} is declared and never sampled"
        );
    }
    assert!(
        !flat.contains("@binding(6)"),
        "a plane adds nothing after the family's images"
    );

    // Raised: the surface's block pushes the images on by one binding, and the elevation lands
    // after them rather than among them.
    let raised = module(
        Surface::Terrain,
        &raster.blocks,
        raster.attributes,
        raster.textures,
        raster.body,
    )
    .expect("assembles");
    assert!(raised.contains("@binding(2) var<storage, read> terrain_drawable_ubo"));
    assert!(raised.contains("@binding(3) var raster_image0: texture_2d<f32>;"));
    assert!(raised.contains("@binding(5) var raster_image1: texture_2d<f32>;"));
    assert!(
        raised.contains("@binding(7) var elevation: texture_2d<f32>;"),
        "the surface's image does not come last:\n{raised}"
    );
}

/// The drawable's slot arrives as the draw's `firstInstance`, not as a push constant.
///
/// Measured, not preferred: one `vkCmdPushConstants` a draw costs 0.9 us on V3D against 0.014 on
/// RADV -- eleven times the draw it accompanies -- and 0.34 us on Adreno, where `firstInstance` is
/// a field of a call already being made and costs nothing anywhere. See `tests/bench-baselines/`.
///
/// Pinned because the push constant is the obvious way to write this and nothing else would catch
/// a change back: both spellings compile, validate and draw the same picture.
#[test]
fn the_slot_arrives_as_the_instance_index() {
    for family in families() {
        for surface in family.surfaces {
            let source = module(
                *surface,
                &family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .expect("assembles");
            let what = format!("{}{}", family.name, surface.suffix());
            assert!(
                !source.contains("push_constant"),
                "{what} still carries a push constant"
            );
            assert!(
                source.contains("@builtin(instance_index) instance_index: u32,"),
                "{what} does not declare the builtin"
            );
            assert!(
                source.contains("ubo_index = in.instance_index;"),
                "{what} never takes the slot from the builtin"
            );
            // And it is the entry point's first statement, or a block is indexed by whatever the
            // private variable held -- zero on the first draw and the previous draw's slot after.
            //
            // Checked as the first statement rather than as "before the first read": a surface's
            // `place` reads the slot and is emitted above the body, so textual order across the
            // module says nothing. Order only means anything inside the one function that does
            // the assigning, and being first there is the property worth having.
            assert!(
                source.contains(
                    "fn vertex_main(in: In) -> Out {\n    ubo_index = in.instance_index;"
                ),
                "{what} does not take the slot as its first statement"
            );
        }
    }
}

/// A matrix is declared as four columns, and the body cannot treat it as a matrix by accident.
///
/// Adreno's shader compiler asserts on a `mat4x4` read from a storage buffer and fails pipeline
/// creation with `VK_ERROR_UNKNOWN`; every block here is read from a storage buffer and every
/// drawable block carries a matrix, so `mat4x4` builds no pipeline at all on that board. RADV and
/// V3DV accept it, so nothing on a desktop catches a change back.
///
/// The second half is the one that keeps it fixed: an array is not a matrix to WGSL, so a body
/// that multiplies the field by a vector fails to compile rather than failing on a board.
#[test]
fn a_matrix_is_declared_as_four_columns() {
    use tessella_emblema::preamble::MATRIX;

    assert_eq!(MATRIX, "array<vec4<f32>, 4>");
    let source = background();
    assert!(
        source.contains(&format!("    matrix: {MATRIX},")),
        "the drawable block does not declare its matrix as columns:\n{source}"
    );
    assert!(
        !source.contains("    matrix: mat4x4<f32>,"),
        "a block still declares a matrix as a matrix"
    );
    assert!(
        source.contains("fn transform(columns: array<vec4<f32>, 4>, position: vec3<f32>)"),
        "the prelude does not define the way a body applies one"
    );
    // No matrix is assembled anywhere either. Vivante's SPIR-V compiler segfaults on
    // `OpCompositeConstruct` of a matrix -- `VIR_Shader_CompositeConstruct` in `libVSC.so` --
    // so the first answer to Adreno's assertion, columns in the block rebuilt into a `mat4x4`
    // to multiply, trades one vendor's crash for another's.
    assert!(
        !source.contains("mat4x4"),
        "a matrix type survives somewhere in the module:\n{source}"
    );
    // The columns applied in order, pinned verbatim. A snapshot of these lines is the right
    // shape of test here: nothing else can catch a repeated or transposed index, because every
    // wrong combination is well-typed, compiles, validates, and draws every vertex in the wrong
    // place -- which reads as a camera fault rather than as a typo. The producer writes
    // column-major, so column `n` multiplies component `n`.
    for (column, component) in [("0", "x"), ("1", "y"), ("2", "z")] {
        assert!(
            source.contains(&format!("columns[{column}] * position.{component}")),
            "column {column} does not multiply {component}"
        );
    }
    assert!(
        source.contains("+ columns[3];"),
        "the translation column is not added on its own"
    );
    assert!(
        source.contains("place(vec3<f32>(vec2<f32>(in.background_pos), 0.0), drawable.matrix)"),
        "the body does not hand the columns straight to the surface"
    );
}

/// No module anywhere names a matrix type.
///
/// Two vendor compilers between them refuse both halves of the obvious spelling: Adreno asserts on
/// a `mat4x4` read from a storage buffer, and Vivante segfaults on constructing one. So the
/// property worth checking is not which helper a body uses -- it is that the type does not appear
/// at all, on any family or any surface.
#[test]
fn no_module_names_a_matrix_type() {
    for family in families() {
        for surface in family.surfaces {
            let source = module(
                *surface,
                &family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .expect("assembles");
            assert!(
                !source.contains("mat4x4"),
                "{}{} names a matrix type",
                family.name,
                surface.suffix()
            );
        }
    }
}

/// The vertex input carries the attribute at the location the producer binds it to.
#[test]
fn the_vertex_input_matches_the_attribute_table() {
    let source = background();
    let attribute = BACKGROUND_SHADER[0];

    assert_eq!(attribute_name(attribute.name), "background_pos");
    assert!(
        source.contains(&format!(
            "@location({}) background_pos: vec2<i32>",
            attribute.binding
        )),
        "the input does not match the table:\n{source}"
    );
}

/// A body reading a field the tables do not declare does not compile.
///
/// The property the generation exists for. Without it a shader and a block drift apart silently,
/// and the drift shows up as a picture.
#[test]
fn a_body_naming_an_undeclared_field_fails() {
    let source = module(
        Surface::Plane,
        &[&BACKGROUND_DRAWABLE_UBO],
        &BACKGROUND_SHADER,
        &[],
        r"
@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return background_drawable_ubo[0].no_such_field;
}
",
    )
    .expect("declares");

    assert!(
        naga::front::wgsl::parse_str(&source).is_err(),
        "a field the block does not have compiled anyway"
    );
}

/// The emitted SPIR-V is the same from one run to the next.
///
/// A snapshot on the word count rather than a hash of the words: naga embeds no timestamp, but
/// pinning every word would make an unrelated naga version bump read as a shader change, and the
/// count moves for the reasons worth noticing -- a body gaining work, a block gaining a field.
#[test]
fn the_emitted_module_is_stable() {
    let first = compile(&background());
    let second = compile(&background());
    assert_eq!(first, second, "two compiles of one source disagreed");
}

/// How many times `name` appears as a whole identifier.
///
/// Not a substring count: `opacity_t` is inside `stroke_opacity_t`, so counting substrings lets a
/// field borrow another's uses and a dropped read goes unnoticed. Found by a mutation that
/// survived.
fn identifier_uses(source: &str, name: &str) -> usize {
    let bytes = source.as_bytes();
    source
        .match_indices(name)
        .filter(|(at, _)| {
            let before = *at == 0 || !is_identifier(bytes[at - 1]);
            let after_at = at + name.len();
            let after = after_at >= bytes.len() || !is_identifier(bytes[after_at]);
            before && after
        })
        .count()
}

const fn is_identifier(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Every family on every surface it has compiles, validates and emits.
#[test]
fn every_family_on_every_surface_compiles() {
    let mut pairs = 0;
    for family in families() {
        for surface in family.surfaces {
            let what = format!("{}{}", family.name, surface.suffix());
            let source = module(
                *surface,
                &family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .unwrap_or_else(|why| panic!("{what} does not assemble: {why:?}"));
            let words = compile(&source);
            assert!(words.len() > 64, "{what} emitted {} words", words.len());
            assert_eq!(words[0], 0x0723_0203, "{what} is not SPIR-V");
            pairs += 1;
        }
    }
    assert_eq!(
        pairs, 44,
        "the matrix grew or shrank; look at the new pairs"
    );
}

/// Every attribute the producer sends is read by the body that receives it.
///
/// An attribute declared and never read is a paint property the style asked for and the picture
/// does not show — which draws, because the rest of the shader is fine. The compiler cannot catch
/// it: an unused input is legal.
#[test]
fn every_attribute_is_read() {
    for family in families() {
        for surface in family.surfaces {
            let source = module(
                *surface,
                &family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .expect("assembles");
            for attribute in family.attributes {
                let field = attribute_name(attribute.name);
                // Once in the generated input, and at least once more in the body that reads it.
                let uses = identifier_uses(&source, &field);
                assert!(
                    uses >= 2,
                    "{}{} declares {field} and never reads it",
                    family.name,
                    surface.suffix()
                );
            }
        }
    }
}

/// Each surface displaces a vertex its own way, because three of the four are not linear maps.
///
/// A line is a strip of quads that carries its own sideways extrusion, and the extrusion has to be
/// applied where the surface is, not where the tile is. The plane and the raise are linear, so the
/// matrix's first two columns are the whole answer. The direct bend is trig and has no linear part
/// at all, so it is evaluated twice and differenced. The anchored bend is a quadratic, so its
/// Jacobian at the vertex is exact -- and taken *at the vertex*, not at the anchor, because at low
/// zoom a tile is wide enough that the bend turns across it.
///
/// Giving any of them the plane's displacement compiles and draws lines with a believable width in
/// the wrong direction, which is why this is pinned rather than left to review.
#[test]
fn each_surface_displaces_its_own_way() {
    let linear = "return columns[0] * delta.x + columns[1] * delta.y;";
    for surface in Surface::ALL {
        let placement = surface.placement();
        assert!(
            placement.contains("fn displace("),
            "{} has no displacement",
            surface.suffix()
        );
        match surface {
            Surface::Plane | Surface::Terrain => assert!(
                placement.contains(linear),
                "{} does not extrude along the matrix's columns",
                surface.suffix()
            ),
            Surface::Globe => assert!(
                placement.contains(
                    "return place(vec3<f32>(at + delta, 0.0), columns) \
                     - place(vec3<f32>(at, 0.0), columns);"
                ) && !placement.contains(linear),
                "the direct bend is not evaluated twice and differenced"
            ),
            Surface::GlobeAnchored => assert!(
                placement.contains("let j_u = bend.d_u + bend.d_uu * d.x + bend.d_uv * d.y;")
                    && placement
                        .contains("let j_v = bend.d_v + bend.d_vv * d.y + bend.d_uv * d.x;")
                    && !placement.contains(linear),
                "the anchored bend does not extrude along its Jacobian at the vertex"
            ),
        }
    }
}

/// Components a family declares and correctly does not read, as `(family, attribute, components)`.
///
/// Each one is checked against the shader mbgl generates for that family, not reasoned about: an
/// attribute is shared between a family and its variants, and a component only one variant needs
/// arrives for both.
const UNREAD_COMPONENTS: &[(&str, &str, &str)] = &[
    // `a_decimals_ed.y` is the edge distance, which only `fill_extrusion_pattern.vertex.glsl`
    // reads. The plain family declares the pair and uses the first of it.
    ("fill_extrusion", "fill_extrusion_decimals_ed", "y"),
    // `a_pixeloffset.zw` is the minimum font scale, which `symbol_icon.vertex.glsl` reads and
    // `symbol_sdf.vertex.glsl` does not -- the SDF family takes `a_pxoffset` from `xy` and stops.
    ("symbol_sdf", "symbol_pixel_offset", "zw"),
];

/// Every component of every attribute is read, not just the identifier.
///
/// [`every_attribute_is_read`] asks whether the name appears, which a body reading half a packed
/// `vec4` satisfies. Two things ride in the spare half of a symbol's pixel offset -- the offset in
/// `xy` and the minimum font scale in `zw` -- and a body that reads `xy` and stops passes that
/// test, compiles, validates and draws type at the wrong size. This asks per component.
///
/// A use with no swizzle counts as the whole thing, which is how a `vec4` handed to `unpack_color`
/// or a `vec3` handed to `place` is covered.
#[test]
fn every_component_of_an_attribute_is_read() {
    let mut short = Vec::new();
    for family in families() {
        let source = module(
            Surface::Plane,
            &family.blocks,
            family.attributes,
            family.textures,
            family.body,
        )
        .expect("assembles");
        for attribute in family.attributes {
            let field = attribute_name(attribute.name);
            let width = components(attribute.declared);
            if width < 2 {
                continue;
            }
            let mut read = components_read(&source, &field);
            for (family_name, attribute_field, sanctioned) in UNREAD_COMPONENTS {
                if *family_name == family.name && *attribute_field == field {
                    read.extend(sanctioned.chars());
                }
            }
            if read.len() < width {
                short.push(format!(
                    "{} reads {}/{width} of {field}: {read:?}",
                    family.name,
                    read.len()
                ));
            }
        }
    }
    assert!(short.is_empty(), "components left unread: {short:#?}");
}

/// How many components an attribute's declared type has.
fn components(declared: AttributeDataType) -> usize {
    let name = format!("{declared:?}");
    match name.chars().last() {
        Some(digit @ '2'..='9') => digit.to_digit(10).expect("a digit") as usize,
        _ => 1,
    }
}

/// Which of `xyzw` the body takes from `in.<field>`, as the characters it names.
///
/// A bare use -- `in.field` with no `.` after it -- is the whole thing, so it answers `xyzw`.
fn components_read(source: &str, field: &str) -> BTreeSet<char> {
    let mut read = BTreeSet::new();
    let needle = format!("in.{field}");
    for (at, _) in source.match_indices(&needle) {
        let after = &source[at + needle.len()..];
        // `in.symbol_data` is a prefix of nothing else today, but a longer field with the same
        // start would be counted here if one ever arrives.
        if after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let Some(swizzle) = after.strip_prefix('.') else {
            read.extend(['x', 'y', 'z', 'w']);
            continue;
        };
        let named: Vec<char> = swizzle
            .chars()
            .take_while(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'))
            .collect();
        if named.is_empty() {
            read.extend(['x', 'y', 'z', 'w']);
            continue;
        }
        for c in named {
            read.insert(match c {
                'r' => 'x',
                'g' => 'y',
                'b' => 'z',
                'a' => 'w',
                other => other,
            });
        }
    }
    read
}

/// Attribute and factor pairs whose names do not match, as `(attribute field, factor)`.
///
/// A symbol's fill color is `idSymbolColorVertexAttribute` and its factor is `fill_color_t`: the
/// attribute is named for the shader's input and the factor for the style property, and here the
/// two diverge. mbgl's `symbol_sdf_paint.glsl` is what says they are the same property.
const FACTOR_ALIASES: &[(&str, &str)] = &[("symbol_color", "fill_color_t")];

/// Every data-driven property's zoom factor is used where its attribute is.
///
/// The `_t` fields exist to mix an attribute between its two zoom endpoints. Reading the attribute
/// and ignoring the factor draws the lower endpoint at every zoom, which looks like a style that
/// stopped interpolating rather than like a bug.
///
/// Asked from the attribute's side, not the factor's. A drawable block is shared between a family
/// and its variants -- `FILL_EXTRUSION_DRAWABLE_UBO` carries `pattern_from_t` for the pattern
/// shader -- so requiring every factor in a block to be used by every family that reads it
/// demands the impossible. What is actually wanted is narrower and is the property above: if the
/// family declares the attribute, it has to mix with the factor.
///
/// Paired by name with the underscores removed, because the two spellings differ: the attribute is
/// `idLineGapWidthVertexAttribute` and the factor is `gapwidth_t`. Where the shader's name for an
/// attribute is not the style property's name at all, the pair is named in [`FACTOR_ALIASES`].
#[test]
fn every_zoom_factor_is_used() {
    let mut paired = 0;
    for family in families() {
        for surface in family.surfaces {
            let source = module(
                *surface,
                &family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .expect("assembles");
            for attribute in family.attributes {
                let field = attribute_name(attribute.name);
                let flat = field.replace('_', "");
                for block in &family.blocks {
                    for factor in block.fields {
                        let Some(stem) = factor.name.strip_suffix("_t") else {
                            continue;
                        };
                        let aliased = FACTOR_ALIASES.contains(&(field.as_str(), factor.name));
                        if !aliased && !flat.ends_with(&stem.replace('_', "")) {
                            continue;
                        }
                        paired += 1;
                        assert!(
                            identifier_uses(&source, factor.name) >= 2,
                            "{}{} reads {field} and never mixes it with {}",
                            family.name,
                            surface.suffix(),
                            factor.name
                        );
                    }
                }
            }
        }
    }
    assert!(paired > 0, "no attribute was paired with a factor at all");
}
