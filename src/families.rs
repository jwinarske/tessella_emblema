// SPDX-License-Identifier: BSD-2-Clause
//! Which parts make up each family's module, keyed by the shader the producer names.
//!
//! A drawable arrives naming `builtin_shader`, an `i32` discriminant. Turning that into a module
//! takes four things — the uniform blocks it declares, the attribute table, the texture table and
//! the body — and which surfaces it can be drawn on. This is that table.
//!
//! Keyed by [`BuiltIn`] rather than by a name, because the wire names a shader and nothing on the
//! wire carries a name. A consumer that had to map a discriminant to a family itself would be
//! holding the half of this crate's knowledge that is hardest to check.
//!
//! # What is not here
//!
//! Fifteen of mbgl's thirty-three shaders. [`UNDRAWN`] lists them with the reason, and
//! [`for_wire`] returns `None` for each, so a producer naming one is a drawable this crate skips
//! rather than a panic. The list is pinned by a test: adding a family here without removing it
//! there fails, and so does the reverse.

use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::shader_attributes::{
    BACKGROUND_PATTERN_SHADER, BACKGROUND_SHADER, CIRCLE_SHADER, COLOR_RELIEF_SHADER,
    FILL_EXTRUSION_SHADER, FILL_OUTLINE_SHADER, FILL_PATTERN_SHADER, FILL_SHADER, HEATMAP_SHADER,
    HEATMAP_TEXTURE_SHADER, HILLSHADE_PREPARE_SHADER, HILLSHADE_SHADER, LINE_PATTERN_SHADER,
    LINE_SHADER, RASTER_SHADER, SYMBOL_ICON_SHADER, SYMBOL_SDFSHADER, SYMBOL_TEXT_AND_ICON_SHADER,
    ShaderAttribute,
};
use tessella_capture_abi::generated::texture_slots::{
    BACKGROUND_PATTERN_SHADER_TEXTURES, COLOR_RELIEF_SHADER_TEXTURES, FILL_PATTERN_SHADER_TEXTURES,
    HEATMAP_TEXTURE_SHADER_TEXTURES, HILLSHADE_PREPARE_SHADER_TEXTURES, HILLSHADE_SHADER_TEXTURES,
    LINE_PATTERN_SHADER_TEXTURES, RASTER_SHADER_TEXTURES, SYMBOL_ICON_SHADER_TEXTURES,
    SYMBOL_SDFSHADER_TEXTURES, SYMBOL_TEXT_AND_ICON_SHADER_TEXTURES, ShaderTexture,
};
use tessella_capture_abi::generated::ubo_layouts::{
    BACKGROUND_DRAWABLE_UBO, BACKGROUND_PATTERN_DRAWABLE_UBO, BACKGROUND_PATTERN_PROPS_UBO,
    BACKGROUND_PROPS_UBO, CIRCLE_DRAWABLE_UBO, CIRCLE_EVALUATED_PROPS_UBO,
    COLOR_RELIEF_DRAWABLE_UBO, COLOR_RELIEF_EVALUATED_PROPS_UBO, COLOR_RELIEF_TILE_PROPS_UBO,
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, FILL_EXTRUSION_DRAWABLE_UBO,
    FILL_EXTRUSION_PROPS_UBO, FILL_OUTLINE_DRAWABLE_UBO, FILL_PATTERN_DRAWABLE_UBO,
    FILL_PATTERN_TILE_PROPS_UBO, GLOBAL_PAINT_PARAMS_UBO, HEATMAP_DRAWABLE_UBO,
    HEATMAP_EVALUATED_PROPS_UBO, HEATMAP_TEXTURE_PROPS_UBO, HILLSHADE_DRAWABLE_UBO,
    HILLSHADE_EVALUATED_PROPS_UBO, HILLSHADE_PREPARE_DRAWABLE_UBO,
    HILLSHADE_PREPARE_TILE_PROPS_UBO, HILLSHADE_TILE_PROPS_UBO, LINE_DRAWABLE_UBO,
    LINE_EVALUATED_PROPS_UBO, LINE_PATTERN_DRAWABLE_UBO, LINE_PATTERN_TILE_PROPS_UBO,
    RASTER_DRAWABLE_UBO, RASTER_EVALUATED_PROPS_UBO, SYMBOL_DRAWABLE_UBO,
    SYMBOL_EVALUATED_PROPS_UBO, SYMBOL_TILE_PROPS_UBO, UboLayout,
};

use crate::shaders::{
    BACKGROUND_BODY, BACKGROUND_PATTERN_BODY, CIRCLE_BODY, COLOR_RELIEF_BODY, FILL_BODY,
    FILL_EXTRUSION_BODY, FILL_OUTLINE_BODY, FILL_PATTERN_BODY, HEATMAP_BODY, HEATMAP_TEXTURE_BODY,
    HILLSHADE_BODY, HILLSHADE_PREPARE_BODY, LINE_BODY, LINE_PATTERN_BODY, RASTER_BODY,
    SYMBOL_ICON_BODY, SYMBOL_SDF_BODY, SYMBOL_TEXT_AND_ICON_BODY,
};
use crate::surface::Surface;

/// Every surface. What a family drawn on a tile has.
const EVERY: &[Surface] = &[
    Surface::Plane,
    Surface::Globe,
    Surface::GlobeAnchored,
    Surface::Terrain,
];

/// A plane and the direct bend, for a family the producer writes no bend block for.
///
/// A background covers the viewport rather than a tile, so it is never anchored and never raised.
const FLAT_OR_BENT: &[Surface] = &[Surface::Plane, Surface::Globe];

/// Everything but the anchored bend, which needs a block the producer writes per tile.
const UNANCHORED: &[Surface] = &[Surface::Plane, Surface::Globe, Surface::Terrain];

/// A plane alone, for a family that draws into its own offscreen target.
const FLAT: &[Surface] = &[Surface::Plane];

/// A plane and the anchored bend. An extrusion leaves the surface, and the direct bend has no
/// height term, so it cannot be drawn on that one.
const PLANE_OR_ANCHORED: &[Surface] = &[Surface::Plane, Surface::GlobeAnchored];

/// One family's module parts.
#[derive(Debug, Clone, Copy)]
pub struct Family {
    /// The shader the producer names on the wire.
    pub shader: BuiltIn,
    /// Short name, for diagnostics and for matching an oracle case.
    pub name: &'static str,
    /// The uniform blocks the body reads, in binding order.
    pub blocks: &'static [&'static UboLayout],
    /// The attribute table the vertex input is planned against.
    pub attributes: &'static [ShaderAttribute],
    /// The texture table, which names the module's samplers.
    pub textures: &'static [ShaderTexture],
    /// The hand-written body.
    pub body: &'static str,
    /// Which surfaces a module exists for.
    pub surfaces: &'static [Surface],
    /// Whether the body hands `place` a height above the surface rather than zero.
    pub height: bool,
}

/// Every family this crate draws, in no particular order.
pub static ALL: &[Family] = &[
    Family {
        shader: BuiltIn::BackgroundShader,
        name: "background",
        blocks: &[&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
        attributes: &BACKGROUND_SHADER,
        textures: &[],
        body: BACKGROUND_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
    },
    Family {
        shader: BuiltIn::BackgroundPatternShader,
        name: "background_pattern",
        blocks: &[
            &BACKGROUND_PATTERN_DRAWABLE_UBO,
            &BACKGROUND_PATTERN_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &BACKGROUND_PATTERN_SHADER,
        textures: &BACKGROUND_PATTERN_SHADER_TEXTURES,
        body: BACKGROUND_PATTERN_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
    },
    Family {
        shader: BuiltIn::FillShader,
        name: "fill",
        blocks: &[&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
        attributes: &FILL_SHADER,
        textures: &[],
        body: FILL_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::FillOutlineShader,
        name: "fill_outline",
        blocks: &[
            &FILL_OUTLINE_DRAWABLE_UBO,
            &FILL_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &FILL_OUTLINE_SHADER,
        textures: &[],
        body: FILL_OUTLINE_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::FillPatternShader,
        name: "fill_pattern",
        blocks: &[
            &FILL_PATTERN_DRAWABLE_UBO,
            &FILL_PATTERN_TILE_PROPS_UBO,
            &FILL_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &FILL_PATTERN_SHADER,
        textures: &FILL_PATTERN_SHADER_TEXTURES,
        body: FILL_PATTERN_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::FillExtrusionShader,
        name: "fill_extrusion",
        blocks: &[&FILL_EXTRUSION_DRAWABLE_UBO, &FILL_EXTRUSION_PROPS_UBO],
        attributes: &FILL_EXTRUSION_SHADER,
        textures: &[],
        body: FILL_EXTRUSION_BODY,
        surfaces: PLANE_OR_ANCHORED,
        height: true,
    },
    Family {
        shader: BuiltIn::LineShader,
        name: "line",
        blocks: &[
            &LINE_DRAWABLE_UBO,
            &LINE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &LINE_SHADER,
        textures: &[],
        body: LINE_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::LinePatternShader,
        name: "line_pattern",
        blocks: &[
            &LINE_PATTERN_DRAWABLE_UBO,
            &LINE_PATTERN_TILE_PROPS_UBO,
            &LINE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &LINE_PATTERN_SHADER,
        textures: &LINE_PATTERN_SHADER_TEXTURES,
        body: LINE_PATTERN_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::CircleShader,
        name: "circle",
        blocks: &[
            &CIRCLE_DRAWABLE_UBO,
            &CIRCLE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &CIRCLE_SHADER,
        textures: &[],
        body: CIRCLE_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::RasterShader,
        name: "raster",
        blocks: &[&RASTER_DRAWABLE_UBO, &RASTER_EVALUATED_PROPS_UBO],
        attributes: &RASTER_SHADER,
        textures: &RASTER_SHADER_TEXTURES,
        body: RASTER_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::HillshadePrepareShader,
        name: "hillshade_prepare",
        blocks: &[
            &HILLSHADE_PREPARE_DRAWABLE_UBO,
            &HILLSHADE_PREPARE_TILE_PROPS_UBO,
        ],
        attributes: &HILLSHADE_PREPARE_SHADER,
        textures: &HILLSHADE_PREPARE_SHADER_TEXTURES,
        body: HILLSHADE_PREPARE_BODY,
        surfaces: FLAT,
        height: false,
    },
    Family {
        shader: BuiltIn::HillshadeShader,
        name: "hillshade",
        blocks: &[
            &HILLSHADE_DRAWABLE_UBO,
            &HILLSHADE_TILE_PROPS_UBO,
            &HILLSHADE_EVALUATED_PROPS_UBO,
        ],
        attributes: &HILLSHADE_SHADER,
        textures: &HILLSHADE_SHADER_TEXTURES,
        body: HILLSHADE_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::ColorReliefShader,
        name: "color_relief",
        blocks: &[
            &COLOR_RELIEF_DRAWABLE_UBO,
            &COLOR_RELIEF_TILE_PROPS_UBO,
            &COLOR_RELIEF_EVALUATED_PROPS_UBO,
        ],
        attributes: &COLOR_RELIEF_SHADER,
        textures: &COLOR_RELIEF_SHADER_TEXTURES,
        body: COLOR_RELIEF_BODY,
        surfaces: UNANCHORED,
        height: false,
    },
    Family {
        shader: BuiltIn::HeatmapShader,
        name: "heatmap",
        blocks: &[&HEATMAP_DRAWABLE_UBO, &HEATMAP_EVALUATED_PROPS_UBO],
        attributes: &HEATMAP_SHADER,
        textures: &[],
        body: HEATMAP_BODY,
        surfaces: FLAT,
        height: false,
    },
    Family {
        shader: BuiltIn::HeatmapTextureShader,
        name: "heatmap_texture",
        blocks: &[&HEATMAP_TEXTURE_PROPS_UBO, &GLOBAL_PAINT_PARAMS_UBO],
        attributes: &HEATMAP_TEXTURE_SHADER,
        textures: &HEATMAP_TEXTURE_SHADER_TEXTURES,
        body: HEATMAP_TEXTURE_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
    },
    Family {
        shader: BuiltIn::SymbolIconShader,
        name: "symbol_icon",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_ICON_SHADER,
        textures: &SYMBOL_ICON_SHADER_TEXTURES,
        body: SYMBOL_ICON_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::SymbolSDFShader,
        name: "symbol_sdf",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_SDFSHADER,
        textures: &SYMBOL_SDFSHADER_TEXTURES,
        body: SYMBOL_SDF_BODY,
        surfaces: EVERY,
        height: false,
    },
    Family {
        shader: BuiltIn::SymbolTextAndIconShader,
        name: "symbol_text_and_icon",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_TEXT_AND_ICON_SHADER,
        textures: &SYMBOL_TEXT_AND_ICON_SHADER_TEXTURES,
        body: SYMBOL_TEXT_AND_ICON_BODY,
        surfaces: EVERY,
        height: false,
    },
];

/// The three symbol families declare the same blocks.
const SYMBOL_BLOCKS: &[&UboLayout] = &[
    &SYMBOL_DRAWABLE_UBO,
    &SYMBOL_TILE_PROPS_UBO,
    &SYMBOL_EVALUATED_PROPS_UBO,
    &GLOBAL_PAINT_PARAMS_UBO,
];

/// The shaders mbgl declares that this crate does not draw, with why.
///
/// Pinned by a test against [`ALL`] and the `BuiltIn` enum, so this cannot drift from either.
pub static UNDRAWN: &[(BuiltIn, &str)] = &[
    (
        BuiltIn::None,
        "the absence of a shader, which an order entry naming it has no drawable for",
    ),
    (
        BuiltIn::Prelude,
        "mbgl's shared shader prelude, which is not a program at all",
    ),
    (
        BuiltIn::ClippingMaskProgram,
        "the stencil mask, which the consumer draws from the partition rather than from a family",
    ),
    (
        BuiltIn::CollisionBoxShader,
        "a debug overlay, not a map layer",
    ),
    (
        BuiltIn::CollisionCircleShader,
        "a debug overlay, not a map layer",
    ),
    (BuiltIn::DebugShader, "a debug overlay, not a map layer"),
    (
        BuiltIn::CustomGeometryShader,
        "an embedder's own geometry, which this crate has no producer for",
    ),
    (
        BuiltIn::CustomSymbolIconShader,
        "an embedder's own symbols, which this crate has no producer for",
    ),
    (
        BuiltIn::FillOutlinePatternShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::FillOutlineTriangulatedShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::FillExtrusionInstancedShader,
        "the wall form, whose instance attributes the tables carry and whose body is unported",
    ),
    (
        BuiltIn::FillExtrusionPatternShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::FillExtrusionPatternInstancedShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::LineGradientShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::LineSDFShader,
        "the producer emits no drawable for it yet",
    ),
    (
        BuiltIn::LocationIndicatorShader,
        "the puck, which the consumer draws from its own mesh rather than from a family",
    ),
    (
        BuiltIn::LocationIndicatorTexturedShader,
        "the puck, which the consumer draws from its own mesh rather than from a family",
    ),
    (
        BuiltIn::WideVectorShader,
        "not a maplibre-native layer type",
    ),
];

/// The family for a shader, or `None` if this crate does not draw it.
#[must_use]
pub fn family(shader: BuiltIn) -> Option<&'static Family> {
    ALL.iter().find(|family| family.shader == shader)
}

/// The family for a `builtin_shader` as it arrives on the wire.
///
/// `None` both for a discriminant that is not a `BuiltIn` and for one this crate does not draw.
/// The caller skips the drawable either way, which is why the two are not distinguished: an order
/// may name a family a build does not have, and that is a drawable to leave out rather than an
/// error to raise.
#[must_use]
pub fn for_wire(builtin_shader: i32) -> Option<&'static Family> {
    BuiltIn::from_repr(builtin_shader).and_then(family)
}
