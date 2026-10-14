// SPDX-License-Identifier: BSD-2-Clause
//! Which parts make up each family's module, keyed by the shader the producer names.
//!
//! A drawable arrives naming `builtin_shader`, an `i32` discriminant. Turning that into a module
//! takes four things — the uniform blocks it declares, the attribute table, the texture table and
//! the body — and which surfaces it can be drawn on. This is that table.
//!
//! And one thing that is not part of a module at all: [`Composite`], how a draw of the family
//! composites over the target. That is here because mbgl keeps it in the same place -- a render
//! layer's own source, not a style and not the wire -- and because it is the kind of fact this
//! table exists to hold, transcribed once and pinned.
//!
//! Keyed by the shader id rather than by a name, because the wire names a shader and nothing on
//! the wire carries a name. A consumer that had to map a discriminant to a family itself would be
//! holding the half of this crate's knowledge that is hardest to check. Twenty of the ids are
//! mbgl's [`BuiltIn`] discriminants and the twenty-first is tessella's own: mbgl has no terrain,
//! so the ground is a family here with no counterpart there -- see [`Family::shader`].
//!
//! # What is not here
//!
//! Sixteen of mbgl's thirty-six shader entries. [`UNDRAWN`] lists them with the reason, and
//! [`for_wire`] returns `None` for each, so a producer naming one is a drawable this crate skips
//! rather than a panic. The list is pinned by a test: adding a family here without removing it
//! there fails, and so does the reverse.

use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::shader_attributes::{
    BACKGROUND_PATTERN_SHADER, BACKGROUND_SHADER, CIRCLE_SHADER, COLOR_RELIEF_SHADER,
    FILL_EXTRUSION_INSTANCED_SHADER, FILL_EXTRUSION_INSTANCED_SHADER_INSTANCE,
    FILL_EXTRUSION_SHADER, FILL_OUTLINE_SHADER, FILL_OUTLINE_TRIANGULATED_SHADER,
    FILL_PATTERN_SHADER, FILL_SHADER, HEATMAP_SHADER, HEATMAP_TEXTURE_SHADER,
    HILLSHADE_PREPARE_SHADER, HILLSHADE_SHADER, LINE_PATTERN_SHADER, LINE_SHADER, RASTER_SHADER,
    SYMBOL_ICON_SHADER, SYMBOL_SDFSHADER, SYMBOL_TEXT_AND_ICON_SHADER, ShaderAttribute,
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
    FILL_EXTRUSION_PROPS_UBO, FILL_OUTLINE_DRAWABLE_UBO, FILL_OUTLINE_TRIANGULATED_DRAWABLE_UBO,
    FILL_PATTERN_DRAWABLE_UBO, FILL_PATTERN_TILE_PROPS_UBO, GLOBAL_PAINT_PARAMS_UBO,
    HEATMAP_DRAWABLE_UBO, HEATMAP_EVALUATED_PROPS_UBO, HEATMAP_TEXTURE_PROPS_UBO,
    HILLSHADE_DRAWABLE_UBO, HILLSHADE_EVALUATED_PROPS_UBO, HILLSHADE_PREPARE_DRAWABLE_UBO,
    HILLSHADE_PREPARE_TILE_PROPS_UBO, HILLSHADE_TILE_PROPS_UBO, LINE_DRAWABLE_UBO,
    LINE_EVALUATED_PROPS_UBO, LINE_PATTERN_DRAWABLE_UBO, LINE_PATTERN_TILE_PROPS_UBO,
    RASTER_DRAWABLE_UBO, RASTER_EVALUATED_PROPS_UBO, SYMBOL_DRAWABLE_UBO,
    SYMBOL_EVALUATED_PROPS_UBO, SYMBOL_TILE_PROPS_UBO, UboLayout,
};
use tessella_capture_abi::terrain_ubo::{
    BUILTIN_TERRAIN_SHADER, TERRAIN_MESH_ATTRIBUTES, TERRAIN_MESH_ELEVATION_SLOT,
};

use crate::pipelines::Blend;
use crate::shaders::{
    self, BACKGROUND_BODY, BACKGROUND_PATTERN_BODY, CIRCLE_BODY, COLOR_RELIEF_BODY, FILL_BODY,
    FILL_EXTRUSION_BODY, FILL_EXTRUSION_INSTANCED_BODY, FILL_OUTLINE_BODY,
    FILL_OUTLINE_TRIANGULATED_BODY, FILL_PATTERN_BODY, HEATMAP_BODY, HEATMAP_TEXTURE_BODY,
    HILLSHADE_BODY, HILLSHADE_PREPARE_BODY, LINE_BODY, LINE_PATTERN_BODY, Paint, Property,
    RASTER_BODY, SYMBOL_ICON_BODY, SYMBOL_SDF_BODY, SYMBOL_TEXT_AND_ICON_BODY, TERRAIN_BODY,
};
use crate::surface::{Surface, TERRAIN_DRAWABLE_UBO};

/// How a family's draws composite over the target, which is mbgl's `setColorMode`.
///
/// Nothing on the wire carries it -- `grep -i color_mode crates/tessella-capture-abi/src` is empty
/// -- and nothing has to, because mbgl does not take it from a style either. Each render layer
/// calls `setColorMode` with a mode written in its own source, so the whole of it is a property of
/// the family and belongs here beside the blocks and the tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Composite {
    /// One mode, whatever the pass. Fourteen of the twenty-one.
    Fixed(Blend),
    /// [`Blend::Alpha`] where the pass is translucent and [`Blend::Unblended`] where it is not.
    ///
    /// mbgl writes this seven times in three spellings: the background compares
    /// `drawPasses == RenderPass::Translucent` by equality, the line masks the opaque bit out
    /// first and then compares, and the symbol tests the translucent bit. This is the third.
    ///
    /// The three differ on one input, a drawable declaring `Opaque | Translucent`, which the
    /// background's equality gives `unblended` in *both* passes. A line and a symbol cannot reach
    /// it: their own `evaluate` sets `Translucent` or `None` and never the opaque bit, so their
    /// other arm is unreachable in mbgl as well. A background can, and only when its opacity and
    /// its color's alpha are both one -- that is the condition `RenderBackgroundLayer::evaluate`
    /// supplies both passes under, and under any other it supplies `Translucent` alone.
    ///
    /// At an alpha of one the two modes are the same function. `alphaBlended` is
    /// `Add{One, OneMinusSrcAlpha}` over a premultiplied source, so it is `src + dst * 0`, which
    /// is `src` -- what `unblended` writes. Measured rather than argued: `benches/first_pixel.rs`
    /// draws the background both ways over an opaque clear and reads the same four bytes, and
    /// draws it faded both ways and reads two different ones.
    ByPass,
}

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

/// Every surface that can lift a height, which is all of them but the direct bend.
///
/// An extrusion leaves the surface: its vertices carry a height in meters, and a placement that
/// reads no `z` draws every building flat on the ground. [`Surface::Globe`] is the one that reads
/// none -- lifting along a sphere's normal needs the `d_h` coefficient, which belongs to the
/// anchored bend -- and `tests/shaders.rs` holds that rule from both sides.
const LIFTS_A_HEIGHT: &[Surface] = &[Surface::Plane, Surface::GlobeAnchored, Surface::Terrain];

/// A plane alone, for the family that *is* a surface.
///
/// The ground, and not [`FLAT`]'s reason: nothing offscreen here. A ground is drawn in a Mercator
/// frame and only there -- `Map::surface` takes the globe where a style asks for both, so the
/// producer sends no ground at all on a sphere and there is no pair to have. And it is not
/// [`Surface::Terrain`] either, which is what a drawable *standing* on the ground is placed on: the
/// ground raises itself out of its own block, from an elevation that arrives at its own slot.
const THE_GROUND: &[Surface] = &[Surface::Plane];

/// The ground's one texture: its own DEM, at the slot the ABI publishes for this family.
///
/// Written here because nothing generates it -- the generator mirrors mbgl's shaders and mbgl has
/// no terrain -- and in the shape the generated tables take, so the module, the set layout and the
/// placement of an arriving `TextureRef` all read it the way they read every other family's.
///
/// The slot is [`TERRAIN_MESH_ELEVATION_SLOT`] rather than
/// [`crate::surface::TERRAIN_ELEVATION_SLOT`], and the two are different numbers for the same
/// picture: a raised mbgl family's elevation has to clear the slots its own generated table uses,
/// and this family declares one texture and has nothing to clear. The ABI holds both constants
/// beside each other and a test there pins the order.
///
/// `texture_id` is mbgl's shader-side id and this family has none, so it is nought. Nothing here
/// reads it -- the slot is `binding`, and the name is what the module declares.
static TERRAIN_MESH_TEXTURES: [ShaderTexture; 1] = [ShaderTexture {
    binding: TERRAIN_MESH_ELEVATION_SLOT,
    texture_id: 0,
    name: "idElevationTexture",
}];

/// One family's module parts.
#[derive(Debug, Clone, Copy)]
pub struct Family {
    /// The shader the producer names on the wire, as `GeometryAdd::builtin_shader` carries it.
    ///
    /// An `i32` rather than a [`BuiltIn`] because not every family is mbgl's. Twenty of these are
    /// -- the enum's own discriminants -- and the ground is not: mbgl has no terrain, so
    /// `terrain_ubo::BUILTIN_TERRAIN_SHADER` is tessella's own 128, past the enum's thirty-six and
    /// not representable in it. The wire has always been an `i32` for this reason, and the ABI
    /// publishes that constant as one.
    pub shader: i32,
    /// Short name, for diagnostics and for matching an oracle case.
    pub name: &'static str,
    /// The uniform blocks the body reads, in binding order.
    pub blocks: &'static [&'static UboLayout],
    /// The attribute table the vertex input is planned against, per vertex.
    pub attributes: &'static [ShaderAttribute],
    /// The same per instance, which is empty for every family but the instanced extrusions.
    ///
    /// mbgl declares these in a second array -- `instanceAttributes` beside `attributes` -- and
    /// the generated tables carry both. A family with any of these is drawn with more than one
    /// instance, which has two consequences nothing else here has: `firstInstance` cannot carry
    /// `ubo_index` (see [`crate::draws`]), and the entries that name a `shared_buffer` are the
    /// fields of an element the body also reads through a storage buffer, because it has to look
    /// at the *next* instance and an attribute fetch cannot look ahead.
    pub instance_attributes: &'static [ShaderAttribute],
    /// The texture table, which names the module's samplers.
    pub textures: &'static [ShaderTexture],
    /// The data-driven paint properties, each of which the module declares as a `paint_<name>`
    /// function reading whichever of its two arms the drawable's descriptors leave.
    ///
    /// Empty for a family whose every attribute is geometry, which is ten of the twenty-one. See
    /// [`Property`] for why both arms are written out rather than derived from the names.
    pub properties: &'static [Property],
    /// The hand-written body.
    pub body: &'static str,
    /// Which surfaces a module exists for.
    pub surfaces: &'static [Surface],
    /// Whether the body hands `place` a height above the surface rather than zero.
    ///
    /// The extrusions, and only them -- `tests/families.rs` pins that. It decides two things: a
    /// family with a height skips [`Surface::Globe`], whose placement reads no `z` at all, and a
    /// frame drawing one needs a depth attachment rather than a stencil-only one.
    ///
    /// The ground is the entry that reads false and still hands `place` a third component. What
    /// it hands is where the surface *is*, read from its own DEM, rather than how far a feature
    /// stands off one -- so neither of the two things above applies to it.
    pub height: bool,
    /// How a draw of this family composites, transcribed from mbgl's own `setColorMode`.
    pub composite: Composite,
}

impl Family {
    /// This family's WGSL module for a surface, with the attributes at `absent` left out.
    ///
    /// `absent` is [`crate::vertices::Plan::absent`] -- the locations no descriptor supplied -- and
    /// an empty slice is the whole family. Which it is decides the module's text, so two
    /// permutations of one family are two modules and [`crate::pipelines::Cache`] keys them apart.
    ///
    /// A shorthand rather than a layer: every field it passes is public, and the hand-assembled
    /// cases in `tests/` call [`crate::shaders::module`] directly with parts no family has.
    ///
    /// # Errors
    ///
    /// [`crate::shaders::Error`], as [`crate::shaders::module`].
    pub fn module(&self, surface: Surface, absent: &[u32]) -> Result<String, shaders::Error> {
        shaders::module(
            surface,
            self.blocks,
            self.attributes,
            self.instance_attributes,
            self.textures,
            &Paint {
                properties: self.properties,
                absent,
            },
            self.body,
        )
    }

    /// Which blend a draw of this family in `pass` composites with.
    ///
    /// `pass` is `batch::Key::pass`, the one bit `DrawOrder::resolve` left -- the same input
    /// [`crate::pipelines::Depth::of`] reads. The translucent bit is what decides, so a pass byte
    /// carrying neither bit blends by replacement; such a drawable is one mbgl does not draw at
    /// all, its `evaluate` having returned `RenderPass::None`.
    #[must_use]
    pub fn blend(&self, pass: u8) -> Blend {
        match self.composite {
            Composite::Fixed(blend) => blend,
            Composite::ByPass => {
                if pass & tessella_capture_abi::RenderPass::TRANSLUCENT.bits() == 0 {
                    Blend::Unblended
                } else {
                    Blend::Alpha
                }
            }
        }
    }

    /// Which of [`Self::surfaces`] a drawable asking for `asked` is drawn on.
    ///
    /// [`Surface::of`] reads the camera and the drawable and names a surface; this is the half
    /// that knows whether this family has a module for it. Not every pair exists -- an extrusion
    /// has no direct bend, and the two offscreen families have only a plane -- while the producer
    /// marks a drawable raised whatever family draws it, so what is asked for is sometimes not one
    /// of the sixty-six. Where it is not, [`Surface::fallback`]'s chain is walked until it is,
    /// which is the picture `DrawFlags::ON_TERRAIN`'s own note promises.
    #[must_use]
    pub fn surface(&self, asked: Surface) -> Surface {
        let mut at = Some(asked);
        while let Some(surface) = at {
            if self.surfaces.contains(&surface) {
                return surface;
            }
            at = surface.fallback();
        }
        // Unreachable while every family declares a plane, which `tests/placement.rs` pins. A
        // return rather than a panic: the flat module is what the chain above exists to reach, and
        // it is the right end of it too.
        Surface::Plane
    }
}

/// The constant arm of a symbol family's property, which is not a field.
///
/// mbgl's three symbol shaders each read `tileProps.is_text ? props.text_opacity :
/// props.icon_opacity`, because a symbol layer's evaluated properties carry both sets and which
/// one applies is a property of the *tile* -- one layer draws text and icons from the same
/// evaluated properties, and the tile props say which this drawable is.
///
/// These are the entries no derivation from a name could have produced, and the reason
/// [`Property`] writes both arms out instead of deriving one.
macro_rules! symbol_constant {
    ($property:literal) => {
        concat!(
            "select(symbol_evaluated_props_ubo[0].icon_",
            $property,
            ", symbol_evaluated_props_ubo[0].text_",
            $property,
            ", symbol_tile_props_ubo[ubo_index].is_text != 0)"
        )
    };
}

/// `fill`'s two: a color and an opacity, which is mbgl's whole set for a plain fill.
const FILL_PAINT: &[Property] = &[
    Property {
        location: 1,
        name: "color",
        returns: "vec4<f32>",
        varies: "mix_color(in.fill_color, fill_drawable_ubo[ubo_index].color_t)",
        constant: "fill_evaluated_props_ubo[0].color",
    },
    Property {
        location: 2,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.fill_opacity, fill_drawable_ubo[ubo_index].opacity_t)",
        constant: "fill_evaluated_props_ubo[0].opacity",
    },
];

/// `fill_outline`'s two. The color is `outline_color` in both blocks and `fill_outline_color` as
/// an attribute, so the name is written here rather than taken from either.
const FILL_OUTLINE_PAINT: &[Property] = &[
    Property {
        location: 1,
        name: "outline_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.fill_outline_color, fill_outline_drawable_ubo[ubo_index].outline_color_t)",
        constant: "fill_evaluated_props_ubo[0].outline_color",
    },
    Property {
        location: 2,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.fill_opacity, fill_outline_drawable_ubo[ubo_index].opacity_t)",
        constant: "fill_evaluated_props_ubo[0].opacity",
    },
];

/// `fill_pattern`'s one. The pattern corners are attributes mbgl never makes uniform -- a
/// sprite's place on the sheet is per tile, not per layer -- so the opacity is the whole set.
const FILL_PATTERN_PAINT: &[Property] = &[Property {
    location: 3,
    name: "opacity",
    returns: "f32",
    varies: "mix_value(in.fill_opacity, fill_pattern_drawable_ubo[ubo_index].opacity_t)",
    constant: "fill_evaluated_props_ubo[0].opacity",
}];

/// `fill_extrusion`'s three: the color, and the two heights a wall is extruded between.
const FILL_EXTRUSION_PAINT: &[Property] = &[
    Property {
        location: 2,
        name: "color",
        returns: "vec4<f32>",
        varies: "mix_color(in.fill_extrusion_color, fill_extrusion_drawable_ubo[ubo_index].color_t)",
        constant: "fill_extrusion_props_ubo[0].color",
    },
    Property {
        location: 3,
        name: "base",
        returns: "f32",
        varies: "mix_value(in.fill_extrusion_base, fill_extrusion_drawable_ubo[ubo_index].base_t)",
        constant: "fill_extrusion_props_ubo[0].base",
    },
    Property {
        location: 4,
        name: "height",
        returns: "f32",
        varies: "mix_value(in.fill_extrusion_height, fill_extrusion_drawable_ubo[ubo_index].height_t)",
        constant: "fill_extrusion_props_ubo[0].height",
    },
];

/// `fill_extrusion_instanced`'s three: the same properties as the plain walls, two locations
/// higher, because the outline position and the packed decimals come before them in the
/// instance table.
const FILL_EXTRUSION_INSTANCED_PAINT: &[Property] = &[
    Property {
        location: 3,
        name: "color",
        returns: "vec4<f32>",
        varies: "mix_color(in.fill_extrusion_color, fill_extrusion_drawable_ubo[ubo_index].color_t)",
        constant: "fill_extrusion_props_ubo[0].color",
    },
    Property {
        location: 4,
        name: "base",
        returns: "f32",
        varies: "mix_value(in.fill_extrusion_base, fill_extrusion_drawable_ubo[ubo_index].base_t)",
        constant: "fill_extrusion_props_ubo[0].base",
    },
    Property {
        location: 5,
        name: "height",
        returns: "f32",
        varies: "mix_value(in.fill_extrusion_height, fill_extrusion_drawable_ubo[ubo_index].height_t)",
        constant: "fill_extrusion_props_ubo[0].height",
    },
];

/// `line`'s six. `gapwidth` is one word in both blocks and two in the attribute, which is mbgl's
/// spelling in each place and not a typo to tidy.
const LINE_PAINT: &[Property] = &[
    Property {
        location: 2,
        name: "color",
        returns: "vec4<f32>",
        varies: "mix_color(in.line_color, line_drawable_ubo[ubo_index].color_t)",
        constant: "line_evaluated_props_ubo[0].color",
    },
    Property {
        location: 3,
        name: "blur",
        returns: "f32",
        varies: "mix_value(in.line_blur, line_drawable_ubo[ubo_index].blur_t)",
        constant: "line_evaluated_props_ubo[0].blur",
    },
    Property {
        location: 4,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.line_opacity, line_drawable_ubo[ubo_index].opacity_t)",
        constant: "line_evaluated_props_ubo[0].opacity",
    },
    Property {
        location: 5,
        name: "gapwidth",
        returns: "f32",
        varies: "mix_value(in.line_gap_width, line_drawable_ubo[ubo_index].gapwidth_t)",
        constant: "line_evaluated_props_ubo[0].gapwidth",
    },
    Property {
        location: 6,
        name: "offset",
        returns: "f32",
        varies: "mix_value(in.line_offset, line_drawable_ubo[ubo_index].offset_t)",
        constant: "line_evaluated_props_ubo[0].offset",
    },
    Property {
        location: 7,
        name: "width",
        returns: "f32",
        varies: "mix_value(in.line_width, line_drawable_ubo[ubo_index].width_t)",
        constant: "line_evaluated_props_ubo[0].width",
    },
];

/// `line_pattern`'s five: [`LINE_PAINT`] without the color, which a patterned line has no block
/// field for -- the sprite is the color.
const LINE_PATTERN_PAINT: &[Property] = &[
    Property {
        location: 2,
        name: "blur",
        returns: "f32",
        varies: "mix_value(in.line_blur, line_pattern_drawable_ubo[ubo_index].blur_t)",
        constant: "line_evaluated_props_ubo[0].blur",
    },
    Property {
        location: 3,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.line_opacity, line_pattern_drawable_ubo[ubo_index].opacity_t)",
        constant: "line_evaluated_props_ubo[0].opacity",
    },
    Property {
        location: 4,
        name: "gapwidth",
        returns: "f32",
        varies: "mix_value(in.line_gap_width, line_pattern_drawable_ubo[ubo_index].gapwidth_t)",
        constant: "line_evaluated_props_ubo[0].gapwidth",
    },
    Property {
        location: 5,
        name: "offset",
        returns: "f32",
        varies: "mix_value(in.line_offset, line_pattern_drawable_ubo[ubo_index].offset_t)",
        constant: "line_evaluated_props_ubo[0].offset",
    },
    Property {
        location: 6,
        name: "width",
        returns: "f32",
        varies: "mix_value(in.line_width, line_pattern_drawable_ubo[ubo_index].width_t)",
        constant: "line_evaluated_props_ubo[0].width",
    },
];

/// `circle`'s seven, which is every attribute it declares but the position.
const CIRCLE_PAINT: &[Property] = &[
    Property {
        location: 1,
        name: "color",
        returns: "vec4<f32>",
        varies: "mix_color(in.circle_color, circle_drawable_ubo[ubo_index].color_t)",
        constant: "circle_evaluated_props_ubo[0].color",
    },
    Property {
        location: 2,
        name: "radius",
        returns: "f32",
        varies: "mix_value(in.circle_radius, circle_drawable_ubo[ubo_index].radius_t)",
        constant: "circle_evaluated_props_ubo[0].radius",
    },
    Property {
        location: 3,
        name: "blur",
        returns: "f32",
        varies: "mix_value(in.circle_blur, circle_drawable_ubo[ubo_index].blur_t)",
        constant: "circle_evaluated_props_ubo[0].blur",
    },
    Property {
        location: 4,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.circle_opacity, circle_drawable_ubo[ubo_index].opacity_t)",
        constant: "circle_evaluated_props_ubo[0].opacity",
    },
    Property {
        location: 5,
        name: "stroke_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.circle_stroke_color, circle_drawable_ubo[ubo_index].stroke_color_t)",
        constant: "circle_evaluated_props_ubo[0].stroke_color",
    },
    Property {
        location: 6,
        name: "stroke_width",
        returns: "f32",
        varies: "mix_value(in.circle_stroke_width, circle_drawable_ubo[ubo_index].stroke_width_t)",
        constant: "circle_evaluated_props_ubo[0].stroke_width",
    },
    Property {
        location: 7,
        name: "stroke_opacity",
        returns: "f32",
        varies: "mix_value(in.circle_stroke_opacity, circle_drawable_ubo[ubo_index].stroke_opacity_t)",
        constant: "circle_evaluated_props_ubo[0].stroke_opacity",
    },
];

/// `heatmap`'s two: a point's weight and its radius.
const HEATMAP_PAINT: &[Property] = &[
    Property {
        location: 1,
        name: "weight",
        returns: "f32",
        varies: "mix_value(in.heatmap_weight, heatmap_drawable_ubo[ubo_index].weight_t)",
        constant: "heatmap_evaluated_props_ubo[0].weight",
    },
    Property {
        location: 2,
        name: "radius",
        returns: "f32",
        varies: "mix_value(in.heatmap_radius, heatmap_drawable_ubo[ubo_index].radius_t)",
        constant: "heatmap_evaluated_props_ubo[0].radius",
    },
];

/// `symbol_icon`'s one, through [`symbol_constant`].
const SYMBOL_ICON_PAINT: &[Property] = &[Property {
    location: 5,
    name: "opacity",
    returns: "f32",
    varies: "mix_value(in.symbol_opacity, symbol_drawable_ubo[ubo_index].opacity_t)",
    constant: symbol_constant!("opacity"),
}];

/// `symbol_sdf`'s five: the glyph's fill and halo colors, and the three scalars that shape the
/// halo.
const SYMBOL_SDF_PAINT: &[Property] = &[
    Property {
        location: 5,
        name: "fill_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.symbol_color, symbol_drawable_ubo[ubo_index].fill_color_t)",
        constant: symbol_constant!("fill_color"),
    },
    Property {
        location: 6,
        name: "halo_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.symbol_halo_color, symbol_drawable_ubo[ubo_index].halo_color_t)",
        constant: symbol_constant!("halo_color"),
    },
    Property {
        location: 7,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.symbol_opacity, symbol_drawable_ubo[ubo_index].opacity_t)",
        constant: symbol_constant!("opacity"),
    },
    Property {
        location: 8,
        name: "halo_width",
        returns: "f32",
        varies: "mix_value(in.symbol_halo_width, symbol_drawable_ubo[ubo_index].halo_width_t)",
        constant: symbol_constant!("halo_width"),
    },
    Property {
        location: 9,
        name: "halo_blur",
        returns: "f32",
        varies: "mix_value(in.symbol_halo_blur, symbol_drawable_ubo[ubo_index].halo_blur_t)",
        constant: symbol_constant!("halo_blur"),
    },
];

/// `symbol_text_and_icon`'s five: [`SYMBOL_SDF_PAINT`]'s properties, each one location lower.
/// This family declares no pixel offset, so everything after it shifts down.
const SYMBOL_TEXT_AND_ICON_PAINT: &[Property] = &[
    Property {
        location: 4,
        name: "fill_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.symbol_color, symbol_drawable_ubo[ubo_index].fill_color_t)",
        constant: symbol_constant!("fill_color"),
    },
    Property {
        location: 5,
        name: "halo_color",
        returns: "vec4<f32>",
        varies: "mix_color(in.symbol_halo_color, symbol_drawable_ubo[ubo_index].halo_color_t)",
        constant: symbol_constant!("halo_color"),
    },
    Property {
        location: 6,
        name: "opacity",
        returns: "f32",
        varies: "mix_value(in.symbol_opacity, symbol_drawable_ubo[ubo_index].opacity_t)",
        constant: symbol_constant!("opacity"),
    },
    Property {
        location: 7,
        name: "halo_width",
        returns: "f32",
        varies: "mix_value(in.symbol_halo_width, symbol_drawable_ubo[ubo_index].halo_width_t)",
        constant: symbol_constant!("halo_width"),
    },
    Property {
        location: 8,
        name: "halo_blur",
        returns: "f32",
        varies: "mix_value(in.symbol_halo_blur, symbol_drawable_ubo[ubo_index].halo_blur_t)",
        constant: symbol_constant!("halo_blur"),
    },
];

/// Every family this crate draws, in no particular order.
pub static ALL: &[Family] = &[
    Family {
        shader: BuiltIn::BackgroundShader as i32,
        name: "background",
        blocks: &[&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
        attributes: &BACKGROUND_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: &[],
        body: BACKGROUND_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::BackgroundPatternShader as i32,
        name: "background_pattern",
        blocks: &[
            &BACKGROUND_PATTERN_DRAWABLE_UBO,
            &BACKGROUND_PATTERN_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &BACKGROUND_PATTERN_SHADER,
        instance_attributes: &[],
        textures: &BACKGROUND_PATTERN_SHADER_TEXTURES,
        properties: &[],
        body: BACKGROUND_PATTERN_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::FillShader as i32,
        name: "fill",
        blocks: &[&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
        attributes: &FILL_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: FILL_PAINT,
        body: FILL_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::FillOutlineShader as i32,
        name: "fill_outline",
        blocks: &[
            &FILL_OUTLINE_DRAWABLE_UBO,
            &FILL_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &FILL_OUTLINE_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: FILL_OUTLINE_PAINT,
        body: FILL_OUTLINE_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::FillOutlineTriangulatedShader as i32,
        name: "fill_outline_triangulated",
        // The same three `fill_outline` reads, which is what the GL source declares and what
        // fluorite's material takes: the matrix and ratio from the drawable, the outline color
        // and opacity from the evaluated props, the pixel ratio and units-to-pixels from the
        // globals.
        blocks: &[
            &FILL_OUTLINE_TRIANGULATED_DRAWABLE_UBO,
            &FILL_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &FILL_OUTLINE_TRIANGULATED_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: &[],
        body: FILL_OUTLINE_TRIANGULATED_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::FillPatternShader as i32,
        name: "fill_pattern",
        blocks: &[
            &FILL_PATTERN_DRAWABLE_UBO,
            &FILL_PATTERN_TILE_PROPS_UBO,
            &FILL_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &FILL_PATTERN_SHADER,
        instance_attributes: &[],
        textures: &FILL_PATTERN_SHADER_TEXTURES,
        properties: FILL_PATTERN_PAINT,
        body: FILL_PATTERN_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::FillExtrusionShader as i32,
        name: "fill_extrusion",
        blocks: &[&FILL_EXTRUSION_DRAWABLE_UBO, &FILL_EXTRUSION_PROPS_UBO],
        attributes: &FILL_EXTRUSION_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: FILL_EXTRUSION_PAINT,
        body: FILL_EXTRUSION_BODY,
        surfaces: LIFTS_A_HEIGHT,
        height: true,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::FillExtrusionInstancedShader as i32,
        name: "fill_extrusion_instanced",
        blocks: &[&FILL_EXTRUSION_DRAWABLE_UBO, &FILL_EXTRUSION_PROPS_UBO],
        attributes: &FILL_EXTRUSION_INSTANCED_SHADER,
        instance_attributes: &FILL_EXTRUSION_INSTANCED_SHADER_INSTANCE,
        textures: &[],
        properties: FILL_EXTRUSION_INSTANCED_PAINT,
        body: FILL_EXTRUSION_INSTANCED_BODY,
        surfaces: LIFTS_A_HEIGHT,
        height: true,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::LineShader as i32,
        name: "line",
        blocks: &[
            &LINE_DRAWABLE_UBO,
            &LINE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &LINE_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: LINE_PAINT,
        body: LINE_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::LinePatternShader as i32,
        name: "line_pattern",
        blocks: &[
            &LINE_PATTERN_DRAWABLE_UBO,
            &LINE_PATTERN_TILE_PROPS_UBO,
            &LINE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &LINE_PATTERN_SHADER,
        instance_attributes: &[],
        textures: &LINE_PATTERN_SHADER_TEXTURES,
        properties: LINE_PATTERN_PAINT,
        body: LINE_PATTERN_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::CircleShader as i32,
        name: "circle",
        blocks: &[
            &CIRCLE_DRAWABLE_UBO,
            &CIRCLE_EVALUATED_PROPS_UBO,
            &GLOBAL_PAINT_PARAMS_UBO,
        ],
        attributes: &CIRCLE_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: CIRCLE_PAINT,
        body: CIRCLE_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::RasterShader as i32,
        name: "raster",
        blocks: &[&RASTER_DRAWABLE_UBO, &RASTER_EVALUATED_PROPS_UBO],
        attributes: &RASTER_SHADER,
        instance_attributes: &[],
        textures: &RASTER_SHADER_TEXTURES,
        properties: &[],
        body: RASTER_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::HillshadePrepareShader as i32,
        name: "hillshade_prepare",
        blocks: &[
            &HILLSHADE_PREPARE_DRAWABLE_UBO,
            &HILLSHADE_PREPARE_TILE_PROPS_UBO,
        ],
        attributes: &HILLSHADE_PREPARE_SHADER,
        instance_attributes: &[],
        textures: &HILLSHADE_PREPARE_SHADER_TEXTURES,
        properties: &[],
        body: HILLSHADE_PREPARE_BODY,
        surfaces: FLAT,
        height: false,
        // Into its own target, where there is nothing to composite over.
        composite: Composite::Fixed(Blend::Unblended),
    },
    Family {
        shader: BuiltIn::HillshadeShader as i32,
        name: "hillshade",
        blocks: &[
            &HILLSHADE_DRAWABLE_UBO,
            &HILLSHADE_TILE_PROPS_UBO,
            &HILLSHADE_EVALUATED_PROPS_UBO,
        ],
        attributes: &HILLSHADE_SHADER,
        instance_attributes: &[],
        textures: &HILLSHADE_SHADER_TEXTURES,
        properties: &[],
        body: HILLSHADE_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::ColorReliefShader as i32,
        name: "color_relief",
        blocks: &[
            &COLOR_RELIEF_DRAWABLE_UBO,
            &COLOR_RELIEF_TILE_PROPS_UBO,
            &COLOR_RELIEF_EVALUATED_PROPS_UBO,
        ],
        attributes: &COLOR_RELIEF_SHADER,
        instance_attributes: &[],
        textures: &COLOR_RELIEF_SHADER_TEXTURES,
        properties: &[],
        body: COLOR_RELIEF_BODY,
        surfaces: UNANCHORED,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::HeatmapShader as i32,
        name: "heatmap",
        blocks: &[&HEATMAP_DRAWABLE_UBO, &HEATMAP_EVALUATED_PROPS_UBO],
        attributes: &HEATMAP_SHADER,
        instance_attributes: &[],
        textures: &[],
        properties: HEATMAP_PAINT,
        body: HEATMAP_BODY,
        surfaces: FLAT,
        height: false,
        // Every kernel brightens the same texel, which is what the ramp then reads.
        composite: Composite::Fixed(Blend::Additive),
    },
    Family {
        shader: BuiltIn::HeatmapTextureShader as i32,
        name: "heatmap_texture",
        blocks: &[&HEATMAP_TEXTURE_PROPS_UBO, &GLOBAL_PAINT_PARAMS_UBO],
        attributes: &HEATMAP_TEXTURE_SHADER,
        instance_attributes: &[],
        textures: &HEATMAP_TEXTURE_SHADER_TEXTURES,
        properties: &[],
        body: HEATMAP_TEXTURE_BODY,
        surfaces: FLAT_OR_BENT,
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
    },
    Family {
        shader: BuiltIn::SymbolIconShader as i32,
        name: "symbol_icon",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_ICON_SHADER,
        instance_attributes: &[],
        textures: &SYMBOL_ICON_SHADER_TEXTURES,
        properties: SYMBOL_ICON_PAINT,
        body: SYMBOL_ICON_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::SymbolSDFShader as i32,
        name: "symbol_sdf",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_SDFSHADER,
        instance_attributes: &[],
        textures: &SYMBOL_SDFSHADER_TEXTURES,
        properties: SYMBOL_SDF_PAINT,
        body: SYMBOL_SDF_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::ByPass,
    },
    Family {
        shader: BuiltIn::SymbolTextAndIconShader as i32,
        name: "symbol_text_and_icon",
        blocks: SYMBOL_BLOCKS,
        attributes: &SYMBOL_TEXT_AND_ICON_SHADER,
        instance_attributes: &[],
        textures: &SYMBOL_TEXT_AND_ICON_SHADER_TEXTURES,
        properties: SYMBOL_TEXT_AND_ICON_PAINT,
        body: SYMBOL_TEXT_AND_ICON_BODY,
        surfaces: EVERY,
        height: false,
        composite: Composite::ByPass,
    },
    // The ground, which mbgl has no shader for at all. Every field of it is tessella's: the id
    // past the enum's thirty-six, the attribute table published beside it in the ABI, and the
    // block the raised families read -- which this one reads as its own, because it is the
    // surface they are placed on rather than something placed on a surface.
    Family {
        shader: BUILTIN_TERRAIN_SHADER,
        name: "terrain",
        // The same block a raised drawable's placement reads, declared here as the family's own:
        // the matrix, the DEM's unpack vector, what bare ground is painted, the sampling pair and
        // the skirt. One block and no evaluated properties -- the spec gives terrain no paint, so
        // there is no layer-wide set for a second block to carry.
        blocks: &[&TERRAIN_DRAWABLE_UBO],
        attributes: &TERRAIN_MESH_ATTRIBUTES,
        instance_attributes: &[],
        textures: &TERRAIN_MESH_TEXTURES,
        // Nothing data-driven: the ground has no features for a property to vary over, which is
        // what `encode_terrain` sends a `permutation_key` of nought for.
        properties: &[],
        body: TERRAIN_BODY,
        surfaces: THE_GROUND,
        // Not a height above the surface, though the body does hand `place` a z: see the field.
        height: false,
        composite: Composite::Fixed(Blend::Alpha),
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
///
/// # "The producer emits no drawable for it yet" is a claim, and one of them was wrong
///
/// Until a real frame could be replayed here, that reason was unfalsifiable: every fixture writes
/// the records its author chose, so a family nothing sent looked the same as a family nothing
/// *will* send. `benches/capture_frame.rs` replays a capture of a real style and reports the
/// shaders it names that have no family -- and a Berlin z14 frame through `families_p` named
/// `FillOutlineTriangulatedShader` twice and `FillExtrusionInstancedShader` twice.
///
/// Both are drawn now, and that frame plans and uploads all nineteen of its geometries. What is
/// left here is claims again: a capture of a style that uses one of these is what would settle it.
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

/// The family for one of mbgl's shaders, or `None` if this crate does not draw it.
///
/// [`for_wire`] is the other half and the one a stream reaches for: a `builtin_shader` is an `i32`
/// and not every family has a `BuiltIn` to name it by.
#[must_use]
pub fn family(shader: BuiltIn) -> Option<&'static Family> {
    for_wire(shader as i32)
}

/// The family for a `builtin_shader` as it arrives on the wire.
///
/// `None` for an id this crate does not draw, which an order may name: a build without a family is
/// a drawable to leave out rather than an error to raise.
///
/// The id is compared rather than decoded. It was `BuiltIn::from_repr(..).and_then(family)`, which
/// is the same answer for the twenty mbgl families and `None` for every id that is not mbgl's --
/// including the ground's, which is the one this finds here and that could not.
#[must_use]
pub fn for_wire(builtin_shader: i32) -> Option<&'static Family> {
    ALL.iter().find(|family| family.shader == builtin_shader)
}
