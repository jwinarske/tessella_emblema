//! The readback oracle: families drawn, and their pixels checked.
//!
//! Every test here reads the shader. This one runs it. A module can compile to valid SPIR-V,
//! satisfy every pin, and still draw the wrong thing — because the pipeline fetched an attribute at
//! a format the shader reads differently, or because a block the producer wrote and the block the
//! shader declared agree about offsets but not about which binding they arrive at. None of that is
//! visible from the source.
//!
//! So each case below draws into a 32x32 target with inputs chosen to make one pixel's value
//! predictable, and checks that pixel. `cargo bench --bench first_pixel` runs it; a bench rather
//! than a test because it needs a GPU and CI has none.
//!
//! # What a pass proves
//!
//! That the module builds a pipeline on a real driver; that the vertex input built from the ABI's
//! declared types delivers what the body reads; that a block written at the ABI's own offsets is
//! read back by the generated WGSL struct; and that `instance_index` reaches `ubo_index`. Each
//! case adds whatever its own body computes on top of that.
//!
//! # Through the library, not beside it
//!
//! Every step between a case's inputs and its pixel is the library's: `vertices::plan` over the
//! `AttributeDesc`s a producer would have sent, `buffers::needs` and [`Store`] for the geometry,
//! [`blocks`] for the uniform buffers at the slots the wire names them, [`Images`] for the
//! textures, `descriptors::Sets` for the set, [`Cache`] for the pipeline and `target::frame` for
//! the pass. What the probe still owns is the *inputs* and the one pixel to read.
//!
//! It owned all of it once -- about a thousand lines of its own Vulkan, from the instance to the
//! render pass to a linear-tiled image it mapped. Those pixels said the eighteen families' bodies
//! and the ABI's tables agree, which is worth having, and said nothing at all about the code that
//! will draw them in a frame. `descriptors::Sets::write` pointed every block binding at one buffer
//! for four merged slices and no test could see it; this bench now fails on the second case.
//!
//! The substitution was measured rather than assumed: all twenty-six expected pixels are the ones
//! derived by hand against the probe's own Vulkan, unchanged.
//!
//! # What it still does not reach
//!
//! The join. A case names its own geometry, blocks and textures, where a frame gets them from a
//! capture stream through `Joiner` and `Batches` and draws them through `record::content`. So this
//! says a family draws right when it is set up right, and not that a stream sets it up right.
//!
//! And one driver quirk is lost with the probe's own pipeline: it compiled the SPIR-V into a module
//! per stage, because Adreno rejects a module with two entry points. `pipelines::build` uses one
//! module for both, so this no longer runs on the sa8155p -- which is the library's gap rather than
//! the bench's, and is tracked as #85.
//!
//! # A case has to be sensitive to what it claims to read
//!
//! Twice on 2026-10-04 a case drew the right pixel while being blind to one of its inputs, both
//! times because a sampled coordinate sat on a texel boundary and two different values reached
//! the same texel. `color_relief`'s coordinate lands 8e-6 from one; `raster_interleaved` could
//! not tell a four-byte offset error until the bytes nothing binds were filled with a value that
//! samples a *different* texel.
//!
//! `raster_interleaved` was one-sided for the same reason until #84: it caught an attribute read
//! four bytes *late* and not one read four bytes *early*, because the position read as a coordinate
//! landed on the texel boundary that resolves to the texel the case meant to read. Its three values
//! now sample three texels, and the two directions give two different wrong pixels.
//!
//! # And the pass has to be sensitive to how the draw composites
//!
//! The target is cleared to an *opaque* color for the same reason. The pipelines are built
//! `Blend::Unblended`, which this bench's note called deliberate -- and over the transparent black
//! it used to clear to, it was unmeasurable: mbgl's alpha mode is premultiplied, so blending gives
//! `src * 1 + 0 * (1 - srcAlpha)`, which is `src` for any alpha. Every case passed with
//! `Blend::Alpha` too, and with `Additive`. Over an opaque clear both are caught on the first case
//! that returns an alpha below 255.
//!
//! That also sharpened `symbol_below`, whose note said "nothing is drawn" and was wrong: with an
//! opaque clear the pixel still reads zero, so the glyph's quad does cover it and the body wrote a
//! transparent black. Three outcomes are distinguishable there now where two were.
//!
//! So a new case is not finished when it draws the expected pixel. Nudge each input to a
//! neighbouring value and check the pixel moves: a texel of each image, each stream, each uniform
//! field the body reads. An input that can be changed without moving the pixel is not under test,
//! whatever the case's comment says.
//!
//! Every case here was audited that way for its images, one texel at a time. Each moves on
//! exactly the texels it should, and the three that move on more than one have reasons:
//! `hillshade_prepare` reads a 3x3 neighbourhood to compute a slope, `color_relief` mixes two
//! ramp stops, and a raised surface samples the DEM per vertex and interpolates, so several
//! texels reach one pixel. The two text-and-icon cases are the sharpest result: `both_as_icon`
//! moves only on the icon sheet and `both_as_glyph` only on the glyph atlas, which is the pair
//! saying the family picks its atlas per vertex rather than per draw.
//!
//! # What a pass does not prove
//!
//! Anything about a family not listed, and nothing about a tiler. The device is chosen external,
//! then internal, then software — see `device::preferred` — so a run says which tier answered.
//! `TSL_VULKAN_LIB` names a driver to open directly, for an image whose loader and driver
//! disagree; see `draw_cost.rs`.

use ash::vk;
use tessella_capture_abi::envelope::{
    AttributeDesc, Extent, GeometryId, Rect16, SlabRef, TextureFilter, TextureId, TextureRef,
    ViewId,
};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::shader_attributes::ShaderAttribute;
use tessella_capture_abi::generated::ubo_layouts::{
    BACKGROUND_DRAWABLE_UBO, BACKGROUND_PATTERN_DRAWABLE_UBO, BACKGROUND_PATTERN_PROPS_UBO,
    BACKGROUND_PROPS_UBO, CIRCLE_DRAWABLE_UBO, CIRCLE_EVALUATED_PROPS_UBO,
    COLOR_RELIEF_DRAWABLE_UBO, COLOR_RELIEF_EVALUATED_PROPS_UBO, COLOR_RELIEF_TILE_PROPS_UBO,
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, FILL_EXTRUSION_DRAWABLE_UBO,
    FILL_EXTRUSION_PROPS_UBO, FILL_OUTLINE_DRAWABLE_UBO, FILL_PATTERN_DRAWABLE_UBO,
    FILL_PATTERN_TILE_PROPS_UBO, GLOBAL_PAINT_PARAMS_UBO, HEATMAP_DRAWABLE_UBO,
    HEATMAP_EVALUATED_PROPS_UBO, HEATMAP_TEXTURE_PROPS_UBO, HILLSHADE_DRAWABLE_UBO,
    HILLSHADE_EVALUATED_PROPS_UBO, HILLSHADE_PREPARE_DRAWABLE_UBO,
    HILLSHADE_PREPARE_TILE_PROPS_UBO, HILLSHADE_TILE_PROPS_UBO, LINE_EVALUATED_PROPS_UBO,
    LINE_PATTERN_DRAWABLE_UBO, LINE_PATTERN_TILE_PROPS_UBO, RASTER_DRAWABLE_UBO,
    RASTER_EVALUATED_PROPS_UBO, SYMBOL_DRAWABLE_UBO, SYMBOL_EVALUATED_PROPS_UBO,
    SYMBOL_TILE_PROPS_UBO, UboLayout,
};
use tessella_capture_abi::{TextureChannelDataType, TexturePixelType};
use tessella_emblema::device::{self, Attachment, check_vertex_formats, vertex_format};
use tessella_emblema::families::family;
use tessella_emblema::images::Images;
use tessella_emblema::pipelines::{Blend, Cache, Targets};
use tessella_emblema::shaders::module;
use tessella_emblema::store::Store;
use tessella_emblema::surface::{GLOBE_BEND_UBO, GLOBE_CAMERA_UBO, Surface, TERRAIN_DRAWABLE_UBO};
use tessella_emblema::target::{self, Depth, Host};
use tessella_emblema::{blocks, buffers, descriptors, pipelines, slots, vertices};
// The probe's own `Image` is a case's input; the wrapper's is a device object. Aliased rather
// than renamed, because the twenty-six cases name theirs.
use tessella_vk::{Buffer, Image as Device2d, ImageView, Memory, Recorder};

mod common;
use common::Open;

/// The target's edge, in pixels. Small: one pixel is read and the rest is margin.
const SIDE: u32 = 32;

/// How a case's vertex data is laid out in memory.
///
/// Both forms draw the same picture from the same numbers, which is what makes the interleaved
/// case worth having: it is the layout the producer actually sends for the raster families, and
/// until it was drawn here the oracle only ever bound one buffer per attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// One buffer per attribute, each at offset zero with its own width as the stride.
    PerAttribute,
    /// One buffer for every attribute, each at its own offset within a vertex of `stride`.
    ///
    /// What `encode_raster`, `encode_hillshade` and `encode_color_relief` send: the position at 0,
    /// the texture coordinate at 4 and the skirt flag at 8, over a stride of 12. The offsets are
    /// per declared attribute in table order; bytes no attribute names are skipped, which is how
    /// the skirt's four travel without being bound.
    Interleaved {
        /// Bytes between consecutive vertices.
        stride: u32,
        /// Byte offset of each declared attribute within a vertex, in table order.
        offsets: &'static [u32],
    },
}

/// A family, the inputs that make one pixel predictable, and that pixel.
struct Case {
    name: &'static str,
    /// Which pixel to read, as a column and a row.
    ///
    /// The center for almost every case: a family's own arithmetic is read best where the geometry
    /// certainly covers. A surface case is the exception -- what it checks is *where* the geometry
    /// went, so it reads a pixel chosen for being covered by the right placement and by no wrong
    /// one, which is rarely the middle.
    at: (u32, u32),
    /// Which surface the family is assembled against, and so where its vertices land.
    ///
    /// Every case but the globe one below is a plane. That is not a preference: a surface is only
    /// observable through *where* geometry lands, and the families' own arithmetic is easier to
    /// read off a surface that does nothing.
    surface: Surface,
    /// Which family's module to draw, resolved through `families::family`.
    ///
    /// Named rather than restated: the oracle's value is that it runs the modules the library
    /// ships, and a case carrying its own copy of a family's blocks, tables and body would
    /// verify the copy.
    family: BuiltIn,
    /// How `streams` is laid out: one buffer per attribute, or one shared interleaved buffer.
    layout: Layout,
    /// One stream per attribute in the table's order, or one interleaved stream.
    streams: Vec<Vec<u8>>,
    /// One block per entry in the family's `blocks`, in the same order.
    uniforms: Vec<Vec<u8>>,
    /// One image per entry in `textures`, in the same order.
    images: Vec<Image>,
    vertices: u32,
    /// Which entry of the blocks the draw reads, which the body receives as `ubo_index`.
    ///
    /// Zero for every case but one. `drawable_at_one` is the exception and the reason this is a
    /// field: entry zero sits at offset zero under any stride, so a case at zero cannot tell what
    /// stride the entries are packed at -- see #81.
    ubo_index: u32,
    expect: [u8; 4],
}

/// An image the probe samples.
///
/// Small, with every texel distinct where a case needs to tell which one was read. A ramp is one
/// of these too -- wide and a few rows tall rather than square, because a ramp's second coordinate
/// is a constant the body chooses and a single row would make any choice look right.
struct Image {
    width: u32,
    height: u32,
    texels: Texels,
}

/// What an image holds, which decides its format.
///
/// Three, because the producer uploads three. mbgl's `Texture2D::setFormat` takes a pixel type
/// *and* a channel type, and the families here use three of the combinations: bytes in four
/// channels for a sheet or a tile, bytes in one for a glyph atlas, and floats in four for a color
/// relief's elevation stops -- which are meters over a range that spans the planet, where eight
/// bits would be a forty-meter step.
enum Texels {
    /// `R8G8B8A8_UNORM`: a sheet, a tile, a color ramp.
    Rgba(Vec<u8>),
    /// `R8_UNORM`: one channel, which is what a glyph atlas is.
    Red(Vec<u8>),
    /// `R32G32B32A32_SFLOAT`: four floats a texel, which is `setFormat(RGBA, Float)`.
    Floats(Vec<f32>),
}

impl Texels {
    /// The pair the producer sends, which is what the format is derived *from*.
    ///
    /// Not a `vk::Format` any more. mbgl's `Texture2D::setFormat` takes a pixel type and a channel
    /// type and `device::texture_format` is the mapping between them, so a probe naming the Vulkan
    /// format would be asserting its own copy of that mapping rather than the library's.
    fn kinds(&self) -> (TexturePixelType, TextureChannelDataType) {
        match self {
            Self::Rgba(_) => (TexturePixelType::RGBA, TextureChannelDataType::UnsignedByte),
            Self::Red(_) => (
                TexturePixelType::Alpha,
                TextureChannelDataType::UnsignedByte,
            ),
            Self::Floats(_) => (TexturePixelType::RGBA, TextureChannelDataType::Float),
        }
    }

    /// Bytes a texel, which is what a row's stride is counted in.
    fn width(&self) -> usize {
        match self {
            Self::Rgba(_) => 4,
            Self::Red(_) => 1,
            Self::Floats(_) => 16,
        }
    }

    /// The bytes to write, little-endian, as the producer has them.
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Rgba(bytes) | Self::Red(bytes) => bytes.clone(),
            Self::Floats(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }
}

impl Image {
    /// A square `R8G8B8A8_UNORM` image whose texel `(x, y)` is `paint(x, y)`.
    fn new(side: u32, paint: impl Fn(u32, u32) -> [u8; 4]) -> Self {
        Self::sized(side, side, paint)
    }

    /// The same, for an image that is not square.
    fn sized(width: u32, height: u32, paint: impl Fn(u32, u32) -> [u8; 4]) -> Self {
        let mut texels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                texels.extend_from_slice(&paint(x, y));
            }
        }
        Self {
            width,
            height,
            texels: Texels::Rgba(texels),
        }
    }

    /// A row of four-float texels, which is how a color relief's elevation stops arrive.
    fn floats(values: &[[f32; 4]]) -> Self {
        Self {
            width: u32::try_from(values.len()).unwrap_or(1),
            height: 1,
            texels: Texels::Floats(values.iter().flatten().copied().collect()),
        }
    }

    /// One byte a texel, which is how a glyph atlas arrives.
    fn red(width: u32, height: u32, paint: impl Fn(u32, u32) -> u8) -> Self {
        let mut texels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                texels.push(paint(x, y));
            }
        }
        Self {
            width,
            height,
            texels: Texels::Red(texels),
        }
    }
}

/// A value written into a block at a named field.
enum At<'a> {
    /// Floats, which is most of what a block holds.
    F(&'a [f32]),
    /// Signed integers, which is what the flags are.
    I(&'a [i32]),
}

/// The identity with `y` negated, so a tile position is already a clip position.
///
/// Every case uses it, which keeps each one's expected pixel independent of a projection the probe
/// would otherwise also have to be right about.
///
/// # There are two flips, and they cancel
///
/// mbgl's shaders end their vertex stage with `gl_Position.y *= -1.0`, because the matrix they are
/// handed was built for a `y`-up convention. emblema has no such step in its bodies -- but naga's
/// SPIR-V backend emits one, under `WriterFlags::ADJUST_COORDINATE_SPACE`, after the body has run.
/// See `tests/naga_overrides.rs`.
///
/// So this matrix's negation and naga's negation cancel: **the net mapping a case's geometry sees
/// is `y`-up**, and a position of `+1` lands at the top of the target. Measured, not assumed --
/// the `heatmap` case samples its kernel off-center and so can tell, and it read 80 where a
/// `y`-down mapping would have given 75.
///
/// Every other case is symmetric in `y` and cannot tell, which is why this went unnoticed through
/// nine of them. It is left as it is rather than simplified to the identity, because
/// `fill_outline` reads its own position back and the three signs together are what make its
/// feather land -- see `SYMBOL_SDF_BODY`'s sibling note in `FILL_OUTLINE_BODY`.
const CLIP: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// Three unflagged vertices' skirt attribute, as `Short2` pairs.
///
/// Every raster-family case on a plane carries this: the producer sends the flag for every bucket
/// of those families and the table declares it, so the pipeline has an input for it whether the
/// surface hangs anything from it or not. Zero because an unraised tile has no crack to cover --
/// `curtain` on a plane returns nought regardless, which is what `raster_raised_skirt` and these
/// cases together say.
const FLAT_SKIRT: [u8; 12] = [0; 12];

/// A triangle covering the viewport, in tile units that the identity makes clip units.
const COVERING: [i16; 6] = [-1, -1, 3, -1, -1, 3];

/// A triangle covering the viewport in a coordinate space of nought to one.
///
/// `heatmap_texture`'s quad is the viewport rather than a tile, so its positions are a unit square
/// and the matrix is what reaches clip space. See `UNIT_CLIP`.
const UNIT: [i16; 6] = [0, 0, 2, 0, 0, 2];

/// Nought to one onto clip space, with `y` negated as [`CLIP`] is.
///
/// `clip = (2x - 1, 1 - 2y)`, so the unit square covers the viewport exactly and the center pixel
/// comes from a position near a half in both axes -- which is what makes the texture coordinate
/// this case reads predictable.
const UNIT_CLIP: [f32; 16] = [
    2.0, 0.0, 0.0, 0.0, //
    0.0, -2.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    -1.0, 1.0, 0.0, 1.0,
];

/// The glyph a symbol case samples, and the two texels that matter.
///
/// One channel, which is what `GLYPH_ATLAS_FORMAT` is. Texel (5, 5) is solid, so one case lands
/// inside the glyph; (6, 6) holds 153, which is below the coverage ramp but above where the ramp
/// would sit if the field's zero level moved -- so the case that lands there separates two
/// mutations rather than one.
fn glyph_atlas() -> Image {
    Image::red(16, 16, |x, y| match (x, y) {
        (5, 5) => 255,
        (6, 6) => 153,
        _ => 0,
    })
}

/// A symbol's drawable block, which every symbol case shares but for the gamma.
///
/// The three matrices are all [`CLIP`], which collapses the two-stage placement: the layout
/// position is the origin, so the label plane contributes nothing and the corner offset reaches
/// clip space directly. Both size flags are set and the offset flag with them, so the size is the
/// block's own and the perspective term is skipped -- the placement this leaves is
/// `corner / 32 * font_scale`, and a font scale of 32 makes that the corner itself.
fn symbol_drawable() -> Vec<u8> {
    symbol_drawable_sized(32.0, 16.0, 16.0)
}

/// The same, with the size and the two atlas sizes chosen.
///
/// `symbol_text_and_icon` needs both: its font scale is `size / 24` unconditionally where the
/// other two branch on `is_text_prop`, so reaching a font scale of 32 takes a size of 768; and its
/// two atlases are different sizes, which is how a case can tell which one was read.
fn symbol_drawable_sized(size: f32, texsize: f32, texsize_icon: f32) -> Vec<u8> {
    block(
        &SYMBOL_DRAWABLE_UBO,
        &[
            ("matrix", At::F(&CLIP)),
            ("label_plane_matrix", At::F(&CLIP)),
            ("coord_matrix", At::F(&CLIP)),
            ("texsize", At::F(&[texsize, texsize])),
            ("texsize_icon", At::F(&[texsize_icon, texsize_icon])),
            // Not a text property, so the font scale is the size rather than the size over 24.
            ("is_text_prop", At::I(&[0])),
            ("rotate_symbol", At::I(&[0])),
            ("pitch_with_map", At::I(&[0])),
            ("is_size_zoom_constant", At::I(&[1])),
            ("is_size_feature_constant", At::I(&[1])),
            ("is_offset", At::I(&[1])),
            ("size", At::F(&[size])),
            // One, so a case giving its opacity two different endpoints lands on the second and
            // the factor is under test rather than a no-op.
            ("opacity_t", At::F(&[1.0])),
        ],
    )
}

/// The nine streams a text-and-icon vertex needs, with the glyph-or-sprite mark chosen.
///
/// `sized.x` carries the size doubled with the mark in its low bit, so an even value is a sprite
/// and an odd one a glyph. There is no pixel offset: this family declares none.
fn text_and_icon_streams(tex: u16, sized_x: u16) -> Vec<Vec<u8>> {
    vec![
        shorts(&[0, 0, -1, -1, 0, 0, 3, -1, 0, 0, -1, 3]),
        ushorts(&[
            tex, tex, sized_x, 0, tex, tex, sized_x, 0, tex, tex, sized_x, 0,
        ]),
        per_vertex(&[0.0, 0.0, 0.0], 3),
        per_vertex(&[254.0], 3),
        per_vertex(&packed_color([200, 100, 50, 128]), 3),
        per_vertex(&packed_color([255, 0, 255, 255]), 3),
        per_vertex(&[0.0, 0.5], 3),
        per_vertex(&[0.0, 0.0], 3),
        per_vertex(&[0.0, 0.0], 3),
    ]
}

/// The four blocks every symbol case binds, with the drawable and the gamma chosen.
fn symbol_blocks(drawable: Vec<u8>) -> Vec<Vec<u8>> {
    vec![
        drawable,
        block(
            &SYMBOL_TILE_PROPS_UBO,
            &[
                ("is_text", At::I(&[0])),
                ("is_halo", At::I(&[0])),
                ("gamma_scale", At::F(&[0.1])),
            ],
        ),
        block(&SYMBOL_EVALUATED_PROPS_UBO, &[]),
        block(
            &GLOBAL_PAINT_PARAMS_UBO,
            &[
                ("camera_to_center_distance", At::F(&[1.0])),
                ("symbol_fade_change", At::F(&[0.0])),
                ("aspect_ratio", At::F(&[1.0])),
                ("pixel_ratio", At::F(&[1.0])),
            ],
        ),
    ]
}

/// A sprite sheet, whose texels differ in every channel and whose alpha is even.
///
/// Even, so a half opacity divides it exactly rather than landing on a half step.
fn sprite_sheet(side: u32) -> Image {
    Image::new(side, |x, y| {
        [
            u8::try_from(x * 16).unwrap_or(255),
            u8::try_from(y * 8).unwrap_or(255),
            128,
            128,
        ]
    })
}

/// The ten streams a symbol-SDF vertex needs, with the corner and the glyph's texel chosen.
fn symbol_streams(tex: u16) -> Vec<Vec<u8>> {
    vec![
        // The anchor at the origin and the corner covering the viewport, in one `Short4`.
        shorts(&[0, 0, -1, -1, 0, 0, 3, -1, 0, 0, -1, 3]),
        // The glyph's texel, then a size this case does not use.
        ushorts(&[tex, tex, 0, 0, tex, tex, 0, 0, tex, tex, 0, 0]),
        shorts(&[0; 12]),
        per_vertex(&[0.0, 0.0, 0.0], 3),
        // 254 is an opacity of 127 with the rising bit clear, which with no fade change in
        // flight is a fade of exactly one.
        per_vertex(&[254.0], 3),
        per_vertex(&packed_color([200, 100, 50, 255]), 3),
        per_vertex(&packed_color([255, 0, 255, 255]), 3),
        per_vertex(&[1.0, 1.0], 3),
        per_vertex(&[0.0, 0.0], 3),
        per_vertex(&[0.0, 0.0], 3),
    ]
}

/// Two triangles whose low bits spell the four corners of one quad, centered at the origin.
///
/// `circle` and `heatmap` sneak the corner sign into the low bit of the position, so a quad's
/// vertices are `2 * center + (corner + 1) / 2`. Centered at the origin, that is zeros and ones.
const CORNERED_QUAD: [i16; 12] = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1];

/// The same quad, centered at (1, 1) instead.
///
/// `2 * center + (corner + 1) / 2`, so a center of one is a two with the corner bit on top. The
/// heatmap case needs its kernel sampled away from its own peak -- see that case for why.
const CORNERED_QUAD_AT_ONE: [i16; 12] = [2, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3];

fn main() {
    match run() {
        Ok(passed) => println!("{passed} cases: ok"),
        Err(why) => {
            eprintln!("readback oracle: {why}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<usize, String> {
    let open = Open::preferred()?;
    println!("  device: {} ({})", open.name, open.class);
    let probe = Probe::new(&open)?;
    let mut cache = Cache::new();

    // The control first. A cleared pass has to read the clear, or a right answer below could be
    // whatever the mapped memory happened to hold -- and the clear is opaque now, so this also
    // says the readback is showing the pass rather than a zeroed buffer.
    probe.draw(&open, None)?;
    let cleared = probe.pixel(SIDE / 2, SIDE / 2)?;
    if cleared != CLEARED {
        return Err(format!(
            "a cleared pass reads {cleared:?} against {CLEARED:?}, so the readback is not showing the pass"
        ));
    }

    // `TSL_ONLY` runs one case by name, which is how a board that dies in its own shader
    // compiler gets bisected: the probe prints a case's name only after it has drawn, so a
    // crash says which case but not which part of it.
    let only = std::env::var("TSL_ONLY").ok();
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(|case| only.as_deref().is_none_or(|name| case.name == name))
        .collect();
    if cases.is_empty() {
        return Err(format!("no case named {}", only.unwrap_or_default()));
    }
    for case in &cases {
        let drawn = draw_case(&open, &probe, &mut cache, case)?;
        if drawn != case.expect {
            return Err(format!(
                "{} reads {drawn:?}, wanted {:?}",
                case.name, case.expect
            ));
        }
        println!("  {:<18} {drawn:?}", case.name);
    }
    println!(
        "  through the library: {} pipelines from {} modules, {} binds",
        cache.built(),
        cache.modules(),
        cache.bound()
    );
    Ok(cases.len())
}

/// Every family this probe can set up without a texture, and the pixel each should draw.
// A table, and nothing but a table.
#[allow(clippy::too_many_lines)]
fn cases() -> Vec<Case> {
    vec![
        // A flat fill, at the far end of both of its interpolations.
        //
        // The color's two endpoints are red and blue and `color_t` is one, so blue is the answer
        // and red is what a body that swapped the endpoints or dropped the factor would draw. The
        // opacity runs nothing to one over the same factor, so a dropped opacity is black.
        Case {
            name: "fill",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_pair([255, 0, 0, 255], [0, 0, 255, 255]), 3),
                per_vertex(&[0.0, 1.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("color_t", At::F(&[1.0])),
                        ("opacity_t", At::F(&[1.0])),
                    ],
                ),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [0, 0, 255, 255],
        },
        // The second entry of a drawable buffer, which is where the entry *stride* becomes visible.
        //
        // Every other case draws entry zero, and entry zero sits at offset zero under any stride --
        // so none of them can tell what stride the entries are packed at. This one can, and it is
        // the case #81 asked for.
        //
        // A fill's drawable buffer is an array of `FillDrawableUnionUBO`, whose stride is 96
        // because the pattern variants are larger than the plain 80-byte `FillDrawableUBO`. So
        // entry one starts at byte 96. A shader declaring the struct at its own 80 bytes reads it
        // at byte 80 instead, where the first sixteen bytes are entry zero's tail -- zeros -- so
        // the matrix's first column is zero, the triangle is degenerate and nothing draws.
        //
        // Three outcomes, which is what makes this worth a case rather than an assertion:
        //
        // * the stride right, the index right -- blue, below;
        // * the stride right, the index dropped to zero -- red, because entry zero's `color_t` is
        //   zero where entry one's is one;
        // * the stride wrong -- the clear, because the matrix read across the boundary collapses.
        Case {
            name: "drawable_at_one",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_pair([255, 0, 0, 255], [0, 0, 255, 255]), 3),
                per_vertex(&[0.0, 1.0], 3),
            ],
            uniforms: vec![
                // Two entries, end to end at the union's stride. The decoy's matrix is the same,
                // so a wrong *index* draws rather than blanks -- which is the outcome a decoy of
                // zeros could not be told apart from a wrong stride.
                [
                    block(
                        &FILL_DRAWABLE_UBO,
                        &[
                            ("matrix", At::F(&CLIP)),
                            ("color_t", At::F(&[0.0])),
                            ("opacity_t", At::F(&[1.0])),
                        ],
                    ),
                    block(
                        &FILL_DRAWABLE_UBO,
                        &[
                            ("matrix", At::F(&CLIP)),
                            ("color_t", At::F(&[1.0])),
                            ("opacity_t", At::F(&[1.0])),
                        ],
                    ),
                ]
                .concat(),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 1,
            expect: [0, 0, 255, 255],
        },
        // A background: the only family whose color is a uniform rather than an attribute, and
        // the only one declaring a `Color` field -- so this is where that kind's four floats are
        // checked to arrive unpacked.
        Case {
            name: "background",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::BackgroundShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![shorts(&COVERING)],
            uniforms: vec![
                block(&BACKGROUND_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &BACKGROUND_PROPS_UBO,
                    &[
                        ("color", At::F(&[0.0, 0.0, 1.0, 1.0])),
                        ("opacity", At::F(&[1.0])),
                    ],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [0, 0, 255, 255],
        },
        // An outline, which is the one family whose fragment stage reads its own screen position
        // back and so the one that can tell which way `y` runs.
        //
        // Full coverage at the center is the answer and not a missing feather. The vertex stage
        // divides by `w` and the result is then interpolated; with no perspective that
        // interpolation is the fragment's own position, so the distance is zero and the fade is
        // one. The feather only separates from the fragment under a `w` that varies, which is the
        // case it exists for.
        //
        // This case is why `FILL_OUTLINE_BODY` negates `y`. Without that it draws nothing at all:
        // naga flips the position after the body runs, so the position the body computed is
        // mirrored against the one the fragment is at, and every fragment is more than a pixel
        // from its own vertex.
        Case {
            name: "fill_outline",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillOutlineShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_color([255, 255, 255, 255]), 3),
                per_vertex(&[1.0, 1.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_OUTLINE_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("outline_color_t", At::F(&[0.0])),
                        ("opacity_t", At::F(&[0.0])),
                    ],
                ),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[(
                        "world_size",
                        At::F(&[f64::from(SIDE) as f32, f64::from(SIDE) as f32]),
                    )],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [255, 255, 255, 255],
        },
        // An extrusion's roof, which is the one family that computes a color rather than
        // carrying one. Every number is derived from `fill_extrusion.hpp` by hand, so a
        // disagreement is between mbgl's arithmetic and this crate's rather than between the
        // shader and a restatement of itself.
        //
        // Lit brightly, where the surface's own luminance is what narrows the range:
        //
        //   luminance   = 1.0*0.2126 + 0.25098*0.7152 + 0*0.0722   = 0.39210
        //   ambient     = (1.03, 0.28098, 0.03)
        //   facing      = clamp(dot((0, 0, 1), (0, 0, 2)), 0, 1)   = 1
        //   directional = mix(1 - 1, max(1 - 0.39210 + 1, 1), 1)   = 1.60790
        //   lit         = clamp(ambient * 1.60790)  ->  (255, 115, 12)
        //
        // The color is deliberately not gray: with equal channels any permutation of the
        // luminance weights gives the same answer, and that mutation survived a gray case. Red
        // saturates the clamp and the other two channels carry the signal.
        //
        // The light sits at twice the height it needs to, which costs the expectation nothing and
        // puts the `facing` clamp's upper end under test: without it the dot product is two, and
        // this reads (255, 230, 25).
        Case {
            name: "fill_extrusion",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillExtrusionShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                ushorts(&[0, 0, 0, 0, 0, 0]),
                per_vertex(&packed_color([255, 64, 0, 255]), 3),
                per_vertex(&[0.0, 0.0], 3),
                per_vertex(&[0.0, 0.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_EXTRUSION_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("base_t", At::F(&[0.0])),
                        ("height_t", At::F(&[0.0])),
                        ("color_t", At::F(&[0.0])),
                    ],
                ),
                block(
                    &FILL_EXTRUSION_PROPS_UBO,
                    &[
                        ("light_color", At::F(&[1.0, 1.0, 1.0])),
                        ("light_position", At::F(&[0.0, 0.0, 2.0])),
                        ("light_intensity", At::F(&[1.0])),
                        ("vertical_gradient", At::F(&[0.0])),
                        ("opacity", At::F(&[1.0])),
                    ],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [255, 115, 12, 255],
        },
        // The same roof under a dim light at an angle, which is where the other half of the
        // shading shows. Above, the light is overhead and at full intensity, so the mix lands on
        // its upper end and the `max` that guards it never binds.
        //
        //   luminance   = 0.50196,  ambient = 0.53196
        //   least = 1 - 0.1 = 0.9,  most = max(1 - 0.50196 + 0.1, 1) = 1.0
        //   directional = mix(0.9, 1.0, 0.25)                        = 0.92500
        //   lit         = 0.53196 * 0.925  ->  125
        //
        // Three mutations separate here and nowhere else: dropping the `max` reads 112, swapping
        // the mix's arms reads 132, and a facing of a quarter is what keeps those two apart -- at
        // a half the mix is symmetric and the swap is invisible.
        Case {
            name: "extrusion_dim",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillExtrusionShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                ushorts(&[0, 0, 0, 0, 0, 0]),
                per_vertex(&packed_color([128, 128, 128, 255]), 3),
                per_vertex(&[0.0, 0.0], 3),
                per_vertex(&[0.0, 0.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_EXTRUSION_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("base_t", At::F(&[0.0])),
                        ("height_t", At::F(&[0.0])),
                        ("color_t", At::F(&[0.0])),
                    ],
                ),
                block(
                    &FILL_EXTRUSION_PROPS_UBO,
                    &[
                        ("light_color", At::F(&[1.0, 1.0, 1.0])),
                        ("light_position", At::F(&[0.0, 0.0, 0.25])),
                        ("light_intensity", At::F(&[0.1])),
                        ("vertical_gradient", At::F(&[0.0])),
                        ("opacity", At::F(&[1.0])),
                    ],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [125, 125, 125, 255],
        },
        // A raster tile, which is the first case to sample anything.
        //
        // The texture is four texels square and every one is distinct, so the pixel says which
        // was read rather than only that something was. The coordinate is a position of 6144 under
        // a buffer scale of two:
        //
        //   uv = ((6144 / 8192) - 0.5) / 2 + 0.5 = 0.625  ->  texel 2 of 4, which is
        //                                                      (128, 64, 128, 128)
        //
        // Texel 2 rather than a nearer one, and the center of it rather than an edge, because the
        // interleaved case below shares these numbers and has to be able to tell three values
        // apart -- see it for which three.
        //
        // A buffer scale of two is what puts that formula under test. At one the recentering
        // cancels and `((t / 8192) - 0.5) / 1 + 0.5` is just `t / 8192`, so a body that dropped
        // it would read the same texel; at two, dropping it gives 0.375 and texel 1.
        //
        // Every color adjustment is set to its identity -- the spin a permutation that is no
        // permutation, no saturation shift, unit contrast, brightness from nothing to one -- and
        // `fade_t` of zero takes the near tile alone. So what this case checks is the sampling
        // and the chain's neutrality rather than its arithmetic.
        //
        // The texel is half transparent and the opacity is a half, which is what puts the
        // un-premultiply under test. An opaque texel hides it: dividing the color by its alpha
        // and multiplying it back by the same alpha is the identity, and the case reads the texel
        // either way. With the two alphas different the division shows --
        //
        //   unpremultiplied = (1.0, 0.5, 1.0),  alpha = 0.50196 * 0.5 = 0.25098
        //   out             = (0.25098, 0.12549, 0.25098, 0.25098)  ->  64, 32, 64, 64
        //
        // -- and a body that skipped it multiplies the premultiplied color by the output alpha
        // instead and reads (32, 16, 32, 64).
        Case {
            name: "raster",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::RasterShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[COORDINATE; 6]),
                FLAT_SKIRT.to_vec(),
            ],
            uniforms: vec![
                block(&RASTER_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &RASTER_EVALUATED_PROPS_UBO,
                    &[
                        // `dot(rgb, spin.xyz)`, `dot(rgb, spin.zxy)`, `dot(rgb, spin.yzx)`, which
                        // leaves every channel alone exactly when the weights are (1, 0, 0).
                        ("spin_weights", At::F(&[1.0, 0.0, 0.0, 0.0])),
                        ("buffer_scale", At::F(&[2.0])),
                        ("scale_parent", At::F(&[1.0])),
                        ("tl_parent", At::F(&[0.0, 0.0])),
                        ("fade_t", At::F(&[0.0])),
                        ("opacity", At::F(&[0.5])),
                        ("brightness_low", At::F(&[0.0])),
                        ("brightness_high", At::F(&[1.0])),
                        ("saturation_factor", At::F(&[0.0])),
                        ("contrast_factor", At::F(&[1.0])),
                    ],
                ),
            ],
            images: vec![
                // Every channel a different function of the texel, so no two are equal: with
                // `r` and `g` alike, rotating one of the spin's three rows draws the same pixel
                // and that mutation survives.
                Image::new(4, |x, y| {
                    [
                        u8::try_from(x * 64).unwrap_or(255),
                        u8::try_from(y * 32).unwrap_or(255),
                        128,
                        128,
                    ]
                }),
                // The parent tile, which `fade_t` of zero mixes none of. Bound because the shader
                // declares it and a pipeline with an unbound sampler is undefined, not because
                // anything reads it -- so it is painted differently, and a case that mixed the
                // two would say so.
                Image::new(4, |_, _| [255, 0, 255, 255]),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [64, 32, 64, 64],
        },
        // The same picture from the same numbers, in the layout the producer actually sends.
        //
        // `encode_raster` writes one interleaved buffer -- `RasterVertex` is a position, a texture
        // coordinate and a skirt flag over a stride of 12 -- and gives each attribute its own
        // offset into it. Every other case here binds one buffer per attribute at offset zero, so
        // until this one that layout had never been drawn.
        //
        // It expects `raster`'s pixel exactly. A stride or an offset read wrong does not fail,
        // it samples another vertex's bytes, and the two cases disagreeing is what says so. The
        // skirt's four bytes are present and bound by nothing, which is how they travel.
        //
        // # Three values, three texels
        //
        // The texture coordinate sits between the position and the skirt, so an offset read wrong
        // in either direction lands on one of them -- and this case can only tell if all three
        // sample a *different* texel. Under a buffer scale of two, `uv = t / 16384 + 0.25`:
        //
        //   the coordinate  6144  ->  0.625  ->  texel 2, dead center
        //   the skirt      10240  ->  0.875  ->  texel 3, dead center     (read four bytes late)
        //   the position       0  ->  0.25   ->  the texel 0 / 1 boundary (read four bytes early)
        //
        // The position is the interpolated attribute at the pixel read, which is zero at the
        // target's center: `COVERING` spans clip space, so the center of the triangle's coverage
        // is the midpoint of its own coordinates.
        //
        // It read 819 and 6144 once, which gave texel 1 for the coordinate -- and the position's
        // 0.25 is the boundary *of* texel 1, which the hardware resolved to texel 1. So an early
        // read drew the texel the case meant to read and the case passed. Measured: zeroing every
        // offset `vertices::plan` produces left all twenty-seven cases green. #84.
        Case {
            name: "raster_interleaved",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::RasterShader,
            surface: Surface::Plane,
            layout: Layout::Interleaved {
                stride: 12,
                offsets: &[0, 4, 8],
            },
            streams: vec![interleaved(&COVERING, COORDINATE)],
            uniforms: vec![
                block(&RASTER_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &RASTER_EVALUATED_PROPS_UBO,
                    &[
                        // `dot(rgb, spin.xyz)`, `dot(rgb, spin.zxy)`, `dot(rgb, spin.yzx)`, which
                        // leaves every channel alone exactly when the weights are (1, 0, 0).
                        ("spin_weights", At::F(&[1.0, 0.0, 0.0, 0.0])),
                        ("buffer_scale", At::F(&[2.0])),
                        ("scale_parent", At::F(&[1.0])),
                        ("tl_parent", At::F(&[0.0, 0.0])),
                        ("fade_t", At::F(&[0.0])),
                        ("opacity", At::F(&[0.5])),
                        ("brightness_low", At::F(&[0.0])),
                        ("brightness_high", At::F(&[1.0])),
                        ("saturation_factor", At::F(&[0.0])),
                        ("contrast_factor", At::F(&[1.0])),
                    ],
                ),
            ],
            images: vec![
                // Every channel a different function of the texel, so no two are equal: with
                // `r` and `g` alike, rotating one of the spin's three rows draws the same pixel
                // and that mutation survives.
                Image::new(4, |x, y| {
                    [
                        u8::try_from(x * 64).unwrap_or(255),
                        u8::try_from(y * 32).unwrap_or(255),
                        128,
                        128,
                    ]
                }),
                // The parent tile, which `fade_t` of zero mixes none of. Bound because the shader
                // declares it and a pipeline with an unbound sampler is undefined, not because
                // anything reads it -- so it is painted differently, and a case that mixed the
                // two would say so.
                Image::new(4, |_, _| [255, 0, 255, 255]),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [64, 32, 64, 64],
        },
        // The same texel through adjustments that are not identities, which is where the chain's
        // arithmetic shows rather than only its neutrality. Derived from `raster.hpp` by hand:
        //
        //   texel      = (0.25098, 0.12549, 0.50196),  average = 0.29281
        //   saturation = rgb + (average - rgb) * 0.5  = (0.27190, 0.20915, 0.39739)
        //   contrast   = (rgb - 0.5) * 1.5 + 0.5      = (0.15784, 0.06373, 0.34608)
        //   brightness = mix(0.1, 0.8, rgb)           = (0.21049, 0.14461, 0.34225)  ->  54, 37, 87
        //
        // The brightness ends are deliberately the wrong way round, and that is mbgl's: it names
        // its vectors `u_high_vec` from `brightness_low` and `u_low_vec` from `brightness_high`.
        // `RASTER_BODY` transcribes the swap rather than correcting it, because the two are the
        // ends of a `mix` and exchanging them is what inverts the ramp -- so this expectation is
        // computed with the swap in place, and a body that "fixed" it would read (34, 27, 68).
        //
        // Dropping the saturation reads (48, 14, 115) and dropping the contrast (74, 63, 96), so
        // the two cannot be confused with each other or with the neutral case above.
        Case {
            name: "raster_adjusted",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::RasterShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[819, 819, 819, 819, 819, 819]),
                FLAT_SKIRT.to_vec(),
            ],
            uniforms: vec![
                block(&RASTER_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &RASTER_EVALUATED_PROPS_UBO,
                    &[
                        ("spin_weights", At::F(&[1.0, 0.0, 0.0, 0.0])),
                        ("buffer_scale", At::F(&[2.0])),
                        ("scale_parent", At::F(&[1.0])),
                        ("tl_parent", At::F(&[0.0, 0.0])),
                        ("fade_t", At::F(&[0.0])),
                        ("opacity", At::F(&[1.0])),
                        ("brightness_low", At::F(&[0.1])),
                        ("brightness_high", At::F(&[0.8])),
                        ("saturation_factor", At::F(&[0.5])),
                        ("contrast_factor", At::F(&[1.5])),
                    ],
                ),
            ],
            images: vec![
                Image::new(4, |x, y| {
                    [
                        u8::try_from(x * 64).unwrap_or(255),
                        u8::try_from(y * 32).unwrap_or(255),
                        128,
                        255,
                    ]
                }),
                Image::new(4, |_, _| [255, 0, 255, 255]),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [54, 37, 87, 255],
        },
        // The heatmap's second pass, which turns an accumulated density into a color through a
        // ramp. The first pass is not here: it writes a texture rather than a picture, and what
        // this one checks is the lookup.
        //
        //   the center pixel is at clip (0.03125, 0.03125), which this matrix and a world size
        //   of two reach from position (0.25781, 0.24219)  ->  density texel (1, 0), red 207
        //   density = 0.81176  ->  ramp texel (3, 1) = (255, 128, 0, 255), times a half opacity
        //
        // Three numbers are chosen to separate three mutations that otherwise collide, and each
        // one did collide before it was:
        //
        //   * **A world size of two, not one.** At one the multiply is the identity and a body
        //     that dropped it reads the same texel. At two, dropping it moves the position to
        //     (0.51563, 0.48438) and the pixel to (170, 128, 0).
        //   * **A ramp three rows tall, with only the middle one painted.** One row makes any
        //     second coordinate look right, and three is the smallest count where a half lands
        //     inside a row rather than on a boundary -- so `0.5` is under test, and row nought
        //     draws the magenta above it.
        //   * **A density above three quarters.** The ramp is four wide and three tall, so for a
        //     density near a half the lookup and its own transpose land on the same texel. At
        //     0.81176 the transpose reads (2, 2), which is magenta.
        Case {
            name: "heatmap_texture",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::HeatmapTextureShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![shorts(&UNIT)],
            uniforms: vec![
                block(
                    &HEATMAP_TEXTURE_PROPS_UBO,
                    // A half, so dropping the multiply is visible: at one it is the identity.
                    &[("matrix", At::F(&UNIT_CLIP)), ("opacity", At::F(&[0.5]))],
                ),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[("world_size", At::F(&[2.0, 2.0]))],
                ),
            ],
            images: vec![
                // The density, in red, as the first pass writes it. Every texel distinct, so the
                // pixel says which was read.
                Image::new(4, |x, y| {
                    [
                        u8::try_from(255 - x * 48).unwrap_or(255),
                        u8::try_from(y * 32).unwrap_or(255),
                        0,
                        255,
                    ]
                }),
                Image::sized(4, 3, |x, y| {
                    if y == 1 {
                        [u8::try_from(x * 85).unwrap_or(255), 128, 0, 255]
                    } else {
                        [255, 0, 255, 255]
                    }
                }),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [128, 64, 0, 128],
        },
        // A color relief: an elevation decoded from a DEM, then looked up in a ramp by a binary
        // search. The first case with three textures, and the first to read a float one.
        //
        //   uv          = (2458 / 8192) * (4 - 2) / 4 + 1 / 4 = 0.40002  ->  DEM texel (1, 1)
        //   elevation   = dot((100, 40, 0, -1), (1, 0.5, 0, 10))         = 110 meters
        //   the search  = 0..3, m = 1 (50), 110 >= 50 so l = 1
        //                       m = 2 (100), 110 >= 100 so l = 2  ->  brackets 2..3
        //   t           = (110 - 100) / (200 - 100)                      = 0.1
        //   mix(stop 2, stop 3, 0.1) * a half opacity  ->  (10, 90, 55, 128)
        //
        // The stops are *floats*, four to a texel. That is not a choice here: a stop is meters
        // over a range that spans the planet, so eight bits across it would be a forty-meter step
        // and a style whose stops are ten apart would collapse into one. mbgl says the same in a
        // line -- `setFormat(TexturePixelType::RGBA, TextureChannelDataType::Float)` -- and
        // tessella's `whole_float` is the producer's half of it.
        //
        // Three stops rather than two bracketing values, so the search has somewhere to go wrong:
        // an elevation of 110 against stops at 0, 50, 100 and 200 takes the upper branch twice,
        // and a search that took either the other way brackets 1..2 and draws (0, 100, 50).
        //
        // Not under test here: the half-texel offset in `(index + 0.5) / stops`. This probe
        // samples `NEAREST`, and with four stops every index lands on its own texel with or
        // without the half -- 0.125 and 0.000 are both texel nought, 0.875 and 0.750 both texel
        // three. The offset is load-bearing only under a `LINEAR` sampler, where dropping it would
        // blend each stop with its neighbor, so whether it matters is the producer's choice of
        // filter and not something a pixel here can say.
        Case {
            name: "color_relief",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::ColorReliefShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[2458, 2458, 2458, 2458, 2458, 2458]),
                FLAT_SKIRT.to_vec(),
            ],
            uniforms: vec![
                block(&COLOR_RELIEF_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &COLOR_RELIEF_TILE_PROPS_UBO,
                    &[
                        // Meters from the texel, with the alpha term as the offset: the body dots
                        // `(rgb, -1)` with this, so the fourth component is subtracted.
                        ("unpack", At::F(&[1.0, 0.5, 0.0, 10.0])),
                        ("dimension", At::F(&[4.0, 4.0])),
                        ("color_ramp_size", At::I(&[4])),
                    ],
                ),
                block(
                    &COLOR_RELIEF_EVALUATED_PROPS_UBO,
                    &[("opacity", At::F(&[0.5]))],
                ),
            ],
            images: vec![
                // The DEM. Red and green both carry signal, so the unpack's first two weights are
                // both under test; blue is zero because its weight is.
                Image::new(4, |x, y| {
                    [
                        u8::try_from(x * 100).unwrap_or(255),
                        u8::try_from(y * 40).unwrap_or(255),
                        0,
                        255,
                    ]
                }),
                Image::floats(&[
                    [0.0, 0.0, 0.0, 0.0],
                    [50.0, 0.0, 0.0, 0.0],
                    [100.0, 0.0, 0.0, 0.0],
                    [200.0, 0.0, 0.0, 0.0],
                ]),
                Image::sized(4, 1, |x, _| match x {
                    0 => [255, 255, 255, 255],
                    1 => [255, 255, 0, 255],
                    2 => [0, 200, 100, 255],
                    _ => [200, 0, 200, 255],
                }),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [10, 90, 55, 128],
        },
        // A glyph, read from the middle of its own solid texel.
        //
        // This is the case that puts a pixel behind the obligation `SYMBOL_SDF_BODY` documents.
        // `GLYPH_ATLAS_FORMAT` is one channel, Vulkan has no alpha-only format, and so the channel
        // the field arrives in is the image view's to decide: mbgl's own backend uploads
        // `R8_UNORM` and maps red into alpha, and its shader reads `.a`. This crate reads `.r`,
        // which is right through a view with the identity mapping -- the default, and what the
        // probe creates.
        //
        //   gamma = (0.105 / 1) / (32 * 0.1) = 0.03281,  inner edge = 192 / 256 = 0.75
        //   the ramp is [0.71719, 0.78281], and the solid texel is 1.0, above it
        //   alpha = 1  ->  the fill color, times an opacity and a fade of one
        //
        // The ends of the ramp rather than its middle, on purpose: at the middle one byte of the
        // atlas moves the result by a step and a half, and an expectation that tight says more
        // about float precision than about the shader.
        //
        // Not covered by either symbol case: the halo, which needs `is_halo` and a second draw;
        // and `edge_gamma`'s own value, which only sets the ramp's *width* -- both texels sit
        // outside it, so widening it moves the answer by less than a quantization step. Both are
        // pinned in `tests/shaders.rs` instead.
        Case {
            name: "symbol_sdf",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::SymbolSDFShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: symbol_streams(5),
            uniforms: symbol_blocks(symbol_drawable()),
            images: vec![glyph_atlas()],
            vertices: 3,
            ubo_index: 0,
            expect: [200, 100, 50, 255],
        },
        // The same glyph, read one texel over, outside the letter.
        //
        // The fragment stage returns zero coverage, and that is the assertion. The case above is
        // what says the setup draws at all, and the pair is what separates the channels: `.a`
        // through an identity-mapped view is 1.0 at *every* texel, so a body reading it would draw
        // the fill color here too and the two cases would be indistinguishable.
        //
        // "Nothing is drawn" is what this said, and the opaque clear disproved it: the pixel still
        // reads zero rather than the clear, so the glyph's quad *does* cover it and the body wrote
        // a transparent black over the clear. Which makes the expectation a stronger claim than it
        // was -- three outcomes are now distinguishable where two were:
        //
        //   the clear          nothing covered the pixel
        //   zero               covered, and the field gave no coverage   <- this case
        //   the fill color     covered, and the field gave full coverage <- the case above
        //
        // Over a transparent clear the first two were the same pixel.
        //
        // The texel is 153 rather than nought, which is 0.6 -- below the ramp at
        // [0.71719, 0.78281] and so still no coverage, but *above* the [0.46719, 0.53281] the ramp
        // would sit at if the field's zero level moved from 192 to 128. So this one pixel
        // separates two mutations: the channel and the edge.
        Case {
            name: "symbol_below",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::SymbolSDFShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: symbol_streams(6),
            uniforms: symbol_blocks(symbol_drawable()),
            images: vec![glyph_atlas()],
            vertices: 3,
            ubo_index: 0,
            expect: [0, 0, 0, 0],
        },
        // The first hillshade pass, which writes a slope rather than a picture.
        //
        // Nine DEM texels through a Sobel kernel. A coordinate of 4096 under a dimension of four
        // puts the center sample at uv 0.5 and its neighbors at 0.25 and 0.75, which are texels
        // 1, 2 and 3 -- so the kernel reads a 3x3 window and not the image's edge.
        //
        //   the DEM holds x * 20 + y * 5 in red, and the unpack takes red alone, so
        //     a b c = 25 45 65     d . f = 30 . 70     g h i = 35 55 75
        //   deriv.x = (65 + 70 + 70 + 75) - (25 + 30 + 30 + 35) = 160
        //   deriv.y = (35 + 55 + 55 + 75) - (25 + 45 + 45 + 65) = 40
        //   scaled by (4 - 2) / 2^(0 + 28.2562 - 21)  ->  (2.09323, 0.52331)
        //   encoded as deriv / 8 + 0.5  ->  (194, 144, 255, 255)
        //
        // The DEM's two axes carry different weights on purpose. With a pure gradient in `x` the
        // `y` derivative is nought, and reversing the kernel's rows -- the hillshade defect
        // everybody ships once -- would read nought either way. Here it reads 111 instead of 144.
        Case {
            name: "hillshade_prep",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::HillshadePrepareShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[4096, 4096, 4096, 4096, 4096, 4096]),
            ],
            uniforms: vec![
                block(&HILLSHADE_PREPARE_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &HILLSHADE_PREPARE_TILE_PROPS_UBO,
                    &[
                        ("unpack", At::F(&[1.0, 0.0, 0.0, 0.0])),
                        ("dimension", At::F(&[4.0, 4.0])),
                        // At or above 15 the exaggeration term is nought, which keeps the
                        // expectation to one power of two. A zoom of 21 is what scales the
                        // derivative into a readable part of the channel.
                        ("zoom", At::F(&[21.0])),
                    ],
                ),
            ],
            images: vec![Image::new(4, |x, y| {
                [u8::try_from(x * 20 + y * 5).unwrap_or(255), 0, 0, 255]
            })],
            vertices: 3,
            ubo_index: 0,
            expect: [194, 144, 255, 255],
        },
        // The second pass, shading the slope the first one wrote.
        //
        // Method four, which is one Lambertian light -- the simplest of the five, and the one
        // whose answer can be derived without a page of trigonometry.
        //
        //   a coordinate of 3072 is uv 0.375, and the body flips `v`  ->  texel (1, 2)
        //   the latitude there is 0.625 of the way from nought to 60, so 37.5 degrees, whose
        //     cosine is 0.79335 -- the Mercator scale, because a pixel that far north covers less
        //     ground and the same rise over it is a steeper real slope
        //   that texel is (144, 128), so deriv = (0.51765, 0.00784) / 0.79335 and the slope the
        //     light sees is twice that times an exaggeration of a half  = (0.65242, 0.00988)
        //   with the light at an altitude of 1.4 and an azimuth of pi/2,
        //     shade = (sin - lit.x * cos) / sqrt(1 + lit . lit) = 0.732329,  above a half
        //   so the highlight applies:  (0.8, 0.4, 0.2, 1) * (2 * 0.732329 - 1)  ->  95, 47, 24, 118
        //
        // The altitude is 1.4 after measuring two others. At pi/3 the shade is 0.539, a hair
        // above the branch at a half, and the pixel it draws is 16 -- an expectation that close to
        // a discontinuity says more about float precision than about the shading. At 2.0 the shade
        // clamps and the light's angle stops mattering at all.
        //
        // Texel (1, 1) is painted differently, so dropping the body's `v` flip reads it instead.
        Case {
            name: "hillshade",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::HillshadeShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[3072, 3072, 3072, 3072, 3072, 3072]),
                FLAT_SKIRT.to_vec(),
            ],
            uniforms: vec![
                block(&HILLSHADE_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &HILLSHADE_TILE_PROPS_UBO,
                    &[
                        // A real latitude, not nought. At nought the cosine is one and dividing
                        // by it is the identity, so a body that dropped the Mercator scale draws
                        // the same pixel -- that mutation survived until this was 60.
                        ("latrange", At::F(&[60.0, 0.0])),
                        ("exaggeration", At::F(&[0.5])),
                        ("method", At::I(&[4])),
                        ("num_lights", At::I(&[1])),
                    ],
                ),
                block(
                    &HILLSHADE_EVALUATED_PROPS_UBO,
                    &[
                        ("altitudes", At::F(&[1.4, 0.0, 0.0, 0.0])),
                        (
                            "azimuths",
                            At::F(&[std::f32::consts::FRAC_PI_2, 0.0, 0.0, 0.0]),
                        ),
                        // Four colors each, one per light, which the preamble emits as four
                        // `vec4`s under the no-matrix rule. Only the first is read here.
                        ("shadows", At::F(&[0.2, 0.4, 0.8, 1.0])),
                        ("highlights", At::F(&[0.8, 0.4, 0.2, 1.0])),
                    ],
                ),
            ],
            images: vec![Image::new(4, |x, y| match (x, y) {
                (1, 2) => [144, 128, 0, 255],
                (1, 1) => [200, 200, 0, 255],
                _ => [128, 128, 0, 255],
            })],
            vertices: 3,
            ubo_index: 0,
            expect: [95, 47, 24, 118],
        },
        // The heatmap's first pass: one point feature's Gaussian, written into a texture.
        //
        // The quad is sized by where the kernel falls under a sixteenth of a color step, which is
        // what the vertex stage's logarithm solves for:
        //
        //   S = sqrt(-2 * ln(ZERO / (2 * 1 * GAUSS_COEF))) / 3  = 1.34065
        //   radius 20 times an extrude scale of 0.15 = 3, so the corners reach 4.02 in clip
        //
        // The quad is centered at (1, 1) and *not* at the origin, which is the whole point of
        // this case. At the origin the center pixel sits on the kernel's peak, the extrusion
        // interpolates to nearly nothing, and the `3.0` the vertex stage divides by is the same
        // `3.0` the fragment squares -- so changing either reads 203 and the coupling is
        // invisible. Offset, the pixel samples the kernel's flank:
        //
        //   extrude     = (-0.32292, -0.32292),  dot = 0.20855
        //   density     = 2 * 1 * GAUSS_COEF * exp(-0.5 * 9 * 0.20855) = 0.31215  ->  80
        //
        // and a fragment that squared a two instead reads 134.
        //
        // Both components of the extrusion are equal, which is also this case's own finding: the
        // first derivation here took `y` to be mirrored, predicted 75, and read 80. `CLIP`'s note
        // says why -- the matrix negates `y` and naga negates it again.
        //
        // # What no covered pixel can see
        //
        // `S` itself, and so everything that goes into it: `ZERO`, the division by three, and the
        // floors on the weight and the intensity. The reason is structural rather than a poor
        // choice of inputs here. The vertex stage writes
        //
        //   clip = center + extrude * radius * extrude_scale
        //
        // so a fragment at a given clip position receives `extrude = (clip - center) / reach`,
        // which has no `S` in it. `S` decides only how far the quad *reaches*; inside it, the
        // kernel a pixel sees is fixed by where that pixel is. Halving the divisor, coarsening
        // `ZERO`, or removing the floors all still read 80, and the body's own comment is where
        // those live. (`GAUSS_COEF` is different -- it is in the fragment too, so it does show,
        // though a mutation to 0.4 is within half a percent of it and does not; 0.2 reads 40.)
        //
        // Red carries the density and the other channels are one, because the target is blended
        // additively and a zero there would still be summed.
        Case {
            name: "heatmap",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::HeatmapShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&CORNERED_QUAD_AT_ONE),
                per_vertex(&[2.0, 2.0], 6),
                per_vertex(&[20.0, 20.0], 6),
            ],
            uniforms: vec![
                block(
                    &HEATMAP_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("extrude_scale", At::F(&[0.15])),
                        ("weight_t", At::F(&[0.0])),
                        ("radius_t", At::F(&[0.0])),
                    ],
                ),
                block(
                    &HEATMAP_EVALUATED_PROPS_UBO,
                    &[("intensity", At::F(&[1.0]))],
                ),
            ],
            images: Vec::new(),
            vertices: 6,
            ubo_index: 0,
            expect: [80, 255, 255, 255],
        },
        // A tiled background, which is the case for `pattern_pos` -- the one piece of the
        // prelude no pixel had reached.
        //
        // A pattern is anchored to the *world*, so a repeating fill does not restart at every
        // tile edge, and the anchor is a pixel coordinate too large for an `f32` at the precision
        // a pattern needs. It arrives split, and `pattern_pos` brings it down a byte at a time
        // through three nested wraps. The numbers here are what make that nesting observable
        // rather than decorative:
        //
        //   upper 1001, lower 7, pattern size 10
        //   nested:     wrap(1001, 10) = 1  ->  *256, wrap = 6  ->  *256 + 7, wrap = 3
        //   exact:      (1001 * 65536 + 7) mod 10                              = 3
        //   collapsed:  wrap(f32(1001 * 65536 + 7), 10)                        = 4
        //
        // The collapsed form is wrong because the sum is 65,601,543, which needs 26 bits and so
        // rounds in an `f32` -- to 65,601,544, whose remainder is 4. That one unit moves the
        // pattern coordinate by a tenth and the atlas texel from 6 to 7, so the pixel reads
        // (56, 28, 64, 64) instead. A pattern built the collapsed way would tile *almost* right
        // and drift as the camera moved.
        //
        //   pattern pos = (500 * 0.03125 + 3) / 10 = 1.8625,  wrapped to 0.8625
        //   uv          = mix(0 / 16, 8 / 16, 0.8625) = 0.43125  ->  atlas texel 6
        //   that texel is (96, 48, 128, 128), times a half opacity  ->  (48, 24, 64, 64)
        //
        // Three of the four numbers above exist to separate mutations, and each collided first:
        //
        //   * **A tile-to-pixel factor of 500.** It was chosen by searching: at 10 the position
        //     term lands in the same texel as the offset alone, at 100 the pattern position stays
        //     under one so the repeat wrap is the identity, and at 400 dropping the size divide
        //     reads the right texel by coincidence. 500 is the first value where all six
        //     mutations below read a different texel from the correct one.
        //   * **A size of 5 with a scale of 2, not a size of 10 with a scale of 1.** The product
        //     is what the arithmetic uses, so with a scale of one dropping it changes nothing. As
        //     a 5 doubled, dropping the scale wraps over 5 and reads texel 1.
        //   * **An atlas alpha of 128, not 255.** The half opacity then divides it exactly; at 255
        //     the answer is 127.5 and the expectation becomes a claim about rounding.
        //
        // The second pattern's sprite is a different corner of the same sheet, so a `mix` taken
        // the other way reads texel 14.
        Case {
            name: "background_pattern",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::BackgroundPatternShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![shorts(&COVERING)],
            uniforms: vec![
                block(
                    &BACKGROUND_PATTERN_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("pixel_coord_upper", At::F(&[1001.0, 1001.0])),
                        ("pixel_coord_lower", At::F(&[7.0, 7.0])),
                        ("tile_units_to_pixels", At::F(&[500.0])),
                    ],
                ),
                block(
                    &BACKGROUND_PATTERN_PROPS_UBO,
                    &[
                        ("pattern_tl_a", At::F(&[0.0, 0.0])),
                        ("pattern_br_a", At::F(&[8.0, 8.0])),
                        ("pattern_tl_b", At::F(&[8.0, 8.0])),
                        ("pattern_br_b", At::F(&[16.0, 16.0])),
                        ("pattern_size_a", At::F(&[5.0, 5.0])),
                        ("pattern_size_b", At::F(&[5.0, 5.0])),
                        ("scale_a", At::F(&[2.0])),
                        ("scale_b", At::F(&[2.0])),
                        ("mix", At::F(&[0.0])),
                        ("opacity", At::F(&[0.5])),
                    ],
                ),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[("pattern_atlas_texsize", At::F(&[16.0, 16.0]))],
                ),
            ],
            images: vec![Image::new(16, |x, y| {
                [
                    u8::try_from(x * 16).unwrap_or(255),
                    u8::try_from(y * 8).unwrap_or(255),
                    128,
                    128,
                ]
            })],
            vertices: 3,
            ubo_index: 0,
            expect: [48, 24, 64, 64],
        },
        // A tiled fill, which shares `pattern_pos` with the background above and differs in
        // where its pattern's size comes from. A background is handed one; a fill derives it from
        // the sprite's own extent in the atlas, divided twice:
        //
        //   the sprite is (0, 0) to (8, 8), so its extent is 8
        //   display size = 8 / a pixel ratio of 2   = 4    (atlas pixels to screen pixels)
        //   pattern size = 4 * a from-scale of 2.5  = 10   (screen pixels to tile units)
        //
        // Ten is the same size the background case uses, with the same anchor and the same tile
        // factor of 500, so the pattern coordinate and the texel are the same -- which is the
        // point. What is new here is the two divisions, and each one is separated:
        //
        //   clean                    ->  texel 6
        //   pixel ratio dropped      ->  size 20, texel 7
        //   from-scale dropped       ->  size 4, texel 5
        //   tile factor dropped      ->  texel 2
        //   sprite taken from `to`   ->  texel 14
        //
        // The sprite corners arrive as attributes rather than uniforms, because `pattern-from` is
        // per feature -- and they are read straight through, since `FILL_PATTERN_DRAWABLE_UBO`'s
        // `pattern_from_t` is a factor mbgl declares and never reads. Halfway between two
        // sprites' corners is a rectangle containing neither.
        Case {
            name: "fill_pattern",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillPatternShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                ushorts(&[0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8, 8]),
                ushorts(&[8, 8, 16, 16, 8, 8, 16, 16, 8, 8, 16, 16]),
                // Nought to a half over a factor of one, so the opacity is a half and both the
                // endpoint order and the factor are under test.
                per_vertex(&[0.0, 0.5], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_PATTERN_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("pixel_coord_upper", At::F(&[1001.0, 1001.0])),
                        ("pixel_coord_lower", At::F(&[7.0, 7.0])),
                        ("tile_ratio", At::F(&[500.0])),
                        ("opacity_t", At::F(&[1.0])),
                    ],
                ),
                block(
                    &FILL_PATTERN_TILE_PROPS_UBO,
                    &[("texsize", At::F(&[16.0, 16.0]))],
                ),
                block(
                    &FILL_EVALUATED_PROPS_UBO,
                    &[
                        ("fade", At::F(&[0.0])),
                        ("from_scale", At::F(&[2.5])),
                        ("to_scale", At::F(&[2.5])),
                    ],
                ),
                block(&GLOBAL_PAINT_PARAMS_UBO, &[("pixel_ratio", At::F(&[2.0]))]),
            ],
            images: vec![Image::new(16, |x, y| {
                [
                    u8::try_from(x * 16).unwrap_or(255),
                    u8::try_from(y * 8).unwrap_or(255),
                    128,
                    128,
                ]
            })],
            vertices: 3,
            ubo_index: 0,
            expect: [48, 24, 64, 64],
        },
        // A sprite running along a line, which is the last family a pixel can reach.
        //
        // The geometry is a line quad: the line runs along `x` with its center at plus and minus
        // four, and each vertex carries the extrusion that pushes it off the centerline. The
        // normal is the low bit of the position, so one side of the quad is -1 and the other +1
        // and the fragment reads the interpolation between them.
        //
        //   width 5 and no gap, so outset = 2.5 + 0.5 = 3 and inset = 0
        //   the extrusion is 3 * (-128 .. 127) / 63, so the quad spans y [-6.0952, 6.0476]
        //   the center pixel is at 0.03125, which is a normal of 0.009069 across that span
        //   distance = 0.009069 * 3 = 0.027206, and blur2 = (0 + 1) * 1 = 1
        //   alpha    = clamp(min(0.027206 + 1, 3 - 0.027206) / 1)  =  1, exactly
        //
        // An outset of 3 against a blur of 1 is what makes the coverage *exactly* one rather than
        // a fraction, and that is deliberate: at an outset of 1 the sample sits at the line's own
        // edge, the coverage is 0.98057, and the pixel comes out at 23.5 -- an expectation about
        // rounding rather than about the shader.
        //
        // Then the sprite. A line's pattern is not square to the tile: its length runs along the
        // line and divides by the tile's zoom ratio, its width runs across and does not.
        //
        //   display  = (8 - 0) / a pixel ratio of 2                      = 4
        //   size     = (4 * a from-scale of 4.5 / a zoom ratio of 2, 4)  = (9, 4)
        //   along    = (floor(9 / 4) + 5 * 64) * 2 = 644, so x = 644 / 9 wrapped  = 0.55556
        //   across   = 0.5 + 0.009069 * clamp(3, 0, 3) / 4                        = 0.50680
        //   uv       = mix((0, 0) / 16, (8, 8) / 16, (0.55556, 0.50680))  ->  texel (4, 4)
        //   that texel is (64, 32, 128, 128), times a half opacity  ->  (32, 16, 64, 64)
        //
        // A zoom ratio of 2 rather than 1 is what puts that division under test: at 1 it is the
        // identity and a body that dropped it reads the same texel. And a from-scale of 4.5
        // rather than 2.5 was found by searching -- with a size of 5 the distance along the line
        // and a distance whose high byte was scaled by 256 instead of 64 both wrap to 0.8, because
        // they differ by a multiple of the size. Nine is the smallest size here where they do not.
        //
        // Not under test, and not reachable in one pixel: the clamp on the width, and whether the
        // sprite's two axes are scaled alike. Both only move the *across* coordinate, which the
        // sample's normal of 0.009 barely shifts -- and making that normal large enough to matter
        // puts the sample at the line's edge, where the coverage stops being exactly one and the
        // expectation becomes a statement about rounding. The textual pins in `tests/shaders.rs`
        // cover both.
        Case {
            name: "line_pattern",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::LinePatternShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                // Two triangles: the centerline at -4 and +4, the normal bit low.
                shorts(&[-8, 0, 8, 0, 8, 1, -8, 0, 8, 1, -8, 1]),
                // The extrusion biased by 128, then the distance along the line: the third byte's
                // low two bits are the cap direction and the rest of it with the fourth is the
                // distance, which is why the fourth is multiplied by 64.
                vec![
                    128, 0, 9, 5, 128, 0, 9, 5, 128, 255, 9, 5, //
                    128, 0, 9, 5, 128, 255, 9, 5, 128, 255, 9, 5,
                ],
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[0.0, 0.5], 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[5.0, 5.0], 6),
                ushorts(&[
                    0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8, 8, //
                    0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8, 8,
                ]),
                ushorts(&[
                    8, 8, 16, 16, 8, 8, 16, 16, 8, 8, 16, 16, //
                    8, 8, 16, 16, 8, 8, 16, 16, 8, 8, 16, 16,
                ]),
            ],
            uniforms: vec![
                block(
                    &LINE_PATTERN_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("ratio", At::F(&[1.0])),
                        ("opacity_t", At::F(&[1.0])),
                    ],
                ),
                block(
                    &LINE_PATTERN_TILE_PROPS_UBO,
                    &[
                        // The device pixel ratio, the tile's zoom ratio, and the two scales, in
                        // one slot.
                        ("scale", At::F(&[2.0, 2.0, 4.5, 4.5])),
                        ("texsize", At::F(&[16.0, 16.0])),
                        ("fade", At::F(&[0.0])),
                    ],
                ),
                block(&LINE_EVALUATED_PROPS_UBO, &[]),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[
                        // One to one, so the perspective correction on the feather is exactly one
                        // and the coverage above is the whole of it.
                        ("units_to_pixels", At::F(&[1.0, 1.0])),
                        ("pixel_ratio", At::F(&[1.0])),
                    ],
                ),
            ],
            images: vec![Image::new(16, |x, y| {
                [
                    u8::try_from(x * 16).unwrap_or(255),
                    u8::try_from(y * 8).unwrap_or(255),
                    128,
                    128,
                ]
            })],
            vertices: 6,
            ubo_index: 0,
            expect: [32, 16, 64, 64],
        },
        // A symbol's icon: a sprite from a sheet, with no color of its own and no halo.
        //
        // The placement is the one the two SDF cases above already check, so what is new is the
        // fragment: a premultiplied sprite scaled by the opacity and the fade. The coordinate is
        // `tex / texsize` with no further arithmetic, which at 5 over 16 is texel 5 exactly.
        //
        //   sheet texel (5, 5) = (80, 40, 128, 128), times a half opacity and a fade of one
        //   ->  (40, 20, 64, 64)
        //
        // Two of this family's own decisions are *not* reachable from this pixel, and both were
        // real fixes: the minimum font scale in the spare half of the pixel offset, and that same
        // offset's division by 16. Each moves the quad rather than the coordinate it samples, so
        // a center pixel inside the quad reads the same texel either way. Both stay pinned by
        // `the_symbol_icon_keeps_its_placement_decisions`.
        Case {
            name: "symbol_icon",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::SymbolIconShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&[0, 0, -1, -1, 0, 0, 3, -1, 0, 0, -1, 3]),
                ushorts(&[5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 0, 0]),
                shorts(&[0; 12]),
                per_vertex(&[0.0, 0.0, 0.0], 3),
                per_vertex(&[254.0], 3),
                per_vertex(&[0.0, 0.5], 3),
            ],
            uniforms: symbol_blocks(symbol_drawable()),
            images: vec![sprite_sheet(16)],
            vertices: 3,
            ubo_index: 0,
            expect: [40, 20, 64, 64],
        },
        // One draw holding both, read at a vertex marked as a sprite.
        //
        // This family puts glyphs and inline images in one bucket and marks each vertex with the
        // low bit of the first size byte. The two atlases are deliberately different sizes -- 32
        // for the glyphs, 16 for the sprites -- so the coordinate says which sheet was read and
        // not only which texel.
        //
        //   `sized.x` of 0 is even, so this vertex is a sprite
        //   uv = 5 / texsize_icon of 16 = 0.3125  ->  sheet texel 5, (80, 40, 128, 128)
        //   times a half opacity  ->  (40, 20, 64, 64)
        //
        // Together with the case below, this pair *is* the test of the mark: invert it and this
        // one takes the glyph path at 5 / 32 and draws the fill color instead.
        Case {
            name: "both_as_icon",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::SymbolTextAndIconShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: text_and_icon_streams(5, 0),
            uniforms: symbol_blocks(symbol_drawable_sized(768.0, 32.0, 16.0)),
            images: vec![
                Image::red(32, 32, |x, y| if x == 5 && y == 5 { 255 } else { 0 }),
                sprite_sheet(16),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [40, 20, 64, 64],
        },
        // The same draw, read at a vertex marked as a glyph.
        //
        //   `sized.x` of 1 is odd, so this vertex is a glyph
        //   uv = 5 / texsize of 32 = 0.15625  ->  glyph texel 5, whose field is solid
        //   the ramp is [0.71719, 0.78281] and the field is 1.0, so the coverage is one
        //   the fill color (200, 100, 50, 128) times a half opacity  ->  (100, 50, 25, 64)
        //
        // The fill's alpha is 128 rather than 255 so the half opacity divides it exactly.
        //
        // Invert the mark and this reads the sprite sheet at 5 / 16 instead -- which is the case
        // above's answer, (40, 20, 64, 64). The two cases are each other's mutation, which is the
        // only way one pixel can say a branch went the right way.
        Case {
            name: "both_as_glyph",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::SymbolTextAndIconShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: text_and_icon_streams(5, 1),
            uniforms: symbol_blocks(symbol_drawable_sized(768.0, 32.0, 16.0)),
            images: vec![
                Image::red(32, 32, |x, y| if x == 5 && y == 5 { 255 } else { 0 }),
                sprite_sheet(16),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [100, 50, 25, 64],
        },
        // A fill bent onto a sphere, which is the first case on any surface but the plane.
        //
        // A surface is observable only through *where* geometry lands, so the assertion is
        // coverage: the bend has to put this triangle over the pixel being read. That is weaker
        // than the color assertions above and it took some care to make it mean anything.
        //
        // # Why the patch is not at the middle of the world
        //
        // At mercator (0.5, 0.5) the latitude and the longitude are both nought, and the bend is
        // locally a scaled identity -- so a small patch there cannot be told from its own
        // linearization, and a large one covers the whole target whatever the bend does. Worse,
        // the point is a symmetry of three plausible mistakes at once: negating the sphere's `y`,
        // dropping the longitude's 180, and skipping the trig entirely all land a patch centered
        // there in the same place.
        //
        // So the patch sits at (0.6, 0.3) instead, where the latitude is -58.4 degrees and the
        // longitude 36. The sphere point there is (0.309508, -0.850134), and the globe matrix
        // scales by ten and translates that point to the origin -- which is what brings a patch
        // from the southern Indian Ocean onto a 32-pixel target.
        //
        //   the triangle bends to clip (-0.21373, 0.08481), (0.29494, 0.08481), (0.10740, -0.28317)
        //   and the center pixel at (0.03125, 0.03125) is inside it
        //
        // Five wrong bends were checked against that pixel and none covers it: the latitude
        // without its doubling and offset lands the patch at (-1.4, 1.1), the sphere's `y`
        // unnegated at (0.1, -17), the longitude without its 180 at (-6.3, 0.1), no trig at all
        // at (-2.1, -10.6), and a latitude offset without the doubling at (2.3, -11.2).
        //
        // Three things this does *not* check, and cannot from here:
        //
        //   * **The bend's curvature.** Over a patch small enough to be a coverage test the bend
        //     is linear to well under a pixel. The curvature is what `displace` exists for and
        //     what the anchored surface approximates, and neither has a pixel.
        //   * **The clamp on the latitude.** This patch sits at 0.3 of the way down the world,
        //     nowhere near the edge, so clamping or not gives the same answer. It is there for a
        //     tile that spans the pole, which is a configuration a 32-pixel target cannot hold.
        //   * **The depth offset.** `merc.z` reaches `clip.z` and this case has no depth test, so
        //     the term is computed and ignored.
        Case {
            name: "fill_on_globe",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::FillShader,
            surface: Surface::Globe,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_color([0, 255, 255, 255]), 3),
                per_vertex(&[1.0, 1.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_DRAWABLE_UBO,
                    &[
                        // Tile units to mercator: a span of 0.005 about (0.6, 0.3).
                        (
                            "matrix",
                            At::F(&[
                                0.005, 0.0, 0.0, 0.0, //
                                0.0, 0.005, 0.0, 0.0, //
                                0.0, 0.0, 0.0, 0.0, //
                                0.6, 0.3, 0.0, 1.0,
                            ]),
                        ),
                        ("color_t", At::F(&[0.0])),
                        ("opacity_t", At::F(&[0.0])),
                    ],
                ),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
                // The surface's own block, which binds after the family's -- the order `module`
                // numbers them in.
                block(
                    &GLOBE_CAMERA_UBO,
                    &[(
                        // Ten times the unit sphere, with the patch's own point brought to the
                        // origin. The third column is nought: a sphere's `z` is depth this case
                        // does not test, and leaving it in would push the patch out of clip.
                        "globe_matrix",
                        At::F(&[
                            10.0, 0.0, 0.0, 0.0, //
                            0.0, 10.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 0.0, //
                            -3.095_077, 8.501_343, 0.0, 1.0,
                        ]),
                    )],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [0, 255, 255, 255],
        },
        // The same fill on the anchored bend: the sphere as a quadratic about the tile's own
        // center, which is what the producer expands above the zoom where that holds.
        //
        // This placement reads no matrix at all -- the expansion is already in clip space -- so
        // what a pixel can check is the expansion's own shape. A span of 3072 tile units either
        // side of the center at 4096 makes the quadratic terms comparable to the linear ones:
        //
        //   linear      1e-4 * 3072                      = 0.3072
        //   quadratic   0.5 * 1e-7 * 3072^2              = 0.4719
        //   cross       5e-8 * 3072 * 3072               = 0.4719
        //
        // and with an anchor of (0.1, -0.05) the triangle bends to (0.42932, -0.27932),
        // (0.7144, 0.6644), (-0.5144, -0.5644). The pixel read is one of eight that the correct
        // expansion covers and eight wrong ones do not: either second-order term dropped, the
        // cross term dropped, either linear term dropped, the half in front of the squares
        // dropped, the anchor dropped, and the tile center taken as the origin rather than 4096.
        //
        // The coefficients and the anchor were both found by search rather than chosen. A coarser
        // sweep found spans where only a pixel or two qualified, and a margin that thin is a coin
        // toss against the rasterizer's own fill rule.
        //
        // Not checked: `d_h`, the lift per meter above the surface. A fill hands `place` a third
        // component of nought, so the term is multiplied away -- it is the extrusions' alone, and
        // this probe has no extrusion on a bent surface.
        Case {
            name: "fill_anchored",
            at: (25, 22),
            family: BuiltIn::FillShader,
            surface: Surface::GlobeAnchored,
            layout: Layout::PerAttribute,
            streams: vec![
                // 4096 plus and minus 3072, which a `Short2` holds.
                shorts(&[1024, 1024, 7168, 1024, 1024, 7168]),
                per_vertex(&packed_color([255, 255, 0, 255]), 3),
                per_vertex(&[1.0, 1.0], 3),
            ],
            uniforms: vec![
                // The drawable's matrix is not read by this placement, so it is left at nought.
                block(&FILL_DRAWABLE_UBO, &[("color_t", At::F(&[0.0]))]),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
                block(
                    &GLOBE_BEND_UBO,
                    &[
                        // Not the origin: with the anchor at nought a body that dropped it entirely
                        // draws the same picture, and that mutation survived until this moved.
                        ("anchor", At::F(&[0.1, -0.05, 0.0, 1.0])),
                        ("d_u", At::F(&[2.0e-4, 0.0, 0.0, 0.0])),
                        ("d_v", At::F(&[0.0, 2.0e-4, 0.0, 0.0])),
                        ("d_uu", At::F(&[1.0e-7, 0.0, 0.0, 0.0])),
                        ("d_vv", At::F(&[0.0, 1.0e-7, 0.0, 0.0])),
                        ("d_uv", At::F(&[5.0e-8, 5.0e-8, 0.0, 0.0])),
                        ("d_h", At::F(&[0.0, 0.0, 0.0, 0.0])),
                    ],
                ),
            ],
            images: Vec::new(),
            vertices: 3,
            ubo_index: 0,
            expect: [255, 255, 0, 255],
        },
        // The raise: a fill standing on a DEM, which is the fourth and last surface.
        //
        // The placement reads the elevation per vertex and adds it to the position's third
        // component, so the raise is observable only where the drawable's matrix turns a height
        // into lateral movement -- which is what a real projection does, and what this matrix's
        // third column is for. The base is 0.2 a tile unit and the height coefficient 3, so the
        // shape is dominated by the three vertices' elevations rather than by the triangle.
        //
        //   uv       = position * 0.1 + 0.3, so the three vertices read texels (0,0), (2,0), (0,2)
        //   meters   = dot((r, g, 0), (1, 0.5, 0)) - 10        =  -10,  70,  10
        //   height   = (meters - a skirt of 20) * 0.01         = -0.30, 0.50, -0.10
        //   clip     = (0.2x + 3h, -(0.2y + 3h))  ->  (-1.1, 1.1), (2.1, -1.3), (-0.5, -0.3)
        //
        // The pixel read is one of nineteen the correct raise covers and seven wrong ones do not:
        // the texture coordinate unscaled or unoffset, the skirt not subtracted, the exaggeration
        // dropped, the unpack's base added rather than subtracted, its green weight ignored, and
        // the height not added at all.
        //
        // Those coefficients came from a search. The obvious ones -- a base of 0.3 against a
        // height coefficient of 0.5 -- gave *no* qualifying pixel, because the triangle's own size
        // swamped the heights and four of the seven mutations drew almost the same shape.
        //
        // The DEM binds after the family's textures, which is the order `module` declares them in
        // and the order this probe already binds them -- so the surface's own image needed no new
        // machinery, only a case with an empty family texture table and one image.
        Case {
            name: "fill_raised",
            at: (8, 13),
            family: BuiltIn::FillShader,
            surface: Surface::Terrain,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_color([255, 0, 255, 255]), 3),
                per_vertex(&[1.0, 1.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_DRAWABLE_UBO,
                    &[
                        // The third column is what makes a height visible: a raised vertex moves
                        // sideways, as it does under any real projection.
                        (
                            "matrix",
                            At::F(&[
                                0.2, 0.0, 0.0, 0.0, //
                                0.0, 0.2, 0.0, 0.0, //
                                3.0, 3.0, 0.0, 0.0, //
                                0.0, 0.0, 0.0, 1.0,
                            ]),
                        ),
                        ("color_t", At::F(&[0.0])),
                        ("opacity_t", At::F(&[0.0])),
                    ],
                ),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
                block(
                    &TERRAIN_DRAWABLE_UBO,
                    &[
                        // Red and green both weighted, so both are under test; the base is
                        // subtracted, which is the sign a mutation gets wrong.
                        ("unpack", At::F(&[1.0, 0.5, 0.0, 10.0])),
                        // The coordinate's scale, then its offset in `yz`, then the exaggeration.
                        ("params", At::F(&[0.1, 0.3, 0.3, 0.01])),
                        // The ground under the camera's center, which the height is measured from
                        // so that raising the relief leaves the center where it is.
                        ("skirt", At::F(&[0.0, 20.0, 0.0, 0.0])),
                    ],
                ),
            ],
            images: vec![Image::new(4, |x, y| {
                [
                    u8::try_from(x * 40).unwrap_or(255),
                    u8::try_from(y * 20).unwrap_or(255),
                    0,
                    255,
                ]
            })],
            vertices: 3,
            ubo_index: 0,
            expect: [255, 0, 255, 255],
        },
        // The curtain: a raster quad on raised ground, with every vertex flagged.
        //
        // What it draws is coverage rather than color. The texture coordinate is constant over the
        // triangle, so every covered pixel reads the same texel as `raster` above -- the question
        // is only *where* the triangle went, and the curtain is the one thing moving it.
        //
        // Isolated on purpose: `unpack` is zero, so the DEM contributes no height and the only
        // thing reaching `position.z` is `curtain`. `fill_raised` above is where the height path
        // is under test; here it would be a second variable.
        //
        //   drop  = -flag * skirt.z = -0.2,  and the matrix's third column is (3, 3)
        //   clip  = (0.2x - 0.6, -(0.2y - 0.6))
        //   so the three vertices land at (3.2, 28.8), (16, 28.8), (3.2, 16) in pixels
        //
        // Pixel (5, 25) is inside that and inside none of the wrong triangles: with no curtain the
        // quad sits at cols 13 to 26 and rows 6 to 19; with the sign flipped it goes off the top
        // right; with the drop scaled by the exaggeration it moves a hundredth as far; and reading
        // `skirt.x` instead of `.z` -- 0.6 rather than 0.2, which is why it is not zero -- takes it
        // off the screen to the left. `skirt.y` is nought here, so reading that draws the
        // uncurtained triangle.
        Case {
            name: "raster_raised_skirt",
            at: (5, 25),
            family: BuiltIn::RasterShader,
            surface: Surface::Terrain,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&COVERING),
                shorts(&[819, 819, 819, 819, 819, 819]),
                // Flagged, which is what `add_skirt` sets on the vertices it adds.
                shorts(&[1, 0, 1, 0, 1, 0]),
            ],
            uniforms: vec![
                block(
                    &RASTER_DRAWABLE_UBO,
                    &[(
                        "matrix",
                        At::F(&[
                            0.2, 0.0, 0.0, 0.0, //
                            0.0, 0.2, 0.0, 0.0, //
                            3.0, 3.0, 0.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0,
                        ]),
                    )],
                ),
                block(
                    &RASTER_EVALUATED_PROPS_UBO,
                    &[
                        ("spin_weights", At::F(&[1.0, 0.0, 0.0, 0.0])),
                        ("buffer_scale", At::F(&[2.0])),
                        ("scale_parent", At::F(&[1.0])),
                        ("tl_parent", At::F(&[0.0, 0.0])),
                        ("fade_t", At::F(&[0.0])),
                        ("opacity", At::F(&[0.5])),
                        ("brightness_low", At::F(&[0.0])),
                        ("brightness_high", At::F(&[1.0])),
                        ("saturation_factor", At::F(&[0.0])),
                        ("contrast_factor", At::F(&[1.0])),
                    ],
                ),
                block(
                    &TERRAIN_DRAWABLE_UBO,
                    &[
                        // No height from the DEM at all: this case is the curtain alone.
                        ("unpack", At::F(&[0.0, 0.0, 0.0, 0.0])),
                        ("params", At::F(&[0.1, 0.3, 0.3, 0.01])),
                        // The ground skirt in `x` -- which only the ground family reads, and which
                        // is three times the curtain so that reading it instead is visible -- then
                        // the center height, then the curtain this hangs from.
                        ("skirt", At::F(&[0.6, 0.0, 0.2, 0.0])),
                    ],
                ),
            ],
            images: vec![
                Image::new(4, |x, y| {
                    [
                        u8::try_from(x * 64).unwrap_or(255),
                        u8::try_from(y * 32).unwrap_or(255),
                        128,
                        128,
                    ]
                }),
                Image::new(4, |_, _| [255, 0, 255, 255]),
                // The DEM, which `unpack` weights at nothing. Bound because the surface samples it
                // and an unbound sampler is undefined rather than zero.
                Image::new(4, |_, _| [128, 128, 128, 255]),
            ],
            vertices: 3,
            ubo_index: 0,
            expect: [32, 16, 64, 64],
        },
        // A circle, read at its own center: the extrusion interpolates to zero there, which is
        // inside the fill and nowhere near the stroke. With no stroke width the stroke's own
        // selection is skipped, so what is left is the fill color times its opacity.
        //
        // An extrude scale of a fifth against a reach of ten puts the quad's corners two clip
        // units out, which covers the viewport.
        Case {
            name: "circle",
            at: (SIDE / 2, SIDE / 2),
            family: BuiltIn::CircleShader,
            surface: Surface::Plane,
            layout: Layout::PerAttribute,
            streams: vec![
                shorts(&CORNERED_QUAD),
                per_vertex(&packed_color([0, 255, 0, 255]), 6),
                per_vertex(&[10.0, 10.0], 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[1.0, 1.0], 6),
                per_vertex(&packed_color([255, 0, 255, 255]), 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[0.0, 0.0], 6),
            ],
            uniforms: vec![
                block(
                    &CIRCLE_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("extrude_scale", At::F(&[0.2, 0.2])),
                    ],
                ),
                block(
                    &CIRCLE_EVALUATED_PROPS_UBO,
                    &[
                        ("scale_with_map", At::I(&[0])),
                        ("pitch_with_map", At::I(&[0])),
                    ],
                ),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[
                        ("pixel_ratio", At::F(&[1.0])),
                        ("camera_to_center_distance", At::F(&[1.0])),
                    ],
                ),
            ],
            images: Vec::new(),
            vertices: 6,
            ubo_index: 0,
            expect: [0, 255, 0, 255],
        },
    ]
}

/// Two colors packed the way `unpack_color` reads them: two channels to a float, scaled by 255.
///
/// `from` and `to` are a data-driven property's two zoom endpoints, which the body mixes by the
/// block's `_t`. Giving them *different* colors is what makes the mix observable: with the same
/// color at both ends the result is whatever `_t` is, and a body that swapped the endpoints or
/// ignored the factor would draw the right pixel anyway.
fn packed_pair(from: [u8; 4], to: [u8; 4]) -> [f32; 4] {
    let pack = |rgba: [u8; 4]| {
        [
            f32::from(rgba[0]) * 256.0 + f32::from(rgba[1]),
            f32::from(rgba[2]) * 256.0 + f32::from(rgba[3]),
        ]
    };
    let [lo_from, hi_from] = pack(from);
    let [lo_to, hi_to] = pack(to);
    [lo_from, hi_from, lo_to, hi_to]
}

/// One color at both endpoints, for a case whose interpolation is not what it is checking.
fn packed_color(rgba: [u8; 4]) -> [f32; 4] {
    packed_pair(rgba, rgba)
}

/// One attribute's stream: the same values for every vertex.
fn per_vertex(values: &[f32], vertices: usize) -> Vec<u8> {
    (0..vertices)
        .flat_map(|_| values.iter().flat_map(|value| value.to_le_bytes()))
        .collect()
}

/// The texture coordinate `raster` and `raster_interleaved` share.
///
/// Chosen so that the three values an interleaved offset can land on sample three *different*
/// texels -- see `raster_interleaved` for the arithmetic and for what it read before. Under a
/// buffer scale of two this is the dead center of texel 2 of a four-texel image.
///
/// Signed, because the attribute is: a raster's texture coordinate is declared `Short2` and the
/// vertex format is `R16G16_SINT`, so the body reads whatever is here as an `i16`.
const COORDINATE: i16 = 6144;

/// The value in the skirt's four bytes, which nothing binds.
///
/// An attribute offset read four bytes late lands here, so this has to sample a texel that is
/// neither the coordinate's nor the position's. Under a buffer scale of two it is the dead center
/// of texel 3, where [`COORDINATE`] is texel 2 and the position reaches the texel 0 / 1 boundary.
///
/// Zero would not do, and nor would 6144 once that became the coordinate: both have been the value
/// here, and each in turn made the late read draw the texel the case meant to read.
///
/// The flag itself is inert on a plane -- `curtain` returns zero there whatever it is handed -- so
/// this changes nothing but which texel a misread offset finds.
const SKIRT_FILLER: i16 = 10240;

/// One interleaved raster stream: a position, a texture coordinate and a skirt flag per vertex.
///
/// Laid out as `RasterVertex` is -- `[i16; 2]`, `[u16; 2]`, `u16` and a pad to twelve -- so the
/// offsets a case names are the producer's own.
fn interleaved(positions: &[i16], texture: i16) -> Vec<u8> {
    let mut out = Vec::with_capacity(positions.len() / 2 * 12);
    // Stepped rather than chunked: `chunks_exact` with a constant draws a stable-only lint and
    // its suggested `as_chunks` is newer than the pinned toolchain, so neither form passes both.
    for at in (0..positions.len() - 1).step_by(2) {
        out.extend(positions[at].to_le_bytes());
        out.extend(positions[at + 1].to_le_bytes());
        out.extend(texture.to_le_bytes());
        out.extend(texture.to_le_bytes());
        // The skirt flag and the pad that aligns the next vertex. Bound by nothing, and filled
        // with something a misread offset would show.
        out.extend(SKIRT_FILLER.to_le_bytes());
        out.extend(SKIRT_FILLER.to_le_bytes());
    }
    out
}

/// A stream of sixteen-bit positions.
fn shorts(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A stream of unsigned sixteen-bit values, which is what a packed pair arrives as.
fn ushorts(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One entry of a block's buffer, written at the offsets the ABI declares rather than at counted
/// ones.
///
/// The point of going through the table: a block written by hand would agree with the WGSL struct
/// only because the same person wrote both. Taking every offset from the layout makes this a
/// comparison between the producer's placement and the shader's.
///
/// Sized to the stride the producer packs entries at, which for a drawable block is its union's
/// rather than its own -- `slots::stride`. For a single entry the extra bytes are a tail of zeros
/// and change nothing; what they are for is `drawable_at_one`, where two of these sit end to end
/// and the second has to start where the producer would have put it.
fn block(layout: &'static UboLayout, writes: &[(&str, At<'_>)]) -> Vec<u8> {
    let mut bytes = vec![0u8; slots::stride(layout) as usize];
    for (name, value) in writes {
        let field = layout
            .fields
            .iter()
            .find(|field| field.name == *name)
            .unwrap_or_else(|| panic!("{} has no {name}", layout.name));
        let at = field.offset as usize;
        let words: Vec<[u8; 4]> = match value {
            At::F(values) => values.iter().map(|v| v.to_le_bytes()).collect(),
            At::I(values) => values.iter().map(|v| v.to_le_bytes()).collect(),
        };
        for (index, word) in words.iter().enumerate() {
            let start = at + index * 4;
            bytes[start..start + 4].copy_from_slice(word);
        }
    }
    bytes
}

/// WGSL to SPIR-V, the same way `tests/shaders.rs` does it.
fn compile(source: &str) -> Result<Vec<u32>, String> {
    let parsed = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spirv: {why}"))
}

/// How wide a vertex format is, from its own name.
///
/// Parsed rather than tabulated: a table of fifty formats is fifty chances to write a wrong number,
/// and the name already carries the answer. Only used for the one-buffer-per-attribute layout,
/// where an attribute's stride is its own width.
fn stride_of(format: vk::Format) -> u32 {
    let name = format!("{format:?}");
    let channels = name
        .rsplit_once('_')
        .map_or_else(|| panic!("{name} has no suffix"), |(channels, _)| channels);
    let mut bits = 0u32;
    let mut width = String::new();
    for ch in channels.chars().chain(std::iter::once('R')) {
        if ch.is_ascii_digit() {
            width.push(ch);
            continue;
        }
        if !width.is_empty() {
            bits += width.parse::<u32>().unwrap_or_else(|_| panic!("{name}"));
            width.clear();
        }
    }
    assert!(
        bits.is_multiple_of(8),
        "{name} is {bits} bits, which is not whole bytes"
    );
    bits / 8
}

/// The slab this probe's streams live in, which is the only one it has.
const SLAB: u32 = 0;

/// The geometry every case announces, one at a time.
const GEOMETRY: GeometryId = GeometryId(1);

/// The view and layer a case's blocks belong to.
const WHICH: blocks::Which = blocks::Which {
    view: ViewId(1),
    layer: 0,
};

/// The target's format, which is what the expected pixels are written in.
const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// What the target is cleared to before each case draws.
///
/// Opaque, and the whole point is that it is not transparent black. The pipelines are built
/// `Blend::Unblended`, which the bench's own note said was deliberate -- and over a transparent
/// clear it made no difference it could measure: mbgl's alpha mode is premultiplied,
/// `Add{One, OneMinusSrcAlpha}`, so blending gives `src * 1 + 0 * (1 - srcAlpha)`, which is `src`
/// for any alpha. Every case passed with `Blend::Alpha` as well. Over an opaque clear they differ,
/// and the four cases returning an alpha below 255 are what measure it.
///
/// Thirds of a fifth, so each channel is an exact multiple of 1/255 and the conversion back is not
/// a rounding question. No case's expected pixel is this, which is what lets a case that drew
/// nothing be told from one that drew the clear's color.
const CLEAR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];

/// The same color as the target's bytes, which is what a case reading the clear expects.
const CLEARED: [u8; 4] = [51, 102, 153, 255];

/// What the producer would have sent for this case's streams.
///
/// The point of going through `AttributeDesc` rather than building Vulkan state directly: these are
/// the records a capture carries, and `vertices::plan` is what a consumer turns them into. A probe
/// that built the bindings itself would be checking its own arithmetic against the shader and would
/// not be checking the library's at all.
///
/// One `SlabRef` per attribute for the per-attribute layout, and one shared by all of them for the
/// interleaved one -- which is what makes the dedup in `buffers::needs` observable: three
/// descriptors over one reference are one buffer bound three times at three offsets.
fn descriptors_for(case: &Case, table: &[ShaderAttribute]) -> Result<Vec<AttributeDesc>, String> {
    let mut out = Vec::with_capacity(table.len());
    let mut at = 0u32;
    for (index, attribute) in table.iter().enumerate() {
        let format = vertex_format(attribute.declared)
            .ok_or_else(|| format!("{} has no vertex format", attribute.name))?;
        let (source, offset, stride) = match case.layout {
            Layout::PerAttribute => {
                let length = case
                    .streams
                    .get(index)
                    .ok_or_else(|| format!("{}: no stream {index}", case.name))?
                    .len();
                let source = SlabRef {
                    slab: SLAB,
                    offset: at,
                    length: u32::try_from(length).map_err(|_| "a stream past 4GiB".to_string())?,
                };
                at += source.length;
                (source, 0, stride_of(format))
            }
            Layout::Interleaved { stride, offsets } => {
                let whole = case
                    .streams
                    .first()
                    .ok_or_else(|| format!("{}: no interleaved stream", case.name))?
                    .len();
                let offset = *offsets.get(index).ok_or_else(|| {
                    format!(
                        "{}: interleaved layout gives {} offsets for {} attributes",
                        case.name,
                        offsets.len(),
                        table.len()
                    )
                })?;
                let source = SlabRef {
                    slab: SLAB,
                    offset: 0,
                    length: u32::try_from(whole).map_err(|_| "a stream past 4GiB".to_string())?,
                };
                (source, offset, stride)
            }
        };
        out.push(AttributeDesc {
            attr_id: attribute.attr_id,
            binding: attribute.binding,
            source,
            offset,
            vertex_offset: 0,
            stride,
            // The same on both sides, which is what a producer sends for an attribute it supplies
            // at the declared type. A disagreement is `Refused::DeclaredDisagrees`, and
            // `tests/vertex_plans.rs` is where that is exercised.
            data_type: attribute.declared as u8,
            declared_data_type: attribute.declared as u8,
            _pad: [0; 2],
        });
    }
    Ok(out)
}

/// The bytes a slab reference names, out of the case's own streams.
///
/// Stands in for `tessella_consume::slab::resolve`, which reaches into the shared region. Here the
/// streams are `Vec<u8>` the case built, and the references were laid out over them end to end.
fn resolve(case: &Case, reference: SlabRef) -> Option<&[u8]> {
    match case.layout {
        // Laid out end to end in `descriptors_for`, so the offset locates the stream.
        Layout::PerAttribute => {
            let mut at = 0u32;
            for stream in &case.streams {
                let length = u32::try_from(stream.len()).ok()?;
                if at == reference.offset {
                    return Some(stream);
                }
                at += length;
            }
            None
        }
        Layout::Interleaved { .. } => case.streams.first().map(Vec::as_slice),
    }
}

/// The target a case draws into, and the buffer its pixel is read back out of.
///
/// Through `target::frame` rather than a render pass of the probe's own, which is the whole point of
/// this bench now: the pass the library records is the pass under test. That costs a readback --
/// the image is optimally tiled and device-local, so a pixel is a `vkCmdCopyImageToBuffer` and a map
/// rather than a map alone -- and buys the dynamic rendering, the depth-stencil attachment and the
/// layout transitions being the library's.
struct Probe<'d> {
    image: Device2d<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
    readback: Buffer<'d>,
    read_memory: Memory<'d>,
    depth: Depth<'d>,
    targets: Targets,
}

impl<'d> Probe<'d> {
    fn new(open: &'d Open) -> Result<Self, String> {
        let gpu = open.gpu();
        let image = gpu
            .image(
                SIDE,
                SIDE,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("target image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("target memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("bind target: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("target view: {why}"))?;

        let bytes = u64::from(SIDE) * u64::from(SIDE) * 4;
        let readback = gpu
            .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
            .map_err(|why| format!("readback buffer: {why}"))?;
        let needs = [readback.requirements()];
        let read_memory = gpu
            .allocate(
                bytes.max(needs[0].size),
                &needs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|why| format!("readback memory: {why}"))?;
        read_memory
            .bind(&readback, 0)
            .map_err(|why| format!("bind readback: {why}"))?;

        // Stencil only: these cases draw flat and test no depth, and the stencil is what the
        // library's content pipelines always have. A compare mask of zero is the unclipped path --
        // `benches/clip_masks.rs` measures that it draws everywhere rather than nowhere.
        let depth_stencil = device::depth_stencil_format(Attachment::StencilOnly, |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
        let depth = Depth::new(gpu, SIDE, SIDE, depth_stencil)
            .map_err(|why| format!("depth attachment: {why}"))?;

        Ok(Self {
            image,
            view,
            _memory: memory,
            readback,
            read_memory,
            depth,
            targets: Targets {
                color: COLOR,
                depth_stencil,
                attachment: Attachment::StencilOnly,
            },
        })
    }

    /// One pixel of the last frame read back, as the target's format has it.
    fn pixel(&self, x: u32, y: u32) -> Result<[u8; 4], String> {
        let mapping = self
            .read_memory
            .map()
            .map_err(|why| format!("map readback: {why}"))?;
        let mut pixel = [0u8; 4];
        // `copy_to_buffer` sets a row length of the image's width, so a row is `SIDE` texels.
        let at = (u64::from(y) * u64::from(SIDE) + u64::from(x)) * 4;
        mapping
            .read(at, &mut pixel)
            .map_err(|why| format!("read pixel: {why}"))?;
        Ok(pixel)
    }
}

/// What one draw needs bound, for the closure that records it.
///
/// The buffers and offsets are borrowed from the [`Store`], which is what makes them worth passing:
/// a probe that bound buffers it had created itself would not be checking that
/// `buffers::needs` and the store agree about which binding reads which slab.
struct Bound<'a> {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    streams: &'a [vk::Buffer],
    offsets: &'a [u64],
    vertices: u32,
    ubo_index: u32,
}

impl Probe<'_> {
    /// Records one frame through `target::frame` and copies the result back.
    ///
    /// `None` draws nothing, which is the cleared control. The copy is part of the same submission,
    /// so the fence the harness waits on covers the draw and the readback together -- a wait that
    /// covered only the draw would map a buffer the copy had not reached.
    fn draw(&self, open: &Open, bound: Option<&Bound<'_>>) -> Result<(), String> {
        open.submit(|record: Recorder<'_>| {
            target::frame(
                record,
                Host {
                    image: &self.image,
                    view: &self.view,
                    width: SIDE,
                    height: SIDE,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &self.depth,
                Some(CLEAR),
                |record| {
                    if let Some(bound) = bound {
                        record.bind_pipeline(bound.pipeline);
                        record.bind_descriptor_set(bound.layout, bound.set);
                        // Unclipped: a compare mask of no bits compares nothing and passes, which
                        // `benches/clip_masks.rs` measures. These cases are about what a family
                        // draws, not about where a tile's mask lets it.
                        record.stencil_compare_mask(0);
                        record.stencil_reference(0);
                        record.bind_vertex_buffers(0, bound.streams, bound.offsets);
                        // `firstInstance` is the drawable's entry, which the body reads as
                        // `ubo_index`.
                        record.draw(bound.vertices, 1, bound.ubo_index);
                    }
                },
            );
            record.transition(
                &self.image,
                target::LEAVES_IN,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            record.copy_to_buffer(&self.image, &self.readback, SIDE, SIDE);
        })
    }
}

/// Sets one case up through the library and draws it, answering the pixel it asked for.
///
/// Every step here is the library's. What the probe still owns is the *inputs*: the case's streams,
/// its blocks, its images and the one pixel to read. That is the division the bench exists for -- a
/// pixel that moves when a case's input moves says the input reached the shader, and it says it
/// about the path a capture takes rather than about a path written here.
#[allow(clippy::too_many_lines)]
fn draw_case<'d>(
    open: &'d Open,
    probe: &Probe<'d>,
    cache: &mut Cache<'d>,
    case: &Case,
) -> Result<[u8; 4], String> {
    let gpu = open.gpu();
    let found = family(case.family)
        .ok_or_else(|| format!("{}: {:?} is not a drawn family", case.name, case.family))?;
    let source = module(
        case.surface,
        found.blocks,
        found.attributes,
        found.textures,
        found.body,
    )
    .map_err(|why| format!("{} does not assemble: {why:?}", case.name))?;
    let words = compile(&source).map_err(|why| format!("{}: {why}", case.name))?;

    // The producer's records, then the plan a consumer makes of them, then the key that plan is.
    let descs = descriptors_for(case, found.attributes)?;
    let plan = vertices::plan(found.attributes, &descs)
        .map_err(|why| format!("{}: {why:?}", case.name))?;
    if plan.bound.len() != found.attributes.len() {
        return Err(format!(
            "{}: {} of {} attributes planned",
            case.name,
            plan.bound.len(),
            found.attributes.len()
        ));
    }
    // A device that will not take one of these in a vertex buffer binds the attribute anyway and
    // the shader reads zero, so ask before building the pipeline rather than reading the silence as
    // a pixel. Per case, not once per device: a board missing one format can still run every family
    // that does not declare it.
    let formats: Vec<vk::Format> = plan.bound.iter().map(|bound| bound.format).collect();
    check_vertex_formats(&formats, |format| {
        open.format_properties(format).buffer_features
    })
    .map_err(|why| format!("{} needs a format this device refuses: {why:?}", case.name))?;

    let key = pipelines::key(case.family, case.surface, 0, &plan, Blend::Unblended);
    let bindings = pipelines::bindings(found, case.surface);

    // The geometry, through the store the frame loop uses.
    let needs = buffers::needs(
        &plan,
        SlabRef {
            slab: SLAB,
            offset: 0,
            length: 0,
        },
    );
    let mut store = Store::new();
    store
        .upload(gpu, GEOMETRY, &needs, &[], &|reference| {
            resolve(case, reference)
        })
        .map_err(|why| format!("{}: geometry: {why}", case.name))?;

    // One block buffer per slot, which is what the wire sends and what a set binds.
    let mut held = blocks::Blocks::new();
    let declared: Vec<&UboLayout> = found
        .blocks
        .iter()
        .copied()
        .chain(case.surface.blocks().iter().copied())
        .collect();
    if declared.len() != case.uniforms.len() {
        return Err(format!(
            "{}: {} blocks declared and {} supplied",
            case.name,
            declared.len(),
            case.uniforms.len()
        ));
    }
    for (layout, bytes) in declared.iter().zip(&case.uniforms) {
        let slot = slots::of(layout)
            .ok_or_else(|| format!("{}: {} has no slot", case.name, layout.name))?;
        // The stride the producer packs entries at, which for a drawable block is its union's --
        // so a case supplying two entries supplies two of *those*, and the second starts where the
        // producer would have put it rather than where the struct happens to end.
        let stride = slots::stride(layout) as usize;
        if bytes.is_empty() || bytes.len() % stride != 0 {
            return Err(format!(
                "{}: {} is {} bytes, which is not whole entries of {stride}",
                case.name,
                layout.name,
                bytes.len()
            ));
        }
        let entries = bytes.len() / stride;
        held.declare(gpu, WHICH, slot, entries, stride)
            .map_err(|why| format!("{}: {} buffer: {why}", case.name, layout.name))?;
        for (index, entry) in bytes.chunks(stride).enumerate() {
            let index = u32::try_from(index).map_err(|_| "absurd entry count".to_string())?;
            held.write(WHICH, slot, index, entry)
                .map_err(|why| format!("{}: {} write: {why}", case.name, layout.name))?;
        }
        held.flush(WHICH, slot, 0)
            .map_err(|why| format!("{}: {} flush: {why}", case.name, layout.name))?;
    }

    // The images, in one submission: declared, filled, and left where a sampler can read them.
    let mut images = Images::new();
    // Point sampling, which is what every case's expected pixel was derived against: a case checks
    // *which* texel was read, and an interpolated sample is a blend of two. `raster` reads
    // [22, 11, 64, 64] under linear filtering against the [32, 16, 64, 64] it was derived for,
    // which is the neighboring texel bleeding in at eleven sixteenths.
    //
    // The filter is per binding on the wire -- `TextureRef::filter`, because one atlas is sampled
    // both ways in one frame -- so a case that wanted to check the filter itself would carry its
    // own, and none does yet.
    //
    // The slot comes from the set's own bindings, in binding order, which is where #95 put the
    // authority: a run built with ids counting from one and no slots would be placed by arrival
    // again, which is the thing that stopped deciding anything.
    let refs: Vec<TextureRef> = bindings
        .iter()
        .filter(|b| b.kind == pipelines::Kind::SampledImage)
        .enumerate()
        .map(|(at, binding)| TextureRef {
            texture: TextureId(at as u64 + 1),
            slot: binding.slot.expect("a texture binding carries its slot"),
            filter: TextureFilter::Nearest as u32,
        })
        .collect();
    let mut staged: Result<(), String> = Ok(());
    open.submit(|record: Recorder<'_>| {
        staged = stage(&mut images, gpu, record, case, &refs);
    })?;
    staged?;
    let bound = descriptors::bound_from(&images, &bindings, &refs)
        .map_err(|why| format!("{}: textures: {why}", case.name))?;

    let pipeline = cache
        .pipeline(gpu, &key, &bindings, &words, probe.targets)
        .map_err(|why| format!("{}: pipeline: {why}", case.name))?;
    let layout = cache
        .layout(gpu, case.family, case.surface, &bindings)
        .map_err(|why| format!("{}: layout: {why}", case.name))?;
    let mut sets = descriptors::Sets::new(gpu, 1, &bindings)
        .map_err(|why| format!("{}: pool: {why}", case.name))?;
    let set = sets
        .write(layout, &bindings, WHICH, &held, &bound)
        .map_err(|why| format!("{}: set: {why}", case.name))?;

    let (streams, offsets) = store
        .bindings(GEOMETRY)
        .ok_or_else(|| format!("{}: the geometry is not resident", case.name))?;
    probe.draw(
        open,
        Some(&Bound {
            pipeline,
            layout: layout.pipeline(),
            set,
            streams,
            offsets,
            vertices: case.vertices,
            ubo_index: case.ubo_index,
        }),
    )?;
    probe.pixel(case.at.0, case.at.1)
}

/// Declares and fills a case's images, and leaves them readable by a shader.
///
/// One region covering each whole image, which is the `rect_count` of zero a producer sends for a
/// texture it has just created. The transition to `SHADER_READ_ONLY_OPTIMAL` is the caller's --
/// `Images::upload` leaves the image in `TRANSFER_DST_OPTIMAL` so a frame writing several regions
/// of one atlas pays for one barrier instead of one per region -- and `descriptors::Sets::write`
/// names that layout in the descriptor, so skipping it is a sampled image in the wrong layout.
fn stage<'d>(
    images: &mut Images<'d>,
    gpu: tessella_vk::Gpu<'d>,
    record: Recorder<'_>,
    case: &Case,
    refs: &[TextureRef],
) -> Result<(), String> {
    for (at, image) in case.images.iter().enumerate() {
        let texture = refs[at].texture;
        let (pixel, channel) = image.texels.kinds();
        images
            .declare(
                gpu,
                record,
                texture,
                Extent {
                    width: image.width,
                    height: image.height,
                },
                pixel,
                channel,
            )
            .map_err(|why| format!("{}: image {at}: {why}", case.name))?;

        let bytes = image.texels.bytes();
        let row = image.width as usize * image.texels.width();
        let whole = [Rect16 {
            x: 0,
            y: 0,
            w: u16::try_from(image.width).map_err(|_| "an image past 65535".to_string())?,
            h: u16::try_from(image.height).map_err(|_| "an image past 65535".to_string())?,
        }];
        images
            .upload(gpu, record, texture, &whole, &|_, line| {
                let start = line as usize * row;
                bytes.get(start..start + row)
            })
            .map_err(|why| format!("{}: image {at} upload: {why}", case.name))?;
        let held = images
            .image(texture)
            .ok_or_else(|| format!("{}: image {at} was not held", case.name))?;
        record.transition(
            held,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
    Ok(())
}
