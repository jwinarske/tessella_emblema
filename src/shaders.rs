//! Assembling a family's WGSL module: generated declarations, then a hand-written body.
//!
//! The split is the point. Everything that has to agree with the producer — the uniform blocks,
//! their fields, the vertex attributes and their locations — is generated from the ABI's own
//! tables, so it cannot drift. What is left for a person to write is the arithmetic, which is the
//! part no table describes.
//!
//! A body that reads a field the tables do not declare fails to compile, which is the whole reason
//! to assemble it this way rather than writing the struct out beside the body and keeping them in
//! step by hand.
//!
//! # What the bindings are
//!
//! Group zero, one binding per block, read-only storage. Storage rather than uniform because the
//! blocks arrive as a consolidated buffer indexed per drawable — an array of blocks, one slot per
//! draw — which is what the producer's `ubo_index` indexes into. A uniform buffer would need one
//! binding per draw.
//!
//! The family's blocks come first, then the surface's, then a texture and a sampler for each image
//! the surface samples. So a module is one (family, surface) pair and the pair decides the set.

use std::fmt::Write as _;

use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_capture_abi::generated::shader_attributes::ShaderAttribute;
use tessella_capture_abi::generated::texture_slots::ShaderTexture;
use tessella_capture_abi::generated::ubo_layouts::UboLayout;

use crate::preamble::{Unrepresentable, declare, type_name};
use crate::surface::Surface;

/// Why a family could not be assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A block could not be declared.
    Block(Unrepresentable),
    /// An attribute's declared type is one the ABI itself calls invalid.
    ///
    /// Refused rather than read as a float. A shader cannot read a type nobody named, and giving
    /// it one draws from whatever the buffer happens to hold.
    InvalidAttribute {
        /// The attribute, as the header spells it.
        name: &'static str,
    },
}

impl From<Unrepresentable> for Error {
    fn from(why: Unrepresentable) -> Self {
        Self::Block(why)
    }
}

/// The WGSL type a vertex attribute is read as.
///
/// Sixteen-bit attributes are read as integers and widened in the body rather than declared
/// through the scaled formats, which are optional in Vulkan and absent on parts this targets. The
/// conversion is then visible in the shader instead of being a property of a format that may not
/// exist.
const fn attribute_type(declared: AttributeDataType) -> Option<&'static str> {
    match declared {
        AttributeDataType::Byte | AttributeDataType::Short | AttributeDataType::Int => Some("i32"),
        AttributeDataType::UByte | AttributeDataType::UShort | AttributeDataType::UInt => {
            Some("u32")
        }
        AttributeDataType::Byte2 | AttributeDataType::Short2 | AttributeDataType::Int2 => {
            Some("vec2<i32>")
        }
        AttributeDataType::UByte2 | AttributeDataType::UShort2 | AttributeDataType::UInt2 => {
            Some("vec2<u32>")
        }
        AttributeDataType::Byte3 | AttributeDataType::Short3 | AttributeDataType::Int3 => {
            Some("vec3<i32>")
        }
        AttributeDataType::UByte3 | AttributeDataType::UShort3 | AttributeDataType::UInt3 => {
            Some("vec3<u32>")
        }
        AttributeDataType::Byte4 | AttributeDataType::Short4 | AttributeDataType::Int4 => {
            Some("vec4<i32>")
        }
        AttributeDataType::UByte4 | AttributeDataType::UShort4 | AttributeDataType::UInt4 => {
            Some("vec4<u32>")
        }
        AttributeDataType::Float => Some("f32"),
        AttributeDataType::Float2 => Some("vec2<f32>"),
        AttributeDataType::Float3 => Some("vec3<f32>"),
        AttributeDataType::Float4 => Some("vec4<f32>"),
        // Eight shorts: mbgl packs a symbol's placement into one attribute, read as two vec4s
        // worth of integers and taken apart in the body.
        AttributeDataType::UShort8 => Some("array<u32, 8>"),
        // Invalid is the ABI saying it does not know, and a shader cannot read a
        // type nobody named. Refused by the caller rather than guessed at here.
        AttributeDataType::Invalid => None,
    }
}

/// A WGSL name for a texture, from the id the header spells.
///
/// `idRasterImage0Texture` becomes `raster_image0`, and its sampler `raster_image0_sampler`. The
/// `id` prefix and the `Texture` suffix say nothing a shader needs, as with an attribute's.
#[must_use]
pub fn texture_name(declared: &str) -> String {
    let trimmed = declared
        .strip_prefix("id")
        .unwrap_or(declared)
        .strip_suffix("Texture")
        .unwrap_or(declared);
    let mut out = String::with_capacity(trimmed.len() + 4);
    for (at, ch) in trimmed.char_indices() {
        if ch.is_ascii_uppercase() {
            if at != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// A WGSL field name for an attribute, from the id the header spells.
///
/// `idBackgroundPosVertexAttribute` becomes `background_pos`: the `id` prefix and the
/// `VertexAttribute` suffix say nothing a shader needs, and what is left is the name.
#[must_use]
pub fn attribute_name(declared: &str) -> String {
    let trimmed = declared
        .strip_prefix("id")
        .unwrap_or(declared)
        .strip_suffix("VertexAttribute")
        .unwrap_or(declared);
    let mut out = String::with_capacity(trimmed.len() + 4);
    for (at, ch) in trimmed.char_indices() {
        if ch.is_ascii_uppercase() {
            if at != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Assembles a module for one (family, surface) pair: the blocks, the vertex input, the surface's
/// placement, then the family's body.
///
/// The family's blocks bind first, from zero, then the surface's, then two bindings for each
/// texture the surface samples. The body sees each block as a binding named after its type in
/// snake case, the vertex input as `In`, and the surface as `place`.
///
/// # Errors
///
/// [`Error`] from a block that cannot be declared, or an attribute the ABI calls invalid.
pub fn module(
    surface: Surface,
    blocks: &[&UboLayout],
    attributes: &[ShaderAttribute],
    textures: &[ShaderTexture],
    body: &str,
) -> Result<String, Error> {
    let mut out = String::new();
    out.push_str("// Generated declarations. The body below is the only hand-written part.\n\n");

    let all: Vec<&UboLayout> = blocks
        .iter()
        .copied()
        .chain(surface.blocks().iter().copied())
        .collect();

    for layout in &all {
        out.push_str(&declare(layout)?);
        out.push('\n');
    }

    for (binding, layout) in all.iter().enumerate() {
        let name = type_name(layout.name);
        let _ = writeln!(
            out,
            "@group(0) @binding({binding}) var<storage, read> {}: array<{name}>;",
            binding_name(&name)
        );
    }
    // A texture and the sampler that reads it, which a placement names together. After the blocks
    // rather than in a group of their own: a family with no texture then has no gap in its set,
    // and the whole of a draw's state is one descriptor set either way.
    //
    // Counted rather than computed from the index, because the count differs by family and by
    // surface and an index formula would be arithmetic no test could distinguish from a wrong one.
    //
    // The family's own images first, in the order its table declares, then the surface's. A
    // family's samplers are a property of the shader rather than of what it is drawn on -- a
    // raster tile samples its own picture and its parent's whether it is flat or raised -- so they
    // keep their place when the surface changes under them.
    let mut binding = all.len();
    let named = textures
        .iter()
        .map(|texture| texture_name(texture.name))
        .chain(surface.textures().iter().map(|name| (*name).to_string()));
    for texture in named {
        let _ = writeln!(
            out,
            "@group(0) @binding({binding}) var {texture}: texture_2d<f32>;"
        );
        binding += 1;
        let _ = writeln!(
            out,
            "@group(0) @binding({binding}) var {texture}_sampler: sampler;"
        );
        binding += 1;
    }
    out.push('\n');

    // The vertex input, at the locations the producer binds its buffers to. A body naming a field
    // the family does not declare does not compile, which is the point of generating this.
    out.push_str("struct In {\n");
    // The drawable's slot, as the draw's own `firstInstance`. A builtin rather than an attribute
    // because it is one number a draw: bound as a vertex buffer it would be one number a vertex.
    out.push_str("    @builtin(instance_index) instance_index: u32,\n");
    for attribute in attributes {
        let Some(wgsl) = attribute_type(attribute.declared) else {
            return Err(Error::InvalidAttribute {
                name: attribute.name,
            });
        };
        let _ = writeln!(
            out,
            "    @location({}) {}: {},",
            attribute.binding,
            attribute_name(attribute.name),
            wgsl
        );
    }
    out.push_str("}\n\n");

    out.push_str(PRELUDE);
    out.push('\n');
    out.push_str(surface.placement());
    out.push('\n');
    out.push_str(body);
    Ok(out)
}

/// The binding's name from its type name: `BackgroundDrawableUbo` gives `background_drawable_ubo`.
fn binding_name(type_name: &str) -> String {
    let mut out = String::with_capacity(type_name.len() + 4);
    for (at, ch) in type_name.char_indices() {
        if ch.is_ascii_uppercase() {
            if at != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// The background family's body.
///
/// The simplest thing the producer emits: a quad, a matrix, a color and an opacity. Written
/// against the generated declarations above it, so a field renamed in the ABI breaks the build
/// here rather than drawing something else.
///
/// The position goes through `place` rather than through a matrix multiply, which is what lets the
/// same body draw on a plane and on a globe. A background has no surface beyond those two: it
/// covers the viewport rather than a tile, so the producer neither anchors it nor raises it.
pub const BACKGROUND_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = background_drawable_ubo[ubo_index];
    var out: Out;
    out.clip = place(in.background_pos, drawable.matrix);
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    let props = background_props_ubo[0];
    return props.color * props.opacity;
}
";

/// What every body and every placement may use, prepended to both by [`module`].
///
/// `ubo_index` is the drawable's slot in the consolidated blocks. It arrives as the draw's
/// `firstInstance`, which the vertex stage reads as `@builtin(instance_index)` and the entry point
/// copies here — so a placement and a body both reach it as a plain name and neither has to carry
/// it through a parameter.
///
/// # Why not a push constant
///
/// Measured. `vkCmdPushConstants` once a draw costs 0.9 us a draw on V3D against 0.014 on RADV --
/// eleven times the cost of the draw it accompanies -- and 0.34 us on Adreno. `firstInstance` is a
/// field of the draw call that is already being made, so it costs nothing on any of the three.
/// See `tests/bench-baselines/`.
///
/// `var<private>` and not a parameter: the surfaces read it too, and threading it through `place`
/// would put it in a signature every family's body has to repeat.
///
/// A data-driven paint property arrives twice: as a vertex attribute holding the value at two zoom
/// levels, and as a `_t` field in the drawable block saying how far between them this frame is.
/// Mixing the two is the same arithmetic in every family, so it is written once.
///
/// A color arrives packed: four floats holding two colors, each color's four channels folded into
/// two floats. Unpacking is mbgl's, and the constant is its own -- 255 per channel, the high
/// channel scaled by 256.
///
/// A matrix arrives as four columns and is never assembled into one. `transform` applies it,
/// which is the four multiply-adds a matrix multiply is anyway. Two vendor compilers each refuse
/// one half of the obvious spelling -- see [`crate::preamble::MATRIX`] -- and this is the form
/// neither objects to. The columns are columns: the producer writes column-major.
pub const PRELUDE: &str = r"
var<private> ubo_index: u32;

fn transform(columns: array<vec4<f32>, 4>, position: vec3<f32>) -> vec4<f32> {
    return columns[0] * position.x
        + columns[1] * position.y
        + columns[2] * position.z
        + columns[3];
}

fn mix_value(packed: vec2<f32>, t: f32) -> f32 {
    return mix(packed.x, packed.y, t);
}

fn unpack_color(packed: vec2<f32>) -> vec4<f32> {
    let lo = vec2<f32>(floor(packed.x / 256.0), packed.x - floor(packed.x / 256.0) * 256.0);
    let hi = vec2<f32>(floor(packed.y / 256.0), packed.y - floor(packed.y / 256.0) * 256.0);
    return vec4<f32>(lo, hi) / 255.0;
}

fn mix_color(packed: vec4<f32>, t: f32) -> vec4<f32> {
    return mix(unpack_color(packed.xy), unpack_color(packed.zw), t);
}
";

/// The fill family's body.
///
/// Position in tile units through the surface's `place`; color and opacity are data-driven, so each
/// is mixed between its two zoom endpoints by the `_t` the block carries.
///
/// Nothing here knows which surface it is on. A fill has no height of its own, so the third
/// component is zero -- the plane and the raise both read it, and only an extrusion ever sends one.
pub const FILL_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_drawable_ubo[ubo_index];
    var out: Out;
    out.clip = place(vec3<f32>(vec2<f32>(in.fill_pos), 0.0), drawable.matrix);
    out.color = mix_color(in.fill_color, drawable.color_t);
    out.opacity = mix_value(in.fill_opacity, drawable.opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return in.color * in.opacity;
}
";

/// The fill-outline family's body.
///
/// The same geometry as a fill, drawn as lines with the outline color. `fill_outline_color` is its
/// own attribute: an outline is not the fill's color at a different opacity.
pub const FILL_OUTLINE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) outline_color: vec4<f32>,
    @location(1) opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_drawable_ubo[ubo_index];
    var out: Out;
    out.clip = place(vec3<f32>(vec2<f32>(in.fill_pos), 0.0), drawable.matrix);
    out.outline_color = mix_color(in.fill_outline_color, drawable.color_t);
    out.opacity = mix_value(in.fill_opacity, drawable.opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return in.outline_color * in.opacity;
}
";

/// The circle family's body.
///
/// The position is the circle's center and the data attribute carries the corner it is extruded
/// to, which is what makes one vertex buffer draw a disc. Six data-driven properties, each with
/// its own `_t`.
pub const CIRCLE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) stroke_color: vec4<f32>,
    @location(2) extrude: vec2<f32>,
    @location(3) radius: f32,
    @location(4) blur: f32,
    @location(5) opacity: f32,
    @location(6) stroke_width: f32,
    @location(7) stroke_opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = circle_drawable_ubo[ubo_index];
    var out: Out;

    let radius = mix_value(in.circle_radius, drawable.radius_t);
    let stroke_width = mix_value(in.circle_stroke_width, drawable.stroke_width_t);
    // The low two bits of the position say which corner this vertex is, as mbgl packs it.
    let corner = vec2<f32>(vec2<i32>(in.circle_pos) % 2) * 2.0 - 1.0;
    let reach = (radius + stroke_width) * drawable.extrude_scale;

    out.clip = place(
        vec3<f32>(vec2<f32>(in.circle_pos) + corner * reach, 0.0),
        drawable.matrix
    );
    out.extrude = corner;
    out.color = mix_color(in.circle_color, drawable.color_t);
    out.stroke_color = mix_color(in.circle_stroke_color, drawable.stroke_color_t);
    out.radius = radius;
    out.blur = mix_value(in.circle_blur, drawable.blur_t);
    out.opacity = mix_value(in.circle_opacity, drawable.opacity_t);
    out.stroke_width = stroke_width;
    out.stroke_opacity = mix_value(in.circle_stroke_opacity, drawable.stroke_opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let distance = length(in.extrude) * (in.radius + in.stroke_width);
    let antialias = in.blur + 1.0;
    let fill = 1.0 - smoothstep(in.radius - antialias, in.radius + antialias, distance);
    let outer = in.radius + in.stroke_width;
    let stroke = 1.0 - smoothstep(outer - antialias, outer + antialias, distance);
    return mix(
        in.stroke_color * in.stroke_opacity * stroke,
        in.color * in.opacity,
        fill
    );
}
";

/// The raster family's body.
///
/// Two pictures and a fade between them: a tile's own and its parent's, which is how a raster
/// layer stays covered while a finer tile is still arriving. The texture coordinates arrive as
/// `i16` over the same 8192 extent the positions use, not as normalized floats, which is what
/// gives a tile enough precision to be sampled at a fraction of a texel.
///
/// The color adjustments are mbgl's, in its order -- spin, saturation, contrast, brightness --
/// and the pair of pictures is un-premultiplied before the mix and premultiplied after, because
/// they are blended against each other rather than composited.
pub const RASTER_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = raster_drawable_ubo[ubo_index];
    let props = raster_evaluated_props_ubo[0];
    var out: Out;

    let texel = vec2<f32>(in.raster_texture_pos);
    let own = ((texel / 8192.0) - 0.5) / props.buffer_scale + 0.5;
    let parent = own * props.scale_parent + props.tl_parent;
    out.uv = vec4<f32>(own, parent);

    out.clip = place(vec3<f32>(vec2<f32>(in.raster_pos), 0.0), drawable.matrix);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let props = raster_evaluated_props_ubo[0];

    // Un-premultiplied before mixing: the two are blended against each other, not composited,
    // so they have to be in the same space first.
    var own = textureSample(raster_image0, raster_image0_sampler, in.uv.xy);
    var parent = textureSample(raster_image1, raster_image1_sampler, in.uv.zw);
    if own.a > 0.0 {
        own = vec4<f32>(own.rgb / own.a, own.a);
    }
    if parent.a > 0.0 {
        parent = vec4<f32>(parent.rgb / parent.a, parent.a);
    }
    let mixed = mix(own, parent, props.fade_t);
    let alpha = mixed.a * props.opacity;

    // Spin, as a rotation of the channels against each other.
    var rgb = vec3<f32>(
        dot(mixed.rgb, props.spin_weights.xyz),
        dot(mixed.rgb, props.spin_weights.zxy),
        dot(mixed.rgb, props.spin_weights.yzx)
    );

    // Saturation, toward or away from the pixel's own gray.
    let average = (mixed.r + mixed.g + mixed.b) / 3.0;
    rgb += (average - rgb) * props.saturation_factor;

    // Contrast, about the middle.
    rgb = (rgb - 0.5) * props.contrast_factor + 0.5;

    // Brightness, as a mix between the two levels the style names. mbgl passes `low` into the
    // vector it calls high and vice versa; that is transcribed rather than corrected, because the
    // two are the ends of a mix and swapping them is what inverts the ramp.
    let high = vec3<f32>(props.brightness_low);
    let low = vec3<f32>(props.brightness_high);
    let adjusted = mix(high, low, rgb);

    return vec4<f32>(adjusted * alpha, alpha);
}
";

/// The color-relief family's body.
///
/// The one family whose picture *is* the elevation. It reads the DEM per fragment, finds the pair
/// of style stops that bracket the height there, and mixes their colors -- so unlike every other
/// raised family it samples the elevation in the fragment stage rather than the vertex stage, for
/// its own content rather than to place itself.
///
/// The stops arrive as two one-dimensional pictures, an elevation table and a color table of the
/// same width, sampled at texel centers so nearest filtering cannot round to a neighbor.
///
/// # Why the search is a bisection
///
/// A linear walk is the same answer and costs the whole table on every pixel of every frame. The
/// bound is the table's own width, which the style fixes.
pub const COLOR_RELIEF_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = color_relief_drawable_ubo[ubo_index];
    let tile = color_relief_tile_props_ubo[ubo_index];
    var out: Out;

    // Into the tile's own square inside the padded image: the border ring is one texel a side,
    // so the interior spans `1/stride` to `(dim + 1)/stride`.
    let epsilon = 1.0 / tile.dimension;
    let scale = (tile.dimension.x - 2.0) / tile.dimension.x;
    out.uv = (vec2<f32>(in.color_relief_texture_pos) / 8192.0) * scale + epsilon;

    out.clip = place(vec3<f32>(vec2<f32>(in.color_relief_pos), 0.0), drawable.matrix);
    return out;
}

// The stop at `index`, at its texel's center.
fn elevation_stop(index: i32, stops: i32) -> f32 {
    let x = (f32(index) + 0.5) / f32(stops);
    return textureSample(
        color_relief_elevation_stops,
        color_relief_elevation_stops_sampler,
        vec2<f32>(x, 0.5)
    ).r;
}

fn color_stop(index: i32, stops: i32) -> vec4<f32> {
    let x = (f32(index) + 0.5) / f32(stops);
    return textureSample(
        color_relief_color_stops,
        color_relief_color_stops_sampler,
        vec2<f32>(x, 0.5)
    );
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let tile = color_relief_tile_props_ubo[ubo_index];
    let props = color_relief_evaluated_props_ubo[0];
    let stops = tile.color_ramp_size;

    // The elevation here, unpacked the way the DEM encodes it: the texel times 255, its alpha
    // replaced by -1, dotted with the unpack vector.
    let texel = textureSample(color_relief_image, color_relief_image_sampler, in.uv) * 255.0;
    let elevation = dot(vec4<f32>(texel.rgb, -1.0), tile.unpack);

    // The pair of stops that bracket it, by halving the range.
    var right = stops - 1;
    var left = 0;
    while right - left > 1 {
        let middle = (right + left) / 2;
        if elevation < elevation_stop(middle, stops) {
            right = middle;
        } else {
            left = middle;
        }
    }

    let low = elevation_stop(left, stops);
    let high = elevation_stop(right, stops);
    // Two stops at one elevation is a ramp with a hard edge in it, and the division that would
    // find where between them this pixel sits has nothing to divide by.
    let span = high - low;
    var t = 0.0;
    if abs(span) >= 0.0001 {
        t = clamp((elevation - low) / span, 0.0, 1.0);
    }

    return props.opacity * mix(color_stop(left, stops), color_stop(right, stops), t);
}
";

/// The line family's body.
///
/// The position attribute carries the point and its normal together, as mbgl packs it: the low bit
/// of each component is the normal's sign and the rest is the coordinate. The data attribute
/// carries the extrusion and the line's distance along itself.
pub const LINE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) width: f32,
    @location(3) blur: f32,
    @location(4) opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = line_drawable_ubo[ubo_index];
    var out: Out;

    // The normal is the low bit of each component; the position is what is left.
    let packed = vec2<i32>(in.line_pos_normal);
    let normal = vec2<f32>(packed % 2) * 2.0 - 1.0;
    let position = vec2<f32>(packed / 2);

    let width = mix_value(in.line_width, drawable.width_t);
    let gapwidth = mix_value(in.line_gap_width, drawable.gapwidth_t);
    let offset = mix_value(in.line_offset, drawable.offset_t);
    // A gap splits the line in two, each half the remaining width.
    let half = select(width * 0.5, gapwidth * 0.5 + width, gapwidth > 0.0);
    let extrude = vec2<f32>(in.line_data.xy) / 128.0 - 1.0;

    out.clip = place(
        vec3<f32>(position + extrude * (half + offset) * drawable.ratio, 0.0),
        drawable.matrix
    );
    out.normal = normal;
    out.width = half;
    out.color = mix_color(in.line_color, drawable.color_t);
    out.blur = mix_value(in.line_blur, drawable.blur_t);
    out.opacity = mix_value(in.line_opacity, drawable.opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let distance = length(in.normal) * in.width;
    let antialias = in.blur + 1.0;
    let coverage = 1.0 - smoothstep(in.width - antialias, in.width + antialias, distance);
    return in.color * in.opacity * coverage;
}
";
