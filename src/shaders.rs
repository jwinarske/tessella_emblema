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
        // Eight shorts. No family in the ABI's tables declares one -- the arm is here because the
        // enum has the variant, not because anything reads it. An earlier comment here said this
        // was a symbol's packed placement; it is not, and the symbol families declare `UShort4`.
        AttributeDataType::UShort8 => Some("array<u32, 8>"),
        // Invalid is the ABI saying it does not know, and a shader cannot read a
        // type nobody named. Refused by the caller rather than guessed at here.
        AttributeDataType::Invalid => None,
    }
}

/// Drops the `id` prefix and whichever of `suffixes` the name ends with.
///
/// Each strip is applied to the result of the last, which is the whole of what this exists for: a
/// chain of `strip_suffix(..).unwrap_or(declared)` falls back to the *original* string, so a name
/// that does not end in the suffix loses the prefix strip too. That is how
/// `idFillExtrusionDecimalsEdAttribute` came out as `id_fill_extrusion_decimals_ed_attribute` --
/// nine of the ABI's seventy-three attribute ids do not end in `VertexAttribute`, and nothing
/// read one until the extrusions.
fn strip_id<'a>(declared: &'a str, suffixes: &[&str]) -> &'a str {
    let trimmed = declared.strip_prefix("id").unwrap_or(declared);
    for suffix in suffixes {
        if let Some(shorter) = trimmed.strip_suffix(suffix) {
            return shorter;
        }
    }
    trimmed
}

/// A WGSL name for a texture, from the id the header spells.
///
/// `idRasterImage0Texture` becomes `raster_image0`, and its sampler `raster_image0_sampler`. The
/// `id` prefix and the `Texture` suffix say nothing a shader needs, as with an attribute's.
#[must_use]
pub fn texture_name(declared: &str) -> String {
    let trimmed = strip_id(declared, &["Texture"]);
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
    let trimmed = strip_id(declared, &["VertexAttribute", "Attribute"]);
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
///
/// Two components, not three. The ABI recorded this attribute as `Float3` until tessella#326; it
/// is `Short2`, which is what the producer sends and what mbgl's own shader declares --
/// `vec4(in_position, 0.0, 1.0)`, supplying the third component itself. This body read three
/// floats from two 16-bit integers for as long as the table said so, and nothing caught it:
/// the bench that drew with this pipeline writes its own vertices as three floats.
pub const BACKGROUND_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = background_drawable_ubo[ubo_index];
    var out: Out;
    out.clip = place(vec3<f32>(vec2<f32>(in.background_pos), 0.0), drawable.matrix);
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

/// The fill-extrusion family's body: a building's roof.
///
/// The first family with a height, which is what `place`'s third component has been carrying
/// unused since the surfaces landed. On a plane the drawable's matrix takes meters in its `z` row
/// -- `world_to_camera` post-multiplies it by pixels-per-meter -- and on the anchored bend the
/// height goes along the sphere's normal through `d_h`, the coefficient declared in
/// `GlobeBendUbo` for exactly this and read by nothing until now.
///
/// It has no direct-bend variant, and that is not an omission: the direct bend has no height term
/// at all, so a family that leaves the surface cannot be drawn on it. The producer agrees -- it
/// writes extrusions a bend block, which is what the anchored surface needs.
///
/// # What this draws, and what it does not
///
/// The roof only. mbgl draws the walls as a second, instanced drawable over the same outline, and
/// that is its own family. The vertical gradient down a wall is therefore not here: it applies
/// only where the surface normal is horizontal, and every vertex of a roof faces straight up, so
/// including it would be a branch nothing can take. It belongs with the walls and has to arrive
/// with them.
pub const FILL_EXTRUSION_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) shade: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_extrusion_drawable_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    var out: Out;

    let footprint = vec2<f32>(in.fill_extrusion_pos);

    // The fraction the position was packed with, seven bits an axis. Integer tile units leave it
    // zero, and a simplification pass produces fractional positions -- at which point a roof that
    // dropped this would part company with the walls standing under it.
    let packed = floor(f32(in.fill_extrusion_decimals_ed.x) / 2.0);
    let high = floor(packed / 256.0);
    let decimals = vec2<f32>(high, packed - high * 256.0) / 128.0;

    let base = max(mix_value(in.fill_extrusion_base, drawable.base_t), 0.0);
    let height = max(mix_value(in.fill_extrusion_height, drawable.height_t), 0.0);
    // Every vertex of a roof is at the building's top, so this always takes the height. Written
    // as mbgl's selection rather than as `height` alone: the base is the other arm of the term it
    // computes, and a shader that dropped it would agree with the oracle by coincidence of what
    // this drawable happens to carry rather than by computing the same thing.
    let on_roof = 1.0;
    let z = select(base, height, on_roof > 0.0);

    // How bright the surface already is. A pale building takes a narrower range of shading than a
    // dark one, which is what keeps a light roof from blowing out.
    var color = mix_color(in.fill_extrusion_color, drawable.color_t);
    let luminance = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
    // Slight ambient, so nothing is ever completely black.
    color += vec4<f32>(0.03, 0.03, 0.03, 1.0);

    // A roof faces straight up, so the directional term is the light's own elevation.
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    let facing = clamp(dot(normal, props.light_position), 0.0, 1.0);
    let least = 1.0 - props.light_intensity;
    let most = max(1.0 - luminance + props.light_intensity, 1.0);
    let directional = mix(least, most, facing);

    // Shading is floored at a tint complementary to the light, so a colored light leaves its
    // opposite in the shadows rather than driving them to black.
    let floor_light = mix(vec3<f32>(0.0), vec3<f32>(0.3), 1.0 - props.light_color);
    let lit = clamp(color.rgb * directional * props.light_color, floor_light, vec3<f32>(1.0));

    // Premultiplied: rgb scaled by the opacity and alpha set to it, over a color whose alpha
    // was one.
    out.shade = vec4<f32>(lit, 1.0) * props.opacity;

    // The height in meters, which is what `place`'s third component is for: the plane's matrix
    // takes meters in its `z` row, and the anchored bend lifts along the sphere's normal.
    out.clip = place(vec3<f32>(footprint + decimals, z), drawable.matrix);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return in.shade;
}
";

/// The symbol-icon family's body.
///
/// A sprite placed where a label was laid out. The hardest placement of any family here, because a
/// symbol is positioned twice: the anchor goes through the *label plane* matrix, where the layout
/// decided it, and the glyph's own corner offset is added in that plane before the result goes
/// through the coordinate matrix to clip space. Adding the offset in tile units instead would
/// make type change size across a tile.
///
/// It is also the first family to read [`GLOBAL_PAINT_PARAMS_UBO`], which carries what a frame
/// knows and a drawable does not: how far the camera is from the center, the viewport's aspect,
/// and how far the fades have advanced.
///
/// # Where this departs from fluorite
///
/// fluorite packs `projected_pos` and `fade_opacity` into one attribute slot, because Filament ran
/// out of custom slots. Nothing here has that limit: each attribute binds at the location the
/// producer gives it, so the two are read separately.
///
/// [`GLOBAL_PAINT_PARAMS_UBO`]: tessella_capture_abi::generated::ubo_layouts::GLOBAL_PAINT_PARAMS_UBO
pub const SYMBOL_ICON_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex: vec2<f32>,
    @location(1) fade: f32,
    @location(2) opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = symbol_drawable_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let anchor = vec2<f32>(in.symbol_pos_offset.xy);
    let corner = vec2<f32>(in.symbol_pos_offset.zw);
    let tex = vec2<f32>(in.symbol_data.xy);
    let sized = vec2<f32>(in.symbol_data.zw);
    let pixel_offset = vec2<f32>(in.symbol_pixel_offset.xy);
    // The minimum font scale rides in the spare half, over 256. An icon in a label does not shrink
    // below it, which is what keeps a shield legible where the text beside it is small.
    let min_font_scale = vec2<f32>(in.symbol_pixel_offset.zw) / 256.0;

    // The layout's own position, and the angle of the segment the label sits on.
    let placed = in.symbol_projected_pos;
    let segment_angle = -placed.z;

    // Three ways a size arrives, which is what the two flags distinguish: interpolated between
    // the feature's own stops, the feature's lower stop alone, or the layer's constant.
    let smallest = floor(sized.x * 0.5);
    var size = drawable.size;
    if drawable.is_size_zoom_constant == 0 && drawable.is_size_feature_constant == 0 {
        size = mix(smallest, sized.y, drawable.size_t) / 128.0;
    } else if drawable.is_size_zoom_constant != 0 && drawable.is_size_feature_constant == 0 {
        size = smallest / 128.0;
    }

    // How far the anchor is from the camera, which is what makes distant type smaller. Laid out
    // in pitched space distance shrinks a label and this counteracts part of it; laid out in
    // viewport space it grows one, so the ratio inverts.
    let anchor_point = transform(drawable.matrix, vec3<f32>(anchor, 0.0));
    let to_anchor = anchor_point.w;
    var ratio = global.camera_to_center_distance / to_anchor;
    if drawable.pitch_with_map != 0 {
        ratio = to_anchor / global.camera_to_center_distance;
    }
    // The zero floor is what stops an overzoomed near-field symbol becoming enormous.
    let perspective = clamp(0.5 + 0.5 * ratio, 0.0, 4.0);
    if drawable.is_offset == 0 {
        size *= perspective;
    }
    var font_scale = size;
    if drawable.is_text_prop != 0 {
        font_scale = size / 24.0;
    }

    // A label horizontal in tile units is not horizontal on screen. Its angle there is found by
    // projecting a short horizontal line and measuring what became of it.
    var rotation = 0.0;
    if drawable.rotate_symbol != 0 {
        let along = transform(drawable.matrix, vec3<f32>(anchor + vec2<f32>(1.0, 0.0), 0.0));
        let here = anchor_point.xy / anchor_point.w;
        let there = along.xy / along.w;
        rotation = atan2((there.y - here.y) / global.aspect_ratio, there.x - here.x);
    }

    let turn = segment_angle + rotation;
    let spun = mat2x2<f32>(cos(turn), -sin(turn), sin(turn), cos(turn));

    // The anchor in the label plane, with the corner added there rather than in tile units.
    let in_plane = transform(drawable.label_plane_matrix, vec3<f32>(placed.xy, 0.0));
    // Sixteenths of a pixel, where the SDF family's same attribute is whole pixels. mbgl's two
    // shaders differ here; `symbol_icon.vertex.glsl` is what this one follows.
    let offset = corner / 32.0 * max(min_font_scale, vec2<f32>(font_scale, font_scale))
        + pixel_offset / 16.0;
    let on_plane = in_plane.xy / in_plane.w + spun * offset;
    out.clip = place(vec3<f32>(on_plane, 0.0), drawable.coord_matrix);

    // Two values in one float: the opacity in the high bits and the direction it is moving in the
    // low bit, so a label fading in and one fading out are told apart.
    let whole = floor(in.symbol_fade_opacity / 2.0);
    let rising = in.symbol_fade_opacity - whole * 2.0;
    var change = -global.symbol_fade_change;
    if rising > 0.5 {
        change = global.symbol_fade_change;
    }
    out.fade = clamp(whole / 127.0 + change, 0.0, 1.0);

    out.tex = tex / drawable.texsize;
    out.opacity = mix_value(in.symbol_opacity, drawable.opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The sheet is premultiplied, so the opacity and the fade scale it directly.
    let sprite = textureSample(symbol_image, symbol_image_sampler, in.tex);
    return sprite * (in.opacity * in.fade);
}
";

/// The symbol-SDF family's body: text, and its halo.
///
/// The icon's placement exactly -- see [`SYMBOL_ICON_BODY`] -- with four more paint properties and
/// a different fragment stage. A glyph is a signed distance field rather than a picture, so the
/// edge is found by thresholding the distance and the width of that threshold is what makes type
/// look right at a size rather than blurred or bitten.
///
/// # The atlas is read from red, where mbgl reads alpha
///
/// `GLYPH_ATLAS_FORMAT` is `TexturePixelType::Alpha`, one channel. GL's `GL_ALPHA` samples as
/// `(0, 0, 0, a)`, which is why mbgl reads `.a`; Vulkan has no alpha-only format, so one channel
/// is `R8_UNORM` and samples as `(r, 0, 0, 1)`. Reading `.a` here would be 1.0 at every pixel --
/// every glyph a solid block. The bytes are the same; only the channel they arrive in differs.
///
/// A one-channel atlas is also a known hazard on this hardware: a sibling consumer found Adreno
/// GLES sampling `r8` as zero under minification and widened its atlases to RGBA. Whether Vulkan
/// on that part does the same is untested here.
///
/// # The halo is subtracted, not drawn under
///
/// A translucent fill wants a translucent halo *inside* it. Drawing the halo first and the fill
/// over it doubles the coverage where they meet, so the fill's own alpha is subtracted from the
/// halo's instead.
pub const SYMBOL_SDF_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex: vec2<f32>,
    @location(1) fade: f32,
    @location(2) font_scale: f32,
    @location(3) gamma: f32,
    @location(4) paint: vec4<f32>,
    @location(5) opacity: f32,
    @location(6) halo_width: f32,
    @location(7) halo_blur: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = symbol_drawable_ubo[ubo_index];
    let tile = symbol_tile_props_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let anchor = vec2<f32>(in.symbol_pos_offset.xy);
    let corner = vec2<f32>(in.symbol_pos_offset.zw);
    let tex = vec2<f32>(in.symbol_data.xy);
    let sized = vec2<f32>(in.symbol_data.zw);
    let pixel_offset = vec2<f32>(in.symbol_pixel_offset.xy);
    let placed = in.symbol_projected_pos;
    let segment_angle = -placed.z;

    let smallest = floor(sized.x * 0.5);
    var size = drawable.size;
    if drawable.is_size_zoom_constant == 0 && drawable.is_size_feature_constant == 0 {
        size = mix(smallest, sized.y, drawable.size_t) / 128.0;
    } else if drawable.is_size_zoom_constant != 0 && drawable.is_size_feature_constant == 0 {
        size = smallest / 128.0;
    }

    let anchor_point = transform(drawable.matrix, vec3<f32>(anchor, 0.0));
    let to_anchor = anchor_point.w;
    var ratio = global.camera_to_center_distance / to_anchor;
    if drawable.pitch_with_map != 0 {
        ratio = to_anchor / global.camera_to_center_distance;
    }
    let perspective = clamp(0.5 + 0.5 * ratio, 0.0, 4.0);
    if drawable.is_offset == 0 {
        size *= perspective;
    }
    // Twenty-four is the size the glyphs were rasterized at, so this is how far the field has to
    // be stretched to reach the size the style asked for.
    var font_scale = size;
    if drawable.is_text_prop != 0 {
        font_scale = size / 24.0;
    }

    var rotation = 0.0;
    if drawable.rotate_symbol != 0 {
        let along = transform(drawable.matrix, vec3<f32>(anchor + vec2<f32>(1.0, 0.0), 0.0));
        let here = anchor_point.xy / anchor_point.w;
        let there = along.xy / along.w;
        rotation = atan2((there.y - here.y) / global.aspect_ratio, there.x - here.x);
    }
    let turn = segment_angle + rotation;
    let spun = mat2x2<f32>(cos(turn), -sin(turn), sin(turn), cos(turn));

    let in_plane = transform(drawable.label_plane_matrix, vec3<f32>(placed.xy, 0.0));
    let offset = corner / 32.0 * font_scale + pixel_offset;
    let on_plane = in_plane.xy / in_plane.w + spun * offset;
    let clip = place(vec3<f32>(on_plane, 0.0), drawable.coord_matrix);
    out.clip = clip;

    let whole = floor(in.symbol_fade_opacity / 2.0);
    let rising = in.symbol_fade_opacity - whole * 2.0;
    var change = -global.symbol_fade_change;
    if rising > 0.5 {
        change = global.symbol_fade_change;
    }
    out.fade = clamp(whole / 127.0 + change, 0.0, 1.0);

    // The halo pass and the fill pass are two draws of the same geometry, told apart here.
    let fill = mix_color(in.symbol_color, drawable.fill_color_t);
    let halo = mix_color(in.symbol_halo_color, drawable.halo_color_t);
    out.paint = select(fill, halo, tile.is_halo != 0);

    out.tex = tex / drawable.texsize;
    out.font_scale = font_scale;
    // The perspective divide the threshold has to be measured in, which is the clip `w`.
    out.gamma = clip.w;
    out.opacity = mix_value(in.symbol_opacity, drawable.opacity_t);
    out.halo_width = mix_value(in.symbol_halo_width, drawable.halo_width_t);
    out.halo_blur = mix_value(in.symbol_halo_blur, drawable.halo_blur_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let tile = symbol_tile_props_ubo[ubo_index];
    let global = global_paint_params_ubo[0];

    // The atlas stores eight distance units a pixel.
    let sdf_px = 8.0;
    let edge_gamma = 0.105 / max(global.pixel_ratio, 1e-6);
    let font_gamma = in.font_scale * tile.gamma_scale;
    let fill_gamma = edge_gamma / font_gamma;
    let halo_gamma = (in.halo_blur * 1.19 / sdf_px + edge_gamma) / font_gamma;

    let is_halo = tile.is_halo != 0;
    let gamma = select(fill_gamma, halo_gamma, is_halo) * in.gamma;

    // Where the letter's edge sits in the field, and where a halo's does.
    let fill_edge = (256.0 - 64.0) / 256.0;
    let inner = select(fill_edge, fill_edge + halo_gamma * in.gamma, is_halo);

    // `.r`, where mbgl reads `.a`: a one-channel format is `R8_UNORM` here and alpha reads 1.0.
    let distance = textureSample(symbol_image, symbol_image_sampler, in.tex).r;
    var alpha = smoothstep(inner - gamma, inner + gamma, distance);
    if is_halo {
        // Subtracted rather than drawn under, so a translucent fill keeps a translucent halo.
        let halo_edge = (6.0 - in.halo_width / in.font_scale) / sdf_px;
        alpha = min(smoothstep(halo_edge - gamma, halo_edge + gamma, distance), 1.0 - alpha);
    }

    return in.paint * (alpha * in.opacity * in.fade);
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
    @location(2) outset: f32,
    @location(3) inset: f32,
    @location(4) gamma_scale: f32,
    @location(5) blur: f32,
    @location(6) opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = line_drawable_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    // The byte pair encodes a unit vector over 63, biased by 128.
    let extrude_scale = 63.0;
    // The distance the edge fades out over: half a device pixel each side.
    let antialiasing = 1.0 / max(global.pixel_ratio, 1e-6) / 2.0;

    // The normal is the low bit of each component; the center is what is left.
    let packed = vec2<f32>(in.line_pos_normal);
    let center = floor(packed * 0.5);
    var normal = packed - 2.0 * center;
    normal.y = normal.y * 2.0 - 1.0;

    // Four unnormalized bytes. The first two are the extrusion; the low two bits of the third say
    // which way a round end point's extrude points, and the rest of the third with the fourth is
    // the distance along the line, which the plain family has no use for.
    let data = vec4<f32>(in.line_data);
    let extrude = data.xy - 128.0;
    let direction = (data.z % 4.0) - 1.0;
    let along = (floor(data.z / 4.0) + data.w * 64.0) * 2.0;

    let gapwidth = mix_value(in.line_gap_width, drawable.gapwidth_t) * 0.5;
    let halfwidth = mix_value(in.line_width, drawable.width_t) * 0.5;
    let line_offset = -1.0 * mix_value(in.line_offset, drawable.offset_t);

    // The quad reaches `outset` from the center and the fill starts at `inset`. With no gap the
    // inset is zero and the line is solid; with one, both edges are drawn and the middle left
    // open, which is how a road casing is a casing rather than a slab.
    let inset = gapwidth + select(0.0, antialiasing, gapwidth > 0.0);
    let outset = gapwidth
        + halfwidth * select(1.0, 2.0, gapwidth > 0.0)
        + select(antialiasing, 0.0, halfwidth == 0.0);

    // The extrusion down to a normal and back up by this vertex's line width.
    let dist = outset * extrude / extrude_scale;

    // A line drawn to the side of the real one. The vector points along the extrude, rotated
    // where a round end point's extrude points somewhere else.
    let u = 0.5 * direction;
    let t = 1.0 - abs(u);
    let turn = mat2x2<f32>(t, -u, u, t);
    let offset2 = line_offset * extrude / extrude_scale * normal.y * turn;

    let ratio = max(drawable.ratio, 1e-6);

    // Placed and extruded separately, because the projected extrusion is what says how much
    // perspective squashed this edge -- and that length is wanted on its own below. `displace` is
    // the surface's, since the extrusion is linear on a plane and is not on a sphere.
    let at = center + offset2 / ratio;
    let projected_extrude = displace(at, dist / ratio, drawable.matrix);
    let clip = place(vec3<f32>(at, 0.0), drawable.matrix) + projected_extrude;
    out.clip = clip;

    // How much the perspective view squashed or stretched the extrusion, which is what keeps the
    // fade a constant width in pixels. Guarded: a zero-length extrusion would be 0/0 and NaN a
    // whole line away.
    let unprojected = length(dist);
    let projected = length(projected_extrude.xy / clip.w * global.units_to_pixels);
    out.gamma_scale = unprojected / max(projected, 1e-6);

    out.normal = normal;
    out.outset = outset;
    out.inset = inset;
    out.color = mix_color(in.line_color, drawable.color_t);
    out.blur = mix_value(in.line_blur, drawable.blur_t);
    out.opacity = mix_value(in.line_opacity, drawable.opacity_t);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // How far this pixel is from the center of the line, in pixels.
    let distance = length(in.normal) * in.outset;

    // The fade: in across the inner edge for a line with a gap, out across the outer one.
    let blur2 = (in.blur + 1.0 / max(global_paint_params_ubo[0].pixel_ratio, 1e-6))
        * in.gamma_scale;
    let alpha = clamp(
        min(distance - (in.inset - blur2), in.outset - distance) / blur2,
        0.0,
        1.0
    );

    return in.color * (alpha * in.opacity);
}
";
