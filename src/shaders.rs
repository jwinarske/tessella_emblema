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
//! # A family has one module per permutation, not one module
//!
//! A data-driven paint property arrives as a vertex attribute where the style's value varies with
//! the feature and as *nothing at all* where it does not -- the layer's one evaluated value is
//! already in its properties block, and mbgl compiles the program with that attribute's
//! `HAS_UNIFORM_u_*` defined. So which attributes arrived decides the module's text, and it
//! arrives per drawable: [`crate::vertices::Plan::absent`] is the locations no descriptor
//! supplied, [`permutation`] is that set as a bitmask, and
//! [`crate::pipelines::Key::permutation`] carries it so the module cache can key on it.
//!
//! [`Property`] is one such property and both of its arms; [`Paint`] is a family's set of them
//! paired with what a drawable sent. A body never reads a paint attribute directly -- it calls the
//! `paint_<name>` function this module emits, which is the same text in every permutation.
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
//!
//! Group *one* exists for one family, and for a reason that has nothing to do with tidiness: the
//! instanced extrusion reads the next instance out of a storage buffer that belongs to one
//! geometry, where every binding above belongs to a layer. A set shared by the layer cannot hold
//! it, and binding it at an offset into a shared buffer would bring
//! `minStorageBufferOffsetAlignment` into a pass that binds everything at offset zero to stay
//! clear of it. The same family takes `ubo_index` as a push constant, because by then
//! `firstInstance` is carrying the instance -- see [`crate::draws`].

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
    /// An attribute naming a shared buffer whose declared type is not one packed scalar.
    ///
    /// The shared buffer's element is the same bytes as the attributes that name it, read as the
    /// words they were packed into: `Short2` is one `i32` and `UShort2` is one `u32`, which is
    /// mbgl's own `OutlineInstance`. A third type would be bytes nothing here knows how to read
    /// twice, and reading them the one way that is known would silently be the wrong way.
    Unpackable {
        /// The attribute, as the header spells it.
        name: &'static str,
        /// The declared type that has no packed scalar form.
        declared: AttributeDataType,
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
///
/// Public because it is half a statement. `device::vertex_format` is the other half -- how the
/// same bytes are fetched -- and the two have to agree about their count and their class. Neither
/// can be checked against the other while one of them is private.
#[must_use]
pub const fn attribute_type(declared: AttributeDataType) -> Option<&'static str> {
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

/// A data-driven paint property, and the two places its value can come from.
///
/// mbgl evaluates a paint property per layer and asks whether it varies: a `fill-color` given as a
/// plain color is one value for the whole layer, and one given as an expression over a feature's
/// own fields is a value per vertex. The first travels in the layer's evaluated-properties block,
/// the second as a vertex attribute -- and mbgl compiles a *different program* for each, which is
/// what its `#if !defined(HAS_UNIFORM_u_color)` guards select between and what its
/// `propertiesAsUniforms` names.
///
/// So a family does not have one module. It has one per set of its properties that arrived, and
/// which set that is comes off the wire per drawable: [`crate::vertices::Plan::absent`] is the
/// locations no descriptor supplied. Nothing else can decide it -- the family's table says what the
/// module *may* read and the plan says what this drawable *sent*.
///
/// # Why both arms are written out rather than derived
///
/// The uniform arm could almost be computed from the attribute's name: `idFillColorVertexAttribute`
/// is `color`, the drawable block carries `color_t` and the props block carries `color`. Almost --
/// mbgl spells a line's gap width `gapwidth` in the block and `GapWidth` in the attribute, and all
/// three symbol families read `tileProps.is_text ? props.text_opacity : props.icon_opacity`, which
/// is not a field name at all. A derivation would need a sanction list per exception, and the
/// exception is the half that matters.
///
/// Written out, the derivation becomes a *test* instead: `tests/properties.rs` checks every
/// attribute whose drawable block carries a `_t` has an entry here, which is the thing that would
/// otherwise be silently missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Property {
    /// The `@location` whose absence makes this a uniform read.
    pub location: u32,
    /// What the body calls it, without the `paint_` prefix [`module`] gives the function.
    ///
    /// mbgl's name for the property rather than the attribute's: a fill outline's color is
    /// `outline_color`, and the symbol families' is `fill_color` where the attribute is
    /// `symbol_color`.
    pub name: &'static str,
    /// The WGSL type both arms return.
    pub returns: &'static str,
    /// The arm read when the attribute arrived: the two zoom endpoints, mixed by the block's `_t`.
    pub varies: &'static str,
    /// The arm read when it did not: the layer's one evaluated value.
    pub constant: &'static str,
}

/// Which of a family's paint properties arrived, for one module.
///
/// Two slices rather than one resolved list, because the pair is what a caller has: the properties
/// are the family's and the absent locations are the drawable's. Holding them apart is also what
/// lets [`permutation`] be a function of the second alone, which is what the module cache keys on.
#[derive(Debug, Clone, Copy)]
pub struct Paint<'a> {
    /// The family's data-driven properties.
    pub properties: &'a [Property],
    /// The locations no descriptor supplied, from [`crate::vertices::Plan::absent`].
    pub absent: &'a [u32],
}

impl Paint<'_> {
    /// No properties and nothing absent: a family whose every attribute is geometry.
    pub const NONE: Paint<'static> = Paint {
        properties: &[],
        absent: &[],
    };
}

/// The module a set of absent locations selects, as a bitmask.
///
/// One bit per `@location`, so two drawables of a family whose attributes arrived alike share a
/// module and two that did not have their own. This is what [`crate::pipelines::Key::permutation`]
/// carries, and it is a consumer's number rather than the wire's: `batch::Key::permutation_key` is
/// mbgl's own hash over its property names, which says nothing about this crate's locations.
///
/// A location at 64 or above cannot have its own bit and takes bit 63 instead, which makes such a
/// set collide with another rather than silently equal it. No family mbgl declares comes close --
/// the widest table here is ten attributes -- so this is a total function rather than a case.
#[must_use]
pub fn permutation(absent: &[u32]) -> u64 {
    absent
        .iter()
        .map(|location| 1u64 << location.min(&63))
        .fold(0, |mask, bit| mask | bit)
}

/// Assembles a module for one (family, surface) pair: the blocks, the vertex input, the surface's
/// placement, then the family's body.
///
/// The family's blocks bind first, from zero, then the surface's, then two bindings for each
/// texture the surface samples. The body sees each block as a binding named after its type in
/// snake case, the vertex input as `In`, and the surface as `place`.
///
/// `instance_attributes` is empty for every family but the instanced extrusions. Where it is not,
/// its entries join `In` at their own locations, and the ones naming a `shared_buffer` are also
/// declared as the element of a storage array in group one -- the same bytes a second way, which
/// is how a body reads the *next* instance. Such a family also gets `ubo_index` as a push
/// constant, because `firstInstance` is carrying the instance by then; see [`crate::draws`].
///
/// `paint` is what makes this one module of several for a family rather than the only one. A
/// location in [`Paint::absent`] is left out of `In`, and each of [`Paint::properties`] becomes a
/// `paint_<name>` function reading whichever arm that leaves -- see [`Property`]. A body calls
/// those functions and never the attribute directly, which is what lets the same body serve every
/// permutation.
///
/// An absent location that no property claims is *not* refused here. The body then reads a field
/// `In` does not declare and the module fails to compile, which is the same loud failure a body
/// naming an undeclared block field gets and the same end [`crate::vertices::Plan::undeclared`]
/// chooses to fail at: a refusal would make a whole family undrawable over one attribute, where
/// this leaves a caller that can tell the two cases apart from the error it gets.
///
/// # Errors
///
/// [`Error`] from a block that cannot be declared, an attribute the ABI calls invalid, or a shared
/// attribute whose declared type is not one packed word.
pub fn module(
    surface: Surface,
    blocks: &[&UboLayout],
    attributes: &[ShaderAttribute],
    instance_attributes: &[ShaderAttribute],
    textures: &[ShaderTexture],
    paint: &Paint<'_>,
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
        // Padded to the stride the producer packs entries at, which for a drawable block is its
        // union's rather than its own -- see `slots::stride`. WGSL sizes `array<T>` from `T`, so
        // this is the only place that stride can be expressed.
        out.push_str(&declare(layout, crate::slots::stride(layout))?);
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

    // The buffer mbgl binds twice, where there is one: its attributes are registered so the bytes
    // reach the device as vertex input, and the fourth field of each `AttributeInfo` names the
    // binding the *same* buffer is bound at as a storage buffer. A body needs both -- the
    // attribute for this instance and the array for the next one, because an attribute fetch
    // cannot look ahead.
    //
    // The element's fields are the shared attributes in their own order, read as the words they
    // were packed into rather than as the pairs the attribute declares. Derived from the table so
    // it cannot drift from the attributes it is the same bytes as.
    let shared: Vec<&ShaderAttribute> = instance_attributes
        .iter()
        .filter(|attribute| attribute.shared_buffer.is_some())
        .collect();
    if !shared.is_empty() {
        out.push_str("struct SharedInstance {\n");
        for attribute in &shared {
            let Some(wgsl) = packed_type(attribute.declared) else {
                return Err(Error::Unpackable {
                    name: attribute.name,
                    declared: attribute.declared,
                });
            };
            let _ = writeln!(out, "    {}: {},", attribute_name(attribute.name), wgsl);
        }
        out.push_str("}\n\n");
        out.push_str(
            "@group(1) @binding(0) var<storage, read> shared_instance: array<SharedInstance>;\n\n",
        );
        // And the drawable's slot, which has nowhere else to travel on this family: `firstInstance`
        // addresses the instance-rate attribute fetch as well as the builtin, so a draw of more
        // than one instance cannot put the slot there. Vertex stage only, which is where
        // `ubo_index` is read.
        out.push_str("struct Pushed {\n    ubo_index: u32,\n}\n");
        out.push_str("var<push_constant> pushed: Pushed;\n\n");
    }

    // The vertex input, at the locations the producer binds its buffers to. A body naming a field
    // the family does not declare does not compile, which is the point of generating this.
    out.push_str("struct In {\n");
    // The drawable's slot, as the draw's own `firstInstance`. A builtin rather than an attribute
    // because it is one number a draw: bound as a vertex buffer it would be one number a vertex.
    out.push_str("    @builtin(instance_index) instance_index: u32,\n");
    for attribute in attributes.iter().chain(instance_attributes) {
        let Some(wgsl) = attribute_type(attribute.declared) else {
            return Err(Error::InvalidAttribute {
                name: attribute.name,
            });
        };
        // Left out where no descriptor supplied it. Vulkan asks that every input a vertex stage
        // declares have a `VkVertexInputAttributeDescription`, and nothing enforces that --
        // `benches/pipelines_build.rs` measures a module input with no description and RADV builds
        // the pipeline, so the attribute would be fetched from nowhere rather than refused. The
        // body reads this property through `paint_<name>` instead, which is declared below.
        if u32::try_from(attribute.binding).is_ok_and(|at| paint.absent.contains(&at)) {
            continue;
        }
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

    paint_functions(&mut out, paint);
    out.push_str(body);
    Ok(out)
}

/// The family's paint properties, each as the one arm this permutation leaves.
///
/// Emitted after the prelude, because the varying arm mixes with `mix_color` and `mix_value`, and
/// before the body, because that is what calls them.
///
/// Every property gets a function whether it varies or not, so the body is the same text either
/// way. `in` is a parameter rather than read from anywhere: the constant arm does not use it, and
/// giving the two arms different signatures would put the permutation back in the body.
fn paint_functions(out: &mut String, paint: &Paint<'_>) {
    if paint.properties.is_empty() {
        return;
    }
    // Two calls rather than one continued literal, and no bare `data` in the text:
    // `tests/shaders.rs` reads the module for which components of an attribute a body uses, and a
    // word in a comment counts as a use. "data-driven" here excused every component of a line's
    // packed `in.line_data`, which is how that was found.
    out.push_str("// The family's paint properties. Each is read from the attribute the\n");
    out.push_str("// producer sent, or from the layer's evaluated value where it sent none.\n");
    for property in paint.properties {
        let arm = if paint.absent.contains(&property.location) {
            property.constant
        } else {
            property.varies
        };
        let _ = writeln!(
            out,
            "fn paint_{}(in: In) -> {} {{\n    return {arm};\n}}",
            property.name, property.returns
        );
    }
    out.push('\n');
}

/// The WGSL scalar a shared attribute's bytes are read as inside the storage element.
///
/// The other half of [`attribute_type`] for the attributes that name a shared buffer: the *same
/// bytes*, read once as a vertex attribute and once as a word of a storage element. `Short2` is
/// two `i16` in one `i32` and `UShort2` is two `u16` in one `u32`, which is what mbgl's
/// `OutlineInstance` declares for those two attributes.
///
/// Only those two, and the rest return `None` rather than a guess. A `Float2` is not a packed
/// word at all, and a four-component type is two of them -- either would need the element's
/// field count to stop matching the attribute count, which is the correspondence that makes this
/// derivable from the table instead of written out beside it.
#[must_use]
pub const fn packed_type(declared: AttributeDataType) -> Option<&'static str> {
    match declared {
        AttributeDataType::Short2 => Some("i32"),
        AttributeDataType::UShort2 => Some("u32"),
        _ => None,
    }
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
/// floats from two 16-bit integers for as long as the table said so, and nothing caught it --
/// nor did anything catch `draw_cost`, which went on declaring a float-three vertex input
/// against this `vec2<i32>` and drew nothing on any device for it. Its pipeline takes the
/// format and the stride from the table now.
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
/// # A fragment stage has none
///
/// `instance_index` is a vertex builtin and a private variable is per invocation, so a fragment
/// stage's copy of this is zero whatever the vertex stage wrote. A body whose fragment stage
/// reads a per-drawable block therefore carries the slot through its own `Out` as a flat `u32`
/// and assigns it back -- seven of them do, every `*TilePropsUBO` being one entry per drawable.
/// Left out, such a body draws the *first* drawable's tile properties over every tile of its
/// layer: the second tile of a patterned fill is sampled against the first tile's atlas size,
/// which is a plausible wrong texel rather than nothing at all.
///
/// `benches/first_pixel.rs`'s `fill_pattern_at_one` is the case, and the complement is a
/// layer-wide block: those are one entry and are read at `[0]` in both stages.
///
/// # Except for one family, where it cannot be
///
/// The instanced extrusion. `firstInstance` is part of the address an instance-rate vertex
/// attribute is fetched from as well as being the builtin's base -- measured on RADV and V3D in
/// `benches/instance_fetch.rs` -- so a draw of more than one instance cannot put the slot there.
/// Its body assigns this from a push constant instead, which [`module`] declares for it, and the
/// bill is one `vkCmdPushConstants` per walls drawable rather than per draw in the frame.
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
///
/// # `vulkan_depth`, which every placement ends with
///
/// The matrices are mbgl's and mbgl's projection is OpenGL's: `mat4::perspective` sets
/// `out[10] = (far + near) / (near - far)` and `out[11] = -1`, which puts clip `z` in `-w..w`.
/// Vulkan's clip volume is `0..w`, so the near half of every frustum would be clipped away --
/// along with anything whose whole `z` is a sublayer nudge. The producer's viewport quad is
/// exactly that: `DrawableEntry::for_viewport` builds a camera-free matrix whose clip `z` is
/// `-depth_offset`, a small negative number, so the layer that covers the frame is the first thing
/// a Vulkan consumer drops. Measured: the captured Berlin frame draws 538,201 of 786,432 pixels as
/// the clear without this, and none of them with it.
///
/// Remapped here rather than in the matrices, which are the oracle's bit for bit, and in the
/// placements rather than per body, because a placement is where a clip position is formed. The
/// map is linear, so a body that adds a displacement to a placement -- which the lines do -- gets
/// the same answer whether the two are remapped separately or their sum is.
///
/// mbgl's own Vulkan backend does not do this. Its `applySurfaceTransform` flips `y` and leaves `z`
/// alone, and it binds no `VK_EXT_depth_clip_control`, so it clips the near half too -- which it
/// never notices, because its background is a tile quad through the projection and lands at
/// `z/w = 0.99998` where tessella's camera-free quad lands just below zero.
pub const PRELUDE: &str = r"
var<private> ubo_index: u32;

fn transform(columns: array<vec4<f32>, 4>, position: vec3<f32>) -> vec4<f32> {
    return columns[0] * position.x
        + columns[1] * position.y
        + columns[2] * position.z
        + columns[3];
}

fn vulkan_depth(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, (clip.z + clip.w) * 0.5, clip.w);
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

// GLSL's `mod`, which is not WGSL's `%`. The difference is the sign: `%` takes the dividend's,
// `mod` always takes the divisor's. Every use here is on a non-negative quantity, so the two agree
// -- written out anyway, because a pattern offset that went negative would tile backwards from
// here and nothing upstream would have changed.
fn wrap(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b * floor(a / b);
}

// Where a pattern's own texture coordinate sits at a tile position, in units of the pattern.
//
// `common.hpp`'s `get_pattern_pos`. A pattern is anchored to the world rather than to the tile, so
// that a repeating fill does not visibly restart at every tile edge -- which is what the pixel
// coordinate is for. It arrives split in two because the number is larger than an `f32` holds at
// the precision a pattern needs: the upper half is in units of 65,536 and the two nested wraps
// bring it down a byte at a time without ever forming the whole value.
fn pattern_pos(
    upper: vec2<f32>,
    lower: vec2<f32>,
    pattern_size: vec2<f32>,
    tile_units_to_pixels: f32,
    position: vec2<f32>
) -> vec2<f32> {
    let offset = wrap(wrap(wrap(upper, pattern_size) * 256.0, pattern_size) * 256.0 + lower, pattern_size);
    return (tile_units_to_pixels * position + offset) / pattern_size;
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
    out.color = paint_color(in);
    out.opacity = paint_opacity(in);
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
    @location(2) screen: vec2<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_outline_drawable_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let clip = place(vec3<f32>(vec2<f32>(in.fill_pos), 0.0), drawable.matrix);
    out.clip = clip;

    // Where this vertex landed on screen, in pixels, which is what the fragment measures its own
    // distance from. The perspective divide has to happen here: a fragment gets the interpolated
    // result, and interpolating before dividing is not the same number.
    //
    // `y` is negated, which mbgl's does not appear to do. It does: `applySurfaceTransform()` ends
    // with `gl_Position.y *= -1.0` and mbgl calls it *before* computing this position. Here the
    // same flip arrives from naga, which negates the output position under
    // `WriterFlags::ADJUST_COORDINATE_SPACE` -- after this body has run. So the flip is applied by
    // hand to the copy this stage reads back, and the two agree again.
    //
    // Getting it wrong is a mirror about the target's middle, which puts every fragment more than
    // a pixel from its own vertex and fades the whole outline to nothing. See `benches/first_pixel`
    // for the measurement that found it.
    out.screen = (vec2<f32>(clip.x, -clip.y) / clip.w + 1.0) / 2.0 * global.world_size;

    out.outline_color = paint_outline_color(in);
    out.opacity = paint_opacity(in);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // An outline is drawn as lines, which rasterize hard -- two pixels wide, which is mbgl's
    // `lineWidth` and what `pipelines::LINE_DYNAMIC` sets per batch where the device allows it. The
    // fade is this fragment's distance from the vertex's own screen position: the outline's
    // geometry runs along the polygon's edge, so that distance is how far across the line this
    // pixel sits. It fades over one pixel whatever the line's width, so a two-wide line's second
    // pixel is drawn at 0.15625 and its rim at nothing.
    //
    // `in.clip` is `@builtin(position)`, and in a fragment stage that builtin is the framebuffer
    // coordinate rather than the clip position the vertex stage wrote -- WGSL's `gl_FragCoord`.
    let distance = length(in.screen - in.clip.xy);
    let alpha = 1.0 - smoothstep(0.0, 1.0, distance);
    return in.outline_color * in.opacity * alpha;
}
";

/// The line-pattern family's body: a line, with a sprite running along it.
///
/// [`LINE_BODY`]'s vertex stage without the color and with two additions: the sprite's corners,
/// and the distance along the line. That distance is what the pattern repeats against, and it is
/// the `a_data` component the plain family sanctions as unread -- the top six bits of the third
/// byte with the whole fourth, which is why the fourth is multiplied by 64.
///
/// # The pattern's size is computed in the fragment stage
///
/// [`FILL_PATTERN_BODY`] computes it in the vertex stage; this one does not, because a line's
/// pattern is not square to the tile. Its width runs across the line and its length along it, so
/// the two axes are scaled differently -- the length by the tile's zoom ratio and the width not at
/// all -- and the fragment is where the normal that distinguishes them is available.
///
/// # The width is clamped to the sprite plus its padding
///
/// A line wider than its sprite would read past the sprite's own rectangle and into whatever sits
/// beside it on the sheet. The clamp is to half the pattern's height plus two, which is the
/// padding the atlas leaves between sprites.
pub const LINE_PATTERN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec2<f32>,
    @location(1) outset: f32,
    @location(2) inset: f32,
    @location(3) gamma_scale: f32,
    @location(4) blur: f32,
    @location(5) opacity: f32,
    @location(6) along: f32,
    @location(7) pattern_from: vec4<f32>,
    @location(8) pattern_to: vec4<f32>,
    @location(9) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = line_pattern_drawable_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let extrude_scale = 63.0;
    let antialiasing = 1.0 / max(global.pixel_ratio, 1e-6) / 2.0;

    let packed = vec2<f32>(in.line_pos_normal);
    let center = floor(packed * 0.5);
    var normal = packed - 2.0 * center;
    normal.y = normal.y * 2.0 - 1.0;

    let data = vec4<f32>(in.line_data);
    let extrude = data.xy - 128.0;
    let direction = (data.z % 4.0) - 1.0;
    // The distance along the line, in tile units doubled. The fourth byte is the high half of a
    // fourteen-bit number whose low six bits are the top of the third.
    out.along = (floor(data.z / 4.0) + data.w * 64.0) * 2.0;

    let gapwidth = paint_gapwidth(in) * 0.5;
    let halfwidth = paint_width(in) * 0.5;
    let line_offset = -1.0 * paint_offset(in);

    let inset = gapwidth + select(0.0, antialiasing, gapwidth > 0.0);
    let outset = gapwidth
        + halfwidth * select(1.0, 2.0, gapwidth > 0.0)
        + select(antialiasing, 0.0, halfwidth == 0.0);

    let dist = outset * extrude / extrude_scale;
    let u = 0.5 * direction;
    let t = 1.0 - abs(u);
    let turn = mat2x2<f32>(t, -u, u, t);
    let offset2 = line_offset * extrude / extrude_scale * normal.y * turn;

    let ratio = max(drawable.ratio, 1e-6);
    let at = center + offset2 / ratio;
    let projected_extrude = displace(at, dist / ratio, drawable.matrix);
    let clip = place(vec3<f32>(at, 0.0), drawable.matrix) + projected_extrude;
    out.clip = clip;

    let unprojected = length(dist);
    let projected = length(projected_extrude.xy / clip.w * global.units_to_pixels);
    out.gamma_scale = unprojected / max(projected, 1e-6);

    out.normal = normal;
    out.outset = outset;
    out.inset = inset;
    out.blur = paint_blur(in);
    out.opacity = paint_opacity(in);
    // Straight through: a sprite name is not a number. See `FILL_PATTERN_BODY`.
    out.pattern_from = paint_pattern_from(in);
    out.pattern_to = paint_pattern_to(in);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = line_pattern_tile_props_ubo[ubo_index];
    let global = global_paint_params_ubo[0];

    // Four numbers in one slot: the device pixel ratio, the tile's zoom ratio, and the two
    // sprites' scales.
    let pixel_ratio = max(tile.scale.x, 1e-6);
    let tile_zoom_ratio = max(tile.scale.y, 1e-6);

    let display_a = (in.pattern_from.zw - in.pattern_from.xy) / pixel_ratio;
    let display_b = (in.pattern_to.zw - in.pattern_to.xy) / pixel_ratio;
    // The length is scaled by the tile's zoom ratio and the width is not: one runs along the line
    // in tile units and the other across it in display pixels.
    let size_a = vec2<f32>(display_a.x * tile.scale.z / tile_zoom_ratio, display_a.y);
    let size_b = vec2<f32>(display_b.x * tile.scale.w / tile_zoom_ratio, display_b.y);

    // The line's own coverage, exactly as the plain family computes it.
    let distance = length(in.normal) * in.outset;
    let blur2 = (in.blur + 1.0 / max(global.pixel_ratio, 1e-6)) * in.gamma_scale;
    let alpha = clamp(
        min(distance - (in.inset - blur2), in.outset - distance) / blur2,
        0.0,
        1.0
    );

    // Along the line, wrapped into one repeat of the sprite.
    let x_a = wrap(vec2<f32>(in.along / size_a.x, 0.0), vec2<f32>(1.0, 1.0)).x;
    let x_b = wrap(vec2<f32>(in.along / size_b.x, 0.0), vec2<f32>(1.0, 1.0)).x;

    // Across it. The normal runs -1 to 1 edge to edge, so the half centers it, and the clamp stops
    // a line wider than its sprite reading into the sprite beside it on the sheet.
    let y_a = 0.5
        + in.normal.y * clamp(in.outset, 0.0, (size_a.y + 2.0) / 2.0) / max(size_a.y, 1e-6);
    let y_b = 0.5
        + in.normal.y * clamp(in.outset, 0.0, (size_b.y + 2.0) / 2.0) / max(size_b.y, 1e-6);

    let pos_a = mix(
        in.pattern_from.xy / tile.texsize,
        in.pattern_from.zw / tile.texsize,
        vec2<f32>(x_a, y_a)
    );
    let pos_b = mix(
        in.pattern_to.xy / tile.texsize,
        in.pattern_to.zw / tile.texsize,
        vec2<f32>(x_b, y_b)
    );

    let color = mix(
        textureSample(line_image, line_image_sampler, pos_a),
        textureSample(line_image, line_image_sampler, pos_b),
        tile.fade
    );
    return color * (alpha * in.opacity);
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
    @location(3) antialias_blur: f32,
    @location(4) radius: f32,
    @location(5) blur: f32,
    @location(6) opacity: f32,
    @location(7) stroke_width: f32,
    @location(8) stroke_opacity: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = circle_drawable_ubo[ubo_index];
    let props = circle_evaluated_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let radius = paint_radius(in);
    let stroke_width = paint_stroke_width(in);
    let reach = radius + stroke_width;

    // The corner sign rides in the low bit of each coordinate, the same trick the line vertex
    // uses: the point is doubled and zero or one added. So the center is the halved position and
    // the extrusion is what is left, mapped from {0,1} back to {-1,+1}.
    let position = vec2<f32>(in.circle_pos);
    let extrude = (position % vec2<f32>(2.0, 2.0)) * 2.0 - 1.0;
    let scaled_extrude = extrude * drawable.extrude_scale;
    let center = floor(position * 0.5);

    if props.pitch_with_map != 0 {
        // Lying on the ground, so the extrusion is in tile units and goes on before the matrix.
        var corner = center;
        if props.scale_with_map != 0 {
            corner += scaled_extrude * reach;
        } else {
            // Pitching with the map scales the circle with it, so a circle held at a constant
            // pixel size is rescaled by the pitch effect at its own center -- which needs the
            // perspective divide the matrix is about to do, measured first and multiplied in.
            let projected_center = place(vec3<f32>(center, 0.0), drawable.matrix);
            corner += scaled_extrude
                * reach
                * (projected_center.w / max(global.camera_to_center_distance, 1e-6));
        }
        out.clip = place(vec3<f32>(corner, 0.0), drawable.matrix);
    } else {
        // Standing up on the screen: the center is placed and the extrusion added in clip space
        // afterwards, scaled so it survives the perspective divide as a pixel size.
        let placed = place(vec3<f32>(center, 0.0), drawable.matrix);
        let factor = select(
            placed.w,
            global.camera_to_center_distance,
            props.scale_with_map != 0
        );
        out.clip = vec4<f32>(
            placed.xy + scaled_extrude * reach * factor,
            placed.z,
            placed.w
        );
    }

    // The blur that hides the quad's own edge: one device pixel spread over the circle's reach, so
    // a small circle fades over more of itself than a large one. Guarded, because a circle with no
    // radius and no stroke is a style that evaluated to nothing rather than an impossible one.
    out.antialias_blur = 1.0 / max(global.pixel_ratio, 1e-6) / max(reach, 1e-6);

    out.extrude = extrude;
    out.color = paint_color(in);
    out.stroke_color = paint_stroke_color(in);
    out.radius = radius;
    out.blur = paint_blur(in);
    out.opacity = paint_opacity(in);
    out.stroke_width = stroke_width;
    out.stroke_opacity = paint_stroke_opacity(in);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // Everything here is in units of the quad, where 1.0 is the circle's outer edge, which is what
    // makes the shape resolution-independent.
    let extrude_length = length(in.extrude);

    // Negative, and used as `smoothstep`'s *second* edge: the ramp descends. The style's blur and
    // the one-pixel floor are the same quantity, so the wider of the two wins rather than both
    // applying.
    let antialiased_blur = -max(in.blur, in.antialias_blur);

    let opacity_t = smoothstep(0.0, antialiased_blur, extrude_length - 1.0);
    // Where the fill ends and the stroke begins, as a fraction of the reach. Below a hundredth of
    // a pixel there is no stroke to find an edge for, and the ratio would be a hard seam.
    let color_t = select(
        smoothstep(
            antialiased_blur,
            0.0,
            extrude_length - in.radius / max(in.radius + in.stroke_width, 1e-6)
        ),
        0.0,
        in.stroke_width < 0.01
    );

    return opacity_t
        * mix(in.color * in.opacity, in.stroke_color * in.stroke_opacity, color_t);
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

    // The curtain, over the crack between two tiles of raised ground. Zero on every surface
    // but the terrain, and zero there for an unraised one -- see `curtain`.
    let drop = curtain(f32(in.tessella_skirt.x));
    out.clip = place(vec3<f32>(vec2<f32>(in.raster_pos), drop), drawable.matrix);
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
    @location(1) @interpolate(flat) ubo: u32,
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
    // From the texture attribute, where mbgl's Vulkan shader derives this from `in_position` and
    // declares no input at location 1 at all. A deliberate departure, not an unfinished port.
    //
    // The two agree against this producer -- `RasterBucket::add_quad_on` fills position and
    // texture from one `x` and `y` on every path -- so the choice is free of consequence for
    // what is drawn, and the attribute is the honest source because it is what the producer
    // sends for this coordinate.
    //
    // Reading the position instead was tried and rejected. It would not clear the i.MX8M Plus
    // shader-compiler crash it was proposed for: color relief already builds there, and a probe
    // with the attribute neither declared nor read drew the same wrong pixel, so the attribute
    // was never the trigger. And it would cost the oracle its case -- `COVERING` is in clip
    // units, so a position-derived coordinate over it collapses to a 1.2e-4 window on a texel
    // boundary, which two drivers landed on opposite sides of. See the closed issue #44.
    out.uv = (vec2<f32>(in.color_relief_texture_pos) / 8192.0) * scale + epsilon;

    // A tile covering a pole carries a sentinel `y` at the end of the signed range rather than a
    // coordinate, so the row it samples is pinned instead of running off the DEM.
    if f32(in.color_relief_pos.y) < -32767.5 {
        out.uv.y = 0.0;
    }
    if f32(in.color_relief_pos.y) > 32766.5 {
        out.uv.y = 1.0;
    }

    // The curtain, as the raster has one.
    let drop = curtain(f32(in.tessella_skirt.x));
    out.clip = place(vec3<f32>(vec2<f32>(in.color_relief_pos), drop), drawable.matrix);
    out.ubo = ubo_index;
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
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
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

    let base = max(paint_base(in), 0.0);
    let height = max(paint_height(in), 0.0);
    // Every vertex of a roof is at the building's top, so this always takes the height. Written
    // as mbgl's selection rather than as `height` alone: the base is the other arm of the term it
    // computes, and a shader that dropped it would agree with the oracle by coincidence of what
    // this drawable happens to carry rather than by computing the same thing.
    let on_roof = 1.0;
    let z = select(base, height, on_roof > 0.0);

    // How bright the surface already is. A pale building takes a narrower range of shading than a
    // dark one, which is what keeps a light roof from blowing out.
    var color = paint_color(in);
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
    out.opacity = paint_opacity(in);
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
/// `GLYPH_ATLAS_FORMAT` is `TexturePixelType::Alpha`, one channel, and Vulkan has no alpha-only
/// format -- so the image is `R8_UNORM` either way and the channel the data arrives in is the
/// image view's to decide. mbgl's own Vulkan backend uploads `R8_UNORM` and then sets a component
/// mapping of `(Zero, Zero, Zero, R)` on the view (`src/mbgl/vulkan/texture2d.cpp`), which is why
/// its shader reads `.a`.
///
/// This reads `.r`, so the producer must bind the atlas through a view with the **identity**
/// component mapping -- the default, and what anyone creating an `R8_UNORM` view gets without
/// asking. The two have to agree and nothing checks that they do: through an identity view `.a`
/// samples 1.0 at every pixel and every glyph is a solid block, and through mbgl's swizzled view
/// `.r` samples zero and no glyph appears at all.
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
    @location(8) @interpolate(flat) ubo: u32,
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
    let fill = paint_fill_color(in);
    let halo = paint_halo_color(in);
    out.paint = select(fill, halo, tile.is_halo != 0);

    out.tex = tex / drawable.texsize;
    out.font_scale = font_scale;
    // The perspective divide the threshold has to be measured in, which is the clip `w`.
    out.gamma = clip.w;
    out.opacity = paint_opacity(in);
    out.halo_width = paint_halo_width(in);
    out.halo_blur = paint_halo_blur(in);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
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

    // `.a`, as mbgl reads it. A one-channel texture is `R8_UNORM` and the value lands in `r`, so
    // the view swizzles it to alpha -- `device::component_swizzle`, which is mbgl's own mapping.
    let distance = textureSample(symbol_image, symbol_image_sampler, in.tex).a;
    var alpha = smoothstep(inner - gamma, inner + gamma, distance);
    if is_halo {
        // Subtracted rather than drawn under, so a translucent fill keeps a translucent halo.
        let halo_edge = (6.0 - in.halo_width / in.font_scale) / sdf_px;
        alpha = min(smoothstep(halo_edge - gamma, halo_edge + gamma, distance), 1.0 - alpha);
    }

    return in.paint * (alpha * in.opacity * in.fade);
}
";

/// The symbol text-and-icon family's body: one draw whose vertices are glyphs or sprites.
///
/// A label can carry an inline image -- a shield, a transit badge -- and placing it as a separate
/// drawable would let the two fall out of step. So the producer puts both in one bucket and marks
/// each vertex, and the fragment stage branches: a sprite is sampled from a second atlas and a
/// glyph goes through [`SYMBOL_SDF_BODY`]'s arithmetic unchanged.
///
/// # The mark is the low bit of the first size byte
///
/// `in_data.z` carries the size doubled with the flag added, the same packing the fade opacity
/// uses. So `floor(z * 0.5)` is the size and what is left is whether this vertex is a glyph.
/// Nothing else distinguishes them, and getting it backwards samples each atlas with the other's
/// coordinates -- which draws, in pieces of the wrong sheet.
///
/// # The font scale has no text branch here
///
/// `symbol_icon` and `symbol_sdf` both compute `is_text_prop ? size / 24.0 : size`; this family
/// divides unconditionally. Every vertex it draws belongs to a text label -- the icon is *in* the
/// label -- so the branch has one answer and mbgl writes it that way.
pub const SYMBOL_TEXT_AND_ICON_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex: vec2<f32>,
    @location(1) @interpolate(flat) is_icon: u32,
    @location(2) fade: f32,
    @location(3) font_scale: f32,
    @location(4) gamma_scale: f32,
    @location(5) fill_color: vec4<f32>,
    @location(6) halo_color: vec4<f32>,
    @location(7) opacity: f32,
    @location(8) halo_width: f32,
    @location(9) halo_blur: f32,
    @location(10) @interpolate(flat) ubo: u32,
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
    let placed = in.symbol_projected_pos;
    let segment_angle = -placed.z;

    // The size doubled with the glyph-or-sprite flag in the low bit.
    let smallest = floor(sized.x * 0.5);
    let is_sdf = sized.x - 2.0 * smallest;

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
    // Unconditional, unlike the other two symbol families: see this body's own note.
    let font_scale = size / 24.0;

    var rotation = 0.0;
    if drawable.rotate_symbol != 0 {
        let along = transform(drawable.matrix, vec3<f32>(anchor + vec2<f32>(1.0, 0.0), 0.0));
        let here = anchor_point.xy / anchor_point.w;
        let there = along.xy / along.w;
        rotation = atan2((there.y - here.y) / global.aspect_ratio, there.x - here.x);
    }
    let turn = segment_angle + rotation;
    let spun = mat2x2<f32>(cos(turn), -sin(turn), sin(turn), cos(turn));

    // No pixel offset: this family declares none, which is the other attribute it does without.
    let in_plane = transform(drawable.label_plane_matrix, vec3<f32>(placed.xy, 0.0));
    let on_plane = in_plane.xy / in_plane.w + spun * (corner / 32.0 * font_scale);
    let clip = place(vec3<f32>(on_plane, 0.0), drawable.coord_matrix);
    out.clip = clip;

    let is_icon = is_sdf == 0.0;
    out.is_icon = u32(is_icon);
    // The sheet this vertex came from decides which size divides its coordinates, so one draw
    // reads two atlases of different dimensions.
    out.tex = tex / select(drawable.texsize, drawable.texsize_icon, is_icon);

    let whole = floor(in.symbol_fade_opacity / 2.0);
    let rising = in.symbol_fade_opacity - whole * 2.0;
    var change = -global.symbol_fade_change;
    if rising > 0.5 {
        change = global.symbol_fade_change;
    }
    out.fade = clamp(whole / 127.0 + change, 0.0, 1.0);

    out.font_scale = font_scale;
    out.gamma_scale = clip.w;
    out.fill_color = paint_fill_color(in);
    out.halo_color = paint_halo_color(in);
    out.opacity = paint_opacity(in);
    out.halo_width = paint_halo_width(in);
    out.halo_blur = paint_halo_blur(in);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = symbol_tile_props_ubo[ubo_index];
    let global = global_paint_params_ubo[0];

    if in.is_icon != 0u {
        // A sprite, premultiplied, with no color of its own and no halo.
        let sprite = textureSample(symbol_image_icon, symbol_image_icon_sampler, in.tex);
        return sprite * (in.opacity * in.fade);
    }

    let sdf_px = 8.0;
    let edge_gamma = 0.105 / max(global.pixel_ratio, 1e-6);
    let font_gamma = in.font_scale * tile.gamma_scale;
    let fill_gamma = edge_gamma / font_gamma;
    let halo_gamma = (in.halo_blur * 1.19 / sdf_px + edge_gamma) / font_gamma;

    let is_halo = tile.is_halo != 0;
    let gamma_scaled = select(fill_gamma, halo_gamma, is_halo) * in.gamma_scale;
    let color = select(in.fill_color, in.halo_color, is_halo);

    let fill_edge = (256.0 - 64.0) / 256.0;
    let inner = select(fill_edge, fill_edge + halo_gamma * in.gamma_scale, is_halo);

    // `.a`, as mbgl reads it: see `SYMBOL_SDF_BODY` for the swizzle that puts it there.
    let distance = textureSample(symbol_image, symbol_image_sampler, in.tex).a;
    var alpha = smoothstep(inner - gamma_scaled, inner + gamma_scaled, distance);
    if is_halo {
        let halo_edge = (6.0 - in.halo_width / in.font_scale) / sdf_px;
        alpha = min(
            smoothstep(halo_edge - gamma_scaled, halo_edge + gamma_scaled, distance),
            1.0 - alpha
        );
    }

    return color * (alpha * in.opacity * in.fade);
}
";

/// The hillshade-prepare family's body: a DEM in, a slope field out.
///
/// The first family that draws into a texture rather than onto the map. A Sobel kernel over nine
/// DEM samples gives the slope at each pixel, and that is what [`HILLSHADE_BODY`] shades -- once
/// per tile instead of once per frame, which is the whole reason for the two passes.
///
/// # No surface
///
/// mbgl's other shaders end their vertex stage with `applySurfaceTransform()`; this one writes
/// `gl_Position.y *= -1.0` itself and stops. There is no map here to bend onto: the target is a
/// texture the size of the DEM and the matrix takes the tile's own square to it. So this family
/// has the plane alone, for the same reason the background has no raise -- not a limitation, an
/// absence of anything to raise it above.
pub const HILLSHADE_PREPARE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = hillshade_prepare_drawable_ubo[ubo_index];
    let tile = hillshade_prepare_tile_props_ubo[ubo_index];
    var out: Out;

    out.clip = place(vec3<f32>(vec2<f32>(in.hillshade_pos), 0.0), drawable.matrix);

    // Into the tile's own square inside the padded DEM: one border texel a side, so the interior
    // spans `1/stride` to `(dim - 1)/stride`.
    let epsilon = vec2<f32>(1.0, 1.0) / tile.dimension;
    let scale = (tile.dimension.x - 2.0) / tile.dimension.x;
    out.uv = vec2<f32>(in.hillshade_texture_pos) / 8192.0 * scale + epsilon;
    out.ubo = ubo_index;
    return out;
}

// Meters at a texture coordinate, the way the DEM encodes it: the texel times 255 with its alpha
// replaced by -1, dotted with the unpack vector.
fn elevation(uv: vec2<f32>, unpack: vec4<f32>) -> f32 {
    let texel = textureSample(hillshade_image, hillshade_image_sampler, uv) * 255.0;
    return dot(vec4<f32>(texel.rgb, -1.0), unpack);
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = hillshade_prepare_tile_props_ubo[ubo_index];
    let epsilon = vec2<f32>(1.0, 1.0) / tile.dimension;
    let tile_size = tile.dimension.x - 2.0;

    // The eight neighbors of this pixel, named as the kernel is usually drawn. The center is not
    // read: a Sobel operator weights the two columns and the two rows either side of it and the
    // middle cancels.
    //
    //   a b c
    //   d . f
    //   g h i
    let a = elevation(in.uv + vec2<f32>(-epsilon.x, -epsilon.y), tile.unpack);
    let b = elevation(in.uv + vec2<f32>(0.0, -epsilon.y), tile.unpack);
    let c = elevation(in.uv + vec2<f32>(epsilon.x, -epsilon.y), tile.unpack);
    let d = elevation(in.uv + vec2<f32>(-epsilon.x, 0.0), tile.unpack);
    let f = elevation(in.uv + vec2<f32>(epsilon.x, 0.0), tile.unpack);
    let g = elevation(in.uv + vec2<f32>(-epsilon.x, epsilon.y), tile.unpack);
    let h = elevation(in.uv + vec2<f32>(0.0, epsilon.y), tile.unpack);
    let i = elevation(in.uv + vec2<f32>(epsilon.x, epsilon.y), tile.unpack);

    // Pixel-space slope to world-space slope. Meters per pixel is `pow(2, 28.2562 - zoom)`, and
    // the exaggeration term flattens the relief at low zoom where a pixel covers so much ground
    // that the true slope would be noise.
    var factor = 0.3;
    if tile.zoom < 2.0 {
        factor = 0.4;
    } else if tile.zoom < 4.5 {
        factor = 0.35;
    }
    var exaggeration = 0.0;
    if tile.zoom < 15.0 {
        exaggeration = (tile.zoom - 15.0) * factor;
    }

    let deriv = vec2<f32>(
        (c + f + f + i) - (a + d + d + g),
        (g + h + h + i) - (a + b + b + c)
    ) * tile_size / pow(2.0, exaggeration + (28.2562 - tile.zoom));

    // Stored in red and green over the range [-4, 4], which is the largest slope assumed to
    // occur: eight units of slope across the channel's zero to one.
    return clamp(
        vec4<f32>(deriv.x / 8.0 + 0.5, deriv.y / 8.0 + 0.5, 1.0, 1.0),
        vec4<f32>(0.0),
        vec4<f32>(1.0)
    );
}
";

/// The hillshade family's body: five methods over one slope field.
///
/// Reads what [`HILLSHADE_PREPARE_BODY`] wrote and turns a slope into a color. `method` selects
/// between five, which the style names and which do genuinely different things rather than being
/// variations on one: `standard` is the legacy algorithm, `basic` and `multidirectional` are Lambertian
/// with one light and up to four, and `combined` and `igor` split the result into a shadow and a
/// highlight term that are summed.
///
/// # The latitude scale is not cosmetic
///
/// A Mercator pixel covers less ground the further it is from the equator, so the same elevation
/// change across one pixel is a steeper real slope. Dividing by `cos(latitude)` undoes it; without
/// it, relief flattens visibly toward the poles.
pub const HILLSHADE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = hillshade_drawable_ubo[ubo_index];
    var out: Out;

    // The curtain, as the raster has one: this is the same quad over the same raised ground.
    let drop = curtain(f32(in.tessella_skirt.x));
    out.clip = place(vec3<f32>(vec2<f32>(in.hillshade_pos), drop), drawable.matrix);
    // Flipped, because the slope field was rendered into a texture whose rows run the other way.
    let uv = vec2<f32>(in.hillshade_texture_pos) / 8192.0;
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    out.ubo = ubo_index;
    return out;
}

// The downhill direction. Where the x slope is zero the arctangent is undefined, so the aspect is
// straight up or straight down by the sign of the y slope.
fn aspect_of(deriv: vec2<f32>) -> f32 {
    let pi = 3.141592653589793;
    if deriv.x != 0.0 {
        return atan2(deriv.y, -deriv.x);
    }
    return pi / 2.0 * select(-1.0, 1.0, deriv.y > 0.0);
}

// How much of the light reaches a surface of this slope, for a light at one altitude and azimuth.
fn lambert(deriv: vec2<f32>, altitude: f32, cos_az: f32, sin_az: f32) -> f32 {
    let cos_alt = cos(altitude);
    let sin_alt = sin(altitude);
    let cang = (sin_alt - (deriv.y * cos_az * cos_alt - deriv.x * sin_az * cos_alt))
        / sqrt(1.0 + dot(deriv, deriv));
    return clamp(cang, 0.0, 1.0);
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = hillshade_tile_props_ubo[ubo_index];
    let props = hillshade_evaluated_props_ubo[0];
    let pi = 3.141592653589793;

    let pixel = textureSample(hillshade_image, hillshade_image_sampler, in.uv);

    // The latitude of this row, and the Mercator distortion there.
    let latitude = (tile.latrange.x - tile.latrange.y) * in.uv.y + tile.latrange.y;
    let scale_factor = cos(radians(latitude));
    // Back from the stored [0, 1] to the slope range [-4, 4].
    let deriv = ((pixel.rg * 8.0) - 4.0) / scale_factor;

    let azimuth = props.azimuths.x + pi;
    // Every method but the standard one works on the slope already exaggerated.
    let lit = deriv * tile.exaggeration * 2.0;

    if tile.method == 4 {
        // BASIC: one Lambertian light, the shade split either side of a half.
        let shade = lambert(lit, props.altitudes.x, cos(azimuth), sin(azimuth));
        if shade > 0.5 {
            return props.highlights[0] * (2.0 * shade - 1.0);
        }
        return props.shadows[0] * (1.0 - 2.0 * shade);
    }

    if tile.method == 1 {
        // COMBINED: the angle to the light and the steepness as two terms that are summed, so a
        // slope facing away is dark *and* a flat surface is neutral whichever way it faces.
        let cos_alt = cos(props.altitudes.x);
        let sin_alt = sin(props.altitudes.x);
        let cos_az = cos(azimuth);
        let sin_az = sin(azimuth);
        let cang = clamp(
            acos(
                (sin_alt - (lit.y * cos_az * cos_alt - lit.x * sin_az * cos_alt))
                    / sqrt(1.0 + dot(lit, lit))
            ),
            0.0,
            pi / 2.0
        );
        let steepness = atan(length(lit)) * 4.0 / pi / pi;
        return props.shadows[0] * (cang * steepness)
            + props.highlights[0] * ((pi / 2.0 - cang) * steepness);
    }

    if tile.method == 2 {
        // IGOR: no light altitude at all. The shadow is how steep the slope is times how far it
        // faces away from the light, and the highlight is the remainder.
        let aspect = aspect_of(lit);
        let slope_strength = atan(length(lit)) * 2.0 / pi;
        let aspect_strength = 1.0 - abs(((aspect + azimuth) / pi + 0.5) % 2.0 - 1.0);
        return props.shadows[0] * (slope_strength * aspect_strength)
            + props.highlights[0] * (slope_strength * (1.0 - aspect_strength));
    }

    if tile.method == 3 {
        // MULTIDIRECTIONAL: up to four lights, each contributing its share. The azimuths are
        // negated here where the single-light methods add pi, which is the same half turn.
        var altitudes = array<f32, 4>(
            props.altitudes.x,
            props.altitudes.y,
            props.altitudes.z,
            props.altitudes.w
        );
        var azimuths = array<f32, 4>(
            props.azimuths.x,
            props.azimuths.y,
            props.azimuths.z,
            props.azimuths.w
        );
        let lights = min(tile.num_lights, 4);
        var total = vec4<f32>(0.0);
        for (var light = 0; light < lights; light++) {
            let shade = lambert(
                lit,
                altitudes[light],
                -cos(azimuths[light]),
                -sin(azimuths[light])
            );
            if shade > 0.5 {
                total += props.highlights[light] * (2.0 * shade - 1.0) / f32(lights);
            } else {
                total += props.shadows[light] * (1.0 - 2.0 * shade) / f32(lights);
            }
        }
        return total;
    }

    // STANDARD, the legacy algorithm, and the default for a method this build does not know. The
    // slope is scaled exponentially by the intensity rather than linearly, so turning the
    // exaggeration up steepens the gentle ground more than the cliffs.
    let slope = atan(0.625 * length(deriv));
    let aspect = aspect_of(deriv);
    let intensity = tile.exaggeration;
    let base = 1.875 - intensity * 1.75;
    let most = 0.5 * pi;
    // At an intensity of a half the base is one and the curve is flat, so the division has
    // nothing to divide by -- and the unscaled slope is the answer that limit approaches.
    var scaled = slope;
    if abs(intensity - 0.5) > 1e-6 {
        scaled = ((pow(base, slope) - 1.0) / (pow(base, most) - 1.0)) * most;
    }

    let reach = clamp(intensity * 2.0, 0.0, 1.0);
    let accent = (1.0 - cos(scaled)) * props.accent * reach;
    let shade = abs(((aspect + azimuth) / pi + 0.5) % 2.0 - 1.0);
    let shaded = mix(props.shadows[0], props.highlights[0], shade) * sin(scaled) * reach;

    // The accent shows through whatever the shade leaves transparent.
    return accent * (1.0 - shaded.a) + shaded;
}
";

/// The background-pattern family's body: the background, tiled with a sprite.
///
/// Two patterns, not one. A style changing `background-pattern` cross-fades between the old sprite
/// and the new, so both are sampled every frame and `mix` holds where between them the transition
/// is. The two reduce to the same picture once it finishes, which is why the fade is a uniform and
/// not a branch.
///
/// Unlike [`FILL_PATTERN_BODY`] the sizes arrive already computed, because a background has no
/// feature to evaluate them against -- so there is no display size to divide by the pixel ratio
/// here, just `scale` times `pattern_size`.
pub const BACKGROUND_PATTERN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos_a: vec2<f32>,
    @location(1) pos_b: vec2<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = background_pattern_drawable_ubo[ubo_index];
    let props = background_pattern_props_ubo[0];
    var out: Out;

    let position = vec2<f32>(in.background_pos);
    out.pos_a = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.scale_a * props.pattern_size_a,
        drawable.tile_units_to_pixels,
        position
    );
    out.pos_b = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.scale_b * props.pattern_size_b,
        drawable.tile_units_to_pixels,
        position
    );

    out.clip = place(vec3<f32>(position, 0.0), drawable.matrix);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let props = background_pattern_props_ubo[0];
    let texsize = global_paint_params_ubo[0].pattern_atlas_texsize;

    // The fraction of one repeat, interpolated across the sprite's own corners in the atlas. The
    // wrap is what makes the pattern repeat, and it has to happen here rather than on the sampler:
    // the sprite is a window into a shared sheet, so a repeat addressing mode would walk into its
    // neighbors.
    let in_a = wrap(in.pos_a, vec2<f32>(1.0, 1.0));
    let pos_a = mix(props.pattern_tl_a / texsize, props.pattern_br_a / texsize, in_a);
    let color_a = textureSample(background_image, background_image_sampler, pos_a);

    let in_b = wrap(in.pos_b, vec2<f32>(1.0, 1.0));
    let pos_b = mix(props.pattern_tl_b / texsize, props.pattern_br_b / texsize, in_b);
    let color_b = textureSample(background_image, background_image_sampler, pos_b);

    return mix(color_a, color_b, props.mix) * props.opacity;
}
";

/// The fill-pattern family's body: a fill, tiled with a sprite.
///
/// The two sprites are a paint property like any other: a `fill-pattern` given as an expression
/// over a feature's own field arrives as an attribute, four shorts holding the sprite's top left
/// and bottom right in the atlas, and one given as a plain name arrives in the tile's block
/// instead. Both arms are read through `paint_pattern_from`, which is this family's entry in
/// [`crate::families::ALL`].
///
/// # The sprite names are not interpolated
///
/// `FILL_PATTERN_DRAWABLE_UBO` carries `pattern_from_t` and `pattern_to_t` and mbgl's shader reads
/// neither, which is why they are named in `UNUSED_FACTORS`. A pattern is a window into a sheet,
/// not a number; halfway between two sprites' corners is a rectangle containing neither. The
/// cross-fade is `fade`, applied to the two sampled colors, and that is the only blending a
/// pattern transition gets.
///
/// # The size is in display pixels, divided twice
///
/// The sprite's extent in the atlas is in atlas pixels, so dividing by the device pixel ratio
/// gives the size it should occupy on screen, and dividing by the tile ratio converts that to the
/// tile units `pattern_pos` works in. Either one dropped tiles the pattern at a plausible but
/// wrong size.
pub const FILL_PATTERN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos_a: vec2<f32>,
    @location(1) pos_b: vec2<f32>,
    @location(2) pattern_from: vec4<f32>,
    @location(3) pattern_to: vec4<f32>,
    @location(4) opacity: f32,
    @location(5) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_pattern_drawable_ubo[ubo_index];
    let props = fill_evaluated_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    // Straight through, not mixed: see this body's own note.
    let pattern_from = paint_pattern_from(in);
    let pattern_to = paint_pattern_to(in);
    out.pattern_from = pattern_from;
    out.pattern_to = pattern_to;

    // The sprite's extent in the atlas, as the size it should cover on screen.
    let display_a = (pattern_from.zw - pattern_from.xy) / max(global.pixel_ratio, 1e-6);
    let display_b = (pattern_to.zw - pattern_to.xy) / max(global.pixel_ratio, 1e-6);

    let position = vec2<f32>(in.fill_pos);
    out.pos_a = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.from_scale * display_a,
        drawable.tile_ratio,
        position
    );
    out.pos_b = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.to_scale * display_b,
        drawable.tile_ratio,
        position
    );

    out.opacity = paint_opacity(in);
    out.clip = place(vec3<f32>(position, 0.0), drawable.matrix);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = fill_pattern_tile_props_ubo[ubo_index];
    let props = fill_evaluated_props_ubo[0];

    // This tile's own atlas size, where the background reads the shared one: a fill's sprites are
    // bound per tile.
    let texsize = tile.texsize;

    let in_a = wrap(in.pos_a, vec2<f32>(1.0, 1.0));
    let pos_a = mix(in.pattern_from.xy / texsize, in.pattern_from.zw / texsize, in_a);
    let color_a = textureSample(fill_image, fill_image_sampler, pos_a);

    let in_b = wrap(in.pos_b, vec2<f32>(1.0, 1.0));
    let pos_b = mix(in.pattern_to.xy / texsize, in.pattern_to.zw / texsize, in_b);
    let color_b = textureSample(fill_image, fill_image_sampler, pos_b);

    return mix(color_a, color_b, props.fade) * in.opacity;
}
";

/// The fill-outline-pattern family's body: a patterned fill's outline.
///
/// The fourteenth shader, and the last of mbgl's fill family this crate had no entry for. A fill
/// with `fill-antialias` emits an outline beside it, and a *patterned* fill emits a patterned one:
/// the same geometry as [`FILL_PATTERN_BODY`], drawn as lines, tiled with the same sprite.
///
/// It is the two bodies either side of it with nothing new of its own. The sprite's size and its
/// place in the atlas are [`FILL_PATTERN_BODY`]'s arithmetic exactly -- the same two divisions, the
/// same `pattern_pos` anchor, the same cross-fade through `fade` -- and the one-pixel feather is
/// [`FILL_OUTLINE_BODY`]'s, the fragment's distance from its own vertex's screen position with the
/// `y` flip that `ADJUST_COORDINATE_SPACE` makes necessary.
///
/// # What the two of them multiply to
///
/// `color * alpha * opacity`, in that order in mbgl and in any order here: the feather is coverage
/// and the opacity is the layer's, so an outline half a pixel from the line's center in a
/// half-opaque layer is a quarter of the sprite's own color. Dropping either leaves a plausible
/// picture -- the whole outline at full strength, or a hard line with no fade at all -- which is
/// why `benches/first_pixel.rs` reads a pixel that only both together cover.
pub const FILL_OUTLINE_PATTERN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos_a: vec2<f32>,
    @location(1) pos_b: vec2<f32>,
    @location(2) pattern_from: vec4<f32>,
    @location(3) pattern_to: vec4<f32>,
    @location(4) opacity: f32,
    @location(5) screen: vec2<f32>,
    @location(6) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_outline_pattern_drawable_ubo[ubo_index];
    let props = fill_evaluated_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    // Straight through, not mixed: see `FILL_PATTERN_BODY`'s own note.
    let pattern_from = paint_pattern_from(in);
    let pattern_to = paint_pattern_to(in);
    out.pattern_from = pattern_from;
    out.pattern_to = pattern_to;

    let display_a = (pattern_from.zw - pattern_from.xy) / max(global.pixel_ratio, 1e-6);
    let display_b = (pattern_to.zw - pattern_to.xy) / max(global.pixel_ratio, 1e-6);

    let position = vec2<f32>(in.fill_pos);
    out.pos_a = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.from_scale * display_a,
        drawable.tile_ratio,
        position
    );
    out.pos_b = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.to_scale * display_b,
        drawable.tile_ratio,
        position
    );

    out.opacity = paint_opacity(in);
    let clip = place(vec3<f32>(position, 0.0), drawable.matrix);
    out.clip = clip;
    // The flipped copy this stage reads back, for `FILL_OUTLINE_BODY`'s reason.
    out.screen = (vec2<f32>(clip.x, -clip.y) / clip.w + 1.0) / 2.0 * global.world_size;
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = fill_outline_pattern_tile_props_ubo[ubo_index];
    let props = fill_evaluated_props_ubo[0];

    let texsize = tile.texsize;

    let in_a = wrap(in.pos_a, vec2<f32>(1.0, 1.0));
    let pos_a = mix(in.pattern_from.xy / texsize, in.pattern_from.zw / texsize, in_a);
    let color_a = textureSample(fill_image, fill_image_sampler, pos_a);

    let in_b = wrap(in.pos_b, vec2<f32>(1.0, 1.0));
    let pos_b = mix(in.pattern_to.xy / texsize, in.pattern_to.zw / texsize, in_b);
    let color_b = textureSample(fill_image, fill_image_sampler, pos_b);

    // The outline's own one-pixel feather, as `FILL_OUTLINE_BODY` measures it.
    let distance = length(in.screen - in.clip.xy);
    let alpha = 1.0 - smoothstep(0.0, 1.0, distance);

    return mix(color_a, color_b, props.fade) * alpha * in.opacity;
}
";

/// The heatmap family's body: one Gaussian per point, summed into a texture.
///
/// The first pass of two. Each point feature draws a quad carrying a Gaussian kernel, the quads
/// are blended additively into an offscreen texture, and [`HEATMAP_TEXTURE_BODY`] colors the sum
/// through a ramp. The density has to be accumulated before it is colored -- two points' kernels
/// add and the result is looked up once -- which is why this cannot be one pass.
///
/// # The quad is sized by where the kernel stops mattering
///
/// A Gaussian never reaches zero, so the quad is sized to where it falls below one sixteenth of a
/// color step: below that the blend cannot represent it and the remaining tail is wasted
/// rasterization. Solving `weight * intensity * GAUSS_COEF * exp(-0.5 * 3^2 * S^2) == ZERO` for
/// `S` is where the vertex stage's logarithm comes from, and `extrude` is handed to the fragment
/// in units of that radius so the kernel it evaluates is the one the quad was sized for.
///
/// # No surface
///
/// Like [`HILLSHADE_PREPARE_BODY`], this draws into a texture and flips `y` itself rather than
/// calling mbgl's surface transform, so it has the plane alone.
pub const HEATMAP_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) extrude: vec2<f32>,
    @location(1) weight: f32,
}

// The height of a unit Gaussian at its peak, `1 / sqrt(2 * pi)`.
const GAUSS_COEF: f32 = 0.3989422804014327;
// A sixteenth of one step of an eight-bit channel: the point below which the accumulated density
// cannot be told from nothing.
const ZERO: f32 = 1.0 / 255.0 / 16.0;

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = heatmap_drawable_ubo[ubo_index];
    let props = heatmap_evaluated_props_ubo[0];
    var out: Out;

    let weight = paint_weight(in);
    let radius = paint_radius(in);

    // The corner sign in the low bit of each coordinate, as the circle and the line pack it.
    let position = vec2<f32>(in.heatmap_pos);
    let corner = (position % vec2<f32>(2.0, 2.0)) * 2.0 - 1.0;
    let center = floor(position * 0.5);

    // How far out the kernel has to reach before it falls under `ZERO`. Both factors are floored
    // at `ZERO` rather than guarded against zero: a weight or an intensity of nothing would send
    // the logarithm to infinity, and a quad of the smallest size that still means something is
    // the answer that limit approaches.
    let s = sqrt(
        -2.0 * log(ZERO / (max(weight, ZERO) * max(props.intensity, ZERO) * GAUSS_COEF))
    ) / 3.0;

    // In units of the radius, which is what the fragment stage evaluates the kernel in.
    let extrude = s * corner;
    out.extrude = extrude;
    out.weight = weight;
    out.clip = place(
        vec3<f32>(center + extrude * radius * drawable.extrude_scale, 0.0),
        drawable.matrix
    );
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let props = heatmap_evaluated_props_ubo[0];

    // A Gaussian of three standard deviations across the quad.
    let d = -0.5 * 3.0 * 3.0 * dot(in.extrude, in.extrude);
    let density = in.weight * props.intensity * GAUSS_COEF * exp(d);

    // Only red carries the density. The other channels are one because the target is blended
    // additively and a zero there would still be summed.
    return vec4<f32>(density, 1.0, 1.0, 1.0);
}
";

/// The heatmap-texture family's body: the accumulated density, through a ramp.
///
/// The second pass. [`HEATMAP_BODY`] summed every point's kernel into a texture; this reads that
/// sum and looks the color up, so the ramp is applied to the total rather than to each point.
///
/// The quad is the viewport in pixels -- `position` is a unit square scaled by the world size --
/// which is why this family needs no tile geometry and reads the density by its own position.
///
/// # Nothing here is per drawable
///
/// One quad over the viewport, and `HeatmapTexturePropsUBO` is one entry for the layer rather than
/// one per tile -- mbgl declares it a plain `uniform` block and the producer writes it once. So
/// both stages read it at `[0]`, and the slot the vertex stage takes from `firstInstance` goes
/// unread. `tests/shaders.rs` checks that distinction from both sides.
pub const HEATMAP_TEXTURE_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let props = heatmap_texture_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let position = vec2<f32>(in.heatmap_pos);
    out.uv = position;
    out.clip = place(vec3<f32>(position * global.world_size, 0.0), props.matrix);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let props = heatmap_texture_props_ubo[0];

    // Red, which is the channel the first pass accumulated into.
    let density = textureSample(heatmap_image, heatmap_image_sampler, in.uv).r;
    // The ramp is one row, so the second coordinate is its own center.
    let color = textureSample(
        heatmap_color_ramp,
        heatmap_color_ramp_sampler,
        vec2<f32>(density, 0.5)
    );
    return color * props.opacity;
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

    // Four unnormalized bytes. The first two are the extrusion and the low two bits of the third
    // say which way a round end point's extrude points. The rest of the third with the fourth is
    // the distance along the line, which only the patterned variants read -- see
    // `LINE_PATTERN_BODY`.
    let data = vec4<f32>(in.line_data);
    let extrude = data.xy - 128.0;
    let direction = (data.z % 4.0) - 1.0;

    let gapwidth = paint_gapwidth(in) * 0.5;
    let halfwidth = paint_width(in) * 0.5;
    let line_offset = -1.0 * paint_offset(in);

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
    out.color = paint_color(in);
    out.blur = paint_blur(in);
    out.opacity = paint_opacity(in);
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

/// The line-gradient family's body: a line colored by a ramp rather than by a paint property.
///
/// [`LINE_BODY`] with the color taken out and the ramp put in its place. `line-gradient` is a
/// *color stop list along the line*, which mbgl bakes into one 256x1 image per layer -- so there is
/// no color attribute, no `color_t` in the drawable block, and the fragment samples instead.
///
/// # The progress is the whole line's, not this tile's
///
/// The same fourteen bits [`LINE_PATTERN_BODY`] reads as a distance in tile units, divided by
/// `MAX_LINE_DISTANCE` rather than scaled: what a ramp wants is where along the line this fragment
/// is as a fraction. The producer only puts a usable number there when the source sets
/// `lineMetrics` -- a line cut by a tile boundary has to keep the progress it had whole, or every
/// piece restarts the gradient and the seam is visible at every edge. That is why the style spec
/// allows `line-gradient` on a `GeoJSON` source alone, and why `capture-render` had to learn to
/// read
/// one before this family could be drawn here at all.
pub const LINE_GRADIENT_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec2<f32>,
    @location(1) outset: f32,
    @location(2) inset: f32,
    @location(3) gamma_scale: f32,
    @location(4) blur: f32,
    @location(5) opacity: f32,
    @location(6) progress: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = line_gradient_drawable_ubo[ubo_index];
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

    // Four unnormalized bytes. The first two are the extrusion and the low two bits of the third
    // say which way a round end point's extrude points. The rest of the third with the fourth is
    // the distance along the line, which this family reads as the ramp's coordinate.
    let data = vec4<f32>(in.line_data);
    let extrude = data.xy - 128.0;
    let direction = (data.z % 4.0) - 1.0;

    // How far along the *whole* line this vertex is, as a fraction of `MAX_LINE_DISTANCE`. The
    // same fourteen bits `LINE_PATTERN_BODY` reads as a distance, divided rather than scaled:
    // a ramp is 256 texels of the line's whole length, so what it wants is the fraction.
    //
    // Whole, not this tile's piece. A line cut by a tile boundary keeps the progress it had in
    // the source, which is what `lineMetrics` is for -- without it each piece would start the
    // gradient over, and the seam would be visible at every tile edge.
    out.progress = (floor(data.z / 4.0) + data.w * 64.0) * 2.0 / 32767.0;

    let gapwidth = paint_gapwidth(in) * 0.5;
    let halfwidth = paint_width(in) * 0.5;
    let line_offset = -1.0 * paint_offset(in);

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
    out.blur = paint_blur(in);
    out.opacity = paint_opacity(in);
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

    // The color is the ramp at this fragment's progress. One row, sampled at its middle: mbgl
    // bakes `line-gradient` into a 256x1 image per layer, so the `y` is the texel's center and
    // not an edge.
    let color = textureSample(line_image, line_image_sampler, vec2<f32>(in.progress, 0.5));
    return color * (alpha * in.opacity);
}
";

/// The line-SDF family's body: a dashed line, where the dash is a distance field.
///
/// [`LINE_BODY`] with three additions, and the third is the one worth reading twice.
///
/// **The distance along the line**, as [`LINE_PATTERN_BODY`] reads it, scaled by the pattern's own
/// `patternscale` -- which is the producer's conversion from tile units to the atlas's.
///
/// **Two rows of the atlas, crossfaded.** A dash pattern is rasterized per zoom, so a layer
/// crossing a zoom boundary is between two of them; `tex_y_a` and `tex_y_b` are the rows and
/// `mix` is how far across. Sampling one row alone makes a dash that jumps at every zoom step.
///
/// **A threshold divided by the line's width.** The atlas holds a signed distance field in its
/// alpha, and the dash's edge is where that field crosses a half. `sdfgamma` is how soft that
/// crossing is at this zoom -- and it is divided by `floorwidth`, the width the dash was
/// rasterized at, because a wide line stretches the field over more pixels. Without the division a
/// thick road's dashes have soft ends and a thin road's are sharp, which reads as a renderer that
/// cannot draw wide dashes.
///
/// `floorwidth` is this family's ninth attribute and the only paint property no other line family
/// has.
pub const LINE_SDF_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) outset: f32,
    @location(3) inset: f32,
    @location(4) gamma_scale: f32,
    @location(5) blur: f32,
    @location(6) opacity: f32,
    @location(7) tex_a: vec2<f32>,
    @location(8) tex_b: vec2<f32>,
    @location(9) floorwidth: f32,
    @location(10) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = line_sdf_drawable_ubo[ubo_index];
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

    // Four unnormalized bytes. The first two are the extrusion and the low two bits of the third
    // say which way a round end point's extrude points. The rest of the third with the fourth is
    // the distance along the line, which this family runs its dash pattern along.
    let data = vec4<f32>(in.line_data);
    let extrude = data.xy - 128.0;
    let direction = (data.z % 4.0) - 1.0;

    let gapwidth = paint_gapwidth(in) * 0.5;
    let halfwidth = paint_width(in) * 0.5;
    let line_offset = -1.0 * paint_offset(in);

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
    out.color = paint_color(in);
    out.blur = paint_blur(in);
    out.opacity = paint_opacity(in);

    // Where this vertex reads the dash atlas, in both rows the layer is crossfading between.
    //
    // `x` is the distance along the line scaled by the pattern's own, divided by the width the
    // dash was *rasterized* at: a dash is drawn into the atlas at one width and stretched to the
    // line's, so dividing here is what keeps a wide road's dashes the same shape as a narrow
    // one's rather than the same length.
    //
    // `y` is the row. The normal runs -1 to 1 across the line, so this reads the atlas across the
    // dash as well as along it, and `tex_y` picks which of the two patterns the row belongs to.
    let linesofar = (floor(data.z / 4.0) + data.w * 64.0) * 2.0;
    let floorwidth = paint_floorwidth(in);
    out.floorwidth = floorwidth;
    let safe_floor = max(floorwidth, 1e-6);
    out.tex_a = vec2<f32>(
        linesofar * drawable.patternscale_a.x / safe_floor,
        normal.y * drawable.patternscale_a.y + drawable.tex_y_a
    );
    out.tex_b = vec2<f32>(
        linesofar * drawable.patternscale_b.x / safe_floor,
        normal.y * drawable.patternscale_b.y + drawable.tex_y_b
    );
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = line_sdf_tile_props_ubo[ubo_index];

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

    // The dash itself, which is a distance field in the atlas's alpha rather than a picture: the
    // two rows are the pattern the layer is crossfading from and to, and `mix` is how far between
    // them this frame is.
    //
    // The threshold is the field's own half, widened by the gamma the producer computed for this
    // zoom and *narrowed* by the line's width: a wide line stretches the field across more pixels,
    // so the same gamma would soften its edges more. Dividing by the width is what keeps a dash's
    // edge as sharp on a thick road as on a thin one.
    // `.a`, as mbgl reads it. The dash atlas is one channel -- it holds a distance field, and
    // three of four channels would be copies of the one the shader reads -- and the view swizzles
    // that channel to alpha. Read as `r` through such a view the field is zero everywhere and the
    // line disappears; read as `a` through an *unswizzled* view it is one everywhere and every
    // dash is solid, which is what a capture of `dash_style.json` drew while this crate had no
    // swizzle. `device::component_swizzle` is the one place that decides it.
    let sdfdist_a = textureSample(line_image, line_image_sampler, in.tex_a).a;
    let sdfdist_b = textureSample(line_image, line_image_sampler, in.tex_b).a;
    let sdfdist = mix(sdfdist_a, sdfdist_b, tile.mix);
    let gamma = tile.sdfgamma / max(in.floorwidth, 1e-6);
    let dash = smoothstep(0.5 - gamma, 0.5 + gamma, sdfdist);

    return in.color * (alpha * dash * in.opacity);
}
";

/// `FillOutlineTriangulatedShader`: a fill's outline as extruded quads rather than lines.
///
/// A fill's antialiasing *is* its outline -- `fill-antialias` turns it on -- and the triangles
/// rasterize hard, so the only thing softening a polygon's boundary is the fade in this fragment.
/// mbgl draws the outline one pixel wide with the fade outside it, which needs fragments further
/// than half a pixel from the line's center; a line primitive generates none, so mbgl's
/// `MLN_TRIANGULATE_FILL_OUTLINES` builds the ring out of the line layer's extruded quads instead
/// and draws it through this shader. The producer picks that path, which is why this family exists
/// here at all -- a capture of a real style names it twice a frame.
///
/// # Transcribed from the GL source, deliberately
///
/// `include/mbgl/shaders/gl/fill_outline_triangulated.hpp`, not the Vulkan one. mbgl's Vulkan
/// version of this shader **cannot compile**: its `FillOutlineTriangulatedDrawableUBO` reads
/// `float pad1,` with a comma, `float pad2` with no semicolon, and declares `pad1` twice, and it
/// sizes the block at ninety-six bytes where `fill_layer_ubo.hpp` asserts eighty. It is dead code
/// in that backend. The GL and Metal versions agree with the authoritative struct and with each
/// other. See #111.
///
/// # No width property, and no data-driven paint
///
/// `u_width` is the literal `1.0` in mbgl -- an outline has no width of its own -- so the
/// half-width is a half pixel and the quad is two across once the fade is added either side. The
/// color is the layer's `outline_color` and the opacity its `opacity`, both read from the
/// evaluated props rather than from an attribute: mbgl only takes this path when both are
/// constant, and a per-feature color keeps the line-primitive path.
///
/// `v_linesofar` is in the GL vertex stage and read by nothing in its fragment, so it is not here.
pub const FILL_OUTLINE_TRIANGULATED_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec2<f32>,
    @location(1) width: f32,
    @location(2) gamma_scale: f32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_outline_triangulated_drawable_ubo[ubo_index];
    let global = global_paint_params_ubo[0];
    var out: Out;

    // The distance the edge fades out over: half a device pixel each side. A denser display needs
    // a narrower fade, which is why mbgl divides by the ratio rather than fixing one pixel.
    let antialiasing = 1.0 / max(global.pixel_ratio, 1e-6) / 2.0;

    // The normal is the low bit of each component; the center is what is left. Identical to the
    // line family's unpacking, because this is the line family's vertex data.
    let packed = vec2<f32>(in.line_pos_normal);
    let center = floor(packed * 0.5);
    var normal = packed - 2.0 * center;
    normal.y = normal.y * 2.0 - 1.0;

    // Four unnormalized bytes; the first two are the extrusion, biased by 128. The third and
    // fourth carry the distance along the line, which this shader does not read -- taken as
    // `.xy` rather than converting the whole vector and swizzling, so that
    // `every_component_of_an_attribute_is_read` sees the two that are left and demands the
    // sanction, instead of reading the bare use as all four.
    let extrude = vec2<f32>(in.line_data.xy) - 128.0;

    // mbgl's `u_width = 1.0`, so a half pixel each side plus the fade outside it.
    let halfwidth = 0.5;
    let outset = halfwidth + antialiasing;

    // The extrusion down to a unit normal -- mbgl's `scale` is one over sixty-three -- and back up
    // by this vertex's width.
    let dist = outset * extrude / 63.0;
    let ratio = max(drawable.ratio, 1e-6);

    // Placed and extruded separately, because the projected extrusion's own length is what says
    // how much perspective squashed this edge. `displace` is the surface's: the extrusion is
    // linear on a plane and is not on a sphere, which is what lets this family draw on all four.
    let projected_extrude = displace(center, dist / ratio, drawable.matrix);
    let clip = place(vec3<f32>(center, 0.0), drawable.matrix) + projected_extrude;
    out.clip = clip;

    // How much the perspective view squashed or stretched the extrusion, which is what keeps the
    // fade a constant width in pixels. Guarded where mbgl's is not: a zero-length extrusion is
    // 0/0 there and NaNs a whole ring away, as `LINE_BODY` records for the same division.
    let unprojected = length(dist);
    let projected = length(projected_extrude.xy / clip.w * global.units_to_pixels);
    out.gamma_scale = unprojected / max(projected, 1e-6);

    out.normal = normal;
    out.width = outset;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    let props = fill_evaluated_props_ubo[0];

    // How far this pixel is from the center of the ring, in pixels, and the fade across the outer
    // edge. `dist + blur2` is the inner edge of a line with no gap, which never bites.
    let distance = length(in.normal) * in.width;
    let blur2 = (1.0 / max(global_paint_params_ubo[0].pixel_ratio, 1e-6)) * in.gamma_scale;
    let alpha = clamp(min(distance + blur2, in.width - distance) / blur2, 0.0, 1.0);

    return props.outline_color * (alpha * props.opacity);
}
";

/// The instanced fill-extrusion family's body: a building's walls.
///
/// The other half of [`FILL_EXTRUSION_BODY`], and the only family here drawn with more than one
/// instance. The geometry is four vertices -- mbgl's `fillExtrusionVertices`, a unit quad -- and
/// the building's outline arrives per *instance*, so one quad raised once per outline point is
/// every wall in the tile. A wall needs the edge it stands on, which is two outline points, so
/// this body reads the next instance as well as its own.
///
/// # The buffer bound twice
///
/// An attribute fetch cannot look ahead, so the next point cannot arrive as an attribute. mbgl's
/// answer is in `AttributeInfo`'s fourth argument: the outline position and the packed decimals
/// name a UBO binding as well as a location, which makes `VertexAttribute::getBufferUsage` add
/// `STORAGE_BUFFER_BIT` and `Drawable::setSharedBuffers` register the same buffer there. The bytes
/// are vertex input *and* a storage buffer, and the body reads this instance from the first and the
/// next instance from the second. tessella#377 is what carries that field to a consumer;
/// `shaders::module` turns it into `SharedInstance` and group one.
///
/// The lookahead is unclamped, exactly as mbgl leaves it. What keeps it inside the buffer is the
/// discard test below, which returns before the read: the closing point of every ring carries the
/// flag, and the buffer ends at some ring's closing point. tessella#378 guards that on the
/// producer's side, over every shape that changes which point lands last.
///
/// # Why `ubo_index` is a push constant here and nowhere else
///
/// Because `firstInstance` is carrying the instance. It addresses the instance-rate attribute
/// fetch as well as `@builtin(instance_index)` -- measured on two drivers in
/// `benches/instance_fetch.rs` -- so the slot cannot ride it on a draw of more than one instance.
/// [`crate::draws`] refuses the pair rather than recording it.
///
/// # What this has that the roof does not
///
/// The vertical gradient. mbgl applies it where the surface normal is horizontal, which is every
/// normal here and no normal on a roof, so the roof's body says it "belongs with the walls and has
/// to arrive with them". This is that arrival. Transcribed from the Vulkan source, including the
/// constant 150 that the height is scaled by and the 0.7-to-0.98 floor that the light intensity
/// chooses between.
pub const FILL_EXTRUSION_INSTANCED_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) shade: vec4<f32>,
}

// Where a wall's foot stands: the integer tile position plus the fraction packed beside the
// discard flag, seven bits an axis. `(frac.x * 256 + frac.y) * 2 + discarded` is what the producer
// packs and what mbgl unpacks, and both ends of a wall come through here -- this instance's from
// its attributes and the next one's from the shared buffer, which is why that buffer is bound
// twice. `decimals` is the sixteen-bit word alone, so the shift is the flag and nothing else.
fn outline_point(position: vec2<f32>, decimals: u32) -> vec2<f32> {
    let pair = f32(decimals >> 1u);
    let high = floor(pair / 256.0);
    return position + vec2<f32>(high, pair - high * 256.0) / 128.0;
}

@vertex
fn vertex_main(in: In) -> Out {
    // Not `in.instance_index`, which is this draw's instance. See the constant's own note.
    ubo_index = pushed.ubo_index;
    let drawable = fill_extrusion_drawable_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    var out: Out;

    // A ring's closing point has no edge leaving it and so no wall to raise. Tested first and
    // returned from first, which is also what keeps the lookahead below inside the buffer.
    // mbgl's `mod(instanceDecimalsEd.x, 2.0) > 0.0`, as the bit it is.
    if (in.fill_extrusion_decimals_ed.x & 1u) != 0u {
        out.clip = vec4<f32>(0.0);
        out.shade = vec4<f32>(0.0);
        return out;
    }

    let p1 = outline_point(
        vec2<f32>(in.fill_extrusion_outline_pos),
        in.fill_extrusion_decimals_ed.x
    );

    // The next outline point, from the same bytes read as a storage element. Its two words carry
    // two sixteen-bit halves each -- mbgl's `unpack_int` and `unpack_uint` -- and the position's
    // are signed where the decimals' are not. The edge distance rides in the high half of the
    // second word and is masked off: it is for a pattern, and this family draws none.
    let next = shared_instance[in.instance_index + 1u];
    let next_position = vec2<f32>(
        f32((next.fill_extrusion_outline_pos << 16u) >> 16u),
        f32(next.fill_extrusion_outline_pos >> 16u)
    );
    let p2 = outline_point(next_position, next.fill_extrusion_decimals_ed & 0xFFFFu);

    // A wall faces along the outline's normal, which is its edge turned a quarter turn. Flat in
    // `z`, which is what makes the gradient below apply to every vertex of this family.
    let edge = normalize(p2 - p1);
    let normal = vec3<f32>(-edge.y, edge.x, 0.0);

    let base = max(paint_base(in), 0.0);
    let height = max(paint_height(in), 0.0);

    // The template's corner chooses which end of the edge and which end of the wall: `x` picks
    // the point, `y` picks the foot or the top.
    let t = f32(in.fill_extrusion_pos.y);
    let z = select(base, height, t > 0.0);
    let foot = select(p2, p1, in.fill_extrusion_pos.x == 0);

    // How bright the surface already is. A pale wall takes a narrower range of shading than a
    // dark one, and a slight ambient keeps nothing completely black.
    var color = paint_color(in);
    let luminance = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
    color += vec4<f32>(0.03, 0.03, 0.03, 1.0);

    let facing = clamp(dot(normal, props.light_position), 0.0, 1.0);
    let least = 1.0 - props.light_intensity;
    let most = max(1.0 - luminance + props.light_intensity, 1.0);
    var directional = mix(least, most, facing);

    // The gradient down a wall, which only a horizontal normal takes. mbgl writes it as a
    // multiply rather than a branch and says so: without a vertical gradient the factor is a
    // constant, and with one it falls off from the top of the building toward its base. The
    // floor the clamp takes is chosen by the light intensity, so a dim light darkens less far.
    let dimmest = mix(0.7, 0.98, 1.0 - props.light_intensity);
    let factor = clamp((t + base) * pow(height / 150.0, 0.5), dimmest, 1.0);
    directional *= (1.0 - props.vertical_gradient) + props.vertical_gradient * factor;

    // Shading is floored at a tint complementary to the light, so a colored light leaves its
    // opposite in the shadows rather than driving them to black.
    let floor_light = mix(vec3<f32>(0.0), vec3<f32>(0.3), 1.0 - props.light_color);
    let lit = clamp(color.rgb * directional * props.light_color, floor_light, vec3<f32>(1.0));

    // Premultiplied, over a color whose alpha was one.
    out.shade = vec4<f32>(lit, 1.0) * props.opacity;
    out.clip = place(vec3<f32>(foot, z), drawable.matrix);
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return in.shade;
}
";

/// The patterned extrusion's body: a roof with a sprite tiled over it.
///
/// [`FILL_EXTRUSION_BODY`]'s placement and lighting with [`FILL_PATTERN_BODY`]'s sprite
/// arithmetic, and the eighteenth of mbgl's shaders this crate draws. The producer emits it for a
/// layer whose `fill-extrusion-pattern` resolves, which `tools/parity/scenes/extrusion_pattern_p`
/// is a scene that does.
///
/// # The lighting is not the plain family's
///
/// It is shorter by exactly the surface color, and that is mbgl's difference rather than a
/// simplification. `FillExtrusionShader` reads the color, takes its relative luminance and widens
/// the shading range by `1.0 - luminance + light_intensity`, so a pale building is shaded less
/// than a dark one. `FillExtrusionPatternShader` has no color attribute at all -- the surface is
/// the sprite, which the vertex stage has not sampled -- so mbgl uses the constant
/// `max(0.5 + light_intensity, 1.0)` in its place and multiplies the sampled color by the result
/// in the fragment stage.
///
/// Copying the plain body's lighting here would compile, draw, and be wrong by the luminance of
/// whatever `fill-extrusion-color` the layer happens to carry beside its pattern -- which the
/// pattern replaces. `benches/first_pixel.rs`'s `fill_extrusion_pattern` is the case that reads
/// the difference: its inputs make the two formulas disagree.
///
/// # The vertical gradient is dead here, as it is in the plain roof
///
/// mbgl guards it with `normal.z == 0.0` and the normal is `vec3(0, 0, 1)`, so a roof never takes
/// it. Written out all the same, for [`FILL_EXTRUSION_BODY`]'s reason: the branch is the oracle's
/// and a body that dropped it would agree by coincidence of what this family draws rather than by
/// computing the same thing. The walls beside it are
/// [`FILL_EXTRUSION_PATTERN_INSTANCED_BODY`], where the normal is horizontal and the gradient is
/// the whole point.
pub const FILL_EXTRUSION_PATTERN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) lighting: vec4<f32>,
    @location(1) pos_a: vec2<f32>,
    @location(2) pos_b: vec2<f32>,
    @location(3) pattern_from: vec4<f32>,
    @location(4) pattern_to: vec4<f32>,
    @location(5) @interpolate(flat) ubo: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let drawable = fill_extrusion_drawable_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    let footprint = vec2<f32>(in.fill_extrusion_pos);

    // The fraction the position was packed with, seven bits an axis, as the plain roof reads it.
    let packed = floor(f32(in.fill_extrusion_decimals_ed.x) / 2.0);
    let high = floor(packed / 256.0);
    let decimals = vec2<f32>(high, packed - high * 256.0) / 128.0;

    let base = max(paint_base(in), 0.0);
    let height = max(paint_height(in), 0.0);
    let on_roof = 1.0;
    let z = select(base, height, on_roof > 0.0);

    // Straight through, not mixed: a pattern's rectangles are corners in the atlas, and the
    // cross-fade between the two is `fade` in the fragment stage.
    let pattern_from = paint_pattern_from(in);
    let pattern_to = paint_pattern_to(in);
    out.pattern_from = pattern_from;
    out.pattern_to = pattern_to;

    // The sprite's extent in the atlas, as the size it should cover on screen.
    let display_a = (pattern_from.zw - pattern_from.xy) / max(global.pixel_ratio, 1e-6);
    let display_b = (pattern_to.zw - pattern_to.xy) / max(global.pixel_ratio, 1e-6);

    // A roof's pattern runs over its footprint, which is the position *without* the decimals --
    // mbgl's `vec2(in_position)`, where the clip position below adds them.
    out.pos_a = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.from_scale * display_a,
        drawable.tile_ratio,
        footprint
    );
    out.pos_b = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.to_scale * display_b,
        drawable.tile_ratio,
        footprint
    );

    // A roof faces straight up, so the directional term is the light's own elevation. The range
    // it is mixed into is a constant rather than the surface's luminance: see this body's note.
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    let facing = clamp(dot(normal, props.light_position), 0.0, 1.0);
    let least = 1.0 - props.light_intensity;
    let most = max(0.5 + props.light_intensity, 1.0);
    var directional = mix(least, most, facing);

    // The gradient down a wall, which a roof's normal does not take. Written as mbgl writes it.
    if normal.z == 0.0 {
        let dimmest = mix(0.7, 0.98, 1.0 - props.light_intensity);
        let factor = clamp((on_roof + base) * pow(height / 150.0, 0.5), dimmest, 1.0);
        directional *= (1.0 - props.vertical_gradient) + props.vertical_gradient * factor;
    }

    // Shading is floored at a tint complementary to the light, so a colored light leaves its
    // opposite in the shadows rather than driving them to black. No surface color in it: the
    // sprite is the surface, and the fragment stage multiplies the two.
    let floor_light = mix(vec3<f32>(0.0), vec3<f32>(0.3), 1.0 - props.light_color);
    let lit = clamp(directional * props.light_color, floor_light, vec3<f32>(1.0));

    // Premultiplied, over a lighting whose alpha was one.
    out.lighting = vec4<f32>(lit, 1.0) * props.opacity;

    out.clip = place(vec3<f32>(footprint + decimals, z), drawable.matrix);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    // The slot, carried: a fragment invocation's own copy of a private is zero.
    ubo_index = in.ubo;
    let tile = fill_extrusion_tile_props_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    let texsize = tile.texsize;

    let in_a = wrap(in.pos_a, vec2<f32>(1.0, 1.0));
    let pos_a = mix(in.pattern_from.xy / texsize, in.pattern_from.zw / texsize, in_a);
    let color_a = textureSample(fill_extrusion_image, fill_extrusion_image_sampler, pos_a);

    let in_b = wrap(in.pos_b, vec2<f32>(1.0, 1.0));
    let pos_b = mix(in.pattern_to.xy / texsize, in.pattern_to.zw / texsize, in_b);
    let color_b = textureSample(fill_extrusion_image, fill_extrusion_image_sampler, pos_b);

    return mix(color_a, color_b, props.fade) * in.lighting;
}
";

/// The patterned extrusion's walls: one per outline edge, with the sprite running along them.
///
/// [`FILL_EXTRUSION_INSTANCED_BODY`]'s geometry and [`FILL_EXTRUSION_PATTERN_BODY`]'s sprite, and
/// the one body here that needs the *edge distance*. The plain walls read the same word and mask
/// it off -- "it is for a pattern, and this family draws none" is what that body says about it --
/// so this is the family that word was always for.
///
/// # The pattern coordinate is not a position
///
/// A roof tiles over its footprint, which is a place on the ground. A wall has no footprint: it is
/// a vertical strip, and mbgl tiles it over `vec2(edgedistance, z * height_factor)` -- how far
/// along the ring this edge begins and how high up this vertex is. Feed it a position instead and
/// the sprite runs over the building's plan rather than up its side, which draws something
/// plausible and wrong: the pattern then slides with the wall's compass direction instead of
/// standing still on it.
///
/// `height_factor` is the drawable's own, and it is what puts the vertical axis in the same units
/// as the horizontal one.
///
/// # The edge distance is read from the *other* end
///
/// `in_position.x == 0.0 ? nextInstanceDecimalsEd.y : instanceDecimalsEd.y`, which pairs the
/// vertex at `p1` with the *next* instance's accumulated distance. Transcribed as mbgl has it
/// rather than as it reads: the distance accumulates to a point along the ring, and swapping the
/// two ends is a pattern that runs backwards along every wall -- a difference no count would
/// catch.
pub const FILL_EXTRUSION_PATTERN_INSTANCED_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) lighting: vec4<f32>,
    @location(1) pos_a: vec2<f32>,
    @location(2) pos_b: vec2<f32>,
    @location(3) pattern_from: vec4<f32>,
    @location(4) pattern_to: vec4<f32>,
    @location(5) @interpolate(flat) ubo: u32,
}

// Where a wall's foot stands, as the plain walls read it: the integer tile position plus the
// fraction packed beside the discard flag.
fn outline_point(position: vec2<f32>, decimals: u32) -> vec2<f32> {
    let pair = f32(decimals >> 1u);
    let high = floor(pair / 256.0);
    return position + vec2<f32>(high, pair - high * 256.0) / 128.0;
}

@vertex
fn vertex_main(in: In) -> Out {
    // Not `in.instance_index`, which is this draw's instance. See the constant's own note.
    ubo_index = pushed.ubo_index;
    let drawable = fill_extrusion_drawable_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    let global = global_paint_params_ubo[0];
    var out: Out;

    // A ring's closing point has no edge leaving it and so no wall to raise. Tested first, which
    // is also what keeps the lookahead below inside the buffer.
    if (in.fill_extrusion_decimals_ed.x & 1u) != 0u {
        out.clip = vec4<f32>(0.0);
        out.lighting = vec4<f32>(0.0);
        out.pos_a = vec2<f32>(0.0);
        out.pos_b = vec2<f32>(0.0);
        out.pattern_from = vec4<f32>(0.0);
        out.pattern_to = vec4<f32>(0.0);
        out.ubo = ubo_index;
        return out;
    }

    let p1 = outline_point(
        vec2<f32>(in.fill_extrusion_outline_pos),
        in.fill_extrusion_decimals_ed.x
    );

    // The next outline point, from the same bytes read as a storage element. Both halves of the
    // second word are wanted here: the low one is the decimals and the flag, the high one is the
    // edge distance this vertex may be the one to use.
    let next = shared_instance[in.instance_index + 1u];
    let next_position = vec2<f32>(
        f32((next.fill_extrusion_outline_pos << 16u) >> 16u),
        f32(next.fill_extrusion_outline_pos >> 16u)
    );
    let next_decimals = next.fill_extrusion_decimals_ed & 0xFFFFu;
    let next_edge = f32(next.fill_extrusion_decimals_ed >> 16u);
    let p2 = outline_point(next_position, next_decimals);

    // A wall faces along the outline's normal, which is its edge turned a quarter turn. Flat in
    // `z`, which is what makes the gradient below apply to every vertex of this family.
    let edge = normalize(p2 - p1);
    let normal = vec3<f32>(-edge.y, edge.x, 0.0);

    let base = max(paint_base(in), 0.0);
    let height = max(paint_height(in), 0.0);

    // The template's corner chooses which end of the edge and which end of the wall: `x` picks
    // the point, `y` picks the foot or the top.
    let t = f32(in.fill_extrusion_pos.y);
    let z = select(base, height, t > 0.0);
    let at_p1 = in.fill_extrusion_pos.x == 0;
    let foot = select(p2, p1, at_p1);
    // And the other end's distance, which is mbgl's pairing. See this body's note.
    let edge_distance = select(f32(in.fill_extrusion_decimals_ed.y), next_edge, at_p1);

    let pattern_from = paint_pattern_from(in);
    let pattern_to = paint_pattern_to(in);
    out.pattern_from = pattern_from;
    out.pattern_to = pattern_to;

    let display_a = (pattern_from.zw - pattern_from.xy) / max(global.pixel_ratio, 1e-6);
    let display_b = (pattern_to.zw - pattern_to.xy) / max(global.pixel_ratio, 1e-6);

    // Along the ring and up the wall, in tile units. See this body's note.
    let along = vec2<f32>(edge_distance, z * drawable.height_factor);
    out.pos_a = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.from_scale * display_a,
        drawable.tile_ratio,
        along
    );
    out.pos_b = pattern_pos(
        drawable.pixel_coord_upper,
        drawable.pixel_coord_lower,
        props.to_scale * display_b,
        drawable.tile_ratio,
        along
    );

    // The shading range is a constant rather than the surface's luminance, for the reason
    // `FILL_EXTRUSION_PATTERN_BODY` gives: there is no surface color to read.
    let facing = clamp(dot(normal, props.light_position), 0.0, 1.0);
    let least = 1.0 - props.light_intensity;
    let most = max(0.5 + props.light_intensity, 1.0);
    var directional = mix(least, most, facing);

    // The gradient down a wall, which this family's normal always takes.
    let dimmest = mix(0.7, 0.98, 1.0 - props.light_intensity);
    let factor = clamp((t + base) * pow(height / 150.0, 0.5), dimmest, 1.0);
    directional *= (1.0 - props.vertical_gradient) + props.vertical_gradient * factor;

    let floor_light = mix(vec3<f32>(0.0), vec3<f32>(0.3), 1.0 - props.light_color);
    let lit = clamp(directional * props.light_color, floor_light, vec3<f32>(1.0));

    out.lighting = vec4<f32>(lit, 1.0) * props.opacity;
    out.clip = place(vec3<f32>(foot, z), drawable.matrix);
    out.ubo = ubo_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    ubo_index = in.ubo;
    let tile = fill_extrusion_tile_props_ubo[ubo_index];
    let props = fill_extrusion_props_ubo[0];
    let texsize = tile.texsize;

    let in_a = wrap(in.pos_a, vec2<f32>(1.0, 1.0));
    let pos_a = mix(in.pattern_from.xy / texsize, in.pattern_from.zw / texsize, in_a);
    let color_a = textureSample(fill_extrusion_image, fill_extrusion_image_sampler, pos_a);

    let in_b = wrap(in.pos_b, vec2<f32>(1.0, 1.0));
    let pos_b = mix(in.pattern_to.xy / texsize, in.pattern_to.zw / texsize, in_b);
    let color_b = textureSample(fill_extrusion_image, fill_extrusion_image_sampler, pos_b);

    return mix(color_a, color_b, props.fade) * in.lighting;
}
";

/// The ground's own body, which is the surface every raised family is placed on.
///
/// The one family mbgl has no shader for. Its mesh is `tessella_layout::terrain::mesh` -- 129
/// vertices a side over a tile's own coordinates, shared by every tile of the frame -- drawn once
/// per tile through that tile's matrix, with the height read from that tile's DEM per vertex.
///
/// # It raises itself, and not through [`Surface::Terrain`]
///
/// A fill standing on the ground is placed on `Surface::Terrain`, which reads the same block and
/// does the same arithmetic. This body does it inline instead, for two reasons that are both the
/// same fact: the ground is not standing on a surface, it *is* one, so the family declares the
/// raise block as its own and the plane's placement is all it needs on top -- and its elevation
/// arrives at the slot its own table declares rather than at the one a raised mbgl family's table
/// leaves free, so the texture it samples is a different binding of a different set.
///
/// `tessella_fluorite`'s `terrain.mat` is the sibling of this, and `terrain_height.glsl` the
/// shared half it factors out.
///
/// # The skirt, which only the ground has
///
/// The curtain hanging from each tile edge, hiding the crack where two tiles at different zooms
/// sample the DEM at different resolutions. A flagged vertex takes the surface's height and then
/// drops `skirt.x` meters -- a fifth of a tile, which is invisible only because the ground is the
/// backmost thing there is. A layer *on* the ground hangs a far shallower one, `skirt.z`, which is
/// [`Surface::Terrain`]'s business and not this body's.
///
/// The flag is the mesh's second attribute rather than a third component of the first, and that is
/// Filament's doing rather than a choice: it draws nothing at all from a `Short3` position. The
/// ABI's [`TERRAIN_MESH_ATTRIBUTES`](tessella_capture_abi::terrain_ubo::TERRAIN_MESH_ATTRIBUTES)
/// carries the pair and the stride that pads them to a multiple of four.
///
/// # The color is a varying and not a second block read
///
/// What bare ground is painted -- the style's `background-color`, so a terrain with nothing over
/// it reads continuous with the flat map. It rides in the per-drawable block, where every tile's
/// copy holds the same four numbers, and `ubo_index` is the vertex stage's: a fragment stage has
/// no instance to index by, so the vertex stage passes it on the way through.
pub const TERRAIN_BODY: &str = r"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    ubo_index = in.instance_index;
    let ground = terrain_drawable_ubo[ubo_index];
    let position = vec2<f32>(in.tessella_terrain_pos);

    // One multiply-add an axis: the producer folded the DEM's width, its border pixel, the half
    // texel that puts a cell at its own center, and which square of a coarser DEM tile this one
    // occupies, into these three numbers.
    let uv = position * ground.params.x + ground.params.yz;
    // Explicit level: a vertex stage has no derivatives to pick one from. The channels come back
    // normalized and the unpack vector is in bytes, which is what scales them back up.
    let channels = textureSampleLevel(elevation, elevation_sampler, uv, 0.0).rgb * 255.0;
    let meters = dot(channels, ground.unpack.rgb) - ground.unpack.a;
    // Relative to the ground under the camera's center, which `skirt.y` carries: the camera sits a
    // fixed distance above the plane while a height in meters reaches the screen multiplied by
    // pixels-per-meter, so measured from sea level the ground climbs toward a camera that does not
    // climb with it. Subtracted before the exaggeration, so stretching the relief leaves the center
    // where it is.
    var height = (meters - ground.skirt.y) * ground.params.w;
    // And the skirt drops, on the flagged vertices alone. Not scaled by the exaggeration: the
    // curtain is a depth below the surface rather than a part of the relief.
    height -= f32(in.tessella_terrain_skirt.x) * ground.skirt.x;

    var out: Out;
    out.clip = place(vec3<f32>(position, height), ground.matrix);
    out.color = ground.color;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return in.color;
}
";
