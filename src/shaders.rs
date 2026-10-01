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

use std::fmt::Write as _;

use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_capture_abi::generated::shader_attributes::ShaderAttribute;
use tessella_capture_abi::generated::ubo_layouts::UboLayout;

use crate::preamble::{Unrepresentable, declare, type_name};

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

/// Assembles a module: the blocks, the vertex input, then the body.
///
/// Blocks bind in the order given, from zero. The body sees each block as a binding named after
/// its type in snake case, and the vertex input as `In`.
///
/// # Errors
///
/// [`Error`] from a block that cannot be declared, or an attribute the ABI calls invalid.
pub fn module(
    blocks: &[&UboLayout],
    attributes: &[ShaderAttribute],
    body: &str,
) -> Result<String, Error> {
    let mut out = String::new();
    out.push_str("// Generated declarations. The body below is the only hand-written part.\n\n");

    for layout in blocks {
        out.push_str(&declare(layout)?);
        out.push('\n');
    }

    for (binding, layout) in blocks.iter().enumerate() {
        let name = type_name(layout.name);
        let _ = writeln!(
            out,
            "@group(0) @binding({binding}) var<storage, read> {}: array<{name}>;",
            binding_name(&name)
        );
    }
    out.push('\n');

    // The vertex input, at the locations the producer binds its buffers to. A body naming a field
    // the family does not declare does not compile, which is the point of generating this.
    out.push_str("struct In {\n");
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
/// `ubo_index` reaches the vertex stage as a push constant, which is why the slot is read from one
/// rather than passed per vertex.
pub const BACKGROUND_BODY: &str = r"
var<push_constant> ubo_index: u32;

struct Out {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    let drawable = background_drawable_ubo[ubo_index];
    var out: Out;
    out.clip = drawable.matrix * vec4<f32>(in.background_pos, 1.0);
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    let props = background_props_ubo[0];
    return props.color * props.opacity;
}
";
