//! WGSL declarations for the uniform blocks, generated from the ABI's own tables.
//!
//! A shader reads a block the producer wrote. If the shader's idea of where a field sits differs
//! from the producer's by so much as a padding word, every field after it is read from the wrong
//! bytes — and the result draws, because the bytes are all valid floats. Nothing errors, nothing
//! validates, and the picture is wrong in a way that looks like a maths bug in whatever the field
//! fed.
//!
//! So the declarations are not written by hand. They are generated from the same table that
//! describes the block's C++ layout, and the generator refuses any block whose fields WGSL would
//! place somewhere else.
//!
//! # WGSL will not simply agree
//!
//! WGSL has its own alignment rules and they are not the ones a C++ struct followed. `vec3<f32>`
//! occupies twelve bytes and aligns to sixteen, so a `vec3` followed by an `f32` packs into one
//! sixteen-byte slot in WGSL and may or may not have in the header. A `mat4x4<f32>` aligns to
//! sixteen. Where the two disagree the generator emits explicit padding rather than hoping, and
//! where it cannot make them agree it says so instead of emitting a struct that reads wrong.

use std::fmt::Write as _;

use tessella_capture_abi::generated::ubo_layouts::{UboField, UboFieldKind, UboLayout};

/// Why a block could not be declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unrepresentable {
    /// A field sits where WGSL's rules cannot put it, even with padding before it.
    ///
    /// Padding can only move a field later. A declared offset earlier than WGSL's next legal one
    /// means the header packed something tighter than WGSL permits, and no declaration reads it
    /// correctly.
    FieldTooEarly {
        /// The block.
        block: &'static str,
        /// The field that cannot be placed.
        field: &'static str,
        /// Where the producer puts it.
        declared: u32,
        /// The earliest WGSL would.
        earliest: u32,
    },
    /// The block's own stride is not a multiple of its alignment.
    ///
    /// A consolidated buffer is an array of these, and an array element stride that is not
    /// aligned puts every block after the first at an offset WGSL will not index to.
    StrideUnaligned {
        /// The block.
        block: &'static str,
        /// Its stride.
        stride: u32,
        /// Its alignment.
        align: u32,
    },
}

/// What WGSL requires of each kind: its alignment and the space it takes.
const fn wgsl_align(kind: UboFieldKind) -> u32 {
    match kind {
        UboFieldKind::F32 | UboFieldKind::I32 | UboFieldKind::U32 => 4,
        UboFieldKind::Vec2 => 8,
        UboFieldKind::Vec3 | UboFieldKind::Vec4 | UboFieldKind::Color | UboFieldKind::Mat4 => 16,
    }
}

/// The WGSL type that holds this field.
///
/// A `mat4` is declared as four `vec4`s and not as `mat4x4<f32>`, which is the one place this
/// deviates from the obvious transcription. See [`MATRIX`] for why.
const fn wgsl_type(kind: UboFieldKind) -> &'static str {
    match kind {
        UboFieldKind::F32 => "f32",
        UboFieldKind::I32 => "i32",
        UboFieldKind::U32 => "u32",
        UboFieldKind::Vec2 => "vec2<f32>",
        UboFieldKind::Vec3 => "vec3<f32>",
        UboFieldKind::Vec4 | UboFieldKind::Color => "vec4<f32>",
        UboFieldKind::Mat4 => MATRIX,
    }
}

/// How a `mat4` is declared: four columns, not a matrix.
///
/// # Why not `mat4x4<f32>`
///
/// Adreno's shader compiler asserts on a `mat4x4` read from a storage buffer --
/// `matOpnd->isMatrix()` in its own `CodeGenHelper.cpp` -- and fails pipeline creation with
/// `VK_ERROR_UNKNOWN`. Every block here is read from a storage buffer indexed by `ubo_index`, and
/// every family's drawable block carries a matrix, so the obvious declaration builds no pipeline
/// at all on the sa8155p. RADV and V3DV both accept it, which is why a desktop says nothing.
///
/// # Why the layout does not move
///
/// WGSL gives `array<vec4<f32>, 4>` an element stride of sixteen, an alignment of sixteen and a
/// size of sixty-four -- the same three numbers as `mat4x4<f32>`. So [`offsets`] answers
/// identically either way and the producer's offsets are untouched. This is a change of spelling,
/// not of layout.
///
/// # And a matrix is never assembled from them either
///
/// Vivante's SPIR-V compiler segfaults on `OpCompositeConstruct` of a matrix --
/// `VIR_Shader_CompositeConstruct` in `libVSC.so`, reached through `gcSPV_Decode` -- so the first
/// answer to Adreno's assertion, declaring the columns and rebuilding a `mat4x4` from them, trades
/// one vendor's crash for another's. The two together leave one form: no matrix type anywhere, and
/// the multiply written out as the four multiply-adds it is. `shaders::PRELUDE` defines
/// `transform`, which is how a body applies one.
///
/// # Why this is the safer spelling anyway
///
/// An array is not a matrix to WGSL, so a body that multiplies one by a vector does not compile.
/// The mistake this exists to prevent therefore cannot be made silently: there is no way to write
/// the code that works on a desktop and dies on a board.
pub const MATRIX: &str = "array<vec4<f32>, 4>";

/// Rounds `at` up to a multiple of `align`.
const fn align_to(at: u32, align: u32) -> u32 {
    at.div_ceil(align) * align
}

/// A WGSL identifier for a block, from the name the header spells.
///
/// `LineSDFTilePropsUBO` becomes `LineSdfTilePropsUbo`: WGSL has no reserved word among these and
/// the shape is only a convention, but one convention beats each shader choosing.
#[must_use]
pub fn type_name(block: &str) -> String {
    let chars: Vec<char> = block.chars().collect();
    let mut out = String::with_capacity(chars.len());
    for (at, ch) in chars.iter().enumerate() {
        if !ch.is_ascii_uppercase() {
            out.push(*ch);
            continue;
        }
        // Inside a run of capitals, keep the first and keep the one that starts the next word --
        // the one followed by a lowercase. `LineSDFTilePropsUBO` has to give up the D and F and
        // keep the T, or `TileProps` reads as `tileProps`.
        let opens_run = at == 0 || !chars[at - 1].is_ascii_uppercase();
        let opens_word = chars.get(at + 1).is_some_and(char::is_ascii_lowercase);
        if opens_run || opens_word {
            out.push(*ch);
        } else {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

/// The WGSL `struct` for one block, with explicit padding wherever WGSL would place a field
/// anywhere other than where the producer does.
///
/// # Errors
///
/// [`Unrepresentable`] when no padding can reconcile the two, which is a block this consumer must
/// not pretend to read.
pub fn declare(layout: &UboLayout) -> Result<String, Unrepresentable> {
    if layout.align == 0 || !layout.stride.is_multiple_of(layout.align) {
        return Err(Unrepresentable::StrideUnaligned {
            block: layout.name,
            stride: layout.stride,
            align: layout.align,
        });
    }

    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated from {}'s `{}`. Do not edit: the offsets are the producer's.",
        layout.header, layout.name
    );
    let _ = writeln!(out, "struct {} {{", type_name(layout.name));

    let mut at = 0u32;
    let mut pads = 0usize;
    for field in layout.fields {
        let earliest = align_to(at, wgsl_align(field.kind));
        if field.offset < earliest {
            return Err(Unrepresentable::FieldTooEarly {
                block: layout.name,
                field: field.name,
                declared: field.offset,
                earliest,
            });
        }
        // Pad in whole words up to where the producer puts it. `@align` would do for a field whose
        // offset happens to be a power-of-two multiple; explicit words work for every offset and
        // read the same way in the generated source.
        if field.offset > earliest {
            let words = (field.offset - earliest) / 4;
            if words > 0 {
                let _ = writeln!(out, "    _pad{pads}: array<u32, {words}>,");
                pads += 1;
            }
        }
        let _ = writeln!(out, "    {}: {},", field.name, wgsl_type(field.kind));
        at = field.offset + field.kind.size();
    }

    let _ = writeln!(out, "}}");
    Ok(out)
}

/// Where WGSL places each of a block's fields, given the declaration [`declare`] would emit.
///
/// The check behind the generator: these are the offsets a shader will actually read from, and
/// they have to be the ones the producer wrote to.
///
/// # Errors
///
/// As [`declare`].
pub fn offsets(layout: &UboLayout) -> Result<Vec<(&'static str, u32)>, Unrepresentable> {
    let mut out = Vec::with_capacity(layout.fields.len());
    let mut at = 0u32;
    for field in layout.fields {
        let earliest = align_to(at, wgsl_align(field.kind));
        if field.offset < earliest {
            return Err(Unrepresentable::FieldTooEarly {
                block: layout.name,
                field: field.name,
                declared: field.offset,
                earliest,
            });
        }
        // The padding `declare` emits moves the field to exactly its declared offset, so that is
        // where WGSL will read it -- provided the padding is expressible in whole words, which is
        // what this agrees with.
        let placed = if (field.offset - earliest).is_multiple_of(4) {
            field.offset
        } else {
            earliest
        };
        out.push((field.name, placed));
        at = placed + field.kind.size();
    }
    Ok(out)
}

/// A block's fields, for a caller comparing against something else.
#[must_use]
pub fn fields(layout: &UboLayout) -> &'static [UboField] {
    layout.fields
}
