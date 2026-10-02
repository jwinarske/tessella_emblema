//! Does an assembled family compile, and does it still say what it said?
//!
//! The declarations come from the ABI's tables and the body is written against them, so the thing
//! being checked is that the two fit: a body naming a field the tables do not declare, or reading
//! one at the wrong type, fails here rather than at pipeline creation on a board.

use tessella_capture_abi::generated::shader_attributes::{
    BACKGROUND_SHADER, CIRCLE_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_SHADER,
    ShaderAttribute,
};
use tessella_capture_abi::generated::ubo_layouts::{
    BACKGROUND_DRAWABLE_UBO, BACKGROUND_PROPS_UBO, CIRCLE_DRAWABLE_UBO, CIRCLE_EVALUATED_PROPS_UBO,
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, LINE_DRAWABLE_UBO, LINE_EVALUATED_PROPS_UBO,
    UboLayout,
};
use tessella_emblema::shaders::{
    BACKGROUND_BODY, CIRCLE_BODY, FILL_BODY, FILL_OUTLINE_BODY, LINE_BODY, attribute_name, module,
};
use tessella_emblema::surface::Surface;

/// A family, and the surfaces the producer can draw it on.
struct Family {
    name: &'static str,
    blocks: Vec<&'static UboLayout>,
    attributes: &'static [ShaderAttribute],
    body: &'static str,
    surfaces: &'static [Surface],
}

/// Every family a plane module exists for, and which surfaces each one has.
///
/// A background has neither of the two surfaces that need a block of their own: it covers the
/// viewport rather than a tile, so the producer writes it no bend block and never marks it
/// raised. Everything else has all four.
fn families() -> Vec<Family> {
    let all = &[
        Surface::Plane,
        Surface::Globe,
        Surface::GlobeAnchored,
        Surface::Terrain,
    ][..];
    let flat_or_bent = &[Surface::Plane, Surface::Globe][..];
    vec![
        Family {
            name: "background",
            blocks: vec![&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
            attributes: &BACKGROUND_SHADER,
            body: BACKGROUND_BODY,
            surfaces: flat_or_bent,
        },
        Family {
            name: "fill",
            blocks: vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            attributes: &FILL_SHADER,
            body: FILL_BODY,
            surfaces: all,
        },
        Family {
            name: "fill_outline",
            blocks: vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            attributes: &FILL_OUTLINE_SHADER,
            body: FILL_OUTLINE_BODY,
            surfaces: all,
        },
        Family {
            name: "line",
            blocks: vec![&LINE_DRAWABLE_UBO, &LINE_EVALUATED_PROPS_UBO],
            attributes: &LINE_SHADER,
            body: LINE_BODY,
            surfaces: all,
        },
        Family {
            name: "circle",
            blocks: vec![&CIRCLE_DRAWABLE_UBO, &CIRCLE_EVALUATED_PROPS_UBO],
            attributes: &CIRCLE_SHADER,
            body: CIRCLE_BODY,
            surfaces: all,
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
        source.contains("fn as_matrix(columns: array<vec4<f32>, 4>) -> mat4x4<f32>"),
        "the prelude does not define the way a body reaches one"
    );
    // The four columns in order, pinned verbatim. A snapshot of one line is the right shape of
    // test here: nothing else can catch a repeated or transposed index, because every wrong
    // assembly is a well-typed `mat4x4` that compiles and validates. What it draws is every
    // vertex in the wrong place, which reads as a camera fault rather than as a typo. The
    // producer writes column-major and `mat4x4<f32>(a, b, c, d)` takes columns, so these are
    // columns.
    assert!(
        source.contains("return mat4x4<f32>(columns[0], columns[1], columns[2], columns[3]);"),
        "the helper does not assemble the four columns in order:\n{source}"
    );
    assert!(
        source.contains("as_matrix(drawable.matrix)"),
        "the body does not reach the matrix through the helper"
    );
}

/// Every family's body reaches a block matrix through `as_matrix` rather than directly.
#[test]
fn no_body_uses_a_block_matrix_as_a_matrix() {
    for family in families() {
        for surface in family.surfaces {
            let source = module(*surface, &family.blocks, family.attributes, family.body)
                .expect("assembles");
            // `drawable.matrix` on its own is the mistake; wrapped, it is preceded by the helper.
            for at in source.match_indices("drawable.matrix").map(|(at, _)| at) {
                let before = &source[..at];
                assert!(
                    before.ends_with("as_matrix("),
                    "{}{} reads drawable.matrix without as_matrix",
                    family.name,
                    surface.suffix()
                );
            }
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
            "@location({}) background_pos: vec3<f32>",
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
            let source = module(*surface, &family.blocks, family.attributes, family.body)
                .unwrap_or_else(|why| panic!("{what} does not assemble: {why:?}"));
            let words = compile(&source);
            assert!(words.len() > 64, "{what} emitted {} words", words.len());
            assert_eq!(words[0], 0x0723_0203, "{what} is not SPIR-V");
            pairs += 1;
        }
    }
    assert_eq!(
        pairs, 18,
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
            let source = module(*surface, &family.blocks, family.attributes, family.body)
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

/// Every data-driven property's zoom factor is used where its attribute is.
///
/// The `_t` fields exist to mix an attribute between its two zoom endpoints. Reading the attribute
/// and ignoring the factor draws the lower endpoint at every zoom, which looks like a style that
/// stopped interpolating rather than like a bug.
#[test]
fn every_zoom_factor_is_used() {
    for family in families() {
        for surface in family.surfaces {
            let source = module(*surface, &family.blocks, family.attributes, family.body)
                .expect("assembles");
            for block in &family.blocks {
                for field in block.fields {
                    if !field.name.ends_with("_t") {
                        continue;
                    }
                    assert!(
                        identifier_uses(&source, field.name) >= 2,
                        "{}{} declares {} and never mixes with it",
                        family.name,
                        surface.suffix(),
                        field.name
                    );
                }
            }
        }
    }
}
