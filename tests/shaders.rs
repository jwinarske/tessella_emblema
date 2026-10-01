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

/// Every plane family: its blocks, its attributes, its body.
fn families() -> Vec<(
    &'static str,
    Vec<&'static UboLayout>,
    &'static [ShaderAttribute],
    &'static str,
)> {
    vec![
        (
            "background",
            vec![&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
            &BACKGROUND_SHADER,
            BACKGROUND_BODY,
        ),
        (
            "fill",
            vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            &FILL_SHADER,
            FILL_BODY,
        ),
        (
            "fill_outline",
            vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            &FILL_OUTLINE_SHADER,
            FILL_OUTLINE_BODY,
        ),
        (
            "line",
            vec![&LINE_DRAWABLE_UBO, &LINE_EVALUATED_PROPS_UBO],
            &LINE_SHADER,
            LINE_BODY,
        ),
        (
            "circle",
            vec![&CIRCLE_DRAWABLE_UBO, &CIRCLE_EVALUATED_PROPS_UBO],
            &CIRCLE_SHADER,
            CIRCLE_BODY,
        ),
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
        "matrix: mat4x4<f32>",
        "color: vec4<f32>",
        "opacity: f32",
    ] {
        assert!(source.contains(named), "{named} is missing from:\n{source}");
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

/// Every plane family compiles, validates and emits.
#[test]
fn every_plane_family_compiles() {
    for (name, blocks, attributes, body) in families() {
        let source = module(&blocks, attributes, body)
            .unwrap_or_else(|why| panic!("{name} does not assemble: {why:?}"));
        let words = compile(&source);
        assert!(words.len() > 64, "{name} emitted {} words", words.len());
        assert_eq!(words[0], 0x0723_0203, "{name} is not SPIR-V");
    }
}

/// Every attribute the producer sends is read by the body that receives it.
///
/// An attribute declared and never read is a paint property the style asked for and the picture
/// does not show — which draws, because the rest of the shader is fine. The compiler cannot catch
/// it: an unused input is legal.
#[test]
fn every_attribute_is_read() {
    for (name, blocks, attributes, body) in families() {
        let source = module(&blocks, attributes, body).expect("assembles");
        for attribute in attributes {
            let field = attribute_name(attribute.name);
            // Once in the generated input, and at least once more in the body that reads it.
            let uses = identifier_uses(&source, &field);
            assert!(uses >= 2, "{name} declares {field} and never reads it");
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
    for (name, blocks, attributes, body) in families() {
        let source = module(&blocks, attributes, body).expect("assembles");
        for block in blocks {
            for field in block.fields {
                if !field.name.ends_with("_t") {
                    continue;
                }
                assert!(
                    identifier_uses(&source, field.name) >= 2,
                    "{name} declares {} and never mixes with it",
                    field.name
                );
            }
        }
    }
}
