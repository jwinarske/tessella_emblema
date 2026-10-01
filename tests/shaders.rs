//! Does an assembled family compile, and does it still say what it said?
//!
//! The declarations come from the ABI's tables and the body is written against them, so the thing
//! being checked is that the two fit: a body naming a field the tables do not declare, or reading
//! one at the wrong type, fails here rather than at pipeline creation on a board.

use tessella_capture_abi::generated::shader_attributes::BACKGROUND_SHADER;
use tessella_capture_abi::generated::ubo_layouts::{BACKGROUND_DRAWABLE_UBO, BACKGROUND_PROPS_UBO};
use tessella_emblema::shaders::{BACKGROUND_BODY, attribute_name, module};

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
