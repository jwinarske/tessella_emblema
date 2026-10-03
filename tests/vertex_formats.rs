//! Does the pipeline's vertex input agree with what the shader declares?
//!
//! A vertex format and a WGSL type are two statements about the same bytes, written in different
//! places: `device::vertex_format` tells Vulkan how to fetch an attribute and
//! `shaders::attribute_type` tells the shader what it received. Where the two disagree nothing
//! errors — both are a legal reading of the same buffer — and the picture is wrong in a way that
//! looks like a maths bug in whatever the attribute fed.
//!
//! So the two are checked against each other for every type any shader in the ABI declares,
//! rather than each being checked against its author's intention.

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_capture_abi::generated::shader_attributes::{attributes, instance_attributes};
use tessella_capture_abi::generated::texture_slots::TABLED;
use tessella_emblema::device::vertex_format;
use tessella_emblema::shaders::attribute_type;

/// Every declared type that reaches a vertex input, across every shader with a table.
fn declared_types() -> Vec<AttributeDataType> {
    let mut seen = Vec::new();
    for shader in TABLED {
        for attribute in attributes(shader).iter().chain(instance_attributes(shader)) {
            if !seen.contains(&attribute.declared) {
                seen.push(attribute.declared);
            }
        }
    }
    seen
}

/// How many components a format carries and what class its scalars are, from what the enum says.
///
/// Read from the format's own name rather than from a table of my own: a table would be a second
/// opinion about what `R16G16_SINT` means, and the name is the statement.
fn shape_of(format: vk::Format) -> (usize, &'static str) {
    let name = format!("{format:?}");
    let (channels, suffix) = name
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("{name} has no suffix"));
    let components = channels.matches(['R', 'G', 'B', 'A']).count();
    let class = match suffix {
        "SINT" => "i32",
        "UINT" => "u32",
        "SFLOAT" => "f32",
        other => panic!("{name} is a {other} format, which is not an integer or a float"),
    };
    (components, class)
}

/// And the same two facts from the WGSL type the shader declares.
fn shape_of_wgsl(declared: &str) -> (usize, &'static str) {
    let class = if declared.contains("i32") {
        "i32"
    } else if declared.contains("u32") {
        "u32"
    } else {
        "f32"
    };
    let components = if let Some(rest) = declared.strip_prefix("vec") {
        rest.as_bytes()[0] as usize - b'0' as usize
    } else {
        1
    };
    (components, class)
}

/// Every type a shader declares has a vertex format, and the two say the same thing.
///
/// Both halves matter. A missing format is a family that cannot build a pipeline at all, which is
/// loud. A format that disagrees with the WGSL type is not: two shorts fetched as two shorts and
/// read as two floats is a legal pipeline whose every vertex is somewhere else.
#[test]
fn every_declared_type_has_a_format_that_agrees_with_it() {
    let mut checked = 0;
    for declared in declared_types() {
        let wgsl = attribute_type(declared)
            .unwrap_or_else(|| panic!("{declared:?} is declared and has no WGSL type"));
        let format = vertex_format(declared)
            .unwrap_or_else(|| panic!("{declared:?} is declared and has no vertex format"));
        assert_eq!(
            shape_of(format),
            shape_of_wgsl(wgsl),
            "{declared:?} fetches as {format:?} and reads as {wgsl}"
        );
        checked += 1;
    }
    assert!(checked >= 10, "only {checked} types were checked");
}

/// An integer type never maps to a normalized format.
///
/// The failure is silent and total: a `Short2` tile position fetched as `R16G16_SNORM` arrives
/// divided by 32,767, so the whole tile lands inside one pixel at the origin. Both formats are
/// two shorts, so nothing in Vulkan objects. mbgl's line shader reads `a_data` as raw bytes and
/// subtracts 128 from them, which only works unnormalized.
#[test]
fn no_integer_type_normalizes() {
    for declared in declared_types() {
        let format = vertex_format(declared).expect("a declared type has a format");
        let name = format!("{format:?}");
        assert!(
            !name.ends_with("NORM") && !name.ends_with("SCALED"),
            "{declared:?} fetches as {name}, which rescales the value"
        );
    }
}

/// The two types a shader can be handed but no vertex input can carry have no format.
///
/// Recorded rather than mapped to something plausible. `UShort8` is eight shorts, which is two
/// vertex attributes and not one, and `Invalid` is the enum's absence value — a format for either
/// would be an invention this crate then depended on.
#[test]
fn the_types_without_a_format_are_the_two_that_cannot_have_one() {
    assert_eq!(vertex_format(AttributeDataType::UShort8), None);
    assert_eq!(vertex_format(AttributeDataType::Invalid), None);
    // And neither is reachable from a table, which is why their absence costs nothing.
    assert!(!declared_types().contains(&AttributeDataType::UShort8));
    assert!(!declared_types().contains(&AttributeDataType::Invalid));
}
