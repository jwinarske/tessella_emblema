// SPDX-License-Identifier: BSD-2-Clause
//! The paint-property table against the generated tables it has to agree with.
//!
//! `Property` writes both of a property's arms out by hand, which is the only way to express the
//! symbol families' `is_text ? text_x : icon_x` -- so what no derivation produced has to be
//! checked instead. These tests are that check, and the one that matters most is the pair at the
//! top: a data-driven attribute with no entry here, and a body that reads one directly.
//!
//! Neither of those fails loudly on its own. A missing entry is a body that will not compile the
//! moment that attribute does not arrive, which is a defect nothing short of a real frame would
//! reach; a body reading `mix_color(in.x, ...)` instead of `paint_x(in)` is a module that compiles
//! and draws until the same moment.

use std::collections::BTreeSet;

use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_capture_abi::generated::shader_attributes::ShaderAttribute;
use tessella_emblema::families::{ALL, Family};
use tessella_emblema::shaders::{attribute_name, permutation};

/// Every attribute of a family, per-vertex then per-instance, which is the order `In` declares.
fn attributes(family: &Family) -> Vec<&ShaderAttribute> {
    family
        .attributes
        .iter()
        .chain(family.instance_attributes)
        .collect()
}

/// Whether an attribute's declared type is the shape a data-driven paint property arrives in.
///
/// mbgl sends such a property as its value at *two* zoom levels, to be mixed by the `_t` in the
/// drawable block: a scalar becomes `Float2` and a color becomes `Float4`, four floats holding two
/// colors with each color's channels folded in pairs. So the declared type is what says a property
/// is data-driven, and it says it for every family without a list -- which is why this is the test
/// and not a table.
///
/// Nothing else in any of mbgl's thirty-six tables declares either type. Geometry is `Short2`,
/// `Short4` or `UShort2`, a line's packed data is `UByte4`, a pattern's corners are `UShort4`, a
/// symbol's projected position is `Float3` and its fade is a bare `Float`.
fn is_paint(attribute: &ShaderAttribute) -> bool {
    matches!(
        attribute.declared,
        AttributeDataType::Float2 | AttributeDataType::Float4
    )
}

/// The WGSL type the two arms of such an attribute return.
fn returns(attribute: &ShaderAttribute) -> &'static str {
    match attribute.declared {
        AttributeDataType::Float4 => "vec4<f32>",
        _ => "f32",
    }
}

/// Every data-driven attribute has an entry, and nothing else does.
///
/// Both directions, because each is a different defect. An attribute with no entry is a family
/// whose body cannot be assembled once the producer stops sending it -- and the producer stops
/// sending it whenever the style's own value is constant, which is the ordinary case rather than
/// an edge. An entry at a location that is not data-driven is a `paint_*` function reading two
/// zoom endpoints out of bytes that are not a pair.
#[test]
fn every_data_driven_attribute_is_a_property_and_nothing_else_is() {
    for family in ALL {
        let declared: BTreeSet<u32> = attributes(family)
            .iter()
            .filter(|attribute| is_paint(attribute))
            .filter_map(|attribute| u32::try_from(attribute.binding).ok())
            .collect();
        let listed: BTreeSet<u32> = family
            .properties
            .iter()
            .map(|property| property.location)
            .collect();
        assert_eq!(
            declared, listed,
            "{}: the data-driven attributes are at {declared:?} and the table lists {listed:?}",
            family.name
        );
    }
}

/// Every property's body call is in the body, and no body reads a paint attribute directly.
///
/// The second half is the one that would go wrong by accident: the obvious way to read a
/// data-driven property is `mix_color(in.fill_color, drawable.color_t)`, which is what every body
/// here said before the permutation existed and what the next family's author would write. It
/// compiles, and it keeps compiling until a drawable arrives without that attribute.
#[test]
fn every_body_reads_its_properties_through_their_functions() {
    for family in ALL {
        for property in family.properties {
            let call = format!("paint_{}(in)", property.name);
            assert!(
                family.body.contains(&call),
                "{}: nothing in the body calls {call}",
                family.name
            );
        }
        for direct in ["mix_color(in.", "mix_value(in."] {
            assert!(
                !family.body.contains(direct),
                "{}: the body reads an attribute through {direct}, which bypasses the permutation",
                family.name
            );
        }
    }
}

/// Each arm names what it says it names.
///
/// The varying arm must mix *that* attribute with *that* `_t`, and the constant arm must read a
/// field of a block the family declares. Checked as text, which is what the arms are: a typo in
/// either is a module that does not compile, so this is a faster failure rather than a different
/// one -- except for the one case it is not, which is an arm naming the *wrong* attribute of the
/// same type. That compiles and draws.
#[test]
fn each_arm_names_its_own_attribute_and_its_own_block() {
    for family in ALL {
        // What the module declares, read off the module: the generator decides these names and
        // the arms are text, so the one thing worth comparing them against is the text itself.
        let surface = *family.surfaces.first().expect("a family has a surface");
        let text = family
            .module(surface, &[])
            .unwrap_or_else(|why| panic!("{} does not assemble: {why:?}", family.name));
        let bindings: BTreeSet<&str> = text
            .lines()
            .filter_map(|line| line.split("var<storage, read> ").nth(1))
            .filter_map(|tail| tail.split(':').next())
            .collect();
        for property in family.properties {
            let attribute = attributes(family)
                .into_iter()
                .find(|attribute| {
                    u32::try_from(attribute.binding).is_ok_and(|at| at == property.location)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{}: no attribute at location {}",
                        family.name, property.location
                    )
                });
            let named = format!("in.{}", attribute_name(attribute.name));
            assert!(
                property.varies.contains(&named),
                "{} {}: the varying arm does not read {named}",
                family.name,
                property.name
            );
            let interpolation = format!(".{}_t", property.name);
            assert!(
                property.varies.contains(&interpolation),
                "{} {}: the varying arm does not mix by {interpolation}",
                family.name,
                property.name
            );
            assert_eq!(
                property.returns,
                returns(attribute),
                "{} {}: the arms return the wrong type for a {:?}",
                family.name,
                property.name,
                attribute.declared
            );

            // Every field the constant arm reads is this property's own, under one of the three
            // spellings mbgl uses for it. This is the check for the defect no compiler sees: a
            // fill whose constant arm reads `props.fade` instead of `props.opacity` is the right
            // type out of the right block, and it draws.
            //
            // `is_text` is the one field that is not a value: it is the tile flag the symbol
            // families select on, and it is named here rather than allowed by shape so that a
            // fourth spelling cannot arrive unnoticed.
            let mine: Vec<String> = ["", "text_", "icon_"]
                .iter()
                .map(|prefix| format!("{prefix}{}", property.name))
                .collect();
            let mut named = 0;
            for field in property.constant.split('.').skip(1).map(|tail| {
                tail.chars()
                    .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                    .collect::<String>()
            }) {
                if mine.contains(&field) {
                    named += 1;
                    continue;
                }
                assert_eq!(
                    field, "is_text",
                    "{} {}: the constant arm reads `{field}`, which is not this property",
                    family.name, property.name
                );
            }
            assert!(
                named > 0,
                "{} {}: the constant arm reads no field named for the property",
                family.name,
                property.name
            );

            // Every array the constant arm indexes is a binding the module declares. Checked
            // against the assembled text rather than against the family's block list, because the
            // text is what has to agree -- a symbol property reads two blocks, one for the value
            // and one for the tile's `is_text`, and only one of them is where the value lives.
            for indexed in property
                .constant
                .split('[')
                .take(property.constant.matches('[').count())
                .map(|head| {
                    head.rsplit(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                        .next()
                        .unwrap_or_default()
                })
            {
                assert!(
                    bindings.contains(indexed),
                    "{} {}: the constant arm indexes `{indexed}`, which the module does not \
                     declare",
                    family.name,
                    property.name
                );
            }
        }
    }
}

/// The interpolation each varying arm mixes by is a field of a block the family declares.
///
/// The other half of the arm check, from the block's side: `_t` is what makes a property
/// data-driven at all, and a family whose drawable block has no such field has no business
/// listing one.
#[test]
fn every_interpolation_is_a_field_of_a_declared_block() {
    for family in ALL {
        for property in family.properties {
            let wanted = format!("{}_t", property.name);
            assert!(
                family
                    .blocks
                    .iter()
                    .any(|layout| { layout.fields.iter().any(|field| field.name == wanted) }),
                "{}: no block it declares carries `{wanted}`",
                family.name
            );
        }
    }
}

/// Two properties of one family cannot share a name, because the functions would.
#[test]
fn a_family_names_each_property_once() {
    for family in ALL {
        let mut seen = BTreeSet::new();
        for property in family.properties {
            assert!(
                seen.insert(property.name),
                "{}: two properties named {}",
                family.name,
                property.name
            );
        }
    }
}

/// The bitmask is the set, which is what the module cache depends on.
///
/// Order must not matter and repetition must not either -- `Plan::absent` is built by walking the
/// table, so it is sorted today, and a cache that depended on that would miss the first time
/// anything reordered it.
#[test]
fn the_permutation_is_the_set_and_not_the_order() {
    assert_eq!(permutation(&[]), 0, "nothing absent is the whole family");
    assert_eq!(permutation(&[0]), 1);
    assert_eq!(permutation(&[1, 2]), 0b110);
    assert_eq!(
        permutation(&[2, 1]),
        permutation(&[1, 2]),
        "the order a plan happens to list them in is not part of the key"
    );
    assert_eq!(
        permutation(&[1, 1, 2]),
        permutation(&[1, 2]),
        "a location twice is the same location"
    );
    assert_ne!(
        permutation(&[1]),
        permutation(&[2]),
        "two single absences must not collide"
    );
}
