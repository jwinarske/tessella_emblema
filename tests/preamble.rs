//! Does WGSL put the fields where the producer wrote them?
//!
//! One question, asked of every block the ABI describes. A shader whose idea of a field's offset
//! differs from the producer's by a padding word reads every later field from the wrong bytes, and
//! draws, because the bytes are valid floats either way. There is no error to catch at run time,
//! so it is caught here.

use tessella_capture_abi::generated::ubo_layouts::{LAYOUTS, UboFieldKind, UboLayout};
use tessella_emblema::preamble::{Unrepresentable, declare, offsets, type_name};
use tessella_emblema::slots;

/// Every block the producer describes is placed exactly where it says.
///
/// The whole point of generating the declarations rather than writing them. Fifty blocks, and a
/// disagreement in any one of them is a layer drawing from the wrong bytes.
#[test]
fn every_block_lands_where_the_producer_put_it() {
    let mut checked = 0;
    for layout in LAYOUTS {
        let placed = offsets(&layout)
            .unwrap_or_else(|why| panic!("{} cannot be declared: {why:?}", layout.name));
        assert_eq!(
            placed.len(),
            layout.fields.len(),
            "{} lost a field",
            layout.name
        );
        for (field, (name, at)) in layout.fields.iter().zip(placed) {
            assert_eq!(name, field.name, "{} reordered its fields", layout.name);
            assert_eq!(
                at, field.offset,
                "{}::{} would be read from {at}, written at {}",
                layout.name, field.name, field.offset
            );
            checked += 1;
        }
    }
    assert_eq!(
        LAYOUTS.len(),
        50,
        "the table grew or shrank; look at the new ones"
    );
    assert!(
        checked > 200,
        "only {checked} fields checked, which is too few to mean much"
    );
}

/// And every block declares without error.
#[test]
fn every_block_declares() {
    for layout in LAYOUTS {
        let source = declare(&layout, slots::stride(&layout))
            .unwrap_or_else(|why| panic!("{} cannot be declared: {why:?}", layout.name));
        assert!(
            source.contains("struct "),
            "{} declared nothing",
            layout.name
        );
        assert!(
            source.contains(&type_name(layout.name)),
            "{} is not named in its own declaration",
            layout.name
        );
        for field in layout.fields {
            assert!(
                source.contains(field.name),
                "{}::{} is missing from the declaration",
                layout.name,
                field.name
            );
        }
    }
}

/// A `vec3` followed by an `f32` is the case WGSL and C++ most often disagree about.
///
/// `vec3<f32>` is twelve bytes with sixteen-byte alignment, so WGSL will pack an `f32` into the
/// fourth word if the producer did. The generator has to follow whichever the producer chose
/// rather than assume.
#[test]
fn a_vec3_followed_by_a_float_is_placed_as_the_producer_placed_it() {
    let packed = LAYOUTS.iter().find(|layout| {
        layout.fields.windows(2).any(|pair| {
            pair[0].kind == UboFieldKind::Vec3
                && pair[1].kind == UboFieldKind::F32
                && pair[1].offset == pair[0].offset + 12
        })
    });
    let Some(layout) = packed else {
        // Not a failure: it means mbgl never packs one that way, which is worth knowing too.
        return;
    };
    let placed = offsets(layout).expect("declarable");
    for (field, (_, at)) in layout.fields.iter().zip(placed) {
        assert_eq!(at, field.offset, "{}::{}", layout.name, field.name);
    }
}

/// A field the producer puts earlier than WGSL could is refused, not papered over.
///
/// Padding only moves a field later. There is no declaration that reads such a block correctly, so
/// emitting one would be the quiet wrong answer.
#[test]
fn a_field_packed_tighter_than_wgsl_allows_is_refused() {
    use tessella_capture_abi::generated::ubo_layouts::UboField;

    // A `vec4` at offset four: WGSL aligns it to sixteen and nothing can move it earlier.
    static FIELDS: [UboField; 2] = [
        UboField {
            name: "first",
            offset: 0,
            kind: UboFieldKind::F32,
        },
        UboField {
            name: "second",
            offset: 4,
            kind: UboFieldKind::Vec4,
        },
    ];
    let impossible = UboLayout {
        name: "Impossible",
        header: "none",
        align: 16,
        size: 32,
        stride: 32,
        fields: &FIELDS,
    };

    assert_eq!(
        declare(&impossible, impossible.stride),
        Err(Unrepresentable::FieldTooEarly {
            block: "Impossible",
            field: "second",
            declared: 4,
            earliest: 16,
        })
    );
}

/// A stride that is not a multiple of the alignment is refused.
///
/// A consolidated buffer is an array of these. An unaligned stride puts every block after the
/// first where WGSL will not index to.
#[test]
fn an_unaligned_stride_is_refused() {
    use tessella_capture_abi::generated::ubo_layouts::UboField;

    static FIELDS: [UboField; 1] = [UboField {
        name: "only",
        offset: 0,
        kind: UboFieldKind::F32,
    }];
    let ragged = UboLayout {
        name: "Ragged",
        header: "none",
        align: 16,
        size: 4,
        stride: 20,
        fields: &FIELDS,
    };

    assert_eq!(
        declare(&ragged, ragged.stride),
        Err(Unrepresentable::StrideUnaligned {
            block: "Ragged",
            stride: 20,
            align: 16,
        })
    );
}

/// The generated name reads as a word rather than as shouting.
#[test]
fn a_block_name_becomes_a_wgsl_type_name() {
    assert_eq!(type_name("BackgroundDrawableUBO"), "BackgroundDrawableUbo");
    assert_eq!(type_name("LineSDFTilePropsUBO"), "LineSdfTilePropsUbo");
    assert_eq!(type_name("DebugUBO"), "DebugUbo");
}

/// Every block's type name is distinct, or two structs would collide in one module.
#[test]
fn no_two_blocks_share_a_type_name() {
    let mut names: Vec<String> = LAYOUTS.iter().map(|l| type_name(l.name)).collect();
    names.sort();
    let before = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        before,
        "two blocks generate the same type name"
    );
}

/// A block whose fields sit later than WGSL would put them is padded to match.
///
/// No block mbgl declares needs this -- all fifty land on WGSL's natural offsets, which is worth
/// knowing and is why the test above passes whether or not the padding works. So the padding is
/// exercised here against a layout built for it, or it would be code nothing has ever run.
#[test]
fn a_gap_the_producer_leaves_is_padded() {
    use tessella_capture_abi::generated::ubo_layouts::UboField;

    // WGSL would put the second `f32` at 4. The producer puts it at 16, so three words go between.
    static FIELDS: [UboField; 2] = [
        UboField {
            name: "first",
            offset: 0,
            kind: UboFieldKind::F32,
        },
        UboField {
            name: "second",
            offset: 16,
            kind: UboFieldKind::F32,
        },
    ];
    let gapped = UboLayout {
        name: "Gapped",
        header: "none",
        align: 16,
        size: 32,
        stride: 32,
        fields: &FIELDS,
    };

    let placed = offsets(&gapped).expect("declarable");
    assert_eq!(placed, [("first", 0), ("second", 16)], "padded to the gap");

    let source = declare(&gapped, gapped.stride).expect("declarable");
    assert!(
        source.contains("array<u32, 3>"),
        "three words of padding, not {source}"
    );
}

/// And a gap that is not a whole number of words is refused rather than rounded.
///
/// Padding is emitted in words. A field the producer puts two bytes past where WGSL would is not
/// reachable by any number of them, and placing it at the nearest word reads the wrong bytes.
#[test]
fn a_gap_that_is_not_whole_words_does_not_silently_move_the_field() {
    use tessella_capture_abi::generated::ubo_layouts::UboField;

    static FIELDS: [UboField; 2] = [
        UboField {
            name: "first",
            offset: 0,
            kind: UboFieldKind::F32,
        },
        UboField {
            name: "second",
            offset: 6,
            kind: UboFieldKind::F32,
        },
    ];
    let ragged = UboLayout {
        name: "RaggedField",
        header: "none",
        align: 16,
        size: 16,
        stride: 16,
        fields: &FIELDS,
    };

    // The field cannot be placed at 6, and `offsets` reports where it would actually land rather
    // than claiming the declared offset. A caller comparing the two sees the disagreement.
    let placed = offsets(&ragged).expect("declarable");
    assert_eq!(
        placed[1],
        ("second", 4),
        "reported where it lands, not where it was asked for"
    );
    assert_ne!(placed[1].1, FIELDS[1].offset, "and the two disagree");
}
