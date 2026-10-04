// SPDX-License-Identifier: BSD-2-Clause
//! What a drawable's attribute descriptors come to against a real family's table.
//!
//! Every case uses a generated table rather than a hand-written one, so a table that changes
//! shape under this crate fails here instead of being accommodated.

use ash::vk;
use tessella_capture_abi::AttributeDataType;
use tessella_capture_abi::envelope::{AttributeDesc, SlabRef};
use tessella_capture_abi::generated::shader_attributes::{
    COLOR_RELIEF_SHADER, FILL_SHADER, RASTER_SHADER,
};
use tessella_emblema::vertices::{Refused, plan};

/// A descriptor agreeing with a table entry, which is the shape the producer sends.
fn agreeing(
    entry: &tessella_capture_abi::generated::shader_attributes::ShaderAttribute,
) -> AttributeDesc {
    AttributeDesc {
        attr_id: entry.attr_id,
        binding: entry.binding,
        source: SlabRef {
            slab: 1,
            offset: 0,
            length: 64,
        },
        offset: 0,
        vertex_offset: 0,
        stride: 12,
        data_type: entry.declared as u8,
        declared_data_type: entry.declared as u8,
        _pad: [0; 2],
    }
}

#[test]
fn a_full_agreement_binds_every_slot() {
    let descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    let got = plan(&RASTER_SHADER, &descs).expect("agrees");
    assert_eq!(got.bound.len(), RASTER_SHADER.len());
    assert!(got.dropped.is_empty(), "nothing was bound at -1");
    assert!(got.absent.is_empty(), "every slot was supplied");
    // The format is the table's declared type, not the wire's.
    assert_eq!(got.bound[0].format, vk::Format::R16G16_SINT);
    assert_eq!(got.bound[1].format, vk::Format::R16G16_SINT);
    // And in slot order, whatever order the descriptors arrived in.
    assert_eq!(got.bound[0].slot, 0);
    assert_eq!(got.bound[1].slot, 1);
}

#[test]
fn the_plan_is_sorted_by_slot_however_the_wire_ordered_it() {
    let mut descs: Vec<AttributeDesc> = FILL_SHADER.iter().map(agreeing).collect();
    descs.reverse();
    let got = plan(&FILL_SHADER, &descs).expect("agrees");
    let slots: Vec<u32> = got.bound.iter().map(|bound| bound.slot).collect();
    let mut sorted = slots.clone();
    sorted.sort_unstable();
    assert_eq!(slots, sorted, "a pipeline reads these by location");
}

/// The ABI's own rule: `-1` is an override the shader does not declare, and it is dropped.
///
/// `buildAttributeBindings` in the mbgl backends does the same, and the `LineShader` floor-width
/// override is the case that exists in practice.
#[test]
fn a_minus_one_binding_is_dropped_and_named() {
    let mut descs: Vec<AttributeDesc> = FILL_SHADER.iter().map(agreeing).collect();
    let overridden = descs[1].attr_id;
    descs[1].binding = -1;
    let got = plan(&FILL_SHADER, &descs).expect("a -1 is not an error");
    assert_eq!(got.dropped, vec![overridden]);
    assert_eq!(got.bound.len(), FILL_SHADER.len() - 1);
    assert_eq!(
        got.absent,
        vec![u32::try_from(FILL_SHADER[1].binding).unwrap()],
        "the slot it would have filled is reported unbound"
    );
}

/// A data-driven attribute whose paint is constant sends no descriptor at all.
#[test]
fn a_slot_nobody_supplied_is_absent_rather_than_an_error() {
    let descs = vec![agreeing(&FILL_SHADER[0])];
    let got = plan(&FILL_SHADER, &descs).expect("a partial run is not an error");
    assert_eq!(got.bound.len(), 1);
    let want: Vec<u32> = FILL_SHADER[1..]
        .iter()
        .map(|entry| u32::try_from(entry.binding).unwrap())
        .collect();
    assert_eq!(got.absent, want);
}

/// The skirt flag. `encode_color_relief` sends three descriptors and the table declares two.
///
/// Reported, not refused and not silently dropped. This crate has no per-layer skirt by design —
/// `TERRAIN_PLACEMENT` says only the ground has one — so no module here will declare that
/// attribute and refusing it would make raster and color relief undrawable. `tessella_fluorite`
/// does read it, as `custom1`, which is why it is named rather than discarded.
#[test]
fn a_slot_the_table_does_not_declare_is_reported() {
    let mut descs: Vec<AttributeDesc> = COLOR_RELIEF_SHADER.iter().map(agreeing).collect();
    let mut skirt = agreeing(&COLOR_RELIEF_SHADER[1]);
    skirt.attr_id = 2;
    skirt.binding = 2;
    skirt.offset = 8;
    descs.push(skirt);
    let got = plan(&COLOR_RELIEF_SHADER, &descs).expect("an extra slot is not fatal");
    assert_eq!(got.undeclared, vec![(2, 2)]);
    assert_eq!(
        got.bound.len(),
        COLOR_RELIEF_SHADER.len(),
        "the declared attributes still bind"
    );
    assert_eq!(got.absent, [] as [u32; 0]);
}

/// The whole of raster's run plans, which is the family the refusal would have broken.
#[test]
fn a_raster_bucket_with_its_skirt_still_binds_both_declared_attributes() {
    let mut descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    let mut skirt = agreeing(&RASTER_SHADER[1]);
    skirt.attr_id = 2;
    skirt.binding = 2;
    skirt.offset = 8;
    descs.push(skirt);
    let got = plan(&RASTER_SHADER, &descs).expect("raster plans");
    assert_eq!(got.bound.len(), 2);
    assert_eq!(got.undeclared, vec![(2, 2)]);
}

#[test]
fn a_declared_type_that_disagrees_with_the_table_is_refused() {
    let mut descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    descs[1].declared_data_type = AttributeDataType::Float2 as u8;
    assert_eq!(
        plan(&RASTER_SHADER, &descs),
        Err(Refused::DeclaredDisagrees {
            attr_id: RASTER_SHADER[1].attr_id,
            wire: AttributeDataType::Float2,
            table: RASTER_SHADER[1].declared,
        })
    );
}

/// The buffer's own type may differ from the declared one, and that is not an error.
///
/// mbgl states a type twice and the two disagree for six attributes; the ABI says to bind the
/// declared one. A consumer that refused the difference would refuse the producer's own output.
#[test]
fn a_wire_data_type_that_differs_from_the_declared_one_is_fine() {
    let mut descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    descs[1].data_type = AttributeDataType::Float3 as u8;
    let got = plan(&RASTER_SHADER, &descs).expect("only the declared type is bound");
    assert_eq!(got.bound[1].format, vk::Format::R16G16_SINT);
}

#[test]
fn an_undecodable_discriminant_is_refused_rather_than_trusted() {
    let mut descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    descs[0].declared_data_type = 200;
    assert_eq!(
        plan(&RASTER_SHADER, &descs),
        Err(Refused::BadDataType {
            attr_id: RASTER_SHADER[0].attr_id,
            raw: 200
        })
    );
}

#[test]
fn two_descriptors_for_one_slot_are_refused() {
    let mut descs: Vec<AttributeDesc> = RASTER_SHADER.iter().map(agreeing).collect();
    descs[1].binding = descs[0].binding;
    assert_eq!(
        plan(&RASTER_SHADER, &descs),
        Err(Refused::DuplicateSlot {
            slot: RASTER_SHADER[0].binding
        })
    );
}

/// The offsets and strides the producer sent travel through untouched.
#[test]
fn the_wire_decides_where_the_bytes_are() {
    let mut desc = agreeing(&RASTER_SHADER[0]);
    desc.offset = 4;
    desc.stride = 20;
    desc.vertex_offset = 7;
    desc.source = SlabRef {
        slab: 3,
        offset: 128,
        length: 400,
    };
    let got = plan(&RASTER_SHADER, &[desc]).expect("agrees");
    assert_eq!(got.bound[0].offset, 4);
    assert_eq!(got.bound[0].stride, 20);
    assert_eq!(got.bound[0].vertex_offset, 7);
    assert_eq!(got.bound[0].source.slab, 3);
    assert_eq!(got.bound[0].source.offset, 128);
}
