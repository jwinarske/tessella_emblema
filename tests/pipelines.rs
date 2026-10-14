// SPDX-License-Identifier: BSD-2-Clause
//! What a pipeline cache must and must not tell apart.
//!
//! Each test is a pair that differs in one thing. The ones that must differ are the silent
//! defects: a cache that conflated them would bind a pipeline whose vertex input reads every
//! vertex from the wrong place, and nothing would fail.

use std::collections::HashMap;

use tessella_capture_abi::envelope::{AttributeDesc, SlabRef};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::shader_attributes::{FILL_SHADER, RASTER_SHADER};
use tessella_emblema::pipelines::{Blend, Depth, Enabled, key};
use tessella_emblema::surface::Surface;
use tessella_emblema::vertices::{Plan, plan, plan_instanced};

fn descs(stride: u32, offset: u32, source: SlabRef) -> Vec<AttributeDesc> {
    RASTER_SHADER
        .iter()
        .map(|entry| AttributeDesc {
            attr_id: entry.attr_id,
            binding: entry.binding,
            source,
            offset,
            vertex_offset: 0,
            stride,
            data_type: entry.declared as u8,
            declared_data_type: entry.declared as u8,
            _pad: [0; 2],
        })
        .collect()
}

const SOMEWHERE: SlabRef = SlabRef {
    slab: 1,
    offset: 0,
    length: 144,
};

fn planned(stride: u32, offset: u32, source: SlabRef) -> Plan {
    plan(&RASTER_SHADER, &descs(stride, offset, source)).expect("agrees")
}

#[test]
fn the_same_drawable_twice_is_one_key() {
    let a = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    assert_eq!(a, b);
}

/// The defect the key exists to prevent: same family, same permutation, different stride.
#[test]
fn a_different_stride_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(16, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    assert_ne!(
        a, b,
        "the binding description differs, so the pipeline does"
    );
}

#[test]
fn a_different_offset_within_a_vertex_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 4, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    assert_ne!(a, b);
}

/// The surface and the permutation each name a different module, so each must separate a key.
///
/// The permutation is not a number a caller chooses: it is which of the family's attributes the
/// drawable sent, so the second key here is built from a drawable that sent one fewer. That is the
/// whole of mbgl's `propertiesAsUniforms` as this crate sees it -- a descriptor that did not
/// arrive -- and a module assembled for it leaves that location out of `In` and reads the layer's
/// evaluated value instead.
#[test]
fn the_surface_and_the_permutation_each_separate_a_key() {
    let base = planned(12, 0, SOMEWHERE);
    let flat = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &base,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    let bent = key(
        BuiltIn::RasterShader as i32,
        Surface::Globe,
        &base,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );

    // The same drawable with its last descriptor missing. Asserted rather than assumed, because a
    // plan that still bound three would make the comparison below vacuous.
    let mut fewer = descs(12, 0, SOMEWHERE);
    let dropped = fewer.pop().expect("raster declares attributes");
    let short = plan(&RASTER_SHADER, &fewer).expect("agrees");
    assert_eq!(
        short.absent,
        vec![u32::try_from(dropped.binding).expect("a declared location")],
        "the short plan must be short by exactly the popped attribute"
    );
    let varied = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &short,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );

    assert_ne!(flat, bent, "a different module");
    assert_ne!(
        flat, varied,
        "a different module, and a different vertex input"
    );
    assert_ne!(
        flat.permutation, varied.permutation,
        "the permutation alone has to separate them: a cache keyed on the vertex layout only \
         would hand this drawable the whole family's module and then not bind one of its inputs"
    );
}

#[test]
fn the_family_separates_a_key() {
    let base = planned(12, 0, SOMEWHERE);
    assert_ne!(
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            &base,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        key(
            BuiltIn::HillshadeShader as i32,
            Surface::Plane,
            &base,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        "raster and hillshade declare the same attributes and are not the same shader"
    );
}

/// Which slab the bytes live in is not pipeline state: buffers bind at draw time.
#[test]
fn where_the_bytes_live_does_not_separate_a_key() {
    let elsewhere = SlabRef {
        slab: 9,
        offset: 512,
        length: 144,
    };
    assert_eq!(
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            &planned(12, 0, SOMEWHERE),
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            &planned(12, 0, elsewhere),
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        "two drawables in different slabs share a pipeline"
    );
}

/// A drawable binding fewer attributes is a different vertex input, so a different pipeline.
#[test]
fn a_shorter_run_is_a_different_pipeline() {
    let full = plan(&FILL_SHADER, &{
        FILL_SHADER
            .iter()
            .map(|entry| AttributeDesc {
                attr_id: entry.attr_id,
                binding: entry.binding,
                source: SOMEWHERE,
                offset: 0,
                vertex_offset: 0,
                stride: 12,
                data_type: entry.declared as u8,
                declared_data_type: entry.declared as u8,
                _pad: [0; 2],
            })
            .collect::<Vec<_>>()
    })
    .expect("agrees");
    let partial = plan(
        &FILL_SHADER,
        &[AttributeDesc {
            attr_id: FILL_SHADER[0].attr_id,
            binding: FILL_SHADER[0].binding,
            source: SOMEWHERE,
            offset: 0,
            vertex_offset: 0,
            stride: 12,
            data_type: FILL_SHADER[0].declared as u8,
            declared_data_type: FILL_SHADER[0].declared as u8,
            _pad: [0; 2],
        }],
    )
    .expect("a partial run is not an error");
    assert_ne!(
        key(
            BuiltIn::FillShader as i32,
            Surface::Plane,
            &full,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        key(
            BuiltIn::FillShader as i32,
            Surface::Plane,
            &partial,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        )
    );
}

/// It has to work as a map key, which is the whole point.
#[test]
fn the_key_caches() {
    let mut cache: HashMap<_, u32> = HashMap::new();
    let flat = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    let strided = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(16, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    cache.insert(flat.clone(), 1);
    cache.insert(strided, 2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&flat), Some(&1));
    // And looking up an equal key built separately finds the same entry.
    let again = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    assert_eq!(cache.get(&again), Some(&1));
}

/// The format comes from the table, so it is constant per family -- but it is in the key because
/// the pipeline carries it, and a family whose declared type changed would be a new pipeline.
#[test]
fn the_layout_records_the_format_the_table_declared() {
    let got = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
    );
    assert_eq!(got.layout.len(), RASTER_SHADER.len());
    for slot in &got.layout {
        assert_eq!(slot.format, ash::vk::Format::R16G16_SINT);
    }
}

/// The rate is pipeline state, so two drawables differing only in it are two pipelines.
///
/// Built by planning one table as the vertex run and then as the instanced run, which is the only
/// thing that differs between the two plans.
#[test]
fn the_input_rate_separates_a_key() {
    let descs = descs(12, 0, SOMEWHERE);
    let per_vertex = plan(&RASTER_SHADER, &descs).expect("agrees");
    let per_instance = plan_instanced(&[], &[], &RASTER_SHADER, &descs).expect("agrees");
    assert_eq!(
        per_vertex.bound.len(),
        per_instance.bound.len(),
        "the same bindings"
    );
    assert_ne!(
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            &per_vertex,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            &per_instance,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL
        ),
        "one advances per vertex and one per instance"
    );
}
