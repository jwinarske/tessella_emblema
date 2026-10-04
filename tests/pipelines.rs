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
use tessella_emblema::pipelines::key;
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
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    let b = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    assert_eq!(a, b);
}

/// The defect the key exists to prevent: same family, same permutation, different stride.
#[test]
fn a_different_stride_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    let b = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(16, 0, SOMEWHERE),
    );
    assert_ne!(
        a, b,
        "the binding description differs, so the pipeline does"
    );
}

#[test]
fn a_different_offset_within_a_vertex_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    let b = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 4, SOMEWHERE),
    );
    assert_ne!(a, b);
}

#[test]
fn the_surface_and_the_permutation_each_separate_a_key() {
    let base = planned(12, 0, SOMEWHERE);
    let flat = key(BuiltIn::RasterShader, Surface::Plane, 0, &base);
    let bent = key(BuiltIn::RasterShader, Surface::Globe, 0, &base);
    let varied = key(BuiltIn::RasterShader, Surface::Plane, 1, &base);
    assert_ne!(flat, bent, "a different module");
    assert_ne!(flat, varied, "different specialization constants");
}

#[test]
fn the_family_separates_a_key() {
    let base = planned(12, 0, SOMEWHERE);
    assert_ne!(
        key(BuiltIn::RasterShader, Surface::Plane, 0, &base),
        key(BuiltIn::HillshadeShader, Surface::Plane, 0, &base),
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
            BuiltIn::RasterShader,
            Surface::Plane,
            0,
            &planned(12, 0, SOMEWHERE)
        ),
        key(
            BuiltIn::RasterShader,
            Surface::Plane,
            0,
            &planned(12, 0, elsewhere)
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
        key(BuiltIn::FillShader, Surface::Plane, 0, &full),
        key(BuiltIn::FillShader, Surface::Plane, 0, &partial)
    );
}

/// It has to work as a map key, which is the whole point.
#[test]
fn the_key_caches() {
    let mut cache: HashMap<_, u32> = HashMap::new();
    let flat = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    let strided = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(16, 0, SOMEWHERE),
    );
    cache.insert(flat.clone(), 1);
    cache.insert(strided, 2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&flat), Some(&1));
    // And looking up an equal key built separately finds the same entry.
    let again = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
    );
    assert_eq!(cache.get(&again), Some(&1));
}

/// The format comes from the table, so it is constant per family -- but it is in the key because
/// the pipeline carries it, and a family whose declared type changed would be a new pipeline.
#[test]
fn the_layout_records_the_format_the_table_declared() {
    let got = key(
        BuiltIn::RasterShader,
        Surface::Plane,
        0,
        &planned(12, 0, SOMEWHERE),
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
        key(BuiltIn::RasterShader, Surface::Plane, 0, &per_vertex),
        key(BuiltIn::RasterShader, Surface::Plane, 0, &per_instance),
        "one advances per vertex and one per instance"
    );
}
