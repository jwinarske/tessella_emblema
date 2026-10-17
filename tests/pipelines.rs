// SPDX-License-Identifier: BSD-2-Clause
//! What a pipeline cache must and must not tell apart.
//!
//! Each test is a pair that differs in one thing. The ones that must differ are the silent
//! defects: a cache that conflated them would bind a pipeline whose vertex input reads every
//! vertex from the wrong place, and nothing would fail.

use std::collections::HashMap;

use tessella_capture_abi::envelope::Topology;
use tessella_capture_abi::envelope::{AttributeDesc, SlabRef};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::generated::shader_attributes::{FILL_SHADER, RASTER_SHADER};
use tessella_emblema::pipelines::{Blend, Depth, Enabled, Stages, Stencil, key, stage_infos};
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
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    assert_eq!(a, b);
}

/// The defect the key exists to prevent: same family, same permutation, different stride.
#[test]
fn a_different_stride_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(16, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    assert_ne!(
        a, b,
        "the binding description differs, so the pipeline does"
    );
}

/// One family's two topologies are two pipelines, because the primitive is baked into the
/// pipeline rather than set per draw.
///
/// The case the wire carries a topology for: the location indicator's accuracy circle is one
/// family drawing a triangle fan and a line strip over one vertex buffer. A cache keyed without
/// this hands the second drawable the first's pipeline and draws its indices as the other
/// primitive -- which for a fill outline's pairs is a band along every ring instead of a line, and
/// for a fan read as pairs is every other edge.
#[test]
fn a_different_topology_is_a_different_pipeline() {
    let of = |topology| {
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            topology,
            &planned(12, 0, SOMEWHERE),
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        )
    };
    // By the key's own equality, which is what the cache's `HashMap` looks it up by. Pairwise
    // rather than through a set, because `Key` is `Eq` and `Hash` and deliberately not `Ord`.
    let keys: Vec<_> = Topology::ALL.iter().map(|at| (at, of(*at))).collect();
    for (one, left) in &keys {
        for (other, right) in &keys {
            assert_eq!(
                left == right,
                one == other,
                "{one:?} and {other:?} key the same"
            );
        }
    }
    assert_eq!(keys.len(), 3, "the ABI names three topologies");
}

#[test]
fn a_different_offset_within_a_vertex_is_a_different_pipeline() {
    let a = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    let b = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(12, 4, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
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
        Topology::Triangles,
        &base,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    let bent = key(
        BuiltIn::RasterShader as i32,
        Surface::Globe,
        Topology::Triangles,
        &base,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
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
        Topology::Triangles,
        &short,
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
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
            Topology::Triangles,
            &base,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        ),
        key(
            BuiltIn::HillshadeShader as i32,
            Surface::Plane,
            Topology::Triangles,
            &base,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
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
            Topology::Triangles,
            &planned(12, 0, SOMEWHERE),
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        ),
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            Topology::Triangles,
            &planned(12, 0, elsewhere),
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
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
            Topology::Triangles,
            &full,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        ),
        key(
            BuiltIn::FillShader as i32,
            Surface::Plane,
            Topology::Triangles,
            &partial,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
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
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    let strided = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(16, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
    );
    cache.insert(flat.clone(), 1);
    cache.insert(strided, 2);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&flat), Some(&1));
    // And looking up an equal key built separately finds the same entry.
    let again = key(
        BuiltIn::RasterShader as i32,
        Surface::Plane,
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
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
        Topology::Triangles,
        &planned(12, 0, SOMEWHERE),
        Blend::Alpha,
        Depth::Write,
        Enabled::ALL,
        Stencil::Clip,
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
            Topology::Triangles,
            &per_vertex,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        ),
        key(
            BuiltIn::RasterShader as i32,
            Surface::Plane,
            Topology::Triangles,
            &per_instance,
            Blend::Alpha,
            Depth::Write,
            Enabled::ALL,
            Stencil::Clip,
        ),
        "one advances per vertex and one per instance"
    );
}

/// Each stage names its own module, and the right one.
///
/// A pipeline whose two stages name one module builds nowhere on the sa8155p -- #85 -- and the
/// arrangement that gets past it is a handle per stage. Nothing in a frame can show which handle
/// a stage was given: every other device takes either, so a builder that named the vertex module
/// twice would draw correctly here and build no pipeline at all there.
///
/// So the check is on the stage descriptions rather than on a pipeline, from two handles invented
/// for the purpose -- which is also why it needs no device.
#[test]
fn each_stage_names_its_own_module() {
    use ash::vk::Handle as _;

    let stages = Stages {
        vertex: ash::vk::ShaderModule::from_raw(0x5e1f),
        fragment: ash::vk::ShaderModule::from_raw(0xf00d),
    };
    let infos = stage_infos(stages);

    assert_eq!(infos[0].stage, ash::vk::ShaderStageFlags::VERTEX);
    assert_eq!(
        infos[0].module.as_raw(),
        0x5e1f,
        "the vertex stage was handed the other module"
    );
    assert_eq!(infos[1].stage, ash::vk::ShaderStageFlags::FRAGMENT);
    assert_eq!(
        infos[1].module.as_raw(),
        0xf00d,
        "the fragment stage was handed the other module"
    );
    assert_ne!(
        infos[0].module, infos[1].module,
        "both stages named one module, which is the defect itself"
    );
}
