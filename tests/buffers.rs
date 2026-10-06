// SPDX-License-Identifier: BSD-2-Clause
//! What a planned drawable needs on the device.
//!
//! The case that matters is the interleaved one: the three raster-family encoders each send
//! several descriptors pointing at one buffer, and a store that took them at face value would hold
//! a copy per attribute.

use tessella_capture_abi::envelope::{AttributeDesc, SlabRef};
use tessella_capture_abi::generated::shader_attributes::{
    COLOR_RELIEF_SHADER, FILL_SHADER, RASTER_SHADER, ShaderAttribute,
};
use tessella_emblema::buffers::{needs, vertex_bytes};
use tessella_emblema::vertices::plan;

const INTERLEAVED: SlabRef = SlabRef {
    slab: 1,
    offset: 0,
    length: 144,
};

const INDEXES: SlabRef = SlabRef {
    slab: 1,
    offset: 144,
    length: 36,
};

/// A descriptor agreeing with its table entry, reading from `source` at `offset` within a vertex.
fn from(entry: &ShaderAttribute, source: SlabRef, offset: u32) -> AttributeDesc {
    AttributeDesc {
        attr_id: entry.attr_id,
        binding: entry.binding,
        source,
        offset,
        vertex_offset: 0,
        stride: 12,
        data_type: entry.declared as u8,
        declared_data_type: entry.declared as u8,
        _pad: [0; 2],
    }
}

/// Raster's two declared attributes share one interleaved buffer, as the producer sends them.
#[test]
fn attributes_sharing_one_buffer_need_it_once() {
    let descs = [
        from(&RASTER_SHADER[0], INTERLEAVED, 0),
        from(&RASTER_SHADER[1], INTERLEAVED, 4),
    ];
    let planned = plan(&RASTER_SHADER, &descs).expect("agrees");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices, vec![INTERLEAVED], "one buffer, not two");
    assert_eq!(got.reads.len(), 2);
    assert_eq!(got.reads[0].buffer, 0);
    assert_eq!(got.reads[1].buffer, 0, "both read the same one");
    assert_eq!(vertex_bytes(&got), 144, "counted once");
}

/// And the skirt shares it too, so color relief's whole run of three is one buffer.
///
/// It binds rather than being reported undeclared, which is what tessella#331 changed: the skirt
/// is the producer's own attribute and it is now in the generated table, so this crate reaches it
/// through the table like any other.
#[test]
fn the_skirt_shares_the_interleaved_buffer_as_well() {
    let descs = [
        from(&COLOR_RELIEF_SHADER[0], INTERLEAVED, 0),
        from(&COLOR_RELIEF_SHADER[1], INTERLEAVED, 4),
        from(&COLOR_RELIEF_SHADER[2], INTERLEAVED, 8),
    ];
    let planned = plan(&COLOR_RELIEF_SHADER, &descs).expect("agrees");
    assert!(planned.undeclared.is_empty(), "{planned:?}");
    assert_eq!(
        planned
            .bound
            .iter()
            .map(|bound| bound.slot)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
        "the skirt binds at the slot the table declares"
    );
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices, vec![INTERLEAVED]);
    assert_eq!(vertex_bytes(&got), 144, "counted once for all three");
}

#[test]
fn separate_buffers_stay_separate_and_keep_first_seen_order() {
    let second = SlabRef {
        slab: 2,
        offset: 64,
        length: 48,
    };
    let descs = [
        from(&RASTER_SHADER[0], INTERLEAVED, 0),
        from(&RASTER_SHADER[1], second, 0),
    ];
    let planned = plan(&RASTER_SHADER, &descs).expect("agrees");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices, vec![INTERLEAVED, second]);
    assert_eq!(got.reads[0].buffer, 0);
    assert_eq!(got.reads[1].buffer, 1);
    assert_eq!(vertex_bytes(&got), 144 + 48);
}

/// One slab and one offset but two lengths are not the same bytes, and taking the first would
/// leave the longer binding short.
#[test]
fn one_offset_with_two_lengths_is_two_buffers() {
    let longer = SlabRef {
        length: INTERLEAVED.length * 2,
        ..INTERLEAVED
    };
    assert_eq!(
        INTERLEAVED.slab_and_offset(),
        longer.slab_and_offset(),
        "the same slab and offset, which is what makes this the trap"
    );
    let descs = [
        from(&RASTER_SHADER[0], INTERLEAVED, 0),
        from(&RASTER_SHADER[1], longer, 0),
    ];
    let planned = plan(&RASTER_SHADER, &descs).expect("agrees");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices, vec![INTERLEAVED, longer]);
    assert_eq!(vertex_bytes(&got), 144 * 3);
}

/// The index buffer is its own even when it points into the same slab, because it binds through a
/// different call.
#[test]
fn the_index_buffer_is_not_a_vertex_buffer() {
    let descs = [from(&RASTER_SHADER[0], INTERLEAVED, 0)];
    let planned = plan(&RASTER_SHADER, &descs).expect("agrees");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.indexes, Some(INDEXES));
    assert_eq!(got.vertices, vec![INTERLEAVED]);
    assert!(
        !got.vertices.contains(&INDEXES),
        "it binds through vkCmdBindIndexBuffer"
    );
}

#[test]
fn a_zero_length_index_reference_is_no_index_buffer() {
    let descs = [from(&RASTER_SHADER[0], INTERLEAVED, 0)];
    let planned = plan(&RASTER_SHADER, &descs).expect("agrees");
    let empty = SlabRef {
        slab: 1,
        offset: 144,
        length: 0,
    };
    assert_eq!(needs(&planned, empty).indexes, None);
}

/// A family whose data-driven attributes each arrive in their own buffer, which is what the paint
/// binder produces: one buffer per attribute rather than one interleaved vertex.
#[test]
fn a_buffer_per_attribute_needs_one_each() {
    let sources: Vec<SlabRef> = (0..FILL_SHADER.len())
        .map(|at| SlabRef {
            slab: 3,
            offset: u32::try_from(at).unwrap() * 32,
            length: 32,
        })
        .collect();
    let descs: Vec<AttributeDesc> = FILL_SHADER
        .iter()
        .zip(&sources)
        .map(|(entry, source)| from(entry, *source, 0))
        .collect();
    let planned = plan(&FILL_SHADER, &descs).expect("agrees");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices.len(), FILL_SHADER.len());
    assert_eq!(
        vertex_bytes(&got),
        32 * FILL_SHADER.len() as u64,
        "nothing was deduplicated that should not have been"
    );
    for (at, read) in got.reads.iter().enumerate() {
        assert_eq!(read.buffer, at, "each binding reads its own");
    }
}

#[test]
fn nothing_bound_needs_nothing() {
    let planned = plan(&RASTER_SHADER, &[]).expect("a partial run is not an error");
    let got = needs(&planned, INDEXES);
    assert_eq!(got.vertices, [] as [SlabRef; 0]);
    assert_eq!(vertex_bytes(&got), 0);
    assert_eq!(got.indexes, Some(INDEXES), "indices are not an attribute");
}
