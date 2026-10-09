// SPDX-License-Identifier: BSD-2-Clause
//! A frame written onto a ring, as the producer would write one.
//!
//! The fixture behind the end-to-end checks: two tiles of one fill layer, each with its own clip
//! mask, written as the envelopes a capture carries and read back through `tessella_consume::host`.
//! Nothing here is a shortcut past the wire -- the records go through `ring::Producer` and come out
//! of `Host::read`, which is the point.
//!
//! # Why two tiles and not one
//!
//! Because one tile cannot show that a drawable is clipped to *its own* tile. With two, each one's
//! geometry covers the whole target and its mask covers half, so the halves say which reference
//! each drew with -- and swapping them swaps the picture rather than blanking it.
//!
//! A single tile also gets a one-bit stencil field, where there are only two values and no
//! reference can fail to match one: the same reason `benches/clip_masks.rs` covers four.

use tessella_capture_abi::envelope::{
    AttributeDesc, CameraUpdate, DrawFlags, GeometryAdd, GeometryId, OrderEntry, OrderEpoch,
    OrderUpdate, Segment, SlabRef, Span, StencilTile, StencilTiles, TileId, ViewId, ViewUse,
    WireRecord,
};
use tessella_capture_abi::generated::mbgl_enums::{AttributeDataType, BuiltIn};
use tessella_capture_abi::generated::ubo_layouts::{FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO};
use tessella_capture_abi::generated::{ubo_layouts, ubo_slots};
use tessella_capture_abi::ring::{Producer, Ring};
use tessella_capture_abi::{EnvelopeKind, RenderPass};
use tessella_consume::slab::Slab;

/// The view the frame draws.
pub const VIEW: ViewId = ViewId(1);

/// The layer group both tiles belong to.
pub const LAYER: i32 = 0;

/// The epoch the order and the camera agree on.
pub const EPOCH: OrderEpoch = OrderEpoch(1);

/// The two geometries, one per tile.
pub const GEOMETRIES: [GeometryId; 2] = [GeometryId(10), GeometryId(11)];

/// The target's edge. Each tile's mask covers half of it.
pub const SIDE: u32 = 64;

/// What one tile's quad is painted.
///
/// Distinct in every channel, and deliberately *not* a permutation of each other. The first pair
/// here was `[255, 64, 0, 255]` and `[0, 64, 255, 255]`, which are one R-and-B swap apart -- so a
/// channel order read wrongly and a tile clipped to its neighbor's mask produced the same pixels,
/// and the bench reported the second when the first was possible. These two share no channel value
/// in a different position, so the two failures cannot be confused.
pub const COLORS: [[u8; 4]; 2] = [[255, 128, 64, 255], [32, 96, 192, 255]];

/// Where each tile's mask lands, as a fraction of the target's width.
///
/// Tile zero takes the left half and tile one the right. The *geometry* of both covers the whole
/// target, so the halves in the result are the masks' work and nothing else's.
pub const HALVES: [(f32, f32); 2] = [(-1.0, 0.0), (0.0, 1.0)];

/// The slab region's geometry, and where each piece of it is.
pub struct Geometry {
    /// The bytes a `SlabRef` resolves into.
    pub region: Vec<u8>,
    /// The slab table, which the consumer is handed rather than deriving.
    pub slabs: Vec<Slab>,
    /// Per tile: the position, color and opacity runs, then the indexes.
    refs: Vec<[SlabRef; 4]>,
}

/// A quad covering clip space, as four vertices of two signed shorts.
///
/// Tile units of 0 and 8192 would be the producer's; these are already clip, because the drawable
/// matrix each tile carries is the identity with `y` negated and what this fixture is about is the
/// *stream*, not the projection. `benches/first_pixel.rs` is where a family's arithmetic is read.
const QUAD: [i16; 8] = [-1, -1, 1, -1, 1, 1, -1, 1];

/// Two triangles over those four vertices.
const INDEXES: [u16; 6] = [0, 1, 2, 0, 2, 3];

impl Geometry {
    /// Lays both tiles' geometry into one region, as the producer packs a slab.
    #[must_use]
    pub fn new() -> Self {
        let mut region: Vec<u8> = Vec::new();
        let mut refs = Vec::new();
        for color in COLORS {
            let mut at = |bytes: &[u8]| {
                let offset = u32::try_from(region.len()).expect("a small region");
                region.extend_from_slice(bytes);
                SlabRef {
                    slab: 0,
                    offset,
                    length: u32::try_from(bytes.len()).expect("a small run"),
                }
            };
            let positions: Vec<u8> = QUAD.iter().flat_map(|v| v.to_le_bytes()).collect();
            // The color, as `unpack_color` reads it: two channels to a float, scaled by 256, and
            // four floats for a data-driven property's two zoom endpoints. Both ends are the same
            // color here, so the pixel is that color whatever `color_t` is -- which is deliberate:
            // what this fixture checks is which *tile* drew, not how the family interpolates.
            // `benches/first_pixel.rs`'s `fill` is where the interpolation is read.
            let packed: Vec<u8> = (0..4)
                .flat_map(|_| {
                    let lo = f32::from(color[0]) * 256.0 + f32::from(color[1]);
                    let hi = f32::from(color[2]) * 256.0 + f32::from(color[3]);
                    [lo, hi, lo, hi]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<u8>>()
                })
                .collect();
            let opacity: Vec<u8> = (0..4)
                .flat_map(|_| [1.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()))
                .collect();
            let indexes: Vec<u8> = INDEXES.iter().flat_map(|v| v.to_le_bytes()).collect();
            refs.push([at(&positions), at(&packed), at(&opacity), at(&indexes)]);
        }
        let length = region.len() as u64;
        Self {
            region,
            slabs: vec![Slab { offset: 0, length }],
            refs,
        }
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

/// A tile, as the producer names one.
#[must_use]
pub fn tile(at: u32) -> TileId {
    TileId {
        z: 14,
        x: 8000 + at,
        y: 5000,
        overscaled_z: 14,
        wrap: 0,
    }
}

/// The identity with `y` negated, which is what every drawable here is placed by.
///
/// Column-major. `benches/first_pixel.rs` has the note on why the negation and naga's own cancel.
const CLIP: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// A matrix placing a full-tile mask quad over `from`..`to` of the target's width.
///
/// The mask's own body builds a quad over tile units of 0..8192 from `vertex_index`, so this scales
/// that to the half's width and shifts it there.
#[must_use]
pub fn half(from: f32, to: f32) -> [f32; 16] {
    let width = to - from;
    [
        width / 8192.0,
        0.0,
        0.0,
        0.0, //
        0.0,
        2.0 / 8192.0,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0, //
        from,
        -1.0,
        0.0,
        1.0,
    ]
}

/// Writes the whole frame onto a ring and returns it with the geometry it refers to.
///
/// The order the envelopes go in is the producer's: a geometry is announced before the view that
/// uses it, the clips and the uniforms follow, and the camera comes last because it is what commits
/// the frame -- §11.7's "hold `CameraUpdate` until its `orderEpoch` is held".
#[must_use]
pub fn write(capacity: usize) -> (Ring, Geometry) {
    let geometry = Geometry::new();
    let mut ring = Ring::new(capacity);
    let producer = ring.producer();
    for (at, id) in GEOMETRIES.iter().enumerate() {
        announce(producer, *id, &geometry.refs[at]);
        used(producer, *id, tile(u32::try_from(at).expect("two tiles")));
    }
    clips(producer);
    uniforms(producer);
    order(producer);
    camera(producer);
    (ring, geometry)
}

/// One geometry: its attribute descriptors, its segment, and the slab its bytes are in.
fn announce(producer: &mut Producer, id: GeometryId, refs: &[SlabRef; 4]) {
    // One descriptor per attribute the fill family declares, each over its own run at offset zero
    // -- which is the per-attribute layout. `first_pixel`'s `raster_interleaved` is the other one.
    // The types `FILL_SHADER` declares, in its own order. Taken from the table rather than
    // guessed: `vertices::plan` refuses a descriptor whose declared type disagrees, which is how
    // the first version of this fixture was caught claiming `UShort4` for a `Float4` color.
    let declared = [
        AttributeDataType::Short2,
        AttributeDataType::Float4,
        AttributeDataType::Float2,
    ];
    let attrs: Vec<AttributeDesc> = declared
        .iter()
        .enumerate()
        .map(|(slot, kind)| AttributeDesc {
            attr_id: u32::try_from(slot).expect("three attributes"),
            binding: i32::try_from(slot).expect("three attributes"),
            source: refs[slot],
            offset: 0,
            vertex_offset: 0,
            stride: stride_of(*kind),
            data_type: *kind as u8,
            declared_data_type: *kind as u8,
            _pad: [0; 2],
        })
        .collect();
    let segments = [Segment {
        vertex_offset: 0,
        index_offset: 0,
        vertex_length: 4,
        index_length: 6,
    }];

    let mut payload = Vec::new();
    for desc in &attrs {
        payload.extend_from_slice(desc.as_bytes());
    }
    let attrs_span = Span {
        offset: 0,
        count: u32::try_from(attrs.len()).expect("three"),
    };
    let segments_at = u32::try_from(payload.len()).expect("a small payload");
    for segment in &segments {
        payload.extend_from_slice(segment.as_bytes());
    }

    let add = GeometryAdd {
        geometry: id,
        permutation_key: 0,
        indexes: refs[3],
        vertex_count: 4,
        attrs: attrs_span,
        instance_attrs: Span::default(),
        segments: Span {
            offset: segments_at,
            count: 1,
        },
        texture_refs: Span::default(),
        builtin_shader: BuiltIn::FillShader as i32,
        vertex_type: AttributeDataType::Short2 as u8,
        reason: 0,
        topology: 0,
        _pad: [0; 1],
    };
    producer
        .write(EnvelopeKind::GeometryAdd, add.as_bytes(), &payload)
        .expect("room");
}

/// How wide one attribute's vertex is, which is also its stride in this layout.
fn stride_of(kind: AttributeDataType) -> u32 {
    match kind {
        AttributeDataType::Short2 => 4,
        AttributeDataType::Float4 => 16,
        AttributeDataType::Float2 => 8,
        other => panic!("{other:?} is not one this fixture sends"),
    }
}

/// The view's use of one geometry, which is where its tile comes from.
fn used(producer: &mut Producer, id: GeometryId, at: TileId) {
    let use_ = ViewUse {
        geometry: id,
        view: VIEW,
        layer_index: LAYER,
        sub_layer_index: 0,
        tile: at,
        render_pass: RenderPass::TRANSLUCENT,
        draw_flags: DrawFlags::default(),
        has_tile: 1,
        _pad: [0; 5],
    };
    producer
        .write(EnvelopeKind::ViewUse, use_.as_bytes(), &[])
        .expect("room");
}

/// The layer group's clip set: one mask per tile, each over its own half.
fn clips(producer: &mut Producer) {
    let tiles: Vec<StencilTile> = HALVES
        .iter()
        .enumerate()
        .map(|(at, (from, to))| StencilTile {
            matrix: half(*from, *to),
            tile: tile(u32::try_from(at).expect("two tiles")),
        })
        .collect();
    let mut payload = Vec::new();
    for one in &tiles {
        payload.extend_from_slice(one.as_bytes());
    }
    let update = StencilTiles {
        view: VIEW,
        layer_index: LAYER,
        tiles: Span {
            offset: 0,
            count: u32::try_from(tiles.len()).expect("two"),
        },
    };
    producer
        .write(EnvelopeKind::StencilTiles, update.as_bytes(), &payload)
        .expect("room");
}

/// The layer's two block buffers: a drawable entry per tile, and the evaluated properties.
fn uniforms(producer: &mut Producer) {
    // The drawable array, at the stride the *union* of a fill's drawable blocks sits at -- which is
    // 96 where `FillDrawableUBO` is 80. See tessella_emblema#81: a consumer reading at the struct's
    // own size finds entry one sixteen bytes early.
    let stride = ubo_layouts::FILL_DRAWABLE_UNION_UBO.stride as usize;
    let mut drawables = vec![0u8; stride * GEOMETRIES.len()];
    for at in 0..GEOMETRIES.len() {
        let entry = &mut drawables[at * stride..(at + 1) * stride];
        put(entry, &FILL_DRAWABLE_UBO, "matrix", &CLIP);
        put(entry, &FILL_DRAWABLE_UBO, "color_t", &[0.0]);
        put(entry, &FILL_DRAWABLE_UBO, "opacity_t", &[0.0]);
    }
    ubo(producer, ubo_slots::ID_FILL_DRAWABLE_UBO, &drawables);

    // The evaluated properties, which a fill reads at entry zero whatever the drawable is.
    let mut props = vec![0u8; FILL_EVALUATED_PROPS_UBO.stride as usize];
    put(&mut props, &FILL_EVALUATED_PROPS_UBO, "color", &[0.0; 4]);
    ubo(producer, ubo_slots::ID_FILL_EVALUATED_PROPS_UBO, &props);
}

/// Writes floats into a block at the offset the ABI declares for a named field.
fn put(entry: &mut [u8], layout: &ubo_layouts::UboLayout, field: &str, values: &[f32]) {
    let found = layout
        .fields
        .iter()
        .find(|held| held.name == field)
        .unwrap_or_else(|| panic!("{} has no {field}", layout.name));
    let at = found.offset as usize;
    for (index, value) in values.iter().enumerate() {
        let start = at + index * 4;
        entry[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn ubo(producer: &mut Producer, slot: u32, data: &[u8]) {
    let update = tessella_capture_abi::envelope::UboUpdate {
        view: VIEW,
        layer_index: LAYER,
        slot,
        _pad: 0,
        data: Span {
            offset: 0,
            count: u32::try_from(data.len()).expect("a small buffer"),
        },
    };
    producer
        .write(EnvelopeKind::UboUpdate, update.as_bytes(), data)
        .expect("room");
}

/// The painter order: both tiles, in tile order, each naming its own entry of the drawable buffer.
fn order(producer: &mut Producer) {
    let entries: Vec<OrderEntry> = GEOMETRIES
        .iter()
        .enumerate()
        .map(|(at, id)| OrderEntry {
            geometry: *id,
            draw_priority: 0,
            layer_index: u32::try_from(LAYER).expect("a non-negative layer"),
            sub_layer_index: 0,
            ubo_index: u32::try_from(at).expect("two tiles"),
            pass: RenderPass::TRANSLUCENT,
            _pad: [0; 3],
        })
        .collect();
    let mut payload = Vec::new();
    for entry in &entries {
        payload.extend_from_slice(entry.as_bytes());
    }
    let update = OrderUpdate {
        view: VIEW,
        order_epoch: EPOCH,
        entries: Span {
            offset: 0,
            count: u32::try_from(entries.len()).expect("two"),
        },
        _pad: 0,
    };
    producer
        .write(EnvelopeKind::OrderUpdate, update.as_bytes(), &payload)
        .expect("room");
}

/// The camera, last, because holding it until its epoch is held is what commits the frame.
///
/// Read from zeros rather than built field by field: a camera is a hundred and sixty bytes of
/// matrices this fixture does not place geometry with -- every drawable carries its own -- and
/// zeros are a camera the decode accepts.
fn camera(producer: &mut Producer) {
    let zeros = vec![0u8; core::mem::size_of::<CameraUpdate>()];
    let mut update = CameraUpdate::from_bytes(&zeros).expect("a camera reads from zeros");
    update.view = VIEW;
    update.order_epoch = EPOCH;
    producer
        .write(EnvelopeKind::CameraUpdate, update.as_bytes(), &[])
        .expect("room");
}
