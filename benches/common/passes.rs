// SPDX-License-Identifier: BSD-2-Clause
//! A frame of two passes: a view that draws into a texture, and the view that samples it.
//!
//! DR-25's shape, as the envelopes carry it. `heatmap` draws its kernels into a half-resolution
//! target and then draws that target through a color ramp, and `hillshade_prepare` has the same two
//! passes -- so two of the eighteen families cannot draw at all without this.
//!
//! # Why a fill and a raster rather than a heatmap
//!
//! Because the mechanism is what has never been driven, and a heatmap would introduce two families'
//! uniforms at the same time. A fill drawing into the target and a raster sampling it are families
//! this fixture already writes, so what a failure here says is about the *passes*: that the child's
//! ran, into an image the parent could sample, before the parent's.
//!
//! `benches/first_pixel.rs` is where a heatmap's own arithmetic is read, in two single-pass cases
//! that cannot say anything about the hand-off between them.
//!
//! # What is deliberately not under test
//!
//! The texture coordinate. The child paints its whole target one color, so any texel of it gives
//! the same answer and a coordinate read wrongly would still pass. `benches/a_frame.rs` is where
//! that is checked, against an image whose every texel differs.

// Each bench is its own crate and compiles this module separately, so whatever *that* bench does
// not name reads as dead here -- `two_passes` uses the writers and none of the one-view scene's
// constants, and `a_frame` the other way about. The alternative is a copy of the writers per scene,
// which is the duplication this module exists to remove. `common/mod.rs` carries the same allow for
// the same reason.
#![allow(dead_code)]

use tessella_capture_abi::envelope::{
    GeometryId, OrderEntry, OrderEpoch, OrderUpdate, Span, TextureId, ViewDeclare, ViewId,
    ViewTarget, WireRecord,
};
use tessella_capture_abi::generated::mbgl_enums::{AttributeDataType, BuiltIn};
use tessella_capture_abi::generated::{ubo_layouts, ubo_slots};
use tessella_capture_abi::ring::{Producer, Ring};
use tessella_capture_abi::{
    CameraMode, EnvelopeKind, RenderPass, TextureChannelDataType, TexturePixelType,
};

use super::frame::{self, Geometry};

/// The view that reaches the screen.
pub const PARENT: ViewId = ViewId(1);

/// The view that draws into a texture, which the parent samples.
pub const CHILD: ViewId = ViewId(2);

/// The id the child's output is bound by.
///
/// In `TextureUpdate`'s id space and never the subject of one: nothing uploads pixels to a render
/// target. So a consumer holds an image under this id that no upload will ever name, which is the
/// second way into `Images` this slice adds.
pub const TARGET: TextureId = TextureId(80);

/// The parent's second image, which a raster declares and `fade_t` of zero does not read.
pub const PARENT_IMAGE: TextureId = TextureId(81);

/// The layer each view draws in.
pub const LAYER: i32 = 0;

/// The parent's edge, in pixels.
pub const SIDE: u32 = 64;

/// The child's size against the parent, as the fraction `ViewTarget` carries.
///
/// One half is mbgl's heatmap target, which is what the ABI gives as the example.
pub const SCALE: (u16, u16) = (1, 2);

/// What the child paints its whole target.
///
/// The first color the shared geometry carries, because the child draws that geometry: its packed
/// color is a vertex attribute laid into the slab once, so a color of its own would mean a fifth
/// drawable in the region rather than a constant here.
///
/// Distinct from the parent's second image in every channel and from the clear, so a frame that
/// sampled the wrong one of the three says which.
pub const CHILD_COLOR: [u8; 4] = frame::COLORS[0];

/// Writes both passes onto a ring: the child's view, then the parent's.
///
/// The order within it is the producer's. Both views are declared before anything names them, the
/// `ViewTarget` comes after the child's own `ViewDeclare` -- "a target naming an undeclared view is
/// the same protocol fault a use would be" -- and each view's camera comes last, because that is
/// what commits its frame.
#[must_use]
pub fn write(capacity: usize) -> (Ring, Geometry) {
    let geometry = Geometry::new();
    let mut ring = Ring::new(capacity);
    let producer = ring.producer();

    for view in [PARENT, CHILD] {
        let declare = ViewDeclare {
            view,
            camera_mode: CameraMode::Producer as u8,
            _reserved: [0; 3],
        };
        producer
            .write(EnvelopeKind::ViewDeclare, declare.as_bytes(), &[])
            .expect("room");
    }
    let target = ViewTarget {
        view: CHILD,
        parent: PARENT,
        texture: TARGET,
        scale_num: SCALE.0,
        scale_den: SCALE.1,
        // Eight bits a channel, where a heatmap's target is `HalfFloat`: what this frame samples is
        // a flat color, so the range a half carries is not what it is about. `RGBA` and
        // `UnsignedByte` is also what the parent's own image is, which keeps the two comparable.
        format: TexturePixelType::RGBA as u8,
        channel_type: TextureChannelDataType::UnsignedByte as u8,
        _pad: [0; 2],
    };
    producer
        .write(EnvelopeKind::ViewTarget, target.as_bytes(), &[])
        .expect("room");

    child(producer, &geometry);
    parent(producer, &geometry);
    (ring, geometry)
}

/// The child's pass: one fill over its whole target, in a layer of its own.
fn child(producer: &mut Producer, geometry: &Geometry) {
    let refs = geometry.runs(0);
    frame::announce(
        producer,
        DRAWN,
        BuiltIn::FillShader,
        &[
            AttributeDataType::Short2,
            AttributeDataType::Float4,
            AttributeDataType::Float2,
        ],
        &refs[0..3],
        refs[3],
        &[],
    );
    frame::used(producer, DRAWN, CHILD, LAYER, None);

    let stride = ubo_layouts::FILL_DRAWABLE_UNION_UBO.stride as usize;
    let mut drawable = vec![0u8; stride];
    frame::entry(&mut drawable, &IDENTITY);
    frame::ubo(
        producer,
        CHILD,
        LAYER,
        ubo_slots::ID_FILL_DRAWABLE_UBO,
        &drawable,
    );
    frame::ubo(
        producer,
        CHILD,
        LAYER,
        ubo_slots::ID_FILL_EVALUATED_PROPS_UBO,
        &frame::props(),
    );
    commit(producer, CHILD, DRAWN);
}

/// The parent's pass: one raster over its whole target, sampling the child's.
fn parent(producer: &mut Producer, geometry: &Geometry) {
    let refs = geometry.runs(1);
    frame::announce(
        producer,
        SAMPLES,
        BuiltIn::RasterShader,
        &[AttributeDataType::Short2; 3],
        &[refs[0], refs[4], refs[5]],
        refs[3],
        // The target first, which is the slot the near tile binds at -- and the parent's own image
        // second, which the raster declares and `fade_t` of zero does not read.
        &[TARGET, PARENT_IMAGE],
    );
    frame::used(producer, SAMPLES, PARENT, LAYER, None);

    let mut drawable = vec![0u8; ubo_layouts::RASTER_DRAWABLE_UBO.stride as usize];
    frame::put(
        &mut drawable,
        &ubo_layouts::RASTER_DRAWABLE_UBO,
        "matrix",
        &IDENTITY,
    );
    frame::ubo(
        producer,
        PARENT,
        LAYER,
        ubo_slots::ID_RASTER_DRAWABLE_UBO,
        &drawable,
    );
    frame::ubo(
        producer,
        PARENT,
        LAYER,
        ubo_slots::ID_RASTER_EVALUATED_PROPS_UBO,
        &frame::raster_paint(),
    );

    // The second image, sent whole. The first is the child's target and no upload names it.
    let pixels: Vec<u8> = (0..4 * 4).flat_map(|_| [0, 255, 0, 255]).collect();
    frame::texture(producer, PARENT_IMAGE, &[], false, &pixels);
    commit(producer, PARENT, SAMPLES);
}

/// One view's order of a single drawable, and the camera that commits it.
fn commit(producer: &mut Producer, view: ViewId, geometry: GeometryId) {
    let entries = [OrderEntry {
        geometry,
        draw_priority: 0,
        layer_index: u32::try_from(LAYER).expect("a non-negative layer"),
        sub_layer_index: 0,
        ubo_index: 0,
        pass: RenderPass::TRANSLUCENT,
        _pad: [0; 3],
    }];
    let mut payload = Vec::new();
    for entry in &entries {
        payload.extend_from_slice(entry.as_bytes());
    }
    let update = OrderUpdate {
        view,
        order_epoch: EPOCH,
        entries: Span {
            offset: 0,
            count: u32::try_from(entries.len()).expect("one"),
        },
        _pad: 0,
    };
    producer
        .write(EnvelopeKind::OrderUpdate, update.as_bytes(), &payload)
        .expect("room");
    frame::camera(producer, view);
}

/// The geometry the child draws.
pub const DRAWN: GeometryId = GeometryId(20);

/// The geometry the parent draws.
pub const SAMPLES: GeometryId = GeometryId(21);

/// The epoch both views' orders and cameras agree on.
const EPOCH: OrderEpoch = OrderEpoch(1);

/// The identity with `y` negated, which is what both drawables are placed by.
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];
