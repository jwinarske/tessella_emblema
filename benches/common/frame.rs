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

// Each bench is its own crate and compiles this module separately, so whatever *that* bench does
// not name reads as dead here -- `two_passes` uses the writers and none of the one-view scene's
// constants, and `a_frame` the other way about. The alternative is a copy of the writers per scene,
// which is the duplication this module exists to remove. `common/mod.rs` carries the same allow for
// the same reason.
#![allow(dead_code)]

use tessella_capture_abi::envelope::{
    AttributeDesc, CameraUpdate, DrawFlags, Extent, GeometryAdd, GeometryId, OrderEntry,
    OrderEpoch, OrderUpdate, Rect16, Segment, SlabRef, Span, StencilTile, StencilTiles, TextureId,
    TextureRef, TextureUpdate, TileId, ViewDeclare, ViewId, ViewUse, WireRecord,
};
use tessella_capture_abi::generated::mbgl_enums::{AttributeDataType, BuiltIn};
use tessella_capture_abi::generated::ubo_layouts::{FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO};
use tessella_capture_abi::generated::{ubo_layouts, ubo_slots};
use tessella_capture_abi::ring::{Producer, Ring};
use tessella_capture_abi::{
    CameraMode, EnvelopeKind, RenderPass, TextureChannelDataType, TexturePixelType,
};
use tessella_consume::slab::Slab;

/// The view the frame draws.
pub const VIEW: ViewId = ViewId(1);

/// The layer group the two tiles belong to.
pub const LAYER: i32 = 0;

/// The layer that samples a texture, which covers a band across the top.
///
/// A raster, because it is the shortest family that samples: three attributes, two blocks and two
/// images. Thirteen of the eighteen families sample something and nothing had driven a
/// `TextureUpdate` into a sampler, so this is the layer that does -- see #93.
pub const SAMPLED: i32 = 2;

/// The layer drawn over it.
///
/// A second layer is what says the painter order is obeyed and that a descriptor set is per layer:
/// `record::content` looks one up with `Which { view, layer }`, and with a single layer a `get` that
/// ignored the layer would pass.
pub const OVER: i32 = 1;

/// The epoch the order and the camera agree on.
pub const EPOCH: OrderEpoch = OrderEpoch(1);

/// The two geometries of the lower layer, one per tile.
pub const GEOMETRIES: [GeometryId; 2] = [GeometryId(10), GeometryId(11)];

/// The upper layer's one geometry, which covers the middle of the target and no tile.
pub const ABOVE: GeometryId = GeometryId(12);

/// The sampling layer's one geometry.
pub const SAMPLER: GeometryId = GeometryId(13);

/// The two textures the raster binds, in the order its table declares them.
///
/// Both, because `descriptors::Sets::write` refuses a drawable naming a different number than the
/// family declares -- and the second is bound because the shader declares it, not because anything
/// reads it: `fade_t` of zero takes the near tile alone.
pub const TEXTURES: [TextureId; 2] = [TextureId(100), TextureId(101)];

/// The texel the sampling layer reads, which is texel (2, 2) of its own image.
///
/// Opaque, unlike `first_pixel`'s raster image: the band is drawn over the tiles beneath it, so a
/// half-transparent texel would make the expectation a blend rather than a texel. With an alpha of
/// 255 and every adjustment at its identity the result is the texel itself.
pub const SAMPLED_COLOR: [u8; 4] = [128, 64, 128, 255];

/// How much of the target's height the sampling layer covers, either side of the top.
///
/// An eighth, so it is rows 0 through 7 and disjoint from the band in the middle.
pub const TOP_OF: u16 = 8;

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

/// What the upper layer is painted, which is neither tile's color nor the clear.
///
/// Opaque, and covering a band across the middle: so the order between the layers is readable
/// straight off the pixels and needs no blend arithmetic to interpret. Reversed, the band would hold
/// the tiles' colors instead.
pub const ABOVE_COLOR: [u8; 4] = [16, 240, 112, 255];

/// How much of the target's height the upper layer covers, either side of the middle.
///
/// A band rather than the whole target, so the lower layer is visible beside it -- a layer that
/// covered everything would make "the upper layer drew" and "the lower layer did not" the same
/// picture. Applied as a scale in that layer's own matrix, so its quad is the same as everyone
/// else's.
///
/// A quarter, because the target is 64 and 64 / 4 is whole: the band is rows 24 through 39 and
/// there is no pixel on a boundary to argue about.
///
/// A divisor rather than a fraction, and a `u16` so both users convert exactly: the matrix wants a
/// float and `f32::from` of a `u16` is lossless, while the pixel count wants whole rows and gets
/// them by dividing.
pub const BAND_OF: u16 = 4;

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
    /// Per drawable: the position, color and opacity runs, the indexes, then a raster's texture
    /// coordinate and skirt.
    refs: Vec<[SlabRef; 6]>,
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
    /// One drawable's runs: the position, the packed color and the opacity, the indexes, then a
    /// raster's texture coordinate and skirt.
    ///
    /// Six because two families read it: a fill takes the first three and the indexes, a raster
    /// takes the position, the coordinate and the skirt. Which of them a drawable is decides which
    /// it names, and `vertices::plan` refuses a mismatch.
    #[must_use]
    pub fn runs(&self, at: usize) -> &[SlabRef; 6] {
        &self.refs[at]
    }

    /// Lays both tiles' geometry into one region, as the producer packs a slab.
    #[must_use]
    pub fn new() -> Self {
        let mut region: Vec<u8> = Vec::new();
        let mut refs = Vec::new();
        // Three drawables over one quad: the two tiles and the layer above them. What differs is
        // the color each carries and the matrix its block holds -- the upper layer's squashes the
        // quad into a band across the middle, which is why it needs no geometry of its own.
        for color in [COLORS[0], COLORS[1], ABOVE_COLOR, SAMPLED_COLOR] {
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
            // A raster reads three `Short2` runs where a fill reads a position, a packed color and
            // an opacity -- so the fourth drawable's three runs are the texture coordinate and the
            // skirt rather than a color, and the color above is unused for it. Laid out in the same
            // region, because a slab is a slab.
            let coordinates: Vec<u8> = (0..4)
                .flat_map(|_| {
                    [COORDINATE, COORDINATE]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                })
                .collect();
            let skirt: Vec<u8> = vec![0u8; 4 * 4];
            refs.push([
                at(&positions),
                at(&packed),
                at(&opacity),
                at(&indexes),
                at(&coordinates),
                at(&skirt),
            ]);
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

/// The texture coordinate the sampling layer reads with.
///
/// Under a buffer scale of two, `uv = t / 16384 + 0.25`, so 6144 is 0.625 -- the dead center of
/// texel 2 of a four-texel image. The same number `benches/first_pixel.rs`'s raster cases use, and
/// for the same reason: the center of a texel rather than its edge, so the pixel says which texel
/// was read.
const COORDINATE: i16 = 6144;

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

    // The view first, and that is a rule rather than a tidiness: DR-18 puts the per-view state on
    // `ViewDeclare`, "ordered ahead of any `ViewUse` naming the view", and a use that arrives first
    // is dropped and counted in `Progress::undeclared`. This fixture had no declaration at all
    // until `host` started reading the record -- every drawable in the frame was dropped.
    //
    // `Producer` mode, because every placement here is a matrix the fixture sends: in consumer mode
    // those are advisory and a tile's position comes from its own id instead.
    let declare = ViewDeclare {
        view: VIEW,
        camera_mode: CameraMode::Producer as u8,
        _reserved: [0; 3],
    };
    producer
        .write(EnvelopeKind::ViewDeclare, declare.as_bytes(), &[])
        .expect("room");
    let fill = [
        AttributeDataType::Short2,
        AttributeDataType::Float4,
        AttributeDataType::Float2,
    ];
    for (at, id) in GEOMETRIES.iter().enumerate() {
        let refs = &geometry.refs[at];
        announce(
            producer,
            *id,
            BuiltIn::FillShader,
            &fill,
            &refs[0..3],
            refs[3],
            &[],
        );
        used(
            producer,
            *id,
            VIEW,
            LAYER,
            Some(tile(u32::try_from(at).expect("two tiles"))),
        );
    }
    // The layer above, with no tile of its own: a layer that covers the viewport rather than a
    // tile's ground -- which is what `ViewUse::has_tile` of zero says, and what `Partition`'s "a
    // tile absent here has no mask" leaves unclipped.
    let refs = &geometry.refs[GEOMETRIES.len()];
    announce(
        producer,
        ABOVE,
        BuiltIn::FillShader,
        &fill,
        &refs[0..3],
        refs[3],
        &[],
    );
    used(producer, ABOVE, VIEW, OVER, None);

    // And the layer that samples: three `Short2` runs -- a position, a texture coordinate and a
    // skirt -- and both of the images its table declares.
    let refs = &geometry.refs[GEOMETRIES.len() + 1];
    announce(
        producer,
        SAMPLER,
        BuiltIn::RasterShader,
        &[AttributeDataType::Short2; 3],
        &[refs[0], refs[4], refs[5]],
        refs[3],
        &TEXTURES,
    );
    used(producer, SAMPLER, VIEW, SAMPLED, None);
    textures(producer);

    clips(producer);
    uniforms(producer);
    order(producer);
    camera(producer, VIEW);
    (ring, geometry)
}

/// One geometry: its attribute descriptors, its segment, the textures it binds, and its slab.
///
/// `declared` is the family's own attribute types, in its own order, each over its own run at
/// offset zero -- the per-attribute layout; `first_pixel`'s `raster_interleaved` is the other one.
/// Taken from the table rather than guessed: `vertices::plan` refuses a descriptor whose declared
/// type disagrees, which is how the first version of this fixture was caught claiming `UShort4`
/// for a fill's `Float4` color.
pub fn announce(
    producer: &mut Producer,
    id: GeometryId,
    shader: BuiltIn,
    declared: &[AttributeDataType],
    runs: &[SlabRef],
    indexes: SlabRef,
    textures: &[TextureId],
) {
    let attrs: Vec<AttributeDesc> = declared
        .iter()
        .enumerate()
        .map(|(slot, kind)| AttributeDesc {
            attr_id: u32::try_from(slot).expect("a short table"),
            binding: i32::try_from(slot).expect("a short table"),
            source: runs[slot],
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
        count: u32::try_from(attrs.len()).expect("a short table"),
    };
    let segments_at = u32::try_from(payload.len()).expect("a small payload");
    for segment in &segments {
        payload.extend_from_slice(segment.as_bytes());
    }

    // The textures it binds, in the order the shader declares them. `descriptors::Sets::write`
    // refuses a drawable naming a different number than the family does, so a raster names both of
    // its images even though `fade_t` of zero reads only the first.
    let textures_at = u32::try_from(payload.len()).expect("a small payload");
    for (slot, texture) in textures.iter().enumerate() {
        let bound = TextureRef {
            texture: *texture,
            slot: u32::try_from(slot).expect("two textures"),
            // Zero, which the ABI says is `Linear`: "this was padding through R0, and zero is
            // `TextureFilter::Linear`". The sampling layer reads a texel's center, so either
            // filter gives that texel; `first_pixel` is where the filter is chosen deliberately.
            filter: 0,
        };
        payload.extend_from_slice(bound.as_bytes());
    }

    let add = GeometryAdd {
        geometry: id,
        permutation_key: 0,
        indexes,
        vertex_count: 4,
        attrs: attrs_span,
        instance_attrs: Span::default(),
        segments: Span {
            offset: segments_at,
            count: 1,
        },
        texture_refs: Span {
            offset: textures_at,
            count: u32::try_from(textures.len()).expect("a short list"),
        },
        builtin_shader: shader as i32,
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
pub fn stride_of(kind: AttributeDataType) -> u32 {
    match kind {
        AttributeDataType::Short2 => 4,
        AttributeDataType::Float4 => 16,
        AttributeDataType::Float2 => 8,
        other => panic!("{other:?} is not one this fixture sends"),
    }
}

/// The view's use of one geometry, which is where its layer and its tile come from.
///
/// `None` for a drawable that covers the viewport. The tile field is then meaningless -- the ABI
/// says it is "meaningful only when `has_tile` is set" -- so it is left at its default and the flag
/// is what a consumer reads.
pub fn used(producer: &mut Producer, id: GeometryId, view: ViewId, layer: i32, at: Option<TileId>) {
    let use_ = ViewUse {
        geometry: id,
        view,
        layer_index: layer,
        sub_layer_index: 0,
        tile: at.unwrap_or_default(),
        render_pass: RenderPass::TRANSLUCENT,
        draw_flags: DrawFlags::default(),
        has_tile: u8::from(at.is_some()),
        _pad: [0; 5],
    };
    producer
        .write(EnvelopeKind::ViewUse, use_.as_bytes(), &[])
        .expect("room");
}

/// The sampling layer's two images.
///
/// The near tile goes out as a **packed** two-rect update and the parent as a whole-texture one, so
/// both payload forms `upload::rows` resolves are driven. `TextureUpdate::packed` says what the
/// difference costs if a consumer reads the wrong one:
///
/// > the rects would land at the right addresses holding the wrong pixels, which is a map that
/// > draws rather than one that fails
///
/// The rects are the image's two halves by row, so between them they cover it: a packed payload
/// holds each region's own rows end to end at the region's width, where a whole one holds the
/// texture and the rects are windows into it.
fn textures(producer: &mut Producer) {
    // Texel (x, y) is (x * 64, y * 32, 128, 255), so every one differs and the sampled one says
    // which was read. Opaque, so the band's pixel is the texel rather than a blend over the tiles.
    let texel = |x: u16, y: u16| {
        let (x, y) = (
            u8::try_from(x).expect("a small image"),
            u8::try_from(y).expect("a small image"),
        );
        [x * 64, y * 32, 128, 255]
    };

    // Two sub-regions, not two row-halves. Row-halves were the first shape here and they cannot
    // tell the two payload forms apart: a region spanning the full width starts at the same offset
    // and runs at the same stride whichever form it is in, so a consumer ignoring `packed` reads
    // exactly the right bytes. Measured -- the frame came out unchanged with the flag thrown away.
    //
    // A quarter-sized region does differ. The *second* covers texels (2, 2) to (3, 3), which is the
    // one the sampling layer reads:
    //
    //   packed   at 16, stride 8     after the first region's four texels, at its own width
    //   whole    at 40, stride 16    two rows down and two texels across, at the texture's width
    //
    // Second rather than first, and that matters too: at offset zero a consumer that dropped the
    // offset entirely would still read the right texels. The first covers the opposite corner, so
    // between them nothing the frame reads is left at the zeros `Images::declare` clears to.
    let corners = [
        Rect16 {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        },
        Rect16 {
            x: 2,
            y: 2,
            w: 2,
            h: 2,
        },
    ];
    let packed: Vec<u8> = corners
        .iter()
        .flat_map(|rect| {
            (rect.y..rect.y + rect.h)
                .flat_map(move |y| (rect.x..rect.x + rect.w).flat_map(move |x| texel(x, y)))
        })
        .collect();
    debug_assert_eq!(packed.len(), 2 * 2 * 2 * 4, "two regions of four texels");
    texture(producer, TEXTURES[0], &corners, true, &packed);

    // The parent, which `fade_t` of zero mixes none of. Painted differently from the near tile so a
    // frame that sampled the wrong one would say so, and sent whole -- `rect_count` of zero, which
    // is what a producer sends for a texture it has just created.
    let parent: Vec<u8> = (0..4 * 4).flat_map(|_| [255, 0, 255, 255]).collect();
    texture(producer, TEXTURES[1], &[], false, &parent);
}

/// One texture's pixels, in whichever form the rects describe.
pub fn texture(
    producer: &mut Producer,
    id: TextureId,
    rects: &[Rect16],
    packed: bool,
    pixels: &[u8],
) {
    let mut held = [Rect16::default(); 4];
    held[..rects.len()].copy_from_slice(rects);
    let update = TextureUpdate {
        texture: id,
        size: Extent {
            width: 4,
            height: 4,
        },
        rects: held,
        pixels: Span {
            offset: 0,
            count: u32::try_from(pixels.len()).expect("a small image"),
        },
        format: TexturePixelType::RGBA as u8,
        rect_count: u8::try_from(rects.len()).expect("at most four"),
        channel_type: TextureChannelDataType::UnsignedByte as u8,
        packed: u8::from(packed),
        _pad: Default::default(),
    };
    producer
        .write(EnvelopeKind::TextureUpdate, update.as_bytes(), pixels)
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
    // The lower layer: one entry per tile, both placed by the identity.
    //
    // At the stride the *union* of a fill's drawable blocks sits at, which is 96 where
    // `FillDrawableUBO` is 80 -- see #81. A consumer reading at the struct's own size finds entry
    // one sixteen bytes early.
    let stride = ubo_layouts::FILL_DRAWABLE_UNION_UBO.stride as usize;
    let mut drawables = vec![0u8; stride * GEOMETRIES.len()];
    for at in 0..GEOMETRIES.len() {
        entry(&mut drawables[at * stride..(at + 1) * stride], &CLIP);
    }
    ubo(
        producer,
        VIEW,
        LAYER,
        ubo_slots::ID_FILL_DRAWABLE_UBO,
        &drawables,
    );
    ubo(
        producer,
        VIEW,
        LAYER,
        ubo_slots::ID_FILL_EVALUATED_PROPS_UBO,
        &props(),
    );

    // The upper layer: one entry, squashed into a band across the middle. Its own buffers at its
    // own slots, because a layer's blocks are keyed by the layer -- a consumer that kept one set of
    // them per view would draw this layer through the one below it.
    let mut above = vec![0u8; stride];
    let mut matrix = CLIP;
    matrix[5] /= f32::from(BAND_OF);
    entry(&mut above, &matrix);
    ubo(
        producer,
        VIEW,
        OVER,
        ubo_slots::ID_FILL_DRAWABLE_UBO,
        &above,
    );
    ubo(
        producer,
        VIEW,
        OVER,
        ubo_slots::ID_FILL_EVALUATED_PROPS_UBO,
        &props(),
    );

    // The sampling layer: its quad squashed into a band across the top, and every color adjustment
    // at its identity -- so what its pixel says is which texel was sampled and not what the chain
    // does to it. `benches/first_pixel.rs`'s `raster_adjusted` is where the chain is read.
    let mut raster = vec![0u8; ubo_layouts::RASTER_DRAWABLE_UBO.stride as usize];
    let mut matrix = CLIP;
    matrix[5] /= f32::from(TOP_OF);
    // Up to the top edge. The sign is the one thing here that was not reasoned to: `-0.875` put the
    // band in rows 56 through 63 and `+0.875` puts it in rows 0 through 7, measured by tallying the
    // frame row by row. Two negations are involved -- the drawable matrix's own and the one naga's
    // SPIR-V backend emits -- and reasoning about them in series is how the first attempt got it
    // backwards.
    matrix[13] = 1.0 - 1.0 / f32::from(TOP_OF);
    put(
        &mut raster,
        &ubo_layouts::RASTER_DRAWABLE_UBO,
        "matrix",
        &matrix,
    );
    ubo(
        producer,
        VIEW,
        SAMPLED,
        ubo_slots::ID_RASTER_DRAWABLE_UBO,
        &raster,
    );
    ubo(
        producer,
        VIEW,
        SAMPLED,
        ubo_slots::ID_RASTER_EVALUATED_PROPS_UBO,
        &raster_paint(),
    );
}

/// A raster's evaluated properties, with every color adjustment at its identity.
///
/// The same values `benches/first_pixel.rs`'s `raster` case uses, and for the same reason: with the
/// chain neutral the pixel is the texel, so what this frame says is which texel was sampled. The
/// one difference is the opacity, which is one here -- the band is drawn over the tiles beneath it,
/// and a half-transparent raster would make the expectation a blend.
pub fn raster_paint() -> Vec<u8> {
    let mut paint = vec![0u8; ubo_layouts::RASTER_EVALUATED_PROPS_UBO.stride as usize];
    for (field, values) in [
        // `dot(rgb, spin.xyz)`, `dot(rgb, spin.zxy)`, `dot(rgb, spin.yzx)`, which leaves every
        // channel alone exactly when the weights are (1, 0, 0).
        ("spin_weights", &[1.0f32, 0.0, 0.0, 0.0][..]),
        ("buffer_scale", &[2.0][..]),
        ("scale_parent", &[1.0][..]),
        ("tl_parent", &[0.0, 0.0][..]),
        ("fade_t", &[0.0][..]),
        ("opacity", &[1.0][..]),
        ("brightness_low", &[0.0][..]),
        ("brightness_high", &[1.0][..]),
        ("saturation_factor", &[0.0][..]),
        ("contrast_factor", &[1.0][..]),
    ] {
        put(
            &mut paint,
            &ubo_layouts::RASTER_EVALUATED_PROPS_UBO,
            field,
            values,
        );
    }
    paint
}

/// One drawable entry: where it is placed, and both interpolation factors at their first endpoint.
pub fn entry(into: &mut [u8], matrix: &[f32; 16]) {
    put(into, &FILL_DRAWABLE_UBO, "matrix", matrix);
    put(into, &FILL_DRAWABLE_UBO, "color_t", &[0.0]);
    put(into, &FILL_DRAWABLE_UBO, "opacity_t", &[0.0]);
}

/// The evaluated properties, which a fill reads at entry zero whatever the drawable is.
///
/// Zeros: every color here is a per-vertex attribute, and the block's own color is the fallback a
/// layer with a constant paint would use. A layer that read it instead would draw black.
pub fn props() -> Vec<u8> {
    let mut props = vec![0u8; FILL_EVALUATED_PROPS_UBO.stride as usize];
    put(&mut props, &FILL_EVALUATED_PROPS_UBO, "color", &[0.0; 4]);
    props
}

/// Writes floats into a block at the offset the ABI declares for a named field.
pub fn put(entry: &mut [u8], layout: &ubo_layouts::UboLayout, field: &str, values: &[f32]) {
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

pub fn ubo(producer: &mut Producer, view: ViewId, layer: i32, slot: u32, data: &[u8]) {
    let update = tessella_capture_abi::envelope::UboUpdate {
        view,
        layer_index: layer,
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
    let mut entries: Vec<OrderEntry> = GEOMETRIES
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
    // After them, which is what puts them on top: the order is the painter's.
    entries.push(OrderEntry {
        geometry: ABOVE,
        draw_priority: 0,
        layer_index: u32::try_from(OVER).expect("a non-negative layer"),
        sub_layer_index: 0,
        ubo_index: 0,
        pass: RenderPass::TRANSLUCENT,
        _pad: [0; 3],
    });
    entries.push(OrderEntry {
        geometry: SAMPLER,
        draw_priority: 0,
        layer_index: u32::try_from(SAMPLED).expect("a non-negative layer"),
        sub_layer_index: 0,
        ubo_index: 0,
        pass: RenderPass::TRANSLUCENT,
        _pad: [0; 3],
    });
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
pub fn camera(producer: &mut Producer, view: ViewId) {
    let zeros = vec![0u8; core::mem::size_of::<CameraUpdate>()];
    let mut update = CameraUpdate::from_bytes(&zeros).expect("a camera reads from zeros");
    update.view = view;
    update.order_epoch = EPOCH;
    producer
        .write(EnvelopeKind::CameraUpdate, update.as_bytes(), &[])
        .expect("room");
}
