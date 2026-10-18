// SPDX-License-Identifier: BSD-2-Clause
//! An authored model, from the wire to a pixel.
//!
//! The last of #93's five: a `MeshAdd` announced on a ring, bound by a `ViewUse`, placed by the
//! order, and drawn from the bytes the slab held -- which is the one path in that issue nothing
//! had driven end to end. Every other family here is geometry the producer computed and described;
//! this is an asset it never looked inside.
//!
//! # What each half contributes, and why that is the thing to check
//!
//! The asset carries the shape and the producer carries the placement. So the two cases are the
//! same asset drawn twice, under two matrices the *stream* sent: a model that ignored them would
//! draw the same picture both times, and one that read its own coordinates as clip space would
//! draw the triangle in the corner whatever the producer said.
//!
//! Which is why this is not a test of `meshes::load` -- `tests/meshes.rs` is, with the fixtures
//! and the refusals -- but of the path: `Host` plans an `AUTHORED_MESH` batch, the slab resolves,
//! the loader reads it, and the pipeline built from `meshes::BODY` puts it where the matrix says.

mod common;

#[path = "common/frame.rs"]
mod frame;

use ash::vk;

use tessella_capture_abi::envelope::{
    AddReason, GeometryId, MeshAdd, MeshFormat, OrderEntry, OrderEpoch, OrderUpdate, SlabRef, Span,
    ViewDeclare, ViewId, ViewUse, WireRecord,
};
use tessella_capture_abi::ring::Ring;
use tessella_capture_abi::{CameraMode, EnvelopeKind, RenderPass};
use tessella_consume::batch::AUTHORED_MESH;
use tessella_consume::host::Host;
use tessella_consume::slab::Slab;
use tessella_consume::upload::Upload;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Targets};
use tessella_emblema::target::{self, Depth, Host as TargetHost};
use tessella_emblema::{blocks, descriptors, meshes};
use tessella_vk::{Buffer, Gpu, Image, ImageView, Memory};

use common::Open;

/// The target's side, in pixels. Small enough to count every texel and wide enough that half of it
/// is unambiguous.
const SIDE: u32 = 64;
/// The color format the pass draws into, which is what `target::frame` expects.
const COLOR: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// The view the frame draws.
const VIEW: ViewId = ViewId(1);
/// The layer the model belongs to.
const LAYER: i32 = 0;
/// The id the asset is announced under, in the geometry space it shares.
const MESH: GeometryId = GeometryId(7);
/// The slot the matrix travels in.
///
/// A model layer is not an mbgl family and has no slot of its own on the wire, so this borrows a
/// per-drawable one. What it may not be is a *global* slot: `slots::scope` calls every slot under
/// `GLOBAL_UBO_COUNT` frame-wide, and a frame-wide block is held under `FRAME_WIDE` rather than
/// the layer -- so a matrix sent at slot zero is written to one place and looked up in another.
/// Which is what the first run of this bench did, and what the number is here to say.
const MESH_SLOT: u32 = tessella_capture_abi::generated::ubo_slots::ID_FILL_DRAWABLE_UBO;
/// The ring, large enough for the handful of records this writes.
const CAPACITY: usize = 1 << 16;

/// The color `meshes::BODY`'s fragment stage returns, as the BGRA bytes it lands as.
///
/// Computed from the shader's own floats rather than written twice: the constant in the shader is
/// what a reader would change, and a copy here would go stale without failing.
fn drawn() -> [u8; 4] {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the shader's own channels, which are between zero and one"
    )]
    let channel = |value: f32| (value * 255.0).round() as u8;
    // BGRA, which is the format the target is read back as.
    [channel(0.5), channel(0.75), channel(0.25), 255]
}

/// A GLB carrying one right triangle over the unit square, with 16-bit indices.
///
/// The unit square rather than a tile's 8192: an asset's coordinates are its own, and a model that
/// is drawn correctly is one whose coordinates the *matrix* turned into clip space.
fn triangle_glb() -> Vec<u8> {
    let mut binary = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for value in position {
            binary.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in [0u16, 1, 2] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    binary.resize(binary.len().next_multiple_of(4), 0);

    let json = br#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 44}]
}"#;
    let json_padded = json.len().next_multiple_of(4);
    let bin_padded = binary.len().next_multiple_of(4);

    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(12 + 8 + json_padded + 8 + bin_padded)
            .expect("a small asset")
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(json_padded)
            .expect("a small chunk")
            .to_le_bytes(),
    );
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    out.resize(out.len() + (json_padded - json.len()), b' ');
    out.extend_from_slice(
        &u32::try_from(bin_padded)
            .expect("a small chunk")
            .to_le_bytes(),
    );
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&binary);
    out.resize(out.len() + (bin_padded - binary.len()), 0);
    out
}

/// The matrix that maps the asset's unit square onto clip space, shifted by `x`.
///
/// Column-major, as a `mat4x4<f32>` reads it. The `z` is left at zero and the shader's own remap
/// takes it to the middle of the depth range, where nothing else is drawn.
fn placing(x: f32) -> [f32; 16] {
    [
        2.0,
        0.0,
        0.0,
        0.0, //
        0.0,
        2.0,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0, //
        -1.0 + x,
        -1.0,
        0.0,
        1.0,
    ]
}

/// The asset's bytes, as a region with one slab over them.
///
/// The producer's own arrangement: a `MeshAdd` points into a slab rather than carrying the asset
/// inline, because a model tile is hundreds of kilobytes and the ring is sized by envelope count.
struct Asset {
    region: Vec<u8>,
    slabs: Vec<Slab>,
    reference: SlabRef,
}

impl Asset {
    fn new(bytes: &[u8]) -> Self {
        let length = u64::try_from(bytes.len()).expect("a small asset");
        Self {
            region: bytes.to_vec(),
            slabs: vec![Slab { offset: 0, length }],
            reference: SlabRef {
                slab: 0,
                offset: 0,
                length: u32::try_from(bytes.len()).expect("a small asset"),
            },
        }
    }
}

/// Writes the frame a model layer is: the view, the asset, its use, the order, the matrix, the
/// camera.
fn write(asset: &Asset, matrix: [f32; 16]) -> Ring {
    let mut ring = Ring::new(CAPACITY);
    {
        let (producer, _) = ring.split();
        producer
            .write(
                EnvelopeKind::ViewDeclare,
                ViewDeclare {
                    view: VIEW,
                    camera_mode: CameraMode::Producer as u8,
                    _reserved: [0; 3],
                }
                .as_bytes(),
                &[],
            )
            .expect("room");
        producer
            .write(
                EnvelopeKind::MeshAdd,
                MeshAdd {
                    mesh: MESH,
                    bytes: asset.reference,
                    format: MeshFormat::Glb as u8,
                    reason: AddReason::Created as u8,
                    _pad: [0; 2],
                }
                .as_bytes(),
                &[],
            )
            .expect("room");
        producer
            .write(
                EnvelopeKind::ViewUse,
                ViewUse {
                    geometry: MESH,
                    view: VIEW,
                    layer_index: LAYER,
                    sub_layer_index: 0,
                    tile: tessella_capture_abi::envelope::TileId::default(),
                    render_pass: RenderPass::TRANSLUCENT,
                    draw_flags: tessella_capture_abi::envelope::DrawFlags::default(),
                    has_tile: 0,
                    _pad: [0; 5],
                }
                .as_bytes(),
                &[],
            )
            .expect("room");

        let entry = OrderEntry {
            geometry: MESH,
            draw_priority: 0,
            #[allow(clippy::cast_sign_loss)]
            layer_index: LAYER as u32,
            sub_layer_index: 0,
            ubo_index: 0,
            pass: RenderPass::TRANSLUCENT,
            _pad: [0; 3],
        };
        producer
            .write(
                EnvelopeKind::OrderUpdate,
                OrderUpdate {
                    order_epoch: OrderEpoch(1),
                    view: VIEW,
                    layer_count: 1,
                    entries: Span {
                        offset: 0,
                        count: 1,
                    },
                }
                .as_bytes(),
                entry.as_bytes(),
            )
            .expect("room");

        // The placement, which travels as a `UboUpdate` like every other layer's: the asset says
        // nothing about where it goes.
        let bytes: Vec<u8> = matrix.iter().flat_map(|f| f.to_le_bytes()).collect();
        frame::ubo(producer, VIEW, LAYER, MESH_SLOT, &bytes);
        frame::camera(producer, VIEW);
    }
    ring
}

/// The image a pass draws into, with its view.
struct Owned<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
}

impl<'d> Owned<'d> {
    fn new(gpu: Gpu<'d>) -> Result<Self, String> {
        let image = gpu
            .image(
                SIDE,
                SIDE,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("bind: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("view: {why}"))?;
        Ok(Self {
            image,
            view,
            _memory: memory,
        })
    }
}

/// Puts bytes in a host-visible buffer of `usage`.
fn upload<'d>(
    gpu: Gpu<'d>,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(Buffer<'d>, Memory<'d>), String> {
    let length = u64::try_from(bytes.len()).expect("a small buffer");
    let buffer = gpu
        .buffer(length, usage)
        .map_err(|why| format!("buffer: {why}"))?;
    let requirements = [buffer.requirements()];
    let memory = gpu
        .allocate(
            length.max(requirements[0].size),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|why| format!("memory: {why}"))?;
    memory
        .bind(&buffer, 0)
        .map_err(|why| format!("bind: {why}"))?;
    let mut mapping = memory.map().map_err(|why| format!("map: {why}"))?;
    mapping
        .write(0, bytes)
        .map_err(|why| format!("write: {why}"))?;
    drop(mapping);
    Ok((buffer, memory))
}

/// Reads the target back as BGRA.
fn read_back(device: &Open, owned: &Owned<'_>) -> Result<Vec<u8>, String> {
    let bytes = u64::from(SIDE) * u64::from(SIDE) * 4;
    let gpu = device.gpu();
    let buffer = gpu
        .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
        .map_err(|why| format!("buffer: {why}"))?;
    let requirements = [buffer.requirements()];
    let memory = gpu
        .allocate(
            bytes.max(requirements[0].size),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|why| format!("memory: {why}"))?;
    memory
        .bind(&buffer, 0)
        .map_err(|why| format!("bind: {why}"))?;
    device.submit(|record| {
        record.transition(
            &owned.image,
            target::LEAVES_IN,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        record.copy_to_buffer(&owned.image, &buffer, SIDE, SIDE);
    })?;
    let mut out = vec![0u8; usize::try_from(bytes).expect("a small image")];
    let mapping = memory.map().map_err(|why| format!("map: {why}"))?;
    mapping
        .read(0, &mut out)
        .map_err(|why| format!("read: {why}"))?;
    Ok(out)
}

/// Where the model's own color landed: how many texels, and how many of those on the left.
fn landed(found: &[u8]) -> (usize, usize) {
    let wanted = drawn();
    let mut total = 0;
    let mut left = 0;
    for (at, texel) in found.chunks(4).enumerate() {
        if texel == wanted {
            total += 1;
            if u32::try_from(at).expect("a small image") % SIDE < SIDE / 2 {
                left += 1;
            }
        }
    }
    (total, left)
}

/// Draws the model the stream describes and reads the target back.
///
/// One function because it is one sequence -- read the stream, find the batch, resolve the slab,
/// load the asset, put its bytes and its matrix on the device, build the pipeline, draw -- and
/// every step of it is what the case is about. Split for the line count, the halves would have to
/// hand each other eight borrowed things whose lifetimes are the device's.
#[allow(clippy::too_many_lines, reason = "one sequence, which is the subject")]
fn draw(device: &Open, matrix: [f32; 16]) -> Result<Vec<u8>, String> {
    let gpu = device.gpu();
    let asset = Asset::new(&triangle_glb());
    let mut ring = write(&asset, matrix);

    // The stream, read the way any consumer reads it.
    let mut host = Host::new();
    let progress = host.read(ring.consumer());
    if progress.malformed != 0 || progress.unsupported != 0 || progress.unknown != 0 {
        return Err(format!("the stream did not read clean: {progress:?}"));
    }
    let planned = host.plan(VIEW).ok_or("no plan for the view")?;

    // One batch, and it says it is a mesh rather than a family.
    let batches: Vec<_> = planned.batches.iter().collect();
    if batches.len() != 1 {
        return Err(format!("{} batches, wanted one", batches.len()));
    }
    if batches[0].key.builtin_shader != AUTHORED_MESH {
        return Err(format!(
            "the batch names family {} rather than an authored mesh",
            batches[0].key.builtin_shader
        ));
    }
    let id = *batches[0]
        .geometries
        .first()
        .ok_or("the batch holds no drawable")?;

    // The asset, out of the slab the record pointed at rather than from a copy.
    let held = planned
        .joiner
        .mesh(id)
        .ok_or("the mesh the batch names was not announced")?;
    let bytes = tessella_consume::slab::resolve(&asset.region, &asset.slabs, held.add.bytes)
        .ok_or("the mesh's slab reference does not resolve")?;
    let loaded = meshes::load(bytes).map_err(|why| format!("the asset: {why}"))?;
    if loaded.indices.len() != 3 {
        return Err(format!(
            "{} indices, and the asset carries a triangle",
            loaded.indices.len()
        ));
    }

    // The matrix the stream sent, in the block the shader indexes by instance.
    let which = blocks::Which {
        view: VIEW,
        layer: LAYER,
    };
    let mut held_blocks = blocks::Blocks::new();
    let mut wrote = 0;
    for work in host.uploads().work() {
        let Upload::Uniforms {
            view,
            layer_index,
            slot,
            bytes,
        } = work
        else {
            continue;
        };
        let data = host
            .uploads()
            .bytes(bytes)
            .ok_or("a uniform upload's bytes are not in the host's buffer")?;
        let at = blocks::Which {
            view: *view,
            layer: *layer_index,
        };
        let grain = 16;
        held_blocks
            .declare(gpu, at, *slot, data.len() / grain, grain)
            .map_err(|why| format!("blocks: {why}"))?;
        held_blocks
            .replace(at, *slot, data)
            .map_err(|why| format!("blocks: {why}"))?;
        held_blocks
            .flush(at, *slot, grain)
            .map_err(|why| format!("blocks: {why}"))?;
        wrote += 1;
    }
    if wrote != 1 {
        return Err(format!("{wrote} uniform uploads, wanted the one matrix"));
    }

    // One storage buffer, which is all the shader declares.
    let bindings = [pipelines::Binding {
        group: 0,
        binding: 0,
        kind: pipelines::Kind::StorageBuffer,
        slot: Some(MESH_SLOT),
    }];
    let layout = pipelines::layout(gpu, &bindings).map_err(|why| format!("layout: {why}"))?;
    let mut sets =
        descriptors::Sets::new(gpu, 1, &bindings).map_err(|why| format!("sets: {why}"))?;
    sets.write(&layout, &bindings, which, AUTHORED_MESH, &held_blocks, &[])
        .map_err(|why| format!("set: {why}"))?;
    let set = sets
        .get(which, AUTHORED_MESH)
        .ok_or("the descriptor set was not written")?;

    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    let depth =
        Depth::new(gpu, SIDE, SIDE, depth_stencil).map_err(|why| format!("depth: {why}"))?;
    let targets = Targets {
        color: COLOR,
        depth_stencil,
        attachment: Attachment::DepthStencil,
    };

    // The vertex input the asset has: three floats a vertex, tightly packed, at location 0.
    let key = pipelines::Key {
        shader: AUTHORED_MESH,
        surface: tessella_emblema::surface::Surface::Plane,
        topology: tessella_capture_abi::envelope::Topology::Triangles,
        permutation: 0,
        layout: vec![pipelines::Slot {
            slot: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
            stride: 12,
            rate: vk::VertexInputRate::VERTEX,
        }],
        // Unblended, so the pixel read back is what the fragment stage returned rather than what it
        // returned over the clear.
        blend: pipelines::Blend::Unblended,
        depth: pipelines::Depth::Write,
        enabled: pipelines::Enabled::ALL,
        stencil: pipelines::Stencil::Clip,
    };
    let modules = pipelines::stage_modules(gpu, &compile(meshes::BODY)?)
        .map_err(|why| format!("module: {why}"))?;
    let pipeline = pipelines::build(gpu, &key, &layout, pipelines::Stages::of(&modules), targets)
        .map_err(|why| format!("pipeline: {why}"))?;

    let (vertices, _vertex_memory) = upload(
        gpu,
        &loaded.position_bytes(),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    )?;
    let (indices, _index_memory) = upload(
        gpu,
        &loaded.index_bytes(),
        vk::BufferUsageFlags::INDEX_BUFFER,
    )?;
    let count = u32::try_from(loaded.indices.len()).expect("a small mesh");

    let owned = Owned::new(gpu)?;
    device.submit(|record| {
        target::frame(
            record,
            TargetHost {
                image: &owned.image,
                view: &owned.view,
                width: SIDE,
                height: SIDE,
                layout: vk::ImageLayout::UNDEFINED,
            },
            &depth,
            // Transparent, so any color in the result came from the model.
            Some([0.0, 0.0, 0.0, 0.0]),
            |record| {
                record.bind_pipeline(pipeline.raw());
                record.bind_descriptor_set(layout.pipeline(), set);
                record.bind_vertex_buffers(0, &[vertices.raw()], &[0]);
                // Thirty-two bits, because the indices are the asset's: see
                // `tessella_vk::Recorder::bind_index_buffer`.
                record.bind_index_buffer_32(indices.raw());
                record.stencil_reference(0);
                record.stencil_compare_mask(0);
                record.stencil_write_mask(0);
                record.draw_indexed(count, 1, 0, 0, 0);
            },
        );
    })?;
    read_back(device, &owned)
}

/// Compiles WGSL, as every bench here does.
fn compile(source: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spv: {why:?}"))
}

/// The model reaches a pixel, from the asset the stream pointed at.
fn a_mesh_draws_from_the_stream(device: &Open) -> Result<(), String> {
    let found = draw(device, placing(0.0))?;
    let (total, left) = landed(&found);

    // Half the target, within the diagonal's own row of texels: the asset is a right triangle over
    // the unit square and the matrix maps that square onto the whole of clip space.
    let half = (SIDE * SIDE / 2) as usize;
    if total == 0 {
        return Err("nothing was drawn: the model did not reach a pixel".into());
    }
    if total.abs_diff(half) > SIDE as usize {
        return Err(format!(
            "{total} texels of the model, wanted about {half}: the triangle is the wrong size"
        ));
    }
    // Three quarters of it on the left, which is the triangle's own share rather than a tolerance:
    // `x + y <= 1` over the unit square puts 0.375 of the square left of the midline and 0.125
    // right of it, so a right triangle covering half the target covers three quarters of its own
    // area in the left half. A shape drawn from the wrong corner, or mirrored, reads differently.
    let share = (left * 100) / total;
    if !(70..=80).contains(&share) {
        return Err(format!(
            "{left} of {total} texels on the left, which is {share}% against the triangle's own \
             75%: the asset was not placed the way the matrix asks"
        ));
    }
    println!(
        "  from the stream       ok   {total} texels of about {half}, {share}% of them on the left"
    );
    Ok(())
}

/// And it is placed where the producer's matrix says, not where the asset's coordinates are.
///
/// The same asset, the same loader, one matrix translated by half of clip space. A consumer that
/// drew the asset's coordinates as clip space -- or ignored the block and read a zero matrix --
/// would draw the identical picture twice, which the first case alone cannot tell.
fn the_matrix_is_the_producers(device: &Open) -> Result<(), String> {
    let (first, first_left) = landed(&draw(device, placing(0.0))?);
    let (moved, moved_left) = landed(&draw(device, placing(1.0))?);

    if moved == 0 {
        return Err("nothing was drawn under the moved matrix".into());
    }
    if first_left == 0 {
        return Err("the first draw put nothing in the left half".into());
    }
    if moved_left != 0 {
        return Err(format!(
            "{moved_left} of {moved} texels still in the left half: the matrix moved and the \
             model did not"
        ));
    }
    println!("  the producer's place  ok   {first_left} of {first} left, {moved} right, one asset");
    Ok(())
}

type Case = fn(&Open) -> Result<(), String>;

fn main() {
    let device = match Open::first() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping: {why}");
            return;
        }
    };
    println!("device: {}", device.name);

    let mut failed = 0;
    for (name, case) in [
        ("from_the_stream", a_mesh_draws_from_the_stream as Case),
        ("producers_place", the_matrix_is_the_producers),
    ] {
        if let Err(why) = case(&device) {
            println!("  {name:<21} FAIL {why}");
            failed += 1;
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
