// SPDX-License-Identifier: BSD-2-Clause
//! A frame off a ring, drawn.
//!
//! The last of #72's four findings and what closes it. Every other bench here starts from values a
//! case wrote by hand; this one starts from bytes on a ring and ends at pixels, through the whole
//! chain: `Host::read` to decode, `Joiner` to pair geometry with the view that uses it,
//! `collapse_into` to batch it, `vertices::plan` and `buffers::needs` over the attribute
//! descriptors the stream carried, `Store`, `Blocks` and `Images` to put it on the device,
//! `stencil::partition` for the masks, `Cache` for the pipelines and `record::content` to record
//! the draws.
//!
//! # What a pass proves that `first_pixel` cannot
//!
//! That the join is right. `first_pixel` names its own geometry, blocks and textures, so it says a
//! family draws correctly when it is set up correctly -- and says nothing about whether a stream
//! sets it up correctly. Here the setup is the stream's: a drawable's attribute descriptors, its
//! tile, its entry in the layer's buffer and its place in the painter order all come off the wire.
//!
//! The assertion is about *clipping*, because that is the part of a frame no single-drawable check
//! can reach. Two tiles of one fill layer, each drawable covering the whole target, each mask
//! covering half: so the halves in the result say which tile's reference each drew with. Swapping
//! the references swaps the picture, and a stencil that did nothing would leave the second
//! drawable over the whole target.
//!
//! Run with `cargo bench --bench a_frame`.

mod common;

#[path = "common/frame.rs"]
mod frame;

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::{GeometryId, TileId};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_consume::host::Host;
use tessella_consume::upload::Upload;
use tessella_emblema::blocks::Blocks;
use tessella_emblema::descriptors::Sets;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{Blend, Cache, Targets};
use tessella_emblema::record::Program;
use tessella_emblema::store::Store;
use tessella_emblema::surface::Surface;
use tessella_emblema::target::{self, Depth, Host as Host_};
use tessella_emblema::{blocks, buffers, families, masks, pipelines, record, shaders, vertices};
use tessella_vk::{Buffer, Image, ImageView, Memory, Recorder};

use common::Open;

/// The slot a clipping mask's matrices arrive at, through mbgl's own constant.
const MASK_SLOT: u32 = tessella_capture_abi::generated::ubo_slots::ID_CLIPPING_MASK_UBO;

/// What the target is cleared to, which no tile's color is.
const CLEAR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];

/// The same color as the target's bytes, so "nothing drew here" is a distinguishable answer.
const CLEARED: [u8; 4] = [51, 102, 153, 255];

/// The target, its depth attachment and the buffer a frame is read back out of.
struct Target<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
    readback: Buffer<'d>,
    read_memory: Memory<'d>,
    depth: Depth<'d>,
    targets: Targets,
}

impl<'d> Target<'d> {
    fn new(open: &'d Open, attachment: Attachment) -> Result<Self, String> {
        let gpu = open.gpu();
        let color = vk::Format::R8G8B8A8_UNORM;
        let image = gpu
            .image(
                frame::SIDE,
                frame::SIDE,
                color,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("the target image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("the target memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("binding the target: {why}"))?;
        let view = gpu
            .view(&image, color)
            .map_err(|why| format!("the target view: {why}"))?;

        let bytes = u64::from(frame::SIDE) * u64::from(frame::SIDE) * 4;
        let readback = gpu
            .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
            .map_err(|why| format!("the readback buffer: {why}"))?;
        let needs = [readback.requirements()];
        let read_memory = gpu
            .allocate(
                bytes.max(needs[0].size),
                &needs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|why| format!("the readback memory: {why}"))?;
        read_memory
            .bind(&readback, 0)
            .map_err(|why| format!("binding the readback: {why}"))?;

        let depth_stencil = device::depth_stencil_format(attachment, |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
        let depth = Depth::new(gpu, frame::SIDE, frame::SIDE, depth_stencil)
            .map_err(|why| format!("the depth attachment: {why}"))?;

        Ok(Self {
            image,
            view,
            _memory: memory,
            readback,
            read_memory,
            depth,
            targets: Targets {
                color,
                depth_stencil,
                attachment,
            },
        })
    }

    /// The whole frame, as the target's bytes.
    fn read(&self) -> Result<Vec<u8>, String> {
        let mapping = self
            .read_memory
            .map()
            .map_err(|why| format!("mapping the readback: {why}"))?;
        let mut pixels = vec![0u8; (frame::SIDE * frame::SIDE * 4) as usize];
        mapping
            .read(0, &mut pixels)
            .map_err(|why| format!("reading the frame: {why}"))?;
        Ok(pixels)
    }
}

/// WGSL to SPIR-V, as every other bench here compiles it.
fn compile(source: &str) -> Result<Vec<u32>, String> {
    let parsed = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spirv: {why}"))
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(why) => {
            eprintln!("a frame: {why}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    // The stream half first, which needs no device: a frame that did not decode is a frame there is
    // no point drawing, and the two failures look nothing alike.
    let (mut ring, geometry) = frame::write(1 << 16);
    let mut host = Host::new();
    let progress = host.read(ring.consumer());
    if progress.unknown != 0 || progress.malformed != 0 {
        return Err(format!(
            "{} unknown and {} malformed of {} records",
            progress.unknown, progress.malformed, progress.records
        ));
    }
    if !host.ready(frame::VIEW) {
        return Err("the camera and the order do not agree".into());
    }

    let read = {
        let found = host.plan(frame::VIEW).ok_or("no plan for the view")?;
        let drawables: usize = found
            .batches
            .iter()
            .map(|batch| batch.geometries.len())
            .sum();
        Read {
            batches: found.batches.len(),
            drawables,
            masks: found.clips.len(),
            assignments: found.clips.partition().tiles.len(),
            partitioned: found.clips.partition().partitioned,
            uniforms: host.uploads().work().len(),
        }
    };
    if read.batches != 1 || read.drawables != 2 {
        return Err(format!(
            "{} batches of {} drawables, wanted one of two",
            read.batches, read.drawables
        ));
    }
    if read.masks != 2 || read.assignments != 2 {
        return Err(format!(
            "{} masks and {} assignments, wanted two of each",
            read.masks, read.assignments
        ));
    }
    if !read.partitioned {
        return Err("two tiles at one zoom must fit the stencil byte".into());
    }
    if read.uniforms != 2 {
        return Err(format!("{} uniform uploads, wanted two", read.uniforms));
    }
    println!(
        "  the stream            ok   {} records, 1 batch of 2 drawables, 2 masks, 2 uniforms",
        progress.records
    );

    // And the geometry the stream pointed at resolves, which is the other half of a decode: a
    // reference that does not resolve is a drawable that cannot be uploaded.
    let resolved = frame::GEOMETRIES
        .iter()
        .filter_map(|id| host.joiner().drawable(*id, frame::VIEW))
        .flat_map(|drawable| drawable.geometry.attrs.clone())
        .filter(|desc| {
            tessella_consume::slab::resolve(&geometry.region, &geometry.slabs, desc.source)
                .is_some()
        })
        .count();
    if resolved != 6 {
        return Err(format!("{resolved} of 6 attribute runs resolve"));
    }
    println!("  the slab              ok   6 attribute runs resolve against the region");

    let device = match Open::preferred() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping the draw: {why}");
            return Ok(());
        }
    };
    println!("  device: {} ({})", device.name, device.class);

    // Both attachments, because the stencil-only one was broken and nothing could see it: its view
    // carried a depth aspect `S8_UINT` does not have, the stencil test then passed everywhere, and
    // both drawables drew over the whole target. Tile one's color in both halves, measured. #90.
    //
    // Run as a pair rather than as one, so the two cannot drift: a fix to the aspects that worked
    // only for the packed format would pass the first and fail the second.
    //
    // The pair is only two different things on a device that offers `S8_UINT` as a depth-stencil
    // attachment, which is what `depth_stencil_format` asks. RADV does; V3D 7.1.7.0 does not, and
    // falls back to `D24_UNORM_S8_UINT` for both -- so on the Pi this runs the same format twice
    // and the stencil-only path is covered on RADV alone.
    for attachment in [Attachment::DepthStencil, Attachment::StencilOnly] {
        draw(&device, &mut host, &geometry, attachment)?;
    }
    Ok(())
}

/// Puts the frame on the device, records it, and reads the halves back.
#[allow(clippy::too_many_lines)]
fn draw(
    device: &Open,
    host: &mut Host,
    geometry: &frame::Geometry,
    attachment: Attachment,
) -> Result<(), String> {
    let gpu = device.gpu();
    let target = Target::new(device, attachment)?;

    // The uniforms the stream carried, each at the slot it named. The entry count comes from the
    // bytes, which is the only place it is: nothing on the wire says it.
    let mut blocks = Blocks::new();
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
        let which = blocks::Which {
            view: *view,
            layer: *layer_index,
        };
        // Sixteen bytes, which is every block's declared alignment -- so a chunk never straddles
        // anything meaningful and `flush` merges runs of them back into entry-sized ranges.
        let grain = 16;
        blocks
            .declare(gpu, which, *slot, data.len() / grain, grain)
            .map_err(|why| format!("slot {slot}: {why}"))?;
        blocks
            .replace(which, *slot, data)
            .map_err(|why| format!("slot {slot}: {why}"))?;
        blocks
            .flush(which, *slot, grain)
            .map_err(|why| format!("slot {slot}: {why}"))?;
    }

    // The masks' own buffer: one matrix per tile, at the slot a clipping mask's block travels at.
    let which = blocks::Which {
        view: frame::VIEW,
        layer: frame::LAYER,
    };
    let order: Vec<(TileId, [f32; 16])> = host
        .clips(frame::VIEW)
        .masks()
        .map(|(tile, matrix)| (tile, *matrix))
        .collect();
    let mask_bytes: Vec<u8> = order
        .iter()
        .flat_map(|(_, matrix)| matrix.iter().flat_map(|value| value.to_le_bytes()))
        .collect();
    blocks
        .declare(gpu, which, MASK_SLOT, order.len(), 64)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    blocks
        .replace(which, MASK_SLOT, &mask_bytes)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    blocks
        .flush(which, MASK_SLOT, 0)
        .map_err(|why| format!("the mask buffer: {why}"))?;

    // The geometry, resolved out of the slab the stream pointed at rather than copied by the host.
    let mut store = Store::new();
    let mut plans: BTreeMap<GeometryId, vertices::Plan> = BTreeMap::new();
    for id in frame::GEOMETRIES {
        let drawable = host
            .joiner()
            .drawable(id, frame::VIEW)
            .ok_or_else(|| format!("geometry {} is not used by the view", id.0))?;
        let family = families::family(
            BuiltIn::from_repr(drawable.geometry.add.builtin_shader)
                .ok_or("the stream named a family this build does not know")?,
        )
        .ok_or("the stream named a family this build does not draw")?;
        let plan = vertices::plan(family.attributes, &drawable.geometry.attrs)
            .map_err(|why| format!("geometry {}: {why:?}", id.0))?;
        let needs = buffers::needs(&plan, drawable.geometry.add.indexes);
        store
            .upload(gpu, id, &needs, &drawable.geometry.segments, &|reference| {
                tessella_consume::slab::resolve(&geometry.region, &geometry.slabs, reference)
            })
            .map_err(|why| format!("geometry {}: {why}", id.0))?;
        plans.insert(id, plan);
    }

    // The pipelines, built before the recording: `record::content` resolves a program through a
    // closure, and building one inside it would need the cache mutably from a `&dyn Fn`.
    let mut cache = Cache::new();
    let fill = families::family(BuiltIn::FillShader).ok_or("no fill family")?;
    let bindings = pipelines::bindings(fill, Surface::Plane);
    let words = compile(
        &shaders::module(
            Surface::Plane,
            fill.blocks,
            fill.attributes,
            fill.textures,
            fill.body,
        )
        .map_err(|why| format!("the fill does not assemble: {why:?}"))?,
    )?;
    let key = pipelines::key(
        BuiltIn::FillShader,
        Surface::Plane,
        0,
        plans
            .get(&frame::GEOMETRIES[0])
            .ok_or("the first geometry has no plan")?,
        Blend::Alpha,
    );
    let pipeline = cache
        .pipeline(gpu, &key, &bindings, &words, target.targets)
        .map_err(|why| format!("the fill pipeline: {why}"))?;
    let layout = cache
        .layout(gpu, BuiltIn::FillShader, Surface::Plane, &bindings)
        .map_err(|why| format!("the fill layout: {why}"))?;
    let mut sets = Sets::new(gpu, 2, &bindings).map_err(|why| format!("the pool: {why}"))?;
    // Written for its effect: `record::content` finds it through `sets.get(which)` rather than
    // being handed one, because a set is per layer and a batch is a run within a layer.
    if sets
        .write(layout, &bindings, which, &blocks, &[])
        .map_err(|why| format!("the fill set: {why}"))?
        == vk::DescriptorSet::null()
    {
        return Err("the fill set is null".into());
    }
    let program = Program {
        pipeline,
        layout: layout.pipeline(),
        instances: 1,
    };

    // The mask pipeline and its own set, over the one storage binding its body declares.
    let mask_bindings = [pipelines::Binding {
        binding: 0,
        kind: pipelines::Kind::StorageBuffer,
        slot: Some(MASK_SLOT),
    }];
    let mask_layout =
        pipelines::layout(gpu, &mask_bindings).map_err(|why| format!("the mask layout: {why}"))?;
    let mask_module = gpu
        .shader(&compile(masks::BODY)?)
        .map_err(|why| format!("the mask module: {why}"))?;
    let mask_pipeline = pipelines::build_mask(gpu, &mask_layout, &mask_module, target.targets)
        .map_err(|why| format!("the mask pipeline: {why}"))?;
    let mut mask_sets =
        Sets::new(gpu, 1, &mask_bindings).map_err(|why| format!("the mask pool: {why}"))?;
    let mask_set = mask_sets
        .write(&mask_layout, &mask_bindings, which, &blocks, &[])
        .map_err(|why| format!("the mask set: {why}"))?;

    let found = host.plan(frame::VIEW).ok_or("no plan for the view")?;
    let partition = found.clips.partition();
    let mut counts = Err("the recording did not run".to_string());
    device.submit(|record: Recorder<'_>| {
        target::frame(
            record,
            Host_ {
                image: &target.image,
                view: &target.view,
                width: frame::SIDE,
                height: frame::SIDE,
                layout: vk::ImageLayout::UNDEFINED,
            },
            &target.depth,
            Some(CLEAR),
            |record| {
                // The masks first, each writing its own field. One draw per tile, because the
                // reference and the write mask are the tile's and both are dynamic.
                record.bind_pipeline(mask_pipeline.raw());
                record.bind_descriptor_set(mask_layout.pipeline(), mask_set);
                for (at, (tile, _)) in order.iter().enumerate() {
                    let Some(given) = partition.tiles.get(tile) else {
                        continue;
                    };
                    record.stencil_reference(u32::from(given.value));
                    record.stencil_write_mask(u32::from(given.write_mask));
                    record.draw(masks::VERTICES, 1, at as u32);
                }

                // Then the content, through the frame's own batches.
                counts = record::content(
                    record,
                    found.batches,
                    &record::Scene {
                        store: &store,
                        sets: &sets,
                        joiner: found.joiner,
                        partition,
                        view: frame::VIEW,
                    },
                    &|_| Some(program),
                )
                .map_err(|why| format!("recording: {why}"));
            },
        );
        record.transition(
            &target.image,
            target::LEAVES_IN,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        record.copy_to_buffer(&target.image, &target.readback, frame::SIDE, frame::SIDE);
    })?;
    let counts = counts?;
    if counts.batches != 1 || counts.drawables != 2 || counts.draws != 2 {
        return Err(format!(
            "{} batches, {} drawables, {} draws recorded",
            counts.batches, counts.drawables, counts.draws
        ));
    }
    if counts.unclipped != 0 {
        return Err(format!(
            "{} drawables drew unclipped, so a tile had no mask",
            counts.unclipped
        ));
    }
    println!(
        "  the recording         ok   {} batches, {} drawables, {} draws, none unclipped",
        counts.batches, counts.drawables, counts.draws
    );

    halves(&target, target.targets.depth_stencil)
}

/// Each half of the target holds its own tile's color, and neither holds the other's.
fn halves(target: &Target<'_>, format: vk::Format) -> Result<(), String> {
    let pixels = target.read()?;
    let mut seen = [0usize; 2];
    for y in 0..frame::SIDE {
        for x in 0..frame::SIDE {
            let at = ((y * frame::SIDE + x) * 4) as usize;
            let found = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];
            let side = usize::from(x >= frame::SIDE / 2);
            if found == frame::COLORS[side] {
                seen[side] += 1;
                continue;
            }
            if found == frame::COLORS[1 - side] {
                return Err(format!(
                    "({x}, {y}) is in half {side} and holds half {}'s color, so the references are \
                     crossed",
                    1 - side
                ));
            }
            if found == CLEARED {
                return Err(format!(
                    "({x}, {y}) is still the clear, so nothing drew there -- which is what a mask \
                     that wrote nothing leaves behind"
                ));
            }
            return Err(format!(
                "({x}, {y}) holds {found:?}, which is neither tile's color nor the clear"
            ));
        }
    }
    let each = (frame::SIDE * frame::SIDE / 2) as usize;
    if seen != [each, each] {
        return Err(format!("{seen:?} texels of {each} in each half"));
    }
    println!(
        "  the halves            ok   {each} texels of each tile's color in its own half, on \
         {format:?}"
    );
    Ok(())
}

/// What the stream said, gathered before the borrow on the host ends.
struct Read {
    batches: usize,
    drawables: usize,
    masks: usize,
    assignments: usize,
    partitioned: bool,
    uniforms: usize,
}
