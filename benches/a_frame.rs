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
//! # What it does not catch
//!
//! A `TextureRef`'s `slot`. `descriptors::bound_from` places the views in the order the refs
//! arrive and never reads the field, so making the two refs claim each other's slots changes
//! nothing in the frame -- measured. That is #95 and a defect rather than a gap here: the bench
//! cannot see it until the consumer places by slot.
//!
//! Run with `cargo bench --bench a_frame`.

mod common;

#[path = "common/frame.rs"]
mod frame;

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::{GeometryId, Rect16, TextureFilter, TextureId, TileId};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_consume::host::Host;
use tessella_consume::upload::Upload;
use tessella_emblema::blocks::Blocks;
use tessella_emblema::descriptors::Sets;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::images::Images;
use tessella_emblema::pipelines::{Blend, Cache, Targets};
use tessella_emblema::record::Program;
use tessella_emblema::store::Store;
use tessella_emblema::surface::Surface;
use tessella_emblema::target::{self, Depth, Host as Host_};
use tessella_emblema::{
    blocks, buffers, descriptors, families, masks, pipelines, record, shaders, vertices,
};
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
    if progress.unknown != 0 || progress.malformed != 0 || progress.undeclared != 0 {
        return Err(format!(
            "{} unknown, {} malformed and {} naming an undeclared view, of {} records",
            progress.unknown, progress.malformed, progress.undeclared, progress.records
        ));
    }
    if !host.declared(frame::VIEW) {
        return Err("the view was not declared".into());
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
    // Three batches, because the collapse is scoped to a layer: the two tiles share a program and
    // collapse into one, and each layer above them is its own run -- the second because a layer
    // breaks a run however alike its program is, the third because it is another family entirely.
    if read.batches != 3 || read.drawables != 4 {
        return Err(format!(
            "{} batches of {} drawables, wanted three of four",
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
    // Eight: a drawable buffer and a properties buffer for each of the three layers, and the two
    // images the sampling layer binds.
    if read.uniforms != 8 {
        return Err(format!("{} uploads, wanted eight", read.uniforms));
    }
    println!(
        "  the stream            ok   {} records, 3 batches of 4 drawables, 2 masks, 8 uploads",
        progress.records
    );

    // And the geometry the stream pointed at resolves, which is the other half of a decode: a
    // reference that does not resolve is a drawable that cannot be uploaded.
    let resolved = frame::GEOMETRIES
        .iter()
        .copied()
        .chain([frame::ABOVE, frame::SAMPLER])
        .filter_map(|id| host.joiner().drawable(id, frame::VIEW))
        .flat_map(|drawable| drawable.geometry.attrs.clone())
        .filter(|desc| {
            tessella_consume::slab::resolve(&geometry.region, &geometry.slabs, desc.source)
                .is_some()
        })
        .count();
    if resolved != 12 {
        return Err(format!("{resolved} of 12 attribute runs resolve"));
    }
    println!("  the slab              ok   12 attribute runs resolve against the region");

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

    // The textures the stream carried, each declared at the size it named and filled from the rows
    // `Upload::rows` resolves. Both payload forms arrive -- one packed two-rect update and one
    // whole-texture one -- and this is the one place that difference is applied rather than tested.
    let mut images = Images::new();
    let mut staged: Result<(), String> = Ok(());
    device.submit(|record: Recorder<'_>| {
        staged = stage(&mut images, gpu, record, host);
    })?;
    staged?;

    // The masks' own buffer: one matrix per tile, at the slot a clipping mask's block travels at.
    //
    // Keyed at layer -1, which is the producer's own convention for a buffer that "belongs to the
    // renderer rather than to any layer" -- `UboUpdate::layer_index` says so of the frame-wide
    // blocks. It has to be a layer no content set uses: `Sets` holds one set per `Which`, so a mask
    // keyed at layer zero would replace the set the first layer's batches bind.
    let mask_at = blocks::Which {
        view: frame::VIEW,
        layer: -1,
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
        .declare(gpu, mask_at, MASK_SLOT, order.len(), 64)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    blocks
        .replace(mask_at, MASK_SLOT, &mask_bytes)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    blocks
        .flush(mask_at, MASK_SLOT, 0)
        .map_err(|why| format!("the mask buffer: {why}"))?;

    // The geometry, resolved out of the slab the stream pointed at rather than copied by the host.
    let mut store = Store::new();
    let mut plans: BTreeMap<GeometryId, vertices::Plan> = BTreeMap::new();
    for id in frame::GEOMETRIES
        .iter()
        .copied()
        .chain([frame::ABOVE, frame::SAMPLER])
    {
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

    // The pipelines and the sets, both before the recording: `record::content` resolves a program
    // through a closure, and building one inside it would need the cache mutably from a `&dyn Fn`.
    //
    // A family's set is written while its layout is held, and the two families are done in turn --
    // `Cache::layout` hands back a reference out of a `&mut self` borrow, so holding one family's
    // layout while asking for another's does not compile. The same shape `Host::plan` had, and not
    // a problem here: a set is written once per layer and after that only its handle is needed.
    let mut cache = Cache::new();
    let fill = families::family(BuiltIn::FillShader).ok_or("no fill family")?;
    let raster = families::family(BuiltIn::RasterShader).ok_or("no raster family")?;
    let bindings = pipelines::bindings(fill, Surface::Plane);
    let raster_bindings = pipelines::bindings(raster, Surface::Plane);

    // One pool for every layer, sized for the widest set among the families this frame draws --
    // which is what `Sets` is for: "the pool is sized from the widest set rather than per family,
    // because a pool is reset whole and sizing it per family would mean a pool per family". Two
    // pools was the first shape here, and `record::content` takes one `Sets` -- so the raster's set
    // was somewhere it could not look: "view ViewId(1) layer 2 has no descriptor set".
    let widest = if raster_bindings.len() > bindings.len() {
        &raster_bindings
    } else {
        &bindings
    };
    // Four: one per layer, and one for the masks at layer -1. A pool sized for three answers
    // `ERROR_OUT_OF_POOL_MEMORY` on the fourth, which is what it did.
    let mut sets = Sets::new(gpu, 4, widest).map_err(|why| format!("the pool: {why}"))?;

    let program = {
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

        // One set per layer, which is the shape `record::content` looks them up in: a batch's layer
        // decides which set it binds, so a layer whose set was never written is a batch with
        // nothing to bind -- and a lookup that ignored the layer would hand a layer another's
        // blocks and draw it through the wrong matrix.
        for layer in [frame::LAYER, frame::OVER] {
            let at = blocks::Which {
                view: frame::VIEW,
                layer,
            };
            if sets
                .write(layout, &bindings, at, &blocks, &[])
                .map_err(|why| format!("the set for layer {layer}: {why}"))?
                == vk::DescriptorSet::null()
            {
                return Err(format!("the set for layer {layer} is null"));
            }
        }
        Program {
            pipeline,
            layout: layout.pipeline(),
            instances: 1,
        }
    };

    // And the sampling layer's own program, which is a different family: its own module, its own
    // bindings -- two more, for the image and the sampler -- and its own set, carrying the views
    // the stream's `TextureRef`s name. One program for every batch would bind the fill's pipeline
    // to a raster's vertex layout, which `vkCreateGraphicsPipelines` would not have built at all.
    let raster_program = {
        let words = compile(
            &shaders::module(
                Surface::Plane,
                raster.blocks,
                raster.attributes,
                raster.textures,
                raster.body,
            )
            .map_err(|why| format!("the raster does not assemble: {why:?}"))?,
        )?;
        let key = pipelines::key(
            BuiltIn::RasterShader,
            Surface::Plane,
            0,
            plans
                .get(&frame::SAMPLER)
                .ok_or("the sampling geometry has no plan")?,
            Blend::Alpha,
        );
        let pipeline = cache
            .pipeline(gpu, &key, &raster_bindings, &words, target.targets)
            .map_err(|why| format!("the raster pipeline: {why}"))?;
        let layout = cache
            .layout(gpu, BuiltIn::RasterShader, Surface::Plane, &raster_bindings)
            .map_err(|why| format!("the raster layout: {why}"))?;

        // The textures it binds, as the stream named them: ids and filters off the `TextureRef`
        // run, resolved to views by the store. A drawable naming a texture nobody uploaded is
        // `Error::NoTexture` rather than a blank sampler.
        let mut refs: Vec<(TextureId, TextureFilter)> = Vec::new();
        for bound in &host
            .joiner()
            .drawable(frame::SAMPLER, frame::VIEW)
            .ok_or("the sampling geometry is not used by the view")?
            .geometry
            .texture_refs
        {
            let filter = filter_of(bound.filter)
                .ok_or_else(|| format!("{} is not a filter this build knows", bound.filter))?;
            refs.push((bound.texture, filter));
        }
        let bound = descriptors::bound_from(&images, &refs)
            .map_err(|why| format!("the sampling layer's textures: {why}"))?;
        let at = blocks::Which {
            view: frame::VIEW,
            layer: frame::SAMPLED,
        };
        sets.write(layout, &raster_bindings, at, &blocks, &bound)
            .map_err(|why| format!("the raster set: {why}"))?;
        Program {
            pipeline,
            layout: layout.pipeline(),
            instances: 1,
        }
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
    let mask_set = sets
        .write(&mask_layout, &mask_bindings, mask_at, &blocks, &[])
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
                    // By family, which is what a batch's key carries: a program is a pipeline and
                    // the layout its set binds through, and the two families here share neither.
                    &|batch| match BuiltIn::from_repr(batch.key.builtin_shader)? {
                        BuiltIn::FillShader => Some(program),
                        BuiltIn::RasterShader => Some(raster_program),
                        _ => None,
                    },
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
    if counts.batches != 3 || counts.drawables != 4 || counts.draws != 4 {
        return Err(format!(
            "{} batches, {} drawables, {} draws recorded",
            counts.batches, counts.drawables, counts.draws
        ));
    }
    // Two: the layer above and the sampling layer both cover the viewport rather than a tile, so
    // neither has a mask. Three would mean a tile's mask went missing.
    if counts.unclipped != 2 {
        return Err(format!(
            "{} drawables drew unclipped, wanted two -- the layers that cover the viewport",
            counts.unclipped
        ));
    }
    println!(
        "  the recording         ok   {} batches, {} drawables, {} draws, {} unclipped",
        counts.batches, counts.drawables, counts.draws, counts.unclipped
    );

    halves(&target, target.targets.depth_stencil)
}

/// The filter a `TextureRef` names.
///
/// Matched here because the ABI has no decoder for it: `TextureUpdate::format` and `channel_type`
/// both have a `from_repr` and `host` refuses a value neither enum knows, and this field is a bare
/// `u32` -- so every backend writes this match itself. Filed as #95.
///
/// Zero is `Linear`, which the ABI states: "this was padding through R0, and zero is
/// `TextureFilter::Linear`". Anything other than the two it names is `None` rather than defaulted,
/// because defaulting is how a producer sending a third filter is read as sending the first.
fn filter_of(raw: u32) -> Option<TextureFilter> {
    match raw {
        0 => Some(TextureFilter::Linear),
        1 => Some(TextureFilter::Nearest),
        _ => None,
    }
}

/// Declares and fills every texture the stream sent, and leaves them readable by a shader.
///
/// `Images::upload` asks for a region's rows one at a time -- "given the region's index and row, it
/// answers that row's pixels" -- and `Upload::rows` is what says where a row is: an offset and a
/// stride per region, which differ between the two payload forms. A backend that read the wrong one
/// would upload the right number of bytes to the right place from the wrong rows.
fn stage<'d>(
    images: &mut Images<'d>,
    gpu: tessella_vk::Gpu<'d>,
    record: Recorder<'_>,
    host: &Host,
) -> Result<(), String> {
    for work in host.uploads().work() {
        let Upload::Texture {
            texture,
            size,
            shape,
            rects,
            bytes,
        } = work
        else {
            continue;
        };
        let pixels = host
            .uploads()
            .bytes(bytes)
            .ok_or("a texture upload's bytes are not in the host's buffer")?;
        let rows = work
            .rows()
            .ok_or("a texture upload has no row layout")?
            .map_err(|why| format!("texture {}: {why:?}", texture.0))?;

        images
            .declare(gpu, record, *texture, *size, shape.format, shape.channel)
            .map_err(|why| format!("texture {}: {why}", texture.0))?;

        // An empty rect list is one region covering the whole texture, which is what
        // `upload::rows` answers for it -- so the rects handed to `Images` are the same list with
        // that one substituted.
        let whole = [Rect16 {
            x: 0,
            y: 0,
            w: u16::try_from(size.width).map_err(|_| "a texture past 65535".to_string())?,
            h: u16::try_from(size.height).map_err(|_| "a texture past 65535".to_string())?,
        }];
        let regions: &[Rect16] = if rects.is_empty() { &whole } else { rects };
        let texel = shape.texel();
        images
            .upload(gpu, record, *texture, regions, &|index, row| {
                let at = rows.get(index)?;
                let width = usize::from(regions.get(index)?.w) * texel;
                let start = at.at + usize::from(row) * at.stride;
                pixels.get(start..start + width)
            })
            .map_err(|why| format!("texture {}: {why}", texture.0))?;

        let held = images
            .image(*texture)
            .ok_or_else(|| format!("texture {} was not held", texture.0))?;
        record.transition(
            held,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
    Ok(())
}

/// The band holds the upper layer, and each half outside it holds its own tile.
///
/// Three regions, which is what two layers and two tiles come to. Each is a different failure:
///
/// * the band holding a tile's color -- the layers drew in the wrong order, or the upper layer's
///   set was the lower one's and it was placed by the lower one's matrix;
/// * a half holding the other's color -- the tiles' references are crossed;
/// * the clear anywhere -- a mask wrote nothing, so nothing passed the stencil test there.
fn halves(target: &Target<'_>, format: vk::Format) -> Result<(), String> {
    let pixels = target.read()?;
    let mut seen = [0usize; 4];
    for y in 0..frame::SIDE {
        for x in 0..frame::SIDE {
            let at = ((y * frame::SIDE + x) * 4) as usize;
            let found = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];

            // Three bands and two halves, all on whole rows: the target is 64, the upper layer's
            // matrix divides the clip range by four and the sampling layer's by eight, so the
            // middle band is rows 24 through 39 and the top one rows 0 through 7. No pixel sits on
            // an edge. Integer arithmetic, because a pixel is an integer.
            let middle = frame::SIDE / 2;
            let reach = frame::SIDE / u32::from(frame::BAND_OF) / 2;
            let top = frame::SIDE / u32::from(frame::TOP_OF);
            let region = if y < top {
                3
            } else if y >= middle - reach && y < middle + reach {
                2
            } else {
                usize::from(x >= frame::SIDE / 2)
            };
            let wanted = match region {
                3 => frame::SAMPLED_COLOR,
                2 => frame::ABOVE_COLOR,
                side => frame::COLORS[side],
            };
            if found == wanted {
                seen[region] += 1;
                continue;
            }
            if found == CLEARED {
                return Err(format!(
                    "({x}, {y}) is still the clear, so nothing drew there -- which is what a mask \
                     that wrote nothing leaves behind"
                ));
            }
            if region == 3 {
                return Err(format!(
                    "({x}, {y}) is in the sampled band and holds {found:?} rather than texel \
                     (2, 2), so the texture reached the sampler wrongly -- the rows, the rects or \
                     the view bound"
                ));
            }
            if region == 2 {
                return Err(format!(
                    "({x}, {y}) is in the band and holds {found:?} rather than the layer above, so \
                     the layers drew in the wrong order or through the wrong blocks"
                ));
            }
            if found == frame::COLORS[1 - region] {
                return Err(format!(
                    "({x}, {y}) is in half {region} and holds half {}'s color, so the references \
                     are crossed",
                    1 - region
                ));
            }
            if found == frame::ABOVE_COLOR {
                return Err(format!(
                    "({x}, {y}) is outside the band and holds the layer above, so that layer was \
                     placed by another layer's matrix -- which is what binding one set for every \
                     layer does"
                ));
            }
            return Err(format!(
                "({x}, {y}) holds {found:?}, which is nothing this frame draws"
            ));
        }
    }
    let row = frame::SIDE as usize;
    let band = (frame::SIDE / u32::from(frame::BAND_OF)) as usize * row;
    let sampled = (frame::SIDE / u32::from(frame::TOP_OF)) as usize * row;
    let half = (row * row - band - sampled) / 2;
    if seen != [half, half, band, sampled] {
        return Err(format!(
            "{seen:?} texels, wanted {half} in each half, {band} in the band and {sampled} sampled"
        ));
    }
    println!(
        "  the regions           ok   {half} a half, {band} in the band, {sampled} sampled, on \
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
