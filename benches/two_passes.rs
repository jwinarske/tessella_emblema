// SPDX-License-Identifier: BSD-2-Clause
//! A view drawn into a texture, and the view that samples it.
//!
//! The drawing half of DR-25, and of #93's fourth item. tessella#368 made `host` read `ViewTarget`;
//! this is a frame that acts on it: the child's pass renders into an image held under a `TextureId`
//! that no upload will ever name, and the parent's pass samples it.
//!
//! Two of the eighteen families are this shape -- `heatmap` draws its kernels into a half-resolution
//! target and reads it back through a color ramp, and `hillshade_prepare` is the same -- so neither
//! could draw at all before. `benches/first_pixel.rs` has a case for each, and each is a *single*
//! pass that says nothing about the hand-off between them.
//!
//! # What a pass proves
//!
//! That `Host::feeding` names the view to draw first, that `Images::declare_target` gives an image a
//! pass can render into *and* a sampler can read, and that the order between the two passes holds:
//! the parent's pixels are the color the child drew. A parent drawn first samples an image whose
//! contents are undefined, which is a different pixel.
//!
//! # What it does not catch
//!
//! The transition of the child's target to `SHADER_READ_ONLY_OPTIMAL` before the parent samples it.
//! Removing it changes nothing on RADV *or* on V3D 7.1.7.0 -- `target::LEAVES_IN` is `GENERAL`, and
//! `GENERAL` is a legal layout to sample from, so the only thing wrong is that the descriptor says
//! `SHADER_READ_ONLY_OPTIMAL` while the image is in `GENERAL`. That is invalid usage no driver has
//! to report, the same position as the stencil reference in #83 and the pipeline's declared depth
//! format in #90: the transition is required and no bench can be what requires it.
//!
//! Run with `cargo bench --bench two_passes`.

mod common;

#[path = "common/frame.rs"]
mod frame;

#[path = "common/passes.rs"]
mod passes;

use ash::vk;
use tessella_capture_abi::envelope::{Rect16, ViewId};
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
use tessella_emblema::target::{self, Depth, Host as Attach};
use tessella_emblema::{
    blocks, buffers, descriptors, families, pipelines, record, shaders, vertices,
};
use tessella_vk::{Buffer, Image, ImageView, Memory, Recorder};

use common::Open;

/// What the screen is cleared to, which neither the child's color nor the parent's image is.
const CLEAR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];

/// The same, as the target's bytes.
const CLEARED: [u8; 4] = [51, 102, 153, 255];

/// The color format both the screen and the child's target are.
const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

fn main() {
    match run() {
        Ok(()) => {}
        Err(why) => {
            eprintln!("two passes: {why}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let (mut ring, geometry) = passes::write(1 << 16);
    let mut host = Host::new();
    let progress = host.read(ring.consumer());
    if progress.unknown != 0 || progress.malformed != 0 || progress.undeclared != 0 {
        return Err(format!(
            "{} unknown, {} malformed and {} undeclared of {} records",
            progress.unknown, progress.malformed, progress.undeclared, progress.records
        ));
    }

    // The relation, before any device: a frame that did not read it has nothing to draw first.
    let feeding: Vec<ViewId> = host.feeding(passes::PARENT).collect();
    if feeding != [passes::CHILD] {
        return Err(format!("{feeding:?} feed the parent, wanted the child"));
    }
    let found = host
        .target(passes::CHILD)
        .copied()
        .ok_or("the child has no target")?;
    if found.texture != passes::TARGET {
        return Err(format!("the child draws into {:?}", found.texture));
    }
    let size = found.size(tessella_capture_abi::envelope::Extent {
        width: passes::SIDE,
        height: passes::SIDE,
    });
    if size.width != passes::SIDE / 2 || size.height != passes::SIDE / 2 {
        return Err(format!("the child is {}x{}", size.width, size.height));
    }
    if !host.feeding(passes::CHILD).collect::<Vec<_>>().is_empty() {
        return Err("the child feeds something, so nesting was accepted".into());
    }
    println!(
        "  the relation          ok   {} records, the child feeds the parent at {}x{}",
        progress.records, size.width, size.height
    );

    let device = match Open::preferred() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping the draw: {why}");
            return Ok(());
        }
    };
    println!("  device: {} ({})", device.name, device.class);
    draw(&device, &mut host, &geometry, size)
}

/// Puts both passes on the device, records them in one submission, and reads the screen back.
#[allow(clippy::too_many_lines)]
fn draw(
    device: &Open,
    host: &mut Host,
    geometry: &frame::Geometry,
    child: tessella_capture_abi::envelope::Extent,
) -> Result<(), String> {
    let gpu = device.gpu();
    let screen = Screen::new(device, passes::SIDE, passes::SIDE)?;
    let depth = Depth::new(gpu, child.width, child.height, screen.depth_stencil)
        .map_err(|why| format!("the child's depth attachment: {why}"))?;

    // The child's target: an image under a `TextureId` that no upload will ever name. Declared from
    // what the stream said -- the size from the parent's through `Target::size`, the format from the
    // record -- and never from anything this bench chose.
    let found = host
        .target(passes::CHILD)
        .copied()
        .ok_or("the child has no target")?;
    let mut images = Images::new();
    images
        .declare_target(gpu, found.texture, child, found.format, found.channel)
        .map_err(|why| format!("the child's target: {why}"))?;

    // And the parent's second image, which *is* uploaded.
    let mut staged: Result<(), String> = Ok(());
    device.submit(|record: Recorder<'_>| {
        staged = stage(&mut images, gpu, record, host);
    })?;
    staged?;

    // Both views' blocks. Keyed by the view as well as the layer, which is why two views can both
    // draw in layer zero without one reading the other's.
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
        let at = blocks::Which {
            view: *view,
            layer: *layer_index,
        };
        blocks
            .declare(gpu, at, *slot, data.len() / 16, 16)
            .map_err(|why| format!("slot {slot}: {why}"))?;
        blocks
            .replace(at, *slot, data)
            .map_err(|why| format!("slot {slot}: {why}"))?;
        blocks
            .flush(at, *slot, 16)
            .map_err(|why| format!("slot {slot}: {why}"))?;
    }

    // Both geometries, each planned against its own family's table from the descriptors the stream
    // carried.
    let mut store = Store::new();
    let mut plans = std::collections::BTreeMap::new();
    for (view, id) in [
        (passes::CHILD, passes::DRAWN),
        (passes::PARENT, passes::SAMPLES),
    ] {
        let drawable = host
            .joiner()
            .drawable(id, view)
            .ok_or_else(|| format!("geometry {} is not used by its view", id.0))?;
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

    let mut cache = Cache::new();
    let fill = families::family(BuiltIn::FillShader).ok_or("no fill family")?;
    let raster = families::family(BuiltIn::RasterShader).ok_or("no raster family")?;
    let fill_bindings = pipelines::bindings(fill, Surface::Plane);
    let raster_bindings = pipelines::bindings(raster, Surface::Plane);
    let widest = if raster_bindings.len() > fill_bindings.len() {
        &raster_bindings
    } else {
        &fill_bindings
    };
    let mut sets = Sets::new(gpu, 2, widest).map_err(|why| format!("the pool: {why}"))?;

    // The child's program and set, written while its layout is held.
    let child_program = {
        let words = compile_family(fill)?;
        let key = pipelines::key(
            BuiltIn::FillShader,
            Surface::Plane,
            0,
            plans.get(&passes::DRAWN).ok_or("the child has no plan")?,
            Blend::Alpha,
        );
        // Against the *child's* formats, which are its own: a pipeline names the formats it renders
        // into, and the child renders into its target rather than onto the screen. They happen to
        // agree here -- both `R8G8B8A8_UNORM` with the same depth-stencil -- and a case where they
        // did not would need two `Targets` and two pipelines of the same family.
        let pipeline = cache
            .pipeline(gpu, &key, &fill_bindings, &words, screen.targets)
            .map_err(|why| format!("the child's pipeline: {why}"))?;
        let layout = cache
            .layout(gpu, BuiltIn::FillShader, Surface::Plane, &fill_bindings)
            .map_err(|why| format!("the child's layout: {why}"))?;
        let at = blocks::Which {
            view: passes::CHILD,
            layer: passes::LAYER,
        };
        sets.write(layout, &fill_bindings, at, &blocks, &[])
            .map_err(|why| format!("the child's set: {why}"))?;
        Program {
            pipeline,
            layout: layout.pipeline(),
            instances: 1,
        }
    };

    // The parent's, binding the child's target as its first image.
    let parent_program = {
        let words = compile_family(raster)?;
        let key = pipelines::key(
            BuiltIn::RasterShader,
            Surface::Plane,
            0,
            plans.get(&passes::SAMPLES).ok_or("no parent plan")?,
            Blend::Alpha,
        );
        let pipeline = cache
            .pipeline(gpu, &key, &raster_bindings, &words, screen.targets)
            .map_err(|why| format!("the parent's pipeline: {why}"))?;
        let layout = cache
            .layout(gpu, BuiltIn::RasterShader, Surface::Plane, &raster_bindings)
            .map_err(|why| format!("the parent's layout: {why}"))?;

        let refs = &host
            .joiner()
            .drawable(passes::SAMPLES, passes::PARENT)
            .ok_or("the parent's geometry is not used")?
            .geometry
            .texture_refs;
        // The target has to be the image the raster's *first* sampler reads, because that is the
        // one the body samples. Asked by slot rather than by position in the run: that is what
        // `bound_from` places by, so a run in another order is bound the same way and this guard
        // follows it instead of fixing the order in place.
        let wants = raster_bindings
            .iter()
            .find(|b| b.kind == pipelines::Kind::SampledImage)
            .and_then(|b| b.slot)
            .ok_or("the raster set declares no texture")?;
        let at_first = refs.iter().find(|r| r.slot == wants).map(|r| r.texture);
        if at_first != Some(passes::TARGET) {
            return Err(format!(
                "the raster's first sampler is slot {wants}, which the parent names {at_first:?} \
                 rather than the target"
            ));
        }
        let bound = descriptors::bound_from(&images, &raster_bindings, refs)
            .map_err(|why| format!("the parent's textures: {why}"))?;
        let at = blocks::Which {
            view: passes::PARENT,
            layer: passes::LAYER,
        };
        sets.write(layout, &raster_bindings, at, &blocks, &bound)
            .map_err(|why| format!("the parent's set: {why}"))?;
        Program {
            pipeline,
            layout: layout.pipeline(),
            instances: 1,
        }
    };

    // Both views planned, each `Frame` dropped before the next: `plan` takes the host mutably and
    // hands back references into it, so two cannot be held. `Host::batches` reads them back
    // afterwards, which is what lets one recording carry both passes.
    for view in [passes::CHILD, passes::PARENT] {
        if host.plan(view).is_none() {
            return Err(format!(
                "view {} has no order and camera that agree",
                view.0
            ));
        }
    }

    // One submission, both passes, in the order `feeding` gives: the child first, then the image it
    // wrote transitioned for sampling, then the parent.
    let (target_image, target_view) = images
        .rendered(found.texture)
        .ok_or("the child's target was not held")?;
    let mut counts = Err("the recording did not run".to_string());
    device.submit(|record: Recorder<'_>| {
        counts = (|| -> Result<(usize, usize), String> {
            let child_counts = pass(
                record,
                host,
                &store,
                &sets,
                passes::CHILD,
                child_program,
                Attach {
                    image: target_image,
                    view: target_view,
                    width: child.width,
                    height: child.height,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &depth,
            )?;
            record.transition(
                target_image,
                target::LEAVES_IN,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            let parent_counts = pass(
                record,
                host,
                &store,
                &sets,
                passes::PARENT,
                parent_program,
                Attach {
                    image: &screen.image,
                    view: &screen.view,
                    width: passes::SIDE,
                    height: passes::SIDE,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &screen.depth,
            )?;
            record.transition(
                &screen.image,
                target::LEAVES_IN,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            record.copy_to_buffer(&screen.image, &screen.readback, passes::SIDE, passes::SIDE);
            Ok((child_counts, parent_counts))
        })();
    })?;
    let (drew_child, drew_parent) = counts?;
    if drew_child != 1 || drew_parent != 1 {
        return Err(format!(
            "{drew_child} drawables in the child's pass and {drew_parent} in the parent's"
        ));
    }
    println!("  the two passes        ok   one drawable each, the child's recorded first");

    screen_holds(&screen)
}

/// Records one view's frame into the attachment it draws.
#[allow(clippy::too_many_arguments)]
fn pass(
    record: Recorder<'_>,
    host: &Host,
    store: &Store<'_>,
    sets: &Sets<'_>,
    view: ViewId,
    program: Program,
    attach: Attach<'_>,
    depth: &Depth<'_>,
) -> Result<usize, String> {
    // Planned through the host, so the batches are the stream's rather than this bench's. Taken
    // before the scope opens, because `plan` borrows the host mutably and the scope borrows the
    // recorder -- the two are easier to keep apart than to nest.
    let batches = host
        .batches(view)
        .ok_or_else(|| format!("view {} has no batches", view.0))?;
    let partition = host.clips(view).partition();
    let mut counts = Err("the recording did not run".to_string());
    target::frame(record, attach, depth, Some(CLEAR), |record| {
        counts = record::content(
            record,
            batches,
            &record::Scene {
                store,
                sets,
                joiner: host.joiner(),
                partition,
                view,
            },
            &|_| Some(program),
        )
        .map_err(|why| format!("view {}: {why}", view.0));
    });
    Ok(counts?.drawables)
}

/// The screen: what the parent draws onto, and the buffer it is read back out of.
struct Screen<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
    readback: Buffer<'d>,
    read_memory: Memory<'d>,
    depth: Depth<'d>,
    depth_stencil: vk::Format,
    targets: Targets,
}

impl<'d> Screen<'d> {
    fn new(open: &'d Open, width: u32, height: u32) -> Result<Self, String> {
        let gpu = open.gpu();
        let image = gpu
            .image(
                width,
                height,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("the screen image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("the screen memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("binding the screen: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("the screen view: {why}"))?;

        let bytes = u64::from(width) * u64::from(height) * 4;
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

        let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
        let depth = Depth::new(gpu, width, height, depth_stencil)
            .map_err(|why| format!("the screen's depth attachment: {why}"))?;

        Ok(Self {
            image,
            view,
            _memory: memory,
            readback,
            read_memory,
            depth,
            depth_stencil,
            targets: Targets {
                color: COLOR,
                depth_stencil,
                attachment: Attachment::DepthStencil,
            },
        })
    }
}

/// The screen holds the color the child drew, everywhere.
///
/// Three outcomes the pixels tell apart: the child's color is both passes in the right order; the
/// clear is the child's pass not having drawn, so the parent sampled an empty target; and the
/// parent's own second image is the two images bound the wrong way round.
fn screen_holds(screen: &Screen<'_>) -> Result<(), String> {
    let mapping = screen
        .read_memory
        .map()
        .map_err(|why| format!("mapping the readback: {why}"))?;
    let mut pixels = vec![0u8; (passes::SIDE * passes::SIDE * 4) as usize];
    mapping
        .read(0, &mut pixels)
        .map_err(|why| format!("reading the screen: {why}"))?;

    let mut held = 0usize;
    for y in 0..passes::SIDE {
        for x in 0..passes::SIDE {
            let at = ((y * passes::SIDE + x) * 4) as usize;
            let found = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];
            if found == passes::CHILD_COLOR {
                held += 1;
                continue;
            }
            if found == CLEARED {
                return Err(format!(
                    "({x}, {y}) is the clear, so the parent sampled a target the child had not \
                     drawn into -- which is what drawing the passes in the other order gives"
                ));
            }
            return Err(format!(
                "({x}, {y}) holds {found:?}, which is neither what the child drew nor the clear"
            ));
        }
    }
    let every = (passes::SIDE * passes::SIDE) as usize;
    if held != every {
        return Err(format!("{held} of {every} texels hold the child's color"));
    }
    println!("  the screen            ok   {held} texels of the color the child drew");
    Ok(())
}

/// Uploads the textures the stream sent. The child's target is not one of them.
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
                pixels
                    .get(at.at + usize::from(row) * at.stride..)
                    .and_then(|rest| rest.get(..width))
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

/// One family's module, compiled.
fn compile_family(family: &families::Family) -> Result<Vec<u32>, String> {
    let source = shaders::module(
        Surface::Plane,
        family.blocks,
        family.attributes,
        family.textures,
        family.body,
    )
    .map_err(|why| format!("{} does not assemble: {why:?}", family.name))?;
    let parsed = naga::front::wgsl::parse_str(&source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(&source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spirv: {why}"))
}
