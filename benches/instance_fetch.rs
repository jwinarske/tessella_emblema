// SPDX-License-Identifier: BSD-2-Clause
//! What `firstInstance` reaches on a real device: the builtin, and the attribute fetch with it.
//!
//! [`tessella_emblema::draws`] puts a drawable's slot in `firstInstance`, because every body reads
//! it as `ubo_index` and a draw parameter costs nothing where `vkCmdPushConstants` costs 0.9 us a
//! draw on V3D. That holds for all eighteen families drawn today, and every one of them draws a
//! single instance.
//!
//! The instanced wall family is the first that would draw more than one, and its body needs
//! `@builtin(instance_index)` to be *the instance* -- mbgl reads `instance[gl_InstanceIndex]` and
//! `[gl_InstanceIndex + 1]` out of the same buffer it binds as instance-rate vertex input. Whether
//! the slot can go on riding `firstInstance` turns on one question about the device: does
//! `firstInstance` shift only the builtin, or the address an instance-rate attribute is fetched
//! from as well?
//!
//! # Why this is measured rather than read
//!
//! The spec answers it, but at one remove. `VK_EXT_vertex_attribute_divisor` gives the
//! instance-rate address as `firstInstance + (instanceIndex - firstInstance) / divisor`, which for
//! the divisor of one that every binding here has reduces to `instanceIndex` -- the `InstanceIndex`
//! builtin itself, `firstInstance` included. So the two uses are not independent and the slot
//! cannot ride `firstInstance` for a family with instance-rate attributes.
//!
//! That is a design-deciding conclusion drawn from a formula written for a different feature, and
//! `src/lib.rs` states the slot's travel as settled fact. So it is drawn here instead, from a
//! device, twice over and two different ways:
//!
//! * `by_color` draws one instance at `firstInstance` zero through three and reports which
//!   element of the instance buffer the attribute read, beside what the builtin held.
//! * `by_column` draws three instances at `firstInstance` one, each placing a one-pixel column at
//!   the column its *fetched* element names. Which columns light says which elements were read,
//!   through the position rather than through a color -- a second channel for the same question,
//!   because a wrong reading here moves the whole port.
//!
//! The instance buffer holds more elements than any draw reads, so a shift lands on a real element
//! rather than past the end: the measurement must not depend on out-of-range fetch behavior, which
//! is exactly what the port is being designed to avoid.
//!
//! # What the two drivers said
//!
//! Identical readings, 2026-10-12:
//!
//! | driver | one instance at `firstInstance` k | three instances from one |
//! | --- | --- | --- |
//! | RADV `RAPHAEL_MENDOCINO` | element k, builtin k | elements 1, 2, 3 |
//! | V3D 7.1.7.0 (Mesa 26.2.2) | element k, builtin k | elements 1, 2, 3 |
//!
//! And the control held on both: the same buffer read per vertex stayed on element zero through
//! all four draws while the builtin moved under it. So the shift belongs to the instance rate and
//! not to the draw call.
//!
//! The Adreno 640 on the sa8155p cannot answer: the device open asks for dynamic rendering, which
//! that driver has neither as a feature nor as `VK_KHR_dynamic_rendering`, so the bench prints
//! `skipping` there rather than a number. Two agreeing drivers and the spec are what this rests
//! on, and the conclusion is a constraint rather than a value -- a third driver could only widen
//! it by disagreeing.
//!
//! # What follows from it
//!
//! [`tessella_emblema::draws::indexed`] refuses a non-zero slot on a draw of more than one
//! instance, rather than recording a draw that is wrong in two ways at once. The instanced wall
//! family will pass a slot of zero and carry `ubo_index` some other way.
//!
//! Run with `cargo bench --bench instance_fetch`.

mod common;

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Blend, Key, Slot, Targets};
use tessella_emblema::surface::Surface;
use tessella_emblema::target::{self, Depth, Host};
use tessella_vk::{Buffer, Image, ImageView, Memory, Recorder};

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// The target's edge, in pixels. Wide enough for one column per element a draw could reach.
const SIDE: u32 = 16;

/// The color format, in the channel order the readback is interpreted in.
const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Black, so an unlit column is unambiguous.
const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Elements in the instance buffer, each holding its own ordinal.
///
/// More than any draw below reads. A shift of three on a draw of three has to land on element five
/// and not past the end: what this bench measures must not be entangled with what an out-of-range
/// instance fetch does, which is the behavior the whole port exists to stay inside of.
const ELEMENTS: u32 = 8;

/// A unit quad as six vertices, which needs no index buffer.
const QUAD: [[f32; 2]; 6] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

/// Reports the element the attribute fetched and the builtin, in the first two channels.
///
/// Flat rather than interpolated: the three vertices of a triangle carry the same value, so the
/// interpolation would be exact anyway, and saying so means the pixel cannot be an artifact of how
/// a driver interpolates an integer it converted to a float.
const BY_COLOR: &str = r"
struct In {
    @builtin(instance_index) instance_index: u32,
    @location(0) corner: vec2<f32>,
    @location(1) tag: u32,
}

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) fetched: u32,
    @location(1) @interpolate(flat) builtin: u32,
}

@vertex
fn vertex_main(in: In) -> Out {
    var out: Out;
    out.clip = vec4<f32>(in.corner * 2.0 - 1.0, 0.0, 1.0);
    out.fetched = in.tag;
    out.builtin = in.instance_index;
    return out;
}

@fragment
fn fragment_main(in: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(in.fetched) / 255.0, f32(in.builtin) / 255.0, 0.0, 1.0);
}
";

/// Places a one-pixel column at the column the fetched element names.
///
/// The quad spans `[tag, tag + 1)` in pixels, so it covers the center of column `tag` and of no
/// other. Nothing here reads the builtin: this half asks the same question through the position,
/// where the other half asks it through a color.
fn column_module(side: u32) -> String {
    format!(
        r"
struct In {{
    @location(0) corner: vec2<f32>,
    @location(1) tag: u32,
}}

struct Out {{
    @builtin(position) clip: vec4<f32>,
}}

@vertex
fn vertex_main(in: In) -> Out {{
    let x = (f32(in.tag) + in.corner.x) / {side}.0 * 2.0 - 1.0;
    let y = in.corner.y * 2.0 - 1.0;
    var out: Out;
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {{
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}}
"
    )
}

/// Compiles a module to SPIR-V, as the other device benches do.
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

/// The two bindings: a quad per vertex, and an ordinal at the rate under test.
///
/// The binding number is the `@location`, which is what [`Slot`] is throughout this crate. `rate`
/// is the whole subject: the same module and the same buffer, read per instance for the
/// measurement and per vertex for the control.
fn layout_slots(rate: vk::VertexInputRate) -> Vec<Slot> {
    vec![
        Slot {
            slot: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
            stride: 8,
            rate: vk::VertexInputRate::VERTEX,
        },
        Slot {
            slot: 1,
            format: vk::Format::R32_UINT,
            offset: 0,
            stride: 4,
            rate,
        },
    ]
}

/// A host-visible buffer holding `bytes`, for the vertex and instance streams.
fn upload<'d>(
    open: &'d Open,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(Buffer<'d>, Memory<'d>), String> {
    let gpu = open.gpu();
    let length = bytes.len() as u64;
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

/// The target, its readback, and the two streams a draw binds.
struct Rig<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
    readback: Buffer<'d>,
    read_memory: Memory<'d>,
    depth: Depth<'d>,
    targets: Targets,
    quad: Buffer<'d>,
    _quad_memory: Memory<'d>,
    instances: Buffer<'d>,
    _instance_memory: Memory<'d>,
}

impl<'d> Rig<'d> {
    fn new(open: &'d Open) -> Result<Self, String> {
        let gpu = open.gpu();
        let image = gpu
            .image(
                SIDE,
                SIDE,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("target image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("target memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("bind target: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("target view: {why}"))?;

        let bytes = u64::from(SIDE) * u64::from(SIDE) * 4;
        let readback = gpu
            .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
            .map_err(|why| format!("readback buffer: {why}"))?;
        let needs = [readback.requirements()];
        let read_memory = gpu
            .allocate(
                bytes.max(needs[0].size),
                &needs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|why| format!("readback memory: {why}"))?;
        read_memory
            .bind(&readback, 0)
            .map_err(|why| format!("bind readback: {why}"))?;

        // Stencil only, with the compare mask the unclipped path uses: this bench is about an
        // address, not about where a tile's mask lets a draw land.
        let depth_stencil = device::depth_stencil_format(Attachment::StencilOnly, |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
        let depth = Depth::new(gpu, SIDE, SIDE, depth_stencil)
            .map_err(|why| format!("depth attachment: {why}"))?;

        let corners: Vec<u8> = QUAD
            .iter()
            .flat_map(|corner| corner.iter().flat_map(|at| at.to_le_bytes()))
            .collect();
        let (quad, quad_memory) = upload(open, &corners, vk::BufferUsageFlags::VERTEX_BUFFER)?;

        // Each element holds its own ordinal, so the value a fetch returns *is* the element it
        // came from. Nothing else distinguishes them, which is what makes the reading unambiguous.
        let ordinals: Vec<u8> = (0..ELEMENTS).flat_map(u32::to_le_bytes).collect();
        let (instances, instance_memory) =
            upload(open, &ordinals, vk::BufferUsageFlags::VERTEX_BUFFER)?;

        Ok(Self {
            image,
            view,
            _memory: memory,
            readback,
            read_memory,
            depth,
            targets: Targets {
                color: COLOR,
                depth_stencil,
                attachment: Attachment::StencilOnly,
            },
            quad,
            _quad_memory: quad_memory,
            instances,
            _instance_memory: instance_memory,
        })
    }

    /// Draws `instances` instances from `first_instance` and copies the target back.
    ///
    /// The copy is in the same submission as the draw, so the fence covers both -- a wait that
    /// covered only the draw would map a buffer the copy had not reached.
    fn draw(
        &self,
        open: &Open,
        pipeline: vk::Pipeline,
        instances: u32,
        first_instance: u32,
    ) -> Result<(), String> {
        open.submit(|record: Recorder<'_>| {
            target::frame(
                record,
                Host {
                    image: &self.image,
                    view: &self.view,
                    width: SIDE,
                    height: SIDE,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &self.depth,
                Some(CLEAR),
                |record| {
                    record.bind_pipeline(pipeline);
                    record.stencil_compare_mask(0);
                    record.stencil_reference(0);
                    // Declared dynamic for the 3D draw-once's sake, so it has to be set here too.
                    record.stencil_write_mask(0);
                    record.bind_vertex_buffers(
                        0,
                        &[self.quad.raw(), self.instances.raw()],
                        &[0, 0],
                    );
                    record.draw(QUAD.len() as u32, instances, first_instance);
                },
            );
            record.transition(
                &self.image,
                target::LEAVES_IN,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            record.copy_to_buffer(&self.image, &self.readback, SIDE, SIDE);
        })
    }

    /// One pixel of the last draw, in the target's channel order.
    fn pixel(&self, x: u32, y: u32) -> Result<[u8; 4], String> {
        let mapping = self
            .read_memory
            .map()
            .map_err(|why| format!("map readback: {why}"))?;
        let mut pixel = [0u8; 4];
        let at = (u64::from(y) * u64::from(SIDE) + u64::from(x)) * 4;
        mapping
            .read(at, &mut pixel)
            .map_err(|why| format!("read pixel: {why}"))?;
        Ok(pixel)
    }
}

/// A pipeline over [`layout_slots`], for one module and one rate.
///
/// The layout comes back with it because a pipeline must not outlive the layout it was built
/// against, which the borrow makes the compiler's problem rather than a comment's.
fn pipeline<'d>(
    open: &'d Open,
    rig: &Rig<'d>,
    source: &str,
    rate: vk::VertexInputRate,
) -> Result<(tessella_vk::Pipeline<'d>, pipelines::Layout<'d>), String> {
    let gpu = open.gpu();
    let words = compile(source)?;
    let module = gpu
        .shader(&words)
        .map_err(|why| format!("shader module: {why}"))?;
    // No descriptors at all: this measures an address, and a block would only add a resource whose
    // own binding could be wrong instead.
    let layout = pipelines::layout(gpu, &[]).map_err(|why| format!("layout: {why}"))?;
    let key = Key {
        shader: BuiltIn::FillExtrusionInstancedShader as i32,
        surface: Surface::Plane,
        permutation: 0,
        layout: layout_slots(rate),
        blend: Blend::Unblended,
        depth: pipelines::Depth::Write,
        enabled: pipelines::Enabled::ALL,
        stencil: pipelines::Stencil::Clip,
    };
    let built = pipelines::build(gpu, &key, &layout, &module, rig.targets)
        .map_err(|why| format!("pipeline: {why:?}"))?;
    Ok((built, layout))
}

/// One instance at a time: which element did the attribute read, and what did the builtin hold?
fn by_color(open: &Open) -> Result<(), String> {
    let rig = Rig::new(open)?;
    let (built, _layout) = pipeline(open, &rig, BY_COLOR, vk::VertexInputRate::INSTANCE)?;

    let mut shifted = Vec::new();
    for first in 0..4u32 {
        rig.draw(open, built.raw(), 1, first)?;
        let pixel = rig.pixel(SIDE / 2, SIDE / 2)?;
        let (fetched, builtin) = (u32::from(pixel[0]), u32::from(pixel[1]));
        println!("    firstInstance {first}  attribute element {fetched}  builtin {builtin}");
        shifted.push((first, fetched, builtin));
    }

    for (first, fetched, builtin) in shifted {
        if fetched != first {
            return Err(format!(
                "firstInstance {first} fetched element {fetched}: the instance-rate address does \
                 not follow firstInstance, so the slot could ride it after all"
            ));
        }
        if builtin != first {
            return Err(format!(
                "firstInstance {first} gave builtin {builtin}: instance_index is not firstInstance \
                 plus the instance"
            ));
        }
    }
    println!("  by_color             ok   both follow firstInstance");
    Ok(())
}

/// Three instances from element one: which columns light?
fn by_column(open: &Open) -> Result<(), String> {
    /// The instance to start at. Non-zero, which the case checks rather than assumes.
    const FIRST: u32 = 1;
    /// Instances to draw, fewer than [`ELEMENTS`] leaves room for even shifted.
    const COUNT: u32 = 3;

    let rig = Rig::new(open)?;
    let (built, _layout) = pipeline(
        open,
        &rig,
        &column_module(SIDE),
        vk::VertexInputRate::INSTANCE,
    )?;

    let unshifted: Vec<u32> = (0..COUNT).collect();
    // Written as an offset from the unshifted answer rather than as `FIRST..FIRST + COUNT`, which
    // clippy would have as `FIRST..=COUNT` -- equal only because `FIRST` is one, and wrong for
    // any other value.
    let wanted: Vec<u32> = unshifted.iter().map(|at| FIRST + at).collect();
    // The inputs have to make the two answers different columns, or the comparison below passes
    // whichever is true. A first instance of zero is exactly that case, and it is the one a
    // later edit would reach for as a simplification.
    if wanted == unshifted {
        return Err(format!(
            "firstInstance {FIRST} makes the shifted and unshifted answers the same columns"
        ));
    }

    rig.draw(open, built.raw(), COUNT, FIRST)?;
    let mut lit = Vec::new();
    for column in 0..ELEMENTS {
        if rig.pixel(column, SIDE / 2)?[0] > 0 {
            lit.push(column);
        }
    }
    println!(
        "    {COUNT} instances from firstInstance {FIRST}  columns lit {lit:?}  \
         unshifted would be {unshifted:?}"
    );
    if lit != wanted {
        return Err(format!(
            "columns {lit:?} lit where elements {wanted:?} were expected"
        ));
    }
    println!("  by_column            ok   elements {wanted:?} were the ones read");
    Ok(())
}

/// The control: the same buffer read per vertex, which must not follow `firstInstance`.
///
/// Without it the measurement is one-sided. "The element fetched equals `firstInstance`" would
/// also be what a device that shifted *every* vertex fetch by `firstInstance` produced, and that
/// device would break the eighteen families drawn today rather than only the instanced one. So the
/// same module and the same buffer are read at the vertex rate and the four draws compared to each
/// other.
///
/// Which element the vertex rate lands on is not asserted, only that it does not move. The flat
/// value comes from a triangle's provoking vertex and the pixel sits near the quad's diagonal, so
/// *which* vertex provokes it is a tie this bench has no reason to break -- while the four draws
/// agreeing on it is the whole of what the control has to say.
fn per_vertex_control(open: &Open) -> Result<(), String> {
    let rig = Rig::new(open)?;
    let (built, _layout) = pipeline(open, &rig, BY_COLOR, vk::VertexInputRate::VERTEX)?;

    let mut seen = Vec::new();
    for first in 0..4u32 {
        rig.draw(open, built.raw(), 1, first)?;
        let pixel = rig.pixel(SIDE / 2, SIDE / 2)?;
        seen.push((first, u32::from(pixel[0]), u32::from(pixel[1])));
    }
    for (first, fetched, builtin) in &seen {
        println!("    firstInstance {first}  vertex element {fetched}  builtin {builtin}");
    }

    let first_fetched = seen[0].1;
    if let Some((first, fetched, _)) = seen
        .iter()
        .find(|(_, fetched, _)| *fetched != first_fetched)
    {
        return Err(format!(
            "firstInstance {first} moved the vertex-rate fetch from element {first_fetched} to \
             {fetched}, so the shift is not particular to the instance rate"
        ));
    }
    // And the builtin still follows it, which is what says this draw differed from the others only
    // in the binding's rate.
    if seen.iter().any(|(first, _, builtin)| builtin != first) {
        return Err(
            "the builtin stopped following firstInstance when the rate changed".to_string(),
        );
    }
    println!("  per_vertex_control   ok   element {first_fetched} every time, builtin still moves");
    Ok(())
}

fn main() {
    let device = match Open::preferred() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping: {why}");
            return;
        }
    };
    println!("device: {} ({})", device.name, device.class);

    let mut failed = 0;
    for (name, case) in [
        ("by_color", by_color as Case),
        ("by_column", by_column),
        ("per_vertex_control", per_vertex_control),
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
