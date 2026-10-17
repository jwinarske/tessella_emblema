//! What shading a covered opaque layer costs, and what a depth early-out would reclaim.
//!
//! Run with `cargo bench --bench overdraw_cost`.
//!
//! # The question
//!
//! §11.7 asks a consumer to "honor the opaque/translucent split -- opaque layers front-to-back
//! with depth writes per `pass` + `opaquePassCutoff`, or TBDR parts eat full-screen overdraw per
//! layer". This pass does not: every draw lands at the same depth, so the depth buffer orders
//! nothing and a covered opaque layer is shaded anyway.
//!
//! #103 is the issue for fixing that, and it is deliberately gated on this measurement. The cost
//! of the fix is known to be near zero on the gating target -- V3D 7.1.7.0 does not support
//! `FORMAT_S8_UINT` at all, so every view there already takes `D24_UNORM_S8_UINT` and already
//! carries depth -- but the *saving* was not measured, and adding depth state to defend a number
//! nobody has is the wrong order of work.
//!
//! # What is measured
//!
//! `LAYERS` full-target triangles, every one covering every pixel, in three arms that differ only
//! in the depth state and in what each draw's `firstInstance` carries:
//!
//! | arm | attachment | depth per layer | what it stands for |
//! | --- | --- | --- | --- |
//! | `test-off` | `StencilOnly` | n/a, test disabled | this pass today |
//! | `one-depth` | `DepthStencil` | all at zero | depth enabled and ordering nothing |
//! | `per-layer` | `DepthStencil` | its own, front to back | the early-out #103 proposes |
//!
//! `one-depth` is the control that separates two things a single comparison would confuse: what
//! enabling the depth test costs, and what distinct depths then save. Without it, `per-layer`
//! against `test-off` is a difference with two causes in it.
//!
//! Zero layers is the harness floor. `Open::submit` builds a command pool per call, so every row
//! carries that, and a row minus the zero row is what the draws cost.
//!
//! # What this is not
//!
//! A representative fragment shader. This one returns a constant, so a layer costs a triangle
//! setup and a write per pixel and nothing else -- which makes every number here a **floor** on
//! what a real layer costs and so a floor on what the early-out saves. A fill evaluating
//! data-driven paint properties costs more per fragment, not less.
//!
//! Nor a frame: one target, one pipeline, no clipping, no uploads. What it isolates is the price
//! of covering the same pixels twice.

// A bench is a program, and it reports by printing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::time::{Duration, Instant};

use ash::vk;
use tessella_capture_abi::envelope::Topology;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Blend, Key, Targets};
use tessella_emblema::surface::Surface;
use tessella_emblema::target::{self, Depth, Host as Attach};
use tessella_vk::Recorder;

use common::Open;

/// The target's side. A map frame, not a unit test: the quad sweep's own scenes are 1024 by 768,
/// and a saving measured over 4,096 pixels is not a saving.
const SIDE: u32 = 1024;

/// The color target's format, as every other bench here uses.
const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// How many covering layers each row draws. Zero is the harness floor.
const LAYERS: [u32; 6] = [0, 1, 2, 4, 8, 16];

/// Submissions per row, each carrying `REPEATS` frames. Reported as a median.
const SAMPLES: usize = 15;

/// The gap between one layer's depth and the next.
///
/// Small enough that sixteen layers stay well inside the near half of the range, and large enough
/// that no two land on one value after the depth buffer's quantization -- `D24_UNORM_S8_UINT` has
/// 24 bits, so a thirty-second is sixteen million times the smallest step it can hold.
const DEPTH_STEP: f32 = 1.0 / 32.0;

/// How much arithmetic a fragment does, in rounds.
///
/// Zero is the first version of this bench and is a floor rather than a measurement: with nothing
/// in the fragment stage, a rejected fragment saves almost nothing and the answer is about
/// rasterization instead. It read eighty per cent saved on RADV and four on V3D, which is not two
/// devices disagreeing -- it is one number that does not mean what it looks like.
///
/// The second weight is the one to read. `README.md` names the reason it matters most here: "a
/// fragment shader on V3D costs what its whole body needs, not what the taken branch needs", so a
/// real layer's shader is dear on the board that gates this and a rejection saves all of it.
/// Paired with how many frames to put in one submission at that weight.
///
/// `Open::submit` builds and destroys a command pool per call, and the first version of this
/// measured that instead of the draws: twenty-five milliseconds a submission against one to three
/// for the draws, so the floor swamped the signal and the savings came out negative. Repeating the
/// frame inside the one command buffer divides that fixed cost by the repeat count while leaving
/// the GPU work intact. Each repeat is its own `target::frame`, so each clears the depth buffer
/// and the test starts from one every time -- which is what makes the repeats equivalent rather
/// than cumulative.
///
/// The repeat count cannot be fixed across weights. At forty-eight rounds and two hundred and
/// fifty-six frames a submission, RADV lost the device -- "the CS has been cancelled because the
/// context is lost" -- because one submission's work ran past the driver's watchdog. A sixteen
/// layer frame at zero rounds is already a quarter of a second over that many repeats, so a weight
/// that multiplies the fragment cost has to divide the repeats.
const WORK: [(u32, u32); 2] = [(0, 256), (24, 8)];

/// A full-target triangle, with its depth carried in `firstInstance`.
///
/// No vertex buffer and no descriptors: the positions are computed from `vertex_index`, so the
/// pipeline's vertex layout is empty and nothing has to be uploaded or bound. `firstInstance`
/// reaching the shader as `instance_index` is what `draw_cost.rs` verifies before it measures
/// anything, and the same mechanism carries the layer here.
///
/// The triangle reaches past the target on two sides rather than being a quad of two triangles,
/// so one draw covers every pixel with no shared edge down the middle.
const MODULE: &str = r"
struct Output {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) layer: u32,
) -> Output {
    var xs = array<f32, 3>(-1.0, 3.0, -1.0);
    var ys = array<f32, 3>(-1.0, -1.0, 3.0);
    var out: Output;
    out.position = vec4<f32>(xs[vertex], ys[vertex], f32(layer) * {step}, 1.0);
    return out;
}

@fragment
fn fragment_main(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    // `{work}` rounds of arithmetic the compiler cannot fold away, because each depends on the
    // last and the first depends on the fragment's own position. What a rejected fragment does not
    // pay is exactly this.
    var acc = at.x * 0.001;
    for (var i = 0u; i < {work}u; i = i + 1u) {
        acc = fma(acc, 1.0001, 0.5);
        acc = sqrt(abs(acc)) + acc * 0.25;
    }
    return vec4<f32>(0.25, 0.5, 0.75, clamp(acc, 0.9, 1.0));
}
";

/// Which depth state a row is measured under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// The depth test disabled, which is this pass today for a view of flat layers.
    TestOff,
    /// The depth test enabled with every layer at the same depth.
    OneDepth,
    /// The depth test enabled with each layer at its own depth, drawn front to back.
    PerLayer,
}

impl Arm {
    const ALL: [Self; 3] = [Self::TestOff, Self::OneDepth, Self::PerLayer];

    fn name(self) -> &'static str {
        match self {
            Self::TestOff => "test-off",
            Self::OneDepth => "one-depth",
            Self::PerLayer => "per-layer",
        }
    }

    /// `StencilOnly` leaves the depth test disabled; `DepthStencil` enables it and the write.
    fn attachment(self) -> Attachment {
        match self {
            Self::TestOff => Attachment::StencilOnly,
            Self::OneDepth | Self::PerLayer => Attachment::DepthStencil,
        }
    }

    /// What a layer's draw puts in `firstInstance`, which is the depth the vertex stage computes.
    fn first_instance(self, layer: u32) -> u32 {
        match self {
            Self::TestOff | Self::PerLayer => layer,
            Self::OneDepth => 0,
        }
    }
}

fn main() {
    let open = match Open::preferred() {
        Ok(open) => open,
        Err(why) => {
            // Not a failure: a machine with no Vulkan cannot answer this.
            println!("no device to measure on: {why}");
            return;
        }
    };
    println!("device: {}", open.name);
    println!("target: {SIDE}x{SIDE}, {SAMPLES} submissions a row\n");

    match run(&open) {
        Ok(()) => {}
        Err(why) => {
            println!("  overdraw              FAIL {why}");
            std::process::exit(1);
        }
    }
}

fn run(open: &Open) -> Result<(), String> {
    let gpu = open.gpu();

    // One color target for every arm. The attachment *format* is the same throughout -- what the
    // arms differ in is the pipeline's depth state, not the image.
    let image = gpu
        .image(
            SIDE,
            SIDE,
            COLOR,
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
        .view(&image, COLOR)
        .map_err(|why| format!("the target view: {why}"))?;

    // Reported, because it is the whole cost argument in #103: where `S8_UINT` is not an
    // attachment format, a stencil-only view already carries depth and the early-out's attachment
    // is already paid for.
    let stencil_only = device::depth_stencil_format(Attachment::StencilOnly, |format| {
        open.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no stencil format: {why:?}"))?;
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        open.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    println!("stencil-only resolves to {stencil_only:?}");
    println!("depth-stencil resolves to {depth_stencil:?}");
    if stencil_only == depth_stencil {
        println!("  the same format, so a flat view here already carries depth\n");
    } else {
        println!("  different, so a flat view here pays for depth it does not use\n");
    }

    // No bindings at all: the module declares no blocks and no textures.
    let layout = pipelines::layout(gpu, &[]).map_err(|why| format!("the layout: {why}"))?;

    for (work, repeats) in WORK {
        println!("fragment work: {work} rounds, {repeats} frames a submission");
        let words = compile(work).map_err(|why| format!("the module: {why}"))?;
        let modules = pipelines::stage_modules(gpu, &words)
            .map_err(|why| format!("the shader module: {why}"))?;
        let rows = measure(open, gpu, &image, &view, &layout, &modules, repeats)?;
        report(&rows, repeats);
        println!();
    }
    Ok(())
}

/// Every arm at every layer count, interleaved.
fn measure(
    open: &Open,
    gpu: tessella_vk::Gpu<'_>,
    image: &tessella_vk::Image<'_>,
    view: &tessella_vk::ImageView<'_>,
    layout: &pipelines::Layout<'_>,
    modules: &[tessella_vk::ShaderModule<'_>; 2],
    repeats: u32,
) -> Result<Vec<(Arm, u32, Duration)>, String> {
    // Every arm's state built up front, because the sample loop interleaves them. Measuring one
    // arm to completion before the next put a machine-level change inside the comparison: in one
    // run of three, `one-depth` and `per-layer` came back uniformly four times faster than
    // `test-off`, floor included, which is not an effect of the depth state but of the machine
    // getting quieter partway through. Interleaved, such a shift lands on every arm at once.
    let mut built = Vec::new();
    for arm in Arm::ALL {
        let format = device::depth_stencil_format(arm.attachment(), |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("{}: no format: {why:?}", arm.name()))?;
        let depth = Depth::new(gpu, SIDE, SIDE, format)
            .map_err(|why| format!("{}: the depth attachment: {why}", arm.name()))?;
        let targets = Targets {
            color: COLOR,
            depth_stencil: format,
            attachment: arm.attachment(),
        };
        let key = Key {
            // Any family: the key's only effect on the built pipeline is its vertex layout, which
            // is empty here, so this names one rather than inventing a shader the tree does not
            // have.
            shader: BuiltIn::FillShader as i32,
            surface: Surface::Plane,
            topology: Topology::Triangles,
            permutation: 0,
            layout: Vec::new(),
            blend: Blend::Unblended,
            // Every arm writes; the attachment is what turns the test off in `test-off`.
            depth: pipelines::Depth::Write,
            enabled: pipelines::Enabled::ALL,
            stencil: pipelines::Stencil::Clip,
        };
        let pipeline = pipelines::build(gpu, &key, layout, pipelines::Stages::of(modules), targets)
            .map_err(|why| format!("{}: the pipeline: {why}", arm.name()))?;
        built.push((arm, depth, pipeline));
    }

    let mut samples: Vec<(Arm, u32, Vec<Duration>)> = built
        .iter()
        .flat_map(|(arm, _, _)| LAYERS.iter().map(move |layers| (*arm, *layers, Vec::new())))
        .collect();
    for _ in 0..SAMPLES {
        for (arm, depth, pipeline) in &built {
            for layers in LAYERS {
                let at = Instant::now();
                open.submit(|record: Recorder<'_>| {
                    for _ in 0..repeats {
                        let attach = Attach {
                            image,
                            view,
                            width: SIDE,
                            height: SIDE,
                            layout: vk::ImageLayout::UNDEFINED,
                        };
                        target::frame(
                            record,
                            attach,
                            depth,
                            Some([0.0, 0.0, 0.0, 1.0]),
                            |record| {
                                draw(record, pipeline.raw(), *arm, layers);
                            },
                        );
                    }
                })?;
                let took = at.elapsed() / repeats;
                if let Some((_, _, times)) = samples
                    .iter_mut()
                    .find(|(a, l, _)| *a == *arm && *l == layers)
                {
                    times.push(took);
                }
            }
        }
    }

    let rows: Vec<(Arm, u32, Duration)> = samples
        .into_iter()
        .map(|(arm, layers, mut times)| {
            times.sort_unstable();
            (arm, layers, times[times.len() / 2])
        })
        .collect();

    Ok(rows)
}

/// Records `layers` covering draws under one arm.
fn draw(record: Recorder<'_>, pipeline: vk::Pipeline, arm: Arm, layers: u32) {
    record.bind_pipeline(pipeline);
    record.viewport(SIDE, SIDE);
    // `CONTENT_DYNAMIC` carries both, and a dynamic state left unset before a draw is undefined.
    // A compare mask of zero passes everywhere, which is how an unclipped draw is expressed --
    // there are no clip masks here.
    record.stencil_reference(0);
    record.stencil_compare_mask(0);
    record.stencil_write_mask(0);
    // Front to back, which is the order the opaque pass is sent in and the order the early-out
    // needs: the nearest layer first, so the ones behind it have something to be rejected by.
    for layer in 0..layers {
        record.draw(3, 1, arm.first_instance(layer));
    }
}

/// The module, compiled through the same front end as the families'.
fn compile(work: u32) -> Result<Vec<u32>, String> {
    // The step is interpolated rather than written twice: a literal in the module text and a
    // constant beside it could disagree, and the disagreement would be a depth range nobody meant.
    let source = MODULE
        .replace("{step}", &format!("{DEPTH_STEP:?}"))
        .replace("{work}", &work.to_string());
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

/// One line per arm, with the floor taken off and the saving named.
fn report(rows: &[(Arm, u32, Duration)], repeats: u32) {
    let at = |arm: Arm, layers: u32| -> f64 {
        rows.iter()
            .find(|(a, l, _)| *a == arm && *l == layers)
            .map_or(0.0, |(_, _, took)| took.as_secs_f64() * 1e6)
    };

    println!(
        "{:<11} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "arm", "0 layers", "1", "2", "4", "8"
    );
    for arm in Arm::ALL {
        println!(
            "{:<11} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}   (us, p50)",
            arm.name(),
            at(arm, 0),
            at(arm, 1),
            at(arm, 2),
            at(arm, 4),
            at(arm, 8),
        );
    }

    println!();
    // The ratio of two medians rather than a difference of two floor-subtracted ones. The floor
    // is a few hundred microseconds and varies between runs, so subtracting it from both arms put
    // its noise into the answer twice: the same two-layer row read 9.8%, 37.7% and 86.4% over
    // three runs of identical code. A ratio of the arms as measured is stable to a few points.
    for layers in LAYERS.iter().copied().filter(|n| *n > 0) {
        let off = at(Arm::TestOff, layers);
        let per = at(Arm::PerLayer, layers);
        let saved = if off > 0.0 {
            (off - per) / off * 100.0
        } else {
            0.0
        };
        println!(
            "{layers:>2} layers: test-off {off:>8.1}  per-layer {per:>8.1} us   \
             the early-out leaves {:>5.1}% of the frame ({saved:>5.1}% saved)",
            if off > 0.0 { per / off * 100.0 } else { 0.0 },
        );
    }
    println!();
    println!(
        "the floor, with nothing drawn: {:.1} us a frame -- a command pool per submission, \
         divided by {repeats}",
        at(Arm::TestOff, 0)
    );
}
