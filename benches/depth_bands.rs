// SPDX-License-Identifier: BSD-2-Clause
//! What a layer's depth band decides, measured where it decides it.
//!
//! §11.7 asks for "opaque layers front-to-back with depth writes per `pass` + `opaquePassCutoff`",
//! and `record::DepthRange` is the per-layer half: each layer gets a slice of
//! `CameraUpdate::depth_range_size` by its `depth_slot`, slot zero nearest, so a nearer layer's
//! fragments reject the ones behind them whichever order the two were drawn in.
//!
//! # Why this exists when `a_frame` already fails
//!
//! Because of what it says when it does. Collapse every band to the whole range, reverse the band
//! order, or invert the depth compare op, and `a_frame` catches all three -- through one of its
//! sampled-layer pixels, reporting
//!
//! > (0, 0) is in the sampled band and holds [255, 128, 64, 255] rather than texel (2, 2), so the
//! > texture reached the sampler wrongly -- the rows, the rects or the view bound
//!
//! which names three causes and not the depth range. A depth defect that reports as a texture
//! defect costs whoever meets it the time to rule the texture out first. #142.
//!
//! So this draws two overlapping layers and nothing else: no textures, no masks, no stream, one
//! quad each, two colors. What it can fail about is the band.
//!
//! Run with `cargo bench --bench depth_bands`.

mod common;

use ash::vk;

use tessella_capture_abi::envelope::Topology;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Targets};
use tessella_emblema::record::DepthRange;
use tessella_emblema::target::{self, Depth, Host as TargetHost};
use tessella_vk::{Gpu, Image, ImageView, Memory, Pipeline, Recorder};

use common::Open;

/// The target's side. Small: every texel of it is read and compared.
const SIDE: u32 = 32;
/// The color format read back, which `target::frame` draws into.
const COLOR: vk::Format = vk::Format::B8G8R8A8_UNORM;
/// Layers in the style the bands divide.
const LAYERS: u32 = 2;
/// The range they divide, as a producer's `depth_range_size`.
///
/// One, which is the whole of Vulkan's depth volume: the bands are then a half each, far enough
/// apart that nothing here depends on the depth buffer's precision.
const RANGE: f32 = 1.0;

/// A full-target quad in one color, built from the vertex index.
///
/// No vertex buffer and no block: the subject is the depth band, and anything else bound is
/// another thing that could be wrong. The quad's clip `z` is zero, which `viewport_in` maps onto
/// the band's `near` -- that is how a layer gets a depth without the shader knowing the layer.
fn quad(color: &str) -> String {
    format!(
        r"
fn corner(at: u32) -> vec2<f32> {{
    switch (at) {{
        case 0u, 3u: {{ return vec2<f32>(-1.0, -1.0); }}
        case 1u: {{ return vec2<f32>(1.0, -1.0); }}
        case 2u, 4u: {{ return vec2<f32>(1.0, 1.0); }}
        default: {{ return vec2<f32>(-1.0, 1.0); }}
    }}
}}

@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {{
    return vec4<f32>(corner(vertex % 6u), 0.0, 1.0);
}}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {{
    return {color};
}}
"
    )
}

/// The upper layer's color, and the lower one's.
///
/// Upper and lower rather than near and far, because which is which is the thing to get right:
/// `depth_slot` is "measured from the top of the style" and "the style's first layer is the
/// furthest back and gets the highest slot" -- so layer *index* 1 of two is the upper layer, the
/// nearer band, and the one that has to win. The first run of this bench had them the other way
/// round and read the bands back to say so.
const UPPER_COLOR: &str = "vec4<f32>(1.0, 0.0, 0.0, 1.0)";
const LOWER_COLOR: &str = "vec4<f32>(0.0, 0.0, 1.0, 1.0)";

/// Those two as the BGRA bytes they land as, so a failure can say which layer won.
const UPPER_BGRA: [u8; 4] = [0, 0, 255, 255];
const LOWER_BGRA: [u8; 4] = [255, 0, 0, 255];

/// The alpha a translucent case draws each layer at.
///
/// Half, so the result carries both layers rather than only whichever was drawn last. What the
/// case reads is that both contributed, which neither layer alone can fake: one is blue and the
/// other red.
const BOTH_ALPHA: f32 = 0.5;

/// Which layer a case expects to see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Won {
    /// Layer index 1 of two: the later layer in the style, in the nearer band.
    Upper,
    /// Layer index 0: the style's first layer, furthest back.
    Lower,
}

/// The image a pass draws into.
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

/// A pipeline for one layer's quad, in the depth and blend state its pass asks for.
fn pipeline<'d>(
    gpu: Gpu<'d>,
    layout: &pipelines::Layout<'_>,
    targets: Targets,
    source: &str,
    depth: pipelines::Depth,
    blend: pipelines::Blend,
) -> Result<Pipeline<'d>, String> {
    let key = pipelines::Key {
        // Any family: the key's only effect on the pipeline here is its vertex layout, which is
        // empty because the quad is built from the index.
        shader: BuiltIn::BackgroundShader as i32,
        surface: tessella_emblema::surface::Surface::Plane,
        topology: Topology::Triangles,
        permutation: 0,
        layout: Vec::new(),
        blend,
        depth,
        enabled: pipelines::Enabled::ALL,
        stencil: pipelines::Stencil::Clip,
    };
    let modules =
        pipelines::stage_modules(gpu, &compile(source)?).map_err(|why| format!("module: {why}"))?;
    pipelines::build(gpu, &key, layout, pipelines::Stages::of(&modules), targets)
        .map_err(|why| format!("pipeline: {why}"))
}

/// Draws the two layers in the order given and reads the target back.
///
/// `order` names the layer *indexes* in the order they are recorded, so a case can draw the upper
/// one first or last. The band each is drawn in is `DepthRange::band`'s, set through `viewport_in` --
/// which is what `record::content` does per run, and the thing under test.
fn draw(
    device: &Open,
    order: [u32; 2],
    depth_state: pipelines::Depth,
    blend: pipelines::Blend,
) -> Result<Vec<u8>, String> {
    let gpu = device.gpu();
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    let attachment =
        Depth::new(gpu, SIDE, SIDE, depth_stencil).map_err(|why| format!("depth: {why}"))?;
    let targets = Targets {
        color: COLOR,
        depth_stencil,
        attachment: Attachment::DepthStencil,
    };

    // No descriptors: neither shader declares one.
    let layout = pipelines::layout(gpu, &[]).map_err(|why| format!("layout: {why}"))?;
    let lower = pipeline(
        gpu,
        &layout,
        targets,
        &with_alpha(LOWER_COLOR, blend),
        depth_state,
        blend,
    )?;
    let upper = pipeline(
        gpu,
        &layout,
        targets,
        &with_alpha(UPPER_COLOR, blend),
        depth_state,
        blend,
    )?;

    let bands = DepthRange::new(
        core::num::NonZeroU32::new(LAYERS).expect("two layers"),
        RANGE,
    )
    .ok_or("a positive range divides into bands")?;

    let owned = Owned::new(gpu)?;
    device.submit(|record: Recorder<'_>| {
        target::frame(
            record,
            TargetHost {
                image: &owned.image,
                view: &owned.view,
                width: SIDE,
                height: SIDE,
                layout: vk::ImageLayout::UNDEFINED,
            },
            &attachment,
            // Transparent, so a pixel that is neither color says nothing was drawn there.
            Some([0.0, 0.0, 0.0, 0.0]),
            |record| {
                for slot in order {
                    let (from, to) = bands.band(slot);
                    record.viewport_in(SIDE, SIDE, from, to);
                    record.bind_pipeline(if slot == 0 { lower.raw() } else { upper.raw() });
                    // The stencil is not the subject: a reference and a mask of zero compare
                    // equal against a buffer this never writes, which passes for every texel.
                    record.stencil_reference(0);
                    record.stencil_compare_mask(0);
                    record.stencil_write_mask(0);
                    record.draw(6, 1, 0);
                }
            },
        );
    })?;
    read_back(device, &owned)
}

/// The shader source, with the alpha the blend state wants.
fn with_alpha(color: &str, blend: pipelines::Blend) -> String {
    if matches!(blend, pipelines::Blend::Unblended) {
        return quad(color);
    }
    // Half, so a translucent result carries both layers rather than only the last one drawn.
    quad(&color.replace("1.0)", &format!("{BOTH_ALPHA})")))
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

/// Which layer every texel holds, or the complaint if they do not agree.
fn held(found: &[u8]) -> Result<Won, String> {
    let mut upper = 0usize;
    let mut lower = 0usize;
    let mut other = 0usize;
    let mut first_other = None;
    for (at, texel) in found.chunks(4).enumerate() {
        match texel {
            t if t == UPPER_BGRA => upper += 1,
            t if t == LOWER_BGRA => lower += 1,
            _ => {
                other += 1;
                if first_other.is_none() {
                    first_other = Some((at, [texel[0], texel[1], texel[2], texel[3]]));
                }
            }
        }
    }
    let total = (SIDE * SIDE) as usize;
    if let Some((at, texel)) = first_other {
        return Err(format!(
            "{other} of {total} texels are neither layer's color -- the first at {at} is \
             {texel:?}, where a quad covering the whole target leaves none"
        ));
    }
    match (upper, lower) {
        (u, 0) if u == total => Ok(Won::Upper),
        (0, l) if l == total => Ok(Won::Lower),
        (u, l) => Err(format!(
            "{u} texels of the upper layer and {l} of the lower one, where one quad covers the \
             whole target: the two were drawn into different parts of it"
        )),
    }
}

/// The upper layer wins, whichever order the two were drawn in.
///
/// Upper is layer index 1 of two: `depth_slot` is measured from the top of the style, so the
/// style's *first* layer is furthest back and the last one is nearest. That is the direction a map
/// needs -- a later layer paints over an earlier one -- and it is what makes the two draw orders
/// worth running.
///
/// Both orders, because each catches a different thing. Lower-then-upper is what painting alone
/// gets right, so it says nothing about depth on its own. Upper-then-lower is the case depth has
/// to decide: with the bands collapsed, or reversed, or the compare op inverted, the lower layer
/// paints over the upper one and this reads its color instead.
fn the_upper_layer_wins(device: &Open) -> Result<(), String> {
    let bands = DepthRange::new(
        core::num::NonZeroU32::new(LAYERS).expect("two layers"),
        RANGE,
    )
    .ok_or("a positive range divides into bands")?;
    for order in [[0u32, 1], [1, 0]] {
        let found = draw(
            device,
            order,
            pipelines::Depth::Write,
            pipelines::Blend::Unblended,
        )?;
        let won = held(&found).map_err(|why| format!("drawn {order:?}: {why}"))?;
        if won != Won::Upper {
            let (lower_from, lower_to) = bands.band(0);
            let (upper_from, upper_to) = bands.band(1);
            return Err(format!(
                "drawn {order:?}, the {won:?} layer won: layer 0 is in {lower_from}..{lower_to} \
                 and layer 1 in {upper_from}..{upper_to}, so either the bands do not order the \
                 layers or the depth test does not reject what is behind"
            ));
        }
    }
    println!("  the upper wins        ok   both draw orders, layer 1 over layer 0");
    Ok(())
}

/// A translucent pass does not write depth, so what is behind survives being drawn over.
///
/// The other half of §11.7's "depth writes per `pass`". The producer sends each pass front to back
/// -- `record::content` is what reverses the translucent one -- so a translucent pipeline that
/// wrote depth would have the *first* draw reject the second: the upper layer lands in the nearer
/// band, writes it, and the lower layer behind it is thrown away rather than blended under.
///
/// Which is why this draws them in the order the producer sent, upper first. `Depth::TestOnly`
/// keeps both; `Depth::Write` keeps one, and the case says which texel gave it away.
fn a_translucent_pass_does_not_write(device: &Open) -> Result<(), String> {
    let found = draw(
        device,
        [1, 0],
        pipelines::Depth::TestOnly,
        pipelines::Blend::Alpha,
    )?;

    let mut both = 0usize;
    let mut first_single = None;
    for (at, texel) in found.chunks(4).enumerate() {
        // Both contributed where blue and red are both present: each layer is half alpha, so a
        // texel carrying one of them alone is a layer that was rejected or never drew.
        if texel[0] > 16 && texel[2] > 16 {
            both += 1;
        } else if first_single.is_none() {
            first_single = Some((at, [texel[0], texel[1], texel[2], texel[3]]));
        }
    }
    let total = (SIDE * SIDE) as usize;
    if both != total {
        let (at, texel) = first_single.unwrap_or((0, [0; 4]));
        return Err(format!(
            "{both} of {total} texels carry both layers -- {at} reads {texel:?} -- so the first \
             draw rejected the second rather than letting it blend under"
        ));
    }
    println!("  translucent keeps     ok   both layers in every texel, drawn front to back");
    Ok(())
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
        ("upper_wins", the_upper_layer_wins as Case),
        ("translucent_keeps", a_translucent_pass_does_not_write),
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
