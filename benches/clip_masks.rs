// SPDX-License-Identifier: BSD-2-Clause
//! A clip mask, drawn and then tested against.
//!
//! A bench rather than a test because it needs a GPU. `tests/mask_references.rs` holds the counter,
//! and `tests/pipeline_state.rs` the state fields; what needs a device is whether the two halves
//! actually meet in a stencil buffer.
//!
//! # What this proves
//!
//! That the mask clips. The mask pipeline writes `REPLACE` with a dynamic reference and no color;
//! the content pipeline tests `EQUAL` against that reference and never writes. Nothing short of
//! running both says whether the pair is right -- and every way of getting it wrong *draws*:
//!
//! - a mask whose color write mask is not empty paints a tile-sized rectangle over the frame;
//! - a content draw that writes the stencil erases the mask for every later draw in the tile;
//! - a reference mismatch clips everything or nothing, depending which way it is wrong.
//!
//! So this draws a mask over half the target, then a full-target quad that tests against it, and
//! checks the color landed on that half and nowhere else.
//!
//! Run with `cargo bench --bench clip_masks`.

mod common;

use ash::vk;
use std::collections::{BTreeMap, BTreeSet};

use tessella_capture_abi::envelope::TileId;
use tessella_capture_abi::envelope::Topology;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_consume::stencil;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Targets};
use tessella_emblema::target::{self, Depth, Host};
use tessella_vk::{Gpu, Image, ImageView, Memory};

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

const SIDE: u32 = 64;

/// The slot a clipping mask's matrices arrive at.
///
/// mbgl's own, through the generated constant: a mask is a shader family like any other and its
/// block travels on `UboUpdate` like any other.
const MASK_SLOT: u32 = tessella_capture_abi::generated::ubo_slots::ID_CLIPPING_MASK_UBO;

/// The shader a mask's set is keyed by.
///
/// A set is keyed by its layer and its family, and the mask is not a family -- but mbgl numbers it
/// in the same enum, so its own id is the honest answer rather than a spare number. It is in
/// `families::UNDRAWN` for the matching reason: "the stencil mask, which the consumer draws from
/// the partition rather than from a family".
const MASK_SHADER: i32 = BuiltIn::ClippingMaskProgram as i32;
const COLOR: vk::Format = vk::Format::B8G8R8A8_UNORM;
/// The tile whose mask is drawn, and the one the partition is asked about.
fn tile() -> TileId {
    TileId {
        z: 14,
        x: 5,
        y: 5,
        overscaled_z: 14,
        wrap: 0,
    }
}

/// The partition over a cover of four tiles at one zoom.
///
/// Four rather than one, and that matters. A single tile gets a **one-bit** field, where there are
/// only two values and no reference can fail to match one of them -- so the control below would be
/// testing the complement rather than a mismatch. Four tiles need three bits, which leaves values
/// the partition did not assign.
///
/// A cover of four tiles is also what a view actually has, where one is what no view has.
fn cover() -> stencil::Partition {
    let mut tiles = BTreeSet::new();
    for x in 5..9u32 {
        tiles.insert(TileId {
            z: 14,
            x,
            y: 5,
            overscaled_z: 14,
            wrap: 0,
        });
    }
    let mut groups = BTreeMap::new();
    groups.insert(0i32, tiles.clone());
    stencil::partition(&tiles, &groups, stencil::ALL_BITS)
}

/// The assignment `tessella_consume::stencil` gives the tile this bench draws.
///
/// Taken from the partition rather than invented, which is the point: an earlier version of this
/// bench used a hardcoded reference and a full mask, and both are correct only in the fallback the
/// partition falls back *to*.
fn assignment() -> stencil::Assignment {
    cover().tiles.get(&tile()).copied().unwrap_or_default()
}

/// A reference no tile in the cover was given.
///
/// What the control needs: a value that cannot match any mask under this field's read mask. Found
/// by asking rather than assumed, because the field's width depends on the cover's size.
fn unassigned() -> u32 {
    let partition = cover();
    let given: BTreeSet<u8> = partition.tiles.values().map(|a| a.value).collect();
    let mask = assignment().read_mask;
    (0..=u32::from(mask))
        .find(|candidate| {
            // Nothing in the cover compares equal to it under the read mask, and it is not the
            // cleared zero either -- which every unmasked texel holds.
            *candidate != 0
                && given.iter().all(|value| {
                    u32::from(*value) & u32::from(mask) != *candidate & u32::from(mask)
                })
        })
        .unwrap_or(u32::from(mask) + 1)
}

/// A content shader that fills the target and is clipped only by the stencil.
///
/// Deliberately not one of the eighteen: this is about the stencil, so the draw should be the
/// simplest thing that covers the target and nothing else. Same quad trick as the mask.
const FILL: &str = r"
@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    var at = vec2<f32>(0.0, 0.0);
    switch (vertex % 6u) {
        case 0u, 3u: { at = vec2<f32>(-1.0, -1.0); }
        case 1u: { at = vec2<f32>(1.0, -1.0); }
        case 2u, 4u: { at = vec2<f32>(1.0, 1.0); }
        default: { at = vec2<f32>(-1.0, 1.0); }
    }
    return vec4<f32>(at, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
";

/// The same quad at half alpha, premultiplied, for counting blends.
///
/// `Blend::Alpha` is `src + dst * (1 - src.a)` over a premultiplied source, so one blend over a
/// black clear reads 128 and two read 191. The gap is what says how many times a pixel was
/// painted, which is the whole question the draw-once answers.
const FILL_HALF: &str = r"
@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    var at = vec2<f32>(0.0, 0.0);
    switch (vertex % 6u) {
        case 0u, 3u: { at = vec2<f32>(-1.0, -1.0); }
        case 1u: { at = vec2<f32>(1.0, -1.0); }
        case 2u, 4u: { at = vec2<f32>(1.0, 1.0); }
        default: { at = vec2<f32>(-1.0, 1.0); }
    }
    return vec4<f32>(at, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.5, 0.5, 0.5);
}
";

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

/// A host image, with its view and allocation.
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

/// The matrix the mask quad is drawn with: the left half of the target, in clip space.
///
/// Column-major, as the wire sends. The quad's corners are tile units of 0..8192, so this scales
/// 8192 to one clip unit and places the result over x in -1..0 and y in -1..1.
fn left_half() -> [f32; 16] {
    let s = 1.0 / 8192.0;
    [
        s,
        0.0,
        0.0,
        0.0, // column 0: x scaled to one clip unit
        0.0,
        2.0 * s,
        0.0,
        0.0, // column 1: y scaled to two
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// The whole target's matrix, for the case that covers everything.
fn whole() -> [f32; 16] {
    let s = 2.0 / 8192.0;
    [
        s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0,
    ]
}

/// Translates a matrix so the quad starts at the bottom-left corner of clip space.
fn placed(mut matrix: [f32; 16], x: f32, y: f32) -> [f32; 16] {
    matrix[12] = x;
    matrix[13] = y;
    matrix
}

struct Scene<'d> {
    depth: Depth<'d>,
    /// The formats the pipelines were built against, so a case can build one of its own.
    targets: Targets,
    mask_layout: pipelines::Layout<'d>,
    mask_pipeline: tessella_vk::Pipeline<'d>,
    fill_pipeline: tessella_vk::Pipeline<'d>,
    /// Held, not read: the descriptor set names this buffer and keeps it alive
    /// through nothing, so the scene has to.
    _blocks: tessella_emblema::blocks::Blocks<'d>,
    sets: tessella_emblema::descriptors::Sets<'d>,
    which: tessella_emblema::blocks::Which,
}

/// Builds everything a mask-and-test frame needs.
fn scene(device: &Open, matrix: [f32; 16]) -> Result<Scene<'_>, String> {
    use tessella_capture_abi::envelope::ViewId;
    use tessella_emblema::blocks::{Blocks, Which};
    use tessella_emblema::descriptors::Sets;
    use tessella_emblema::pipelines::{Binding, Kind};

    let gpu = device.gpu();
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

    // The mask declares one storage buffer and nothing else, arriving at mbgl's own slot for it.
    let bindings = [Binding {
        group: 0,
        binding: 0,
        kind: Kind::StorageBuffer,
        slot: Some(MASK_SLOT),
    }];
    let mask_layout = pipelines::layout(gpu, &bindings).map_err(|why| format!("layout: {why}"))?;

    let mask_modules = pipelines::stage_modules(gpu, &compile(tessella_emblema::masks::BODY)?)
        .map_err(|why| format!("mask module: {why}"))?;
    let mask_pipeline = pipelines::build_mask(
        gpu,
        &mask_layout,
        pipelines::Stages::of(&mask_modules),
        targets,
    )
    .map_err(|why| format!("mask pipeline: {why}"))?;

    // The content pipeline: the key carries no vertex input, because the fill builds its own quad.
    let fill_modules = pipelines::stage_modules(gpu, &compile(FILL)?)
        .map_err(|why| format!("fill module: {why}"))?;
    let fill_key = pipelines::Key {
        shader: BuiltIn::BackgroundShader as i32,
        surface: tessella_emblema::surface::Surface::Plane,
        topology: Topology::Triangles,
        permutation: 0,
        layout: Vec::new(),
        // Unblended, so the fill's pixel is what the fragment stage returned rather than what it
        // returned composited over the clear -- which is what this bench reads back.
        blend: pipelines::Blend::Unblended,
        depth: pipelines::Depth::Write,
        enabled: pipelines::Enabled::ALL,
        stencil: pipelines::Stencil::Clip,
    };
    let fill_pipeline = pipelines::build(
        gpu,
        &fill_key,
        &mask_layout,
        pipelines::Stages::of(&fill_modules),
        targets,
    )
    .map_err(|why| format!("fill pipeline: {why}"))?;

    // One matrix, in a block the mask indexes by instance.
    let which = Which {
        view: ViewId(1),
        layer: 0,
    };
    let mut blocks = Blocks::new();
    blocks
        .declare(gpu, which, MASK_SLOT, 1, 64)
        .map_err(|why| format!("blocks: {why}"))?;
    let bytes: Vec<u8> = matrix.iter().flat_map(|f| f.to_le_bytes()).collect();
    blocks
        .write(which, MASK_SLOT, 0, &bytes)
        .map_err(|why| format!("write matrix: {why}"))?;
    blocks
        .flush(which, MASK_SLOT, 0)
        .map_err(|why| format!("flush: {why}"))?;

    let mut sets = Sets::new(gpu, 2, &bindings).map_err(|why| format!("sets: {why}"))?;
    sets.write(&mask_layout, &bindings, which, MASK_SHADER, &blocks, &[])
        .map_err(|why| format!("set: {why}"))?;

    Ok(Scene {
        depth,
        targets,
        mask_layout,
        mask_pipeline,
        fill_pipeline,
        _blocks: blocks,
        sets,
        which,
    })
}

/// Reads the image back as BGRA.
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
    let mut out = vec![0u8; bytes as usize];
    let mapping = memory.map().map_err(|why| format!("map: {why}"))?;
    mapping
        .read(0, &mut out)
        .map_err(|why| format!("read: {why}"))?;
    Ok(out)
}

/// Draws a mask, then a fill tested against it, and returns the image.
fn mask_then_fill(
    device: &Open,
    owned: &Owned<'_>,
    scene: &Scene<'_>,
    reference: u32,
) -> Result<Vec<u8>, String> {
    mask_then_fill_masked(device, owned, scene, reference, None)
}

/// As above, with the content draw's compare mask overridden.
///
/// `None` is the tile's own `read_mask`, which every case but one wants. The exception is the case
/// that measures what a mask of no bits does, and it needs everything else held still.
fn mask_then_fill_masked(
    device: &Open,
    owned: &Owned<'_>,
    scene: &Scene<'_>,
    reference: u32,
    compare: Option<u32>,
) -> Result<Vec<u8>, String> {
    let given = assignment();
    let set = scene
        .sets
        .get(scene.which, MASK_SHADER)
        .ok_or("the descriptor set was not written")?;
    device.submit(|record| {
        target::frame(
            record,
            Host {
                image: &owned.image,
                view: &owned.view,
                width: SIDE,
                height: SIDE,
                layout: vk::ImageLayout::UNDEFINED,
            },
            &scene.depth,
            // Cleared to transparent, so any color in the result came from the fill.
            Some([0.0, 0.0, 0.0, 0.0]),
            |record| {
                // The mask writes its own field: the assignment's value, through its write mask.
                record.bind_pipeline(scene.mask_pipeline.raw());
                record.bind_descriptor_set(scene.mask_layout.pipeline(), set);
                record.stencil_reference(u32::from(given.value));
                record.stencil_write_mask(u32::from(given.write_mask));
                record.draw(tessella_emblema::masks::VERTICES, 1, 0);

                // And the content compares its own zoom's field, which is the read mask.
                record.bind_pipeline(scene.fill_pipeline.raw());
                record.bind_descriptor_set(scene.mask_layout.pipeline(), set);
                record.stencil_reference(reference);
                record.stencil_compare_mask(compare.unwrap_or(u32::from(given.read_mask)));
                // A content draw writes nothing. Set rather than left, because the write mask is
                // declared dynamic for the 3D draw-once and the mask draw above just set it.
                record.stencil_write_mask(0);
                record.draw(6, 1, 0);
            },
        );
    })?;
    read_back(device, owned)
}

/// How many texels are white, and how many of those are in the left half.
fn white(found: &[u8]) -> (usize, usize) {
    let mut total = 0;
    let mut left = 0;
    for (at, texel) in found.chunks(4).enumerate() {
        if texel == [255, 255, 255, 255] {
            total += 1;
            if (at as u32 % SIDE) < SIDE / 2 {
                left += 1;
            }
        }
    }
    (total, left)
}

/// A mask over half the target clips the fill to that half.
fn a_mask_clips_the_fill(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let scene = scene(device, placed(left_half(), -1.0, -1.0))?;
    let found = mask_then_fill(device, &owned, &scene, u32::from(assignment().value))?;
    let (total, left) = white(&found);

    let half = (SIDE * SIDE / 2) as usize;
    if total == 0 {
        return Err("nothing was drawn: the fill was clipped away entirely".into());
    }
    if total == (SIDE * SIDE) as usize {
        return Err("everything was drawn: the stencil clipped nothing".into());
    }
    if left != total {
        return Err(format!(
            "{total} texels drawn and only {left} in the masked half: the mask is in the wrong place"
        ));
    }
    // Allow a column of slack for the quad's edge landing on a texel boundary.
    if total.abs_diff(half) > SIDE as usize {
        return Err(format!(
            "{total} texels drawn against about {half} for half the target"
        ));
    }
    println!(
        "  a mask clips          ok   {total} of {} texels, all in the masked half",
        SIDE * SIDE
    );
    Ok(())
}

/// A fill tested against the wrong reference is clipped away entirely.
///
/// The control. Without it, a mask that wrote nothing and a stencil test that passed everywhere
/// would look the same as success for the case above -- both leave the fill on the target.
fn the_wrong_reference_draws_nothing(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let scene = scene(device, placed(left_half(), -1.0, -1.0))?;
    let found = mask_then_fill(device, &owned, &scene, unassigned())?;
    let (total, _) = white(&found);
    if total != 0 {
        return Err(format!(
            "{total} texels survived a reference the mask never wrote"
        ));
    }
    println!("  the wrong reference   ok   nothing drawn, so the test is really testing");
    Ok(())
}

/// A compare mask of zero draws everywhere, which is what "unclipped" is.
///
/// Measured because the constant behind it once claimed the opposite: that a pipeline whose dynamic
/// compare mask was never set "compares no bits and the draw is clipped away entirely, which is a
/// blank layer". The stencil test compares `reference & compare_mask` against
/// `stencil & compare_mask`, so a mask of zero makes both sides zero and `EQUAL` passes everywhere.
///
/// The control is the case above: the same scene and the same reference no tile was assigned, which
/// draws nothing under a real mask. Only the compare mask differs between the two, so the whole
/// target coming back covered is attributable to it and to nothing else.
///
/// Pinned rather than left to the comment, because the value is the one `record::clipping` returns
/// for a drawable with no tile -- the unclipped path depends on this being a draw and not a blank.
fn a_zero_compare_mask_draws_unclipped(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let scene = scene(device, placed(left_half(), -1.0, -1.0))?;
    let found = mask_then_fill_masked(device, &owned, &scene, unassigned(), Some(0))?;
    let (total, left) = white(&found);
    let texels = (SIDE * SIDE) as usize;
    if total != texels || left != texels / 2 {
        return Err(format!(
            "{total} of {texels} texels drawn, {left} in the masked half, for a mask of no bits"
        ));
    }
    println!(
        "  zero compare mask     ok   all {total} texels, so no bits compared draws unclipped"
    );
    Ok(())
}

/// A mask covering the whole target clips nothing away.
///
/// The other side of the first case: it bounds the mask from above, so a mask that happened to
/// cover half for the wrong reason is distinguishable from one placed by its matrix.
fn a_whole_tile_mask_clips_nothing(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let scene = scene(device, whole())?;
    let found = mask_then_fill(device, &owned, &scene, u32::from(assignment().value))?;
    let (total, _) = white(&found);
    let all = (SIDE * SIDE) as usize;
    if total != all {
        return Err(format!(
            "{total} of {all} texels drawn under a mask covering the whole target"
        ));
    }
    println!("  a whole-tile mask     ok   all {all} texels drawn");
    Ok(())
}

/// The mask itself writes no color.
///
/// Drawn with no fill after it, so anything on the target came from the mask. Its own fragment
/// stage returns transparent black and the clear is transparent black too -- which is why this
/// checks the *alpha* of a texel the mask covers is still zero rather than comparing to the clear.
fn the_mask_writes_no_color(device: &Open) -> Result<(), String> {
    let given = assignment();
    let owned = Owned::new(device.gpu())?;
    let scene = scene(device, whole())?;
    let set = scene
        .sets
        .get(scene.which, MASK_SHADER)
        .ok_or("the descriptor set was not written")?;
    device.submit(|record| {
        target::frame(
            record,
            Host {
                image: &owned.image,
                view: &owned.view,
                width: SIDE,
                height: SIDE,
                layout: vk::ImageLayout::UNDEFINED,
            },
            &scene.depth,
            // Opaque red, so a mask that wrote color would have to overwrite it.
            Some([0.0, 0.0, 1.0, 1.0]),
            |record| {
                record.bind_pipeline(scene.mask_pipeline.raw());
                record.bind_descriptor_set(scene.mask_layout.pipeline(), set);
                record.stencil_reference(u32::from(given.value));
                record.stencil_write_mask(u32::from(given.write_mask));
                record.draw(tessella_emblema::masks::VERTICES, 1, 0);
            },
        );
    })?;
    let found = read_back(device, &owned)?;
    let first: [u8; 4] = found[..4].try_into().map_err(|_| "short read")?;
    if first != [255, 0, 0, 255] {
        return Err(format!(
            "the mask wrote color: {first:?} rather than the clear's BGRA red"
        ));
    }
    println!("  no color from a mask ok   the clear survived a whole-target mask");
    Ok(())
}

/// A pixel a 3D run covers twice is blended once, and the same pixel without the scheme is not.
///
/// The mechanism `pipelines::Stencil::DrawOnce` is: `NOT_EQUAL` against the run's value with
/// `REPLACE` on pass, so the first fragment at a pixel marks it and every later one of that run
/// fails. That is what keeps a translucent building from blending where its own surfaces meet --
/// a wall shared by two footprints, a roof meeting the wall raised off it -- which the depth
/// buffer cannot do anything about, because those surfaces are at the *same* depth.
///
/// Drawn twice over a black clear at half alpha, premultiplied: one blend reads 128 and two read
/// 191. Both arms are here, because the number on its own says nothing -- the control is the same
/// two draws under `Stencil::Clip` with a compare mask of no bits, which is this bench's own
/// spelling of "unclipped" and is what every extrusion drew with before.
///
/// The depth test is on and writing in both arms, which is the point of running it this way: the
/// second draw is at the depth the first wrote, `LESS_OR_EQUAL` passes, and the stencil is the only
/// thing that can tell them apart.
///
/// # What this case cannot reach
///
/// The third op. Both draws here pass the depth test, so a `depthFail` of `REPLACE` -- which would
/// mark a pixel for a fragment that was never drawn and reject the one that is -- draws the same
/// two numbers. `tests/stencil_masks.rs` asserts the ops field by field, and that is where that
/// mutation is caught; this says the three of them together do what they are for.
fn the_draw_once_blends_a_pixel_once(device: &Open) -> Result<(), String> {
    use tessella_emblema::record::DRAW_ONCE_BIT;

    let gpu = device.gpu();
    let owned = Owned::new(gpu)?;
    let scene = scene(device, whole())?;
    let modules = pipelines::stage_modules(gpu, &compile(FILL_HALF)?)
        .map_err(|why| format!("the half-alpha module: {why}"))?;

    let key = |stencil| pipelines::Key {
        shader: BuiltIn::BackgroundShader as i32,
        surface: tessella_emblema::surface::Surface::Plane,
        topology: Topology::Triangles,
        permutation: 0,
        layout: Vec::new(),
        blend: pipelines::Blend::Alpha,
        depth: pipelines::Depth::Write,
        enabled: pipelines::Enabled::ALL,
        stencil,
    };
    let built = |stencil| {
        pipelines::build(
            gpu,
            &key(stencil),
            &scene.mask_layout,
            pipelines::Stages::of(&modules),
            scene.targets,
        )
        .map_err(|why| format!("the {stencil:?} pipeline: {why}"))
    };
    let once = built(pipelines::Stencil::DrawOnce)?;
    let clip = built(pipelines::Stencil::Clip)?;

    let set = scene
        .sets
        .get(scene.which, MASK_SHADER)
        .ok_or("the descriptor set was not written")?;
    let twice = |pipeline: &tessella_vk::Pipeline<'_>, reference: u32, mask: u32| {
        device.submit(|record| {
            target::frame(
                record,
                Host {
                    image: &owned.image,
                    view: &owned.view,
                    width: SIDE,
                    height: SIDE,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &scene.depth,
                Some([0.0, 0.0, 0.0, 1.0]),
                |record| {
                    record.bind_pipeline(pipeline.raw());
                    record.bind_descriptor_set(scene.mask_layout.pipeline(), set);
                    record.stencil_reference(reference);
                    record.stencil_compare_mask(mask);
                    record.stencil_write_mask(mask);
                    // The same quad twice, which is what two surfaces of one building at one
                    // pixel are: two draws of one run, not one draw of two instances.
                    record.draw(6, 1, 0);
                    record.draw(6, 1, 0);
                },
            );
        })?;
        read_back(device, &owned)
    };

    let marked = twice(&once, u32::from(DRAW_ONCE_BIT), u32::from(DRAW_ONCE_BIT))?;
    // The control writes nothing and compares nothing, so both draws land.
    let unmarked = twice(&clip, 0, 0)?;

    // Both ends of the one blend, not just the upper one. A scheme that rejected *every* fragment
    // -- `EQUAL` against a mark the run has not made yet, which is the obvious way to get the
    // compare the wrong way round -- leaves the pixel at the clear, and a check that only asked
    // for "fewer than two" would call that a pass. Measured: that mutation reads 0 here.
    let blue = |found: &[u8]| i32::from(found.first().copied().unwrap_or(0));
    if !(180..=200).contains(&blue(&unmarked)) {
        return Err(format!(
            "the control blended to {} where two blends are 191, so this case cannot tell one \
             from two",
            blue(&unmarked)
        ));
    }
    if !(120..=140).contains(&blue(&marked)) {
        return Err(format!(
            "a pixel of the run blended to {} where one blend is 128 and two are 191",
            blue(&marked)
        ));
    }
    println!(
        "  draw_once             ok   one blend reads {}, two read {}",
        blue(&marked),
        blue(&unmarked)
    );
    Ok(())
}

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
        ("mask_clips", a_mask_clips_the_fill as Case),
        ("wrong_reference", the_wrong_reference_draws_nothing),
        ("zero_compare", a_zero_compare_mask_draws_unclipped),
        ("whole_tile", a_whole_tile_mask_clips_nothing),
        ("no_color", the_mask_writes_no_color),
        ("draw_once", the_draw_once_blends_a_pixel_once),
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
