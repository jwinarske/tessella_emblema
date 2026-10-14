//! A captured frame's geometry and descriptors, onto the device.
//!
//! ```text
//! TSE_CAPTURE=<path.tslcap> cargo bench --bench capture_frame
//! ```
//!
//! Written by `capture-render --capture <path>` on the producer's side. `examples/replay-a-capture`
//! is the device-free half -- it reads the file, drives `Host`, and checks every slab reference
//! resolves. This is the half that needs a GPU: it plans each drawable's vertex input against its
//! family's table, uploads the bytes, brings up the layer blocks the stream carried, and writes a
//! descriptor set for every (layer, family) the frame draws.
//!
//! # Why this is a bench and not an example
//!
//! The device harness is `benches/common`, which an example cannot reach. A bench takes no
//! arguments, so the path comes from the environment -- and with none set this skips, the way it
//! skips when there is no device. Neither is a failure: a run that cannot answer says so.
//!
//! # What it is for
//!
//! Every other bench here draws a fixture: a handful of families its author chose. A capture of a
//! real style carries whatever that style uses, in whatever shapes the producer emitted -- which
//! is the first thing in this repository that can disagree with the fixtures about what a frame
//! looks like. The failures worth finding are a family with no table, an attribute run the plan
//! refuses, and a slab reference the store resolves short.
//!
//! Two of those failures were in the library rather than in the capture, and both were about where
//! a resource is *looked up* rather than whether it exists -- which is why they needed a real frame
//! and not a wider fixture:
//!
//! - `GlobalPaintParamsUBO` arrives under `layer_index: -1` because it belongs to the renderer
//!   rather than to a layer, and `line` and `circle` both declare it. Resolved at the batch's own
//!   layer, neither layer's set could be written at all.
//! - Three of this frame's six layers are drawn by *two* families -- a fill with its outline, twice,
//!   and an extrusion with its walls -- and a set keyed by the layer alone kept only the second of
//!   each. The write succeeded; the lookup was what was lost.
//!
//! And then it draws: the masks, a pipeline per batch key, `record::content` over the frame's own
//! batches, and the pixels back. Three more things were wrong and none of them could have been
//! found by a fixture, because a fixture writes what it would read:
//!
//! - an `AttributeDesc` of stride zero means tightly packed, which is what the producer sends for
//!   the background's quad and what a Vulkan binding of zero is not -- so the quad collapsed to its
//!   first corner;
//! - the matrices are mbgl's, whose projection is OpenGL's, so clip `z` runs `-w..w` where Vulkan's
//!   volume is `0..w` -- and the viewport quad's whole `z` is a sublayer nudge just below zero;
//! - a drawable with instance-rate attributes draws one instance per entry of the shortest run,
//!   which is a number the wire carries only as a length over a stride.
//!
//! What it does not do is say the picture is *right*. It says every batch resolved to a program,
//! every drawable was recorded, and no pixel was left at the clear -- and the one cross-check it
//! prints is the commonest color, which for the Berlin capture is `[234, 230, 221]` over 538,300
//! pixels: `#eae6dd`, the `background-color` of `tools/parity/scenes/families_p.json`, which is the
//! style that capture was taken through.
//!
//! # What the Berlin frame actually contains, which is not all of that style
//!
//! Worth saying, so nobody reads the picture as a reference. Seven families arrive -- background,
//! fill, its triangulated outline, line, circle and the two extrusions -- and the colors they draw
//! are the scene's: the water is `#9ec6dd`, the roads `#cfc7b6`, the buildings `#c9b9a8` through
//! the light the producer sent (`light_intensity` 0.5, `light_position` 0.2875, -0.498, 0.9959).
//!
//! The parks layer is **black**, and that is the capture rather than this crate. Its paint is
//! `fill-pattern: grass_pattern` and the capture carries no sprite, so the producer resolved no
//! pattern, emitted a plain `FillShader` drawable, and sent the style spec's default `fill-color`
//! with it -- `#000000`, which is what `FillEvaluatedPropsUBO` holds for that layer. mbgl does not
//! have that arm: `render_fill_layer.cpp:458` branches on whether the *style* has a pattern, not on
//! whether one resolved, so it builds a pattern drawable either way. Drawing it by hand with
//! `TSE_ONLY_LAYER=3` is how that was found.
//!
//! The capture also has no raster tile, so the style's imagery layer is absent, and no symbols,
//! because the capture was taken without glyphs.
//!
//! # Taking one with walls in it
//!
//! That frame is top-down, so its extrusion layer draws roofs and no walls at all -- and the depth
//! prepass each extrusion is built with has nothing to reject. A pitched capture of the same tile
//! shows them, and is what measured `pipelines::Depth`'s 3D arm:
//!
//! ```text
//! curl -s http://127.0.0.1:8080/berlin_z15/14/8801/5374.mvt | gunzip > berlin-z14.mvt
//! capture-render --style tools/parity/scenes/families_p.json --tile berlin-z14.mvt \
//!     --tile-at 14/8801/5374 --source berlin --lon 13.4 --lat 52.5 --zoom 14 --pitch 60 \
//!     --capture pitched.tslcap --out pitched.png
//! ```
//!
//! # And one with a ground under it
//!
//! `--dem` (tessella#380) puts a terrain in the capture, which is what raises the layers standing
//! on it. The same tile under a terrain style at 1.5 exaggeration:
//!
//! ```text
//! python3 tools/parity/scenes/dem.py 14 8801 5374 > dem.png
//! capture-render --style <a terrain scene> --tile berlin-z14.mvt --tile-at 14/8801/5374 \
//!     --source berlin --dem dem.png --lon 13.4 --lat 52.5 --zoom 14 --pitch 60 \
//!     --capture raised.tslcap
//! ```
//!
//! Replayed, that is 99 records and 17 batches, twelve of them raised, and it draws: the surfaces
//! come out `Plane and Terrain`, every raised layer binds the ground's elevation at
//! `surface::TERRAIN_ELEVATION_SLOT`, and 12.8% of the frame's pixels differ from the flat one --
//! the relief, which is what a raise is.
//!
//! **All seventeen of its batches draw**, in 8 families and 15 pipelines, 55 drawables and 55
//! draws with none undrawn. The seventeenth used to be the ground itself -- it arrives as
//! `terrain_ubo::BUILTIN_TERRAIN_SHADER`, which `families::ALL` had no entry for, so the layers
//! standing on the surface drew and the surface under them did not. The recording's count is still
//! checked against the batches that *resolved to a program* rather than against all of them,
//! because that is a real case whether or not this capture has one: `record::content`'s own note
//! names it, "a batch this consumer does not draw, which is a real case".
//!
//! # The ground drew and this frame is byte-identical, which is the right answer
//!
//! Bare ground is painted the style's `background-color` -- the spec gives terrain no paint of its
//! own -- and the background layer covers the viewport with that same color under it. So the ground
//! of *this* capture changes no pixel, and the frame before the family existed and the frame after
//! it compare equal byte for byte.
//!
//! Measured against a control rather than argued: the same tile and DEM through the same style with
//! its background layer removed, where the ground's color falls back to the spec default black and
//! nothing else covers the view. That capture is 11 batches, ten of them raised, and the clear
//! shows through 614,532 of its 786,432 pixels with the ground drawn against 703,062 with the
//! family taken back out of the table. The 88,530 between them are the bare ground, which is what
//! this capture paints the background's color over.
//!
//! Where a pixel does say the ground's own arithmetic is right is `benches/first_pixel.rs`:
//! `terrain` reads a pixel that only the right elevation covers, and `terrain_skirt` one that only
//! the right curtain does.
//!
//! The tile is the parity harness's own, off `serve.sh` on 8080, and has to be decompressed: the
//! server sends it gzipped and the decoder refuses that with "wire type 7 at byte 0 is a group".
//! `--tile-at` and `--source` are what place it; without them the tool draws the style's background
//! over an empty cover, which is what the PNG beside the first capture turned out to be. The result
//! here is 32 geometries, 54 drawables and the same 12 batches.

// A bench is a program, and it reports by printing.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    // One pass over the capture, in the order the steps depend on each other, as
    // `benches/first_pixel.rs` allows for the same reason.
    clippy::too_many_lines
)]

mod common;

#[path = "common/screen.rs"]
mod screen;

use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use tessella_capture_abi::EnvelopeKind;
use tessella_capture_abi::RenderPass;
use tessella_capture_abi::capture;
use tessella_capture_abi::envelope::{DrawFlags, Rect16, TileId, ViewId};
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_capture_abi::ring::Ring;
use tessella_consume::batch;
use tessella_consume::host::Host;
use tessella_consume::slab::{self, Slab};
use tessella_consume::upload::Upload;
use tessella_emblema::blocks::Blocks;
use tessella_emblema::descriptors::Sets;
use tessella_emblema::device::Attachment;
use tessella_emblema::images::Images;
use tessella_emblema::pipelines::Blend;
use tessella_emblema::store::Store;
use tessella_emblema::surface::Surface;
use tessella_emblema::{
    blocks, buffers, descriptors, families, masks, pipelines, record, slots, target, vertices,
};
use tessella_vk::Recorder;

use common::Open;
use screen::Screen;

/// The viewport the capture's camera was computed for.
///
/// Not on the wire, and not an omission: a screen view's size belongs to the consumer, and
/// `CameraUpdate` carries the projection rather than the extent it was built from.
/// `capture-render`'s own default is 1024 by 768 and the PNG it wrote beside this capture is that
/// size, so this is the size the matrices place geometry for. Drawn at another, it is this frame
/// through the wrong aspect.
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

/// What the target is cleared to, which is no color this style draws.
///
/// Magenta, loud on purpose: the background is the bottom layer of this style and covers the
/// viewport, so a pixel left at this is a pixel nothing drew.
const CLEAR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// The same, as the bytes a readback holds.
const CLEARED: [u8; 4] = [255, 0, 255, 255];

/// What a batch's program is looked up by.
///
/// `batch::Key` without its texture refs, which no batch in this frame carries: a program is a
/// pipeline and the layout its set binds through, and both are functions of exactly these five.
/// The pass is one of them, which is why this frame's background -- batched once in each pass --
/// is two programs and not one.
type ProgramKey = (i32, u32, u8, u8, u64);

/// That key, off a batch.
fn program_key(key: &batch::Key) -> ProgramKey {
    (
        key.builtin_shader,
        key.layer_index,
        key.pass,
        key.draw_flags,
        key.permutation_key,
    )
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

/// Declares and fills every texture the stream sent, and leaves them readable by a shader.
///
/// `benches/a_frame.rs` has the same twenty lines over a fixture's two atlases, and the shape is
/// the ABI's rather than either bench's: `Images::upload` asks for a region's rows one at a time
/// and `Upload::rows` is what says where a row is -- an offset and a stride per region, which
/// differ between the two payload forms. A backend reading the wrong one uploads the right number
/// of bytes to the right place from the wrong rows.
fn stage<'d>(
    images: &mut Images<'d>,
    gpu: tessella_vk::Gpu<'d>,
    record: Recorder<'_>,
    host: &Host,
) -> Result<usize, String> {
    let mut empty = 0;
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

        // A texture of no area is not an image: `VkImageCreateInfo` requires every extent to be
        // at least one, and a producer sends `0x0` for an atlas with nothing in it -- the captured
        // Berlin frame carries one. Skipped rather than declared, which is what `Error::NoArea`
        // says: RADV allocates it and says nothing, V3D fails the submission three calls later
        // with an error naming neither the texture nor the extent.
        if size.width == 0 || size.height == 0 {
            empty += 1;
            continue;
        }
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
    Ok(empty)
}

/// A render state, as the flags it is made of.
///
/// Named rather than printed through `Debug`, which is three fields of `true` and `false` in an
/// order a reader has to look up.
fn state(enabled: pipelines::Enabled) -> String {
    let mut parts = Vec::new();
    if enabled.depth {
        parts.push("depth");
    }
    if enabled.stencil {
        parts.push("stencil");
    }
    if enabled.color {
        parts.push("color");
    }
    if parts.is_empty() {
        "nothing".to_owned()
    } else {
        parts.join("+")
    }
}

/// A pass, as a name. An order entry carries one bit of what is a mask on the wire.
fn named(pass: u8) -> String {
    let one = RenderPass::from_bits(pass);
    if one == Some(RenderPass::OPAQUE) {
        "the opaque pass".to_owned()
    } else if one == Some(RenderPass::TRANSLUCENT) {
        "the translucent pass".to_owned()
    } else {
        format!("pass {pass:#04b}")
    }
}

fn main() {
    let Ok(path) = std::env::var("TSE_CAPTURE") else {
        println!("skipping: set TSE_CAPTURE to a path written by capture-render --capture");
        return;
    };
    match run(&path) {
        Ok(()) => {}
        Err(why) => {
            println!("  the capture           FAIL {why}");
            std::process::exit(1);
        }
    }
}

fn run(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|why| format!("reading {path}: {why}"))?;
    let read = capture::decode(&bytes)
        .ok_or_else(|| format!("{path} is {} bytes and not a capture", bytes.len()))?;
    let region = read.region();
    let slabs: Vec<Slab> = read
        .slabs()
        .into_iter()
        .map(|slab| Slab {
            offset: slab.offset,
            length: slab.length,
        })
        .collect();

    let mut ring = Ring::new((bytes.len() * 2).next_power_of_two().max(1 << 16));
    {
        let (producer, _) = ring.split();
        for record in read.records() {
            let Some(kind) = u16::try_from(record.kind)
                .ok()
                .and_then(EnvelopeKind::from_repr)
            else {
                continue;
            };
            producer
                .write(kind, record.record, record.payload)
                .map_err(|_| "the ring filled".to_string())?;
        }
    }

    let mut host = Host::new();
    let progress = host.read(ring.consumer());
    let view = (0..8u32)
        .map(ViewId)
        .find(|id| host.declared(*id))
        .ok_or("no view was declared")?;
    let frame = host
        .plan(view)
        .ok_or_else(|| format!("view {} planned nothing", view.0))?;
    println!(
        "  the stream            ok   {} records, {} batches, {} slabs, {} bytes of region",
        progress.records,
        frame.batches.len(),
        slabs.len(),
        region.len()
    );

    // Collected before the device opens, so a capture that cannot be planned says so without
    // needing a GPU -- and so the borrow on the host ends before the store takes one.
    //
    // Distinct geometries, not drawables: a geometry drawn in two batches is one set of bytes, and
    // uploading it twice would make `Store::resident` disagree with the count here -- it replaces
    // rather than appends. The first version of this reported "15 of 24 resident, 0 failed", which
    // reads as nine losses and was nine repeats.
    let mut wanted: Vec<(tessella_capture_abi::envelope::GeometryId, i32)> = Vec::new();
    let mut seen = BTreeSet::new();
    for batch in frame.batches.iter() {
        for geometry in batch.geometries {
            if seen.insert(geometry.0) {
                wanted.push((*geometry, batch.key.builtin_shader));
            }
        }
    }
    // And which (layer, family) pairs the frame draws, which is what a descriptor set is per.
    // Owned, because `Host::plan` borrows the host mutably and the uniform uploads below need it
    // back.
    let mut programs: Vec<(i32, i32)> = frame
        .batches
        .iter()
        // The same conversion `record::layer_of` makes, so a pair here is the pair a recording
        // would look up: a layer index past `i32::MAX` is absurd and saturating leaves it
        // somewhere that is not the reserved `-1`.
        .map(|batch| {
            (
                batch.key.builtin_shader,
                i32::try_from(batch.key.layer_index).unwrap_or(i32::MAX),
            )
        })
        .collect();
    programs.sort_unstable();
    programs.dedup();
    // And how many layers those are, which is not the same number: a fill layer emits a fill
    // drawable and an outline drawable at one `layer_index`, so a set per layer is fewer sets than
    // the frame needs.
    let mut layers: Vec<i32> = programs.iter().map(|(_, layer)| *layer).collect();
    layers.sort_unstable();
    layers.dedup();

    // And what each pair is *placed* on, which is per drawable rather than per view: a camera
    // names one projection for the whole frame, and `ON_TERRAIN` is set per binding wherever a
    // ground covers the drawable's own tile. So the flags are collected per pair -- and a pair
    // whose batches disagree about them is a layer drawn on two surfaces, which binds two
    // different shapes of descriptor set under one `(view, layer, shader)` key. The loop below
    // reports such a pair rather than picking one of the two.
    let projection = frame.camera.projection().ok_or_else(|| {
        format!(
            "the camera's projection is {}, which this build does not know",
            frame.camera.projection
        )
    })?;
    let mut flags: BTreeMap<(i32, i32), BTreeSet<u8>> = BTreeMap::new();
    // And one geometry per pair, for the textures: a set binds the views a *drawable* names, and
    // the drawables of one (layer, family) name the same ones -- the elevation a raised layer
    // samples is the ground's tile, not the drawable's own.
    let mut drawn_by: BTreeMap<(i32, i32), tessella_capture_abi::envelope::GeometryId> =
        BTreeMap::new();
    for batch in frame.batches.iter() {
        let at = (
            batch.key.builtin_shader,
            i32::try_from(batch.key.layer_index).unwrap_or(i32::MAX),
        );
        flags.entry(at).or_default().insert(batch.key.draw_flags);
        if let Some(geometry) = batch.geometries.first() {
            drawn_by.entry(at).or_insert(*geometry);
        }
    }
    let raised = frame
        .batches
        .iter()
        .filter(|batch| batch.key.draw_flags & DrawFlags::ON_TERRAIN.bits() != 0)
        .count();
    let batches = frame.batches.len();

    // And the render state each batch asks for, which is the input to the key that only the
    // drawable can supply: the view does not know it and the family does not either. More than one
    // answer in one frame is the whole reason it is in the key, and this frame has three.
    let mut states: BTreeMap<String, usize> = BTreeMap::new();
    for batch in frame.batches.iter() {
        *states
            .entry(state(pipelines::Enabled::of(batch.key.draw_flags)))
            .or_default() += 1;
    }

    // And how each batch composites, which is the last thing a draw needs that nothing on the wire
    // carries: mbgl writes a `ColorMode` per render layer, and `families::Composite` is that table.
    // Read here rather than below because `frame` borrows the host, which the uploads take back.
    //
    // By (family, pass) and not by batch, because that is what the answer depends on -- and in
    // this frame the two differ: its background is batched twice, once in each pass, and those are
    // the two arms of the one family that has two.
    let mut composites: Vec<(&'static str, u8, Blend)> = frame
        .batches
        .iter()
        .filter_map(|batch| {
            let family = families::for_wire(batch.key.builtin_shader)?;
            Some((family.name, batch.key.pass, family.blend(batch.key.pass)))
        })
        .collect();
    composites.sort_unstable_by_key(|(name, pass, _)| (*name, *pass));
    composites.dedup_by_key(|(name, pass, _)| (*name, *pass));

    let joiner = frame.joiner;
    // What each geometry samples, as the producer named it: a `TextureRef` is a texture id and the
    // slot it arrives at, and both halves of a set's image pair are filled from one.
    let mut refs: BTreeMap<
        tessella_capture_abi::envelope::GeometryId,
        Vec<tessella_capture_abi::envelope::TextureRef>,
    > = BTreeMap::new();
    let plans: Vec<(
        tessella_capture_abi::envelope::GeometryId,
        vertices::Plan,
        buffers::Needs,
        Vec<tessella_capture_abi::envelope::Segment>,
    )> = {
        let mut out = Vec::new();
        let mut families_seen = BTreeSet::new();
        let mut refused = Vec::new();
        for (geometry, shader) in &wanted {
            let Some(drawable) = joiner.drawable(*geometry, view) else {
                refused.push(format!("geometry {} has no use", geometry.0));
                continue;
            };
            let add = &drawable.geometry.add;
            // By the id the wire carries, not by decoding it: the ground's is tessella's own 128
            // and `BuiltIn::from_repr` answers `None` for it, so a lookup that decoded first
            // refused the one family mbgl has no shader for -- which is what this said when it
            // reported "geometry 1 names shader 128, which has no family here".
            let Some(family) = families::for_wire(*shader) else {
                refused.push(format!(
                    "geometry {} names shader {shader}, which has no family here",
                    geometry.0
                ));
                continue;
            };
            families_seen.insert(family.name);
            // Both runs, because a family with instance attributes has them on the wire too: the
            // ABI splits a geometry's `attrs` from its `instance_attrs`, and which run a
            // descriptor arrived in is the only thing that says a binding advances per instance.
            let plan = match vertices::plan_instanced(
                family.attributes,
                &drawable.geometry.attrs,
                family.instance_attributes,
                &drawable.geometry.instance_attrs,
            ) {
                Ok(plan) => plan,
                Err(why) => {
                    refused.push(format!(
                        "geometry {} on {}: the plan refuses it: {why:?}",
                        geometry.0, family.name
                    ));
                    continue;
                }
            };
            // The plan itself is kept, not only what it needs on the device: the pipeline key
            // and the module both come off it, and this is the only place it is built.
            let needs = buffers::needs(&plan, add.indexes);
            refs.insert(*geometry, drawable.geometry.texture_refs.clone());
            out.push((*geometry, plan, needs, drawable.geometry.segments.clone()));
        }
        println!(
            "  the plans             {}   {} of {} geometries planned, {} famil{}: {}",
            if refused.is_empty() { "ok  " } else { "WARN" },
            out.len(),
            wanted.len(),
            families_seen.len(),
            if families_seen.len() == 1 { "y" } else { "ies" },
            families_seen.iter().copied().collect::<Vec<_>>().join(" ")
        );
        for why in refused.iter().take(4) {
            println!("                             {why}");
        }
        out
    };

    let device = match Open::preferred() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping the upload: {why}");
            return Ok(());
        }
    };
    println!("  device: {} ({})", device.name, device.class);

    let gpu = device.gpu();
    let mut store = Store::new();
    let mut failed = Vec::new();
    for (geometry, _, needs, segments) in &plans {
        if let Err(why) = store.upload(gpu, *geometry, needs, segments, &|reference| {
            slab::resolve(region, &slabs, reference)
        }) {
            failed.push(format!("geometry {}: {why}", geometry.0));
        }
    }
    println!(
        "  the upload            {}   {} of {} resident, {} bytes on the device, {} failed",
        if failed.is_empty() { "ok  " } else { "FAIL" },
        store.resident(),
        plans.len(),
        store.total_bytes(),
        failed.len()
    );
    for why in failed.iter().take(4) {
        println!("                             {why}");
    }
    if !failed.is_empty() {
        return Err(format!("{} geometries did not upload", failed.len()));
    }

    // The textures the stream carried, declared at the sizes it named and filled from the rows
    // `Upload::rows` resolves. A flat capture has none and this does nothing; a raised one has the
    // ground's DEM, which every layer standing on it samples -- so without this the sets of a
    // raised frame are refused, "the set has 1 texture bindings and 0 arrived".
    //
    // Its own submission, because `Images` records copies and a frame cannot begin until they have
    // landed: the draw below samples what this leaves in `SHADER_READ_ONLY_OPTIMAL`.
    let mut images = Images::new();
    let mut staged = Ok(0);
    device
        .submit(|record: Recorder<'_>| {
            staged = stage(&mut images, gpu, record, &host);
        })
        .map_err(|why| format!("staging the textures: {why}"))?;
    let empty = staged?;
    println!(
        "  the textures          ok   {} declared, {} bytes on the device, {empty} of no area",
        images.textures(),
        images.total_bytes()
    );

    // The blocks, each under the view and layer the producer named -- which for the frame-wide
    // ones is not a layer at all. `line` and `circle` both declare `GlobalPaintParamsUBO` and both
    // are in this frame, so a set written for either needs a buffer the layer does not hold.
    let mut held = Blocks::new();
    let mut frame_wide = 0;
    for work in host.uploads().work() {
        let Upload::Uniforms {
            view: at,
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
            view: *at,
            layer: *layer_index,
        };
        if *layer_index < 0 {
            frame_wide += 1;
        }
        // Sixteen bytes, which is every block's declared alignment, so a chunk never straddles
        // anything meaningful and `flush` merges runs of them back into entry-sized ranges.
        let grain = 16;
        held.declare(gpu, which, *slot, data.len() / grain, grain)
            .map_err(|why| format!("layer {layer_index} slot {slot}: {why}"))?;
        held.replace(which, *slot, data)
            .map_err(|why| format!("layer {layer_index} slot {slot}: {why}"))?;
        held.flush(which, *slot, grain)
            .map_err(|why| format!("layer {layer_index} slot {slot}: {why}"))?;
    }
    println!(
        "  the blocks            ok   {} buffers, {frame_wide} of them frame-wide",
        held.buffers()
    );

    // Then a descriptor set per (layer, family), which is what `record::content` looks one up by --
    // each on the surface its own drawables ask for, derived here rather than assumed. The three
    // inputs are the camera's projection, the batch's flags, and whether the layer holds a bend
    // block; the first two came off the capture above and the third is a question for the blocks
    // just brought up, which is why it is asked here and not there.
    let bend_slot = tessella_capture_abi::globe_ubo::ID_GLOBE_BEND_UBO;
    let mut layouts = Vec::with_capacity(programs.len());
    let mut refused = Vec::new();
    let mut placements: BTreeSet<Surface> = BTreeSet::new();
    let mut placed = 0;
    for (shader, layer) in &programs {
        let Some(family) = families::for_wire(*shader) else {
            refused.push(format!("shader {shader} has no family here"));
            continue;
        };
        let which = blocks::Which {
            view,
            layer: *layer,
        };
        let bend = held
            .buffer(slots::held_at(which, bend_slot), bend_slot)
            .is_some();
        let mut asked: Vec<Surface> = flags
            .get(&(*shader, *layer))
            .into_iter()
            .flatten()
            .map(|flags| family.surface(Surface::of(projection, *flags, bend)))
            .collect();
        asked.sort_unstable();
        asked.dedup();
        let [surface] = asked[..] else {
            refused.push(format!(
                "{} on layer {layer} asks for {asked:?}, and a set is keyed by (view, layer, \
                 shader) alone",
                family.name
            ));
            continue;
        };
        placements.insert(surface);
        placed += 1;
        let bindings = pipelines::bindings(family, surface);
        let layout = pipelines::layout(gpu, &bindings)
            .map_err(|why| format!("{} on {surface:?}: {why}", family.name))?;
        layouts.push((*shader, *layer, family.name, surface, bindings, layout));
    }
    println!(
        "  the surfaces          ok   a {projection:?} camera, {raised} of {batches} batches \
         raised, {placed} of {} programs placed, on {}",
        programs.len(),
        placements
            .iter()
            .map(|surface| format!("{surface:?}"))
            .collect::<Vec<_>>()
            .join(" and ")
    );

    let widest = layouts
        .iter()
        .map(|(_, _, _, _, bindings, _)| bindings.clone())
        .max_by_key(Vec::len)
        .unwrap_or_default();
    // Sized for every set the frame needs, which is more than one per program: the masks have one
    // of their own at layer -1, and the instanced extrusion has a group-one set per drawable over
    // that drawable's own instance buffer. A pool short by one answers `ERROR_OUT_OF_POOL_MEMORY`
    // on the set it cannot allocate, which is a frame that stops rather than draws.
    let instanced = wanted
        .iter()
        .filter(|(_, shader)| {
            families::for_wire(*shader).is_some_and(|family| !family.instance_attributes.is_empty())
        })
        .count();
    let mut sets = Sets::new(
        gpu,
        u32::try_from(layouts.len() + 1 + instanced).unwrap_or(u32::MAX),
        &widest,
    )
    .map_err(|why| format!("the descriptor pool: {why}"))?;
    let mut written = Vec::with_capacity(layouts.len());
    for (shader, layer, name, _, bindings, layout) in &layouts {
        let which = blocks::Which {
            view,
            layer: *layer,
        };
        // The views this pair's drawables name, as the producer named them: the family's own
        // textures and the surface's, which for a raised layer is the ground's DEM at
        // `surface::TERRAIN_ELEVATION_SLOT`. `bound_from` places them by slot rather than by
        // arrival order, which is the defect #95 was.
        let named = drawn_by
            .get(&(*shader, *layer))
            .and_then(|geometry| refs.get(geometry))
            .map_or(&[][..], Vec::as_slice);
        let bound = descriptors::bound_from(&images, bindings, named)
            .map_err(|why| format!("{name} on layer {layer}: its textures: {why}"))?;
        let set = sets
            .write(layout, bindings, which, *shader, &held, &bound)
            .map_err(|why| format!("{name} on layer {layer}: {why}"))?;
        written.push((which, *shader, *name, set));
    }

    // Every one of them still reachable, which is the half the writes alone do not show. Three of
    // this frame's six layers are drawn by *two* families -- two and three by a fill and its
    // outline, five by an extrusion and its walls -- and a set keyed by the layer alone kept only
    // the second of each. The write succeeded either way; what was lost was the lookup.
    for (which, shader, name, set) in &written {
        if sets.get(*which, *shader) != Some(*set) {
            return Err(format!(
                "{name} on layer {} wrote a set and cannot find it again",
                which.layer
            ));
        }
    }
    let written = written.len();
    let shared = programs.len() - layers.len();
    println!(
        "  the sets              {}   {written} of {} written, {shared} sharing a layer, {} refused",
        if refused.is_empty() { "ok  " } else { "WARN" },
        programs.len(),
        refused.len()
    );
    for why in refused.iter().take(4) {
        println!("                             {why}");
    }

    println!(
        "  the state             ok   {batches} batches, {} distinct: {}",
        states.len(),
        states
            .iter()
            .map(|(name, count)| format!("{name} {count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // The blend each of them would be built with. Not a draw, and not a pipeline: it is the one
    // input to `pipelines::Key` that the capture cannot supply, so this is where the table meets a
    // real frame's batches rather than a fixture's chosen pair.
    let replaced: Vec<String> = composites
        .iter()
        .filter(|(_, _, blend)| *blend == Blend::Unblended)
        .map(|(name, pass, _)| format!("{name} in {}", named(*pass)))
        .collect();
    println!(
        "  the blends            ok   {} (family, pass) pairs, {} by replacement{}",
        composites.len(),
        replaced.len(),
        if replaced.is_empty() {
            String::new()
        } else {
            format!(": {}", replaced.join(", "))
        }
    );

    // ------------------------------------------------------------------ the draw
    //
    // Everything above is what a draw would bind; this is the draw. The masks first, then one
    // pipeline per batch key, then `record::content` over the frame's own batches, then the pixels
    // back.

    // The attachment, from the families the frame draws rather than from a constant:
    // `Attachment::DepthStencil` is "every view that draws an extrusion", and `Family::height` is
    // which families those are. This frame has two of them.
    let extrudes = programs
        .iter()
        .any(|(shader, _)| families::for_wire(*shader).is_some_and(|family| family.height));
    let attachment = if extrudes {
        Attachment::DepthStencil
    } else {
        Attachment::StencilOnly
    };
    let screen = Screen::new(&device, WIDTH, HEIGHT, attachment)
        .map_err(|why| format!("the target: {why}"))?;

    // The masks, which every content draw that asked for the stencil test is held against. One
    // matrix per tile at the slot mbgl's own constant names, keyed frame-wide: a mask belongs to
    // the renderer rather than to any layer, which is what `slots::FRAME_WIDE` is.
    let mask_slot = tessella_capture_abi::generated::ubo_slots::ID_CLIPPING_MASK_UBO;
    let mask_at = blocks::Which {
        view,
        layer: slots::FRAME_WIDE,
    };
    let order: Vec<(TileId, [f32; 16])> = host
        .clips(view)
        .masks()
        .map(|(tile, matrix)| (tile, *matrix))
        .collect();
    let mask_bytes: Vec<u8> = order
        .iter()
        .flat_map(|(_, matrix)| matrix.iter().flat_map(|value| value.to_le_bytes()))
        .collect();
    held.declare(gpu, mask_at, mask_slot, order.len(), 64)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    held.replace(mask_at, mask_slot, &mask_bytes)
        .map_err(|why| format!("the mask buffer: {why}"))?;
    held.flush(mask_at, mask_slot, 0)
        .map_err(|why| format!("the mask buffer: {why}"))?;

    let mask_bindings = [pipelines::Binding {
        group: 0,
        binding: 0,
        kind: pipelines::Kind::StorageBuffer,
        slot: Some(mask_slot),
    }];
    let mask_layout =
        pipelines::layout(gpu, &mask_bindings).map_err(|why| format!("the mask layout: {why}"))?;
    let mask_module = gpu
        .shader(&compile(masks::BODY)?)
        .map_err(|why| format!("the mask module: {why}"))?;
    let mask_pipeline = pipelines::build_mask(gpu, &mask_layout, &mask_module, screen.targets)
        .map_err(|why| format!("the mask pipeline: {why}"))?;
    let mask_set = sets
        .write(
            &mask_layout,
            &mask_bindings,
            mask_at,
            BuiltIn::ClippingMaskProgram as i32,
            &held,
            &[],
        )
        .map_err(|why| format!("the mask set: {why}"))?;

    // Re-planned, because a plan borrows the host and the first one ended at the uploads. The
    // batches, the joiner, the clips and the camera all come out of one borrow of it.
    let frame = host
        .plan(view)
        .ok_or_else(|| format!("view {} planned nothing the second time", view.0))?;
    let partition = frame.clips.partition();

    // One program per batch key: the pipeline that batch's own state asks for, and the layout its
    // set binds through. Built here and not inside the resolver, which `record::content` takes as a
    // `&dyn Fn` and so cannot hold the device mutably.
    //
    // The module is per (family, permutation): two batches of one family whose paint properties are
    // constant in different places read different uniforms and are different text. The pipeline is
    // per key on top of that, which is where the surface, the blend, the pass and the drawable's own
    // state come in.
    let mut modules: BTreeMap<(i32, u64), tessella_vk::ShaderModule<'_>> = BTreeMap::new();
    let mut built: Vec<tessella_vk::Pipeline<'_>> = Vec::new();
    let mut resolved: BTreeMap<ProgramKey, record::Program> = BTreeMap::new();
    let mut undrawable = Vec::new();
    for batch in frame.batches.iter() {
        let at = program_key(batch.key);
        if resolved.contains_key(&at) {
            continue;
        }
        let Some(family) = families::for_wire(batch.key.builtin_shader) else {
            undrawable.push(format!(
                "shader {} has no family here",
                batch.key.builtin_shader
            ));
            continue;
        };
        let layer = i32::try_from(batch.key.layer_index).unwrap_or(i32::MAX);
        let Some((_, _, _, surface, _, layout)) = layouts
            .iter()
            .find(|(shader, held_at, ..)| *shader == batch.key.builtin_shader && *held_at == layer)
        else {
            undrawable.push(format!("{} on layer {layer} has no layout", family.name));
            continue;
        };
        // The plan of the batch's first drawable. A batch is one program by construction -- the
        // merge key carries mbgl's permutation hash -- so any of its geometries describes the
        // vertex input, and the first is the one that certainly exists.
        let Some(geometry) = batch.geometries.first() else {
            undrawable.push(format!("{} on layer {layer} has no geometry", family.name));
            continue;
        };
        let Some((_, plan, _, _)) = plans.iter().find(|(id, ..)| id == geometry) else {
            undrawable.push(format!("geometry {} has no plan", geometry.0));
            continue;
        };
        let key = pipelines::key(
            family.shader,
            *surface,
            plan,
            family.blend(batch.key.pass),
            pipelines::Depth::of(batch.key.pass, batch.key.draw_flags),
            pipelines::Enabled::of(batch.key.draw_flags),
        );
        let module = match modules.entry((batch.key.builtin_shader, key.permutation)) {
            std::collections::btree_map::Entry::Occupied(had) => had.into_mut(),
            std::collections::btree_map::Entry::Vacant(empty) => {
                let source = family
                    .module(*surface, &plan.absent)
                    .map_err(|why| format!("{} does not assemble: {why:?}", family.name))?;
                let module = gpu
                    .shader(&compile(&source)?)
                    .map_err(|why| format!("{}: the module: {why}", family.name))?;
                empty.insert(module)
            }
        };
        let pipeline = pipelines::build(gpu, &key, layout, module, screen.targets)
            .map_err(|why| format!("{} on layer {layer}: the pipeline: {why}", family.name))?;

        // And the group-one set per drawable, for the one family whose attributes advance per
        // instance: its slot cannot ride `firstInstance`, which also addresses the instance-rate
        // fetch, so the instance buffer is bound through a set of its own.
        let per_drawable = !family.instance_attributes.is_empty();
        if per_drawable {
            for geometry in batch.geometries {
                let Some((_, _, needs, _)) = plans.iter().find(|(id, ..)| id == geometry) else {
                    continue;
                };
                let shared = *needs.storage.first().ok_or_else(|| {
                    format!("{}: an instanced plan names no shared buffer", family.name)
                })?;
                let buffer = store
                    .vertex_buffer(*geometry, shared)
                    .ok_or_else(|| format!("geometry {} has no buffer {shared}", geometry.0))?;
                sets.write_instances(layout, *geometry, buffer)
                    .map_err(|why| format!("geometry {}: the instance set: {why}", geometry.0))?;
            }
        }

        resolved.insert(
            at,
            record::Program {
                pipeline: pipeline.raw(),
                layout: layout.pipeline(),
                instances: vertices::instances(plan),
                per_drawable,
            },
        );
        built.push(pipeline);
    }
    // Batches rather than keys: a program this crate has no family for may be several batches, and
    // what the recording is checked against below is how many draws were skipped.
    let unprogrammed = frame
        .batches
        .iter()
        .filter(|batch| !resolved.contains_key(&program_key(batch.key)))
        .count();
    println!(
        "  the pipelines         {}   {} for {batches} batches, {} modules, {} undrawable, \
         {unprogrammed} batches skipped",
        if undrawable.is_empty() {
            "ok  "
        } else {
            "WARN"
        },
        built.len(),
        modules.len(),
        undrawable.len()
    );
    for why in undrawable.iter().take(4) {
        println!("                             {why}");
    }

    // The depth range the layers divide, which is the order's layer count and the camera's size --
    // the only place either arrives.
    let bands = frame
        .layer_count
        .and_then(|layers| record::DepthRange::new(layers, frame.camera.depth_range_size))
        .ok_or("the stream carried no layer count or no depth range")?;

    // Whether this run is a diagnostic rather than a check. Drawing one layer leaves most of the
    // frame at the clear and most batches undrawn, which is what the two assertions below are
    // about -- so under it they report and do not fail.
    let isolating = std::env::var("TSE_ONLY_LAYER").is_ok();
    let mut counts = Err("the recording did not run".to_string());
    device
        .submit(|record: Recorder<'_>| {
            target::frame(
                record,
                target::Host {
                    image: &screen.image,
                    view: &screen.view,
                    width: WIDTH,
                    height: HEIGHT,
                    layout: vk::ImageLayout::UNDEFINED,
                },
                &screen.depth,
                Some(CLEAR),
                |record| {
                    // The masks first: a content draw tests against a stencil they wrote, so recording
                    // it first tests an empty buffer. One draw per tile, because the reference and the
                    // write mask are the tile's and both are dynamic.
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

                    counts = record::content(
                        record,
                        frame.batches,
                        &record::Scene {
                            store: &store,
                            sets: &sets,
                            joiner: frame.joiner,
                            partition,
                            view,
                            depth: Some(bands),
                            size: (WIDTH, HEIGHT),
                        },
                        // `TSE_ONLY_LAYER=<n>` draws that layer alone, which is how a color in
                        // the frame is traced to the layer that put it there -- the parks layer's
                        // black was found this way. The batch count below is not checked under it,
                        // because leaving layers undrawn is the point.
                        &|batch| {
                            if let Ok(only) = std::env::var("TSE_ONLY_LAYER")
                                && only.parse::<u32>() != Ok(batch.key.layer_index)
                            {
                                return None;
                            }
                            resolved.get(&program_key(batch.key)).copied()
                        },
                    )
                    .map_err(|why| format!("recording: {why}"));
                },
            );
            record.transition(
                &screen.image,
                target::LEAVES_IN,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            record.copy_to_buffer(&screen.image, screen.readback(), WIDTH, HEIGHT);
        })
        .map_err(|why| format!("the submission: {why}"))?;
    let counts = counts?;
    println!(
        "  the recording         {}   {} batches, {} drawables, {} draws, {} undrawn, {} unclipped",
        if counts.undrawn == 0 { "ok  " } else { "WARN" },
        counts.batches,
        counts.drawables,
        counts.draws,
        counts.undrawn,
        counts.unclipped
    );
    // Every batch this crate has a program for, and the ones it has none for accounted for rather
    // than ignored. A raised capture carries the ground itself -- `BUILTIN_TERRAIN_SHADER`, which
    // `families::ALL` has no entry for -- so the layers standing on it draw and the surface under
    // them does not. `record::content`'s own note names the case: "a batch this consumer does not
    // draw, which is a real case".
    if !isolating {
        if counts.batches + unprogrammed != batches {
            return Err(format!(
                "{} of {batches} batches recorded and {unprogrammed} had no program, which does \
                 not account for all of them",
                counts.batches
            ));
        }
        if counts.undrawn != unprogrammed {
            return Err(format!(
                "{} batches were skipped at the recording and {unprogrammed} had no program",
                counts.undrawn
            ));
        }
    }

    // And the pixels. Two things are asked of them, and both are properties of this style rather
    // than of any one family: the background is the bottom layer and covers the viewport, so no
    // pixel may be left at the clear; and more than one color has to be there, or the only thing
    // that drew was that background.
    let pixels = screen.pixels()?;
    // Optionally the frame itself, for a human: a binary PPM, which needs no encoder.
    if let Ok(path) = std::env::var("TSE_IMAGE") {
        let mut out = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        for pixel in &pixels {
            out.extend_from_slice(&pixel[0..3]);
        }
        std::fs::write(&path, out).map_err(|why| format!("writing {path}: {why}"))?;
        println!("  wrote {path}");
    }
    let cleared = pixels.iter().filter(|pixel| **pixel == CLEARED).count();
    let mut counts: BTreeMap<[u8; 4], usize> = BTreeMap::new();
    for pixel in &pixels {
        *counts.entry(*pixel).or_default() += 1;
    }
    // And the commonest color, which is not an assertion -- a capture of another style has another
    // one -- but is what a reader checks the frame against. For the Berlin capture it is
    // `[234, 230, 221]` over 538,300 pixels, which is `#eae6dd`, the `background-color` of the
    // scene that capture was taken through.
    let (common, share) = counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|(color, count)| (*color, *count))
        .unwrap_or_default();
    println!(
        "  the pixels            {}   {} of {} at the clear, {} colors, {common:?} over {share}",
        if isolating {
            "one "
        } else if cleared == 0 && counts.len() > 1 {
            "ok  "
        } else {
            "FAIL"
        },
        cleared,
        pixels.len(),
        counts.len(),
    );
    if cleared != 0 && !isolating {
        return Err(format!(
            "{cleared} pixels are still {CLEARED:?}, which the background covers"
        ));
    }
    if counts.len() < 2 && !isolating {
        return Err("the frame is one color, so only the background drew".to_string());
    }

    Ok(())
}
