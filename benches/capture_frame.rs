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
//! What it still does *not* do is draw. That wants a pipeline per batch and the recording itself,
//! which is its own piece of work; this stops where every resource a draw would bind is on the
//! device and reachable, and where every input to the pipeline's key is in hand -- the last of
//! them being the blend, which the frame's background takes two different values of because it is
//! batched once in each pass.

// A bench is a program, and it reports by printing.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    // One pass over the capture, in the order the steps depend on each other, as
    // `benches/first_pixel.rs` allows for the same reason.
    clippy::too_many_lines
)]

mod common;

use std::collections::BTreeSet;

use tessella_capture_abi::EnvelopeKind;
use tessella_capture_abi::RenderPass;
use tessella_capture_abi::capture;
use tessella_capture_abi::envelope::ViewId;
use tessella_capture_abi::ring::Ring;
use tessella_consume::host::Host;
use tessella_consume::slab::{self, Slab};
use tessella_consume::upload::Upload;
use tessella_emblema::blocks::Blocks;
use tessella_emblema::descriptors::Sets;
use tessella_emblema::pipelines::Blend;
use tessella_emblema::store::Store;
use tessella_emblema::surface::Surface;
use tessella_emblema::{blocks, buffers, families, pipelines, vertices};

use common::Open;

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
    let plans: Vec<(
        tessella_capture_abi::envelope::GeometryId,
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
            let Some(family) = add.builtin_shader().and_then(families::family) else {
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
            out.push((
                *geometry,
                buffers::needs(&plan, add.indexes),
                drawable.geometry.segments.clone(),
            ));
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
    for (geometry, needs, segments) in &plans {
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

    // Then a descriptor set per (layer, family), which is what `record::content` looks one up by.
    //
    // `Surface::Plane` for every one of them, and the capture says so rather than this bench
    // assuming it: a bent or raised frame carries the surface's own blocks, and this one's uploads
    // name slots 0, 2, 3, 4 and 5 and neither the globe bend's eleven nor the terrain block's 128.
    // Which surface a view is drawn on is the consumer's decision from the camera, and making it
    // is not this bench's business yet.
    let surface = Surface::Plane;
    let mut layouts = Vec::with_capacity(programs.len());
    let mut refused = Vec::new();
    for (shader, layer) in &programs {
        let Some(family) = families::for_wire(*shader) else {
            refused.push(format!("shader {shader} has no family here"));
            continue;
        };
        if !family.textures.is_empty() {
            refused.push(format!(
                "{} samples {} texture(s), and no batch in this frame binds any",
                family.name,
                family.textures.len()
            ));
            continue;
        }
        let bindings = pipelines::bindings(family, surface);
        let layout = pipelines::layout(gpu, &bindings)
            .map_err(|why| format!("{} on {surface:?}: {why}", family.name))?;
        layouts.push((*shader, *layer, family.name, bindings, layout));
    }

    let widest = layouts
        .iter()
        .map(|(_, _, _, bindings, _)| bindings.clone())
        .max_by_key(Vec::len)
        .unwrap_or_default();
    let mut sets = Sets::new(
        gpu,
        u32::try_from(layouts.len()).unwrap_or(u32::MAX),
        &widest,
    )
    .map_err(|why| format!("the descriptor pool: {why}"))?;
    let mut written = Vec::with_capacity(layouts.len());
    for (shader, layer, name, bindings, layout) in &layouts {
        let which = blocks::Which {
            view,
            layer: *layer,
        };
        let set = sets
            .write(layout, bindings, which, *shader, &held, &[])
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

    Ok(())
}
