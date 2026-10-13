//! A captured frame's geometry, onto the device.
//!
//! ```text
//! TSE_CAPTURE=<path.tslcap> cargo bench --bench capture_frame
//! ```
//!
//! Written by `capture-render --capture <path>` on the producer's side. `examples/replay-a-capture`
//! is the device-free half -- it reads the file, drives `Host`, and checks every slab reference
//! resolves. This is the half that needs a GPU: it plans each drawable's vertex input against its
//! family's table and uploads the bytes.
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
//! What it does *not* do is draw. Recording and reading back a captured frame wants pipelines and
//! sets per family, which is its own piece of work; this stops where the bytes are resident.

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
use tessella_capture_abi::capture;
use tessella_capture_abi::envelope::ViewId;
use tessella_capture_abi::ring::Ring;
use tessella_consume::host::Host;
use tessella_consume::slab::{self, Slab};
use tessella_emblema::store::Store;
use tessella_emblema::{buffers, families, vertices};

use common::Open;

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

    Ok(())
}
