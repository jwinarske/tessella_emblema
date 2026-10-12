//! Drives `Host` from a capture file instead of a hand-written ring.
//!
//! ```text
//! cargo run --example replay-a-capture -- <path.tslcap>
//! ```
//!
//! Written with `capture-render --capture <path>` on the producer's side, which builds the frame
//! from a real style and a real tile. Every fixture in `benches/` writes its own ring, which is
//! what makes it a fixture: it carries the records its author thought of. Three defects this
//! repository found recently -- #95, #101, #105 -- were each a case the fixtures happened not to
//! send, and each was found by making one send what a producer sends.
//!
//! This reads what a producer actually emitted. No device: what it reports is what `Host` made of
//! the stream and whether every geometry's bytes survived the file, which is the half a fixture
//! cannot check at all.
//!
//! # What it prints, and why each line is worth a line
//!
//! - **the capture**: records, slabs and region bytes, so a file that lost its geometry is
//!   visible before anything is blamed on the consumer.
//! - **the stream**: `Progress`, which separates a kind this build does not know from a record it
//!   could not parse from one naming an undeclared view -- three different faults that all look
//!   like a blank map.
//! - **the frame**: whether a view was declared, whether its camera and order agree, and what the
//!   batches collapsed to.
//! - **the table**: that no slab runs past the region and that none overlaps another. `resolve`
//!   answers whether a reference *fits*, not whether it lands on the bytes the producer wrote --
//!   so this is what stands between a wrong table and a frame of plausible wrong geometry.
//! - **the slabs**: every attribute of every geometry resolved against the region.
//!
//! # What it still cannot tell you
//!
//! That a run resolves to the *right* bytes. A table that is internally consistent but shifted
//! would pass every check here, because nothing in the capture says what the bytes should be. The
//! honest fix is a digest per slab in the format, which is a change to the format rather than to
//! this.

// A probe is a program, and it reports by printing.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    // The report is one pass over the capture, in the order the checks depend on each other, and
    // `benches/first_pixel.rs` allows the same for the same reason.
    clippy::too_many_lines
)]

use core::fmt::Write as _;

use tessella_capture_abi::EnvelopeKind;
use tessella_capture_abi::capture;
use tessella_capture_abi::envelope::ViewId;
use tessella_capture_abi::ring::Ring;
use tessella_consume::host::Host;
use tessella_consume::slab::{self, Slab};

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        println!("usage: replay-a-capture <path.tslcap>");
        println!("  written by: capture-render --style <s> --tile <t> --capture <path.tslcap>");
        return;
    };
    match replay(&path) {
        Ok(report) => print!("{report}"),
        Err(why) => {
            println!("  replay                FAIL {why}");
            std::process::exit(1);
        }
    }
}

fn replay(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|why| format!("reading {path}: {why}"))?;
    let read = capture::decode(&bytes).ok_or_else(|| {
        format!(
            "{path} is {} bytes and not a capture -- the magic, the header's lengths, or both",
            bytes.len()
        )
    })?;

    let region = read.region();
    // `capture::Slab` into the consumer's, which is the translation the format exists to keep
    // trivial: the two are the same two fields, and `capture` holds its own copy so that the
    // crate both halves depend on does not depend on the consumer.
    let slabs: Vec<Slab> = read
        .slabs()
        .into_iter()
        .map(|slab| Slab {
            offset: slab.offset,
            length: slab.length,
        })
        .collect();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "  the capture           ok   {} records, {} slabs, {} bytes of region",
        read.len(),
        slabs.len(),
        region.len()
    );

    // The table, before anything is resolved against it.
    //
    // `resolve` answers whether a reference *fits*, not whether it lands on the bytes the
    // producer put there -- so a table whose offsets are all wrong still resolves everything and
    // hands back the wrong geometry. Measured: writing every slab's offset as zero left the
    // resolve count at "0 unresolved" and nothing else noticed.
    //
    // Two slabs claiming the same bytes cannot both be right, so an overlap is a fault and fails
    // here. A gap is only reported: the format says a slab is an offset and a length within the
    // region and does not require them to tile it, so a producer may legitimately leave one.
    let mut ordered: Vec<(usize, &Slab)> = slabs.iter().enumerate().collect();
    ordered.sort_by_key(|(_, slab)| slab.offset);
    let mut overlaps = Vec::new();
    let mut gaps = 0u64;
    let mut reach = 0u64;
    for (id, slab) in &ordered {
        let end = slab.offset.saturating_add(slab.length);
        if end > region.len() as u64 {
            return Err(format!(
                "slab {id} runs to {end} and the region is {} bytes",
                region.len()
            ));
        }
        if slab.length == 0 {
            continue;
        }
        if slab.offset < reach {
            overlaps.push(format!(
                "slab {id} starts at {} inside {reach}",
                slab.offset
            ));
        } else {
            gaps += slab.offset - reach;
        }
        reach = reach.max(end);
    }
    let _ = writeln!(
        out,
        "  the table             {}   {} byte(s) covered, {gaps} in gaps, {} overlap(s)",
        if overlaps.is_empty() { "ok  " } else { "FAIL" },
        reach,
        overlaps.len()
    );
    for why in overlaps.iter().take(3) {
        let _ = writeln!(out, "                             {why}");
    }
    if !overlaps.is_empty() {
        print!("{out}");
        return Err(format!(
            "{} slab(s) overlap, so the table cannot be describing where the producer put the \
             bytes -- every reference may still resolve, to the wrong geometry",
            overlaps.len()
        ));
    }

    // Onto a ring, through `Producer::write`, which is how the records reach a `Host` at all --
    // and the reason a capture holds records rather than a copy of the ring: this is the public
    // way in, and it reconstructs an equivalent stream without the file knowing the ring's
    // layout.
    //
    // Sized from the records rather than fixed: a capture of a busy frame does not fit a
    // guess, and `write` answering `Full` would look like a short stream.
    let mut ring = Ring::new((bytes.len() * 2).next_power_of_two().max(1 << 16));
    let mut written = 0usize;
    let mut refused = Vec::new();
    {
        let (producer, _) = ring.split();
        for record in read.records() {
            let Some(kind) = u16::try_from(record.kind)
                .ok()
                .and_then(EnvelopeKind::from_repr)
            else {
                // A kind this build does not know. Counted rather than fatal: the capture may be
                // a newer producer's, and every other record is still worth replaying.
                refused.push(record.kind);
                continue;
            };
            producer
                .write(kind, record.record, record.payload)
                .map_err(|_| format!("the ring filled after {written} records"))?;
            written += 1;
        }
    }
    let _ = writeln!(
        out,
        "  onto the ring         ok   {written} written, {} kind(s) this build does not know",
        refused.len()
    );
    if !refused.is_empty() {
        let _ = writeln!(out, "                             {refused:?}");
    }

    let mut host = Host::new();
    let progress = host.read(ring.consumer());
    let _ = writeln!(
        out,
        "  the stream            {}   {} read, {} unknown, {} malformed, {} undeclared",
        if progress.malformed == 0 && progress.undeclared == 0 {
            "ok  "
        } else {
            "WARN"
        },
        progress.records,
        progress.unknown,
        progress.malformed,
        progress.undeclared,
    );

    // The views the stream declared. A capture of one frame has one, but asking rather than
    // assuming is what makes this useful on a capture of something else.
    let mut found = None;
    for id in 0..8u32 {
        if host.declared(ViewId(id)) {
            found = Some(ViewId(id));
            break;
        }
    }
    let Some(view) = found else {
        let _ = writeln!(
            out,
            "  the frame             WARN no view was declared, so nothing can draw"
        );
        return Ok(out);
    };

    if !host.ready(view) {
        let _ = writeln!(
            out,
            "  the frame             WARN view {} declared but its camera and order disagree",
            view.0
        );
        return Ok(out);
    }

    let frame = host
        .plan(view)
        .ok_or_else(|| format!("view {} is ready and planned nothing", view.0))?;
    let batches = frame.batches.len();
    let drawables: usize = frame.batches.iter().map(|b| b.geometries.len()).sum();
    let layers: std::collections::BTreeSet<u32> =
        frame.batches.iter().map(|b| b.key.layer_index).collect();
    let depth = frame.layer_count.map_or_else(
        || "none sent".to_string(),
        |n| format!("{} layers", n.get()),
    );
    let _ = writeln!(
        out,
        "  the frame             ok   view {}, epoch {}, {batches} batches of {drawables} drawables, \
         {} layer(s), {depth}",
        view.0,
        frame.plan.epoch.0,
        layers.len(),
    );

    // Every attribute of every drawable, resolved against the region. The joiner is what pairs an
    // announcement with the use that draws it, so this walks what the frame actually draws rather
    // than every geometry announced.
    let joiner = frame.joiner;
    let mut asked = 0usize;
    let mut missing = Vec::new();
    for batch in frame.batches.iter() {
        for geometry in batch.geometries {
            let Some(drawable) = joiner.drawable(*geometry, view) else {
                missing.push(format!("geometry {} has no use", geometry.0));
                continue;
            };
            for desc in &drawable.geometry.attrs {
                asked += 1;
                if slab::resolve(region, &slabs, desc.source).is_none() {
                    missing.push(format!(
                        "geometry {} attribute {} wants slab {} +{} for {} bytes",
                        geometry.0,
                        desc.attr_id,
                        desc.source.slab,
                        desc.source.offset,
                        desc.source.length
                    ));
                }
            }
        }
    }
    let _ = writeln!(
        out,
        "  the slabs             {}   {asked} attribute run(s) asked, {} unresolved",
        if missing.is_empty() { "ok  " } else { "FAIL" },
        missing.len()
    );
    for why in missing.iter().take(4) {
        let _ = writeln!(out, "                             {why}");
    }
    if !missing.is_empty() {
        return Err(format!(
            "{} of {asked} attribute runs did not resolve, so the slab table and the region \
             disagree about where the geometry is",
            missing.len()
        ));
    }

    Ok(out)
}
