// SPDX-License-Identifier: BSD-2-Clause
//! The geometry store, against a real device.
//!
//! A bench rather than a test for the reason `first_pixel` is one: it needs a GPU and CI has none.
//! `tests/store_layout.rs` holds the half that does not, which is the arithmetic.
//!
//! # What this proves that the layout tests cannot
//!
//! That the bytes arrive. The layout tests say where each buffer should sit; this writes a geometry,
//! reads it back off the device and compares — so an offset that is arithmetically sound and bound to
//! the wrong place is caught here and nowhere else. `vkBindBufferMemory` also validates alignment
//! against the driver's own requirement, which no host-side test can know.
//!
//! Run with `cargo bench --bench geometry_store`.

mod common;

use ash::vk;
use tessella_capture_abi::envelope::{GeometryId, SlabRef};
use tessella_emblema::buffers::{self, Needs, Reads};
use tessella_emblema::store::Store;
use tessella_emblema::vertices::{Bound, Plan};

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// One segment over a whole index buffer, which is what a bucket with no sub-ranges sends.
fn segments() -> Vec<tessella_capture_abi::envelope::Segment> {
    vec![tessella_capture_abi::envelope::Segment {
        vertex_offset: 0,
        index_offset: 0,
        vertex_length: 4,
        index_length: 12,
    }]
}

const fn at(slab: u32, offset: u32, length: u32) -> SlabRef {
    SlabRef {
        slab,
        offset,
        length,
    }
}

/// A geometry of two distinct vertex buffers and an index buffer, with bytes to match.
fn two_and_indexes() -> (Needs, Vec<u8>, Vec<u8>, Vec<u8>) {
    let first = at(0, 0, 64);
    let second = at(1, 0, 48);
    let indexes = at(2, 0, 24);
    let needs = Needs {
        vertices: vec![first, second],
        reads: vec![Reads { slot: 0, buffer: 0 }, Reads { slot: 1, buffer: 1 }],
        indexes: Some(indexes),
    };
    // Distinct per buffer and varying within each, so a swap or a shared offset cannot pass.
    let a: Vec<u8> = (0..64u8).collect();
    let b: Vec<u8> = (0..48u8).map(|byte| 128 + byte).collect();
    let i: Vec<u8> = (0..24u8).map(|byte| 200 + byte).collect();
    (needs, a, b, i)
}

/// What the bytes a geometry's references name, for the store to resolve against.
fn resolver<'b>(pairs: &'b [(SlabRef, &'b [u8])]) -> impl Fn(SlabRef) -> Option<&'b [u8]> + 'b {
    move |wanted| {
        pairs
            .iter()
            .find(|(reference, _)| *reference == wanted)
            .map(|(_, bytes)| *bytes)
    }
}

fn round_trip(device: &Open) -> Result<(), String> {
    let (needs, a, b, i) = two_and_indexes();
    let pairs = [
        (needs.vertices[0], a.as_slice()),
        (needs.vertices[1], b.as_slice()),
        (needs.indexes.expect("indexes"), i.as_slice()),
    ];
    let resolve = resolver(&pairs);

    let mut store = Store::new();
    let geometry = GeometryId(1);
    store
        .upload(device.gpu(), geometry, &needs, &segments(), &resolve)
        .map_err(|why| format!("upload: {why}"))?;

    if !store.holds(geometry) {
        return Err("the store does not hold what it uploaded".into());
    }
    if store.vertex_buffer(geometry, 0) == store.vertex_buffer(geometry, 1) {
        return Err("two distinct references became one buffer".into());
    }
    if store.index_buffer(geometry).is_none() {
        return Err("the index buffer is missing".into());
    }

    // The bytes, off the device. This is the assertion the layout tests cannot make.
    for (index, wanted) in [a.as_slice(), b.as_slice()].iter().enumerate() {
        let mut got = vec![0u8; wanted.len()];
        store
            .read_vertex_bytes(geometry, index, &mut got)
            .map_err(|why| format!("read back {index}: {why}"))?;
        if got != *wanted {
            return Err(format!(
                "buffer {index} came back wrong: first eight {:?} against {:?}",
                &got[..8.min(got.len())],
                &wanted[..8.min(wanted.len())]
            ));
        }
    }

    let footprint = store.bytes(geometry).ok_or("no footprint")?;
    let claimed: u64 = 64 + 48 + 24;
    if footprint < claimed {
        return Err(format!(
            "the allocation is {footprint} bytes for {claimed} of references"
        ));
    }
    println!("  round trip            ok   {footprint} bytes for {claimed} claimed");

    store.free(&[geometry]);
    if store.holds(geometry) || store.total_bytes() != 0 {
        return Err("freeing left something behind".into());
    }
    println!("  free                  ok   nothing resident");
    Ok(())
}

fn dedup(device: &Open) -> Result<(), String> {
    let interleaved = at(3, 0, 120);
    let bound = |slot: u32, offset: u32| Bound {
        slot,
        format: vk::Format::R32_SFLOAT,
        stride: 12,
        offset,
        vertex_offset: 0,
        source: interleaved,
        rate: vk::VertexInputRate::VERTEX,
    };
    let plan = Plan {
        bound: vec![bound(0, 0), bound(1, 4), bound(2, 8)],
        ..Plan::default()
    };
    let needs = buffers::needs(&plan, at(0, 0, 0));
    let bytes: Vec<u8> = (0..120u8).collect();
    let pairs = [(interleaved, bytes.as_slice())];
    let resolve = resolver(&pairs);

    let mut store = Store::new();
    let geometry = GeometryId(2);
    store
        .upload(device.gpu(), geometry, &needs, &segments(), &resolve)
        .map_err(|why| format!("upload: {why}"))?;

    // Three bindings, one buffer: the whole point of the dedup, and visible only through the handles.
    let handles: Vec<Option<vk::Buffer>> = needs
        .reads
        .iter()
        .map(|read| store.vertex_buffer(geometry, read.buffer))
        .collect();
    if handles.iter().any(|handle| *handle != handles[0]) {
        return Err(format!(
            "three descriptors over one interleaved buffer got {handles:?}"
        ));
    }
    if store.index_buffer(geometry).is_some() {
        return Err("a zero-length index reference became a buffer".into());
    }
    let footprint = store.bytes(geometry).ok_or("no footprint")?;
    println!("  dedup                 ok   one buffer, {footprint} bytes for 120 claimed");
    Ok(())
}

/// The bind list is one entry per binding, with a handle repeated for an interleaved buffer.
///
/// What `vkCmdBindVertexBuffers` is handed, and the thing a draw would otherwise rebuild from
/// `Needs::reads` every time. The dedup is what makes it worth checking: three descriptors over one
/// twelve-byte vertex are *three* bindings naming *one* handle, so a bind list built by walking the
/// buffers rather than the reads would be one entry long and leave two bindings unbound.
fn the_bind_list_follows_the_reads(device: &Open) -> Result<(), String> {
    let interleaved = at(3, 0, 120);
    let bound = |slot: u32, offset: u32| Bound {
        slot,
        format: vk::Format::R32_SFLOAT,
        stride: 12,
        offset,
        vertex_offset: 0,
        source: interleaved,
        rate: vk::VertexInputRate::VERTEX,
    };
    let plan = Plan {
        bound: vec![bound(0, 0), bound(1, 4), bound(2, 8)],
        ..Plan::default()
    };
    let needs = buffers::needs(&plan, at(0, 0, 0));
    let bytes: Vec<u8> = (0..120u8).collect();
    let pairs = [(interleaved, bytes.as_slice())];
    let resolve = resolver(&pairs);

    let mut store = Store::new();
    let geometry = GeometryId(5);
    store
        .upload(device.gpu(), geometry, &needs, &segments(), &resolve)
        .map_err(|why| format!("upload: {why}"))?;

    let (handles, offsets) = store.bindings(geometry).ok_or("no bindings")?;
    if handles.len() != 3 {
        return Err(format!(
            "{} bindings for three descriptors: a draw would leave {} unbound",
            handles.len(),
            3 - handles.len()
        ));
    }
    if offsets.len() != handles.len() {
        return Err("the offsets are not parallel to the handles".into());
    }
    if handles.iter().any(|handle| *handle != handles[0]) {
        return Err(format!(
            "three bindings over one interleaved buffer got {handles:?}"
        ));
    }
    if offsets.iter().any(|offset| *offset != 0) {
        return Err("a binding offset is not zero, so it is being read twice".into());
    }

    // And the segments came back, which is the other thing a draw needs and the store used to drop.
    let held = store.segments(geometry);
    if held.len() != 1 || held[0].index_length != 12 {
        return Err(format!("the segments did not survive the upload: {held:?}"));
    }
    if !store.segments(GeometryId(404)).is_empty() {
        return Err("a geometry the store does not hold claimed segments".into());
    }
    println!("  the bind list         ok   3 bindings, 1 handle, 1 segment of 12 indices");
    Ok(())
}

fn refusals(device: &Open) -> Result<(), String> {
    let needs = Needs {
        vertices: vec![at(9, 0, 32)],
        reads: vec![Reads { slot: 0, buffer: 0 }],
        indexes: None,
    };
    let mut store = Store::new();

    // Nothing resolves it.
    let nothing = |_: SlabRef| None;
    let unresolved = store.upload(device.gpu(), GeometryId(3), &needs, &segments(), &nothing);
    if unresolved.is_ok() {
        return Err("an unresolved reference was accepted".into());
    }
    if store.resident() != 0 {
        return Err("a refused upload left a geometry resident".into());
    }

    // It resolves, but short.
    let short_bytes = vec![0u8; 8];
    let pairs = [(needs.vertices[0], short_bytes.as_slice())];
    let resolve = resolver(&pairs);
    let short = store.upload(device.gpu(), GeometryId(4), &needs, &segments(), &resolve);
    if short.is_ok() {
        return Err("a reference resolving short was accepted".into());
    }
    if store.resident() != 0 {
        return Err("a refused upload left a geometry resident".into());
    }
    println!("  refusals              ok   neither left anything on the device");
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

    // Called through the loop rather than listed as results: an array of `f(&device)` evaluates every
    // case before the loop starts, which prints each FAIL after every other case's line.
    let mut failed = 0;
    for (name, case) in [
        ("round_trip", round_trip as Case),
        ("dedup", dedup),
        ("bind_list", the_bind_list_follows_the_reads),
        ("refusals", refusals),
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
