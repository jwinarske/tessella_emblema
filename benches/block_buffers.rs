// SPDX-License-Identifier: BSD-2-Clause
//! A layer's block buffer, against a real device.
//!
//! A bench rather than a test for the reason `geometry_store` is one: it needs a GPU and CI has none.
//! `tests/blocks.rs` holds the half that does not, which is the shape arithmetic, and
//! `tests/uniforms.rs` holds the shadow the flushes come out of.
//!
//! # What this proves that neither of those can
//!
//! That a flushed range lands where the shadow says it does. The shadow's tests prove which ranges a
//! set of dirty slots produces; this writes them to a device, reads the buffer back and compares it
//! slot by slot -- so a range applied at the wrong offset is caught here and nowhere else, and that is
//! a failure that draws rather than one that blanks: a drawable reading another's block is a feature
//! painted in the wrong color, at the wrong width, in the right place.
//!
//! The merge case is the sharper one. A merged range covers clean slots between the dirty ones, so it
//! writes bytes the device already had -- correct only because the source is the shadow and the shadow
//! still holds them. A merge sourced from anywhere else would pass every host-side test and quietly
//! zero a slot nobody wrote this frame.
//!
//! Run with `cargo bench --bench block_buffers`.

mod common;

use tessella_capture_abi::envelope::ViewId;
use tessella_emblema::blocks::{Blocks, Error, Which};

use common::Open;

/// Eight slots of sixteen bytes, which is a small `UboUpdate` block and enough slots to leave gaps in.
const SLOTS: usize = 8;
const BLOCK: usize = 16;

const fn which(layer: i32) -> Which {
    Which {
        view: ViewId(1),
        layer,
    }
}

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// A block whose every byte names the slot it belongs to.
///
/// Distinct per slot and uniform within one, so a range written at the wrong offset shows up as the
/// wrong slot's number rather than as plausible-looking noise.
fn block_for(slot: u32) -> Vec<u8> {
    vec![0xA0 | (slot as u8 & 0x0F); BLOCK]
}

/// Reads the whole buffer back and returns it slot by slot.
fn slots_of(blocks: &Blocks<'_>, at: Which) -> Result<Vec<Vec<u8>>, String> {
    let mut all = vec![0u8; SLOTS * BLOCK];
    let found = blocks
        .read_bytes(at, 0, &mut all)
        .map_err(|why| format!("read back: {why}"))?;
    if !found {
        return Err("the layer has no buffer to read".into());
    }
    Ok(all.chunks(BLOCK).map(<[u8]>::to_vec).collect())
}

/// A buffer nobody has written reads back as zeros.
///
/// The direct form of what `scattered_writes` catches sideways, and the case that found the defect:
/// no flush is involved at all, so this is purely whether `declare` put the shadow's zeros on the
/// device or trusted the allocation to arrive that way. It arrives that way on RADV and V3D and does
/// not on the GC7000UL, and `vkAllocateMemory` promises nothing either way -- so a slot the producer
/// has not filled yet is garbage on some share of parts unless this holds.
fn a_fresh_buffer_is_zeroed(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(8);
    blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    // Deliberately not written to and not flushed.
    let found = slots_of(&blocks, at)?;
    for (slot, bytes) in found.iter().enumerate() {
        if *bytes != [0u8; BLOCK] {
            return Err(format!(
                "slot {slot} of an untouched buffer holds {:#04x}",
                bytes[0]
            ));
        }
    }
    println!("  fresh buffer zeroed   ok   8 untouched slots, all zero");
    Ok(())
}

/// Writes to scattered slots arrive at those slots, and nowhere else.
fn scattered_writes(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(0);
    let made = blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    if !made {
        return Err("the first declare made nothing".into());
    }
    if blocks.declare(device.gpu(), at, SLOTS, BLOCK) != Ok(false) {
        return Err("declaring the same shape twice was not idempotent".into());
    }

    let written = [0u32, 2, 5];
    for slot in written {
        if blocks.write(at, slot, &block_for(slot)) != Ok(true) {
            return Err(format!("slot {slot} was not taken"));
        }
    }
    if !blocks.is_dirty(at) {
        return Err("three writes left the layer clean".into());
    }

    // A gap of zero, so each dirty slot is its own range: three slots, three writes.
    let ranges = blocks.flush(at, 0).map_err(|why| format!("flush: {why}"))?;
    if ranges != 3 {
        return Err(format!("three scattered slots flushed as {ranges} ranges"));
    }
    if blocks.is_dirty(at) {
        return Err("a flush left the layer dirty".into());
    }

    let found = slots_of(&blocks, at)?;
    for slot in 0..SLOTS as u32 {
        let wanted = if written.contains(&slot) {
            block_for(slot)
        } else {
            vec![0u8; BLOCK]
        };
        if found[slot as usize] != wanted {
            return Err(format!(
                "slot {slot} holds {:#04x} against {:#04x}",
                found[slot as usize][0], wanted[0]
            ));
        }
    }
    println!("  scattered writes      ok   3 ranges, slots 0 2 5 placed, 5 others zero");
    Ok(())
}

/// A merged range rewrites the clean slots it spans with what they already held.
///
/// The assertion the host side cannot make. Slot 1 is written and flushed first, then slots 0 and 2
/// are written and flushed with a gap wide enough to merge all three -- so slot 1 is inside a range
/// nothing dirtied. It must come back with its own bytes.
fn a_merge_preserves_what_it_spans(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(1);
    blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    blocks
        .write(at, 1, &block_for(1))
        .map_err(|why| format!("write 1: {why}"))?;
    if blocks
        .flush(at, 0)
        .map_err(|why| format!("flush 1: {why}"))?
        != 1
    {
        return Err("one dirty slot flushed as more than one range".into());
    }

    for slot in [0u32, 2] {
        blocks
            .write(at, slot, &block_for(slot))
            .map_err(|why| format!("write {slot}: {why}"))?;
    }
    // One clean block sits between them, so a gap of BLOCK merges the three into one range.
    let ranges = blocks
        .flush(at, BLOCK)
        .map_err(|why| format!("flush 0 and 2: {why}"))?;
    if ranges != 1 {
        return Err(format!(
            "slots 0 and 2 either side of one clean slot flushed as {ranges} ranges"
        ));
    }

    let found = slots_of(&blocks, at)?;
    for slot in 0..3u32 {
        if found[slot as usize] != block_for(slot) {
            return Err(format!(
                "slot {slot} holds {:#04x} after a merge, against {:#04x}",
                found[slot as usize][0],
                block_for(slot)[0]
            ));
        }
    }
    println!("  merge spans cleanly   ok   1 range over 3 slots, the middle one intact");
    Ok(())
}

/// The last write to a slot is the one that reaches the device.
fn latest_write_wins(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(2);
    blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    blocks
        .write(at, 3, &[0x11; BLOCK])
        .map_err(|why| format!("first write: {why}"))?;
    blocks
        .write(at, 3, &[0x22; BLOCK])
        .map_err(|why| format!("second write: {why}"))?;
    let ranges = blocks.flush(at, 0).map_err(|why| format!("flush: {why}"))?;
    if ranges != 1 {
        return Err(format!("one slot written twice flushed as {ranges} ranges"));
    }

    let found = slots_of(&blocks, at)?;
    if found[3] != [0x22; BLOCK] {
        return Err(format!(
            "slot 3 holds {:#04x}, so the first write reached the device",
            found[3][0]
        ));
    }
    println!("  latest write wins     ok   one range, the second write's bytes");
    Ok(())
}

/// A flush with nothing dirty maps nothing and writes nothing.
fn a_clean_flush_does_nothing(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(3);
    blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    if blocks.flush(at, 0) != Ok(0) {
        return Err("a clean layer flushed something".into());
    }
    // And a layer that was never declared, which a producer sending for an undrawn layer reaches.
    let absent = which(99);
    if blocks.write(absent, 0, &block_for(0)) != Ok(false) {
        return Err("a write to an undeclared layer was taken".into());
    }
    if blocks.flush(absent, 0) != Ok(0) || blocks.buffer(absent).is_some() {
        return Err("an undeclared layer acquired a buffer".into());
    }
    println!("  clean and absent      ok   neither flushed nor allocated");
    Ok(())
}

/// A layer arriving with a different shape is refused, and a forgotten one is gone.
fn refusals_and_forgetting(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(4);
    blocks
        .declare(device.gpu(), at, SLOTS, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    let before = blocks.total_bytes();
    if before < (SLOTS * BLOCK) as u64 {
        return Err(format!(
            "{before} bytes resident for {} declared",
            SLOTS * BLOCK
        ));
    }

    match blocks.declare(device.gpu(), at, SLOTS * 2, BLOCK) {
        Err(Error::Reshaped { slots, block }) if slots == SLOTS && block == BLOCK => {}
        other => return Err(format!("a reshaped layer gave {other:?}")),
    }
    if blocks.total_bytes() != before {
        return Err("a refused reshape changed what is resident".into());
    }

    match blocks.declare(device.gpu(), which(5), 0, BLOCK) {
        Err(Error::Degenerate { .. }) => {}
        other => return Err(format!("a slotless layer gave {other:?}")),
    }

    // A write of the wrong length, which is the producer disagreeing about the block size.
    match blocks.write(at, 0, &[0u8; BLOCK + 1]) {
        Err(Error::Write(_)) => {}
        other => return Err(format!("a wrong-length write gave {other:?}")),
    }
    if blocks.is_dirty(at) {
        return Err("a refused write dirtied the layer".into());
    }

    blocks.forget(at);
    if blocks.buffer(at).is_some() || blocks.layers() != 0 || blocks.total_bytes() != 0 {
        return Err("forgetting left something behind".into());
    }
    println!("  refusals and forget   ok   {before} bytes resident, then none");
    Ok(())
}

/// Two layers of one view do not share a buffer.
///
/// `UboUpdate` is keyed by both, and a key that collapsed to the view would put one layer's blocks
/// over another's -- which draws, because the block sizes would often agree.
fn layers_are_separate(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let (first, second) = (which(6), which(7));
    for at in [first, second] {
        blocks
            .declare(device.gpu(), at, SLOTS, BLOCK)
            .map_err(|why| format!("declare {}: {why}", at.layer))?;
    }
    if blocks.buffer(first) == blocks.buffer(second) {
        return Err("two layers of one view got the same buffer".into());
    }

    blocks
        .write(first, 0, &[0x55; BLOCK])
        .map_err(|why| format!("write: {why}"))?;
    blocks
        .flush(first, 0)
        .map_err(|why| format!("flush: {why}"))?;

    let other = slots_of(&blocks, second)?;
    if other[0] != [0u8; BLOCK] {
        return Err(format!(
            "writing layer 6 put {:#04x} in layer 7's slot 0",
            other[0][0]
        ));
    }
    // And the other way round, so the check is not passing on an ordering accident.
    let mine = slots_of(&blocks, first)?;
    if mine[0] != [0x55; BLOCK] {
        return Err("layer 6's own write did not arrive".into());
    }
    println!("  layers are separate   ok   two buffers, neither in the other");
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

    // Called through the loop rather than listed as results, so a failure prints where it happened.
    // An array of `f(&device)` evaluates every case before the loop starts, which puts each FAIL after
    // every other case's line -- and that reads as a different case having failed.
    let mut failed = 0;
    for (name, case) in [
        ("fresh_is_zeroed", a_fresh_buffer_is_zeroed as Case),
        ("scattered_writes", scattered_writes),
        ("merge", a_merge_preserves_what_it_spans),
        ("latest_write_wins", latest_write_wins),
        ("clean_flush", a_clean_flush_does_nothing),
        ("refusals", refusals_and_forgetting),
        ("layers_are_separate", layers_are_separate),
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
