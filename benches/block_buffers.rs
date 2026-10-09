// SPDX-License-Identifier: BSD-2-Clause
//! A layer's block buffers, against a real device.
//!
//! A bench rather than a test for the reason `geometry_store` is one: it needs a GPU and CI has none.
//! `tests/blocks.rs` holds the half that does not, which is the shape arithmetic, and
//! `tests/uniforms.rs` holds the shadow the flushes come out of.
//!
//! # What this proves that neither of those can
//!
//! That a flushed range lands where the shadow says it does. The shadow's tests prove which ranges a
//! set of dirty entries produces; this writes them to a device, reads the buffer back and compares it
//! entry by entry -- so a range applied at the wrong offset is caught here and nowhere else, and that
//! is a failure that draws rather than one that blanks: a drawable reading another's block is a
//! feature painted in the wrong color, at the wrong width, in the right place.
//!
//! The merge case is the sharper one. A merged range covers clean entries between the dirty ones, so
//! it writes bytes the device already had -- correct only because the source is the shadow and the
//! shadow still holds them. A merge sourced from anywhere else would pass every host-side test and
//! quietly zero an entry nobody wrote this frame.
//!
//! Run with `cargo bench --bench block_buffers`.

mod common;

use tessella_capture_abi::envelope::ViewId;
use tessella_emblema::blocks::{Blocks, Error, Which};

use common::Open;

/// Eight entries of sixteen bytes: a small `UboUpdate` block, and enough entries to leave gaps in.
const ENTRIES: usize = 8;
const BLOCK: usize = 16;

/// The slot these cases use, which is the one every family's drawable array arrives at.
///
/// Through the generated constant rather than as a `2`, so the number here is the producer's.
const SLOT: u32 = tessella_capture_abi::generated::ubo_slots::ID_FILL_DRAWABLE_UBO;

/// A second slot, for the case that two of a layer's buffers are two buffers.
const OTHER_SLOT: u32 = tessella_capture_abi::generated::ubo_slots::ID_FILL_EVALUATED_PROPS_UBO;

const fn which(layer: i32) -> Which {
    Which {
        view: ViewId(1),
        layer,
    }
}

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// A block whose every byte names the entry it belongs to.
///
/// Distinct per entry and uniform within one, so a range written at the wrong offset shows up as the
/// wrong entry's number rather than as plausible-looking noise.
fn block_for(index: u32) -> Vec<u8> {
    vec![0xA0 | (index as u8 & 0x0F); BLOCK]
}

/// Reads one buffer back and returns it entry by entry.
fn entries_of(blocks: &Blocks<'_>, at: Which, slot: u32) -> Result<Vec<Vec<u8>>, String> {
    let mut all = vec![0u8; ENTRIES * BLOCK];
    let found = blocks
        .read_bytes(at, slot, 0, &mut all)
        .map_err(|why| format!("read back: {why}"))?;
    if !found {
        return Err("the slot has no buffer to read".into());
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
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    // Deliberately not written to and not flushed.
    let found = entries_of(&blocks, at, SLOT)?;
    for (index, bytes) in found.iter().enumerate() {
        if *bytes != [0u8; BLOCK] {
            return Err(format!(
                "entry {index} of an untouched buffer holds {:#04x}",
                bytes[0]
            ));
        }
    }
    println!("  fresh buffer zeroed   ok   8 untouched entries, all zero");
    Ok(())
}

/// Writes to scattered entries arrive at those entries, and nowhere else.
fn scattered_writes(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(0);
    let made = blocks
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    if !made {
        return Err("the first declare made nothing".into());
    }
    if blocks.declare(device.gpu(), at, SLOT, ENTRIES, BLOCK) != Ok(false) {
        return Err("declaring the same shape twice was not idempotent".into());
    }

    let written = [0u32, 2, 5];
    for index in written {
        if blocks.write(at, SLOT, index, &block_for(index)) != Ok(true) {
            return Err(format!("entry {index} was not taken"));
        }
    }
    if !blocks.is_dirty(at, SLOT) {
        return Err("three writes left the buffer clean".into());
    }

    // A gap of zero, so each dirty entry is its own range: three entries, three writes.
    let ranges = blocks
        .flush(at, SLOT, 0)
        .map_err(|why| format!("flush: {why}"))?;
    if ranges != 3 {
        return Err(format!(
            "three scattered entries flushed as {ranges} ranges"
        ));
    }
    if blocks.is_dirty(at, SLOT) {
        return Err("a flush left the buffer dirty".into());
    }

    let found = entries_of(&blocks, at, SLOT)?;
    for index in 0..ENTRIES as u32 {
        let wanted = if written.contains(&index) {
            block_for(index)
        } else {
            vec![0u8; BLOCK]
        };
        if found[index as usize] != wanted {
            return Err(format!(
                "entry {index} holds {:#04x} against {:#04x}",
                found[index as usize][0], wanted[0]
            ));
        }
    }
    println!("  scattered writes      ok   3 ranges, entries 0 2 5 placed, 5 others zero");
    Ok(())
}

/// A merged range rewrites the clean entries it spans with what they already held.
///
/// The assertion the host side cannot make. Entry 1 is written and flushed first, then entries 0 and
/// 2 are written and flushed with a gap wide enough to merge all three -- so entry 1 is inside a
/// range nothing dirtied. It must come back with its own bytes.
fn a_merge_preserves_what_it_spans(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(1);
    blocks
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    blocks
        .write(at, SLOT, 1, &block_for(1))
        .map_err(|why| format!("write 1: {why}"))?;
    if blocks
        .flush(at, SLOT, 0)
        .map_err(|why| format!("flush 1: {why}"))?
        != 1
    {
        return Err("one dirty entry flushed as more than one range".into());
    }

    for index in [0u32, 2] {
        blocks
            .write(at, SLOT, index, &block_for(index))
            .map_err(|why| format!("write {index}: {why}"))?;
    }
    // One clean block sits between them, so a gap of BLOCK merges the three into one range.
    let ranges = blocks
        .flush(at, SLOT, BLOCK)
        .map_err(|why| format!("flush 0 and 2: {why}"))?;
    if ranges != 1 {
        return Err(format!(
            "entries 0 and 2 either side of one clean entry flushed as {ranges} ranges"
        ));
    }

    let found = entries_of(&blocks, at, SLOT)?;
    for index in 0..3u32 {
        if found[index as usize] != block_for(index) {
            return Err(format!(
                "entry {index} holds {:#04x} after a merge, against {:#04x}",
                found[index as usize][0],
                block_for(index)[0]
            ));
        }
    }
    println!("  merge spans cleanly   ok   1 range over 3 entries, the middle one intact");
    Ok(())
}

/// The last write to an entry is the one that reaches the device.
fn latest_write_wins(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(2);
    blocks
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    blocks
        .write(at, SLOT, 3, &[0x11; BLOCK])
        .map_err(|why| format!("first write: {why}"))?;
    blocks
        .write(at, SLOT, 3, &[0x22; BLOCK])
        .map_err(|why| format!("second write: {why}"))?;
    let ranges = blocks
        .flush(at, SLOT, 0)
        .map_err(|why| format!("flush: {why}"))?;
    if ranges != 1 {
        return Err(format!(
            "one entry written twice flushed as {ranges} ranges"
        ));
    }

    let found = entries_of(&blocks, at, SLOT)?;
    if found[3] != [0x22; BLOCK] {
        return Err(format!(
            "entry 3 holds {:#04x}, so the first write reached the device",
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
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    if blocks.flush(at, SLOT, 0) != Ok(0) {
        return Err("a clean buffer flushed something".into());
    }
    // And a layer that was never declared, which a producer sending for an undrawn layer reaches.
    let absent = which(99);
    if blocks.write(absent, SLOT, 0, &block_for(0)) != Ok(false) {
        return Err("a write to an undeclared layer was taken".into());
    }
    if blocks.flush(absent, SLOT, 0) != Ok(0) || blocks.buffer(absent, SLOT).is_some() {
        return Err("an undeclared layer acquired a buffer".into());
    }
    println!("  clean and absent      ok   neither flushed nor allocated");
    Ok(())
}

/// A slot arriving with a different shape is refused, and a forgotten layer is gone.
fn refusals_and_forgetting(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(4);
    blocks
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;
    let before = blocks.total_bytes();
    if before < (ENTRIES * BLOCK) as u64 {
        return Err(format!(
            "{before} bytes resident for {} declared",
            ENTRIES * BLOCK
        ));
    }

    match blocks.declare(device.gpu(), at, SLOT, ENTRIES * 2, BLOCK) {
        Err(Error::Reshaped { entries, block }) if entries == ENTRIES && block == BLOCK => {}
        other => return Err(format!("a reshaped slot gave {other:?}")),
    }
    if blocks.total_bytes() != before {
        return Err("a refused reshape changed what is resident".into());
    }

    match blocks.declare(device.gpu(), which(5), SLOT, 0, BLOCK) {
        Err(Error::Degenerate { .. }) => {}
        other => return Err(format!("a buffer of no entries gave {other:?}")),
    }

    // A write of the wrong length, which is the producer disagreeing about the block size.
    match blocks.write(at, SLOT, 0, &[0u8; BLOCK + 1]) {
        Err(Error::Write(_)) => {}
        other => return Err(format!("a wrong-length write gave {other:?}")),
    }
    if blocks.is_dirty(at, SLOT) {
        return Err("a refused write dirtied the buffer".into());
    }

    blocks.forget(at);
    if blocks.buffer(at, SLOT).is_some() || blocks.layers() != 0 || blocks.total_bytes() != 0 {
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
            .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
            .map_err(|why| format!("declare {}: {why}", at.layer))?;
    }
    if blocks.buffer(first, SLOT) == blocks.buffer(second, SLOT) {
        return Err("two layers of one view got the same buffer".into());
    }

    blocks
        .write(first, SLOT, 0, &[0x55; BLOCK])
        .map_err(|why| format!("write: {why}"))?;
    blocks
        .flush(first, SLOT, 0)
        .map_err(|why| format!("flush: {why}"))?;

    let other = entries_of(&blocks, second, SLOT)?;
    if other[0] != [0u8; BLOCK] {
        return Err(format!(
            "writing layer 6 put {:#04x} in layer 7's entry 0",
            other[0][0]
        ));
    }
    // And the other way round, so the check is not passing on an ordering accident.
    let mine = entries_of(&blocks, first, SLOT)?;
    if mine[0] != [0x55; BLOCK] {
        return Err("layer 6's own write did not arrive".into());
    }
    println!("  layers are separate   ok   two buffers, neither in the other");
    Ok(())
}

/// Two slots of one layer do not share a buffer.
///
/// The case for the defect this keying replaced. A layer's blocks arrive in several buffers --
/// `UboUpdate::slot` is which buffer, not which entry within one -- and keyed by the layer alone
/// this held one, so a family's second binding was pointed at its first block's bytes. Nothing
/// failed: the set was complete and the pipeline valid, and a shader read a color out of a matrix.
///
/// Written at both slots and read back from both, because a key that ignored the slot would pass a
/// check that only wrote one of them.
fn slots_are_separate(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(10);
    for slot in [SLOT, OTHER_SLOT] {
        blocks
            .declare(device.gpu(), at, slot, ENTRIES, BLOCK)
            .map_err(|why| format!("declare {slot}: {why}"))?;
    }
    if blocks.buffer(at, SLOT) == blocks.buffer(at, OTHER_SLOT) {
        return Err("two slots of one layer got the same buffer".into());
    }
    if blocks.layers() != 1 || blocks.buffers() != 2 {
        return Err(format!(
            "{} layers and {} buffers for one layer of two slots",
            blocks.layers(),
            blocks.buffers()
        ));
    }

    blocks
        .write(at, SLOT, 0, &[0x11; BLOCK])
        .map_err(|why| format!("write {SLOT}: {why}"))?;
    blocks
        .write(at, OTHER_SLOT, 0, &[0x22; BLOCK])
        .map_err(|why| format!("write {OTHER_SLOT}: {why}"))?;
    for slot in [SLOT, OTHER_SLOT] {
        blocks
            .flush(at, slot, 0)
            .map_err(|why| format!("flush {slot}: {why}"))?;
    }

    let drawables = entries_of(&blocks, at, SLOT)?;
    let props = entries_of(&blocks, at, OTHER_SLOT)?;
    if drawables[0] != [0x11; BLOCK] || props[0] != [0x22; BLOCK] {
        return Err(format!(
            "slot {SLOT} holds {:#04x} and slot {OTHER_SLOT} holds {:#04x}",
            drawables[0][0], props[0][0]
        ));
    }

    // And forgetting is per layer, so both go.
    blocks.forget(at);
    if blocks.buffers() != 0 {
        return Err("forgetting a layer left one of its slots behind".into());
    }
    println!("  slots are separate    ok   two buffers in one layer, neither in the other");
    Ok(())
}

/// A whole buffer arrives, reaches the device, and a second arrival flushes only what moved.
///
/// The shape an `UboUpdate` has: `Upload::Uniforms` carries a layer's buffer entire, and
/// `Blocks::write` takes one entry -- so this is the path that could not be walked at all before
/// `replace`. What the host side cannot say is that the comparison's dirty set lands at the right
/// offsets on the device, which is the same thing `scattered_writes` says about per-entry writes.
///
/// The second arrival differs in one entry out of eight, so a `replace` that marked everything
/// would flush one range covering the whole buffer rather than one covering an entry -- both
/// correct on the device, which is why the range count is asserted beside the bytes.
fn a_whole_buffer_arrives(device: &Open) -> Result<(), String> {
    let mut blocks = Blocks::new();
    let at = which(11);
    blocks
        .declare(device.gpu(), at, SLOT, ENTRIES, BLOCK)
        .map_err(|why| format!("declare: {why}"))?;

    // The producer's first send: every entry named after itself.
    let first: Vec<u8> = (0..ENTRIES as u32).flat_map(block_for).collect();
    let changed = blocks
        .replace(at, SLOT, &first)
        .map_err(|why| format!("first arrival: {why}"))?;
    if changed != ENTRIES {
        return Err(format!("{changed} of {ENTRIES} entries taken as new"));
    }
    let ranges = blocks
        .flush(at, SLOT, 0)
        .map_err(|why| format!("flush: {why}"))?;
    if ranges != 1 {
        return Err(format!("a wholly new buffer flushed as {ranges} ranges"));
    }
    let found = entries_of(&blocks, at, SLOT)?;
    for index in 0..ENTRIES as u32 {
        if found[index as usize] != block_for(index) {
            return Err(format!(
                "entry {index} holds {:#04x} after the first arrival",
                found[index as usize][0]
            ));
        }
    }

    // The same bytes again, which is a parked frame: nothing moves.
    if blocks.replace(at, SLOT, &first) != Ok(0) || blocks.is_dirty(at, SLOT) {
        return Err("an identical buffer dirtied something".into());
    }
    if blocks.flush(at, SLOT, 0) != Ok(0) {
        return Err("an identical buffer flushed something".into());
    }

    // One entry moves, and one range carries it.
    let mut second = first.clone();
    second[5 * BLOCK..6 * BLOCK].fill(0xC5);
    if blocks.replace(at, SLOT, &second) != Ok(1) {
        return Err("one changed entry was not the only one marked".into());
    }
    if blocks.flush(at, SLOT, 0) != Ok(1) {
        return Err("one changed entry flushed as more than one range".into());
    }
    let found = entries_of(&blocks, at, SLOT)?;
    if found[5] != [0xC5; BLOCK] {
        return Err(format!(
            "entry 5 holds {:#04x} after the second arrival",
            found[5][0]
        ));
    }
    for index in (0..ENTRIES as u32).filter(|index| *index != 5) {
        if found[index as usize] != block_for(index) {
            return Err(format!(
                "entry {index} moved when only entry 5 changed, and holds {:#04x}",
                found[index as usize][0]
            ));
        }
    }
    println!("  a whole buffer        ok   8 entries in, then 1 range for 1 changed entry");
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
        ("slots_are_separate", slots_are_separate),
        ("whole_buffer", a_whole_buffer_arrives),
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
