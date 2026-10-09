//! Turning a frame's entry writes into the writes the device actually gets.
//!
//! The question each case asks is how many copies the backend ends up recording, and which bytes
//! they carry. A rewrite of the whole buffer is always correct and is the thing being avoided.

use tessella_emblema::uniforms::{Consolidated, Rejected};

const BLOCK: usize = 64;

fn block(fill: u8) -> Vec<u8> {
    vec![fill; BLOCK]
}

/// One entry is one range, of one block.
#[test]
fn one_entry_is_one_block() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(3, &block(0xAA)).expect("an entry");

    assert!(buffer.is_dirty());
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 192..256);
    assert!(!buffer.is_dirty(), "and the flush takes the debt with it");
    assert_eq!(&buffer.bytes()[192..256], block(0xAA).as_slice());
}

/// Touching entries merge even at a gap of zero, because there is no gap.
#[test]
fn adjacent_entries_become_one_range() {
    let mut buffer = Consolidated::new(8, BLOCK);
    for index in [2, 3, 4] {
        buffer.write(index, &block(1)).expect("an entry");
    }
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1, "one copy, not three");
    assert_eq!(ranges[0], 128..320);
}

/// Slots with a clean one between them stay apart at a gap of zero.
#[test]
fn a_clean_entry_between_two_dirty_ones_splits_them() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("an entry");
    buffer.write(3, &block(1)).expect("an entry");

    assert_eq!(buffer.flush(0), [64..128, 192..256]);
}

/// And come together once the gap is allowed.
///
/// The trade the threshold exists for: one copy of 192 bytes against two of 64, where the extra
/// 64 bytes buy one fewer region.
#[test]
fn a_gap_within_the_threshold_merges() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("an entry");
    buffer.write(3, &block(1)).expect("an entry");

    let ranges = buffer.flush(BLOCK);
    assert_eq!(ranges.len(), 1, "one copy spanning the clean entry");
    assert_eq!(ranges[0], 64..256);
}

/// A gap one byte wider than the threshold does not merge.
#[test]
fn a_gap_past_the_threshold_does_not() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("an entry");
    buffer.write(3, &block(1)).expect("an entry");

    assert_eq!(buffer.flush(BLOCK - 1), [64..128, 192..256]);
}

/// Writing an entry twice before a flush is one range, carrying the later bytes.
#[test]
fn an_entry_written_twice_is_one_range_of_the_later_bytes() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(2, &block(0x11)).expect("an entry");
    buffer.write(2, &block(0x22)).expect("an entry");

    assert_eq!(buffer.dirty_entries(), 1, "one entry, written twice");
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 128..192);
    assert_eq!(&buffer.bytes()[128..192], block(0x22).as_slice());
}

/// A flush with nothing dirty is no writes at all.
///
/// The parked case. A frame that changed nothing should cost the device nothing, and a backend
/// looping over an empty list records no regions.
#[test]
fn a_clean_buffer_flushes_nothing() {
    let mut buffer = Consolidated::new(8, BLOCK);
    assert!(!buffer.is_dirty());
    assert_eq!(buffer.flush(0), []);

    buffer.write(0, &block(1)).expect("an entry");
    buffer.flush(0);
    assert_eq!(buffer.flush(0), [], "and nothing is owed twice");
}

/// Ranges come back ascending and disjoint, whatever order the entries were written in.
///
/// A backend hands them to a region list, which has no opinion about order and every opinion about
/// overlap.
#[test]
fn ranges_are_ascending_and_disjoint() {
    let mut buffer = Consolidated::new(16, BLOCK);
    for index in [9, 2, 14, 3, 7] {
        buffer.write(index, &block(1)).expect("an entry");
    }
    let ranges = buffer.flush(0);

    assert_eq!(ranges, [128..256, 448..512, 576..640, 896..960]);
    for pair in ranges.windows(2) {
        assert!(pair[0].end <= pair[1].start, "disjoint and ordered");
    }
}

/// A write of the wrong length is refused, and changes nothing.
///
/// A layer's blocks are one size. Writing another length would put the right bytes at the wrong
/// offset for every entry after it, which draws and draws wrong.
#[test]
fn a_write_of_the_wrong_length_is_refused() {
    let mut buffer = Consolidated::new(4, BLOCK);
    let before = buffer.bytes().to_vec();

    assert_eq!(
        buffer.write(0, &[1; BLOCK + 8]),
        Err(Rejected::WrongLength {
            expected: BLOCK,
            got: BLOCK + 8
        })
    );
    assert_eq!(buffer.bytes(), before.as_slice(), "nothing written");
    assert!(!buffer.is_dirty(), "and nothing owed");
}

/// An index past the buffer is refused rather than grown into.
///
/// The size comes from the view's own declaration, so an index past it is a stale order naming a
/// drawable the layer no longer has. Growing would hide that behind an allocation.
#[test]
fn an_index_past_the_buffer_is_refused() {
    let mut buffer = Consolidated::new(4, BLOCK);

    assert_eq!(
        buffer.write(4, &block(1)),
        Err(Rejected::NoSuchIndex { entries: 4, got: 4 })
    );
    assert_eq!(
        buffer.write(u32::MAX, &block(1)),
        Err(Rejected::NoSuchIndex {
            entries: 4,
            got: u32::MAX
        }),
        "and a far one does not overflow into a valid offset"
    );
    assert!(!buffer.is_dirty());
}

/// Every entry dirty is one range, which is the whole-buffer rewrite -- correctly reached.
#[test]
fn a_fully_dirty_buffer_is_one_range() {
    let mut buffer = Consolidated::new(4, BLOCK);
    for index in 0..4 {
        buffer.write(index, &block(1)).expect("an entry");
    }
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 0..256);
}

/// A whole buffer arriving marks only the entries that changed.
///
/// What an `UboUpdate` delivers, and the reason `replace` exists: the producer sends a layer's
/// buffer entire and `write` takes one entry, so before this there was no call that could apply
/// what arrived.
#[test]
fn a_whole_buffer_marks_only_what_changed() {
    let mut buffer = Consolidated::new(4, BLOCK);
    // The first arrival differs from the zeros everywhere, so every entry is dirty.
    let first: Vec<u8> = (0..4u8).flat_map(|at| vec![0x10 | at; BLOCK]).collect();
    assert_eq!(buffer.replace(&first), Ok(4), "a fresh buffer is all new");
    assert_eq!(buffer.dirty_entries(), 4);
    buffer.flush(0);

    // The same bytes again: nothing changed, so nothing is marked and a flush moves nothing.
    assert_eq!(
        buffer.replace(&first),
        Ok(0),
        "an identical buffer is clean"
    );
    assert!(
        !buffer.is_dirty(),
        "a still map re-sending its buffer must flush nothing"
    );
    assert_eq!(buffer.flush(0), [], "and nothing is owed");

    // One entry different, and only that one is marked.
    let mut second = first.clone();
    second[2 * BLOCK] = 0xFF;
    assert_eq!(buffer.replace(&second), Ok(1));
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1, "one entry changed, so one range");
    assert_eq!(
        ranges[0],
        2 * BLOCK..3 * BLOCK,
        "the range covers entry two and nothing else"
    );
    assert_eq!(buffer.bytes()[2 * BLOCK], 0xFF);
}

/// A buffer of another size is refused rather than partly applied.
///
/// The record describes the whole buffer, so a length that disagrees is the producer and this
/// consumer disagreeing about how many entries the layer has -- a *different buffer*, which is
/// `blocks::declare`'s `Reshaped` and not damage to this one. Applying the overlap would leave the
/// rest holding the last frame's drawables.
#[test]
fn a_buffer_of_another_size_is_refused() {
    let mut buffer = Consolidated::new(4, BLOCK);
    let filled = vec![0xAB; 4 * BLOCK];
    buffer.replace(&filled).expect("the whole buffer");
    buffer.flush(0);

    for wrong in [3 * BLOCK, 5 * BLOCK, 4 * BLOCK - 1] {
        assert_eq!(
            buffer.replace(&vec![0xCD; wrong]),
            Err(Rejected::NotTheBuffer {
                expected: 4 * BLOCK,
                got: wrong
            })
        );
    }
    assert!(
        !buffer.is_dirty(),
        "a refused arrival must not dirty anything"
    );
    assert!(
        buffer.bytes().iter().all(|byte| *byte == 0xAB),
        "and must not write anything"
    );
}

/// Entries that change on either side of a clean one are two ranges, or one with a gap allowed.
///
/// `replace` feeds the same dirty set `write` does, so the merge rule is the one the rest of this
/// module already tests -- this is the case that says a whole-buffer arrival reaches it.
#[test]
fn a_replaced_buffer_merges_like_any_other() {
    let mut buffer = Consolidated::new(4, BLOCK);
    let zeros = vec![0u8; 4 * BLOCK];
    buffer.replace(&zeros).expect("the whole buffer");
    buffer.flush(0);

    let mut arrived = zeros.clone();
    arrived[0] = 1;
    arrived[2 * BLOCK] = 1;
    assert_eq!(buffer.replace(&arrived), Ok(2));
    assert_eq!(
        buffer.flush(0).len(),
        2,
        "a clean entry between them splits"
    );

    let mut again = arrived.clone();
    again[1] = 2;
    again[2 * BLOCK + 1] = 2;
    assert_eq!(buffer.replace(&again), Ok(2));
    assert_eq!(
        buffer.flush(BLOCK).len(),
        1,
        "a gap of one entry merges them"
    );
}
