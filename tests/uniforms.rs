//! Turning a frame's slot writes into the writes the device actually gets.
//!
//! The question each case asks is how many copies the backend ends up recording, and which bytes
//! they carry. A rewrite of the whole buffer is always correct and is the thing being avoided.

use tessella_emblema::uniforms::{Consolidated, Rejected};

const BLOCK: usize = 64;

fn block(fill: u8) -> Vec<u8> {
    vec![fill; BLOCK]
}

/// One slot is one range, of one block.
#[test]
fn one_slot_is_one_block() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(3, &block(0xAA)).expect("a slot");

    assert!(buffer.is_dirty());
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 192..256);
    assert!(!buffer.is_dirty(), "and the flush takes the debt with it");
    assert_eq!(&buffer.bytes()[192..256], block(0xAA).as_slice());
}

/// Touching slots merge even at a gap of zero, because there is no gap.
#[test]
fn adjacent_slots_become_one_range() {
    let mut buffer = Consolidated::new(8, BLOCK);
    for slot in [2, 3, 4] {
        buffer.write(slot, &block(1)).expect("a slot");
    }
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1, "one copy, not three");
    assert_eq!(ranges[0], 128..320);
}

/// Slots with a clean one between them stay apart at a gap of zero.
#[test]
fn a_clean_slot_between_two_dirty_ones_splits_them() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("a slot");
    buffer.write(3, &block(1)).expect("a slot");

    assert_eq!(buffer.flush(0), [64..128, 192..256]);
}

/// And come together once the gap is allowed.
///
/// The trade the threshold exists for: one copy of 192 bytes against two of 64, where the extra
/// 64 bytes buy one fewer region.
#[test]
fn a_gap_within_the_threshold_merges() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("a slot");
    buffer.write(3, &block(1)).expect("a slot");

    let ranges = buffer.flush(BLOCK);
    assert_eq!(ranges.len(), 1, "one copy spanning the clean slot");
    assert_eq!(ranges[0], 64..256);
}

/// A gap one byte wider than the threshold does not merge.
#[test]
fn a_gap_past_the_threshold_does_not() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(1, &block(1)).expect("a slot");
    buffer.write(3, &block(1)).expect("a slot");

    assert_eq!(buffer.flush(BLOCK - 1), [64..128, 192..256]);
}

/// Writing a slot twice before a flush is one range, carrying the later bytes.
#[test]
fn a_slot_written_twice_is_one_range_of_the_later_bytes() {
    let mut buffer = Consolidated::new(8, BLOCK);
    buffer.write(2, &block(0x11)).expect("a slot");
    buffer.write(2, &block(0x22)).expect("a slot");

    assert_eq!(buffer.dirty_slots(), 1, "one slot, written twice");
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

    buffer.write(0, &block(1)).expect("a slot");
    buffer.flush(0);
    assert_eq!(buffer.flush(0), [], "and nothing is owed twice");
}

/// Ranges come back ascending and disjoint, whatever order the slots were written in.
///
/// A backend hands them to a region list, which has no opinion about order and every opinion about
/// overlap.
#[test]
fn ranges_are_ascending_and_disjoint() {
    let mut buffer = Consolidated::new(16, BLOCK);
    for slot in [9, 2, 14, 3, 7] {
        buffer.write(slot, &block(1)).expect("a slot");
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
/// offset for every slot after it, which draws and draws wrong.
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

/// A slot past the buffer is refused rather than grown into.
///
/// The size comes from the view's own declaration, so a slot past it is a stale order naming a
/// drawable the layer no longer has. Growing would hide that behind an allocation.
#[test]
fn a_slot_past_the_buffer_is_refused() {
    let mut buffer = Consolidated::new(4, BLOCK);

    assert_eq!(
        buffer.write(4, &block(1)),
        Err(Rejected::NoSuchSlot { slots: 4, got: 4 })
    );
    assert_eq!(
        buffer.write(u32::MAX, &block(1)),
        Err(Rejected::NoSuchSlot {
            slots: 4,
            got: u32::MAX
        }),
        "and a far one does not overflow into a valid offset"
    );
    assert!(!buffer.is_dirty());
}

/// Every slot dirty is one range, which is the whole-buffer rewrite -- correctly reached.
#[test]
fn a_fully_dirty_buffer_is_one_range() {
    let mut buffer = Consolidated::new(4, BLOCK);
    for slot in 0..4 {
        buffer.write(slot, &block(1)).expect("a slot");
    }
    let ranges = buffer.flush(0);
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 0..256);
}
