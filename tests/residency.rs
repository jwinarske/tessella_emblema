//! What the device holds, and the orders in which it must not let go.
//!
//! The cases worth having are the ones where the producer's clock and the device's disagree. A
//! retire says the producer is finished; it says nothing about the frame recorded two frames ago
//! that is still reading the buffer.

use tessella_capture_abi::envelope::GeometryId;
use tessella_emblema::residency::Residency;

fn id(n: u64) -> GeometryId {
    GeometryId(n)
}

/// Announced, uploaded, current.
#[test]
fn an_uploaded_geometry_is_current() {
    let mut held = Residency::new();
    held.announced(id(1));

    assert!(!held.is_current(id(1)), "wanted, not yet there");
    assert_eq!(held.wanted().collect::<Vec<_>>(), [id(1)]);

    held.uploaded(id(1));
    assert!(held.is_current(id(1)));
    assert_eq!(held.wanted().count(), 0);
    assert_eq!(held.resident(), 1);
}

/// A re-announcement makes resident bytes stale.
///
/// The producer announcing again is how it says the bytes changed. Treating it as already present
/// draws the drawable from the last zoom's vertices, which is a picture rather than an error.
#[test]
fn a_re_announcement_makes_it_not_current() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.uploaded(id(1));

    held.announced(id(1));
    assert!(!held.is_current(id(1)), "the device holds the old bytes");
    assert_eq!(held.wanted().collect::<Vec<_>>(), [id(1)]);
    assert_eq!(held.resident(), 1, "and still holds something");

    held.uploaded(id(1));
    assert!(held.is_current(id(1)));
}

/// Retiring frees nothing on its own.
///
/// The whole point. A frame recorded earlier may still be reading the buffer.
#[test]
fn retiring_does_not_free() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.uploaded(id(1));

    held.retired(id(1), 7);
    assert_eq!(held.retiring(), 1);
    assert_eq!(held.resident(), 1, "still held by the device");
    assert!(held.completed(6).is_empty(), "frame 7 has not completed");
}

/// It is freed when the frame it was retired in completes, and not before.
#[test]
fn completion_frees_what_that_frame_retired() {
    let mut held = Residency::new();
    for n in 1..=3 {
        held.announced(id(n));
        held.uploaded(id(n));
    }
    held.retired(id(1), 5);
    held.retired(id(2), 7);
    held.retired(id(3), 9);

    assert_eq!(held.completed(7), [id(1), id(2)], "through seven, not nine");
    assert_eq!(held.retiring(), 1);
    assert_eq!(held.resident(), 1);

    assert_eq!(held.completed(9), [id(3)]);
    assert_eq!(held.resident(), 0);
}

/// Completing the same frame twice frees nothing the second time.
///
/// `completed` drains, so what it returned is no longer tracked. A caller that drops the iterator
/// leaks rather than frees twice, which is the safer way round: a leak is a number going up and a
/// double free is a crash somewhere else.
#[test]
fn completing_twice_does_not_free_twice() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.uploaded(id(1));
    held.retired(id(1), 4);

    assert_eq!(held.completed(4), [id(1)]);
    assert!(held.completed(4).is_empty(), "already handed over");
    assert!(held.completed(99).is_empty());
}

/// Re-announcing something retired cancels the free.
///
/// The producer is using it again. Freeing on the strength of the earlier retire would free the
/// bytes the new announcement just filled -- and the free would land *after* the upload, because
/// the frame it was retired in completes later.
#[test]
fn a_re_announcement_cancels_a_pending_free() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.uploaded(id(1));
    held.retired(id(1), 3);
    assert_eq!(held.retiring(), 1);

    held.announced(id(1));
    assert_eq!(held.retiring(), 0, "no longer going away");

    held.uploaded(id(1));
    assert!(held.completed(100).is_empty(), "and never freed for it");
    assert!(held.is_current(id(1)));
}

/// Retired, re-announced, retired again: freed against the later frame.
///
/// Against the earlier one, the free would land while the second upload was still in flight.
#[test]
fn a_second_retire_moves_the_frame_forward() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.uploaded(id(1));
    held.retired(id(1), 3);
    held.announced(id(1));
    held.uploaded(id(1));
    held.retired(id(1), 11);

    assert!(
        held.completed(3).is_empty(),
        "the first retire is not the one"
    );
    assert_eq!(held.completed(11), [id(1)]);
}

/// Retiring something never uploaded frees nothing and leaves nothing pending.
///
/// The producer can announce and retire between two reads, so the device never saw it. There is no
/// allocation to free and scheduling one would hand the backend an id it never made.
#[test]
fn retiring_what_was_never_uploaded_schedules_nothing() {
    let mut held = Residency::new();
    held.announced(id(1));
    held.retired(id(1), 2);

    assert_eq!(held.retiring(), 0);
    assert_eq!(held.resident(), 0);
    assert_eq!(held.wanted().count(), 0, "and it is no longer wanted");
    assert!(held.completed(2).is_empty());
}

/// Uploading something nobody asked for is ignored.
///
/// Not an error worth refusing -- a backend may complete an upload for something retired while it
/// was in flight -- but it must not become resident, or it is never freed.
#[test]
fn uploading_something_unwanted_does_not_make_it_resident() {
    let mut held = Residency::new();
    held.uploaded(id(1));
    assert_eq!(held.resident(), 0);
    assert!(!held.is_current(id(1)));
}

/// Frames reported out of order still free everything up to the high-water mark.
#[test]
fn completion_is_a_high_water_mark() {
    let mut held = Residency::new();
    for n in 1..=2 {
        held.announced(id(n));
        held.uploaded(id(n));
    }
    held.retired(id(1), 2);
    held.retired(id(2), 8);

    assert_eq!(held.completed(8), [id(1), id(2)], "both, in one call");
}
