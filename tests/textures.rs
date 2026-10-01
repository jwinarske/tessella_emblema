//! What the device holds for each texture, and which parts of it are owed.
//!
//! Two questions: how many regions a backend ends up uploading, and whether an allocation is ever
//! freed while something might still be sampling it.

use tessella_capture_abi::envelope::{Extent, Rect16, TextureId};
use tessella_emblema::textures::{Damage, Needs, Textures};

const CAP: usize = 4;
const RGBA: u8 = 0;

fn rect(x: u16, y: u16, w: u16, h: u16) -> Rect16 {
    Rect16 { x, y, w, h }
}

fn size(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

fn id(n: u64) -> TextureId {
    TextureId(n)
}

/// Rects that do not meet stay apart.
#[test]
fn separate_regions_stay_separate() {
    let mut damage = Damage::new();
    damage.add(rect(0, 0, 4, 4), CAP);
    damage.add(rect(32, 32, 4, 4), CAP);

    assert_eq!(damage.rects().len(), 2);
}

/// Overlapping rects become one covering both.
#[test]
fn overlapping_regions_merge() {
    let mut damage = Damage::new();
    damage.add(rect(0, 0, 8, 8), CAP);
    damage.add(rect(4, 4, 8, 8), CAP);

    assert_eq!(damage.rects(), [rect(0, 0, 12, 12)]);
}

/// So do touching ones: one region covers both no worse than two do.
#[test]
fn touching_regions_merge() {
    let mut damage = Damage::new();
    damage.add(rect(0, 0, 8, 8), CAP);
    damage.add(rect(8, 0, 8, 8), CAP);

    assert_eq!(damage.rects(), [rect(0, 0, 16, 8)]);
}

/// A rect bridging two others collapses all three.
///
/// The reason adding settles rather than doing one pass: unioning can make the result meet
/// something it did not meet before, and stopping after one pass leaves two regions that overlap.
#[test]
fn a_bridging_region_collapses_both_sides() {
    let mut damage = Damage::new();
    damage.add(rect(0, 0, 4, 4), CAP);
    damage.add(rect(16, 0, 4, 4), CAP);
    assert_eq!(damage.rects().len(), 2, "apart to begin with");

    damage.add(rect(4, 0, 12, 4), CAP);
    assert_eq!(
        damage.rects(),
        [rect(0, 0, 20, 4)],
        "one region, not two overlapping ones"
    );
}

/// Past the cap everything becomes one union.
///
/// Deliberately the whole-texture write this otherwise avoids: beyond a handful of regions the
/// per-region overhead is the larger cost, and §6.4 caps the producer's list for the same reason.
#[test]
fn past_the_cap_everything_unions() {
    let mut damage = Damage::new();
    for n in 0..5u16 {
        damage.add(rect(n * 16, 0, 4, 4), 4);
    }
    assert_eq!(
        damage.rects(),
        [rect(0, 0, 68, 4)],
        "one region covering all five"
    );
}

/// A zero-area region is dropped rather than given a region of its own.
#[test]
fn an_empty_region_is_dropped() {
    let mut damage = Damage::new();
    damage.add(rect(4, 4, 0, 8), CAP);
    damage.add(rect(4, 4, 8, 0), CAP);
    assert_eq!(damage.rects(), [], "nothing changed, so nothing is owed");
}

/// A texture nobody has seen has to be created.
#[test]
fn a_new_texture_is_created() {
    let mut held = Textures::new();
    let needs = held.updated(id(1), size(64, 64), RGBA, &[rect(0, 0, 8, 8)], CAP, 1);

    assert_eq!(needs, Needs::Create);
    assert_eq!(held.held(), 1);
    assert_eq!(held.damage(id(1)), [rect(0, 0, 8, 8)]);
}

/// A second update at the same size and format is damage, not a new image.
#[test]
fn the_same_texture_again_is_an_upload() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[rect(0, 0, 8, 8)], CAP, 1);
    let needs = held.updated(id(1), size(64, 64), RGBA, &[rect(32, 32, 8, 8)], CAP, 2);

    assert_eq!(needs, Needs::Upload);
    assert_eq!(held.damage(id(1)).len(), 2, "both regions owed");
    assert_eq!(held.retiring(), 0, "and nothing is going away");
}

/// A different size is a different image.
///
/// Writing the new pixels into the old allocation either overruns it or leaves a stale border, so
/// this is reported rather than treated as damage.
#[test]
fn a_different_size_has_to_be_remade() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[], CAP, 1);
    let needs = held.updated(id(1), size(128, 128), RGBA, &[], CAP, 5);

    assert_eq!(needs, Needs::Recreate);
    assert_eq!(
        held.retiring(),
        1,
        "the old allocation is scheduled, not dropped"
    );
    assert_eq!(
        held.damage(id(1)),
        [rect(0, 0, 128, 128)],
        "and the new one is wholly owed"
    );
}

/// So is a different format at the same size.
#[test]
fn a_different_format_has_to_be_remade() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[], CAP, 1);
    assert_eq!(
        held.updated(id(1), size(64, 64), 1, &[], CAP, 2),
        Needs::Recreate
    );
}

/// Remaking drops the old damage: it described pixels in an image that no longer exists.
#[test]
fn remaking_forgets_the_old_damage() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[rect(0, 0, 4, 4)], CAP, 1);
    held.updated(id(1), size(32, 32), RGBA, &[rect(8, 8, 4, 4)], CAP, 2);

    assert_eq!(
        held.damage(id(1)),
        [rect(8, 8, 4, 4)],
        "only what the new image was told about"
    );
}

/// An empty rect list is a whole-texture write.
#[test]
fn no_rects_means_the_whole_texture() {
    let mut held = Textures::new();
    held.updated(id(1), size(256, 128), RGBA, &[], CAP, 1);

    assert_eq!(held.damage(id(1)), [rect(0, 0, 256, 128)]);
}

/// Uploading takes the damage and leaves the texture held.
#[test]
fn uploading_clears_the_damage_and_keeps_the_texture() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[rect(0, 0, 8, 8)], CAP, 1);

    assert_eq!(held.uploaded(id(1)), [rect(0, 0, 8, 8)]);
    assert_eq!(held.damage(id(1)), [], "nothing owed twice");
    assert_eq!(held.held(), 1, "and the texture is still there");
}

/// Retiring does not free, and completion does.
#[test]
fn retiring_waits_for_the_frame() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[], CAP, 1);
    held.retired(id(1), 6);

    assert_eq!(held.held(), 0, "the producer is finished with it");
    assert_eq!(held.retiring(), 1, "the device is not");
    assert_eq!(held.completed(5), [], "frame 6 has not completed");
    assert_eq!(held.completed(6), [id(1)]);
    assert_eq!(held.retiring(), 0);
}

/// A remade texture's old allocation is freed on the frame it was replaced in.
#[test]
fn a_replaced_allocation_is_freed_on_its_frame() {
    let mut held = Textures::new();
    held.updated(id(1), size(64, 64), RGBA, &[], CAP, 1);
    held.updated(id(1), size(128, 128), RGBA, &[], CAP, 9);

    assert_eq!(held.completed(8), [], "still being sampled, possibly");
    assert_eq!(held.completed(9), [id(1)]);
    assert_eq!(held.held(), 1, "and the new one is still held");
}

/// Retiring something never seen schedules nothing.
#[test]
fn retiring_an_unknown_texture_schedules_nothing() {
    let mut held = Textures::new();
    held.retired(id(9), 1);
    assert_eq!(held.retiring(), 0);
    assert_eq!(held.completed(99), []);
}

/// Damage on a texture nobody has seen is empty rather than a panic.
#[test]
fn damage_on_an_unknown_texture_is_empty() {
    let mut held = Textures::new();
    assert_eq!(held.damage(id(4)), []);
    assert_eq!(held.uploaded(id(4)), []);
}

/// Whatever is added, no two stored regions ever meet.
///
/// The invariant the merging exists to keep, checked over many shapes rather than the few a
/// hand-written case reaches. It is also the test that says whether settling is needed: a single
/// merge pass can leave two regions overlapping if unioning made the result reach one already
/// passed over, and that shows up here as two stored rects that meet.
#[test]
fn stored_regions_never_meet() {
    // A small deterministic generator: the sequence matters less than that it is the same one
    // every run, so a failure is reproducible without a seed to carry around.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for _ in 0..2000 {
        let mut damage = Damage::new();
        for _ in 0..8 {
            let r = next();
            damage.add(
                rect(
                    (r & 0x3F) as u16,
                    ((r >> 8) & 0x3F) as u16,
                    ((r >> 16) & 0x0F) as u16 + 1,
                    ((r >> 24) & 0x0F) as u16 + 1,
                ),
                64,
            );
        }
        let rects = damage.rects();
        for (at, a) in rects.iter().enumerate() {
            for b in &rects[at + 1..] {
                let (ax0, ay0) = (u32::from(a.x), u32::from(a.y));
                let (ax1, ay1) = (ax0 + u32::from(a.w), ay0 + u32::from(a.h));
                let (bx0, by0) = (u32::from(b.x), u32::from(b.y));
                let (bx1, by1) = (bx0 + u32::from(b.w), by0 + u32::from(b.h));
                let meets = ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1;
                assert!(!meets, "{a:?} and {b:?} meet, so one pass was not enough");
            }
        }
    }
}
