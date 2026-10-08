// SPDX-License-Identifier: BSD-2-Clause
//! What shape a layer's block buffer may have.
//!
//! # What this is for
//!
//! `blocks::sizing` is the whole of what putting a block buffer on a device decides without one, for
//! the reason `store::layout` is split the same way: the rest of `blocks` needs a GPU and CI has none,
//! so anything decided in there is decided where nothing checks it. `benches/block_buffers.rs` holds
//! the device half.
//!
//! # What would be caught
//!
//! A shape that makes the shadow's slot arithmetic lie. `Consolidated` maps slot *n* to offset
//! `n * block`, so a block of zero bytes maps every slot onto offset zero -- a write for any slot at
//! all would be accepted and land over slot zero's block. An overflowing product is the same failure
//! with a larger constant: `Consolidated::new` multiplies unchecked, so on a thirty-two bit target the
//! shadow comes back smaller than the slot count claims and the slots past the wrap write over
//! earlier ones. Both draw, and neither announces itself.

use tessella_emblema::blocks::{self, Error};

/// An ordinary shape is its own byte count.
#[test]
fn a_shape_is_its_slots_times_its_block() {
    assert_eq!(blocks::sizing(64, 48), Ok(3072));
    assert_eq!(blocks::sizing(1, 1), Ok(1));
}

/// A layer with no slots is refused rather than given a one-byte buffer.
///
/// Which is what it would get: both `Gpu::buffer` and `Gpu::allocate` raise a zero length to one, so
/// a degenerate shape would succeed and hold a buffer that cannot be drawn from.
#[test]
fn no_slots_is_refused() {
    assert_eq!(
        blocks::sizing(0, 48),
        Err(Error::Degenerate {
            slots: 0,
            block: 48
        })
    );
}

/// A block of no bytes is refused, because every slot would share offset zero.
#[test]
fn a_zero_byte_block_is_refused() {
    assert_eq!(
        blocks::sizing(64, 0),
        Err(Error::Degenerate {
            slots: 64,
            block: 0
        })
    );
}

/// A product that will not fit a `usize` is refused rather than wrapped.
///
/// The case the `checked_mul` exists for. Written against `usize::MAX` so it is the same test on a
/// thirty-two and a sixty-four bit target -- the targets this builds for include both, and the one
/// where the wrap is reachable with realistic slot counts is the one without a GPU to catch it.
#[test]
fn an_overflowing_product_is_refused() {
    let slots = usize::MAX / 8 + 1;
    assert_eq!(
        blocks::sizing(slots, 16),
        Err(Error::Overflows { slots, block: 16 }),
        "a product past usize::MAX must not wrap into a plausible size"
    );
}

/// The largest shape that does fit is accepted.
///
/// The other side of the overflow test: a guard that refused everything large would pass that one.
#[test]
fn the_largest_fitting_shape_is_accepted() {
    let slots = usize::MAX / 16;
    assert_eq!(blocks::sizing(slots, 16), Ok((slots * 16) as u64));
}

/// The error says both numbers, in a sentence naming the layer's shape.
///
/// The message is what a consumer sees when the producer and it disagree about a layout, and a
/// disagreement is only diagnosable from both sides of it.
#[test]
fn the_refusal_names_the_shape() {
    let why = blocks::sizing(0, 0).expect_err("refused");
    let text = why.to_string();
    assert!(
        text.contains('0'),
        "the message should carry the shape, got {text:?}"
    );
    assert!(
        !text.is_empty() && text.chars().next().is_some_and(char::is_lowercase),
        "messages read as a clause, got {text:?}"
    );
}
