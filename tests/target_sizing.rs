// SPDX-License-Identifier: BSD-2-Clause
//! Which of a frame's state is per size and which is per image.
//!
//! # What this is for
//!
//! #60's requirement is one sentence: the pass
//!
//! > must not cache per-image state that breaks when the image changes every frame
//!
//! A host rotating a ring of three hands over a different image each frame, all the same size. So
//! anything keyed by *size* is shared by the whole ring and costs nothing, and anything keyed by
//! *image* is remade three times as often as it should be -- or worse, used with the wrong image.
//!
//! `Depth::serves` is the whole of that decision, and it is the one part of the hand-off that can be
//! checked without a GPU. `benches/hand_off.rs` proves the rest.
//!
//! # What would be caught
//!
//! A `serves` that compared anything per-image, which would answer `false` for every image in the
//! ring and remake the depth attachment every frame -- a correct picture at three times the
//! allocation traffic, which is exactly the kind of cost that never gets noticed. And a `serves`
//! that ignored the format, which would reuse a `D24_UNORM_S8_UINT` attachment on a device that
//! wanted `D32_SFLOAT_S8_UINT` -- the two parts on the bench disagree about that, so it is not
//! hypothetical.

use ash::vk;
use tessella_emblema::target;

/// The layout the pass states is `GENERAL`, and it is a constant the host can read.
///
/// The host has to agree with it *before* the frame is recorded -- it passes the same value to
/// emblema's `assume_layout` so its own wrapper tracks the image correctly. A function returning it
/// after the fact would be too late to be useful.
#[test]
fn the_stated_layout_is_a_constant() {
    assert_eq!(target::LEAVES_IN, vk::ImageLayout::GENERAL);
}

/// `GENERAL` rather than a layout only Vulkan can name.
///
/// The image is exported as a dma-buf and read by a compositor. There is no way to tell an importer
/// that an image is in `COLOR_ATTACHMENT_OPTIMAL`, and no negotiation across that boundary, so
/// `GENERAL` is the one layout both sides can mean the same thing by.
#[test]
fn the_stated_layout_is_not_a_vulkan_only_layout() {
    for only_vulkan in [
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::UNDEFINED,
    ] {
        assert_ne!(
            target::LEAVES_IN,
            only_vulkan,
            "{only_vulkan:?} cannot be communicated to a dma-buf importer"
        );
    }
}

/// An attachment serves every image of its own size and format.
///
/// Which is every image in the ring: they differ from each other and share both of these. A `serves`
/// that compared anything per-image would answer `false` here and have the attachment remade every
/// frame -- a correct picture at three times the allocation traffic.
#[test]
fn one_shape_serves_a_whole_ring() {
    let shape = target::Shape {
        width: 1920,
        height: 1080,
        format: vk::Format::D24_UNORM_S8_UINT,
    };
    assert!(shape.serves(shape), "the same shape must serve itself");
    // A second image of the ring is described by an identical shape, because a shape holds nothing
    // that distinguishes one image from another.
    let next = target::Shape {
        width: 1920,
        height: 1080,
        format: vk::Format::D24_UNORM_S8_UINT,
    };
    assert!(shape.serves(next));
}

/// A different format is a different attachment.
///
/// Not hypothetical: V3D gives `D24_UNORM_S8_UINT` and RADV gives `D32_SFLOAT_S8_UINT`, so an
/// attachment reused across a format change is an attachment of the wrong format.
#[test]
fn a_different_format_is_not_served() {
    let have = target::Shape {
        width: 64,
        height: 64,
        format: vk::Format::D24_UNORM_S8_UINT,
    };
    let wanted = target::Shape {
        format: vk::Format::D32_SFLOAT_S8_UINT,
        ..have
    };
    assert!(
        !have.serves(wanted),
        "a D24 attachment must not serve a D32 target"
    );
}

/// A resize is a different attachment, in either dimension.
#[test]
fn a_resize_is_not_served() {
    let have = target::Shape {
        width: 1920,
        height: 1080,
        format: vk::Format::D24_UNORM_S8_UINT,
    };
    assert!(
        !have.serves(target::Shape {
            width: 1280,
            ..have
        }),
        "width"
    );
    assert!(
        !have.serves(target::Shape {
            height: 720,
            ..have
        }),
        "height"
    );
    // And a transpose, which a comparison on area alone would accept.
    assert!(
        !have.serves(target::Shape {
            width: 1080,
            height: 1920,
            ..have
        }),
        "a transposed target has the same area and is not the same attachment"
    );
}
