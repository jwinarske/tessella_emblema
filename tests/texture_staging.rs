// SPDX-License-Identifier: BSD-2-Clause
//! Which format a texture is, and where its regions sit in one staging buffer.
//!
//! # What this is for
//!
//! Both halves of putting a texture on a device that can be decided without one, split out for the
//! reason `store::layout` and `blocks::sizing` were: the rest needs a GPU and CI has none, so a
//! decision left inside it is a decision nothing checks.
//!
//! # What would be caught
//!
//! A `bufferOffset` that does not satisfy its alignment, which `vkCmdCopyBufferToImage` rejects --
//! but only on a queue strict enough to notice. The texel-block rule always applies; the
//! multiple-of-four rule applies only on a transfer-only queue, which is the queue an uploader
//! should be using and is not the queue a desktop test runs on. So a layout aligned to the texel
//! alone passes everywhere this gets tested and fails where it ships.
//!
//! And a format chosen by the pixel type alone. A color relief's elevation stops are `RGBA` and
//! `Float`; made as `R8G8B8A8_UNORM` they take a quarter of the payload and sample a ramp that has
//! collapsed, which draws.

use ash::vk;
use tessella_capture_abi::envelope::{Extent, Rect16};
use tessella_capture_abi::generated::mbgl_enums::{TextureChannelDataType, TexturePixelType};
use tessella_emblema::device::{self, Unsupported};
use tessella_emblema::textures::{self, Placed};

const BYTE: TextureChannelDataType = TextureChannelDataType::UnsignedByte;
const HALF: TextureChannelDataType = TextureChannelDataType::HalfFloat;
const FLOAT: TextureChannelDataType = TextureChannelDataType::Float;

const fn rect(x: u16, y: u16, w: u16, h: u16) -> Rect16 {
    Rect16 { x, y, w, h }
}

const fn size(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

/// The format is mbgl's, both halves of it.
///
/// Transcribed from `Texture2D::vulkanFormat`. Pinned rather than trusted because the pair is the
/// point: the channel type is what separates a color relief's stops from an ordinary atlas, and
/// reading only the pixel type gives both the same format.
#[test]
fn the_format_comes_from_both_halves() {
    use TexturePixelType::{Alpha, RGBA};
    for (pixel, channel, wanted) in [
        (Alpha, BYTE, vk::Format::R8_UNORM),
        (Alpha, HALF, vk::Format::R16_SFLOAT),
        (Alpha, FLOAT, vk::Format::R32_SFLOAT),
        (RGBA, BYTE, vk::Format::R8G8B8A8_UNORM),
        (RGBA, HALF, vk::Format::R16G16B16A16_SFLOAT),
        (RGBA, FLOAT, vk::Format::R32G32B32A32_SFLOAT),
    ] {
        assert_eq!(
            device::texture_format(pixel, channel),
            Some(wanted),
            "{pixel:?} and {channel:?}"
        );
    }
    // And the pair actually disagrees, which is the property the table exists for.
    assert_ne!(
        device::texture_format(RGBA, BYTE),
        device::texture_format(RGBA, FLOAT),
        "a format read from the pixel type alone would make these equal"
    );
}

/// Stencil is packed and ignores the channel type, which is mbgl's own early return.
#[test]
fn stencil_ignores_the_channel_type() {
    for channel in [BYTE, HALF, FLOAT] {
        assert_eq!(
            device::texture_format(TexturePixelType::Stencil, channel),
            Some(vk::Format::S8_UINT)
        );
    }
}

/// Depth and luminance have no format here, because mbgl's Vulkan backend gives them none.
///
/// Depth it refuses outright; luminance falls past both of its `if`s to the final `eUndefined`.
/// The ABI can describe a luminance texture and this backend cannot make one, so this says so
/// rather than inventing `R8_UNORM` -- which would sample, and sample something nobody chose.
#[test]
fn depth_and_luminance_have_no_format() {
    for pixel in [TexturePixelType::Depth, TexturePixelType::Luminance] {
        for channel in [BYTE, HALF, FLOAT] {
            assert_eq!(
                device::texture_format(pixel, channel),
                None,
                "{pixel:?} must have no format rather than a guessed one"
            );
        }
    }
}

/// A device missing either feature on a format is an error naming that format.
#[test]
fn a_format_the_device_will_not_sample_is_refused() {
    let wanted = [vk::Format::R8_UNORM, vk::Format::R8G8B8A8_UNORM];
    let both = vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;

    assert_eq!(device::check_texture_formats(&wanted, |_| both), Ok(()));

    // Sampled but not writable, which is the half a renderable-only format would have.
    assert_eq!(
        device::check_texture_formats(&wanted, |_| vk::FormatFeatureFlags::SAMPLED_IMAGE),
        Err(Unsupported::TextureFormat(vk::Format::R8_UNORM)),
        "a format that cannot be a transfer destination cannot be filled"
    );
    // Writable but not sampled.
    assert_eq!(
        device::check_texture_formats(&wanted, |_| vk::FormatFeatureFlags::TRANSFER_DST),
        Err(Unsupported::TextureFormat(vk::Format::R8_UNORM))
    );
    // And the second one specifically, so the search is not answering with the first either way.
    assert_eq!(
        device::check_texture_formats(&wanted, |format| if format == vk::Format::R8_UNORM {
            both
        } else {
            vk::FormatFeatureFlags::empty()
        }),
        Err(Unsupported::TextureFormat(vk::Format::R8G8B8A8_UNORM))
    );
}

/// A texel is both factors.
#[test]
fn a_texel_is_channels_times_channel_size() {
    assert_eq!(textures::texel(TexturePixelType::Alpha, BYTE), 1);
    assert_eq!(textures::texel(TexturePixelType::RGBA, BYTE), 4);
    assert_eq!(textures::texel(TexturePixelType::RGBA, HALF), 8);
    assert_eq!(textures::texel(TexturePixelType::RGBA, FLOAT), 16);
    assert_eq!(textures::texel(TexturePixelType::Alpha, FLOAT), 4);
}

/// The alignment is the stricter of the two rules.
///
/// The texel-block rule always applies; the multiple-of-four rule applies on a transfer-only
/// queue. A one-byte texel aligned to one would be accepted on a graphics queue and rejected on a
/// transfer queue, which is the queue the upload belongs on.
#[test]
fn the_alignment_is_the_stricter_rule() {
    assert_eq!(
        textures::copy_alignment(1),
        4,
        "a byte texel still needs four"
    );
    assert_eq!(textures::copy_alignment(2), 4);
    assert_eq!(textures::copy_alignment(4), 4);
    assert_eq!(
        textures::copy_alignment(8),
        8,
        "past four the texel rule wins"
    );
    assert_eq!(textures::copy_alignment(16), 16);
}

/// Regions are packed end to end, each starting at an aligned offset.
#[test]
fn regions_are_packed_at_aligned_offsets() {
    // Alpha, so a texel is one byte and the alignment is four -- the case where the two rules
    // differ and every region needs padding it would not otherwise get.
    let rects = [rect(0, 0, 3, 2), rect(8, 8, 2, 2), rect(1, 1, 5, 1)];
    let found = textures::staging(&rects, size(64, 64), TexturePixelType::Alpha, BYTE);

    assert_eq!(
        found.placed,
        [
            // 3 * 2 = 6 bytes, so the next region rounds up from 6 to 8.
            Placed {
                rect: rects[0],
                at: 0,
                row_length: 3
            },
            // 2 * 2 = 4 bytes, ending at 12, which is already aligned.
            Placed {
                rect: rects[1],
                at: 8,
                row_length: 2
            },
            Placed {
                rect: rects[2],
                at: 12,
                row_length: 5
            },
        ]
    );
    assert_eq!(found.total, 20, "5 bytes from 12 is 17, rounded up to 20");
}

/// Every offset satisfies the alignment, at every texel size and rect shape.
///
/// Asserted as a property rather than against a table. The shapes are deliberately awkward -- odd
/// widths, a single row, a zero -- because a rounding error hides in a tidy number, and this is the
/// failure a desktop queue does not report.
#[test]
fn every_offset_is_aligned() {
    let rects = [
        rect(0, 0, 1, 1),
        rect(0, 0, 7, 3),
        rect(0, 0, 0, 0),
        rect(0, 0, 13, 1),
        rect(0, 0, 2, 9),
    ];
    for (pixel, channel) in [
        (TexturePixelType::Alpha, BYTE),
        (TexturePixelType::Alpha, FLOAT),
        (TexturePixelType::RGBA, BYTE),
        (TexturePixelType::RGBA, HALF),
        (TexturePixelType::RGBA, FLOAT),
    ] {
        let texel = textures::texel(pixel, channel);
        let alignment = textures::copy_alignment(texel);
        let found = textures::staging(&rects, size(64, 64), pixel, channel);
        for placed in &found.placed {
            assert_eq!(
                placed.at % alignment,
                0,
                "{pixel:?}/{channel:?}: offset {} is not a multiple of {alignment}",
                placed.at
            );
        }
        assert_eq!(
            found.total % alignment,
            0,
            "{pixel:?}/{channel:?}: the total is not aligned, so a texture packed after this one \
             would start misaligned"
        );
    }
}

/// No two regions overlap, and every one fits inside the total.
///
/// An overlap is two regions reading each other's pixels, which uploads the right number of bytes
/// to the right place taken from the wrong rows -- an atlas whose glyphs are each other's.
#[test]
fn no_two_regions_overlap() {
    let rects = [rect(0, 0, 5, 4), rect(0, 0, 1, 1), rect(0, 0, 9, 2)];
    for (pixel, channel) in [
        (TexturePixelType::Alpha, BYTE),
        (TexturePixelType::RGBA, FLOAT),
    ] {
        let texel = textures::texel(pixel, channel);
        let found = textures::staging(&rects, size(64, 64), pixel, channel);
        let mut spans: Vec<(u64, u64)> = found
            .placed
            .iter()
            .map(|placed| {
                let bytes = u64::from(placed.rect.w) * u64::from(placed.rect.h) * texel;
                (placed.at, placed.at + bytes)
            })
            .collect();
        spans.sort_unstable();
        for pair in spans.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "{pixel:?}/{channel:?}: a region ending at {} overlaps one starting at {}",
                pair[0].1,
                pair[1].0
            );
        }
        let last = spans.last().copied().unwrap_or((0, 0));
        assert!(
            last.1 <= found.total,
            "{pixel:?}/{channel:?}: the last region ends at {} past a total of {}",
            last.1,
            found.total
        );
    }
}

/// A zero-area region keeps its slot.
///
/// The rect list is positional -- a backend walks it beside `tessella_consume::upload::rows`, which
/// returns one entry per rect -- so dropping an empty one would pair every later region with
/// another's source bytes.
#[test]
fn a_zero_area_region_keeps_its_slot() {
    let rects = [rect(0, 0, 4, 1), rect(0, 0, 0, 0), rect(0, 0, 4, 1)];
    let found = textures::staging(&rects, size(64, 64), TexturePixelType::RGBA, BYTE);
    assert_eq!(found.placed.len(), 3, "the empty slot was dropped");
    assert_eq!(found.placed[0].at, 0);
    assert_eq!(found.placed[1].at, 16, "where the first region ended");
    assert_eq!(
        found.placed[2].at, 16,
        "and the third starts there too, because nothing was staged between"
    );
    assert_eq!(found.total, 32);
}

/// An empty rect list stages the whole texture as one region.
#[test]
fn an_empty_rect_list_stages_the_whole_texture() {
    let found = textures::staging(&[], size(8, 4), TexturePixelType::RGBA, BYTE);
    assert_eq!(
        found.placed,
        [Placed {
            rect: rect(0, 0, 8, 4),
            at: 0,
            row_length: 8
        }]
    );
    assert_eq!(found.total, 8 * 4 * 4);
}

/// An extent past what a rect can address is clamped rather than wrapped.
#[test]
fn an_extent_past_a_rect_is_clamped() {
    let found = textures::staging(&[], size(70_000, 1), TexturePixelType::Alpha, BYTE);
    assert_eq!(found.placed[0].rect.w, u16::MAX);
    assert_eq!(found.placed[0].row_length, u32::from(u16::MAX));
}

/// A maximal rect list's total is well inside a `u64`, which is why `staging` is infallible.
///
/// The bound the arithmetic rests on, asserted rather than asserted-in-a-comment. One region is at
/// most `65535 * 65535 * 16` bytes and §6.4 caps an update's list at four, so the total cannot
/// approach a `u64`; it would take 2.68e8 regions. A `Result` here would be a branch nothing can
/// take and no test can reach -- this test exists because the first version of it was exactly that,
/// named for a refusal it then asserted the absence of.
#[test]
fn a_maximal_rect_list_is_far_inside_a_u64() {
    let maximal =
        [rect(0, 0, u16::MAX, u16::MAX); tessella_capture_abi::envelope::TEXTURE_RECT_CAP];
    let found = textures::staging(
        &maximal,
        size(u32::from(u16::MAX), u32::from(u16::MAX)),
        TexturePixelType::RGBA,
        FLOAT,
    );

    let one = u64::from(u16::MAX) * u64::from(u16::MAX) * 16;
    assert_eq!(one, 68_717_379_600, "one maximal RGBA-float region");
    assert_eq!(
        found.total,
        one * 4,
        "four of them, each already aligned to 16"
    );
    assert!(
        found.total < u64::MAX / 1_000_000,
        "the cap leaves six orders of magnitude of headroom, which is the argument for no Result"
    );
}
