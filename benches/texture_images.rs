// SPDX-License-Identifier: BSD-2-Clause
//! A texture's image on a real device, with the pixels read back.
//!
//! A bench rather than a test for the reason the other device benches are: it needs a GPU and CI has
//! none. `tests/texture_staging.rs` holds the half that does not, which is the format and the
//! staging layout.
//!
//! # What this proves that the layout tests cannot
//!
//! That a region lands where its rect says. The layout tests prove the staging offsets are aligned
//! and do not overlap; this writes distinct pixels per region, copies them into an image, reads the
//! whole image back and compares it texel by texel. A region copied to the right offset in the
//! wrong place in the image is caught here and nowhere else -- and it is a failure that draws: an
//! atlas whose glyphs sit at each other's coordinates is a map with labels made of the wrong shapes,
//! not a blank one.
//!
//! It also proves the clear. An image is written only where the producer reported damage, so every
//! texel outside a rect is whatever `vkCreateImage`'s memory arrived holding. On the `GC7000UL` that
//! is a previously exited process's bytes -- measured for buffers in #64 -- so `declare` clears, and
//! `untouched_texels_are_zero` is the case that says so.
//!
//! Run with `cargo bench --bench texture_images`.

mod common;

use ash::vk;
use tessella_capture_abi::envelope::{Extent, Rect16, TextureId};
use tessella_capture_abi::generated::mbgl_enums::{TextureChannelDataType, TexturePixelType};
use tessella_emblema::images::{Error, Images};
use tessella_vk::Recorder;

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

const BYTE: TextureChannelDataType = TextureChannelDataType::UnsignedByte;
const ALPHA: TexturePixelType = TexturePixelType::Alpha;
const RGBA: TexturePixelType = TexturePixelType::RGBA;

const fn rect(x: u16, y: u16, w: u16, h: u16) -> Rect16 {
    Rect16 { x, y, w, h }
}

const fn size(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

/// The byte every texel of region `index` is filled with.
///
/// Distinct per region and uniform within one, so a region copied to the wrong place shows up as
/// another region's number rather than as plausible noise. Never zero, so it cannot be confused with
/// the cleared background.
const fn fill(index: usize) -> u8 {
    0xA1 + index as u8
}

/// Reads a whole `Alpha` image back as one byte per texel.
///
/// The slow way to look at an image and the only way without drawing it: a staging buffer, a
/// transition to `TRANSFER_SRC_OPTIMAL`, an image-to-buffer copy, and a map.
fn read_back(
    device: &Open,
    images: &Images<'_>,
    texture: TextureId,
    extent: Extent,
) -> Result<Vec<u8>, String> {
    let texels = (extent.width * extent.height) as usize;
    let gpu = device.gpu();
    let buffer = gpu
        .buffer(texels as u64, vk::BufferUsageFlags::TRANSFER_DST)
        .map_err(|why| format!("read-back buffer: {why}"))?;
    let requirements = [buffer.requirements()];
    let memory = gpu
        .allocate(
            (texels as u64).max(requirements[0].size),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|why| format!("read-back memory: {why}"))?;
    memory
        .bind(&buffer, 0)
        .map_err(|why| format!("read-back bind: {why}"))?;

    let image = images.image(texture).ok_or("the texture has no image")?;
    device.submit(|record: Recorder<'_>| {
        // The image is left in TRANSFER_DST_OPTIMAL by `declare` and `upload`, so this is the
        // transition a sampler would otherwise make -- to TRANSFER_SRC instead, to read it. An
        // unrecognized pair, so it takes `Recorder::transition`'s conservative ALL_COMMANDS branch,
        // which is the right answer for a read-back and the wrong one for a frame.
        record.transition(
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        record.copy_to_buffer(image, &buffer, extent.width, extent.height);
    })?;

    let mut out = vec![0u8; texels];
    let mapping = memory
        .map()
        .map_err(|why| format!("read-back map: {why}"))?;
    mapping
        .read(0, &mut out)
        .map_err(|why| format!("read-back read: {why}"))?;
    Ok(out)
}

/// Scattered regions land at their own rects, and nowhere else.
fn scattered_regions(device: &Open) -> Result<(), String> {
    let extent = size(8, 8);
    let rects = [rect(0, 0, 2, 2), rect(5, 1, 2, 3), rect(3, 6, 4, 1)];
    let mut images = Images::new();
    let texture = TextureId(1);

    device.submit(|record| {
        let _ = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
    })?;
    if images.view(texture).is_none() {
        return Err("the texture has no view after declare".into());
    }
    if images.format(texture) != Some(vk::Format::R8_UNORM) {
        return Err(format!("Alpha/byte became {:?}", images.format(texture)));
    }

    // Every row of a region is that region's own byte.
    let rows: Vec<Vec<u8>> = rects
        .iter()
        .enumerate()
        .map(|(index, r)| vec![fill(index); usize::from(r.w)])
        .collect();
    let resolve = |index: usize, _row: u16| rows.get(index).map(Vec::as_slice);

    let mut uploaded = Ok(false);
    device.submit(|record| {
        uploaded = images.upload(device.gpu(), record, texture, &rects, &resolve);
    })?;
    if uploaded != Ok(true) {
        return Err(format!("the upload answered {uploaded:?}"));
    }

    let found = read_back(device, &images, texture, extent)?;
    for y in 0..extent.height as u16 {
        for x in 0..extent.width as u16 {
            let at = (y as usize) * extent.width as usize + x as usize;
            let wanted = rects
                .iter()
                .position(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
                .map_or(0, fill);
            if found[at] != wanted {
                return Err(format!(
                    "texel ({x}, {y}) is {:#04x} against {wanted:#04x}",
                    found[at]
                ));
            }
        }
    }
    println!("  scattered regions     ok   3 rects placed, 50 texels left clear");
    Ok(())
}

/// Every texel outside a region is zero, on a newly declared image.
///
/// The case the clear exists for. One small region of an 8x8 image leaves 60 texels nobody wrote,
/// and `vkCreateImage` promises nothing about what they hold.
fn untouched_texels_are_zero(device: &Open) -> Result<(), String> {
    let extent = size(8, 8);
    let rects = [rect(2, 2, 2, 2)];
    let mut images = Images::new();
    let texture = TextureId(2);

    device.submit(|record| {
        let _ = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
    })?;
    let row = vec![fill(0); 2];
    let resolve = |_index: usize, _row: u16| Some(row.as_slice());
    device.submit(|record| {
        let _ = images.upload(device.gpu(), record, texture, &rects, &resolve);
    })?;

    let found = read_back(device, &images, texture, extent)?;
    let outside = found
        .iter()
        .enumerate()
        .filter(|(at, _)| {
            let (x, y) = ((at % 8) as u16, (at / 8) as u16);
            !((2..4).contains(&x) && (2..4).contains(&y))
        })
        .filter(|(_, byte)| **byte != 0)
        .count();
    if outside != 0 {
        return Err(format!(
            "{outside} of 60 texels outside the damaged region are not zero"
        ));
    }
    println!("  untouched are zero    ok   60 texels outside the rect, all clear");
    Ok(())
}

/// A whole-texture update with no rects covers every texel.
fn an_empty_rect_list_covers_everything(device: &Open) -> Result<(), String> {
    let extent = size(4, 4);
    let mut images = Images::new();
    let texture = TextureId(3);
    device.submit(|record| {
        let _ = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
    })?;

    let row = vec![0x5Cu8; 4];
    let resolve = |_index: usize, _row: u16| Some(row.as_slice());
    device.submit(|record| {
        let _ = images.upload(device.gpu(), record, texture, &[], &resolve);
    })?;

    let found = read_back(device, &images, texture, extent)?;
    if found.iter().any(|byte| *byte != 0x5C) {
        return Err(format!(
            "an empty rect list left {} texels unwritten",
            found.iter().filter(|b| **b != 0x5C).count()
        ));
    }
    println!("  whole texture         ok   16 of 16 texels written");
    Ok(())
}

/// The refusals, and that none of them leaves anything on the device.
fn refusals(device: &Open) -> Result<(), String> {
    let mut images = Images::new();
    let texture = TextureId(4);
    let extent = size(4, 4);

    // A pair with no format, which is what mbgl gives luminance and depth.
    let mut outcome = Ok(false);
    device.submit(|record| {
        outcome = images.declare(
            device.gpu(),
            record,
            TextureId(9),
            extent,
            TexturePixelType::Luminance,
            BYTE,
        );
    })?;
    match outcome {
        Err(Error::NoFormat { .. }) => {}
        other => return Err(format!("a luminance texture gave {other:?}")),
    }
    if images.textures() != 0 {
        return Err("a refused declare left a texture resident".into());
    }

    device.submit(|record| {
        let _ = images.declare(device.gpu(), record, texture, extent, RGBA, BYTE);
    })?;
    let before = images.total_bytes();
    if before == 0 {
        return Err("a declared texture has no footprint".into());
    }

    // A different size is a different image.
    device.submit(|record| {
        outcome = images.declare(device.gpu(), record, texture, size(8, 8), RGBA, BYTE);
    })?;
    match outcome {
        Err(Error::Reshaped { size: had, .. }) if had == extent => {}
        other => return Err(format!("a reshaped texture gave {other:?}")),
    }
    // And so is a different format at the same size.
    device.submit(|record| {
        outcome = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
    })?;
    match outcome {
        Err(Error::Reshaped { .. }) => {}
        other => return Err(format!("a reformatted texture gave {other:?}")),
    }
    // The same shape twice is idempotent, not a refusal.
    device.submit(|record| {
        outcome = images.declare(device.gpu(), record, texture, extent, RGBA, BYTE);
    })?;
    if outcome != Ok(false) {
        return Err(format!("declaring the same shape twice gave {outcome:?}"));
    }

    // A row shorter than the rect claims.
    let short = vec![0u8; 3];
    let resolve = |_index: usize, _row: u16| Some(short.as_slice());
    let mut uploaded = Ok(false);
    device.submit(|record| {
        uploaded = images.upload(device.gpu(), record, texture, &[rect(0, 0, 4, 1)], &resolve);
    })?;
    match uploaded {
        Err(Error::Short { rect: 0 }) => {}
        other => return Err(format!("a short row gave {other:?}")),
    }

    // A texture nothing declared.
    let nothing = |_: usize, _: u16| None;
    device.submit(|record| {
        uploaded = images.upload(
            device.gpu(),
            record,
            TextureId(77),
            &[rect(0, 0, 1, 1)],
            &nothing,
        );
    })?;
    if uploaded != Ok(false) {
        return Err(format!("an undeclared texture gave {uploaded:?}"));
    }

    images.forget(texture);
    if images.textures() != 0 || images.total_bytes() != 0 || images.view(texture).is_some() {
        return Err("forgetting left something behind".into());
    }
    println!("  refusals and forget   ok   {before} bytes resident, then none");
    Ok(())
}

/// Two textures do not share an image.
fn textures_are_separate(device: &Open) -> Result<(), String> {
    let extent = size(4, 4);
    let mut images = Images::new();
    let (first, second) = (TextureId(5), TextureId(6));
    for texture in [first, second] {
        device.submit(|record| {
            let _ = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
        })?;
    }
    if images.image(first).map(tessella_vk::Image::raw)
        == images.image(second).map(tessella_vk::Image::raw)
    {
        return Err("two textures got the same image".into());
    }
    if images.view(first) == images.view(second) {
        return Err("two textures got the same view".into());
    }

    let row = vec![0x7Eu8; 4];
    let resolve = |_index: usize, _row: u16| Some(row.as_slice());
    device.submit(|record| {
        let _ = images.upload(device.gpu(), record, first, &[], &resolve);
    })?;

    let other = read_back(device, &images, second, extent)?;
    if other.iter().any(|byte| *byte != 0) {
        return Err("writing the first texture reached the second".into());
    }
    let mine = read_back(device, &images, first, extent)?;
    if mine.iter().any(|byte| *byte != 0x7E) {
        return Err("the first texture's own write did not arrive".into());
    }
    println!("  textures are separate ok   two images, neither in the other");
    Ok(())
}

/// The staging buffer grows and is not reallocated for a smaller upload.
fn the_staging_buffer_is_reused(device: &Open) -> Result<(), String> {
    let extent = size(64, 64);
    let mut images = Images::new();
    let texture = TextureId(7);
    device.submit(|record| {
        let _ = images.declare(device.gpu(), record, texture, extent, ALPHA, BYTE);
    })?;
    let bare = images.total_bytes();

    let big = vec![0x11u8; 32];
    let resolve = |_index: usize, _row: u16| Some(big.as_slice());
    device.submit(|record| {
        let _ = images.upload(
            device.gpu(),
            record,
            texture,
            &[rect(0, 0, 32, 32)],
            &resolve,
        );
    })?;
    let grown = images.total_bytes();
    if grown <= bare {
        return Err(format!(
            "the staging buffer did not appear: {bare} then {grown}"
        ));
    }

    // A smaller upload must not shrink it: an atlas's damage varies frame to frame, and a buffer
    // that shrank would be reallocated the next time it grew back.
    let small = vec![0x22u8; 2];
    let resolve = |_index: usize, _row: u16| Some(small.as_slice());
    device.submit(|record| {
        let _ = images.upload(device.gpu(), record, texture, &[rect(0, 0, 2, 2)], &resolve);
    })?;
    if images.total_bytes() != grown {
        return Err(format!(
            "a smaller upload changed the footprint: {grown} then {}",
            images.total_bytes()
        ));
    }
    println!(
        "  staging is reused     ok   {bare} bytes, then {grown}, unchanged by a smaller upload"
    );
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

    let mut failed = 0;
    for (name, case) in [
        ("scattered_regions", scattered_regions as Case),
        ("untouched_are_zero", untouched_texels_are_zero),
        ("whole_texture", an_empty_rect_list_covers_everything),
        ("refusals", refusals),
        ("textures_are_separate", textures_are_separate),
        ("staging_is_reused", the_staging_buffer_is_reused),
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
