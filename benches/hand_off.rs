// SPDX-License-Identifier: BSD-2-Clause
//! Rendering into an image the host allocated, read back.
//!
//! A bench rather than a test because it needs a GPU. `tests/target_sizing.rs` holds the half that
//! does not, which is which state is per-size and which is per-image.
//!
//! # What this proves
//!
//! That the hand-off #60 describes works end to end on a real device, with no render pass and no
//! framebuffer anywhere: the *bench* allocates the image, standing in for the host, hands it over
//! with a layout, and gets it back in the layout the pass states. The pixel read out of it is the
//! evidence -- a rendering scope that never opened, a viewport that was never set or a store op that
//! discarded would all leave the clear color absent, and nothing else in the crate would notice.
//!
//! It also proves the ring. Three images are allocated once, one depth attachment serves all three,
//! and each is rendered into with a different color -- so an image that picked up another's pixels,
//! or a depth attachment that had to be remade per image, shows up here.
//!
//! Run with `cargo bench --bench hand_off`.

mod common;

use ash::vk;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::target::{self, Depth, Host};
use tessella_vk::{Gpu, Image, ImageView, Memory};

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

const SIDE: u32 = 64;
/// What a host hands over. `B8G8R8A8_UNORM` is what a compositor takes.
const COLOR: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// An image the host owns, with its view and allocation.
///
/// Stands in for one entry of #60's ring. Allocated `DEVICE_LOCAL` and read back through a buffer
/// copy, which is what a host exporting a dma-buf would also have to do to inspect one.
struct Owned<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
}

impl<'d> Owned<'d> {
    fn new(gpu: Gpu<'d>) -> Result<Self, String> {
        let image = gpu
            .image(
                SIDE,
                SIDE,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("host image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("host memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("host bind: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("host view: {why}"))?;
        Ok(Self {
            image,
            view,
            _memory: memory,
        })
    }

    fn host(&self, layout: vk::ImageLayout) -> Host<'_> {
        Host {
            image: &self.image,
            view: &self.view,
            width: SIDE,
            height: SIDE,
            layout,
        }
    }
}

/// Reads the whole image back as BGRA bytes.
///
/// The image is in [`target::LEAVES_IN`] when the frame ends, which is the layout the pass states
/// and therefore the one this has to transition out of -- if the pass left it somewhere else, this
/// barrier would be wrong and the copy would read an image in the wrong layout.
fn read_back(device: &Open, owned: &Owned<'_>) -> Result<Vec<u8>, String> {
    let bytes = u64::from(SIDE) * u64::from(SIDE) * 4;
    let gpu = device.gpu();
    let buffer = gpu
        .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
        .map_err(|why| format!("read-back buffer: {why}"))?;
    let requirements = [buffer.requirements()];
    let memory = gpu
        .allocate(
            bytes.max(requirements[0].size),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|why| format!("read-back memory: {why}"))?;
    memory
        .bind(&buffer, 0)
        .map_err(|why| format!("read-back bind: {why}"))?;

    device.submit(|record| {
        record.transition(
            &owned.image,
            target::LEAVES_IN,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        record.copy_to_buffer(&owned.image, &buffer, SIDE, SIDE);
    })?;

    let mut out = vec![0u8; bytes as usize];
    let mapping = memory.map().map_err(|why| format!("map: {why}"))?;
    mapping
        .read(0, &mut out)
        .map_err(|why| format!("read: {why}"))?;
    Ok(out)
}

/// The pass's depth attachment, at the format this device gives.
fn depth_for(device: &Open) -> Result<Depth<'_>, String> {
    let format = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    Depth::new(device.gpu(), SIDE, SIDE, format).map_err(|why| format!("depth: {why}"))
}

/// A frame clears the host's image, and the clear is what comes back.
fn a_frame_reaches_the_host_image(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let depth = depth_for(device)?;

    // Blue in BGRA, which is asymmetric: a channel swap shows up rather than cancelling.
    let clear = [0.0, 0.0, 1.0, 1.0];
    device.submit(|record| {
        target::frame(
            record,
            // UNDEFINED is what a host hands over for an image it is having fully redrawn.
            owned.host(vk::ImageLayout::UNDEFINED),
            &depth,
            Some(clear),
            |_| {},
        );
    })?;

    let found = read_back(device, &owned)?;
    let first: [u8; 4] = found[..4].try_into().map_err(|_| "short read")?;
    // BGRA, so blue is byte zero.
    if first != [255, 0, 0, 255] {
        return Err(format!(
            "the host's image holds {first:?}, wanted [255, 0, 0, 255] -- BGRA blue"
        ));
    }
    if found.chunks(4).any(|texel| texel != first) {
        return Err("the clear did not cover the whole image".into());
    }
    println!("  a frame reaches it    ok   64x64 cleared to BGRA blue, every texel");
    Ok(())
}

/// The pass leaves the image in the layout it says it does.
///
/// Checked by transitioning *out of* that layout and reading a correct pixel. A barrier whose old
/// layout is wrong makes the contents undefined, so a pass that left the image somewhere else would
/// show up as the clear not arriving -- which is the same symptom as several other faults, so this
/// case also states the layout explicitly.
fn the_stated_layout_is_general(_device: &Open) -> Result<(), String> {
    if target::LEAVES_IN != vk::ImageLayout::GENERAL {
        return Err(format!(
            "this bench transitions out of GENERAL; the pass now states {:?}",
            target::LEAVES_IN
        ));
    }
    println!("  the stated layout     ok   GENERAL, which a dma-buf consumer can mean too");
    Ok(())
}

/// A ring of three images shares one depth attachment, and none picks up another's pixels.
fn a_ring_shares_one_depth(device: &Open) -> Result<(), String> {
    let depth = depth_for(device)?;
    if !depth.serves(SIDE, SIDE, depth_format(device)?) {
        return Err("the depth attachment does not serve the size it was made for".into());
    }

    let ring = [
        Owned::new(device.gpu())?,
        Owned::new(device.gpu())?,
        Owned::new(device.gpu())?,
    ];
    // One channel each, so a mix-up is a wrong channel rather than a wrong shade.
    let colors = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ];
    for (owned, clear) in ring.iter().zip(colors) {
        device.submit(|record| {
            target::frame(
                record,
                owned.host(vk::ImageLayout::UNDEFINED),
                &depth,
                Some(clear),
                |_| {},
            );
        })?;
    }

    // BGRA: blue is byte 0, green byte 1, red byte 2.
    let wanted = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    for (at, (owned, want)) in ring.iter().zip(wanted).enumerate() {
        let found = read_back(device, owned)?;
        let first: [u8; 4] = found[..4].try_into().map_err(|_| "short read")?;
        if first != want {
            return Err(format!(
                "ring image {at} holds {first:?}, wanted {want:?}: one image took another's pixels"
            ));
        }
    }
    println!("  a ring of three       ok   one depth attachment, three images, no bleed");
    Ok(())
}

fn depth_format(device: &Open) -> Result<vk::Format, String> {
    device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))
}

/// Drawing over what the image already holds, which is what `None` means.
fn a_frame_can_draw_over(device: &Open) -> Result<(), String> {
    let owned = Owned::new(device.gpu())?;
    let depth = depth_for(device)?;

    device.submit(|record| {
        target::frame(
            record,
            owned.host(vk::ImageLayout::UNDEFINED),
            &depth,
            Some([0.0, 0.0, 1.0, 1.0]),
            |_| {},
        );
    })?;
    // A second frame that loads rather than clears, and draws nothing: the first frame's pixels
    // must survive it. The host's layout this time is what the pass left it in.
    device.submit(|record| {
        target::frame(record, owned.host(target::LEAVES_IN), &depth, None, |_| {});
    })?;

    let found = read_back(device, &owned)?;
    let first: [u8; 4] = found[..4].try_into().map_err(|_| "short read")?;
    if first != [255, 0, 0, 255] {
        return Err(format!(
            "loading lost the previous frame: {first:?} rather than BGRA blue"
        ));
    }
    println!("  loading keeps it      ok   a second frame with no clear kept the first's pixels");
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
        ("frame_reaches_host", a_frame_reaches_the_host_image as Case),
        ("stated_layout", the_stated_layout_is_general),
        ("ring_shares_depth", a_ring_shares_one_depth),
        ("draw_over", a_frame_can_draw_over),
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
