// SPDX-License-Identifier: BSD-2-Clause
//! A color target at any size, with its depth-stencil attachment and a readback buffer.
//!
//! What a bench that draws a whole frame needs, and what `benches/common/mod.rs` does not have:
//! an image to render into, the depth and the stencil to test against, and host-visible memory the
//! frame comes back through.
//!
//! Here rather than in the bench because the size is the point. `benches/a_frame.rs` and
//! `benches/first_pixel.rs` each hold their own, both sized to one fixture's square; a captured
//! frame's camera was computed for the producer's own viewport, and drawing it at another size
//! draws another picture.

// Each bench compiles this separately, so whatever one of them does not name reads as dead.
// `common/mod.rs` and `common/passes.rs` carry the same allow for the same reason.
#![allow(dead_code)]

use ash::vk;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::Targets;
use tessella_emblema::target::Depth;
use tessella_vk::{Buffer, Image, ImageView, Memory};

use super::common::Open;

/// The color format every bench here reads back in, which is also the order the bytes land in.
pub const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// A target and everything a frame needs to be drawn into it and read back out.
pub struct Screen<'d> {
    /// The color attachment.
    pub image: Image<'d>,
    /// Its view, which is what a rendering scope takes.
    pub view: ImageView<'d>,
    _memory: Memory<'d>,
    /// The depth and stencil, at the same size.
    pub depth: Depth<'d>,
    readback: Buffer<'d>,
    read_memory: Memory<'d>,
    /// The formats a pipeline declares to render here.
    pub targets: Targets,
    /// Width and height, in pixels.
    pub size: (u32, u32),
}

impl<'d> Screen<'d> {
    /// Allocates one, with the depth-stencil format the device offers for `attachment`.
    ///
    /// # Errors
    ///
    /// A string naming the step that failed, which is what a bench prints.
    pub fn new(
        open: &'d Open,
        width: u32,
        height: u32,
        attachment: Attachment,
    ) -> Result<Self, String> {
        let gpu = open.gpu();
        let image = gpu
            .image(
                width,
                height,
                COLOR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .map_err(|why| format!("the target image: {why}"))?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|why| format!("the target memory: {why}"))?;
        memory
            .bind_image(&image, 0)
            .map_err(|why| format!("binding the target: {why}"))?;
        let view = gpu
            .view(&image, COLOR)
            .map_err(|why| format!("the target view: {why}"))?;

        let bytes = u64::from(width) * u64::from(height) * 4;
        let readback = gpu
            .buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST)
            .map_err(|why| format!("the readback buffer: {why}"))?;
        let needs = [readback.requirements()];
        let read_memory = gpu
            .allocate(
                bytes.max(needs[0].size),
                &needs,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(|why| format!("the readback memory: {why}"))?;
        read_memory
            .bind(&readback, 0)
            .map_err(|why| format!("binding the readback: {why}"))?;

        let depth_stencil = device::depth_stencil_format(attachment, |format| {
            open.format_properties(format).optimal_tiling_features
        })
        .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
        let depth = Depth::new(gpu, width, height, depth_stencil)
            .map_err(|why| format!("the depth attachment: {why}"))?;

        Ok(Self {
            image,
            view,
            _memory: memory,
            depth,
            readback,
            read_memory,
            targets: Targets {
                color: COLOR,
                depth_stencil,
                attachment,
            },
            size: (width, height),
        })
    }

    /// The buffer a frame is copied into, for `Recorder::copy_to_buffer`.
    pub fn readback(&self) -> &Buffer<'d> {
        &self.readback
    }

    /// The frame, as `width * height` pixels of four bytes.
    ///
    /// # Errors
    ///
    /// A string if the readback memory will not map.
    pub fn pixels(&self) -> Result<Vec<[u8; 4]>, String> {
        let (width, height) = self.size;
        let count = width as usize * height as usize;
        let mut bytes = vec![0u8; count * 4];
        let mapping = self
            .read_memory
            .map()
            .map_err(|why| format!("mapping the readback: {why}"))?;
        mapping
            .read(0, &mut bytes)
            .map_err(|why| format!("reading the frame: {why}"))?;
        Ok(bytes.as_chunks::<4>().0.to_vec())
    }
}
