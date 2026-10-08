// SPDX-License-Identifier: BSD-2-Clause
//! Rendering into an image the host allocated.
//!
//! The hand-off #60 asks for, and the last part of its order. The host owns the image: it allocates
//! a ring of at least three on emblema's device, exports each as a dma-buf with an explicit DRM
//! modifier, and submits the frames itself. This pass is a guest -- it records into a command buffer
//! it was handed and never submits.
//!
//! # What is per-size and what is per-image
//!
//! The distinction the issue turns on. It says the pass
//!
//! > must not cache per-image state that breaks when the image changes every frame
//!
//! There is none. [`pipelines`](crate::pipelines) keys pipelines by formats, which the whole ring
//! shares; the viewport is dynamic; and there is no framebuffer to key by image because there is no
//! render pass. What is left is the depth-stencil attachment, and that is **per size, not per
//! image**: the ring's images are all one size, so one [`Depth`] serves all three and is remade only
//! when the host resizes.
//!
//! So a ring of three costs one depth attachment, and rotating through it costs nothing.
//!
//! # The layouts
//!
//! The host says what layout the image is in; the pass states what layout it leaves it in. Both
//! halves matter, and the second is [`LEAVES_IN`] -- `GENERAL`, because the image leaves Vulkan.
//! A compositor importing a dma-buf cannot be told a vendor's optimal layout, and there is no
//! negotiation across that boundary, so `GENERAL` is the one layout both sides can mean the same
//! thing by. The host records it with emblema's `assume_layout` so its own wrapper agrees.

use ash::vk;
use tessella_vk::{Gpu, Image, ImageView, Memory, Recorder};

/// The layout this pass leaves the host's image in.
///
/// `GENERAL`, for the reason in this module's notes: the image is exported and read by something
/// that is not Vulkan. Stated as a constant rather than returned, because the host has to agree with
/// it before the frame is recorded, not after.
pub const LEAVES_IN: vk::ImageLayout = vk::ImageLayout::GENERAL;

/// Why a target could not be prepared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The device refused.
    Device(tessella_vk::Error),
}

impl From<tessella_vk::Error> for Error {
    fn from(why: tessella_vk::Error) -> Self {
        Self::Device(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// What a depth-stencil attachment is for: a size and a format, and nothing per-image.
///
/// The whole of #60's requirement lives in this struct's fields. A ring's images differ from each
/// other, and share these -- so an attachment keyed by a `Shape` is made once for the ring, and one
/// keyed by anything else is made per frame.
///
/// Split out from [`Depth`] so the comparison can be exercised without a device, which is where the
/// mistake would be: a `serves` that ignored the format would reuse a `D24_UNORM_S8_UINT`
/// attachment where the device wanted `D32_SFLOAT_S8_UINT`, and the two parts on the bench disagree
/// about which they give.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Width in pixels, shared by the ring.
    pub width: u32,
    /// Height in pixels, shared by the ring.
    pub height: u32,
    /// The depth-stencil format this device gives, from [`crate::device::depth_stencil_format`].
    pub format: vk::Format,
}

impl Shape {
    /// Whether an attachment of this shape can be used for a target of `wanted`.
    ///
    /// Equality, and the point is what is *absent*: no image handle, no view, no frame number.
    #[must_use]
    pub fn serves(&self, wanted: Self) -> bool {
        *self == wanted
    }
}

/// The depth-stencil attachment the pass owns, for one target size.
///
/// Not the host's: the host allocates color images for its compositor and has no use for this one.
/// Transient as far as anything outside a frame is concerned -- nothing reads it between frames,
/// which is why the rendering scope clears it on load and discards it on store.
pub struct Depth<'d> {
    image: Image<'d>,
    view: ImageView<'d>,
    _memory: Memory<'d>,
    shape: Shape,
}

impl core::fmt::Debug for Depth<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Depth")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl<'d> Depth<'d> {
    /// Creates the attachment for one size and format.
    ///
    /// `LAZILY_ALLOCATED` is asked for first and fallen back from. On a tiler the attachment never
    /// needs to exist in memory at all -- it is written and discarded inside one pass, which is what
    /// `DONT_CARE` on store tells the driver -- and a lazy allocation is how that is expressed. A
    /// device without it gets an ordinary device-local one, which is correct and merely uses memory.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the image, the allocation or the view is refused.
    pub fn new(gpu: Gpu<'d>, width: u32, height: u32, format: vk::Format) -> Result<Self, Error> {
        let image = gpu.image(
            width,
            height,
            format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        )?;
        let requirements = [image.requirements()];
        let memory = gpu
            .allocate(
                requirements[0].size,
                &requirements,
                vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            )
            .or_else(|_| {
                gpu.allocate(
                    requirements[0].size,
                    &requirements,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
            })?;
        memory.bind_image(&image, 0)?;
        let view = gpu.depth_view(&image, format)?;
        Ok(Self {
            image,
            view,
            _memory: memory,
            shape: Shape {
                width,
                height,
                format,
            },
        })
    }

    /// Whether this attachment already serves a target of that size and format.
    ///
    /// What a host rotating a ring asks before remaking anything: the answer is yes for every image
    /// in the ring, and no only when the host has resized.
    #[must_use]
    pub fn serves(&self, width: u32, height: u32, format: vk::Format) -> bool {
        self.shape.serves(Shape {
            width,
            height,
            format,
        })
    }

    /// What this attachment is for.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// The view, for the rendering scope.
    #[must_use]
    pub fn view(&self) -> &ImageView<'d> {
        &self.view
    }

    /// The image, for the transition into an attachment layout.
    #[must_use]
    pub fn image(&self) -> &Image<'d> {
        &self.image
    }
}

/// One frame's target: the host's image, and what the host says about it.
///
/// Borrowed rather than owned. The pass does not keep the image alive and must not: the host may
/// retire any image in its ring, and an owned handle here would be a second owner of something that
/// is going away.
#[derive(Debug, Clone, Copy)]
pub struct Host<'a> {
    /// The image itself, for the transitions.
    pub image: &'a Image<'a>,
    /// A view of it, for the rendering scope.
    pub view: &'a ImageView<'a>,
    /// Its width in pixels.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The layout the host says it is in now.
    ///
    /// `UNDEFINED` is legal and means its contents may be discarded, which is what a host hands over
    /// for an image it is about to have fully redrawn. Anything else is preserved by the transition.
    pub layout: vk::ImageLayout,
}

/// Records one frame into the host's image.
///
/// Transitions both attachments in, opens the rendering scope, sets the viewport, calls `draws`, then
/// closes the scope and transitions the host's image to [`LEAVES_IN`].
///
/// `clear` is the color to clear to, or `None` to draw over what the image already holds. A host
/// rotating a ring normally clears, because the image it is handed is three frames old.
///
/// # What this does not do
///
/// Submit, wait, or signal. The command buffer, the queue and the semaphore are the host's --
/// #60 asks for "a semaphore the host can export as a `sync_file`, not a CPU wait", and a pass that
/// submitted could not give it one.
pub fn frame(
    record: Recorder<'_>,
    host: Host<'_>,
    depth: &Depth<'_>,
    clear: Option<[f32; 4]>,
    draws: impl FnOnce(Recorder<'_>),
) {
    record.transition(
        host.image,
        host.layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );
    record.transition(
        depth.image(),
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    );

    record.begin_rendering(host.view, depth.view(), host.width, host.height, clear);
    record.viewport(host.width, host.height);
    draws(record);
    record.end_rendering();

    record.transition(
        host.image,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        LEAVES_IN,
    );
}
