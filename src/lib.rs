//! The map pass of tessella's capture stream, drawn with Vulkan on emblema's device.
//!
//! tessella produces envelopes and does not draw; this crate draws them. It renders with raw
//! `ash` on the `VkDevice` and `VkQueue` emblema already owns, into an image emblema created, and
//! hands that image back for the canvas to composite a HUD over. emblema is asked for four small
//! seams rather than a map renderer, and its HAL trait is untouched — see the README for why the
//! map pass is not built inside that HAL.
//!
//! # What is here
//!
//! [`device`], which is the part of the design that can be decided and tested without a GPU, a
//! stream, or the layout hand-off this crate is blocked on. It is not a placeholder: owning a
//! depth-stencil attachment means owning the obligation to *query* what the device supports, and
//! that obligation is what the module encodes.
//!
//! [`residency`], which is the bookkeeping either side of a device allocation: what the device
//! holds, what it still needs, and what it may let go of. The use-after-free lives there rather
//! than in the allocation, which is why it is the half that can be tested without a device.
//!
//! [`uniforms`], which shadows a layer's consolidated buffer so a frame's scattered slot writes
//! become the few contiguous ranges §11.7 asks for rather than a whole-buffer rewrite.
//!
//! And [`spec`], which turns a shader's permutation switches back into specialization constants
//! after naga has resolved them away. That is the whole of what stands between one module per
//! (family, surface) and one module per permutation — see `tests/naga_overrides.rs` for why it is
//! needed and `spec` for what it does.
//!
//! # What is not here yet
//!
//! In dependency order: the reader half (`tessella-consume`, ported once to Rust and living in
//! the tessella workspace); the geometry, uniform-block and texture stores keyed by the ABI's
//! ids; the shader modules, one per (family, surface); and the hand-off itself, which waits on
//! emblema gaining `VulkanTexture::assume_layout`.
//!
//! Nothing here reads the stream yet. [`tessella_capture_abi`] is a dependency from the first
//! commit regardless, because it is the boundary this crate exists to sit on and its types are
//! what every store above will be keyed by.

#![forbid(unsafe_code)]

pub mod device;
pub mod residency;
pub mod spec;
mod spirv;
pub mod uniforms;

/// The ABI this crate consumes, re-exported so a caller pins one version of it with this crate.
///
/// A consumer linking a different `tessella-capture-abi` than the mirror it drives would read the
/// right bytes at the wrong offsets, which is the failure this re-export exists to make awkward.
pub use tessella_capture_abi;
