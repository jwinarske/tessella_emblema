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
//! [`textures`], the same bookkeeping for images, plus the part geometry does not have: an update
//! naming a different size or format is a different image rather than damage, and regions
//! accumulate across updates and have to be kept from becoming a whole-texture write by accident.
//!
//! [`shaders`], the largest part: a body per family, and the assembler that joins one to a
//! surface and to the ABI's own declarations. Eighteen families, each transcribed from
//! `include/mbgl/shaders/vulkan/*.hpp` — the Vulkan backend's text, which is what the ABI's
//! tables are generated from and which differs from the GL shaders in places that matter.
//!
//! [`preamble`], which declares those families' uniform blocks from the same tables, and refuses
//! a block whose fields WGSL would place somewhere other than the producer put them.
//!
//! [`surface`], which is the other half of a shader module: a family says what a vertex is and a
//! surface says where it lands, so a module is one (family, surface) pair and the surface supplies
//! the two functions a body places through — `place` for a point and `displace` for a direction,
//! which is linear on a plane and is not on a sphere.
//!
//! [`uniforms`], which shadows a layer's consolidated buffer so a frame's scattered slot writes
//! become the few contiguous ranges §11.7 asks for rather than a whole-buffer rewrite.
//!
//! And [`spec`], which turns a shader's permutation switches back into specialization constants
//! after naga has resolved them away. That is the whole of what stands between one module per
//! (family, surface) and one module per permutation — see `tests/naga_overrides.rs` for why it is
//! needed and `spec` for what it does.
//!
//! # How the shaders are checked
//!
//! Two ways, because reading a shader cannot catch what reading cannot see.
//!
//! `tests/shaders.rs` assembles every (family, surface) pair — 57 of them — compiles each to
//! SPIR-V, and pins the decisions that are silent when wrong. A wrong one of those still draws,
//! which is why they are pinned at all.
//!
//! `benches/first_pixel.rs` *runs* them. Twenty-four cases pick inputs that make one pixel
//! predictable, derive that pixel from mbgl's own arithmetic by hand, draw, and read it back. It
//! is a bench rather than a test because it needs a GPU and CI has none; it has caught defects
//! the pins could not, including a feather mirrored by a coordinate flip naga applies after a
//! body runs.
//!
//! # What is not here yet
//!
//! In dependency order: the geometry, uniform-block and texture stores keyed by the ABI's ids;
//! the pipelines those modules become; and the hand-off itself, which waits on emblema gaining
//! `VulkanTexture::assume_layout`. The reader half is done and lives in the tessella workspace as
//! `tessella-consume`, which hands this crate a `join::Drawable` -- an announcement with its
//! attribute, segment and texture runs already read out of the payload, paired with one view's
//! use of it.
//!
//! [`families`] is what a producer's `builtin_shader` resolves to: the blocks, tables, body and
//! surfaces one family's module is assembled from, keyed by the shader rather than by a name
//! because nothing on the wire carries a name. It also states the coverage -- eighteen of mbgl's
//! thirty-six shader entries are drawn here and the rest are listed with the reason, pinned
//! against the enum so neither list can drift.
//!
//! [`vertices`] is the first of those stores' decisions rather than a store: what a drawable's
//! attribute descriptors come to once they are checked against the family's own table. [`draws`]
//! is the other end of the same drawable: its segments as the indexed draw parameters they
//! become, including the slot that travels as `firstInstance` because the bodies read it as
//! `ubo_index`. [`buffers`] is what those two come to on the device -- the distinct slab
//! references behind a plan's bindings, which is fewer than the bindings whenever the producer
//! interleaved a vertex. [`pipelines`] is what a cache of them is keyed by, which includes the
//! vertex input state because a `VkPipeline` bakes that in -- two drawables of one family and
//! permutation whose strides differ are two pipelines.
//!
//! [`store`] is the first module that puts anything on a device -- and this crate still has no
//! `unsafe` in it.
//!
//! # How it keeps the forbid
//!
//! `ash` is an unsafe FFI binding -- `vkCreateBuffer`, `vkBindBufferMemory` and `vkMapMemory` are all
//! `unsafe fn` -- so a crate that forbids `unsafe` cannot call Vulkan. The way to put a buffer on a
//! device from here was either to drop the forbid or to put the `unsafe` somewhere with a boundary
//! around it.
//!
//! It went into [`tessella_vk`], a thin safe layer in this workspace. Every object there borrows the
//! device it was made from, so the compiler refuses one that outlives the device rather than a doc
//! comment asking nicely, and a write through a mapping is bounds-checked. What this crate gets is
//! RAII: [`store`] creates buffers and an allocation, and a `?` part-way through frees whatever was
//! made because the types own it.
//!
//! That is the second reason to prefer the wrapper over the forbid: the unwinding. Between the first
//! buffer and the last bind there are five fallible calls and each leaves more to undo than the one
//! before, so writing it by hand means the error paths outnumber the happy one and the leak lives in
//! whichever was not written.
//!
//! [`tessella_capture_abi`] is a dependency from the first commit regardless, because it is the
//! boundary this crate exists to sit on and its types are what every store above is keyed by.

#![forbid(unsafe_code)]

pub mod buffers;
pub mod device;
pub mod draws;
pub mod families;
pub mod pipelines;
pub mod preamble;
pub mod residency;
pub mod shaders;
pub mod spec;
mod spirv;
pub mod store;
pub mod surface;
pub mod textures;
pub mod uniforms;
pub mod vertices;

/// The ABI this crate consumes, re-exported so a caller pins one version of it with this crate.
///
/// A consumer linking a different `tessella-capture-abi` than the mirror it drives would read the
/// right bytes at the wrong offsets, which is the failure this re-export exists to make awkward.
pub use tessella_capture_abi;
