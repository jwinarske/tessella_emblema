// SPDX-License-Identifier: BSD-2-Clause
//! What distinguishes one pipeline from another.
//!
//! A `VkPipeline` bakes in its vertex input state, so two drawables of the same family, surface and
//! permutation still need different pipelines if their strides or offsets differ. That is the part
//! a cache key is easy to get wrong: keyed on the family and permutation alone it would hand back a
//! pipeline whose `VkVertexInputBindingDescription` has the wrong stride, and the draw would read
//! every vertex from the wrong place without failing.
//!
//! # Why the layout is carried rather than hashed
//!
//! A `u64` digest of the layout would make the key small and a collision would silently bind the
//! wrong pipeline — the same class of defect, found by nobody, at a rate nobody can reproduce. The
//! key holds the slots instead and compares them. There are at most ten per family.
//!
//! # What is not in the key yet
//!
//! The per-binding input rate. An instanced family binds some attributes per instance rather than
//! per vertex, and that is pipeline state too. [`crate::vertices::plan`] reads a drawable's `attrs`
//! and not its `instance_attrs`, so no instanced family reaches here at all; adding them is what
//! adds the rate. Noted rather than guessed, because a rate field that nothing sets is a field that
//! is wrong the first time something does.

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;

use crate::surface::Surface;
use crate::vertices::Plan;

/// One attribute's share of the pipeline's vertex input state.
///
/// The buffer a binding reads is not here: buffers are bound at draw time, so two drawables whose
/// bytes live in different slabs share a pipeline. Nor is `vertex_offset`, which is a draw
/// parameter. Only what `VkPipelineVertexInputStateCreateInfo` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot {
    /// The `@location` the module declares, which is also the binding number.
    pub slot: u32,
    /// The format bound, from the family's declared type.
    pub format: vk::Format,
    /// Byte offset of this attribute within a vertex.
    pub offset: u32,
    /// Bytes between consecutive vertices.
    pub stride: u32,
}

/// What a pipeline is cached by.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    /// The shader family.
    pub shader: BuiltIn,
    /// Which surface its module was assembled against.
    pub surface: Surface,
    /// The data-driven variant, which decides the module's specialization constants.
    pub permutation: u64,
    /// The vertex input state, in slot order.
    pub layout: Vec<Slot>,
}

/// The key for a drawable whose input has been planned.
///
/// Built from the plan rather than from the family's table, because the table says what a module
/// *declares* and the plan says what this drawable *binds*. They differ whenever a data-driven
/// attribute is constant for this permutation and no descriptor arrived for it, and that difference
/// is a different pipeline.
#[must_use]
pub fn key(shader: BuiltIn, surface: Surface, permutation: u64, plan: &Plan) -> Key {
    Key {
        shader,
        surface,
        permutation,
        layout: plan
            .bound
            .iter()
            .map(|bound| Slot {
                slot: bound.slot,
                format: bound.format,
                offset: bound.offset,
                stride: bound.stride,
            })
            .collect(),
    }
}
