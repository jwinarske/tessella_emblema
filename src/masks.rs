// SPDX-License-Identifier: BSD-2-Clause
//! The clip mask's own shader.
//!
//! # Where the numbering went
//!
//! Not here. `tessella_consume::stencil::partition` assigns each tile its `Assignment` --
//! `{ value, read_mask, write_mask }` -- and that is a renderer-agnostic decision this crate's own
//! `Cargo.toml` says belongs there:
//!
//! > The renderer-agnostic half: reading the ring, joining, collapsing, the stencil partition.
//! > Shared with every other consumer rather than written again here, which is what keeps two
//! > implementations evidence about the drawing rather than about the arithmetic.
//!
//! An earlier version of this module had a running counter transcribed from mbgl's
//! `PaintParameters`, which was accurate about mbgl and beside the point: `stencil` spends the same
//! eight bits on a *field per zoom* instead, so a finer tile's mask clears an ancestor's field
//! rather than needing a value of its own. A value per tile is what that module calls "the scheme
//! this replaces", and it is the fallback it falls back *to*.
//!
//! What is left here is the part that is about this renderer: the quad, and the state that draws
//! it. The masks a draw sets come from the `Assignment` -- see [`crate::pipelines::MASK_DYNAMIC`].

/// The mask's own shader, written here rather than assembled from a family table.
///
/// [`crate::families`] excludes `ClippingMaskProgram` deliberately -- "the stencil mask, which the
/// consumer draws from the partition rather than from a family" -- so there is no attribute table,
/// no props block and no generated declarations for it. This is the whole of it.
///
/// # What it draws
///
/// A full-tile quad from `vertex_index` alone, with no vertex buffer. §2.2 says the mask is a
/// full-tile quad carrying the matrix the producer sent, and a quad whose corners are derived from
/// the index needs nothing bound: six indices, two triangles, the unit square. A vertex buffer here
/// would be four corners the producer would have to send per tile and this would have to store.
///
/// The matrix comes from a storage buffer indexed by `instance_index`, which is the same
/// arrangement the families use for their drawable blocks -- `draws` puts the slot in
/// `firstInstance` for exactly this. One entry per tile of the pass.
///
/// # Why the fragment stage returns anything at all
///
/// It has to have one: a pipeline with no fragment stage cannot write a stencil attachment in a
/// rendering scope that has one. What it returns never lands, because the mask pipeline's color
/// write mask is empty -- see [`crate::pipelines::no_color`]. Returning zero rather than something
/// recognizable is deliberate: if the write mask were ever wrong, a transparent black over the
/// frame is a subtler failure than magenta, and the bench checks the color is untouched rather than
/// trusting the mask.
pub const BODY: &str = r"
struct MaskUbo {
    matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> mask_ubo: array<MaskUbo>;

// The unit square's corners, as two triangles. Indexed rather than bound: the quad is the same for
// every tile and only its matrix differs.
fn corner(at: u32) -> vec2<f32> {
    switch (at) {
        case 0u, 3u: { return vec2<f32>(0.0, 0.0); }
        case 1u: { return vec2<f32>(1.0, 0.0); }
        case 2u, 4u: { return vec2<f32>(1.0, 1.0); }
        default: { return vec2<f32>(0.0, 1.0); }
    }
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) tile: u32,
) -> @builtin(position) vec4<f32> {
    let at = corner(vertex % 6u);
    // The tile's own extent, which the matrix expects in tile units.
    let position = vec4<f32>(at * 8192.0, 0.0, 1.0);
    return mask_ubo[tile].matrix * position;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
";

/// Indices a mask quad is drawn with.
///
/// Six, two triangles, and the vertex stage reads nothing but the index -- so this is a
/// `vkCmdDraw` of six vertices rather than an indexed draw, and there is no index buffer to bind.
pub const VERTICES: u32 = 6;
