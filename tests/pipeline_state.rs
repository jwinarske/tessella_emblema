// SPDX-License-Identifier: BSD-2-Clause
//! The pipeline state a content draw gets, and why each piece is what it is.
//!
//! # What this is for
//!
//! Pipeline state is baked, so a wrong field is wrong for every draw that pipeline ever serves, and
//! most of these fail by drawing rather than by failing. The cases here are the ones where a
//! plausible alternative is silently wrong:
//!
//! - depth enabled on a view whose attachment has no depth, which reads an attachment that is not
//!   there;
//! - a stencil op that writes, which would have a content draw erase the tile mask it is testing
//!   against;
//! - an alpha factor that premultiplies twice, which darkens every blended edge in the map by an
//!   amount nobody would call a bug;
//! - a baked viewport, which quietly ties one pipeline to one target size.
//!
//! None of these needs a GPU to check, and the bench that does need one cannot check any of them --
//! a pipeline with all four wrong builds perfectly well.

use ash::vk;
use tessella_emblema::device::Attachment;
use tessella_emblema::pipelines;

/// Depth is tested and written only where the view keeps depth.
///
/// A view of flat layers can take a stencil-only format, which on a tiler is less to write back. A
/// pipeline that enabled the depth test against such an attachment would be reading one the frame
/// does not have.
#[test]
fn depth_is_enabled_only_where_there_is_depth() {
    let flat = pipelines::depth_stencil(Attachment::StencilOnly);
    assert_eq!(flat.depth_test_enable, vk::FALSE);
    assert_eq!(flat.depth_write_enable, vk::FALSE);

    let deep = pipelines::depth_stencil(Attachment::DepthStencil);
    assert_eq!(deep.depth_test_enable, vk::TRUE);
    assert_eq!(deep.depth_write_enable, vk::TRUE);
}

/// The stencil test is on for both, because the tile clip masks are what it is for.
///
/// §2.2 draws a mask quad per tile and every content draw tests against it, so a view with no depth
/// still needs stencil -- which is why `Attachment::StencilOnly` exists at all rather than "no
/// attachment".
#[test]
fn the_stencil_test_is_always_on() {
    for attachment in [Attachment::StencilOnly, Attachment::DepthStencil] {
        let state = pipelines::depth_stencil(attachment);
        assert_eq!(
            state.stencil_test_enable,
            vk::TRUE,
            "{attachment:?} must still test the tile mask"
        );
    }
}

/// A content draw reads the tile mask and never writes it.
///
/// The failure this prevents is ordering-dependent and therefore nasty: a content draw that wrote
/// the stencil would erase the mask for every later draw in the same tile, so the first layer would
/// clip correctly and the rest would not -- which looks like a layer-ordering bug, not a stencil bug.
#[test]
fn a_content_draw_does_not_write_the_stencil() {
    for attachment in [Attachment::StencilOnly, Attachment::DepthStencil] {
        let state = pipelines::depth_stencil(attachment);
        for (side, face) in [("front", state.front), ("back", state.back)] {
            assert_eq!(face.write_mask, 0, "{side} writes the mask");
            // The compare mask is dynamic now, set per draw from the tile's `read_mask`. See
            // `tests/stencil_masks.rs` for why a baked one cannot be right.
            assert_eq!(face.compare_mask, 0, "{side} bakes a compare mask");
            assert_eq!(face.fail_op, vk::StencilOp::KEEP, "{side} fail");
            assert_eq!(face.pass_op, vk::StencilOp::KEEP, "{side} pass");
            assert_eq!(face.depth_fail_op, vk::StencilOp::KEEP, "{side} depth fail");
            assert_eq!(
                face.compare_op,
                vk::CompareOp::EQUAL,
                "{side} must test equal to the mask's reference"
            );
        }
    }
}

/// The alpha factor does not premultiply a second time.
///
/// `SRC_ALPHA` on color and `ONE` on alpha is premultiplied-correct compositing of a straight-alpha
/// source. `SRC_ALPHA` on both -- the symmetric-looking choice -- squares the alpha, which darkens
/// every blended edge in the map by an amount that reads as a style difference rather than a defect.
#[test]
fn the_alpha_factor_does_not_premultiply_twice() {
    let state = pipelines::blend();
    assert_eq!(state.blend_enable, vk::TRUE);
    assert_eq!(state.src_color_blend_factor, vk::BlendFactor::SRC_ALPHA);
    assert_eq!(
        state.dst_color_blend_factor,
        vk::BlendFactor::ONE_MINUS_SRC_ALPHA
    );
    assert_eq!(
        state.src_alpha_blend_factor,
        vk::BlendFactor::ONE,
        "the alpha channel must not be scaled by alpha again"
    );
    assert_eq!(
        state.dst_alpha_blend_factor,
        vk::BlendFactor::ONE_MINUS_SRC_ALPHA
    );
    assert_eq!(state.color_write_mask, vk::ColorComponentFlags::RGBA);
}

/// Nothing is culled.
///
/// A fill's triangles come from an earcut tessellation and a line's from a stroker; neither promises
/// a winding. mbgl does not cull either. Culling the wrong way is a layer that vanishes at some
/// zooms and not others.
#[test]
fn nothing_is_culled() {
    let state = pipelines::rasterization();
    assert_eq!(state.cull_mode, vk::CullModeFlags::NONE);
    assert_eq!(state.polygon_mode, vk::PolygonMode::FILL);
    assert!(
        (state.line_width - 1.0).abs() < f32::EPSILON,
        "a line width other than one needs the wideLines feature"
    );
}

/// The targets are formats, and the depth and stencil formats are the same attachment.
///
/// These are packed depth-stencil formats, so naming the format twice is naming one image twice --
/// which is what dynamic rendering wants, and is why there is no separate stencil format field to
/// get wrong.
#[test]
fn the_targets_are_formats_not_handles() {
    let targets = pipelines::Targets {
        color: vk::Format::B8G8R8A8_UNORM,
        depth_stencil: vk::Format::D24_UNORM_S8_UINT,
        attachment: Attachment::DepthStencil,
    };
    assert_eq!(targets.depth_stencil, vk::Format::D24_UNORM_S8_UINT);
    // Copy, so a frame can hold one per view without borrowing anything per-image.
    let other = targets;
    assert_eq!(other, targets);
}

/// Dynamic rendering is required, not preferred.
#[test]
fn dynamic_rendering_is_required() {
    use tessella_emblema::device::{self, Unsupported};
    assert_eq!(device::check_dynamic_rendering(true), Ok(()));
    assert_eq!(
        device::check_dynamic_rendering(false),
        Err(Unsupported::NoDynamicRendering),
        "a device without it cannot run this pass at all"
    );
}
