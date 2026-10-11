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

/// Depth is tested only where the view keeps depth, and written only by the opaque pass.
///
/// A view of flat layers can take a stencil-only format. A pipeline that enabled the depth test
/// against such an attachment would be reading one the frame does not have.
///
/// The write is the other half, and §11.7's: "opaque layers front-to-back with depth writes per
/// `pass`". The translucent pass is painted back to front and must not occlude what follows it, so
/// it tests without writing -- and the two are a `Key` apart, because a cache keyed without the
/// pass would hand one the other's pipeline.
#[test]
fn depth_is_tested_where_it_is_kept_and_written_by_the_opaque_pass() {
    for depth in [pipelines::Depth::Write, pipelines::Depth::TestOnly] {
        let flat = pipelines::depth_stencil(Attachment::StencilOnly, depth);
        assert_eq!(
            flat.depth_test_enable,
            vk::FALSE,
            "{depth:?} has no depth to test against a stencil-only attachment"
        );
        assert_eq!(
            flat.depth_write_enable,
            vk::FALSE,
            "{depth:?} has no depth to write there either"
        );
    }

    let opaque = pipelines::depth_stencil(Attachment::DepthStencil, pipelines::Depth::Write);
    assert_eq!(opaque.depth_test_enable, vk::TRUE);
    assert_eq!(
        opaque.depth_write_enable,
        vk::TRUE,
        "the opaque pass writes, which is what rejects the layers behind it"
    );

    let translucent =
        pipelines::depth_stencil(Attachment::DepthStencil, pipelines::Depth::TestOnly);
    assert_eq!(
        translucent.depth_test_enable,
        vk::TRUE,
        "the translucent pass still tests, so an opaque surface in front of it hides it"
    );
    assert_eq!(
        translucent.depth_write_enable,
        vk::FALSE,
        "and does not write, so it does not occlude the layer painted in front of it"
    );
}

/// The pass a bit names decides the depth write, and an unknown pass does not get one.
///
/// `RenderPass` is a bitflag and `DrawOrder::resolve` expands a drawable marked both into one entry
/// per pass, so an entry carries one bit. Anything that is not the opaque bit is translucent: a
/// pass this build does not know must not be handed depth writes for not being recognized.
#[test]
fn only_the_opaque_bit_writes_depth() {
    use tessella_capture_abi::RenderPass;
    assert_eq!(
        pipelines::Depth::of(RenderPass::OPAQUE.bits()),
        pipelines::Depth::Write
    );
    assert_eq!(
        pipelines::Depth::of(RenderPass::TRANSLUCENT.bits()),
        pipelines::Depth::TestOnly
    );
    assert_eq!(
        pipelines::Depth::of(0),
        pipelines::Depth::TestOnly,
        "no bit set is not the opaque pass"
    );
    assert_eq!(
        pipelines::Depth::of(0b1000_0000),
        pipelines::Depth::TestOnly,
        "nor is a bit this build does not know"
    );
}

/// The stencil test is on for both, because the tile clip masks are what it is for.
///
/// §2.2 draws a mask quad per tile and every content draw tests against it, so a view with no depth
/// still needs stencil -- which is why `Attachment::StencilOnly` exists at all rather than "no
/// attachment".
#[test]
fn the_stencil_test_is_always_on() {
    for attachment in [Attachment::StencilOnly, Attachment::DepthStencil] {
        let state = pipelines::depth_stencil(attachment, pipelines::Depth::Write);
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
        let state = pipelines::depth_stencil(attachment, pipelines::Depth::Write);
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

/// Alpha blending is premultiplied, which is what the bodies return.
///
/// mbgl's `alphaBlended` is `Add{One, OneMinusSrcAlpha}`, and `src = ONE` is the whole point: its
/// fragment stages return premultiplied color -- `out_color = color * opacity` scales the alpha
/// channel along with the others -- and this crate's bodies are transcribed from those and do the
/// same. One of them says so: "the sheet is premultiplied, so the opacity and the fade scale it
/// directly."
///
/// `SRC_ALPHA` is the symmetric-looking choice and it multiplies by alpha a second time, darkening
/// every blended edge by an amount that reads as a style difference rather than a defect. This test
/// asserted `SRC_ALPHA` until mbgl's own `color_mode.hpp` was read.
#[test]
fn alpha_blending_is_premultiplied() {
    let state = pipelines::Blend::Alpha.attachment();
    assert_eq!(state.blend_enable, vk::TRUE);
    assert_eq!(
        state.src_color_blend_factor,
        vk::BlendFactor::ONE,
        "the source is already premultiplied; scaling it by alpha again darkens every edge"
    );
    assert_eq!(
        state.dst_color_blend_factor,
        vk::BlendFactor::ONE_MINUS_SRC_ALPHA
    );
    assert_eq!(state.src_alpha_blend_factor, vk::BlendFactor::ONE);
    assert_eq!(
        state.dst_alpha_blend_factor,
        vk::BlendFactor::ONE_MINUS_SRC_ALPHA
    );
    assert_eq!(state.color_write_mask, vk::ColorComponentFlags::RGBA);
}

/// Unblended writes the source through, which is what a readback oracle needs.
///
/// mbgl's `unblended`: a `Replace` function. A pixel read back is then what the fragment stage
/// returned rather than what it returned composited over whatever the target held, which is the
/// only way a derived-by-hand expected value means anything.
#[test]
fn unblended_replaces_the_target() {
    let state = pipelines::Blend::Unblended.attachment();
    assert_eq!(state.blend_enable, vk::FALSE);
    assert_eq!(
        state.color_write_mask,
        vk::ColorComponentFlags::RGBA,
        "unblended still writes every channel; `disabled` is the one that writes none"
    );
}

/// Additive accumulates, which is what a heatmap does.
#[test]
fn additive_accumulates() {
    let state = pipelines::Blend::Additive.attachment();
    assert_eq!(state.blend_enable, vk::TRUE);
    assert_eq!(state.src_color_blend_factor, vk::BlendFactor::ONE);
    assert_eq!(
        state.dst_color_blend_factor,
        vk::BlendFactor::ONE,
        "every contribution brightens the same texel"
    );
}

/// The blend mode is in the pipeline key, because it is pipeline state.
///
/// Two layers of one family and permutation can pick different modes -- a heatmap accumulates where
/// a fill composites -- so a cache keyed without it would hand the second the first's pipeline.
///
/// This guards the field's presence and no more: the equality is derived, so it holds as long as the
/// field is there. What makes a pipeline's blend agree with its key is structural rather than
/// tested -- `build` reads `key.blend` and takes no blend argument, so there is no way to build one
/// that disagrees.
#[test]
fn the_blend_mode_is_part_of_the_key() {
    use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
    use tessella_emblema::surface::Surface;

    let of = |blend| pipelines::Key {
        shader: BuiltIn::FillShader,
        surface: Surface::Plane,
        permutation: 0,
        layout: Vec::new(),
        blend,
        depth: pipelines::Depth::Write,
    };
    assert_ne!(
        of(pipelines::Blend::Alpha),
        of(pipelines::Blend::Additive),
        "two modes must be two keys"
    );
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
