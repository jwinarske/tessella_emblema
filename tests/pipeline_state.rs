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
use tessella_capture_abi::envelope::Topology;
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
        let flat = pipelines::depth_stencil(
            Attachment::StencilOnly,
            depth,
            pipelines::Enabled::ALL,
            pipelines::Stencil::Clip,
        );
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

    let opaque = pipelines::depth_stencil(
        Attachment::DepthStencil,
        pipelines::Depth::Write,
        pipelines::Enabled::ALL,
        pipelines::Stencil::Clip,
    );
    assert_eq!(opaque.depth_test_enable, vk::TRUE);
    assert_eq!(
        opaque.depth_write_enable,
        vk::TRUE,
        "the opaque pass writes, which is what rejects the layers behind it"
    );

    let translucent = pipelines::depth_stencil(
        Attachment::DepthStencil,
        pipelines::Depth::TestOnly,
        pipelines::Enabled::ALL,
        pipelines::Stencil::Clip,
    );
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
    let flat = 0;
    assert_eq!(
        pipelines::Depth::of(RenderPass::OPAQUE.bits(), flat),
        pipelines::Depth::Write
    );
    assert_eq!(
        pipelines::Depth::of(RenderPass::TRANSLUCENT.bits(), flat),
        pipelines::Depth::TestOnly
    );
    assert_eq!(
        pipelines::Depth::of(0, flat),
        pipelines::Depth::TestOnly,
        "no bit set is not the opaque pass"
    );
    assert_eq!(
        pipelines::Depth::of(0b1000_0000, flat),
        pipelines::Depth::TestOnly,
        "nor is a bit this build does not know"
    );
}

/// A 3D drawable writes depth in either pass, which is the whole of the extrusions' two-pass scheme.
///
/// mbgl's Vulkan backend hands every 3D drawable in a layer group `depthModeFor3D()`
/// (`vulkan/tile_layer_group.cpp:80`), and that mode is `{LessEqual, ReadWrite}` with the mask
/// hardcoded -- where `depthModeForSublayer` takes the caller's mask and returns
/// `DepthMode::disabled()` below the opaque cutoff.
///
/// The consequence is not subtle. A translucent extrusion is built as *two* drawables, a depth-only
/// one and a color one (`render_fill_extrusion_layer.cpp:283`), and both are in the translucent
/// pass -- so a depth taken from the pass alone makes the first a draw that changes nothing and
/// leaves every overlapping wall of a building to blend again over the one behind it. The captured
/// Berlin frame has that pair twice over, which is where it was measured.
#[test]
fn a_3d_drawable_writes_depth_in_either_pass() {
    use tessella_capture_abi::RenderPass;
    use tessella_capture_abi::envelope::DrawFlags;

    for pass in [
        RenderPass::OPAQUE.bits(),
        RenderPass::TRANSLUCENT.bits(),
        0,
        0b1000_0000,
    ] {
        assert_eq!(
            pipelines::Depth::of(pass, DrawFlags::IS_3D.bits()),
            pipelines::Depth::Write,
            "pass {pass:#010b} with IS_3D"
        );
    }

    // And the bit is `IS_3D` and not another: the captured frame's extrusions carry four of the
    // five flags between them, and only this one moves the write.
    for other in [
        DrawFlags::ENABLE_DEPTH,
        DrawFlags::ENABLE_STENCIL,
        DrawFlags::ENABLE_COLOR,
        DrawFlags::ON_TERRAIN,
    ] {
        assert_eq!(
            pipelines::Depth::of(RenderPass::TRANSLUCENT.bits(), other.bits()),
            pipelines::Depth::TestOnly,
            "{other:?} is not what makes a drawable 3D"
        );
    }
}

/// The stencil test is on for both, because the tile clip masks are what it is for.
///
/// §2.2 draws a mask quad per tile and every content draw tests against it, so a view with no depth
/// still needs stencil -- which is why `Attachment::StencilOnly` exists at all rather than "no
/// attachment".
#[test]
fn the_stencil_test_is_always_on() {
    for attachment in [Attachment::StencilOnly, Attachment::DepthStencil] {
        let state = pipelines::depth_stencil(
            attachment,
            pipelines::Depth::Write,
            pipelines::Enabled::ALL,
            pipelines::Stencil::Clip,
        );
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
        let state = pipelines::depth_stencil(
            attachment,
            pipelines::Depth::Write,
            pipelines::Enabled::ALL,
            pipelines::Stencil::Clip,
        );
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
        stencil: pipelines::Stencil::Clip,
        shader: BuiltIn::FillShader as i32,
        surface: Surface::Plane,
        topology: Topology::Triangles,
        permutation: 0,
        layout: Vec::new(),
        blend,
        depth: pipelines::Depth::Write,
        enabled: pipelines::Enabled::ALL,
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

/// A drawable that did not ask for a test does not get it, whatever the view and the pass allow.
///
/// Three of `DrawFlags`'s bits are render state the producer sets per drawable, and this crate read
/// none of them until it had a frame to draw: the depth test was the view's attachment alone, the
/// stencil test was baked on, and the color write mask was baked to RGBA. The captured Berlin frame
/// carries three different values across its twelve batches, so all three arms are reachable.
///
/// What each one costs when it is baked on instead: a drawable covering the viewport is held
/// against whichever tile's mask last reached the texel, and a draw that is there for its depth
/// paints its geometry over the layers under it.
#[test]
fn a_drawable_gets_only_the_tests_it_asked_for() {
    let asked = |depth| pipelines::Enabled { depth, color: true };
    let scheme = |stencil: bool| {
        if stencil {
            pipelines::Stencil::Clip
        } else {
            pipelines::Stencil::Off
        }
    };

    let none = pipelines::depth_stencil(
        Attachment::DepthStencil,
        pipelines::Depth::Write,
        asked(false),
        scheme(false),
    );
    assert_eq!(
        none.depth_test_enable,
        vk::FALSE,
        "`ENABLE_DEPTH` is clear, which is mbgl's `DepthMode::disabled()`"
    );
    assert_eq!(
        none.depth_write_enable,
        vk::FALSE,
        "a draw that does not test depth does not write it either"
    );
    assert_eq!(
        none.stencil_test_enable,
        vk::FALSE,
        "`ENABLE_STENCIL` is clear, and such a drawable has no mask of its own"
    );

    // And each bit on its own, so neither gates the other.
    let depth_only = pipelines::depth_stencil(
        Attachment::DepthStencil,
        pipelines::Depth::Write,
        asked(true),
        scheme(false),
    );
    assert_eq!(depth_only.depth_test_enable, vk::TRUE);
    assert_eq!(depth_only.stencil_test_enable, vk::FALSE);

    let stencil_only = pipelines::depth_stencil(
        Attachment::DepthStencil,
        pipelines::Depth::Write,
        asked(false),
        scheme(true),
    );
    assert_eq!(stencil_only.depth_test_enable, vk::FALSE);
    assert_eq!(stencil_only.depth_write_enable, vk::FALSE);
    assert_eq!(stencil_only.stencil_test_enable, vk::TRUE);

    // The view still has the last word on depth: asking for a test an attachment cannot serve does
    // not produce one.
    let flat = pipelines::depth_stencil(
        Attachment::StencilOnly,
        pipelines::Depth::Write,
        pipelines::Enabled::ALL,
        pipelines::Stencil::Clip,
    );
    assert_eq!(flat.depth_test_enable, vk::FALSE);
}

/// A draw that asked not to write color writes none of it, whatever it composites with.
///
/// mbgl's fourth `ColorMode` is a mask of all-false rather than a blend function, so this is the one
/// of the three that is not a test: the pipeline still blends and still runs the fragment stage, and
/// nothing it returns reaches the attachment.
#[test]
fn a_draw_that_writes_no_color_has_an_empty_mask() {
    for blend in [
        pipelines::Blend::Unblended,
        pipelines::Blend::Alpha,
        pipelines::Blend::Additive,
    ] {
        let written = pipelines::color_attachment(blend, pipelines::Enabled::ALL);
        assert_eq!(
            written.color_write_mask,
            vk::ColorComponentFlags::RGBA,
            "{blend:?} writes every channel"
        );

        let masked = pipelines::color_attachment(
            blend,
            pipelines::Enabled {
                depth: true,
                color: false,
            },
        );
        assert_eq!(
            masked.color_write_mask,
            vk::ColorComponentFlags::empty(),
            "{blend:?} with `ENABLE_COLOR` clear still wrote a channel"
        );
        // And it is the same state the masks draw with, which is this crate's one spelling of
        // mbgl's `disabled()`. Field by field, because ash's struct is not `PartialEq`.
        assert_eq!(
            masked.color_write_mask,
            pipelines::no_color().color_write_mask
        );
        assert_eq!(masked.blend_enable, pipelines::no_color().blend_enable);
    }
}

/// The state a drawable asked for is part of the key, so one family can have two pipelines.
///
/// The captured frame's extrusion layer is drawn by two families twice over: once with no color for
/// the depth it leaves behind, and once with color. Keyed without this, the second draw of each
/// would be handed the first's pipeline and either paint what should be invisible or leave invisible
/// what should be painted.
///
/// The stencil scheme is in the key for the same reason and is checked here with them: all three of
/// it are baked state, and two of them differ only in the compare op -- a key that could not tell
/// `Clip` from `DrawOnce` would hand a building its tile's clip mask or a fill the draw-once's
/// `NOT_EQUAL`, and the second of those draws nothing at all.
#[test]
fn the_asked_for_state_is_part_of_the_key() {
    use tessella_capture_abi::envelope::DrawFlags;
    use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
    use tessella_emblema::surface::Surface;

    let of = |enabled, stencil| pipelines::Key {
        shader: BuiltIn::FillShader as i32,
        surface: Surface::Plane,
        topology: Topology::Triangles,
        permutation: 0,
        layout: Vec::new(),
        blend: pipelines::Blend::Alpha,
        depth: pipelines::Depth::Write,
        enabled,
        stencil,
    };
    let all = of(pipelines::Enabled::ALL, pipelines::Stencil::Clip);
    for one in [
        pipelines::Enabled {
            color: false,
            ..pipelines::Enabled::ALL
        },
        pipelines::Enabled {
            depth: false,
            ..pipelines::Enabled::ALL
        },
    ] {
        assert_ne!(
            of(one, pipelines::Stencil::Clip),
            all,
            "{one:?} and both bits must be two keys"
        );
    }
    for scheme in [pipelines::Stencil::Off, pipelines::Stencil::DrawOnce] {
        assert_ne!(
            of(pipelines::Enabled::ALL, scheme),
            all,
            "{scheme:?} and the clip mask must be two keys"
        );
    }
    assert_eq!(
        of(
            pipelines::Enabled::of(DrawFlags::ENABLE_DEPTH.bits() | DrawFlags::ENABLE_COLOR.bits()),
            pipelines::Stencil::Clip
        ),
        all,
        "and the two bits together are what `Enabled::ALL` is"
    );
}

/// `Enabled::of` reads its two bits out of the byte, and reads no others.
///
/// The byte carries five. `ON_TERRAIN` is a surface, and `IS_3D` and `ENABLE_STENCIL` are
/// [`pipelines::Stencil`]'s -- which is the pair that decides the stencil side and is checked in
/// [`the_stencil_scheme_is_the_two_bits_and_the_frames_value`]. A reading that took the wrong bit
/// would turn a test off wherever a style has terrain, or write no color wherever an extrusion is
/// drawn -- both of which draw rather than fail.
#[test]
fn the_state_is_the_two_bits_and_no_others() {
    use tessella_capture_abi::envelope::DrawFlags;

    for byte in 0..=u8::MAX {
        assert_eq!(
            pipelines::Enabled::of(byte),
            pipelines::Enabled {
                depth: byte & DrawFlags::ENABLE_DEPTH.bits() != 0,
                color: byte & DrawFlags::ENABLE_COLOR.bits() != 0,
            },
            "{byte:#010b}"
        );
    }

    // And the other three bits move nothing, whichever way the two are set.
    for byte in 0..=u8::MAX {
        for other in [
            DrawFlags::IS_3D,
            DrawFlags::ON_TERRAIN,
            DrawFlags::ENABLE_STENCIL,
        ] {
            assert_eq!(
                pipelines::Enabled::of(byte),
                pipelines::Enabled::of(byte | other.bits()),
                "{other:?} changed the state of {byte:#010b}"
            );
        }
    }
}

/// `Stencil::of` reads the stencil bit and `IS_3D`, and the frame's answer for a 3D run.
///
/// The three states and nothing else, over every byte: a 2D drawable that asked gets its tile's
/// clip mask, a 3D one that asked gets the draw-once where the frame had a value for its run and
/// *nothing* where it did not, and a drawable that did not ask gets nothing whatever else it
/// carries.
///
/// The last of those is the one worth a test of its own. A 3D run with no value must not fall back
/// to [`pipelines::Stencil::Clip`]: a building's geometry runs past the tile that owns its
/// footprint and the clip cuts the overhang off, which nothing else paints.
#[test]
fn the_stencil_scheme_is_the_two_bits_and_the_frames_value() {
    use tessella_capture_abi::envelope::DrawFlags;
    use tessella_emblema::pipelines::Stencil;

    for byte in 0..=u8::MAX {
        let asked = byte & DrawFlags::ENABLE_STENCIL.bits() != 0;
        let three_d = byte & DrawFlags::IS_3D.bits() != 0;
        for value in [None, Some(0x80)] {
            // Written as the three questions in the order `Stencil::of` asks them, rather than as
            // four arms: two of those arms are the same answer for different reasons -- a drawable
            // that asked for nothing, and a 3D run the frame had nothing to give.
            let want = if !asked {
                Stencil::Off
            } else if !three_d {
                Stencil::Clip
            } else if value.is_some() {
                Stencil::DrawOnce
            } else {
                Stencil::Off
            };
            assert_eq!(
                Stencil::of(byte, value),
                want,
                "{byte:#010b} with {value:?}"
            );
        }
    }

    // And the bits that are not these two move nothing.
    for byte in 0..=u8::MAX {
        for other in [
            DrawFlags::ENABLE_DEPTH,
            DrawFlags::ENABLE_COLOR,
            DrawFlags::ON_TERRAIN,
        ] {
            assert_eq!(
                Stencil::of(byte, Some(0x80)),
                Stencil::of(byte | other.bits(), Some(0x80)),
                "{other:?} changed the scheme of {byte:#010b}"
            );
        }
    }
}
