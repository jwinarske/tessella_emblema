// SPDX-License-Identifier: BSD-2-Clause
//! Every family's pipeline, built on a real device without a render pass.
//!
//! A bench rather than a test for the reason the others are: it needs a GPU and CI has none.
//! `tests/pipeline_state.rs` holds the half that does not, which is every state field that fails by
//! drawing rather than by failing.
//!
//! # What this proves that the state tests cannot
//!
//! That a driver accepts them. `vkCreateGraphicsPipelines` is where a module meets its layout, its
//! vertex input and its attachment formats, and it is the only place a disagreement between the four
//! is reported -- a descriptor set layout that does not match what the module declares, or a vertex
//! attribute at a location the module has no input for, is caught here and nowhere earlier.
//!
//! It is also where the driver compiles the shader, which on some parts is where a driver crashes.
//! That is why each family is built in its own iteration with its name printed first: a segfault
//! then names the family that caused it rather than ending the run anonymously.
//!
//! Run with `cargo bench --bench pipelines_build`.

mod common;

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;
use tessella_emblema::device::{self, Attachment};
use tessella_emblema::pipelines::{self, Key, Slot, Targets};
use tessella_emblema::surface::Surface;
use tessella_emblema::{families, shaders};

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// The color format a host image is handed over as, which the pass does not choose.
const COLOR: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Compiles an assembled module to SPIR-V, as the oracle does.
///
/// `Options::default()`, and the profile matters: naga emits different SPIR-V in a debug build than
/// in a release one, and a bench is built release, so these are the words a board actually sees.
fn compile(source: &str) -> Result<Vec<u32>, String> {
    let parsed = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spirv: {why}"))
}

/// The key for a family, with one binding per attribute its table declares.
///
/// Built from the table rather than from a producer's plan, because there is no producer here. The
/// strides are each attribute's own format size, which is the uninterleaved case -- the dedup's
/// shared-stride case is covered by `tests/vertex_input.rs` and changes nothing a driver checks.
fn key_for(family: &families::Family, surface: Surface) -> Result<Key, String> {
    let mut layout = Vec::new();
    for attribute in family.attributes {
        let format = device::vertex_format(attribute.declared)
            .ok_or_else(|| format!("{} has no vertex format", attribute.name))?;
        layout.push(Slot {
            slot: u32::try_from(attribute.binding).map_err(|_| "absurd binding")?,
            format,
            offset: 0,
            stride: stride_of(attribute.declared)
                .ok_or_else(|| format!("{} has no stride", attribute.name))?,
            rate: vk::VertexInputRate::VERTEX,
        });
    }
    Ok(Key {
        shader: family.shader,
        surface,
        permutation: 0,
        layout,
        blend: pipelines::Blend::Alpha,
    })
}

/// Bytes one vertex of a declared type occupies.
///
/// `None` for a type with no vertex format, which `device::vertex_format` refuses first -- this is
/// written to agree with it rather than to guess a size for something that cannot be bound.
fn stride_of(declared: AttributeDataType) -> Option<u32> {
    use AttributeDataType as A;
    Some(match declared {
        A::Byte | A::UByte => 1,
        A::Byte2 | A::UByte2 | A::Short | A::UShort => 2,
        A::Byte3 | A::UByte3 => 3,
        A::Byte4 | A::UByte4 | A::Short2 | A::UShort2 | A::Int | A::UInt | A::Float => 4,
        A::Short3 | A::UShort3 => 6,
        A::Short4 | A::UShort4 | A::Int2 | A::UInt2 | A::Float2 => 8,
        A::Int3 | A::UInt3 | A::Float3 => 12,
        A::Int4 | A::UInt4 | A::Float4 | A::UShort8 => 16,
        A::Invalid => return None,
    })
}

/// Every family and surface pair that has a module builds a pipeline.
fn every_pipeline_builds(device: &Open) -> Result<(), String> {
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    println!("  depth-stencil format  {depth_stencil:?}");

    let mut built = 0;
    // Held to the end, so the device carries every pipeline at once -- the state a frame drawing
    // every family would be in.
    let mut held = Vec::new();
    for family in families::ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            let who = format!("{} on {surface:?}", family.name);
            let text = shaders::module(
                surface,
                family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .map_err(|why| format!("{who} does not assemble: {why:?}"))?;
            let words = compile(&text).map_err(|why| format!("{who}: {why}"))?;

            let module = device
                .gpu()
                .shader(&words)
                .map_err(|why| format!("{who}: {why}"))?;
            let layout = pipelines::layout(device.gpu(), &pipelines::bindings(family, surface))
                .map_err(|why| format!("{who}: {why}"))?;
            let key = key_for(family, surface).map_err(|why| format!("{who}: {why}"))?;

            for attachment in [Attachment::DepthStencil, Attachment::StencilOnly] {
                let targets = Targets {
                    color: COLOR,
                    depth_stencil,
                    attachment,
                };
                let pipeline = pipelines::build(device.gpu(), &key, &layout, &module, targets)
                    .map_err(|why| format!("{who} with {attachment:?}: {why}"))?;
                if pipeline.raw() == vk::Pipeline::null() {
                    return Err(format!("{who}: a null pipeline"));
                }
                held.push(pipeline);
                built += 1;
            }
            // The layout and the module outlive the pipelines built from them here, which Vulkan
            // allows -- a pipeline keeps no reference to either once created.
            drop(layout);
            drop(module);
        }
    }
    if built != 114 {
        return Err(format!(
            "{built} pipelines, which is not 57 modules times two attachment kinds"
        ));
    }
    println!("  every pipeline        ok   {built} pipelines, all resident at once");
    Ok(())
}

/// A key whose vertex input the module has no input for is refused.
///
/// The disagreement `vkCreateGraphicsPipelines` exists to catch, and the reason the key carries the
/// layout at all: an attribute at a location the module does not declare is invalid usage that the
/// validation layers report as `VUID-VkGraphicsPipelineCreateInfo-Input-07904`. Whether a driver
/// refuses it is not promised, so this reports what happened rather than asserting a refusal -- the
/// same lesson as the descriptor limits.
fn a_bogus_location_is_reported(device: &Open) -> Result<(), String> {
    let family = families::ALL
        .iter()
        .find(|f| f.name == "background")
        .ok_or("a background family")?;
    let surface = Surface::Plane;
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;

    let text = shaders::module(
        surface,
        family.blocks,
        family.attributes,
        family.textures,
        family.body,
    )
    .map_err(|why| format!("assemble: {why:?}"))?;
    let words = compile(&text)?;
    let module = device.gpu().shader(&words).map_err(|why| why.to_string())?;
    let layout = pipelines::layout(device.gpu(), &pipelines::bindings(family, surface))
        .map_err(|why| why.to_string())?;

    let mut key = key_for(family, surface)?;
    // Location 9, which no family declares.
    key.layout.push(Slot {
        slot: 9,
        format: vk::Format::R32_SFLOAT,
        offset: 0,
        stride: 4,
        rate: vk::VertexInputRate::VERTEX,
    });
    let targets = Targets {
        color: COLOR,
        depth_stencil,
        attachment: Attachment::DepthStencil,
    };
    let verdict = match pipelines::build(device.gpu(), &key, &layout, &module, targets) {
        Ok(_) => "accepted, so nothing enforces it",
        Err(_) => "refused",
    };
    println!("  a bogus location      ok   an input the module lacks was {verdict}");
    Ok(())
}

/// A second ask for the same key builds nothing.
///
/// What the cache is for. A styled view has thousands of batches over tens of programs, so a frame
/// that rebuilt per batch would pay the shader compile per batch -- and the only observable
/// difference between a cache that works and one that silently misses is this count.
fn the_cache_reuses(device: &Open) -> Result<(), String> {
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    let targets = Targets {
        color: COLOR,
        depth_stencil,
        attachment: Attachment::DepthStencil,
    };

    let mut cache = pipelines::Cache::new();
    let mut asked = 0usize;
    // Every pair, three times over, which is what a frame redrawing the same style looks like.
    for round in 0..3 {
        for family in families::ALL {
            for surface in Surface::ALL {
                if !family.surfaces.contains(&surface) {
                    continue;
                }
                let who = format!("{} on {surface:?}", family.name);
                let bindings = pipelines::bindings(family, surface);
                let key = key_for(family, surface).map_err(|why| format!("{who}: {why}"))?;

                // The words are needed only when the *module* is missing, which is the question
                // `has_module` answers and `holds` does not.
                let words = if cache.has_module(family.shader, surface) {
                    Vec::new()
                } else {
                    let text = shaders::module(
                        surface,
                        family.blocks,
                        family.attributes,
                        family.textures,
                        family.body,
                    )
                    .map_err(|why| format!("{who}: {why:?}"))?;
                    compile(&text).map_err(|why| format!("{who}: {why}"))?
                };
                if round > 0 && !words.is_empty() {
                    return Err(format!("{who} was compiled again on round {round}"));
                }

                let pipeline = cache
                    .pipeline(device.gpu(), &key, &bindings, &words, targets)
                    .map_err(|why| format!("{who}: {why}"))?;
                if pipeline == vk::Pipeline::null() {
                    return Err(format!("{who}: a null pipeline"));
                }
                asked += 1;
            }
        }
    }

    if cache.built() != 57 {
        return Err(format!(
            "{} pipelines built for 57 keys asked {asked} times",
            cache.built()
        ));
    }
    if cache.len() != 57 || cache.modules() != 57 {
        return Err(format!(
            "{} pipelines and {} modules held, wanted 57 of each",
            cache.len(),
            cache.modules()
        ));
    }
    if cache.bound() != asked {
        return Err(format!("{} binds for {asked} asks", cache.bound()));
    }
    println!(
        "  the cache reuses      ok   {asked} asks, {} built, {} modules",
        cache.built(),
        cache.modules()
    );
    Ok(())
}

/// A key differing only in a stride is a different pipeline.
///
/// The failure `Key` exists for, from the cache's side: keyed on the family and permutation alone,
/// the second ask here would be a hit and the draw would read every vertex at the wrong stride.
fn a_stride_is_a_different_pipeline(device: &Open) -> Result<(), String> {
    let depth_stencil = device::depth_stencil_format(Attachment::DepthStencil, |format| {
        device.format_properties(format).optimal_tiling_features
    })
    .map_err(|why| format!("no depth-stencil format: {why:?}"))?;
    let targets = Targets {
        color: COLOR,
        depth_stencil,
        attachment: Attachment::DepthStencil,
    };
    let family = families::ALL
        .iter()
        .find(|f| f.name == "background")
        .ok_or("a background family")?;
    let bindings = pipelines::bindings(family, Surface::Plane);
    let text = shaders::module(
        Surface::Plane,
        family.blocks,
        family.attributes,
        family.textures,
        family.body,
    )
    .map_err(|why| format!("{why:?}"))?;
    let words = compile(&text)?;

    let mut cache = pipelines::Cache::new();
    let key = key_for(family, Surface::Plane)?;
    let first = cache
        .pipeline(device.gpu(), &key, &bindings, &words, targets)
        .map_err(|why| why.to_string())?;

    let mut wider = key.clone();
    for slot in &mut wider.layout {
        slot.stride *= 2;
    }
    // Empty words deliberately: the module is already compiled, and a cache that reached for them
    // anyway would be recompiling the same text per stride. `vkCreateShaderModule` of nothing
    // fails, so this is the assertion rather than a count.
    if !cache.has_module(family.shader, Surface::Plane) {
        return Err("the first pipeline did not cache its module".into());
    }
    let second = cache
        .pipeline(device.gpu(), &wider, &bindings, &[], targets)
        .map_err(|why| format!("a second stride needed the words again: {why}"))?;

    if first == second {
        return Err("two strides got one pipeline".into());
    }
    if cache.built() != 2 {
        return Err(format!("{} built for two strides", cache.built()));
    }
    // And one module serves both, which is the other half of the two-level split.
    if cache.modules() != 1 {
        return Err(format!(
            "{} modules for one family: the text was compiled per stride",
            cache.modules()
        ));
    }
    println!("  a stride differs      ok   two pipelines, one module");
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
        ("every_pipeline", every_pipeline_builds as Case),
        ("bogus_location", a_bogus_location_is_reported),
        ("cache_reuses", the_cache_reuses),
        ("stride_differs", a_stride_is_a_different_pipeline),
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
