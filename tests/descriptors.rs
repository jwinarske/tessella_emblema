// SPDX-License-Identifier: BSD-2-Clause
//! What a family's descriptor set holds, checked against what its modules declare.
//!
//! # What this is for
//!
//! `pipelines::bindings` and `shaders::module` have to agree. One writes the `@group(0) @binding(n)`
//! lines into the WGSL; the other describes the same set to `vkCreateDescriptorSetLayout`. Nothing
//! connects them but the rule they both follow, and `shaders::module` says what makes that rule
//! treacherous:
//!
//! > Counted rather than computed from the index, because the count differs by family and by
//! > surface and an index formula would be arithmetic no test could distinguish from a wrong one.
//!
//! So this does not check `bindings` against a table. It assembles the real module for every family
//! and surface pair, reads the bindings out of the text, and checks the two against each other.
//! A table would be a third statement of the rule and would agree with whichever of the two I wrote
//! it from.
//!
//! # What would be caught
//!
//! A count off by one, which is every later binding one place out. That binds one block's bytes
//! where another's were meant, and the shapes usually fit -- a uniform block is a block -- so it
//! draws. The type cases are the loud half: a layout offering `COMBINED_IMAGE_SAMPLER` where the
//! module declares a `texture_2d` and a `sampler` separately is rejected by
//! `vkCreateGraphicsPipelines`, which is a failure somebody sees.

use tessella_emblema::families;
use tessella_emblema::pipelines::{self, Binding, Kind};
use tessella_emblema::shaders;
use tessella_emblema::surface::Surface;

/// Every `@group(0) @binding(n)` the assembled text declares, in binding order, with its kind read
/// from the declaration itself.
fn declared(text: &str) -> Vec<Binding> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("@group(0) @binding(") else {
            continue;
        };
        let Some((number, tail)) = rest.split_once(')') else {
            continue;
        };
        let Ok(binding) = number.parse::<u32>() else {
            continue;
        };
        // Read from what is declared rather than from the binding's position, so a module that
        // declared a sampler where a block belongs is a disagreement rather than a relabeling.
        let kind = if tail.contains("var<storage, read>") {
            Kind::StorageBuffer
        } else if tail.contains(": texture_2d<f32>") {
            Kind::SampledImage
        } else if tail.contains(": sampler") {
            Kind::Sampler
        } else {
            panic!("a binding of an unrecognized kind: {line}");
        };
        // The slot is not read back from the text -- a module declares a binding number, not the
        // slot the bytes arrive at -- so this leaves it out and `tests/block_slots.rs` is where the
        // slots are checked.
        out.push(Binding {
            binding,
            kind,
            slot: None,
        });
    }
    out.sort_unstable_by_key(|b| b.binding);
    out
}

/// Assembles the module for a family and surface, or `None` where none exists.
fn assembled(family: &families::Family, surface: Surface) -> Option<String> {
    if !family.surfaces.contains(&surface) {
        return None;
    }
    Some(
        shaders::module(
            surface,
            family.blocks,
            family.attributes,
            family.textures,
            family.body,
        )
        .unwrap_or_else(|why| panic!("{} on {surface:?} does not assemble: {why:?}", family.name)),
    )
}

/// Every module's declared bindings are exactly what `bindings` describes.
///
/// The assertion the slice exists for, over every pair that assembles.
#[test]
fn every_module_declares_what_the_layout_describes() {
    let mut checked = 0;
    for family in families::ALL {
        for surface in Surface::ALL {
            let Some(text) = assembled(family, surface) else {
                continue;
            };
            // The number and the kind, which is what a module declares. The slot a block's bytes
            // arrive at is not in the text and cannot be: it is the producer's name for a buffer,
            // not the shader's for a binding. `tests/block_slots.rs` is where those are checked.
            let derived: Vec<(u32, Kind)> = pipelines::bindings(family, surface)
                .iter()
                .map(|b| (b.binding, b.kind))
                .collect();
            let from_text: Vec<(u32, Kind)> = declared(&text)
                .iter()
                .map(|b| (b.binding, b.kind))
                .collect();
            assert_eq!(
                from_text, derived,
                "{} on {surface:?}: the module and the descriptor layout disagree",
                family.name
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked, 57,
        "the crate assembles 57 modules; a different number here means the sweep drifted from \
         what tests/shaders.rs compiles"
    );
}

/// The bindings are consecutive from zero, with no gaps.
///
/// A gap is a binding the layout declares and nothing reads, which is legal and wasteful -- but it
/// is also what an off-by-one looks like from this side, so it is worth refusing outright.
#[test]
fn the_bindings_are_consecutive_from_zero() {
    for family in families::ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            let derived = pipelines::bindings(family, surface);
            for (index, binding) in derived.iter().enumerate() {
                assert_eq!(
                    binding.binding, index as u32,
                    "{} on {surface:?} has a gap at {index}",
                    family.name
                );
            }
        }
    }
}

/// Blocks come before textures, and a texture is always a pair in that order.
///
/// The shape `shaders::module` writes. A sampler before its image, or an image with no sampler,
/// would be a set the module cannot be built against.
#[test]
fn textures_follow_the_blocks_as_pairs() {
    for family in families::ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            let derived = pipelines::bindings(family, surface);
            let blocks = derived
                .iter()
                .take_while(|b| b.kind == Kind::StorageBuffer)
                .count();
            assert!(
                derived[blocks..]
                    .iter()
                    .all(|b| b.kind != Kind::StorageBuffer),
                "{} on {surface:?} has a block after a texture",
                family.name
            );
            let images = &derived[blocks..];
            assert_eq!(
                images.len() % 2,
                0,
                "{} on {surface:?} has an odd number of texture bindings",
                family.name
            );
            for pair in images.chunks(2) {
                assert_eq!(pair[0].kind, Kind::SampledImage);
                assert_eq!(pair[1].kind, Kind::Sampler);
            }
        }
    }
}

/// The surface's own block and textures land after the family's.
///
/// `Terrain` adds a block and an elevation texture; the same family on `Plane` has neither. So the
/// same family on two surfaces is two different sets, which is why the surface is in the key.
#[test]
fn a_surface_adds_its_own_descriptors_after_the_family_s() {
    let raster = families::ALL
        .iter()
        .find(|family| family.name == "raster")
        .expect("a raster family");
    let flat = pipelines::bindings(raster, Surface::Plane);
    let raised = pipelines::bindings(raster, Surface::Terrain);

    assert!(
        raised.len() > flat.len(),
        "terrain adds descriptors, so the raised set is larger: {} against {}",
        raised.len(),
        flat.len()
    );
    // The family's own blocks are still first and still blocks, so nothing it declares moved out
    // from under it.
    let flat_blocks = flat
        .iter()
        .filter(|b| b.kind == Kind::StorageBuffer)
        .count();
    let raised_blocks = raised
        .iter()
        .filter(|b| b.kind == Kind::StorageBuffer)
        .count();
    assert_eq!(
        raised_blocks,
        flat_blocks + 1,
        "terrain declares exactly one block of its own"
    );
    assert_eq!(
        raised
            .iter()
            .filter(|b| b.kind == Kind::SampledImage)
            .count(),
        flat.iter().filter(|b| b.kind == Kind::SampledImage).count() + 1,
        "and exactly one texture"
    );
}

/// A pool is sized by kind, and leaves out a kind nothing declares.
#[test]
fn a_pool_is_sized_by_what_is_declared() {
    let background = families::ALL
        .iter()
        .find(|family| family.name == "background")
        .expect("a background family");
    let bare = pipelines::bindings(background, Surface::Plane);
    assert_eq!(
        pipelines::pool_sizes(&bare),
        vec![(Kind::StorageBuffer, 2)],
        "a family with no texture must not ask a pool for image descriptors"
    );

    let patterned = families::ALL
        .iter()
        .find(|family| family.name == "background_pattern")
        .expect("a background_pattern family");
    let sizes = pipelines::pool_sizes(&pipelines::bindings(patterned, Surface::Plane));
    assert_eq!(
        sizes,
        vec![
            (Kind::StorageBuffer, 3),
            (Kind::SampledImage, 1),
            (Kind::Sampler, 1)
        ]
    );
}

/// Each kind maps to the Vulkan type the module's declaration means.
#[test]
fn each_kind_is_its_vulkan_type() {
    use ash::vk::DescriptorType as D;
    assert_eq!(Kind::StorageBuffer.descriptor_type(), D::STORAGE_BUFFER);
    assert_eq!(Kind::SampledImage.descriptor_type(), D::SAMPLED_IMAGE);
    assert_eq!(Kind::Sampler.descriptor_type(), D::SAMPLER);
    // Not the combined type, which is the mistake available here: WGSL's texture and sampler are
    // two objects and naga emits them as two descriptors.
    assert_ne!(
        Kind::SampledImage.descriptor_type(),
        D::COMBINED_IMAGE_SAMPLER
    );
}
