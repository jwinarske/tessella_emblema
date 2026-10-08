// SPDX-License-Identifier: BSD-2-Clause
//! The vertex input state a pipeline key describes.
//!
//! # What this is for
//!
//! `pipelines::Key` exists because a `VkPipeline` bakes in its vertex input state, and that module
//! says what gets it wrong:
//!
//! > keyed on the family and permutation alone it would hand back a pipeline whose
//! > `VkVertexInputBindingDescription` has the wrong stride, and the draw would read every vertex
//! > from the wrong place without failing.
//!
//! `vertex_input` is the step that turns the key into those descriptions, so it is the step where a
//! stride can go astray between the plan and the pipeline.
//!
//! # What would be caught
//!
//! A binding described with another slot's stride, which reads every vertex after the first from
//! the wrong address -- and reads *something*, because a vertex buffer is bytes. The two refusals
//! are the cases Vulkan would reject anyway but less legibly: it says a binding is duplicated, not
//! which of the two strides was meant.

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
use tessella_emblema::pipelines::{self, Invalid, Key, Slot};
use tessella_emblema::surface::Surface;

const fn slot(slot: u32, format: vk::Format, offset: u32, stride: u32) -> Slot {
    Slot {
        slot,
        format,
        offset,
        stride,
        rate: vk::VertexInputRate::VERTEX,
    }
}

fn key(layout: Vec<Slot>) -> Key {
    Key {
        shader: BuiltIn::FillShader,
        surface: Surface::Plane,
        permutation: 0,
        layout,
    }
}

/// Each slot becomes one binding and one attribute, carrying its own numbers.
#[test]
fn a_slot_becomes_a_binding_and_an_attribute() {
    let found = pipelines::vertex_input(&key(vec![
        slot(0, vk::Format::R32G32_SFLOAT, 0, 8),
        slot(1, vk::Format::R8G8B8A8_UNORM, 0, 4),
    ]))
    .expect("describable");

    assert_eq!(found.bindings.len(), 2);
    assert_eq!(found.attributes.len(), 2);
    for (index, (binding, attribute)) in found.bindings.iter().zip(&found.attributes).enumerate() {
        assert_eq!(binding.binding, index as u32);
        assert_eq!(attribute.location, index as u32);
        assert_eq!(
            attribute.binding, binding.binding,
            "an attribute must read the binding of its own slot"
        );
    }
    assert_eq!(found.bindings[0].stride, 8);
    assert_eq!(found.bindings[1].stride, 4);
    assert_eq!(found.attributes[0].format, vk::Format::R32G32_SFLOAT);
    assert_eq!(found.attributes[1].format, vk::Format::R8G8B8A8_UNORM);
}

/// Each binding keeps its own slot's stride.
///
/// The failure `Key` was written for. Two slots with different strides must not end up sharing one,
/// because a binding described with the other's stride reads every vertex after the first from the
/// wrong address and reads something rather than failing.
#[test]
fn each_binding_keeps_its_own_stride() {
    let found = pipelines::vertex_input(&key(vec![
        slot(0, vk::Format::R32G32_SFLOAT, 0, 8),
        slot(1, vk::Format::R32_SFLOAT, 0, 4),
        slot(2, vk::Format::R16G16_SINT, 0, 32),
    ]))
    .expect("describable");

    let strides: Vec<u32> = found.bindings.iter().map(|b| b.stride).collect();
    assert_eq!(
        strides,
        [8, 4, 32],
        "the strides must be each slot's own, in slot order"
    );
}

/// Three attributes over one interleaved buffer are three bindings of one stride.
///
/// The case the geometry store's dedup produces: `buffers::needs` collapses three descriptors over
/// one twelve-byte vertex into one *buffer* and keeps three `Reads`, each with its own slot. So the
/// pipeline sees three bindings and the draw binds one buffer handle to all three -- the dedup saves
/// the allocation, not the binding.
///
/// This test is here because the other reading is tempting: I wrote `vertex_input` believing three
/// attributes over one buffer shared a binding, and this case is what said otherwise.
#[test]
fn an_interleaved_vertex_is_three_bindings_of_one_stride() {
    let found = pipelines::vertex_input(&key(vec![
        slot(0, vk::Format::R32_SFLOAT, 0, 12),
        slot(1, vk::Format::R32_SFLOAT, 4, 12),
        slot(2, vk::Format::R32_SFLOAT, 8, 12),
    ]))
    .expect("describable");

    assert_eq!(found.bindings.len(), 3);
    assert!(
        found.bindings.iter().all(|b| b.stride == 12),
        "every binding over one interleaved vertex has that vertex's stride"
    );
    let offsets: Vec<u32> = found.attributes.iter().map(|a| a.offset).collect();
    assert_eq!(offsets, [0, 4, 8]);
    for (at, attribute) in found.attributes.iter().enumerate() {
        assert_eq!(
            attribute.binding, at as u32,
            "each attribute reads its own binding, not a shared one"
        );
    }
}

/// A location claimed twice is refused.
///
/// The one way a layout can fail to be described. The binding number *is* the location, so two
/// slots clash at a binding exactly when they clash at a location -- which is why there is one
/// variant and not two.
#[test]
fn a_repeated_location_is_refused() {
    assert_eq!(
        pipelines::vertex_input(&key(vec![
            slot(2, vk::Format::R32_SFLOAT, 0, 8),
            slot(2, vk::Format::R32G32_SFLOAT, 0, 8),
        ]))
        .expect_err("refused"),
        Invalid::RepeatedLocation { location: 2 }
    );
    // Including when the strides differ, which an earlier version reported as a separate kind of
    // clash. The slots would have to be equal to share a binding and distinct to disagree, so that
    // variant could not be reached.
    assert_eq!(
        pipelines::vertex_input(&key(vec![
            slot(3, vk::Format::R32_SFLOAT, 0, 12),
            slot(3, vk::Format::R32_SFLOAT, 4, 16),
        ]))
        .expect_err("refused"),
        Invalid::RepeatedLocation { location: 3 }
    );
}

/// An empty layout describes nothing rather than failing.
///
/// A family with no vertex input is a thing: the clipping mask draws a quad the vertex stage builds
/// from its own index.
#[test]
fn an_empty_layout_is_empty_state() {
    let found = pipelines::vertex_input(&key(Vec::new())).expect("describable");
    assert!(found.bindings.is_empty());
    assert!(found.attributes.is_empty());
}

/// The input rate reaches the binding description.
///
/// `Key` carries it because it is pipeline state, and a binding described per vertex where the
/// producer meant per instance advances once a triangle instead of once a wall.
#[test]
fn the_rate_reaches_the_binding() {
    let instanced = Slot {
        rate: vk::VertexInputRate::INSTANCE,
        ..slot(0, vk::Format::R32_SFLOAT, 0, 4)
    };
    let found = pipelines::vertex_input(&key(vec![instanced])).expect("describable");
    assert_eq!(found.bindings[0].input_rate, vk::VertexInputRate::INSTANCE);
}
