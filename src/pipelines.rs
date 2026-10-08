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
//! # The input rate is in the key
//!
//! An instanced family binds its wall outline per instance and its position per vertex, and that
//! is pipeline state: two drawables differing only in a binding's rate are two pipelines.
//! [`crate::vertices::plan_instanced`] sets it from which run a descriptor arrived in, so the
//! field has a source rather than being a placeholder.

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;

use tessella_vk::{DescriptorSetLayout, Gpu, PipelineLayout};

use crate::families::Family;
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
    /// Whether the binding advances per vertex or per instance.
    pub rate: vk::VertexInputRate,
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
                rate: bound.rate,
            })
            .collect(),
    }
}

/// What one descriptor in a family's set is.
///
/// Three kinds, because that is what the assembled modules declare: a block is
/// `var<storage, read>`, and a sampled texture is a `texture_2d<f32>` and a `sampler` as two
/// separate bindings rather than one combined. Separate because naga emits WGSL's two objects as
/// two descriptors, and a layout offering `COMBINED_IMAGE_SAMPLER` where the module declares a pair
/// does not match -- which `vkCreateGraphicsPipelines` rejects, loudly, so this one at least fails
/// rather than draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A uniform block, read as a storage buffer.
    StorageBuffer,
    /// A texture a placement or a body samples.
    SampledImage,
    /// The sampler that reads it.
    Sampler,
}

impl Kind {
    /// The Vulkan descriptor type.
    #[must_use]
    pub const fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            Self::Sampler => vk::DescriptorType::SAMPLER,
        }
    }
}

/// One entry of a family's descriptor set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Binding {
    /// The `@binding(n)` the module declares, in group zero.
    pub binding: u32,
    /// What is bound there.
    pub kind: Kind,
}

/// The descriptor set a module for this family and surface declares.
///
/// Group zero, in the order [`crate::shaders::module`] writes them, and derived by *counting* the
/// same sequence rather than by an index formula. That module says why:
///
/// > Counted rather than computed from the index, because the count differs by family and by
/// > surface and an index formula would be arithmetic no test could distinguish from a wrong one.
///
/// The same reasoning applies here with more force, because this is the half that has to agree with
/// the other. A descriptor set layout that disagrees with what the module declares is rejected by
/// `vkCreateGraphicsPipelines` when the types differ -- and when only the *counts* differ it is a
/// set whose later bindings are all one place out, which binds one block's bytes where another's
/// were meant and draws.
///
/// So `tests/descriptors.rs` checks this against the assembled text for every family and surface
/// rather than against a table, which is the only check that can catch the two drifting apart.
#[must_use]
pub fn bindings(family: &Family, surface: Surface) -> Vec<Binding> {
    let mut out = Vec::new();
    // The family's blocks first, then the surface's, exactly as the module declares them.
    for _ in family.blocks.iter().chain(surface.blocks()) {
        out.push(Binding {
            binding: out.len() as u32,
            kind: Kind::StorageBuffer,
        });
    }
    // Then two bindings for every texture, the family's own first and the surface's after -- a
    // family's samplers are a property of the shader rather than of what it is drawn on.
    // The two counts rather than the two iterators, which have different item types. The binding
    // number is still `out.len()`, which is the part that must not become a formula.
    for _ in 0..family.textures.len() + surface.textures().len() {
        for kind in [Kind::SampledImage, Kind::Sampler] {
            out.push(Binding {
                binding: out.len() as u32,
                kind,
            });
        }
    }
    out
}

/// How many descriptors of each kind a set needs, for sizing a pool.
///
/// Returned in a fixed order rather than a map, because a pool wants a list and there are three
/// kinds. A kind with no descriptors is left out: `vkCreateDescriptorPool` takes a zero count, but
/// a pool sized for something nothing declares is a pool that hides a layout gone wrong.
#[must_use]
pub fn pool_sizes(bindings: &[Binding]) -> Vec<(Kind, u32)> {
    [Kind::StorageBuffer, Kind::SampledImage, Kind::Sampler]
        .into_iter()
        .filter_map(|kind| {
            let count = bindings.iter().filter(|b| b.kind == kind).count() as u32;
            (count > 0).then_some((kind, count))
        })
        .collect()
}

/// The descriptor set layout a family's draws are recorded against, and the pipeline layout over it.
///
/// Held together because neither is useful alone: the set layout allocates the sets and the pipeline
/// layout binds them, and a pipeline built against one must be bound with sets from the other.
pub struct Layout<'d> {
    set: DescriptorSetLayout<'d>,
    pipeline: PipelineLayout<'d>,
}

impl core::fmt::Debug for Layout<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Layout").finish_non_exhaustive()
    }
}

impl Layout<'_> {
    /// The set layout, for allocating a descriptor set.
    #[must_use]
    pub fn set(&self) -> vk::DescriptorSetLayout {
        self.set.raw()
    }

    /// The pipeline layout, for creating a pipeline and binding its descriptors.
    #[must_use]
    pub fn pipeline(&self) -> vk::PipelineLayout {
        self.pipeline.raw()
    }
}

/// Which stages see a family's descriptors.
///
/// Both, for every binding. A block is read where the body reads it and the body is one text
/// compiled into two entry points -- `tessella_emblema` assembles one module per (family, surface)
/// and names `vs_main` and `fs_main` in it -- so a binding visible to one stage and not the other
/// would be a module that does not build against its own layout.
///
/// Narrowing this per binding would need the module's own reflection to say which stage touches
/// which block, and `vkCreateGraphicsPipelines` rejects a layout that is too narrow. The cost of
/// both is that a driver cannot prove a block is unread in a stage; the cost of guessing is a
/// pipeline that does not build.
const BOTH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw() | vk::ShaderStageFlags::FRAGMENT.as_raw(),
);

/// The `VkDescriptorSetLayoutBinding` list for a family's descriptors.
///
/// Split from [`layout`] so it can be inspected without a device, and because the bench wants to
/// count the kinds against the device's own limits before it tries to create anything.
#[must_use]
pub fn set_bindings(bindings: &[Binding]) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    bindings
        .iter()
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding.binding)
                .descriptor_type(binding.kind.descriptor_type())
                .descriptor_count(1)
                .stage_flags(BOTH_STAGES)
        })
        .collect()
}

/// Creates the layouts for a family's descriptor set.
///
/// # Errors
///
/// [`tessella_vk::Error::Call`] when the device refuses either layout, which is what asking for more
/// descriptors of a kind than `maxPerStageDescriptor*` allows produces. The widest set this crate
/// declares is twelve bindings -- five storage buffers, four sampled images and four samplers -- and
/// Vulkan's own floor for `maxPerStageDescriptorStorageBuffers` is four, so this is a limit a
/// conformant device may genuinely be under.
pub fn layout<'d>(gpu: Gpu<'d>, bindings: &[Binding]) -> Result<Layout<'d>, tessella_vk::Error> {
    let described = set_bindings(bindings);
    let set = gpu.set_layout(&described)?;
    let pipeline = gpu.pipeline_layout(&set)?;
    Ok(Layout { set, pipeline })
}
