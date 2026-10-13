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

use std::collections::HashMap;

use ash::vk;
use tessella_capture_abi::envelope::DrawFlags;
use tessella_capture_abi::generated::mbgl_enums::BuiltIn;

use tessella_vk::{DescriptorSetLayout, Gpu, PipelineLayout};

use crate::device::Attachment;
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
    /// Which of the family's attributes the drawable did not send, as a bitmask over locations.
    ///
    /// mbgl's `propertiesAsUniforms`, by another route. A data-driven paint property that is
    /// constant for a layer arrives as no descriptor at all, and the module then reads the layer's
    /// evaluated value instead of an attribute -- a different program, with a different vertex
    /// input. So this names the *module*, not only the pipeline, and [`Cache`] keys both on it.
    ///
    /// Derived from the plan by [`key`] rather than passed in, because the plan is the only thing
    /// that knows: `batch::Key::permutation_key` is mbgl's own hash over its property names and
    /// says nothing about this crate's locations. See [`crate::shaders::permutation`].
    pub permutation: u64,
    /// The vertex input state, in slot order.
    pub layout: Vec<Slot>,
    /// How the draw composites.
    ///
    /// In the key because it is pipeline state, for the same reason the vertex layout is: one
    /// family, permutation and layout can want two of these, and a cache keyed without it would
    /// hand the second draw the first's pipeline and composite it wrongly. Which draws.
    ///
    /// Not hypothetical. Seven of the twenty families are [`crate::families::Composite::ByPass`],
    /// and the captured Berlin frame batches its background twice -- once in each pass -- which is
    /// the same family, permutation and layout under two blends.
    pub blend: Blend,
    /// Whether the draw writes depth.
    ///
    /// In the key for the reason [`Self::blend`] is: it is pipeline state, the two passes of one
    /// frame want different values of it, and a cache keyed without it would hand the translucent
    /// pass the opaque pass's pipeline. Which draws, and occludes what follows it.
    pub depth: Depth,
    /// Which of the two tests and the color write the drawable asked for.
    ///
    /// In the key for the same reason again, and not hypothetically: the captured Berlin frame's
    /// twelve batches carry three different values of it, and five of them ask for something other
    /// than all three.
    pub enabled: Enabled,
}

/// The two tests and the color write a drawable's own flags ask for.
///
/// Three of `DrawFlags`'s five bits, which travel on `ViewUse` beside the render pass and reach
/// here as `batch::Key::draw_flags`. They are not a property of the family or of the pass: mbgl
/// sets them per drawable, and tessella#379 put the whole byte in the merge key because each one
/// is pipeline state that cannot be shared across a batch boundary.
///
/// # mbgl's three arms
///
/// Each bit is a mode or its `disabled()`, read off the oracle at `ad5e73527f3a`:
///
/// * `vulkan/drawable.cpp:312` -- `enableDepth` false gives `gfx::DepthMode::disabled()`.
/// * `vulkan/drawable.cpp:325` -- `enableStencil` false gives `gfx::StencilMode::disabled()`.
/// * `gl/drawable_gl.cpp:242` -- `enableColor ? parameters.colorModeForRenderPass() :
///   gfx::ColorMode::disabled()`, which is the fourth `ColorMode` and this crate's [`no_color`].
///   The Vulkan drawable does not read the flag at all, which is mbgl's own gap rather than a
///   different rule: the ABI's note is "Draw writes color. Cleared for depth- or stencil-only
///   passes."
///
/// Its builder's defaults are `enableColor = true`, `enableStencil = false`, `enableDepth = true`
/// (`gfx/drawable_builder.hpp:234`), so the clip test is the one a layer has to ask for.
///
/// # The two bits that are not here
///
/// `ON_TERRAIN` is a surface and is read by [`crate::surface::Surface::of`]. `IS_3D` is pipeline
/// state in mbgl and is **not transcribed yet**: it selects `depthModeFor3D` (the whole range
/// rather than the layer's band) and `stencilModeFor3D` (`NotEqual{0xFF}` against a freshly
/// allocated id, which writes the stencil and discards the tile clip masks) in place of the
/// per-sublayer modes. Both are a scheme of their own, and the captured frame's four extrusion
/// batches are what will measure them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Enabled {
    /// `DrawFlags::ENABLE_DEPTH`: whether the draw tests depth at all.
    ///
    /// [`Key::depth`] is the other half and says whether it *writes* what it tests against. This
    /// one gates both: mbgl's other arm is `DepthMode::disabled()`, which neither tests nor writes.
    pub depth: bool,
    /// `DrawFlags::ENABLE_STENCIL`: whether the draw tests its tile's clip mask.
    ///
    /// Asked for rather than assumed, which is the part worth stating. mbgl's default is false and
    /// three layers set it: the fill and the line unconditionally
    /// (`render_fill_layer.cpp:181`, `render_line_layer.cpp:279`), and the extrusion's color pass
    /// only where it has a depth pass to test against (`render_fill_extrusion_layer.cpp:374`). So
    /// the captured frame's background and circle batches carry no stencil bit, and a baked test
    /// would hold them against whichever tile's value last reached the texel.
    pub stencil: bool,
    /// `DrawFlags::ENABLE_COLOR`: whether the draw writes color.
    ///
    /// Cleared for a draw that is there for its depth. The captured frame has two:
    /// `render_fill_extrusion_layer.cpp:283` builds a depth-only drawable beside the color one
    /// where `!opaque || hasPattern`, because "the non-pattern path in `render()` only uses
    /// two-pass rendering if there's translucency" -- and the prepass is what makes a translucent
    /// wall blend once per pixel rather than once per overlapping wall. Baked on, that drawable
    /// paints the geometry it was built to leave invisible.
    pub color: bool,
}

impl Enabled {
    /// All three, which is what a caller with nothing to say about them means.
    ///
    /// Not a [`Default`], deliberately: `DrawFlags::NONE` is the opposite answer from the same
    /// absence of information, and the two spellings would disagree.
    pub const ALL: Self = Self {
        depth: true,
        stencil: true,
        color: true,
    };

    /// What a drawable's `draw_flags` ask for.
    ///
    /// The byte whole, as `batch::Key::draw_flags` carries it. Bits this does not name are not
    /// read: `ON_TERRAIN` is a surface, and `IS_3D` has no arm here yet.
    #[must_use]
    pub const fn of(draw_flags: u8) -> Self {
        Self {
            depth: draw_flags & DrawFlags::ENABLE_DEPTH.bits() != 0,
            stencil: draw_flags & DrawFlags::ENABLE_STENCIL.bits() != 0,
            color: draw_flags & DrawFlags::ENABLE_COLOR.bits() != 0,
        }
    }
}

/// Whether a draw writes the depth it tests against.
///
/// §11.7's split, as pipeline state: "opaque layers front-to-back with depth writes per `pass` +
/// `opaquePassCutoff`". The opaque pass writes, so a nearer layer rejects the ones behind it --
/// which is the early-out, measured at 89% of a sixteen-layer frame on V3D in #104. The
/// translucent pass tests and does not write: it is painted back to front and a translucent layer
/// must not occlude the one in front of it.
///
/// Not a dynamic state, though `VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE` is core in Vulkan 1.3 and
/// V3D reports 1.3. This crate does not create the device -- `check_dynamic_rendering` takes the
/// device's answer as a bool -- so it cannot enable a feature, only ask whether one is there. Two
/// pipelines is the cost of not assuming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Depth {
    /// Tested and written, which is the opaque pass.
    #[default]
    Write,
    /// Tested and not written, which is the translucent pass.
    TestOnly,
}

impl Depth {
    /// Which a pass draws with, from `batch::Key::pass`.
    ///
    /// `RenderPass` is a bitflag and a drawable can be in both passes -- the background is
    /// `Opaque | Translucent` -- but an `OrderEntry` carries one bit, because `DrawOrder::resolve`
    /// expands such a drawable into one entry per pass. So this reads the one bit that is set, and
    /// anything that is not the opaque bit is treated as translucent: a pass this build does not
    /// know must not be given depth writes on the strength of not being recognized.
    #[must_use]
    pub fn of(pass: u8) -> Self {
        if pass & tessella_capture_abi::RenderPass::OPAQUE.bits() != 0 {
            Self::Write
        } else {
            Self::TestOnly
        }
    }
}

/// The key for a drawable whose input has been planned.
///
/// Built from the plan rather than from the family's table, because the table says what a module
/// *declares* and the plan says what this drawable *binds*. They differ whenever a data-driven
/// attribute is constant for this permutation and no descriptor arrived for it, and that difference
/// is a different pipeline -- and, through [`Key::permutation`], a different module.
#[must_use]
pub fn key(
    shader: BuiltIn,
    surface: Surface,
    plan: &Plan,
    blend: Blend,
    depth: Depth,
    enabled: Enabled,
) -> Key {
    Key {
        shader,
        surface,
        permutation: crate::shaders::permutation(&plan.absent),
        blend,
        depth,
        enabled,
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
    /// Which `@group(n)` it is in.
    ///
    /// Zero for everything a layer shares, which is every binding of every family whose draws
    /// differ only by `ubo_index`. One for a family's per-drawable resources: those cannot live in
    /// a set the layer shares, and an offset into a shared buffer is the alternative this pass
    /// stays clear of -- [`crate::blocks`] binds at offset zero on purpose, to keep
    /// `minStorageBufferOffsetAlignment` out of the crate.
    pub group: u32,
    /// The `@binding(n)` the module declares, counting from zero within its own group.
    pub binding: u32,
    /// What is bound there.
    pub kind: Kind,
    /// The wire slot whose resource belongs here, or `None` where it could not be resolved.
    ///
    /// For a block, which buffer: a family's blocks are in different buffers -- `UboUpdate::slot`
    /// is which buffer, not which entry -- so a binding that did not carry this could only be
    /// pointed at a layer's first one. [`crate::slots`] is what resolves it.
    ///
    /// For a texture or its sampler, the `TextureRef::slot` the producer names it by. Both of a
    /// texture's two bindings carry the same one, because one `TextureRef` fills both.
    ///
    /// Not [`Self::binding`], which is the `@binding(n)` of the module and counts from zero
    /// through the whole set. A slot is the producer's number for the resource and nothing orders
    /// the two together: a family's second texture is binding 2 or 4 and slot 1, and terrain's
    /// elevation is the last binding of its set and slot 8.
    pub slot: Option<u32>,
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
    for block in family.blocks.iter().chain(surface.blocks()) {
        out.push(Binding {
            group: 0,
            binding: out.len() as u32,
            kind: Kind::StorageBuffer,
            slot: crate::slots::of(block),
        });
    }
    // Then two bindings for every texture, the family's own first and the surface's after -- a
    // family's samplers are a property of the shader rather than of what it is drawn on.
    // The slots rather than the two counts: they are the one thing both halves have, a family's
    // from its generated table and a surface's hand-agreed beside its names. The binding number is
    // still `out.len()`, which is the part that must not become a formula.
    let slots = family
        .textures
        .iter()
        .map(|texture| texture.binding)
        .chain(surface.texture_slots().iter().copied());
    for slot in slots {
        for kind in [Kind::SampledImage, Kind::Sampler] {
            out.push(Binding {
                group: 0,
                binding: out.len() as u32,
                kind,
                slot: Some(slot),
            });
        }
    }
    // And last, outside group zero, the instance buffer a family with instance-rate attributes
    // reads ahead in. One binding whatever the family, because it is one buffer: the attributes
    // that name a shared buffer in the ABI's table are the fields of its element, and the ABI's
    // own note is that mbgl registers them as attributes so that the bytes are *also* bound as a
    // storage buffer. No slot: it is a geometry's buffer rather than a layer's block, so nothing
    // in `slots` names it.
    if shares_a_buffer(family) {
        out.push(Binding {
            group: 1,
            binding: 0,
            kind: Kind::StorageBuffer,
            slot: None,
        });
    }
    out
}

/// Whether a family's instance attributes name a buffer that is also bound as a storage buffer.
///
/// Read off `ShaderAttribute::shared_buffer`, which `mbgl-codegen` carries from
/// `AttributeInfo`'s fourth argument -- the UBO binding mbgl registers the same buffer at. A
/// family with instance attributes and none of them shared would be one whose body reads only its
/// own instance, and would need no second group.
#[must_use]
pub fn shares_a_buffer(family: &Family) -> bool {
    family
        .instance_attributes
        .iter()
        .any(|attribute| attribute.shared_buffer.is_some())
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
    instances: Option<DescriptorSetLayout<'d>>,
    pipeline: PipelineLayout<'d>,
}

impl core::fmt::Debug for Layout<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Layout").finish_non_exhaustive()
    }
}

impl Layout<'_> {
    /// The group-zero set layout, for allocating a layer's descriptor set.
    #[must_use]
    pub fn set(&self) -> vk::DescriptorSetLayout {
        self.set.raw()
    }

    /// The group-one set layout, for a family that has one.
    ///
    /// `None` for every family whose state is per layer, which is all of them but the instanced
    /// extrusions. A caller writing a per-drawable set needs this and not [`Self::set`]: the two
    /// layouts describe different sets and a set allocated from the wrong one binds nothing the
    /// module declares.
    #[must_use]
    pub fn instances(&self) -> Option<vk::DescriptorSetLayout> {
        self.instances.as_ref().map(DescriptorSetLayout::raw)
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
    set_bindings_of(bindings, 0)
}

/// The same for one group, which is how a layout with two sets is built.
///
/// Group zero is [`set_bindings`]; group one is a family's per-drawable resources. Each set
/// numbers its bindings from zero, so the two cannot be described together.
#[must_use]
pub fn set_bindings_of(
    bindings: &[Binding],
    group: u32,
) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    bindings
        .iter()
        .filter(|binding| binding.group == group)
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

    // Group one only where the bindings say there is one, so no family gains a second set layout
    // or a push constant range it does not read. The two travel together: the group holds the
    // instance buffer and the push constant holds the `ubo_index` that cannot ride `firstInstance`
    // on the same draw -- see [`crate::draws`].
    let described = set_bindings_of(bindings, 1);
    let instances = if described.is_empty() {
        None
    } else {
        Some(gpu.set_layout(&described)?)
    };

    let pipeline = match instances.as_ref() {
        Some(second) => gpu.pipeline_layout(&[&set, second], UBO_INDEX_BYTES)?,
        None => gpu.pipeline_layout(&[&set], 0)?,
    };
    Ok(Layout {
        set,
        instances,
        pipeline,
    })
}

/// Bytes of push constant a family with instance-rate attributes declares: `ubo_index`, a `u32`.
pub const UBO_INDEX_BYTES: u32 = 4;

/// The vertex input state a key describes, as Vulkan wants it.
///
/// One binding and one attribute per slot, with the binding number equal to the `@location` --
/// which is what [`Slot::slot`] is, and what the producer's own descriptors say.
///
/// # The interleaved case is three bindings, not one
///
/// `buffers::needs` deduplicates three descriptors over one twelve-byte vertex into one *buffer*,
/// and keeps three `Reads`, each with its own slot. So the pipeline sees three bindings of stride
/// twelve at offsets zero, four and eight, and the draw binds one buffer handle to all three. The
/// dedup saves the allocation, not the binding.
///
/// That is worth stating because the other reading is tempting and wrong: one binding read by three
/// attributes would need the three to agree about the stride, and nothing in a `Plan` makes them
/// distinct slots if they are one binding. They are distinct slots, so they are distinct bindings.
/// `ash`'s description structs implement neither `PartialEq` nor `Eq`, so neither does this. A test
/// comparing two of these compares the fields it cares about, which is the honest thing anyway: a
/// derived equality over a `#[repr(C)]` Vulkan struct also compares its padding.
#[derive(Debug, Clone, Default)]
pub struct VertexInput {
    /// One per binding, in slot order.
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    /// One per attribute, in slot order.
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
}

/// Why a key cannot become vertex input state.
///
/// One variant, because the binding number *is* the location: two slots clash at a binding exactly
/// when they clash at a location, so there is nothing a second variant could distinguish. An
/// earlier version of this had one for a binding claimed twice with two strides, which cannot
/// happen -- the slots would have to be equal to share a binding and distinct to disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalid {
    /// Two slots claim one `@location`, and so one binding number.
    ///
    /// A module declares each location once, so this is a plan naming the same shader input twice.
    /// `vkCreateGraphicsPipelines` rejects it too, as a duplicated binding; this says which.
    RepeatedLocation {
        /// The location claimed twice.
        location: u32,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RepeatedLocation { location } => {
                write!(f, "location {location} is claimed twice")
            }
        }
    }
}

impl std::error::Error for Invalid {}

/// The vertex input state for a key.
///
/// # Errors
///
/// [`Invalid::RepeatedLocation`] for a layout claiming one location twice.
pub fn vertex_input(key: &Key) -> Result<VertexInput, Invalid> {
    let mut bindings: Vec<vk::VertexInputBindingDescription> = Vec::new();
    let mut attributes: Vec<vk::VertexInputAttributeDescription> = Vec::new();

    for slot in &key.layout {
        if attributes.iter().any(|a| a.location == slot.slot) {
            return Err(Invalid::RepeatedLocation {
                location: slot.slot,
            });
        }
        bindings.push(
            vk::VertexInputBindingDescription::default()
                .binding(slot.slot)
                .stride(slot.stride)
                .input_rate(slot.rate),
        );
        attributes.push(
            vk::VertexInputAttributeDescription::default()
                .location(slot.slot)
                .binding(slot.slot)
                .format(slot.format)
                .offset(slot.offset),
        );
    }

    Ok(VertexInput {
        bindings,
        attributes,
    })
}

/// What a pipeline renders into, by format rather than by handle.
///
/// There is no `VkRenderPass` and no `VkFramebuffer` here. Dynamic rendering is core in Vulkan 1.3
/// and reported by every part this runs on -- RADV, V3D 7.1.7.0 and the `VeriSilicon` `GC7000UL` --
/// so a pipeline names the *formats* it is compatible with and a frame names the image views.
///
/// That is not a tidiness preference. #60's contract is a ring of at least three host images, and
/// says the pass "must not cache per-image state that breaks when the image changes every frame".
/// A framebuffer is precisely that state: one per image, and one more whenever the ring is resized.
/// Formats are shared by the whole ring, so there is nothing per-image left to invalidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Targets {
    /// The host image's format, which the pass is handed rather than choosing.
    pub color: vk::Format,
    /// The depth-stencil format, from [`crate::device::depth_stencil_format`].
    pub depth_stencil: vk::Format,
    /// Whether this view keeps depth, or only stencil.
    pub attachment: Attachment,
}

/// What a content pipeline sets per draw rather than baking.
///
/// The viewport and scissor, so one pipeline serves a ring of images of any size -- baking them
/// would need a pipeline per target size, and #60's host may resize its ring.
///
/// And the **stencil compare mask and reference**, which are per tile. `stencil::partition` gives
/// each tile an `Assignment` whose `value` the draw compares against and whose `read_mask` is its
/// own zoom's field of the stencil byte -- so a content draw comparing all eight bits would test
/// bits belonging to another zoom, and one comparing another tile's value would be clipped by a
/// tile that is not its own. Both differ per tile, so baking either means a pipeline per tile.
///
/// # The reference has to be declared, not merely set
///
/// It was set and not declared. [`crate::record::content`] calls `Recorder::stencil_reference` per
/// drawable and the whole clipping scheme rests on it, but this array listed only the compare mask
/// -- and a `vkCmdSet*` for state a pipeline did not declare dynamic is ignored, the pipeline's
/// static value being used instead. That value is zero, so every content draw would have compared
/// zero: a layer drawn exactly where the stencil is *not* its tile's value, which is everywhere
/// except its own tile.
///
/// Measured rather than reasoned: RADV and V3D 7.1.7.0 both honor the call anyway, and
/// `benches/clip_masks.rs` passes on both either way. Two drivers agreeing is not the spec
/// agreeing, and this is the same shape as `maxPerStageDescriptor*` -- invalid usage that no driver
/// has to report. The declaration is the fix; the bench cannot be.
///
/// The *write* mask is not here. A content draw never writes the stencil, so zero is baked and
/// cannot be set wrong at a call site.
pub const CONTENT_DYNAMIC: [vk::DynamicState; 4] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::STENCIL_COMPARE_MASK,
    vk::DynamicState::STENCIL_REFERENCE,
];

/// The depth and stencil state a content draw gets.
///
/// Three inputs, because the state has three sources. The **view** says whether there is depth to
/// test at all -- [`Attachment`] is where that is decided, and a view of flat layers can take a
/// stencil-only format, which on a tiler is less to write back. The **pass** says whether the test
/// also writes ([`Depth`]). The **drawable** says whether either test runs at all ([`Enabled`]),
/// which is the half mbgl keeps per drawable instead of in a mode: the per-tile clip masks are what
/// the stencil buffer is for, and a drawable that covers the viewport rather than a tile has no
/// mask of its own to test.
///
/// `KEEP` on every stencil op and `EQUAL` as the compare: a content draw reads the mask and does not
/// write it. Whoever draws the masks sets its own stencil state, which is a later slice -- this is
/// the state the *content* pipelines want.
pub fn depth_stencil(
    attachment: Attachment,
    depth: Depth,
    enabled: Enabled,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let keep = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::EQUAL,
        compare_mask: CONTENT_COMPARE_MASK,
        // A content draw never writes, so this is baked rather than dynamic: zero at a call site
        // that cannot set it is one fewer thing to get wrong.
        write_mask: 0,
        reference: 0,
    };
    // Two conditions, and they are different questions: the view has to keep depth, and the
    // drawable has to have asked to test it. `Enabled::depth` is mbgl's `enableDepth`, whose other
    // arm is `DepthMode::disabled()` -- which neither tests nor writes, so it gates both lines
    // below.
    let kept = matches!(attachment, Attachment::DepthStencil) && enabled.depth;
    vk::PipelineDepthStencilStateCreateInfo::default()
        // Tested only where there is depth to keep. A flat view that enabled the test against a
        // stencil-only attachment would be reading an attachment it does not have.
        .depth_test_enable(kept)
        // And written only in the pass that owns the ordering. The translucent pass tests without
        // writing, so a translucent layer does not occlude the one painted in front of it.
        .depth_write_enable(kept && matches!(depth, Depth::Write))
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        // And the mask is tested only by a draw that is clipped to a tile. A drawable covering the
        // viewport carries no `ENABLE_STENCIL` and has no mask of its own, so a baked test would
        // hold it against whichever tile's value last reached the texel.
        .stencil_test_enable(enabled.stencil)
        .front(keep)
        .back(keep)
}

/// The stencil state a content draw compares with, before the per-tile masks are set.
///
/// `compare_mask` is zero here and set per draw from the tile's `read_mask`. Zero because that is
/// what [`crate::record::clipping`] returns for a drawable with no tile and what
/// [`crate::record::content`] counts as unclipped, so "no clipping" has one spelling across the
/// crate and the baked default agrees with it.
///
/// Not for the reason this once gave, which was backwards. It said a never-set compare mask
/// "compares no bits and the draw is clipped away entirely, which is a blank layer". The stencil
/// test compares `reference & compare_mask` with `stencil & compare_mask`, so a mask of zero makes
/// both sides zero and `EQUAL` *passes* -- the draw covers everything. Measured on RADV and on
/// V3D 7.1.7.0, from the same scene and the same unassigned reference that draws nothing with a
/// real mask: 4096 of 4096 texels. `0xFF` baked against a stencil cleared to zero passes as well,
/// so neither value fails closed and the old sentence's distinction did not exist.
const CONTENT_COMPARE_MASK: u32 = 0;

/// How a draw composites over the target.
///
/// mbgl's four named `ColorMode`s, which is where these come from rather than from first
/// principles: `include/mbgl/gfx/color_mode.hpp` has `unblended`, `alphaBlended`, `additive` and
/// `disabled`, and a render layer picks one.
///
/// Three of the four are here. The fourth, `disabled`, is a mask of all-false rather than a blend
/// function -- it writes no color at all -- and this crate spells it [`no_color`], which is what a
/// clipping mask's pipeline is built with.
///
/// Which of the three a draw takes is [`crate::families::Composite`] and the batch's pass, so it
/// is a parameter of [`build`] rather than a constant: a family can have two of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Blend {
    /// The source replaces the target. mbgl's `unblended`.
    ///
    /// What the opaque pass uses, and what a readback oracle wants: the pixel is what the fragment
    /// stage returned rather than what it returned composited over whatever was there.
    Unblended,
    /// Premultiplied alpha over the target. mbgl's `alphaBlended`, and the usual one.
    #[default]
    Alpha,
    /// The source adds to the target. mbgl's `additive`.
    ///
    /// What a heatmap accumulates with, where every contribution brightens the same texel.
    Additive,
}

impl Blend {
    /// The attachment state this mode is.
    ///
    /// # Premultiplied, which is the part worth stating
    ///
    /// [`Self::Alpha`] is `src = ONE`, not `SRC_ALPHA`. mbgl's `alphaBlended` is
    /// `Add{One, OneMinusSrcAlpha}` because its fragment stages return **premultiplied** color --
    /// `out_color = color * opacity` scales the alpha channel along with the others, and this
    /// crate's bodies are transcribed from those and do the same. One of them says so outright:
    /// "the sheet is premultiplied, so the opacity and the fade scale it directly".
    ///
    /// `SRC_ALPHA` would multiply by alpha a second time, darkening every blended edge by an amount
    /// that reads as a style difference rather than a defect. An earlier version of this crate had
    /// exactly that, with a test asserting it.
    pub fn attachment(self) -> vk::PipelineColorBlendAttachmentState {
        let state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        match self {
            Self::Unblended => state.blend_enable(false),
            Self::Alpha => state
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD),
            Self::Additive => state
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD),
        }
    }
}

/// The color state a content draw gets: its blend mode, or no color at all.
///
/// The second arm is mbgl's fourth `ColorMode` -- [`no_color`], every channel masked off -- and a
/// draw that asked for it is there for its depth or its stencil. Written with the mask on, such a
/// draw paints the geometry it was meant to leave invisible: the captured Berlin frame's two
/// depth-only extrusion batches would put their walls over the layers under them.
///
/// A function of its own rather than a line inside [`build`], so the choice is checkable without a
/// device. A pipeline built with the wrong one builds perfectly well.
pub fn color_attachment(blend: Blend, enabled: Enabled) -> vk::PipelineColorBlendAttachmentState {
    if enabled.color {
        blend.attachment()
    } else {
        no_color()
    }
}

/// No culling.
///
/// A fill's triangles come from an earcut tessellation and a line's from a stroker, and neither
/// promises a winding order. mbgl does not cull either. Culling the wrong way is a layer that
/// disappears at some zooms and not others, which is a bug nobody finds quickly, and the saving on a
/// tiler is a fraction of a pass.
pub fn rasterization() -> vk::PipelineRasterizationStateCreateInfo<'static> {
    vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
}

/// The entry points `crate::shaders::module` emits, as the names Vulkan wants.
const VERTEX_ENTRY: &core::ffi::CStr = c"vertex_main";
const FRAGMENT_ENTRY: &core::ffi::CStr = c"fragment_main";

/// Builds the pipeline a key describes, against the targets given.
///
/// Both stages come from one module, which is what `shaders::module` assembles: one text, two entry
/// points. The same handle is named twice rather than compiled twice.
///
/// Everything not derived from the key is a decision with a reason beside it --
/// [`depth_stencil`], [`Blend::attachment`], [`rasterization`], and the dynamic viewport and scissor. The
/// topology is a triangle list because every bucket this crate draws is indexed triangles; a line is
/// a stroked quad pair and a circle is a quad, both tessellated by the producer.
///
/// # Errors
///
/// [`Error::Invalid`] when the key's layout cannot be described, and [`Error::Device`] when the
/// device refuses the pipeline.
pub fn build<'d>(
    gpu: Gpu<'d>,
    key: &Key,
    layout: &Layout<'_>,
    module: &tessella_vk::ShaderModule<'_>,
    targets: Targets,
) -> Result<tessella_vk::Pipeline<'d>, Error> {
    let input = vertex_input(key)?;
    let vertex_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&input.bindings)
        .vertex_attribute_descriptions(&input.attributes);

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module.raw())
            .name(VERTEX_ENTRY),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module.raw())
            .name(FRAGMENT_ENTRY),
    ];
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // Counts without pointers: the values are set per draw, so only the counts are baked.
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&CONTENT_DYNAMIC);
    let raster = rasterization();
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = depth_stencil(targets.attachment, key.depth, key.enabled);
    let attachments = [color_attachment(key.blend, key.enabled)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

    // The attachments by format, which is what replaces the render pass handle. A packed
    // depth-stencil format names the same attachment twice, because it is one image -- but a
    // stencil-only format has no depth attachment at all, and `UNDEFINED` is how a pipeline says
    // so. Declaring a depth format the rendering scope will not have is a mismatch, and
    // `Recorder::begin_rendering` now omits that attachment for exactly the same reason.
    let colors = [targets.color];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&colors)
        .depth_attachment_format(depth_format(targets.depth_stencil))
        .stencil_attachment_format(stencil_format(targets.depth_stencil));

    let create = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_state)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blending)
        .dynamic_state(&dynamic)
        .layout(layout.pipeline())
        .push_next(&mut rendering);

    Ok(gpu.graphics_pipeline(&create)?)
}

/// The depth attachment's format, or `UNDEFINED` where the format has no depth.
fn depth_format(format: vk::Format) -> vk::Format {
    if tessella_vk::aspects(format).contains(vk::ImageAspectFlags::DEPTH) {
        format
    } else {
        vk::Format::UNDEFINED
    }
}

/// The stencil attachment's format, or `UNDEFINED` where the format has no stencil.
///
/// Every format `device::depth_stencil_format` gives has a stencil, so this is `UNDEFINED` for
/// nothing it can return -- written as the pair of [`depth_format`] because a reader checking one
/// will ask about the other, and because a depth-only format is a legal thing to hand a pipeline
/// that does not clip.
fn stencil_format(format: vk::Format) -> vk::Format {
    if tessella_vk::aspects(format).contains(vk::ImageAspectFlags::STENCIL) {
        format
    } else {
        vk::Format::UNDEFINED
    }
}

/// Why a pipeline could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The key's vertex input cannot be described.
    Invalid(Invalid),
    /// The device refused.
    Device(tessella_vk::Error),
}

impl From<Invalid> for Error {
    fn from(why: Invalid) -> Self {
        Self::Invalid(why)
    }
}

impl From<tessella_vk::Error> for Error {
    fn from(why: tessella_vk::Error) -> Self {
        Self::Device(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "{why}"),
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// The depth and stencil state a clip mask is drawn with.
///
/// The other half of [`depth_stencil`]. A content draw tests `EQUAL` and never writes; a mask
/// writes and never tests. So this is `compare_op: ALWAYS` with `pass_op: REPLACE` and a full write
/// mask, and the value written is the dynamic stencil reference -- which is why the reference is
/// dynamic state rather than baked: one pipeline draws every tile's mask and
/// `vkCmdSetStencilReference` is what distinguishes them.
///
/// Depth is off in both directions. A mask has no depth of its own and must not occlude anything:
/// writing depth here would put a surface at the tile's plane that every later draw tests against.
pub fn depth_stencil_write() -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let replace = vk::StencilOpState {
        fail_op: vk::StencilOp::REPLACE,
        pass_op: vk::StencilOp::REPLACE,
        depth_fail_op: vk::StencilOp::REPLACE,
        // Always, so the compare mask is never read and is left at zero.
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        // Set per tile from the `Assignment`'s `write_mask`: its own zoom's field, plus any
        // ancestor field it replaces. Zero baked for the same reason the compare mask is -- a mask
        // that never set it writes nothing, which is a missing mask rather than a mask over every
        // other zoom's bits.
        write_mask: 0,
        reference: 0,
    };
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .stencil_test_enable(true)
        .front(replace)
        .back(replace)
}

/// A color blend state that writes no color at all.
///
/// What a mask wants, and mbgl's `disabled()`: a `Replace` function with every channel masked off.
/// The quad exists to put a number in the stencil buffer, and a mask that wrote color would paint a
/// tile-sized rectangle over the frame -- once per tile, under every layer. An empty write mask is
/// how a draw says it is only here for its side effects.
pub fn no_color() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::empty())
}

/// What a mask pipeline sets per draw rather than baking.
///
/// The viewport and scissor, as for content, plus the **reference and the write mask** -- which are
/// the two halves of a tile's `Assignment` that a mask uses. One pipeline draws every tile's mask
/// and both differ per tile, so baking either means a pipeline per tile.
pub const MASK_DYNAMIC: [vk::DynamicState; 4] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::STENCIL_REFERENCE,
    vk::DynamicState::STENCIL_WRITE_MASK,
];

/// Builds the pipeline that draws clip masks.
///
/// Takes the module rather than assembling it, as [`build`] does. There is no vertex input at all:
/// the quad comes from `vertex_index`, so there are no bindings and no attributes, and a key would
/// have nothing to carry.
///
/// # Errors
///
/// [`Error::Device`] when the device refuses the pipeline.
pub fn build_mask<'d>(
    gpu: Gpu<'d>,
    layout: &Layout<'_>,
    module: &tessella_vk::ShaderModule<'_>,
    targets: Targets,
) -> Result<tessella_vk::Pipeline<'d>, Error> {
    let empty = vk::PipelineVertexInputStateCreateInfo::default();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module.raw())
            .name(VERTEX_ENTRY),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module.raw())
            .name(FRAGMENT_ENTRY),
    ];
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&MASK_DYNAMIC);
    let raster = rasterization();
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = depth_stencil_write();
    let attachments = [no_color()];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

    let colors = [targets.color];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&colors)
        .depth_attachment_format(depth_format(targets.depth_stencil))
        .stencil_attachment_format(stencil_format(targets.depth_stencil));

    let create = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&empty)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blending)
        .dynamic_state(&dynamic)
        .layout(layout.pipeline())
        .push_next(&mut rendering);

    Ok(gpu.graphics_pipeline(&create)?)
}

/// The pipelines a frame has built, by key.
///
/// # Why a cache at all
///
/// `vkCreateGraphicsPipelines` compiles the shader. On the gating target that is tens of
/// milliseconds for the 61 modules together, and a frame that rebuilt a pipeline per batch would
/// pay it per batch -- a styled view has thousands. So a pipeline is built once per key and bound
/// many times, which is what the key exists for.
///
/// # Two levels, because the module is not per key
///
/// A module is per `(family, surface)`: one text, two entry points, 65 of them. A *pipeline* is per
/// [`Key`], which adds the permutation and the vertex input -- so one module serves every pipeline
/// of its family, and the two are cached separately. Holding the module beside each pipeline would
/// compile the same text once per stride.
///
/// # What it does not do
///
/// Evict. A key holds a family, a surface, a permutation and a vertex layout, and all four are
/// bounded by the style: a style that drew every family on every surface in every permutation it
/// declares would build that many pipelines and then build no more. Eviction would need a
/// measurement of a frame that actually thrashes, and nothing here has one -- so the cache grows to
/// the style's own size and stops, and [`Cache::len`] is what a caller watches to find out it was
/// wrong about that.
pub struct Cache<'d> {
    modules: HashMap<(BuiltIn, Surface, u64), tessella_vk::ShaderModule<'d>>,
    layouts: HashMap<(BuiltIn, Surface), Layout<'d>>,
    pipelines: HashMap<Key, tessella_vk::Pipeline<'d>>,
    built: usize,
    bound: usize,
}

impl core::fmt::Debug for Cache<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cache")
            .field("pipelines", &self.pipelines.len())
            .field("modules", &self.modules.len())
            .field("built", &self.built)
            .field("bound", &self.bound)
            .finish()
    }
}

impl Default for Cache<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'d> Cache<'d> {
    /// Nothing built.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            layouts: HashMap::new(),
            pipelines: HashMap::new(),
            built: 0,
            bound: 0,
        }
    }

    /// How many pipelines are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// How many modules are held, which is at most one per family, surface and permutation.
    #[must_use]
    pub fn modules(&self) -> usize {
        self.modules.len()
    }

    /// How many pipelines have been built, across the cache's life.
    ///
    /// Equal to [`Self::len`] while nothing is evicted, and reported separately so a caller can
    /// tell a cache that grew from one that is rebuilding -- which is what a thrash would look
    /// like.
    #[must_use]
    pub fn built(&self) -> usize {
        self.built
    }

    /// How many times a pipeline has been asked for and found.
    #[must_use]
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// The layout for a family and surface, built on first use.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the device refuses either layout.
    pub fn layout(
        &mut self,
        gpu: Gpu<'d>,
        shader: BuiltIn,
        surface: Surface,
        bindings: &[Binding],
    ) -> Result<&Layout<'d>, Error> {
        // `entry` would need the layout built before the lookup, and building it is the fallible
        // part -- so the miss is handled first and the entry taken after.
        if let std::collections::hash_map::Entry::Vacant(slot) =
            self.layouts.entry((shader, surface))
        {
            slot.insert(layout(gpu, bindings)?);
        }
        Ok(&self.layouts[&(shader, surface)])
    }

    /// The pipeline for a key, built on first use.
    ///
    /// `words` is only consulted on a miss, so a caller that has to compile to produce them should
    /// check [`Self::holds`] first -- compiling WGSL is the expensive half and a hit does not need
    /// it.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a key whose layout cannot be described, and [`Error::Device`] when
    /// the device refuses the module or the pipeline.
    pub fn pipeline(
        &mut self,
        gpu: Gpu<'d>,
        key: &Key,
        bindings: &[Binding],
        words: &[u32],
        targets: Targets,
    ) -> Result<vk::Pipeline, Error> {
        if let Some(had) = self.pipelines.get(key) {
            self.bound += 1;
            return Ok(had.raw());
        }

        // The module is keyed on the permutation as well, because it is the text that changed:
        // an absent attribute is left out of `In` and its property read from the layer's block
        // instead. The layout is not -- a permutation moves vertex input and nothing in the
        // descriptor set, so one family and surface have one set layout however their attributes
        // arrived.
        let text = (key.shader, key.surface, key.permutation);
        let at = (key.shader, key.surface);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.modules.entry(text) {
            slot.insert(gpu.shader(words)?);
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = self.layouts.entry(at) {
            slot.insert(layout(gpu, bindings)?);
        }
        let pipeline = build(gpu, key, &self.layouts[&at], &self.modules[&text], targets)?;
        let raw = pipeline.raw();
        self.pipelines.insert(key.clone(), pipeline);
        self.built += 1;
        self.bound += 1;
        Ok(raw)
    }

    /// Whether a key already has a pipeline.
    #[must_use]
    pub fn holds(&self, key: &Key) -> bool {
        self.pipelines.contains_key(key)
    }

    /// Whether the module for a family and surface is already compiled.
    ///
    /// This is the question a caller deciding whether to compile should ask, and [`Self::holds`] is
    /// not it. A key carries the vertex layout, so a drawable of a known family with a new stride
    /// is a pipeline *miss* and a module *hit* -- and a caller that checked only `holds` would
    /// recompile the text to build a pipeline from a module it already has.
    ///
    /// [`Self::pipeline`] ignores `words` when the module is present, so a caller seeing `true`
    /// here can pass an empty slice.
    ///
    /// The permutation is part of the question for the reason [`Key::permutation`] gives: a
    /// drawable that sent fewer attributes needs its own text, so a cache asked without it would
    /// report a hit and then be handed the wrong module's words to ignore.
    #[must_use]
    pub fn has_module(&self, shader: BuiltIn, surface: Surface, permutation: u64) -> bool {
        self.modules.contains_key(&(shader, surface, permutation))
    }
}
