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
    /// The data-driven variant, which decides the module's specialization constants.
    pub permutation: u64,
    /// The vertex input state, in slot order.
    pub layout: Vec<Slot>,
    /// How the draw composites.
    ///
    /// In the key because it is pipeline state, for the same reason the vertex layout is: two
    /// layers of one family and permutation can pick different `ColorMode`s -- a heatmap
    /// accumulates where a fill composites -- and a cache keyed without this would hand the second
    /// one the first's pipeline and blend it wrongly. Which draws.
    pub blend: Blend,
}

/// The key for a drawable whose input has been planned.
///
/// Built from the plan rather than from the family's table, because the table says what a module
/// *declares* and the plan says what this drawable *binds*. They differ whenever a data-driven
/// attribute is constant for this permutation and no descriptor arrived for it, and that difference
/// is a different pipeline.
#[must_use]
pub fn key(shader: BuiltIn, surface: Surface, permutation: u64, plan: &Plan, blend: Blend) -> Key {
    Key {
        shader,
        surface,
        permutation,
        blend,
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
    /// For a block, the slot its buffer arrives at; `None` for a texture or a sampler.
    ///
    /// A family's blocks are in different buffers -- `UboUpdate::slot` is which buffer, not which
    /// entry -- so a binding that did not carry this could only be pointed at a layer's first one.
    /// [`crate::slots`] is what resolves it, and `None` on a block means it could not.
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
            binding: out.len() as u32,
            kind: Kind::StorageBuffer,
            slot: crate::slots::of(block),
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
                slot: None,
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
/// And the **stencil compare mask**, which is per tile. `stencil::partition` gives each tile an
/// `Assignment` whose `read_mask` is its own zoom's field of the stencil byte, so a content draw
/// comparing all eight bits would test bits belonging to another zoom -- and be clipped by a tile
/// that is not its own. Baking it would mean a pipeline per tile.
///
/// The *write* mask is not here. A content draw never writes the stencil, so zero is baked and
/// cannot be set wrong at a call site.
pub const CONTENT_DYNAMIC: [vk::DynamicState; 3] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::STENCIL_COMPARE_MASK,
];

/// The depth and stencil state for a view.
///
/// Stencil always, because the per-tile clip masks are what the stencil buffer is for: §2.2 draws a
/// mask quad per tile and every content draw tests against it. Depth only when the view has an
/// extrusion -- [`Attachment`] is where that is decided, and a view of flat layers can take a
/// stencil-only format, which on a tiler is less to write back.
///
/// `KEEP` on every stencil op and `EQUAL` as the compare: a content draw reads the mask and does not
/// write it. Whoever draws the masks sets its own stencil state, which is a later slice -- this is
/// the state the *content* pipelines want.
pub fn depth_stencil(attachment: Attachment) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
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
    let depth = matches!(attachment, Attachment::DepthStencil);
    vk::PipelineDepthStencilStateCreateInfo::default()
        // Tested and written only where there is depth to keep. A flat view that enabled the test
        // against a stencil-only attachment would be reading an attachment it does not have.
        .depth_test_enable(depth)
        .depth_write_enable(depth)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .stencil_test_enable(true)
        .front(keep)
        .back(keep)
}

/// The stencil state a content draw compares with, before the per-tile masks are set.
///
/// `compare_mask` is zero here and set per draw from the tile's `read_mask` -- zero rather than
/// `0xFF` deliberately: a pipeline whose dynamic compare mask was never set then compares no bits
/// and the draw is clipped away entirely, which is a blank layer. `0xFF` baked would compare every
/// bit and draw everywhere, which is a layer with no clipping -- the failure that looks like
/// working.
const CONTENT_COMPARE_MASK: u32 = 0;

/// How a draw composites over the target.
///
/// mbgl's four named `ColorMode`s, which is where these come from rather than from first
/// principles: `include/mbgl/gfx/color_mode.hpp` has `unblended`, `alphaBlended`, `additive` and
/// `disabled`, and a layer picks one. Per layer rather than per family, which is why this is a
/// parameter of [`build`] and not a constant.
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
    let depth = depth_stencil(targets.attachment);
    let attachments = [key.blend.attachment()];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

    // The attachments by format, which is what replaces the render pass handle. The stencil format
    // is the same attachment as the depth one -- these are packed formats, so naming it twice is
    // naming one image twice.
    let colors = [targets.color];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&colors)
        .depth_attachment_format(targets.depth_stencil)
        .stencil_attachment_format(targets.depth_stencil);

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
        .depth_attachment_format(targets.depth_stencil)
        .stencil_attachment_format(targets.depth_stencil);

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
/// milliseconds for the 57 modules together, and a frame that rebuilt a pipeline per batch would
/// pay it per batch -- a styled view has thousands. So a pipeline is built once per key and bound
/// many times, which is what the key exists for.
///
/// # Two levels, because the module is not per key
///
/// A module is per `(family, surface)`: one text, two entry points, 57 of them. A *pipeline* is per
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
    modules: HashMap<(BuiltIn, Surface), tessella_vk::ShaderModule<'d>>,
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

    /// How many modules are held, which is at most one per family and surface.
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

        let at = (key.shader, key.surface);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.modules.entry(at) {
            slot.insert(gpu.shader(words)?);
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = self.layouts.entry(at) {
            slot.insert(layout(gpu, bindings)?);
        }
        let pipeline = build(gpu, key, &self.layouts[&at], &self.modules[&at], targets)?;
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
    #[must_use]
    pub fn has_module(&self, shader: BuiltIn, surface: Surface) -> bool {
        self.modules.contains_key(&(shader, surface))
    }
}
