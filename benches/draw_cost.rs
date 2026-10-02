//! What a draw costs to record, and what the per-draw state around it costs.
//!
//! Run with `cargo bench -p tessella-emblema`.
//!
//! # The question
//!
//! A sibling consumer on Flutter GPU measures `~62 us a draw` on an embedded board, and half of
//! that is the Dart side walking batches and calling the backend. That number is what made
//! merging drawables into one draw the whole of its performance work: 640 draws to 86 took its
//! frame from 46.8 ms to 13.7.
//!
//! None of it transfers without measuring. Recording a draw from Rust into a command buffer is
//! not the same act as encoding one through a Dart API, and if it costs a fraction of a
//! microsecond then merging buys a fraction of nothing and the effort belongs elsewhere.
//!
//! # What is measured
//!
//! Four shapes, the same drawables each time, differing only in how much state each one carries.
//! Strictly additive, so the difference between two rows is the price of what the later one adds
//! -- which is the thing a merge exists to remove:
//!
//! | shape | per drawable |
//! | --- | --- |
//! | `instance` | an indexed draw alone, the slot riding in `firstInstance` |
//! | `push` | plus a push constant carrying the slot |
//! | `buffers` | plus its own vertex and index buffer |
//! | `descriptors` | plus its own descriptor set |
//!
//! Both halves are timed: recording on the CPU, and submit to fence, which is what the device did
//! with it. Recording is the half the sibling's encode compares against.
//!
//! # What this is not
//!
//! One driver on one machine, and a record cost is the driver's as much as the API's. What carries
//! across is the order of magnitude: a tenth of a microsecond and thirty microseconds are
//! different architectures, not different tunings.

use std::ffi::CStr;
use std::time::{Duration, Instant};

use ash::vk;
use tessella_capture_abi::generated::shader_attributes::BACKGROUND_SHADER;
use tessella_capture_abi::generated::ubo_layouts::{BACKGROUND_DRAWABLE_UBO, BACKGROUND_PROPS_UBO};
use tessella_emblema::shaders::{BACKGROUND_BODY, module};
use tessella_emblema::surface::Surface;

/// Drawables a recorded frame holds. The sibling's unmerged basemap is 640.
const DRAWABLES: u32 = 640;

/// Frames recorded per shape.
const SAMPLES: usize = 60;

/// The offscreen target's side.
const SIDE: u32 = 64;

/// Vertices shaded in the fetch measurement, which is about sixty terrain tiles' worth.
///
/// Large on purpose. A vertex-stage cost is per vertex, and the shapes above shade three vertices
/// a draw -- 1,920 in a frame -- which would put the answer inside the noise of queue submission.
/// A terrain mesh is 17,415 vertices a tile and a pitched z14 cover is tens of tiles, so this is
/// the order the question is actually asked at.
const VERTICES: u32 = 1 << 20;

/// The vertex stage under test, with and without the DEM read, identical in all else.
///
/// Issue tessella#324: a terrain style costs the Vulkan backend +38.9 ms a frame and GLES
/// +2.7 ms, and nothing has attributed the gap. A vertex-stage texture read is not known to be
/// expensive on Adreno -- only that one backend's path to it is -- and this is the controlled
/// pair that says which.
///
/// Written out here rather than assembled from a family, because the question has one variable
/// and a family would change three: a real `_terrain` variant also reads a different block and
/// binds a different set. The arithmetic is the real one -- `terrain_height.glsl`'s unpack
/// against `Encoding::Mapbox`'s own constants -- and the result feeds the position, so it cannot
/// be folded away.
const FETCH_CONTROL: &str = r"
struct Block {
    columns: array<vec4<f32>, 4>,
}
@group(0) @binding(0) var<storage, read> block: array<Block>;

struct In {
    @location(0) pos: vec2<f32>,
}
struct Out {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    let b = block[0];
    var out: Out;
    out.clip = b.columns[0] * in.pos.x + b.columns[1] * in.pos.y + b.columns[3];
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
";

/// See [`FETCH_CONTROL`]. The same, plus one `textureSampleLevel` a vertex.
const FETCH_SAMPLED: &str = r"
struct Block {
    columns: array<vec4<f32>, 4>,
}
@group(0) @binding(0) var<storage, read> block: array<Block>;
@group(0) @binding(1) var elevation: texture_2d<f32>;
@group(0) @binding(2) var elevation_sampler: sampler;

struct In {
    @location(0) pos: vec2<f32>,
}
struct Out {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vertex_main(in: In) -> Out {
    let b = block[0];
    // `sampling_within`'s pair for a 256-pixel DEM at extent 8192, and the Mapbox unpack.
    let uv = in.pos * 1.2112e-4 + vec2<f32>(3.876e-3, 3.876e-3);
    let channels = textureSampleLevel(elevation, elevation_sampler, uv, 0.0).rgb * 255.0;
    let meters = dot(channels, vec3<f32>(6553.6, 25.6, 0.1)) - 10000.0;
    var out: Out;
    out.clip = b.columns[0] * in.pos.x
        + b.columns[1] * in.pos.y
        + b.columns[2] * meters
        + b.columns[3];
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
";

/// The slot the verification puts its identity matrix in, and hands the draw as `firstInstance`.
///
/// Not zero, and not one: a driver that ignored the base instance entirely would read slot zero,
/// and one that truncated it to a byte or confused it with the vertex index would land somewhere
/// low. 613 is past all of that and inside the block buffer.
const SLOT: u32 = 613;

fn main() {
    let gpu = match Gpu::open() {
        Ok(gpu) => gpu,
        Err(why) => {
            // Not a failure: a machine with no Vulkan cannot answer this, and saying so beats a
            // panic that reads like a defect in the thing being measured.
            println!("no device to measure on: {why}");
            return;
        }
    };
    println!("device: {} ({} target)", gpu.name, gpu.tiling);
    // Before any timing: does the draw's `firstInstance` actually reach the shader as the slot?
    //
    // The whole design indexes its blocks by it, and a driver that dropped it would read slot zero
    // for every drawable -- one layer's paint for the whole map, a picture that looks deliberate.
    // Both boards this is run on have had base-instance quirks, so it is checked rather than
    // assumed, and checked here rather than in a test because it needs a device.
    match gpu.verify() {
        Ok(()) => println!("firstInstance reaches the shader as slot {SLOT}"),
        Err(why) => {
            println!("SLOT NOT DELIVERED: {why}");
            println!("the numbers below are still a measurement, but the design is not sound here");
        }
    }
    println!();
    println!("{DRAWABLES} drawables a frame, {SAMPLES} frames a shape\n");

    measure_shapes(&gpu);

    // tessella#324: what one vertex-stage DEM read costs, with everything else held equal.
    println!();
    match fetch_cost(&gpu) {
        Ok((control, sampled)) => {
            let delta = sampled.saturating_sub(control);
            println!(
                "vertex fetch over {VERTICES} vertices: control {:8.1} us  sampled {:8.1} us  \
                 delta {:8.1} us  per vertex {:6.3} ns",
                micros(control),
                micros(sampled),
                micros(delta),
                delta.as_secs_f64() * 1e9 / f64::from(VERTICES),
            );
        }
        Err(why) => println!("vertex fetch not measured: {why}"),
    }
}

/// The four recording shapes, in order.
fn measure_shapes(gpu: &Gpu) {
    for shape in Shape::ALL {
        let (record, submit) = gpu.measure(shape);
        let (r50, r95, rmax) = percentiles(record);
        let (s50, _, _) = percentiles(submit);
        println!(
            "{:<12} record {:8.1} us p50 {:8.1} p95 {:8.1} max   per draw {:7.4} us   \
             submit {:8.1} us p50",
            shape.name(),
            micros(r50),
            micros(r95),
            micros(rmax),
            micros(r50) / f64::from(DRAWABLES),
            micros(s50),
        );
    }
}

/// How much state each drawable carries. Additive, in the order listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Instance,
    Push,
    Buffers,
    Descriptors,
}

impl Shape {
    const ALL: [Self; 4] = [Self::Instance, Self::Push, Self::Buffers, Self::Descriptors];

    const fn name(self) -> &'static str {
        match self {
            Self::Instance => "instance",
            Self::Push => "push",
            Self::Buffers => "buffers",
            Self::Descriptors => "descriptors",
        }
    }

    /// Whether this shape writes the slot as a push constant per drawable.
    const fn pushes(self) -> bool {
        !matches!(self, Self::Instance)
    }

    /// Whether this shape rebinds the geometry per drawable.
    const fn rebinds_buffers(self) -> bool {
        matches!(self, Self::Buffers | Self::Descriptors)
    }

    /// Whether this shape rebinds the descriptor set per drawable.
    const fn rebinds_descriptors(self) -> bool {
        matches!(self, Self::Descriptors)
    }
}

fn percentiles(mut samples: Vec<Duration>) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
    (at(0.5), at(0.95), samples[samples.len() - 1])
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Everything needed to record one frame, held for the length of the run.
struct Gpu {
    name: String,
    /// Which tiling the readback target got, which is a property of the driver.
    tiling: &'static str,
    /// Kept for the fetch measurement, which allocates its own image and buffers.
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    view: vk::ImageView,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    held: [vk::Buffer; 5],
    offsets: [u64; 5],
    memory: vk::DeviceMemory,
    fence: vk::Fence,
    shader: [vk::ShaderModule; 2],
}

impl Gpu {
    #[allow(clippy::too_many_lines)]
    fn open() -> Result<Self, String> {
        // `TSL_VULKAN_LIB` names the driver to open instead of going through the loader.
        //
        // For an image whose loader and driver disagree about the interface version. On the
        // i.MX8M Plus the stock loader reports that VeriSilicon's ICD "supports Vulkan 1.3, but
        // only supports loader interface version 2", then inserts
        // `VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO` into `vkCreateDevice`'s `pNext` -- which
        // that driver does not skip. Opening the ICD directly puts nothing in the chain.
        let entry = match std::env::var("TSL_VULKAN_LIB") {
            Ok(path) => unsafe { ash::Entry::load_from(&path) }
                .map_err(|why| format!("no driver at {path}: {why}"))?,
            Err(_) => unsafe { ash::Entry::load() }.map_err(|why| format!("no loader: {why}"))?,
        };
        // Vulkan 1.1 where there is one, and 1.0 where there is not.
        //
        // 1.1 is wanted because naga emits `StorageBuffer` through
        // `SPV_KHR_storage_buffer_storage_class`, which is core from 1.1 and an extension that
        // has to be enabled before it. But asking for a version the implementation does not have
        // is not free: the i.MX8M Plus ships a driver the loader reports as "supports Vulkan 1.3,
        // but only supports loader interface version 2" and does not export
        // `vkEnumerateInstanceVersion`, which is an implementation that is 1.0 whatever its
        // manifest claims.
        //
        // So it is asked rather than assumed. `None` is the 1.0 answer -- the entry point that
        // would report a version is the one those implementations lack.
        let version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::make_api_version(0, 1, 0, 0));
        let wanted = if vk::api_version_minor(version) >= 1 {
            vk::make_api_version(0, 1, 1, 0)
        } else {
            vk::make_api_version(0, 1, 0, 0)
        };
        let app = vk::ApplicationInfo::default().api_version(wanted);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(|why| format!("no instance: {why}"))?;

        let physical = *unsafe { instance.enumerate_physical_devices() }
            .map_err(|why| format!("no devices: {why}"))?
            .first()
            .ok_or_else(|| "no physical device".to_string())?;
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = u32::try_from(
            families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .ok_or_else(|| "no graphics queue".to_string())?,
        )
        .map_err(|_| "queue family out of range".to_string())?;

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )
        }
        .map_err(|why| format!("no device: {why}"))?;
        let queue = unsafe { device.get_device_queue(family, 0) };

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|why| format!("no command pool: {why}"))?;
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|why| format!("no command buffer: {why}"))?[0];

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

        // The target, linear where the device will take one.
        //
        // The verification reads this back, and on Vivante an OPTIMAL (tiled) image keeps the
        // render pass's clear in tile status where a transfer read does not see it -- the drawn
        // triangle never appeared and the control read the clear color. A linear color attachment
        // has no tile status to be stale. Queried rather than assumed, because linear color
        // attachments are optional: the two desktop-class drivers here decline and keep OPTIMAL,
        // where their reads are correct anyway.
        let linear = unsafe {
            instance.get_physical_device_format_properties(physical, vk::Format::R8G8B8A8_UNORM)
        }
        .linear_tiling_features
        .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT);
        let tiling = if linear {
            vk::ImageTiling::LINEAR
        } else {
            vk::ImageTiling::OPTIMAL
        };
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: SIDE,
                        height: SIDE,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(tiling)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|why| format!("no image: {why}"))?;
        let needs = unsafe { device.get_image_memory_requirements(image) };
        let image_memory = allocate(
            &device,
            &memory_properties,
            needs,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe { device.bind_image_memory(image, image_memory, 0) }
            .map_err(|why| format!("image memory: {why}"))?;
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(|why| format!("no view: {why}"))?;

        let attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL)];
        let references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&references)];
        let pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
        }
        .map_err(|why| format!("no render pass: {why}"))?;
        let views = [view];
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&views)
                    .width(SIDE)
                    .height(SIDE)
                    .layers(1),
                None,
            )
        }
        .map_err(|why| format!("no framebuffer: {why}"))?;

        // The shader is the one this crate assembles, compiled the way the tests compile it. A
        // bench measuring a stand-in pipeline would measure the stand-in.
        let source = module(
            Surface::Plane,
            &[&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
            &BACKGROUND_SHADER,
            BACKGROUND_BODY,
        )
        .map_err(|why| format!("the family does not assemble: {why:?}"))?;
        let make_module = |stage, entry| -> Result<vk::ShaderModule, String> {
            let words = compile(&source, stage, entry)?;
            unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            }
            .map_err(|why| format!("no {entry} module: {why}"))
        };
        let vertex = make_module(naga::ShaderStage::Vertex, "vertex_main")?;
        let fragment = make_module(naga::ShaderStage::Fragment, "fragment_main")?;

        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stages),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stages),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|why| format!("no set layout: {why}"))?;

        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 4,
        }];
        let layouts = [set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }
        .map_err(|why| format!("no pipeline layout: {why}"))?;

        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        }];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(1),
                None,
            )
        }
        .map_err(|why| format!("no descriptor pool: {why}"))?;
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|why| format!("no descriptor set: {why}"))?[0];

        // Four buffers in one allocation: vertices, indexes, and the two blocks the family reads.
        // The contents are zeros; what is being timed is the recording.
        let (held, offsets, memory) = buffers(&device, &memory_properties)?;
        let drawable = [vk::DescriptorBufferInfo {
            buffer: held[2],
            offset: 0,
            range: vk::WHOLE_SIZE,
        }];
        let props = [vk::DescriptorBufferInfo {
            buffer: held[3],
            offset: 0,
            range: vk::WHOLE_SIZE,
        }];
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&drawable),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&props),
                ],
                &[],
            );
        }

        let pipeline = graphics(&device, pass, layout, vertex, fragment)?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|why| format!("no fence: {why}"))?;

        Ok(Self {
            name,
            tiling: if linear { "linear" } else { "optimal" },
            memory_properties,
            _entry: entry,
            instance,
            device,
            queue,
            pool,
            command,
            pipeline,
            layout,
            set,
            descriptor_pool,
            set_layout,
            pass,
            framebuffer,
            view,
            image,
            image_memory,
            held,
            offsets,
            memory,
            fence,
            shader: [vertex, fragment],
        })
    }

    /// Compiles one entry point of `source` into a module of its own.
    ///
    /// # Errors
    ///
    /// From the front end, the validator, the back end, or the driver.
    fn module(
        &self,
        source: &str,
        stage: naga::ShaderStage,
        entry: &str,
    ) -> Result<vk::ShaderModule, String> {
        let words = compile(source, stage, entry)?;
        unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|why| format!("no {entry} module: {why}"))
    }

    /// Puts `image` into the layout a sampled read needs.
    ///
    /// # Errors
    ///
    /// From the submission.
    fn transition(&self, image: vk::Image) -> Result<(), String> {
        unsafe {
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .map_err(|why| format!("reset: {why}"))?;
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|why| format!("begin: {why}"))?;
            let barrier = [vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            self.device
                .end_command_buffer(self.command)
                .map_err(|why| format!("end: {why}"))?;
        }
        self.submit_and_wait()
    }

    /// Records one draw of [`VERTICES`] vertices through `pipeline`.
    fn record_fetch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        vertices: vk::Buffer,
    ) {
        let clears = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let sets = [set];
        let buffers = [vertices];
        let offsets = [0u64];
        unsafe {
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .expect("reset");
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");
            self.device.cmd_begin_render_pass(
                self.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: SIDE,
                            height: SIDE,
                        },
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
            self.device
                .cmd_bind_pipeline(self.command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &sets,
                &[],
            );
            self.device
                .cmd_bind_vertex_buffers(self.command, 0, &buffers, &offsets);
            self.device.cmd_draw(self.command, VERTICES, 1, 0, 0);
            self.device.cmd_end_render_pass(self.command);
            self.device.end_command_buffer(self.command).expect("end");
        }
    }

    /// Submits the recorded buffer and waits for the fence.
    ///
    /// # Errors
    ///
    /// From the submission or the wait.
    fn submit_and_wait(&self) -> Result<(), String> {
        let commands = [self.command];
        let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .map_err(|why| format!("fence: {why}"))?;
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|why| format!("submit: {why}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|why| format!("wait: {why}"))
        }
    }

    /// Draws one triangle with `firstInstance` set to [`SLOT`] and checks the pixel it lands on.
    ///
    /// Slot zero holds a zero matrix and slot [`SLOT`] an identity, so the triangle covers the
    /// target when the shader read the slot it was handed and collapses to a point when it read
    /// zero instead. A green center pixel is the slot arriving; the clear color is it not.
    ///
    /// # The control comes first
    ///
    /// A blank target means "did not draw with slot [`SLOT`]'s matrix", and "did not draw at all"
    /// is one of the ways that happens -- so on its own it does not say the base instance was
    /// dropped. The same triangle is therefore drawn first with the identity in slot zero and
    /// `firstInstance` of zero, which differs from the real case in nothing but the mechanism
    /// under test. Blank there and the probe cannot draw on this device, which is not a finding
    /// about `firstInstance`.
    ///
    /// # Errors
    ///
    /// A description of what was read, and which of the two draws read it.
    fn verify(&self) -> Result<(), String> {
        // The control: identity in slot zero, drawn with slot zero. Green or the probe is broken.
        self.paint(0)?;
        if !Self::is_green(self.read_center()?) {
            return Err(format!(
                "the control drew nothing -- identity in slot zero, `firstInstance` of zero, and \
                 the center pixel is {:?}. This probe cannot draw on this device, so it says \
                 nothing either way about the base instance.",
                self.read_center()?
            ));
        }
        self.slot_check()
    }

    /// Whether a pixel is the opaque green the fragment stage writes.
    fn is_green(pixel: [u8; 4]) -> bool {
        pixel[1] > 200 && pixel[0] < 64
    }

    /// Writes the buffers so that `slot` holds the identity, draws with it, and submits.
    ///
    /// Every other slot holds a zero matrix, which sends every vertex to the origin.
    fn paint(&self, slot: u32) -> Result<(), String> {
        unsafe {
            let base = self
                .device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|why| format!("map: {why}"))?
                .cast::<u8>();

            // Written as little-endian bytes rather than through an `f32` pointer: the mapping is
            // page aligned and every offset is a multiple of the allocation's alignment, so a
            // cast would be sound -- and it is one `unsafe` a reader has to verify for no gain.
            let put = |offset: u64, values: &[f32]| {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                base.add(offset as usize)
                    .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
            };

            // A triangle over the whole of clip space, so an identity matrix covers the target.
            put(
                self.offsets[0],
                &[-1.0, -1.0, 0.0, 3.0, -1.0, 0.0, -1.0, 3.0, 0.0],
            );

            // Zeros everywhere, which send every vertex to the origin, and an identity in the
            // one slot under test. The block is 64 bytes of column-major matrix and nothing else.
            base.add(self.offsets[2] as usize)
                .write_bytes(0, (u64::from(SLOT) + 1) as usize * 64);
            put(
                self.offsets[2] + u64::from(slot) * 64,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            );

            // The paint: opaque green at `color`, one at `opacity` sixteen bytes in.
            put(self.offsets[3], &[0.0, 1.0, 0.0, 1.0, 1.0]);

            self.device.unmap_memory(self.memory);
        }

        self.record_verify(slot);
        let commands = [self.command];
        let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .map_err(|why| format!("fence: {why}"))?;
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|why| format!("submit: {why}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|why| format!("wait: {why}"))?;
        }

        Ok(())
    }

    /// The center pixel of the last frame drawn, from the readback buffer.
    ///
    /// # Errors
    ///
    /// When the memory cannot be mapped.
    fn read_center(&self) -> Result<[u8; 4], String> {
        let center = (SIDE as usize / 2) * SIDE as usize * 4 + (SIDE as usize / 2) * 4;
        let pixel = unsafe {
            let base = self
                .device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|why| format!("map: {why}"))?
                .cast::<u8>()
                .add(self.offsets[4] as usize);
            let mut pixel = [0u8; 4];
            base.add(center)
                .copy_to_nonoverlapping(pixel.as_mut_ptr(), 4);
            self.device.unmap_memory(self.memory);
            pixel
        };
        Ok(pixel)
    }

    /// The real case: the identity in [`SLOT`] alone, drawn with [`SLOT`] as the base instance.
    ///
    /// # Errors
    ///
    /// When the pixel says the shader read some other slot.
    fn slot_check(&self) -> Result<(), String> {
        self.paint(SLOT)?;
        let pixel = self.read_center()?;
        if Self::is_green(pixel) {
            Ok(())
        } else {
            Err(format!(
                "the center pixel is {pixel:?}, not green. The control drew, so this device \
                 does not deliver `firstInstance` to `instance_index`: every drawable would \
                 read slot zero's block."
            ))
        }
    }

    /// Records the one draw [`Self::paint`] reads back, with `slot` as the base instance.
    fn record_verify(&self, slot: u32) {
        let clears = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let sets = [self.set];
        let vertices = [self.held[0]];
        let offsets = [0u64];
        unsafe {
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .expect("reset");
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");
            self.device.cmd_begin_render_pass(
                self.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: SIDE,
                            height: SIDE,
                        },
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &sets,
                &[],
            );
            self.device
                .cmd_bind_vertex_buffers(self.command, 0, &vertices, &offsets);
            // Three vertices, one instance, and the slot as the base instance.
            self.device.cmd_draw(self.command, 3, 1, 0, slot);
            self.device.cmd_end_render_pass(self.command);

            // The render pass leaves the image in GENERAL, which a copy may read from; the
            // barrier is for the write finishing, not for the layout.
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: SIDE,
                    height: SIDE,
                    depth: 1,
                })];
            self.device.cmd_copy_image_to_buffer(
                self.command,
                self.image,
                vk::ImageLayout::GENERAL,
                self.held[4],
                &region,
            );
            self.device.end_command_buffer(self.command).expect("end");
        }
    }

    /// Records `SAMPLES` frames of one shape, returning record and submit times.
    fn measure(&self, shape: Shape) -> (Vec<Duration>, Vec<Duration>) {
        let mut records = Vec::with_capacity(SAMPLES);
        let mut submits = Vec::with_capacity(SAMPLES);
        let commands = [self.command];

        for _ in 0..SAMPLES {
            let at = Instant::now();
            self.record(shape);
            records.push(at.elapsed());

            let submitted = Instant::now();
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            unsafe {
                self.device
                    .reset_fences(&[self.fence])
                    .expect("reset fence");
                self.device
                    .queue_submit(self.queue, &submit, self.fence)
                    .expect("submit");
                self.device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .expect("wait");
            }
            submits.push(submitted.elapsed());
        }
        (records, submits)
    }

    /// Records one frame of `shape` into the command buffer.
    fn record(&self, shape: Shape) {
        let clears = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let sets = [self.set];
        let vertices = [self.held[0]];
        let offsets = [0u64];
        unsafe {
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .expect("reset");
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");
            self.device.cmd_begin_render_pass(
                self.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: SIDE,
                            height: SIDE,
                        },
                    })
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            // Bound once, which is what a merged frame does: the arena holds every drawable of
            // a family in one buffer and one set serves all of them. Every shape starts here
            // and the dearer ones add to it, so a row's cost over the one above is exactly
            // what it added.
            self.device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &sets,
                &[],
            );
            self.device
                .cmd_bind_vertex_buffers(self.command, 0, &vertices, &offsets);
            self.device
                .cmd_bind_index_buffer(self.command, self.held[1], 0, vk::IndexType::UINT16);
            if !shape.pushes() {
                // The slot rides in `firstInstance` instead, so the range is written once to
                // keep the draw valid rather than per drawable to carry anything.
                self.device.cmd_push_constants(
                    self.command,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &0u32.to_ne_bytes(),
                );
            }

            for drawable in 0..DRAWABLES {
                if shape.rebinds_descriptors() {
                    self.device.cmd_bind_descriptor_sets(
                        self.command,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        0,
                        &sets,
                        &[],
                    );
                }
                if shape.rebinds_buffers() {
                    self.device
                        .cmd_bind_vertex_buffers(self.command, 0, &vertices, &offsets);
                    self.device.cmd_bind_index_buffer(
                        self.command,
                        self.held[1],
                        0,
                        vk::IndexType::UINT16,
                    );
                }
                if shape.pushes() {
                    self.device.cmd_push_constants(
                        self.command,
                        self.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        &drawable.to_ne_bytes(),
                    );
                    self.device.cmd_draw_indexed(self.command, 3, 1, 0, 0, 0);
                } else {
                    self.device
                        .cmd_draw_indexed(self.command, 3, 1, 0, 0, drawable);
                }
            }

            self.device.cmd_end_render_pass(self.command);
            self.device.end_command_buffer(self.command).expect("end");
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_pipeline(self.pipeline, None);
            for module in self.shader {
                self.device.destroy_shader_module(module, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.pass, None);
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.image_memory, None);
            for buffer in self.held {
                self.device.destroy_buffer(buffer, None);
            }
            self.device.free_memory(self.memory, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Times one draw of [`VERTICES`] vertices with and without a DEM read in the vertex stage.
///
/// Returns the two submit-to-fence medians. Everything but the read is held equal: one pipeline a
/// variant from the pair above, the same vertex buffer, the same block, the same target, and a
/// placement that puts every vertex outside the clip volume so neither variant rasterizes
/// anything. The difference is the fetch.
///
/// # Errors
///
/// Whatever could not be created, which on a driver missing something is the useful answer.
#[allow(clippy::too_many_lines)]
fn fetch_cost(gpu: &Gpu) -> Result<(Duration, Duration), String> {
    let device = &gpu.device;

    // One storage block, one sampled image, one sampler.
    let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(stages),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(stages),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(stages),
    ];
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(|why| format!("fetch set layout: {why}"))?;
    let layouts = [set_layout];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
            None,
        )
    }
    .map_err(|why| format!("fetch pipeline layout: {why}"))?;

    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: 1,
        },
    ];
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1),
            None,
        )
    }
    .map_err(|why| format!("fetch descriptor pool: {why}"))?;
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts),
        )
    }
    .map_err(|why| format!("fetch descriptor set: {why}"))?[0];

    // The elevation: a DEM's stored size for a 256-pixel tile, which is what the sampling pair
    // above was computed for. Contents do not matter; its dimensions and format do.
    let stride = 258u32;
    let elevation = unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: stride,
                    height: stride,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }
    .map_err(|why| format!("elevation image: {why}"))?;
    let needs = unsafe { device.get_image_memory_requirements(elevation) };
    let elevation_memory = allocate(
        device,
        &gpu.memory_properties,
        needs,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    unsafe { device.bind_image_memory(elevation, elevation_memory, 0) }
        .map_err(|why| format!("elevation memory: {why}"))?;
    let elevation_view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(elevation)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
    }
    .map_err(|why| format!("elevation view: {why}"))?;
    // Linear, which is what the elevation binds in the renderer.
    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
    }
    .map_err(|why| format!("sampler: {why}"))?;

    // The vertices, and a block whose placement sends every one of them off screen.
    let vertex_bytes = u64::from(VERTICES) * 8;
    let make = |size: u64, usage: vk::BufferUsageFlags| -> Result<vk::Buffer, String> {
        unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|why| format!("fetch buffer: {why}"))
    };
    let vertices = make(vertex_bytes, vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let block = make(256, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let pair = [vertices, block];
    let needs: Vec<vk::MemoryRequirements> = pair
        .iter()
        .map(|b| unsafe { device.get_buffer_memory_requirements(*b) })
        .collect();
    let align = needs.iter().map(|n| n.alignment).max().unwrap_or(256);
    let total = needs.iter().map(|n| n.size.div_ceil(align) * align).sum();
    let bits = needs.iter().fold(u32::MAX, |a, n| a & n.memory_type_bits);
    let memory = allocate(
        device,
        &gpu.memory_properties,
        vk::MemoryRequirements {
            size: total,
            alignment: align,
            memory_type_bits: bits,
        },
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mut offsets = [0u64; 2];
    let mut at = 0u64;
    for (index, (buffer, need)) in pair.iter().zip(&needs).enumerate() {
        unsafe { device.bind_buffer_memory(*buffer, memory, at) }
            .map_err(|why| format!("fetch bind: {why}"))?;
        offsets[index] = at;
        at += need.size.div_ceil(align) * align;
    }

    unsafe {
        let base = device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|why| format!("fetch map: {why}"))?
            .cast::<u8>();
        // Positions spread over a tile's coordinate range, so the sampling pair reaches the
        // whole image and no cache holds the answer.
        let mut bytes = Vec::with_capacity(vertex_bytes as usize);
        for index in 0..VERTICES {
            #[allow(clippy::cast_precision_loss)]
            let x = (index % 8192) as f32;
            #[allow(clippy::cast_precision_loss)]
            let y = ((index / 8192) % 8192) as f32;
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        base.add(offsets[0] as usize)
            .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());

        // Columns that place every vertex far outside the clip volume, so neither variant
        // rasterizes a fragment and the difference is the vertex stage alone.
        let columns: [f32; 16] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 64.0, 64.0, 0.5, 1.0,
        ];
        let mut block_bytes = Vec::with_capacity(64);
        for value in columns {
            block_bytes.extend_from_slice(&value.to_le_bytes());
        }
        base.add(offsets[1] as usize)
            .copy_from_nonoverlapping(block_bytes.as_ptr(), block_bytes.len());
        device.unmap_memory(memory);
    }

    let buffer_info = [vk::DescriptorBufferInfo {
        buffer: block,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let image_info = [vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: elevation_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }];
    let sampler_info = [vk::DescriptorImageInfo {
        sampler,
        image_view: vk::ImageView::null(),
        image_layout: vk::ImageLayout::UNDEFINED,
    }];
    unsafe {
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info),
            ],
            &[],
        );
    }

    // The image has to be readable before it is sampled.
    gpu.transition(elevation)?;

    let mut built = Vec::new();
    for source in [FETCH_CONTROL, FETCH_SAMPLED] {
        let vertex = gpu.module(source, naga::ShaderStage::Vertex, "vertex_main")?;
        let fragment = gpu.module(source, naga::ShaderStage::Fragment, "fragment_main")?;
        built.push((
            fetch_pipeline(device, gpu.pass, layout, vertex, fragment)?,
            vertex,
            fragment,
        ));
    }

    let mut medians = Vec::new();
    for (pipeline, _, _) in &built {
        let mut timed = Vec::with_capacity(FETCH_SAMPLES);
        for _ in 0..FETCH_SAMPLES {
            gpu.record_fetch(*pipeline, layout, set, vertices);
            let at = Instant::now();
            gpu.submit_and_wait()?;
            timed.push(at.elapsed());
        }
        medians.push(percentiles(timed).0);
    }

    unsafe {
        let _ = device.device_wait_idle();
        for (pipeline, vertex, fragment) in built {
            device.destroy_pipeline(pipeline, None);
            device.destroy_shader_module(vertex, None);
            device.destroy_shader_module(fragment, None);
        }
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_sampler(sampler, None);
        device.destroy_image_view(elevation_view, None);
        device.destroy_image(elevation, None);
        device.free_memory(elevation_memory, None);
        device.destroy_buffer(vertices, None);
        device.destroy_buffer(block, None);
        device.free_memory(memory, None);
    }
    Ok((medians[0], medians[1]))
}

/// Frames timed per variant in the fetch measurement.
const FETCH_SAMPLES: usize = 20;

/// A pipeline for the fetch pair: one `float32x2` attribute and nothing else.
fn fetch_pipeline(
    device: &ash::Device,
    pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
) -> Result<vk::Pipeline, String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"vertex_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"fragment_main"),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 8,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32_SFLOAT,
        offset: 0,
    }];
    let input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    #[allow(clippy::cast_precision_loss)]
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIDE as f32,
        height: SIDE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIDE,
            height: SIDE,
        },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blends = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let create = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(pass)
        .subpass(0)];
    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create, None) }
        .map(|pipelines| pipelines[0])
        .map_err(|(_, why)| format!("fetch pipeline: {why}"))
}

/// Compiles one entry point of the assembled module to SPIR-V words.
///
/// One module an entry point, not one module with both. A single module carrying `vertex_main`
/// and `fragment_main` is legal Vulkan and is what RADV and V3DV take; Adreno refuses the
/// pipeline built from it with `VK_ERROR_UNKNOWN` and prints "Pipeline create failed" and no
/// reason. Splitting them is what `naga`'s `PipelineOptions` is for.
fn compile(source: &str, stage: naga::ShaderStage, entry: &str) -> Result<Vec<u32>, String> {
    let parsed = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    let options = naga::back::spv::Options {
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.to_string(),
    };
    naga::back::spv::write_vec(&parsed, &info, &options, Some(&pipeline))
        .map_err(|why| format!("spirv: {why:?}"))
}

/// A memory type with the wanted properties, allocated for `needs`.
fn allocate(
    device: &ash::Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    needs: vk::MemoryRequirements,
    wanted: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, String> {
    let index = (0..properties.memory_type_count)
        .find(|index| {
            needs.memory_type_bits & (1 << index) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(wanted)
        })
        .ok_or_else(|| format!("no memory type for {wanted:?}"))?;
    unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(index),
            None,
        )
    }
    .map_err(|why| format!("allocation: {why}"))
}

/// Vertices, indexes, the two blocks and a readback target, in one allocation.
///
/// The offsets come back because the verification writes the blocks through them: slot zero gets a
/// zero matrix and slot `SLOT` an identity, which is what makes a drawn pixel mean the shader read
/// the slot it was given.
fn buffers(
    device: &ash::Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<([vk::Buffer; 5], [u64; 5], vk::DeviceMemory), String> {
    let make = |size: u64, usage: vk::BufferUsageFlags| -> Result<vk::Buffer, String> {
        unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|why| format!("buffer: {why}"))
    };
    let held = [
        make(4096, vk::BufferUsageFlags::VERTEX_BUFFER)?,
        make(4096, vk::BufferUsageFlags::INDEX_BUFFER)?,
        make(1 << 16, vk::BufferUsageFlags::STORAGE_BUFFER)?,
        make(4096, vk::BufferUsageFlags::STORAGE_BUFFER)?,
        make(
            u64::from(SIDE) * u64::from(SIDE) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?,
    ];

    let needs: Vec<vk::MemoryRequirements> = held
        .iter()
        .map(|b| unsafe { device.get_buffer_memory_requirements(*b) })
        .collect();
    let align = needs.iter().map(|n| n.alignment).max().unwrap_or(256);
    let size = needs.iter().map(|n| n.size.div_ceil(align) * align).sum();
    let bits = needs.iter().fold(u32::MAX, |a, n| a & n.memory_type_bits);
    let memory = allocate(
        device,
        properties,
        vk::MemoryRequirements {
            size,
            alignment: align,
            memory_type_bits: bits,
        },
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mut offsets = [0u64; 5];
    let mut at = 0u64;
    for (index, (buffer, need)) in held.iter().zip(&needs).enumerate() {
        unsafe { device.bind_buffer_memory(*buffer, memory, at) }
            .map_err(|why| format!("bind: {why}"))?;
        offsets[index] = at;
        at += need.size.div_ceil(align) * align;
    }
    Ok((held, offsets, memory))
}

/// The pipeline, with the family's own vertex input.
fn graphics(
    device: &ash::Device,
    pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
) -> Result<vk::Pipeline, String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"vertex_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"fragment_main"),
    ];
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 12,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    }];
    let input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    #[allow(clippy::cast_precision_loss)]
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIDE as f32,
        height: SIDE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIDE,
            height: SIDE,
        },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blends = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);

    let create = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(pass)
        .subpass(0)];
    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &create, None) }
        .map(|pipelines| pipelines[0])
        .map_err(|(_, why)| format!("no pipeline: {why}"))
}
