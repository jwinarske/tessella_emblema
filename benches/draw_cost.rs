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

/// The offscreen target's side. Nothing reads it; a draw has to have somewhere to go.
const SIDE: u32 = 64;

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
    println!("device: {}", gpu.name);
    println!("{DRAWABLES} drawables a frame, {SAMPLES} frames a shape\n");

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
    held: [vk::Buffer; 4],
    memory: vk::DeviceMemory,
    fence: vk::Fence,
    shader: vk::ShaderModule,
}

impl Gpu {
    #[allow(clippy::too_many_lines)]
    fn open() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(|why| format!("no loader: {why}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));
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

        // The target. Nothing samples it or reads it back.
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
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
        let words = compile(&source)?;
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|why| format!("no shader module: {why}"))?;

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
        let (held, memory) = buffers(&device, &memory_properties)?;
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

        let pipeline = graphics(&device, pass, layout, shader)?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|why| format!("no fence: {why}"))?;

        Ok(Self {
            name,
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
            memory,
            fence,
            shader,
        })
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
            self.device.destroy_shader_module(self.shader, None);
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

/// Compiles the assembled module to SPIR-V words, as `tests/shaders.rs` does.
fn compile(source: &str) -> Result<Vec<u32>, String> {
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
    naga::back::spv::write_vec(&parsed, &info, &options, None)
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

/// Vertices, indexes, and the two blocks, in one allocation.
fn buffers(
    device: &ash::Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<([vk::Buffer; 4], vk::DeviceMemory), String> {
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
    let mut at = 0u64;
    for (buffer, need) in held.iter().zip(&needs) {
        unsafe { device.bind_buffer_memory(*buffer, memory, at) }
            .map_err(|why| format!("bind: {why}"))?;
        at += need.size.div_ceil(align) * align;
    }
    Ok((held, memory))
}

/// The pipeline, with the family's own vertex input.
fn graphics(
    device: &ash::Device,
    pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
) -> Result<vk::Pipeline, String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"vertex_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
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
