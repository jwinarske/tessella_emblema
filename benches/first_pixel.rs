//! The first pixel: one `fill` drawable, through the module this crate generates.
//!
//! Every other test here reads the shader. This one runs it. A module can compile to valid SPIR-V,
//! satisfy every pin, and still draw nothing — because the pipeline fetched an attribute at a
//! format the shader reads differently, or because the block the producer wrote and the block the
//! shader declared agree about offsets but not about which binding they arrive at. None of that is
//! visible from the source.
//!
//! So this draws a triangle over the viewport with a known color and reads the center pixel back.
//! It is not a benchmark; `cargo bench --bench first_pixel` runs it because a bench is the target
//! that may need a GPU. CI has none, which is why this is not a test.
//!
//! # What a pass here proves
//!
//! That `module(Surface::Plane, …)`'s output builds a pipeline on a real driver; that the vertex
//! input built from the ABI's declared types delivers what the body reads; that a block written at
//! the ABI's own offsets is read back by the generated WGSL struct; that `place` and `mix_color`
//! compute what they are believed to; and that `instance_index` reaches `ubo_index`.
//!
//! # What a pass here does not prove
//!
//! Anything about any other family, and nothing about a tiler. `TSL_VULKAN_LIB` names a driver to
//! open directly, for an image whose loader and driver disagree — see `draw_cost.rs`.

use ash::vk;
use tessella_capture_abi::generated::shader_attributes::FILL_SHADER;
use tessella_capture_abi::generated::ubo_layouts::{
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, UboLayout,
};
use tessella_emblema::device::vertex_format;
use tessella_emblema::shaders::{FILL_BODY, module};
use tessella_emblema::surface::Surface;

/// The target's edge, in pixels. Small: one pixel is read and the rest is margin.
const SIDE: u32 = 32;

/// The color the drawable asks for, as eight-bit channels.
const WANTED: [u8; 4] = [255, 0, 0, 255];

fn main() {
    match run() {
        Ok(()) => println!("first pixel: ok"),
        Err(why) => {
            eprintln!("first pixel: {why}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let source = module(
        Surface::Plane,
        &[&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
        &FILL_SHADER,
        &[],
        FILL_BODY,
    )
    .map_err(|why| format!("the fill module does not assemble: {why:?}"))?;
    let words = compile(&source)?;

    let gpu = Gpu::open()?;

    // The control first. A cleared pass has to read black, or a green answer below could be
    // whatever the mapped memory happened to hold.
    gpu.draw(None)?;
    let cleared = gpu.center()?;
    if cleared != [0, 0, 0, 0] {
        return Err(format!(
            "a cleared pass reads {cleared:?}, so the readback is not showing the pass"
        ));
    }

    gpu.draw(Some(&words))?;
    let drawn = gpu.center()?;
    if drawn != WANTED {
        return Err(format!("the center reads {drawn:?}, wanted {WANTED:?}"));
    }
    println!("  device:  {}", gpu.name);
    println!("  center:  {drawn:?}");
    Ok(())
}

/// WGSL to SPIR-V, the same way `tests/shaders.rs` does it.
fn compile(source: &str) -> Result<Vec<u32>, String> {
    let parsed = naga::front::wgsl::parse_str(source)
        .map_err(|why| format!("wgsl: {}", why.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|why| format!("validation: {why:?}"))?;
    naga::back::spv::write_vec(
        &parsed,
        &info,
        &naga::back::spv::Options::default(),
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Vertex,
            entry_point: "vertex_main".to_string(),
        }),
    )
    .map_err(|why| format!("spirv: {why}"))
    .map(|_| ())
    .and_then(|()| {
        naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
            .map_err(|why| format!("spirv: {why}"))
    })
}

/// A color packed the way `unpack_color` reads it: two channels to a float, scaled by 255.
///
/// `unpack_color` splits each component into a high and a low byte and divides by 255, so the two
/// floats carry four channels. Both endpoints are set to the same color, which makes the result
/// independent of `color_t` -- the interpolation is not what this is checking.
fn packed_color(rgba: [u8; 4]) -> [f32; 4] {
    let lo = f32::from(rgba[0]) * 256.0 + f32::from(rgba[1]);
    let hi = f32::from(rgba[2]) * 256.0 + f32::from(rgba[3]);
    [lo, hi, lo, hi]
}

/// A drawable block, written at the offsets the ABI declares rather than at counted ones.
///
/// The point of going through the table: a block written by hand would agree with the WGSL struct
/// only because the same person wrote both. Taking the offsets from `FILL_DRAWABLE_UBO` makes this
/// a comparison between the producer's layout and the shader's.
fn drawable_block() -> Vec<u8> {
    let mut bytes = vec![0u8; FILL_DRAWABLE_UBO.stride as usize];
    let mut put = |name: &str, values: &[f32]| {
        let field = field(&FILL_DRAWABLE_UBO, name);
        let at = field.offset as usize;
        for (index, value) in values.iter().enumerate() {
            let start = at + index * 4;
            bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
    };
    // The identity, so a tile position is already a clip position and nothing here depends on a
    // projection this probe would also have to be right about.
    put(
        "matrix",
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    put("color_t", &[0.0]);
    put("opacity_t", &[0.0]);
    bytes
}

fn field(
    layout: &UboLayout,
    name: &str,
) -> &'static tessella_capture_abi::generated::ubo_layouts::UboField {
    layout
        .fields
        .iter()
        .find(|field| field.name == name)
        .unwrap_or_else(|| panic!("{} has no {name}", layout.name))
}

/// A device, a linear-tiled target, and everything needed to draw once into it.
///
/// Linear tiling and host-visible memory, so the readback is a map rather than a staging copy and
/// a blit this probe would also have to get right. Every target here supports it for a sampled
/// color attachment at this size; a device that does not would fail at image creation, which is
/// where it should.
struct Gpu {
    name: String,
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    view: vk::ImageView,
    pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    buffers: Vec<vk::Buffer>,
    buffer_memory: vk::DeviceMemory,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    row: u64,
    offset: u64,
}

/// What goes in the one host-visible allocation, in order: three vertex buffers then two blocks.
const PARTS: usize = 5;

impl Gpu {
    #[allow(clippy::too_many_lines)]
    fn open() -> Result<Self, String> {
        let entry = match std::env::var("TSL_VULKAN_LIB") {
            Ok(path) => unsafe { ash::Entry::load_from(&path) }
                .map_err(|why| format!("no driver at {path}: {why}"))?,
            Err(_) => unsafe { ash::Entry::load() }.map_err(|why| format!("no loader: {why}"))?,
        };
        // 1.1 where there is one: naga emits `StorageBuffer` through
        // `SPV_KHR_storage_buffer_storage_class`, core from 1.1. Asked rather than assumed, for
        // the implementations that are 1.0 whatever their manifest says.
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
        let name = properties.device_name_as_c_str().map_or_else(
            |_| "unnamed".to_string(),
            |name| name.to_string_lossy().into_owned(),
        );

        let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| "no graphics queue".to_string())? as u32;
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
                    .command_buffer_count(1),
            )
        }
        .map_err(|why| format!("no command buffer: {why}"))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|why| format!("no fence: {why}"))?;

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
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|why| format!("no image: {why}"))?;
        let needs = unsafe { device.get_image_memory_requirements(image) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let host = pick(
            &memory_properties,
            needs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| "no host-visible memory for a linear attachment".to_string())?;
        let image_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(host),
                None,
            )
        }
        .map_err(|why| format!("no image memory: {why}"))?;
        unsafe { device.bind_image_memory(image, image_memory, 0) }
            .map_err(|why| format!("image not bound: {why}"))?;
        let subresource = unsafe {
            device.get_image_subresource_layout(
                image,
                vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .array_layer(0),
            )
        };

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(|why| format!("no view: {why}"))?;

        let attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL)];
        let references = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
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

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..2)
            .map(|slot| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(slot)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|why| format!("no descriptor layout: {why}"))?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|why| format!("no descriptor pool: {why}"))?;
        let layouts = [descriptor_layout];
        let descriptors = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|why| format!("no descriptor set: {why}"))?[0];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }
        .map_err(|why| format!("no pipeline layout: {why}"))?;

        let mut gpu = Self {
            name,
            _entry: entry,
            instance,
            device,
            queue,
            pool,
            command,
            fence,
            image,
            image_memory,
            view,
            pass,
            framebuffer,
            buffers: Vec::new(),
            buffer_memory: vk::DeviceMemory::null(),
            descriptor_layout,
            descriptor_pool,
            descriptors,
            layout,
            row: subresource.row_pitch,
            offset: subresource.offset,
        };
        gpu.fill_buffers(physical)?;
        Ok(gpu)
    }

    /// Three vertex buffers and two blocks, in one host-visible allocation.
    ///
    /// The vertex buffers' contents are built per attribute from `FILL_SHADER`, so each one holds
    /// exactly what its declared type says and the pipeline below describes it from the same
    /// table.
    fn fill_buffers(&mut self, physical: vk::PhysicalDevice) -> Result<(), String> {
        // The triangle covers the viewport: with the identity matrix a tile position is a clip
        // position, so these three corners put the whole target inside the triangle.
        let positions: [i16; 6] = [-1, -1, 3, -1, -1, 3];
        let color = packed_color(WANTED);
        let colors: Vec<f32> = (0..3).flat_map(|_| color).collect();
        let opacities: Vec<f32> = (0..3).flat_map(|_| [1.0f32, 1.0]).collect();

        let contents: [Vec<u8>; PARTS] = [
            positions.iter().flat_map(|v| v.to_le_bytes()).collect(),
            colors.iter().flat_map(|v| v.to_le_bytes()).collect(),
            opacities.iter().flat_map(|v| v.to_le_bytes()).collect(),
            drawable_block(),
            vec![0u8; FILL_EVALUATED_PROPS_UBO.stride as usize],
        ];

        let mut buffers = Vec::with_capacity(PARTS);
        for (index, bytes) in contents.iter().enumerate() {
            let usage = if index < 3 {
                vk::BufferUsageFlags::VERTEX_BUFFER
            } else {
                vk::BufferUsageFlags::STORAGE_BUFFER
            };
            let buffer = unsafe {
                self.device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes.len() as u64)
                        .usage(usage),
                    None,
                )
            }
            .map_err(|why| format!("no buffer {index}: {why}"))?;
            buffers.push(buffer);
        }

        // One allocation, each buffer at its own aligned offset.
        let mut offsets = [0u64; PARTS];
        let mut total = 0u64;
        let mut bits = u32::MAX;
        for (index, buffer) in buffers.iter().enumerate() {
            let needs = unsafe { self.device.get_buffer_memory_requirements(*buffer) };
            let align = needs.alignment.max(1);
            total = total.div_ceil(align) * align;
            offsets[index] = total;
            total += needs.size;
            bits &= needs.memory_type_bits;
        }
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(physical)
        };
        let host = pick(
            &memory_properties,
            bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| "no host-visible memory for the buffers".to_string())?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(total)
                    .memory_type_index(host),
                None,
            )
        }
        .map_err(|why| format!("no buffer memory: {why}"))?;

        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|why| format!("not mapped: {why}"))?
        .cast::<u8>();
        for (index, buffer) in buffers.iter().enumerate() {
            unsafe {
                self.device
                    .bind_buffer_memory(*buffer, memory, offsets[index])
            }
            .map_err(|why| format!("buffer {index} not bound: {why}"))?;
            let bytes = &contents[index];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    mapped.add(offsets[index] as usize),
                    bytes.len(),
                );
            }
        }
        unsafe { self.device.unmap_memory(memory) };

        let writes: Vec<vk::DescriptorBufferInfo> = (3..PARTS)
            .map(|index| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffers[index])
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let updates: Vec<vk::WriteDescriptorSet<'_>> = writes
            .iter()
            .enumerate()
            .map(|(slot, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptors)
                    .dst_binding(slot as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&updates, &[]) };

        self.buffers = buffers;
        self.buffer_memory = memory;
        Ok(())
    }

    /// Clears the target, and draws the triangle when given a module.
    ///
    /// `None` is the control: the same pass with no draw in it, which has to read black.
    fn draw(&self, words: Option<&[u32]>) -> Result<(), String> {
        let pipeline = words.map(|words| self.pipeline(words)).transpose()?;
        unsafe {
            self.device
                .begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
        }
        .map_err(|why| format!("not recording: {why}"))?;

        let clears = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }];
        unsafe {
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
            if let Some((pipeline, _)) = pipeline {
                self.device.cmd_bind_pipeline(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[self.descriptors],
                    &[],
                );
                let vertex: Vec<vk::Buffer> = self.buffers[..3].to_vec();
                self.device
                    .cmd_bind_vertex_buffers(self.command, 0, &vertex, &[0, 0, 0]);
                // `firstInstance` is the drawable's slot, which the body reads as `ubo_index`.
                self.device.cmd_draw(self.command, 3, 1, 0, 0);
            }
            self.device.cmd_end_render_pass(self.command);
            self.device
                .end_command_buffer(self.command)
                .map_err(|why| format!("not recorded: {why}"))?;
        }

        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .map_err(|why| format!("not submitted: {why}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|why| format!("not finished: {why}"))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|why| format!("fence not reset: {why}"))?;
        }
        if let Some((pipeline, modules)) = pipeline {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                for shader in modules {
                    self.device.destroy_shader_module(shader, None);
                }
            }
        }
        Ok(())
    }

    /// A pipeline for the module, with the vertex input taken from the ABI's table.
    ///
    /// One binding per attribute, numbered by the table's own `binding` -- which is also the
    /// `@location` the generated `In` struct gives it, so the two cannot drift apart here.
    fn pipeline(&self, words: &[u32]) -> Result<(vk::Pipeline, Vec<vk::ShaderModule>), String> {
        // Adreno rejects a module with two entry points, so each stage gets its own.
        let vertex = self.shader(words)?;
        let fragment = self.shader(words)?;

        let mut bindings = Vec::new();
        let mut attributes = Vec::new();
        for attribute in &FILL_SHADER {
            let format = vertex_format(attribute.declared)
                .ok_or_else(|| format!("{} has no vertex format", attribute.name))?;
            let slot = u32::try_from(attribute.binding)
                .map_err(|_| format!("{} binds at {}", attribute.name, attribute.binding))?;
            bindings.push(
                vk::VertexInputBindingDescription::default()
                    .binding(slot)
                    .stride(stride_of(format))
                    .input_rate(vk::VertexInputRate::VERTEX),
            );
            attributes.push(
                vk::VertexInputAttributeDescription::default()
                    .location(slot)
                    .binding(slot)
                    .format(format)
                    .offset(0),
            );
        }

        let entry_vertex = c"vertex_main";
        let entry_fragment = c"fragment_main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(entry_vertex),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(entry_fragment),
        ];
        let input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let side = f64::from(SIDE) as f32;
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: side,
            height: side,
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
            // No culling: this probe is not checking which way the triangle winds.
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Blending off, so the pixel read back is what the fragment stage returned rather than
        // what it returned composited over the clear.
        let blends = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);

        let create = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .layout(self.layout)
            .render_pass(self.pass)
            .subpass(0)];
        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &create, None)
        }
        .map_err(|(_, why)| format!("no pipeline: {why}"))?;
        Ok((pipelines[0], vec![vertex, fragment]))
    }

    fn shader(&self, words: &[u32]) -> Result<vk::ShaderModule, String> {
        unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        }
        .map_err(|why| format!("no shader module: {why}"))
    }

    /// The center pixel, read straight out of the linear image.
    fn center(&self) -> Result<[u8; 4], String> {
        let mapped = unsafe {
            self.device.map_memory(
                self.image_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|why| format!("image not mapped: {why}"))?
        .cast::<u8>();
        let at = self.offset + u64::from(SIDE / 2) * self.row + u64::from(SIDE / 2) * 4;
        let mut pixel = [0u8; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(mapped.add(at as usize), pixel.as_mut_ptr(), 4);
            self.device.unmap_memory(self.image_memory);
        }
        Ok(pixel)
    }
}

/// How many bytes one vertex of a format occupies, read from the format's own name.
///
/// A table of my own would be a second opinion about what `R16G16_SINT` means, and would also let
/// a wrong format fail here -- loudly, in this probe's own code -- instead of reaching the draw
/// and producing the wrong pixel, which is the failure worth seeing.
fn stride_of(format: vk::Format) -> u32 {
    let name = format!("{format:?}");
    let channels = name
        .rsplit_once('_')
        .map_or_else(|| panic!("{name} has no suffix"), |(channels, _)| channels);
    let mut bits = 0u32;
    let mut width = String::new();
    for ch in channels.chars().chain(std::iter::once('R')) {
        if ch.is_ascii_digit() {
            width.push(ch);
            continue;
        }
        if !width.is_empty() {
            bits += width.parse::<u32>().unwrap_or_else(|_| panic!("{name}"));
            width.clear();
        }
    }
    assert!(
        bits.is_multiple_of(8),
        "{name} is {bits} bits, which is not whole bytes"
    );
    bits / 8
}

/// A memory type satisfying the requirements and carrying the properties.
fn pick(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        bits & (1 << index) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(wanted)
    })
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for buffer in &self.buffers {
                self.device.destroy_buffer(*buffer, None);
            }
            if self.buffer_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.buffer_memory, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.pass, None);
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.image_memory, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
