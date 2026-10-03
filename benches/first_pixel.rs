//! The readback oracle: families drawn, and their pixels checked.
//!
//! Every test here reads the shader. This one runs it. A module can compile to valid SPIR-V,
//! satisfy every pin, and still draw the wrong thing — because the pipeline fetched an attribute at
//! a format the shader reads differently, or because a block the producer wrote and the block the
//! shader declared agree about offsets but not about which binding they arrive at. None of that is
//! visible from the source.
//!
//! So each case below draws into a 32x32 target with inputs chosen to make one pixel's value
//! predictable, and checks that pixel. `cargo bench --bench first_pixel` runs it; a bench rather
//! than a test because it needs a GPU and CI has none.
//!
//! # What a pass proves
//!
//! That the module builds a pipeline on a real driver; that the vertex input built from the ABI's
//! declared types delivers what the body reads; that a block written at the ABI's own offsets is
//! read back by the generated WGSL struct; and that `instance_index` reaches `ubo_index`. Each
//! case adds whatever its own body computes on top of that.
//!
//! # What a pass does not prove
//!
//! Anything about a family not listed, and nothing about a tiler. The device is chosen external,
//! then internal, then software — see `device::preferred` — so a run says which tier answered.
//! `TSL_VULKAN_LIB` names a driver to open directly, for an image whose loader and driver
//! disagree; see `draw_cost.rs`.

use ash::vk;
use tessella_capture_abi::generated::shader_attributes::{
    BACKGROUND_SHADER, CIRCLE_SHADER, FILL_SHADER, ShaderAttribute,
};
use tessella_capture_abi::generated::ubo_layouts::{
    BACKGROUND_DRAWABLE_UBO, BACKGROUND_PROPS_UBO, CIRCLE_DRAWABLE_UBO, CIRCLE_EVALUATED_PROPS_UBO,
    FILL_DRAWABLE_UBO, FILL_EVALUATED_PROPS_UBO, GLOBAL_PAINT_PARAMS_UBO, UboLayout,
};
use tessella_emblema::device::{preferred, vertex_format};
use tessella_emblema::shaders::{BACKGROUND_BODY, CIRCLE_BODY, FILL_BODY, module};
use tessella_emblema::surface::Surface;

/// The target's edge, in pixels. Small: one pixel is read and the rest is margin.
const SIDE: u32 = 32;

/// A family, the inputs that make one pixel predictable, and that pixel.
struct Case {
    name: &'static str,
    blocks: Vec<&'static UboLayout>,
    attributes: &'static [ShaderAttribute],
    body: &'static str,
    /// One stream per attribute, in the table's order, holding every vertex.
    streams: Vec<Vec<u8>>,
    /// One block per entry in `blocks`, in the same order.
    uniforms: Vec<Vec<u8>>,
    vertices: u32,
    expect: [u8; 4],
}

/// A value written into a block at a named field.
enum At<'a> {
    /// Floats, which is most of what a block holds.
    F(&'a [f32]),
    /// Signed integers, which is what the flags are.
    I(&'a [i32]),
}

/// The identity with `y` negated, so a tile position is already a clip position.
///
/// Every case uses it, which keeps each one's expected pixel independent of a projection the probe
/// would otherwise also have to be right about.
///
/// # The flip is not decoration
///
/// Vulkan's clip space runs `y` down and so does the framebuffer, but mbgl's shaders end their
/// vertex stage with `gl_Position.y *= -1.0` -- `applySurfaceTransform()` -- because the matrix
/// they are handed was built for a `y`-up convention. emblema has no such step: the surface's
/// `place` is the whole transform, so the flip has to be in the matrix the producer supplies.
///
/// Most families cannot tell. `fill_outline` can: its fragment stage compares the position its
/// vertex stage computed against `FragCoord`, and with an unflipped matrix the two are mirrored
/// about the target's middle. Read at the center that is a distance of one pixel, which is exactly
/// the width the feather fades over -- so the pixel came out empty, and looked like a missing
/// feather rather than a matrix this probe had built wrong.
const CLIP: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// A triangle covering the viewport, in tile units that the identity makes clip units.
const COVERING: [i16; 6] = [-1, -1, 3, -1, -1, 3];

/// Two triangles whose low bits spell the four corners of one quad, centered at the origin.
///
/// `circle` and `heatmap` sneak the corner sign into the low bit of the position, so a quad's
/// vertices are `2 * center + (corner + 1) / 2`. Centered at the origin, that is zeros and ones.
const CORNERED_QUAD: [i16; 12] = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1];

fn main() {
    match run() {
        Ok(passed) => println!("{passed} cases: ok"),
        Err(why) => {
            eprintln!("readback oracle: {why}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<usize, String> {
    let gpu = Gpu::open()?;
    println!("  device: {} ({})", gpu.name, gpu.class);

    // The control first. A cleared pass has to read black, or a right answer below could be
    // whatever the mapped memory happened to hold.
    gpu.clear()?;
    let cleared = gpu.center()?;
    if cleared != [0, 0, 0, 0] {
        return Err(format!(
            "a cleared pass reads {cleared:?}, so the readback is not showing the pass"
        ));
    }

    let cases = cases();
    for case in &cases {
        let drawn = gpu.run(case)?;
        if drawn != case.expect {
            return Err(format!(
                "{} reads {drawn:?}, wanted {:?}",
                case.name, case.expect
            ));
        }
        println!("  {:<13} {drawn:?}", case.name);
    }
    Ok(cases.len())
}

/// Every family this probe can set up without a texture, and the pixel each should draw.
fn cases() -> Vec<Case> {
    vec![
        // A flat fill, at the far end of both of its interpolations.
        //
        // The color's two endpoints are red and blue and `color_t` is one, so blue is the answer
        // and red is what a body that swapped the endpoints or dropped the factor would draw. The
        // opacity runs nothing to one over the same factor, so a dropped opacity is black.
        Case {
            name: "fill",
            blocks: vec![&FILL_DRAWABLE_UBO, &FILL_EVALUATED_PROPS_UBO],
            attributes: &FILL_SHADER,
            body: FILL_BODY,
            streams: vec![
                shorts(&COVERING),
                per_vertex(&packed_pair([255, 0, 0, 255], [0, 0, 255, 255]), 3),
                per_vertex(&[0.0, 1.0], 3),
            ],
            uniforms: vec![
                block(
                    &FILL_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("color_t", At::F(&[1.0])),
                        ("opacity_t", At::F(&[1.0])),
                    ],
                ),
                block(&FILL_EVALUATED_PROPS_UBO, &[]),
            ],
            vertices: 3,
            expect: [0, 0, 255, 255],
        },
        // A background: the only family whose color is a uniform rather than an attribute, and
        // the only one declaring a `Color` field -- so this is where that kind's four floats are
        // checked to arrive unpacked.
        Case {
            name: "background",
            blocks: vec![&BACKGROUND_DRAWABLE_UBO, &BACKGROUND_PROPS_UBO],
            attributes: &BACKGROUND_SHADER,
            body: BACKGROUND_BODY,
            streams: vec![shorts(&COVERING)],
            uniforms: vec![
                block(&BACKGROUND_DRAWABLE_UBO, &[("matrix", At::F(&CLIP))]),
                block(
                    &BACKGROUND_PROPS_UBO,
                    &[
                        ("color", At::F(&[0.0, 0.0, 1.0, 1.0])),
                        ("opacity", At::F(&[1.0])),
                    ],
                ),
            ],
            vertices: 3,
            expect: [0, 0, 255, 255],
        },
        // A circle, read at its own center: the extrusion interpolates to zero there, which is
        // inside the fill and nowhere near the stroke. With no stroke width the stroke's own
        // selection is skipped, so what is left is the fill color times its opacity.
        //
        // An extrude scale of a fifth against a reach of ten puts the quad's corners two clip
        // units out, which covers the viewport.
        Case {
            name: "circle",
            blocks: vec![
                &CIRCLE_DRAWABLE_UBO,
                &CIRCLE_EVALUATED_PROPS_UBO,
                &GLOBAL_PAINT_PARAMS_UBO,
            ],
            attributes: &CIRCLE_SHADER,
            body: CIRCLE_BODY,
            streams: vec![
                shorts(&CORNERED_QUAD),
                per_vertex(&packed_color([0, 255, 0, 255]), 6),
                per_vertex(&[10.0, 10.0], 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[1.0, 1.0], 6),
                per_vertex(&packed_color([255, 0, 255, 255]), 6),
                per_vertex(&[0.0, 0.0], 6),
                per_vertex(&[0.0, 0.0], 6),
            ],
            uniforms: vec![
                block(
                    &CIRCLE_DRAWABLE_UBO,
                    &[
                        ("matrix", At::F(&CLIP)),
                        ("extrude_scale", At::F(&[0.2, 0.2])),
                    ],
                ),
                block(
                    &CIRCLE_EVALUATED_PROPS_UBO,
                    &[
                        ("scale_with_map", At::I(&[0])),
                        ("pitch_with_map", At::I(&[0])),
                    ],
                ),
                block(
                    &GLOBAL_PAINT_PARAMS_UBO,
                    &[
                        ("pixel_ratio", At::F(&[1.0])),
                        ("camera_to_center_distance", At::F(&[1.0])),
                    ],
                ),
            ],
            vertices: 6,
            expect: [0, 255, 0, 255],
        },
    ]
}

/// Two colors packed the way `unpack_color` reads them: two channels to a float, scaled by 255.
///
/// `from` and `to` are a data-driven property's two zoom endpoints, which the body mixes by the
/// block's `_t`. Giving them *different* colors is what makes the mix observable: with the same
/// color at both ends the result is whatever `_t` is, and a body that swapped the endpoints or
/// ignored the factor would draw the right pixel anyway.
fn packed_pair(from: [u8; 4], to: [u8; 4]) -> [f32; 4] {
    let pack = |rgba: [u8; 4]| {
        [
            f32::from(rgba[0]) * 256.0 + f32::from(rgba[1]),
            f32::from(rgba[2]) * 256.0 + f32::from(rgba[3]),
        ]
    };
    let [lo_from, hi_from] = pack(from);
    let [lo_to, hi_to] = pack(to);
    [lo_from, hi_from, lo_to, hi_to]
}

/// One color at both endpoints, for a case whose interpolation is not what it is checking.
fn packed_color(rgba: [u8; 4]) -> [f32; 4] {
    packed_pair(rgba, rgba)
}

/// One attribute's stream: the same values for every vertex.
fn per_vertex(values: &[f32], vertices: usize) -> Vec<u8> {
    (0..vertices)
        .flat_map(|_| values.iter().flat_map(|value| value.to_le_bytes()))
        .collect()
}

/// A stream of sixteen-bit positions.
fn shorts(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A block, written at the offsets the ABI declares rather than at counted ones.
///
/// The point of going through the table: a block written by hand would agree with the WGSL struct
/// only because the same person wrote both. Taking every offset from the layout makes this a
/// comparison between the producer's placement and the shader's.
fn block(layout: &'static UboLayout, writes: &[(&str, At<'_>)]) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.stride as usize];
    for (name, value) in writes {
        let field = layout
            .fields
            .iter()
            .find(|field| field.name == *name)
            .unwrap_or_else(|| panic!("{} has no {name}", layout.name));
        let at = field.offset as usize;
        let words: Vec<[u8; 4]> = match value {
            At::F(values) => values.iter().map(|v| v.to_le_bytes()).collect(),
            At::I(values) => values.iter().map(|v| v.to_le_bytes()).collect(),
        };
        for (index, word) in words.iter().enumerate() {
            let start = at + index * 4;
            bytes[start..start + 4].copy_from_slice(word);
        }
    }
    bytes
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
    naga::back::spv::write_vec(&parsed, &info, &naga::back::spv::Options::default(), None)
        .map_err(|why| format!("spirv: {why}"))
}

/// A device, a linear-tiled target, and everything shared between cases.
///
/// Linear tiling and host-visible memory, so the readback is a map rather than a staging copy and
/// a blit this probe would also have to get right. Every target here supports it for a color
/// attachment at this size; a device that does not would fail at image creation, which is where it
/// should.
struct Gpu {
    name: String,
    class: &'static str,
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
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
    row: u64,
    offset: u64,
}

/// What a device's class is called here, which is the tier a run reports.
fn tier(class: vk::PhysicalDeviceType) -> &'static str {
    match class {
        vk::PhysicalDeviceType::DISCRETE_GPU => "external",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "internal",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "software",
        _ => "unknown",
    }
}

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

        // External, then internal, then software.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|why| format!("no devices: {why}"))?;
        let classes: Vec<vk::PhysicalDeviceType> = devices
            .iter()
            .map(|device| unsafe { instance.get_physical_device_properties(*device) }.device_type)
            .collect();
        let physical = preferred(&classes)
            .map(|index| devices[index])
            .ok_or_else(|| "no physical device".to_string())?;
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties.device_name_as_c_str().map_or_else(
            |_| "unnamed".to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        let class = tier(properties.device_type);

        let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| "no graphics queue".to_string())?;
        let family = u32::try_from(family).map_err(|_| "absurd queue family".to_string())?;
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

        Ok(Self {
            name,
            class,
            _entry: entry,
            instance,
            physical,
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
            row: subresource.row_pitch,
            offset: subresource.offset,
        })
    }

    /// The control: the same pass with nothing drawn in it.
    fn clear(&self) -> Result<(), String> {
        self.record(None)?;
        self.submit()
    }

    /// Draws one case and reads its pixel.
    fn run(&self, case: &Case) -> Result<[u8; 4], String> {
        let source = module(
            Surface::Plane,
            &case.blocks,
            case.attributes,
            &[],
            case.body,
        )
        .map_err(|why| format!("{} does not assemble: {why:?}", case.name))?;
        let words = compile(&source).map_err(|why| format!("{}: {why}", case.name))?;
        let held = Held::new(self, case)?;
        let pipeline = self.pipeline(case, &words, held.pipeline_layout)?;
        self.record(Some((case, &pipeline, &held)))?;
        self.submit()?;
        unsafe {
            self.device.destroy_pipeline(pipeline.pipeline, None);
            for shader in &pipeline.modules {
                self.device.destroy_shader_module(*shader, None);
            }
        }
        drop(held);
        self.center()
    }

    fn record(&self, draw: Option<(&Case, &Pipeline, &Held)>) -> Result<(), String> {
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
            if let Some((case, pipeline, held)) = draw {
                self.device.cmd_bind_pipeline(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    held.pipeline_layout,
                    0,
                    &[held.descriptors],
                    &[],
                );
                let streams = held.streams();
                let zeros = vec![0u64; streams.len()];
                self.device
                    .cmd_bind_vertex_buffers(self.command, 0, &streams, &zeros);
                // `firstInstance` is the drawable's slot, which the body reads as `ubo_index`.
                self.device.cmd_draw(self.command, case.vertices, 1, 0, 0);
            }
            self.device.cmd_end_render_pass(self.command);
            self.device
                .end_command_buffer(self.command)
                .map_err(|why| format!("not recorded: {why}"))?;
        }
        Ok(())
    }

    fn submit(&self) -> Result<(), String> {
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
        Ok(())
    }

    /// A pipeline for the module, with the vertex input taken from the ABI's table.
    ///
    /// One binding per attribute, numbered by the table's own `binding` -- which is also the
    /// `@location` the generated `In` struct gives it, so the two cannot drift apart here.
    fn pipeline(
        &self,
        case: &Case,
        words: &[u32],
        layout: vk::PipelineLayout,
    ) -> Result<Pipeline, String> {
        // Adreno rejects a module with two entry points, so each stage gets its own.
        let vertex = self.shader(words)?;
        let fragment = self.shader(words)?;

        let mut bindings = Vec::new();
        let mut attributes = Vec::new();
        for attribute in case.attributes {
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
            // No culling: this probe is not checking which way a triangle winds.
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
            .layout(layout)
            .render_pass(self.pass)
            .subpass(0)];
        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &create, None)
        }
        .map_err(|(_, why)| format!("no pipeline for {}: {why}", case.name))?;
        Ok(Pipeline {
            pipeline: pipelines[0],
            modules: vec![vertex, fragment],
        })
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

/// A pipeline and the modules it was built from, destroyed together.
struct Pipeline {
    pipeline: vk::Pipeline,
    modules: Vec<vk::ShaderModule>,
}

/// One case's buffers and descriptors, freed when it is done.
struct Held<'a> {
    gpu: &'a Gpu,
    buffers: Vec<vk::Buffer>,
    streams: usize,
    memory: vk::DeviceMemory,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
}

impl<'a> Held<'a> {
    #[allow(clippy::too_many_lines)]
    fn new(gpu: &'a Gpu, case: &Case) -> Result<Self, String> {
        let streams = case.streams.len();
        let contents: Vec<&Vec<u8>> = case.streams.iter().chain(case.uniforms.iter()).collect();

        let mut buffers = Vec::with_capacity(contents.len());
        for (index, bytes) in contents.iter().enumerate() {
            let usage = if index < streams {
                vk::BufferUsageFlags::VERTEX_BUFFER
            } else {
                vk::BufferUsageFlags::STORAGE_BUFFER
            };
            let buffer = unsafe {
                gpu.device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes.len() as u64)
                        .usage(usage),
                    None,
                )
            }
            .map_err(|why| format!("no buffer {index} for {}: {why}", case.name))?;
            buffers.push(buffer);
        }

        // One allocation, each buffer at its own aligned offset.
        let mut offsets = vec![0u64; buffers.len()];
        let mut total = 0u64;
        let mut bits = u32::MAX;
        for (index, buffer) in buffers.iter().enumerate() {
            let needs = unsafe { gpu.device.get_buffer_memory_requirements(*buffer) };
            let align = needs.alignment.max(1);
            total = total.div_ceil(align) * align;
            offsets[index] = total;
            total += needs.size;
            bits &= needs.memory_type_bits;
        }
        let memory_properties = unsafe {
            gpu.instance
                .get_physical_device_memory_properties(gpu.physical)
        };
        let host = pick(
            &memory_properties,
            bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| "no host-visible memory for the buffers".to_string())?;
        let memory = unsafe {
            gpu.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(total)
                    .memory_type_index(host),
                None,
            )
        }
        .map_err(|why| format!("no buffer memory: {why}"))?;

        let mapped = unsafe {
            gpu.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|why| format!("not mapped: {why}"))?
        .cast::<u8>();
        for (index, buffer) in buffers.iter().enumerate() {
            unsafe {
                gpu.device
                    .bind_buffer_memory(*buffer, memory, offsets[index])
            }
            .map_err(|why| format!("buffer {index} not bound: {why}"))?;
            let bytes = contents[index];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    mapped.add(offsets[index] as usize),
                    bytes.len(),
                );
            }
        }
        unsafe { gpu.device.unmap_memory(memory) };

        // One storage binding per block, from zero, which is how `module` numbers them.
        let count = u32::try_from(case.uniforms.len()).map_err(|_| "absurd block count")?;
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..count)
            .map(|slot| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(slot)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        let descriptor_layout = unsafe {
            gpu.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|why| format!("no descriptor layout: {why}"))?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(count)];
        let descriptor_pool = unsafe {
            gpu.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|why| format!("no descriptor pool: {why}"))?;
        let layouts = [descriptor_layout];
        let descriptors = unsafe {
            gpu.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|why| format!("no descriptor set: {why}"))?[0];
        let pipeline_layout = unsafe {
            gpu.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }
        .map_err(|why| format!("no pipeline layout: {why}"))?;

        let infos: Vec<vk::DescriptorBufferInfo> = buffers[streams..]
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let updates: Vec<vk::WriteDescriptorSet<'_>> = infos
            .iter()
            .enumerate()
            .map(|(slot, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptors)
                    .dst_binding(u32::try_from(slot).unwrap_or_default())
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { gpu.device.update_descriptor_sets(&updates, &[]) };

        Ok(Self {
            gpu,
            buffers,
            streams,
            memory,
            descriptor_layout,
            descriptor_pool,
            descriptors,
            pipeline_layout,
        })
    }

    fn streams(&self) -> Vec<vk::Buffer> {
        self.buffers[..self.streams].to_vec()
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let device = &self.gpu.device;
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            for buffer in &self.buffers {
                device.destroy_buffer(*buffer, None);
            }
            device.free_memory(self.memory, None);
        }
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
