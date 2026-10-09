// SPDX-License-Identifier: BSD-2-Clause
//! Opening a device, for the benches that need one.
//!
//! Shared rather than copied per bench: this is the only `unsafe` the benches contain -- the library
//! itself forbids it and reaches Vulkan through `tessella-vk` -- and one copy is one place to read the
//! SAFETY notes and one place to get them wrong.

// Each bench is its own crate and compiles this module separately, so whatever *that* bench does
// not call reads as dead here -- `submit` is used by `texture_images` and by nothing else, and the
// two store benches need no queue at all. The alternative is a copy of the device setup per bench,
// which is the duplication this module exists to remove.
#![allow(dead_code)]

use ash::vk;
use tessella_emblema::device;
use tessella_vk::{Gpu, Recorder};

/// What a device's class is called in a summary line.
fn tier(class: vk::PhysicalDeviceType) -> &'static str {
    match class {
        vk::PhysicalDeviceType::DISCRETE_GPU => "external",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "internal",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "software",
        _ => "unknown",
    }
}

/// A device, open for as long as this lives.
pub struct Open {
    pub name: String,
    /// The tier that answered: external, internal, virtual or software.
    ///
    /// What a run reports, because which tier drew is part of reading a result -- a pixel from a
    /// software rasterizer says less about a board than one from its own GPU.
    pub class: &'static str,
    _entry: ash::Entry,
    instance: ash::Instance,
    handle: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    physical: vk::PhysicalDevice,
    family: u32,
}

/// Which of the enumerated devices to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// The first that enumerates, which is all a store or a layout needs.
    First,
    /// External, then internal, then software, through [`device::preferred`].
    ///
    /// What a bench reading pixels wants: a result is only as good as the part that produced it,
    /// and a machine with a software rasterizer beside a GPU enumerates both.
    Preferred,
}

impl Open {
    /// Opens the first device that enumerates, or says why not.
    ///
    /// Named for what it picks rather than what it does, because `Open::open` reads as a repeat.
    ///
    /// No queue is used and none is submitted to: a store creates buffers, binds them and maps the
    /// allocation, and not one of those touches a queue. One is requested anyway because
    /// `vkCreateDevice` requires at least one queue family.
    pub fn first() -> Result<Self, String> {
        Self::with(Pick::First)
    }

    /// Opens the device [`device::preferred`] names, which is what a pixel oracle wants.
    pub fn preferred() -> Result<Self, String> {
        Self::with(Pick::Preferred)
    }

    /// Opens a device, choosing it as `pick` says.
    ///
    /// `TSL_VULKAN_LIB` names a driver to load directly rather than through the loader, for an
    /// image whose loader and driver disagree about which ICDs exist.
    pub fn with(pick: Pick) -> Result<Self, String> {
        // SAFETY: the loader is linked at run time and these are its documented entry points. The
        // path is the operator's, and loading a named driver is the whole point of the variable.
        let entry = match std::env::var("TSL_VULKAN_LIB") {
            Ok(path) => unsafe { ash::Entry::load_from(&path) }
                .map_err(|why| format!("no driver at {path}: {why}"))?,
            Err(_) => unsafe { ash::Entry::load() }.map_err(|why| format!("no loader: {why}"))?,
        };
        // 1.3, because the pass requires dynamic rendering and `vkCmdBeginRendering` is 1.3 core.
        // ash loads an entry point only when the declared version covers it, so an instance asking
        // for 1.1 gets a null pointer and panics on the first call rather than failing to create --
        // which is what this said before and how it was found.
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        // SAFETY: the info is fully initialized and borrowed only for the call.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(|why| format!("create_instance: {why}"))?;

        // SAFETY: the instance is live.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|why| format!("enumerate: {why}"))?;
        let physical = match pick {
            Pick::First => *devices.first().ok_or("no physical device")?,
            Pick::Preferred => {
                let classes: Vec<vk::PhysicalDeviceType> = devices
                    .iter()
                    // SAFETY: the instance is live and each handle came from it.
                    .map(|at| unsafe { instance.get_physical_device_properties(*at) }.device_type)
                    .collect();
                device::preferred(&classes)
                    .and_then(|at| devices.get(at).copied())
                    .ok_or("no physical device")?
            }
        };
        // SAFETY: as above.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties.device_name_as_c_str().map_or_else(
            |_| "unnamed".to_owned(),
            |raw| raw.to_string_lossy().into_owned(),
        );
        // SAFETY: as above.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let limits = properties.limits;

        // A family that can draw, rather than family zero. Zero is a graphics family on every part
        // this runs on, but a device that reported otherwise would fail at the first draw rather
        // than here, and the search costs one call.
        // SAFETY: the instance and the handle are live.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .and_then(|at| u32::try_from(at).ok())
            .ok_or("no graphics queue family")?;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // Dynamic rendering, which the pass requires rather than prefers -- see
        // `device::check_dynamic_rendering`. Core in Vulkan 1.3 and reported by every part this runs
        // on, but a feature still has to be *enabled* at device creation to be used, and a pipeline
        // chaining `VkPipelineRenderingCreateInfo` without it is invalid usage that a driver need
        // not report.
        let mut thirteen = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        // SAFETY: the info is fully initialized and names a family this device reported.
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .push_next(&mut thirteen),
                None,
            )
        }
        .map_err(|why| format!("create_device: {why}"))?;

        Ok(Self {
            name,
            class: tier(properties.device_type),
            _entry: entry,
            instance,
            handle: device,
            memory,
            limits,
            physical,
            family,
        })
    }

    /// What the device will do with a format, for the queries `device.rs` takes as a closure.
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        // SAFETY: the instance and the physical device are live.
        unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical, format)
        }
    }

    pub fn gpu(&self) -> Gpu<'_> {
        Gpu::new(&self.handle, &self.memory)
    }

    /// Records commands into a one-shot command buffer, submits it, and waits for it.
    ///
    /// Standing in for the host. Beginning a command buffer, submitting it and knowing when it
    /// completed belong to whoever owns the queue -- `tessella-vk` owns no pool and the map pass only
    /// records -- so in production this is emblema's job and here it is the bench's.
    ///
    /// Waits on a fence rather than `vkQueueWaitIdle`, because a bench that reads pixels back has to
    /// know the copy finished and not merely that the queue went quiet.
    pub fn submit(&self, record: impl FnOnce(Recorder<'_>)) -> Result<(), String> {
        // SAFETY: this family was the one the device was created with.
        let pool = unsafe {
            self.handle.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .map_err(|why| format!("create_command_pool: {why}"))?;

        let outcome = self.record_and_wait(pool, record);

        // SAFETY: the pool was made by this device; every buffer from it was allocated here and the
        // submission has completed or failed, so nothing is still executing.
        unsafe { self.handle.destroy_command_pool(pool, None) };
        outcome
    }

    fn record_and_wait(
        &self,
        pool: vk::CommandPool,
        record: impl FnOnce(Recorder<'_>),
    ) -> Result<(), String> {
        // SAFETY: the pool is live and belongs to this device.
        let buffers = unsafe {
            self.handle.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|why| format!("allocate_command_buffers: {why}"))?;
        let buffer = buffers[0];

        // SAFETY: the buffer was just allocated and is not recording.
        unsafe {
            self.handle.begin_command_buffer(
                buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|why| format!("begin_command_buffer: {why}"))?;

        record(Recorder::new(&self.handle, buffer));

        // SAFETY: the buffer is recording and every command in it was recorded through `Recorder`.
        unsafe { self.handle.end_command_buffer(buffer) }
            .map_err(|why| format!("end_command_buffer: {why}"))?;

        // SAFETY: the create info is fully initialized.
        let fence = unsafe {
            self.handle
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|why| format!("create_fence: {why}"))?;

        let outcome = self.wait_on(buffer, fence);
        // SAFETY: the fence is signaled or the submit failed, so nothing is waiting on it.
        unsafe { self.handle.destroy_fence(fence, None) };
        outcome
    }

    fn wait_on(&self, buffer: vk::CommandBuffer, fence: vk::Fence) -> Result<(), String> {
        let buffers = [buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        // SAFETY: this family was requested at device creation, so index zero of it exists.
        let queue = unsafe { self.handle.get_device_queue(self.family, 0) };
        // SAFETY: the buffer has ended, the fence is unsignaled, and both belong to this device.
        unsafe { self.handle.queue_submit(queue, &[submit], fence) }
            .map_err(|why| format!("queue_submit: {why}"))?;
        // SAFETY: the fence was just submitted with.
        unsafe { self.handle.wait_for_fences(&[fence], true, u64::MAX) }
            .map_err(|why| format!("wait_for_fences: {why}"))
    }
}

impl Drop for Open {
    fn drop(&mut self) {
        // SAFETY: every store built on this has been dropped by now -- they borrow the device, so the
        // compiler will not let one outlive this.
        unsafe {
            self.handle.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
