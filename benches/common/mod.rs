// SPDX-License-Identifier: BSD-2-Clause
//! Opening a device, for the benches that need one.
//!
//! Shared rather than copied per bench: this is the only `unsafe` the benches contain -- the library
//! itself forbids it and reaches Vulkan through `tessella-vk` -- and one copy is one place to read the
//! SAFETY notes and one place to get them wrong.

use ash::vk;
use tessella_vk::Gpu;

/// A device, open for as long as this lives.
pub struct Open {
    pub name: String,
    _entry: ash::Entry,
    instance: ash::Instance,
    handle: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
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
        // SAFETY: the loader is linked at run time and this is the documented entry point.
        let entry = unsafe { ash::Entry::load() }.map_err(|why| format!("no loader: {why}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
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
        let physical = *devices.first().ok_or("no physical device")?;
        // SAFETY: as above.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties.device_name_as_c_str().map_or_else(
            |_| "unnamed".to_owned(),
            |raw| raw.to_string_lossy().into_owned(),
        );
        // SAFETY: as above.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&priorities)];
        // SAFETY: the info is fully initialized; family zero exists on every conformant device.
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )
        }
        .map_err(|why| format!("create_device: {why}"))?;

        Ok(Self {
            name,
            _entry: entry,
            instance,
            handle: device,
            memory,
        })
    }

    pub fn gpu(&self) -> Gpu<'_> {
        Gpu::new(&self.handle, &self.memory)
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
