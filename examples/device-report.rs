//! What a device offers, asked of the device rather than inferred from a failure.
//!
//! ```text
//! cargo run --example device-report
//! ```
//!
//! `benches/common/mod.rs` creates its device with
//! `VkPhysicalDeviceVulkan13Features::dynamic_rendering(true)` and its comment says that is
//! "reported by every part this runs on". Where that is not true the only symptom is
//! `vkCreateDevice` answering `Requested feature is not available on this device`, which names
//! neither the feature nor the device -- so this asks each physical device what it is and what it
//! has, and creates nothing.
//!
//! One line per physical device: its name, the API version *it* reports rather than the loader's,
//! and whether dynamic rendering is there as Vulkan 1.3 core or through
//! `VK_KHR_dynamic_rendering`. The pass requires it, which `device::check_dynamic_rendering` is
//! the library's half of.

// A probe is a program, and it reports by printing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::CStr;

use ash::vk;

/// Reads a `VkPhysicalDeviceProperties`-style name array as a string.
///
/// Through the pointer rather than element by element, because `c_char` is signed on `x86_64` and
/// *unsigned* on aarch64 -- so a `as u8` per element is a sign-losing cast on one target and a
/// no-op on the other, and no single lint expectation fits both.
fn name_of(raw: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan specifies these arrays as NUL-terminated UTF-8 within their own length, so
    // the pointer is valid and the scan terminates inside the array.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn main() {
    // SAFETY: loading the loader is the one call that has to come first.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(entry) => entry,
        Err(why) => {
            println!("no Vulkan loader: {why}");
            return;
        }
    };
    // 1.1 rather than 1.3: an instance asking for more than the loader has can be refused, and
    // what is wanted here is to be *told* each device's version, not to be turned away for
    // guessing high.
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: the info is fully initialized and names no extensions or layers.
    let instance = match unsafe { entry.create_instance(&info, None) } {
        Ok(instance) => instance,
        Err(why) => {
            println!("no instance: {why}");
            return;
        }
    };

    // SAFETY: the instance is live and owns the handles this returns.
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    println!("{} physical device(s)", devices.len());
    for physical in devices {
        // SAFETY: the handle came from this instance.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        // SAFETY: as above.
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();

        let version = props.api_version;
        let by_core = version >= vk::API_VERSION_1_3;
        let by_extension = extensions
            .iter()
            .any(|e| name_of(&e.extension_name) == "VK_KHR_dynamic_rendering");

        println!(
            "  {}\n    api {}.{}.{}   dynamic rendering: {}",
            name_of(&props.device_name),
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version),
            match (by_core, by_extension) {
                (true, _) => "yes, Vulkan 1.3 core",
                (false, true) => "yes, VK_KHR_dynamic_rendering",
                (false, false) => "NO -- this crate's pass cannot run here",
            }
        );
    }

    // SAFETY: nothing was created from it.
    unsafe { instance.destroy_instance(None) };
}
