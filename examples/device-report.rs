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
//!
//! Then what each [`Attachment`] resolves to, which is the question `Attachment::StencilOnly`
//! turns on. Its own note says a stencil-only format is "a smaller attachment and, on a tiler,
//! less to write back" -- so the two have to resolve to *different* formats for it to save
//! anything, and on a device offering no `S8_UINT` it is inert. `benches/overdraw_cost.rs` prices
//! the difference where there is one; this says whether there is one at all, on any device,
//! without creating one.

// A probe is a program, and it reports by printing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::ffi::CStr;

use ash::vk;
use tessella_emblema::device::{self, Attachment};

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

        // The library's own resolver, handed this device's `optimalTilingFeatures`. Asking it
        // rather than reimplementing the preference order is the point: what is reported is what a
        // view would actually be given.
        let features = |format| {
            // SAFETY: the handle came from this instance.
            unsafe { instance.get_physical_device_format_properties(physical, format) }
                .optimal_tiling_features
        };
        let flat = device::depth_stencil_format(Attachment::StencilOnly, features);
        let deep = device::depth_stencil_format(Attachment::DepthStencil, features);
        print!("    stencil-only -> {flat:?}, depth-stencil -> {deep:?}   ");
        match (flat, deep) {
            (Ok(a), Ok(b)) if a == b => {
                println!("the same, so StencilOnly saves nothing here");
            }
            (Ok(_), Ok(_)) => println!("different, so StencilOnly has something to save"),
            _ => println!("one of them has no format at all"),
        }
    }

    // SAFETY: nothing was created from it.
    unsafe { instance.destroy_instance(None) };
}
