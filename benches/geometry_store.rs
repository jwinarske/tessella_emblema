// SPDX-License-Identifier: BSD-2-Clause
//! The geometry store, against a real device.
//!
//! A bench rather than a test for the reason `first_pixel` is one: it needs a GPU and CI has none.
//! `tests/store_layout.rs` holds the half that does not, which is the arithmetic.
//!
//! # What this proves that the layout tests cannot
//!
//! That the bytes arrive. The layout tests say where each buffer should sit; this writes a geometry,
//! reads it back off the device and compares — so an offset that is arithmetically sound and bound to
//! the wrong place is caught here and nowhere else. `vkBindBufferMemory` also validates alignment
//! against the driver's own requirement, which no host-side test can know.
//!
//! Run with `cargo bench --bench geometry_store`.

use ash::vk;
use tessella_capture_abi::envelope::{GeometryId, SlabRef};
use tessella_emblema::buffers::{self, Needs, Reads};
use tessella_emblema::store::Store;
use tessella_emblema::vertices::{Bound, Plan};
use tessella_vk::Gpu;

/// A device, open for as long as this lives.
struct Open {
    name: String,
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
    fn first() -> Result<Self, String> {
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

    fn gpu(&self) -> Gpu<'_> {
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

const fn at(slab: u32, offset: u32, length: u32) -> SlabRef {
    SlabRef {
        slab,
        offset,
        length,
    }
}

/// A geometry of two distinct vertex buffers and an index buffer, with bytes to match.
fn two_and_indexes() -> (Needs, Vec<u8>, Vec<u8>, Vec<u8>) {
    let first = at(0, 0, 64);
    let second = at(1, 0, 48);
    let indexes = at(2, 0, 24);
    let needs = Needs {
        vertices: vec![first, second],
        reads: vec![Reads { slot: 0, buffer: 0 }, Reads { slot: 1, buffer: 1 }],
        indexes: Some(indexes),
    };
    // Distinct per buffer and varying within each, so a swap or a shared offset cannot pass.
    let a: Vec<u8> = (0..64u8).collect();
    let b: Vec<u8> = (0..48u8).map(|byte| 128 + byte).collect();
    let i: Vec<u8> = (0..24u8).map(|byte| 200 + byte).collect();
    (needs, a, b, i)
}

/// What the bytes a geometry's references name, for the store to resolve against.
fn resolver<'b>(pairs: &'b [(SlabRef, &'b [u8])]) -> impl Fn(SlabRef) -> Option<&'b [u8]> + 'b {
    move |wanted| {
        pairs
            .iter()
            .find(|(reference, _)| *reference == wanted)
            .map(|(_, bytes)| *bytes)
    }
}

fn round_trip(device: &Open) -> Result<(), String> {
    let (needs, a, b, i) = two_and_indexes();
    let pairs = [
        (needs.vertices[0], a.as_slice()),
        (needs.vertices[1], b.as_slice()),
        (needs.indexes.expect("indexes"), i.as_slice()),
    ];
    let resolve = resolver(&pairs);

    let mut store = Store::new();
    let geometry = GeometryId(1);
    store
        .upload(device.gpu(), geometry, &needs, &resolve)
        .map_err(|why| format!("upload: {why}"))?;

    if !store.holds(geometry) {
        return Err("the store does not hold what it uploaded".into());
    }
    if store.vertex_buffer(geometry, 0) == store.vertex_buffer(geometry, 1) {
        return Err("two distinct references became one buffer".into());
    }
    if store.index_buffer(geometry).is_none() {
        return Err("the index buffer is missing".into());
    }

    // The bytes, off the device. This is the assertion the layout tests cannot make.
    for (index, wanted) in [a.as_slice(), b.as_slice()].iter().enumerate() {
        let mut got = vec![0u8; wanted.len()];
        store
            .read_vertex_bytes(geometry, index, &mut got)
            .map_err(|why| format!("read back {index}: {why}"))?;
        if got != *wanted {
            return Err(format!(
                "buffer {index} came back wrong: first eight {:?} against {:?}",
                &got[..8.min(got.len())],
                &wanted[..8.min(wanted.len())]
            ));
        }
    }

    let footprint = store.bytes(geometry).ok_or("no footprint")?;
    let claimed: u64 = 64 + 48 + 24;
    if footprint < claimed {
        return Err(format!(
            "the allocation is {footprint} bytes for {claimed} of references"
        ));
    }
    println!("  round trip            ok   {footprint} bytes for {claimed} claimed");

    store.free(&[geometry]);
    if store.holds(geometry) || store.total_bytes() != 0 {
        return Err("freeing left something behind".into());
    }
    println!("  free                  ok   nothing resident");
    Ok(())
}

fn dedup(device: &Open) -> Result<(), String> {
    let interleaved = at(3, 0, 120);
    let bound = |slot: u32, offset: u32| Bound {
        slot,
        format: vk::Format::R32_SFLOAT,
        stride: 12,
        offset,
        vertex_offset: 0,
        source: interleaved,
        rate: vk::VertexInputRate::VERTEX,
    };
    let plan = Plan {
        bound: vec![bound(0, 0), bound(1, 4), bound(2, 8)],
        ..Plan::default()
    };
    let needs = buffers::needs(&plan, at(0, 0, 0));
    let bytes: Vec<u8> = (0..120u8).collect();
    let pairs = [(interleaved, bytes.as_slice())];
    let resolve = resolver(&pairs);

    let mut store = Store::new();
    let geometry = GeometryId(2);
    store
        .upload(device.gpu(), geometry, &needs, &resolve)
        .map_err(|why| format!("upload: {why}"))?;

    // Three bindings, one buffer: the whole point of the dedup, and visible only through the handles.
    let handles: Vec<Option<vk::Buffer>> = needs
        .reads
        .iter()
        .map(|read| store.vertex_buffer(geometry, read.buffer))
        .collect();
    if handles.iter().any(|handle| *handle != handles[0]) {
        return Err(format!(
            "three descriptors over one interleaved buffer got {handles:?}"
        ));
    }
    if store.index_buffer(geometry).is_some() {
        return Err("a zero-length index reference became a buffer".into());
    }
    let footprint = store.bytes(geometry).ok_or("no footprint")?;
    println!("  dedup                 ok   one buffer, {footprint} bytes for 120 claimed");
    Ok(())
}

fn refusals(device: &Open) -> Result<(), String> {
    let needs = Needs {
        vertices: vec![at(9, 0, 32)],
        reads: vec![Reads { slot: 0, buffer: 0 }],
        indexes: None,
    };
    let mut store = Store::new();

    // Nothing resolves it.
    let nothing = |_: SlabRef| None;
    let unresolved = store.upload(device.gpu(), GeometryId(3), &needs, &nothing);
    if unresolved.is_ok() {
        return Err("an unresolved reference was accepted".into());
    }
    if store.resident() != 0 {
        return Err("a refused upload left a geometry resident".into());
    }

    // It resolves, but short.
    let short_bytes = vec![0u8; 8];
    let pairs = [(needs.vertices[0], short_bytes.as_slice())];
    let resolve = resolver(&pairs);
    let short = store.upload(device.gpu(), GeometryId(4), &needs, &resolve);
    if short.is_ok() {
        return Err("a reference resolving short was accepted".into());
    }
    if store.resident() != 0 {
        return Err("a refused upload left a geometry resident".into());
    }
    println!("  refusals              ok   neither left anything on the device");
    Ok(())
}

fn main() {
    let device = match Open::first() {
        Ok(device) => device,
        Err(why) => {
            println!("skipping: {why}");
            return;
        }
    };
    println!("device: {}", device.name);

    let mut failed = 0;
    for (name, outcome) in [
        ("round_trip", round_trip(&device)),
        ("dedup", dedup(&device)),
        ("refusals", refusals(&device)),
    ] {
        if let Err(why) = outcome {
            println!("  {name:<21} FAIL {why}");
            failed += 1;
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
