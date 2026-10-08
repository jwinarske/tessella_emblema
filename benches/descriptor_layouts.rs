// SPDX-License-Identifier: BSD-2-Clause
//! Every family's descriptor set layout, made on a real device.
//!
//! A bench rather than a test for the reason the others are: it needs a GPU and CI has none.
//! `tests/descriptors.rs` holds the half that does not, which is whether the bindings agree with
//! what the modules declare.
//!
//! # What this proves that the agreement test cannot
//!
//! That a device will accept them. `vkCreateDescriptorSetLayout` refuses a set asking for more
//! descriptors of a kind than `maxPerStageDescriptor*` allows, and the limits that matter here are
//! not generous: the widest set this crate declares is five storage buffers, and **Vulkan's own
//! floor for `maxPerStageDescriptorStorageBuffers` is four**. A conformant device may be under what
//! the shaders need, so the number is worth knowing per part rather than assumed.
//!
//! So this creates all 57 and then reports the headroom, which is the measurement a later family
//! would spend.
//!
//! Run with `cargo bench --bench descriptor_layouts`.

mod common;

use tessella_emblema::families;
use tessella_emblema::pipelines::{self, Kind};
use tessella_emblema::surface::Surface;

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

/// Every family and surface pair that has a module, with its bindings.
fn every_pair() -> Vec<(String, Vec<pipelines::Binding>)> {
    let mut out = Vec::new();
    for family in families::ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            out.push((
                format!("{} on {surface:?}", family.name),
                pipelines::bindings(family, surface),
            ));
        }
    }
    out
}

/// Every pair's layout is created, and none is refused.
fn every_layout_is_accepted(device: &Open) -> Result<(), String> {
    let pairs = every_pair();
    if pairs.len() != 57 {
        return Err(format!(
            "{} pairs, which is not the 57 modules the crate assembles",
            pairs.len()
        ));
    }
    // Held to the end rather than dropped per iteration, so the device is carrying all 57 at once --
    // which is the state a frame drawing every family would be in.
    let mut held = Vec::with_capacity(pairs.len());
    for (who, bindings) in &pairs {
        let layout =
            pipelines::layout(device.gpu(), bindings).map_err(|why| format!("{who}: {why}"))?;
        if layout.set() == ash::vk::DescriptorSetLayout::null() {
            return Err(format!("{who}: a null set layout"));
        }
        if layout.pipeline() == ash::vk::PipelineLayout::null() {
            return Err(format!("{who}: a null pipeline layout"));
        }
        held.push(layout);
    }
    println!("  every layout          ok   57 set and pipeline layouts, all resident at once");
    Ok(())
}

/// Two pairs do not share a layout handle.
///
/// The same family on two surfaces is two different sets, so a cache keyed on the family alone would
/// hand back a layout with the wrong number of bindings -- and a set bound against it has every
/// descriptor after the surface's block one place out.
fn layouts_are_distinct(device: &Open) -> Result<(), String> {
    let raster = families::ALL
        .iter()
        .find(|family| family.name == "raster")
        .ok_or("a raster family")?;
    let flat = pipelines::layout(device.gpu(), &pipelines::bindings(raster, Surface::Plane))
        .map_err(|why| format!("flat: {why}"))?;
    let raised = pipelines::layout(device.gpu(), &pipelines::bindings(raster, Surface::Terrain))
        .map_err(|why| format!("raised: {why}"))?;
    if flat.set() == raised.set() {
        return Err("one family on two surfaces got one set layout".into());
    }
    println!("  layouts are distinct  ok   raster on Plane and on Terrain are two layouts");
    Ok(())
}

/// What the widest set needs, against what this device allows.
///
/// Not an assertion about a number this crate chose -- it is the headroom a later family would
/// spend, reported per part because the three on the bench disagree by six orders of magnitude.
fn headroom(device: &Open) -> Result<(), String> {
    // Ranked by storage buffers, which is the limit that can bite, and ties broken by the total --
    // so the pair reported is the most demanding one rather than whichever came first.
    let mut worst = (0usize, 0usize, 0usize, 0usize, String::new());
    for (who, bindings) in every_pair() {
        let storage = bindings
            .iter()
            .filter(|b| b.kind == Kind::StorageBuffer)
            .count();
        let images = bindings
            .iter()
            .filter(|b| b.kind == Kind::SampledImage)
            .count();
        let samplers = bindings.iter().filter(|b| b.kind == Kind::Sampler).count();
        let rank = (storage, bindings.len());
        if rank > (worst.0, worst.1) {
            worst = (storage, bindings.len(), images, samplers, who.clone());
        }
    }
    let limits = &device.limits;
    let (storage, total, _, _, who) = &worst;

    // The one that can genuinely bite: Vulkan's floor is four and the widest family wants five.
    if *storage as u32 > limits.max_per_stage_descriptor_storage_buffers {
        return Err(format!(
            "{who} needs {storage} storage buffers and this device allows {}",
            limits.max_per_stage_descriptor_storage_buffers
        ));
    }
    println!(
        "  headroom              ok   {storage} storage of {} allowed, {total} bindings ({who})",
        limits.max_per_stage_descriptor_storage_buffers
    );
    println!(
        "                             images {} of {}, samplers {} of {}, resources {}",
        worst.2,
        limits.max_per_stage_descriptor_sampled_images,
        worst.3,
        limits.max_per_stage_descriptor_samplers,
        limits.max_per_stage_resources
    );
    Ok(())
}

/// Every set is within this device's per-stage limits -- the check the driver will not do.
///
/// `maxPerStageDescriptor*` is valid usage the application must respect, not something
/// `vkCreateDescriptorSetLayout` or `vkCreatePipelineLayout` is required to enforce. Measured here:
/// a set of one more storage buffer than the device allows is **accepted** on both V3D 7.1.7.0 and
/// the `VeriSilicon` `GC7000UL`. Only the validation layers report it.
///
/// So there is no backstop, and the limits have to be checked by whoever builds the layout. This is
/// that check, and it is the reason the headroom above is a number worth printing rather than
/// trivia: a family gaining two more blocks would be over V3D's eight, and nothing would say so.
fn every_set_is_within_the_limits(device: &Open) -> Result<(), String> {
    let limits = &device.limits;
    for (who, bindings) in every_pair() {
        for (kind, allowed) in [
            (
                Kind::StorageBuffer,
                limits.max_per_stage_descriptor_storage_buffers,
            ),
            (
                Kind::SampledImage,
                limits.max_per_stage_descriptor_sampled_images,
            ),
            (Kind::Sampler, limits.max_per_stage_descriptor_samplers),
        ] {
            let wanted = bindings.iter().filter(|b| b.kind == kind).count();
            if wanted as u64 > u64::from(allowed) {
                return Err(format!(
                    "{who} wants {wanted} of {kind:?} and this device allows {allowed}"
                ));
            }
        }
        if bindings.len() as u64 > u64::from(limits.max_per_stage_resources) {
            return Err(format!(
                "{who} wants {} resources and this device allows {}",
                bindings.len(),
                limits.max_per_stage_resources
            ));
        }
    }

    // And the evidence that the check is needed rather than redundant: one past the limit, which
    // the driver is free to accept and both boards do.
    let past = limits.max_per_stage_descriptor_storage_buffers as usize + 1;
    let verdict = if past > 4096 {
        "not probed, the limit is in the millions".to_owned()
    } else {
        let bindings: Vec<pipelines::Binding> = (0..past)
            .map(|at| pipelines::Binding {
                binding: at as u32,
                kind: Kind::StorageBuffer,
            })
            .collect();
        if pipelines::layout(device.gpu(), &bindings).is_ok() {
            format!(
                "{past} of {} was accepted, so nothing enforces it",
                limits.max_per_stage_descriptor_storage_buffers
            )
        } else {
            format!(
                "{past} of {} was refused",
                limits.max_per_stage_descriptor_storage_buffers
            )
        }
    };
    println!("  within the limits     ok   57 sets checked; {verdict}");
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
    for (name, case) in [
        ("every_layout", every_layout_is_accepted as Case),
        ("layouts_are_distinct", layouts_are_distinct),
        ("headroom", headroom),
        ("within_the_limits", every_set_is_within_the_limits),
    ] {
        if let Err(why) = case(&device) {
            println!("  {name:<21} FAIL {why}");
            failed += 1;
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
