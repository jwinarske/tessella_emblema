// SPDX-License-Identifier: BSD-2-Clause
//! Allocating and writing a family's descriptor set on a real device.
//!
//! A bench rather than a test because it needs a GPU. The device-free half is
//! `tests/descriptors.rs`, which is the arithmetic: which binding each resource goes to, and the
//! refusals.
//!
//! # What this proves that those cannot
//!
//! That `vkUpdateDescriptorSets` accepts the writes. A write whose descriptor type disagrees with
//! the layout's, or whose binding number is not in the layout, is invalid usage -- and the layout
//! here is the real one `pipelines::bindings` produced, so this is the first place the two halves
//! meet on a device.
//!
//! It also proves the pool arithmetic. `pool_sizes` counts descriptors per kind and `Sets::new`
//! multiplies by the set count; getting either wrong gives `VK_ERROR_OUT_OF_POOL_MEMORY` on some
//! later allocation rather than the first, which is why this allocates every family's set from one
//! pool rather than one each.
//!
//! Run with `cargo bench --bench descriptor_sets`.

mod common;

use ash::vk;
use tessella_capture_abi::envelope::{Extent, TextureFilter, TextureId, TextureRef, ViewId};
use tessella_capture_abi::generated::mbgl_enums::{TextureChannelDataType, TexturePixelType};
use tessella_emblema::blocks::{Blocks, Which};
use tessella_emblema::descriptors::{self, Sets};
use tessella_emblema::families;
use tessella_emblema::images::Images;
use tessella_emblema::pipelines;
use tessella_emblema::surface::Surface;
use tessella_vk::Recorder;

use common::Open;

/// One checked behavior, named in the summary line.
type Case = fn(&Open) -> Result<(), String>;

const fn which(layer: i32) -> Which {
    Which {
        view: ViewId(1),
        layer,
    }
}

/// One block buffer per storage binding, at the slot that binding's bytes arrive in.
///
/// Per binding rather than one buffer for the set: a family's blocks are separate buffers, which is
/// what `UboUpdate::slot` names. One buffer here would make every binding point at it and the bench
/// could not tell.
fn blocks_for<'d>(
    device: &'d Open,
    which: Which,
    bindings: &[pipelines::Binding],
) -> Result<Blocks<'d>, String> {
    let mut blocks = Blocks::new();
    for binding in bindings
        .iter()
        .filter(|b| b.kind == pipelines::Kind::StorageBuffer)
    {
        let slot = binding
            .slot
            .ok_or_else(|| format!("binding {} carries no slot", binding.binding))?;
        blocks
            .declare(device.gpu(), which, slot, 4, 64)
            .map_err(|why| format!("declare slot {slot}: {why}"))?;
    }
    Ok(blocks)
}

/// One image per texture the family declares, each actually created.
/// The slots a set takes a texture at, in binding order.
fn texture_slots(bindings: &[pipelines::Binding]) -> Vec<u32> {
    bindings
        .iter()
        .filter(|b| b.kind == pipelines::Kind::SampledImage)
        .filter_map(|b| b.slot)
        .collect()
}

fn images_for<'d>(
    device: &'d Open,
    slots: &[u32],
) -> Result<(Images<'d>, Vec<TextureRef>), String> {
    let mut images = Images::new();
    let mut refs = Vec::new();
    for (at, slot) in slots.iter().enumerate() {
        let texture = TextureId(at as u64 + 1);
        device.submit(|record: Recorder<'_>| {
            let _ = images.declare(
                device.gpu(),
                record,
                texture,
                Extent {
                    width: 4,
                    height: 4,
                },
                TexturePixelType::RGBA,
                TextureChannelDataType::UnsignedByte,
            );
            // Into the layout a sampled descriptor names, which is what the set claims it is in.
            if let Some(image) = images.image(texture) {
                record.transition(
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        })?;
        // Alternating filters, so both samplers are exercised and a set that used one for
        // everything still writes two distinct handles.
        let filter = if at % 2 == 0 {
            TextureFilter::Linear
        } else {
            TextureFilter::Nearest
        };
        refs.push(TextureRef {
            texture,
            slot: *slot,
            filter: filter as u32,
        });
    }
    Ok((images, refs))
}

/// Every family's set is allocated and written from one pool.
fn every_set_is_written(device: &Open) -> Result<(), String> {
    // The widest set, so the pool is sized for the worst family and shared by all of them.
    let widest = families::ALL
        .iter()
        .flat_map(|family| {
            Surface::ALL
                .iter()
                .filter(|surface| family.surfaces.contains(surface))
                .map(|surface| pipelines::bindings(family, *surface))
        })
        .max_by_key(Vec::len)
        .ok_or("no families")?;

    let pairs: usize = families::ALL
        .iter()
        .map(|family| {
            Surface::ALL
                .iter()
                .filter(|s| family.surfaces.contains(s))
                .count()
        })
        .sum();
    let mut sets =
        Sets::new(device.gpu(), pairs as u32, &widest).map_err(|why| format!("sets: {why}"))?;

    let mut written = 0;
    let mut layer = 0i32;
    for family in families::ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            let who = format!("{} on {surface:?}", family.name);
            let bindings = pipelines::bindings(family, surface);
            let layout = pipelines::layout(device.gpu(), &bindings)
                .map_err(|why| format!("{who}: {why}"))?;

            // The slots the set takes a texture at, which is what a run has to name. Taken from
            // the bindings rather than counted 0..n: a family's come from its generated table and
            // terrain's elevation is slot 8, so a count would build a run no set declares.
            let image_slots = texture_slots(&bindings);

            let at = which(layer);
            layer += 1;
            let blocks = blocks_for(device, at, &bindings)?;
            let (images, refs) = images_for(device, &image_slots)?;
            let bound = descriptors::bound_from(&images, &bindings, &refs)
                .map_err(|why| format!("{who}: {why}"))?;

            let set = sets
                .write(&layout, &bindings, at, &blocks, &bound)
                .map_err(|why| format!("{who}: {why}"))?;
            if set == vk::DescriptorSet::null() {
                return Err(format!("{who}: a null set"));
            }
            if sets.get(at) != Some(set) {
                return Err(format!("{who}: the set was not remembered"));
            }
            written += 1;
        }
    }
    if written != pairs {
        return Err(format!("{written} sets written for {pairs} pairs"));
    }
    println!(
        "  every set written     ok   {written} sets from one pool, widest {} bindings",
        widest.len()
    );

    sets.reset().map_err(|why| format!("reset: {why}"))?;
    if sets.allocated() != 0 {
        return Err("resetting the pool left sets behind".into());
    }
    println!("  the pool resets       ok   nothing allocated afterwards");
    Ok(())
}

/// Each storage binding resolves its own slot, and a missing one is named.
///
/// The case for the defect this keying replaced. A family's blocks arrive in separate buffers --
/// `UboUpdate::slot` is which buffer -- and this held one per layer and pointed every binding at it,
/// so the second block's fields were read out of the first's bytes.
///
/// A descriptor set cannot be read back, so what is checkable here is which buffer each binding
/// *asks* for: every slot but one is declared, and the write must refuse naming exactly the one left
/// out. Against one buffer per layer the same write succeeded, whichever slot was missing.
fn each_binding_resolves_its_own_slot(device: &Open) -> Result<(), String> {
    let family = families::ALL
        .iter()
        .find(|f| f.name == "fill")
        .ok_or("the fill family")?;
    let bindings = pipelines::bindings(family, Surface::Plane);
    let slots: Vec<u32> = bindings
        .iter()
        .filter(|b| b.kind == pipelines::Kind::StorageBuffer)
        .filter_map(|b| b.slot)
        .collect();
    if slots.len() < 2 {
        return Err(format!("fill declares {} block slots", slots.len()));
    }
    let layout =
        pipelines::layout(device.gpu(), &bindings).map_err(|why| format!("layout: {why}"))?;
    let mut sets = Sets::new(device.gpu(), 2, &bindings).map_err(|why| format!("sets: {why}"))?;

    for missing in &slots {
        let at = which(100 + i32::try_from(*missing).unwrap_or(0));
        let mut blocks = Blocks::new();
        for slot in slots.iter().filter(|slot| *slot != missing) {
            blocks
                .declare(device.gpu(), at, *slot, 4, 64)
                .map_err(|why| format!("declare {slot}: {why}"))?;
        }
        match sets.write(&layout, &bindings, at, &blocks, &[]) {
            Err(descriptors::Error::NoBlocks { slot, .. }) if slot == *missing => {}
            other => {
                return Err(format!(
                    "slot {missing} missing of {slots:?} gave {other:?}"
                ));
            }
        }
    }
    if sets.allocated() != 0 {
        return Err("a refused write left a set allocated".into());
    }

    // And with every slot declared it writes, which is the other side: a check that refused
    // everything would pass the loop above.
    let at = which(120);
    let blocks = blocks_for(device, at, &bindings)?;
    sets.write(&layout, &bindings, at, &blocks, &[])
        .map_err(|why| format!("all slots present: {why}"))?;
    println!(
        "  bindings per slot     ok   {} slots, each refused by name, then all written",
        slots.len()
    );
    Ok(())
}

/// The two filters are two distinct samplers.
///
/// One texture can want both in one frame -- the icon atlas is linear when the icons are scaled and
/// nearest when they are not -- so a single sampler could not express what the wire asks for.
fn the_filters_are_two_samplers(device: &Open) -> Result<(), String> {
    let bindings = pipelines::bindings(
        families::ALL
            .iter()
            .find(|f| f.name == "background_pattern")
            .ok_or("a patterned family")?,
        Surface::Plane,
    );
    let sets = Sets::new(device.gpu(), 1, &bindings).map_err(|why| format!("sets: {why}"))?;
    let linear = sets.sampler(TextureFilter::Linear);
    let nearest = sets.sampler(TextureFilter::Nearest);
    if linear == nearest {
        return Err("both filters gave one sampler".into());
    }
    if linear == vk::Sampler::null() || nearest == vk::Sampler::null() {
        return Err("a null sampler".into());
    }
    println!("  two filters           ok   linear and nearest are distinct samplers");
    Ok(())
}

/// The refusals, and that none leaves a half-written set behind.
/// The same run in another order binds the same images.
///
/// This is #95's contract stated positively. The defect it names was that `bound_from` placed by
/// arrival, which is right only while the producer emits in slot order -- `encode_raster` does, so
/// no fixture here distinguished the two. Reversing a run with its slots intact is what does: place
/// by arrival and the two images swap, place by slot and nothing moves.
///
/// `raster` because it is the one family here with two textures whose slots differ.
fn an_out_of_order_run(device: &Open) -> Result<(), String> {
    let family = families::ALL
        .iter()
        .find(|f| f.name == "raster")
        .ok_or("the raster family")?;
    let bindings = pipelines::bindings(family, Surface::Plane);
    let slots = texture_slots(&bindings);
    if slots.len() != 2 {
        return Err(format!("raster declares {} textures", slots.len()));
    }
    let (images, refs) = images_for(device, &slots)?;

    let forward = descriptors::bound_from(&images, &bindings, &refs)
        .map_err(|why| format!("in order: {why}"))?;
    let mut backward_refs = refs.clone();
    backward_refs.reverse();
    let backward = descriptors::bound_from(&images, &bindings, &backward_refs)
        .map_err(|why| format!("reversed: {why}"))?;

    if forward != backward {
        return Err(format!(
            "the run reversed bound differently: {forward:?} then {backward:?}"
        ));
    }
    // And the two entries really are distinguishable, or the comparison above proves nothing: the
    // images are different textures and `images_for` alternates the filters.
    if forward[0] == forward[1] {
        return Err("both bindings got the same image and filter".into());
    }
    println!(
        "  out of order          ok   slots {slots:?} reversed, bound the same, {} distinct",
        forward.len()
    );
    Ok(())
}

fn refusals(device: &Open) -> Result<(), String> {
    let family = families::ALL
        .iter()
        .find(|f| f.name == "background_pattern")
        .ok_or("a patterned family")?;
    let bindings = pipelines::bindings(family, Surface::Plane);
    let layout =
        pipelines::layout(device.gpu(), &bindings).map_err(|why| format!("layout: {why}"))?;
    let mut sets = Sets::new(device.gpu(), 4, &bindings).map_err(|why| format!("sets: {why}"))?;

    let at = which(0);
    let blocks = blocks_for(device, at, &bindings)?;
    let slots = texture_slots(&bindings);
    let (images, refs) = images_for(device, &slots)?;
    let bound =
        descriptors::bound_from(&images, &bindings, &refs).map_err(|why| why.to_string())?;

    // Too few textures for the set's bindings.
    match sets.write(&layout, &bindings, at, &blocks, &[]) {
        Err(descriptors::Error::WrongTextureCount { wanted: 1, got: 0 }) => {}
        other => return Err(format!("no textures gave {other:?}")),
    }
    // Too many.
    let twice = [bound[0], bound[0]];
    match sets.write(&layout, &bindings, at, &blocks, &twice) {
        Err(descriptors::Error::WrongTextureCount { wanted: 1, got: 2 }) => {}
        other => return Err(format!("two textures gave {other:?}")),
    }
    // A layer with no block buffer at all.
    let empty = Blocks::new();
    match sets.write(&layout, &bindings, at, &empty, &bound) {
        Err(descriptors::Error::NoBlocks { .. }) => {}
        other => return Err(format!("no blocks gave {other:?}")),
    }
    if sets.allocated() != 0 {
        return Err("a refused write left a set allocated".into());
    }
    // A texture the store does not hold.
    let absent = TextureRef {
        texture: TextureId(99),
        slot: slots[0],
        filter: TextureFilter::Linear as u32,
    };
    match descriptors::bound_from(&images, &bindings, &[absent]) {
        Err(descriptors::Error::NoTexture { slot: 0 }) => {}
        other => return Err(format!("an absent texture gave {other:?}")),
    }
    // A run naming a slot this set has no binding for.
    match descriptors::bound_from(
        &images,
        &bindings,
        &[TextureRef {
            slot: 31,
            ..refs[0]
        }],
    ) {
        Err(descriptors::Error::UndeclaredTextureSlot { slot: 31 }) => {}
        other => return Err(format!("an undeclared slot gave {other:?}")),
    }
    // Two refs claiming one slot. Caught before the one left empty is noticed, because the
    // duplicate is the thing that went wrong and the hole is its consequence.
    match descriptors::bound_from(&images, &bindings, &[refs[0], refs[0]]) {
        Err(descriptors::Error::DuplicateTextureSlot { slot }) if slot == slots[0] => {}
        other => return Err(format!("a duplicate slot gave {other:?}")),
    }
    // A binding nothing named.
    match descriptors::bound_from(&images, &bindings, &[]) {
        Err(descriptors::Error::UnfilledTextureSlot { slot }) if slot == slots[0] => {}
        other => return Err(format!("an unfilled slot gave {other:?}")),
    }
    // A filter discriminant this build does not know, which the ABI now decodes -- so this is
    // refused rather than guessed. #95's other half.
    match descriptors::bound_from(
        &images,
        &bindings,
        &[TextureRef {
            filter: 9,
            ..refs[0]
        }],
    ) {
        Err(descriptors::Error::BadFilter { raw: 9, slot }) if slot == slots[0] => {}
        other => return Err(format!("an unknown filter gave {other:?}")),
    }

    // And the good write still works afterwards.
    sets.write(&layout, &bindings, at, &blocks, &bound)
        .map_err(|why| format!("after the refusals: {why}"))?;
    println!(
        "  refusals              ok   eight refused, none left a set, a good write still works"
    );
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
        ("every_set_written", every_set_is_written as Case),
        ("two_samplers", the_filters_are_two_samplers),
        ("out_of_order", an_out_of_order_run),
        ("refusals", refusals),
        ("bindings_per_slot", each_binding_resolves_its_own_slot),
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
