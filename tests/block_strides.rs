// SPDX-License-Identifier: BSD-2-Clause
//! The stride a layer's block entries sit at, as WGSL computes it.
//!
//! # What this is for
//!
//! A layer's drawable buffer is an array of the *union* of its drawable blocks, and the producer
//! says what that costs where it packs one:
//!
//! > A plain fill writes an 80-byte `FillDrawableUBO` into a 96-byte slot, because the pattern
//! > variants are larger and set the stride for everyone. Packing at 80 would put every entry after
//! > the first at the wrong offset -- a layer whose tiles are drawn with each other's matrices,
//! > which is plausible-looking output no size check would catch.
//!
//! This is the reading half. WGSL sizes `array<T>` from `T`, so a struct left at its own 80 bytes
//! reads entry one at byte 80 where the producer wrote it at 96.
//!
//! # What would be caught
//!
//! A struct whose size is not the stride its entries sit at. Measured through naga rather than by
//! counting the declaration's words, because what matters is the size *WGSL* computes: a test that
//! added up the fields and the padding it expected would agree with a wrong declaration that padded
//! the same way.
//!
//! Entry zero is at offset zero under any stride, so nothing here is visible at `ubo_index` 0 --
//! which is every case `benches/first_pixel.rs` drew before `drawable_at_one`.

use naga::front::wgsl;
use tessella_capture_abi::generated::ubo_layouts::{LAYOUTS, UNIONS, UboLayout};
use tessella_emblema::families::ALL;
use tessella_emblema::preamble::{declare, type_name};
use tessella_emblema::shaders::module;
use tessella_emblema::slots;
use tessella_emblema::surface::Surface;

/// The size WGSL gives a block's struct, out of a module that declares it.
///
/// Through the real front end: `naga::proc::Layouter` is what computes a struct's size and span,
/// and it is the same code the SPIR-V backend uses to lay the block out.
fn wgsl_size(layout: &UboLayout, entries_at: u32) -> u32 {
    let source = declare(layout, entries_at)
        .unwrap_or_else(|why| panic!("{} cannot be declared: {why:?}", layout.name));
    // A declaration alone is not a module naga will parse -- nothing uses the type -- so it is
    // given a binding, which is how `shaders::module` uses it anyway.
    let name = type_name(layout.name);
    let text = format!(
        "{source}\n@group(0) @binding(0) var<storage, read> probe: array<{name}>;\n\
         @compute @workgroup_size(1) fn main() {{ _ = probe[0]; }}\n"
    );
    let parsed = wgsl::parse_str(&text)
        .unwrap_or_else(|why| panic!("{}: {}", layout.name, why.emit_to_string(&text)));
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(parsed.to_ctx())
        .unwrap_or_else(|why| panic!("{}: {why:?}", layout.name));
    let (handle, _) = parsed
        .types
        .iter()
        .find(|(_, found)| found.name.as_deref() == Some(name.as_str()))
        .unwrap_or_else(|| panic!("{name} is not in the module"));
    layouter[handle].size
}

/// Every block's struct is exactly as large as the stride its entries sit at.
///
/// Over the generated layouts rather than over this crate's families, so it covers the blocks no
/// family here declares yet as well.
#[test]
fn a_blocks_struct_is_the_stride_its_entries_sit_at() {
    for layout in LAYOUTS {
        let wanted = slots::stride(&layout);
        assert_eq!(
            wgsl_size(&layout, wanted),
            wanted,
            "{} is declared at a size WGSL does not index at {wanted}",
            layout.name
        );
    }
}

/// A union member is padded to its union, and that is a change rather than a restatement.
///
/// The eight blocks whose own stride is below their union's are the whole of the defect: every one
/// of them was read at its own stride, so every entry after the first came from inside the one
/// before it. Five are declared by a family here -- `background`, `fill`, `fill_outline`, `line`
/// and `line_pattern` -- and the other three are blocks no family declares yet, which is why this
/// counts blocks rather than families.
///
/// Asserted as the inequality rather than as eight numbers, so a union growing a larger member
/// keeps the test meaningful.
#[test]
fn a_union_member_is_padded_to_its_union() {
    let mut widened = 0;
    for union in &UNIONS {
        for member in union.members {
            let Some(layout) = LAYOUTS.iter().find(|found| found.name == *member) else {
                continue;
            };
            assert_eq!(
                slots::stride(layout),
                union.stride,
                "{member} is a {} and does not take its stride",
                union.name
            );
            assert_eq!(wgsl_size(layout, union.stride), union.stride);
            if layout.stride < union.stride {
                widened += 1;
                // And the declaration at its own stride is the defect, which is what makes this a
                // fix: WGSL would have sized it below the entry the producer writes.
                assert!(
                    wgsl_size(layout, layout.stride) < union.stride,
                    "{member} reads the same at either stride, so nothing was wrong"
                );
            }
        }
    }
    assert_eq!(
        widened, 8,
        "eight blocks are below their union's stride; a change here means the headers moved"
    );
}

/// A block that belongs to no union keeps its own stride.
///
/// The other side: a fix that padded everything to the largest union would move the thirty-five
/// blocks that are in no union and were already right, and the producer packs those at their own
/// size.
#[test]
fn a_block_in_no_union_keeps_its_own_stride() {
    let mut kept = 0;
    for layout in LAYOUTS {
        if UNIONS
            .iter()
            .any(|union| union.members.contains(&layout.name))
        {
            continue;
        }
        assert_eq!(
            slots::stride(&layout),
            layout.stride,
            "{} is in no union and was widened anyway",
            layout.name
        );
        kept += 1;
    }
    assert_eq!(
        kept, 35,
        "thirty-five of the fifty blocks are in no union; a change means the headers moved"
    );
}

/// Every family's assembled module declares its blocks at the strides the producer packs them at.
///
/// The wiring, which the two tests above do not reach: they check `declare` and `slots::stride`
/// agree, and this checks `shaders::module` asks for the right one. A module that passed its own
/// `layout.stride` would satisfy both of them and still read entry one from inside entry zero.
#[test]
fn every_module_declares_its_blocks_at_the_producers_stride() {
    let mut checked = 0;
    for family in ALL {
        for surface in Surface::ALL {
            if !family.surfaces.contains(&surface) {
                continue;
            }
            let source = module(
                surface,
                family.blocks,
                family.attributes,
                family.textures,
                family.body,
            )
            .unwrap_or_else(|why| panic!("{} on {surface:?}: {why:?}", family.name));

            for block in family.blocks.iter().chain(surface.blocks()) {
                let wanted = slots::stride(block);
                if wanted == align_up(block.size, block.align) {
                    // Nothing to find: the fields already fill the entry.
                    continue;
                }
                let words = (wanted - align_up(block.size, block.align)) / 4;
                assert!(
                    source.contains(&format!("_tail: array<u32, {words}>")),
                    "{} on {surface:?}: {} is not padded to {wanted}",
                    family.name,
                    block.name
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "no family declares a block below its union's stride, so this checked nothing"
    );
}

/// Where a block's fields end, padded up to its alignment, which is WGSL's size without a tail.
fn align_up(at: u32, align: u32) -> u32 {
    at.div_ceil(align) * align
}
