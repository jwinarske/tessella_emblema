// SPDX-License-Identifier: BSD-2-Clause
//! The vertex input a drawable asks for, planned against the family's own table.
//!
//! A geometry arrives as a run of [`AttributeDesc`] — where the bytes are, how they are strided,
//! and what type they hold. A module declares an `@location` per entry in the family's
//! [`ShaderAttribute`] table. Those two have to agree before a pipeline is built, and the place
//! to find out is here rather than in a driver: an attribute bound at a location no module
//! declares is not an error anywhere in Vulkan, it simply never arrives, and what reaches the
//! frame is a wrong pixel in one family.
//!
//! # Why the format comes from the table and not the wire
//!
//! `AttributeDesc` carries both: `data_type` is what the buffer holds and `declared_data_type` is
//! what the shader declares. The ABI says to bind the declared one, and the two differ on purpose
//! — mbgl states a type twice and the two disagree for six attributes, which is what
//! `mbgl-codegen`'s corrections table records. Binding the buffer's type instead would read the
//! right bytes through the wrong format.
//!
//! So the table is the authority and the wire is checked against it. A disagreement is refused
//! rather than resolved, because either answer draws something and only one of them is right.

use ash::vk;
use tessella_capture_abi::AttributeDataType;
use tessella_capture_abi::envelope::{AttributeDesc, SlabRef};
use tessella_capture_abi::generated::shader_attributes::ShaderAttribute;

use crate::device::vertex_format;

/// One attribute, ready to become a `VkVertexInputAttributeDescription` and its binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    /// The `@location` the module declares, which is also the binding slot.
    pub slot: u32,
    /// Format to bind, from the table's declared type.
    pub format: vk::Format,
    /// Bytes between consecutive vertices.
    pub stride: u32,
    /// Byte offset of this attribute within a vertex.
    pub offset: u32,
    /// First vertex, for a binding that does not start at the buffer's start.
    pub vertex_offset: u32,
    /// Where the bytes are.
    pub source: SlabRef,
    /// Whether the binding advances per vertex or per instance.
    ///
    /// Pipeline state, so two drawables that differ only here are two pipelines. It comes from
    /// *which run* the descriptor arrived in rather than from anything in the descriptor: the ABI
    /// splits a geometry's `attrs` from its `instance_attrs`, and an instanced family's wall
    /// outline is in the second.
    pub rate: vk::VertexInputRate,
}

/// What a drawable's descriptors came to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// The attributes to bind, in the table's order.
    pub bound: Vec<Bound>,
    /// Attribute ids dropped because the geometry bound them at `-1`.
    ///
    /// Not a problem: the ABI says a `-1` binding is an override the shader does not declare and
    /// the consumer drops it, which is what the mbgl backends do in `buildAttributeBindings`.
    pub dropped: Vec<u32>,
    /// Table entries no descriptor supplied, as their `@location`.
    ///
    /// Normal for a data-driven attribute whose paint is constant for this permutation: the
    /// module reads a uniform instead and the location goes unbound. Reported rather than
    /// resolved, because which permutation is in force is not this module's to decide.
    pub absent: Vec<u32>,
    /// Descriptors at a positive slot the family's table does not declare, as `(attr_id, slot)`.
    ///
    /// Empty for every family this producer sends today, and the one case that filled it is worth
    /// keeping in mind: the skirt flag rode the wire at binding 2 for the raster, hillshade and
    /// relief families while their tables declared two attributes each. This crate reported it
    /// here and drew without it, which loses the curtain over a crack between two tiles of raised
    /// ground -- up to 2382 holes of 1,620,000 at a high-pitch camera, which no gross-pixel count
    /// can see. tessella#331 put it in the tables, so it binds like anything else.
    ///
    /// Reported rather than refused, because that is the useful end to fail at. The next attribute
    /// this producer adds will arrive before the table describes it too, and a refusal would make
    /// the whole family undrawable over one attribute where this leaves the caller drawing without
    /// it and knowing that it did. `-1` is the other thing entirely: that is the ABI asking for a
    /// drop, and it goes in [`Self::dropped`].
    pub undeclared: Vec<(u32, i32)>,
}

/// Why a drawable's descriptors cannot be bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// A discriminant that is not an `AttributeDataType`.
    ///
    /// The record came from untrusted bytes, so this is a reachable state rather than a bug.
    BadDataType {
        /// The attribute's id.
        attr_id: u32,
        /// The discriminant that did not decode.
        raw: u8,
    },
    /// The wire's declared type disagrees with the table's.
    DeclaredDisagrees {
        /// The attribute's id.
        attr_id: u32,
        /// What the geometry said the shader declares.
        wire: AttributeDataType,
        /// What the table says it declares.
        table: AttributeDataType,
    },
    /// A declared type with no Vulkan vertex format.
    NoFormat {
        /// The attribute's id.
        attr_id: u32,
        /// The type that has no mapping.
        declared: AttributeDataType,
    },
    /// Two descriptors for the same slot.
    DuplicateSlot {
        /// The slot named twice.
        slot: i32,
    },
}

/// Plans the vertex input for one geometry against the family's table.
///
/// # Errors
///
/// [`Refused`], naming the attribute. The first disagreement rather than all of them: they are
/// resolved one at a time and a list is no more actionable than its head.
pub fn plan(table: &[ShaderAttribute], descs: &[AttributeDesc]) -> Result<Plan, Refused> {
    planned(&[(table, descs, vk::VertexInputRate::VERTEX)])
}

/// As [`plan`], with the instanced run as well.
///
/// The two runs have their own tables — `attributes` and `instance_attributes` for the same
/// shader — and their own slots, so they are planned together and the slots must still not
/// collide. An instanced family declares its position in the first and its outline, packed
/// decimals and data-driven attributes in the second.
///
/// # Errors
///
/// As [`plan`], and [`Refused::DuplicateSlot`] for a slot claimed by both runs.
pub fn plan_instanced(
    table: &[ShaderAttribute],
    descs: &[AttributeDesc],
    instance_table: &[ShaderAttribute],
    instance_descs: &[AttributeDesc],
) -> Result<Plan, Refused> {
    planned(&[
        (table, descs, vk::VertexInputRate::VERTEX),
        (
            instance_table,
            instance_descs,
            vk::VertexInputRate::INSTANCE,
        ),
    ])
}

fn planned(
    runs: &[(&[ShaderAttribute], &[AttributeDesc], vk::VertexInputRate)],
) -> Result<Plan, Refused> {
    let mut out = Plan::default();
    let mut seen: Vec<i32> = Vec::new();

    for (table, descs, rate) in runs {
        for desc in *descs {
            if desc.binding < 0 {
                out.dropped.push(desc.attr_id);
                continue;
            }
            if seen.contains(&desc.binding) {
                return Err(Refused::DuplicateSlot { slot: desc.binding });
            }
            seen.push(desc.binding);

            let Some(entry) = table.iter().find(|entry| entry.binding == desc.binding) else {
                out.undeclared.push((desc.attr_id, desc.binding));
                continue;
            };
            let Some(wire) = desc.declared_data_type() else {
                return Err(Refused::BadDataType {
                    attr_id: desc.attr_id,
                    raw: desc.declared_data_type,
                });
            };
            if wire != entry.declared {
                return Err(Refused::DeclaredDisagrees {
                    attr_id: desc.attr_id,
                    wire,
                    table: entry.declared,
                });
            }
            let Some(format) = vertex_format(entry.declared) else {
                return Err(Refused::NoFormat {
                    attr_id: desc.attr_id,
                    declared: entry.declared,
                });
            };
            // The table's own binding, which generation keeps equal to the module's `@location`.
            // Negative is unreachable -- `desc.binding` was checked non-negative and the two are
            // equal -- so a table that said otherwise is reported rather than bound.
            let Ok(slot) = u32::try_from(entry.binding) else {
                out.undeclared.push((desc.attr_id, entry.binding));
                continue;
            };
            out.bound.push(Bound {
                slot,
                format,
                stride: desc.stride,
                offset: desc.offset,
                vertex_offset: desc.vertex_offset,
                source: desc.source,
                rate: *rate,
            });
        }
    }

    out.bound.sort_unstable_by_key(|bound| bound.slot);
    for entry in runs.iter().flat_map(|(table, _, _)| table.iter()) {
        // `try_from` is also the `binding >= 0` check: a table entry cannot be bound at a
        // negative slot, and one that said so would be skipped rather than panicked over.
        if let Ok(slot) = u32::try_from(entry.binding)
            && !seen.contains(&entry.binding)
        {
            out.absent.push(slot);
        }
    }
    Ok(out)
}
