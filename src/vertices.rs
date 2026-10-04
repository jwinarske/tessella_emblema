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
    /// A descriptor bound at a slot the family's table does not declare.
    ///
    /// Distinct from a `-1` binding, which the ABI defines. This is a positive slot the table has
    /// no entry for, so there is no declared type to bind it with and no `@location` to receive
    /// it. Refused rather than dropped: a dropped attribute the producer meant to send draws a
    /// wrong picture, and the terrain curtain's skirt flag is exactly that case —
    /// `encode_color_relief` sends a third descriptor and `COLOR_RELIEF_SHADER` declares two.
    UndeclaredSlot {
        /// The attribute's id.
        attr_id: u32,
        /// The slot it asked for.
        slot: i32,
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
    let mut out = Plan::default();
    let mut seen: Vec<i32> = Vec::with_capacity(descs.len());

    for desc in descs {
        if desc.binding < 0 {
            out.dropped.push(desc.attr_id);
            continue;
        }
        if seen.contains(&desc.binding) {
            return Err(Refused::DuplicateSlot { slot: desc.binding });
        }
        seen.push(desc.binding);

        let Some(entry) = table.iter().find(|entry| entry.binding == desc.binding) else {
            return Err(Refused::UndeclaredSlot {
                attr_id: desc.attr_id,
                slot: desc.binding,
            });
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
        // The slot is the table's own `binding`, which generation keeps equal to the `@location`
        // the module declares. Taken from the entry rather than the descriptor so the two cannot
        // drift: they were compared above, and this is the one the module will read.
        let slot = u32::try_from(entry.binding).map_err(|_| Refused::UndeclaredSlot {
            attr_id: desc.attr_id,
            slot: entry.binding,
        })?;
        out.bound.push(Bound {
            slot,
            format,
            stride: desc.stride,
            offset: desc.offset,
            vertex_offset: desc.vertex_offset,
            source: desc.source,
        });
    }

    out.bound.sort_unstable_by_key(|bound| bound.slot);
    for entry in table {
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
