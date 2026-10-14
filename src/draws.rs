// SPDX-License-Identifier: BSD-2-Clause
//! A drawable's segments as indexed draw parameters.
//!
//! A geometry arrives with a run of [`Segment`], each a contiguous index range with its own vertex
//! base. One becomes one `vkCmdDrawIndexed`. The arithmetic is small and the two places it can be
//! wrong are both silent: a `vertexOffset` that wrapped draws from somewhere else in the buffer,
//! and a `firstInstance` that is not the drawable's slot makes every body read another drawable's
//! uniforms.
//!
//! # Why `firstInstance` carries the slot
//!
//! The bodies read `@builtin(instance_index)` as `ubo_index` — the drawable's place in the
//! layer's consolidated buffer. It travels as `firstInstance` because it is one number per draw:
//! bound as a vertex attribute it would be one number per vertex, and in a push constant it would
//! be a pipeline-layout difference between two draws that are otherwise the same. So the slot is a
//! draw parameter here and nothing else in the module knows about it.
//!
//! # And why it cannot, past one instance
//!
//! `firstInstance` is not only read by the body. It is part of the address an **instance-rate
//! vertex attribute** is fetched from, so a draw that puts the slot there reads the instance
//! stream from the slot's element onward — and `@builtin(instance_index)` is the slot plus the
//! instance, so a body reading it as `ubo_index` reads another drawable's uniforms for every
//! instance but the first.
//!
//! `benches/instance_fetch.rs` measures both halves on a device, with a per-vertex binding as the
//! control. The spec says the same thing, at one remove: `VK_EXT_vertex_attribute_divisor` gives
//! the instance-rate address as `firstInstance + (instanceIndex - firstInstance) / divisor`, which
//! for the divisor of one every binding here has is the builtin itself.
//!
//! So the two uses are exclusive, and [`indexed`] refuses the combination rather than recording a
//! draw that is wrong in two ways at once. Twenty-one of the twenty-two families draw one
//! instance, where the slot and the instance index agree. The other is the instanced extrusion,
//! and it is built around this: it passes a slot of zero and takes `ubo_index` from a push
//! constant that `shaders::module` declares for it alone.

use tessella_capture_abi::envelope::Segment;

/// One `vkCmdDrawIndexed`, in that call's own argument order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draw {
    /// Indices to read.
    pub index_count: u32,
    /// Instances to draw. One for everything but the instanced families.
    pub instance_count: u32,
    /// First index, into the geometry's whole index buffer.
    pub first_index: u32,
    /// Added to every index before it reaches the vertex buffer.
    ///
    /// Signed, because that is what the call takes: mbgl's segments are non-negative and a
    /// `vertex_offset` past `i32::MAX` is refused rather than wrapped.
    pub vertex_offset: i32,
    /// The drawable's slot, which the body reads as `ubo_index`.
    ///
    /// Only a slot while [`Self::instance_count`] is one. Past that it is also the first element
    /// the instance-rate attributes are fetched from, which is why [`indexed`] refuses the two
    /// together — see the module's second section.
    pub first_instance: u32,
}

/// Why a segment run cannot be drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unusable {
    /// A `vertex_offset` that does not fit the signed field the call takes.
    VertexOffset {
        /// Which segment, by position in the run.
        segment: usize,
        /// The offset that did not fit.
        offset: u32,
    },
    /// A non-zero slot on a draw of more than one instance.
    ///
    /// The two cannot share `firstInstance`: it addresses the instance-rate attribute fetch as
    /// well as the builtin the body reads, so this draw would read the instance stream from the
    /// slot's element and every instance but the first would read another drawable's uniforms.
    /// Measured in `benches/instance_fetch.rs`.
    SlotWithInstances {
        /// The slot asked for.
        slot: u32,
        /// Instances asked for.
        instances: u32,
    },
}

/// The draws for one drawable's segments.
///
/// `slot` is the drawable's index into its layer's uniform buffer, and `instances` is how many
/// copies each segment draws — one for every family but the instanced extrusions.
///
/// A segment with no indices contributes no draw. That is not an error: a bucket can hold a
/// segment whose geometry was clipped away, and a `vkCmdDrawIndexed` of zero indices is a command
/// buffer entry that does nothing. The same goes for `instances` of zero, which is a drawable with
/// nothing to draw rather than a malformed one.
///
/// # Errors
///
/// [`Unusable::VertexOffset`] for an offset past `i32::MAX`, naming the segment, and
/// [`Unusable::SlotWithInstances`] for a non-zero slot on a draw of more than one instance.
pub fn indexed(segments: &[Segment], slot: u32, instances: u32) -> Result<Vec<Draw>, Unusable> {
    // Ahead of the segment loop, so the combination is refused on what was asked for rather than
    // on what the segments happened to hold. A draw of no instances falls past it: nothing is
    // drawn, so there is nothing for the two numbers to disagree about.
    if instances > 1 && slot != 0 {
        return Err(Unusable::SlotWithInstances { slot, instances });
    }
    if instances == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(segments.len());
    for (at, segment) in segments.iter().enumerate() {
        let vertex_offset =
            i32::try_from(segment.vertex_offset).map_err(|_| Unusable::VertexOffset {
                segment: at,
                offset: segment.vertex_offset,
            })?;
        // Checked before the empty test, so a malformed offset is reported whether or not the
        // segment would have drawn. A run that is refused is refused on its own contents rather
        // than on which of its segments happened to be empty.
        if segment.index_length == 0 {
            continue;
        }
        out.push(Draw {
            index_count: segment.index_length,
            instance_count: instances,
            first_index: segment.index_offset,
            vertex_offset,
            first_instance: slot,
        });
    }
    Ok(out)
}
