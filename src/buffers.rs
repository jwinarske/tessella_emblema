// SPDX-License-Identifier: BSD-2-Clause
//! Which device buffers a geometry needs, and which one each binding reads.
//!
//! A drawable names its bytes by [`SlabRef`] — a slab, a byte offset into it and a length. Several
//! attributes usually name the *same* reference: the producer interleaves a vertex and gives every
//! attribute the one buffer with its own offset within a vertex. `encode_raster`,
//! `encode_hillshade` and `encode_color_relief` each send three descriptors pointing at one
//! interleaved buffer, and a consumer that allocated per descriptor would hold three copies of it.
//!
//! So the references are deduplicated and each binding records which one it reads. The dedup is by
//! the whole reference rather than by [`SlabRef::slab_and_offset`]: two references into one slab at
//! one offset but with different lengths are not the same bytes, and taking the first length would
//! leave the longer one short.
//!
//! # What this does not do
//!
//! Allocate, suballocate, or decide an alignment. Those need a device and its limits; this is the
//! part that can be decided and tested without one. It also does not read the bytes — a slab
//! reference is resolved against the host's region by `tessella_consume::slab::resolve`, and
//! whether that succeeds is the reader's business rather than the store's.

use tessella_capture_abi::envelope::SlabRef;

use crate::vertices::Plan;

/// Which buffer a binding's bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reads {
    /// The `@location` the module declares.
    pub slot: u32,
    /// Index into [`Needs::vertices`].
    pub buffer: usize,
}

/// The buffers one geometry needs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Needs {
    /// The distinct vertex buffers, in the order the bindings first named them.
    ///
    /// First-seen order rather than sorted, so the common case — one interleaved buffer — is index
    /// zero and a diagnostic reads in the order the producer wrote.
    pub vertices: Vec<SlabRef>,
    /// One per bound attribute, in the plan's slot order.
    pub reads: Vec<Reads>,
    /// The index buffer, or `None` when the geometry has no indices.
    ///
    /// Separate from `vertices` even when it points into the same slab: it binds through
    /// `vkCmdBindIndexBuffer` rather than as a vertex binding, and a store that merged them would
    /// have to un-merge them to bind.
    pub indexes: Option<SlabRef>,
}

/// What a planned drawable needs on the device.
///
/// `indexes` is [`GeometryAdd::indexes`](tessella_capture_abi::envelope::GeometryAdd::indexes). A
/// zero-length reference is no index buffer rather than an empty one, which is what a geometry
/// drawn without indices sends.
#[must_use]
pub fn needs(plan: &Plan, indexes: SlabRef) -> Needs {
    let mut out = Needs {
        vertices: Vec::new(),
        reads: Vec::with_capacity(plan.bound.len()),
        indexes: (indexes.length > 0).then_some(indexes),
    };
    for bound in &plan.bound {
        let at = out
            .vertices
            .iter()
            .position(|held| *held == bound.source)
            .unwrap_or_else(|| {
                out.vertices.push(bound.source);
                out.vertices.len() - 1
            });
        out.reads.push(Reads {
            slot: bound.slot,
            buffer: at,
        });
    }
    out
}

/// Total bytes the vertex buffers come to.
///
/// What a store asks before allocating, and what makes the dedup visible: three descriptors over
/// one interleaved buffer cost that buffer once.
#[must_use]
pub fn vertex_bytes(needs: &Needs) -> u64 {
    needs
        .vertices
        .iter()
        .map(|source| u64::from(source.length))
        .sum()
}
