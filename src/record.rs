// SPDX-License-Identifier: BSD-2-Clause
//! Walking a frame's batches and recording the draws.
//!
//! The last join in the frame half. Everything it reads is already decided elsewhere --
//! [`crate::pipelines`] built the pipeline, [`crate::descriptors`] wrote the set,
//! [`crate::store`] holds the bind list and the segments, [`crate::draws`] turns a segment run into
//! draw parameters, and `tessella_consume::stencil` assigned the tile its masks. This puts them in
//! order.
//!
//! # What the caller supplies, and why
//!
//! A pipeline, by batch. This crate cannot compile WGSL -- `naga` is a dev-dependency, deliberately,
//! so the library ships no shader compiler -- which means the host compiles and warms
//! [`crate::pipelines::Cache`] and this records against what is already there. A resolver returning
//! `None` is a batch this consumer does not draw, which is a real case: `families::ALL` excludes the
//! debug overlays and the embedder's own geometry.
//!
//! # One pipeline and one set per batch, the rest per drawable
//!
//! A batch's key pins the layer, the family and the permutation. The descriptor set is per
//! `(view, layer)`, so it is constant across the batch; and the vertex layout is a function of the
//! same three things, which #77 works out -- so the pipeline is too. Both bind once.
//!
//! What varies per drawable is its buffers, its segments, its slot, and its **tile** -- so the
//! stencil reference and compare mask are set per drawable, not per batch. Two drawables of one
//! batch in different tiles are the ordinary case, and that is what the collapse is for.
//!
//! # A drawable with no mask
//!
//! `Partition`'s own words: "A tile absent here has no mask; its geometry is left unclipped." That
//! is expressed with a **compare mask of zero** rather than a second pipeline: the test is a baked
//! `EQUAL`, and `(0 & 0) == (stencil & 0)` passes for every texel whatever the buffer holds. So
//! unclipped is a value, not a branch -- and the same two `vkCmdSet` calls serve both.

use ash::vk;
use tessella_capture_abi::envelope::ViewId;
use tessella_consume::batch::{Batch, Batches};
use tessella_consume::join::Joiner;
use tessella_consume::stencil::Partition;
use tessella_vk::Recorder;

use crate::blocks::Which;
use crate::descriptors::Sets;
use crate::draws;
use crate::store::Store;

/// What the caller resolves a batch's program to.
#[derive(Debug, Clone, Copy)]
pub struct Program {
    /// The pipeline to bind, from the caller's warmed cache.
    pub pipeline: vk::Pipeline,
    /// Its layout, for binding the descriptor set.
    pub layout: vk::PipelineLayout,
    /// Instances per draw.
    ///
    /// One for everything but the instanced families, which bind a wall outline per instance. The
    /// caller knows which family a batch is; this does not need to.
    pub instances: u32,
}

/// Why a frame could not be recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A batch's layer has no descriptor set.
    ///
    /// The set is per `(view, layer)` and written before the frame is recorded. A batch whose layer
    /// has none is a layer the producer sent drawables for and no uniform block, which would draw
    /// with whatever was bound last.
    NoSet {
        /// The layer.
        which: Which,
    },
    /// A drawable's geometry is not on the device.
    ///
    /// Not the same as a geometry the order names and the consumer has not been given -- the
    /// collapse already drops those, because `collapse_into`'s resolver returns `None` for one. This
    /// is a geometry that resolved to a program and then had no buffers, which is a store and a
    /// joiner that disagree.
    NotResident {
        /// Which geometry.
        geometry: tessella_capture_abi::envelope::GeometryId,
    },
    /// A segment run cannot be drawn.
    Unusable(draws::Unusable),
}

impl From<draws::Unusable> for Error {
    fn from(why: draws::Unusable) -> Self {
        Self::Unusable(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoSet { which } => write!(
                f,
                "view {:?} layer {} has no descriptor set",
                which.view, which.layer
            ),
            Self::NotResident { geometry } => {
                write!(f, "geometry {} is not on the device", geometry.0)
            }
            Self::Unusable(why) => write!(f, "{why:?}"),
        }
    }
}

impl std::error::Error for Error {}

/// What one frame's recording came to.
///
/// Returned rather than logged, because the interesting numbers are ratios: batches against
/// drawables is what the collapse bought, and `undrawn` against batches is how much of the order
/// this consumer has no program for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    /// Batches whose program resolved and which were recorded.
    pub batches: usize,
    /// Drawables recorded across them.
    pub drawables: usize,
    /// `vkCmdDrawIndexed` calls recorded.
    pub draws: usize,
    /// Batches skipped because the caller resolved no program.
    pub undrawn: usize,
    /// Drawables whose tile has no mask, and which are therefore unclipped.
    pub unclipped: usize,
}

/// Everything a frame reads that is not the batches.
pub struct Scene<'a, 'd> {
    /// The geometry on the device.
    pub store: &'a Store<'d>,
    /// The descriptor sets, written for this frame.
    pub sets: &'a Sets<'d>,
    /// Announcements paired with uses, which is where a drawable's tile comes from.
    pub joiner: &'a Joiner,
    /// Each tile's stencil assignment.
    pub partition: &'a Partition,
    /// The view being drawn.
    pub view: ViewId,
}

impl core::fmt::Debug for Scene<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scene")
            .field("view", &self.view)
            .finish_non_exhaustive()
    }
}

/// Records a frame's content draws, in the order the batches are in.
///
/// Must be called inside a rendering scope, with the masks already drawn -- a content draw tests
/// against a stencil the masks wrote, so recording it first tests an empty buffer.
///
/// # Errors
///
/// [`Error`] for a layer with no set, a geometry with no buffers, or a segment run that cannot be
/// drawn. Nothing is recorded for the batch that failed; what was recorded before it stays
/// recorded, which is why this returns rather than unwinding -- a command buffer cannot be rolled
/// back and a caller that hits this should abandon the buffer rather than submit it.
pub fn content(
    record: Recorder<'_>,
    batches: &Batches,
    scene: &Scene<'_, '_>,
    program: &dyn Fn(&Batch<'_>) -> Option<Program>,
) -> Result<Counts, Error> {
    let mut counts = Counts::default();

    for batch in batches.iter() {
        let Some(found) = program(&batch) else {
            counts.undrawn += 1;
            continue;
        };

        // Per batch: the key pins the layer, the family and the permutation, and the set and the
        // pipeline are functions of exactly those.
        let which = Which {
            view: scene.view,
            layer: layer_of(&batch),
        };
        let set = scene.sets.get(which).ok_or(Error::NoSet { which })?;

        record.bind_pipeline(found.pipeline);
        record.bind_descriptor_set(found.layout, set);

        for (geometry, slot) in batch.geometries.iter().zip(batch.ubo_indexes) {
            let (buffers, offsets) = scene.store.bindings(*geometry).ok_or(Error::NotResident {
                geometry: *geometry,
            })?;

            // Per drawable, because its tile is. Two drawables of one batch in different tiles is
            // the ordinary case.
            let (reference, compare) =
                clipping(scene.joiner, scene.partition, scene.view, *geometry);
            if compare == 0 {
                counts.unclipped += 1;
            }
            record.stencil_reference(reference);
            record.stencil_compare_mask(compare);

            record.bind_vertex_buffers(0, buffers, offsets);
            if let Some(indexes) = scene.store.index_buffer(*geometry) {
                record.bind_index_buffer(indexes);
            }

            for draw in draws::indexed(scene.store.segments(*geometry), *slot, found.instances)? {
                record.draw_indexed(
                    draw.index_count,
                    draw.instance_count,
                    draw.first_index,
                    draw.vertex_offset,
                    draw.first_instance,
                );
                counts.draws += 1;
            }
            counts.drawables += 1;
        }
        counts.batches += 1;
    }

    Ok(counts)
}

/// The layer a batch belongs to, as a block buffer is keyed.
///
/// `Which::layer` is an `i32` because `UboUpdate::layer_index` is; a batch's is a `u32` because
/// `OrderEntry::layer_index` is. Converted with a saturating cast rather than a wrapping one: a
/// layer index past `i32::MAX` is not a layer, and wrapping it would name another layer's buffer.
fn layer_of(batch: &Batch<'_>) -> i32 {
    i32::try_from(batch.key.layer_index).unwrap_or(i32::MAX)
}

/// A drawable's stencil reference and compare mask.
///
/// `(0, 0)` when it has no mask, which is how unclipped is expressed -- see this module's own notes.
/// **Three** ways to get there and all mean the same thing: the joiner has no use pairing this
/// geometry with this view, the `ViewUse` carried no tile, or the tile is not in the partition.
///
/// Takes the joiner and the partition rather than the whole [`Scene`], which is what makes it
/// testable without a device: the stores need one and this does not.
#[must_use]
pub fn clipping(
    joiner: &Joiner,
    partition: &Partition,
    view: ViewId,
    geometry: tessella_capture_abi::envelope::GeometryId,
) -> (u32, u32) {
    let Some(drawable) = joiner.drawable(geometry, view) else {
        return (0, 0);
    };
    if drawable.use_.has_tile == 0 {
        return (0, 0);
    }
    partition
        .tiles
        .get(&drawable.use_.tile)
        .map_or((0, 0), |assignment| {
            (u32::from(assignment.value), u32::from(assignment.read_mask))
        })
}
