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
//! `(view, layer, family)` and the batch is one of each, so it is constant across the batch; and
//! the vertex layout is a function of the same three things, which #77 works out -- so the pipeline
//! is too. Both bind once.
//!
//! The family is part of the set's key because a layer is drawn by more than one -- a fill and its
//! outline share a `layer_index` and declare different blocks. Keyed by the layer alone, the second
//! family's set replaced the first's and the wrong one was bound.
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
//!
//! That spelling is the clip scheme's alone. A 3D drawable compares `NOT_EQUAL`, where a mask of
//! no bits makes both sides zero and *fails* everywhere -- so the scheme is baked per pipeline
//! ([`crate::pipelines::Stencil`]) and a 3D run with no value to spend has its test turned off
//! rather than neutralized. See [`DrawOnce`] for what a value is and how one frame has one.

use ash::vk;
use tessella_capture_abi::envelope::{DrawFlags, GeometryId, ViewId};
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
    ///
    /// More than one and the slot cannot travel in `firstInstance`: it also addresses the
    /// instance-rate attribute fetch, so the two cannot share it. [`crate::draws::indexed`]
    /// refuses the pair rather than recording it, and [`Self::per_drawable`] is how the slot
    /// travels instead.
    pub instances: u32,
    /// Whether this family's per-drawable state is bound rather than indexed.
    ///
    /// False for every family whose draws differ only by `ubo_index`: the slot rides
    /// `firstInstance`, the layer's one descriptor set serves every drawable, and recording a
    /// drawable is a `vkCmdDrawIndexed` and nothing else.
    ///
    /// True for a family drawn with instance-rate attributes, where two things move per drawable
    /// instead -- `ubo_index` into a push constant, and the drawable's own instance buffer into a
    /// group-one descriptor set. Both are set here per drawable rather than per batch, which is
    /// what this flag buys and what it costs.
    pub per_drawable: bool,
}

/// Why a frame could not be recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A batch's layer and family have no descriptor set.
    ///
    /// The set is per `(view, layer, family)` and written before the frame is recorded. A batch
    /// with none is a layer the producer sent drawables for and no uniform block, which would draw
    /// with whatever was bound last.
    ///
    /// The family is part of it because a layer is drawn by more than one: mbgl's fill layer emits
    /// a fill drawable *and* an outline drawable at the same `layer_index`, and the captured Berlin
    /// frame has that twice over plus an extrusion drawn by two families. Keyed by the layer alone,
    /// the second family's set replaced the first's and this error could not be reached -- the
    /// wrong set was bound instead, to a pipeline layout it does not match.
    NoSet {
        /// The layer.
        which: Which,
        /// The shader the batch names, as `batch::Key::builtin_shader`.
        shader: i32,
    },
    /// A `per_drawable` family's drawable has no group-one descriptor set.
    ///
    /// The set names this geometry's instance buffer and is written before the frame is recorded,
    /// as a layer's set is. A drawable without one is a caller that resolved the batch to a
    /// `per_drawable` program and did not write the sets that program needs -- and drawing anyway
    /// would read the instance array through whatever was bound at group one last, which is
    /// another drawable's outline or nothing at all.
    NoInstanceSet {
        /// Which geometry.
        geometry: GeometryId,
    },
    /// A drawable's geometry is not on the device.
    ///
    /// Not the same as a geometry the order names and the consumer has not been given -- the
    /// collapse already drops those, because `collapse_into`'s resolver returns `None` for one.
    /// This is a geometry that resolved to a program and then had no buffers, which is a store and
    /// a joiner that disagree.
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
            Self::NoSet { which, shader } => write!(
                f,
                "view {:?} layer {} has no descriptor set for shader {shader}",
                which.view, which.layer
            ),
            Self::NoInstanceSet { geometry } => {
                write!(f, "geometry {} has no instance descriptor set", geometry.0)
            }
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
    /// Drawables that asked for their tile's clip mask and had none.
    ///
    /// Narrower than it was, and the narrowing is the point: this counted every drawable whose
    /// `clipping` answered `(0, 0)`, which included every drawable that never wanted a mask -- a
    /// background covers the viewport and carries no stencil bit at all. Those are
    /// [`crate::pipelines::Stencil::Off`] now and are not counted, so what is left is the case
    /// worth noticing: a drawable clipped to a tile the partition had no assignment for. The
    /// captured Berlin frame went from 16 to 0 on that reading and nothing about it changed.
    pub unclipped: usize,
    /// Drawables drawn with the 3D draw-once mask.
    ///
    /// The run [`DrawOnce`] gave the frame's one value to, drawable by drawable.
    pub drawn_once: usize,
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
    /// Which 3D run drew with the draw-once mask, from [`DrawOnce::of`].
    ///
    /// The caller's rather than this module's, because the caller builds the pipelines: a
    /// drawable's scheme is baked state ([`crate::pipelines::Stencil`]) and the dynamic reference
    /// set here has to agree with it. Both read [`DrawOnce::value_for`], so there is one answer.
    ///
    /// [`DrawOnce::default`] is a frame with no 3D run, which is every frame of a flat style.
    pub draw_once: DrawOnce,
    /// The view being drawn.
    pub view: ViewId,
    /// How a layer's drawables are placed in the depth range, or `None` to place none.
    ///
    /// `None` is a frame with no depth to order with: a view whose attachment is
    /// `Attachment::StencilOnly`, or an order that carried no layer count so no slot can be
    /// computed. The draws then all land at the viewport's default range and
    /// the painting order turns the passes around instead.
    pub depth: Option<DepthRange>,
    /// The target's size, for the viewport each depth band is set through.
    pub size: (u32, u32),
}

/// What divides the depth range between a view's layers.
///
/// §11.7 asks for "opaque layers front-to-back with depth writes per `pass` + `opaquePassCutoff`",
/// and this is the per-layer half of it: each layer gets a band of the range and the slot says
/// which, so a nearer layer's fragments reject the ones behind them.
///
/// # The band is the consumer's to choose
///
/// tessella's `depth_slot` says so in as many words -- "The absolute base is a consumer-side depth
/// mapping and mbgl's own count includes groups this frontend does not model, so what is
/// reproduced here is the ordering, not mbgl's particular integers." So this divides
/// `CameraUpdate::depth_range_size` evenly by the layer count rather than reproducing mbgl's
/// epsilon arithmetic, which would need two constants the ABI does not publish.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthRange {
    /// Layers in the style, from `OrderUpdate::layer_count` by way of `Frame::layer_count`.
    pub layers: core::num::NonZeroU32,
    /// The range the layers divide, from `CameraUpdate::depth_range_size`.
    pub size: f32,
}

impl DepthRange {
    /// The range a view divides, or `None` where it divides nothing.
    ///
    /// A `size` that is not positive and finite orders no layers: every band would be the same
    /// zero-width slice and every drawable would land on one depth, which is the state this pass
    /// was already in. `CameraUpdate::depth_range_size` is zero from a producer that does not set
    /// it -- the field was there before anything read it -- so this is a real input and not a
    /// defensive check. The caller then leaves [`Scene::depth`] `None` and the painting order turns the
    /// opaque pass around instead, which is correct without depth.
    #[must_use]
    pub fn new(layers: core::num::NonZeroU32, size: f32) -> Option<Self> {
        (size.is_finite() && size > 0.0).then_some(Self { layers, size })
    }

    /// The band a layer's drawables are placed in, as `(near, far)`.
    ///
    /// Slot zero is the top of the style and nearest, which is the direction
    /// `tessella_capture_abi::envelope::depth_slot` counts in -- so a band's `near` grows with the
    /// slot and the opaque pass, drawn front to back, rejects what is behind it.
    ///
    /// A vertex stage emitting clip-space `z` of zero lands exactly on `near`, which is what gives
    /// a layer a depth without any shader knowing which layer it is in.
    #[must_use]
    pub fn band(self, layer_index: u32) -> (f32, f32) {
        let layers = self.layers.get();
        let slot = tessella_capture_abi::envelope::depth_slot(
            i32::try_from(layer_index).unwrap_or(i32::MAX),
            layers,
        );
        // `f64` for the division and back, because a style of many layers divides a range smaller
        // than one into bands whose ends must stay ordered: at 2,048 layers an `f32` band is near
        // the last place it can hold and two adjacent slots could round together.
        let width = f64::from(self.size) / f64::from(layers);
        let near = width * f64::from(slot);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "both are products of a range at most one and a slot below the layer count, \
                      so each is inside 0.0..=1.0 and `viewport_in` clamps regardless"
        )]
        (near as f32, (near + width) as f32)
    }
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

    for at in painted(batches, scene.depth.is_some()) {
        let Some(batch) = batches.get(at) else {
            continue;
        };
        let Some(found) = program(&batch) else {
            counts.undrawn += 1;
            continue;
        };

        // Per batch: the key pins the layer, the family and the permutation, and the set and the
        // pipeline are functions of exactly those.
        //
        // The family as well as the layer, because one layer is drawn by more than one of them --
        // a fill and its outline share a `layer_index` and declare different blocks, so they
        // cannot share a set. See [`Error::NoSet`].
        let which = Which {
            view: scene.view,
            layer: layer_of(&batch),
        };
        let shader = batch.key.builtin_shader;
        let set = scene
            .sets
            .get(which, shader)
            .ok_or(Error::NoSet { which, shader })?;

        record.bind_pipeline(found.pipeline);
        record.bind_descriptor_set(found.layout, set);

        // Per batch, because a batch is one layer: `batch::Key` carries `layer_index` and collapse
        // is scoped to `(layer, pass)`, so every drawable in a run belongs in one band. Set through
        // the viewport rather than the shader, so no family's module has to know its layer.
        if let Some(range) = scene.depth {
            let (near, far) = range.band(batch.key.layer_index);
            record.viewport_in(scene.size.0, scene.size.1, near, far);
        }

        // Per batch, because the scheme is baked in the pipeline the caller resolved: the flags
        // are the batch's and so is the run.
        let scheme = crate::pipelines::Stencil::of(
            batch.key.draw_flags,
            scene.draw_once.value_for(batch.key),
        );

        for (geometry, slot) in batch.geometries.iter().zip(batch.ubo_indexes) {
            let (buffers, offsets) = scene.store.bindings(*geometry).ok_or(Error::NotResident {
                geometry: *geometry,
            })?;

            // Per drawable, because its tile is -- two drawables of one batch in different tiles
            // is the ordinary case -- except under the draw-once, whose value is the run's and is
            // the same for every drawable of it. That is the point of it: a roof and the wall
            // raised off it are two batches of one run and the second must see the first's mark.
            let (reference, compare, write) = match scheme {
                crate::pipelines::Stencil::DrawOnce => {
                    counts.drawn_once += 1;
                    // The field is one bit, so the value and the mask it is compared and written
                    // through are the same number.
                    (
                        u32::from(DRAW_ONCE_BIT),
                        u32::from(DRAW_ONCE_BIT),
                        u32::from(DRAW_ONCE_BIT),
                    )
                }
                crate::pipelines::Stencil::Clip => {
                    let (reference, compare) =
                        clipping(scene.joiner, scene.partition, scene.view, *geometry);
                    if compare == 0 {
                        counts.unclipped += 1;
                    }
                    (reference, compare, 0)
                }
                // The pipeline's test is off, so none of the three is read. Set anyway, because
                // dynamic state is the command buffer's and not the pipeline's: a later draw that
                // does test would otherwise inherit whatever this one left.
                crate::pipelines::Stencil::Off => (0, 0, 0),
            };
            record.stencil_reference(reference);
            record.stencil_compare_mask(compare);
            record.stencil_write_mask(write);

            record.bind_vertex_buffers(0, buffers, offsets);
            if let Some(indexes) = scene.store.index_buffer(*geometry) {
                record.bind_index_buffer(indexes);
            }

            // Per drawable, for the family that cannot put its slot in `firstInstance`: the
            // instance buffer is this geometry's own, so its set is too, and `ubo_index` goes in
            // a push constant. The draw then starts at instance zero, which is what makes the
            // instance-rate attributes fetch from the start of that buffer.
            let slot = if found.per_drawable {
                let instances = scene
                    .sets
                    .instances(*geometry)
                    .ok_or(Error::NoInstanceSet {
                        geometry: *geometry,
                    })?;
                record.bind_descriptor_set_at(1, found.layout, instances);
                record.push_constants(found.layout, &slot.to_le_bytes());
                0
            } else {
                *slot
            };

            for draw in draws::indexed(scene.store.segments(*geometry), slot, found.instances)? {
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

/// The batches in the order they are painted, which is not the order they arrive.
///
/// # Topmost first is what arrives
///
/// A producer sorts each pass by depth slot ascending, which puts the *topmost* layer first, and
/// `Placed::sort_key` on tessella's side is explicit that this is deliberate and that acting on
/// it is the consumer's job:
///
/// > mbgl does not paint that way: it draws front-to-back against a depth buffer, with each layer
/// > at its own slot, so a lower layer is rejected where a higher one already covered. [...] A
/// > consumer with no depth buffer therefore cannot paint the order as given; it has to reverse
/// > each translucent run itself.
///
/// This pass has no depth buffer to order with. Every draw lands at the same depth, because
/// nothing here assigns a per-layer depth range -- and a view of flat layers takes
/// `Attachment::StencilOnly`, where on some devices there is no depth aspect to write at all. So
/// the reversal is unconditional rather than a fallback for the flat case.
///
/// Measured before it was written: with the fixture emitting the producer's order, `a_frame`'s
/// sampled band came out holding a lower layer's color over the top one. See #101.
///
/// # The opaque pass depends on whether depth orders the frame
///
/// `depth_orders` is [`Scene::depth`] being set. With it, the opaque pass is left as it arrives:
/// front-to-back is the order it wants, each layer is in its own band, and a nearer layer rejects
/// what is behind it -- which is the early-out, 89% of a sixteen-layer frame on V3D in #104.
///
/// Without it the opaque pass is reversed too. #101 left it alone on the reasoning that
/// front-to-back is what it wants, which is true *against a depth buffer* and false without one:
/// with nothing to reject a covered fragment, a lower opaque layer drawn later paints over the
/// higher one. Marking the fixture's layers `RenderPass::OPAQUE` put a lower layer's color in the
/// sampled band, the same failure #101 opened with.
///
/// For opaque geometry the two agree in output -- painting back-to-front and testing front-to-back
/// both leave the nearest layer visible -- so neither is a compromise. One is faster and needs a
/// depth attachment; the other needs nothing.
///
/// The translucent pass is reversed either way. It is painted back to front and does not write
/// depth, so there is nothing for a depth buffer to order it by.
///
/// # What is not reversed
///
/// The inside of a layer's run. Between layers the order is front-to-back and has to be turned
/// around; within one it is already the order mbgl paints in -- a fill's triangles then its
/// outline, a symbol's sprites then its halo then its letters, an extrusion's roof then the walls
/// raised on it. tessella's note names the trap too, which is that getting this wrong hides:
///
/// > Reversing per drawable is invisible for as long as a layer's parts agree in color, which is
/// > why it survived.
///
/// So this reverses the *sequence of runs* within each pass and leaves each run as it was given.
///
/// Returns indices rather than batches so the borrow stays with the caller, and one small
/// allocation per frame rather than one per batch.
fn painted(batches: &Batches, depth_orders: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(batches.len());
    let mut at = 0;
    while at < batches.len() {
        let Some(first) = batches.get(at) else {
            break;
        };
        let pass = first.key.pass;
        // A pass's batches are contiguous, because the producer sorts by pass before anything
        // else. Found by scanning rather than assumed to be two: a pass this build does not know
        // is then its own run and is painted in the order it arrived.
        //
        // Starts past the batch that opened the run, so the scan advances whatever the comparison
        // says. A run that could come back empty is an outer loop that never progresses, and a
        // mutation of the layer comparison below proved that the hard way: it did not fail the
        // tests, it exhausted memory and the runner killed the process.
        let mut end = at + 1;
        while batches.get(end).is_some_and(|batch| batch.key.pass == pass) {
            end += 1;
        }

        // The runs of one layer each, collected so they can be emitted back to front.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut start = at;
        while start < end {
            let Some(batch) = batches.get(start) else {
                break;
            };
            let layer = batch.key.layer_index;
            let mut stop = start + 1;
            while stop < end
                && batches
                    .get(stop)
                    .is_some_and(|other| other.key.layer_index == layer)
            {
                stop += 1;
            }
            runs.push((start, stop));
            start = stop;
        }
        if depth_orders && pass & OPAQUE != 0 {
            out.extend(at..end);
        } else {
            for (start, stop) in runs.into_iter().rev() {
                out.extend(start..stop);
            }
        }
        at = end;
    }
    out
}

/// `RenderPass::OPAQUE`, which `batch::Key::pass` carries as its bits.
const OPAQUE: u8 = tessella_capture_abi::RenderPass::OPAQUE.bits();

/// The layer a batch belongs to, as a block buffer is keyed.
///
/// `Which::layer` is an `i32` because `UboUpdate::layer_index` is; a batch's is a `u32` because
/// `OrderEntry::layer_index` is. Converted with a saturating cast rather than a wrapping one: a
/// layer index past `i32::MAX` is not a layer, and wrapping it would name another layer's buffer.
fn layer_of(batch: &Batch<'_>) -> i32 {
    i32::try_from(batch.key.layer_index).unwrap_or(i32::MAX)
}

/// The bit a 3D run's draw-once mask is written over.
///
/// The top of the stencil byte, and a choice rather than a derivation: `stencil::partition` packs
/// its clip fields from the low bits up, so the high bit is the one most likely to be free -- but
/// that packing is "a choice, not a requirement" in its own words, so [`DrawOnce::of`] *checks*
/// the assignments rather than relying on it.
///
/// One bit, so one run of a frame can have it. The sibling consumer reserves the same bit for the
/// same reason and measured the cost of leaving the clip fields seven over 906 frames: no frame's
/// fields needed the eighth.
pub const DRAW_ONCE_BIT: u8 = 0x80;

/// Which run of a frame drew with the 3D draw-once mask, and what it cost the rest.
///
/// mbgl allocates a fresh stencil id for every layer group with 3D features and repaints the clip
/// masks afterwards (`vulkan/tile_layer_group.cpp`, `PaintParameters::stencilModeFor3D`). A
/// consumer drawing the whole frame in one pass cannot repaint anything, so the value has to come
/// out of a bit the clip masks are not using -- see [`DRAW_ONCE_BIT`] -- and there is one of those.
///
/// So: the first 3D run in paint order takes it, any other is drawn with no stencil at all, and
/// [`Self::skipped`] counts those. That is a real difference from the oracle and is counted rather
/// than hidden, which is the sibling consumer's arrangement too -- its `drawOnceSkipped` is this
/// number.
///
/// # What a run is
///
/// `(layer_index, pass)`, which is mbgl's layer group: `TileLayerGroup::render` takes one mode for
/// everything it draws, and it is called once per group per pass. A fill extrusion is two batches
/// of one layer -- the roofs and the walls -- and they share the value, which is what makes a roof
/// reject the wall that meets it rather than blending twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawOnce {
    /// The value, and the run that took it as `(layer_index, pass)`.
    taken: Option<(u8, (u32, u8))>,
    /// How many other 3D runs wanted one and were drawn with no stencil instead.
    pub skipped: usize,
}

impl DrawOnce {
    /// What this frame can spend, from its batches and its clip assignments.
    ///
    /// `depth_orders` is [`Scene::depth`]`.is_some()`, because the answer depends on paint order
    /// and the opaque pass is only reversed where there is no depth attachment to order it.
    ///
    /// Nothing is taken when no batch asks for one, and nothing is taken when the clip fields hold
    /// [`DRAW_ONCE_BIT`] -- in `value`, in `write_mask` or in `read_mask`, because a mask writing
    /// that bit or a drawable comparing it would read this scheme's marks as its own clip. The
    /// fallback partition (`partitioned: false`) spends the whole byte on a value per tile, so it
    /// leaves nothing either.
    #[must_use]
    pub fn of(batches: &Batches, partition: &Partition, depth_orders: bool) -> Self {
        let free = partition.partitioned
            && partition.tiles.values().all(|assignment| {
                (assignment.value | assignment.read_mask | assignment.write_mask) & DRAW_ONCE_BIT
                    == 0
            });
        let mut out = Self::default();
        for at in painted(batches, depth_orders) {
            let Some(batch) = batches.get(at) else {
                continue;
            };
            if !wants_draw_once(batch.key.draw_flags) {
                continue;
            }
            let run = (batch.key.layer_index, batch.key.pass);
            // Another batch of the run that already took it is the ordinary case and costs
            // nothing: an extrusion's roofs and its walls are two batches of one run, and the
            // value being shared is what makes the roof reject the wall that meets it.
            if out.taken.is_some_and(|(_, took)| took == run) {
                continue;
            }
            // The first run takes the bit where the clip fields left one. Any other run, and every
            // run in a frame whose fields took the bit, is drawn with no stencil and counted --
            // there is one value and mbgl has as many as it needs.
            if out.taken.is_none() && free {
                out.taken = Some((DRAW_ONCE_BIT, run));
            } else {
                out.skipped += 1;
            }
        }
        out
    }

    /// The value a batch draws its draw-once mask with, or `None` for one that has none.
    ///
    /// What [`crate::pipelines::Stencil::of`] takes, so the pipeline's scheme and the dynamic state
    /// this module sets come from one answer rather than from two reads of the same rule.
    #[must_use]
    pub fn value_for(&self, key: &tessella_consume::batch::Key) -> Option<u8> {
        let (value, run) = self.taken?;
        (wants_draw_once(key.draw_flags) && run == (key.layer_index, key.pass)).then_some(value)
    }
}

/// Whether a batch's flags are the pair mbgl reads for the 3D stencil.
///
/// `IS_3D` with `ENABLE_STENCIL`, which is `drawable.getIs3D()` and `getEnableStencil()` in
/// `TileLayerGroup::render`. The asymmetry is required rather than incidental: an extrusion's depth
/// prepass is `IS_3D` and carries no stencil bit, and a prepass that wrote the run's value would
/// leave the color pass comparing `NOT_EQUAL` against its own mark at every pixel the building
/// covers -- the layer would vanish.
fn wants_draw_once(draw_flags: u8) -> bool {
    const WANTED: u8 = DrawFlags::IS_3D.bits() | DrawFlags::ENABLE_STENCIL.bits();
    draw_flags & WANTED == WANTED
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

#[cfg(test)]
mod tests {
    use super::{DRAW_ONCE_BIT, DepthRange, DrawOnce, Partition, painted};
    use tessella_capture_abi::RenderPass;
    use tessella_capture_abi::envelope::{DrawFlags, GeometryId, OrderEntry};
    use tessella_capture_abi::generated::mbgl_enums::BuiltIn;
    use tessella_consume::batch::{Batches, Program, collapse_into};

    fn entry(geometry: u64, layer: u32, pass: RenderPass) -> OrderEntry {
        OrderEntry {
            geometry: GeometryId(geometry),
            draw_priority: 0,
            layer_index: layer,
            sub_layer_index: 0,
            ubo_index: 0,
            pass,
            _pad: [0; 3],
        }
    }

    /// Collapses with each geometry's shader taken from its id, so an order can ask for two
    /// batches inside one layer -- which is what a fill and its outline are.
    fn batches_of(order: &[OrderEntry]) -> Batches {
        let mut batches = Batches::new();
        collapse_into(order, &mut batches, |id| {
            Some(Program {
                builtin_shader: match id.0 % 2 {
                    0 => BuiltIn::FillShader as i32,
                    _ => BuiltIn::FillOutlineShader as i32,
                },
                permutation_key: 0,
                draw_flags: 0,
                texture_refs: &[],
            })
        });
        batches
    }

    /// Collapses with the flags each geometry's id asks for, which is what a 3D run needs.
    ///
    /// Every geometry is the same family, so what breaks a batch here is the flags and the layer:
    /// `batch::Key` carries the whole byte for exactly that reason.
    fn flagged(order: &[OrderEntry], flags: &dyn Fn(GeometryId) -> u8) -> Batches {
        let mut batches = Batches::new();
        collapse_into(order, &mut batches, |id| {
            Some(Program {
                builtin_shader: BuiltIn::FillExtrusionShader as i32,
                permutation_key: 0,
                draw_flags: flags(id),
                texture_refs: &[],
            })
        });
        batches
    }

    /// The flags an extrusion's color pass carries, which is the pair the draw-once reads.
    const COLOR_3D: u8 = DrawFlags::IS_3D.bits()
        | DrawFlags::ENABLE_STENCIL.bits()
        | DrawFlags::ENABLE_DEPTH.bits()
        | DrawFlags::ENABLE_COLOR.bits();

    /// And its depth pass, which is 3D and asks for no stencil.
    const DEPTH_3D: u8 = DrawFlags::IS_3D.bits() | DrawFlags::ENABLE_DEPTH.bits();

    /// A partition with one tile, whose fields leave `DRAW_ONCE_BIT` alone.
    fn partitioned(write_mask: u8) -> Partition {
        use tessella_capture_abi::envelope::TileId;
        use tessella_consume::stencil::Assignment;

        let mut tiles = std::collections::BTreeMap::new();
        tiles.insert(
            TileId {
                x: 1,
                y: 1,
                z: 14,
                overscaled_z: 14,
                wrap: 0,
            },
            Assignment {
                value: 1,
                read_mask: write_mask,
                write_mask,
            },
        );
        Partition {
            partitioned: true,
            tiles,
        }
    }

    /// A frame with no 3D run spends nothing and skips nothing.
    #[test]
    fn a_flat_frame_takes_no_draw_once_value() {
        let order = [entry(1, 0, RenderPass::TRANSLUCENT)];
        let batches = flagged(&order, &|_| DrawFlags::ENABLE_STENCIL.bits());
        let once = DrawOnce::of(&batches, &partitioned(0x0F), true);
        assert_eq!(once, DrawOnce::default());
        assert_eq!(
            once.value_for(batches.get(0).expect("a batch").key),
            None,
            "a 2D drawable draws with its tile's clip mask, not with a run's value"
        );
    }

    /// One run takes the bit, and every batch of that run draws with it.
    ///
    /// Two batches of one layer, which is what an extrusion is: the roofs and the walls. They have
    /// to share the value -- that is what makes the wall meeting a roof fail the test rather than
    /// blend over it -- so this is the case the whole scheme turns on.
    #[test]
    fn one_run_takes_the_bit_and_its_batches_share_it() {
        let order = [
            entry(1, 7, RenderPass::TRANSLUCENT),
            entry(2, 7, RenderPass::TRANSLUCENT),
        ];
        // Two batches of one run: the flags break the run, as an extrusion's depth pass and its
        // color pass do.
        let batches = flagged(&order, &|id| {
            if id.0 == 1 { DEPTH_3D } else { COLOR_3D }
        });
        assert_eq!(batches.len(), 2, "the flags did not break the run");

        let once = DrawOnce::of(&batches, &partitioned(0x0F), true);
        assert_eq!(once.skipped, 0);
        let depth = batches.get(0).expect("a batch");
        let color = batches.get(1).expect("a batch");
        assert_eq!(
            once.value_for(depth.key),
            None,
            "the depth pass asks for no stencil, and a prepass that marked the pixels would \
             leave the color pass rejected everywhere"
        );
        assert_eq!(once.value_for(color.key), Some(DRAW_ONCE_BIT));

        // A second color batch of the same layer and pass -- the walls beside the roofs -- draws
        // with the same value.
        let order = [
            entry(3, 7, RenderPass::TRANSLUCENT),
            entry(5, 7, RenderPass::TRANSLUCENT),
        ];
        let batches = flagged(&order, &|_| COLOR_3D);
        let once = DrawOnce::of(&batches, &partitioned(0x0F), true);
        assert_eq!(once.skipped, 0);
        for at in 0..batches.len() {
            assert_eq!(
                once.value_for(batches.get(at).expect("a batch").key),
                Some(DRAW_ONCE_BIT)
            );
        }
    }

    /// A second run gets nothing, and is counted rather than hidden.
    ///
    /// mbgl allocates a value per layer group and repaints the clip masks afterwards; one pass has
    /// one spare bit, so the second 3D layer of a frame is drawn with no stencil at all. That is a
    /// real difference from the oracle -- some of its pixels blend twice where its own surfaces
    /// meet -- and the count is how a caller finds out rather than looking at the picture.
    #[test]
    fn a_second_run_is_skipped_and_counted() {
        let order = [
            entry(1, 2, RenderPass::TRANSLUCENT),
            entry(2, 5, RenderPass::TRANSLUCENT),
        ];
        let batches = flagged(&order, &|_| COLOR_3D);
        assert_eq!(batches.len(), 2, "two layers are two runs");

        let once = DrawOnce::of(&batches, &partitioned(0x0F), true);
        assert_eq!(once.skipped, 1);
        let taken: Vec<Option<u8>> = (0..batches.len())
            .map(|at| once.value_for(batches.get(at).expect("a batch").key))
            .collect();
        assert_eq!(
            taken.iter().filter(|value| value.is_some()).count(),
            1,
            "two runs drew with the one value"
        );
    }

    /// One layer in two passes is two runs, because mbgl's layer group is rendered once per pass.
    #[test]
    fn a_layers_two_passes_are_two_runs() {
        let order = [
            entry(1, 3, RenderPass::OPAQUE),
            entry(2, 3, RenderPass::TRANSLUCENT),
        ];
        let batches = flagged(&order, &|_| COLOR_3D);
        assert_eq!(batches.len(), 2, "the pass is part of the merge key");
        let once = DrawOnce::of(&batches, &partitioned(0x0F), true);
        assert_eq!(once.skipped, 1, "one layer, two passes, one value");
    }

    /// A frame whose clip fields hold the bit spends nothing on the draw-once.
    ///
    /// The clip masks come first: they are what keeps every other layer inside its tile, and a
    /// draw-once writing over a field would erase the mask of every layer drawn after it. So a
    /// frame whose fields reach the top bit draws its 3D run with no stencil and counts it.
    #[test]
    fn a_frame_whose_fields_take_the_bit_spends_nothing() {
        let order = [entry(1, 0, RenderPass::TRANSLUCENT)];
        let batches = flagged(&order, &|_| COLOR_3D);

        for assignment in [DRAW_ONCE_BIT, DRAW_ONCE_BIT | 0x01] {
            let once = DrawOnce::of(&batches, &partitioned(assignment), true);
            assert_eq!(once.skipped, 1, "{assignment:#010b}");
            assert_eq!(
                once.value_for(batches.get(0).expect("a batch").key),
                None,
                "{assignment:#010b}"
            );
        }

        // And the fallback partition, which spends the whole byte on a value per tile whatever the
        // budget said -- so there is nothing left however its fields look.
        let mut fallen = partitioned(0x0F);
        fallen.partitioned = false;
        let once = DrawOnce::of(&batches, &fallen, true);
        assert_eq!(once.skipped, 1);
        assert_eq!(once.value_for(batches.get(0).expect("a batch").key), None);
    }

    fn layers_of(batches: &Batches, order: &[usize]) -> Vec<u32> {
        order
            .iter()
            .map(|at| batches.get(*at).expect("a batch").key.layer_index)
            .collect()
    }

    /// A translucent pass is painted with its layer runs reversed.
    ///
    /// The producer sends the topmost layer first; with no depth buffer this pass has to turn
    /// that around. Three layers, one batch each, so what this pins is the run *sequence*.
    #[test]
    fn the_translucent_runs_are_painted_back_to_front() {
        let order = [
            entry(0, 2, RenderPass::TRANSLUCENT),
            entry(2, 1, RenderPass::TRANSLUCENT),
            entry(4, 0, RenderPass::TRANSLUCENT),
        ];
        let batches = batches_of(&order);
        assert_eq!(batches.len(), 3, "a layer boundary breaks a collapse");
        assert_eq!(
            layers_of(&batches, &painted(&batches, false)),
            vec![0, 1, 2],
            "the bottom layer is painted first"
        );
    }

    /// And a run of more than one batch keeps its own order.
    ///
    /// The trap tessella's `sort_key` names: "Reversing per drawable is invisible for as long as a
    /// layer's parts agree in color, which is why it survived." A fill and its outline are two
    /// batches of one layer, and the outline has to follow the fill -- so the run sequence
    /// reverses and the inside of a run does not.
    ///
    /// `benches/a_frame.rs` cannot catch this: every layer there is a single batch, so a run's
    /// internal order is one element and reversing it is a no-op. This is where it is held.
    #[test]
    fn a_run_of_two_batches_keeps_its_order() {
        // Layer 1 is a fill then its outline; layer 0 is one fill. Topmost first, as sent.
        let order = [
            entry(0, 1, RenderPass::TRANSLUCENT),
            entry(1, 1, RenderPass::TRANSLUCENT),
            entry(2, 0, RenderPass::TRANSLUCENT),
        ];
        let batches = batches_of(&order);
        assert_eq!(batches.len(), 3, "the outline is its own batch");

        let order = painted(&batches, false);
        assert_eq!(
            layers_of(&batches, &order),
            vec![0, 1, 1],
            "layer 0 first, then layer 1's two batches"
        );
        // The fill before its outline, which is the order it arrived in.
        let shaders: Vec<i32> = order
            .iter()
            .map(|at| batches.get(*at).expect("a batch").key.builtin_shader)
            .collect();
        assert_eq!(
            shaders,
            vec![
                BuiltIn::FillShader as i32,
                BuiltIn::FillShader as i32,
                BuiltIn::FillOutlineShader as i32,
            ],
            "the outline follows the fill it outlines, reversal or not"
        );
    }

    /// The opaque pass reverses too, and a mixed order keeps its passes apart.
    ///
    /// This test asserted the opposite when it was written, and was wrong. Front-to-back is the
    /// order the opaque pass wants *against a depth buffer*; with nothing to reject a covered
    /// fragment, a lower opaque layer drawn later paints over the higher one. Marking `a_frame`'s
    /// layers `RenderPass::OPAQUE` put a lower layer's color in the sampled band, which is the
    /// same failure #101 opened with.
    ///
    /// The producer sorts by pass before anything else, so the two never interleave and each is
    /// reversed within itself.
    #[test]
    fn the_opaque_pass_reverses_within_itself() {
        let order = [
            entry(0, 2, RenderPass::OPAQUE),
            entry(2, 1, RenderPass::OPAQUE),
            entry(4, 2, RenderPass::TRANSLUCENT),
            entry(6, 1, RenderPass::TRANSLUCENT),
        ];
        let batches = batches_of(&order);
        assert_eq!(batches.len(), 4);

        let drawn = painted(&batches, false);
        assert_eq!(
            layers_of(&batches, &drawn),
            vec![1, 2, 1, 2],
            "each pass bottom layer first, and the opaque pass still ahead of the translucent one"
        );
        assert_eq!(
            batches.get(drawn[0]).expect("a batch").key.pass,
            RenderPass::OPAQUE.bits(),
            "and the opaque pass still comes first"
        );
    }

    /// With depth to order it, the opaque pass keeps the order it arrived in.
    ///
    /// Front-to-back is what it wants once each layer is in its own band: the nearest layer draws
    /// first and writes, and the ones behind it are rejected rather than painted over. That is the
    /// early-out, and leaving the order alone is what lets it happen -- a reversed opaque pass
    /// would draw the farthest layer first and every nearer one would pass the test and shade.
    ///
    /// The translucent pass is reversed either way, because it does not write depth and so has
    /// nothing to be ordered by.
    #[test]
    fn depth_leaves_the_opaque_pass_front_to_back() {
        let order = [
            entry(0, 2, RenderPass::OPAQUE),
            entry(2, 1, RenderPass::OPAQUE),
            entry(4, 2, RenderPass::TRANSLUCENT),
            entry(6, 1, RenderPass::TRANSLUCENT),
        ];
        let batches = batches_of(&order);
        assert_eq!(
            layers_of(&batches, &painted(&batches, true)),
            vec![2, 1, 1, 2],
            "the opaque pass as sent, the translucent one reversed"
        );
        assert_eq!(
            layers_of(&batches, &painted(&batches, false)),
            vec![1, 2, 1, 2],
            "and without depth both are reversed"
        );
    }

    /// A layer's band is its slot's share of the range, nearest at slot zero.
    #[test]
    fn a_band_follows_the_depth_slot() {
        let range = DepthRange {
            layers: core::num::NonZeroU32::new(4).expect("four layers"),
            size: 1.0,
        };
        // Slot zero is the top of the style and nearest, so the last layer index leads.
        let (near, far) = range.band(3);
        assert!(
            (near - 0.0).abs() < 1e-6,
            "the top layer is nearest: {near}"
        );
        assert!((far - 0.25).abs() < 1e-6, "one band wide: {far}");

        let (near, _) = range.band(0);
        assert!(
            (near - 0.75).abs() < 1e-6,
            "and the style's first layer is deepest: {near}"
        );

        // Every band inside the range, and ordered by slot.
        let mut last = -1.0f32;
        for layer in [3, 2, 1, 0] {
            let (near, far) = range.band(layer);
            assert!(near > last, "bands ascend with the slot");
            assert!((0.0..=1.0).contains(&near) && (0.0..=1.0).contains(&far));
            assert!(far > near, "a band has width");
            last = near;
        }
    }

    /// No batches is no draws, rather than a panic.
    #[test]
    fn an_empty_frame_paints_nothing() {
        assert_eq!(painted(&Batches::new(), false), Vec::<usize>::new());
        assert_eq!(painted(&Batches::new(), true), Vec::<usize>::new());
    }
}
