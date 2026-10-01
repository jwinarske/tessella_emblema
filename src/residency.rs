//! What the device holds, what it still needs, and what it may let go of.
//!
//! The producer announces geometry and retires it; the device has to follow, and the two are not
//! the same clock. A retire says the producer is finished with it. It does not say the GPU is: a
//! frame recorded two frames ago may still be reading the buffer, and freeing it then is a
//! use-after-free that draws correctly on a desktop and faults on a tiler.
//!
//! So nothing is freed when it is retired. It is retired *into* a frame, and freed when the
//! backend says that frame has completed — which is the same shape as acknowledging a slab to the
//! producer, and for the same reason: only the thing holding the fence knows.
//!
//! # What this is not
//!
//! It allocates nothing on a device and knows nothing about Vulkan. It is the bookkeeping either
//! side of that, which is what makes it testable without one — and the bookkeeping is where the
//! use-after-free lives, not in the allocation.

use std::collections::{BTreeMap, BTreeSet};

use tessella_capture_abi::envelope::GeometryId;

/// A frame number, as the backend counts them.
///
/// Monotonic and the backend's own: this never generates one, only remembers which was current
/// when something was retired and compares it against the ones reported complete.
pub type FrameNo = u64;

/// What the device holds and what it owes.
#[derive(Debug, Clone, Default)]
pub struct Residency {
    /// Announced and not yet uploaded, or announced again since it was.
    wanted: BTreeSet<GeometryId>,
    /// Uploaded and current.
    resident: BTreeSet<GeometryId>,
    /// Retired by the producer, with the frame it was retired in.
    ///
    /// A map rather than a list: a geometry retired, re-announced and retired again should be
    /// freed against the later frame, not the earlier one, or the free lands while the second
    /// upload is still in flight.
    retiring: BTreeMap<GeometryId, FrameNo>,
}

impl Residency {
    /// Nothing held, nothing owed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The producer announced this geometry, so the device needs its bytes.
    ///
    /// An announcement for something already resident makes it wanted again: a re-announcement is
    /// how the producer says the bytes changed, and keeping the old ones is a drawable rendered
    /// from last zoom's vertices.
    ///
    /// It also cancels a pending retire. The producer announcing something it had retired means it
    /// is in use again, and freeing it on the strength of the earlier retire would free bytes the
    /// new announcement just filled.
    pub fn announced(&mut self, geometry: GeometryId) {
        self.retiring.remove(&geometry);
        self.wanted.insert(geometry);
    }

    /// The backend uploaded it.
    pub fn uploaded(&mut self, geometry: GeometryId) {
        if self.wanted.remove(&geometry) {
            self.resident.insert(geometry);
        }
    }

    /// The producer retired it, during `frame`.
    ///
    /// Nothing is freed here. A frame recorded earlier may still be reading the buffer, so the
    /// free waits for [`Residency::completed`].
    pub fn retired(&mut self, geometry: GeometryId, frame: FrameNo) {
        self.wanted.remove(&geometry);
        if self.resident.contains(&geometry) {
            self.retiring.insert(geometry, frame);
        }
    }

    /// Everything that may now be freed, given that every frame up to and including `frame` has
    /// completed on the device.
    ///
    /// Drains: what this returns is no longer tracked, so a caller that ignores the iterator leaks
    /// rather than double-frees. That is the safer way round.
    pub fn completed(&mut self, frame: FrameNo) -> Vec<GeometryId> {
        let freeable: Vec<GeometryId> = self
            .retiring
            .iter()
            .filter(|(_, retired)| **retired <= frame)
            .map(|(id, _)| *id)
            .collect();
        for id in &freeable {
            self.retiring.remove(id);
            self.resident.remove(id);
        }
        freeable
    }

    /// What the device still needs uploaded, in id order.
    pub fn wanted(&self) -> impl Iterator<Item = GeometryId> + '_ {
        self.wanted.iter().copied()
    }

    /// Whether the device holds current bytes for this geometry.
    ///
    /// False while it is wanted, including a re-announcement of something resident: the bytes are
    /// there but they are the old ones, and a drawable naming it would read those.
    #[must_use]
    pub fn is_current(&self, geometry: GeometryId) -> bool {
        self.resident.contains(&geometry) && !self.wanted.contains(&geometry)
    }

    /// How many geometries the device holds, current or not.
    #[must_use]
    pub fn resident(&self) -> usize {
        self.resident.len()
    }

    /// How many are waiting to be freed.
    ///
    /// For a caller watching whether its frame accounting is advancing. A number that only grows
    /// means frames are never being reported complete, which is a leak with a cause rather than a
    /// mystery.
    #[must_use]
    pub fn retiring(&self) -> usize {
        self.retiring.len()
    }
}
