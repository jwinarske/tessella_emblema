// SPDX-License-Identifier: BSD-2-Clause
//! One geometry's buffers on the device, and where each one sits in a single allocation.
//!
//! The first part of #60's frame half. [`crate::buffers`] decides *which* buffers a geometry needs
//! and [`crate::residency`] decides *when* they may go; this is the part in between that puts them on
//! a device and takes them off again.
//!
//! # One allocation a geometry, not one a buffer
//!
//! A geometry's buffers are created separately -- a vertex binding and an index binding are
//! different usages and `vkCmdBindIndexBuffer` will not take a vertex binding -- but they are backed
//! by one `VkDeviceMemory` and bound at offsets within it. That is the shape `benches/draw_cost.rs`
//! already uses, and it is not only tidiness: `maxMemoryAllocationCount` is 4096 on a good many
//! drivers, and a cover of a few hundred tiles with several bindings each would spend it on a map.
//!
//! The offsets are the part worth testing, so [`layout`] computes them with no device in sight. An
//! overlap there is two bindings reading each other's bytes, which draws something and is not what
//! the producer sent.
//!
//! # What this does not do
//!
//! Suballocate across geometries, reuse a freed block, or stage through a device-local copy. Every
//! allocation here is `HOST_VISIBLE | HOST_COHERENT` and written by mapping it, which is what the
//! capture stream wants on the integrated parts this is for -- the bytes arrive on the CPU and are
//! read once by the GPU. A discrete part would rather have them device-local behind a staging copy,
//! and that is a later change with a measurement behind it rather than a guess now.

use std::collections::BTreeMap;

use ash::vk;
use tessella_capture_abi::envelope::{GeometryId, SlabRef};
use tessella_vk::{Buffer, Gpu, Requirements};

use crate::buffers::Needs;

/// Why a geometry could not be put on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A slab reference named bytes the host could not resolve.
    ///
    /// The producer's reference and the consumer's region disagree, which is a protocol fault rather
    /// than a device one -- so it is named separately from everything below.
    Unresolved(SlabRef),
    /// A reference resolved to fewer bytes than it claims.
    ///
    /// Distinct from [`Self::Unresolved`] because it is the shape a truncated region has, and
    /// copying what arrived would leave the tail of a vertex buffer holding whatever the allocation
    /// came with.
    Short {
        /// Bytes the reference claims.
        wanted: usize,
        /// Bytes the host actually produced.
        got: usize,
    },
    /// The device refused, or has no memory type that serves.
    Device(tessella_vk::Error),
}

impl From<tessella_vk::Error> for Error {
    fn from(why: tessella_vk::Error) -> Self {
        Self::Device(why)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unresolved(reference) => write!(
                f,
                "slab {} offset {} length {} does not resolve",
                reference.slab, reference.offset, reference.length
            ),
            Self::Short { wanted, got } => {
                write!(f, "a reference claiming {wanted} bytes resolved to {got}")
            }
            Self::Device(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

/// Where one buffer sits in a geometry's allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct At {
    /// Byte offset into the allocation.
    pub offset: u64,
    /// Bytes the buffer holds, as the reference claims them.
    pub length: u64,
}

/// Where every one of a geometry's buffers sits, and how large the allocation must be.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// One per [`Needs::vertices`], in the same order.
    pub vertices: Vec<At>,
    /// The index buffer, when the geometry has one.
    pub indexes: Option<At>,
    /// Bytes the allocation must cover, including the padding between buffers.
    pub total: u64,
}

/// Lays a geometry's buffers out end to end, each aligned up to `alignment`.
///
/// `alignment` is the largest `VkMemoryRequirements::alignment` of the buffers being placed. Taking
/// the largest rather than each buffer's own is what keeps this device-free: the requirement cannot
/// be known per buffer without creating it, and over-aligning is correct where under-aligning is
/// not.
///
/// A zero-length vertex reference still takes a slot, because [`Needs::reads`] indexes into
/// `vertices` positionally and dropping one would shift every binding after it onto the wrong bytes.
#[must_use]
pub fn layout(needs: &Needs, alignment: u64) -> Layout {
    let step = alignment.max(1);
    let mut at = 0u64;
    let mut place = |length: u64| {
        let placed = At { offset: at, length };
        // Aligned up even for a zero-length buffer, so the next one's offset does not depend on
        // whether the one before it was empty.
        at += length.div_ceil(step) * step;
        placed
    };
    let vertices = needs
        .vertices
        .iter()
        .map(|source| place(u64::from(source.length)))
        .collect();
    let indexes = needs.indexes.map(|source| place(u64::from(source.length)));
    Layout {
        vertices,
        indexes,
        total: at,
    }
}

/// The device buffers the resident geometries are drawn from.
///
/// Keyed by [`GeometryId`], which is what the stream names and what [`crate::residency`] hands back
/// as freeable. Nothing here decides *when* a geometry may go -- see that module for why a retire is
/// not a free.
///
/// `'d` is the device's lifetime, borrowed through [`Gpu`]: a store cannot outlive the device its
/// buffers are on, which is checked rather than promised.
#[derive(Debug, Default)]
pub struct Store<'d> {
    held: BTreeMap<GeometryId, Held<'d>>,
}

/// One geometry's buffers and the allocation behind them.
///
/// Field order is drop order, and it matters: the buffers must go before the memory they are bound
/// to. Stating it here rather than writing a `Drop` is what keeps the rule from being re-derived.
#[derive(Debug)]
struct Held<'d> {
    /// One per [`Needs::vertices`], in the same order, so [`crate::buffers::Reads::buffer`] indexes
    /// this directly.
    vertices: Vec<Buffer<'d>>,
    indexes: Option<Buffer<'d>>,
    memory: tessella_vk::Memory<'d>,
    /// Where each vertex buffer starts, for reading one back without recomputing the layout from an
    /// alignment this no longer has.
    offsets: Vec<u64>,
}

impl<'d> Store<'d> {
    /// A store holding nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            held: BTreeMap::new(),
        }
    }

    /// How many geometries are on the device.
    #[must_use]
    pub fn resident(&self) -> usize {
        self.held.len()
    }

    /// Whether this geometry's bytes are on the device.
    #[must_use]
    pub fn holds(&self, geometry: GeometryId) -> bool {
        self.held.contains_key(&geometry)
    }

    /// The buffer a binding reads, by its index into [`Needs::vertices`].
    #[must_use]
    pub fn vertex_buffer(&self, geometry: GeometryId, at: usize) -> Option<vk::Buffer> {
        Some(self.held.get(&geometry)?.vertices.get(at)?.raw())
    }

    /// This geometry's index buffer, if it has one.
    #[must_use]
    pub fn index_buffer(&self, geometry: GeometryId) -> Option<vk::Buffer> {
        Some(self.held.get(&geometry)?.indexes.as_ref()?.raw())
    }

    /// Device bytes this geometry's allocation covers.
    ///
    /// The allocation rather than the sum of the references: it includes the padding between buffers
    /// and whatever the driver wanted beyond what was asked, which is what the device actually spent.
    #[must_use]
    pub fn bytes(&self, geometry: GeometryId) -> Option<u64> {
        Some(self.held.get(&geometry)?.memory.size())
    }

    /// Reads back what was uploaded for one of a geometry's vertex buffers.
    ///
    /// For checking a hand-off and for a test that has to prove the bytes arrived rather than that
    /// the calls returned `Ok`. Not a drawing path: this maps host-visible memory and copies, which is
    /// the slow way to look at a buffer and the only way to look at one without drawing it.
    ///
    /// Reads from the start of that buffer's slice of the allocation, for `into.len()` bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when the allocation will not map, or when the read runs past it -- which it
    /// does if `into` is longer than the buffer.
    pub fn read_vertex_bytes(
        &self,
        geometry: GeometryId,
        at: usize,
        into: &mut [u8],
    ) -> Result<(), Error> {
        let Some(held) = self.held.get(&geometry) else {
            return Ok(());
        };
        let mapping = held.memory.map()?;
        mapping.read(held.offsets[at], into)?;
        Ok(())
    }

    /// Device bytes every resident geometry comes to.
    ///
    /// What a caller watches to know whether a cover is affordable, and the number a suballocator
    /// would be judged against if one is ever written.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.held.values().map(|held| held.memory.size()).sum()
    }

    /// Puts one geometry's bytes on the device.
    ///
    /// `resolve` answers a slab reference with the bytes the host holds for it, which is
    /// `tessella_consume::slab::resolve`'s job rather than this one's -- a store that reached into the
    /// region itself would have to know how the producer laid it out.
    ///
    /// Replacing a geometry already held drops the old buffers at the end of this call. That is only
    /// correct because the producer announces a geometry under a *new* id and does not reuse one, so a
    /// replacement means a caller asked twice for the same thing. If that stops being true this is a
    /// use-after-free and belongs on the retirement path instead.
    ///
    /// # Errors
    ///
    /// [`Error`] when a reference does not resolve, when it resolves short, or when the device
    /// refuses. Nothing is left on the device in any of those cases: the wrapper's types free what
    /// they own as the `?` unwinds.
    pub fn upload<'bytes>(
        &mut self,
        gpu: Gpu<'d>,
        geometry: GeometryId,
        needs: &Needs,
        resolve: &dyn Fn(SlabRef) -> Option<&'bytes [u8]>,
    ) -> Result<(), Error> {
        // Resolved before anything is created, so a protocol fault costs no device objects.
        let mut sources: Vec<(SlabRef, &'bytes [u8])> =
            Vec::with_capacity(needs.vertices.len() + 1);
        for reference in needs.vertices.iter().chain(needs.indexes.iter()) {
            let bytes = resolve(*reference).ok_or(Error::Unresolved(*reference))?;
            if bytes.len() < reference.length as usize {
                return Err(Error::Short {
                    wanted: reference.length as usize,
                    got: bytes.len(),
                });
            }
            sources.push((*reference, bytes));
        }

        let vertices: Vec<Buffer<'d>> = needs
            .vertices
            .iter()
            .map(|reference| {
                gpu.buffer(
                    u64::from(reference.length),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )
            })
            .collect::<Result<_, _>>()?;
        let indexes = needs
            .indexes
            .map(|reference| {
                gpu.buffer(
                    u64::from(reference.length),
                    vk::BufferUsageFlags::INDEX_BUFFER,
                )
            })
            .transpose()?;

        // The requirements decide the alignment and the layout decides the offsets from it. One
        // alignment for every buffer: see `layout`.
        let requirements: Vec<Requirements> = vertices
            .iter()
            .chain(indexes.iter())
            .map(Buffer::requirements)
            .collect();
        let alignment = requirements
            .iter()
            .map(|need| need.alignment)
            .max()
            .unwrap_or(1);
        let placed = layout(needs, alignment);
        // The driver may want more for a buffer than its reference claims, so the allocation is the
        // larger of what the layout needs and what the requirements add up to.
        let required: u64 = requirements
            .iter()
            .map(|need| need.size.div_ceil(alignment) * alignment)
            .sum();
        let memory = gpu.allocate(
            placed.total.max(required),
            &requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        for (buffer, at) in vertices
            .iter()
            .chain(indexes.iter())
            .zip(placed.vertices.iter().chain(placed.indexes.iter()))
        {
            memory.bind(buffer, at.offset)?;
        }

        {
            let mut mapping = memory.map()?;
            for (at, (reference, bytes)) in placed
                .vertices
                .iter()
                .chain(placed.indexes.iter())
                .zip(&sources)
            {
                mapping.write(at.offset, &bytes[..reference.length as usize])?;
            }
        }

        self.held.insert(
            geometry,
            Held {
                vertices,
                indexes,
                memory,
                offsets: placed.vertices.iter().map(|at| at.offset).collect(),
            },
        );
        Ok(())
    }

    /// Frees the geometries named, which is what [`crate::residency::Residency::completed`] answers.
    ///
    /// An id this store does not hold is ignored rather than an error: residency tracks what the
    /// producer announced and this tracks what reached the device, and a geometry retired before its
    /// upload succeeded is in the first and not the second.
    ///
    /// # Correctness, which this cannot check
    ///
    /// Every frame that read these buffers must have completed. That is residency's judgment, and
    /// calling this too early is a use-after-free that draws correctly on a desktop and faults on a
    /// tiler. It is not an `unsafe fn` because nothing here dereferences anything -- the hazard is a
    /// Vulkan lifetime rule, which `unsafe` does not describe and cannot enforce.
    pub fn free(&mut self, geometries: &[GeometryId]) {
        for geometry in geometries {
            self.held.remove(geometry);
        }
    }

    /// Frees everything, for a store going away with its device still alive.
    ///
    /// As [`Self::free`], for every geometry at once: the device must be idle.
    pub fn clear(&mut self) {
        self.held.clear();
    }
}
