//! What the device holds for each texture, and which parts of it are owed.
//!
//! Two things that are not the geometry story.
//!
//! A texture is created once and written many times. A `TextureUpdate` carrying a different size
//! or format is not an upload, it is a different image: the old one has to go and a new one be
//! made, and a backend that writes the new pixels into the old allocation either overruns it or
//! leaves a stale border. So that case is reported rather than handled as damage.
//!
//! And an update names *regions*. §6.4 caps the producer's list and spills to a union past it,
//! which stops the opposite-corners pathology — two small writes in opposite corners whose union
//! is the whole texture. The same discipline is needed here, because a consumer accumulates across
//! several updates before it flushes, and four tidy rects per update become sixteen untidy ones by
//! the time anything is uploaded.
//!
//! Freeing follows the geometry rule exactly: a retire is the producer's clock, a frame completing
//! is the device's, and nothing is freed on the first.

use std::collections::BTreeMap;

use tessella_capture_abi::envelope::{Extent, Rect16, TextureId};
use tessella_capture_abi::generated::mbgl_enums::{TextureChannelDataType, TexturePixelType};

use crate::residency::FrameNo;

/// What an update asks the backend to do before it uploads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Needs {
    /// Nothing: the texture exists at this size and format, and the regions are damage.
    Upload,
    /// The texture does not exist yet.
    Create,
    /// It exists at a different size or format, so it has to be made again.
    ///
    /// The old allocation is scheduled for freeing on the frame given, not dropped here: a frame
    /// recorded earlier may still be sampling it.
    Recreate,
}

/// The damage owed on one texture.
///
/// Rects are kept disjoint-ish rather than exactly: two that overlap or touch are unioned, which
/// can grow the area a little and saves a region. Past `cap` they all become one union, which is
/// the whole-texture write this is otherwise avoiding — deliberately, because beyond a handful of
/// regions the per-region overhead is the larger cost.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Damage {
    rects: Vec<Rect16>,
}

/// The widest a rect can be before the arithmetic below would not fit.
fn bounds(rect: Rect16) -> (u32, u32, u32, u32) {
    let x = u32::from(rect.x);
    let y = u32::from(rect.y);
    (x, y, x + u32::from(rect.w), y + u32::from(rect.h))
}

/// Whether two rects overlap or touch, so that one rect covers both no worse than two do.
fn meets(a: Rect16, b: Rect16) -> bool {
    let (ax0, ay0, ax1, ay1) = bounds(a);
    let (bx0, by0, bx1, by1) = bounds(b);
    ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1
}

/// The smallest rect covering both.
fn union(a: Rect16, b: Rect16) -> Rect16 {
    let (ax0, ay0, ax1, ay1) = bounds(a);
    let (bx0, by0, bx1, by1) = bounds(b);
    let (x0, y0) = (ax0.min(bx0), ay0.min(by0));
    let (x1, y1) = (ax1.max(bx1), ay1.max(by1));
    Rect16 {
        x: x0 as u16,
        y: y0 as u16,
        w: (x1 - x0) as u16,
        h: (y1 - y0) as u16,
    }
}

impl Damage {
    /// No damage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a region, unioning it with any it meets.
    ///
    /// Unioning can make the result meet a rect it did not before, so this settles rather than
    /// doing one pass: a rect bridging two others collapses all three.
    ///
    /// Past `cap` regions everything becomes one union. An empty rect is dropped — a zero-area
    /// update is a producer saying nothing changed, and keeping it would spend a region on it.
    pub fn add(&mut self, rect: Rect16, cap: usize) {
        if rect.w == 0 || rect.h == 0 {
            return;
        }
        let mut merged = rect;
        let mut settled = false;
        while !settled {
            settled = true;
            let mut keep = Vec::with_capacity(self.rects.len());
            for held in self.rects.drain(..) {
                if meets(merged, held) {
                    merged = union(merged, held);
                    settled = false;
                } else {
                    keep.push(held);
                }
            }
            self.rects = keep;
        }
        self.rects.push(merged);

        if self.rects.len() > cap
            && let Some(all) = self.rects.drain(..).reduce(union)
        {
            self.rects.push(all);
        }
    }

    /// The regions owed, in no particular order.
    #[must_use]
    pub fn rects(&self) -> &[Rect16] {
        &self.rects
    }

    /// Whether anything is owed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }

    /// Takes the damage, leaving none.
    pub fn take(&mut self) -> Vec<Rect16> {
        core::mem::take(&mut self.rects)
    }
}

/// One texture the device holds.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Held {
    size: Extent,
    format: u8,
    damage: Damage,
}

/// Every texture the device holds, and what each is owed.
#[derive(Debug, Clone, Default)]
pub struct Textures {
    held: BTreeMap<TextureId, Held>,
    retiring: BTreeMap<TextureId, FrameNo>,
}

impl Textures {
    /// Nothing held.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes an update, and says what the backend has to do about it.
    ///
    /// An empty rect list is a whole-texture write, which is how the producer says it has no
    /// damage worth describing — so it becomes damage covering the whole image rather than nothing.
    ///
    /// `frame` is the frame being recorded, used only if the texture has to be remade.
    pub fn updated(
        &mut self,
        texture: TextureId,
        size: Extent,
        format: u8,
        rects: &[Rect16],
        cap: usize,
        frame: FrameNo,
    ) -> Needs {
        let needs = match self.held.get(&texture) {
            None => Needs::Create,
            Some(held) if held.size != size || held.format != format => Needs::Recreate,
            Some(_) => Needs::Upload,
        };

        if needs != Needs::Upload {
            if needs == Needs::Recreate {
                // The old allocation outlives this call: a frame recorded earlier may still be
                // sampling it, which is the same reason geometry is not freed on retire.
                self.retiring.insert(texture, frame);
            }
            self.held.insert(
                texture,
                Held {
                    size,
                    format,
                    damage: Damage::new(),
                },
            );
        }

        // Present either way: `Upload` means it was already there, and the other two just
        // inserted it. Written as a lookup that can fail rather than one that cannot, so the
        // function has no panic to document.
        let Some(held) = self.held.get_mut(&texture) else {
            return needs;
        };
        if rects.is_empty() {
            // An extent is `u32` and a rect is `u16`, so a texture larger than a rect can
            // address is clamped rather than wrapped. The producer could not have described
            // damage on such a texture either -- its own rects are the same type -- so a backend
            // taking this as "write what you can address" is reading it the way the ABI means it.
            held.damage.add(
                Rect16 {
                    x: 0,
                    y: 0,
                    w: u16::try_from(size.width).unwrap_or(u16::MAX),
                    h: u16::try_from(size.height).unwrap_or(u16::MAX),
                },
                cap,
            );
        } else {
            for rect in rects {
                held.damage.add(*rect, cap);
            }
        }
        needs
    }

    /// The regions owed on one texture.
    #[must_use]
    pub fn damage(&self, texture: TextureId) -> &[Rect16] {
        self.held
            .get(&texture)
            .map_or(&[], |held| held.damage.rects())
    }

    /// Takes the damage owed, leaving the texture held and clean.
    pub fn uploaded(&mut self, texture: TextureId) -> Vec<Rect16> {
        self.held
            .get_mut(&texture)
            .map(|held| held.damage.take())
            .unwrap_or_default()
    }

    /// The producer retired it, during `frame`.
    pub fn retired(&mut self, texture: TextureId, frame: FrameNo) {
        if self.held.remove(&texture).is_some() {
            self.retiring.insert(texture, frame);
        }
    }

    /// Every allocation that may now be freed, given every frame through `frame` has completed.
    ///
    /// Drains, as the geometry side does: what this returns is no longer tracked, so ignoring it
    /// leaks rather than double-frees.
    pub fn completed(&mut self, frame: FrameNo) -> Vec<TextureId> {
        let freeable: Vec<TextureId> = self
            .retiring
            .iter()
            .filter(|(_, retired)| **retired <= frame)
            .map(|(id, _)| *id)
            .collect();
        for id in &freeable {
            self.retiring.remove(id);
        }
        freeable
    }

    /// How many textures the device holds.
    #[must_use]
    pub fn held(&self) -> usize {
        self.held.len()
    }

    /// How many allocations are waiting to be freed.
    #[must_use]
    pub fn retiring(&self) -> usize {
        self.retiring.len()
    }
}

/// How large a texel of this pixel and channel type is.
///
/// Both of mbgl's factors: `Texture2DDesc::getStorageSize` is
/// `channelCount() * channelStorageSize()`, and a consumer multiplying by the first alone sizes a
/// color relief's elevation stops at a quarter.
#[must_use]
pub const fn texel(pixel: TexturePixelType, channel: TextureChannelDataType) -> u64 {
    pixel.channels() as u64 * channel.storage_size() as u64
}

/// What `bufferOffset` has to be a multiple of for a texel of this size.
///
/// Two rules, and the stricter is taken because this does not choose the queue.
///
/// `VkBufferImageCopy` requires `bufferOffset` to be a multiple of the format's texel block size,
/// always. It *additionally* requires a multiple of four when the queue family supports neither
/// `GRAPHICS` nor `COMPUTE` -- a transfer-only queue, which is exactly the queue a sensible
/// uploader would pick on a device that has one. A layout that satisfied only the texel rule would
/// be valid on the graphics queue and rejected on the transfer queue, so it would work until
/// somebody moved the upload where it belongs.
#[must_use]
pub const fn copy_alignment(texel: u64) -> u64 {
    if texel > 4 { texel } else { 4 }
}

/// One region's place in the staging buffer, and the image region it is copied into.
///
/// Becomes one `VkBufferImageCopy`: `bufferOffset` is [`Self::at`], `bufferRowLength` is
/// [`Self::row_length`], and the image offset and extent come from [`Self::rect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    /// Where in the image this goes.
    pub rect: Rect16,
    /// Byte offset of the region's first row in the staging buffer.
    pub at: u64,
    /// Texels from the start of one staged row to the next.
    ///
    /// Always the region's own width, because a region is staged packed whatever shape the payload
    /// arrived in. That is what lets one buffer hold regions of different widths.
    pub row_length: u32,
}

/// Where a texture's regions sit inside one staging buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staging {
    /// One per region, in the order the rects named them.
    pub placed: Vec<Placed>,
    /// Bytes the staging buffer needs.
    pub total: u64,
}

/// Packs a texture's regions into one staging buffer, each at an aligned offset.
///
/// # Why one buffer rather than one per region
///
/// mbgl's Vulkan backend creates a buffer per sub-region and copies from offset zero
/// (`Texture2D::uploadSubRegion`), which makes the alignment question disappear and costs a buffer
/// and an allocation per damage rect per frame. §6.4 caps the rect list at four per update, but a
/// frame's updates accumulate across several textures before anything is flushed, and on a tiler
/// that is the allocation traffic the dirty-rect list exists to avoid.
///
/// Packing them into one buffer is what brings `bufferOffset` into it, and [`copy_alignment`] is
/// the rule.
///
/// An empty rect list is one region covering the whole texture, which is how the producer says it
/// has no damage worth describing -- the same reading [`Textures::updated`] takes.
///
/// # Why this cannot overflow
///
/// A region is at most `65535 * 65535 * 16` bytes, which is 6.87e10 -- a rect is two `u16`s and
/// the largest texel is RGBA floats. A whole-texture region is the same bound, because the extent
/// is clamped into a `u16` above. It would take 2.68e8 such regions to pass a `u64`, and §6.4 caps
/// an update's list at `TEXTURE_RECT_CAP`, which is four.
///
/// So this returns a layout rather than a `Result`: a fallible signature here would be a branch
/// nothing can take and no test can reach, which is worse than arithmetic with its bound written
/// down. The additions saturate anyway, because a saturated total is unallocatable and so fails at
/// the allocation rather than quietly fitting.
#[must_use]
pub fn staging(
    rects: &[Rect16],
    size: Extent,
    pixel: TexturePixelType,
    channel: TextureChannelDataType,
) -> Staging {
    staging_for(rects, size, texel(pixel, channel))
}

/// [`staging`] for a caller that already has the texel size.
///
/// The image store holds a texture's texel rather than the pixel and channel types it came from,
/// because the texel is all the staging arithmetic uses and keeping both would be two things that
/// can disagree.
#[must_use]
pub fn staging_for(rects: &[Rect16], size: Extent, texel: u64) -> Staging {
    let alignment = copy_alignment(texel);
    let whole = Rect16 {
        x: 0,
        y: 0,
        // An extent is `u32` and a rect is `u16`, so a texture larger than a rect can address is
        // clamped rather than wrapped, as `updated` does.
        w: u16::try_from(size.width).unwrap_or(u16::MAX),
        h: u16::try_from(size.height).unwrap_or(u16::MAX),
    };
    let regions = if rects.is_empty() {
        core::slice::from_ref(&whole)
    } else {
        rects
    };

    let mut placed = Vec::with_capacity(regions.len());
    let mut at = 0u64;
    for rect in regions {
        // Aligned before the region rather than after, so the first one starts at zero and a
        // zero-area region still leaves the next one aligned.
        at = at.next_multiple_of(alignment);
        placed.push(Placed {
            rect: *rect,
            at,
            row_length: u32::from(rect.w),
        });
        at = at.saturating_add(
            u64::from(rect.w)
                .saturating_mul(texel)
                .saturating_mul(u64::from(rect.h)),
        );
    }
    // Rounded up, so a second texture packed after this one starts aligned too.
    Staging {
        placed,
        total: at.next_multiple_of(alignment),
    }
}
