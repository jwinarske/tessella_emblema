//! What the device supports, asked rather than assumed.
//!
//! This crate owns its own depth-stencil attachment and declares its own vertex formats, so it
//! owns the obligation to check both. That obligation is not theoretical. Filament assumed
//! `D32_SFLOAT_S8_UINT` and segfaulted in `vkCreateImageView` on a Raspberry Pi 5, because V3D
//! offers `D24_UNORM_S8_UINT` and not the other; RADV is the reverse. `tessella_fluorite` carries
//! an upstream patch for exactly that, and the mirror that owns the attachment inherits the
//! lesson directly.
//!
//! So: no format is used because it is usual. Every one is selected from what the device reports,
//! and a device that reports none is an error at creation — loudly, and never a silent fallback
//! that draws a wrong picture.
//!
//! The functions here take the device's answers as a closure rather than a `vk::Instance`, which
//! is what makes them testable without a GPU. The caller supplies
//! `vkGetPhysicalDeviceFormatProperties`; these decide what to do with it.

use ash::vk;
use tessella_capture_abi::generated::mbgl_enums::{
    AttributeDataType, TextureChannelDataType, TexturePixelType,
};

/// Whether a view's pass needs depth, or only stencil.
///
/// A view drawing fill-extrusions needs depth: the opaque pass resolves front-to-back and the
/// extrusion's own prepass decides which surface is nearest. A view with none still needs stencil,
/// for the per-tile clip masks, and can take a stencil-only format where the device has one —
/// which is a smaller attachment.
///
/// # What the saving is, and is not, measured to be
///
/// This said "and, on a tiler, less to write back". That is still the reason the distinction
/// exists, and it is unverified on any tiler this runs on — `examples/device-report.rs` asked four
/// GPUs what the two attachments resolve to:
///
/// | device | `StencilOnly` | `DepthStencil` |
/// |---|---|---|
/// | RADV, Ryzen 9 iGPU | `S8_UINT` | `D32_SFLOAT_S8_UINT` |
/// | llvmpipe | `S8_UINT` | `D24_UNORM_S8_UINT` |
/// | Adreno 640, SA8155P | `S8_UINT` | `D24_UNORM_S8_UINT` |
/// | **V3D 7.1.7.0, Pi 5** | `D24_UNORM_S8_UINT` | `D24_UNORM_S8_UINT` |
///
/// Three of the four give a smaller attachment, so the mechanism is real. **The one tiler does
/// not** — V3D offers no `S8_UINT` at all, so a flat view there already carries depth and this
/// saves nothing. The claim is about tilers and the only tiler here cannot exercise it.
///
/// Where the difference does exist and the pass can run, it measures the wrong way:
/// `benches/overdraw_cost.rs`'s zero-layer rows put the stencil-only attachment about **12 µs a
/// frame dearer** than the packed one on RADV, over 1024 by 1024 — 22.8 against 10.6. One device,
/// one scene, and the opposite sign to the reason for the knob.
///
/// Kept rather than retired, because the caller chooses it and three devices do give a smaller
/// attachment. What is removed is the claim that this is known to be cheaper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attachment {
    /// Depth and stencil, which is every view that draws an extrusion.
    DepthStencil,
    /// Stencil alone, for a view whose layers are all flat.
    StencilOnly,
}

/// Why a device cannot host the map pass.
///
/// Returned rather than logged. A device that cannot give a depth-stencil attachment cannot draw
/// the map at all, and the useful moment to say so is device creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// No candidate format reported `DEPTH_STENCIL_ATTACHMENT` in `optimalTilingFeatures`.
    NoDepthStencilFormat,
    /// A vertex format the shaders declare is not usable in a vertex buffer here.
    ///
    /// Carries the first one found, because the fix is per format and a list of every failure is
    /// no more actionable than the first.
    VertexFormat(vk::Format),
    /// The device cannot do dynamic rendering.
    ///
    /// Required rather than optional, and the reason is #60's own contract: the host passes a ring
    /// of at least three images and the pass "must not cache per-image state that breaks when the
    /// image changes every frame". A `VkFramebuffer` is exactly that state -- one per image, and one
    /// per size if the ring is ever resized -- so a render pass would make the requirement a
    /// cache-invalidation problem instead of a non-problem.
    ///
    /// It costs nothing on the parts this runs on: core in Vulkan 1.3, and `dynamicRendering` is
    /// reported `true` by RADV, by V3D 7.1.7.0 (the gating target) and by the `VeriSilicon`
    /// `GC7000UL`.
    NoDynamicRendering,
    /// A texture format the producer can send cannot be sampled or written here.
    ///
    /// The first one found, as for [`Self::VertexFormat`]. A device without one of these cannot
    /// draw a layer that samples it -- a glyph atlas missing is a map with no labels -- so it is
    /// an error at creation rather than a layer quietly skipped.
    TextureFormat(vk::Format),
}

/// Checks the device can render without a render pass.
///
/// Takes the feature bit rather than a `vk::PhysicalDevice`, which is what makes it testable without
/// a GPU -- the caller reads `VkPhysicalDeviceVulkan13Features::dynamicRendering` and this decides
/// what to do about it.
///
/// # Errors
///
/// [`Unsupported::NoDynamicRendering`] when the device does not have it.
pub fn check_dynamic_rendering(supported: bool) -> Result<(), Unsupported> {
    if supported {
        Ok(())
    } else {
        Err(Unsupported::NoDynamicRendering)
    }
}

/// The width mbgl draws a fill's outline at, in pixels.
///
/// `render_fill_layer.cpp:177` is `constexpr auto lineWidth = 2.0f`, and every `gfx::Lines` in that
/// file takes it: the plain outline at :401 and the patterned one at :439 and :451. The wire
/// carries no width -- `GeometryAdd` has a topology byte and nothing beside it -- so the constant
/// is the consumer's, and it is true of every `gfx::Lines` mbgl builds for a fill.
///
/// A field in the ABI would be the honest answer if some other family ever drew lines at another
/// width. The location indicator is the one to watch: its accuracy border is the other
/// [`Topology`](tessella_capture_abi::envelope::Topology) the wire carries, and mbgl draws it with
/// `gfx::LineStrip` through the *line* shader, which takes its width from a paint property rather
/// than from a constant. Nothing on this wire sends one yet.
pub const OUTLINE_WIDTH: f32 = 2.0;

/// What a line-topology draw is recorded at, given what the device can do.
///
/// Takes the feature bit and the limit rather than a `vk::PhysicalDevice`, for
/// [`check_dynamic_rendering`]'s reason: the caller creates the device, so only the caller can
/// enable `wideLines`, and this decides what to do about the answer.
///
/// One pixel without the feature, which is not a workaround but the only answer: a device that does
/// not have it rasterizes every line one pixel wide whatever a pipeline or a `vkCmdSetLineWidth`
/// asks for, and asking for two is invalid usage besides. Both gate devices do have it --
/// `lineWidthRange` is 1 to 32 on V3D 7.1.7.0 and 1 to 255 on RADV -- so the fallback is for a
/// device this pass has not met.
///
/// Clamped into the range because that is the valid-usage condition: a width outside
/// `lineWidthRange` is undefined behavior, and a device reporting a maximum below two is a device
/// whose widest line is narrower than mbgl's outline. `lineWidthGranularity` is not read -- a
/// non-strict line is rasterized at the nearest multiple the device supports, which is the
/// device's rounding to do and not this crate's.
///
/// # What the fallback costs
///
/// Not half the coverage. The feather in `FILL_OUTLINE_BODY` fades over one pixel of distance from
/// the line's own center, so the second pixel of a two-wide line is drawn at the tail of the
/// `smoothstep` rather than at full strength. #132 is where that gap is measured.
#[must_use]
pub fn line_width(wide_lines: bool, range: [f32; 2]) -> f32 {
    let [low, high] = range;
    // A range that is not a range is treated as no feature. `f32::clamp` panics on an inverted or
    // `NaN` bound, and a library has no business panicking over what a driver reported.
    if !wide_lines || low.is_nan() || high.is_nan() || low > high {
        return 1.0;
    }
    OUTLINE_WIDTH.clamp(low, high)
}

/// Which `VkFormat` a texture of this pixel and channel type is, as mbgl's Vulkan backend decides.
///
/// Transcribed from `Texture2D::vulkanFormat` in `src/mbgl/vulkan/texture2d.cpp`, which is the
/// backend whose tables this crate's shaders come from. `setFormat` takes both halves and so does
/// this: a color relief's elevation stops are `RGBA` and `Float` together, and the pixel type alone
/// would make them bytes.
///
/// `None` where mbgl returns `eUndefined`, which is two cases and both deliberate:
///
/// - `Depth`, which it refuses outright -- a depth texture is an attachment this crate selects
///   through [`depth_stencil_format`], not something the producer uploads.
/// - `Luminance`, which falls past both of its `if`s to the final `return`. The ABI can describe a
///   luminance texture and mbgl's Vulkan backend cannot make one, so this says so rather than
///   inventing `R8_UNORM` for it. Nothing on this wire sends one.
///
/// `Stencil` is `S8_UINT` whatever the channel type says, which is mbgl's own early return. It
/// disagrees with the texel size the channel type implies, and the disagreement is unreachable:
/// the producer sends `Alpha` and `RGBA` only.
#[must_use]
pub const fn texture_format(
    pixel: TexturePixelType,
    channel: TextureChannelDataType,
) -> Option<vk::Format> {
    match pixel {
        // Packed, and before the channel type is looked at -- mbgl returns early here.
        TexturePixelType::Stencil => Some(vk::Format::S8_UINT),
        TexturePixelType::Alpha => Some(match channel {
            TextureChannelDataType::UnsignedByte => vk::Format::R8_UNORM,
            TextureChannelDataType::HalfFloat => vk::Format::R16_SFLOAT,
            TextureChannelDataType::Float => vk::Format::R32_SFLOAT,
        }),
        TexturePixelType::RGBA => Some(match channel {
            TextureChannelDataType::UnsignedByte => vk::Format::R8G8B8A8_UNORM,
            TextureChannelDataType::HalfFloat => vk::Format::R16G16B16A16_SFLOAT,
            TextureChannelDataType::Float => vk::Format::R32G32B32A32_SFLOAT,
        }),
        TexturePixelType::Depth | TexturePixelType::Luminance => None,
    }
}

/// What a sampled texture needs of its format: to be read by a shader, and to be written into.
///
/// `SAMPLED_IMAGE` because every one of these is read by a fragment shader, and `TRANSFER_DST`
/// because every one is filled by `vkCmdCopyBufferToImage` rather than rendered into.
const TEXTURE_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::SAMPLED_IMAGE.as_raw() | vk::FormatFeatureFlags::TRANSFER_DST.as_raw(),
);

/// Checks that every texture format the producer can send is usable here.
///
/// Asked of `optimalTilingFeatures`, because a sampled texture is optimally tiled -- a linear one
/// would be legal and slow, and on a tiler the difference is the whole point of the copy.
///
/// # Errors
///
/// [`Unsupported::TextureFormat`] naming the first format the device will not take.
pub fn check_texture_formats(
    formats: &[vk::Format],
    optimal_features: impl Fn(vk::Format) -> vk::FormatFeatureFlags,
) -> Result<(), Unsupported> {
    formats
        .iter()
        .copied()
        .find(|format| !optimal_features(*format).contains(TEXTURE_FEATURES))
        .map_or(Ok(()), |format| Err(Unsupported::TextureFormat(format)))
}

/// The depth-stencil formats this crate will accept, in the order it prefers them.
///
/// `D24_UNORM_S8_UINT` first because that is what the gating target has: V3D reports no attachment
/// support for `D32_SFLOAT_S8_UINT`. The 32-bit form second for the devices that are the other way
/// round. Both carry eight stencil bits, which the tile-clip masks need.
const DEPTH_STENCIL: [vk::Format; 2] = [
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D32_SFLOAT_S8_UINT,
];

/// And the stencil-only formats, for a view with no depth to keep.
///
/// `S8_UINT` is optional in Vulkan and plenty of devices lack it, so this falls back to the
/// depth-stencil list rather than failing — an attachment with unused depth costs memory and
/// bandwidth, not correctness.
///
/// "Plenty of devices lack it" is not hypothetical and the one that matters most is among them:
/// V3D 7.1.7.0 has no `S8_UINT`, so every view on the gating target takes the fallback. See
/// [`Attachment`] for the four devices asked and what each answered.
const STENCIL_ONLY: [vk::Format; 1] = [vk::Format::S8_UINT];

/// Picks the depth-stencil format for a view's pass.
///
/// `optimal_features` is the device's `optimalTilingFeatures` for a format, which is what an
/// attachment is created with. `linearTilingFeatures` is deliberately not consulted: an attachment
/// in linear tiling would be a performance bug on every target here.
///
/// # Errors
///
/// [`Unsupported::NoDepthStencilFormat`] when nothing in the preference list can be an attachment.
pub fn depth_stencil_format(
    attachment: Attachment,
    optimal_features: impl Fn(vk::Format) -> vk::FormatFeatureFlags,
) -> Result<vk::Format, Unsupported> {
    let wanted = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
    let preferred: &[vk::Format] = match attachment {
        Attachment::StencilOnly => &STENCIL_ONLY,
        Attachment::DepthStencil => &[],
    };
    preferred
        .iter()
        .chain(DEPTH_STENCIL.iter())
        .copied()
        .find(|format| optimal_features(*format).contains(wanted))
        .ok_or(Unsupported::NoDepthStencilFormat)
}

/// Checks that every vertex format the shaders declare can be read from a vertex buffer.
///
/// tessella emits `Short2`, `Short4`, `UShort2`, `UShort4`, `UByte4` and `Float`..`Float4`. None
/// is a three-component 16-bit format, which is the shape that most often lacks vertex support —
/// but "most often" is not "never", so this asks.
///
/// The 16-bit attributes are declared `SINT`/`UINT` and widened in the vertex stage rather than
/// through the optional `SSCALED`/`USCALED` formats, which are far less widely supported and would
/// trade a documented conversion for an undocumented absence.
///
/// # Errors
///
/// [`Unsupported::VertexFormat`] naming the first format the device will not take.
pub fn check_vertex_formats(
    formats: &[vk::Format],
    buffer_features: impl Fn(vk::Format) -> vk::FormatFeatureFlags,
) -> Result<(), Unsupported> {
    let wanted = vk::FormatFeatureFlags::VERTEX_BUFFER;
    formats
        .iter()
        .copied()
        .find(|format| !buffer_features(*format).contains(wanted))
        .map_or(Ok(()), |format| Err(Unsupported::VertexFormat(format)))
}

/// How much this crate wants to run on a device of each class, lowest first.
///
/// A software implementation renders the map correctly and far too slowly to be a target, and it
/// is also what a misconfigured system silently falls back to. Picking it only when nothing else
/// exists means a wrong answer on a board shows up as a wrong *picture* rather than as a frame
/// time nobody can explain.
///
/// `Other` sits above `Cpu` and below the rest because it is the class an implementation reports
/// when it will not say: unknown hardware is still more likely to be hardware than the one class
/// that is definitionally not.
const fn rank(class: vk::PhysicalDeviceType) -> u8 {
    match class {
        // External.
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        // Internal.
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        // A paravirtualized device, which is hardware on the other side of the hypervisor.
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        // Software.
        vk::PhysicalDeviceType::CPU => 4,
        // Anything that will not say, including a class added after this was written.
        _ => 3,
    }
}

/// Which of the enumerated devices to open: external, then internal, then software.
///
/// Returns an index into `classes`, which the caller zips back against its own enumeration. Ties
/// go to the earlier device, so the order the implementation reported is preserved among equals
/// and two discrete GPUs do not swap between runs.
///
/// Takes the classes rather than an `ash::Instance` for the same reason the rest of this module
/// does: the decision is the part worth testing, and it needs no device to make.
#[must_use]
pub fn preferred(classes: &[vk::PhysicalDeviceType]) -> Option<usize> {
    classes
        .iter()
        .enumerate()
        .min_by_key(|(index, class)| (rank(**class), *index))
        .map(|(index, _)| index)
}

/// Bytes one vertex of a declared type occupies, which is mbgl's `VertexAttribute::getStrideOf`.
///
/// What a binding's stride is when the wire says nothing: an `AttributeDesc` of stride zero means
/// tightly packed, and this is what tightly packed comes to. [`crate::vertices::plan`] is the one
/// caller, and mbgl's Vulkan backend does the same in two places --
/// `Drawable::setVertices` sets `getStrideOf(type)` on the attribute, and `upload_pass.cpp`
/// supplies it for an attribute with no override at all.
///
/// `None` for exactly the types [`vertex_format`] has none for, which is what makes the pair
/// usable together: a caller that got a format can get a stride. `UShort8` is a real sixteen bytes
/// and still `None` here, because a type with no vertex format cannot be bound and a size for it
/// would be a number nothing may use.
#[must_use]
pub const fn stride_of(declared: AttributeDataType) -> Option<u32> {
    // Grouped by the answer rather than by the type's own family, which is what keeps the arms
    // distinct: two shorts and four bytes are both four bytes, and clippy refuses to see the two
    // spellings written out separately.
    let bytes = match declared {
        AttributeDataType::Byte | AttributeDataType::UByte => 1,
        AttributeDataType::Byte2
        | AttributeDataType::UByte2
        | AttributeDataType::Short
        | AttributeDataType::UShort => 2,
        AttributeDataType::Byte3 | AttributeDataType::UByte3 => 3,
        AttributeDataType::Byte4
        | AttributeDataType::UByte4
        | AttributeDataType::Short2
        | AttributeDataType::UShort2
        | AttributeDataType::Int
        | AttributeDataType::UInt
        | AttributeDataType::Float => 4,
        AttributeDataType::Short3 | AttributeDataType::UShort3 => 6,
        AttributeDataType::Short4
        | AttributeDataType::UShort4
        | AttributeDataType::Int2
        | AttributeDataType::UInt2
        | AttributeDataType::Float2 => 8,
        AttributeDataType::Int3 | AttributeDataType::UInt3 | AttributeDataType::Float3 => 12,
        AttributeDataType::Int4 | AttributeDataType::UInt4 | AttributeDataType::Float4 => 16,
        // The two [`vertex_format`] refuses, and no wildcard arm for the same reason it has none:
        // a variant added to the generated enum should fail to compile here.
        AttributeDataType::UShort8 | AttributeDataType::Invalid => return None,
    };
    Some(bytes)
}

/// The Vulkan vertex format for a type a shader declares.
///
/// The declared type decides this and the supplied type does not: the producer's buffer may hold
/// bytes where the shader declares shorts, and the ABI records both — `declared_data_type` is what
/// the pipeline's vertex input must agree with, because that is what the shader reads.
///
/// # Integers are integers
///
/// Every integer type here maps to a `_SINT` or `_UINT` format, never a `_SNORM` or `_UNORM` one.
/// The distinction is invisible until it draws: a `Short2` tile position bound as `R16G16_SNORM`
/// arrives divided by 32,767, which puts the whole tile inside one pixel at the origin — and
/// nothing errors, because both formats are two shorts. mbgl's own line shader depends on the
/// unnormalized reading, which is why `fill_outline`'s reference calls `a_data` "raw bytes, not
/// normalized".
///
/// Returns `None` for a type no shader in the tables declares. That is not the same as a type
/// Vulkan lacks: `UShort8` has no single vertex format and nothing asks for one, so the absence is
/// recorded rather than worked around.
#[must_use]
pub const fn vertex_format(declared: AttributeDataType) -> Option<vk::Format> {
    let format = match declared {
        AttributeDataType::Byte => vk::Format::R8_SINT,
        AttributeDataType::Byte2 => vk::Format::R8G8_SINT,
        AttributeDataType::Byte3 => vk::Format::R8G8B8_SINT,
        AttributeDataType::Byte4 => vk::Format::R8G8B8A8_SINT,
        AttributeDataType::UByte => vk::Format::R8_UINT,
        AttributeDataType::UByte2 => vk::Format::R8G8_UINT,
        AttributeDataType::UByte3 => vk::Format::R8G8B8_UINT,
        AttributeDataType::UByte4 => vk::Format::R8G8B8A8_UINT,
        AttributeDataType::Short => vk::Format::R16_SINT,
        AttributeDataType::Short2 => vk::Format::R16G16_SINT,
        AttributeDataType::Short3 => vk::Format::R16G16B16_SINT,
        AttributeDataType::Short4 => vk::Format::R16G16B16A16_SINT,
        AttributeDataType::UShort => vk::Format::R16_UINT,
        AttributeDataType::UShort2 => vk::Format::R16G16_UINT,
        AttributeDataType::UShort3 => vk::Format::R16G16B16_UINT,
        AttributeDataType::UShort4 => vk::Format::R16G16B16A16_UINT,
        AttributeDataType::Int => vk::Format::R32_SINT,
        AttributeDataType::Int2 => vk::Format::R32G32_SINT,
        AttributeDataType::Int3 => vk::Format::R32G32B32_SINT,
        AttributeDataType::Int4 => vk::Format::R32G32B32A32_SINT,
        AttributeDataType::UInt => vk::Format::R32_UINT,
        AttributeDataType::UInt2 => vk::Format::R32G32_UINT,
        AttributeDataType::UInt3 => vk::Format::R32G32B32_UINT,
        AttributeDataType::UInt4 => vk::Format::R32G32B32A32_UINT,
        AttributeDataType::Float => vk::Format::R32_SFLOAT,
        AttributeDataType::Float2 => vk::Format::R32G32_SFLOAT,
        AttributeDataType::Float3 => vk::Format::R32G32B32_SFLOAT,
        AttributeDataType::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        // Eight shorts is not a vertex format. Nothing declares it; see `shaders::attribute_type`.
        //
        // No wildcard arm: a variant added to the generated enum should fail to compile here
        // rather than quietly become a type with no format.
        AttributeDataType::UShort8 | AttributeDataType::Invalid => return None,
    };
    Some(format)
}

#[cfg(test)]
mod tests {
    use super::{
        Attachment, DEPTH_STENCIL, Unsupported, check_vertex_formats, depth_stencil_format,
        preferred, stride_of, vertex_format,
    };
    use ash::vk;
    use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;

    /// A device reporting attachment support for exactly the formats listed.
    fn supports(list: &[vk::Format]) -> impl Fn(vk::Format) -> vk::FormatFeatureFlags + '_ {
        move |format| {
            if list.contains(&format) {
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::FormatFeatureFlags::empty()
            }
        }
    }

    /// The gating target's answer, which is the whole reason this is queried.
    ///
    /// V3D reports no attachment support for `D32_SFLOAT_S8_UINT`. A build that assumed the 32-bit
    /// form would pick a format the device cannot use, which is how Filament crashed on this part.
    #[test]
    fn a_v3d_shaped_device_gets_the_24_bit_format() {
        let v3d = supports(&[vk::Format::D24_UNORM_S8_UINT]);
        assert_eq!(
            depth_stencil_format(Attachment::DepthStencil, &v3d),
            Ok(vk::Format::D24_UNORM_S8_UINT)
        );
    }

    /// And a device the other way round gets the other, without the list needing to know which.
    #[test]
    fn a_radv_shaped_device_gets_the_32_bit_format() {
        let radv = supports(&[vk::Format::D32_SFLOAT_S8_UINT]);
        assert_eq!(
            depth_stencil_format(Attachment::DepthStencil, &radv),
            Ok(vk::Format::D32_SFLOAT_S8_UINT)
        );
    }

    /// Preference, not availability, decides between two the device offers.
    #[test]
    fn both_supported_takes_the_24_bit_one() {
        let both = supports(&DEPTH_STENCIL);
        assert_eq!(
            depth_stencil_format(Attachment::DepthStencil, &both),
            Ok(vk::Format::D24_UNORM_S8_UINT)
        );
    }

    /// A flat view takes a stencil-only attachment where the device has one.
    #[test]
    fn a_view_with_no_depth_takes_stencil_only_when_offered() {
        let with_s8 = supports(&[vk::Format::S8_UINT, vk::Format::D24_UNORM_S8_UINT]);
        assert_eq!(
            depth_stencil_format(Attachment::StencilOnly, &with_s8),
            Ok(vk::Format::S8_UINT)
        );
    }

    /// `S8_UINT` is optional, so its absence falls back rather than failing.
    ///
    /// The attachment then carries depth nothing reads, which costs memory and bandwidth and
    /// draws the same picture. Failing here would refuse a device over an optimization.
    #[test]
    fn stencil_only_falls_back_to_depth_stencil() {
        let no_s8 = supports(&[vk::Format::D24_UNORM_S8_UINT]);
        assert_eq!(
            depth_stencil_format(Attachment::StencilOnly, &no_s8),
            Ok(vk::Format::D24_UNORM_S8_UINT)
        );
    }

    /// A device offering none of them is refused, and at creation.
    #[test]
    fn no_usable_format_is_an_error_not_a_guess() {
        let none = supports(&[]);
        assert_eq!(
            depth_stencil_format(Attachment::DepthStencil, &none),
            Err(Unsupported::NoDepthStencilFormat)
        );
    }

    /// Attachment support is read from optimal tiling, never linear.
    ///
    /// A device advertising the format only for linear tiling has not offered a usable attachment,
    /// and taking it would be a performance bug on every target here rather than a correctness one
    /// -- which is exactly the kind that survives review.
    #[test]
    fn linear_only_support_does_not_count() {
        let linear_only = |_format: vk::Format| vk::FormatFeatureFlags::empty();
        assert_eq!(
            depth_stencil_format(Attachment::DepthStencil, linear_only),
            Err(Unsupported::NoDepthStencilFormat)
        );
    }

    /// Every vertex format tessella emits, accepted.
    #[test]
    fn the_emitted_vertex_formats_pass_on_a_device_that_takes_them() {
        let emitted = [
            vk::Format::R16G16_SINT,
            vk::Format::R16G16B16A16_SINT,
            vk::Format::R16G16_UINT,
            vk::Format::R16G16B16A16_UINT,
            vk::Format::R8G8B8A8_UINT,
            vk::Format::R32_SFLOAT,
            vk::Format::R32G32_SFLOAT,
            vk::Format::R32G32B32_SFLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
        ];
        let all = |_format: vk::Format| vk::FormatFeatureFlags::VERTEX_BUFFER;
        assert_eq!(check_vertex_formats(&emitted, all), Ok(()));
    }

    /// And one the device will not take is named, not skipped.
    #[test]
    fn a_missing_vertex_format_names_itself() {
        let missing = vk::Format::R16G16B16A16_UINT;
        let all_but_one = |format: vk::Format| {
            if format == missing {
                vk::FormatFeatureFlags::empty()
            } else {
                vk::FormatFeatureFlags::VERTEX_BUFFER
            }
        };
        assert_eq!(
            check_vertex_formats(&[vk::Format::R16G16_SINT, missing], all_but_one),
            Err(Unsupported::VertexFormat(missing))
        );
    }

    /// External before internal before software, which is the order asked for.
    #[test]
    fn a_discrete_device_wins_over_an_integrated_one_and_both_over_software() {
        let all = [
            vk::PhysicalDeviceType::CPU,
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::DISCRETE_GPU,
        ];
        assert_eq!(preferred(&all), Some(2));
        assert_eq!(preferred(&all[..2]), Some(1));
        assert_eq!(preferred(&all[..1]), Some(0));
    }

    /// The machine this was written on: an integrated GPU and a software implementation.
    ///
    /// The software one enumerates second here and first elsewhere, which is why the choice cannot
    /// be left to the enumeration order.
    #[test]
    fn software_is_taken_only_when_it_is_alone() {
        let listed = [
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::CPU,
        ];
        assert_eq!(preferred(&listed), Some(0));
        let reversed = [
            vk::PhysicalDeviceType::CPU,
            vk::PhysicalDeviceType::INTEGRATED_GPU,
        ];
        assert_eq!(preferred(&reversed), Some(1));
        assert_eq!(preferred(&[vk::PhysicalDeviceType::CPU]), Some(0));
    }

    /// A paravirtualized device is hardware, and ranks above software.
    #[test]
    fn a_virtual_device_beats_software() {
        let listed = [
            vk::PhysicalDeviceType::CPU,
            vk::PhysicalDeviceType::VIRTUAL_GPU,
        ];
        assert_eq!(preferred(&listed), Some(1));
    }

    /// A class this crate does not know still beats software.
    ///
    /// `OTHER` is what an implementation reports when it will not say, and so is any class added
    /// to Vulkan after this was written. Either is more likely to be hardware than the one class
    /// that is definitionally not, and ranking them below software would pick llvmpipe on the
    /// first board that reports something new.
    #[test]
    fn an_unknown_class_beats_software() {
        let listed = [
            vk::PhysicalDeviceType::CPU,
            vk::PhysicalDeviceType::OTHER,
            vk::PhysicalDeviceType::from_raw(99),
        ];
        assert_eq!(preferred(&listed), Some(1));
        assert_eq!(
            preferred(&[
                vk::PhysicalDeviceType::CPU,
                vk::PhysicalDeviceType::from_raw(99)
            ]),
            Some(1)
        );
    }

    /// Among equals the enumeration order stands, so a two-GPU machine does not alternate.
    #[test]
    fn a_tie_goes_to_the_earlier_device() {
        let two = [
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::PhysicalDeviceType::DISCRETE_GPU,
        ];
        assert_eq!(preferred(&two), Some(0));
    }

    /// No devices is not a choice, and is the caller's to report.
    #[test]
    fn nothing_enumerated_is_no_answer() {
        assert_eq!(preferred(&[]), None);
    }

    /// A type has a stride exactly where it has a vertex format.
    ///
    /// The two are used together and the pair is what makes that safe: `vertices::plan` takes the
    /// format first and then the stride for a descriptor that carried none, so a type with one and
    /// not the other would be a plan that binds a format at a stride of zero -- which is a binding
    /// where every vertex reads the first one.
    ///
    /// `UShort8` is the type that has neither. It is a real sixteen bytes and no vertex format, and
    /// a size for it here would be a number nothing may use.
    #[test]
    fn a_type_has_a_stride_exactly_where_it_has_a_format() {
        let every = [
            AttributeDataType::Byte,
            AttributeDataType::Byte2,
            AttributeDataType::Byte3,
            AttributeDataType::Byte4,
            AttributeDataType::UByte,
            AttributeDataType::UByte2,
            AttributeDataType::UByte3,
            AttributeDataType::UByte4,
            AttributeDataType::Short,
            AttributeDataType::Short2,
            AttributeDataType::Short3,
            AttributeDataType::Short4,
            AttributeDataType::UShort,
            AttributeDataType::UShort2,
            AttributeDataType::UShort3,
            AttributeDataType::UShort4,
            AttributeDataType::UShort8,
            AttributeDataType::Int,
            AttributeDataType::Int2,
            AttributeDataType::Int3,
            AttributeDataType::Int4,
            AttributeDataType::UInt,
            AttributeDataType::UInt2,
            AttributeDataType::UInt3,
            AttributeDataType::UInt4,
            AttributeDataType::Float,
            AttributeDataType::Float2,
            AttributeDataType::Float3,
            AttributeDataType::Float4,
            AttributeDataType::Invalid,
        ];
        let mut sized = 0;
        for declared in every {
            assert_eq!(
                stride_of(declared).is_some(),
                vertex_format(declared).is_some(),
                "{declared:?} has one of the two and not the other"
            );
            if let Some(bytes) = stride_of(declared) {
                assert!(bytes > 0, "{declared:?} occupies no bytes");
                sized += 1;
            }
        }
        assert_eq!(sized, every.len() - 2, "only UShort8 and Invalid have none");

        // And the sizes themselves, on the two the tables use most: a tile position is two shorts
        // and a packed color is four bytes, both four.
        assert_eq!(stride_of(AttributeDataType::Short2), Some(4));
        assert_eq!(stride_of(AttributeDataType::UByte4), Some(4));
        assert_eq!(stride_of(AttributeDataType::Float4), Some(16));
    }
}
