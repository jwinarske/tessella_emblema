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
use tessella_capture_abi::generated::mbgl_enums::AttributeDataType;

/// Whether a view's pass needs depth, or only stencil.
///
/// A view drawing fill-extrusions needs depth: the opaque pass resolves front-to-back and the
/// extrusion's own prepass decides which surface is nearest. A view with none still needs stencil,
/// for the per-tile clip masks, and can take a stencil-only format where the device has one —
/// which is a smaller attachment and, on a tiler, less to write back.
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
    };
    use ash::vk;

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
}
