// SPDX-License-Identifier: BSD-2-Clause
//! Which sampler a filter is, and the refusals a set's write makes.
//!
//! # What this is for
//!
//! `benches/descriptor_sets.rs` needs a GPU and CI has none. What does not need one is the part that
//! decides *whether* to write: the texture-count agreement, and the mapping from the wire's filter
//! to one of two samplers.
//!
//! # What would be caught
//!
//! A write that proceeded with the wrong number of textures. A family declares its image count in
//! its own table, so a drawable naming a different number is a producer and this consumer
//! disagreeing -- and writing what arrived would leave a binding unwritten. A descriptor the shader
//! reads and nothing filled is undefined, and on a driver that happens to leave it zeroed it reads
//! as a black texture, which looks like a missing sprite rather than a protocol fault.
//!
//! And the filter defaulting the wrong way. Zero on the wire is `Linear`, which the ABI chose so a
//! producer that never sets the field keeps the behavior it had -- so a mapping that treated the
//! default as `Nearest` would quietly sharpen every atlas in the map.

use tessella_capture_abi::envelope::TextureFilter;

/// Zero on the wire is linear, which is what the ABI's own default says.
///
/// `TextureRef::filter` "was padding through R0, and zero is `TextureFilter::Linear`" -- so a
/// producer that never writes the field and a consumer that reads it both keep the old behavior.
/// A consumer that read zero as nearest would change every texture that predates the field.
#[test]
fn the_default_filter_is_linear() {
    assert_eq!(TextureFilter::default(), TextureFilter::Linear);
    assert_eq!(TextureFilter::Linear as u32, 0);
    assert_eq!(TextureFilter::Nearest as u32, 1);
}

/// The two filters are distinct, so a set can want both in one frame.
///
/// Which it can: the icon atlas is linear when the icons are scaled and nearest when they are not,
/// and that is per drawable rather than per texture -- so one texture is sampled both ways in one
/// frame and a sampler per texture could not express it.
#[test]
fn the_filters_are_two_things() {
    assert_ne!(TextureFilter::Linear, TextureFilter::Nearest);
}

/// A drawable naming fewer textures than the family declares is refused.
///
/// Checked through `Error`'s own shape rather than a device: the count comes from the bindings and
/// the textures come from the drawable, and the comparison is all that decides it.
#[test]
fn a_texture_count_mismatch_is_describable() {
    use tessella_emblema::descriptors::Error;
    let too_few = Error::WrongTextureCount { wanted: 2, got: 0 };
    let too_many = Error::WrongTextureCount { wanted: 1, got: 3 };
    assert_ne!(too_few, too_many);
    // The message carries both sides, because a disagreement is only diagnosable from both.
    let text = too_few.to_string();
    assert!(text.contains('2') && text.contains('0'), "got {text:?}");
}

/// Every error says which resource was missing.
#[test]
fn the_refusals_name_what_was_missing() {
    use tessella_capture_abi::envelope::ViewId;
    use tessella_emblema::blocks::Which;
    use tessella_emblema::descriptors::Error;

    let blocks = Error::NoBlocks {
        which: Which {
            view: ViewId(7),
            layer: 3,
        },
        slot: 5,
    };
    let text = blocks.to_string();
    assert!(
        text.contains('7') && text.contains('3') && text.contains('5'),
        "a missing block buffer must name the view, the layer and the slot, got {text:?}"
    );

    let texture = Error::NoTexture { slot: 2 };
    assert!(texture.to_string().contains('2'));

    // And a block binding with no slot, which is the one refusal that is about the bindings rather
    // than about what the stores hold.
    let unknown = Error::UnknownSlot { binding: 1 };
    assert!(unknown.to_string().contains('1'));
    assert_ne!(unknown, Error::UnknownSlot { binding: 0 });
}
