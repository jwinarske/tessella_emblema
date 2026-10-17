// SPDX-License-Identifier: BSD-2-Clause
//! Reading an authored model, and refusing the ones this reader does not cover.
//!
//! The asset is a producer's: bytes it never looked inside, from a style a consumer did not write.
//! So the cases that matter are the ones where the bytes are not what they say -- a length past
//! the chunk, an index past the vertices, a width the reader has no path for -- because a reader
//! that trusts those draws from memory the asset does not own.
//!
//! Every fixture is built here rather than checked in, which is what makes a case readable: the
//! difference between two of them is the two lines that differ.

use tessella_emblema::meshes::{self, Error};

/// A GLB with the JSON and binary chunks a reader expects, in that order.
fn glb(json: &str, binary: &[u8]) -> Vec<u8> {
    let json = json.as_bytes();
    let json_padded = json.len().next_multiple_of(4);
    let bin_padded = binary.len().next_multiple_of(4);
    let total = 12 + 8 + json_padded + 8 + bin_padded;

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total).unwrap().to_le_bytes());

    out.extend_from_slice(&u32::try_from(json_padded).unwrap().to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    // Padded with spaces, which is what the spec asks of a JSON chunk: the padding is inside the
    // chunk's own length, so a reader that padded with zeros would hand this reader bytes that are
    // not JSON.
    out.resize(out.len() + (json_padded - json.len()), b' ');

    out.extend_from_slice(&u32::try_from(bin_padded).unwrap().to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(binary);
    out.resize(out.len() + (bin_padded - binary.len()), 0);
    out
}

/// Three float triples and three sixteen-bit indices, as a buffer.
fn triangle_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for index in [0u16, 1, 2] {
        out.extend_from_slice(&index.to_le_bytes());
    }
    // To four, so the chunk's own padding is not what a stride case is measuring.
    out.resize(out.len().next_multiple_of(4), 0);
    out
}

/// A document over [`triangle_bytes`]: positions at zero, indices at 36.
const TRIANGLE: &str = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 44}]
}"#;

/// The ordinary case, read field by field.
///
/// Field by field rather than by a count: a reader that returned three positions of zeros, or the
/// right positions in the wrong order, would satisfy a length assertion and draw a triangle that
/// is not the asset's.
#[test]
fn a_triangle_is_read_from_its_accessors() {
    let asset = glb(TRIANGLE, &triangle_bytes());
    let loaded = meshes::load(&asset).expect("the asset reads");

    assert_eq!(
        loaded.positions,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    );
    assert_eq!(loaded.indices, [0, 1, 2]);
    assert_eq!(
        loaded.position_bytes().len(),
        36,
        "twelve bytes a vertex, as a vertex buffer takes them"
    );
    assert_eq!(
        loaded.index_bytes().len(),
        12,
        "and widened to 32 bits, whatever the accessor stored"
    );
}

/// Indices of every width glTF allows come back as the same numbers.
///
/// The widening is the point: an asset's choice of index width is not something the renderer
/// should see. The numbers are chosen against the way a reader gets this wrong -- 257 is
/// `0x0101`, so a reader that took the low byte of every index whatever its width would read it
/// as 1, which is a valid index into this asset and would draw the wrong triangle in silence.
/// Indices of 0, 1 and 2 cannot catch that, because their low byte *is* their value.
#[test]
fn every_index_width_reads_the_same_numbers() {
    // Enough vertices that a three-byte index is in range, so the range check is not what fails.
    let vertices = 300usize;
    for (component, width, wanted) in [
        (5121u32, 1usize, [255u32, 2, 1]),
        (5123, 2, [257, 2, 1]),
        (5125, 4, [257, 2, 1]),
    ] {
        let mut binary = Vec::new();
        for vertex in 0..vertices {
            #[allow(clippy::cast_precision_loss)]
            for value in [vertex as f32, 0.0, 0.0] {
                binary.extend_from_slice(&value.to_le_bytes());
            }
        }
        let positions_len = binary.len();
        for index in wanted {
            binary.extend_from_slice(&index.to_le_bytes()[..width]);
        }
        binary.resize(binary.len().next_multiple_of(4), 0);
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1}}]}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": {vertices}, "type": "VEC3"}},
    {{"bufferView": 1, "componentType": {component}, "count": 3, "type": "SCALAR"}}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": {positions_len}}},
    {{"buffer": 0, "byteOffset": {positions_len}, "byteLength": {}}}
  ]
}}"#,
            width * 3
        );
        let loaded = meshes::load(&glb(&json, &binary))
            .unwrap_or_else(|why| panic!("componentType {component}: {why}"));
        assert_eq!(
            loaded.indices, wanted,
            "componentType {component} read the wrong numbers"
        );
    }
}

/// A view's stride is honored, so a position is not read out of its neighbor.
///
/// Interleaved attributes are what `byteStride` is for, and the padding here stands in for the
/// normal or the texture coordinate that would sit between two positions. A reader that stepped by
/// the element's own twelve bytes would read a position made of one vertex's `z` and the padding
/// after it.
#[test]
fn a_strided_view_steps_by_its_stride() {
    let mut binary = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for value in position {
            binary.extend_from_slice(&value.to_le_bytes());
        }
        // Four bytes of something else, as an interleaved attribute would be.
        binary.extend_from_slice(&0x7fff_ffffu32.to_le_bytes());
    }
    for index in [0u16, 1, 2] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    binary.resize(binary.len().next_multiple_of(4), 0);
    let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 48, "byteStride": 16},
    {"buffer": 0, "byteOffset": 48, "byteLength": 6}
  ]
}"#;

    let loaded = meshes::load(&glb(json, &binary)).expect("the asset reads");
    assert_eq!(
        loaded.positions,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "the stride was not stepped by"
    );
}

/// An accessor's own offset is relative to its view, and both are added.
#[test]
fn an_accessor_offset_is_added_to_its_view() {
    let mut binary = vec![0xffu8; 8];
    for value in [0.5f32, 0.25, 0.125] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    binary.extend_from_slice(&0u16.to_le_bytes());
    binary.resize(binary.len().next_multiple_of(4), 0);
    let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "byteOffset": 4, "componentType": 5126, "count": 1, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 1, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 4, "byteLength": 16},
    {"buffer": 0, "byteOffset": 20, "byteLength": 2}
  ]
}"#;

    let loaded = meshes::load(&glb(json, &binary)).expect("the asset reads");
    assert_eq!(
        loaded.positions,
        [[0.5, 0.25, 0.125]],
        "the two offsets were not added"
    );
}

/// A primitive with no indices draws its positions in order.
#[test]
fn an_unindexed_primitive_is_its_own_order() {
    let mut binary = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
  "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}]
}"#;

    let loaded = meshes::load(&glb(json, &binary)).expect("the asset reads");
    assert_eq!(loaded.indices, [0, 1, 2]);
}

/// Bytes that are not a GLB are refused as such, and each fault is told apart.
///
/// Told apart because they mean different things: a producer sending something that is not a GLB
/// is a protocol fault, where a length past the chunk is a corrupt asset and a version this
/// reader does not know is a newer file.
#[test]
fn a_container_fault_says_which_one_it_is() {
    assert_eq!(
        meshes::load(b"not a gltf at all").unwrap_err(),
        Error::NotGlb
    );
    assert!(matches!(
        meshes::load(&[]).unwrap_err(),
        Error::Truncated { .. }
    ));

    let mut wrong_version = glb(TRIANGLE, &triangle_bytes());
    wrong_version[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(
        meshes::load(&wrong_version).unwrap_err(),
        Error::Version(3),
        "glTF 2 is the only container there is"
    );

    // A chunk whose length runs past the asset's own total.
    let mut past = glb(TRIANGLE, &triangle_bytes());
    let json_length = u32::from_le_bytes([past[12], past[13], past[14], past[15]]);
    past[12..16].copy_from_slice(&(json_length + 4096).to_le_bytes());
    assert!(matches!(
        meshes::load(&past).unwrap_err(),
        Error::Truncated { .. }
    ));

    // JSON alone, which glTF allows and this cannot draw.
    let json = TRIANGLE.as_bytes();
    let padded = json.len().next_multiple_of(4);
    let mut without_binary = Vec::new();
    without_binary.extend_from_slice(b"glTF");
    without_binary.extend_from_slice(&2u32.to_le_bytes());
    without_binary.extend_from_slice(&u32::try_from(12 + 8 + padded).unwrap().to_le_bytes());
    without_binary.extend_from_slice(&u32::try_from(padded).unwrap().to_le_bytes());
    without_binary.extend_from_slice(b"JSON");
    without_binary.extend_from_slice(json);
    without_binary.resize(without_binary.len() + (padded - json.len()), b' ');
    assert_eq!(
        meshes::load(&without_binary).unwrap_err(),
        Error::Chunks("no binary chunk")
    );
}

/// An accessor that runs past the binary chunk is refused rather than read.
///
/// This is the case a reader that trusts the document draws memory the asset does not own from --
/// and on a device without robust buffer access, what it draws from is not even defined.
#[test]
fn an_accessor_past_the_buffer_is_truncated() {
    let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 300, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ]
}"#;
    assert!(matches!(
        meshes::load(&glb(json, &triangle_bytes())).unwrap_err(),
        Error::Truncated { .. }
    ));
}

/// An index past the positions is refused, with both numbers in the message.
#[test]
fn an_index_past_the_vertices_is_refused() {
    let mut binary = Vec::new();
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    for index in [0u16, 1, 9] {
        binary.extend_from_slice(&index.to_le_bytes());
    }
    binary.resize(binary.len().next_multiple_of(4), 0);

    let why = meshes::load(&glb(TRIANGLE, &binary)).unwrap_err();
    let Error::Unsupported(what) = &why else {
        panic!("an index past the vertices should be refused, not {why:?}");
    };
    assert!(
        what.contains('9') && what.contains('3'),
        "the message says neither the index nor the count: {what}"
    );
}

/// What the reader does not cover is refused by name, not guessed at.
#[test]
fn an_asset_this_reader_does_not_cover_says_so() {
    // A component type no accessor path has.
    let odd_component = TRIANGLE.replace("\"componentType\": 5126", "\"componentType\": 5130");
    assert!(matches!(
        meshes::load(&glb(&odd_component, &triangle_bytes())).unwrap_err(),
        Error::Unsupported(_)
    ));

    // A POSITION that is not float, which glTF does not allow and a reader could still try.
    let short_position = TRIANGLE
        .replace("\"componentType\": 5126", "\"componentType\": 5123")
        .replace("\"byteLength\": 36", "\"byteLength\": 18");
    assert!(matches!(
        meshes::load(&glb(&short_position, &triangle_bytes())).unwrap_err(),
        Error::Unsupported(_)
    ));

    // A mode that is not triangles.
    let lines = TRIANGLE.replace("\"indices\": 1", "\"indices\": 1, \"mode\": 1");
    assert!(matches!(
        meshes::load(&glb(&lines, &triangle_bytes())).unwrap_err(),
        Error::Unsupported(_)
    ));

    // A sparse accessor, which is a whole second addressing scheme.
    let sparse = TRIANGLE.replace(
        "\"type\": \"VEC3\"",
        "\"type\": \"VEC3\", \"sparse\": {\"count\": 1}",
    );
    assert!(matches!(
        meshes::load(&glb(&sparse, &triangle_bytes())).unwrap_err(),
        Error::Unsupported(_)
    ));
}

/// A document that is not JSON is refused with the byte it stopped at.
///
/// The byte is the point: a producer's asset that will not parse is something someone has to find
/// in a file they did not write, and "invalid JSON" does not help them.
#[test]
fn a_json_fault_names_where_it_stopped() {
    let truncated = &TRIANGLE[..TRIANGLE.len() / 2];
    let why = meshes::load(&glb(truncated, &triangle_bytes())).unwrap_err();
    let Error::Json(what) = &why else {
        panic!("a JSON fault should say so, not {why:?}");
    };
    assert!(
        what.contains("at byte"),
        "the message does not say where: {what}"
    );
}

/// An asset carrying no mesh at all is a document fault rather than an empty draw.
#[test]
fn an_asset_with_no_mesh_is_refused() {
    let json = r#"{"asset": {"version": "2.0"}, "nodes": [{}]}"#;
    assert!(matches!(
        meshes::load(&glb(json, &triangle_bytes())).unwrap_err(),
        Error::Json(_)
    ));
}

/// A count past the buffer is refused by its whole span, not by the first element that fails.
///
/// The accessor's span is checked against the chunk once, before any element is read, and that is
/// what makes it safe to size the output from a count the document chose -- a hundred million
/// vertices of a thirty-six byte buffer would otherwise reserve one and a quarter gigabytes
/// before the first read refused it.
///
/// The number in the error is what tells the two refusals apart: a per-element read wants twelve
/// bytes more than it has, where the span wants the whole 1,200,000,000. Without that, removing
/// the span check changes nothing a test can see -- which it did, and this is the case that
/// noticed.
#[test]
fn a_count_past_the_buffer_is_refused_by_its_whole_span() {
    let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 100000000, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ]
}"#;
    let why = meshes::load(&glb(json, &triangle_bytes())).unwrap_err();
    let Error::Truncated { wanted, had } = why else {
        panic!("a count past the buffer should be truncated, not {why:?}");
    };
    assert_eq!(had, 44, "the binary chunk this asset carries");
    assert_eq!(
        wanted, 1_200_000_000,
        "the whole span was not what refused it, so the check that bounds the \
         allocation is not what ran"
    );
}
