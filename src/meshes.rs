// SPDX-License-Identifier: BSD-2-Clause
//! Authored models: reading the asset a `MeshAdd` points at, and the shader that draws one.
//!
//! A model tile is the one thing this crate draws that the producer did not compute. What arrives
//! is the asset itself -- `MeshFormat::Glb`, a slab reference, bytes the producer never looked
//! inside -- and what the producer contributes is the decision to draw it and the transform that
//! places it. So the family tables, the attribute descriptions and the generated declarations
//! every other draw is built from are all absent here, and this module is what stands in for
//! them.
//!
//! # Why the reader is written here
//!
//! It reads one primitive's positions and indices out of a binary glTF, and nothing else: no
//! materials, no textures, no nodes, no animation, no sparse accessors. That is the whole of what
//! the draw below needs, and it is a small enough subset to test exhaustively -- which is the
//! argument for writing it rather than taking a glTF crate and a JSON crate for it. `naga` is a
//! dev-dependency in this crate deliberately, for the same reason: what a dependency costs is not
//! only its code.
//!
//! The parser underneath is a JSON reader of about the same size, for the same reason, and it is
//! kept behind [`load`] so that swapping either half for a crate is a change to this file alone.
//!
//! # What it is not
//!
//! A PBR renderer. glTF describes materials this does not read and the shader does not have: the
//! positions are drawn flat, in one color, which is what says the asset reached the device and was
//! placed where the producer asked. Shading an authored model is its own piece of work with its
//! own oracle, and this is the path it would be built on.

/// Why an asset could not be read.
///
/// Separate variants rather than one string, because the two halves mean different things to
/// whoever is looking: a container fault is a producer that sent something that is not a GLB,
/// where [`Self::Unsupported`] is an asset this reader does not cover and a perfectly good file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The bytes do not open with glTF's magic.
    NotGlb,
    /// A container version this reader does not know. glTF 2 is the only one there is.
    Version(u32),
    /// A length in the container or an accessor runs past the bytes there are.
    ///
    /// Both numbers, because which one is wrong is the question: an accessor wanting a gigabyte
    /// of a thirty-six byte chunk is a document that disagrees with its own buffer, where one
    /// wanting four bytes more than it has is a chunk that was cut short. The reader checks an
    /// accessor's whole span before it reads an element of it -- that is what makes it safe to
    /// size the output from a count the document chose -- and `wanted` is how a caller can tell
    /// that refusal from a per-element one.
    Truncated {
        /// Bytes the asset said were there.
        wanted: usize,
        /// Bytes there are.
        had: usize,
    },
    /// The chunk layout is not JSON-then-BIN.
    Chunks(&'static str),
    /// The JSON chunk could not be parsed.
    Json(String),
    /// The asset is valid glTF and names something this reader does not cover.
    Unsupported(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotGlb => write!(f, "not a GLB: the magic is not glTF"),
            Self::Version(saw) => write!(f, "a GLB of version {saw}, and 2 is the only one"),
            Self::Truncated { wanted, had } => {
                write!(f, "a length wants {wanted} bytes of {had}")
            }
            Self::Chunks(why) => write!(f, "the chunk layout: {why}"),
            Self::Json(why) => write!(f, "the JSON chunk: {why}"),
            Self::Unsupported(what) => write!(f, "not covered by this reader: {what}"),
        }
    }
}

impl core::error::Error for Error {}

/// One primitive, as the draw below wants it.
///
/// Positions and indices and nothing else, in the order the asset declares them: an indexed
/// triangle list, which is what every other draw in this crate is too.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    /// Vertex positions, in the asset's own coordinates.
    pub positions: Vec<[f32; 3]>,
    /// Indices into [`Self::positions`], widened to 32 bits whatever the accessor stored.
    ///
    /// Widened here rather than carried at its own width because the alternative is two index
    /// buffer formats and two draw paths for a difference the asset should not impose on the
    /// renderer. A model whose indices fit in sixteen bits costs an extra two bytes each.
    pub indices: Vec<u32>,
}

impl Loaded {
    /// Positions as the bytes a vertex buffer takes.
    #[must_use]
    pub fn position_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.positions.len() * 12);
        for position in &self.positions {
            for value in position {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    /// Indices as the bytes an index buffer takes, at `VK_INDEX_TYPE_UINT32`.
    #[must_use]
    pub fn index_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.indices.len() * 4);
        for index in &self.indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
        out
    }
}

/// glTF's magic, as the four bytes a GLB opens with.
const MAGIC: &[u8; 4] = b"glTF";
/// The JSON chunk's type, as its four bytes.
const CHUNK_JSON: &[u8; 4] = b"JSON";
/// The binary chunk's type. Trailing space: the type is four bytes and `BIN` is three.
const CHUNK_BIN: &[u8; 4] = b"BIN\0";

/// Reads one primitive out of a binary glTF.
///
/// The first primitive of the first mesh, which is the whole of what a model tile needs to be for
/// this to draw it. An asset with more is not refused -- the rest is left unread, and
/// [`Loaded`]'s doc says what that means.
///
/// # Errors
///
/// [`Error`], which separates a container fault from an asset this reader does not cover.
pub fn load(bytes: &[u8]) -> Result<Loaded, Error> {
    let (json, binary) = chunks(bytes)?;
    let json = core::str::from_utf8(json)
        .map_err(|_| Error::Json(String::from("the chunk is not UTF-8")))?;
    let document = json::parse(json).map_err(Error::Json)?;
    primitive(&document, binary)
}

/// Splits a GLB into its JSON and binary chunks.
///
/// The header is twelve bytes -- magic, version, total length -- and each chunk is a length, a
/// four-byte type and that many bytes, padded to four. The JSON chunk is first and required; the
/// binary chunk is optional in glTF and required here, because an accessor with no buffer behind
/// it describes nothing this can draw.
fn chunks(bytes: &[u8]) -> Result<(&[u8], &[u8]), Error> {
    if bytes.len() < 12 {
        return Err(Error::Truncated {
            wanted: 12,
            had: bytes.len(),
        });
    }
    if &bytes[0..4] != MAGIC {
        return Err(Error::NotGlb);
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 2 {
        return Err(Error::Version(version));
    }
    // The header's own total, which is the asset's length and not the reader's: a file longer than
    // it says is as much a fault as one shorter, because the chunks are walked against it.
    let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if total > bytes.len() || total < 12 {
        return Err(Error::Truncated {
            wanted: total,
            had: bytes.len(),
        });
    }

    let mut at = 12;
    let mut json: Option<&[u8]> = None;
    let mut binary: Option<&[u8]> = None;
    while at + 8 <= total {
        let length =
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
        let kind = &bytes[at + 4..at + 8];
        let from = at + 8;
        let to = from.checked_add(length).ok_or(Error::Truncated {
            wanted: usize::MAX,
            had: total,
        })?;
        if to > total {
            return Err(Error::Truncated {
                wanted: to,
                had: total,
            });
        }
        if kind == CHUNK_JSON {
            if json.is_some() {
                return Err(Error::Chunks("two JSON chunks"));
            }
            json = Some(&bytes[from..to]);
        } else if kind == CHUNK_BIN {
            if binary.is_some() {
                return Err(Error::Chunks("two binary chunks"));
            }
            binary = Some(&bytes[from..to]);
        }
        // Anything else is an extension's chunk, which glTF says to skip rather than refuse.
        //
        // Padded to four: the length is the chunk's own and the next one starts aligned.
        at = from + length.next_multiple_of(4);
    }

    let json = json.ok_or(Error::Chunks("no JSON chunk"))?;
    let binary = binary.ok_or(Error::Chunks("no binary chunk"))?;
    Ok((json, binary))
}

/// The first primitive of the first mesh, read against the binary chunk.
fn primitive(document: &json::Value<'_>, binary: &[u8]) -> Result<Loaded, Error> {
    let meshes = document
        .get("meshes")
        .and_then(json::Value::array)
        .ok_or_else(|| missing("meshes"))?;
    let mesh = meshes.first().ok_or_else(|| missing("a mesh"))?;
    let primitives = mesh
        .get("primitives")
        .and_then(json::Value::array)
        .ok_or_else(|| missing("primitives"))?;
    let primitive = primitives.first().ok_or_else(|| missing("a primitive"))?;

    // Triangles, which is glTF's default mode and the only one an indexed triangle list draws.
    if let Some(mode) = primitive.get("mode").and_then(json::Value::index)
        && mode != 4
    {
        return Err(Error::Unsupported(format!(
            "primitive mode {mode}, and 4 is triangles"
        )));
    }

    let position = primitive
        .get("attributes")
        .and_then(|attributes| attributes.get("POSITION"))
        .and_then(json::Value::index)
        .ok_or_else(|| missing("a POSITION attribute"))?;
    let positions = read_positions(document, binary, position)?;

    let indices = match primitive.get("indices").and_then(json::Value::index) {
        Some(accessor) => read_indices(document, binary, accessor)?,
        // Non-indexed, which glTF allows: the positions are the triangle list.
        None => (0..u32::try_from(positions.len()).unwrap_or(u32::MAX)).collect(),
    };

    // Every index has to name a vertex this asset carries. A model whose indices run past its
    // positions draws whatever the buffer happens to hold after them, which on a device without
    // robust buffer access is not even defined -- so it is refused here rather than drawn.
    let count = positions.len();
    if let Some(past) = indices.iter().find(|index| **index as usize >= count) {
        return Err(Error::Unsupported(format!(
            "an index of {past} against {count} positions"
        )));
    }
    Ok(Loaded { positions, indices })
}

/// What an accessor says about its own bytes, resolved against the buffer behind it.
struct Window {
    /// Offset of the first element, from the start of the binary chunk.
    at: usize,
    /// Bytes from one element to the next, which is the view's stride where it has one.
    stride: usize,
    /// How many elements.
    count: usize,
    /// glTF's `componentType`.
    component: u64,
}

/// Resolves accessor `index` against its buffer view.
fn window(document: &json::Value<'_>, binary: &[u8], index: u64) -> Result<Window, Error> {
    let accessor = document
        .get("accessors")
        .and_then(json::Value::array)
        .and_then(|all| all.get(usize::try_from(index).unwrap_or(usize::MAX)))
        .ok_or_else(|| missing("the accessor an attribute names"))?;
    if accessor.get("sparse").is_some() {
        return Err(Error::Unsupported(String::from("a sparse accessor")));
    }
    let component = accessor
        .get("componentType")
        .and_then(json::Value::index)
        .ok_or_else(|| missing("componentType"))?;
    let count = accessor
        .get("count")
        .and_then(json::Value::index)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| missing("count"))?;
    let kind = accessor
        .get("type")
        .and_then(json::Value::string)
        .ok_or_else(|| missing("an accessor type"))?;
    let components = match kind {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        other => {
            return Err(Error::Unsupported(format!("an accessor of {other}")));
        }
    };
    let width = component_width(component)?;
    // The accessor's own offset is relative to its view, and the view's is absolute. A view is
    // required here: an accessor without one is zeros in glTF, which is nothing to draw.
    let view_index = accessor
        .get("bufferView")
        .and_then(json::Value::index)
        .ok_or_else(|| missing("a bufferView"))?;
    let view = document
        .get("bufferViews")
        .and_then(json::Value::array)
        .and_then(|all| all.get(usize::try_from(view_index).unwrap_or(usize::MAX)))
        .ok_or_else(|| missing("the bufferView an accessor names"))?;
    if view
        .get("buffer")
        .and_then(json::Value::index)
        .is_some_and(|buffer| buffer != 0)
    {
        return Err(Error::Unsupported(String::from(
            "a buffer other than the GLB's own",
        )));
    }
    let view_at = view
        .get("byteOffset")
        .and_then(json::Value::index)
        .and_then(|at| usize::try_from(at).ok())
        .unwrap_or(0);
    let accessor_at = accessor
        .get("byteOffset")
        .and_then(json::Value::index)
        .and_then(|at| usize::try_from(at).ok())
        .unwrap_or(0);
    let tight = width * components;
    let stride = view
        .get("byteStride")
        .and_then(json::Value::index)
        .and_then(|stride| usize::try_from(stride).ok())
        .unwrap_or(tight);
    if stride < tight {
        return Err(Error::Unsupported(format!(
            "a byteStride of {stride} under the element's own {tight}"
        )));
    }

    let at = view_at.checked_add(accessor_at).ok_or(Error::Truncated {
        wanted: usize::MAX,
        had: binary.len(),
    })?;
    // The last element's own bytes, not `count * stride`: a tightly-read last element may stop
    // short of a stride, and a view sized exactly is not truncated.
    let span = count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|span| span.checked_add(tight))
        .unwrap_or(0);
    let wanted = at.checked_add(span).ok_or(Error::Truncated {
        wanted: usize::MAX,
        had: binary.len(),
    })?;
    if wanted > binary.len() {
        return Err(Error::Truncated {
            wanted,
            had: binary.len(),
        });
    }
    Ok(Window {
        at,
        stride,
        count,
        component,
    })
}

/// Bytes in one component of a `componentType`.
fn component_width(component: u64) -> Result<usize, Error> {
    match component {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        other => Err(Error::Unsupported(format!("a componentType of {other}"))),
    }
}

/// Reads a `VEC3` of floats, which is what glTF requires a POSITION to be.
fn read_positions(
    document: &json::Value<'_>,
    binary: &[u8],
    index: u64,
) -> Result<Vec<[f32; 3]>, Error> {
    let window = window(document, binary, index)?;
    if window.component != 5126 {
        return Err(Error::Unsupported(format!(
            "a POSITION of componentType {}, and glTF requires float",
            window.component
        )));
    }
    let mut out = Vec::with_capacity(window.count);
    for element in 0..window.count {
        let at = window.at + element * window.stride;
        let mut position = [0f32; 3];
        for (axis, value) in position.iter_mut().enumerate() {
            let from = at + axis * 4;
            let bytes: [u8; 4] = binary
                .get(from..from + 4)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(Error::Truncated {
                    wanted: from + 4,
                    had: binary.len(),
                })?;
            *value = f32::from_le_bytes(bytes);
        }
        out.push(position);
    }
    Ok(out)
}

/// Reads a `SCALAR` of unsigned integers, widened to 32 bits.
fn read_indices(document: &json::Value<'_>, binary: &[u8], index: u64) -> Result<Vec<u32>, Error> {
    let window = window(document, binary, index)?;
    let width = component_width(window.component)?;
    // Unsigned only: glTF says indices are `UNSIGNED_BYTE`, `UNSIGNED_SHORT` or `UNSIGNED_INT`,
    // and a signed index is an asset fault rather than a width this could widen.
    if !matches!(window.component, 5121 | 5123 | 5125) {
        return Err(Error::Unsupported(format!(
            "indices of componentType {}",
            window.component
        )));
    }
    let mut out = Vec::with_capacity(window.count);
    for element in 0..window.count {
        let at = window.at + element * window.stride;
        let bytes = binary.get(at..at + width).ok_or(Error::Truncated {
            wanted: at + width,
            had: binary.len(),
        })?;
        let value = match width {
            1 => u32::from(bytes[0]),
            2 => u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            _ => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        };
        out.push(value);
    }
    Ok(out)
}

/// An asset that does not carry something this needs.
fn missing(what: &str) -> Error {
    Error::Json(format!("no {what}"))
}

/// The shader an authored mesh is drawn with.
///
/// Positions from a vertex buffer and a matrix from a storage buffer indexed by `instance_index`,
/// which is the arrangement every other draw here uses -- `draws` puts the slot in `firstInstance`
/// for exactly that. One entry per mesh of the pass.
///
/// # One color, and why that is the right amount
///
/// glTF describes materials this reader does not read. What this draw establishes is that the
/// asset arrived, was read, and was placed where the producer's transform asked -- which a flat
/// color says exactly as well as a shaded one, and a shaded one would need a material path, its
/// textures and an oracle to be right against. The color is opaque and recognizable so a bench
/// can tell the mesh from the clear and from every family's output.
pub const BODY: &str = r"
struct MeshUbo {
    matrix: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> mesh_ubo: array<MeshUbo>;

@vertex
fn vertex_main(
    @location(0) position: vec3<f32>,
    @builtin(instance_index) mesh: u32,
) -> @builtin(position) vec4<f32> {
    let clip = mesh_ubo[mesh].matrix * vec4<f32>(position, 1.0);
    // The same remap every placement here ends with: the matrix is the producer's and its clip
    // `z` is OpenGL's `-w..w`, where Vulkan's volume is `0..w`.
    return vec4<f32>(clip.xy, (clip.z + clip.w) * 0.5, clip.w);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.25, 0.75, 0.5, 1.0);
}
";

/// A JSON reader, which is the other half of what [`load`] does not take a dependency for.
///
/// Enough of RFC 8259 to read a glTF document: objects, arrays, strings, numbers, the three
/// literals, and the whitespace between them. What it leaves out is everything a reader of this
/// document does not need -- a string's escapes are scanned past rather than resolved, because
/// nothing [`load`] reads is a string that could carry one, and a number is a `f64` with an
/// integer accessor beside it rather than two number types.
mod json {
    /// A JSON value, borrowed from the document it was parsed from.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value<'a> {
        /// `null`.
        Null,
        /// `true` or `false`.
        Bool(bool),
        /// Any number, as the only numeric type JSON has.
        Number(f64),
        /// A string, unescaped: see the module's own doc.
        Str(&'a str),
        /// An array, in order.
        Array(Vec<Value<'a>>),
        /// An object, in the order its members were written.
        ///
        /// A `Vec` rather than a map: a glTF object has a handful of members, a linear scan over
        /// that beats hashing it, and the order is the document's.
        Object(Vec<(&'a str, Value<'a>)>),
    }

    impl<'a> Value<'a> {
        /// The member a key names, or `None` for anything that is not an object with it.
        #[must_use]
        pub fn get(&self, key: &str) -> Option<&Value<'a>> {
            match self {
                Self::Object(members) => members
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, value)| value),
                _ => None,
            }
        }

        /// The elements, or `None` for anything that is not an array.
        #[must_use]
        pub fn array(&self) -> Option<&[Value<'a>]> {
            match self {
                Self::Array(values) => Some(values),
                _ => None,
            }
        }

        /// The string, or `None` for anything that is not one.
        #[must_use]
        pub fn string(&self) -> Option<&'a str> {
            match self {
                Self::Str(text) => Some(text),
                _ => None,
            }
        }

        /// The number as the non-negative integer glTF uses for an index, a count or an offset.
        ///
        /// `None` for a number that is not one: every number this reader asks for by this name is
        /// an index into an array, a count of elements or an offset in bytes, and a fractional or
        /// negative value in any of those is an asset fault rather than something to truncate.
        #[must_use]
        pub fn index(&self) -> Option<u64> {
            match self {
                Self::Number(value)
                    if value.fract() == 0.0
                        && *value >= 0.0
                        && *value <= 9_007_199_254_740_992.0 =>
                {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Some(*value as u64)
                }
                _ => None,
            }
        }
    }

    /// Parses a whole document.
    ///
    /// # Errors
    ///
    /// The reason, as a sentence naming the byte it stopped at: a JSON fault in a producer's
    /// asset is a thing someone has to find, and "invalid JSON" does not help them.
    pub fn parse(text: &str) -> Result<Value<'_>, String> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            text,
            at: 0,
        };
        reader.space();
        let value = reader.value()?;
        reader.space();
        if reader.at != reader.bytes.len() {
            return Err(reader.say("trailing bytes after the document"));
        }
        Ok(value)
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        text: &'a str,
        at: usize,
    }

    impl<'a> Reader<'a> {
        fn say(&self, why: &str) -> String {
            format!("{why}, at byte {}", self.at)
        }

        fn space(&mut self) {
            while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.at += 1;
            }
        }

        fn byte(&self) -> Option<u8> {
            self.bytes.get(self.at).copied()
        }

        fn want(&mut self, byte: u8) -> Result<(), String> {
            if self.byte() == Some(byte) {
                self.at += 1;
                return Ok(());
            }
            Err(self.say(&format!("expected {}", byte as char)))
        }

        fn literal(&mut self, word: &str, value: Value<'a>) -> Result<Value<'a>, String> {
            if self.text[self.at..].starts_with(word) {
                self.at += word.len();
                return Ok(value);
            }
            Err(self.say(&format!("expected {word}")))
        }

        fn value(&mut self) -> Result<Value<'a>, String> {
            match self.byte() {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => self.string().map(Value::Str),
                Some(b't') => self.literal("true", Value::Bool(true)),
                Some(b'f') => self.literal("false", Value::Bool(false)),
                Some(b'n') => self.literal("null", Value::Null),
                Some(_) => self.number(),
                None => Err(self.say("the document ends where a value was expected")),
            }
        }

        fn object(&mut self) -> Result<Value<'a>, String> {
            self.want(b'{')?;
            let mut members = Vec::new();
            self.space();
            if self.byte() == Some(b'}') {
                self.at += 1;
                return Ok(Value::Object(members));
            }
            loop {
                self.space();
                let key = self.string()?;
                self.space();
                self.want(b':')?;
                self.space();
                let value = self.value()?;
                members.push((key, value));
                self.space();
                match self.byte() {
                    Some(b',') => self.at += 1,
                    Some(b'}') => {
                        self.at += 1;
                        return Ok(Value::Object(members));
                    }
                    _ => return Err(self.say("expected , or } in an object")),
                }
            }
        }

        fn array(&mut self) -> Result<Value<'a>, String> {
            self.want(b'[')?;
            let mut values = Vec::new();
            self.space();
            if self.byte() == Some(b']') {
                self.at += 1;
                return Ok(Value::Array(values));
            }
            loop {
                self.space();
                values.push(self.value()?);
                self.space();
                match self.byte() {
                    Some(b',') => self.at += 1,
                    Some(b']') => {
                        self.at += 1;
                        return Ok(Value::Array(values));
                    }
                    _ => return Err(self.say("expected , or ] in an array")),
                }
            }
        }

        /// A string, sliced rather than unescaped.
        ///
        /// An escape is scanned past so the string ends where it ends, and `\uXXXX` with it: what
        /// comes back is the document's own bytes between the quotes. Nothing [`super::load`]
        /// reads is a string that could carry one -- the keys are glTF's and the one value is an
        /// accessor type -- so resolving them would be code with no caller.
        fn string(&mut self) -> Result<&'a str, String> {
            self.want(b'"')?;
            let from = self.at;
            loop {
                match self.byte() {
                    Some(b'"') => {
                        let text = &self.text[from..self.at];
                        self.at += 1;
                        return Ok(text);
                    }
                    Some(b'\\') => {
                        // The escape and the byte it escapes, which is enough to not stop at a
                        // quote that is part of the string.
                        self.at += 2;
                    }
                    Some(_) => self.at += 1,
                    None => return Err(self.say("a string with no closing quote")),
                }
            }
        }

        fn number(&mut self) -> Result<Value<'a>, String> {
            let from = self.at;
            if self.byte() == Some(b'-') {
                self.at += 1;
            }
            while matches!(
                self.byte(),
                Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
            ) {
                self.at += 1;
            }
            let text = &self.text[from..self.at];
            text.parse::<f64>()
                .map(Value::Number)
                .map_err(|_| format!("not a number: {text:?}, at byte {from}"))
        }
    }
}
