//! The surface a family is drawn on, and what placing a vertex on it costs.
//!
//! A family says what a vertex *is* — a fill's polygon corner, a circle's extruded quad corner, a
//! line's offset point. A surface says where that lands on screen. The two are independent: the
//! same fill draws on a flat Mercator plane, bent onto a sphere, or lifted onto a DEM, and in all
//! three the paint arithmetic is the same arithmetic.
//!
//! So a module is one (family, surface) pair, and this is the surface half. Each surface supplies
//! one function:
//!
//! ```wgsl
//! fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32>
//! ```
//!
//! taking a tile-local position whose `z` is height above the surface in meters, and returning a
//! clip position. A family's body never names a matrix multiply, so adding a surface is a
//! placement and not a fifth copy of every body.
//!
//! # Why a separate module and not a branch
//!
//! A uniform branch would put the fork on every vertex of every flat map ever drawn, to serve
//! modes most of them never enter. The permutation machinery is already here for the paint
//! properties ([`crate::spec`]), and a surface is the same kind of switch resolved at the same
//! time.
//!
//! # Which surfaces a family has
//!
//! Not every pair exists, and the producer decides which:
//!
//! * [`Surface::Plane`] and [`Surface::Globe`] need nothing of the family — the globe reads the
//!   drawable's own matrix, which the producer fills with the tile-to-Mercator step instead of the
//!   tile-to-clip one when the camera is in globe mode. So every family has both.
//! * [`Surface::GlobeAnchored`] needs a [`GLOBE_BEND_UBO`] per drawable, which the producer writes
//!   only for a layer whose geometry is a tile's.
//! * [`Surface::Terrain`] needs a [`TERRAIN_DRAWABLE_UBO`] per drawable, which the producer writes
//!   only where a binding carries `DrawFlags::ON_TERRAIN`.
//!
//! A background is the family that has neither: it covers the viewport rather than a tile, so the
//! producer sends it no bend block and never raises it.

use tessella_capture_abi::generated::ubo_layouts::{UboField, UboFieldKind, UboLayout};

static GLOBE_CAMERA_FIELDS: [UboField; 1] = [UboField {
    name: "globe_matrix",
    offset: 0,
    kind: UboFieldKind::Mat4,
}];

/// The unit sphere to clip, for a frame in globe mode.
///
/// `CameraUpdate::globe_matrix`, narrowed to `f32`. One matrix for the whole frame rather than one
/// per drawable, which is why the placement reads element zero and not `ubo_index`.
pub static GLOBE_CAMERA_UBO: UboLayout = UboLayout {
    name: "GlobeCameraUBO",
    header: "envelope.rs",
    align: 16,
    size: 64,
    stride: 64,
    fields: &GLOBE_CAMERA_FIELDS,
};

static GLOBE_BEND_FIELDS: [UboField; 7] = [
    UboField {
        name: "anchor",
        offset: 0,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_u",
        offset: 16,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_v",
        offset: 32,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_uu",
        offset: 48,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_vv",
        offset: 64,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_uv",
        offset: 80,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "d_h",
        offset: 96,
        kind: UboFieldKind::Vec4,
    },
];

/// The anchored bend's coefficients, in the order `globe_ubo::GlobeBendUbo` declares them.
///
/// Seven rows and no scalars, so the WGSL offsets are the producer's without padding — which the
/// tests check against the struct rather than assume.
pub static GLOBE_BEND_UBO: UboLayout = UboLayout {
    name: "GlobeBendUBO",
    header: "globe_ubo.rs",
    align: 16,
    size: tessella_capture_abi::globe_ubo::GlobeBendUbo::STRIDE,
    stride: tessella_capture_abi::globe_ubo::GlobeBendUbo::STRIDE,
    fields: &GLOBE_BEND_FIELDS,
};

static TERRAIN_DRAWABLE_FIELDS: [UboField; 5] = [
    UboField {
        name: "matrix",
        offset: 0,
        kind: UboFieldKind::Mat4,
    },
    UboField {
        name: "unpack",
        offset: 64,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "color",
        offset: 80,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "params",
        offset: 96,
        kind: UboFieldKind::Vec4,
    },
    UboField {
        name: "skirt",
        offset: 112,
        kind: UboFieldKind::Vec4,
    },
];

/// The raise block, in the order `terrain_ubo::TerrainDrawableUbo` declares it.
pub static TERRAIN_DRAWABLE_UBO: UboLayout = UboLayout {
    name: "TerrainDrawableUBO",
    header: "terrain_ubo.rs",
    align: 16,
    size: tessella_capture_abi::terrain_ubo::TerrainDrawableUbo::STRIDE,
    stride: tessella_capture_abi::terrain_ubo::TerrainDrawableUbo::STRIDE,
    fields: &TERRAIN_DRAWABLE_FIELDS,
};

static NO_BLOCKS: [&UboLayout; 0] = [];
static GLOBE_BLOCKS: [&UboLayout; 1] = [&GLOBE_CAMERA_UBO];
static GLOBE_ANCHORED_BLOCKS: [&UboLayout; 1] = [&GLOBE_BEND_UBO];
static TERRAIN_BLOCKS: [&UboLayout; 1] = [&TERRAIN_DRAWABLE_UBO];

static NO_TEXTURES: [&str; 0] = [];
static TERRAIN_TEXTURES: [&str; 1] = ["elevation"];

/// What a vertex is placed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Surface {
    /// Flat Mercator: the drawable's matrix reaches clip space and that is the whole of it.
    Plane,
    /// A sphere, bent by the projection's own arithmetic in the vertex stage.
    Globe,
    /// A sphere, bent by a quadratic the producer expanded about the tile's center.
    GlobeAnchored,
    /// A DEM, with the height read per vertex from an elevation texture.
    Terrain,
}

impl Surface {
    /// Every surface, for a caller walking them.
    pub const ALL: [Self; 4] = [Self::Plane, Self::Globe, Self::GlobeAnchored, Self::Terrain];

    /// The suffix a module for this surface is named with, and the empty string for a plane.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Plane => "",
            Self::Globe => "_globe",
            Self::GlobeAnchored => "_globe_anchored",
            Self::Terrain => "_terrain",
        }
    }

    /// The blocks the placement reads, bound after the family's own.
    #[must_use]
    pub const fn blocks(self) -> &'static [&'static UboLayout] {
        match self {
            Self::Plane => &NO_BLOCKS,
            Self::Globe => &GLOBE_BLOCKS,
            Self::GlobeAnchored => &GLOBE_ANCHORED_BLOCKS,
            Self::Terrain => &TERRAIN_BLOCKS,
        }
    }

    /// The textures the placement samples, each binding a `texture_2d<f32>` and a `sampler`.
    #[must_use]
    pub const fn textures(self) -> &'static [&'static str] {
        match self {
            Self::Plane | Self::Globe | Self::GlobeAnchored => &NO_TEXTURES,
            Self::Terrain => &TERRAIN_TEXTURES,
        }
    }

    /// The WGSL defining `place`, which is the whole of what a surface contributes.
    #[must_use]
    pub const fn placement(self) -> &'static str {
        match self {
            Self::Plane => PLANE_PLACEMENT,
            Self::Globe => GLOBE_PLACEMENT,
            Self::GlobeAnchored => GLOBE_ANCHORED_PLACEMENT,
            Self::Terrain => TERRAIN_PLACEMENT,
        }
    }
}

/// Flat Mercator.
///
/// The matrix reaches clip space, the height rides in `z`, and the layer's depth offset is already
/// baked into the matrix at this drawable's own sub-layer.
const PLANE_PLACEMENT: &str = r"
fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    return transform(columns, position);
}

fn displace(at: vec2<f32>, delta: vec2<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    return columns[0] * delta.x + columns[1] * delta.y;
}

// Nothing to hang a curtain over: a plane's tiles are coplanar and their shared edges meet.
fn curtain(flag: f32) -> f32 {
    return 0.0;
}
";

/// A sphere, bent in the vertex stage.
///
/// `tile-local -> normalized Mercator -> sphere -> clip`. The first step is the drawable's own
/// matrix, which the producer fills with the Mercator placement rather than the clip one when the
/// camera is in globe mode; the last is the frame's globe matrix. What is between is
/// `tessella_tile::globe::sphere_point_from_mercator`, which is `camera::latitude_of` and then
/// `globe::sphere_point` with `y` negated.
///
/// # The depth offset arrives in `z`
///
/// Coincident layers are separated by a nudge to the projection. A globe has no per-drawable
/// projection to fold one into — the globe matrix is one matrix for the whole frame — so the
/// producer parks it in the Mercator placement's own translation, where tile geometry never
/// reaches, and the multiply below carries it out in `merc.z`. It is added to clip `z` after the
/// bend, which is where the plane applies it too.
///
/// # No height term
///
/// `position.z` is not read, and that is not an omission: a height above the surface has to be
/// lifted along the sphere's normal, and the coefficient for that — `d_h` — belongs to the
/// anchored bend. The extrusions are the only family that leaves the surface and the producer
/// anchors them, so nothing drawn here has a height to lift.
const GLOBE_PLACEMENT: &str = r"
fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    // `merc.z` comes out as the depth offset: the input z is zero, so the only thing reaching it
    // is the matrix's own translation.
    let merc = transform(columns, vec3<f32>(position.xy, 0.0));

    let longitude = merc.x * 360.0 - 180.0;
    // `camera::latitude_of`. The clamp is `sphere_point_from_mercator`'s: a tile edge running off
    // the top of the world lands on the pole instead of diverging. Bounded either side, so the
    // exponent cannot overflow.
    let fraction = clamp(merc.y, 0.0, 1.0);
    let latitude = degrees(atan(exp(radians(180.0 - fraction * 360.0)))) * 2.0 - 90.0;

    // `globe::sphere_point`, y negated: the convention everything downstream of it assumes, with
    // the compensating flip in `globe::clip_matrix`.
    let lat = radians(latitude);
    let lon = radians(longitude);
    let sphere = vec3<f32>(cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon));

    var clip = transform(globe_camera_ubo[0].globe_matrix, sphere);
    clip.z += merc.z;
    return clip;
}

// Two evaluations and their difference, because this bend has no linear part to extrude along: it
// takes tile-local coordinates through trig to a sphere position. The secant rather than the
// tangent, which for a displacement of a line's width is the same answer.
fn displace(at: vec2<f32>, delta: vec2<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    return place(vec3<f32>(at + delta, 0.0), columns) - place(vec3<f32>(at, 0.0), columns);
}

// A sphere's tiles meet on the sphere, as a plane's meet on the plane. And `place` above reads
// no z at all, so a curtain here would be dropped on the floor rather than drawn.
fn curtain(flag: f32) -> f32 {
    return 0.0;
}
";

/// A sphere, by the quadratic the producer expanded about the tile's center.
///
/// The direct bend above forms a unit-sphere position in `f32` and lets the globe matrix amplify
/// it — and that matrix scales the sphere to 1,663,008 pixels at z14, so one ulp of the position
/// is a fifth of a pixel and the four transcendentals each cost a few of their own. This carries
/// the same function expanded about each tile's center, already in clip space, so the shader adds
/// small to small and does no trig at all:
///
/// ```text
/// clip(du, dv) = anchor + d_u du + d_v dv + (d_uu du^2 + d_vv dv^2) / 2 + d_uv du dv
/// ```
///
/// A quadratic is only as good as the arc its tile subtends, so this is selected per tile above
/// the zoom where that holds and the direct bend keeps everything below it. The crossover is the
/// producer's to pick; both are far under a pixel across a wide band of zooms.
///
/// The drawable's matrix is not read. There is no placement here for a depth offset to be baked
/// into, so the producer puts it in `anchor.z` instead.
const GLOBE_ANCHORED_PLACEMENT: &str = r"
// The tile's center, which is what `du` and `dv` are measured from. Half of the coordinate range
// every tile this consumer sees is produced at.
const TILE_CENTER: f32 = 4096.0;

fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    let bend = globe_bend_ubo[ubo_index];
    // Tile units from the center. Both are small, and every coefficient but the anchor is small,
    // so nothing here is a large number waiting to cancel.
    let d = position.xy - vec2<f32>(TILE_CENTER, TILE_CENTER);

    return bend.anchor
        + bend.d_u * d.x
        + bend.d_v * d.y
        + 0.5 * (bend.d_uu * (d.x * d.x) + bend.d_vv * (d.y * d.y))
        + bend.d_uv * (d.x * d.y)
        // Clip displacement per meter above the surface, along the sphere's normal rather than
        // along the plane's z. Zero for every family but the extrusions.
        + bend.d_h * position.z;
}

// The Jacobian *here*, not at the anchor: differentiating the expansion gives the linear term plus
// the second-order term's contribution, and at low zoom a tile is wide enough that the bend turns
// measurably across it.
fn displace(at: vec2<f32>, delta: vec2<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    let bend = globe_bend_ubo[ubo_index];
    let d = at - vec2<f32>(TILE_CENTER, TILE_CENTER);
    let j_u = bend.d_u + bend.d_uu * d.x + bend.d_uv * d.y;
    let j_v = bend.d_v + bend.d_vv * d.y + bend.d_uv * d.x;
    return j_u * delta.x + j_v * delta.y;
}

// The direct bend's answer, for the direct bend's reason: the tiles meet.
fn curtain(flag: f32) -> f32 {
    return 0.0;
}
";

/// A DEM, with the height read per vertex.
///
/// The elevation is a second texture because a fill or a line is on a vector tile and has no DEM
/// of its own. The producer folded four things into the sampling pair — the DEM's width, its
/// border pixel, the half texel that puts a cell at its own center, and which square of a coarser
/// DEM tile this tile occupies — so reaching a texture coordinate is one multiply-add an axis.
///
/// # Why the height is relative to the camera's center
///
/// The camera sits a fixed distance above the *plane*, while a height in meters reaches the screen
/// multiplied by pixels-per-meter, which doubles with every zoom level. Measured from sea level
/// the ground climbs toward a camera that does not climb with it and eventually passes it, and a
/// camera underground draws the background. Measured from the ground under the camera's center it
/// cannot. Subtracted before the exaggeration rather than after, so stretching the relief leaves
/// the center where it is.
///
/// # The curtain, which a layer on the ground does need
///
/// The ground's own skirt is not here: it is a fifth of a tile deep and belongs to the mesh the
/// ground family draws, which this crate does not have. What is here is the *layer's* curtain,
/// which is a different thing at a different depth and was argued away twice before it was
/// measured.
///
/// Two tiles agree on a shared edge's world position and reach it through different matrices, so
/// the two land a fraction of a pixel apart and the boundary pixels are claimed by neither. On the
/// ground that is a crack; on a picture drawn over raised ground it is the ground showing through.
/// A flagged vertex takes the surface's height and then drops below it, so the layer's own edge
/// hangs down far enough to be behind its neighbor rather than beside it.
///
/// The producer's `seam_at_camera` is the depth -- two pixels of a crack, in meters -- and it
/// arrives in `skirt.z` already zeroed for the cases that do not need it: an unraised terrain,
/// whose tiles are coplanar, and a tile that is not at the edge of the cover. So this reads the
/// uniform rather than deciding anything, and the flag is the producer's own vertex attribute
/// (`tessellaSkirtVertexAttribute`, tessella#331).
///
/// Measured, because the argument went the other way: with the per-layer curtain not emitted,
/// every `gross` row of tessella's parity sweep is identical to the pixel while `terrain_cover_p`
/// counts up to 2382 holes of 1,620,000 at the high-pitch cameras. The gross counter cannot see
/// it -- what shows through a crack is the ground, which takes the background's color.
const TERRAIN_PLACEMENT: &str = r"
fn place(position: vec3<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    let terrain = terrain_drawable_ubo[ubo_index];

    let uv = position.xy * terrain.params.x + terrain.params.yz;
    // Explicit level: a vertex stage has no derivatives to pick one from. The channels come back
    // normalized and the unpack vector is in bytes, which is what scales them back up.
    let channels = textureSampleLevel(elevation, elevation_sampler, uv, 0.0).rgb * 255.0;
    let meters = dot(channels, terrain.unpack.rgb) - terrain.unpack.a;
    let height = (meters - terrain.skirt.y) * terrain.params.w;

    return transform(columns, vec3<f32>(position.xy, position.z + height));
}

// The linear part, with no second sample. A line's extrusion is a line width in tile units and the
// ground does not climb measurably across one, so the height at the extruded point is the height
// already read -- which the displacement does not carry anyway.
fn displace(at: vec2<f32>, delta: vec2<f32>, columns: array<vec4<f32>, 4>) -> vec4<f32> {
    return columns[0] * delta.x + columns[1] * delta.y;
}

// How far below the surface a flagged vertex hangs, in the meters `place` adds to a height. Not
// scaled by the exaggeration: the crack it covers is a fraction of a *pixel* wide however much the
// relief is stretched, and `skirt.z` is already in those terms.
fn curtain(flag: f32) -> f32 {
    return -flag * terrain_drawable_ubo[ubo_index].skirt.z;
}
";
