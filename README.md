# tessella-emblema

The map pass of [tessella]'s capture stream, drawn with Vulkan on [emblema]'s device.

tessella produces a stream of envelopes — geometry, uniform blocks, textures, draw order — and
does not draw. This crate consumes that stream and draws the map with raw Vulkan, into an image
emblema owns. emblema then composites a HUD over it and presents, through its WSI target or
straight to KMS.

It is the second consumer, beside the Filament mirror `tessella_fluorite`. Two independent
consumers of one ABI is the instrument: where both agree with `mbgl-render` and not with each
other, one of them reads the ABI wrong.

## Why raw Vulkan, and not emblema's HAL

tessella's DR-14 says the map draws "at entity/HAL level" inside emblema. That layer does not
exist in the shape DR-14 assumes, and building it would break emblema's own recorded scope:

- emblema's entity layer is coverage-only and nothing routes through it.
- Its HAL is a *batch* HAL — one fixed vertex, a closed material enum, fragment-only runtime
  programs, every submit re-uploading all geometry, no buffer API, no storage buffers, no depth
  attachment, whole-texture writes only.
- Its architecture puts retained mode and 3D out of scope, and pins every vertex's depth to zero
  as a load-bearing invariant.

A map pass needs the opposite of most of that: retained device-local geometry keyed by
`GeometryId`, a vertex layout and vertex stage per family, a consolidated storage buffer per
(view, layer), depth for extrusions and the opaque pass, a stencil it owns for tile clipping, and
sub-rect texture updates.

So the *intent* of DR-14 is kept — never canvas-level, pipelines compiled ahead of time, the
canvas composites the result — and the *mechanism* moves here. emblema is asked for a few small
seams instead of a map renderer, and its HAL trait is untouched.

## What it needs from emblema, and does not have yet

One thing is required and absent: a way for a foreign pass to declare what image layout it left
an image in. emblema's `set_layout` is crate-private, and sampling transitions *from the tracked
layout* — so an image emblema created (tracked `UNDEFINED`) and this crate rendered into would be
transitioned from `UNDEFINED` at first sample, which permits the driver to discard the map's
pixels.

What it needs is a `VulkanTexture::assume_layout`. Until that lands there is no correct hand-off,
and its test has to run on a tiler: on lavapipe the discard does not show.

## Whether naga can emit specialization constants

The answer decides the *form* of the permutation mechanism, not whether it happens. `tests/naga_overrides.rs` answers it against
naga 23 and pins the answer so it cannot go stale:

- The SPIR-V backend **refuses** a module that still carries an override — `Error::Override`, not
  a silent resolution. So the literal form, where the driver binds specialization constants, is
  unavailable.
- `process_overrides` substitutes a value and emits, and **does not fold the branch**: the module
  comes out with one boolean `OpConstant` and one `OpBranchConditional` reading it.

That second point is the one that matters, and it is the opposite of what the plan feared. There
is a constant to patch and a branch for it to steer, so the fallback works as written: substitute,
then rewrite that `OpConstantFalse` into `OpSpecConstantFalse` with a `SpecId` decoration, and let
the driver fold the branch and the dead arm at pipeline creation. One module per (family, surface),
permutations as pipeline-creation arguments, nothing translated at run time, and no module-count
explosion.

## Status

The shader half is done; the frame half is not started.

What is here: eighteen shader families, transcribed from `include/mbgl/shaders/vulkan/*.hpp` —
the Vulkan backend's text, which is what the ABI's tables are generated from and which differs
from the GL shaders in places that matter. Each is assembled against one of four surfaces, giving
57 modules that compile to SPIR-V. Plus the uniform-block declarations generated from the same
tables, the permutation mechanism, and the device-side decisions that can be made without a GPU.

How that is checked, in two ways, because reading a shader cannot catch what reading cannot see:

- `tests/shaders.rs` assembles all 57 pairs, compiles each, and pins the decisions that are
  silent when wrong — a wrong one of those still draws, which is why they are pinned.
- `benches/first_pixel.rs` runs them. Twenty-one cases pick inputs that make one pixel
  predictable, derive it from mbgl's arithmetic by hand, draw, and read it back. A bench rather
  than a test because it needs a GPU and CI has none. It picks a device external first, then
  internal, then software.

Every family has at least one pixel behind it. The readback has caught defects the pins could
not, including an outline's feather mirrored by a coordinate flip naga applies *after* a body
runs, and it has corrected several comments that were confidently wrong.

Three parts of the frame half are now here. `store` creates a geometry's buffers, binds them into
one allocation and writes its bytes. `blocks` gives each `(view, layer)` its consolidated storage
buffer and brings it level from `uniforms`' shadow a dirty range at a time. And `textures` decides a
texture's `VkFormat` from both halves of mbgl's two-part format, and packs an update's regions into
one staging buffer at offsets `vkCmdCopyBufferToImage` will accept.

Still to come, in dependency order: the texture store's device half -- images, views and the
recorded copy, which is the first part of this crate to need a command buffer -- then the pipelines
these modules become, and the hand-off. `tessella-consume` (the reader, draw list and stencil
partition) lives once in the tessella workspace and now carries a texture update's channel type and
payload shape, which this crate's format and staging decisions both read.

`#![forbid(unsafe_code)]` is still at the top of `lib.rs`, and the store does not contradict it: the
`unsafe` lives in `crates/tessella-vk`, a thin safe layer over the Vulkan calls the map pass makes.
Every object there borrows the device it was made from, so the compiler refuses one that outlives it,
and a write through a mapping is bounds-checked. See that crate's own docs for what "thin" excludes --
it chooses no suballocator and no staging policy, because both want a measurement and the pass is what
can take one.

## Targets

The Raspberry Pi 5 (BCM2712, V3D 7.1, Mesa `v3dv`) gates every phase — not a lane, the target.
Desktop and lavapipe are development. RK3588 and SA8155P follow. The Pi 4 is out of scope: it
renders through `v3dv`, but its `vc4` cannot import what Vulkan allocates, so it cannot scan out
the result.

Two board facts shape the code rather than merely inform it. V3D has `D24_UNORM_S8_UINT` and not
`D32_SFLOAT_S8_UINT`, so every format this crate uses is queried and a miss is an error at device
creation rather than a wrong picture. And a fragment shader on V3D costs what its whole body
needs, not what the taken branch needs — so a permutation's unused work has to be folded out at
pipeline creation, never branched around at run time.

## License

BSD-2-Clause. Shader bodies ported from `tessella_fluorite` carry its Apache-2.0 provenance and,
through it, MapLibre's BSD-2 notice; each such file says so in its header.

[tessella]: https://github.com/jwinarske/tessella
[emblema]: https://github.com/jwinarske/emblema
