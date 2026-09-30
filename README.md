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

tessella's plan of record (DR-14) says the map draws "at entity/HAL level" inside emblema. That
layer does not exist in the shape DR-14 assumes, and building it would break emblema's own
recorded scope:

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
canvas composites the result — and the *mechanism* moves here. emblema is asked for four small
seams instead of a map renderer, and its HAL trait is untouched. This is decision **TI-1** of the
integration plan.

## What it needs from emblema, and does not have yet

One thing is required and absent: a way for a foreign pass to declare what image layout it left
an image in. emblema's `set_layout` is crate-private, and sampling transitions *from the tracked
layout* — so an image emblema created (tracked `UNDEFINED`) and this crate rendered into would be
transitioned from `UNDEFINED` at first sample, which permits the driver to discard the map's
pixels.

That is **IR-1**, `VulkanTexture::assume_layout`. Until it lands there is no correct hand-off, and
its test has to run on a tiler: on lavapipe the discard does not show.

## Status

Scaffolding. What is here is the repository, its pins, and the device-side decisions that can be
made and tested without a GPU, a stream, or IR-1 — chiefly format selection, which is an
obligation this crate inherits directly by owning its own depth-stencil attachment.

Not here yet, in dependency order: `tessella-consume` (the reader/draw-list/stencil-partition
half, ported once to Rust and living in the tessella workspace — decision TI-7), the geometry and
uniform stores, the shader modules, and the hand-off that waits on IR-1.

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
