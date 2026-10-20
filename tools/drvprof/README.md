# Two drivers for one Adreno 640

`drvprof.c` measures what a Vulkan driver costs, in pieces small enough to attribute: shader
compilation, pipeline creation, command recording, submit latency, and four GPU-bound passes.
It exists because the SA8155P has **two** drivers for the same part -- Qualcomm's
`libvulkan_adreno.so` and Mesa's turnip -- and the interesting question is not whether either
works but where each is weak.

It uses a `VkRenderPass` rather than dynamic rendering, which is the whole point: the blob has
neither the Vulkan 1.3 feature nor `VK_KHR_dynamic_rendering`, so nothing in `benches/` can open
a device on it. One binary and one loader run both drivers, so the driver is the only variable.

## Building and running

No Rust, no crate, nothing in the workspace: it is one C file and five shaders. Cross-compile it
the way the baseline records for this board, push it with the `.spv` files beside it, then

    export LD_LIBRARY_PATH=/tmp/turnip          # the loader built for the board
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/adreno-vk.json ./drvprof all   # the blob
    VK_ICD_FILENAMES=/tmp/turnip/freedreno_icd.json  ./drvprof all   # turnip
    MESA_SHADER_CACHE_DISABLE=true VK_ICD_FILENAMES=.../freedreno_icd.json ./drvprof all

`all`, or one of `pipeline`, `record`, `submit`, `fill`, `sample`, `vertex`.

## The shaders

Five WGSL files through naga, each with **one** entry point, because the blob refuses a module
carrying two -- and no matrices, because it refuses a `mat4x4` read from a storage buffer. The
`.spv` beside each one is naga 23.1.0 output, `--release`, which is what the crate emits.

| module | what it is for |
|---|---|
| `vs-quad` | one oversized triangle covering the target, no vertex buffer |
| `vs-degenerate` | one multiply a vertex and a shared position, so vertex work has no fragments |
| `fs-const` | a constant color: fill rate and the pass's own cost |
| `fs-alu` | 256 dependent multiply-adds a fragment, seeded from the position |
| `fs-alu-flat` | the same arithmetic written out, so the pair says what the loop costs |
| `fs-sq` | 256 dependent **squarings**, which no compiler can fold |
| `fs-sq-flat` | the same, written out: the pair that measures arithmetic rather than an optimizer |
| `fs-tex` | sixteen samples a fragment at coordinates that move with the pixel |

**Why two pairs.** `acc * k + c` repeated is affine and collapses to a single multiply-add, and
Qualcomm's compiler does collapse it -- `fs-alu-flat` returns that driver's constant-color cost
exactly. So the `fs-alu` pair measures an optimizer. `acc * acc + c` does not collapse, which is
what makes `fs-sq-flat` the ALU-throughput number: the two drivers come within 1.7% on it, where
the rolled form differs by 46%.

**Reading ir3's own numbers.** `IR3_SHADER_DEBUG=disasm` prints a `; FRAG prog N/M: U instr, U
nops, ...` line per shader, which is how the loop story was settled: ir3 unrolls `fs-tex`'s
sixteen iterations (158 instructions, 0 loops) and leaves 256 rolled (30 instructions, half of
them nops, 1 loop). `TU_DEBUG=shaderdb` does not exist in current Mesa; this is the flag.

## Reading the output

Every figure is a p50 over fifteen repeats, and the GPU-side ones also carry a `min`, because
this board's load average sits above ten and its GPU clock ramps: a full-screen fragment pass
measured 140 us in one process and 73 us in the next, from the same binary. Compare runs taken
back to back, and prefer the `min` -- `p50` on a contended board is a ceiling, not a figure.

Pipeline creation is reported **cold and warm separately** for the same reason it has to be:
Mesa caches compiled shaders in process and on disk, so the second create of one shader is a
cache hit rather than a compile, and a median over a loop would report the cache.

## Timing this crate's own modules

The five shaders here are 88 to 358 words; a family in `shaders.rs` is one to three thousand, and
the drivers do not scale the same way. `drvprof module <vs.spv> <fs.spv> [more pairs]` times
pipeline creation for real module pairs against a layout wide enough for any family -- six storage
buffers, four sampled images, four samplers, which the widest family (six, three and three) fits
inside. A layout wider than the module is valid; one that does not cover it is invalid usage, and
some drivers answer that with a segfault.

The modules have to be **per stage**, one entry point each, because that is what the Adreno blob
requires -- the same reason `pipelines::Stages` splits them. naga will emit one if asked:

    naga::back::spv::write_vec(&module, &info, &options, Some(&PipelineOptions {
        shader_stage: naga::ShaderStage::Vertex, entry_point: "vertex_main".into(),
    }))

They are not checked in: a dump goes stale the moment a body changes, and the word counts beside
each figure in the baseline are what make the numbers readable.

The numbers, and what they say about each driver, are in `../../tests/bench-baselines/sa8155p.txt`.

## The cache question, and why it needs its own subcommand

The table in the baseline was taken with `pipelineCache` as `VK_NULL_HANDLE`, which asks a
driver to compile and tells it nothing may be kept. Read naively it makes the blob look like it
has no cache. It does not mean that: **RADV behaves identically** under the same conditions --
`MESA_SHADER_CACHE_DISABLE=true`, no cache object -- recompiling `fs-alu` in 534 us cold and
453 us on every repeat. A driver owes nothing without a cache object.

So `drvprof cache [path]` asks the question an application actually has:

    ./drvprof cache ./pipeline.cache     # first run: empty cache, compiles, writes the file
    ./drvprof cache ./pipeline.cache     # second run: loads it, and the creates should be cheap

Three states are distinguishable, and on RADV with its own caches disabled all three show:

| state | fs-const | fs-alu | fs-tex |
|---|---|---|---|
| no cache object, every create | 6701 / 296 p50 | 534 / 453 p50 | 1024 / 937 p50 |
| cache object, empty: first create | 1216 | 550 | 949 |
| the same object, repeats | 2.1 | 2.1 | 2.1 |
| cache object preloaded from the file: first create | 45.1 | 4.0 | 4.7 |

Microseconds. The second row is a compile, the third is the in-process cache, and the fourth is
the serialized cache surviving a process boundary -- which is the one worth knowing per driver,
because it is what a product can ship. Run it on both drivers before concluding anything about
startup cost.
