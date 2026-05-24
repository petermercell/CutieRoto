# CutieRoto — Build Guide, Part 2: The C++ Nuke Plugin

Turning the 5 model assets from [`001_PREP`](../001_PREP/) into a native Nuke 17
plugin: `CutieRoto.so`.

This is the harder half. Cutie is *stateful* — it carries memory forward frame to
frame — so the plugin can't be a stateless per-frame filter. The architecture:

- **A TensorRT engine runner** that drives the 4 engines (E1, E2, E3, E5).
- **A C++ memory core** — the similarity → top-k softmax → readout, reimplemented in
  libtorch C++, bit-exact to the Python reference.
- **The traced `fusion_transformer.pt`** run via libtorch (`torch::jit`).
- **A stateful Nuke node** that, on a Process button, pulls each keyframe + the plate
  at arbitrary frames, runs the recurrent propagation across the clip, and caches
  every frame's matte to disk for instant scrubbing.

The whole thing — 4 engines + the `.pt` — is **embedded into the `.so`** via
`objcopy`, and **TensorRT is statically linked** so deployment needs only Nuke 17 +
an NVIDIA driver.

> This guide follows the real build order, smallest provable piece first. Each stage
> has a standalone test that validates it **before** Nuke is involved — that's how
> you keep a port like this debuggable. Don't skip the tests; they're the reason the
> final plugin worked on roughly the first try in Nuke.

---

## The linkage problem (read this first)

The single hardest thing about this port isn't the model — it's getting **TensorRT
and libtorch to coexist in one process.** Both want CUDA. Get it wrong and you get a
plugin that links but crashes, or corrupts memory subtly. The rules that make it
work, learned the hard way:

1. **Use Nuke's libtorch, not your own.** Nuke 17.0 ships libtorch 2.7.1. Link those
   exact `.so`s (`/opt/Nuke17.0v1/libtorch*.so`, `libc10*.so`). Headers come from a
   matching conda torch 2.7.1.
2. **Match the C++ ABI.** Nuke's libtorch is built with the new ABI →
   `-D_GLIBCXX_USE_CXX11_ABI=1`. Mismatch = links-but-crashes with corrupted
   `std::string`. (Check with `nm` for `cxx11` symbols if unsure.)
3. **One CUDA runtime, shared.** TensorRT links against the *same* dynamic
   `libcudart.so` that Nuke's libtorch already loads. Do **not** introduce a second
   (e.g. static cudart) — two CUDA runtimes in one process clash. (More on this in
   the static-build section.)
4. **One CUDA stream, shared.** The plugin creates one stream; every engine
   (`enqueueV3`) and every libtorch op orders on it. Per-engine streams = cross-stream
   races that randomly black frames.

Before writing any real code, prove these hold with a 30-line smoke test
([`linkage_smoke.cpp`](linkage_smoke.cpp)): load the `.pt` via libtorch **and**
deserialize an engine via TensorRT, in one binary. If that runs, the port is viable.

```bash
cmake -B build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build build -j --target linkage_smoke
./build/linkage_smoke fusion_transformer.pt encode_image.fp16.engine
# expect: "SUCCESS: libtorch + TRT + CUDA coexist"
```

---

## Stage 1 — The TensorRT engine runner

[`trt_engine.h`](include/trt_engine.h) / [`trt_engine.cpp`](src/trt_engine.cpp)

A small `TRTEngine` class wraps one engine: deserialize from a memory buffer, then
`run(map<name, torch::Tensor>, stream)` binds input tensors by name (setting input
shapes on the dynamic engines E1/E2), allocates fp32 output `torch::Tensor`s at the
context-resolved shapes, binds their `data_ptr()` to TensorRT, and `enqueueV3` on the
shared stream. **All engine IO is fp32** (even the fp16 engines — fp16 is internal
only), so the boundaries are simple.

The key design choice: outputs are `torch::Tensor`s, not raw `cudaMalloc`. That means
an engine's output feeds directly as input to the next engine *and* to the libtorch
`.pt` and the memory read, all sharing libtorch's allocator and the one stream — no
copies, no manual buffer juggling.

**Test it standalone** ([`test_engines.cpp`](src/tests/test_engines.cpp)): chain
E1 → E2 → E3 → `.pt` → E5 on dummy tensors, verify every handoff shape against the IO
map. No Nuke.

```bash
cmake --build build -j --target test_engines
./build/test_engines models      # dir holding the engines + .pt
# expect: ALL SHAPES OK — runner + chain validated
```

The handoff map this verifies (all fp32):

```
E1 encode_image:  image(1,3,H,W) -> f16, f8, f4, pix_feat
E2 transform_key: f16            -> key, shrinkage, selection
E3 mask_encoder:  image, pix_feat, sensory, masks -> mask_value, new_sensory
.pt fusion+xfmr:  pix_feat, visual_readout, sensory, last_mask, obj_memory
                                  -> readout_memory  (feeds E5.memory_readout)
E5 mask_decoder:  f8, f4, memory_readout, sensory  -> new_sensory, prob4
```

---

## Stage 2 — The C++ memory core (the genuinely custom part)

[`memory_core.h`](include/memory_core.h) / [`memory_core.cpp`](src/memory_core.cpp)

This replaces Cutie's `MemoryManager` with two pieces in libtorch C++:

**`KVStore`** — three `torch::Tensor` buffers (key / shrinkage / value) laid out
`[permanent | temporary]` along the token axis, with `perm_end_pt`. `add(permanent?)`
appends (keyframes are permanent); `removeOldMemory()` FIFO-trims temporary tokens
past the cap (`max_mem_frames × HW = 5 × 8160 = 40800`). With `use_long_term=False`
there are no prototypes / usage scoring / compression — this is the whole store.

**`memoryRead`** — the fp32 transcription of Cutie's `get_similarity` (anisotropic L2
with the `selection`/`qe` term) → `do_softmax(top_k=30)` → `readout` (bmm). Pure
`torch::` ops, no custom CUDA, run in true fp32 (more accurate than Cutie's native
fp16 read).

> **Numerical-stability note (important):** the top-k softmax must subtract the
> per-column max before `exp` — `exp(v - max) / Σexp(v - max)`. Mathematically
> identical to plain `exp/Σ`, but without it, widely-spaced keyframes (e.g. roto on
> frame 1 and 96 only) produce large similarity values mid-clip, `exp` overflows to
> Inf, and you get NaN partway through. Cutie's fp16 path mostly hides this; the
> fp32 C++ path needs the max-subtraction explicitly.

**Validate bit-exact against Python.** `001_PREP`'s `memory_read_reference.py` is the
spec. First produce a test fixture: `001_PREP`'s `dump_memory_read_case.py` runs a
real Cutie propagation, snapshots the KV state (k/s/v + `perm_end_pt`) and the query
key/selection at a chosen frame, computes the fp32 reference read, and saves it all
to `memcase.pt`. Then the C++ loads that fixture, runs its own `KVStore` +
`memoryRead` on the identical inputs, and compares.

```bash
# in 001_PREP (env cutie311), make the fixture:
python dump_memory_read_case.py \
    --frames /path/to/plate.%04d.exr --roto /path/to/roto.%04d.exr \
    --seeds 1,48,96 --check-frame 30 --out memcase.pt
cp memcase.pt /path/to/002_CPP/test_data/

# then in 002_CPP, build + run the C++ test:
cmake --build build -j --target test_memory
./build/test_memory test_data/memcase.pt
# expect: affinity / visual_readout / memoryRead  max_abs_err ~0  (float32 epsilon)
#         FIFO OK ... MEMORY CORE OK
```

`~1e-6` error is float32 epsilon (the stable-softmax max-subtraction perturbs the
last bits) — completely negligible and correct.

---

## Stage 3 — Embed the assets into the `.so`

[`embedded_assets.h`](include/embedded_assets.h)

The 4 engines + the `.pt` are baked into the plugin with `objcopy`: each file →
`stem.bin` → `stem.o` with `_binary_<stem>_bin_start/_end` symbols, listed as link
sources. The C++ constructs `TRTEngine` from the embedded byte range, and loads the
`.pt` from an in-memory `std::istringstream` via `torch::jit::load(stream, kCUDA)` —
identical to a disk load, only the byte source differs.

CMake generalizes the single-blob recipe to five via a `foreach` over stems
(`e1 e2 e3 e5 ft`). **Test that all five load from the linked `.so`:**

```bash
cmake --build build -j --target test_embedded
./build/test_embedded
# expect: 4 engines deserialized from embedded bytes OK
#         fusion_transformer.pt loaded from embedded stream OK
```

---

## Stage 4 — The stateful Nuke node

[`CutieRoto.h`](include/CutieRoto.h) / [`CutieRoto.cpp`](src/CutieRoto.cpp)

`CutieRoto : public Iop`. The pieces:

**Inputs.** 0 = plate (RGBA), 1 = a single animated Roto/RotoPaint. The artist keys
the roto on a few frames; the node reads it at each listed frame.

**Knobs.** `Keyframes` (e.g. `1,22,48,58,62,96`), `Range`, a Cache Dir, a **Process**
button, post-process Gain/Gamma/Clamp, Matte Only, Invert.

**The hard NDK mechanic: arbitrary-frame input pulls.** A node normally only sees the
frame Nuke is currently rendering. To read input *k* at frame *f*, build an
`OutputContext`, `setFrame(f)`, and `inputAt(k, ctx)` — that gives the input op
evaluated at *f*. This is how a single animated Roto serves a different shape at each
keyframe, and how the sequential Process loop walks the clip in order. (Stateful
temporal nodes are rare in Nuke precisely because this isn't the default model.)

**Process (the two-phase loop).** On the Process button (`knob_changed`):
1. *Analyze* — for each keyframe frame, pull plate + roto via `inputAt`, run
   E1 → E2 → E3, add to the store as **permanent** memory.
2. *Display* — for each frame in range, in order: pull plate, run
   E1 → E2 → memory read → `.pt` → E5 → matte; carry sensory + last-mask state
   forward; every `mem_every=5` frames add a temporary memory + FIFO trim. At
   keyframe frames, pin the output to the artist's roto (a keyframe should *be* the
   artist's mask, not a prediction).
   A modal progress bar (`progressFraction`/`progressMessage`) with `aborted()`
   cancellation drives the UI.

**Disk cache.** Each propagated matte is written to
`<cacheDir>/matte.######.raw` (a flat header + float dump — zero-dependency, fast).
`engine()` serves the current frame from RAM, falling back to the disk file. Disk is
the source of truth, so any node instance / viewer pipe reads the same result
(this also sidesteps the multiple-instance cache flicker you'd otherwise hit), and it
persists across sessions.

**Geometry — match the validated Python exactly.** Plate → resize to 1920×1080
(area for plate, nearest for roto) → **pad** the height to 1088 (8 rows, no stretch)
→ engine at 1088×1920. On output, **crop** the 8 padded rows, then resize to plate.
Stretching straight to 1088×1920 instead distorts the aspect ratio and softens the
matte everywhere — a subtle bug worth getting right.

Build the plugin and load it:

```bash
cmake --build build -j --target CutieRoto
cp build/CutieRoto.so ~/.nuke/CPP/17.0/    # or your NUKE_PATH
# launch Nuke FROM a terminal to see the plugin's stderr
```

In Nuke: plate → input 0, an animated Roto → input 1, set `Keyframes`, hit Process,
scrub.

---

## Stage 5 — The static-TensorRT build (zero-install deployment)

The goal: artists have Nuke 17 (libtorch + CUDA) but **not** TensorRT. So static-link
*only* TensorRT, keep CUDA dynamic and shared with Nuke's libtorch.

Enable it with a CMake flag (default off keeps the proven shared build available):

```bash
cmake -B build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCUTIE_STATIC_TRT=ON
cmake --build build -j --target CutieRoto
ldd build/CutieRoto.so | grep -i nvinfer   # expect: nothing (TRT is static)
```

The recipe (in [`CMakeLists.txt`](CMakeLists.txt)):

```cmake
-Wl,--start-group
  libnvinfer_static.a
  libnvptxcompiler_static.a        # static TRT's NVPTX serializer needs this
-Wl,--end-group
CUDA::cudart                       # DYNAMIC — shared with Nuke's libtorch
nvrtc (dynamic)  nvJitLink (dynamic)
dl rt pthread
```

The crucial point: **cudart stays dynamic.** Static-linking cudart too would put a
second CUDA runtime in the process and clash with libtorch. The static TRT archive's
CUDA references resolve against the same `libcudart.so` Nuke already loads. The
companion libs (`nvrtc`, `nvJitLink`, `nvptxcompiler`) are what TRT-static needs for
JIT kernel work — link errors name each one in turn if missing.

Result: an ~800MB `.so` whose `ldd` shows `libcudart`/`libnvrtc`/`libnvJitLink`
resolving from Nuke, and **no `libnvinfer.so`**. Drop it in `.nuke/CPP/17.0/` — needs
only Nuke 17 + the NVIDIA driver.

Verify static-TRT + dynamic-libtorch coexist at runtime before trusting it in Nuke —
`test_embedded` does exactly that (deserialize engines via static TRT + load the `.pt`
via libtorch in one binary, one dynamic cudart):

```bash
./build/test_embedded   # close Nuke first so the GPU is free
```

---

## Source layout (`002_CPP/`)

```
include/
  trt_engine.h         TRTEngine runner
  memory_core.h        KVStore + memoryRead
  embedded_assets.h    extern decls for the 5 objcopy blobs
  CutieRoto.h          the Iop node
src/
  trt_engine.cpp
  memory_core.cpp
  CutieRoto.cpp        the node (inputs, knobs, Process loop, disk cache, engine())
  tests/
    test_engines.cpp   Stage 1 — 4-engine chain
    test_memory.cpp    Stage 2 — bit-exact memory core
    test_embedded.cpp  Stage 3 — embedded-asset load
models/                the 5 assets from 001_PREP (embedded at build time)
CMakeLists.txt
```

---

## The order that worked

1. `linkage_smoke` — torch + TRT + CUDA coexist? → proves the port is possible.
2. `test_engines` — the 4-engine chain runs with correct shapes.
3. `test_memory` — the memory core is bit-exact vs Python.
4. `test_embedded` — all 5 assets load from the linked `.so`.
5. `CutieRoto.so` — assemble the node from proven parts.
6. `-DCUTIE_STATIC_TRT=ON` — static TRT for zero-install deployment.

Every step is provable before the next. That discipline is why the Nuke integration
came together quickly instead of being an undebuggable black box.

---

*CutieRoto port © Peter Mercell, 2026 — [petermercell.com](https://petermercell.com).
Built on Cutie (Ho Kei Cheng et al., MIT).*
