# CutieRoto — macOS / Apple Metal (MPS) backend

A port of [`004_CPP_Libtorch`](../004_CPP_Libtorch/) that runs the inference
pipeline on **Apple Silicon's GPU via MPS** instead of NVIDIA CUDA. Same node,
same memory core, same disk cache, same geometry and post-process — only the
device backend changes.

> **Status: unverified on hardware.** This was ported by reading the CUDA sources;
> it has **not** been compiled or run on a Mac (the porting environment had no
> Nuke NDK, no models, and no Apple GPU). Treat it as a complete first draft to
> build and debug on your machine. The places most likely to need a tweak are
> called out under [Known risks](#known-risks).

---

## Why MPS and not CPU

Your Mac's Nuke 17 ships a CPU+MPS libtorch (`libtorch_cpu.dylib`, `libc10.dylib`
— there is no `libtorch_cuda` on macOS, and no CUDA). A pure-CPU build is trivial
but far too slow for a stateful per-frame segmenter. MPS runs the convolutions,
attention and memory-read on the Apple GPU, which is the only way to get usable
speed on this hardware. I confirmed the bundled libtorch carries the MPS backend
(its dylib exports `MPSHooksInterface` / `_mps_convolution*` symbols), so the
runtime supports it.

---

## What changed vs. the CUDA build (004)

| | 004 (CUDA) | this (MPS) |
|--|--|--|
| Device | `torch::kCUDA` | `torch::kMPS` |
| Sync | `cudaStreamSynchronize(stream_)` | `torch::mps::synchronize()` |
| Streams | explicit `cudaStream_t` | none (MPS default stream) |
| Availability gate | `torch::cuda::is_available()` | `torch::mps::is_available()` |
| Mem reclaim | `CUDACachingAllocator::emptyCache()` under a `cudaMemGetInfo` watermark | `torch::mps::empty_cache()` once per frame |
| Autocast | fp16 on `at::kCUDA` | **off** (fp32) by default |
| Models | embedded via GNU `objcopy` | **loaded from disk** at runtime (no objcopy on macOS) |
| Build | GCC, `.so`, CUDA toolkit + libtorch_cuda | clang/libc++, `.dylib`, no CUDA |
| `memory_core.*` | — identical — | — identical — |

`memory_core.cpp` needed **zero** changes: it's pure `torch::Tensor` math that
follows whatever device its tensors live on. That's the reason this port is
mechanical rather than a rewrite.

---

## Prerequisites

1. **Nuke 17.0** installed (provides the libtorch 2.7.1 dylibs and the NDK
   headers under `…/Nuke17.0v1.app/Contents/MacOS/include`).
2. **CMake ≥ 3.18** and the **Xcode command-line tools** (`xcode-select --install`).
3. **libtorch 2.7.1 headers**, version-matched to Nuke's dylibs. Easiest:
   ```bash
   python3 -m pip install torch==2.7.1
   python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"include"))'
   ```
   (We use these **headers only**; the runtime dylibs come from Nuke. The versions
   must match — 2.7.1 — or you'll get ABI surprises at load.)
4. **The 5 traced models** (see next section). They are *not* in the repo.

---

## Step 1 — Get the 5 `.pt` models (device-neutral)

The models aren't committed (`models/put_models_here.txt` is a placeholder, and I
found no `.onnx`/`.pt` anywhere in your folders). The CUDA trace script
([`003/trace_all_stages.py`](../003_CPP_Libtorch/trace_all_stages.py)) traces on
CUDA, which won't run on a Mac. Use the included **CPU trace** instead:

```bash
# in the cutie env, alongside the export_*.py wrappers from 001_PREP
python trace_all_stages_mps.py
# -> encode_image.pt  transform_key.pt  mask_encoder.pt  mask_decoder.pt
```

With your existing `fusion_transformer.pt` (from `001_PREP`) that's all five. Put
them where the plugin looks (first match wins):

1. `$CUTIE_MODEL_DIR`
2. `models/` next to `CutieRoto.dylib`
3. `~/.nuke/CutieRoto/models/`

```bash
mkdir -p ~/.nuke/CutieRoto/models
cp encode_image.pt transform_key.pt mask_encoder.pt mask_decoder.pt \
   fusion_transformer.pt  ~/.nuke/CutieRoto/models/
```

> **About your ONNX file.** ONNX can't be loaded by `TorchEngine` (it loads
> TorchScript `.pt`, not ONNX). It's still useful as a *source*: you can convert
> ONNX → TorchScript, or — if you'd rather not depend on libtorch at all — run the
> ONNX graphs through **onnxruntime with the CoreML execution provider**, which is
> a different (and also Metal-accelerated) backend. That would be a separate runner
> from this one. For this build, the `.pt` route above is the path of least
> resistance. If you locate the ONNX file, share it and I can advise on converting
> or wiring an onnxruntime/CoreML variant.

---

## Step 2 — Build

```bash
cd 005_CPP_Libtorch_MPS

cmake -B build \
  -DNUKE_MACOS="/Applications/Nuke17.0v1/Nuke17.0v1.app/Contents/MacOS" \
  -DTORCH_INCLUDE_ROOT="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"include"))')"

cmake --build build -j --target CutieRoto
# -> build/CutieRoto.dylib
```

Adjust `-DNUKE_MACOS=` to your actual install path. The CMake will stop early with
a clear message if it can't find `libtorch.dylib`, the NDK headers, or
`torch/script.h`.

Install it:

```bash
mkdir -p ~/.nuke
cp build/CutieRoto.dylib ~/.nuke/
# (ensure ~/.nuke is on the Nuke plugin path, which it is by default)
```

> If your Nuke build expects plugins named `.so` even on macOS, rename it:
> `cp build/CutieRoto.dylib ~/.nuke/CutieRoto.so`. Modern Nuke macOS uses
> `.dylib`; try that first.

---

## Step 3 — Run

Launch Nuke **from a terminal** so you see the `[CutieRoto] …` trace:

```bash
/Applications/Nuke17.0v1/Nuke17.0v1.app/Contents/MacOS/Nuke17.0 &
```

Node graph: plate → input 0, an animated Roto/RotoPaint → input 1. Set
`Keyframes` (e.g. `1,22,48,96`) and a `Range`, press **Process**. A healthy run
prints `buildPipeline: MPS ok`, then `analyzed keyframe …`, then per-frame
`displayStep …` lines, ending in `processAllFrames: DONE`.

If anything misbehaves, the first run is the time to set the fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 /Applications/Nuke17.0v1/Nuke17.0v1.app/Contents/MacOS/Nuke17.0
```

This routes any op that lacks an MPS kernel to CPU instead of erroring — slower
for that op, but it gets you a working matte while you see (in the trace) which op
fell back.

---

## Known risks

These are the spots I couldn't verify without a Mac. None are deep; each has a
stated workaround.

1. **MPS op coverage.** The pipeline uses convs, `scaled_dot_product_attention`
   (the fusion transformer), `interpolate` (`area`/`nearest`/`bilinear`), `topk`,
   `scatter_`, `bmm`/`matmul`. All are commonly supported in torch 2.7's MPS
   backend, but if one isn't, you'll get a clear "not implemented for MPS" error
   naming the op. **Fix:** launch with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

2. **`kArea` interpolate (plate downscale).** Area mode maps to adaptive-avg-pool;
   it's the op most likely to be missing on MPS. If `pullInputResized` errors on
   it, either use the fallback env var, or switch the plate branch from
   `torch::kArea` to `torch::kBilinear` (a one-line change in `CutieRoto.cpp`,
   marked with a NOTE; negligible matte difference).

3. **MPS cache reclaim symbol.** The per-frame reclaim uses
   `at::detail::getMPSHooks().emptyCache()` — the C++ entry point (the
   `torch::mps::empty_cache()` you'd guess from Python does **not** exist in C++; I
   verified both against real libtorch headers). If a future libtorch moves it and
   the link fails naming that symbol, just comment the single call at the end of
   `displayStep()` — it's only a memory hint; the allocator still reclaims.

4. **fusion_transformer.pt traced on CUDA.** It should still load on MPS, but if
   `torch::jit::load(..., kMPS)` throws for it, re-trace it on CPU (same idea as
   `trace_all_stages_mps.py`).

5. **fp32 vs fp16.** I left autocast **off** for a correct first build. Once it
   runs, you can experiment with `at::kMPS` autocast (constructor flag in
   `TorchEngine`) for speed — but verify the mattes don't degrade first.

6. **Plugin extension / rpath.** If Nuke loads the plugin but can't resolve torch
   symbols, check `otool -L build/CutieRoto.dylib` — the torch/DDImage entries
   should resolve to the Nuke `Contents/MacOS` dir (that's what the rpath is for).

---

## Files

| File | What it is |
|------|------------|
| `src/CutieRoto.cpp` | the Nuke node — MPS-ported (device, sync, mem reclaim) |
| `include/CutieRoto.h` | node header — MPS-only, TRT/CUDA refs removed |
| `src/torch_engine.cpp`, `include/torch_engine.h` | `.pt` runner on MPS (no stream arg) |
| `src/memory_core.cpp`, `include/memory_core.h` | fp32 memory read — **unchanged** |
| `include/embedded_assets.h` | runtime disk-loader for the 5 `.pt` (no objcopy) |
| `CMakeLists.txt` | macOS / clang / MPS build (no CUDA, no TensorRT) |
| `trace_all_stages_mps.py` | CPU (device-neutral) trace of the 4 stages |

---

*CutieRoto port © Peter Mercell, 2026 — [petermercell.com](https://petermercell.com).
Built on Cutie (Ho Kei Cheng et al., MIT). MPS port scaffolding.*
