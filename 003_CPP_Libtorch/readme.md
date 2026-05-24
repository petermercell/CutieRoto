# CutieRoto — Build Guide, Part 3: The Pure-libtorch Backend (no TensorRT)

The easy-mode build. Same plugin, same pipeline — but the 4 inference stages run as
traced TorchScript `.pt` modules on Nuke's bundled libtorch instead of on TensorRT
engines. **No TensorRT. No engine building. No CUDA toolkit.** Just Nuke 17 and an
NVIDIA driver.

> **Why this exists.** The TRT build ([`002_CPP`](../002_CPP/)) is fast, but the TRT
> half is also the entire reason it's hard: per-GPU engine generation, the static
> link, an ~800MB `.so`. This backend trades some speed for a build any Nuke user can
> do in minutes — and a 232MB plugin that drops straight in. For a "roto a few
> keyframes, Process the clip, scrub the cache" tool, that trade is usually worth it.

This guide assumes you've read [`002_CPP`](../002_CPP/) — the node, memory core, disk
cache, and geometry are **identical and unchanged**. Only the inference backend
differs, and that's all this part covers.

---

## What changes vs. the TRT build

| | TRT build (`002_CPP`) | libtorch build (this) |
|--|----|----|
| Inference stages | 4 TensorRT engines | 4 traced `.pt` modules |
| Prep | export ONNX → build engines (per-GPU) | trace to `.pt` (once, portable) |
| Runner | `trt_engine.cpp` (`TRTEngine`) | `torch_engine.cpp` (`TorchEngine`) |
| Plugin deps | TensorRT (static or bundled) | none beyond Nuke |
| `.so` size | ~800MB | ~232MB |
| Build time | minutes (+ engine builds) | seconds |
| Speed | faster (TRT fp16 + fusion) | slower (plain libtorch) |
| Memory core, node, cache | — identical — | — identical — |

The fusion_transformer (E4) was *already* libtorch in the TRT build, so this backend
just extends that approach to the other four stages.

---

## Step 1 — Trace the 4 stages to `.pt`

The TRT path exported each sub-module to ONNX then built an engine. Here we instead
**trace the same wrapper modules to TorchScript** — reusing the exact wrapper classes
from `001_PREP`'s `export_*.py` (same flat forward signatures), so the libtorch
runner exposes an identical interface to the TRT one.

[`trace_all_stages.py`](trace_all_stages.py) does all four:

```bash
# in the Cutie dir, alongside the export_*.py wrappers (env cutie311)
python trace_all_stages.py
# -> encode_image.pt  transform_key.pt  mask_encoder.pt  mask_decoder.pt
```

With your existing `fusion_transformer.pt` (from `001_PREP`), that's all 5 modules.

**Two shape gotchas** (the script has them right, but know why):
- The encoder/decoder sub-modules `cat` the sensory/mask tensors on the object axis,
  so **sensory is 5D** `(1,1,256,h,w)` and **masks is 4D** `(1,1,H,W)` for E3;
  **memory_readout and sensory are both 5D** for E5. These match the shapes the
  working TRT exports used (the ground truth — those built validated engines). Pass
  them at those exact ranks or tracing errors with a dimension-mismatch.
- Traces are **locked to the fixed padded size** (1088×1920) — `adaptive_avg_pool2d`
  bakes the output size in as a constant (you'll see a `TracerWarning`). That's fine:
  the plugin always feeds that size, same as the static engines.

Expected trace-vs-eager error is tiny (E2/E5 exactly 0; E1/E3 ~3e-3 from fp16-path
rounding in the conv stacks) — negligible for mattes.

---

## Step 2 — The `TorchEngine` runner (drop-in for `TRTEngine`)

[`torch_engine.h`](torch_engine.h) / [`torch_engine.cpp`](torch_engine.cpp)

The trick that keeps the rest of the plugin unchanged: `TorchEngine` exposes the
**same** `run(map<name,Tensor>, stream)` interface as `TRTEngine`. A traced module
returns outputs *positionally* (a tuple), so `TorchEngine` takes the input/output
name lists at construction and maps named-IO ↔ positional-args. Every
`e1_->run({{"image", t}}, stream)` call elsewhere works identically.

```cpp
TorchEngine(const void* data, size_t size, const std::string& name,
            std::vector<std::string> inNames,
            std::vector<std::string> outNames,
            bool autocastFp16 = true);
```

It loads the `.pt` from the embedded byte range (`torch::jit::load` on an
`istringstream`, onto CUDA), maps inputs by name to positional args, forwards under
**autocast fp16** (Cutie's native mode — fast, and avoids the fp16-NaN issues that
forced E3/E5 to fp32 *as TRT engines*, because here we run the real modules), and
maps the tuple outputs back to names. Same single-stream discipline as the TRT path.

> **Autocast API note:** on torch 2.7 the call is
> `at::autocast::set_autocast_enabled(at::kCUDA, true/false)` around the forward.
> Don't call `clear_autocast_cache()` — it's not in torch 2.7's `at::autocast` and
> isn't needed (it's only a memory hint).

---

## Step 3 — The backend flag

The whole switch is a CMake option, `CUTIE_LIBTORCH_BACKEND` (default OFF = the TRT
build). The node source is shared; a type alias picks the runner:

```cpp
// CutieRoto.h
#ifdef CUTIE_LIBTORCH_BACKEND
  #include "torch_engine.h"
  namespace cutie { using EngineT = TorchEngine; }
#else
  #include "trt_engine.h"
  namespace cutie { using EngineT = TRTEngine; }
#endif
```

The node declares `std::unique_ptr<cutie::EngineT> e1_, e2_, e3_, e5_;` and constructs
each with the right args per backend (the libtorch branch passes the IO name lists;
the TRT branch passes the TRT runtime). Build it:

```bash
# copy the traced .pt into models/ (alongside fusion_transformer.pt)
cp encode_image.pt transform_key.pt mask_encoder.pt mask_decoder.pt models/

rm -rf build
cmake -B build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCUTIE_LIBTORCH_BACKEND=ON
cmake --build build -j --target CutieRoto

ls -la build/CutieRoto.so                       # ~232MB
ldd build/CutieRoto.so | grep -i nvinfer        # expect: nothing (zero TRT)
```

`ldd` should show `libtorch*.so` / `libcudart.so.12` resolving **from Nuke**, and no
`libnvinfer` anywhere.

> **Gotcha — keep the libtorch build truly TRT-free.** A shared library links even
> with undefined symbols, so a stray reference to a TRT type only bites at *load*
> time in Nuke (`undefined symbol: ...TRTLogger...`). The libtorch build must not
> *include* `trt_engine.h`/`<NvInfer.h>` or *construct* any TRT object — all of that
> is `#ifndef CUTIE_LIBTORCH_BACKEND`-guarded in `CutieRoto.h`/`.cpp`. If you see an
> undefined-symbol error on load, something TRT slipped past a guard.

CMake also embeds the `.pt` stage files instead of the `.engine` files (the asset
filenames are switched by the same flag), and links **no** `nvinfer` — only DDImage,
torch, cudart.

---

## Step 4 — Run it

```bash
cp build/CutieRoto.so ~/.nuke/CPP/17.0/
# launch Nuke from a terminal; set CUTIE_DEBUG=1 only if you want the per-frame trace
```

Identical workflow to the TRT build: plate → input 0, animated roto → input 1,
`Keyframes`, Process. The mattes match the TRT output closely (same modules, same
fp16 autocast as Cutie native). The first frame is slow (libtorch JIT-warms CUDA
kernels); judge speed by the steady-state per-frame rate.

---

## The honest tradeoff

This backend is **slower per frame** than TRT — no fp16 kernel fusion, plain libtorch
execution. How much slower depends on your GPU. But CutieRoto is a
batch-process-then-cache tool: you Process once, then scrub the cached result
instantly. For that workflow, "the Process pass takes somewhat longer" is a mild cost
against "any Nuke user builds it in minutes with zero TensorRT."

**Pick the backend for the user:**
- **Performance / studio pipeline** → the TRT build (`002_CPP`), `-DCUTIE_STATIC_TRT=ON`.
- **Accessibility / just want it working** → this libtorch build, `-DCUTIE_LIBTORCH_BACKEND=ON`.

Both come from one codebase. The memory core, the stateful node, the disk cache, the
`inputAt` frame-pull, the resize/pad geometry, the post-process and keyframe-pin
knobs — all shared, all identical. Only the four inference stages swap.

---

## Files in `003_CPP_Libtorch/`

| File | What it is |
|------|------------|
| `trace_all_stages.py` | trace E1/E2/E3/E5 to `.pt` (reuses the `001_PREP` wrappers) |
| `torch_engine.h` / `.cpp` | the `TorchEngine` runner — drop-in for `TRTEngine` |

> The shared node/core/CMake live in `002_CPP`; this folder holds only what's unique
> to the libtorch backend. Build with `-DCUTIE_LIBTORCH_BACKEND=ON` from the
> `002_CPP` tree with these two sources added and the `.pt` files in `models/`.

---

*CutieRoto port © Peter Mercell, 2026 — [petermercell.com](https://petermercell.com).
Built on Cutie (Ho Kei Cheng et al., MIT).*
