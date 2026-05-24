# CutieRoto — Build Guide, Part 1: Model Preparation

Turning Cutie's PyTorch model into the 5 deployable assets the Nuke plugin loads:
**4 TensorRT engines + 1 traced TorchScript module.**

This is the first stage of building [CutieRoto](https://petermercell.com) — a native
TensorRT + libtorch Nuke plugin that propagates a roto mask across a clip (roto a
few keyframes, the tracker fills the rest). It's a port of
[Cutie](https://github.com/hkchengrex/Cutie) (Ho Kei Cheng et al., MIT) by
[Peter Mercell](https://petermercell.com).

> You generate the engines yourself from Cutie's weights. **TensorRT engines are
> GPU-architecture- and TRT-version-specific** — an engine built on one card won't
> load on a different GPU arch. So everyone builds their own; that's expected and
> correct. The traced `.pt` is portable (TorchScript), but the guide builds it too.

---

## Why this shape? (the one decision that matters)

Cutie is a *stateful* video object segmentation model: it carries a memory of past
frames forward. That rules out the usual "export the whole model to one engine"
approach — there's no single forward pass. Instead the port splits Cutie into stages
and runs each where it's fastest and most stable:

- **4 stages → TensorRT** (the CNN encoders + the mask encoder/decoder). These are
  feed-forward and TRT-friendly.
- **1 stage → libtorch** (the pixel-fusion + object-transformer). It uses
  MultiheadAttention, which is a pain to export to ONNX/TRT — and since Nuke 17
  *ships the libtorch runtime*, we can just run this stage as a traced TorchScript
  module in-process. We keep it in torch and sidestep the export problem entirely.
- **The memory read** (similarity → top-k softmax → readout) is reimplemented
  directly in C++ later — it's pure tensor math, no learned weights.

This is the "Path B" hybrid: TRT where TRT is good, libtorch where Nuke already has
it. The five assets below are exactly the learned pieces that result.

| Asset | Stage | Format | Precision | Shapes |
|------|-------|--------|-----------|--------|
| `encode_image.fp16.engine`   | E1 image encoder      | TRT | fp16 | dynamic H/W |
| `transform_key.fp16.engine`  | E2 key projection     | TRT | fp16 | dynamic H/W |
| `mask_encoder.fp32.engine`   | E3 mask encoder       | TRT | fp32 | static 1088×1920 |
| `mask_decoder.fp32.engine`   | E5 mask decoder       | TRT | fp32 | static 1088×1920 |
| `fusion_transformer.pt`      | E4 fusion+transformer | TorchScript | fp32 | static |

*(E1/E2 are plain CNNs → fp16 is safe and fast. E3/E5 contain the sensory GRU /
prediction head, which overflow in fp16 → kept fp32. E4 stays in libtorch.)*

---

## Prerequisites

The toolchain must be internally consistent — TRT, CUDA, and the torch used for
export all have to agree. The versions this port was built and validated on:

```
Python 3.11 (conda env)
torch 2.7.1 + cu128, torchvision 0.22.1   # match your target Nuke's libtorch!
TensorRT 10.9.0.34
CUDA 12.8                                  # the TRT + torch CUDA must match
onnxruntime-gpu                            # for ONNX validation
GPU: RTX A5000 (sm_86), driver 590.x
OS: Rocky Linux 9
```

> **Match torch to your Nuke.** The traced `.pt` is loaded by Nuke's bundled
> libtorch later, so trace it with the *same* torch version Nuke ships (Nuke 17.0 =
> torch 2.7.1). A mismatch here causes load failures down the line.

Set the environment every session (adjust paths to your install):

```bash
conda activate cutie311
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/opt/TensorRT-10.9.0.34/lib:$LD_LIBRARY_PATH
```

---

## Step 0 — Get Cutie and its weights

Clone Cutie and download its pretrained weights per its own instructions (MIT
licensed). Put the prep scripts from this repo's `001_PREP/` folder alongside Cutie
so `from cutie...` imports resolve:

```bash
git clone https://github.com/hkchengrex/Cutie
cd Cutie
# follow Cutie's README to fetch the default model weights
# then copy this guide's 001_PREP/*.py scripts into this directory
```

A quick sanity check that the model loads:

```python
from cutie.utils.get_default_model import get_default_model
m = get_default_model().cuda().eval()
print("Cutie loaded:", type(m).__name__)
```

---

## Step 1 — Export the 4 engine stages to ONNX

Each export wraps one Cutie sub-module as a flat single-purpose `nn.Module`, exports
to ONNX (opset 17), and validates the ONNX against PyTorch via onnxruntime. Run them
in order:

```bash
python export_encode_image.py     # -> encode_image.onnx   (E1)
python export_transform_key.py    # -> transform_key.onnx   (E2)
python export_mask_encoder.py     # -> mask_encoder.onnx    (E3)
python export_mask_decoder.py     # -> mask_decoder.onnx    (E5)
```

Notes baked into the exports:
- **E1 bakes the ImageNet normalization into the graph** (`(image-mean)/std`), so the
  engine's input is raw RGB in `[0,1]` — the plugin feeds plain pixels, no preproc.
- **E1/E2 export with dynamic H/W** (spatial dims free), so one engine handles any
  padded plate size.
- **E3/E5 export at static 1088×1920** — they contain the parts that must stay fp32,
  and fixing the shape lets TRT optimize hard.

Each script prints a PyTorch-vs-ONNX max-abs-error; it should be tiny (~1e-3 or
less). If an export errors on an unsupported op, that's the stage that doesn't belong
on TRT — which is exactly why E4 (the attention block) is *not* here.

---

## Step 2 — Build the TensorRT engines

Build each ONNX into a TRT 10.9 engine. E1/E2 build with a dynamic-shape profile +
fp16; E3/E5 build static + fp32.

```bash
python build_engine.py            # encode_image.onnx  -> encode_image.fp16.engine
python build_transform_key.py     # transform_key.onnx -> transform_key.fp16.engine
python build_mask_encoder.py      # mask_encoder.onnx  -> mask_encoder.fp32.engine
python build_mask_decoder.py      # mask_decoder.onnx  -> mask_decoder.fp32.engine
```

The dynamic profile for E1/E2 (in `build_engine.py`) sets min/opt/max input sizes:

```
min  = 1×3× 256× 256
opt  = 1×3× 544× 960    # tune to your typical plate for best perf
max  = 1×3×1088×1920    # covers up to ~1080p long edge; raise if you feed bigger
```

Tune `opt` to your common plate size and raise `max` if you work larger than 1080p
(bigger `max` = longer build + more workspace). These `.engine` files are built for
**your** GPU and TRT version — they are not portable, by design.

---

## Step 3 — Trace the libtorch stage

The pixel-fusion + object-transformer (E4) stays in libtorch. Produce a TorchScript
artifact the C++ plugin loads via `torch::jit::load`:

```bash
python trace_fusion_transformer.py    # -> fusion_transformer.pt
```

This **traces** (not scripts) a thin wrapper of the two stages for the single-object,
fixed-HD path — tracing is correct here because there's no dynamic control flow to
preserve, and it sidesteps the MultiheadAttention export problem completely. The
script validates trace-vs-eager (≈0 error) and that the reloaded `.pt` matches.

> Trace this with the **same torch version your target Nuke ships** (Nuke 17.0 →
> torch 2.7.1). The `.pt` is loaded by Nuke's libtorch at runtime.

---

## Step 4 — Validate the whole assembled pipeline

Before touching C++, prove the 5 assets reproduce Cutie's result end-to-end in
Python. `cutie_mask_propagate.py` is the **oracle** (ground-truth Cutie run);
`validate_pathb_insitu.py` runs the assembled TRT+libtorch pipeline and compares.

```bash
python validate_pathb_insitu.py \
    --frames /path/to/plate.%04d.exr \
    --roto   /path/to/roto.%04d.exr \
    --seeds 1,48,96 --ts-module fusion_transformer.pt
```

Target: per-frame IoU ≈ 0.999 vs the oracle, no NaN, mattes that look right. This
validated run is the reference the C++ plugin must later match — if Python is good
here, the port has a correct target to hit.

(`memory_read_reference.py` is the fp32 spec for the memory read — used by the
validation and, later, transcribed into the C++ memory core. The validation also
needs the `trt_feature_store.py` / `trt_mask_encoder_hook.py` / `trt_decoder_hook.py`
helpers present — they inject the engines into Cutie's network so pass B runs the
TRT pipeline while pass A runs stock PyTorch, on the identical loop.)

---

## What you have now

```
encode_image.fp16.engine     # E1
transform_key.fp16.engine    # E2
mask_encoder.fp32.engine     # E3
mask_decoder.fp32.engine     # E5
fusion_transformer.pt        # E4 (libtorch)
```

These 5 files are everything the Nuke plugin needs — they get embedded into the
`.so` in the C++ build (`002_CPP`). Keep them; the build's `objcopy` step bakes them in.

---

## Files in `001_PREP/`

| Script | Does |
|--------|------|
| `export_encode_image.py`    | E1 → ONNX (+ bakes normalization, validates) |
| `export_transform_key.py`   | E2 → ONNX |
| `export_mask_encoder.py`    | E3 → ONNX |
| `export_mask_decoder.py`    | E5 → ONNX |
| `build_engine.py`           | E1 ONNX → fp16 dynamic engine |
| `build_transform_key.py`    | E2 ONNX → fp16 dynamic engine |
| `build_mask_encoder.py`     | E3 ONNX → fp32 static engine |
| `build_mask_decoder.py`     | E5 ONNX → fp32 static engine |
| `trace_fusion_transformer.py` | E4 → `fusion_transformer.pt` (traced) |
| `cutie_mask_propagate.py`   | the oracle (ground-truth Cutie run) |
| `memory_read_reference.py`  | fp32 memory-read spec (transcribed into the C++ core) |
| `trt_feature_store.py`      | injects E1/E2 engines into Cutie for validation |
| `trt_mask_encoder_hook.py`  | injects E3 engine into Cutie for validation |
| `trt_decoder_hook.py`       | injects E5 engine into Cutie for validation |
| `validate_pathb_insitu.py`  | full Path-B validation vs oracle (the C++ spec) |

> The `trt_*_hook.py` / `trt_feature_store.py` helpers monkey-patch Cutie's network
> to run the TRT engines in place of its PyTorch modules — that's how the A/B
> validation compares the engine pipeline against the pure-PyTorch oracle on the
> exact same recurrent loop. `validate_pathb_insitu.py` imports all of them.

---

**Next: `002_CPP` — the C++ Nuke plugin** (the TRT engine runner, the C++ memory
core, the stateful Nuke node, and the static-TensorRT build).

*Cutie © Ho Kei Cheng et al., MIT. CutieRoto port © Peter Mercell, 2026.*
