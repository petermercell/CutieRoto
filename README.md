# CutieRoto

**AI-tracked roto for Foundry Nuke — roto a few keyframes, propagate the mask across the clip.**

CutieRoto is a native Nuke plugin (C++/NDK): a **video object segmenter that takes
hand-roto'd alpha masks on a few keyframes and propagates the masklet across the
whole clip.** Connect your plate and a single animated Roto node, key a handful of
frames, list them, hit **Process** — the mask is propagated across the whole range
and cached to disk for instant scrubbing.

It's a from-scratch port of [Cutie](https://github.com/hkchengrex/Cutie)
(Ho Kei Cheng et al., MIT) onto a hybrid **TensorRT + libtorch** pipeline, built for
Nuke 17. TensorRT is statically linked, so deployment needs only Nuke 17 and an
NVIDIA driver — **no TensorRT install** on any studio machine.

**Two backends, one codebase.** Build the **TensorRT** backend for speed (a
self-contained ~800MB plugin), or the **pure-libtorch** backend ([`003_CPP_Libtorch`](003_CPP_Libtorch/))
for accessibility — no TensorRT at all, builds in seconds, 232MB, runs on nothing but
Nuke 17. Same node, same results; the libtorch one is slower but trivially buildable.

> This repo is also a **complete, honest step-by-step guide** to the port — every
> stage, the real decisions, and the dead-ends avoided. If you want to learn how to
> get a stateful AI model running natively inside Nuke (not via the in-process Python
> that crashes, not via a whole-model TorchScript that can't carry state), this is
> the map.

---

## What makes this non-trivial

Cutie is a *stateful* video object segmentation model — it carries memory forward
frame to frame. That breaks the usual "export the whole net to one engine / one
`.cat`" approach. The port instead splits Cutie and runs each piece where it's best:

- **4 stages on TensorRT** — the CNN encoders + mask encoder/decoder (feed-forward,
  TRT-friendly).
- **1 stage on libtorch** — the pixel-fusion + object-transformer (MultiheadAttention,
  awkward to export; Nuke ships the libtorch runtime, so we run it in-process as a
  traced TorchScript module).
- **The memory read reimplemented in C++** — pure tensor math, bit-exact to the
  Python reference.

Plus the hard integration wins: TensorRT and libtorch *coexisting* in one Nuke
process (shared CUDA), arbitrary-frame input pulls for the sequential propagation,
and a static-TensorRT / dynamic-CUDA-from-Nuke link so it deploys with zero TRT
install.

---

## The guide, in stages

| Stage | Folder | What it covers |
|-------|--------|----------------|
| **0. Environment** | [`000_CutieInstallGuide/`](000_CutieInstallGuide/) | The `cutie311` conda environment — the exact, verified Python setup (torch 2.7.1+cu128, the TensorRT tarball wheel, Cutie + deps) everything else builds on. |
| **1. Model Prep** | [`001_PREP/`](001_PREP/) | Cutie weights → 4 TensorRT engines + 1 traced TorchScript module. Export, build, trace, validate against the Python oracle. |
| **2. The C++ Plugin** | [`002_CPP/`](002_CPP/) | The TRT engine runner, the bit-exact C++ memory core, the stateful Nuke node (`inputAt` frame-pull, disk cache, post-process), and the static-TensorRT build. |
| **3. Pure-libtorch backend** | [`003_CPP_Libtorch/`](003_CPP_Libtorch/) | The no-TensorRT alternative: trace the 4 stages to `.pt`, the `TorchEngine` drop-in runner, the backend flag. Builds in seconds, 232MB, needs only Nuke. Slower but trivially buildable.

---

## Requirements

- **Build:** Nuke 17.0 (provides libtorch 2.7.1 + CUDA runtime), TensorRT 10.9,
  CUDA 12.8, GCC 11, an NVIDIA GPU. Linux (Rocky 9 tested).
- **Run (the built plugin):** Nuke 17.0 + an NVIDIA driver. Nothing else — TensorRT
  is statically embedded.

Engines are GPU-architecture- and TRT-version-specific, so you build them for your
own card in Stage 1 (it's quick, and the guide walks you through it).

---

## Get the plugin

- **Build it yourself:** follow `000_CutieInstallGuide` → `001_PREP` → `002_CPP`. ~3/4h end to end.

---

## Credits & license

CutieRoto port by **Peter Mercell**, 2026 — [petermercell.com](https://petermercell.com).

Built on **Cutie** by Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee,
Alexander Schwing — [github.com/hkchengrex/Cutie](https://github.com/hkchengrex/Cutie),
MIT License. CutieRoto respects and preserves Cutie's MIT license; see
[`001_PREP/`](001_PREP/) for how the model assets are derived from Cutie's weights.

This port is released under [LICENSE](LICENSE).
