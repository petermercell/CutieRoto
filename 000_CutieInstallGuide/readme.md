# CutieRoto — Build Guide, Part 0: The `cutie311` Environment

The Python environment everything else is built in. Get this right and
[`001_PREP`](../001_PREP/) (exporting/building the engines) just works; get it wrong
and you'll fight import errors and CUDA mismatches all the way through. This is the
exact, verified environment used to build CutieRoto.

> **Scope.** This covers the **conda environment only**. It assumes you already have
> **CUDA 12.8** and **TensorRT 10.9.0.34** installed on the system, plus an NVIDIA
> driver. (CUDA/TRT system installs and the driver are NVIDIA's domain — follow
> their installers.) What's documented here is the Python side that ties into them.

---

## Prerequisites (assumed already installed)

- **NVIDIA driver** (see NVIDIA — driver is independent of this env).
- **CUDA 12.8** at `/usr/local/cuda-12.8`.
- **TensorRT 10.9.0.34** at `/opt/TensorRT-10.9.0.34` (the tarball install — we use
  its bundled Python wheel below, *not* `pip install tensorrt`).
- **conda** (miniconda/anaconda).

> **Why these exact versions?** torch must match the libtorch that **Nuke 17 ships
> (2.7.1)** because the traced `fusion_transformer.pt` is loaded by Nuke's libtorch
> later. torch's CUDA build (cu128) must match the system CUDA (12.8), which must
> match TensorRT 10.9. Change one and the chain breaks. Pin all of them.

---

## Install (the exact sequence)

```bash
conda create -n cutie311 python=3.11
conda activate cutie311

# torch/torchvision: match Nuke 17 exactly (2.7.1 / 0.22.1), CUDA 12.8 wheels.
# Use PyTorch's cu128 index — NOT a plain `pip install torch` (wrong CUDA build).
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Cutie itself, editable, WITHOUT its deps (we install a curated set next so we
# control versions and don't drag in anything that fights torch).
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
pip install -e . --no-deps

# Cutie's runtime deps (curated)
pip install faust-cchardet
pip install hydra-core omegaconf einops opencv-python pillow tqdm OpenEXR
pip install scipy requests
# thin-plate-spline: from git, no deps (it would otherwise pull an old torch)
pip install "git+https://github.com/cheind/py-thin-plate-spline" --no-deps

# ONNX export + validation
pip install onnx onnxruntime-gpu

# Cutie model weights (skip if you already have them)
python cutie/utils/download_models.py
```

### TensorRT Python bindings — from the tarball, not pip

This is the step that most often goes wrong. **Do not `pip install tensorrt`** — the
PyPI package pulls its *own* CUDA libraries that clash with your system CUDA/TRT.
Instead install the wheel **bundled in your TensorRT tarball**, which matches the TRT
`.so`s you already have:

```bash
ls /opt/TensorRT-10.9.0.34/python/
pip install /opt/TensorRT-10.9.0.34/python/tensorrt-10.9.0.34-cp311-none-linux_x86_64.whl

# pycuda (used by some build/validation scripts)
pip install pycuda
```

*(The `cp311` in the wheel name = Python 3.11 — match it to your env's Python.)*

---

## Environment variables (every session)

The system likely defaults to a different CUDA (e.g. 13.0). Force 12.8 onto the path
and make the TRT libs visible, **every time you open a shell** to run the scripts:

```bash
conda activate cutie311
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/opt/TensorRT-10.9.0.34/lib:$LD_LIBRARY_PATH
```

> If `nvcc` reports 13.0 (or anything but 12.8), or `import tensorrt` fails with a
> missing `.so`, this `export` block is almost certainly the fix. Consider adding it
> to a small `env.sh` you `source` before working.

---

## Verify

```bash
nvcc --version | grep release          # -> release 12.8
python -c "import tensorrt as trt; print('TRT', trt.__version__)"   # -> TRT 10.9.0.34
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
python -c "from cutie.utils.get_default_model import get_default_model; print('Cutie OK')"
```

Expected:

```
release 12.8
TRT 10.9.0.34
torch 2.7.1+cu128 cuda 12.8 avail True
torchvision 0.22.1+cu128
onnxruntime 1.26.0
Cutie OK
```

If all six print cleanly, the environment is ready — go to [`001_PREP`](../001_PREP/).

---

## Files in `000_CutieInstallGuide/`

| File | What it is |
|------|------------|
| `README.md`        | this walkthrough |
| `environment.yml`  | exact conda env export (reproduce with `conda env create -f`) |
| `pip-freeze.txt`   | exact pip package versions, for reference / `pip install -r` |

> `environment.yml` / `pip-freeze.txt` are the "if all else fails, replicate exactly"
> artifacts. The walkthrough above is the readable path; the pinned files are the
> safety net. Note the TensorRT wheel is a **local file** install (from the tarball)
> — `pip install -r pip-freeze.txt` won't fetch `tensorrt` from PyPI, so install that
> wheel separately as shown above.

---

## Key versions (verified working)

| Component | Version |
|-----------|---------|
| Python    | 3.11 |
| torch     | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| TensorRT  | 10.9.0.34 (tarball wheel) |
| CUDA      | 12.8 |
| onnxruntime-gpu | 1.26.0 |
| onnx      | 1.21.0 |
| numpy     | 2.4.4 |
| opencv-python | 4.13.0.92 |
| OpenEXR   | 3.4.11 |
| pycuda    | 2026.1 |
| GPU (built on) | RTX A5000 (sm_86) |
| OS (built on)  | Rocky Linux 9 |

---

*CutieRoto port © Peter Mercell, 2026 — [petermercell.com](https://petermercell.com).
Built on Cutie (Ho Kei Cheng et al., MIT).*
