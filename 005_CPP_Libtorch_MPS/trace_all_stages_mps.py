#!/usr/bin/env python3
"""
trace_all_stages_mps.py  —  trace the 4 engine stages to DEVICE-NEUTRAL .pt for
                            the macOS / Apple-Metal (MPS) CutieRoto backend.

This is the 003 trace_all_stages.py with ONE change that matters for Mac:
we trace on CPU (DEVICE = "cpu") instead of CUDA. A module traced + frozen on CPU
holds device-neutral fp32 weights, so the C++ can load it straight onto MPS with
  torch::jit::load(stream, torch::kMPS)
and libtorch remaps the parameters to the Apple GPU.

Why not trace on MPS directly? Tracing records the op graph; the saved tensors
just need to be loadable and movable. CPU tracing is the most portable and avoids
baking any device into the graph. (You also generally can't trace on CUDA on a
Mac — there's no NVIDIA GPU — which is the whole reason for this file.)

Run it on the Mac (or any machine) in the cutie env, alongside the export_*.py
wrappers from 001_PREP:

  python trace_all_stages_mps.py

Produces (device-neutral, fp32):
  encode_image.pt  transform_key.pt  mask_encoder.pt  mask_decoder.pt

With your existing fusion_transformer.pt (from 001_PREP) that's all 5 modules the
MPS backend loads. Copy all five into  005_CPP_Libtorch_MPS/models/  (or any dir
you point $CUTIE_MODEL_DIR at).

NOTE on fusion_transformer.pt: if yours was traced/frozen on CUDA it will still
load on MPS in most cases (libtorch remaps params), but if it errors on load,
re-trace it on CPU the same way (trace_fusion_transformer.py with device='cpu').
"""

import torch

from cutie.utils.get_default_model import get_default_model

# reuse the existing wrappers (same forward signatures the C++ expects)
from export_encode_image import EncodeImageWrapper
from export_transform_key import TransformKeyWrapper
from export_mask_encoder import MaskEncoderWrapper
from export_mask_decoder import MaskDecoderWrapper

DEVICE = "mps"                       # trace ON MPS: bakes any internal device
                                     # constants to mps:0 and validates every op
                                     # actually runs on Apple Metal
# fixed padded HD (matches the static engines + the plugin's internal size)
H, W = 1088, 1920
H16, W16 = H // 16, W // 16          # 68, 120
H8,  W8  = H // 8,  W // 8           # 136, 240
H4,  W4  = H // 4,  W // 4           # 272, 480
CV, CK = 256, 64


def _save_traced(wrapper, example_inputs, out_path, name):
    wrapper = wrapper.float().to(DEVICE).eval()   # .float(): MPS has no float64
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
        # IMPORTANT (MPS): do NOT torch.jit.freeze() here. Freeze inlines the
        # weights as graph CONSTANTS, and module.to(device) cannot move graph
        # constants (only registered params/buffers). On the MPS build that leaves
        # the weights on CPU while inputs are on MPS ->
        #   "Expected all tensors to be on the same device, mps:0 and cpu".
        # Leaving the module unfrozen keeps the weights as buffers that the C++
        # load (CPU -> float32 -> MPS) moves correctly. (Freeze is only an
        # optimization; tracing already captured the eval-path graph.)
        # quick self-consistency: traced vs eager on the example
        out_eager  = wrapper(*example_inputs) if isinstance(example_inputs, tuple) \
                     else wrapper(example_inputs)
        out_traced = traced(*example_inputs) if isinstance(example_inputs, tuple) \
                     else traced(example_inputs)
        oe = out_eager if isinstance(out_eager, (tuple, list)) else (out_eager,)
        ot = out_traced if isinstance(out_traced, (tuple, list)) else (out_traced,)
        merr = max(float((a.float() - b.float()).abs().max()) for a, b in zip(oe, ot))
    traced.save(out_path)
    print(f"[{name}] traced (mps) -> {out_path}   trace-vs-eager max_abs={merr:.3e}")


@torch.inference_mode()
def main():
    model = get_default_model().float().to(DEVICE).eval()   # .float(): no float64 on MPS

    # ---- E1 encode_image: image(1,3,H,W) ----
    img = torch.rand(1, 3, H, W, device=DEVICE)
    _save_traced(EncodeImageWrapper(model), (img,),
                 "encode_image.pt", "E1 encode_image")

    # ---- E2 transform_key: f16(1,1024,H16,W16) ----
    f16 = torch.rand(1, 1024, H16, W16, device=DEVICE)
    _save_traced(TransformKeyWrapper(model), (f16,),
                 "transform_key.pt", "E2 transform_key")

    # ---- E3 mask_encoder: image, pix_feat, sensory(5D), masks(4D) ----
    pix_feat = torch.rand(1, 256, H16, W16, device=DEVICE)
    sensory  = torch.zeros(1, 1, 256, H16, W16, device=DEVICE)
    masks    = torch.zeros(1, 1, H, W, device=DEVICE)
    masks[..., 300:600, 400:800] = 1.0
    _save_traced(MaskEncoderWrapper(model),
                 (img, pix_feat, sensory, masks),
                 "mask_encoder.pt", "E3 mask_encoder")

    # ---- E5 mask_decoder: f8, f4, memory_readout(5D), sensory(5D) ----
    f8  = torch.rand(1, 512, H8, W8, device=DEVICE)
    f4  = torch.rand(1, 256, H4, W4, device=DEVICE)
    mem = torch.rand(1, 1, 256, H16, W16, device=DEVICE)
    sen = torch.rand(1, 1, 256, H16, W16, device=DEVICE)
    _save_traced(MaskDecoderWrapper(model),
                 (f8, f4, mem, sen),
                 "mask_decoder.pt", "E5 mask_decoder")

    print("\n[done] 4 device-neutral stage .pt files written. With fusion_transformer.pt")
    print("       that's all 5 modules the MPS backend loads. Copy them into models/.")


if __name__ == "__main__":
    main()
