#!/usr/bin/env python3
"""
trace_all_stages.py  —  PROTOTYPE: trace the 4 engine stages to TorchScript .pt
                        for a pure-libtorch (no-TensorRT) CutieRoto backend.

The TRT path exports each sub-module to ONNX then builds a TRT engine. For the
libtorch-only backend we instead TRACE the SAME wrapper modules to .pt and load
them with Nuke's bundled libtorch via torch::jit::load — no engine build, no TRT.

We reuse the EXACT wrapper classes from the export_*.py scripts (same flat forward
signatures the C++ already expects), so the libtorch backend exposes an identical
interface to the TRT runner: same input/output names, drop-in swap.

Stages (matching the TRT engines):
  E1 encode_image   : image                                  -> f16,f8,f4,pix_feat
  E2 transform_key  : f16                                     -> key,shrinkage,selection
  E3 mask_encoder   : image,pix_feat,sensory,masks           -> mask_value,new_sensory
  E5 mask_decoder   : f8,f4,memory_readout,sensory           -> new_sensory,prob4
  (E4 fusion_transformer.pt already traced by trace_fusion_transformer.py)

We trace at the FIXED padded HD size the pipeline uses (1088x1920), matching what
the static TRT engines did. E1/E2 are "dynamic" in TRT but the plugin always feeds
the padded size, so a fixed trace is fine and simpler.

Precision: traced in fp32 weights; the C++ can run them under autocast fp16 (Cutie's
native mode) for speed, or fp32 for bit-exactness. Tracing itself is precision-
agnostic — the .pt holds fp32 weights and autocast is applied at call time.

Run (env cutie311; CUDA12.8 + TRT not needed for this script):
  python trace_all_stages.py
Produces: encode_image.pt  transform_key.pt  mask_encoder.pt  mask_decoder.pt
"""

import torch

from cutie.utils.get_default_model import get_default_model

# reuse the existing wrappers (same forward signatures the C++ expects)
from export_encode_image import EncodeImageWrapper
from export_transform_key import TransformKeyWrapper
from export_mask_encoder import MaskEncoderWrapper
from export_mask_decoder import MaskDecoderWrapper

DEVICE = "cuda"
# fixed padded HD (matches the static engines + the plugin's internal size)
H, W = 1088, 1920
H16, W16 = H // 16, W // 16          # 68, 120
H8,  W8  = H // 8,  W // 8           # 136, 240
H4,  W4  = H // 4,  W // 4           # 272, 480
CV, CK = 256, 64


def _save_traced(wrapper, example_inputs, out_path, name):
    wrapper = wrapper.to(DEVICE).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
        traced = torch.jit.freeze(traced)            # inline params, drop training bits
        # quick self-consistency: traced vs eager on the example
        out_eager  = wrapper(*example_inputs) if isinstance(example_inputs, tuple) \
                     else wrapper(example_inputs)
        out_traced = traced(*example_inputs) if isinstance(example_inputs, tuple) \
                     else traced(example_inputs)
        oe = out_eager if isinstance(out_eager, (tuple, list)) else (out_eager,)
        ot = out_traced if isinstance(out_traced, (tuple, list)) else (out_traced,)
        merr = max(float((a.float() - b.float()).abs().max()) for a, b in zip(oe, ot))
    traced.save(out_path)
    print(f"[{name}] traced -> {out_path}   trace-vs-eager max_abs={merr:.3e}")


@torch.inference_mode()
def main():
    model = get_default_model().to(DEVICE).eval()

    # ---- E1 encode_image: image(1,3,H,W) ----
    img = torch.rand(1, 3, H, W, device=DEVICE)
    _save_traced(EncodeImageWrapper(model), (img,),
                 "encode_image.pt", "E1 encode_image")

    # ---- E2 transform_key: f16(1,1024,H16,W16) ----
    f16 = torch.rand(1, 1024, H16, W16, device=DEVICE)
    _save_traced(TransformKeyWrapper(model), (f16,),
                 "transform_key.pt", "E2 transform_key")

    # ---- E3 mask_encoder: image, pix_feat, sensory, masks ----
    # Shapes taken from the working export_mask_encoder.py:
    #   sensory (1,1,256,H16,W16) 5D ;  masks (1,1,H,W) 4D
    pix_feat = torch.rand(1, 256, H16, W16, device=DEVICE)
    sensory  = torch.zeros(1, 1, 256, H16, W16, device=DEVICE)
    masks    = torch.zeros(1, 1, H, W, device=DEVICE)
    masks[..., 300:600, 400:800] = 1.0
    _save_traced(MaskEncoderWrapper(model),
                 (img, pix_feat, sensory, masks),
                 "mask_encoder.pt", "E3 mask_encoder")

    # ---- E5 mask_decoder: f8, f4, memory_readout, sensory ----
    # memory_readout and sensory are 5D (1,1,256,H16,W16) per the engine IO shapes.
    f8  = torch.rand(1, 512, H8, W8, device=DEVICE)
    f4  = torch.rand(1, 256, H4, W4, device=DEVICE)
    mem = torch.rand(1, 1, 256, H16, W16, device=DEVICE)
    sen = torch.rand(1, 1, 256, H16, W16, device=DEVICE)
    _save_traced(MaskDecoderWrapper(model),
                 (f8, f4, mem, sen),
                 "mask_decoder.pt", "E5 mask_decoder")

    print("\n[done] 4 stage .pt files written. With fusion_transformer.pt that's all")
    print("       5 modules the libtorch-only backend loads — no TRT engines needed.")
    print("NOTE: verify the squeeze/unsqueeze dims above against your wrapper forward")
    print("      signatures; adjust if a wrapper expects 4D vs 5D sensory/masks.")


if __name__ == "__main__":
    main()
