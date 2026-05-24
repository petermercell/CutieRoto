#!/usr/bin/env python3
"""
export_encode_image.py  —  Stage 1, Milestone 1 (part 1/3)

Wrap Cutie's encode_image as a flat single-in / four-out nn.Module, export to
ONNX with dynamic H/W, and validate the ONNX against PyTorch via onnxruntime-gpu.

encode_image (cutie.py) returns (ms_feat=[f16,f8,f4], pix_feat). We flatten that
to four named tensors and BAKE the (image-mean)/std normalization into the graph,
so the engine's input is raw RGB in [0,1], matching what the pipeline already feeds.

Confirmed shapes (base-mega), input (1,3,H,W), H/W multiples of 16:
  f16:      (1,1024,H/16,W/16)
  f8:       (1, 512,H/8 ,W/8 )
  f4:       (1, 256,H/4 ,W/4 )
  pix_feat: (1, 256,H/16,W/16)
pixel_mean=[0.485,0.456,0.406]  pixel_std=[0.229,0.224,0.225]  (ImageNet)

Run:
  python export_encode_image.py            # exports + validates at default size
"""

import numpy as np
import torch
import torch.nn as nn

from cutie.utils.get_default_model import get_default_model

ONNX_PATH = "encode_image.onnx"
OPSET = 17  # TRT 10.9 is happy with 17


class EncodeImageWrapper(nn.Module):
    """Flat wrapper: raw-RGB image in -> (f16, f8, f4, pix_feat) out.
    Normalization baked in so the engine takes [0,1] RGB directly."""
    def __init__(self, model):
        super().__init__()
        self.pixel_encoder = model.pixel_encoder
        self.pix_feat_proj = model.pix_feat_proj
        # keep the norm buffers (registered, so they ride into the ONNX graph)
        self.register_buffer("pixel_mean", model.pixel_mean.clone())
        self.register_buffer("pixel_std", model.pixel_std.clone())

    def forward(self, image):
        # image: (1,3,H,W) in [0,1]
        x = (image - self.pixel_mean) / self.pixel_std
        f16, f8, f4 = self.pixel_encoder(x)      # strides 16/8/4
        pix_feat = self.pix_feat_proj(f16)
        return f16, f8, f4, pix_feat


def main():
    dev = "cuda"
    model = get_default_model().to(dev).eval()
    wrapper = EncodeImageWrapper(model).to(dev).eval()

    # export in fp32 for a clean graph; TRT applies fp16 at build time.
    # (exporting fp16 graphs is fiddly; let the engine builder handle precision.)
    wrapper = wrapper.float()

    H, W = 480, 832  # 16-divisible sample size for tracing
    dummy = torch.rand(1, 3, H, W, device=dev)

    print("[export] tracing + exporting ONNX ...")
    torch.onnx.export(
        wrapper, (dummy,), ONNX_PATH,
        input_names=["image"],
        output_names=["f16", "f8", "f4", "pix_feat"],
        dynamic_axes={
            "image":    {2: "H",   3: "W"},
            "f16":      {2: "H16", 3: "W16"},
            "f8":       {2: "H8",  3: "W8"},
            "f4":       {2: "H4",  3: "W4"},
            "pix_feat": {2: "H16", 3: "W16"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"[export] wrote {ONNX_PATH}")

    # --- structural check ---
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("[check] onnx.checker passed")

    # --- numerical check: PyTorch vs onnxruntime (hop 1) ---
    import onnxruntime as ort
    # PyTorch reference (fp32 wrapper, fp32 input)
    with torch.inference_mode():
        ref = wrapper(dummy)
    ref = [r.float().cpu().numpy() for r in ref]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    ort_out = sess.run(None, {"image": dummy.cpu().numpy()})

    names = ["f16", "f8", "f4", "pix_feat"]
    print("[validate] PyTorch vs onnxruntime (fp32):")
    ok = True
    for n, r, o in zip(names, ref, ort_out):
        d = np.abs(r - o)
        maxd, meand = float(d.max()), float(d.mean())
        flag = "OK" if maxd < 1e-3 else "CHECK"
        if maxd >= 1e-3:
            ok = False
        print(f"  {n:9s} shape={tuple(o.shape)} max={maxd:.2e} mean={meand:.2e} [{flag}]")

    # --- also test a DIFFERENT resolution to prove dynamic axes work ---
    H2, W2 = 544, 960
    d2 = torch.rand(1, 3, H2, W2, device=dev)
    with torch.inference_mode():
        ref2 = [r.float().cpu().numpy() for r in wrapper(d2)]
    ort2 = sess.run(None, {"image": d2.cpu().numpy()})
    print(f"[validate] dynamic-shape test at {H2}x{W2}:")
    for n, r, o in zip(names, ref2, ort2):
        maxd = float(np.abs(r - o).max())
        print(f"  {n:9s} shape={tuple(o.shape)} max={maxd:.2e} "
              f"[{'OK' if maxd < 1e-3 else 'CHECK'}]")

    print("[done]" + ("  ONNX matches PyTorch." if ok else
                       "  WARNING: divergence > 1e-3, inspect before building engine."))


if __name__ == "__main__":
    main()
