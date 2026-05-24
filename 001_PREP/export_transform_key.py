#!/usr/bin/env python3
"""
export_transform_key.py  —  Stage 1, Milestone 2 (transform_key engine)

Wrap Cutie's transform_key (KeyProjection) as a flat module and export to ONNX.

Input  : f16 feature (stride-16, 1024ch) -- the encoder's first output.
Outputs: key       (1, key_dim, h, w)   fp16-ok
         shrinkage (1, 1,       h, w)   = d_proj(x)**2 + 1  (keep FP32: feeds
                                          the fp32 affinity/memory read)
         selection (1, key_dim, h, w)   = sigmoid(e_proj(x))  fp16-ok

KeyProjection has NO autocast(enabled=False) region, so fp16 compute is safe;
we just preserve shrinkage as fp32 on output because the memory read needs it.

Run:
  python export_transform_key.py
"""

import numpy as np
import torch
import torch.nn as nn

from cutie.utils.get_default_model import get_default_model

ONNX_PATH = "transform_key.onnx"
OPSET = 17


class TransformKeyWrapper(nn.Module):
    """f16 feature in -> (key, shrinkage, selection). need_s/need_e always True
    (inference always needs both for the anisotropic-L2 memory read)."""
    def __init__(self, model):
        super().__init__()
        self.key_proj = model.key_proj  # the KeyProjection module

    def forward(self, f16):
        # mirror KeyProjection.forward with need_s=need_e=True
        kp = self.key_proj
        x = kp.pix_feat_proj(f16)
        shrinkage = kp.d_proj(x) ** 2 + 1          # (1,1,h,w)
        selection = torch.sigmoid(kp.e_proj(x))    # (1,key_dim,h,w)
        key = kp.key_proj(x)                        # (1,key_dim,h,w)
        return key, shrinkage, selection


def main():
    dev = "cuda"
    model = get_default_model().to(dev).eval()
    wrapper = TransformKeyWrapper(model).to(dev).eval().float()

    # input: stride-16 feature. At 480x832 -> 30x52, 1024 channels.
    h, w = 30, 52
    dummy = torch.randn(1, 1024, h, w, device=dev)

    print("[export] exporting transform_key ONNX ...")
    torch.onnx.export(
        wrapper, (dummy,), ONNX_PATH,
        input_names=["f16"],
        output_names=["key", "shrinkage", "selection"],
        dynamic_axes={
            "f16":       {2: "h", 3: "w"},
            "key":       {2: "h", 3: "w"},
            "shrinkage": {2: "h", 3: "w"},
            "selection": {2: "h", 3: "w"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"[export] wrote {ONNX_PATH}")

    import onnx
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("[check] onnx.checker passed")

    # numerical check vs PyTorch (fp32)
    import onnxruntime as ort
    with torch.inference_mode():
        ref = wrapper(dummy)
    ref = [r.float().cpu().numpy() for r in ref]
    sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(None, {"f16": dummy.cpu().numpy()})
    print("[validate] PyTorch vs onnxruntime (fp32):")
    for n, r, o in zip(["key", "shrinkage", "selection"], ref, out):
        rel = np.abs(r - o) / (np.abs(r) + 1e-3)
        print(f"  {n:10s} shape={tuple(o.shape)} max_abs={np.abs(r-o).max():.2e} "
              f"median_rel={np.median(rel):.2e}")
    print("[done]")


if __name__ == "__main__":
    main()
