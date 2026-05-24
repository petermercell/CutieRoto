#!/usr/bin/env python3
"""
export_mask_encoder.py  —  Stage 1, Milestone 4 (mask encoder / encode_mask body)

Wrap Cutie's MaskEncoder (the body of CUTIE.encode_mask) as a flat module and
export to ONNX. Single-object (num_objects=1), fast path (chunk_size=-1),
deep_update=True. STATIC full-HD (1088x1920), like the decoder.

Unlike the decoder, the mask encoder's GRU is SensoryDeepUpdater, which is just
transform(cat([g,h])) with NO area-mode downsample -> no adaptive-pool export
blocker. The only other pooling is AdaptiveAvgPool2d(1) in channel attention,
which is a constant (1,1) size and exports fine. So this export should be clean.

Engine boundary:
  inputs : image (1,3,1088,1920) raw [0,1]  (ImageNet norm baked in here)
           pix_feat (1,256,68,120)          (from the image encoder; fp32 binding)
           sensory  (1,1,256,68,120) fp32
           masks    (1,1,1088,1920)         (the roto, soft or binary)
  outputs: mask_value  (1,1,256,68,120)     (memory value to store)
           new_sensory (1,1,256,68,120) fp32

others (sum of OTHER objects' masks) is all-zeros for a single object, so we build
it internally as zeros_like(masks). Image is normalized inside the wrapper
(pixel_mean/std), matching CUTIE.encode_mask, so the engine takes a raw [0,1]
image just like the image encoder (E1).

Only fp32 region: SensoryDeepUpdater GRU (autocast disabled in source). We build
the engine FP32-first (build_mask_encoder.py, no FP16 flag) for correctness, then
optionally fp16 the safe trunk later.

Run:
  python export_mask_encoder.py
"""

import numpy as np
import torch
import torch.nn as nn

from cutie.utils.get_default_model import get_default_model

ONNX_PATH = "mask_encoder.onnx"
OPSET = 17
PAD_H, PAD_W = 1088, 1920


class MaskEncoderWrapper(nn.Module):
    """Single-object MaskEncoder body (CUTIE.encode_mask up to mask_value+sensory).
    Inputs image,pix_feat,sensory,masks -> (mask_value, new_sensory)."""
    def __init__(self, model):
        super().__init__()
        self.enc = model.mask_encoder
        # ImageNet norm buffers (same as CUTIE), registered so they move with .to()
        self.register_buffer("pixel_mean", model.pixel_mean.clone())
        self.register_buffer("pixel_std", model.pixel_std.clone())

    def forward(self, image, pix_feat, sensory, masks):
        image = (image - self.pixel_mean) / self.pixel_std
        others = torch.zeros_like(masks)          # single object -> others is zeros
        mask_value, new_sensory = self.enc(
            image, pix_feat, sensory, masks, others,
            deep_update=True, chunk_size=-1)
        return mask_value, new_sensory


def main():
    dev = "cuda"
    m = get_default_model().to(dev).eval()

    H, W = PAD_H, PAD_W
    image = torch.rand(1, 3, H, W, device=dev)
    pix_feat = torch.randn(1, 256, H // 16, W // 16, device=dev)
    sensory = torch.zeros(1, 1, 256, H // 16, W // 16, device=dev)
    masks = torch.zeros(1, 1, H, W, device=dev); masks[..., 300:600, 400:800] = 1.0

    # --- faithfulness check: wrapper vs ORIGINAL CUTIE.encode_mask (fp32) ---
    with torch.inference_mode():
        mv_o, ns_o, _, _ = m.encode_mask(image, pix_feat, sensory, masks,
                                         deep_update=True, chunk_size=-1, need_weights=False)
    w = MaskEncoderWrapper(m).to(dev).eval().float()
    with torch.inference_mode():
        mv_p, ns_p = w(image, pix_feat, sensory, masks)
    print("[faithful] wrapper vs original CUTIE.encode_mask (fp32 vs fp32):")
    print(f"  mask_value  max_abs={ (mv_o.float()-mv_p.float()).abs().max().item():.2e}")
    print(f"  new_sensory max_abs={ (ns_o.float()-ns_p.float()).abs().max().item():.2e}")

    print(f"[export] exporting mask_encoder ONNX (STATIC {H}x{W}, legacy exporter) ...")
    torch.onnx.export(
        w, (image, pix_feat, sensory, masks), ONNX_PATH,
        input_names=["image", "pix_feat", "sensory", "masks"],
        output_names=["mask_value", "new_sensory"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote {ONNX_PATH}")

    import onnx
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("[check] onnx.checker passed")

    import onnxruntime as ort
    ref = [mv_p.float().cpu().numpy(), ns_p.float().cpu().numpy()]
    sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(None, {"image": image.cpu().numpy(), "pix_feat": pix_feat.cpu().numpy(),
                          "sensory": sensory.cpu().numpy(), "masks": masks.cpu().numpy()})
    print("[validate] PyTorch vs onnxruntime (fp32):")
    for n, r, o in zip(["mask_value", "new_sensory"], ref, out):
        rel = np.abs(r - o) / (np.abs(r) + 1e-3)
        print(f"  {n:12s} shape={tuple(o.shape)} max_abs={np.abs(r-o).max():.2e} "
              f"median_rel={np.median(rel):.2e}")
    print("[done]  (static HD engine: only runs at 1088x1920 padded input)")


if __name__ == "__main__":
    main()
