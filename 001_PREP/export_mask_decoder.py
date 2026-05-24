#!/usr/bin/env python3
"""
export_mask_decoder.py  —  Stage 1, Milestone 3 (mask decoder / segment body)
STATIC full-HD export.

Both the legacy and dynamo ONNX exporters fail on the SensoryUpdater's area-mode
downsample under DYNAMIC shapes (it computes output_size = floor(dim*ratio), which
becomes a non-constant adaptive_avg_pool2d / sym_float that neither exporter can
lower). Since we standardized on full HD, we export this engine at FIXED shapes
(padded 1088x1920) -> every size is a compile-time constant -> exports cleanly with
the legacy exporter, no dynamo / onnxscript needed.

Fixed padded input H,W = 1088,1920. Strides:
  f8             : (1, 512, 136, 240)
  f4             : (1, 256, 272, 480)
  memory_readout : (1, 1, 256, 68, 120)
  sensory        : (1, 1, 256, 68, 120)
Outputs:
  new_sensory    : (1, 1, 256, 68, 120) fp32
  prob4          : (1, 1, 272, 480)     fp32  = sigmoid(object logits) at stride-4

Validation runs (grimes) use --max-internal-size 720; that produces DIFFERENT
spatial dims, so the in-situ test must run at the engine's native HD. We'll feed
HD frames for decoder validation (see note in the milestone runner).

Run:
  python export_mask_decoder.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cutie.utils.get_default_model import get_default_model

ONNX_PATH = "mask_decoder.onnx"
OPSET = 17

# fixed padded full-HD input
PAD_H, PAD_W = 1088, 1920


def _patch_sensory_updater_for_export(sensory_update):
    """Replace SensoryUpdater.forward's area-mode downsample-by-ratio with an
    explicit-size adaptive_avg_pool2d. The legacy ONNX exporter rejects
    F.interpolate(scale_factor=r, mode='area') because it lowers to
    adaptive_avg_pool2d with a *computed* (Tensor-valued) output_size, which the
    opset-9 symbolic treats as non-constant even when shapes are static.
    adaptive_avg_pool2d with an explicit (int,int) target exports cleanly and is
    mathematically identical to area-mode downsampling. The target sizes are taken
    from g[0] (the stride-16 reference), so g[1] (stride-8) and g[2] (stride-4)
    are pooled down to g[0]'s HxW exactly as the ratios 1/2 and 1/4 intend.
    Patch is monkeypatched onto the bound module so the original source is
    untouched; only affects this export process."""
    su = sensory_update

    def forward(g, h):
        # g[0]: (B,N,C,H16,W16) ; g[1]: stride-8 ; g[2]: stride-4
        H, W = g[0].shape[-2], g[0].shape[-1]   # concrete ints under tracing

        def area_to(x, out_h, out_w):
            B, N = x.shape[:2]
            x = x.flatten(0, 1)                 # (B*N,C,h,w)
            x = F.adaptive_avg_pool2d(x, (int(out_h), int(out_w)))
            return x.view(B, N, *x.shape[1:])

        g_sum = (su.g16_conv(g[0])
                 + su.g8_conv(area_to(g[1], H, W))
                 + su.g4_conv(area_to(g[2], H, W)))

        with torch.amp.autocast('cuda', enabled=False):
            gg = g_sum.float()
            hh = h.float()
            values = su.transform(torch.cat([gg, hh], dim=2))
            # _recurrent_update inline (same as modules.py)
            dim = values.shape[2] // 3
            forget_gate = torch.sigmoid(values[:, :, :dim])
            update_gate = torch.sigmoid(values[:, :, dim:dim * 2])
            new_value = torch.tanh(values[:, :, dim * 2:])
            new_h = forget_gate * hh * (1 - update_gate) + update_gate * new_value
        return new_h

    su.forward = forward
    return su


class MaskDecoderWrapper(nn.Module):
    """Single-object MaskDecoder body. Inputs f8,f4,readout,sensory ->
    (new_sensory, prob4=sigmoid(logits) at stride 4)."""
    def __init__(self, model):
        super().__init__()
        self.dec = model.mask_decoder
        _patch_sensory_updater_for_export(self.dec.sensory_update)

    def forward(self, f8, f4, memory_readout, sensory):
        dec = self.dec
        f8p, f4p = dec.decoder_feat_proc([f8, f4])
        p16 = memory_readout                       # (1,1,256,h,w)
        p8 = dec.up_16_8(p16, f8p)
        p4 = dec.up_8_4(p8, f4p)
        logits = dec.pred(F.relu(p4.flatten(0, 1).float()))   # (1,1,h4,w4) fp32
        p4c = torch.cat([p4, logits.view(1, 1, 1, *logits.shape[-2:])], 2)
        new_sensory = dec.sensory_update([p16, p8, p4c], sensory)  # fp32 GRU
        prob4 = torch.sigmoid(logits).view(1, 1, *logits.shape[-2:])
        return new_sensory, prob4


def decoder_host_tail(prob4, out_hw):
    """CUTIE.segment tail on host: aggregate -> x4 upsample -> softmax.
    prob4: (1,1,h4,w4) sigmoid prob. Returns prob (2,H,W): [bg, obj]."""
    prob = prob4.float()
    new_prob = torch.cat([1 - prob, prob], dim=1).clamp(1e-7, 1 - 1e-7)
    logits = torch.log(new_prob / (1 - new_prob))
    logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
    prob = F.softmax(logits, dim=1)
    if prob.shape[-2:] != tuple(out_hw):
        prob = F.interpolate(prob, size=tuple(out_hw), mode="bilinear", align_corners=False)
    return prob[0]


def main():
    dev = "cuda"
    m = get_default_model().to(dev).eval()

    H, W = PAD_H, PAD_W
    f8 = torch.randn(1, 512, H // 8,  W // 8,  device=dev)
    f4 = torch.randn(1, 256, H // 4,  W // 4,  device=dev)
    readout = torch.randn(1, 1, 256, H // 16, W // 16, device=dev)
    sensory = torch.zeros(1, 1, 256, H // 16, W // 16, device=dev)

    # --- faithfulness check: patched wrapper vs ORIGINAL model.segment ---
    # Build the original-model reference BEFORE the wrapper patches sensory_update.
    # CUTIE.segment needs ms_image_feat = [f16, f8, f4]; the body uses [1:] only,
    # but we must pass a stride-16 placeholder of the right shape for index 0.
    f16_dummy = torch.randn(1, 1024, H // 16, W // 16, device=dev)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        ns_o, _, prob_o = m.segment([f16_dummy, f8, f4], readout, sensory,
                                    selector=None, chunk_size=-1, update_sensory=True)
    # prob_o is (1,2,H,W) full-res softmax; object channel upsampled. We compare
    # new_sensory directly, and compare our host-tail(prob4) to prob_o.

    w = MaskDecoderWrapper(m).to(dev).eval().float()   # this patches sensory_update
    with torch.inference_mode():
        ns_p, prob4_p = w(f8, f4, readout, sensory)
        prob_p = decoder_host_tail(prob4_p, (H, W)).unsqueeze(0)  # (1,2,H,W)

    ns_err = (ns_o.float() - ns_p.float()).abs().max().item()
    pr_err = (prob_o.float() - prob_p.float()).abs().max().item()
    print(f"[faithful] patched-wrapper vs original model.segment:")
    print(f"  new_sensory max_abs_err = {ns_err:.2e}")
    print(f"  prob(full)  max_abs_err = {pr_err:.2e}   "
          f"(host-tail vs model's own aggregate+upsample+softmax)")

    print(f"[export] exporting mask_decoder ONNX (STATIC {H}x{W}, legacy exporter) ...")
    torch.onnx.export(
        w, (f8, f4, readout, sensory), ONNX_PATH,
        input_names=["f8", "f4", "memory_readout", "sensory"],
        output_names=["new_sensory", "prob4"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,            # legacy exporter; static shapes -> no adaptive-pool issue
    )
    print(f"[export] wrote {ONNX_PATH}")

    import onnx
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("[check] onnx.checker passed")

    import onnxruntime as ort
    with torch.inference_mode():
        ref = w(f8, f4, readout, sensory)
    ref = [r.float().cpu().numpy() for r in ref]
    sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(None, {"f8": f8.cpu().numpy(), "f4": f4.cpu().numpy(),
                          "memory_readout": readout.cpu().numpy(),
                          "sensory": sensory.cpu().numpy()})
    print("[validate] PyTorch vs onnxruntime (fp32):")
    for n, r, o in zip(["new_sensory", "prob4"], ref, out):
        rel = np.abs(r - o) / (np.abs(r) + 1e-3)
        print(f"  {n:12s} shape={tuple(o.shape)} max_abs={np.abs(r-o).max():.2e} "
              f"median_rel={np.median(rel):.2e}")
    print("[done]  (static HD engine: only runs at 1088x1920 padded input)")


if __name__ == "__main__":
    main()
