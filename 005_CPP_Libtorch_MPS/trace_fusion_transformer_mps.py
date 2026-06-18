#!/usr/bin/env python3
"""
trace_fusion_transformer_mps.py  —  macOS/CPU trace of the fusion+transformer (E4).

Same as 001_PREP/trace_fusion_transformer.py, but device-neutral: traced on CPU so
the resulting fusion_transformer.pt loads on Apple Metal (MPS) via
torch::jit::load(stream, torch::kMPS). The only changes from the original are
dev="cpu" and the CUDA autocast context -> a plain no-op (we already run fp32).

Run it from the Cutie repo root (so `weights/cutie-base-mega.pth` is found and
`cutie` imports), in the cutie311-mac env:
    python trace_fusion_transformer_mps.py
Produces: fusion_transformer.pt
"""

import contextlib
import torch
import torch.nn as nn
from cutie.utils.get_default_model import get_default_model

OUT_PT = "fusion_transformer.pt"
H, W = 1088, 1920
h, w = H // 16, W // 16   # 68, 120


class FusionTransformer(nn.Module):
    """pixel_fusion -> object_transformer, single object, returns only the readout."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pix_feat, visual_readout, sensory, last_mask, obj_memory):
        pixel_readout = self.net.pixel_fusion(pix_feat, visual_readout, sensory, last_mask)
        obj_mem = obj_memory.unsqueeze(2)
        readout_memory, _ = self.net.readout_query(pixel_readout, obj_mem)
        return readout_memory


@torch.inference_mode()
def main():
    dev = "mps"                      # trace ON MPS: bakes the positional-encoding
                                     # device constant to mps:0 (a CPU trace bakes
                                     # it to cpu -> device mismatch at runtime)
    net = get_default_model().float().to(dev).eval()   # .float(): MPS has no float64
    emb = net.cfg.model.object_transformer.embed_dim
    Q = net.cfg.model.object_transformer.num_queries
    print(f"[cfg] embed_dim={emb} num_queries={Q} "
          f"object_transformer_enabled={net.object_transformer_enabled}")

    pix_feat = torch.randn(1, 256, h, w, device=dev)
    visual_readout = torch.randn(1, 1, 256, h, w, device=dev)
    sensory = torch.zeros(1, 1, 256, h, w, device=dev)
    last_mask = torch.zeros(1, 1, H, W, device=dev); last_mask[..., 300:600, 400:800] = 1.0
    obj_memory = torch.randn(1, 1, Q, emb + 1, device=dev)

    wrap = FusionTransformer(net).float().to(dev).eval()

    # We run fp32 (no autocast) — matches the C++ MPS choice. nullcontext keeps the
    # structure identical to the CUDA script without referencing a CUDA autocast.
    noamp = contextlib.nullcontext()

    with noamp:
        eager = wrap(pix_feat.float(), visual_readout.float(), sensory.float(),
                     last_mask.float(), obj_memory.float())
    print(f"[eager] readout_memory shape={tuple(eager.shape)} dtype={eager.dtype}")

    print("[trace] tracing fusion+transformer (single object, static HD, cpu)...")
    with noamp:
        traced = torch.jit.trace(
            wrap, (pix_feat.float(), visual_readout.float(), sensory.float(),
                   last_mask.float(), obj_memory.float()),
            check_trace=False)
    traced.save(OUT_PT)
    print(f"[trace] saved {OUT_PT}")

    reloaded = torch.jit.load(OUT_PT).to(dev).eval()
    with noamp:
        out = reloaded(pix_feat.float(), visual_readout.float(), sensory.float(),
                       last_mask.float(), obj_memory.float())
    err = (out.float() - eager.float()).abs().max().item()
    print(f"[validate] reloaded .pt vs eager: max_abs_err = {err:.3e}")
    print(f"[validate] output shape={tuple(out.shape)}")
    print("[done] 5th module ready. Copy fusion_transformer.pt into models/ with the other 4.")


if __name__ == "__main__":
    main()
