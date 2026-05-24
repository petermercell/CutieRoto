#!/usr/bin/env python3
"""
trace_fusion_transformer.py  —  Path B, Route 1 (libtorch submodule artifact)

The C++ plugin keeps pixel_fusion + object_transformer in libtorch. Nuke has the
libtorch RUNTIME, but not Cutie's TRAINED modules. So we produce a loadable
TorchScript artifact (.pt) of exactly those two stages, runnable in C++ via
torch::jit::load -> module.forward({...}).

We TRACE (not script) a thin wrapper for the single-object static-HD path:
  inputs : pix_feat       (1,256,68,120)   from E1
           visual_readout (1,1,256,68,120) from the fp32 memory read
           sensory        (1,1,256,68,120) fp32 state
           last_mask      (1,1,1088,1920)  prev-frame prob (full res; fused
                                            interpolates to stride16 internally)
           obj_memory     (1,1,T,Q,emb+1)  obj_v accumulator (T=1 at inference)
  output : readout_memory (1,1,256,68,120) -> goes to the E5 decoder

Tracing is correct here because the pipeline is ALWAYS single-object at fixed HD:
no dynamic control flow to preserve. Tracing also sidesteps the MultiheadAttention
ONNX-export problem entirely (this is the E4 we deliberately did NOT export to TRT).

We validate trace-vs-eager (should be ~0) and that the reloaded .pt matches.

Run:
  python trace_fusion_transformer.py
Produces: fusion_transformer.pt  (load in C++ with torch::jit::load)
"""

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
        # pixel_fusion (single object; chunk_size=-1 fast path)
        pixel_readout = self.net.pixel_fusion(pix_feat, visual_readout, sensory, last_mask)
        # object transformer; obj_memory already (1,1,T,Q,emb+1) -> unsqueeze(2) done by caller?
        # In MemoryManager.read: this_obj_mem.unsqueeze(2). We do it here so the C++
        # passes the raw obj_v accumulator (1,1,Q,emb+1) and we add the T dim.
        obj_mem = obj_memory.unsqueeze(2)
        readout_memory, _ = self.net.readout_query(pixel_readout, obj_mem)
        # readout_memory: (1, num_obj, C, h, w) -> single object slice kept as (1,1,C,h,w)
        return readout_memory


@torch.inference_mode()
def main():
    dev = "cuda"
    net = get_default_model().to(dev).eval()
    emb = net.cfg.model.object_transformer.embed_dim
    Q = net.cfg.model.object_transformer.num_queries
    print(f"[cfg] embed_dim={emb} num_queries={Q} object_transformer_enabled={net.object_transformer_enabled}")

    pix_feat = torch.randn(1, 256, h, w, device=dev)
    visual_readout = torch.randn(1, 1, 256, h, w, device=dev)
    sensory = torch.zeros(1, 1, 256, h, w, device=dev)
    last_mask = torch.zeros(1, 1, H, W, device=dev); last_mask[..., 300:600, 400:800] = 1.0
    obj_memory = torch.randn(1, 1, Q, emb + 1, device=dev)   # obj_v accumulator (T folded out)

    wrap = FusionTransformer(net).to(dev).eval()

    # eager reference (run OUTSIDE autocast -> fp32, matching the C++ choice)
    with torch.amp.autocast("cuda", enabled=False):
        eager = wrap(pix_feat.float(), visual_readout.float(), sensory.float(),
                     last_mask.float(), obj_memory.float())
    print(f"[eager] readout_memory shape={tuple(eager.shape)} dtype={eager.dtype}")

    # trace
    print("[trace] tracing fusion+transformer (single object, static HD)...")
    with torch.amp.autocast("cuda", enabled=False):
        traced = torch.jit.trace(
            wrap, (pix_feat.float(), visual_readout.float(), sensory.float(),
                   last_mask.float(), obj_memory.float()),
            check_trace=False)
    traced.save(OUT_PT)
    print(f"[trace] saved {OUT_PT}")

    # reload + compare (the real test: what C++ will load)
    reloaded = torch.jit.load(OUT_PT).to(dev).eval()
    with torch.amp.autocast("cuda", enabled=False):
        out = reloaded(pix_feat.float(), visual_readout.float(), sensory.float(),
                       last_mask.float(), obj_memory.float())
    err = (out.float() - eager.float()).abs().max().item()
    print(f"[validate] reloaded .pt vs eager: max_abs_err = {err:.3e}")
    print(f"[validate] output shape={tuple(out.shape)}")
    print("[done] load in C++:  auto m = torch::jit::load(\"fusion_transformer.pt\");")
    print("[done] inputs order: pix_feat, visual_readout, sensory, last_mask, obj_memory")


if __name__ == "__main__":
    main()
