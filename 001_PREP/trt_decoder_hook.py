#!/usr/bin/env python3
"""
trt_decoder_hook.py  —  Milestone 3 seam

Routes Cutie's mask decoder (CUTIE.segment body) through the static-HD TRT engine
mask_decoder.fp32.engine, keeping the cheap tail (aggregate -> x4 upsample ->
softmax) on the host. Installed by monkeypatching `network.segment` on the model
instance, so InferenceCore._segment (which calls self.network.segment) transparently
uses the engine.

Engine (build with build_mask_decoder.py), STATIC 1088x1920:
  inputs : f8 (1,512,136,240), f4 (1,256,272,480),
           memory_readout (1,1,256,68,120) fp32, sensory (1,1,256,68,120) fp32
  outputs: new_sensory (1,1,256,68,120) fp32, prob4 (1,1,272,480) fp32

CRITICAL (M1 stream-race lesson): run the engine on PyTorch's CURRENT stream so
input/engine/consumers are ordered on one stream. Separate stream + sync leaves a
cross-stream race -> intermittent garbage.

Single-object only (num_objects=1): memory_readout/sensory have N=1. selector is
None in the inference path; ignored.

Usage:
    from trt_decoder_hook import install_trt_decoder
    install_trt_decoder(network, "mask_decoder.fp32.engine")
    core = InferenceCore(network, cfg, image_feature_store=store)
"""

import types
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt


PAD_H, PAD_W = 1088, 1920


def _torch_dtype(engine, name):
    return torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype


class _MultiInputTRTEngine:
    """Stream-safe multi-input -> multi-output TRT runner (static shapes).
    Mirrors trt_feature_store._TRTEngine but for several named inputs."""
    def __init__(self, engine_path: str, in_names: List[str], out_names: List[str],
                 device: str = "cuda"):
        self.device = device
        self.in_names = in_names
        self.out_names = out_names
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_dtype = {n: _torch_dtype(self.engine, n) for n in in_names}
        self.out_dtype = {n: _torch_dtype(self.engine, n) for n in out_names}

    @torch.inference_mode()
    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # bind inputs (clone for stable buffer; match engine dtype/layout)
        held = []
        for n in self.in_names:
            x = inputs[n].to(self.device, self.in_dtype[n]).contiguous().clone()
            held.append(x)
            self.ctx.set_input_shape(n, tuple(x.shape))
            self.ctx.set_tensor_address(n, int(x.data_ptr()))
        # static engine: output shapes are fixed; verify resolved
        outs = {}
        for n in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            if any(d < 0 for d in shp):
                raise RuntimeError(f"decoder engine output '{n}' unresolved {shp}")
            t = torch.empty(shp, dtype=self.out_dtype[n], device=self.device).contiguous()
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
            outs[n] = t
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outs


def _host_tail(prob4: torch.Tensor, out_hw) -> torch.Tensor:
    """aggregate -> x4 bilinear upsample -> softmax. prob4 (1,1,h4,w4) sigmoid prob.
    Returns full-res prob (1,2,H,W): [bg, obj]. Matches CUTIE.segment tail."""
    prob = prob4.float()
    new_prob = torch.cat([1 - prob, prob], dim=1).clamp(1e-7, 1 - 1e-7)
    logits = torch.log(new_prob / (1 - new_prob))
    logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
    prob = F.softmax(logits, dim=1)
    if prob.shape[-2:] != tuple(out_hw):
        prob = F.interpolate(prob, size=tuple(out_hw), mode="bilinear", align_corners=False)
    return prob


def install_trt_decoder(network, engine_path: str = "mask_decoder.fp32.engine",
                        device: str = "cuda"):
    """Monkeypatch network.segment to run the TRT decoder engine + host tail.

    Replaces the body of CUTIE.segment: runs the engine for (new_sensory, prob4)
    then reconstructs the (sensory, logits, prob) tuple _segment expects. The
    engine is static at PAD_H x PAD_W; the caller's ms_image_feat must be at the
    matching strides (i.e. internal frame padded to 1088x1920).
    """
    eng = _MultiInputTRTEngine(
        engine_path,
        in_names=["f8", "f4", "memory_readout", "sensory"],
        out_names=["new_sensory", "prob4"],
        device=device,
    )

    def segment(self, ms_image_feat, memory_readout, sensory, *,
                selector=None, chunk_size=-1, update_sensory=True):
        # ms_image_feat = [f16, f8, f4]; decoder body uses f8,f4 only.
        f8 = ms_image_feat[1]
        f4 = ms_image_feat[2]
        H, W = f4.shape[-2] * 4, f4.shape[-1] * 4   # full res from stride-4
        outs = eng({
            "f8": f8, "f4": f4,
            "memory_readout": memory_readout, "sensory": sensory,
        })
        new_sensory = outs["new_sensory"]
        prob4 = outs["prob4"]
        if selector is not None:
            prob4 = prob4 * selector
        prob = _host_tail(prob4, (H, W))            # (1,2,H,W)
        # _segment only consumes index 0 (sensory) and index 2 (prob); logits
        # (index 1) is discarded there, so we return the stride-4 logits proxy.
        return new_sensory, None, prob

    network.segment = types.MethodType(segment, network)
    return eng
