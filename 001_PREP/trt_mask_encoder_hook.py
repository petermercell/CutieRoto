#!/usr/bin/env python3
"""
trt_mask_encoder_hook.py  —  Milestone 4 seam

Routes Cutie's mask encoder (CUTIE.encode_mask body) through the static-HD TRT
engine mask_encoder.fp32.engine. The ObjectSummarizer (which produces the object
summaries the transformer consumes) stays in PyTorch for now -- it's logically
E4's input prep, so we keep it on the model and run it on the engine's mask_value.

Installed by monkeypatching network.encode_mask. InferenceCore._add_memory calls
self.network.encode_mask(image, pix_feat, sensory, prob, deep_update, chunk_size,
need_weights) and consumes (msk_value, sensory, obj_value, _).

Engine (build with build_mask_encoder.py), STATIC 1088x1920:
  inputs : image (1,3,1088,1920) raw, pix_feat (1,256,68,120),
           sensory (1,1,256,68,120) fp32, masks (1,1,1088,1920)
  outputs: mask_value (1,1,256,68,120), new_sensory (1,1,256,68,120) fp32

Image normalization + others=zeros are baked into the engine wrapper, so we feed
the RAW image and the prob/roto directly (single object).

CRITICAL (M1 stream-race lesson): engine runs on torch CURRENT stream.

Single-object only. selector/num_objects=1.

Usage:
    from trt_mask_encoder_hook import install_trt_mask_encoder
    install_trt_mask_encoder(network, "mask_encoder.fp32.engine")
"""

import types
from typing import List, Dict

import numpy as np
import torch
import tensorrt as trt


def _torch_dtype(engine, name):
    return torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype


class _MultiInputTRTEngine:
    """Stream-safe multi-input -> multi-output TRT runner (static shapes).
    Same pattern as trt_decoder_hook._MultiInputTRTEngine."""
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
        held = []
        for n in self.in_names:
            x = inputs[n].to(self.device, self.in_dtype[n]).contiguous().clone()
            held.append(x)
            self.ctx.set_input_shape(n, tuple(x.shape))
            self.ctx.set_tensor_address(n, int(x.data_ptr()))
        outs = {}
        for n in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            if any(d < 0 for d in shp):
                raise RuntimeError(f"mask_encoder engine output '{n}' unresolved {shp}")
            t = torch.empty(shp, dtype=self.out_dtype[n], device=self.device).contiguous()
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
            outs[n] = t
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outs


def install_trt_mask_encoder(network, engine_path: str = "mask_encoder.fp32.engine",
                             device: str = "cuda"):
    """Monkeypatch network.encode_mask to run the TRT mask-encoder engine, then the
    (still-PyTorch) ObjectSummarizer for object summaries the transformer needs."""
    eng = _MultiInputTRTEngine(
        engine_path,
        in_names=["image", "pix_feat", "sensory", "masks"],
        out_names=["mask_value", "new_sensory"],
        device=device,
    )

    def encode_mask(self, image, ms_features, sensory, masks, *,
                    deep_update=True, chunk_size=-1, need_weights=False):
        # ms_features here is pix_feat (the loose naming in CUTIE.encode_mask).
        pix_feat = ms_features
        outs = eng({
            "image": image, "pix_feat": pix_feat,
            "sensory": sensory, "masks": masks,
        })
        mask_value = outs["mask_value"]
        new_sensory = outs["new_sensory"]
        # ObjectSummarizer stays in PyTorch (E4's input prep); run on engine value.
        if self.object_transformer_enabled:
            object_summaries, object_logits = self.object_summarizer(
                masks, mask_value, need_weights)
        else:
            object_summaries, object_logits = None, None
        return mask_value, new_sensory, object_summaries, object_logits

    network.encode_mask = types.MethodType(encode_mask, network)
    return eng
