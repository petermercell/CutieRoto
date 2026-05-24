#!/usr/bin/env python3
"""
trt_feature_store.py  —  Milestone 1+2 seam

Drop-in replacement for Cutie's ImageFeatureStore that runs encode_image AND
transform_key on TensorRT engines. The rest of the pipeline (mask encode, memory
read, transformer, decode) stays PyTorch for now.

This is the single seam where the encoder/key feed the rest of the pipeline
(inference_core calls store.get_features / store.get_key). Buffers use torch CUDA
tensors via .data_ptr() (no pycuda) -- the pattern the C++ plugin will use too.

Engines (build with build_engine.py / build_transform_key.py):
  encode_image.fp16.engine   : image(1,3,H,W)[0,1] -> f16,f8,f4,pix_feat
  transform_key.fp16.engine  : f16(1,1024,h,w)     -> key,shrinkage,selection

CRITICAL (Milestone 1 lesson): the engine must run on PyTorch's CURRENT stream so
input upload, engine, and downstream consumers are ordered on one stream. Running
on a separate stream + synchronize() leaves a cross-stream race that intermittently
reads half-written output buffers -> random black frames.

Env: CUDA 12.8 + TRT 10.9 on LD_LIBRARY_PATH.
"""

import warnings
from typing import Iterable, List, Dict

import numpy as np
import torch
import tensorrt as trt

from cutie.model.cutie import CUTIE


def _torch_dtype(engine, name):
    return torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype


class _TRTEngine:
    """Generic single-input-name -> multiple-named-outputs TRT runner.
    Stream-safe (runs on torch current stream). Allocates fresh output tensors
    each call (returned, so they stay alive); clones the input for a stable buffer.
    """
    def __init__(self, engine_path: str, in_name: str, out_names: List[str],
                 device: str = "cuda"):
        self.device = device
        self.in_name = in_name
        self.out_names = out_names
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.in_dtype = _torch_dtype(self.engine, in_name)
        self.out_dtype = {n: _torch_dtype(self.engine, n) for n in out_names}

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(self.device, self.in_dtype).contiguous().clone()
        ok = self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        probe = self.ctx.get_tensor_shape(self.out_names[0])
        if not ok or any(d < 0 for d in probe):
            raise RuntimeError(
                f"TRT engine '{self.in_name}' rejected input {tuple(x.shape)} "
                f"(out of optimization profile). Rebuild with larger MAX_HW or "
                f"downscale the input.")
        self.ctx.set_tensor_address(self.in_name, int(x.data_ptr()))

        outs = {}
        for n in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            t = torch.empty(shp, dtype=self.out_dtype[n], device=self.device).contiguous()
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
            outs[n] = t

        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outs


class TRTImageFeatureStore:
    """encode_image + transform_key on TRT; rest of pipeline in PyTorch."""

    def __init__(self, network: CUTIE,
                 encode_engine: str = "encode_image.fp16.engine",
                 key_engine: str = "transform_key.fp16.engine",
                 no_warning: bool = False):
        self.network = network
        self.encoder = _TRTEngine(encode_engine, "image",
                                  ["f16", "f8", "f4", "pix_feat"])
        self.keyer = _TRTEngine(key_engine, "f16",
                                ["key", "shrinkage", "selection"])
        self._store = {}
        self.no_warning = no_warning

    def _encode_feature(self, index: int, image: torch.Tensor) -> None:
        o = self.encoder(image)
        ms_features = [o["f16"], o["f8"], o["f4"]]
        pix_feat = o["pix_feat"]
        k = self.keyer(o["f16"])
        key, shrinkage, selection = k["key"], k["shrinkage"], k["selection"]
        self._store[index] = (ms_features, pix_feat, key, shrinkage, selection)

    def get_features(self, index: int, image: torch.Tensor) -> (Iterable[torch.Tensor], torch.Tensor):
        if index not in self._store:
            self._encode_feature(index, image)
        return self._store[index][:2]

    def get_key(self, index: int, image: torch.Tensor):
        if index not in self._store:
            self._encode_feature(index, image)
        return self._store[index][2:]

    def delete(self, index: int) -> None:
        if index in self._store:
            del self._store[index]

    def __len__(self):
        return len(self._store)

    def __del__(self):
        if len(self._store) > 0 and not self.no_warning:
            warnings.warn(f'Leaking {self._store.keys()} in the TRT image feature store')
