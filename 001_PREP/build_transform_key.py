#!/usr/bin/env python3
"""
build_transform_key.py  —  Stage 1, Milestone 2

Build the transform_key TRT engine from transform_key.onnx.
Input is the stride-16 feature (1, 1024, h, w) where h,w = H/16, W/16.

For full-HD path: H,W max = 1088,1920 -> stride-16 max = 68,120.
Profile uses the stride-16 spatial dims:
  min  = 1x1024x 16x 16
  opt  = 1x1024x 68x120
  max  = 1x1024x 68x120

fp16 is safe (KeyProjection has no autocast-disabled region). Shrinkage stays
fp32 on output because the ONNX declares it fp32 (feeds the fp32 memory read).

Run:
  python build_transform_key.py
Env: CUDA 12.8 + TRT 10.9 on LD_LIBRARY_PATH.
"""

import tensorrt as trt

ONNX_PATH = "transform_key.onnx"
ENGINE_PATH = "transform_key.fp16.engine"

# stride-16 spatial dims (= input H/16, W/16)
MIN_HW = (16, 16)
OPT_HW = (68, 120)   # 1088/16=68, 1920/16=120
MAX_HW = (68, 120)
IN_CH = 1024
WORKSPACE_GB = 2


def main():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing {ONNX_PATH}")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  PARSE ERROR:", parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    print(f"[build] input '{inp.name}' shape {inp.shape} dtype {inp.dtype}")
    profile.set_shape(inp.name,
                      min=(1, IN_CH, *MIN_HW),
                      opt=(1, IN_CH, *OPT_HW),
                      max=(1, IN_CH, *MAX_HW))
    config.add_optimization_profile(profile)

    print("[build] building engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"[build] wrote {ENGINE_PATH}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    print("[build] engine I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        io = "IN " if mode == trt.TensorIOMode.INPUT else "OUT"
        print(f"  [{io}] {name:10s} {str(engine.get_tensor_shape(name)):24s} "
              f"{engine.get_tensor_dtype(name)}")
    print("[done]")


if __name__ == "__main__":
    main()
