#!/usr/bin/env python3
"""
build_mask_encoder.py  —  Stage 1, Milestone 4  (STATIC full-HD)

Build the mask_encoder TRT engine from mask_encoder.onnx (static 1088x1920).

FP32-FIRST: the one fp32-mandatory region is the SensoryDeepUpdater GRU. Build
without the FP16 flag -> whole graph fp32 -> guaranteed correct. Validate vs the
oracle, then optionally fp16 the ResNet trunk + fuser later for speed.

Static shapes -> no optimization profile needed (TRT infers the single shape).
  image   (1,3,1088,1920)
  pix_feat(1,256,68,120)
  sensory (1,1,256,68,120)
  masks   (1,1,1088,1920)
  -> mask_value (1,1,256,68,120), new_sensory (1,1,256,68,120)

Run:
  python build_mask_encoder.py
Env: CUDA 12.8 + TRT 10.9 on LD_LIBRARY_PATH.
"""

import tensorrt as trt

ONNX_PATH = "mask_encoder.onnx"
ENGINE_PATH = "mask_encoder.fp32.engine"
WORKSPACE_GB = 6


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
    print("[build] FP32 (no FP16 flag) — correctness-first for the fp32 GRU")

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"[build] input '{inp.name}' shape {inp.shape} dtype {inp.dtype}")
        if any(d < 0 for d in inp.shape):
            print(f"  WARNING: '{inp.name}' has dynamic dim {inp.shape} — export should be static.")

    print("[build] building engine (fp32, static)...")
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
        print(f"  [{io}] {name:12s} {str(engine.get_tensor_shape(name)):26s} "
              f"{engine.get_tensor_dtype(name)}")
    print("[done]")


if __name__ == "__main__":
    main()
