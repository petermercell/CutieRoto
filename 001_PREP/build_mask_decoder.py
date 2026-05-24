#!/usr/bin/env python3
"""
build_mask_decoder.py  —  Stage 1, Milestone 3  (STATIC full-HD)

Build the mask_decoder TRT engine from mask_decoder.onnx (static 1088x1920).

FP32-FIRST: two fp32-mandatory regions (pred conv + SensoryUpdater GRU). Build
without the FP16 flag -> whole graph fp32 -> guaranteed correct. Validate vs the
oracle, then optionally fp16 the safe parts later.

Static shapes: the ONNX has fixed dims, so no optimization profile is needed
(TRT infers the single shape from the ONNX). All inputs fixed at the HD strides.

Run:
  python build_mask_decoder.py
Env: CUDA 12.8 + TRT 10.9 on LD_LIBRARY_PATH.
"""

import tensorrt as trt

ONNX_PATH = "mask_decoder.onnx"
ENGINE_PATH = "mask_decoder.fp32.engine"
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
    print("[build] FP32 (no FP16 flag) — correctness-first for fp32 regions")

    # static shapes: report inputs (no profile needed when fully static)
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"[build] input '{inp.name}' shape {inp.shape} dtype {inp.dtype}")
        # if any input still has a -1 dim, the ONNX wasn't fully static
        if any(d < 0 for d in inp.shape):
            print(f"  WARNING: '{inp.name}' has dynamic dim {inp.shape} — "
                  f"export should have been static. A profile would be needed.")

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
        print(f"  [{io}] {name:14s} {str(engine.get_tensor_shape(name)):28s} "
              f"{engine.get_tensor_dtype(name)}")
    print("[done]")


if __name__ == "__main__":
    main()
