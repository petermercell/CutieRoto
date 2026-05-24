#!/usr/bin/env python3
"""
build_engine.py  —  Stage 1, Milestone 1 (part 2/3)

Build a TensorRT 10.9 engine from encode_image.onnx using the Python API, with a
dynamic-shape optimization profile and fp16 enabled.

The encoder is a plain ResNet CNN -- NOT in the fp32-NaN set -- so fp16 is safe
and fast here. (The fp32-only regions are the memory read / sensory GRU / decoder
pred / object summarizer, which are separate engines/C++ later.)

Profile (H,W are the PADDED input dims, multiples of 16):
  min  =  1x3x 256x 256
  opt  =  1x3x 544x 960   (tune to your typical plate)
  max  =  1x3x1088x1920   (covers up to ~1080p long edge; raise if needed)
Raise max if you feed larger plates; bigger max => more build time + workspace.

Run:
  python build_engine.py
Env (must match the toolchain):
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/opt/TensorRT-10.9.0.34/lib:$LD_LIBRARY_PATH
"""

import tensorrt as trt

ONNX_PATH = "encode_image.onnx"
ENGINE_PATH = "encode_image.fp16.engine"

# (min), (opt), (max) — channels fixed at 3, batch fixed at 1
MIN_HW = (256, 256)
OPT_HW = (544, 960)
MAX_HW = (1088, 1920)

WORKSPACE_GB = 4


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
    else:
        print("[build] FP16 not available, building FP32")

    # dynamic shape profile on the single input "image"
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    print(f"[build] input '{inp.name}' shape {inp.shape} dtype {inp.dtype}")
    profile.set_shape(
        inp.name,
        min=(1, 3, *MIN_HW),
        opt=(1, 3, *OPT_HW),
        max=(1, 3, *MAX_HW),
    )
    config.add_optimization_profile(profile)

    print("[build] building engine (this can take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"[build] wrote {ENGINE_PATH}")

    # quick deserialize sanity check
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    print(f"[build] engine I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        io = "IN " if mode == trt.TensorIOMode.INPUT else "OUT"
        print(f"  [{io}] {name:9s} {str(shape):24s} {dtype}")
    print("[done] engine ready")


if __name__ == "__main__":
    main()
