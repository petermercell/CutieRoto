// ============================================================================
// trt_engine.h  —  Cutie-Nuke, the TRT engine runner (port of Python
// _MultiInputTRTEngine). Binds torch::Tensor CUDA buffers to named TRT IO
// tensors, runs enqueueV3 on a SHARED cuda stream. All IO is fp32 (confirmed:
// even the fp16 engines E1/E2 have FLOAT IO; fp16 is internal only).
//
// Handles both dynamic engines (E1/E2: spatial dims -1, need setInputShape) and
// static engines (E3/E5: fixed HD). Outputs are freshly-allocated torch CUDA
// tensors at the context-resolved shape, so they interop directly with the
// libtorch memory-read + the fusion_transformer .pt, all on the one stream.
//
// STREAM DISCIPLINE (M1 lesson): the caller owns ONE stream and passes it in;
// every engine + every libtorch op orders on it. We do NOT create per-engine
// streams (that was the cross-stream race that randomly blacked frames).
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace cutie {

// Minimal TRT logger (warnings+).
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override;
};

class TRTEngine {
public:
    // Deserialize from an in-memory buffer (embedded via objcopy, or file-read).
    // `name` is just for error messages.
    TRTEngine(const void* data, size_t size, const std::string& name,
              nvinfer1::IRuntime* runtime);
    ~TRTEngine();

    // Run the engine. `inputs` maps IO-tensor-name -> CUDA fp32 tensor. Returns
    // outputs by name as freshly-allocated CUDA fp32 tensors. Ordered on `stream`.
    // Caller must keep input tensors alive across the call (we hold refs anyway).
    std::map<std::string, torch::Tensor>
    run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream);

    const std::vector<std::string>& inputNames()  const { return inNames_; }
    const std::vector<std::string>& outputNames() const { return outNames_; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::unique_ptr<nvinfer1::ICudaEngine>      engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    std::vector<std::string> inNames_;
    std::vector<std::string> outNames_;
    bool hasDynamic_ = false;
};

} // namespace cutie
