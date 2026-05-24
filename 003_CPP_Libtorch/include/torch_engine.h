// ============================================================================
// torch_engine.h  —  PROTOTYPE: libtorch-only backend (no TensorRT).
//
// Drop-in replacement for TRTEngine: loads a traced TorchScript .pt (from
// trace_all_stages.py) and exposes the SAME run(map<name,Tensor>, stream)
// interface, so CutieRoto.cpp doesn't change — it just constructs a TorchEngine
// instead of a TRTEngine when built with the libtorch backend.
//
// A traced module returns outputs POSITIONALLY (a tuple). We map input names ->
// positional args and output positions -> names using the name lists passed at
// construction (the same names the TRT engine used). So the rest of the pipeline
// (memory core, fusion_transformer.pt, decoder) sees identical tensors by name.
//
// Precision: the .pt holds fp32 weights. Run under autocast fp16 (Cutie's native
// mode, fast) or fp32 (bit-exact) via the `autocastFp16` flag. Same shared stream
// discipline as the TRT path — every op orders on the caller's one stream.
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace cutie {

class TorchEngine {
public:
    // Load a traced .pt from an in-memory buffer (embedded via objcopy) or file.
    // inNames/outNames give the positional<->name mapping (match the TRT engine's
    // IO names so the pipeline is backend-agnostic).
    TorchEngine(const void* data, size_t size, const std::string& name,
                std::vector<std::string> inNames,
                std::vector<std::string> outNames,
                bool autocastFp16 = true);

    // Same signature as TRTEngine::run. Maps inputs by name to positional args,
    // forwards through the module, maps outputs back to names. Ordered on stream.
    std::map<std::string, torch::Tensor>
    run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream);

    const std::vector<std::string>& inputNames()  const { return inNames_; }
    const std::vector<std::string>& outputNames() const { return outNames_; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    torch::jit::script::Module module_;
    std::vector<std::string> inNames_;
    std::vector<std::string> outNames_;
    bool autocastFp16_;
};

} // namespace cutie
