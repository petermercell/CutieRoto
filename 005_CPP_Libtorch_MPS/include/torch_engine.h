// ============================================================================
// torch_engine.h  —  MPS (Apple Metal) libtorch-only backend.
//
// macOS port of the pure-libtorch TorchEngine. Same role as the CUDA version:
// load a traced TorchScript .pt and expose run(map<name,Tensor>) so the rest of
// the plugin is backend-agnostic. The ONLY differences from the CUDA build are:
//
//   * modules are loaded onto torch::kMPS (Apple GPU) instead of torch::kCUDA;
//   * there is NO CUDA stream parameter — MPS has no user-managed stream in the
//     libtorch C++ API, so ordering is handled by the single default MPS stream
//     and explicit torch::mps::synchronize() at the points the CUDA build synced;
//   * autocast fp16 is OFF by default. On MPS we run fp32 for correctness on the
//     first working build (MPS autocast coverage is narrower than CUDA's). The
//     flag is kept for interface parity and can opt into at::kMPS autocast later.
//
// Outputs are still mapped positionally (a traced module returns a tuple) using
// the name lists passed at construction, exactly like the CUDA/TRT runners.
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace cutie {

class TorchEngine {
public:
    // Load a traced .pt from an in-memory buffer (disk-loaded on Mac; see
    // embedded_assets.h) or file. inNames/outNames give the positional<->name
    // mapping (match the original engine IO names so the pipeline is unchanged).
    // autocastFp16 defaults to FALSE on the MPS build (fp32 path).
    TorchEngine(const void* data, size_t size, const std::string& name,
                std::vector<std::string> inNames,
                std::vector<std::string> outNames,
                bool autocastFp16 = false);

    // Same name-mapped run as the CUDA runner, MINUS the stream argument (MPS
    // has no user stream handle). Maps inputs by name to positional args,
    // forwards through the module, maps tuple outputs back to names.
    std::map<std::string, torch::Tensor>
    run(const std::map<std::string, torch::Tensor>& inputs);

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
