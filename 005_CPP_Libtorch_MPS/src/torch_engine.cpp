// ============================================================================
// torch_engine.cpp  —  MPS (Apple Metal) libtorch backend. See torch_engine.h.
// ============================================================================
#include "torch_engine.h"

#include <ATen/autocast_mode.h>

#include <sstream>

namespace cutie {

TorchEngine::TorchEngine(const void* data, size_t size, const std::string& name,
                         std::vector<std::string> inNames,
                         std::vector<std::string> outNames,
                         bool autocastFp16)
    : name_(name),
      inNames_(std::move(inNames)),
      outNames_(std::move(outNames)),
      autocastFp16_(autocastFp16)
{
    // load the traced module from the in-memory bytes onto the Apple GPU (MPS).
    // The .pt must be traced device-neutrally (trace on CPU; see
    // trace_all_stages_mps.py) so torch::jit::load can remap params to MPS.
    std::string blob(reinterpret_cast<const char*>(data), size);
    std::istringstream iss(blob, std::ios::binary);
    try {
        // Load on CPU, cast any float64 buffers to float32, THEN move to MPS.
        // MPS has no float64 — moving a double tensor straight to MPS throws
        // "Cannot convert a MPS Tensor ...". The .pt is traced fp32, but some
        // submodules carry double constant buffers (e.g. positional-encoding
        // inv_freq); casting to kFloat32 on CPU first makes the MPS move safe and
        // is exactly the precision we run at anyway.
        module_ = torch::jit::load(iss, torch::kCPU);
        module_.to(torch::kFloat32);
        module_.to(torch::kMPS);
        module_.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error("TorchEngine[" + name_ + "] load failed: " + e.what());
    }
}

std::map<std::string, torch::Tensor>
TorchEngine::run(const std::map<std::string, torch::Tensor>& inputs)
{
    torch::NoGradGuard ng;

    // map named inputs -> positional args in the order the trace expects
    std::vector<torch::jit::IValue> args;
    args.reserve(inNames_.size());
    for (const auto& n : inNames_) {
        auto it = inputs.find(n);
        if (it == inputs.end())
            throw std::runtime_error("TorchEngine[" + name_ + "] missing input '" + n + "'");
        args.emplace_back(it->second);
    }

    // forward. Default MPS path is fp32 (autocast OFF). The autocast branch is
    // kept for parity but targets at::kMPS; enable only once you've confirmed
    // every op in this stage has an MPS autocast kernel.
    torch::jit::IValue out;
    if (autocastFp16_) {
        at::autocast::set_autocast_enabled(at::kMPS, true);
        out = module_.forward(args);
        at::autocast::set_autocast_enabled(at::kMPS, false);
        at::autocast::clear_cache();
    } else {
        out = module_.forward(args);
    }

    // map positional outputs -> names. trace returns a tuple (or single tensor).
    std::vector<torch::Tensor> outs;
    if (out.isTensor()) {
        outs.push_back(out.toTensor());
    } else if (out.isTuple()) {
        for (const auto& e : out.toTuple()->elements())
            outs.push_back(e.toTensor());
    } else {
        throw std::runtime_error("TorchEngine[" + name_ + "] unexpected output type");
    }
    if (outs.size() != outNames_.size())
        throw std::runtime_error("TorchEngine[" + name_ + "] output count " +
            std::to_string(outs.size()) + " != names " + std::to_string(outNames_.size()));

    std::map<std::string, torch::Tensor> result;
    for (size_t i = 0; i < outNames_.size(); ++i)
        result[outNames_[i]] = outs[i].to(torch::kFloat32);   // normalize to fp32 out
    return result;
}

} // namespace cutie
