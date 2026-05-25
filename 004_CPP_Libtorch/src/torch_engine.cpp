// ============================================================================
// torch_engine.cpp  —  PROTOTYPE libtorch-only backend (see torch_engine.h).
// ============================================================================
#include "torch_engine.h"

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

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
    // load the traced module from the in-memory bytes onto CUDA
    std::string blob(reinterpret_cast<const char*>(data), size);
    std::istringstream iss(blob, std::ios::binary);
    try {
        module_ = torch::jit::load(iss, torch::kCUDA);
        module_.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error("TorchEngine[" + name_ + "] load failed: " + e.what());
    }
}

std::map<std::string, torch::Tensor>
TorchEngine::run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream)
{
    torch::NoGradGuard ng;
    // order all ops on the caller's single stream (same discipline as TRT path)
    auto s = c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
    c10::cuda::CUDAStreamGuard guard(s);

    // map named inputs -> positional args in the order the trace expects
    std::vector<torch::jit::IValue> args;
    args.reserve(inNames_.size());
    for (const auto& n : inNames_) {
        auto it = inputs.find(n);
        if (it == inputs.end())
            throw std::runtime_error("TorchEngine[" + name_ + "] missing input '" + n + "'");
        args.emplace_back(it->second);
    }

    // forward, optionally under autocast fp16 (Cutie's native mode)
    torch::jit::IValue out;
    if (autocastFp16_) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        out = module_.forward(args);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        // CRITICAL: clear the autocast cast-cache every call. Under autocast,
        // libtorch caches fp16 copies of the weights it casts. Without clearing,
        // these accumulate on the GPU each frame (~GBs/frame) until VRAM is
        // exhausted -> allocator thrash -> progressive slowdown. (On Linux this
        // was masked; on Windows the cache is never reclaimed otherwise.)
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