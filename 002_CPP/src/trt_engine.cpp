// ============================================================================
// trt_engine.cpp  —  TRTEngine runner implementation
// ============================================================================
#include "trt_engine.h"
#include <iostream>

namespace cutie {

void TRTLogger::log(Severity s, const char* msg) noexcept {
    if (s <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

TRTEngine::TRTEngine(const void* data, size_t size, const std::string& name,
                     nvinfer1::IRuntime* runtime)
    : name_(name)
{
    engine_.reset(runtime->deserializeCudaEngine(data, size));
    if (!engine_)
        throw std::runtime_error("TRTEngine: deserialize failed for " + name);
    ctx_.reset(engine_->createExecutionContext());
    if (!ctx_)
        throw std::runtime_error("TRTEngine: context creation failed for " + name);

    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* tn = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(tn);
        if (mode == nvinfer1::TensorIOMode::kINPUT)  inNames_.emplace_back(tn);
        else                                          outNames_.emplace_back(tn);
        // detect dynamic dims on this tensor
        auto d = engine_->getTensorShape(tn);
        for (int k = 0; k < d.nbDims; ++k)
            if (d.d[k] < 0) hasDynamic_ = true;
    }
}

TRTEngine::~TRTEngine() = default;

std::map<std::string, torch::Tensor>
TRTEngine::run(const std::map<std::string, torch::Tensor>& inputs,
               cudaStream_t stream)
{
    // 1) bind inputs: set shape (if dynamic) + tensor address. We require fp32,
    //    contiguous, CUDA. (All engine IO is FLOAT.)
    std::vector<torch::Tensor> held;   // keep refs alive through enqueue
    for (const auto& nm : inNames_) {
        auto it = inputs.find(nm);
        if (it == inputs.end())
            throw std::runtime_error(name_ + ": missing input '" + nm + "'");
        torch::Tensor x = it->second;
        TORCH_CHECK(x.is_cuda(), name_, ": input '", nm, "' must be CUDA");
        x = x.to(torch::kFloat32).contiguous();
        held.push_back(x);

        if (hasDynamic_) {
            nvinfer1::Dims d;
            d.nbDims = x.dim();
            for (int k = 0; k < x.dim(); ++k) d.d[k] = (int)x.size(k);
            ctx_->setInputShape(nm.c_str(), d);
        }
        ctx_->setTensorAddress(nm.c_str(), x.data_ptr());
    }

    // 2) allocate outputs at the context-resolved shape, bind addresses
    std::map<std::string, torch::Tensor> outputs;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (const auto& nm : outNames_) {
        auto d = ctx_->getTensorShape(nm.c_str());
        std::vector<int64_t> shape;
        for (int k = 0; k < d.nbDims; ++k) {
            if (d.d[k] < 0)
                throw std::runtime_error(name_ + ": output '" + nm +
                                         "' has unresolved dim");
            shape.push_back(d.d[k]);
        }
        torch::Tensor out = torch::empty(shape, opts).contiguous();
        ctx_->setTensorAddress(nm.c_str(), out.data_ptr());
        outputs[nm] = out;
    }

    // 3) enqueue on the SHARED stream (no sync here; caller orders work on stream)
    if (!ctx_->enqueueV3(stream))
        throw std::runtime_error(name_ + ": enqueueV3 failed");

    return outputs;
}

} // namespace cutie
