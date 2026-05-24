// ============================================================================
// test_embedded.cpp  —  Cutie-Nuke, STEP 3-pre: prove the embedded assets.
//
// Builds with the 5 objcopy blobs linked in. Confirms:
//   - the 4 TRT engines deserialize from embedded bytes (TRTEngine(ptr,size))
//   - fusion_transformer.pt loads from an in-memory stream (no disk file)
// This is the mechanic the Nuke node uses in _open(). No Nuke yet.
//
// Build: target test_embedded (links e1..e5,ft .o). Run: ./test_embedded
// ============================================================================
#include "embedded_assets.h"
#include "trt_engine.h"
#include <torch/script.h>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <iostream>
#include <memory>

using namespace cutie;

int main() {
    std::cout << "=== embedded assets test ===\n";
    std::cout << "[sizes] e1=" << asset_e1().size << " e2=" << asset_e2().size
              << " e3=" << asset_e3().size << " e5=" << asset_e5().size
              << " ft=" << asset_ft().size << " bytes\n";

    TRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> rt{ nvinfer1::createInferRuntime(logger) };

    // 4 engines from embedded bytes
    TRTEngine e1(asset_e1().data, asset_e1().size, "encode_image", rt.get());
    TRTEngine e2(asset_e2().data, asset_e2().size, "transform_key", rt.get());
    TRTEngine e3(asset_e3().data, asset_e3().size, "mask_encoder", rt.get());
    TRTEngine e5(asset_e5().data, asset_e5().size, "mask_decoder", rt.get());
    std::cout << "[TRT] 4 engines deserialized from embedded bytes OK\n";
    std::cout << "      e1 outputs:"; for (auto& n : e1.outputNames()) std::cout << " " << n;
    std::cout << "\n      e5 inputs :"; for (auto& n : e5.inputNames())  std::cout << " " << n;
    std::cout << "\n";

    // .pt from in-memory stream (the embedding-vs-disk-agnostic load)
    {
        Blob ft = asset_ft();
        std::string bytes(reinterpret_cast<const char*>(ft.data), ft.size);
        std::istringstream ss(bytes);
        torch::jit::script::Module mod = torch::jit::load(ss, torch::kCUDA);
        mod.eval();
        std::cout << "[torch] fusion_transformer.pt loaded from embedded stream OK\n";
    }

    std::cout << "\n=== EMBEDDED ASSETS OK — node _open() can load all 5 from .so ===\n";
    return 0;
}
