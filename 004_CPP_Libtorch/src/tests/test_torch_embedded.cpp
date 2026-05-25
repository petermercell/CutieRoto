// ============================================================================
// test_torch_embedded.cpp — libtorch-backend smoke test (no Nuke).
//
// The libtorch analogue of test_engines + test_embedded: load all 5 embedded
// assets the way the plugin does in buildPipeline() — 4 TorchEngines from the
// objcopy blobs + fusion_transformer.pt from an in-memory stream — then run the
// full e1 -> e2 -> e3 -> ft -> e5 chain on dummy tensors and verify the handoff
// shapes. Proves the traced .pt blobs load, the name<->positional IO mapping is
// correct, and the chain wires up, all without Nuke.
//
// Build: target test_torch_embedded (libtorch backend only). The assets are
// linked into the binary (EMBED_OBJS), so it runs standalone on a CUDA box:
//   ./build/test_torch_embedded
// ============================================================================
#include "embedded_assets.h"
#include "torch_engine.h"

#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

using namespace cutie;

static void showShape(const char* tag, const torch::Tensor& t) {
    std::cout << "    " << tag << " [";
    for (auto d : t.sizes()) std::cout << d << " ";
    std::cout << "] nan=" << (torch::isnan(t).any().item<bool>() ? "YES" : "no") << "\n";
}

int main() {
    const int H = 1088, W = 1920, h = 68, w = 120;
    std::cout << "=== libtorch embedded-assets smoke test ===\n";
    std::cout << "[sizes] e1=" << asset_e1().size << " e2=" << asset_e2().size
              << " e3=" << asset_e3().size << " e5=" << asset_e5().size
              << " ft=" << asset_ft().size << " bytes\n";

    // One CUDA stream from libtorch's own pool — matches the plugin's libtorch
    // path (a toolkit-created stream belongs to a different cudart and crashes
    // when handed to libtorch; libtorch's pool stream keeps one runtime).
    auto torchStream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, 0);
    c10::cuda::setCurrentCUDAStream(torchStream);
    cudaStream_t stream = torchStream.stream();

    // 4 TorchEngines from embedded bytes, IO names matching CutieRoto.cpp.
    TorchEngine e1(asset_e1().data, asset_e1().size, "encode_image",
                   {"image"}, {"f16", "f8", "f4", "pix_feat"});
    TorchEngine e2(asset_e2().data, asset_e2().size, "transform_key",
                   {"f16"}, {"key", "shrinkage", "selection"});
    TorchEngine e3(asset_e3().data, asset_e3().size, "mask_encoder",
                   {"image", "pix_feat", "sensory", "masks"}, {"mask_value", "new_sensory"});
    TorchEngine e5(asset_e5().data, asset_e5().size, "mask_decoder",
                   {"f8", "f4", "memory_readout", "sensory"}, {"new_sensory", "prob4"});
    std::cout << "[load] 4 TorchEngines constructed from embedded .pt OK\n";

    // ft from an in-memory stream (identical to buildPipeline()).
    Blob ftb = asset_ft();
    std::string bytes(reinterpret_cast<const char*>(ftb.data), ftb.size);
    std::istringstream ss(bytes);
    torch::jit::script::Module ft = torch::jit::load(ss, torch::kCUDA);
    ft.eval();
    std::cout << "[load] fusion_transformer.pt loaded from embedded stream OK\n";

    torch::NoGradGuard ng;
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor image   = torch::rand({1, 3, H, W}, opt);
    torch::Tensor masks   = torch::zeros({1, 1, H, W}, opt);
    masks.index_put_({0, 0, torch::indexing::Slice(300, 600),
                            torch::indexing::Slice(400, 800)}, 1.0f);
    torch::Tensor sensory = torch::zeros({1, 1, 256, h, w}, opt);

    std::cout << "\n[E1] encode_image\n";
    auto o1 = e1.run({{"image", image}}, stream);
    cudaStreamSynchronize(stream);
    showShape("pix_feat", o1["pix_feat"]);

    std::cout << "[E2] transform_key\n";
    auto o2 = e2.run({{"f16", o1["f16"]}}, stream);
    cudaStreamSynchronize(stream);
    showShape("key", o2["key"]);

    std::cout << "[E3] mask_encoder (analyze: seed roto)\n";
    auto o3 = e3.run({{"image", image}, {"pix_feat", o1["pix_feat"]},
                      {"sensory", sensory}, {"masks", masks}}, stream);
    cudaStreamSynchronize(stream);
    showShape("mask_value", o3["mask_value"]);

    std::cout << "[.pt] fusion+transformer\n";
    torch::Tensor visual_readout = torch::randn({1, 1, 256, h, w}, opt);
    torch::Tensor last_mask      = masks.clone();
    torch::Tensor obj_memory     = torch::randn({1, 1, 16, 257}, opt);
    std::vector<torch::jit::IValue> ftin{ o1["pix_feat"], visual_readout, sensory,
                                          last_mask, obj_memory };
    torch::Tensor readout_memory = ft.forward(ftin).toTensor();
    cudaStreamSynchronize(stream);
    showShape("readout_memory", readout_memory);

    std::cout << "[E5] mask_decoder\n";
    auto o5 = e5.run({{"f8", o1["f8"]}, {"f4", o1["f4"]},
                      {"memory_readout", readout_memory}, {"sensory", sensory}}, stream);
    cudaStreamSynchronize(stream);
    showShape("prob4", o5["prob4"]);

    auto chk = [](const char* tag, const torch::Tensor& t, std::vector<int64_t> exp) {
        bool ok = (t.sizes().vec() == exp);
        std::cout << "  [" << (ok ? "OK " : "BAD") << "] " << tag << "\n";
        return ok;
    };
    std::cout << "\n[verify handoff shapes]\n";
    bool all = true;
    all &= chk("E1.f8       (1,512,136,240)", o1["f8"], {1, 512, 136, 240});
    all &= chk("E1.f4       (1,256,272,480)", o1["f4"], {1, 256, 272, 480});
    all &= chk("E1.pix_feat (1,256,68,120)",  o1["pix_feat"],   {1, 256, h, w});
    all &= chk("E2.key      (1,64,68,120)",   o2["key"],        {1, 64, h, w});
    all &= chk("E3.mask_val (1,1,256,68,120)",o3["mask_value"], {1, 1, 256, h, w});
    all &= chk(".pt readout (1,1,256,68,120)",readout_memory,   {1, 1, 256, h, w});
    all &= chk("E5.prob4    (1,1,272,480)",   o5["prob4"],      {1, 1, 272, 480});

    cudaStreamSynchronize(stream);
    std::cout << "\n=== " << (all ? "LIBTORCH EMBEDDED OK — all 5 assets load + chain validated"
                                  : "SHAPE MISMATCH — see BAD above") << " ===\n";
    return all ? 0 : 1;
}
