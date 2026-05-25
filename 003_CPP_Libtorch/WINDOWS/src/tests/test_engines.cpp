// ============================================================================
// test_engines.cpp  —  Cutie-Nuke, STEP 1: TRTEngine runner + multi-engine chain
//
// NO Nuke. Loads all 4 engines via TRTEngine, runs the real wiring on dummy
// tensors, verifies every handoff shape against the known IO map:
//   E1 encode_image:  image(1,3,1088,1920) -> f16,f8,f4,pix_feat
//   E2 transform_key: f16                   -> key,shrinkage,selection
//   E3 mask_encoder:  image,pix_feat,sensory,masks -> mask_value,new_sensory
//   E5 mask_decoder:  f8,f4,memory_readout,sensory -> new_sensory,prob4
// plus the fusion_transformer.pt: pix_feat,visual_readout,sensory,last_mask,
//   obj_memory -> readout_memory(1,1,256,68,120) which feeds E5's memory_readout.
//
// All on ONE shared cuda stream (the M1 discipline). If shapes line up and
// nothing throws/NaNs, the runner is correct and we wrap it in the Nuke node.
//
// Build: target test_engines (see CMakeLists). Run from the dir with the engines
//        + fusion_transformer.pt.
//   ./test_engines   (paths default to cwd; or pass a dir arg)
// ============================================================================

#include "trt_engine.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

using namespace cutie;

static std::vector<char> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p);
    size_t n = f.tellg(); f.seekg(0);
    std::vector<char> b(n); f.read(b.data(), n); return b;
}

static void showShape(const std::string& tag, const torch::Tensor& t) {
    std::cout << "    " << tag << " [";
    for (auto d : t.sizes()) std::cout << d << " ";
    std::cout << "] " << (t.dtype() == torch::kFloat32 ? "f32" : "?")
              << " nan=" << (torch::isnan(t).any().item<bool>() ? "YES" : "no") << "\n";
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? std::string(argv[1]) + "/" : "";
    const int H = 1088, W = 1920, h = 68, w = 120;

    TRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> rt{ nvinfer1::createInferRuntime(logger) };

    // load 4 engines
    auto b1 = readFile(dir + "encode_image.fp16.engine");
    auto b2 = readFile(dir + "transform_key.fp16.engine");
    auto b3 = readFile(dir + "mask_encoder.fp32.engine");
    auto b5 = readFile(dir + "mask_decoder.fp32.engine");
    TRTEngine e1(b1.data(), b1.size(), "encode_image", rt.get());
    TRTEngine e2(b2.data(), b2.size(), "transform_key", rt.get());
    TRTEngine e3(b3.data(), b3.size(), "mask_encoder", rt.get());
    TRTEngine e5(b5.data(), b5.size(), "mask_decoder", rt.get());
    std::cout << "[load] 4 engines OK\n";

    // load the .pt (fusion + transformer)
    torch::jit::script::Module ft = torch::jit::load(dir + "fusion_transformer.pt", torch::kCUDA);
    ft.eval();
    std::cout << "[load] fusion_transformer.pt OK\n";

    cudaStream_t stream; cudaStreamCreate(&stream);
    // Make libtorch ops use OUR stream (the M1 single-stream discipline).
    // Use the function form (stable across torch versions) rather than an RAII
    // guard whose class name varies. getStreamFromExternal wraps our raw stream.
    auto torchStream = c10::cuda::getStreamFromExternal(stream, 0);
    c10::cuda::setCurrentCUDAStream(torchStream);

    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor image   = torch::rand({1,3,H,W}, opt);
    torch::Tensor masks   = torch::zeros({1,1,H,W}, opt);
    masks.index_put_({0,0,torch::indexing::Slice(300,600),torch::indexing::Slice(400,800)}, 1.0f);
    torch::Tensor sensory = torch::zeros({1,1,256,h,w}, opt);

    std::cout << "\n[E1] encode_image\n";
    auto o1 = e1.run({{"image", image}}, stream);
    cudaStreamSynchronize(stream);
    showShape("f16", o1["f16"]); showShape("f8", o1["f8"]);
    showShape("f4", o1["f4"]);   showShape("pix_feat", o1["pix_feat"]);

    std::cout << "[E2] transform_key\n";
    auto o2 = e2.run({{"f16", o1["f16"]}}, stream);
    cudaStreamSynchronize(stream);
    showShape("key", o2["key"]); showShape("shrinkage", o2["shrinkage"]);
    showShape("selection", o2["selection"]);

    std::cout << "[E3] mask_encoder (analyze: seed roto)\n";
    auto o3 = e3.run({{"image", image}, {"pix_feat", o1["pix_feat"]},
                      {"sensory", sensory}, {"masks", masks}}, stream);
    cudaStreamSynchronize(stream);
    showShape("mask_value", o3["mask_value"]); showShape("new_sensory", o3["new_sensory"]);

    std::cout << "[.pt] fusion+transformer\n";
    // dummy visual_readout (would come from the fp32 memory read) + obj_memory
    torch::Tensor visual_readout = torch::randn({1,1,256,h,w}, opt);
    torch::Tensor last_mask = masks.clone();
    torch::Tensor obj_memory = torch::randn({1,1,16,257}, opt);
    std::vector<torch::jit::IValue> ftin{ o1["pix_feat"], visual_readout, sensory,
                                          last_mask, obj_memory };
    torch::Tensor readout_memory = ft.forward(ftin).toTensor();
    cudaStreamSynchronize(stream);
    showShape("readout_memory", readout_memory);

    std::cout << "[E5] mask_decoder\n";
    auto o5 = e5.run({{"f8", o1["f8"]}, {"f4", o1["f4"]},
                      {"memory_readout", readout_memory}, {"sensory", sensory}}, stream);
    cudaStreamSynchronize(stream);
    showShape("new_sensory", o5["new_sensory"]); showShape("prob4", o5["prob4"]);

    // expected shape checks
    auto chk = [](const char* tag, const torch::Tensor& t, std::vector<int64_t> exp) {
        bool ok = (t.sizes().vec() == exp);
        std::cout << "  [" << (ok ? "OK " : "BAD") << "] " << tag << "\n";
        return ok;
    };
    std::cout << "\n[verify handoff shapes]\n";
    bool all = true;
    all &= chk("E1.f8       (1,512,136,240)", o1["f8"], {1,512,136,240});
    all &= chk("E1.f4       (1,256,272,480)", o1["f4"], {1,256,272,480});
    all &= chk("E1.pix_feat (1,256,68,120)",  o1["pix_feat"], {1,256,h,w});
    all &= chk("E2.key      (1,64,68,120)",   o2["key"], {1,64,h,w});
    all &= chk("E3.mask_val (1,1,256,68,120)",o3["mask_value"], {1,1,256,h,w});
    all &= chk(".pt readout (1,1,256,68,120)",readout_memory, {1,1,256,h,w});
    all &= chk("E5.prob4    (1,1,272,480)",   o5["prob4"], {1,1,272,480});

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    std::cout << "\n=== " << (all ? "ALL SHAPES OK — runner + chain validated"
                                   : "SHAPE MISMATCH — see BAD above") << " ===\n";
    return all ? 0 : 1;
}
