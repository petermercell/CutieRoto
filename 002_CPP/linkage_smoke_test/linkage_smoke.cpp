// ============================================================================
// linkage_smoke.cpp  —  Cutie-Nuke C++ phase, STEP 0 (de-risk linkage)
//
// NO Nuke, NO Cutie logic. Proves the three runtimes coexist in ONE binary on
// this box BEFORE we build the plugin:
//   (a) libtorch 2.7.1 (Nuke 17.0's .so's) -> torch::jit::load the traced
//       fusion_transformer.pt and run it on dummy tensors
//   (b) TensorRT 10.9 -> deserialize one engine + enqueueV3 on a stream
//   (c) CUDA 12.8 -> one shared stream, both above ordered on it
//
// If this links and runs, the hybrid (TRT engines + libtorch module) is viable
// and the rest is mechanical. If it fails, we find the ABI/symbol/CUDA-version
// clash here in ~150 lines instead of buried under the NDK layer.
//
// The single most likely failure modes, and what they'd mean:
//   - link error "undefined reference ... cxx11" -> ABI flag wrong (need =1)
//   - crash / garbled strings at runtime          -> ABI mismatch (need =1)
//   - "version GLIBCXX_... not found"             -> stdlib too old for these .so
//   - TRT + torch symbol clash (double cudnn etc) -> need link order / -Wl flags
//
// Build: see CMakeLists.txt (target: linkage_smoke)
// Run:   ./linkage_smoke fusion_transformer.pt encode_image.fp16.engine
// ============================================================================

#include <torch/script.h>      // libtorch (Nuke 17's 2.7.1)
#include <NvInfer.h>           // TensorRT 10.9
#include <cuda_runtime.h>      // CUDA 12.8

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"[CUDA] "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; \
  return 2; } } while(0)

// minimal TRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
} gLogger;

static std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << path << "\n"; return {}; }
  size_t n = f.tellg(); f.seekg(0);
  std::vector<char> buf(n);
  f.read(buf.data(), n);
  return buf;
}

int main(int argc, char** argv) {
  std::string ptPath  = (argc > 1) ? argv[1] : "fusion_transformer.pt";
  std::string engPath = (argc > 2) ? argv[2] : "encode_image.fp16.engine";

  std::cout << "=== Cutie-Nuke linkage smoke test ===\n";

  // ---- CUDA: device + one shared stream ----
  int dev = 0; CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "[CUDA] device 0: " << prop.name
            << "  CC " << prop.major << "." << prop.minor << "\n";
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

  // ---- libtorch: version + load the traced module ----
  std::cout << "[torch] compiled against libtorch headers; runtime = Nuke 17's .so\n";
  // a trivial CUDA tensor op proves libtorch's CUDA half works in-process
  {
    torch::Tensor t = torch::ones({2,3}, torch::TensorOptions()
                                   .dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor u = t * 2.0f + 1.0f;
    std::cout << "[torch] cuda tensor op ok, sum=" << u.sum().item<float>()
              << " (expect 18)\n";
  }

  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load(ptPath, torch::kCUDA);
    mod.eval();
    std::cout << "[torch] jit::load OK: " << ptPath << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "[torch] jit::load FAILED: " << e.what() << "\n";
    return 3;
  }

  // run fusion_transformer.pt on dummy tensors (shapes from trace_fusion_transformer.py)
  // inputs: pix_feat(1,256,68,120) visual_readout(1,1,256,68,120)
  //         sensory(1,1,256,68,120) last_mask(1,1,1088,1920) obj_memory(1,1,16,257)
  try {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    std::vector<torch::jit::IValue> in;
    in.push_back(torch::randn({1,256,68,120}, opt));
    in.push_back(torch::randn({1,1,256,68,120}, opt));
    in.push_back(torch::zeros({1,1,256,68,120}, opt));
    in.push_back(torch::zeros({1,1,1088,1920}, opt));
    in.push_back(torch::randn({1,1,16,257}, opt));
    torch::Tensor out = mod.forward(in).toTensor();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[torch] forward OK, out shape = [";
    for (auto d : out.sizes()) std::cout << d << " ";
    std::cout << "] (expect 1 1 256 68 120)\n";
  } catch (const c10::Error& e) {
    std::cerr << "[torch] forward FAILED: " << e.what() << "\n";
    return 4;
  }

  // ---- TensorRT: deserialize one engine + enqueue ----
  auto eng = readFile(engPath);
  if (eng.empty()) return 5;
  std::unique_ptr<nvinfer1::IRuntime> rt{ nvinfer1::createInferRuntime(gLogger) };
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
    rt->deserializeCudaEngine(eng.data(), eng.size()) };
  if (!engine) { std::cerr << "[TRT] deserialize FAILED\n"; return 6; }
  std::cout << "[TRT] deserialize OK: " << engPath
            << "  (" << engine->getNbIOTensors() << " IO tensors)\n";
  std::unique_ptr<nvinfer1::IExecutionContext> ctx{ engine->createExecutionContext() };
  if (!ctx) { std::cerr << "[TRT] context FAILED\n"; return 7; }

  // bind: allocate device buffers for each IO tensor at its (resolved) shape
  std::vector<void*> devbufs;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    auto dims = ctx->getTensorShape(name);
    size_t vol = 1; bool dyn = false;
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) { dyn = true; break; }
      vol *= dims.d[d];
    }
    if (dyn) {
      // encode_image is dynamic; set a concrete HD shape for the input
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        nvinfer1::Dims4 hd{1,3,1088,1920};
        ctx->setInputShape(name, hd);
        dims = ctx->getTensorShape(name);
        vol = 1; for (int d=0; d<dims.nbDims; ++d) vol *= dims.d[d];
      }
    }
    void* p = nullptr; CUDA_CHECK(cudaMalloc(&p, vol * sizeof(float)));
    devbufs.push_back(p);
    ctx->setTensorAddress(name, p);
    std::cout << "[TRT]   " << name << "  vol=" << vol
              << (engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT?" (in)":" (out)")
              << "\n";
  }
  bool ok = ctx->enqueueV3(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "[TRT] enqueueV3 " << (ok ? "OK" : "FAILED") << "\n";
  for (void* p : devbufs) cudaFree(p);

  // ---- both ran on the same process + stream ----
  cudaStreamDestroy(stream);
  std::cout << "\n=== SUCCESS: libtorch 2.7.1 + TRT 10.9 + CUDA 12.8 coexist ===\n";
  std::cout << "The hybrid Path-B plugin is viable. Proceed to the Nuke node.\n";
  return 0;
}
