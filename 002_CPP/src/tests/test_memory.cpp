// ============================================================================
// test_memory.cpp  —  Cutie-Nuke, STEP 2: validate the C++ memory core.
//
// Loads memcase.pt (from dump_memory_read_case.py): the real captured KV buffers
// k/s/v + perm_end_pt, the query key/selection at a frame, and the Python fp32
// reference readout + affinity. Runs the C++ KVStore + memoryRead on the SAME
// inputs and checks it matches the Python reference (target: bit-exact, since
// memory_read_reference.py showed fp32-vs-fp32 = 0.0).
//
// Also exercises KVStore.add + removeOldMemory on synthetic tokens to confirm
// the [perm|temp] layout + FIFO trim match the spec.
//
// Build: target test_memory. Run from dir with memcase.pt.
//   ./test_memory memcase.pt
// ============================================================================

#include "memory_core.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace cutie;

static torch::Tensor buf(torch::jit::script::Module& m, const std::string& n) {
    return m.attr(n).toTensor().to(torch::kCUDA).to(torch::kFloat32);
}

int main(int argc, char** argv) {
    std::string casePath = (argc > 1) ? argv[1] : "memcase.pt";

    cudaStream_t stream; cudaStreamCreate(&stream);
    c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromExternal(stream, 0));
    torch::NoGradGuard ng;
    // No autocast scope is active in this fresh thread, so the raw torch ops below
    // run in true fp32 (matching the Python reference computed with autocast OFF).

    auto mod = torch::jit::load(casePath);
    torch::Tensor k  = buf(mod, "k");           // (1,CK,N)
    torch::Tensor s  = buf(mod, "s");           // (1,1,N)
    torch::Tensor v  = buf(mod, "v");           // (1,CV,N)
    torch::Tensor qk = buf(mod, "query_key");   // (1,CK,h,w)
    torch::Tensor qe = buf(mod, "selection");   // (1,CK,h,w)
    torch::Tensor ref_readout  = buf(mod, "ref_readout");   // (1,CV,HW)
    torch::Tensor ref_affinity = buf(mod, "ref_affinity");  // (1,N,HW)
    int64_t perm = mod.attr("perm_end_pt").toTensor().item<int64_t>();
    int top_k    = mod.attr("top_k").toTensor().item<int64_t>();

    const int64_t N = k.size(-1), CK = k.size(1), CV = v.size(1);
    const int64_t h = qk.size(-2), w = qk.size(-1), HW = h * w;
    std::cout << "[case] N=" << N << " perm=" << perm << " CK=" << CK
              << " CV=" << CV << " h*w=" << h << "x" << w << " top_k=" << top_k << "\n";

    // ---- 1) primitives + full read vs Python reference ----
    auto sim = getSimilarity(k, s, qk, qe);
    auto aff = doSoftmax(sim, top_k);
    auto mem = readout(aff, v);                       // (1,CV,HW)
    cudaStreamSynchronize(stream);

    double aff_err = (aff - ref_affinity).abs().max().item<double>();
    double mem_err = (mem - ref_readout).abs().max().item<double>();
    std::cout << "[read] affinity      max_abs_err = " << aff_err << "\n";
    std::cout << "[read] visual_readout max_abs_err = " << mem_err << "\n";

    // full memoryRead via a KVStore built from the captured buffers
    KVStore store(/*max_work_tokens=*/5 * HW);
    store.k = k; store.s = s; store.v = v; store.perm_end_pt = perm;
    auto vr = memoryRead(store, qk, qe, top_k);       // (1,CV,h,w)
    cudaStreamSynchronize(stream);
    double vr_err = (vr.view({1, CV, HW}) - ref_readout).abs().max().item<double>();
    std::cout << "[read] memoryRead()  max_abs_err = " << vr_err
              << "  shape [" << vr.size(0) << " " << vr.size(1) << " "
              << vr.size(2) << " " << vr.size(3) << "]\n";

    // ---- 2) KVStore add + FIFO trim on synthetic tokens ----
    {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        KVStore st(5 * HW);
        auto mkTok = [&](int64_t n, int64_t C){ return torch::randn({1,C,n}, opt); };
        // 1 permanent keyframe (HW tokens)
        st.add(mkTok(HW,CK), mkTok(HW,1), mkTok(HW,CV), /*permanent=*/true);
        // 7 temporary frames -> 7*HW temp, cap is 5*HW -> after trim temp == 5*HW
        for (int i=0;i<7;++i) st.add(mkTok(HW,CK), mkTok(HW,1), mkTok(HW,CV), false);
        int64_t before = st.k.size(-1);
        st.removeOldMemory();
        int64_t after = st.k.size(-1), tempAfter = after - st.perm_end_pt;
        std::cout << "[store] perm=" << st.perm_end_pt << " (1 kf) tokens before trim="
                  << before << " after=" << after << " temp_after=" << tempAfter
                  << " (cap " << st.max_work_tokens << ")\n";
        bool ok = (st.perm_end_pt == HW) && (tempAfter == 5*HW);
        std::cout << "[store] FIFO " << (ok ? "OK" : "BAD") << "\n";
    }

    bool pass = (aff_err < 1e-3) && (mem_err < 1e-3) && (vr_err < 1e-3);
    cudaStreamSynchronize(stream); cudaStreamDestroy(stream);
    std::cout << "\n=== " << (pass ? "MEMORY CORE OK — matches Python fp32 reference"
                                    : "MISMATCH — read differs from reference") << " ===\n";
    return pass ? 0 : 1;
}
