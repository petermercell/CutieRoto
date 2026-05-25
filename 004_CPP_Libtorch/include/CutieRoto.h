// ============================================================================
// CutieRoto.h  —  the Nuke node (STEP 3). Stateful, sequential mask propagation:
// artist rotos a few keyframes (mask inputs), hits Process, the node propagates
// the masklet across the clip via the Cutie TRT+libtorch pipeline, caches every
// frame's alpha, and engine() serves the cache.
//
// Inputs: 0 = RGBA plate; 1..N = keyframe rotos (alpha channel).
// Process is sequential (Cutie carries memory+sensory state forward), so it
// cannot run in engine()'s pull model -> a Process button runs the whole loop
// once via inputAt() frame-pulls, filling a RAM cache that engine() serves.
// ============================================================================
#pragma once

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Knob.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"
#include "DDImage/Interest.h"
#include "DDImage/OutputContext.h"

#include <torch/torch.h>
#include <torch/script.h>
#ifndef CUTIE_LIBTORCH_BACKEND
#include <NvInfer.h>
#endif
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>

#include "memory_core.h"

// Backend selection: by default the TRT engine runner; with
// -DCUTIE_LIBTORCH_BACKEND the pure-libtorch runner (loads traced .pt, no TRT).
#ifdef CUTIE_LIBTORCH_BACKEND
  #include "torch_engine.h"
  namespace cutie { using EngineT = TorchEngine; }
#else
  #include "trt_engine.h"
  namespace cutie { using EngineT = TRTEngine; }
#endif

using namespace DD::Image;

class CutieRoto : public Iop {
public:
    CutieRoto(Node* node);
    ~CutieRoto() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override;

    // inputs: 0 = plate (required), 1 = animated roto (the artist keys it on
    // each keyframe frame; CutieRoto reads it at each frame in the Keyframes knob)
    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 2; }
    static constexpr int ROTO_INPUT = 1;
    bool test_input(int n, Op* op) const override;
    const char* input_label(int n, char* buf) const override;

    void knobs(Knob_Callback f) override;
    int  knob_changed(Knob* k) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t, ChannelMask m, int count) override;
    void _open() override;
    void _close() override;
    void engine(int y, int x, int r, ChannelMask m, Row& row) override;

    static const Iop::Description description;

private:
    // ---- knobs ----
    const char* framesKnob_;     // "1,48,96" : which frame each mask input maps to
    const char* rangeKnob_;      // "1-96"    : display propagation range
    const char* cacheDirKnob_;   // dir for .raw matte cache (persists across sessions)
    bool        matteOnly_;
    bool        invertMatte_;
    bool        pinKeyframes_;   // output artist roto at keyframes (vs prediction)
    // post-process (applied to the served alpha): out = clamp( (alpha*gain)^(1/gamma) )
    float       ppGain_;
    float       ppGamma_;
    bool        ppClamp_;
    int         gpuDevice_;

    // ---- engine geometry (fixed internal, like the Python) ----
    static constexpr int PAD_H = 1088, PAD_W = 1920;   // engine input (padded)
    static constexpr int RESIZE_W = 1920, RESIZE_H = 1080; // resize target before pad
    static constexpr int H16 = 68, W16 = 120;          // stride-16
    static constexpr int MEM_EVERY = 5;
    static constexpr int MAX_MEM_FRAMES = 5;
    static constexpr int TOP_K = 30;

    // ---- pipeline (built in _open) ----
#ifndef CUTIE_LIBTORCH_BACKEND
    std::unique_ptr<cutie::TRTLogger>     trtLog_;
    std::unique_ptr<nvinfer1::IRuntime>   runtime_;
#endif
    std::unique_ptr<cutie::EngineT>       e1_, e2_, e3_, e5_;
    std::unique_ptr<torch::jit::script::Module> ft_;
    cudaStream_t stream_ = nullptr;
    bool pipelineReady_ = false;

    // ---- propagation state (during Process) ----
    cutie::KVStore store_{ MAX_MEM_FRAMES * H16 * W16 };
    torch::Tensor sensory_;      // (1,1,256,68,120)
    torch::Tensor objMemory_;    // (1,1,16,257) accumulator
    torch::Tensor lastMask_;     // (1,1,PAD_H,PAD_W)
    int objCount_ = 0;

    // ---- disk cache: <cacheDir>/<nodename>.<frame>.raw  (header w,h + floats) ----
    // Disk is the source of truth (persists, instance-independent -> fixes the
    // alternating cached=0 from multiple viewer instances). A small RAM map is a
    // per-session fast layer on top.
    // Each cached matte is immutable once produced. We store it behind a
    // shared_ptr so engine() can copy the pointer under the lock, release the
    // lock, and serve from it WITHOUT holding the lock: even if another thread
    // replaces this frame's entry or clear()s the whole map mid-serve, the
    // buffer this thread holds stays alive until its shared_ptr drops. The old
    // code kept a raw pointer into the map-owned vector after unlocking, so a
    // concurrent move-assign on the same key (two engine threads loading the
    // same uncached frame) or a Process clear()/overwrite freed the buffer
    // under the serving thread -> use-after-free. Only the debug build's stderr
    // lock was accidentally serializing it away (the Heisenbug). Dimensions now
    // travel WITH each buffer, so the old shared cacheW_/cacheH_ scalars (which
    // could disagree with a given frame's buffer) are gone.
    struct CachedMatte {
        std::vector<float> a;
        int w = 0, h = 0;
    };
    using MattePtr = std::shared_ptr<const CachedMatte>;
    std::map<int, MattePtr> alphaCache_;   // RAM fast layer (frame -> immutable matte)
    // SpinLock instead of std::mutex: on Windows, std::mutex's CRT-dependent
    // layout can crash when locked across the plugin/libtorch module boundary
    // (different CRT instances). A std::atomic_flag spinlock has no CRT state, so
    // it's ABI-safe everywhere. The cache is only touched briefly (map clear/
    // insert/find), so a spinlock is fine — no long holds.
    struct SpinLock {
        std::atomic_flag f = ATOMIC_FLAG_INIT;
        void lock()   { while (f.test_and_set(std::memory_order_acquire)) {} }
        void unlock() { f.clear(std::memory_order_release); }
    };
    struct SpinGuard {
        SpinLock& s;
        explicit SpinGuard(SpinLock& l) : s(l) { s.lock(); }
        ~SpinGuard() { s.unlock(); }
    };
    SpinLock cacheMutex_;

    // Diagnostic counter of in-flight cache serves in engine(). Lets
    // processAllFrames() observe serve/clear overlap WITHOUT serializing the
    // threads (the way stderr flushing did, which hid the original race). Not a
    // correctness mechanism — the shared_ptr cache makes overlap safe; this just
    // makes it visible.
    std::atomic<int> serving_{0};

    std::string cacheDir() const;                    // resolved dir (knob or default)
    std::string cacheFilePath(int frame) const;      // <dir>/<node>.<frame>.raw
    bool writeRawMatte(int frame, const std::vector<float>& a, int w, int h);
    bool readRawMatte(int frame, std::vector<float>& a, int& w, int& h);

    // ---- frame geometry of the plate (input 0) ----
    int fX_=0, fY_=0, fW_=0, fH_=0;

    // ---- helpers ----
    void buildPipeline();        // _open: runtime + engines + .pt + stream
    void resetState();           // clear store/sensory/obj/lastMask for a fresh run

    // pull input `idx` at `frame` into a CHW float CUDA tensor at PAD_H x PAD_W
    // (RGB for plate idx 0; single-channel alpha for roto idx>=1). Returns
    // undefined tensor on failure.
    torch::Tensor pullInputResized(int idx, double frame, bool wantAlpha);

    // one analyze step: encode a seed roto into permanent memory
    void analyzeKeyframe(const torch::Tensor& imgPad, const torch::Tensor& rotoPad);

    // one display step: propagate a frame -> alpha (PAD res); updates state
    torch::Tensor displayStep(const torch::Tensor& imgPad, bool addMemory);

    // convert padded engine-res alpha (1,1,PAD_H,PAD_W) -> plate-res cpu float
    // (unpad the RESIZE_H..PAD_H rows, then resize RESIZE to plate). Mirrors the
    // inverse of pullInputResized so the matte isn't aspect-distorted.
    std::vector<float> alphaToPlate(const torch::Tensor& alphaPad, int& outW, int& outH);

    // the whole two-phase run: analyze keyframes + propagate every frame
    void processAllFrames();

    // parse "1,48,96" and "1-96"
    std::vector<std::pair<int,int>> parseKeyframes() const;  // (inputIdx, frame)
    bool parseRange(int& first, int& last) const;
};