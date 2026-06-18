// ============================================================================
// CutieRoto.cpp  —  the Nuke node implementation (macOS / MPS build). Process
// runs the full two-phase sequential propagation (analyze keyframes -> propagate
// every frame) on Apple Metal (MPS), caches each frame's matte to disk, and
// engine() serves the cache with live post-process.
//
// Port of 004_CPP_Libtorch's CutieRoto.cpp. The algorithm is unchanged; the
// device backend is MPS:  torch::kMPS replaces torch::kCUDA, and
// torch::mps::synchronize() replaces the CUDA stream syncs. No CUDA streams, no
// CUDA runtime, no TensorRT.
// ============================================================================
#include "CutieRoto.h"
#include "embedded_assets.h"

#include <torch/mps.h>                      // torch::mps::is_available / synchronize
#include <ATen/detail/MPSHooksInterface.h>  // at::detail::getMPSHooks().emptyCache()
#include <sstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>

// debug tracing -> Nuke terminal (stderr). Two levels:
//   CDBG        — lifecycle/Process trace (analyze, propagate, store sizes,
//                 NaN errors). LOW volume, ALWAYS ON.
//   CDBG_ENGINE — per-scanline engine() serve trace. FIREHOSE; OFF unless you set
//                 CUTIE_DEBUG_ENGINE=1 in the environment.
namespace {
inline bool cutieDebugOn() {
    return true;   // lifecycle trace forced on; for opt-in restore the getenv below
    // static const bool on = []{ const char* e = std::getenv("CUTIE_DEBUG");
    //                            return e && e[0] && e[0] != '0'; }();
    // return on;
}
inline bool cutieDebugEngineOn() {
    static const bool on = []{ const char* e = std::getenv("CUTIE_DEBUG_ENGINE");
                               return e && e[0] && e[0] != '0'; }();
    return on;
}
}
#define CDBG(...) do { if (cutieDebugOn()) { std::cerr << "[CutieRoto] "; \
                       std::fprintf(stderr, __VA_ARGS__); \
                       std::fprintf(stderr, "\n"); std::fflush(stderr); } } while(0)
#define CDBG_ENGINE(...) do { if (cutieDebugEngineOn()) { std::cerr << "[CutieRoto] "; \
                       std::fprintf(stderr, __VA_ARGS__); \
                       std::fprintf(stderr, "\n"); std::fflush(stderr); } } while(0)

// MPS_SYNC — on CUDA this was cudaStreamSynchronize() wrapped in CUDA_CHECK to
// turn a sticky fault into a located exception. On MPS the analogue is a plain
// torch::mps::synchronize() (waits for the Metal command buffer to finish). Any
// op error surfaces as a c10::Error thrown from the op itself, which the
// Process path already catches and converts to an Op::error.
#define MPS_SYNC() torch::mps::synchronize()

// ServeScope — increments an atomic for the lifetime of a cache serve in
// engine(). Purely diagnostic (overlap is safe with the shared_ptr cache).
namespace { struct ServeScope {
    std::atomic<int>& c;
    explicit ServeScope(std::atomic<int>& a) : c(a) { c.fetch_add(1, std::memory_order_acq_rel); }
    ~ServeScope() { c.fetch_sub(1, std::memory_order_acq_rel); }
}; }

// ----------------------------------------------------------------------------
// ctor / dtor / help
// ----------------------------------------------------------------------------
CutieRoto::CutieRoto(Node* node) : Iop(node) {
    framesKnob_ = "1";
    rangeKnob_  = "";
    cacheDirKnob_ = "";   // empty -> default <tmp>/cutieroto/<nodename>/
    matteOnly_  = false;
    pinKeyframes_ = false;
    invertMatte_= false;
    ppGain_     = 1.0f;
    ppGamma_    = 1.0f;
    ppClamp_    = true;
    gpuDevice_  = 0;
}

CutieRoto::~CutieRoto() {}

const char* CutieRoto::node_help() const {
    return "CutieRoto — tracked roto mask propagation (Cutie, libtorch on Apple Metal/MPS).\n\n"
           "Connect your plate (input 0) and a single Roto/RotoPaint (input 1).\n"
           "Animate the roto on a few keyframes, list those frames in 'Keyframes'\n"
           "(e.g. '1,22,48,58,62,96'), then press Process. The mask is propagated\n"
           "across the clip and cached to disk; scrub to view.\n\n"
           "Internal resolution 1088x1920 (plate is resized to fit).\n\n"
           "CutieRoto for Nuke by Peter Mercell, 2026 — petermercell.com\n"
           "Cutie by Ho Kei Cheng et al. (https://github.com/hkchengrex/Cutie), MIT.";
}

// ----------------------------------------------------------------------------
// inputs: 0 = plate, 1 = roto keyframes
// ----------------------------------------------------------------------------
bool CutieRoto::test_input(int n, Op* op) const {
    return dynamic_cast<Iop*>(op) != nullptr;   // any image op
}

const char* CutieRoto::input_label(int n, char* buf) const {
    if (n == 0) return "";          // plate (unlabeled, like B-side)
    return "roto";                  // single animated roto input
}

// ----------------------------------------------------------------------------
// knobs
// ----------------------------------------------------------------------------
void CutieRoto::knobs(Knob_Callback f) {
    String_knob(f, &framesKnob_, "keyframes", "Keyframes");
    Tooltip(f, "Comma-separated frames where the roto (input 1) is keyed.\n"
               "e.g. '1,22,48,58,62,96'. CutieRoto reads the roto at each of\n"
               "these frames and propagates the mask across the Range.");

    String_knob(f, &rangeKnob_, "range", "Range");
    Tooltip(f, "Frame range to propagate, e.g. '1-96'. Empty = input clip range.");

    String_knob(f, &cacheDirKnob_, "cache_dir", "Cache Dir");
    Tooltip(f, "Folder for the propagated matte cache (.raw per frame).\n"
               "Empty = <tmp>/cutieroto/<nodename>/. Persists across sessions;\n"
               "delete the folder to force a fresh Process.");

    Button(f, "process", "Process");
    Tooltip(f, "Run the full sequential propagation and cache every frame.");

    Divider(f, "");
    Bool_knob(f, &matteOnly_, "matte_only", "Matte Only (BW)");
    Bool_knob(f, &invertMatte_, "invert", "Invert Matte");
    Bool_knob(f, &pinKeyframes_, "pin_keyframes", "Pin Keyframes To Roto");
    Tooltip(f, "OFF (default): every frame is the model's prediction — fully\n"
               "consistent, no flicker at keyframes.\n"
               "ON: keyframe frames output the exact artist roto instead of the\n"
               "prediction (pixel-accurate at keyframes, but may pop/flicker).\n"
               "Re-Process after changing this.");

    Divider(f, "Matte Post-Process");
    Float_knob(f, &ppGain_, "gain", "Gain");
    Tooltip(f, "Multiply the matte before gamma. Raise to push the soft matte\n"
               "toward solid (the raw Cutie output is a soft probability).");
    SetRange(f, 0.0, 8.0);
    Float_knob(f, &ppGamma_, "gamma", "Gamma");
    Tooltip(f, "Gamma on the matte: out = (alpha*gain) ^ (1/gamma).\n"
               "Low gamma (e.g. 0.1) crisps edges toward a hard matte; 1.0 = soft.");
    SetRange(f, 0.01, 4.0);
    Bool_knob(f, &ppClamp_, "clamp", "Clamp 0-1");
    Tooltip(f, "Clamp the result to [0,1] after gain/gamma.");
    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");
    SetFlags(f, Knob::HIDDEN);   // ignored on MPS (single Apple GPU)

    Divider(f, "");
    Text_knob(f, "CutieRoto for Nuke by Peter Mercell, 2026\n"
                 "petermercell.com\n\n"
                 "Cutie by Ho Kei Cheng et al. (MIT)\n"
                 "github.com/hkchengrex/Cutie");
}

int CutieRoto::knob_changed(Knob* k) {
    if (!k) return Iop::knob_changed(k);
    CDBG("knob_changed: '%s'", k->name().c_str());
    if (std::string(k->name()) == "process") {
        CDBG("-> process button");
        processAllFrames();
        return 1;
    }
    return Iop::knob_changed(k);
}

// ----------------------------------------------------------------------------
// parse helpers
// ----------------------------------------------------------------------------
std::vector<std::pair<int,int>> CutieRoto::parseKeyframes() const {
    std::vector<std::pair<int,int>> out;
    std::string s = framesKnob_ ? framesKnob_ : "";
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        int frame = std::atoi(tok.substr(a, b - a + 1).c_str());
        out.emplace_back(ROTO_INPUT, frame);   // always read input 1 at this frame
    }
    return out;
}

bool CutieRoto::parseRange(int& first, int& last) const {
    std::string s = rangeKnob_ ? rangeKnob_ : "";
    if (!s.empty()) {
        int a, b;
        if (std::sscanf(s.c_str(), "%d-%d", &a, &b) == 2 && b >= a) {
            first = a; last = b; return true;
        }
    }
    int cur = (int)outputContext().frame();
    first = cur; last = cur;
    return false;
}

// ----------------------------------------------------------------------------
// build + tear down the pipeline
// ----------------------------------------------------------------------------
void CutieRoto::buildPipeline() {
    if (pipelineReady_) return;
    // Validate the Apple GPU up front so a missing/old-macOS MPS fails with a
    // clear message at Process time instead of crashing deep in libtorch.
    if (!torch::mps::is_available())
        throw std::runtime_error(
            "MPS (Apple Metal GPU) is not available. Requires Apple Silicon, "
            "macOS 12.3+, and an MPS-enabled libtorch (Nuke 17's bundled torch).");
    CDBG("buildPipeline: MPS ok");

    // pure-libtorch backend: load the traced .pt stages onto MPS. IO name lists
    // match the original engines' so every e*_->run({{"name",t}}) call is unchanged.
    e1_ = std::make_unique<cutie::EngineT>(cutie::asset_e1().data, cutie::asset_e1().size, "encode_image",
            std::vector<std::string>{"image"},
            std::vector<std::string>{"f16","f8","f4","pix_feat"});
    e2_ = std::make_unique<cutie::EngineT>(cutie::asset_e2().data, cutie::asset_e2().size, "transform_key",
            std::vector<std::string>{"f16"},
            std::vector<std::string>{"key","shrinkage","selection"});
    e3_ = std::make_unique<cutie::EngineT>(cutie::asset_e3().data, cutie::asset_e3().size, "mask_encoder",
            std::vector<std::string>{"image","pix_feat","sensory","masks"},
            std::vector<std::string>{"mask_value","new_sensory"});
    e5_ = std::make_unique<cutie::EngineT>(cutie::asset_e5().data, cutie::asset_e5().size, "mask_decoder",
            std::vector<std::string>{"f8","f4","memory_readout","sensory"},
            std::vector<std::string>{"new_sensory","prob4"});

    cutie::Blob ftb = cutie::asset_ft();
    std::string bytes(reinterpret_cast<const char*>(ftb.data), ftb.size);
    std::istringstream ss(bytes);
    // Same float64->float32 guard as TorchEngine: load on CPU, cast doubles to
    // float32, then move to MPS (MPS rejects float64; the fusion transformer's
    // positional encoding carries double buffers).
    auto ftMod = torch::jit::load(ss, torch::kCPU);
    ftMod.to(torch::kFloat32);
    ftMod.to(torch::kMPS);
    ft_ = std::make_unique<torch::jit::script::Module>(std::move(ftMod));
    ft_->eval();

    pipelineReady_ = true;
}

void CutieRoto::_open() {
    // Do NOT build the GPU pipeline here. engine() only serves the cached matte
    // (RAM or disk) and needs no engines. processAllFrames() builds the pipeline
    // on demand when Process is pressed.
}

void CutieRoto::_close() {
    // keep the cache; engines live with the node
}

void CutieRoto::resetState() {
    store_ = cutie::KVStore(MAX_MEM_FRAMES * H16 * W16);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS);
    sensory_   = torch::zeros({1,1,256,H16,W16}, opt);
    objMemory_ = torch::zeros({1,1,16,257}, opt);
    lastMask_  = torch::zeros({1,1,PAD_H,PAD_W}, opt);
    objCount_  = 0;
}

// ----------------------------------------------------------------------------
// disk cache: <cacheDir>/matte.<frame>.raw   (int32 w, int32 h, w*h floats)
// ----------------------------------------------------------------------------
std::string CutieRoto::cacheDir() const {
    std::string d = (cacheDirKnob_ && cacheDirKnob_[0]) ? cacheDirKnob_ : "";
    if (d.empty()) {
        std::string nm = node_name();
        std::error_code ec;
        std::filesystem::path tmp = std::filesystem::temp_directory_path(ec);
        if (ec) tmp = ".";
        d = (tmp / "cutieroto" / (nm.empty() ? std::string("CutieRoto") : nm)).string();
    }
    return d;
}

std::string CutieRoto::cacheFilePath(int frame) const {
    char buf[32]; std::snprintf(buf, sizeof(buf), "%06d", frame);
    std::filesystem::path p = std::filesystem::path(cacheDir()) / ("matte." + std::string(buf) + ".raw");
    return p.string();
}

bool CutieRoto::writeRawMatte(int frame, const std::vector<float>& a, int w, int h) {
    std::error_code ec;
    std::filesystem::create_directories(cacheDir(), ec);
    std::string path = cacheFilePath(frame);
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { CDBG("writeRawMatte: cannot open %s", path.c_str()); return false; }
    int32_t hdr[2] = { w, h };
    std::fwrite(hdr, sizeof(int32_t), 2, fp);
    std::fwrite(a.data(), sizeof(float), (size_t)w * h, fp);
    std::fclose(fp);
    return true;
}

bool CutieRoto::readRawMatte(int frame, std::vector<float>& a, int& w, int& h) {
    std::string path = cacheFilePath(frame);
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    int32_t hdr[2];
    if (std::fread(hdr, sizeof(int32_t), 2, fp) != 2) { std::fclose(fp); return false; }
    w = hdr[0]; h = hdr[1];
    if (w <= 0 || h <= 0 || w > 100000 || h > 100000) { std::fclose(fp); return false; }
    a.resize((size_t)w * h);
    size_t got = std::fread(a.data(), sizeof(float), a.size(), fp);
    std::fclose(fp);
    return got == a.size();
}

// ----------------------------------------------------------------------------
// frame-pull: read input `idx` at `frame` and resize to PAD_H x PAD_W.
// plate (idx 0): RGB -> (1,3,PAD_H,PAD_W). roto (idx>=1): alpha -> (1,1,PAD_H,PAD_W).
// ----------------------------------------------------------------------------
torch::Tensor CutieRoto::pullInputResized(int idx, double frame, bool wantAlpha) {
    OutputContext oc = outputContext();
    oc.setFrame(frame);
    Op* op = inputAt(idx, oc);
    if (!op) return torch::Tensor();
    Iop* iop = dynamic_cast<Iop*>(op);
    if (!iop) return torch::Tensor();

    iop->validate(true);
    Box b = iop->info().box();
    const int w = b.w(), h = b.h(), x0 = b.x(), y0 = b.y();
    if (w <= 0 || h <= 0) return torch::Tensor();

    ChannelSet need = wantAlpha ? Mask_Alpha : Mask_RGB;
    iop->request(x0, y0, x0 + w, y0 + h, need, 1);

    // fetch into a CPU buffer (Nuke rows are bottom-up; we flip to top-down)
    const int C = wantAlpha ? 1 : 3;
    std::vector<float> host((size_t)C * w * h);
    Channel rgb[3] = { Chan_Red, Chan_Green, Chan_Blue };
    for (int yy = 0; yy < h; ++yy) {
        Row row(x0, x0 + w);
        iop->get(y0 + yy, x0, x0 + w, need, row);
        const int dy = h - 1 - yy;   // flip to top-down
        if (wantAlpha) {
            const float* a = row[Chan_Alpha] + x0;
            std::memcpy(&host[(size_t)dy * w], a, sizeof(float) * w);
        } else {
            for (int c = 0; c < 3; ++c) {
                const float* src = row[rgb[c]] + x0;
                std::memcpy(&host[((size_t)c * h + dy) * w], src, sizeof(float) * w);
            }
        }
    }

    auto cpu = torch::from_blob(host.data(), {1, C, h, w}, torch::kFloat32).clone();

    // Match the VALIDATED Python geometry exactly:
    //   plate (any WxH) -> resize to RESIZE_W x RESIZE_H (1920x1080) -> PAD the
    //   height to PAD_H (1088) with zeros (8 rows at the bottom). NO vertical
    //   stretch. RESIZE_W == PAD_W (1920), so only the height is padded.
    //
    // MPS: do the resize + pad ON CPU, then upload the small result. The plate
    // downscale uses kArea (adaptive-avg-pool); MPS's adaptive pool requires the
    // input dims to be divisible by the output dims (e.g. 3096->1080, 4448->1920
    // are NOT) and throws "Adaptive pool MPS: ...". CPU has no such constraint and
    // is bit-for-bit the validated path. Resizing before the upload also means we
    // transfer a 1088x1920 tensor instead of the full plate (faster). kNearest
    // (roto) is done on CPU here too for the same one-shot upload.
    namespace F = torch::nn::functional;
    torch::Tensor resized;
    if (wantAlpha) {
        resized = F::interpolate(cpu,
            F::InterpolateFuncOptions().size(std::vector<int64_t>{RESIZE_H, RESIZE_W})
            .mode(torch::kNearest));                     // (1,1,1080,1920)
    } else {
        resized = F::interpolate(cpu,
            F::InterpolateFuncOptions().size(std::vector<int64_t>{RESIZE_H, RESIZE_W})
            .mode(torch::kArea));                        // (1,3,1080,1920)
    }
    int padB = PAD_H - RESIZE_H;                          // 8
    auto out = F::pad(resized, F::PadFuncOptions({0, 0, 0, padB})
                      .mode(torch::kConstant).value(0));  // (1,C,1088,1920)
    return out.to(torch::kMPS).contiguous();             // upload the small result
}

// ----------------------------------------------------------------------------
// pipeline stages (reuse the proven runner + memory core)
// ----------------------------------------------------------------------------
void CutieRoto::analyzeKeyframe(const torch::Tensor& imgPad, const torch::Tensor& rotoPad) {
    torch::NoGradGuard ng;
    auto o1 = e1_->run({{"image", imgPad}});
    auto o2 = e2_->run({{"f16", o1["f16"]}});
    auto o3 = e3_->run({{"image", imgPad}, {"pix_feat", o1["pix_feat"]},
                        {"sensory", sensory_}, {"masks", rotoPad}});
    // E3 mask_value is (1, num_obj=1, CV=256, h, w). The KV store value needs
    // (1, CV, N) with N = h*w. Squeeze the object dim and flatten spatial dims.
    auto mv = o3["mask_value"];                          // (1,1,256,h,w)
    mv = mv.squeeze(1).flatten(2);                       // (1,256,N)
    store_.add(o2["key"], o2["shrinkage"], mv, /*permanent=*/true);
    sensory_ = o3["new_sensory"];
    lastMask_ = rotoPad;
    objCount_++;
    MPS_SYNC();
}

torch::Tensor CutieRoto::displayStep(const torch::Tensor& imgPad, bool addMemory) {
    torch::NoGradGuard ng;
    auto nanchk = [](const char* tag, const torch::Tensor& t) {
        bool n = torch::isnan(t).any().item<bool>();
        bool inf = torch::isinf(t).any().item<bool>();
        if (n || inf) {
            std::cerr << "[CutieRoto] NaN/Inf at " << tag
                      << " (nan=" << n << " inf=" << inf << ")\n"; std::fflush(stderr);
        }
        return n || inf;
    };
    auto o1 = e1_->run({{"image", imgPad}});
    auto o2 = e2_->run({{"f16", o1["f16"]}});
    MPS_SYNC();
    nanchk("E1.pix_feat", o1["pix_feat"]);
    nanchk("E2.key", o2["key"]); nanchk("E2.selection", o2["selection"]);
    // fp32 memory read (device-agnostic; runs on MPS because the store is on MPS)
    auto visual = cutie::memoryRead(store_, o2["key"], o2["selection"], TOP_K); // (1,256,h,w)
    MPS_SYNC();
    nanchk("memoryRead", visual);
    CDBG("displayStep: visual readout [%lld %lld %lld %lld] store N=%lld perm=%lld",
         (long long)visual.size(0), (long long)visual.size(1),
         (long long)visual.size(2), (long long)visual.size(3),
         (long long)(store_.k.defined() ? store_.k.size(-1) : 0L),
         (long long)store_.perm_end_pt);
    auto visual5 = visual.view({1,1,256,H16,W16});
    // libtorch fusion + transformer (.pt) — runs on MPS (inputs are MPS tensors)
    std::vector<torch::jit::IValue> ftin{ o1["pix_feat"], visual5, sensory_,
                                          lastMask_, objMemory_ };
    torch::Tensor readout = ft_->forward(ftin).toTensor();   // (1,1,256,h,w)
    MPS_SYNC();
    nanchk("fusion.readout", readout);
    // decoder
    auto o5 = e5_->run({{"f8", o1["f8"]}, {"f4", o1["f4"]},
                        {"memory_readout", readout}, {"sensory", sensory_}});
    MPS_SYNC();
    nanchk("E5.prob4", o5["prob4"]);
    sensory_ = o5["new_sensory"];
    // prob4 (1,1,272,480) -> upsample to PAD, that's the alpha at engine res
    namespace F = torch::nn::functional;
    auto prob = o5["prob4"];
    auto alphaPad = F::interpolate(prob,
        F::InterpolateFuncOptions().size(std::vector<int64_t>{PAD_H, PAD_W})
        .mode(torch::kBilinear).align_corners(false));
    lastMask_ = alphaPad.detach();
    if (addMemory) {
        auto o3 = e3_->run({{"image", imgPad}, {"pix_feat", o1["pix_feat"]},
                            {"sensory", sensory_}, {"masks", alphaPad}});
        auto mv = o3["mask_value"].squeeze(1).flatten(2);   // (1,256,N)
        store_.add(o2["key"], o2["shrinkage"], mv, /*permanent=*/false);
        store_.removeOldMemory();
        sensory_ = o3["new_sensory"];
    }
    MPS_SYNC();
    // Bound the unified-memory footprint: the fp32 memory read builds large
    // (1,N,HW) intermediates whose size grows with the store, so the MPS caching
    // allocator's reserved pool can creep up. Reclaiming once per frame keeps it
    // flat. (No portable free-memory query on MPS, so this is unconditional —
    // cheap relative to a full propagation step.) The C++ entry point is the MPS
    // hooks interface; torch::mps::empty_cache() is Python-only.
    at::detail::getMPSHooks().emptyCache();
    return alphaPad;   // (1,1,PAD_H,PAD_W)
}

// Convert padded engine-res alpha (1,1,PAD_H,PAD_W) to plate-res top-down float.
std::vector<float> CutieRoto::alphaToPlate(const torch::Tensor& alphaPad,
                                           int& outW, int& outH) {
    namespace F = torch::nn::functional;
    using torch::indexing::Slice;
    // crop the 8 padded bottom rows: (1,1,PAD_H,PAD_W) -> (1,1,RESIZE_H,RESIZE_W)
    auto cropped = alphaPad.index({Slice(), Slice(), Slice(0, RESIZE_H), Slice(0, RESIZE_W)});
    int pw = fW_ > 0 ? fW_ : RESIZE_W;
    int ph = fH_ > 0 ? fH_ : RESIZE_H;
    auto plate = F::interpolate(cropped.contiguous(),
        F::InterpolateFuncOptions().size(std::vector<int64_t>{ph, pw})
        .mode(torch::kBilinear).align_corners(false))
        .squeeze().to(torch::kCPU).contiguous();
    outW = pw; outH = ph;
    std::vector<float> a((size_t)pw * ph);
    std::memcpy(a.data(), plate.data_ptr<float>(), sizeof(float) * a.size());
    return a;
}

// ----------------------------------------------------------------------------
// the full sequential Process — analyze all keyframes into permanent memory,
// then propagate every frame in [first..last] in order, caching each.
// ----------------------------------------------------------------------------
void CutieRoto::processAllFrames() {
    CDBG("processAllFrames: ENTER");
    if (!input(0)) { Op::error("CutieRoto: no plate on input 0"); return; }
    try { buildPipeline(); }
    catch (const std::exception& e) { Op::error("CutieRoto buildPipeline: %s", e.what()); return; }

    if (Iop* p0 = dynamic_cast<Iop*>(input(0))) {
        p0->validate(true);
        Box b = p0->info().box();
        if (b.w() > 0 && b.h() > 0) {
            fX_ = b.x(); fY_ = b.y(); fW_ = b.w(); fH_ = b.h();
        }
    }
    CDBG("plate dims for cache: fW=%d fH=%d", fW_, fH_);

    auto kfs = parseKeyframes();
    if (kfs.empty()) { Op::error("CutieRoto: no keyframes set"); return; }

    int first, last;
    parseRange(first, last);
    if (!(rangeKnob_ && rangeKnob_[0])) {
        first = kfs.front().second; last = kfs.back().second;
        for (auto& kf : kfs) { first = std::min(first, kf.second); last = std::max(last, kf.second); }
    }
    if (last < first) std::swap(first, last);
    CDBG("range %d..%d, %zu keyframes", first, last, kfs.size());

    resetState();
    {   // clear cache for a fresh run (safe under concurrent serves: shared_ptr)
        const int inflight = serving_.load(std::memory_order_acquire);
        if (inflight > 0)
            CDBG("clear() with %d engine serve(s) in flight (safe: shared_ptr keeps buffers alive)", inflight);
        SpinGuard lk(cacheMutex_);
        alphaCache_.clear();
    }

    if (!input(ROTO_INPUT)) {
        Op::error("CutieRoto: connect a Roto/RotoPaint to the 'roto' input (1)");
        progressDismiss(); return;
    }

    std::map<int,int> kfAtFrame;
    for (auto& kf : kfs) kfAtFrame[kf.second] = ROTO_INPUT;

    const int total = last - first + 1;
    int done = 0;
    progressMessage("CutieRoto: propagating %d frames...", total);

    // ---- ANALYZE: build permanent memory from every keyframe first ----
    for (auto& kf : kfs) {
        if (aborted()) { CDBG("aborted in analyze"); progressDismiss(); return; }
        int fr = kf.second;
        torch::Tensor img, roto;
        try {
            img  = pullInputResized(0, fr, false);
            roto = pullInputResized(ROTO_INPUT, fr, true);
        } catch (const std::exception& e) {
            Op::error("CutieRoto analyze pull kf %d: %s", fr, e.what()); progressDismiss(); return;
        }
        if (!img.defined() || !roto.defined()) { CDBG("analyze pull failed kf %d", fr); continue; }
        try { analyzeKeyframe(img, roto); }
        catch (const std::exception& e) { Op::error("CutieRoto analyze kf %d: %s", fr, e.what()); progressDismiss(); return; }
        CDBG("analyzed keyframe frame=%d (perm now %lld)", fr, (long long)store_.perm_end_pt);
    }

    // ---- DISPLAY: propagate every frame in order ----
    for (int fr = first; fr <= last; ++fr) {
        if (aborted()) { CDBG("aborted at frame %d", fr); break; }
        torch::Tensor img;
        try { img = pullInputResized(0, fr, false); }
        catch (const std::exception& e) { Op::error("CutieRoto display pull frame %d: %s", fr, e.what()); break; }
        if (!img.defined()) { CDBG("display pull failed frame %d", fr); continue; }

        torch::Tensor alphaPad;
        auto kfIt = kfAtFrame.find(fr);
        if (pinKeyframes_ && kfIt != kfAtFrame.end() && input(kfIt->second)) {
            torch::Tensor roto;
            try { roto = pullInputResized(kfIt->second, fr, true); }
            catch (const std::exception& e) { Op::error("CutieRoto pin pull frame %d: %s", fr, e.what()); break; }
            if (roto.defined()) {
                alphaPad = roto;
                lastMask_ = roto;
                CDBG("frame %d is keyframe -> pinned to artist roto", fr);
            }
        }
        if (!alphaPad.defined()) {
            bool addMem = ((fr - first) % MEM_EVERY == 0);
            try { alphaPad = displayStep(img, addMem); }
            catch (const std::exception& e) { Op::error("CutieRoto display frame %d: %s", fr, e.what()); break; }
        }

        int pw = 0, ph = 0;
        std::vector<float> a = alphaToPlate(alphaPad, pw, ph);
        writeRawMatte(fr, a, pw, ph);            // disk = source of truth
        {
            auto m = std::make_shared<CachedMatte>();
            m->a = std::move(a); m->w = pw; m->h = ph;
            SpinGuard lk(cacheMutex_);
            alphaCache_[fr] = std::move(m);
        }
        ++done;
        progressFraction(done, total, Op::StatusModal);
        progressMessage("CutieRoto: frame %d/%d", done, total);
        if ((fr % 10) == 0) CDBG("propagated frame %d (cache=%zu)", fr, alphaCache_.size());
    }

    progressDismiss();
    CDBG("processAllFrames: DONE, cached %zu frames", alphaCache_.size());
    asapUpdate();
}

// ----------------------------------------------------------------------------
// _validate / _request / engine
// ----------------------------------------------------------------------------
void CutieRoto::_validate(bool for_real) {
    if (input(0)) {
        input(0)->validate(for_real);
        info_ = input(0)->info();
        fX_ = info_.x(); fY_ = info_.y(); fW_ = info_.w(); fH_ = info_.h();
    } else {
        info_.set(0, 0, PAD_W, PAD_H);
    }
    info_.channels(Mask_RGBA);
    set_out_channels(Mask_RGBA);
}

void CutieRoto::_request(int x, int y, int r, int t, ChannelMask m, int count) {
    if (input(0)) input(0)->request(x, y, r, t, m, count);
}

void CutieRoto::engine(int y, int x, int r, ChannelMask m, Row& row) {
    // pass plate RGB through; serve cached alpha for this frame (RAM, else disk).
    const int frame = (int)outputContext().frame();
    CDBG_ENGINE("engine ENTER: frame=%d y=%d x=%d r=%d", frame, y, x, r);
    MattePtr cached;
    {
        SpinGuard lk(cacheMutex_);
        auto it = alphaCache_.find(frame);
        if (it != alphaCache_.end()) cached = it->second;   // refcount bump only
    }
    if (!cached) {
        std::vector<float> a; int w = 0, h = 0;
        if (readRawMatte(frame, a, w, h)) {
            auto mm = std::make_shared<CachedMatte>();
            mm->a = std::move(a); mm->w = w; mm->h = h;
            SpinGuard lk(cacheMutex_);
            auto& slot = alphaCache_[frame];
            if (!slot) slot = std::move(mm);
            cached = slot;
        }
    }
    const int cw = cached ? cached->w : 0;
    const int ch = cached ? cached->h : 0;
    ServeScope _serve(serving_);
    static std::atomic<int> lastLoggedFrame{-99999};
    if (y == fY_ && lastLoggedFrame.exchange(frame, std::memory_order_relaxed) != frame) {
        CDBG_ENGINE("engine: frame=%d cached=%d cw=%d ch=%d fW=%d fH=%d",
             frame, (int)(cached != nullptr), cw, ch, fW_, fH_);
    }

    if (input(0)) {
        CDBG_ENGINE("engine[y=%d x=%d r=%d]: input0->get", y, x, r);
        input(0)->get(y, x, r, m, row);   // RGB passthrough
    } else {
        foreach(z, m) { float* o = row.writable(z) + x; std::memset(o, 0, sizeof(float)*(r-x)); }
    }

    CDBG_ENGINE("engine[y=%d]: get writable Chan_Alpha (x=%d r=%d)", y, x, r);
    float* outA = row.writable(Chan_Alpha) + x;
    CDBG_ENGINE("engine[y=%d]: outA=%p (r-x=%d)", y, (void*)outA, r - x);
    const float invG = (ppGamma_ > 1e-6f) ? (1.0f / ppGamma_) : 1.0f;
    const bool doG = (ppGain_ != 1.0f) || (ppGamma_ != 1.0f);
    auto post = [&](float v) -> float {
        if (doG) {
            v *= ppGain_;
            if (v < 0.f) v = 0.f;
            if (invG != 1.0f) v = std::pow(v, invG);
        }
        if (ppClamp_) v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        if (invertMatte_) v = 1.f - v;
        return v;
    };
    if (cached && cw > 0 && ch > 0 && cached->a.size() == (size_t)cw * (size_t)ch
        && fW_ > 0 && fH_ > 0) {
        CDBG_ENGINE("engine[y=%d]: SERVE cached size=%zu cw=%d ch=%d fX=%d fY=%d fW=%d fH=%d",
             y, cached->a.size(), cw, ch, fX_, fY_, fW_, fH_);
        const int py = y - fY_;
        if (py >= 0 && py < fH_) {
            const int dyTop = (fH_ - 1) - py;
            int sy = (fH_ > 1) ? (int)((int64_t)dyTop * ch / fH_) : 0;
            if (sy < 0) sy = 0;
            if (sy >= ch) sy = ch - 1;
            CDBG_ENGINE("engine[y=%d]: py=%d dyTop=%d sy=%d srcRow=cached+%zu", y, py, dyTop, sy, (size_t)sy*cw);
            const float* srcRow = cached->a.data() + (size_t)sy * cw;
            for (int xi = x; xi < r; ++xi) {
                int px = xi - fX_;
                float v = 0.f;
                if (px >= 0 && px < fW_) {
                    int sx = (cw == fW_) ? px
                           : (fW_ > 1 ? (int)((int64_t)px * cw / fW_) : 0);
                    if (sx < 0) sx = 0;
                    if (sx >= cw) sx = cw - 1;
                    v = srcRow[sx];
                }
                outA[xi - x] = post(v);
            }
            CDBG_ENGINE("engine[y=%d]: serve loop done", y);
        } else {
            std::memset(outA, 0, sizeof(float) * (r - x));
        }
    } else {
        std::memset(outA, 0, sizeof(float) * (r - x));   // not processed yet
    }

    if (matteOnly_) {
        float* rr = row.writable(Chan_Red) + x;
        float* gg = row.writable(Chan_Green) + x;
        float* bb = row.writable(Chan_Blue) + x;
        for (int xi = 0; xi < r - x; ++xi) rr[xi] = gg[xi] = bb[xi] = outA[xi];
    }
}

// ----------------------------------------------------------------------------
// registration
// ----------------------------------------------------------------------------
static Iop* build(Node* node) { return new CutieRoto(node); }
const Iop::Description CutieRoto::description("CutieRoto", "AI/CutieRoto", build);
