// ============================================================================
// embedded_assets.h  —  macOS asset access, two modes:
//
//   CUTIE_EMBED_MODELS defined  -> the 5 .pt are baked into the dylib via
//       .incbin (src/embedded_assets_mac.S). Self-contained ~222MB dylib, no
//       external model files needed. This is the default build.
//
//   CUTIE_EMBED_MODELS undefined -> load the 5 .pt from disk at runtime
//       ($CUTIE_MODEL_DIR, models/ next to the plugin, or ~/.nuke/CutieRoto/
//       models). Tiny dylib, models shipped alongside. Set
//       -DCUTIE_EMBED_MODELS=OFF to build this way.
//
// Both expose the same cutie::asset_e1()/.../asset_ft() returning a Blob{ptr,size}.
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>

namespace cutie {
struct Blob { const void* data; size_t size; };
}

#ifdef CUTIE_EMBED_MODELS
// ----------------------------------------------------------------------------
// Embedded: symbols come from the .incbin assembly. macOS C ABI prepends '_',
// so the asm labels _cutie_e1_start map to C names cutie_e1_start.
// ----------------------------------------------------------------------------
extern "C" {
    extern const uint8_t cutie_e1_start[], cutie_e1_end[];
    extern const uint8_t cutie_e2_start[], cutie_e2_end[];
    extern const uint8_t cutie_e3_start[], cutie_e3_end[];
    extern const uint8_t cutie_e5_start[], cutie_e5_end[];
    extern const uint8_t cutie_ft_start[], cutie_ft_end[];
}

namespace cutie {
inline Blob asset_e1() { return { cutie_e1_start, size_t(cutie_e1_end - cutie_e1_start) }; }
inline Blob asset_e2() { return { cutie_e2_start, size_t(cutie_e2_end - cutie_e2_start) }; }
inline Blob asset_e3() { return { cutie_e3_start, size_t(cutie_e3_end - cutie_e3_start) }; }
inline Blob asset_e5() { return { cutie_e5_start, size_t(cutie_e5_end - cutie_e5_start) }; }
inline Blob asset_ft() { return { cutie_ft_start, size_t(cutie_ft_end - cutie_ft_start) }; }
} // namespace cutie

#else
// ----------------------------------------------------------------------------
// Disk-load (no objcopy needed; also makes iterating on models trivial).
// Resolution order: $CUTIE_MODEL_DIR -> <plugin dir>/models -> ~/.nuke/CutieRoto/
// models -> ./models. Each file is read once into a process-lifetime buffer.
// ----------------------------------------------------------------------------
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <dlfcn.h>                // dladdr — locate this dylib on disk

namespace cutie {
namespace detail {

inline std::filesystem::path selfDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&selfDir), &info) && info.dli_fname) {
        auto p = std::filesystem::path(info.dli_fname);
        auto d = p.parent_path();
        if (!d.empty()) return d;
    }
    return {};
}

inline std::filesystem::path modelsDir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (const char* env = std::getenv("CUTIE_MODEL_DIR"); env && env[0])
        candidates.emplace_back(env);
    if (auto sd = selfDir(); !sd.empty())
        candidates.emplace_back(sd / "models");
    if (const char* home = std::getenv("HOME"); home && home[0])
        candidates.emplace_back(fs::path(home) / ".nuke" / "CutieRoto" / "models");
    candidates.emplace_back(fs::path("models"));

    std::error_code ec;
    for (const auto& c : candidates)
        if (fs::is_directory(c, ec)) return c;
    return candidates.empty() ? fs::path("models") : candidates.front();
}

inline Blob loadAsset(const char* filename) {
    static std::mutex mu;
    static std::unordered_map<std::string, std::vector<uint8_t>> cache;
    std::lock_guard<std::mutex> lk(mu);

    auto it = cache.find(filename);
    if (it == cache.end()) {
        std::filesystem::path path = modelsDir() / filename;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f)
            throw std::runtime_error(
                std::string("CutieRoto: model not found: ") + path.string() +
                "  (set $CUTIE_MODEL_DIR, or place the 5 .pt files there)");
        std::streamsize n = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> buf(static_cast<size_t>(n));
        if (n > 0 && !f.read(reinterpret_cast<char*>(buf.data()), n))
            throw std::runtime_error(
                std::string("CutieRoto: failed to read model: ") + path.string());
        it = cache.emplace(filename, std::move(buf)).first;
    }
    return Blob{ it->second.data(), it->second.size() };
}

} // namespace detail

inline Blob asset_e1() { return detail::loadAsset("encode_image.pt"); }
inline Blob asset_e2() { return detail::loadAsset("transform_key.pt"); }
inline Blob asset_e3() { return detail::loadAsset("mask_encoder.pt"); }
inline Blob asset_e5() { return detail::loadAsset("mask_decoder.pt"); }
inline Blob asset_ft() { return detail::loadAsset("fusion_transformer.pt"); }

} // namespace cutie
#endif // CUTIE_EMBED_MODELS
