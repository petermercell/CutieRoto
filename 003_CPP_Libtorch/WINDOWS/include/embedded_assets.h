// ============================================================================
// embedded_assets.h  —  access the 5 objcopy-embedded blobs (4 TRT engines +
// fusion_transformer.pt) baked into the .so. Symbol names confirmed via nm:
//   _binary_<stem>_bin_start / _binary_<stem>_bin_end
// (the trailing _size symbol is an ABSOLUTE symbol whose ADDRESS is the size;
//  we compute size from end-start instead, which is the portable idiom.)
//
// Stems: e1=encode_image, e2=transform_key, e3=mask_encoder, e5=mask_decoder,
//        ft=fusion_transformer.pt
//
// The TRT engines are deserialized from (ptr,size). The .pt is loaded from an
// in-memory stream (torch::jit::load(std::istream&)). Loader code takes (ptr,
// size) either way, so this is identical to a disk loader -- only the byte
// source differs (this is why "embed now" vs "disk later" is a CMake-only swap).
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>

namespace cutie {
struct Blob { const void* data; size_t size; };
}

#ifdef CUTIE_CMRC
// ----------------------------------------------------------------------------
// Windows (and any build with -DCUTIE_CMRC): the 5 .pt files are embedded via
// CMakeRC instead of objcopy (MSVC has no objcopy). CMakeRC gives a virtual
// filesystem; we open each file once and hand back (ptr,size) — same Blob as
// the objcopy path, so loader code is identical.
// ----------------------------------------------------------------------------
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(cutie_rc);

namespace cutie {
namespace detail {
    inline Blob cmrc_blob(const char* name) {
        // hold the opened file's data alive for the process lifetime
        static auto fs = cmrc::cutie_rc::get_filesystem();
        // cmrc::file's data() is valid for the program's lifetime (it's baked
        // into the binary's rodata), so returning the pointer is safe.
        auto f = fs.open(name);
        return Blob{ static_cast<const void*>(f.begin()),
                     static_cast<size_t>(f.size()) };
    }
}

inline Blob asset_e1() { return detail::cmrc_blob("encode_image.pt"); }
inline Blob asset_e2() { return detail::cmrc_blob("transform_key.pt"); }
inline Blob asset_e3() { return detail::cmrc_blob("mask_encoder.pt"); }
inline Blob asset_e5() { return detail::cmrc_blob("mask_decoder.pt"); }
inline Blob asset_ft() { return detail::cmrc_blob("fusion_transformer.pt"); }
} // namespace cutie

#else
// ----------------------------------------------------------------------------
// Linux: objcopy-embedded blobs. Symbol names confirmed via nm:
//   _binary_<stem>_bin_start / _binary_<stem>_bin_end
// (the trailing _size symbol is an ABSOLUTE symbol whose ADDRESS is the size;
//  we compute size from end-start instead, the portable idiom.)
// Stems: e1=encode_image, e2=transform_key, e3=mask_encoder, e5=mask_decoder,
//        ft=fusion_transformer.pt
// ----------------------------------------------------------------------------
extern "C" {
    extern const uint8_t _binary_e1_bin_start[], _binary_e1_bin_end[];
    extern const uint8_t _binary_e2_bin_start[], _binary_e2_bin_end[];
    extern const uint8_t _binary_e3_bin_start[], _binary_e3_bin_end[];
    extern const uint8_t _binary_e5_bin_start[], _binary_e5_bin_end[];
    extern const uint8_t _binary_ft_bin_start[], _binary_ft_bin_end[];
}

namespace cutie {

inline Blob asset_e1() { return { _binary_e1_bin_start,
    size_t(_binary_e1_bin_end - _binary_e1_bin_start) }; }   // encode_image
inline Blob asset_e2() { return { _binary_e2_bin_start,
    size_t(_binary_e2_bin_end - _binary_e2_bin_start) }; }   // transform_key
inline Blob asset_e3() { return { _binary_e3_bin_start,
    size_t(_binary_e3_bin_end - _binary_e3_bin_start) }; }   // mask_encoder
inline Blob asset_e5() { return { _binary_e5_bin_start,
    size_t(_binary_e5_bin_end - _binary_e5_bin_start) }; }   // mask_decoder
inline Blob asset_ft() { return { _binary_ft_bin_start,
    size_t(_binary_ft_bin_end - _binary_ft_bin_start) }; }   // fusion_transformer.pt

} // namespace cutie
#endif
