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

extern "C" {
    extern const uint8_t _binary_e1_bin_start[], _binary_e1_bin_end[];
    extern const uint8_t _binary_e2_bin_start[], _binary_e2_bin_end[];
    extern const uint8_t _binary_e3_bin_start[], _binary_e3_bin_end[];
    extern const uint8_t _binary_e5_bin_start[], _binary_e5_bin_end[];
    extern const uint8_t _binary_ft_bin_start[], _binary_ft_bin_end[];
}

namespace cutie {

struct Blob { const void* data; size_t size; };

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
