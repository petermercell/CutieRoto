cmake_minimum_required(VERSION 3.18)
project(cutie_nuke_mps VERSION 0.1.0 LANGUAGES CXX ASM)

# =============================================================================
# CutieRoto — macOS / Apple Metal (MPS) build.
#
# Pure-libtorch backend running on Apple Silicon's GPU via MPS. NO CUDA, NO
# TensorRT. By default the 5 traced .pt modules are EMBEDDED into the dylib via
# .incbin (macOS has no objcopy) for a self-contained ~222MB plugin. Build with
# -DCUTIE_EMBED_MODELS=OFF to instead load them from disk at runtime (see
# include/embedded_assets.h).
#
# Provide three paths (cache vars, override on the cmake line):
#   NUKE_MACOS         .../Nuke17.0v1.app/Contents/MacOS   (dylibs + NDK headers)
#   TORCH_INCLUDE_ROOT a torch 2.7.1 install's include dir (pip/conda; headers
#                      only — the dylibs come from Nuke). MUST be 2.7.1 to match.
#
# Example:
#   cmake -B build \
#     -DNUKE_MACOS="/Applications/Nuke17.0v1/Nuke17.0v1.app/Contents/MacOS" \
#     -DTORCH_INCLUDE_ROOT="$HOME/.venv/lib/python3.11/site-packages/torch/include"
#   cmake --build build -j --target CutieRoto
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Apple Silicon only (MPS). Deployment target 12.3+ is required for MPS at RUNTIME.
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "build arch")
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.3" CACHE STRING "min macOS (MPS needs 12.3+)")
endif()

# ---- Nuke 17.0 (provides libtorch 2.7.1 dylibs + the NDK headers) ----
# Default guess; override with -DNUKE_MACOS=... if your install path differs.
set(NUKE_MACOS "/Applications/Nuke17.0v1/Nuke17.0v1.app/Contents/MacOS"
    CACHE PATH "Nuke 17 .app Contents/MacOS (dylibs + include/)")
if(NOT EXISTS "${NUKE_MACOS}/libtorch.dylib")
    message(FATAL_ERROR
        "Nuke libtorch.dylib not found in ${NUKE_MACOS}\n"
        "Set -DNUKE_MACOS=/path/to/Nuke17.0v1.app/Contents/MacOS")
endif()
set(NUKE_INCLUDE "${NUKE_MACOS}/include")
if(NOT EXISTS "${NUKE_INCLUDE}/DDImage/Iop.h")
    message(FATAL_ERROR "Nuke NDK headers not found: ${NUKE_INCLUDE}/DDImage/Iop.h")
endif()

# ---- libtorch headers (version-matched 2.7.1; pip/conda torch install) ----
set(TORCH_INCLUDE_ROOT "" CACHE PATH "libtorch 2.7.1 headers (……/site-packages/torch/include)")
if(NOT TORCH_INCLUDE_ROOT OR NOT EXISTS "${TORCH_INCLUDE_ROOT}/torch/script.h")
    message(FATAL_ERROR
        "torch/script.h not found under TORCH_INCLUDE_ROOT='${TORCH_INCLUDE_ROOT}'.\n"
        "Install matching headers, e.g.:  python3 -m pip install torch==2.7.1\n"
        "then set -DTORCH_INCLUDE_ROOT=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"include\"))')")
endif()
set(TORCH_INCLUDE_DIRS
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include")

# ---- the libtorch dylibs SHIPPED BY NUKE (CPU build incl. the MPS backend) ----
# No libtorch_cuda / libc10_cuda on macOS — they don't exist and aren't needed.
set(TORCH_LIBS
    "${NUKE_MACOS}/libtorch.dylib"
    "${NUKE_MACOS}/libtorch_cpu.dylib"
    "${NUKE_MACOS}/libc10.dylib")

# ---- DDImage ----
find_library(DDIMAGE_LIBRARY NAMES DDImage PATHS "${NUKE_MACOS}" NO_DEFAULT_PATH)
if(NOT DDIMAGE_LIBRARY)
    message(FATAL_ERROR "libDDImage.dylib not found in ${NUKE_MACOS}")
endif()

set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS})

# -----------------------------------------------------------------------------
# Model embedding (default ON): bake the 5 traced .pt into the dylib via .incbin
# (macOS has no objcopy). -DCUTIE_EMBED_MODELS=OFF builds the tiny disk-load dylib
# instead (models shipped alongside / found via $CUTIE_MODEL_DIR).
# -----------------------------------------------------------------------------
option(CUTIE_EMBED_MODELS "Embed the 5 .pt into the dylib (self-contained ~222MB)" ON)
set(ASSET_DIR "${CMAKE_SOURCE_DIR}/models" CACHE PATH "dir holding the 5 traced .pt")

set(EMBED_SRC "")
if(CUTIE_EMBED_MODELS)
    foreach(m encode_image transform_key mask_encoder mask_decoder fusion_transformer)
        if(NOT EXISTS "${ASSET_DIR}/${m}.pt")
            message(FATAL_ERROR
                "CUTIE_EMBED_MODELS=ON but ${ASSET_DIR}/${m}.pt is missing.\n"
                "Trace the 5 .pt into ${ASSET_DIR}/ (see README), or build with "
                "-DCUTIE_EMBED_MODELS=OFF for the disk-load dylib.")
        endif()
    endforeach()
    # configure_file bakes the absolute model paths into the .incbin assembly.
    configure_file("${CMAKE_SOURCE_DIR}/src/embedded_assets_mac.S.in"
                   "${CMAKE_BINARY_DIR}/embedded_assets_mac.S" @ONLY)
    set(EMBED_SRC "${CMAKE_BINARY_DIR}/embedded_assets_mac.S")
    # re-assemble (re-embed) whenever any model file changes
    set_source_files_properties("${EMBED_SRC}" PROPERTIES
        LANGUAGE ASM
        OBJECT_DEPENDS "${ASSET_DIR}/encode_image.pt;${ASSET_DIR}/transform_key.pt;${ASSET_DIR}/mask_encoder.pt;${ASSET_DIR}/mask_decoder.pt;${ASSET_DIR}/fusion_transformer.pt")
endif()

# -----------------------------------------------------------------------------
# The plugin: CutieRoto.dylib
# -----------------------------------------------------------------------------
add_library(CutieRoto SHARED
    src/CutieRoto.cpp
    src/torch_engine.cpp
    src/memory_core.cpp
    ${EMBED_SRC})

target_include_directories(CutieRoto PRIVATE ${NUKE_INCLUDE} ${COMMON_INCLUDES})

# CUTIE_LIBTORCH_BACKEND keeps any shared headers on the libtorch path. No CUDA
# ABI flag on macOS (clang/libc++ has a stable ABI; _GLIBCXX_USE_CXX11_ABI is
# GNU-libstdc++ only and must NOT be set here). CUTIE_EMBED_MODELS switches
# embedded_assets.h to the baked-in (.incbin) symbols.
target_compile_definitions(CutieRoto PRIVATE CUTIE_LIBTORCH_BACKEND)
if(CUTIE_EMBED_MODELS)
    target_compile_definitions(CutieRoto PRIVATE CUTIE_EMBED_MODELS)
endif()

target_compile_options(CutieRoto PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations
    # AppleClang 16+/Xcode's libc++ marks std type-traits [[no_specializations]];
    # libtorch's bundled strong_type.h specializes std::is_arithmetic, which that
    # libc++ rejects as an error. The specialization is valid C++ (the attribute is
    # only a hint), so silence the diagnostic to build torch headers on new Xcode.
    -Wno-invalid-specialization
    -fvisibility=hidden -O3)

target_link_libraries(CutieRoto PRIVATE
    ${DDIMAGE_LIBRARY} ${TORCH_LIBS})

# Plugin name is exactly CutieRoto.dylib; rpath lets the loader resolve Nuke's
# torch/DDImage dylibs (Nuke already has them loaded at plugin-load time, but the
# rpath keeps a standalone link/load resolvable too).
set_target_properties(CutieRoto PROPERTIES
    PREFIX "" SUFFIX ".dylib"
    INSTALL_RPATH "@loader_path;${NUKE_MACOS}"
    BUILD_RPATH   "${NUKE_MACOS}"
    MACOSX_RPATH ON)

message(STATUS "")
message(STATUS "=== CutieRoto (macOS / MPS) ===")
message(STATUS "  Nuke MacOS : ${NUKE_MACOS}")
message(STATUS "  torch incl : ${TORCH_INCLUDE_ROOT}")
message(STATUS "  DDImage    : ${DDIMAGE_LIBRARY}")
message(STATUS "  backend    : pure libtorch on Apple Metal (MPS) — no CUDA, no TRT")
if(CUTIE_EMBED_MODELS)
    message(STATUS "  models     : EMBEDDED via .incbin from ${ASSET_DIR} (self-contained dylib)")
else()
    message(STATUS "  models     : loaded from disk at runtime (CUTIE_MODEL_DIR / models/)")
endif()
message(STATUS "  output     : CutieRoto.dylib")
message(STATUS "===============================")
