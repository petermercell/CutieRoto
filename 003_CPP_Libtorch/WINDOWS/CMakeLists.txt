# ============================================================================
# CutieRoto — CMakeLists (Linux + Windows), libtorch backend.
#
# WINDOWS (Option B): build against a DOWNLOADED libtorch that matches Nuke 17's
# version exactly (2.7.1+cu128), because Nuke ships the libtorch DLLs but NOT the
# import .libs or headers. At RUNTIME, Nuke's own torch.dll/c10.dll/torch_cuda.dll
# /cudart64_12.dll load and satisfy the imports (same version = same ABI).
#
#   1) Download libtorch-win-shared-with-deps-2.7.1+cu128.zip (RELEASE, not Debug)
#      from pytorch.org and unzip to e.g. C:\libtorch-2.7.1
#   2) Configure (x64 Native Tools for VS 2022 prompt):
#        cmake -S . -B build -G Ninja ^
#              -DCUTIE_LIBTORCH_BACKEND=ON -DNUKE_VERSION=17.0v1 ^
#              -DTORCH_DIR=C:/libtorch-2.7.1
#      (or use the VS generator: -G "Visual Studio 17 2022" -A x64)
#   3) cmake --build build --config Release
#
# LINUX: unchanged proven path — link Nuke's bundled .so's, objcopy embedding.
#        cmake -S . -B build -DCUTIE_LIBTORCH_BACKEND=ON -DNUKE_VERSION=17.0v1 \
#              -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
#
# Produces CutieRoto.so (Linux) / CutieRoto.dll (Windows). Drop into your Nuke
# plugin path (~/.nuke or %USERPROFILE%\.nuke).
# ============================================================================
cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)        # MSVC runtime policy, before project()
project(CutieRoto CXX C)

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows"))
    message(FATAL_ERROR "Linux and Windows only (got: ${CMAKE_SYSTEM_NAME}).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")           # CutieRoto.so / .dll, no "lib" prefix

option(CUTIE_LIBTORCH_BACKEND "Pure-libtorch backend (no TensorRT)" ON)
if(WIN32 AND NOT CUTIE_LIBTORCH_BACKEND)
    message(FATAL_ERROR "Windows build supports only the libtorch backend (-DCUTIE_LIBTORCH_BACKEND=ON).")
endif()

# ============================================================================
# Windows runtime/CRT + defines (mirrors the proven AlphaToRoto template)
# ============================================================================
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")    # /MD, match Nuke + Release libtorch
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES -DWIN32_LEAN_AND_MEAN
                    -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS)
endif()

# ============================================================================
# Nuke location
# ============================================================================
set(NUKE_VERSION      "" CACHE STRING "Nuke version e.g. '17.0v1'")
set(NUKE_INSTALL_PATH "" CACHE PATH   "Explicit Nuke root (overrides NUKE_VERSION)")
if(NOT NUKE_INSTALL_PATH)
    if(NOT NUKE_VERSION)
        message(FATAL_ERROR "Pass -DNUKE_VERSION=17.0v1 or -DNUKE_INSTALL_PATH=<path>")
    endif()
    if(WIN32)
        set(NUKE_INSTALL_PATH "C:/Program Files/Nuke${NUKE_VERSION}")
    else()
        set(NUKE_INSTALL_PATH "/opt/Nuke${NUKE_VERSION}")
    endif()
endif()
set(NUKE_INCLUDE "${NUKE_INSTALL_PATH}/include")
set(NUKE_ROOT    "${NUKE_INSTALL_PATH}")
if(NOT EXISTS "${NUKE_INCLUDE}/DDImage/Iop.h")
    message(FATAL_ERROR "DDImage headers not found in ${NUKE_INCLUDE}")
endif()
find_library(DDIMAGE_LIBRARY NAMES DDImage PATHS "${NUKE_ROOT}" NO_DEFAULT_PATH REQUIRED)

# ============================================================================
# libtorch
#   Windows: from the DOWNLOADED libtorch 2.7.1 (TORCH_DIR). Must match Nuke's
#            DLL version exactly so Nuke's DLLs satisfy the imports at runtime.
#   Linux:   Nuke's bundled .so's (proven).
# ============================================================================
if(WIN32)
    set(TORCH_DIR "" CACHE PATH "Unzipped libtorch 2.7.1+cu128 (RELEASE) root, e.g. C:/libtorch-2.7.1")
    if(NOT TORCH_DIR OR NOT EXISTS "${TORCH_DIR}/lib" OR NOT EXISTS "${TORCH_DIR}/include")
        message(FATAL_ERROR
            "Set -DTORCH_DIR to the unzipped libtorch 2.7.1+cu128 (Release) root\n"
            "(the folder containing include/ and lib/).\n"
            "Download: libtorch-win-shared-with-deps-2.7.1+cu128.zip from pytorch.org\n"
            "(must match Nuke 17's libtorch version exactly).")
    endif()
    set(TORCH_INCLUDE
        "${TORCH_DIR}/include"
        "${TORCH_DIR}/include/torch/csrc/api/include")
    set(TORCH_LIB_DIR "${TORCH_DIR}/lib")

    # --- Version coupling guard (the real robustness against libtorch bumps) ---
    # The downloaded libtorch MUST match the version Nuke ships, because the ABI
    # is only stable within an exact version: the plugin compiles 2.7.1's struct
    # layouts/vtables into its code, then loads Nuke's DLLs at runtime. A mismatch
    # crashes at runtime (hard to debug). Catch it HERE, at configure time, loudly.
    # On a Nuke libtorch bump: update EXPECTED_TORCH_VERSION and download the
    # matching libtorch — that's the one maintenance step, by design.
    set(EXPECTED_TORCH_VERSION "2.7.1" CACHE STRING "libtorch version Nuke 17 ships")
    if(EXISTS "${TORCH_DIR}/build-version")
        file(READ "${TORCH_DIR}/build-version" _torch_ver)
        string(STRIP "${_torch_ver}" _torch_ver)
        if(NOT _torch_ver MATCHES "^${EXPECTED_TORCH_VERSION}")
            message(WARNING
                "libtorch version mismatch:\n"
                "  downloaded : ${_torch_ver}\n"
                "  expected   : ${EXPECTED_TORCH_VERSION} (what Nuke 17 ships)\n"
                "The ABI is only stable within an exact version. Building against a\n"
                "different libtorch than Nuke loads at runtime can crash. Download\n"
                "libtorch ${EXPECTED_TORCH_VERSION}+cu128, or (if Nuke changed) update\n"
                "EXPECTED_TORCH_VERSION to match Nuke's bundled libtorch.")
        else()
            message(STATUS "  libtorch ver: ${_torch_ver} (matches Nuke 17)")
        endif()
    else()
        message(WARNING "Could not read ${TORCH_DIR}/build-version to verify libtorch version.")
    endif()

    set(TORCH_LIBS "")
    foreach(_t torch torch_cpu torch_cuda c10 c10_cuda)
        find_library(TORCH_${_t}_LIB NAMES ${_t} PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
        if(NOT TORCH_${_t}_LIB)
            message(FATAL_ERROR "libtorch lib '${_t}.lib' not found in ${TORCH_LIB_DIR}")
        endif()
        list(APPEND TORCH_LIBS "${TORCH_${_t}_LIB}")
    endforeach()
    # (cudart.lib is linked from the CUDA toolkit below, after we resolve the
    #  CUDA include dir — libtorch's Windows package doesn't ship cudart.lib.)
    # cuda_runtime.h: the sources use cudaStream_t / cudaStreamSynchronize, so we
    # need the CUDA runtime HEADERS (we compile no .cu — headers only). Find an
    # installed CUDA 12.x toolkit's include dir. Override with -DCUDA_INCLUDE_DIR.
    set(CUDA_INCLUDE_DIR "" CACHE PATH "CUDA toolkit include dir (has cuda_runtime.h)")
    if(NOT CUDA_INCLUDE_DIR)
        file(GLOB _cuda_candidates
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.*/include")
        if(_cuda_candidates)
            list(SORT _cuda_candidates)
            list(GET _cuda_candidates -1 CUDA_INCLUDE_DIR)   # newest 12.x
        endif()
    endif()
    if(NOT CUDA_INCLUDE_DIR OR NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda_runtime.h")
        message(FATAL_ERROR
            "cuda_runtime.h not found. Set -DCUDA_INCLUDE_DIR to a CUDA 12.x "
            "toolkit include dir, e.g.\n"
            "  C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include")
    endif()
    list(APPEND TORCH_INCLUDE "${CUDA_INCLUDE_DIR}")
    message(STATUS "  CUDA headers: ${CUDA_INCLUDE_DIR}")

    # cudart.lib: the code calls cudaSetDevice/cudaStreamCreate/cudaEvent* etc.,
    # so we must link the CUDA runtime import lib. libtorch's Windows package
    # does NOT ship it — get it from the toolkit (sibling lib/x64 of the headers).
    get_filename_component(_cuda_root "${CUDA_INCLUDE_DIR}" DIRECTORY)
    find_library(CUDART_LIB NAMES cudart PATHS "${_cuda_root}/lib/x64" NO_DEFAULT_PATH)
    if(NOT CUDART_LIB)
        message(FATAL_ERROR "cudart.lib not found in ${_cuda_root}/lib/x64")
    endif()
    list(APPEND TORCH_LIBS "${CUDART_LIB}")
    message(STATUS "  cudart.lib  : ${CUDART_LIB}")
else()
    set(TORCH_LIB_DIR "${NUKE_ROOT}")
    set(TORCH_INCLUDE "${NUKE_INCLUDE}")   # adjust if your torch headers live elsewhere
    set(TORCH_LIBS
        "${TORCH_LIB_DIR}/libtorch.so" "${TORCH_LIB_DIR}/libtorch_cpu.so"
        "${TORCH_LIB_DIR}/libtorch_cuda.so" "${TORCH_LIB_DIR}/libc10.so"
        "${TORCH_LIB_DIR}/libc10_cuda.so")
    find_package(CUDAToolkit REQUIRED)
endif()

# ============================================================================
# Embed the 5 .pt assets.
#   Windows: CMakeRC (no objcopy on MSVC). Provide cmake/CMakeRC.cmake from
#            https://github.com/vector-of-bool/cmrc
#   Linux:   objcopy (proven).
# Either way, embedded_assets.h's asset_e1()..asset_ft() must read the bytes.
# With CMakeRC the accessors wrap cmrc::open(); with objcopy they wrap the
# _binary_*_start/_end symbols. (embedded_assets.h has a CUTIE_CMRC branch.)
# ============================================================================
set(ASSET_DIR "${CMAKE_SOURCE_DIR}/models" CACHE PATH "dir with the 5 .pt files")
set(ASSET_FILES
    "${ASSET_DIR}/encode_image.pt"
    "${ASSET_DIR}/transform_key.pt"
    "${ASSET_DIR}/mask_encoder.pt"
    "${ASSET_DIR}/mask_decoder.pt"
    "${ASSET_DIR}/fusion_transformer.pt")

if(WIN32)
    include(${CMAKE_SOURCE_DIR}/cmake/CMakeRC.cmake)   # provide this file
    cmrc_add_resource_library(cutie_assets ALIAS cutie::rc NAMESPACE cutie_rc
        WHENCE "${ASSET_DIR}" ${ASSET_FILES})
    set(EMBED_OBJS "")
    set(EMBED_LIB cutie::rc)
    set(USE_CMRC TRUE)
else()
    set(EMBED_LIB "")
    set(EMBED_OBJS "")
    set(_files encode_image.pt transform_key.pt mask_encoder.pt mask_decoder.pt fusion_transformer.pt)
    set(_stems e1 e2 e3 e5 ft)
    list(LENGTH _files _nfiles)
    math(EXPR _last "${_nfiles} - 1")
    foreach(_i RANGE ${_last})
        list(GET _files ${_i} _file)
        list(GET _stems ${_i} _stem)
        set(_obj "${CMAKE_BINARY_DIR}/${_stem}.o")
        add_custom_command(OUTPUT "${_obj}"
            COMMAND ${CMAKE_COMMAND} -E copy "${ASSET_DIR}/${_file}" "${CMAKE_BINARY_DIR}/${_stem}.bin"
            COMMAND objcopy --input binary --output elf64-x86-64 --binary-architecture i386:x86-64
                    --rename-section .data=.rodata,alloc,load,readonly,data,contents "${_stem}.bin" "${_obj}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" DEPENDS "${ASSET_DIR}/${_file}"
            COMMENT "Embedding ${_file} -> ${_stem}.o")
        list(APPEND EMBED_OBJS "${_obj}")
        set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    endforeach()
endif()

# ============================================================================
# The plugin
# ============================================================================
add_library(CutieRoto SHARED
    src/CutieRoto.cpp src/torch_engine.cpp src/memory_core.cpp ${EMBED_OBJS})
target_include_directories(CutieRoto PRIVATE ${NUKE_INCLUDE} ${TORCH_INCLUDE} src include)
target_compile_definitions(CutieRoto PRIVATE CUTIE_LIBTORCH_BACKEND)
target_link_libraries(CutieRoto PRIVATE "${DDIMAGE_LIBRARY}" ${TORCH_LIBS})

if(USE_CMRC)
    target_compile_definitions(CutieRoto PRIVATE CUTIE_CMRC)   # embedded_assets.h uses CMakeRC path
    target_link_libraries(CutieRoto PRIVATE ${EMBED_LIB})
endif()

set_target_properties(CutieRoto PROPERTIES PREFIX "")

if(MSVC)
    target_compile_options(CutieRoto PRIVATE
        /W3 /O2 /Ob2 /EHsc /FS /nologo /MP /bigobj   # /bigobj: libtorch headers are huge
        /wd4996 /wd4244 /wd4267 /wd4305 /wd4251)
    # libtorch on Windows: these defines avoid common pitfalls.
    # NOTE: do NOT define C10_USE_GLOG — on this libtorch, *defining* it (even to 0)
    # selects the glog logging backend, whose headers aren't shipped -> the build
    # fails on missing <glog/stl_logging.h>. Leaving it undefined uses libtorch's
    # own logging (the prebuilt DLLs were built that way).
    target_compile_definitions(CutieRoto PRIVATE NOMINMAX)
    # CRITICAL for GPU on Windows: force-reference a torch_cuda symbol so the CUDA
    # backend's static initializers run. Without this the linker drops torch_cuda
    # (nothing in our code references it directly -> the CUDA ops are dispatched
    # dynamically), and torch::cuda::is_available() returns FALSE -> everything
    # silently runs on CPU (very slow). This is the flag libtorch's own CMake adds.
    target_link_options(CutieRoto PRIVATE
        "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
else()
    target_compile_definitions(CutieRoto PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_compile_options(CutieRoto PRIVATE
        -O3 -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations)
    target_link_libraries(CutieRoto PRIVATE CUDA::cudart dl rt pthread)
    set_target_properties(CutieRoto PROPERTIES
        BUILD_RPATH "${NUKE_ROOT}" INSTALL_RPATH "$ORIGIN:${NUKE_ROOT}")
endif()

message(STATUS "")
message(STATUS "==================  CutieRoto  ==================")
message(STATUS "  Platform   : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Backend    : libtorch (no TensorRT)")
message(STATUS "  Nuke root  : ${NUKE_ROOT}")
if(WIN32)
    message(STATUS "  libtorch   : ${TORCH_DIR}  (build against; Nuke's DLLs at runtime)")
    message(STATUS "  embedding  : CMakeRC")
else()
    message(STATUS "  libtorch   : ${TORCH_LIB_DIR} (Nuke bundled)")
    message(STATUS "  embedding  : objcopy")
endif()
message(STATUS "  DDImage    : ${DDIMAGE_LIBRARY}")
message(STATUS "=================================================")
message(STATUS "")