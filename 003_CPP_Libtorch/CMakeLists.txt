cmake_minimum_required(VERSION 3.18)
project(cutie_nuke VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Cutie-Nuke plugin build.  Layout:
#   include/   headers (trt_engine.h, memory_core.h, embedded_assets.h)
#   src/       core sources (trt_engine.cpp, memory_core.cpp, CutieRoto.cpp)
#   src/tests/ standalone test harnesses
#   models/    embedded assets (4 engines + fusion_transformer.pt)
#   test_data/ fixtures (memcase.pt)
# Linkage proven: Nuke17 libtorch 2.7.1 .so + conda 2.7.1 headers +
#   -D_GLIBCXX_USE_CXX11_ABI=1 + shared libnvinfer.so + CUDA 12.8.
# =============================================================================

# ---- Nuke 17.0 (provides libtorch 2.7.1) ----
set(NUKE_ROOT "/opt/Nuke17.0v1" CACHE PATH "Nuke 17.0 install")
if(NOT EXISTS "${NUKE_ROOT}/libtorch.so")
    message(FATAL_ERROR "Nuke libtorch not found in ${NUKE_ROOT}")
endif()
# NDK (only needed for the plugin target, not the tests)
set(NUKE_INCLUDE "${NUKE_ROOT}/include")

# ---- libtorch headers (conda, version-matched 2.7.1) ----
set(TORCH_INCLUDE_ROOT
    "$ENV{HOME}/miniconda3/envs/cutie311/lib/python3.11/site-packages/torch/include"
    CACHE PATH "libtorch headers (conda)")
if(NOT EXISTS "${TORCH_INCLUDE_ROOT}/torch/script.h")
    message(FATAL_ERROR "torch/script.h not found in ${TORCH_INCLUDE_ROOT}")
endif()
set(TORCH_INCLUDE_DIRS
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include")
set(TORCH_LIBS
    "${NUKE_ROOT}/libtorch.so" "${NUKE_ROOT}/libtorch_cpu.so"
    "${NUKE_ROOT}/libtorch_cuda.so" "${NUKE_ROOT}/libc10.so"
    "${NUKE_ROOT}/libc10_cuda.so")

# ---- CUDA 12.8 ----
if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-12.8")
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
endif()
find_package(CUDAToolkit REQUIRED)

# ---- TensorRT 10.9 ----
set(TENSORRT_ROOT "/opt/TensorRT-10.9.0.34" CACHE PATH "TensorRT root")
set(TRT_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")

# EXPERIMENTAL: static-link TRT for a more self-contained .so. Default OFF (the
# proven path is shared libnvinfer.so — torch+TRT coexist via shared CUDA). When
# ON we link libnvinfer_static.a + companions + cudart_static. This MAY clash with
# libtorch's dynamic CUDA runtime (double-CUDA); if link/runtime fails, set OFF.
option(CUTIE_STATIC_TRT "Static-link TensorRT (experimental, may clash with libtorch CUDA)" OFF)

if(CUTIE_STATIC_TRT)
    message(STATUS "  *** static TRT, DYNAMIC cudart (shared w/ Nuke libtorch) ***")
    set(CUDA_LIB_DIR "${CUDAToolkit_LIBRARY_DIR}")
    # static TRT needs nvJitLink (JIT kernel linker) at link time. Find the .so
    # robustly (the CUDA:: target name varies across toolkit versions).
    find_library(NVJITLINK_SO NAMES nvJitLink PATHS "${CUDA_LIB_DIR}" NO_DEFAULT_PATH)
    if(NOT NVJITLINK_SO)
        find_library(NVJITLINK_SO NAMES nvJitLink)   # system search fallback
    endif()
    find_library(NVRTC_SO NAMES nvrtc PATHS "${CUDA_LIB_DIR}" NO_DEFAULT_PATH)
    if(NOT NVRTC_SO)
        find_library(NVRTC_SO NAMES nvrtc)
    endif()
    # nvPTXCompiler* symbols (static TRT's NVPTX serializer). CUDA ships this as a
    # static-only lib (libnvptxcompiler_static.a).
    find_library(NVPTXCOMPILER_LIB NAMES nvptxcompiler_static PATHS "${CUDA_LIB_DIR}" NO_DEFAULT_PATH)
    if(NOT NVPTXCOMPILER_LIB)
        find_library(NVPTXCOMPILER_LIB NAMES nvptxcompiler_static)
    endif()
    # GOAL: static-link ONLY TensorRT (artists lack TRT) while keeping the CUDA
    # runtime DYNAMIC and shared with Nuke's libtorch (always present, already
    # loads libcudart.so). A SECOND static cudart would clash with libtorch.
    set(NVINFER_LIB
        -Wl,--start-group
        ${TRT_LIB_DIR}/libnvinfer_static.a
        ${NVPTXCOMPILER_LIB}
        -Wl,--end-group
        CUDA::cudart        # DYNAMIC (shared with libtorch) -- NOT cudart_static
        ${NVRTC_SO}         # dynamic
        ${NVJITLINK_SO}     # dynamic -- static TRT needs __nvJitLink* symbols
        dl rt pthread)
    if(NOT EXISTS "${TRT_LIB_DIR}/libnvinfer_static.a")
        message(FATAL_ERROR "CUTIE_STATIC_TRT=ON but libnvinfer_static.a not found in ${TRT_LIB_DIR}")
    endif()
    message(STATUS "  static: libnvinfer_static.a, nvptxcompiler=${NVPTXCOMPILER_LIB}")
    message(STATUS "  dynamic: cudart, nvrtc=${NVRTC_SO}, nvJitLink=${NVJITLINK_SO}")
else()
    find_library(NVINFER_LIB NAMES nvinfer PATHS "${TRT_LIB_DIR}" "${TENSORRT_ROOT}/lib"
        NO_DEFAULT_PATH REQUIRED)
endif()

set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS})

# common settings helper
function(cutie_common tgt)
    target_include_directories(${tgt} PRIVATE ${COMMON_INCLUDES})
    target_compile_definitions(${tgt} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter
        -Wno-deprecated-declarations -O3)
    set_target_properties(${tgt} PROPERTIES
        BUILD_RPATH "${NUKE_ROOT};${TRT_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}")
endfunction()

# -----------------------------------------------------------------------------
# Embed the 5 assets (models/) via objcopy -> stem.o, symbols _binary_<stem>_bin_*
# -----------------------------------------------------------------------------
set(ASSET_DIR "${CMAKE_SOURCE_DIR}/models" CACHE PATH "dir holding engines + .pt")
option(CUTIE_LIBTORCH_BACKEND "Pure-libtorch backend: embed traced .pt, no TensorRT" OFF)
if(CUTIE_LIBTORCH_BACKEND)
    message(STATUS "  *** BACKEND: pure libtorch (traced .pt, no TensorRT) ***")
    set(ASSET_e1 "encode_image.pt")
    set(ASSET_e2 "transform_key.pt")
    set(ASSET_e3 "mask_encoder.pt")
    set(ASSET_e5 "mask_decoder.pt")
else()
    set(ASSET_e1 "encode_image.fp16.engine")
    set(ASSET_e2 "transform_key.fp16.engine")
    set(ASSET_e3 "mask_encoder.fp32.engine")
    set(ASSET_e5 "mask_decoder.fp32.engine")
endif()
set(ASSET_ft "fusion_transformer.pt")

set(EMBED_OBJS "")
foreach(stem e1 e2 e3 e5 ft)
    set(src "${ASSET_DIR}/${ASSET_${stem}}")
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR "asset not found: ${src}")
    endif()
    set(obj "${CMAKE_BINARY_DIR}/${stem}.o")
    add_custom_command(OUTPUT ${obj}
        COMMAND ${CMAKE_COMMAND} -E copy ${src} ${CMAKE_BINARY_DIR}/${stem}.bin
        COMMAND objcopy --input binary --output elf64-x86-64
                --binary-architecture i386:x86-64
                --rename-section .data=.rodata,alloc,load,readonly,data,contents
                ${stem}.bin ${stem}.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${src}
        COMMENT "Embedding ${ASSET_${stem}} -> ${stem}.o")
    list(APPEND EMBED_OBJS ${obj})
endforeach()
foreach(obj ${EMBED_OBJS})
    set_source_files_properties(${obj} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
endforeach()

# -----------------------------------------------------------------------------
# Test harnesses (src/tests/)
# -----------------------------------------------------------------------------
add_executable(test_engines  src/tests/test_engines.cpp  src/trt_engine.cpp)
target_link_libraries(test_engines PRIVATE ${TORCH_LIBS} ${NVINFER_LIB} CUDA::cudart dl rt pthread)
cutie_common(test_engines)

add_executable(test_memory   src/tests/test_memory.cpp   src/memory_core.cpp)
target_link_libraries(test_memory PRIVATE ${TORCH_LIBS} CUDA::cudart dl rt pthread)
cutie_common(test_memory)

add_executable(test_embedded src/tests/test_embedded.cpp src/trt_engine.cpp ${EMBED_OBJS})
target_link_libraries(test_embedded PRIVATE ${TORCH_LIBS} ${NVINFER_LIB} CUDA::cudart dl rt pthread)
cutie_common(test_embedded)

# -----------------------------------------------------------------------------
# The plugin (added in STEP 3): CutieRoto.so
#   add_library(CutieRoto SHARED src/CutieRoto.cpp src/trt_engine.cpp
#                                src/memory_core.cpp ${EMBED_OBJS})
#   target_include_directories(CutieRoto PRIVATE ${NUKE_INCLUDE} ${COMMON_INCLUDES})
#   target_link_libraries(CutieRoto PRIVATE ${DDIMAGE_LIBRARY} ${TORCH_LIBS}
#                         ${NVINFER_LIB} CUDA::cudart dl rt pthread)
#   set_target_properties(CutieRoto PROPERTIES PREFIX "" SUFFIX ".so"
#                         INSTALL_RPATH "$ORIGIN;${NUKE_ROOT}")
# (uncomment once src/CutieRoto.cpp exists)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# The plugin: CutieRoto.so  (STEP 3)
# -----------------------------------------------------------------------------
find_library(DDIMAGE_LIBRARY NAMES DDImage PATHS "${NUKE_ROOT}" NO_DEFAULT_PATH)
if(DDIMAGE_LIBRARY AND EXISTS "${CMAKE_SOURCE_DIR}/src/CutieRoto.cpp")
  if(CUTIE_LIBTORCH_BACKEND)
    # ---- pure-libtorch plugin: torch_engine.cpp, no TRT at all ----
    add_library(CutieRoto SHARED
        src/CutieRoto.cpp src/torch_engine.cpp src/memory_core.cpp ${EMBED_OBJS})
    target_include_directories(CutieRoto PRIVATE ${NUKE_INCLUDE} ${COMMON_INCLUDES})
    target_compile_definitions(CutieRoto PRIVATE _GLIBCXX_USE_CXX11_ABI=1 CUTIE_LIBTORCH_BACKEND)
    target_link_libraries(CutieRoto PRIVATE
        ${DDIMAGE_LIBRARY} ${TORCH_LIBS} CUDA::cudart dl rt pthread)   # NO nvinfer
    set_target_properties(CutieRoto PROPERTIES
        PREFIX "" SUFFIX ".so"
        BUILD_RPATH "${NUKE_ROOT};${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "$ORIGIN:${NUKE_ROOT}")
    target_compile_options(CutieRoto PRIVATE -Wall -Wextra -Wno-unused-parameter
        -Wno-deprecated-declarations -fvisibility=hidden -O3)
    message(STATUS "  plugin target: CutieRoto.so [libtorch backend, no TRT]")
  else()
    add_library(CutieRoto SHARED
        src/CutieRoto.cpp src/trt_engine.cpp src/memory_core.cpp ${EMBED_OBJS})
    target_include_directories(CutieRoto PRIVATE ${NUKE_INCLUDE} ${COMMON_INCLUDES})
    target_compile_definitions(CutieRoto PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(CutieRoto PRIVATE
        ${DDIMAGE_LIBRARY} ${TORCH_LIBS} ${NVINFER_LIB} CUDA::cudart dl rt pthread)
    # Dev build: find TRT/cudart at their real paths (so it runs in-place).
    # Bundle/install: find them in libs/ next to the plugin ($ORIGIN/libs).
    set_target_properties(CutieRoto PROPERTIES
        PREFIX "" SUFFIX ".so"
        BUILD_RPATH "${NUKE_ROOT};${TRT_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "$ORIGIN/libs:$ORIGIN:${NUKE_ROOT}")
    target_compile_options(CutieRoto PRIVATE -Wall -Wextra -Wno-unused-parameter
        -Wno-deprecated-declarations -fvisibility=hidden -O3)
    message(STATUS "  plugin target: CutieRoto.so (DDImage=${DDIMAGE_LIBRARY})")

    # ---- bundle target: produce a drop-in folder dist/CutieRoto/ ----
    # Copies the plugin + the runtime .so's that aren't in Nuke (TRT + cudart)
    # into dist/CutieRoto/libs/. libtorch/c10 are NOT bundled (they ship in Nuke).
    set(DIST "${CMAKE_BINARY_DIR}/dist/CutieRoto")
    add_custom_target(bundle
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST}/libs"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:CutieRoto>" "${DIST}/"
        COMMAND bash -c "cp -L ${TRT_LIB_DIR}/libnvinfer.so '${DIST}/libs/' 2>/dev/null || true"
        COMMAND bash -c "cp -L ${TRT_LIB_DIR}/libnvinfer_plugin.so '${DIST}/libs/' 2>/dev/null || true"
        COMMAND bash -c "cp -L ${CUDAToolkit_LIBRARY_DIR}/libcudart.so.12 '${DIST}/libs/' 2>/dev/null || cp -L ${CUDAToolkit_LIBRARY_DIR}/libcudart.so* '${DIST}/libs/' 2>/dev/null || true"
        DEPENDS CutieRoto
        COMMENT "Bundling drop-in dist/CutieRoto/ (plugin + libs/)")
  endif()
else()
    message(STATUS "  plugin target: SKIPPED (DDImage or src/CutieRoto.cpp missing)")
endif()

message(STATUS "")
message(STATUS "=== cutie_nuke ===  Nuke17 torch 2.7.1 | TRT 10.9 (shared) | CUDA ${CUDAToolkit_VERSION}")
message(STATUS "  targets: test_engines test_memory test_embedded  (plugin: add in STEP 3)")
message(STATUS "==================")
