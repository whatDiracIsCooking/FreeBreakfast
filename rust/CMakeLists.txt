cmake_minimum_required(VERSION 4.2)

# What the crates here link, built without the C++ tree: WarpWraps' C ABI
# libraries libwwr_capi_runtime_api.so and libwwr_capi_blas.so, and the
# per-module kernel shared objects (README.md, "Order of work"). The root
# CMakeLists.txt does not add this directory, and this one adds nothing under
# src/.
#
#   cmake --preset cuda        # or hip; from rust/
#   cmake --build --preset cuda
#
# rust/capi_dir.rs finds the library under <build>/deps/WarpWraps/src/capi.

# ========================================================================
# GPU backend
# ========================================================================
# No default, for the root's reason: it decides whether the CUDA language is
# enabled at all, so a silent default would require a CUDA toolkit on a
# HIP-only host, or ROCm on a CUDA-only one. Each preset carries it.
if(NOT DEFINED FREE_BREAKFAST_GPU_BACKEND)
  message(
    FATAL_ERROR
      "FREE_BREAKFAST_GPU_BACKEND must be set explicitly to CUDA or HIP (e.g. "
      "`cmake --preset cuda` or `--preset hip`, from rust/). It is not "
      "defaulted: it decides whether the CUDA language is enabled at all."
  )
endif()
set(FREE_BREAKFAST_GPU_BACKEND
    "${FREE_BREAKFAST_GPU_BACKEND}"
    CACHE STRING "GPU backend to build against: CUDA or HIP"
)
set_property(CACHE FREE_BREAKFAST_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT FREE_BREAKFAST_GPU_BACKEND MATCHES "^(CUDA|HIP)$")
  message(
    FATAL_ERROR "FREE_BREAKFAST_GPU_BACKEND must be CUDA or HIP, got "
                "'${FREE_BREAKFAST_GPU_BACKEND}'"
  )
endif()

# Mirrored onto wwr's own switch before it is added, as at the root, so the
# two can never disagree.
set(WWR_GPU_BACKEND
    "${FREE_BREAKFAST_GPU_BACKEND}"
    CACHE STRING "WarpWraps GPU backend (follows FREE_BREAKFAST_GPU_BACKEND)"
          FORCE
)

# No languages here: wwr's own project() enables CXX (and CUDA) after its
# toolchain prelude -- clang, libc++, clang-scan-deps and the `import std`
# gate -- which a language enabled first, here, would bypass.
project(
  free-breakfast-rust
  LANGUAGES NONE
  DESCRIPTION "The native libraries the free-breakfast Rust crates link"
)
message(STATUS "GPU backend: ${FREE_BREAKFAST_GPU_BACKEND}")

# The architecture floor is wwr's sm_80 (bf16 WMMA), checked here as at the
# root so the message names this project. `native` and the `all*` keywords
# pass through; an `NN-real`/`NN-virtual` suffix keeps its leading digits.
if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  foreach(_fb_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_fb_arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
      message(
        FATAL_ERROR
          "free-breakfast requires CUDA architecture 80 (Ampere) or newer -- "
          "its WarpWraps dependency wraps bf16 WMMA -- got '${_fb_arch}' in "
          "CMAKE_CUDA_ARCHITECTURES"
      )
    endif()
  endforeach()
endif()

# ========================================================================
# WarpWraps
# ========================================================================
# Its tests, examples and install rules off, set explicitly rather than left
# to the nested-build defaults. WWR_COMPILE_TIME_ONLY's one effect outside
# wwr's test tree is skipping the GoogleTest fetch, which nothing here uses.
foreach(_fb_off WWR_BUILD_TESTS WWR_BUILD_EXAMPLES WWR_INSTALL)
  set(${_fb_off}
      OFF
      CACHE BOOL "" FORCE
  )
endforeach()
set(WWR_COMPILE_TIME_ONLY
    ON
    CACHE BOOL "" FORCE
)

set(_fb_wwr "${CMAKE_CURRENT_SOURCE_DIR}/../deps/WarpWraps")
if(NOT EXISTS "${_fb_wwr}/CMakeLists.txt")
  message(
    FATAL_ERROR
      "deps/WarpWraps is empty -- the submodule has not been initialised. "
      "Run `git submodule update --init --recursive` and configure again."
  )
endif()
# The binary directory mirrors the root build's, so a library is at
# <build>/deps/WarpWraps/src/capi in either.
add_subdirectory("${_fb_wwr}" deps/WarpWraps EXCLUDE_FROM_ALL)

# `all` is what the crates link, and nothing else of wwr: the runtime, and
# BLAS for the overlap's spherical transform (wwr-sys's `blas` feature).
add_custom_target(
  rust_native_libs ALL COMMENT "The libraries the rust/ crates link"
)
add_dependencies(rust_native_libs wwr.capi.runtime_api wwr.capi.blas)

# ========================================================================
# The kernel shared objects
# ========================================================================
# One per module, each forked from src/ with the device headers it needs
# (device_headers/), never including or compiling anything under src/
# (docs/gpu_layer.md, "Build"). The languages are enabled here, after wwr's
# project() ran its toolchain prelude and cached the compilers, because the
# targets below compile .cu in this directory's tree; the backend's package is
# found again for the same reason, its imported targets being directory-scoped.
if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  enable_language(CXX)
  enable_language(CUDA)
  find_package(CUDAToolkit 13.0.0 REQUIRED)
else()
  enable_language(CXX)
  find_package(hip CONFIG REQUIRED)
endif()

set(FB_RUST_DEVICE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/device_headers")
# The C ABI's generated headers: common.h (wwr_stream) and runtime_api's.
set(FB_RUST_CAPI_GEN "${_fb_wwr}/src/capi/gen")

add_subdirectory(recipes/kernels)
add_subdirectory(integrals/kernels)
