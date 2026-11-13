# libfb_recipes_kernels.so: the recipes launch wrappers rust/ has forked from
# src/recipes, one shared object with its own device link, which
# fb-recipes-sys (../sys) links (docs/gpu_layer.md, "Build"). Built from these
# forked sources and rust/device_headers alone: nothing under src/ is compiled
# or included.
#
# Linked: wwr::device PRIVATE for runtime.h, cooperative_groups.h,
# device_guard.h and WWR_WARP_SIZE; the C ABI's runtime_api library PRIVATE
# for wwr_get_last_error(), so a status is mapped by the very library the
# crates already load. The vendor runtime is the shared one, so the launches
# here and that query read the same per-thread error.
add_library(fb_recipes_kernels SHARED abi.cu sort/merge_sort.cu)
target_include_directories(
  fb_recipes_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                             ${FB_RUST_DEVICE_HEADERS} ${FB_RUST_CAPI_GEN}
)
target_link_libraries(
  fb_recipes_kernels PRIVATE wwr::device wwr::capi::runtime_api
)
set_target_properties(
  fb_recipes_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  # Separable compilation with the device link resolved inside the .so, so
  # nothing relocatable crosses the ABI (docs/gpu_layer.md, "Build"). The
  # shared cudart, not CMake's static default: a static copy here would keep
  # its own last error, which wwr_get_last_error() could never read.
  set_target_properties(
    fb_recipes_kernels
    PROPERTIES CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_RUNTIME_LIBRARY Shared
  )
else()
  # A HIP build enables no CUDA language: the .cu is CXX, and hip::device's
  # -x hip reaches this target's own sources only (PRIVATE, never PUBLIC).
  set_source_files_properties(
    abi.cu sort/merge_sort.cu PROPERTIES LANGUAGE CXX
  )
  target_link_libraries(fb_recipes_kernels PRIVATE hip::device)
  # A .cu imports nothing, so there is nothing to scan for modules.
  set_target_properties(
    fb_recipes_kernels PROPERTIES CXX_STANDARD 20 CXX_SCAN_FOR_MODULES OFF
  )
endif()

add_dependencies(rust_native_libs fb_recipes_kernels)
