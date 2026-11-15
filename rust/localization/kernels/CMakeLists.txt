# libfb_localization_kernels.so: the localization launch wrappers rust/ has
# forked from src/localization (Pipek-Mezey's populations, pair angles,
# rotations, functional and identity), one shared object with its own device
# link, which fb-localization-sys (../sys) links (docs/gpu_layer.md, "Build").
# Built from these forked sources and rust/device_headers alone: nothing under
# src/ is compiled or included.
#
# Linked: wwr::device PRIVATE for runtime.h, cooperative_groups.h,
# device_guard.h and WWR_WARP_SIZE; the C ABI's runtime_api library PRIVATE
# for wwr_get_last_error(), so a status is mapped by the very library the
# crates already load. The vendor runtime is the shared one, as in
# ../../recipes/kernels, so the launches here and that query read the same
# per-thread error.
set(_fb_localization_sources abi.cu pipek_mezey/pipek_mezey.cu)
add_library(fb_localization_kernels SHARED ${_fb_localization_sources})
target_include_directories(
  fb_localization_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                  ${FB_RUST_DEVICE_HEADERS} ${FB_RUST_CAPI_GEN}
)
target_link_libraries(
  fb_localization_kernels PRIVATE wwr::device wwr::capi::runtime_api
)
set_target_properties(
  fb_localization_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  # Separable compilation with the device link resolved inside the .so, as in
  # ../../recipes/kernels; the shared cudart, so wwr_get_last_error() reads
  # these launches.
  set_target_properties(
    fb_localization_kernels
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
    ${_fb_localization_sources} PROPERTIES LANGUAGE CXX
  )
  target_link_libraries(fb_localization_kernels PRIVATE hip::device)
  # A .cu imports nothing, so there is nothing to scan for modules.
  set_target_properties(
    fb_localization_kernels PROPERTIES CXX_STANDARD 20
                                       CXX_SCAN_FOR_MODULES OFF
  )
endif()

add_dependencies(rust_native_libs fb_localization_kernels)
