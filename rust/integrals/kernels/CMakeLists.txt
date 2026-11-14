# libfb_integrals_kernels.so: the integrals launch wrappers rust/ has forked
# from src/integrals -- mmd's order, records, bound and dressed bound, the
# overlap, the kinetic energy, the nuclear attraction, and density fitting's
# metric, three-center integrals and their symmetrization -- with the
# FelizJueves units they launch, one shared object with its own device link,
# which fb-integrals-sys (../sys) links (docs/gpu_layer.md, "Build"). Built
# from these forked sources, rust/device_headers and FelizJueves's include/
# alone: nothing under src/ is compiled or included.
#
# The emitted units (mmd/kernels/, one_electron/{overlap,kinetic,
# potential_en}/kernels/, eri/density_fitting/{metric,three_center}/kernels/,
# and shared/'s Boys routine) are tools/emit/one_electron_kernels.py's,
# tools/emit/exchange_kernels.py's and tools/emit/df_kernels.py's, copied here
# byte for byte (README.md, "Build and test").
#
# Linked: wwr::device PRIVATE for runtime.h and WWR_WARP_SIZE; the C ABI's
# runtime_api library PRIVATE for wwr_get_last_error(); fb_recipes_kernels
# PRIVATE for fb_merge_sort, which the order calls, a host call between the
# two objects with no device symbol crossing. The vendor runtime is the shared
# one, as in ../../recipes/kernels, so the launches here and that query read
# the same per-thread error.
file(GLOB _fb_integrals_units CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/mmd/kernels/pair_bound/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/mmd/kernels/record_bound/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/one_electron/overlap/kernels/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/one_electron/kinetic/kernels/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/one_electron/potential_en/kernels/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/eri/density_fitting/metric/kernels/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/eri/density_fitting/three_center/kernels/*.cu
)
set(_fb_integrals_sources
    abi.cu
    mmd/mmd_launch.cu
    one_electron/overlap/overlap_launch.cu
    one_electron/kinetic/kinetic_launch.cu
    one_electron/potential_en/potential_en_launch.cu
    eri/density_fitting/metric/metric_launch.cu
    eri/density_fitting/three_center/three_center_launch.cu
    ${_fb_integrals_units}
)
add_library(fb_integrals_kernels SHARED ${_fb_integrals_sources})

# FelizJueves's runtime.h, which every emitted unit includes, and the warp
# width its units static-assert against: wwr's, mirrored as at the root
# (deps/CMakeLists.txt), never set anywhere else. shared/ is eval_boys.h, the
# Boys routine the nuclear attraction's and density fitting's units include by
# bare name.
set(_fb_felizjueves_include
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../deps/FelizJueves/include"
)
if(NOT EXISTS "${_fb_felizjueves_include}/felizjueves/runtime.h")
  message(
    FATAL_ERROR
      "deps/FelizJueves is empty -- the submodule has not been initialised. "
      "Run `git submodule update --init --recursive` and configure again."
  )
endif()
target_include_directories(
  fb_integrals_kernels
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${FB_RUST_DEVICE_HEADERS}
          ${FB_RUST_CAPI_GEN} ${_fb_felizjueves_include}
          ${CMAKE_CURRENT_SOURCE_DIR}/shared
          ${CMAKE_CURRENT_SOURCE_DIR}/../../recipes/kernels
)
target_compile_definitions(
  fb_integrals_kernels PRIVATE FJ_WAVEFRONT=${WWR_WARP_SIZE}
)
target_link_libraries(
  fb_integrals_kernels PRIVATE wwr::device wwr::capi::runtime_api
                               fb_recipes_kernels
)
set_target_properties(
  fb_integrals_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  # Separable compilation with the device link resolved inside the .so, as in
  # ../../recipes/kernels: the launchers launch specializations other units
  # define. The shared cudart, so wwr_get_last_error() reads these launches.
  set_target_properties(
    fb_integrals_kernels
    PROPERTIES CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON
               CUDA_RESOLVE_DEVICE_SYMBOLS ON
               CUDA_RUNTIME_LIBRARY Shared
  )
else()
  # A HIP build enables no CUDA language: every .cu is CXX, and hip::device's
  # -x hip reaches this target's own sources only (PRIVATE, never PUBLIC).
  # No -fgpu-rdc, as src/'s device libraries have none: a HIP launch names
  # the kernel's host stub, and the unit defining it registers its own code.
  set_source_files_properties(
    ${_fb_integrals_sources} PROPERTIES LANGUAGE CXX
  )
  target_link_libraries(fb_integrals_kernels PRIVATE hip::device)
  # A .cu imports nothing, so there is nothing to scan for modules.
  set_target_properties(
    fb_integrals_kernels PROPERTIES CXX_STANDARD 20 CXX_SCAN_FOR_MODULES OFF
  )
endif()

add_dependencies(rust_native_libs fb_integrals_kernels)
