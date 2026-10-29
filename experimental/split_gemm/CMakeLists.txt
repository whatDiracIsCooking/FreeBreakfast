# The split and fold kernels. Device code in its own library, so its language
# and -x hip never reach the module compiles. The bridge header it shares with
# the module reads kFourWarps (block_params.h) and math::idivup
# (primitives.h) and names wwrStream_t and wwrError_t, hence kernel_headers
# PUBLIC: it brings primitives_header, types_headers and wwr::device with it.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.experimental.split_gemm.device
  SOURCES
  split_gemm_kernels.cu
  LINK_PUBLIC
  free_breakfast::core::kernel_headers
)

# The module. The device library is PRIVATE: the implementation unit includes
# the bridge header in its GMF and calls the launchers, and a static library's
# private link dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.split_gemm
  PRIMARY_INTERFACE
  split_gemm.cppm
  IMPLEMENTATION
  split_gemm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::experimental::split_gemm::device
)

# The fp64 ceiling probe and the split GEMM against Dgemm, wall clock. A plain
# executable, not a ctest entry, for bench/'s reason.
add_executable(free_breakfast_split_gemm_bench split_gemm_bench.cpp)
set_target_properties(
  free_breakfast_split_gemm_bench PROPERTIES CXX_MODULE_STD 1
                                             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_split_gemm_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::experimental::split_gemm
)
