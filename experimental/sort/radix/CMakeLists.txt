# The radix-sort kernels, and the vendor reference over CUB / hipCUB. Device
# code in its own library, so its language and -x hip never reach the module
# compiles. The bridge header it shares with the module reads kFourWarps
# (block_params.h) and names wwrStream_t, hence kernel_headers PUBLIC: it brings
# types_headers and wwr::device with it. wwr::device also carries cub.h and the
# include path of the vendor's CUB (the toolkit's cccl dir) or hipCUB (the ROCm
# include root), so the reference needs no find_package.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.experimental.sort.radix.device
  SOURCES
  radix_sort.cu
  cub_radix_sort.cu
  LINK_PUBLIC
  free_breakfast::core::kernel_headers
)

# The module. The device library is PRIVATE: the implementation unit includes
# the bridge header in its GMF and calls the launchers, and a static library's
# private link dependencies still reach the final link.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.sort.radix
  PRIMARY_INTERFACE
  radix_sort.cppm
  IMPLEMENTATION
  radix_sort.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::experimental::sort::radix::device
)

# radix_sort against cub_radix_sort, wall clock. A plain executable, not a ctest
# entry, for bench/'s reason: a timing that fails a threshold on a busy machine
# is noise. Built with the rest of experimental/, so under the same option.
add_executable(free_breakfast_radix_sort_bench radix_sort_bench.cpp)
set_target_properties(
  free_breakfast_radix_sort_bench PROPERTIES CXX_MODULE_STD 1
                                             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_radix_sort_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::experimental::sort::radix
)
