# The vendor reference over CUB / hipCUB. Device code in its own library, so
# its language and -x hip never reach the module compiles. The merge sort
# itself is src/integrals/sort's merge_sort.cu, which the integral builds sort
# with; this tree links it rather than keeping a copy.
# The bridge header this library shares with the module names wwrStream_t and
# wwrError_t, hence kernel_headers PUBLIC: it brings types_headers and
# wwr::device with it. wwr::device also carries cub.h and the vendor's CUB or
# hipCUB include path, as for sort/radix.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.experimental.sort.merge.device
  SOURCES
  cub_merge_sort.cu
  LINK_PUBLIC
  free_breakfast::core::kernel_headers
)

# The module. Both device libraries are PRIVATE: the implementation unit
# includes their bridge headers in its GMF and calls the launchers.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.sort.merge
  PRIMARY_INTERFACE
  merge_sort.cppm
  IMPLEMENTATION
  merge_sort.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::experimental::sort::merge::device
  free_breakfast::integrals::sort::device
)

# merge_sort against radix_sort and the vendor's merge and radix sorts, wall
# clock. Not a
# ctest entry, for bench/'s reason: a timing that fails a threshold on a busy
# machine is noise.
add_executable(free_breakfast_merge_sort_bench merge_sort_bench.cpp)
set_target_properties(
  free_breakfast_merge_sort_bench PROPERTIES CXX_MODULE_STD 1
                                             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_merge_sort_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::experimental::sort::merge
          free_breakfast::experimental::sort::radix
)
