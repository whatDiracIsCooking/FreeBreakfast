# A fixed ring of pinned slots that streams device arrays into .npy files, the
# copies on the device's stream and the writes on a taskflow executor. Host
# code only: the copies and events are host API, so there is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.output_ring
  PRIMARY_INTERFACE
  output_ring.cppm
  IMPLEMENTATION
  output_ring.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::nontransferrable
  free_breakfast::core::wwr
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::npy
)

# The ring against the download-sync-write it replaces, k arrays of n x n, wall
# clock. A plain executable, not a ctest entry, for bench/'s reason.
add_executable(free_breakfast_output_ring_bench output_ring_bench.cpp)
set_target_properties(
  free_breakfast_output_ring_bench PROPERTIES CXX_MODULE_STD 1
                                              CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_output_ring_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::io::npy free_breakfast::experimental::output_ring
)
