# The DF exchange build cut into batches, B resident or streamed from pinned
# memory: the per-batch cost on top of the one-call build. Host code only --
# the BLAS calls, copies, events and graph capture are all host API -- so there
# is no device library and no module. A plain executable, not a ctest entry,
# for bench/'s reason.
add_executable(
  free_breakfast_gemm_host_offload_bench gemm_host_offload_bench.cpp
)
set_target_properties(
  free_breakfast_gemm_host_offload_bench PROPERTIES CXX_MODULE_STD 1
                                                    CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_gemm_host_offload_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
)
