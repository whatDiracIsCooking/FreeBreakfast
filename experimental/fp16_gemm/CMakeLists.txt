# Half-precision GEMM through wwr's wwrblasHgemm / wwrblasGemmEx. Host code
# only: the vendor calls are host API, so there is no device library. wwr::fp16
# is imported directly for wwrHalf, which free_breakfast::core::wwr does not
# re-export.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.fp16_gemm
  PRIMARY_INTERFACE
  fp16_gemm.cppm
  IMPLEMENTATION
  fp16_gemm.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::resources
  wwr::fp16
)

# The two gemm_fp16 accumulators against fp32 Sgemm, wall clock. A plain
# executable, not a ctest entry, for bench/'s reason.
add_executable(free_breakfast_fp16_gemm_bench fp16_gemm_bench.cpp)
set_target_properties(
  free_breakfast_fp16_gemm_bench PROPERTIES CXX_MODULE_STD 1
                                            CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_fp16_gemm_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::experimental::fp16_gemm wwr::fp16
)
