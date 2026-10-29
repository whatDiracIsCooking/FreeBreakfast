# fp64 GEMM through wwr's cuBLASLt / hipBLASLt matmul surface. Host code only:
# the vendor calls are host API, so there is no device library.
# wwr::extension::blaslt is imported directly for the handle, descriptor and
# layout wrappers, which free_breakfast::core::wwr does not re-export.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.experimental.blas_lt
  PRIMARY_INTERFACE
  blas_lt.cppm
  IMPLEMENTATION
  blas_lt.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::wwr
  free_breakfast::core::resources
  wwr::extension::blaslt
)

# gemm_lt against Dgemm on the RDM digest's shapes, wall clock. A plain
# executable, not a ctest entry, for bench/'s reason.
add_executable(free_breakfast_blas_lt_bench blas_lt_bench.cpp)
set_target_properties(
  free_breakfast_blas_lt_bench PROPERTIES CXX_MODULE_STD 1
                                          CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_blas_lt_bench
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::experimental::blas_lt
)
