# Everything PUBLIC: the module is templates only, instantiated in the
# consumer's TU, so what they name -- wwr's dispatch, ScopedPointerMode and
# CreatePolicy (core::wwr), kOne / kZero (core::constants) -- must reach it.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg.blas
  PRIMARY_INTERFACE
  blas.cppm
  PARTITIONS
  shared.cppm
  gemm.cppm
  gemmwich.cppm
  gmmwich.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::constants
  free_breakfast::core::wwr
)
