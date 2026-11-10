# The module: response::cphf, declared in the interface and defined in the
# implementation unit. DeviceResources, Buffers and MemoryFootprint are in its
# signatures, so core::wwr, core::resources and core::footprint are PUBLIC; the
# density-fitted J and K, the packed gemm and pcg_solve are only called, so jk,
# recipes::blas and recipes::pcg are PRIVATE. It launches nothing of its own: J
# and K are df_coulomb's and df_exchange's, the rest BLAS and copies, so there
# is no device library.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.response.cphf
  PRIMARY_INTERFACE
  cphf.cppm
  IMPLEMENTATION
  cphf.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::core::footprint
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::pcg
  free_breakfast::integrals::eri::density_fitting::jk
)
