# The module: response::cphf, declared in the interface and defined in the
# implementation unit. DeviceResources and Buffers are in its signature, so
# core::wwr and core::resources are PUBLIC; the density-fitted J and K and the
# packed gemm are only called, so jk and recipes::blas are PRIVATE. It
# launches nothing of its own: J and K are df_coulomb's and df_exchange's, the
# rest BLAS and copies, so there is no device library.
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
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::eri::density_fitting::jk
)
