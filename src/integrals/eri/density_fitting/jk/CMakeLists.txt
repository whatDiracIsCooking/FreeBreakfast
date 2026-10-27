# The module. Host code over wwr's BLAS, with no kernels of its own: two gemvs
# for J; a batched gemm, a syrk, a geam and a copy for K. DeviceResources is in
# its signature, so core::resources is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.jk
  PRIMARY_INTERFACE
  jk.cppm
  IMPLEMENTATION
  jk.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
