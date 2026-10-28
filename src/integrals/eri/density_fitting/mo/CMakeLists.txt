# The module. Host code over wwr's BLAS, with no kernels of its own: a batched
# gemm for the half transform, and a gemm after it for the full one.
# DeviceResources is in its signature, so core::resources is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.mo
  PRIMARY_INTERFACE
  mo.cppm
  IMPLEMENTATION
  mo.cpp
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
