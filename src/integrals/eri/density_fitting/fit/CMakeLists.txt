# The module. Host code over wwr's solver and BLAS, with no kernels of its own:
# a potrf of the metric and one trsm. DeviceResources is in its signature, so
# core::resources is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.density_fitting.fit
  PRIMARY_INTERFACE
  fit.cppm
  IMPLEMENTATION
  fit.cpp
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
