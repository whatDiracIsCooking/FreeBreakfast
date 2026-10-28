# The module. Host code over recipes/cholesky and wwr's BLAS, with no kernels
# of its own: the metric's Cholesky factor and one trsm. DeviceResources is in
# its signature, so core::resources is PUBLIC.
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
  free_breakfast::recipes::cholesky
)
