# The module: J - K/2 and K of a non-symmetric density, from coulomb and
# exchange and BLAS, so no device library. DeviceResources, the two kinds of
# pair class and CartesianMatrixAO are in its signature, so core::resources,
# coulomb, exchange (which carry the classes) and recipes::matrix_ao are
# PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk
  PRIMARY_INTERFACE
  nonsymmetric_jk.cppm
  IMPLEMENTATION
  nonsymmetric_jk.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
