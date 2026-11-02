# The module: mcscf::state_overlaps, host code over the mixed AO overlap,
# wwr's BLAS and the CI strings, with no kernels of its own (README.md says
# why the string determinants are a host loop). DeviceResources, BasisSet,
# Solution and Result are in its signature, so core::resources, basis,
# mcscf::solve and core::error_handling are PUBLIC. The calls it makes are
# PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.mcscf.overlap
  PRIMARY_INTERFACE
  overlap.cppm
  IMPLEMENTATION
  overlap.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::mcscf::solve
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::one_electron::overlap_mixed
  free_breakfast::ci::strings
)
