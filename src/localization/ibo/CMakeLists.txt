# The module. Host code over wwr's BLAS and solver, orthogonalize and
# overlap_mixed, with no kernels of its own: the localization is
# pipek_mezey's, re-exported for its options and info, so it is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.localization.ibo
  PRIMARY_INTERFACE
  ibo.cppm
  IMPLEMENTATION
  ibo.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::localization::pipek_mezey
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::orthogonalize
  free_breakfast::integrals::one_electron::overlap_mixed
)
