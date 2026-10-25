# The module. Host code over recipes/blas and recipes/diagonalize, with no
# kernels of its own; the IAOs are an argument, so it does not link ibo. The
# localization is pipek_mezey's, re-exported for its options and info, so it
# is PUBLIC.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.localization.livvo
  PRIMARY_INTERFACE
  livvo.cppm
  IMPLEMENTATION
  livvo.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::localization::pipek_mezey
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
)
