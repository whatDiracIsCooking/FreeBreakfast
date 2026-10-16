# Pure host data: no GPU link, so the guess builds and tests on a device-less
# box. basis is PUBLIC because the interface takes a BasisSet and a Molecule
# (basis re-exports molecule) and a span of Shell; error_handling because it
# returns Result.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.sad
  PRIMARY_INTERFACE
  sad.cppm
  IMPLEMENTATION
  sad.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
