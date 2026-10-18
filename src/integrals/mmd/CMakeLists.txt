# Pure host data: no GPU link, so the records build and are tested on a
# device-less box. basis is PUBLIC because the interface takes a BasisSet.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.integrals.mmd
  PRIMARY_INTERFACE
  mmd.cppm
  IMPLEMENTATION
  mmd.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::error_handling
)
