# The module. DeviceResources is in its signature, so core::resources is
# PUBLIC. syevd is wwr's, called only from the implementation unit, so
# core::wwr is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.linalg.diagonalize
  PRIMARY_INTERFACE
  diagonalize.cppm
  IMPLEMENTATION
  diagonalize.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
)
