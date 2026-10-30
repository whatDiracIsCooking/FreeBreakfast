# The Cartesian minimizer: host-only, over a caller's energy and gradient, so
# it links no GPU target and nothing of molecule, the SCF or the pipeline.
# Result and the error kinds are in its signature, so core::error_handling is
# PUBLIC; FBF_TRY is the implementation unit's own business.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.geometry
  PRIMARY_INTERFACE
  geometry.cppm
  IMPLEMENTATION
  geometry.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
