# ci_solve is a template over the caller's sigma, instantiated in the
# consumer's TU, so what it names -- davidson_solve, DeviceResources, the
# runtime's memcpy (core::wwr), StringSet -- is PUBLIC. The FBF_TRY macro
# expands in the interface unit, so its header is PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.ci.solver
  PRIMARY_INTERFACE
  solver.cppm
  IMPLEMENTATION
  solver.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::wwr
  free_breakfast::core::resources
  free_breakfast::recipes::davidson
  free_breakfast::ci::strings
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
