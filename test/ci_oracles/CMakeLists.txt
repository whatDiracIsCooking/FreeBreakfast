# The host references the CI tests check the device against: host_sigma,
# host_diagonal, host_rdm1, host_rdm12 and host_spin_square. Plain loops over
# the strings' replacement lists, linked by the test executables alone, so the
# ci modules export only what a calculation runs. ci::strings is PUBLIC because
# the interface takes StringSets, core::error_handling because it returns a
# Status; the FBF_TRY header is the implementation's own.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.test.ci_oracles
  PRIMARY_INTERFACE
  ci_oracles.cppm
  IMPLEMENTATION
  ci_oracles.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::ci::strings
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
