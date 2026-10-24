# The Fci step. recipes::davidson and pipeline::scf are PUBLIC and re-exported:
# the product holds a DavidsonInfo, and run's View names the Scf and the
# products it re-exports. core::resources and pipeline::step are in run's
# signature. Everything run uses -- the J and K classes, the active-space
# Hamiltonian, the CI strings, sigma, solver and RDM, and core::wwr for the
# memory query -- is PRIVATE: run is defined in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.fci
  PRIMARY_INTERFACE
  fci.cppm
  IMPLEMENTATION
  fci.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::recipes::davidson
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::solver
  free_breakfast::ci::rdm
)
