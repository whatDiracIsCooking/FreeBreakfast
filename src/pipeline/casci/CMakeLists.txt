# The Casci step. recipes::davidson, pipeline::scf and pipeline::active_space
# are PUBLIC and re-exported: the product holds a DavidsonInfo and a
# SphericalMatrixAO, run's View names the Scf and the products it re-exports,
# and CasciStep derives from ActiveSpaceSelection. core::resources and
# pipeline::step are in run's signature. Everything run uses -- the J and K
# classes, the active-space Hamiltonian, the CI strings, sigma and solver, the
# eigensolver for the natural orbitals, and the Molden writer -- is PRIVATE:
# run is defined in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.casci
  PRIMARY_INTERFACE
  casci.cppm
  IMPLEMENTATION
  casci.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::recipes::davidson
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  free_breakfast::pipeline::active_space
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::integrals::active_space
  free_breakfast::ci::strings
  free_breakfast::ci::sigma
  free_breakfast::ci::solver
  free_breakfast::io::molden
  free_breakfast::recipes::diagonalize
)
