# The Scf step. matrix_ao is PUBLIC (re-exported: ScfOutput holds
# SphericalMatrixAOs), and so are the orthogonalizer, core_hamiltonian and guess
# steps, which re-export every other product in run's signature, plus
# pipeline::step and core::resources. Everything run uses -- J, K, DIIS, the
# BLAS conveniences, the log, the Molden writer -- is PRIVATE: run is defined in
# the implementation unit, so no importer compiles against them.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.scf
  PRIMARY_INTERFACE
  scf.cppm
  IMPLEMENTATION
  scf.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::recipes::matrix_ao
  free_breakfast::pipeline::step
  free_breakfast::pipeline::orthogonalizer
  free_breakfast::pipeline::core_hamiltonian
  free_breakfast::pipeline::guess
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::diis
  free_breakfast::io::molden
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
)
