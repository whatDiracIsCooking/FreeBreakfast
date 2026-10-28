# The Scf step. matrix_ao is PUBLIC (re-exported: ScfOutput holds
# SphericalMatrixAOs), and so are the orthogonalizer, core_hamiltonian, guess
# and eri steps, which re-export every other product in run's signature, plus
# pipeline::step and core::resources. Everything run uses -- J, K, DIIS, the
# BLAS conveniences, the log, the Molden writer and DF J and K -- is PRIVATE:
# run and its Fock builders are defined in the implementation unit and the loop
# in the internal partition :loop, so no importer compiles against them.
# :output is an interface partition only so that :loop can name ScfOutput.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.scf
  PRIMARY_INTERFACE
  scf.cppm
  PARTITIONS
  scf_output.cppm
  scf_loop.cppm
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
  free_breakfast::pipeline::eri
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
  free_breakfast::integrals::eri::density_fitting::jk
)
