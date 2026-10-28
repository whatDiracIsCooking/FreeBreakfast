# The Casscf step. mcscf::solve, pipeline::scf and pipeline::active_space are
# PUBLIC and re-exported: the product holds mcscf::Iteration and a
# SphericalMatrixAO, run's View names the ScfOutput and the products it
# re-exports, and CasscfStep derives from ActiveSpaceSelection. core::resources
# and pipeline::step are in run's signature. Everything else run uses -- the J
# and K classes, the integrals and Fock at the solution, the eigensolver for
# the canonical and natural orbitals, the Molden writer, and the DensityFitting
# step that builds B under aux_basis -- is PRIVATE: run is defined in the
# implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.casscf
  PRIMARY_INTERFACE
  casscf.cppm
  IMPLEMENTATION
  casscf.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::mcscf::solve
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  free_breakfast::pipeline::active_space
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::integrals::eri::integral_direct::coulomb
  free_breakfast::integrals::eri::integral_direct::exchange
  free_breakfast::integrals::active_space
  free_breakfast::response::fock
  free_breakfast::pipeline::density_fitting
  free_breakfast::io::molden
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
)
