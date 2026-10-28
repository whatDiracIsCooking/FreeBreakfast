# The Nevpt2 step. pipeline::casscf is PUBLIC and re-exported: run's View
# names the CasscfOutput and the products pipeline::casscf re-exports (the
# EriOutput among them). nevpt2::energy is PUBLIC and re-exported for
# nevpt2::ClassEnergies, in the product. Everything else run uses -- the
# fitted integrals, the RDMs, the Fock, the MO blocks and dm3/f3 -- is
# PRIVATE: run is defined in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.nevpt2
  PRIMARY_INTERFACE
  nevpt2.cppm
  IMPLEMENTATION
  nevpt2.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::pipeline::step
  free_breakfast::pipeline::casscf
  free_breakfast::nevpt2::energy
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::integrals::active_space
  free_breakfast::response::fock
  free_breakfast::ci::strings
  free_breakfast::ci::rdm
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::rdm
)
