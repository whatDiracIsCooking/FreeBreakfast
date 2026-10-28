# The active-space selection shared by the steps that take one (Casci, and
# Casscf to come): not a step, a struct their step configs derive from.
# pipeline::scf is PUBLIC and re-exported: select reads the ScfOutput and
# returns a SphericalMatrixAO. core::resources is in select's signature.
# Everything select uses -- AVAS's MINAO basis, mixed overlaps and eigensolver,
# and the gemm for C R -- is PRIVATE: it is defined in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.active_space
  PRIMARY_INTERFACE
  active_space.cppm
  IMPLEMENTATION
  active_space.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::atom
  free_breakfast::io::basis
  free_breakfast::recipes::blas
  free_breakfast::recipes::diagonalize
  free_breakfast::integrals::one_electron::overlap_mixed
)
