# The Overlap step. matrix_ao is PUBLIC (re-exported: the step's output is a
# SphericalMatrixAO), and so are the spherical_transform and
# primitive_pair_classes steps, the basis_set step, pipeline::step and
# core::resources, since their types are in run's signature. Everything run
# uses -- the overlap kernels, the copy, the log and the .npy writer -- is
# PRIVATE: run is defined in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.overlap
  PRIMARY_INTERFACE
  overlap.cppm
  IMPLEMENTATION
  overlap.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::pipeline::basis_set
  free_breakfast::recipes::matrix_ao
  free_breakfast::pipeline::step
  free_breakfast::pipeline::spherical_transform
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::io::npy
  free_breakfast::integrals::one_electron::overlap
  free_breakfast::core::diagnostics_macros
)
