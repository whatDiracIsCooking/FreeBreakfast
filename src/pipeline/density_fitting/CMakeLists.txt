# The DensityFitting step. core::wwr is PUBLIC and re-exported: the product
# holds Buffers. The basis_set and primitive_pair_classes steps are too, since
# run's View names their products; core::resources and pipeline::step are in
# run's signature. run is defined in the implementation unit, so io::basis
# (resolve_aux_basis), recipes::matrix_ao (the metric's MatrixAOPair), the
# density_fitting aggregate and diagnostics_macros (FBF_TRY) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.density_fitting
  PRIMARY_INTERFACE
  density_fitting.cppm
  IMPLEMENTATION
  density_fitting.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::pipeline::basis_set
  free_breakfast::pipeline::primitive_pair_classes
  free_breakfast::pipeline::step
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::integrals::eri::density_fitting
)
