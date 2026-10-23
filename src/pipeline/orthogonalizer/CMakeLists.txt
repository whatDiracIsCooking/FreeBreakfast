# The Orthogonalizer step. matrix_ao is PUBLIC (re-exported: the step's output
# is a SphericalMatrixAO), and so are the overlap step, basis, pipeline::step
# and core::resources, since their types are in run's signature.
# recipes::orthogonalize is PUBLIC too: run's inline body calls orthogonalize
# from the importer's TU, and the option's default is its constant.
# diagnostics_macros is only FBF_TRY, expanded here.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.orthogonalizer
  PRIMARY_INTERFACE
  orthogonalizer.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::basis
  free_breakfast::recipes::matrix_ao
  free_breakfast::recipes::orthogonalize
  free_breakfast::pipeline::step
  free_breakfast::pipeline::overlap
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
)
