# The Gradient step. gradient and pipeline::scf are PUBLIC and re-exported: the
# product holds gradient::RhfTerms, and run's View names the ScfOutput and the
# products it re-exports (the EriOutput among them); primitive_pair_classes is
# named too, as run's View takes its classes. core::resources and
# pipeline::step are in run's signature. run is defined in the implementation
# unit, so io::json (gradient.json) and diagnostics_macros (FBF_TRY) are
# PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.gradient
  PRIMARY_INTERFACE
  gradient.cppm
  IMPLEMENTATION
  gradient.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::gradient
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::json
)
