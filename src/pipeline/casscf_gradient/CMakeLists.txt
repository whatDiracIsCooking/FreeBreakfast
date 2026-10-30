# The CasscfGradient step. mcscf::gradient and pipeline::casscf are PUBLIC and
# re-exported: the product holds mcscf::GradientTerms, and run's View names the
# CasscfOutput and the products pipeline::casscf re-exports (the EriOutput and
# the CoreHamiltonianOutput among them); primitive_pair_classes is named too,
# as run's View takes its classes. core::resources and pipeline::step are in
# run's signature. run is defined in the implementation unit, so core::wwr
# (the copies back to the device), io::json (casscf_gradient.json) and
# diagnostics_macros (FBF_TRY) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.casscf_gradient
  PRIMARY_INTERFACE
  casscf_gradient.cppm
  IMPLEMENTATION
  casscf_gradient.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::mcscf::gradient
  free_breakfast::pipeline::step
  free_breakfast::pipeline::casscf
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::json
)
