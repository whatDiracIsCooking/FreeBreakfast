# The Nac step. mcscf::nac and pipeline::casscf are PUBLIC and re-exported:
# the product holds mcscf::Nac and the step mcscf::NacOptions, and run's View
# names the CasscfOutput and the products pipeline::casscf re-exports (the
# EriOutput and the CoreHamiltonianOutput among them); primitive_pair_classes
# is named too, as run's View takes its classes. core::resources and
# pipeline::step are in run's signature. run is defined in the implementation
# unit, so core::wwr (the copies back to the device), core::log, io::json
# (nac.json) and diagnostics_macros (FBF_TRY, FBF_LOG_INFO) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.nac
  PRIMARY_INTERFACE
  nac.cppm
  IMPLEMENTATION
  nac.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::mcscf::nac
  free_breakfast::pipeline::step
  free_breakfast::pipeline::casscf
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::json
)
