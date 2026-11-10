# The Nevpt2Nac step. nevpt2::nac and pipeline::casscf are PUBLIC and
# re-exported: the product holds nevpt2::QdNac, and run's View names the
# CasscfOutput and the products pipeline::casscf re-exports (the EriOutput,
# the ScfOutput and the CoreHamiltonianOutput among them);
# primitive_pair_classes is named too, as run's View takes its classes.
# mcscf::nac (the step's NacOptions), pipeline::nevpt2_gradient (its
# Nevpt2Variant), core::resources, pipeline::step and pipeline::footprint
# are in the interface. run is defined in the implementation unit, so
# core::wwr (the copies to and from the device), core::log, core::primitives
# (the CI's binomials), io::json (nevpt2_nac.json),
# recipes::diagonalize, nevpt2::energy and nevpt2::gradient (H_eff and R) and
# diagnostics_macros (FBF_TRY, FBF_LOG_INFO) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.nevpt2_nac
  PRIMARY_INTERFACE
  nevpt2_nac.cppm
  IMPLEMENTATION
  nevpt2_nac.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::mcscf::nac
  free_breakfast::nevpt2::nac
  free_breakfast::pipeline::step
  free_breakfast::pipeline::footprint
  free_breakfast::pipeline::casscf
  free_breakfast::pipeline::nevpt2_gradient
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::primitives
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::json
  free_breakfast::recipes::diagonalize
  free_breakfast::nevpt2::density_fitting
  free_breakfast::nevpt2::energy
  free_breakfast::nevpt2::gradient
)
