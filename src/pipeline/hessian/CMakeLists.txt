# The Hessian step. geometry and pipeline::gradient are PUBLIC and
# re-exported: the product holds geometry::Vibrations, and pipeline::gradient
# re-exports every step the chain reruns, each a member, and MoleculeOutput,
# which run reads. core::resources and pipeline::step are in run's signature.
# run is defined in the implementation unit, so the analytic Hessian
# (hessian), the copy of C it takes (core::wwr), the planner it nests
# (pipeline::pipeline), the file it writes (io::json), core::log and
# diagnostics_macros (FBF_TRY, FBF_LOG_INFO) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.hessian
  PRIMARY_INTERFACE
  hessian.cppm
  IMPLEMENTATION
  hessian.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::geometry
  free_breakfast::pipeline::step
  free_breakfast::pipeline::gradient
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::hessian
  free_breakfast::io::json
  free_breakfast::pipeline::pipeline
)
