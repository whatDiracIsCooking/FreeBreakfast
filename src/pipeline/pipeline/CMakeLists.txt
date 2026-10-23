# Pipeline<Steps...>, the planner. Templates only, so no implementation unit,
# and every module it reaches is instantiated in the importer's TU, hence all
# PUBLIC: core::wwr for ScopedDeviceIndex, core::resources for the node and
# Taskflow. No step module: a registry of made-up steps compiles none of the
# real ones. diagnostics_macros is FBF_TRY, expanded in this interface.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.pipeline
  PRIMARY_INTERFACE
  pipeline.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::pipeline::step
)
