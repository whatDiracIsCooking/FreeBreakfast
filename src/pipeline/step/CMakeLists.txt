# The step contract. Templates only, so no implementation unit: every name a
# step's run signature or the concept checks reaches (Result, DeviceResources,
# NodeResources, tf::Subflow) is instantiated in the importer's TU, hence all
# PUBLIC. Nothing from integrals or linalg: a leaf, so a new method never
# recompiles it.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.step
  PRIMARY_INTERFACE
  step.cppm
  PARTITIONS
  view.cppm
  run_shape.cppm
  pipeline_step.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
)
