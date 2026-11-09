# The pipeline's footprint vocabulary. Templates and constexpr functions only,
# so no implementation unit, and all PUBLIC: core.footprint is re-exported, so
# a step module that imports this one has MemoryFootprint and then / drop.
# Light on purpose: every step module imports it, so nothing from
# pipeline.step (which will require the footprint, #1122) or any method layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.footprint
  PRIMARY_INTERFACE
  footprint.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::footprint
)
