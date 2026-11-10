# The pipeline's footprint vocabulary, and pre-flight's fold held to the
# [memory] caps (check_plan, check_pinned_node: the implementation unit, for
# its messages and log lines). core.footprint is re-exported, so a step module
# that imports this one has MemoryFootprint and then / drop, and basis is, so
# it can read the bases FootprintDimensions carries. Light on purpose: every
# step module imports it, and pipeline.step, whose concept requires the
# footprint member (#1122), imports it too: so nothing from pipeline.step or
# any method layer.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.footprint
  PRIMARY_INTERFACE
  footprint.cppm
  IMPLEMENTATION
  footprint.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::footprint
  free_breakfast::basis
  LINK_PRIVATE
  free_breakfast::core::log
  free_breakfast::core::diagnostics_macros
)
