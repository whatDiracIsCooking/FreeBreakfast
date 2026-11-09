# The plan's shape: a planned Calculation's bases placed and its footprint
# dimensions filled, before the first step. pipeline::config and
# pipeline::footprint are PUBLIC and re-exported: plan_shape reads a
# Calculation and returns the dimensions. core::resources is in its signature.
# The rest -- the aux and MINAO readers, the solver's size query, the steps'
# View -- is PRIVATE: it is the implementation unit's.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.shape
  PRIMARY_INTERFACE
  shape.cppm
  IMPLEMENTATION
  shape.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::pipeline::footprint
  free_breakfast::pipeline::config
  LINK_PRIVATE
  free_breakfast::core::types
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::basis
  free_breakfast::recipes::diagonalize
  free_breakfast::pipeline::step
)
