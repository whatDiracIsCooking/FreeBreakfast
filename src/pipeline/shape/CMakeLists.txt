# The plan's shape: a planned Calculation's bases placed and its footprint
# dimensions filled, before the first step; and preflight, the plan held to
# the [memory] caps. pipeline::config and pipeline::footprint are PUBLIC and
# re-exported: plan_shape reads a Calculation and returns the dimensions.
# core::resources is in their signatures. The rest -- the aux and MINAO
# readers, the Molecule's footprint (the reader's workspaces), the solvers'
# size queries (syevd's, skew_eig's, getrf's), the
# steps' View, the pool's buffers for
# the pre-warm, the log and response::zvector's rotating_pairs (the CASSCF's
# weights) -- is PRIVATE: it is the implementation unit's.
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
  free_breakfast::core::wwr
  free_breakfast::core::log
  free_breakfast::core::primitives
  free_breakfast::io::basis
  free_breakfast::molecule
  free_breakfast::recipes::diagonalize
  free_breakfast::recipes::expm
  free_breakfast::recipes::lu
  free_breakfast::response::zvector
  free_breakfast::pipeline::step
)
