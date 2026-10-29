# The Cis step. cis::sigma and pipeline::scf are PUBLIC and re-exported: the
# step holds cis::Options (cis::Spin and DavidsonOptions in it), and run's View
# names the ScfOutput and the products it re-exports (the EriOutput among
# them). core::resources and pipeline::step are in run's signature. What run
# calls -- the orbital-energy diagonal, the guess and the preconditioner,
# davidson_solve, and core::wwr for the copies -- is PRIVATE: run is defined
# in the implementation unit.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.cis
  PRIMARY_INTERFACE
  cis.cppm
  IMPLEMENTATION
  cis.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::cis::sigma
  free_breakfast::pipeline::step
  free_breakfast::pipeline::scf
  LINK_PRIVATE
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::davidson
  free_breakfast::cis::precondition
)
