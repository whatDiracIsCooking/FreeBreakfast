# The Cis step. cis::sigma and pipeline::scf are PUBLIC and re-exported: the
# step holds cis::Options (cis::Spin and DavidsonOptions in it), and run's View
# names the ScfOutput and the products it re-exports (the EriOutput among
# them); primitive_pair_classes is named too, as run's View takes its classes.
# core::resources and pipeline::step are in run's signature. What run calls --
# the orbital-energy diagonal, the guess and the preconditioner,
# davidson_solve, the dipole matrices and the gemms over them, core::wwr for
# the copies and core::log for the f-shell warning -- is PRIVATE: run is
# defined in the implementation unit.
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
  free_breakfast::pipeline::primitive_pair_classes
  LINK_PRIVATE
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::recipes::blas
  free_breakfast::recipes::davidson
  free_breakfast::integrals::one_electron::dipole
  free_breakfast::cis::precondition
)
