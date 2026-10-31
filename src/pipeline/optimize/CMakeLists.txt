# The Optimize step. geometry, pipeline::gradient, pipeline::casscf_gradient
# and pipeline::nevpt2_gradient are PUBLIC and re-exported: geometry::Options
# is an option, and the three gradient steps' modules re-export every step the
# three chains rerun, each a member, and the products OptimizeOutput and
# MoleculeOutput. core::resources
# and pipeline::step are in run's signature. run is defined in the
# implementation unit, so the planner it nests (pipeline::pipeline), the files
# it writes (io::base, io::json) and reads (io::xyz), atom (element_symbol),
# the carried orbitals' orthonormalization (recipes::blas,
# recipes::orthogonalize, core::wwr), core::constants, core::log and
# diagnostics_macros (FBF_TRY, FBF_LOG_INFO) are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.optimize
  PRIMARY_INTERFACE
  optimize.cppm
  IMPLEMENTATION
  optimize.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::geometry
  free_breakfast::pipeline::step
  free_breakfast::pipeline::gradient
  free_breakfast::pipeline::casscf_gradient
  free_breakfast::pipeline::nevpt2_gradient
  LINK_PRIVATE
  free_breakfast::atom
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::base
  free_breakfast::io::json
  free_breakfast::io::xyz
  free_breakfast::pipeline::pipeline
  free_breakfast::recipes::blas
  free_breakfast::recipes::orthogonalize
)
