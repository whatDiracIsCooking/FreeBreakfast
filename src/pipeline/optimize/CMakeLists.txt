# The Optimize and Irc steps. geometry, pipeline::gradient,
# pipeline::casscf_gradient, pipeline::nevpt2_gradient and pipeline::hessian
# are PUBLIC and re-exported: geometry::Options and geometry::IrcOptions are
# options, the three gradient steps' modules re-export every step the three
# chains rerun, each a member, and the products OptimizeOutput and
# MoleculeOutput, and IrcStep reads HessianOutput. core::resources
# and pipeline::step are in run's signature, pipeline::footprint in the
# footprint's, and io::json in
# recorded_optimize's. run is defined in the
# implementation unit, so the planner it nests (pipeline::pipeline), the files
# it writes (io::base) and reads (io::xyz), atom (element_symbol),
# the carried orbitals' copy to the host (core::wwr), a QD state's root matching
# (mcscf::overlap), the MECI's g_I, g_J and h_IJ (mcscf::intersection) and
# a QD MECI's g_P, g_Q and h_PQ (nevpt2::intersection) over its frozen core
# (pipeline::nevpt2's resolution), the footprint's determinant count
# (core::primitives) and weights (response::zvector's rotating_pairs),
# core::constants, core::log and diagnostics_macros (FBF_TRY, FBF_LOG_INFO)
# are PRIVATE.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.pipeline.optimize
  PRIMARY_INTERFACE
  optimize.cppm
  IMPLEMENTATION
  optimize.cpp
  irc.cpp
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  free_breakfast::core::error_handling
  free_breakfast::core::resources
  free_breakfast::geometry
  free_breakfast::pipeline::step
  free_breakfast::pipeline::footprint
  free_breakfast::pipeline::gradient
  free_breakfast::pipeline::casscf_gradient
  free_breakfast::pipeline::nevpt2_gradient
  free_breakfast::pipeline::hessian
  free_breakfast::io::json
  LINK_PRIVATE
  free_breakfast::atom
  free_breakfast::core::constants
  free_breakfast::core::log
  free_breakfast::core::primitives
  free_breakfast::core::wwr
  free_breakfast::core::diagnostics_macros
  free_breakfast::io::base
  free_breakfast::io::xyz
  free_breakfast::mcscf::intersection
  free_breakfast::mcscf::overlap
  free_breakfast::nevpt2::intersection
  free_breakfast::pipeline::nevpt2
  free_breakfast::pipeline::pipeline
  free_breakfast::response::zvector
)
