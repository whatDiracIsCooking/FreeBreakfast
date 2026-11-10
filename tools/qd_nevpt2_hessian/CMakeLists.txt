# free-breakfast-qd-nevpt2-hessian: the analytic QD-NEVPT2 gradients at the
# displaced points tools/nevpt2_hessian/qd_nevpt2_hessian.py differences into
# test/gtest/data/qd_nevpt2_hessian/ (#953), run by hand on a GPU (see
# main.cpp). Built with the tree, as tools/qd_nevpt2_gradient is: a generator
# nothing builds rots unseen until the day its data needs regenerating.
#
# Not a module library: one main.cpp over tools/qd_nevpt2_gradient's chain
# and guard (ChainSampler) and the gradient it differences
# (nevpt2::qd_sc_gradient, qd_pc_gradient), rerun over the pipeline steps the
# sampler's chain is made of, its H_eff eigenstates matched by the [hessian]
# step's own tracked_qd_state and tracked_qd_pair (pipeline::hessian), the one
# R_P match of the tree. It sets the module-scanning properties itself,
# as src/main.cpp does.
add_executable(free_breakfast_qd_nevpt2_hessian main.cpp)
set_target_properties(
  free_breakfast_qd_nevpt2_hessian
  PROPERTIES OUTPUT_NAME free-breakfast-qd-nevpt2-hessian CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_qd_nevpt2_hessian
  PRIVATE free_breakfast::tools::qd_nevpt2_gradient
          free_breakfast::core::types
          free_breakfast::core::error_handling
          free_breakfast::core::log
          free_breakfast::core::resources
          free_breakfast::core::wwr
          free_breakfast::core::diagnostics_macros
          free_breakfast::atom
          free_breakfast::io::base
          free_breakfast::io::json
          free_breakfast::mcscf::gradient
          free_breakfast::mcscf::overlap
          free_breakfast::nevpt2::gradient
          free_breakfast::pipeline::pipeline
          free_breakfast::molecule
          free_breakfast::pipeline::step
          free_breakfast::pipeline::footprint
          free_breakfast::pipeline::overlap
          free_breakfast::pipeline::nevpt2
          free_breakfast::pipeline::hessian
)
