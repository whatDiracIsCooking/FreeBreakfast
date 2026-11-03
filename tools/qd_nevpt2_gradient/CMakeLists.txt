# free-breakfast-qd-nevpt2-gradient: the generator of
# test/gtest/data/qd_nevpt2_gradient/, run by hand on a GPU (see main.cpp and
# test/gtest/data/README.md). Built with the tree, as tools/sad is: a
# generator nothing builds rots unseen until the day its data needs
# regenerating.
#
# The energy it differences is the pipeline's own, so it links the pipeline
# steps it reruns (pipeline::nevpt2 re-exports the chain up to it), and the
# mcscf gradient and state overlaps its guard reads. Not a module library, so
# it sets the module-scanning properties itself, as src/main.cpp does.
add_executable(free_breakfast_qd_nevpt2_gradient main.cpp)
set_target_properties(
  free_breakfast_qd_nevpt2_gradient
  PROPERTIES OUTPUT_NAME free-breakfast-qd-nevpt2-gradient CXX_MODULE_STD 1
             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_qd_nevpt2_gradient
  PRIVATE free_breakfast::core::types
          free_breakfast::core::error_handling
          free_breakfast::core::constants
          free_breakfast::core::log
          free_breakfast::core::wwr
          free_breakfast::core::resources
          free_breakfast::core::diagnostics_macros
          free_breakfast::atom
          free_breakfast::io::base
          free_breakfast::io::json
          free_breakfast::integrals::one_electron::overlap_mixed
          free_breakfast::mcscf::gradient
          free_breakfast::mcscf::nac
          free_breakfast::mcscf::overlap
          free_breakfast::recipes::blas
          free_breakfast::recipes::diagonalize
          free_breakfast::recipes::orthogonalize
          free_breakfast::pipeline::pipeline
          free_breakfast::pipeline::step
          free_breakfast::pipeline::overlap
          free_breakfast::pipeline::nevpt2
)
