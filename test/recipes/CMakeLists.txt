# ---------------------------------------------------------------------------
# The src/recipes tests, as a host/GPU pair of executables
# ---------------------------------------------------------------------------
# The TUs whose subject is src/recipes: each imports only free_breakfast.core*,
# free_breakfast.recipes* and (matrix_ao_gpu.cpp) free_breakfast.basis, so they
# build without waiting on integrals, mcscf, nevpt2 or any other method layer,
# and a change under src/recipes/ rebuilds and runs this binary first.
#
#   * free_breakfast_recipes_tests_host -- the *_host.cpp TUs; need no
#     device: einsum_host.cpp (einsum's refusals return before any launch)
#     and matrix_ao_host.cpp (ao_count reads only the host BasisSet).
#   * free_breakfast_recipes_tests_gpu -- the *_gpu.cpp TUs; each guards
#     itself with FBF_REQUIRE_GPU() and SKIPS without a device.
#
# The tier decides the binary (docs/testing.md "Which tier a check goes in"):
# a host case in the GPU binary would carry its `gpu` label, which the ci-cuda
# and ci-hip test presets exclude, and would run in test.py's GPU batch. The
# host binary still links the device runtime through free_breakfast::recipes;
# what makes it host is that no case initialises a device. The compile-time
# checks of the exported concepts are test/compile/recipes_compile.cpp.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and none of
# their suite names is used in another tree, so their ctest names need no
# TEST_PREFIX.
#
# devtools/test.py (push and pr) names both binaries, as
# test/recipes/free_breakfast_recipes_tests_{host,gpu}. Keep the target names
# in step if you rename them.
include(GoogleTest)

add_executable(
  free_breakfast_recipes_tests_host einsum_host.cpp matrix_ao_host.cpp
)

set_target_properties(
  free_breakfast_recipes_tests_host PROPERTIES CXX_MODULE_STD 1
                                               CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_recipes_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::recipes
          free_breakfast::basis
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

gtest_discover_tests(
  free_breakfast_recipes_tests_host
  DISCOVERY_MODE PRE_TEST
  XML_OUTPUT_DIR ${_fb_gtest_xml_dir}
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)

add_executable(
  free_breakfast_recipes_tests_gpu
  gemm_gpu.cpp
  gemmwich_gpu.cpp
  gmmwich_gpu.cpp
  cg_unitary_gpu.cpp
  cholesky_gpu.cpp
  complex_cast_gpu.cpp
  davidson_gpu.cpp
  diagonalize_gpu.cpp
  diis_gpu.cpp
  einsum_gpu.cpp
  expm_gpu.cpp
  lu_gpu.cpp
  matrix_ao_gpu.cpp
  norm_gpu.cpp
  orthogonalize_gpu.cpp
  orthonormalize_gpu.cpp
  pcg_gpu.cpp
  polynomial_roots_gpu.cpp
  sort_gpu.cpp
  symmetrize_gpu.cpp
)

set_target_properties(
  free_breakfast_recipes_tests_gpu PROPERTIES CXX_MODULE_STD 1
                                              CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_recipes_tests_gpu
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::recipes
          free_breakfast::basis
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

# The properties are set here and are NOT inherited from test/gtest's: the
# `gpu` label is what the ci-cuda and ci-hip test presets exclude (`-LE gpu`)
# and what devtools/test.py pr batches on.
gtest_discover_tests(
  free_breakfast_recipes_tests_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS gpu
)
