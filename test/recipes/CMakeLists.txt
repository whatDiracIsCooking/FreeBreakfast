# ---------------------------------------------------------------------------
# The src/recipes tests, as their own GPU executable
# ---------------------------------------------------------------------------
# The TUs whose subject is src/recipes: each imports only free_breakfast.core*,
# free_breakfast.recipes* and (matrix_ao_gpu.cpp) free_breakfast.basis, so they
# build without waiting on integrals, mcscf, nevpt2 or any other method layer,
# and a change under src/recipes/ rebuilds and runs this binary first.
#
#   * free_breakfast_recipes_tests_gpu -- the *_gpu.cpp TUs; each guards
#     itself with FBF_REQUIRE_GPU() and SKIPS without a device.
#
# Three TUs need no device and are here too: einsum_host.cpp (einsum's
# refusals return before any launch), matrix_ao_host.cpp (ao_count reads only
# the host BasisSet) and concepts.cpp (static_asserts on the exported
# concepts, checked at build time; it declares no case). A host binary for
# them was not worth its own target; the cost is that their cases carry this
# binary's `gpu` label, which the ci-cuda and ci-hip test presets exclude.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and none of
# their suite names is used in another tree, so their ctest names need no
# TEST_PREFIX.
#
# devtools/cpp-smoke.sh and devtools/cpp-tier.sh name the binary, as
# test/recipes/free_breakfast_recipes_tests_gpu. Keep the target name in step
# if you rename it.
include(GoogleTest)

add_executable(
  free_breakfast_recipes_tests_gpu
  gemm_gpu.cpp
  gemmwich_gpu.cpp
  gmmwich_gpu.cpp
  cg_unitary_gpu.cpp
  cholesky_gpu.cpp
  complex_cast_gpu.cpp
  concepts.cpp
  davidson_gpu.cpp
  diagonalize_gpu.cpp
  diis_gpu.cpp
  einsum_gpu.cpp
  einsum_host.cpp
  expm_gpu.cpp
  lu_gpu.cpp
  matrix_ao_gpu.cpp
  matrix_ao_host.cpp
  norm_gpu.cpp
  orthogonalize_gpu.cpp
  orthonormalize_gpu.cpp
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
          # selected_backend.h, for expm_gpu.cpp's one HIP-only operand.
          wwr::backend
)

# The properties are set here and are NOT inherited from test/gtest's: the
# `gpu` label is what the ci-cuda and ci-hip test presets exclude (`-LE gpu`)
# and what devtools/split-gpu-entries.py batches on.
gtest_discover_tests(
  free_breakfast_recipes_tests_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS gpu
)
