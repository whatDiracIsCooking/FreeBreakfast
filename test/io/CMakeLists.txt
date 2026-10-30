# ---------------------------------------------------------------------------
# The src/io tests that need no device, as their own host executable
# ---------------------------------------------------------------------------
# The TUs whose subject is src/io and whose every case runs without a GPU: the
# file, JSON, TOML and .npy readers and writers, the BSE basis reader and the
# compiled-in basis and SAD tables, and the Molden writer. A change under
# src/io/ rebuilds and runs this binary without waiting on any method layer.
#
#   * free_breakfast_io_tests_host -- the *_host.cpp TUs; need no device.
#
# There is no GPU half. The io cases that need a device are the ones that build
# a Molecule, whose nuclei live on it -- the XYZ reader and the basis's
# placement over a molecule -- and those stay in test/gtest
# (io_xyz_gpu.cpp, io_basis_gpu.cpp) beside the rest of the GPU suite. The BSE
# fixture both halves read, io_basis_json.h, is in test/support.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and none of
# their (suite, case) names is used in test/gtest, so their ctest names need no
# TEST_PREFIX.
#
# devtools/cpp-smoke.sh names the binary, as
# test/io/free_breakfast_io_tests_host. Keep the target name in step if you
# rename it.
include(GoogleTest)

add_executable(
  free_breakfast_io_tests_host
  base_host.cpp
  json_host.cpp
  toml_host.cpp
  npy_host.cpp
  basis_host.cpp
  sad_host.cpp
  molden_host.cpp
)

set_target_properties(
  free_breakfast_io_tests_host PROPERTIES CXX_MODULE_STD 1
                                          CXX_SCAN_FOR_MODULES ON
)

# free_breakfast::sad is sad_host.cpp's: each compiled-in stamp is held to the
# atomic SCF it stands in for, which runs on the host.
target_link_libraries(
  free_breakfast_io_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::atom
          free_breakfast::basis
          free_breakfast::sad
          free_breakfast::io
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

gtest_discover_tests(
  free_breakfast_io_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
