# ---------------------------------------------------------------------------
# The src/basis tests, as their own host-only executable
# ---------------------------------------------------------------------------
# The TUs whose subject is src/basis and that need no device: each imports
# only free_breakfast.basis, so they build without waiting on any method layer,
# and a change under src/basis/ rebuilds and runs this binary first.
#
#   * free_breakfast_basis_tests_host -- one *_host.cpp TU per partition of
#     free_breakfast.basis (:ao, :shell, :harmonics, :basis_set,
#     :shell_stamp, :basis_stamp). Every case runs on a device-less box.
#
# There is no GPU half. The one basis operation that needs a device is
# place(BasisStamp, Molecule) -- a Molecule owns its nuclei on the device --
# and its cases stay in test/gtest/basis_gpu.cpp rather than being faked here.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and none of their
# suite names (Basis<Partition>, Basis<Partition>DeathTest) is used in
# test/gtest, so their ctest names need no TEST_PREFIX.
#
# devtools/cpp-smoke.sh names the binary, as
# test/basis/free_breakfast_basis_tests_host. Keep the target name in step if
# you rename it.
include(GoogleTest)

add_executable(
  free_breakfast_basis_tests_host
  ao_host.cpp
  shell_host.cpp
  harmonics_host.cpp
  basis_set_host.cpp
  shell_stamp_host.cpp
  basis_stamp_host.cpp
)

set_target_properties(
  free_breakfast_basis_tests_host PROPERTIES CXX_MODULE_STD 1
                                             CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_basis_tests_host PRIVATE free_breakfast::basis GTest::gtest
                                          GTest::gtest_main
)

gtest_discover_tests(
  free_breakfast_basis_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
