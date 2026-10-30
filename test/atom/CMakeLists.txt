# ---------------------------------------------------------------------------
# The src/atom tests, as a host-only executable
# ---------------------------------------------------------------------------
# src/atom is the Element enum, the Atom aggregate and the symbol table, none
# of which touches a device. So unlike test/core's pair there is no GPU binary:
# free_breakfast_atom_tests_host is the whole suite, and every case runs on a
# device-less box. Add a *_gpu.cpp pair here only when atom grows something a
# device runs.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and their suite
# name (Atom) is used nowhere else, so their ctest names need no TEST_PREFIX.
#
# devtools/cpp-smoke.sh names the binary, as
# test/atom/free_breakfast_atom_tests_host. Keep the target name in step if
# you rename it.
include(GoogleTest)

add_executable(free_breakfast_atom_tests_host atom_host.cpp)

set_target_properties(
  free_breakfast_atom_tests_host PROPERTIES CXX_MODULE_STD 1
                                            CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_atom_tests_host PRIVATE free_breakfast::atom GTest::gtest
                                         GTest::gtest_main
)

gtest_discover_tests(
  free_breakfast_atom_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
