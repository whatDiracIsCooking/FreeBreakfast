# ---------------------------------------------------------------------------
# The src/sad tests, as their own host executable
# ---------------------------------------------------------------------------
# The TUs whose subject is src/sad. The module is pure host data with no GPU
# link (src/sad/README.md), so its tests need no device and this tree is one
# binary rather than a host/GPU pair:
#
#   * free_breakfast_sad_tests_host -- the *_host.cpp TUs; need no device.
#
# Two SAD tests stay in test/gtest, because src/sad is not their subject alone:
# sad_gpu.cpp places its basis on a Molecule, whose nuclei live on the device,
# and io_sad_host.cpp holds src/io/sad's compiled-in stamps to this module's
# SCF.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL. Their suite name,
# Sad, is shared with sad_gpu.cpp's but no case name is, so their ctest names
# need no TEST_PREFIX.
#
# devtools/cpp-smoke.sh names the binary, as
# test/sad/free_breakfast_sad_tests_host. Keep the target name in step if you
# rename it.
include(GoogleTest)

add_executable(free_breakfast_sad_tests_host sad_host.cpp)

set_target_properties(
  free_breakfast_sad_tests_host PROPERTIES CXX_MODULE_STD 1
                                           CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_sad_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::basis
          free_breakfast::sad
          free_breakfast::io
          GTest::gtest
          GTest::gtest_main
)

# The PySCF reference (data/README.md). An absolute path, so a test finds it
# whatever directory ctest runs it from.
target_compile_definitions(
  free_breakfast_sad_tests_host
  PRIVATE FBF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

gtest_discover_tests(
  free_breakfast_sad_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
