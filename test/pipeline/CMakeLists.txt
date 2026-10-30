# ---------------------------------------------------------------------------
# The src/pipeline tests that need no device, as their own host executable
# ---------------------------------------------------------------------------
# The TUs whose subject is src/pipeline and that run anywhere: the step
# contract and the planner over made-up steps, the contract of every real
# step, the TOML front end, and the Guess step's load. None constructs a
# DeviceResources, so a green run here is meaningful on a device-less box.
#
#   * free_breakfast_pipeline_tests_host -- the *_host.cpp TUs.
#
# There is no GPU half: running a real step needs a Molecule, which owns its
# nuclei on the device, so those cases stay in test/gtest's
# free_breakfast_tests_gpu, beside the method layers they drive.
#
# The module setup and the ctest registration are the same as test/gtest's,
# which says why for each. None of these cases is RUN_SERIAL, and none of their
# full names is used in test/gtest, so their ctest names need no TEST_PREFIX.
#
# devtools/cpp-smoke.sh names the binary, as
# test/pipeline/free_breakfast_pipeline_tests_host. Keep the target name in
# step if you rename it.
include(GoogleTest)

add_executable(
  free_breakfast_pipeline_tests_host
  step_host.cpp
  pipeline_host.cpp
  steps_host.cpp
  config_host.cpp
  guess_host.cpp
)

set_target_properties(
  free_breakfast_pipeline_tests_host
  PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_pipeline_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::io
          free_breakfast::pipeline::step
          free_breakfast::pipeline::pipeline
          free_breakfast::pipeline::guess
          free_breakfast::pipeline::config
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

gtest_discover_tests(
  free_breakfast_pipeline_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
