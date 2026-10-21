# ---------------------------------------------------------------------------
# Tests of experimental/
# ---------------------------------------------------------------------------
# Added by the root CMakeLists only under FREE_BREAKFAST_BUILD_EXPERIMENTAL (the
# targets linked here exist only then) and NOT FREE_BREAKFAST_COMPILE_TIME_ONLY
# (GoogleTest exists only then). Its own binaries, apart from test/gtest's two,
# so an experimental test can never break or slow the normal suite, and is
# dropped along with the code it tests.
#
# Registered exactly as test/gtest's GPU binary is -- one ctest entry per case,
# discovered PRE_TEST, labelled `gpu` -- for the reasons that file gives. Each
# case calls FBF_REQUIRE_GPU() and skips without a device.
include(GoogleTest)

add_executable(
  free_breakfast_tests_experimental_gpu radix_sort_gpu.cpp merge_sort_gpu.cpp
)

set_target_properties(
  free_breakfast_tests_experimental_gpu PROPERTIES CXX_MODULE_STD 1
                                                   CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_tests_experimental_gpu
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::experimental::sort::radix
          free_breakfast::experimental::sort::merge
          GTest::gtest
          GTest::gtest_main
)

# gpu_requirement.h is shared with test/gtest rather than copied.
target_include_directories(
  free_breakfast_tests_experimental_gpu
  PRIVATE ${PROJECT_SOURCE_DIR}/test/gtest
)

math(EXPR _fb_test_timeout "120 * ${FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER}")

# Two labels: `gpu`, which the ci-* test presets exclude, and `experimental`,
# so `ctest -L experimental` runs exactly these. The `;` is escaped because
# PROPERTIES is a flat key/value list: unescaped, `experimental` is read as the
# next property's name and silently dropped.
gtest_discover_tests(
  free_breakfast_tests_experimental_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS "gpu\\;experimental"
)

# ---------------------------------------------------------------------------
# Host-only executable
# ---------------------------------------------------------------------------
# As test/gtest's host binary: nothing in it needs a device, so every case runs
# on a device-less box and none skips. Labelled `experimental` only, so the
# ci-* presets' `gpu` exclusion leaves it in.
add_executable(
  free_breakfast_tests_experimental_host
  atomic_integrals_host.cpp atomic_scf_host.cpp atomic_scf_density_host.cpp
)

set_target_properties(
  free_breakfast_tests_experimental_host PROPERTIES CXX_MODULE_STD 1
                                                    CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_tests_experimental_host
  PRIVATE free_breakfast::core
          free_breakfast::basis
          free_breakfast::io
          free_breakfast::experimental::atomic_scf
          GTest::gtest
          GTest::gtest_main
)

# Absolute paths, so a case finds its data whatever directory ctest runs it
# from: this directory's data/ (written by
# experimental/atomic_scf/emit_reference.py), and test/gtest's golden integrals.
target_compile_definitions(
  free_breakfast_tests_experimental_host
  PRIVATE FBF_EXPERIMENTAL_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
          FBF_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/test/gtest/data"
)

gtest_discover_tests(
  free_breakfast_tests_experimental_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS experimental
)
