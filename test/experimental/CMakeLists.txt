# ---------------------------------------------------------------------------
# Tests of experimental/
# ---------------------------------------------------------------------------
# Added by the root CMakeLists only under FREE_BREAKFAST_BUILD_EXPERIMENTAL (the
# targets linked here exist only then) and NOT FREE_BREAKFAST_COMPILE_TIME_ONLY
# (GoogleTest exists only then). Its own binary, apart from test/gtest's two,
# so an experimental test can never break or slow the normal suite, and is
# dropped along with the code it tests.
#
# Registered exactly as test/gtest's GPU binary is -- one ctest entry per case,
# discovered PRE_TEST, labelled `gpu` -- for the reasons that file gives. Each
# case calls FBF_REQUIRE_GPU() and skips without a device.
include(GoogleTest)

add_executable(
  free_breakfast_tests_experimental_gpu radix_sort_gpu.cpp merge_sort_gpu.cpp
                                        fp16_gemm_gpu.cpp output_ring_gpu.cpp
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
          free_breakfast::experimental::fp16_gemm
          free_breakfast::experimental::output_ring
          free_breakfast::io::base
          free_breakfast::io::npy
          wwr::fp16
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
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
