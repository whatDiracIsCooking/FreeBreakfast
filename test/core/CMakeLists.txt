# ---------------------------------------------------------------------------
# The src/core tests, as their own host/GPU executable pair
# ---------------------------------------------------------------------------
# The TUs whose subject is src/core: each imports only free_breakfast.core*
# modules, so they build without waiting on any method layer, and a change
# under src/core/ rebuilds and runs these two binaries first.
#
#   * free_breakfast_core_tests_host -- the *_host.cpp TUs; need no device.
#   * free_breakfast_core_tests_gpu  -- the *_gpu.cpp TUs; each guards itself
#     with FBF_REQUIRE_GPU() and SKIPS without a device.
#
# The split, the module setup and the ctest registration are the same as
# test/gtest's, which says why for each. None of these cases is RUN_SERIAL, and
# none of their suite names is used in test/gtest, so their ctest names need no
# TEST_PREFIX.
#
# gpu_kernels/ -- the .cu kernels primitives_gpu.cpp and reduce_gpu.cpp launch
# -- is NOT added here but from the root CMakeLists, outside the
# FREE_BREAKFAST_COMPILE_TIME_ONLY gate, so cross-backend-check.sh still
# compiles it.
#
# devtools/test.py (push and pr) names both binaries, as
# test/core/free_breakfast_core_tests_{host,gpu}. Keep the target names in step
# if you rename them.
include(GoogleTest)

add_executable(
  free_breakfast_core_tests_host
  version_host.cpp
  core_vocabulary_host.cpp
  diagnostics_host.cpp
  gpu_policy_host.cpp
  node_resources_host.cpp
  footprint_host.cpp
)

add_executable(
  free_breakfast_core_tests_gpu
  device_gpu.cpp
  device_pool_gpu.cpp
  gpu_policy_gpu.cpp
  node_resources_gpu.cpp
  primitives_gpu.cpp
  reduce_gpu.cpp
  device_resources_gpu.cpp
  fork_stream_gpu.cpp
  device_stream_gpu.cpp
  scoped_device_index_gpu.cpp
)

foreach(_fb_target free_breakfast_core_tests_host free_breakfast_core_tests_gpu)
  set_target_properties(
    ${_fb_target} PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
  )
endforeach()

target_link_libraries(
  free_breakfast_core_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::core::diagnostics_macros
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

# free_breakfast::test::gpu_kernels is the device library (gpu_kernels/)
# primitives_gpu.cpp and reduce_gpu.cpp launch through: its .cu files stay out
# of this executable, and only their plain-C++ launcher headers are #included
# here.
target_link_libraries(
  free_breakfast_core_tests_gpu
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::test::gpu_kernels
          free_breakfast::test::support
          GTest::gtest
          GTest::gtest_main
)

# The properties are set per call and are NOT inherited from test/gtest's: the
# GPU call's `gpu` label is what the ci-cuda and ci-hip test presets exclude
# (`-LE gpu`) and what devtools/test.py pr batches on.
gtest_discover_tests(
  free_breakfast_core_tests_host
  DISCOVERY_MODE PRE_TEST
  XML_OUTPUT_DIR ${_fb_gtest_xml_dir}
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)
gtest_discover_tests(
  free_breakfast_core_tests_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS gpu
)

# The bound memory_budget() leaves for kernel stacks, against the binary's
# kernels (kernel_stack_bound.cmake). CUDA only: cuobjdump reads CUDA
# binaries, and ROCm holds no device-wide stack (Device::stack_growth).
if(FREE_BREAKFAST_GPU_BACKEND STREQUAL "CUDA")
  find_program(
    FREE_BREAKFAST_CUOBJDUMP
    NAMES cuobjdump
    HINTS "${CUDAToolkit_BIN_DIR}" /usr/local/cuda/bin
    REQUIRED
  )
  add_test(
    NAME core_kernel_stack_bound
    COMMAND
      ${CMAKE_COMMAND} -DCUOBJDUMP=${FREE_BREAKFAST_CUOBJDUMP}
      -DBINARY=$<TARGET_FILE:free_breakfast_main>
      -DSOURCE=${PROJECT_SOURCE_DIR}/src/core/resources/device_resources.cppm
      -P ${CMAKE_CURRENT_SOURCE_DIR}/kernel_stack_bound.cmake
  )
  set_tests_properties(
    core_kernel_stack_bound PROPERTIES TIMEOUT ${_fb_test_timeout}
  )
endif()
