# ---------------------------------------------------------------------------
# C++ unit tests, split into two executables by device requirement
# ---------------------------------------------------------------------------
# The suite is aggregated into TWO binaries rather than one:
#
#   * free_breakfast_tests_host -- the *_host.cpp TUs, which need no GPU device
#     at run time. Every case runs (none skip) on a device-less box, so a green
#     run here is meaningful coverage even off a GPU host.
#   * free_breakfast_tests_gpu  -- the *_gpu.cpp TUs, which drive the device.
#     Each guards itself with FBF_REQUIRE_GPU() and SKIPS rather
#     than fails when no device is visible.
#
# The split is by *what the test needs at run time*, not by file extension: the
# _gpu TUs are still ordinary .cpp compiled by the C++ front end -- they reach
# the device through WarpWraps' module surface, not by being .cu. A `.cu` of
# real kernel code belongs in its own library, declared with
# free_breakfast_add_gpu_device_library, and linked in here; keeping it out of
# these executables is deliberate, because they are built with CXX_MODULE_STD
# and CXX_SCAN_FOR_MODULES, and nvcc neither understands `import std` nor
# participates in module dependency scanning.
#
# docker/compose.yaml invokes both binaries by path as
# build*/test/gtest/free_breakfast_tests_{host,gpu}. Keep the target names in
# step if you rename them.
include(GoogleTest)

# ---------------------------------------------------------------------------
# Host-only executable
# ---------------------------------------------------------------------------
# Add new host-only (no device needed at run time) TUs to this list.
add_executable(
  free_breakfast_tests_host
  version_host.cpp
  core_vocabulary_host.cpp
  diagnostics_host.cpp
  atom_host.cpp
  molecule_host.cpp
  basis_host.cpp
  sad_host.cpp
  io_base_host.cpp
  io_basis_host.cpp
  io_json_host.cpp
  io_npy_host.cpp
  io_sad_host.cpp
  io_toml_host.cpp
  gpu_policy_host.cpp
  node_resources_host.cpp
  pipeline_step_host.cpp
  pipeline_pipeline_host.cpp
  pipeline_config_host.cpp
)

# Both executables consume `import std` and this project's C++23 modules, so
# they need the same module-scanning setup as the libraries they link. The
# GoogleTest that wwr's deps/ fetches is built as plain C++ with
# CXX_SCAN_FOR_MODULES OFF, which is why gtest's own headers are safe to
# #include beside an `import std;` here.
set_target_properties(
  free_breakfast_tests_host PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
)

target_link_libraries(
  free_breakfast_tests_host
  PRIVATE free_breakfast::core
          free_breakfast::core::diagnostics_macros
          free_breakfast::atom
          free_breakfast::molecule
          free_breakfast::basis
          free_breakfast::sad
          free_breakfast::io
          free_breakfast::core::resources
          free_breakfast::pipeline::step
          free_breakfast::pipeline::pipeline
          free_breakfast::pipeline::config
          GTest::gtest
          GTest::gtest_main
)

# ---------------------------------------------------------------------------
# GPU executable
# ---------------------------------------------------------------------------
# Add new device-dependent TUs to this list.
add_executable(
  free_breakfast_tests_gpu
  device_gpu.cpp
  gpu_policy_gpu.cpp
  node_resources_gpu.cpp
  primitives_gpu.cpp
  reduce_gpu.cpp
  device_resources_gpu.cpp
  fork_stream_gpu.cpp
  device_stream_gpu.cpp
  mirrored_buffer_gpu.cpp
  scoped_device_index_gpu.cpp
  diagonalize_gpu.cpp
  orthogonalize_gpu.cpp
  diis_gpu.cpp
  expm_gpu.cpp
  davidson_gpu.cpp
  matrix_ao_gpu.cpp
  gemm_gpu.cpp
  gemmwich_gpu.cpp
  gmmwich_gpu.cpp
  complex_cast_gpu.cpp
  polynomial_roots_gpu.cpp
  mmd_gpu.cpp
  molecule_gpu.cpp
  basis_gpu.cpp
  sad_gpu.cpp
  io_basis_gpu.cpp
  io_xyz_gpu.cpp
  pipeline_molecule_gpu.cpp
  pipeline_basis_set_gpu.cpp
  pipeline_spherical_transform_gpu.cpp
  pipeline_primitive_pair_classes_gpu.cpp
  pipeline_overlap_gpu.cpp
  pipeline_orthogonalizer_gpu.cpp
  pipeline_core_hamiltonian_gpu.cpp
  pipeline_guess_gpu.cpp
  pipeline_scf_gpu.cpp
  pipeline_calculation_gpu.cpp
  launch_gpu.cpp
  overlap_gpu.cpp
  core_hamiltonian_gpu.cpp
  coulomb_gpu.cpp
  sort_bounds_gpu.cpp
  exchange_helpers_gpu.cpp
  exchange_pairs_gpu.cpp
  exchange_gpu.cpp
)

set_target_properties(
  free_breakfast_tests_gpu PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
)

# free_breakfast::test::gpu_kernels is the device library (test/gpu_kernels)
# primitives_gpu.cpp and reduce_gpu.cpp launch through: its .cu files stay out
# of this executable, and only their plain-C++ launcher headers are #included
# here. free_breakfast::integrals::eri::integral_direct::exchange::helpers and
# free_breakfast::integrals::eri::integral_direct::coulomb::sort_bounds::device
# are the same arrangement for exchange_helpers_gpu.cpp and sort_bounds_gpu.cpp:
# device libraries with no module face, driven through their bridge headers.
target_link_libraries(
  free_breakfast_tests_gpu
  PRIVATE
  free_breakfast::core
  free_breakfast::core::resources
  free_breakfast::molecule
  free_breakfast::basis
  free_breakfast::sad
  free_breakfast::recipes
  free_breakfast::integrals
  free_breakfast::integrals::eri::integral_direct::coulomb::sort_bounds::device
  free_breakfast::integrals::eri::integral_direct::exchange::helpers
  free_breakfast::integrals::eri::integral_direct::exchange::pairs
  free_breakfast::io
  free_breakfast::pipeline::molecule
  free_breakfast::pipeline::basis_set
  free_breakfast::pipeline::spherical_transform
  free_breakfast::pipeline::primitive_pair_classes
  free_breakfast::pipeline::overlap
  free_breakfast::pipeline::orthogonalizer
  free_breakfast::pipeline::core_hamiltonian
  free_breakfast::pipeline::guess
  free_breakfast::pipeline::scf
  free_breakfast::pipeline::config
  free_breakfast::test::gpu_kernels
  GTest::gtest
  GTest::gtest_main
)

# gpu_requirement.h and slow_test.h are #included by the TUs in this directory.
target_include_directories(
  free_breakfast_tests_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(
  free_breakfast_tests_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Golden reference data (data/README.md). An absolute path, so a test finds it
# whatever directory ctest runs it from.
foreach(_fb_target free_breakfast_tests_gpu free_breakfast_tests_host)
  target_compile_definitions(
    ${_fb_target} PRIVATE FBF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  )
endforeach()

# ---------------------------------------------------------------------------
# ctest registration
# ---------------------------------------------------------------------------
# One ctest entry per gtest case, per binary, discovered at TEST time rather
# than build time.
#
# The obvious alternative is a single `add_test(NAME free_breakfast_tests ...)`,
# on the reasoning that gtest_discover_tests() executes the binary during the
# build and a build host may well have no GPU. The reasoning is right; the
# conclusion is not. DISCOVERY_MODE PRE_TEST defers the enumeration to the start
# of the ctest run, on the machine that is actually going to run the tests --
# and `--gtest_list_tests` only walks the registry, so it never initialises the
# GPU runtime and works fine with no device present.
#
# What the single entry costs, once a suite outgrows one file:
#
#   * `ctest -R <name>` selects nothing. cpp-tier.sh advertises `-- -R Version`
#     as the way to scope a run; against one entry that matches zero tests and
#     still reports success.
#   * `100% tests passed, 1 of 1` reads identically whether every case ran or
#     most of them skipped -- the skip/pass distinction is invisible unless you
#     run the binary by hand.
#
# The trade is that ctest shells out once per binary before the run to
# enumerate. That is milliseconds, and it buys per-case reporting and -R.
# The backstop timeout, scaled by FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER (root
# CMakeLists). A case that hangs should fail its own entry rather than the whole
# run; the suite is milliseconds today, so 120s is purely a backstop -- until
# compute-sanitizer slows every GPU test by an order of magnitude, which is what
# the multiplier is for.
math(EXPR _fb_test_timeout "120 * ${FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER}")

gtest_discover_tests(
  free_breakfast_tests_host
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout}
)

# The `gpu` label is what the ci-cuda and ci-hip test presets exclude (`-LE
# gpu`), so a runner with no device reports the same count as a developer box
# instead of a wall of skips. Label the ENTRIES, not the binary: a GPU case that
# also needs a lot of memory or wall clock gets `no_sanitizer` here too, which
# the asan preset excludes.
gtest_discover_tests(
  free_breakfast_tests_gpu
  DISCOVERY_MODE PRE_TEST
  PROPERTIES TIMEOUT ${_fb_test_timeout} LABELS gpu
)
