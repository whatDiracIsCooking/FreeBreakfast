# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The C++ suite: three trees. core/ and gtest/ are each built into a pair of
# binaries split by device requirement -- a host one (runs anywhere) and a GPU
# one (skips its device cases without a GPU); io/ is a host binary alone:
#
#   * core/  -- the src/core tests: free_breakfast_core_tests_{host,gpu}.
#   * io/    -- the src/io tests that need no device:
#               free_breakfast_io_tests_host.
#   * gtest/ -- everything else: free_breakfast_tests_{host,gpu}.
#
# All five are registered with ctest and run by `ctest --preset <name>`, or by
# devtools/cpp-tier.sh, which is configure + build + ctest in one command.
#
# The PYTHON suite is not built by CMake: it lives in test/pytest/ -- a sibling
# of gtest/ under this directory -- is collected by pytest via `testpaths` in
# pyproject.toml, and exercises the binaries this tree produces. The two suites
# share test/ but not their config: keep pytest's config in pyproject.toml and
# add no pytest.ini or conftest under test/, because a second config there
# silently overrides the root one, and the marker set (slow / gpu /
# no_sanitizer) that the push gate and the compose services both filter on
# would then depend on which directory pytest was started from.
#
# core/gpu_kernels/ -- the .cu kernels the core GPU binary links -- is NOT
# added here but from the root CMakeLists, outside the
# FREE_BREAKFAST_COMPILE_TIME_ONLY gate that skips this whole directory, so
# cross-backend-check.sh still compiles it.
#
# support/ -- gpu_requirement.h, slow_test.h and scratch_dir.h, the helpers
# every gtest binary #includes -- is a header-only INTERFACE target,
# free_breakfast::test::support, added first so each tree links the one copy.
#
# ci_oracles/ -- the host references the CI tests check the device against --
# is a module library gtest/'s two binaries link, kept out of src/ so the ci
# modules export only what a calculation runs.
#
# NOTE: `ctest` here runs THIS project's suites and nothing else. The WarpWraps
# submodule is added with WWR_BUILD_TESTS and WWR_BUILD_EXAMPLES forced OFF
# (deps/CMakeLists.txt), so it calls no enable_testing() and registers no
# entries of its own -- its suites are wwr's CI's job, not this tree's. What a
# run does get scoped by is the device: the GPU binaries' entries carry the
# `gpu` label, which the ci-cuda and ci-hip test presets exclude (gtest/
# CMakeLists.txt says why).
#
# The backstop timeout of every entry in both trees, scaled by
# FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER (root CMakeLists). A case that hangs
# should fail its own entry rather than the whole run; the suite is
# milliseconds today, so 120s is purely a backstop -- until compute-sanitizer
# slows every GPU test by an order of magnitude, which is what the multiplier
# is for. Set here, before the trees are added, so they share one value.
math(EXPR _fb_test_timeout "120 * ${FREE_BREAKFAST_TEST_TIMEOUT_MULTIPLIER}")

add_subdirectory(support)
add_subdirectory(ci_oracles)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(gtest)
