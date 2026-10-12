# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The C++ suite. gtest/ builds into two binaries split by device requirement --
# `free_breakfast_tests_host` (runs anywhere) and `free_breakfast_tests_gpu`
# (skips its device cases without a GPU) -- both registered with ctest and run
# by `ctest --preset <name>`, or by devtools/cpp-tier.sh, which is configure +
# build + ctest in one command.
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
# NOTE: `ctest` here also runs the WarpWraps submodule's own suites -- it adds
# them itself from deps/ (see deps/CMakeLists.txt). Scope a run to this
# project's cases by name, or use the labels.
add_subdirectory(gtest)
