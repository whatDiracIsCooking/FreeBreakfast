/// @file
/// @brief Skip-in-fast-mode helper for the heavy end-to-end validation tests.
///
/// The C++ analogue of the pytest `slow` marker: a full run executes these
/// cases (ctest, CI, `devtools/cpp-slow-tier.sh`), while `devtools/cpp-tier.sh`
/// and the pre-push smoke gate (`devtools/cpp-smoke.sh`) opt out by exporting
/// FREE_BREAKFAST_FAST_TESTS. Default -- the variable unset -- runs everything,
/// so nothing changes for a bare ctest run.
///
/// Put the macro at the TOP of a test body, after FBF_REQUIRE_GPU():
/// a slow test is also a GPU test, so it should still skip cleanly on a
/// device-less host.
///
/// std::getenv comes from the including TU's `import std;` -- the macro is
/// expanded at the point of use, like gpu_requirement.h, not resolved here.

#pragma once

#include <gtest/gtest.h>

#define FBF_SKIP_IF_FAST_TESTS()                                                                   \
  do {                                                                                             \
    if (const char *fb_fast = std::getenv("FREE_BREAKFAST_FAST_TESTS");                            \
        fb_fast != nullptr && fb_fast[0] != '\0' && fb_fast[0] != '0') {                           \
      GTEST_SKIP() << "slow validation test skipped in fast mode "                                 \
                      "(FREE_BREAKFAST_FAST_TESTS is set)";                                        \
    }                                                                                              \
  } while (0)
