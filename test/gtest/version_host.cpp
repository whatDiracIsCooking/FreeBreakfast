// Host-only cases: no GPU device needed at run time, so every one of these runs
// on any box. Add new host-only TUs as `*_host.cpp` and list them in
// CMakeLists.txt's host executable.

// gtest's headers come FIRST, before any import -- the ordering every test TU
// in deps/WarpWraps uses. A header pulled in AFTER `import std;` re-declares
// the same std names through the textual path, and clang is entitled to treat
// the two as distinct; includes first keeps the header build and the module
// build from overlapping at all.
#include <gtest/gtest.h>

import std;
import free_breakfast.core;

namespace {

TEST(Version, ComponentsAreNonNegative) {
  EXPECT_GE(free_breakfast::core::kVersionMajor, 0);
  EXPECT_GE(free_breakfast::core::kVersionMinor, 0);
  EXPECT_GE(free_breakfast::core::kVersionPatch, 0);
}

TEST(Version, StringMatchesComponents) {
  EXPECT_EQ(free_breakfast::core::version_string(),
            std::format("{}.{}.{}", free_breakfast::core::kVersionMajor,
                        free_breakfast::core::kVersionMinor, free_breakfast::core::kVersionPatch));
}

} // namespace
