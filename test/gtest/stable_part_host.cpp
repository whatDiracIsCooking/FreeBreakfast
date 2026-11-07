// Host-only cases for stable_part.h, the mask the NEVPT2 GPU cases compare
// two refusals through: a string rewrite, so nothing touches a device.

#include <gtest/gtest.h>

import std;

// After the imports: it names std::regex from this unit's `import std;`.
#include "stable_part.h"

namespace {

using free_breakfast::test::stable_part;

// The mask covers a non-finite number as it does a finite one: the refusal
// #1010 captured, its amplitude -inf on one side and finite on the other.
TEST(Nevpt2StablePart, MasksNonFinite) {
  const std::string head = "Sr has a denominator λ + Δ = 0.000e+00 at round-off (at most "
                           "3.056e-12, 100 ε √d of its largest), amplitude ";
  const std::string want = "Sr has a denominator λ + Δ = # at round-off (at most #, 100 ε "
                           "√d of its largest), amplitude #";
  for (const char *number : {"-inf", "inf", "nan", "-nan", "1.234e+05", "-1.234e+05"}) {
    EXPECT_EQ(stable_part(head + number), want) << number;
  }
  EXPECT_EQ(stable_part("an infinite information nancy"), "an infinite information nancy");
}

} // namespace
