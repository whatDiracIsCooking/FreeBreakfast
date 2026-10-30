// Host-only cases for basis's :ao and the counts it re-exports from basis.h:
// the per-shell function counts and the canonical Cartesian ordering.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

namespace {

namespace fb = free_breakfast;
using Powers = std::array<std::int32_t, 3>;

TEST(BasisAo, FunctionCounts) {
  EXPECT_EQ(fb::kMaxAngularMomentum, 3);
  const std::array<std::uint32_t, 5> cart{1, 3, 6, 10, 15};
  for (std::int32_t l = 0; l < 5; ++l) {
    EXPECT_EQ(fb::function_count_cart(l), cart[l]) << "l=" << l;
    EXPECT_EQ(fb::function_count_sphe(l), static_cast<std::uint32_t>((2 * l) + 1));
  }
}

TEST(BasisAo, CartesianPowersCanonicalOrder) {
  EXPECT_EQ(fb::cartesian_powers(0), (std::vector<Powers>{{0, 0, 0}}));
  EXPECT_EQ(fb::cartesian_powers(1), (std::vector<Powers>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
  // xx, xy, xz, yy, yz, zz
  EXPECT_EQ(
      fb::cartesian_powers(2),
      (std::vector<Powers>{{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}}));
  // Defined past d, with every entry of degree l.
  for (std::int32_t l = 3; l <= 5; ++l) {
    const std::vector<Powers> powers = fb::cartesian_powers(l);
    ASSERT_EQ(powers.size(), fb::function_count_cart(l));
    EXPECT_EQ(powers.front(), (Powers{l, 0, 0}));
    EXPECT_EQ(powers.back(), (Powers{0, 0, l}));
    for (const Powers &p : powers) {
      EXPECT_EQ(p[0] + p[1] + p[2], l);
    }
  }
}

} // namespace
