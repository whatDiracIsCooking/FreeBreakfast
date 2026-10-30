// Host-only cases for basis's :shell_stamp: the centerless shell, its
// per-primitive normalization, and placing it at a center.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

#include "gaussian_moments.h"

namespace {

namespace fb = free_breakfast;
using fb::test::double_factorial;

TEST(BasisShellStamp, ZipsParallelArrays) {
  const fb::ShellStamp stamp(1, std::vector<double>{3.0, 0.5}, std::vector<double>{0.25, 0.75});
  EXPECT_EQ(stamp.angular_momentum(), 1);
  ASSERT_EQ(stamp.nprimitives(), 2U);
  EXPECT_EQ(stamp.primitives()[0].exponent, 3.0);
  EXPECT_EQ(stamp.primitives()[1].coefficient, 0.75);
  EXPECT_EQ(stamp.function_count_cart(), 3U);
  EXPECT_EQ(stamp.function_count_sphe(), 3U);
}

TEST(BasisShellStampDeathTest, AbortsOnBadShape) {
  EXPECT_DEATH(fb::ShellStamp(-1, {{1.0, 1.0}}), "");
  EXPECT_DEATH(fb::ShellStamp(0, std::vector<fb::Primitive>{}), "");
  EXPECT_DEATH(fb::ShellStamp(0, std::vector<double>{1.0, 2.0}, std::vector<double>{1.0}), "");
}

TEST(BasisShellStamp, NormalizesEachPrimitive) {
  const double pi = std::numbers::pi;
  for (std::int32_t l = 0; l <= 3; ++l) {
    const fb::ShellStamp stamp =
        fb::ShellStamp(l, std::vector<double>{2.0, 0.3}, std::vector<double>{0.4, 0.6})
            .primitive_normalized();
    for (const auto &[i, alpha, c] : {std::tuple{0, 2.0, 0.4}, std::tuple{1, 0.3, 0.6}}) {
      const double norm = std::pow(2.0 * alpha / pi, 0.75) * std::pow(4.0 * alpha, l / 2.0) /
                          std::sqrt(double_factorial((2 * l) - 1));
      EXPECT_NEAR(stamp.primitives()[i].coefficient, c * norm, 1e-12) << "l=" << l;
      EXPECT_EQ(stamp.primitives()[i].exponent, alpha);
    }
  }
}

// place() supplies the center and copies the rest unchanged.
TEST(BasisShellStamp, PlaceCopiesMomentumAndPrimitivesAtTheCenter) {
  const fb::ShellStamp stamp(2, std::vector<double>{1.5, 0.2}, std::vector<double>{0.3, 0.7});
  const fb::Shell shell = fb::place(stamp, fb::Vec<double, 3>{0.5, -1.0, 2.5});
  EXPECT_EQ(shell.angular_momentum(), 2);
  ASSERT_EQ(shell.nprimitives(), 2U);
  for (std::uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(shell.primitives()[i].exponent, stamp.primitives()[i].exponent);
    EXPECT_EQ(shell.primitives()[i].coefficient, stamp.primitives()[i].coefficient);
  }
  EXPECT_EQ(shell.center().x, 0.5);
  EXPECT_EQ(shell.center().y, -1.0);
  EXPECT_EQ(shell.center().z, 2.5);
}

} // namespace
