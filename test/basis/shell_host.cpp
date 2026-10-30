// Host-only cases for basis's :shell: a Shell's accessors and its shape
// contract.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

TEST(BasisShell, ShellZipsParallelArrays) {
  const fb::Shell shell(2, {3.0, 0.5}, {0.25, 0.75}, fb::Vec<double, 3>{1.0, 2.0, 3.0});
  EXPECT_EQ(shell.angular_momentum(), 2);
  ASSERT_EQ(shell.nprimitives(), 2U);
  EXPECT_EQ(shell.primitives()[1].exponent, 0.5);
  EXPECT_EQ(shell.primitives()[1].coefficient, 0.75);
  EXPECT_EQ(shell.center().z, 3.0);
  EXPECT_EQ(shell.function_count_cart(), 6U); // Cartesian, what the kernels evaluate
  EXPECT_EQ(shell.function_count_sphe(), 5U); // 2l+1, what the shell means
}

TEST(BasisShellDeathTest, ShellAbortsOnBadShape) {
  EXPECT_DEATH(fb::Shell(-1, {{1.0, 1.0}}, kOrigin), "");
  EXPECT_DEATH(fb::Shell(0, std::vector<fb::Primitive>{}, kOrigin), "");
  EXPECT_DEATH(fb::Shell(0, std::vector<double>{1.0, 2.0}, std::vector<double>{1.0}, kOrigin), "");
}

} // namespace
