// GPU cases for recipes.cholesky: A = L L^T against a host Cholesky at orders
// around the 64-column block's edges, bitwise repeatable, the strict upper
// triangle untouched, and the flag naming the first leading minor that is not
// positive, as potrf's does.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

// A symmetric positive definite n x n, column-major: R R^T / n + I for R with
// entries in [-1, 1], so its spectrum is in [1, n + 1].
std::vector<double> random_spd(const std::int32_t n) {
  const auto m = static_cast<std::size_t>(n);
  std::mt19937 gen{1357};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> r(m * m);
  for (double &x : r) {
    x = dist(gen);
  }
  std::vector<double> a(m * m, 0.0);
  for (std::size_t j = 0; j < m; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      double sum = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        sum += r[i + (m * k)] * r[j + (m * k)];
      }
      a[i + (m * j)] = (sum / static_cast<double>(m)) + (i == j ? 1.0 : 0.0);
    }
  }
  return a;
}

// The lower factor by the textbook column algorithm, the oracle's own.
std::vector<double> host_cholesky(std::vector<double> a, const std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) {
    double d = a[j + (n * j)];
    for (std::size_t k = 0; k < j; ++k) {
      d -= a[j + (n * k)] * a[j + (n * k)];
    }
    d = std::sqrt(d);
    a[j + (n * j)] = d;
    for (std::size_t i = j + 1; i < n; ++i) {
      double v = a[i + (n * j)];
      for (std::size_t k = 0; k < j; ++k) {
        v -= a[i + (n * k)] * a[j + (n * k)];
      }
      a[i + (n * j)] = v / d;
    }
  }
  return a;
}

// Upload @p host, factor it, and return the flag and the matrix as it came back.
std::pair<std::int32_t, std::vector<double>> factor(fb::DeviceResources &res, std::int32_t n,
                                                    const std::vector<double> &host) {
  fb::Buffers::device<double> a(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(a.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  const fb::Result<std::int32_t> info = fb::cholesky(res, n, a.data());
  EXPECT_TRUE(info.has_value()) << info.error().message;
  std::vector<double> out(host.size());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), a.data(), out.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return {info.value_or(-1), std::move(out)};
}

class Cholesky : public ::testing::TestWithParam<std::int32_t> {};

TEST_P(Cholesky, MatchesTheHostFactorAndLeavesTheUpperTriangle) {
  FBF_REQUIRE_GPU();
  const std::int32_t n = GetParam();
  const auto m = static_cast<std::size_t>(n);
  const std::vector<double> a = random_spd(n);
  const std::vector<double> want = host_cholesky(a, m);

  fb::DeviceResources res{0};
  const auto [info, got] = factor(res, n, a);
  ASSERT_EQ(info, 0);
  double worst = 0.0;
  for (std::size_t j = 0; j < m; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      if (i >= j) {
        worst = std::max(worst, std::abs(got[i + (m * j)] - want[i + (m * j)]));
      } else {
        // Bitwise: the routine never writes the strict upper triangle.
        ASSERT_EQ(std::bit_cast<std::uint64_t>(got[i + (m * j)]),
                  std::bit_cast<std::uint64_t>(a[i + (m * j)]))
            << "upper (" << i << ", " << j << ")";
      }
    }
  }
  // L's entries are O(1) here (A's spectrum is in [1, n + 1]).
  EXPECT_LT(worst, 1e-12) << "n = " << n;
}

// One block, its edges, the DF metric's order for water in cc-pVDZ-JKFIT
// (116, the order #404 failed at), rocSOLVER's small-kernel limit (127), and
// several blocks.
INSTANTIATE_TEST_SUITE_P(Orders, Cholesky,
                         ::testing::Values(1, 2, 63, 64, 65, 116, 127, 128, 129, 300));

// The same input factored repeatedly is the same factor to the bit: the
// kernel's result must not depend on how its waves are scheduled. rocSOLVER's
// potrf failed exactly this under GPU contention (#404).
TEST(CholeskyRepeat, IsBitwiseRepeatable) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 116;
  const std::vector<double> a = random_spd(kN);
  fb::DeviceResources res{0};
  const auto [info0, first] = factor(res, kN, a);
  ASSERT_EQ(info0, 0);
  for (std::int32_t rep = 0; rep < 50; ++rep) {
    const auto [info, again] = factor(res, kN, a);
    ASSERT_EQ(info, 0);
    ASSERT_TRUE(std::ranges::equal(again, first,
                                   [](const double x, const double y) {
                                     return std::bit_cast<std::uint64_t>(x) ==
                                            std::bit_cast<std::uint64_t>(y);
                                   }))
        << "repeat " << rep;
  }
}

// One diagonal element made negative: the leading minors before it are
// untouched and positive, and the minor through it is not, so the flag is its
// 1-based order -- in the first block, at a block's first column, and in a
// later block.
TEST(CholeskyFlag, NamesTheFirstMinorThatIsNotPositive) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kN = 150;
  const auto m = static_cast<std::size_t>(kN);
  fb::DeviceResources res{0};
  for (const std::size_t broken : {std::size_t{10}, std::size_t{64}, std::size_t{100}}) {
    std::vector<double> a = random_spd(kN);
    a[broken + (m * broken)] = -a[broken + (m * broken)];
    const auto [info, got] = factor(res, kN, a);
    EXPECT_EQ(info, static_cast<std::int32_t>(broken) + 1) << "broken " << broken;
  }
}

TEST(CholeskyFlag, EmptyIsZeroAndNegativeIsInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Result<std::int32_t> empty = fb::cholesky(res, 0, nullptr);
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(*empty, 0);
  const fb::Result<std::int32_t> negative = fb::cholesky(res, -1, nullptr);
  ASSERT_FALSE(negative.has_value());
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
