// GPU cases for recipes.cholesky: A = L L^T against a host Cholesky at orders
// around the 64-column block's edges, bitwise repeatable, the strict upper
// triangle untouched, and the flag naming the first leading minor that is not
// positive, as potrf's does; and cholesky_solve against a host solve over the
// host factor, at the same edges and several right-hand-side counts.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

#include "test_workspaces.h"

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

  fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  for (const std::size_t broken : {std::size_t{10}, std::size_t{64}, std::size_t{100}}) {
    std::vector<double> a = random_spd(kN);
    a[broken + (m * broken)] = -a[broken + (m * broken)];
    const auto [info, got] = factor(res, kN, a);
    EXPECT_EQ(info, static_cast<std::int32_t>(broken) + 1) << "broken " << broken;
  }
}

TEST(CholeskyFlag, EmptyIsZeroAndNegativeIsInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::Result<std::int32_t> empty = fb::cholesky(res, 0, nullptr);
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(*empty, 0);
  const fb::Result<std::int32_t> negative = fb::cholesky(res, -1, nullptr);
  ASSERT_FALSE(negative.has_value());
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

// Solve A X = B on the host over host_cholesky's factor @p l: L Y = B, then
// L^T X = Y, column by column. @p b is n x nrhs, packed.
std::vector<double> host_cholesky_solve(const std::vector<double> &l, std::vector<double> b,
                                        const std::size_t n, const std::size_t nrhs) {
  for (std::size_t c = 0; c < nrhs; ++c) {
    double *const x = b.data() + (n * c);
    for (std::size_t i = 0; i < n; ++i) {
      double v = x[i];
      for (std::size_t k = 0; k < i; ++k) {
        v -= l[i + (n * k)] * x[k];
      }
      x[i] = v / l[i + (n * i)];
    }
    for (std::size_t i = n; i-- > 0;) {
      double v = x[i];
      for (std::size_t k = i + 1; k < n; ++k) {
        v -= l[k + (n * i)] * x[k];
      }
      x[i] = v / l[i + (n * i)];
    }
  }
  return b;
}

class CholeskySolve : public ::testing::TestWithParam<std::tuple<std::int32_t, std::int32_t>> {};

TEST_P(CholeskySolve, MatchesTheHostSolve) {
  FBF_REQUIRE_GPU();
  const auto [n, nrhs] = GetParam();
  const auto m = static_cast<std::size_t>(n);
  const auto r = static_cast<std::size_t>(nrhs);
  const std::vector<double> a = random_spd(n);
  std::mt19937 gen{2468};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> b(m * r);
  for (double &x : b) {
    x = dist(gen);
  }
  // The device factor, so both solves see the same L up to cholesky's own
  // rounding, and the host solve over the host factor as the oracle.
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const auto [info, l] = factor(res, n, a);
  ASSERT_EQ(info, 0);
  const std::vector<double> want = host_cholesky_solve(host_cholesky(a, m), b, m, r);

  fb::Buffers::device<double> l_dev(l.size(), res.device());
  fb::Buffers::device<double> b_dev(b.size(), res.device());
  ASSERT_EQ(wwr::wwrMemcpyAsync(l_dev.data(), l.data(), l.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrMemcpyAsync(b_dev.data(), b.data(), b.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  const fb::Status solved = fb::cholesky_solve(res, n, nrhs, l_dev.data(), b_dev.data());
  ASSERT_TRUE(solved.has_value()) << solved.error().message;
  std::vector<double> got(b.size());
  ASSERT_EQ(wwr::wwrMemcpyAsync(got.data(), b_dev.data(), got.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  // Each is a Cholesky solve, so each x^ solves (A + dA) x^ = b with
  // |dA| <= g |L| |L^T|, g = (3n + 1) u / (1 - (3n + 1) u) (Higham, ASNA 2nd ed.,
  // Thm. 10.4), and |||L| |L^T|||_2 <= n ||L||_2^2 = n ||A||_2. To first order
  // that is a forward error |x^ - x|_2 <= g n kappa(A) |x|_2, and A's spectrum
  // is in [1, n + 1], so kappa(A) <= n + 1. Two such errors bound the
  // difference, per column.
  const double u = std::numeric_limits<double>::epsilon() / 2;
  const double three_n = (3.0 * static_cast<double>(n)) + 1.0;
  const double g = three_n * u / (1.0 - (three_n * u));
  const double kappa = static_cast<double>(n) + 1.0;
  for (std::size_t c = 0; c < r; ++c) {
    double x_norm = 0.0;
    double e_norm = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      const double x = want[i + (m * c)];
      const double e = got[i + (m * c)] - x;
      x_norm += x * x;
      e_norm += e * e;
    }
    const double bound = 2.0 * g * static_cast<double>(n) * kappa * std::sqrt(x_norm);
    EXPECT_LE(std::sqrt(e_norm), bound) << "n = " << n << ", nrhs = " << nrhs << ", column " << c;
  }
}

// The block's edges in A's order, and one, few and more than a 16-column tile
// of right-hand sides.
INSTANTIATE_TEST_SUITE_P(OrdersAndColumns, CholeskySolve,
                         ::testing::Combine(::testing::Values(1, 63, 64, 65, 127, 128, 130),
                                            ::testing::Values(1, 3, 17)));

TEST(CholeskySolveSizes, EmptyIsANoOpAndNegativeIsInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  EXPECT_TRUE(fb::cholesky_solve(res, 0, 3, nullptr, nullptr).has_value());
  EXPECT_TRUE(fb::cholesky_solve(res, 3, 0, nullptr, nullptr).has_value());
  for (const auto [n, nrhs] : {std::pair{-1, 1}, std::pair{1, -1}}) {
    const fb::Status refused = fb::cholesky_solve(res, n, nrhs, nullptr, nullptr);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
