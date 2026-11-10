// GPU cases for recipes.lu: lu_solve against a host pivoted elimination on
// random symmetric indefinite and general matrices, at orders from one to
// past a hundred, with several right-hand sides; bitwise repeatable over
// repeated runs (what a race in the factorization, as rocSOLVER's potf2 had,
// would break); an exactly singular U flagged at its first zero pivot with B
// left as it was; the empty and negative shapes; and lu_solve_footprint,
// held to the workspace a solve grows and to its pool peak (#1159).

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

// Entries in [-1, 1], column-major n x m.
std::vector<double> random_matrix(const std::int32_t n, const std::int32_t m,
                                  const std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
  for (double &x : a) {
    x = dist(gen);
  }
  return a;
}

// A symmetric indefinite n x n: R + Rᵀ, whose spectrum straddles zero, plus
// n on the diagonal alternating in sign, so it is well away from singular and
// still indefinite — the capacitance matrix's shape.
std::vector<double> random_indefinite(const std::int32_t n, const std::uint32_t seed) {
  const auto m = static_cast<std::size_t>(n);
  const auto r = random_matrix(n, n, seed);
  std::vector<double> a(m * m);
  for (std::size_t j = 0; j < m; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      a[i + (m * j)] = r[i + (m * j)] + r[j + (m * i)];
    }
    a[j + (m * j)] += (j % 2 == 0 ? 1.0 : -1.0) * static_cast<double>(n);
  }
  return a;
}

// X solving A X = B by Gaussian elimination with partial pivoting: the
// oracle's own.
std::vector<double> host_solve(std::vector<double> a, std::vector<double> b, const std::size_t n,
                               const std::size_t nrhs) {
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(a[i + (n * k)]) > std::abs(a[p + (n * k)])) {
        p = i;
      }
    }
    for (std::size_t j = 0; j < n; ++j) {
      std::swap(a[k + (n * j)], a[p + (n * j)]);
    }
    for (std::size_t c = 0; c < nrhs; ++c) {
      std::swap(b[k + (n * c)], b[p + (n * c)]);
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i + (n * k)] / a[k + (n * k)];
      for (std::size_t j = k; j < n; ++j) {
        a[i + (n * j)] -= f * a[k + (n * j)];
      }
      for (std::size_t c = 0; c < nrhs; ++c) {
        b[i + (n * c)] -= f * b[k + (n * c)];
      }
    }
  }
  for (std::size_t c = 0; c < nrhs; ++c) {
    for (std::size_t k = n; k-- > 0;) {
      double v = b[k + (n * c)];
      for (std::size_t j = k + 1; j < n; ++j) {
        v -= a[k + (n * j)] * b[j + (n * c)];
      }
      b[k + (n * c)] = v / a[k + (n * k)];
    }
  }
  return b;
}

fb::Buffers::device<double> to_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> to_host(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// X from lu_solve on the device, with the flag checked zero.
std::vector<double> device_solve(fb::DeviceResources &res, const std::vector<double> &a,
                                 const std::vector<double> &b, const std::int32_t n,
                                 const std::int32_t nrhs) {
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  const auto flag = fb::lu_solve(res, n, nrhs, d_a.data(), d_b.data());
  EXPECT_TRUE(flag.has_value()) << flag.error().message;
  EXPECT_EQ(flag.value_or(-1), 0) << "n = " << n;
  return to_host(res, d_b);
}

TEST(Lu, MatchesHostOnIndefiniteAndGeneralMatrices) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kRhs = 3;
  for (const std::int32_t n : {1, 2, 6, 17, 64, 65, 130}) {
    for (const bool symmetric : {true, false}) {
      const auto seed = static_cast<std::uint32_t>(n) * 2U + (symmetric ? 1U : 0U);
      auto a = symmetric ? random_indefinite(n, seed) : random_matrix(n, n, seed);
      if (!symmetric) {
        // Diagonally weighted, so the general case is not near singular.
        for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
          a[j + (static_cast<std::size_t>(n) * j)] += static_cast<double>(n);
        }
      }
      const auto b = random_matrix(n, kRhs, seed + 1000U);
      const auto got = device_solve(res, a, b, n, kRhs);
      const auto want = host_solve(a, b, static_cast<std::size_t>(n), kRhs);
      double scale = 0.0;
      double worst = 0.0;
      for (std::size_t e = 0; e < want.size(); ++e) {
        scale = std::max(scale, std::abs(want[e]));
        worst = std::max(worst, std::abs(got[e] - want[e]));
      }
      EXPECT_LE(worst, 1e-12 * std::max(scale, 1.0))
          << "n = " << n << (symmetric ? " indefinite" : " general");
    }
  }
}

// The same solve, repeated: every result bitwise the first. A factorization
// that races between its waves (as rocSOLVER's potf2 did, #410) differs from
// run to run under contention; the concurrent-process run in the PR is what
// supplies that.
TEST(Lu, IsBitwiseRepeatable) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const std::int32_t n : {6, 64, 127}) {
    const auto a = random_indefinite(n, 77U + static_cast<std::uint32_t>(n));
    const auto b = random_matrix(n, 2, 99U);
    const auto first = device_solve(res, a, b, n, 2);
    for (std::int32_t rep = 0; rep < 20; ++rep) {
      ASSERT_EQ(device_solve(res, a, b, n, 2), first) << "n = " << n << ", repeat " << rep;
    }
  }
}

// A zero third column makes U(3, 3) exactly zero, whatever the pivoting does
// to the first two: the flag is 3, 1-based, and B is untouched.
TEST(Lu, ASingularMatrixIsFlaggedAndBLeftAsItWas) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 5;
  auto a = random_indefinite(kN, 5U);
  for (std::size_t i = 0; i < kN; ++i) {
    a[i + (kN * 2)] = 0.0;
  }
  const auto b = random_matrix(kN, 1, 6U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  const auto flag = fb::lu_solve(res, kN, 1, d_a.data(), d_b.data());
  ASSERT_TRUE(flag.has_value()) << flag.error().message;
  EXPECT_EQ(*flag, 3);
  EXPECT_EQ(to_host(res, d_b), b);
}

TEST(Lu, EmptyAndNegativeShapes) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto empty = fb::lu_solve(res, 0, 1, nullptr, nullptr);
  ASSERT_TRUE(empty.has_value()) << empty.error().message;
  EXPECT_EQ(*empty, 0);
  for (const auto [n, nrhs] : {std::pair{-1, 1}, std::pair{2, -1}}) {
    const auto refused = fb::lu_solve(res, n, nrhs, nullptr, nullptr);
    ASSERT_FALSE(refused.has_value()) << n << " " << nrhs;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(refused.error().message.find("lu_solve"), std::string::npos);
  }
}

// getrf's work and the pivots, the device workspace alone, at the RIS
// capacitance matrix's order (water's RIS-J basis, 7 functions) and past a
// hundred: the first solve grows the workspace to exactly the footprint's,
// and the second, measured, allocates nothing.
TEST(Lu, Footprint) {
  FBF_REQUIRE_GPU();
  for (const std::int32_t n : {7, 130}) {
    SCOPED_TRACE(n);
    fb::DeviceResources res{0};
    const auto lwork = fb::getrf_lwork(res, n);
    ASSERT_TRUE(lwork) << lwork.error().message;
    const fb::MemoryFootprint fp = fb::lu_solve_footprint(n, *lwork);
    ASSERT_GT(fp.device.workspace, 0U);
    EXPECT_EQ(fp.device.persistent + fp.device.scratch, 0U);
    const auto a = random_indefinite(n, 7U);
    const auto b = random_matrix(n, 3, 8U);
    auto d_a = to_device(res, a);
    auto d_b = to_device(res, b);
    const auto solve = [&] {
      EXPECT_EQ(wwr::wwrMemcpyAsync(d_a.data(), a.data(), sizeof(double) * a.size(),
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                wwr::wwrSuccess);
      const auto flag = fb::lu_solve(res, n, 3, d_a.data(), d_b.data());
      ASSERT_TRUE(flag.has_value()) << flag.error().message;
      EXPECT_EQ(*flag, 0);
    };
    solve();
    EXPECT_EQ(res.device_workspace(0).num_elements(), fp.device.workspace);
    fbt::expect_footprint(fbt::measure_pool_peak(*res.device(), solve), {}, 0);
  }
  EXPECT_EQ(fb::lu_solve_footprint(0, 0), fb::MemoryFootprint{});
}

} // namespace
