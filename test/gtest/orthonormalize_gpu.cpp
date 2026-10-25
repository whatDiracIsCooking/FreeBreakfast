// GPU cases for recipes.orthogonalize's orthonormalize: the kept columns Q are
// orthonormal and span the input V, a dependent or nearly dependent column
// drops exactly when its direction is at or below the floor, a short column
// beside a long one keeps its direction, and k = 0 and k = 1 work.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

// Column-major n x k on the host, column j at j * n.
template <typename T>
using Block = std::vector<T>;

template <typename T>
Block<T> random_block(std::int32_t n, std::int32_t k, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<T> dist{T{-1}, T{1}};
  Block<T> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

template <typename T>
T *column(Block<T> &v, std::int32_t n, std::int32_t j) {
  return v.data() + (static_cast<std::size_t>(j) * static_cast<std::size_t>(n));
}

template <typename T>
void normalize(T *x, std::int32_t n) {
  double norm2 = 0.0;
  for (std::int32_t i = 0; i < n; ++i) {
    norm2 += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const double inv = 1.0 / std::sqrt(norm2);
  for (std::int32_t i = 0; i < n; ++i) {
    x[i] = static_cast<T>(static_cast<double>(x[i]) * inv);
  }
}

// orthonormalize over a copy of the host block @p v; returns the kept count
// and the kept columns.
template <typename T>
std::pair<std::int32_t, Block<T>> run(std::int32_t n, std::int32_t k, const Block<T> &v, T floor) {
  fb::DeviceResources res{0};
  const std::size_t elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
  fb::Buffers::device<T> d(std::max<std::size_t>(elements, 1), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), v.data(), elements * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  const fb::Result<std::int32_t> kept = fb::orthonormalize<T>(res, n, k, d.data(), floor);
  EXPECT_TRUE(kept.has_value()) << (kept ? "" : kept.error().message);
  if (!kept) {
    return {-1, {}};
  }
  Block<T> q(static_cast<std::size_t>(n) * static_cast<std::size_t>(*kept));
  EXPECT_EQ(wwr::wwrMemcpyAsync(q.data(), d.data(), q.size() * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return {*kept, q};
}

// max |Q^T Q - I|, in double.
template <typename T>
double orthonormality_error(const Block<T> &q, std::int32_t n, std::int32_t kept) {
  double worst = 0.0;
  for (std::int32_t a = 0; a < kept; ++a) {
    for (std::int32_t b = 0; b < kept; ++b) {
      double dot = 0.0;
      for (std::int32_t i = 0; i < n; ++i) {
        dot += static_cast<double>(q[i + (static_cast<std::size_t>(a) * n)]) *
               static_cast<double>(q[i + (static_cast<std::size_t>(b) * n)]);
      }
      worst = std::max(worst, std::abs(dot - (a == b ? 1.0 : 0.0)));
    }
  }
  return worst;
}

// max over V's columns of ||v - Q Q^T v|| / ||v||, in double: V's part
// outside span(Q).
template <typename T>
double projection_residual(const Block<T> &v, std::int32_t k, const Block<T> &q, std::int32_t n,
                           std::int32_t kept) {
  double worst = 0.0;
  std::vector<double> r(static_cast<std::size_t>(n));
  for (std::int32_t j = 0; j < k; ++j) {
    const T *const vj = v.data() + (static_cast<std::size_t>(j) * n);
    double norm2 = 0.0;
    for (std::int32_t i = 0; i < n; ++i) {
      r[static_cast<std::size_t>(i)] = static_cast<double>(vj[i]);
      norm2 += r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
    }
    for (std::int32_t pass = 0; pass < 2; ++pass) {
      for (std::int32_t a = 0; a < kept; ++a) {
        const T *const qa = q.data() + (static_cast<std::size_t>(a) * n);
        double dot = 0.0;
        for (std::int32_t i = 0; i < n; ++i) {
          dot += static_cast<double>(qa[i]) * r[static_cast<std::size_t>(i)];
        }
        for (std::int32_t i = 0; i < n; ++i) {
          r[static_cast<std::size_t>(i)] -= dot * static_cast<double>(qa[i]);
        }
      }
    }
    double rest2 = 0.0;
    for (const double x : r) {
      rest2 += x * x;
    }
    worst = std::max(worst, std::sqrt(rest2 / norm2));
  }
  return worst;
}

// A random tall block is full rank: every column kept, Q^T Q = I, and
// span(Q) = span(V).
template <typename T>
void check_random_block(double tolerance) {
  constexpr std::int32_t kN = 517; // not a multiple of any block size
  constexpr std::int32_t kK = 7;
  const Block<T> v = random_block<T>(kN, kK, 2026);
  const auto [kept, q] = run<T>(kN, kK, v, T{1e-6});
  ASSERT_EQ(kept, kK);
  EXPECT_LT(orthonormality_error(q, kN, kept), tolerance);
  EXPECT_LT(projection_residual(v, kK, q, kN, kept), tolerance);
}

TEST(Orthonormalize, RandomTallBlockDouble) {
  FBF_REQUIRE_GPU();
  check_random_block<double>(1e-13);
}

TEST(Orthonormalize, RandomTallBlockFloat) {
  FBF_REQUIRE_GPU();
  check_random_block<float>(1e-5);
}

// Unit columns a, b, c, then: a again (an exact duplicate), b + above * w
// (its new direction's norm above the floor: kept), c + below * w (below
// the floor: dropped), and below * w alone (a column at or below the floor:
// dropped). Rank above the floor: 4.
template <typename T>
void check_dependent_columns(T floor, T above, T below, double tolerance) {
  constexpr std::int32_t kN = 300;
  constexpr std::int32_t kK = 7;
  Block<T> base = random_block<T>(kN, 4, 7);
  for (std::int32_t j = 0; j < 4; ++j) {
    normalize(column(base, kN, j), kN);
  }
  const T *const a = column(base, kN, 0);
  const T *const b = column(base, kN, 1);
  const T *const c = column(base, kN, 2);
  const T *const w = column(base, kN, 3);
  Block<T> v(static_cast<std::size_t>(kN) * kK);
  for (std::int32_t i = 0; i < kN; ++i) {
    column(v, kN, 0)[i] = a[i];
    column(v, kN, 1)[i] = b[i];
    column(v, kN, 2)[i] = c[i];
    column(v, kN, 3)[i] = a[i];
    column(v, kN, 4)[i] = b[i] + (above * w[i]);
    column(v, kN, 5)[i] = c[i] + (below * w[i]);
    column(v, kN, 6)[i] = below * w[i];
  }
  const auto [kept, q] = run<T>(kN, kK, v, floor);
  ASSERT_EQ(kept, 4);
  EXPECT_LT(orthonormality_error(q, kN, kept), tolerance);
  // span(Q) = span(V): the kept columns to rounding, and the dropped ones,
  // whose new directions were below the floor, to within those.
  EXPECT_LT(projection_residual(v, 5, q, kN, kept), tolerance);
  EXPECT_LT(projection_residual(v, kK, q, kN, kept), 2.0 * static_cast<double>(below));
}

TEST(Orthonormalize, DependentColumnsKeepTheRankAboveTheFloorDouble) {
  FBF_REQUIRE_GPU();
  check_dependent_columns<double>(1e-6, 1e-4, 1e-9, 1e-13);
}

TEST(Orthonormalize, DependentColumnsKeepTheRankAboveTheFloorFloat) {
  FBF_REQUIRE_GPU();
  check_dependent_columns<float>(1e-2F, 3e-1F, 1e-4F, 1e-5);
}

// A column many orders shorter than its neighbour is still a direction of its
// own: scaled to a unit diagonal first, it is not lost below epsilon ||G||.
template <typename T>
void check_graded_columns(T short_scale, double tolerance) {
  constexpr std::int32_t kN = 200;
  Block<T> v = random_block<T>(kN, 2, 11);
  for (std::int32_t i = 0; i < kN; ++i) {
    column(v, kN, 0)[i] *= T{1000};
    column(v, kN, 1)[i] *= short_scale;
  }
  const auto [kept, q] = run<T>(kN, 2, v, T{0});
  ASSERT_EQ(kept, 2);
  EXPECT_LT(orthonormality_error(q, kN, kept), tolerance);
  EXPECT_LT(projection_residual(v, 2, q, kN, kept), tolerance);
}

TEST(Orthonormalize, ShortColumnBesideALongOneKeepsItsDirectionDouble) {
  FBF_REQUIRE_GPU();
  check_graded_columns<double>(1e-9, 1e-13);
}

TEST(Orthonormalize, ShortColumnBesideALongOneKeepsItsDirectionFloat) {
  FBF_REQUIRE_GPU();
  check_graded_columns<float>(1e-3F, 1e-5);
}

TEST(Orthonormalize, NoColumnsKeepsNothing) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Result<std::int32_t> kept = fb::orthonormalize<double>(res, 10, 0, nullptr, 1e-12);
  ASSERT_TRUE(kept.has_value());
  EXPECT_EQ(*kept, 0);
}

// One column comes back normalized, in its own direction up to sign; one at or
// below the floor is dropped.
template <typename T>
void check_one_column(double tolerance) {
  constexpr std::int32_t kN = 129;
  const Block<T> v = random_block<T>(kN, 1, 3);
  const auto [kept, q] = run<T>(kN, 1, v, T{1e-3});
  ASSERT_EQ(kept, 1);
  EXPECT_LT(orthonormality_error(q, kN, kept), tolerance);
  EXPECT_LT(projection_residual(v, 1, q, kN, kept), tolerance);

  Block<T> tiny = v;
  for (auto &x : tiny) {
    x *= T{1e-6};
  }
  EXPECT_EQ(run<T>(kN, 1, tiny, T{1e-3}).first, 0);
}

TEST(Orthonormalize, OneColumnDouble) {
  FBF_REQUIRE_GPU();
  check_one_column<double>(1e-13);
}

TEST(Orthonormalize, OneColumnFloat) {
  FBF_REQUIRE_GPU();
  check_one_column<float>(1e-5);
}

TEST(Orthonormalize, RejectsANegativeColumnCount) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Result<std::int32_t> kept = fb::orthonormalize<double>(res, 10, -1, nullptr, 0.0);
  ASSERT_FALSE(kept.has_value());
  EXPECT_EQ(kept.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
