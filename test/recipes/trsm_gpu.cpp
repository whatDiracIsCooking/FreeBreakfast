// GPU cases for recipes.trsm: every side, fill, op and unit flag, in float and
// double, against a host substitution at orders around the 64-row block's
// edges and right-hand sides around the 16-column tile's, with padded leading
// dimensions; the same against the library's trsm at a size many blocks deep;
// _C the same as _T; alpha zero over NaN; and the argument refusals.
//
// Every element a correct trsm must not read -- A's other triangle, its
// diagonal under DIAG_UNIT, the padding rows of A and B -- holds NaN, so a
// read of one shows up in the result, and the padding of B must come back
// bitwise as it went in.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

#include "test_workspaces.h"

namespace {

namespace fb = free_breakfast;

struct Shape {
  wwr::wwrblasSideMode_t side;
  wwr::wwrblasFillMode_t fill;
  wwr::wwrblasOperation_t op;
  wwr::wwrblasDiagType_t diag;
};

std::string shape_name(const Shape &s) {
  std::string name = s.side == wwr::WWRBLAS_SIDE_LEFT ? "Left" : "Right";
  name += s.fill == wwr::WWRBLAS_FILL_MODE_LOWER ? "Lower" : "Upper";
  name += s.op == wwr::WWRBLAS_OP_N ? "N" : "T";
  name += s.diag == wwr::WWRBLAS_DIAG_UNIT ? "Unit" : "NonUnit";
  return name;
}

// gtest prints a failing case's param with this, not as raw bytes.
void PrintTo(const Shape &s, std::ostream *os) {
  *os << shape_name(s);
}

std::vector<Shape> all_shapes() {
  std::vector<Shape> shapes;
  for (const auto side : {wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_SIDE_RIGHT}) {
    for (const auto fill : {wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_FILL_MODE_UPPER}) {
      for (const auto op : {wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T}) {
        for (const auto diag : {wwr::WWRBLAS_DIAG_NON_UNIT, wwr::WWRBLAS_DIAG_UNIT}) {
          shapes.push_back({side, fill, op, diag});
        }
      }
    }
  }
  return shapes;
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// One problem as the host holds it: A (order k, leading dimension lda) and
// B (m x n, leading dimension ldb), in T.
template <typename T>
struct Problem {
  Shape shape;
  std::int32_t m = 0;
  std::int32_t n = 0;
  std::int32_t k = 0;
  std::int32_t lda = 0;
  std::int32_t ldb = 0;
  std::vector<T> a;
  std::vector<T> b;
};

// A well-conditioned triangular A: its diagonal in [1, 2], the rest of its
// stored triangle in [-2/k, 2/k], so op(A) is the identity plus a small
// perturbation and its inverse is O(1). Every element trsm must not read is
// NaN, and B's padding rows are NaN too.
template <typename T>
Problem<T> make_problem(const Shape &shape, const std::int32_t m, const std::int32_t n,
                        const std::uint32_t seed) {
  Problem<T> p{.shape = shape, .m = m, .n = n};
  p.k = shape.side == wwr::WWRBLAS_SIDE_LEFT ? m : n;
  p.lda = p.k + 3;
  p.ldb = m + 5;
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> unit_dist{-1.0, 1.0};
  std::uniform_real_distribution<double> diag_dist{1.0, 2.0};
  const double off_scale = 2.0 / static_cast<double>(p.k);
  const bool lower = shape.fill == wwr::WWRBLAS_FILL_MODE_LOWER;
  const bool unit = shape.diag == wwr::WWRBLAS_DIAG_UNIT;
  const auto la = static_cast<std::size_t>(p.lda);
  const auto lb = static_cast<std::size_t>(p.ldb);
  p.a.assign(la * static_cast<std::size_t>(p.k), static_cast<T>(kNaN));
  for (std::int32_t j = 0; j < p.k; ++j) {
    for (std::int32_t i = 0; i < p.k; ++i) {
      T &v = p.a[static_cast<std::size_t>(i) + (la * static_cast<std::size_t>(j))];
      if (i == j) {
        if (!unit) {
          v = static_cast<T>(diag_dist(gen));
        }
      } else if ((i > j) == lower) {
        v = static_cast<T>(off_scale * unit_dist(gen));
      }
    }
  }
  p.b.assign(lb * static_cast<std::size_t>(n), static_cast<T>(kNaN));
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < m; ++i) {
      p.b[static_cast<std::size_t>(i) + (lb * static_cast<std::size_t>(j))] =
          static_cast<T>(unit_dist(gen));
    }
  }
  return p;
}

// The oracle: the solve by plain substitution, in double from T's inputs. M is
// op(A) on the left and op(A)^T on the right, read from the stored triangle
// alone, and M Y = alpha C with C = B or B^T; returns the m x n packed X.
template <typename T>
std::vector<double> host_trsm(const Problem<T> &p, const double alpha) {
  const bool left = p.shape.side == wwr::WWRBLAS_SIDE_LEFT;
  const bool trans = p.shape.op != wwr::WWRBLAS_OP_N;
  const bool lower_fill = p.shape.fill == wwr::WWRBLAS_FILL_MODE_LOWER;
  const bool unit = p.shape.diag == wwr::WWRBLAS_DIAG_UNIT;
  const auto k = static_cast<std::size_t>(p.k);
  const auto la = static_cast<std::size_t>(p.lda);
  const auto lb = static_cast<std::size_t>(p.ldb);
  const auto m = static_cast<std::size_t>(p.m);
  const auto n = static_cast<std::size_t>(p.n);
  // A(i, j) from its stored triangle, zero in the other and 1 on a unit
  // diagonal, so a NaN there never reaches the oracle.
  const auto a_at = [&](const std::size_t i, const std::size_t j) -> double {
    if (i == j) {
      return unit ? 1.0 : static_cast<double>(p.a[i + (la * j)]);
    }
    return ((i > j) == lower_fill) ? static_cast<double>(p.a[i + (la * j)]) : 0.0;
  };
  const auto op_a = [&](const std::size_t i, const std::size_t j) {
    return trans ? a_at(j, i) : a_at(i, j);
  };
  const auto m_at = [&](const std::size_t i, const std::size_t j) {
    return left ? op_a(i, j) : op_a(j, i);
  };
  const bool m_lower = left ? lower_fill != trans : lower_fill == trans;
  const std::size_t ncols = left ? n : m;
  std::vector<double> x(m * n);
  std::vector<double> y(k);
  for (std::size_t c = 0; c < ncols; ++c) {
    for (std::size_t r = 0; r < k; ++r) {
      y[r] = alpha * static_cast<double>(left ? p.b[r + (lb * c)] : p.b[c + (lb * r)]);
    }
    for (std::size_t step = 0; step < k; ++step) {
      const std::size_t r = m_lower ? step : k - 1 - step;
      double v = y[r];
      for (std::size_t q = 0; q < k; ++q) {
        if (m_lower ? q < r : q > r) {
          v -= m_at(r, q) * y[q];
        }
      }
      y[r] = v / m_at(r, r);
    }
    for (std::size_t r = 0; r < k; ++r) {
      (left ? x[r + (m * c)] : x[c + (m * r)]) = y[r];
    }
  }
  return x;
}

template <typename T>
std::vector<T> download(fb::DeviceResources &res, const fb::Buffers::device<T> &d,
                        const std::size_t count) {
  std::vector<T> out(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), d.data(), out.size() * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

template <typename T>
fb::Buffers::device<T> upload(fb::DeviceResources &res, const std::vector<T> &host) {
  fb::Buffers::device<T> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  return d;
}

template <typename T>
bool same_bits(const T x, const T y) {
  if constexpr (std::same_as<T, float>) {
    return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
  } else {
    return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
  }
}

// Run fb::trsm on the problem with @p op in place of its own; return B as it
// came back, A checked unchanged.
template <typename T>
std::vector<T> run_trsm(fb::DeviceResources &res, const Problem<T> &p, const T alpha,
                        const wwr::wwrblasOperation_t op) {
  const fb::Buffers::device<T> a = upload(res, p.a);
  fb::Buffers::device<T> b = upload(res, p.b);
  const fb::Status st = fb::trsm<T>(res, p.shape.side, p.shape.fill, op, p.shape.diag, p.m, p.n,
                                    alpha, a.data(), p.lda, b.data(), p.ldb);
  EXPECT_TRUE(st.has_value()) << st.error().message;
  const std::vector<T> a_after = download(res, a, p.a.size());
  EXPECT_TRUE(std::ranges::equal(a_after, p.a, same_bits<T>)) << "A was written";
  return download(res, b, p.b.size());
}

// The bar, by analysis: substitution's forward error is about k eps |M^-1| |M|
// |X|, and A is built so that |M^-1| |M| is O(1); a factor 8 covers that and
// the gemm's different summation order. On |X|, which is O(1) here.
template <typename T>
double tolerance(const std::int32_t k) {
  return 8.0 * static_cast<double>(std::max(k, 1)) * std::numeric_limits<T>::epsilon();
}

// The worst error of @p got against @p want over B's m x n, scaled by
// max(1, |want|), and the padding rows of B checked bitwise against @p p.b.
template <typename T>
double worst_error(const Problem<T> &p, const std::vector<T> &got,
                   const std::vector<double> &want) {
  const auto lb = static_cast<std::size_t>(p.ldb);
  const auto m = static_cast<std::size_t>(p.m);
  double worst = 0.0;
  for (std::size_t j = 0; j < static_cast<std::size_t>(p.n); ++j) {
    for (std::size_t i = 0; i < lb; ++i) {
      const T g = got[i + (lb * j)];
      if (i >= m) {
        EXPECT_TRUE(same_bits(g, p.b[i + (lb * j)])) << "padding (" << i << ", " << j << ")";
        continue;
      }
      const double w = want[i + (m * j)];
      const double e = std::abs(static_cast<double>(g) - w) / std::max(1.0, std::abs(w));
      worst = std::isnan(e) ? std::numeric_limits<double>::infinity() : std::max(worst, e);
    }
  }
  return worst;
}

class Trsm : public ::testing::TestWithParam<Shape> {};

template <typename T>
void check_against_host(const Shape &shape) {
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  constexpr T kAlpha = static_cast<T>(-1.5);
  // One block, its edges, two blocks and their edge, and several; the
  // right-hand sides one, a tile's edge, and past two tiles.
  for (const std::int32_t k : {1, 2, 63, 64, 65, 128, 129, 200}) {
    for (const std::int32_t rhs : {1, 16, 17, 33}) {
      const bool left = shape.side == wwr::WWRBLAS_SIDE_LEFT;
      const Problem<T> p = make_problem<T>(shape, left ? k : rhs, left ? rhs : k,
                                           static_cast<std::uint32_t>(k + rhs));
      const std::vector<T> got = run_trsm(res, p, kAlpha, shape.op);
      const double err = worst_error(p, got, host_trsm(p, kAlpha));
      EXPECT_LE(err, tolerance<T>(k)) << "k = " << k << ", rhs = " << rhs;
    }
  }
}

TEST_P(Trsm, MatchesTheHostSubstitution) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("LeftLowerNNonUnit");
  check_against_host<double>(GetParam());
  check_against_host<float>(GetParam());
}

// Many blocks deep, against the BLAS library's own trsm: k = 700 is not a
// multiple of the block (64) or of rocBLAS's 128, and n = 300 right-hand sides
// is 19 tiles. Both sides of the comparison read only the stored triangle, so
// the NaNs elsewhere are a check on the library too.
template <typename T>
void check_against_library(const Shape &shape) {
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  constexpr std::int32_t kOrder = 700;
  constexpr std::int32_t kRhs = 300;
  constexpr T kAlpha = static_cast<T>(0.75);
  const bool left = shape.side == wwr::WWRBLAS_SIDE_LEFT;
  const Problem<T> p = make_problem<T>(shape, left ? kOrder : kRhs, left ? kRhs : kOrder, 2468);
  const std::vector<T> got = run_trsm(res, p, kAlpha, shape.op);

  const fb::Buffers::device<T> a = upload(res, p.a);
  fb::Buffers::device<T> b = upload(res, p.b);
  const wwr::wwrblasStatus_t st =
      wwr::trsm<T, std::int32_t>(res.blas().get(), shape.side, shape.fill, shape.op, shape.diag,
                                 p.m, p.n, &kAlpha, a.data(), p.lda, b.data(), p.ldb);
  ASSERT_EQ(st, wwr::WWRBLAS_STATUS_SUCCESS);
  const std::vector<T> lib = download(res, b, p.b.size());
  std::vector<double> want(static_cast<std::size_t>(p.m) * static_cast<std::size_t>(p.n));
  for (std::int32_t j = 0; j < p.n; ++j) {
    for (std::int32_t i = 0; i < p.m; ++i) {
      want[static_cast<std::size_t>(i) + (static_cast<std::size_t>(p.m) * j)] = static_cast<double>(
          lib[static_cast<std::size_t>(i) + (static_cast<std::size_t>(p.ldb) * j)]);
    }
  }
  // Two computations each within the bar of the exact solution.
  EXPECT_LE(worst_error(p, got, want), 2.0 * tolerance<T>(kOrder));
}

TEST_P(Trsm, MatchesTheLibraryManyBlocksDeep) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("RightUpperTUnit");
  check_against_library<double>(GetParam());
  check_against_library<float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Shapes, Trsm, ::testing::ValuesIn(all_shapes()),
                         [](const ::testing::TestParamInfo<Shape> &info) {
                           return shape_name(info.param);
                         });

// On a real T, _C is _T: the same result to the bit.
TEST(TrsmOp, ConjugateTransposeIsTranspose) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  for (const auto side : {wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_SIDE_RIGHT}) {
    const Shape shape{side, wwr::WWRBLAS_FILL_MODE_UPPER, wwr::WWRBLAS_OP_T,
                      wwr::WWRBLAS_DIAG_NON_UNIT};
    const Problem<double> p = make_problem<double>(shape, 90, 70, 97);
    const std::vector<double> t = run_trsm(res, p, 1.0, wwr::WWRBLAS_OP_T);
    const std::vector<double> c = run_trsm(res, p, 1.0, wwr::WWRBLAS_OP_C);
    EXPECT_TRUE(std::ranges::equal(t, c, same_bits<double>)) << shape_name(shape);
  }
}

// alpha = 0: B comes back zero over its m x n and untouched in its padding,
// with A and B all NaN, so neither was read.
template <typename T>
void check_alpha_zero(fb::DeviceResources &res, const Shape &shape) {
  const bool left = shape.side == wwr::WWRBLAS_SIDE_LEFT;
  Problem<T> p = make_problem<T>(shape, left ? 130 : 20, left ? 20 : 130, 5);
  std::ranges::fill(p.a, static_cast<T>(kNaN));
  std::ranges::fill(p.b, static_cast<T>(kNaN));
  const std::vector<T> got = run_trsm(res, p, T{0}, shape.op);
  const std::vector<double> zero(static_cast<std::size_t>(p.m) * static_cast<std::size_t>(p.n),
                                 0.0);
  EXPECT_EQ(worst_error(p, got, zero), 0.0) << shape_name(shape);
}

TEST(TrsmAlpha, ZeroWritesZeroAndReadsNothing) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  for (const Shape &shape : all_shapes()) {
    check_alpha_zero<double>(res, shape);
    check_alpha_zero<float>(res, shape);
  }
}

TEST(TrsmArguments, EmptySucceedsAndBadArgumentsAreInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const auto call = [&](const wwr::wwrblasSideMode_t side, const wwr::wwrblasFillMode_t fill,
                        const wwr::wwrblasOperation_t op, const wwr::wwrblasDiagType_t diag,
                        const std::int32_t m, const std::int32_t n, const std::int32_t lda,
                        const std::int32_t ldb) {
    return fb::trsm<double>(res, side, fill, op, diag, m, n, 1.0, nullptr, lda, nullptr, ldb);
  };
  constexpr auto kL = wwr::WWRBLAS_SIDE_LEFT;
  constexpr auto kR = wwr::WWRBLAS_SIDE_RIGHT;
  constexpr auto kLo = wwr::WWRBLAS_FILL_MODE_LOWER;
  constexpr auto kN = wwr::WWRBLAS_OP_N;
  constexpr auto kNu = wwr::WWRBLAS_DIAG_NON_UNIT;

  // Empty: nothing is launched, so null pointers are fine.
  EXPECT_TRUE(call(kL, kLo, kN, kNu, 0, 5, 1, 1).has_value());
  EXPECT_TRUE(call(kL, kLo, kN, kNu, 5, 0, 5, 5).has_value());
  EXPECT_TRUE(call(kR, kLo, kN, kNu, 5, 0, 1, 5).has_value());

  const auto invalid = [](const fb::Status &st) {
    return !st.has_value() && st.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(invalid(call(kL, kLo, kN, kNu, -1, 5, 1, 1)));
  EXPECT_TRUE(invalid(call(kL, kLo, kN, kNu, 5, -1, 5, 5)));
  // lda follows the side: k = m on the left, n on the right.
  EXPECT_TRUE(invalid(call(kL, kLo, kN, kNu, 5, 8, 4, 5)));
  EXPECT_TRUE(invalid(call(kR, kLo, kN, kNu, 5, 8, 7, 5)));
  // ldb is at least m on either side.
  EXPECT_TRUE(invalid(call(kL, kLo, kN, kNu, 5, 8, 5, 4)));
  EXPECT_TRUE(invalid(call(kR, kLo, kN, kNu, 5, 8, 8, 4)));
  // An enum that is none of its values.
  EXPECT_TRUE(invalid(call(static_cast<wwr::wwrblasSideMode_t>(99), kLo, kN, kNu, 5, 5, 5, 5)));
  EXPECT_TRUE(invalid(call(kL, static_cast<wwr::wwrblasFillMode_t>(99), kN, kNu, 5, 5, 5, 5)));
  EXPECT_TRUE(invalid(call(kL, kLo, static_cast<wwr::wwrblasOperation_t>(99), kNu, 5, 5, 5, 5)));
  EXPECT_TRUE(invalid(call(kL, kLo, kN, static_cast<wwr::wwrblasDiagType_t>(99), 5, 5, 5, 5)));
}

} // namespace
