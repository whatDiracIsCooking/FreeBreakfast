// GPU cases for recipes.pcg: block preconditioned conjugate gradients on a
// dense A held as a one-gemm LinearOperator, against a host Cholesky solve;
// columns stopping at their own sweep and tolerance; a Jacobi preconditioner
// on an ill-conditioned diagonal; the column form, which names each block's
// columns; the caller's norms; the curvature check and the silent stops; and
// the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kN = 40;

// The true residual drifts from the recurrence's by rounding: per sweep about
// eps ||A|| ||x||, here 2.2e-16 * 40.5 * 7.3 ~ 7e-14 (||A|| <= 40.5 by
// Gershgorin, ||x|| <= ||b|| / lambda_min ~ 3.7 / 0.5), times at most kN
// sweeps: under 3e-12, so 1e-11 bounds it.
constexpr double kDrift = 1e-11;

template <typename T>
fb::Buffers::device<T> to_device(fb::DeviceResources &res, const std::vector<T> &host) {
  fb::Buffers::device<T> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

template <typename T>
std::vector<T> to_host(fb::DeviceResources &res, const T *d, std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// A dense device matrix as a LinearOperator: one gemm per apply, recording
// each apply's column count and whether it ran in host pointer mode. With
// scribble set, each apply first grows the device workspace and fills it with
// NaN bytes, as an operator taking it for its own scratch would.
struct Dense {
  std::int32_t n;
  const double *a;
  bool scribble = false;
  std::vector<std::int32_t> counts;
  bool host_mode = true;

  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    counts.push_back(k);
    const wwr::wwrblasHandle_t blas = res.blas().get();
    wwr::wwrblasPointerMode_t mode{};
    host_mode = host_mode &&
                wwr::wwrblasGetPointerMode(blas, &mode) == wwr::WWRBLAS_STATUS_SUCCESS &&
                mode == wwr::WWRBLAS_POINTER_MODE_HOST;
    if (scribble) {
      const std::size_t bytes = counts.size() << 16U;
      auto &work = res.device_workspace(bytes);
      if (auto status = fb::gpu_status(
              wwr::wwrMemsetAsync(work.data(), 0xFF, work.num_elements(), res.stream()));
          !status) {
        return status;
      }
    }
    const double one = 1.0;
    const double zero = 0.0;
    return fb::gpu_status(wwr::gemm<double, std::int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, k, n, &one, a, n, x, n, &zero, y, n));
  }
};

static_assert(fb::LinearOperator<Dense, double>);

// A + s_j I for each column j, through pcg_solve's column form: the original
// index of each of the block's columns picks its shift. Records each apply's
// columns. It has no count form, so pcg_solve can only call this one.
struct ShiftedDense {
  std::int32_t n;
  const double *a;
  std::vector<double> shifts;
  std::vector<std::vector<std::size_t>> columns;

  fb::Status apply(fb::DeviceResources &res, std::span<const std::size_t> block, const double *x,
                   double *y) {
    columns.emplace_back(block.begin(), block.end());
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const auto k = static_cast<std::int32_t>(block.size());
    const double one = 1.0;
    const double zero = 0.0;
    if (auto status = fb::gpu_status(wwr::gemm<double, std::int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, k, n, &one, a, n, x, n, &zero, y, n));
        !status) {
      return status;
    }
    for (std::size_t c = 0; c < block.size(); ++c) {
      const std::size_t at = c * static_cast<std::size_t>(n);
      if (auto status = fb::gpu_status(
              wwr::axpy<double, std::int32_t>(blas, n, &shifts[block[c]], x + at, 1, y + at, 1));
          !status) {
        return status;
      }
    }
    return {};
  }
};

static_assert(!fb::LinearOperator<ShiftedDense, double>);

// Column-major A = D^1/2 C D^1/2, C with unit diagonal and symmetric
// off-diagonal elements uniform in +-0.5 / n, so each row of C - I sums under
// 0.5 and C's eigenvalues lie in (0.5, 1.5) (Gershgorin). With D = I + diag(0,
// 1, ..., n-1) (spread 0) A's spectrum is in (0.5, n + 0.5); with D log-spaced
// over 10^spread it is ill-conditioned, and Jacobi's D^-1 recovers C's.
std::vector<double> spd(double spread, bool linear_diagonal) {
  std::mt19937 gen{20261106U};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> d(kN);
  for (std::int32_t i = 0; i < kN; ++i) {
    d[i] = linear_diagonal ? 1.0 + i : std::pow(10.0, spread * i / (kN - 1));
  }
  std::vector<double> a(static_cast<std::size_t>(kN) * kN);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = j; i < kN; ++i) {
      const double c = i == j ? 1.0 : 0.5 / kN * dist(gen);
      a[i + j * kN] = a[j + i * kN] = std::sqrt(d[i] * d[j]) * c;
    }
  }
  return a;
}

std::vector<double> random_columns(std::int32_t k, std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> b(static_cast<std::size_t>(kN) * k);
  for (double &v : b) {
    v = dist(gen);
  }
  return b;
}

// X = A^-1 B on the host by Cholesky in long double: the reference.
std::vector<double> host_solve(const std::vector<double> &a, const std::vector<double> &b,
                               std::int32_t k) {
  std::vector<long double> l(static_cast<std::size_t>(kN) * kN, 0.0L);
  for (std::int32_t j = 0; j < kN; ++j) {
    long double diagonal = a[j + j * kN];
    for (std::int32_t m = 0; m < j; ++m) {
      diagonal -= l[j + m * kN] * l[j + m * kN];
    }
    l[j + j * kN] = std::sqrt(diagonal);
    for (std::int32_t i = j + 1; i < kN; ++i) {
      long double value = a[i + j * kN];
      for (std::int32_t m = 0; m < j; ++m) {
        value -= l[i + m * kN] * l[j + m * kN];
      }
      l[i + j * kN] = value / l[j + j * kN];
    }
  }
  std::vector<double> x(b.size());
  for (std::int32_t c = 0; c < k; ++c) {
    std::vector<long double> y(kN);
    for (std::int32_t i = 0; i < kN; ++i) {
      long double value = b[i + c * kN];
      for (std::int32_t m = 0; m < i; ++m) {
        value -= l[i + m * kN] * y[m];
      }
      y[i] = value / l[i + i * kN];
    }
    for (std::int32_t i = kN - 1; i >= 0; --i) {
      long double value = y[i];
      for (std::int32_t m = i + 1; m < kN; ++m) {
        value -= l[m + i * kN] * y[m];
      }
      y[i] = value / l[i + i * kN];
      x[i + c * kN] = static_cast<double>(y[i]);
    }
  }
  return x;
}

// ||A x_c - b_c||_2 on the host, in long double.
double host_residual(const std::vector<double> &a, const std::vector<double> &x,
                     const std::vector<double> &b, std::int32_t c) {
  long double sum = 0.0L;
  for (std::int32_t i = 0; i < kN; ++i) {
    long double ax = -static_cast<long double>(b[i + c * kN]);
    for (std::int32_t j = 0; j < kN; ++j) {
      ax += static_cast<long double>(a[i + j * kN]) * x[j + c * kN];
    }
    sum += ax * ax;
  }
  return static_cast<double>(std::sqrt(sum));
}

double column_distance(const std::vector<double> &u, const std::vector<double> &v, std::int32_t c) {
  double sum = 0.0;
  for (std::int32_t i = 0; i < kN; ++i) {
    const double d = u[i + c * kN] - v[i + c * kN];
    sum += d * d;
  }
  return std::sqrt(sum);
}

// The diagonal of A, inverted, as a dense matrix: Jacobi's M.
std::vector<double> inverse_diagonal(const std::vector<double> &a) {
  std::vector<double> m(a.size(), 0.0);
  for (std::int32_t i = 0; i < kN; ++i) {
    m[i + i * kN] = 1.0 / a[i + i * kN];
  }
  return m;
}

// The SPD model, three columns to 1e-10 with no preconditioner, against the
// host solve, with an operator that overwrites the workspace on every apply.
// lambda_min > 0.5, so ||x - x_ref|| <= ||A x - b|| / 0.5 plus the reference's
// own rounding (~1e-15).
TEST(RecipesPcg, NoPreconditionerSolvesDenseSpdAgainstHostSolve) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 3;
  constexpr double kTolerance = 1e-10;
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 1U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());
  Dense op{kN, d_a.data(), true};
  fb::PcgOptions options;
  options.tolerance = kTolerance;
  const auto info = fb::pcg_solve(res, op, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_TRUE(op.host_mode);
  const auto x = to_host(res, d_x.data(), b.size());
  const auto reference = host_solve(a, b, kCols);
  ASSERT_EQ(info->residuals.size(), static_cast<std::size_t>(kCols));
  for (std::int32_t c = 0; c < kCols; ++c) {
    EXPECT_TRUE(info->converged[c]) << c;
    EXPECT_GE(info->iterations[c], 1) << c;
    EXPECT_LE(info->residuals[c], kTolerance + kDrift) << c;
    const double residual = host_residual(a, x, b, c);
    EXPECT_LE(residual, kTolerance + kDrift) << c;
    EXPECT_LE(column_distance(x, reference, c), (kTolerance + kDrift) / 0.5 + 1e-14) << c;
  }
  // The solve's applies, then the true residuals' one over every column.
  EXPECT_EQ(op.counts.back(), kCols);
  EXPECT_EQ(static_cast<std::int32_t>(op.counts.size()), info->sweeps + 1);
}

// Per-column tolerances 1e-2, 1e-6, 1e-10: each column stops at its own
// sweep, every later apply covers only the columns still running, and a
// stopped column is left as it was. Each column solved alone to its tolerance
// takes the same products to the same x: a further step would move it by
// about its residual (>= 1e-10 / 40.5), far above the rounding between a
// block and a single-column gemm, which grows at most like sweeps * kappa *
// eps ~ 40 * 81 * 2.2e-16 < 1e-12 relative.
TEST(RecipesPcg, ColumnsStopAtTheirOwnTolerance) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 3;
  const std::array<double, kCols> tolerances{1e-2, 1e-6, 1e-10};
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 2U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());
  Dense op{kN, d_a.data()};
  fb::PcgOptions options;
  options.tolerances = tolerances;
  const auto info = fb::pcg_solve(res, op, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_LT(info->iterations[0], info->iterations[1]);
  EXPECT_LT(info->iterations[1], info->iterations[2]);
  EXPECT_EQ(info->sweeps, info->iterations[2]);
  // Sweep s applies A to the columns whose iteration count reaches s.
  ASSERT_EQ(static_cast<std::int32_t>(op.counts.size()), info->sweeps + 1);
  for (std::int32_t s = 1; s <= info->sweeps; ++s) {
    const auto running =
        std::ranges::count_if(info->iterations, [s](std::int32_t i) { return i >= s; });
    EXPECT_EQ(op.counts[s - 1], running) << s;
  }
  const auto x = to_host(res, d_x.data(), b.size());
  for (std::int32_t c = 0; c < kCols; ++c) {
    SCOPED_TRACE(c);
    EXPECT_TRUE(info->converged[c]);
    EXPECT_LE(info->residuals[c], tolerances[c] + kDrift);
    // The same column alone.
    const std::vector<double> column(b.begin() + c * kN, b.begin() + (c + 1) * kN);
    auto d_column = to_device(res, column);
    fb::Buffers::device<double> d_alone(column.size(), res.device());
    Dense single{kN, d_a.data()};
    fb::PcgOptions alone_options;
    alone_options.tolerance = tolerances[c];
    const auto alone =
        fb::pcg_solve(res, single, nullptr, kN, 1, d_column.data(), d_alone.data(), alone_options);
    ASSERT_TRUE(alone) << alone.error().message;
    EXPECT_EQ(alone->iterations[0], info->iterations[c]);
    const auto x_alone = to_host(res, d_alone.data(), column.size());
    double distance = 0.0;
    double scale = 0.0;
    for (std::int32_t i = 0; i < kN; ++i) {
      distance += std::pow(x_alone[i] - x[i + c * kN], 2);
      scale += x_alone[i] * x_alone[i];
    }
    EXPECT_LE(std::sqrt(distance), 1e-12 * std::sqrt(scale));
  }
}

// The column form: A + s_j I per column, s = 0, 5, 20, at 1e-2, 1e-6, 1e-10,
// so the columns drop out at different sweeps and the gathered blocks are
// subsets. Each apply is told exactly the columns still running, ascending,
// and the true residuals' every column; each column solves its own system.
// lambda_min(A + s I) > 0.5, so ||x - x_ref|| <= (tol + drift) / 0.5 plus the
// reference's rounding.
TEST(RecipesPcg, ColumnFormNamesEachBlocksColumns) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 3;
  const std::array<double, kCols> tolerances{1e-2, 1e-6, 1e-10};
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 6U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());
  ShiftedDense op{kN, d_a.data(), {0.0, 5.0, 20.0}, {}};
  fb::PcgOptions options;
  options.tolerances = tolerances;
  const auto info = fb::pcg_solve(res, op, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  ASSERT_EQ(static_cast<std::int32_t>(op.columns.size()), info->sweeps + 1);
  for (std::int32_t s = 1; s <= info->sweeps; ++s) {
    std::vector<std::size_t> running;
    for (std::size_t c = 0; c < static_cast<std::size_t>(kCols); ++c) {
      if (info->iterations[c] >= s) {
        running.push_back(c);
      }
    }
    EXPECT_EQ(op.columns[s - 1], running) << s;
  }
  EXPECT_EQ(op.columns.back(), (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_LT(op.columns[info->sweeps - 1].size(), static_cast<std::size_t>(kCols));
  const auto x = to_host(res, d_x.data(), b.size());
  for (std::int32_t c = 0; c < kCols; ++c) {
    SCOPED_TRACE(c);
    EXPECT_TRUE(info->converged[c]);
    EXPECT_LE(info->residuals[c], tolerances[c] + kDrift);
    auto shifted = a;
    for (std::int32_t i = 0; i < kN; ++i) {
      shifted[i + i * kN] += op.shifts[c];
    }
    const std::vector<double> column(b.begin() + c * kN, b.begin() + (c + 1) * kN);
    const std::vector<double> x_column(x.begin() + c * kN, x.begin() + (c + 1) * kN);
    EXPECT_LE(host_residual(shifted, x_column, column, 0), tolerances[c] + kDrift);
    EXPECT_LE(column_distance(x_column, host_solve(shifted, column, 1), 0),
              (tolerances[c] + kDrift) / 0.5 + 1e-14);
  }
}

// norms: the caller's ||b_j|| in place of the call's own. Given the host's,
// which differs from the device nrm2 by rounding alone, the solve takes the
// same products to the same x within the drift of ColumnsStopAtTheirOwnTolerance
// (1e-12 relative).
TEST(RecipesPcg, CallersNormsReplaceTheCallsOwn) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 2;
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 7U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  std::vector<double> norms(kCols);
  for (std::int32_t c = 0; c < kCols; ++c) {
    double sum = 0.0;
    for (std::int32_t i = 0; i < kN; ++i) {
      sum += b[i + c * kN] * b[i + c * kN];
    }
    norms[c] = std::sqrt(sum);
  }
  fb::Buffers::device<double> d_own(b.size(), res.device());
  Dense own_op{kN, d_a.data()};
  const auto own = fb::pcg_solve(res, own_op, nullptr, kN, kCols, d_b.data(), d_own.data());
  ASSERT_TRUE(own) << own.error().message;
  fb::Buffers::device<double> d_given(b.size(), res.device());
  Dense given_op{kN, d_a.data()};
  fb::PcgOptions options;
  options.norms = norms;
  const auto given =
      fb::pcg_solve(res, given_op, nullptr, kN, kCols, d_b.data(), d_given.data(), options);
  ASSERT_TRUE(given) << given.error().message;
  EXPECT_EQ(given->iterations, own->iterations);
  EXPECT_EQ(given_op.counts, own_op.counts);
  const auto x_own = to_host(res, d_own.data(), b.size());
  const auto x_given = to_host(res, d_given.data(), b.size());
  for (std::int32_t c = 0; c < kCols; ++c) {
    EXPECT_TRUE(given->converged[c]) << c;
    EXPECT_LE(column_distance(x_given, x_own, c),
              1e-12 * column_distance(x_own, std::vector<double>(b.size(), 0.0), c))
        << c;
  }
}

// A = D^1/2 C D^1/2 over six decades: Jacobi's M A has C's spectrum, kappa < 3,
// so with rho = (sqrt 3 - 1) / (sqrt 3 + 1) < 0.268 CG's bound
// ||r_m|| / ||r_0|| <= 2 rho^m sqrt(kappa(A)), kappa(A) < 1.5e6 / 0.5, reaches
// 1e-10 by m = 24. The identity path, at kappa(A) up to 3e6, takes more.
TEST(RecipesPcg, JacobiCutsSweepsOnIllConditionedDiagonal) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 2;
  constexpr std::int32_t kJacobiBound = 24;
  fb::DeviceResources res{0};
  const auto a = spd(6.0, false);
  const auto b = random_columns(kCols, 3U);
  auto d_a = to_device(res, a);
  auto d_m = to_device(res, inverse_diagonal(a));
  auto d_b = to_device(res, b);
  std::vector<double> tolerances(kCols);
  for (std::int32_t c = 0; c < kCols; ++c) {
    double norm2 = 0.0;
    for (std::int32_t i = 0; i < kN; ++i) {
      norm2 += b[i + c * kN] * b[i + c * kN];
    }
    tolerances[c] = 1e-10 * std::sqrt(norm2);
  }
  fb::PcgOptions options;
  options.tolerances = tolerances;

  fb::Buffers::device<double> d_x(b.size(), res.device());
  Dense op{kN, d_a.data()};
  Dense jacobi{kN, d_m.data()};
  const auto preconditioned =
      fb::pcg_solve(res, op, jacobi, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(preconditioned) << preconditioned.error().message;
  EXPECT_LE(preconditioned->sweeps, kJacobiBound);
  EXPECT_TRUE(jacobi.host_mode);
  // M on every column once to start, then once per sweep on those stepped.
  EXPECT_EQ(static_cast<std::int32_t>(jacobi.counts.size()), preconditioned->sweeps + 1);
  for (std::int32_t c = 0; c < kCols; ++c) {
    EXPECT_TRUE(preconditioned->converged[c]) << c;
  }

  fb::Buffers::device<double> d_plain(b.size(), res.device());
  Dense plain_op{kN, d_a.data()};
  options.max_iterations = 2000;
  options.stop_silently = true;
  const auto plain =
      fb::pcg_solve(res, plain_op, nullptr, kN, kCols, d_b.data(), d_plain.data(), options);
  ASSERT_TRUE(plain) << plain.error().message;
  EXPECT_GT(plain->sweeps, kJacobiBound);
}

// A = diag(-1, 1, 2, ...), b = [e_0, e_1]: column 0's first direction has
// p.Ap = -1. With the check on that is Divergence; off, column 0 stops with X
// untouched while column 1 converges in one exact step, A e_1 = e_1.
TEST(RecipesPcg, CurvatureCheck) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 2;
  fb::DeviceResources res{0};
  std::vector<double> a(static_cast<std::size_t>(kN) * kN, 0.0);
  for (std::int32_t i = 0; i < kN; ++i) {
    a[i + i * kN] = i == 0 ? -1.0 : static_cast<double>(i);
  }
  std::vector<double> b(static_cast<std::size_t>(kN) * kCols, 0.0);
  b[0] = 1.0;
  b[1 + kN] = 1.0;
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());

  Dense checked{kN, d_a.data()};
  const auto divergence = fb::pcg_solve(res, checked, nullptr, kN, kCols, d_b.data(), d_x.data());
  ASSERT_FALSE(divergence);
  EXPECT_EQ(divergence.error().kind, fb::ErrorKind::Divergence);

  Dense unchecked{kN, d_a.data()};
  fb::PcgOptions options;
  options.check_curvature = false;
  const auto info =
      fb::pcg_solve(res, unchecked, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_FALSE(info->converged[0]);
  EXPECT_TRUE(info->converged[1]);
  EXPECT_EQ(info->iterations, (std::vector<std::int32_t>{1, 1}));
  EXPECT_EQ(info->residuals, (std::vector<double>{1.0, 0.0}));
  const auto x = to_host(res, d_x.data(), b.size());
  std::vector<double> expected(b.size(), 0.0);
  expected[1 + kN] = 1.0;
  EXPECT_EQ(x, expected);
}

// One sweep cannot reach 1e-10: MaxIterationsExceeded, or under stop_silently
// a normal return with every column flagged and its true residual reported.
TEST(RecipesPcg, MaxIterations) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 2;
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 4U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());
  fb::PcgOptions options;
  options.max_iterations = 1;

  Dense op{kN, d_a.data()};
  const auto exceeded = fb::pcg_solve(res, op, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_FALSE(exceeded);
  EXPECT_EQ(exceeded.error().kind, fb::ErrorKind::MaxIterationsExceeded);

  options.stop_silently = true;
  Dense silent_op{kN, d_a.data()};
  const auto silent =
      fb::pcg_solve(res, silent_op, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(silent) << silent.error().message;
  EXPECT_EQ(silent->sweeps, 1);
  EXPECT_EQ(silent->iterations, (std::vector<std::int32_t>{1, 1}));
  const auto x = to_host(res, d_x.data(), b.size());
  for (std::int32_t c = 0; c < kCols; ++c) {
    EXPECT_FALSE(silent->converged[c]) << c;
    EXPECT_GT(silent->residuals[c], options.tolerance) << c;
    EXPECT_NEAR(silent->residuals[c], host_residual(a, x, b, c), 1e-12) << c;
  }
}

// start_from_x from a solution converged to 1e-10, at 1e-9: its residual is
// already below, so no sweep runs, X is returned as given and only the
// starting product and the true residuals' apply A.
TEST(RecipesPcg, StartFromX) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kCols = 2;
  fb::DeviceResources res{0};
  const auto a = spd(0.0, true);
  const auto b = random_columns(kCols, 5U);
  auto d_a = to_device(res, a);
  auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_x(b.size(), res.device());
  Dense op{kN, d_a.data()};
  ASSERT_TRUE(fb::pcg_solve(res, op, nullptr, kN, kCols, d_b.data(), d_x.data()));
  const auto start = to_host(res, d_x.data(), b.size());

  Dense restarted{kN, d_a.data()};
  fb::PcgOptions options;
  options.start_from_x = true;
  options.tolerance = 1e-9;
  const auto info =
      fb::pcg_solve(res, restarted, nullptr, kN, kCols, d_b.data(), d_x.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_EQ(info->sweeps, 0);
  EXPECT_EQ(restarted.counts, (std::vector<std::int32_t>{kCols, kCols}));
  EXPECT_EQ(to_host(res, d_x.data(), b.size()), start);
}

TEST(RecipesPcg, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> d_b(kN, res.device());
  fb::Buffers::device<double> d_x(kN, res.device());
  Dense op{kN, nullptr};
  const auto refused = [&](std::int32_t n, std::int32_t k, const fb::PcgOptions &options) {
    const auto info = fb::pcg_solve(res, op, nullptr, n, k, d_b.data(), d_x.data(), options);
    return !info && info.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused(0, 1, {}));
  EXPECT_TRUE(refused(kN, 0, {}));
  fb::PcgOptions no_sweeps;
  no_sweeps.max_iterations = 0;
  EXPECT_TRUE(refused(kN, 1, no_sweeps));
  const std::array<double, 2> two{1e-10, 1e-10};
  fb::PcgOptions mismatched;
  mismatched.tolerances = two;
  EXPECT_TRUE(refused(kN, 1, mismatched));
  const std::array<double, 1> one_norm{1.0};
  fb::PcgOptions wrong_norms;
  wrong_norms.norms = two;
  EXPECT_TRUE(refused(kN, 1, wrong_norms));
  fb::PcgOptions norms_from_x;
  norms_from_x.norms = one_norm;
  norms_from_x.start_from_x = true;
  EXPECT_TRUE(refused(kN, 1, norms_from_x));
  EXPECT_TRUE(op.counts.empty());
}

} // namespace
