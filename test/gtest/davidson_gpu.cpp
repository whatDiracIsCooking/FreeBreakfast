// GPU cases for recipes.davidson and the LinearOperator concept it applies:
// the lowest eigenpairs of a dense symmetric A, held as a one-gemm operator,
// against diagonalize's full spectrum of the same A; the two non-converged
// stops; and an operator that takes the device workspace on every apply.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

// A = diag(1..n) + eps (R + R^T): symmetric and diagonally dominant, so its
// lowest eigenvalues sit near 1, 2, 3, ... and the diagonal preconditioner
// resolves them quickly. Column-major, both triangles.
template <typename T>
std::vector<T> diag_plus_perturbation(std::int32_t n, double eps) {
  std::mt19937 gen{20261004U};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = j; i < n; ++i) {
      const double value = (i == j) ? static_cast<double>(j + 1) : eps * dist(gen);
      a[i + j * n] = a[j + i * n] = static_cast<T>(value);
    }
  }
  return a;
}

template <typename T>
fb::Buffers::device<T> to_device(fb::DeviceResources &res, const std::vector<T> &host) {
  fb::Buffers::device<T> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return d;
}

template <typename T>
std::vector<T> to_host(fb::DeviceResources &res, const T *d, std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// The ascending spectrum of the leading m x m block of the n x n host A,
// through diagonalize: the reference every case compares against.
template <typename T>
std::vector<T> leading_spectrum(fb::DeviceResources &res, const std::vector<T> &a, std::int32_t n,
                                std::int32_t m) {
  std::vector<T> block(static_cast<std::size_t>(m) * m);
  for (std::int32_t j = 0; j < m; ++j) {
    for (std::int32_t i = 0; i < m; ++i) {
      block[i + j * m] = a[i + j * n];
    }
  }
  auto d = to_device(res, block);
  fb::Buffers::device<T> w(static_cast<std::size_t>(m), res.device());
  EXPECT_TRUE(fb::diagonalize<T>(res, m, d.data(), w.data()));
  return to_host(res, w.data(), static_cast<std::size_t>(m));
}

// A dense device A as a LinearOperator: one gemm per apply, counting the
// applies and the columns they covered, and noting any made outside host
// pointer mode. With scribble set, each apply first grows the device workspace
// and fills it with NaN bytes, as a sigma build taking it for its own scratch
// would.
template <typename T>
struct Dense {
  std::int32_t n;
  const T *a;
  bool scribble = false;
  std::int32_t applies = 0;
  std::int32_t columns = 0;
  bool host_mode = true;

  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const T *x, T *y) {
    ++applies;
    columns += k;
    const wwr::wwrblasHandle_t blas = res.blas().get();
    wwr::wwrblasPointerMode_t mode{};
    host_mode = host_mode &&
                wwr::wwrblasGetPointerMode(blas, &mode) == wwr::WWRBLAS_STATUS_SUCCESS &&
                mode == wwr::WWRBLAS_POINTER_MODE_HOST;
    if (scribble) {
      const std::size_t bytes = (static_cast<std::size_t>(applies) << 20U);
      auto &work = res.device_workspace(bytes);
      if (auto status = fb::gpu_status(
              wwr::wwrMemsetAsync(work.data(), 0xFF, work.num_elements(), res.stream()));
          !status) {
        return status;
      }
    }
    const T one{1};
    const T zero{0};
    return fb::gpu_status(wwr::gemm<T, std::int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n,
                                                     k, n, &one, a, n, x, n, &zero, y, n));
  }
};

// The diagonal correction r / (theta - A_jj) on the host, its denominator kept
// off zero. A test needs no kernel for it. With zero set, a zero block, which
// leaves nothing to expand by; with identity set, the residual itself.
template <typename T>
struct DiagonalPreconditioner {
  std::int32_t n;
  const std::vector<T> *a;
  bool zero = false;
  bool identity = false;

  fb::Status operator()(fb::DeviceResources &res, std::span<const T> theta,
                        const T * /*ritz_vectors*/, const T *residual, T *correction) const {
    const std::size_t count = static_cast<std::size_t>(n) * theta.size();
    if (zero) {
      return fb::gpu_status(wwr::wwrMemsetAsync(correction, 0, count * sizeof(T), res.stream()));
    }
    if (identity) {
      return fb::gpu_status(wwr::wwrMemcpyAsync(correction, residual, count * sizeof(T),
                                                wwr::wwrMemcpyDeviceToDevice, res.stream()));
    }
    std::vector<T> r = to_host(res, residual, count);
    for (std::size_t i = 0; i < theta.size(); ++i) {
      for (std::int32_t j = 0; j < n; ++j) {
        T denominator = theta[i] - (*a)[j + j * n];
        if (std::abs(denominator) < T{1e-3}) {
          denominator = denominator < T{0} ? T{-1e-3} : T{1e-3};
        }
        r[i * n + j] /= denominator;
      }
    }
    if (auto status = fb::gpu_status(wwr::wwrMemcpyAsync(correction, r.data(), count * sizeof(T),
                                                         wwr::wwrMemcpyHostToDevice, res.stream()));
        !status) {
      return status;
    }
    return res.sync();
  }
};

static_assert(fb::LinearOperator<Dense<double>, double>);
static_assert(fb::LinearOperator<Dense<float>, float>);
static_assert(!fb::LinearOperator<Dense<double>, float>);
static_assert(!fb::LinearOperator<const Dense<double>, double>); // apply is non-const
static_assert(!fb::LinearOperator<double, double>);              // no apply at all
static_assert(fb::DavidsonPreconditioner<DiagonalPreconditioner<double>, double>);
static_assert(!fb::DavidsonPreconditioner<DiagonalPreconditioner<double>, float>);
static_assert(!fb::DavidsonPreconditioner<Dense<double>, double>);

// One problem: A of order 48, its 4 lowest roots, unit-vector guesses on the
// first 4 coordinates.
template <typename T>
struct Problem {
  static constexpr std::int32_t kN = 48;
  static constexpr std::int32_t kRoots = 4;
  static constexpr bool kFloat = std::is_same_v<T, float>;

  fb::DeviceResources &res;
  std::vector<T> a = diag_plus_perturbation<T>(kN, 0.05);
  fb::Buffers::device<T> d_a = to_device(res, a);
  fb::Buffers::device<T> guess = to_device(res, unit_guess());
  fb::Buffers::device<T> values{static_cast<std::size_t>(kRoots), res.device()};
  fb::Buffers::device<T> vectors{static_cast<std::size_t>(kN) * kRoots, res.device()};
  Dense<T> op{kN, d_a.data()};
  DiagonalPreconditioner<T> precondition{kN, &a};

  static std::vector<T> unit_guess() {
    std::vector<T> g(static_cast<std::size_t>(kN) * kRoots, T{0});
    for (std::int32_t c = 0; c < kRoots; ++c) {
      g[c + c * kN] = T{1};
    }
    return g;
  }

  fb::Result<fb::DavidsonInfo> solve(std::int32_t max_subspace,
                                     const fb::DavidsonOptions &options) {
    return fb::davidson_solve<T>(res, op, precondition, kN, kRoots, max_subspace, guess.data(),
                                 kRoots, values.data(), vectors.data(), options);
  }

  static fb::DavidsonOptions converging_options() {
    fb::DavidsonOptions options;
    options.residual_tolerance = kFloat ? 1e-4 : 1e-8;
    options.max_iterations = 300;
    return options;
  }

  // The values match A's lowest, ascending, and each pair's host residual
  // ||A x - theta x|| is within the relative bound the solve converged to.
  void expect_lowest_pairs(const fb::DavidsonOptions &options) {
    const auto reference = leading_spectrum(res, a, kN, kN);
    const auto got = to_host(res, values.data(), kRoots);
    const auto x = to_host(res, vectors.data(), static_cast<std::size_t>(kN) * kRoots);
    const double value_tol = kFloat ? 5e-3 : 1e-6;
    const double scale = std::max(std::abs(static_cast<double>(reference.front())),
                                  std::abs(static_cast<double>(reference.back())));
    for (std::int32_t r = 0; r < kRoots; ++r) {
      EXPECT_NEAR(static_cast<double>(got[r]), static_cast<double>(reference[r]), value_tol) << r;
      double residual2 = 0.0;
      double norm2 = 0.0;
      for (std::int32_t i = 0; i < kN; ++i) {
        double ax = 0.0;
        for (std::int32_t j = 0; j < kN; ++j) {
          ax += static_cast<double>(a[i + j * kN]) * static_cast<double>(x[j + r * kN]);
        }
        const double xi = static_cast<double>(x[i + r * kN]);
        residual2 +=
            (ax - static_cast<double>(got[r]) * xi) * (ax - static_cast<double>(got[r]) * xi);
        norm2 += xi * xi;
      }
      EXPECT_NEAR(norm2, 1.0, kFloat ? 1e-4 : 1e-10) << r;
      // float's own rounding in the host recomputation sits above its tolerance.
      const double bound = kFloat ? 1e-3 : 10 * options.residual_tolerance * scale;
      EXPECT_LE(std::sqrt(residual2), bound) << r;
    }
  }
};

// Each with the corrections orthonormalized by Gram-Schmidt (the default)
// and by orthonormalize.
template <typename T>
void check_converges(std::int32_t max_subspace) {
  for (const bool canonical : {false, true}) {
    SCOPED_TRACE(std::format("canonical_orthonormalization {}", canonical));
    fb::DeviceResources res{0};
    Problem<T> p{res};
    auto options = Problem<T>::converging_options();
    options.canonical_orthonormalization = canonical;
    const auto info = p.solve(max_subspace, options);
    ASSERT_TRUE(info) << info.error().message;
    EXPECT_GT(info->iterations, 1);
    // ||A||_2 is below n + 1 here, so this is the converged bound loosened.
    EXPECT_LE(info->max_residual_norm, options.residual_tolerance * (Problem<T>::kN + 1));
    // One apply per iteration, on only the columns new since the last: the
    // guess, then each expansion. A collapse reuses A X and applies nothing.
    EXPECT_EQ(p.op.applies, info->iterations);
    EXPECT_TRUE(p.op.host_mode);
    p.expect_lowest_pairs(options);
  }
}

TEST(Davidson, LowestEigenpairsDouble) {
  FBF_REQUIRE_GPU();
  check_converges<double>(20);
}

TEST(Davidson, LowestEigenpairsFloat) {
  FBF_REQUIRE_GPU();
  check_converges<float>(20);
}

// max_subspace = 2 * n_roots collapses at every iteration after the first,
// onto X alone: X_prev does not fit.
TEST(Davidson, ConvergesAtTheSmallestSubspace) {
  FBF_REQUIRE_GPU();
  check_converges<double>(2 * Problem<double>::kRoots);
}

// GD+k: from 3 * n_roots columns a collapse keeps the previous Ritz vectors
// beside the current ones. With the correction the residual itself, a
// collapse onto X alone is a restart of steepest descent, and keeping X_prev
// is what brings the restarted solve back near the unrestarted one: 58
// iterations at 3 * n_roots onto X alone in double against 29 with X_prev and
// 12 unrestarted (229 at 2 * n_roots, where X_prev does not fit).
template <typename T>
void check_collapse_keeps_previous(bool canonical) {
  SCOPED_TRACE(std::format("canonical_orthonormalization {}", canonical));
  const auto iterations = [canonical](std::int32_t max_subspace) {
    fb::DeviceResources res{0};
    Problem<T> p{res};
    p.precondition.identity = true;
    auto options = Problem<T>::converging_options();
    options.canonical_orthonormalization = canonical;
    const auto info = p.solve(max_subspace, options);
    EXPECT_TRUE(info) << max_subspace << ": " << info.error().message;
    if (info) {
      EXPECT_EQ(p.op.applies, info->iterations);
      // Converged only to its 1e-4 relative tolerance, float's residual is
      // above expect_lowest_pairs' bound, which the diagonal preconditioner's
      // solves clear by overshooting it.
      if constexpr (!Problem<T>::kFloat) {
        p.expect_lowest_pairs(options);
      }
    }
    return info ? info->iterations : 0;
  };
  const std::int32_t only_x = iterations(2 * Problem<T>::kRoots);
  const std::int32_t with_previous = iterations(3 * Problem<T>::kRoots);
  const std::int32_t unrestarted = iterations(Problem<T>::kN);
  EXPECT_GT(only_x, 2 * with_previous);
  EXPECT_LE(with_previous, 3 * unrestarted);
}

TEST(Davidson, CollapseKeepsThePreviousRitzVectorsDouble) {
  FBF_REQUIRE_GPU();
  check_collapse_keeps_previous<double>(false);
  check_collapse_keeps_previous<double>(true);
}

TEST(Davidson, CollapseKeepsThePreviousRitzVectorsFloat) {
  FBF_REQUIRE_GPU();
  check_collapse_keeps_previous<float>(false);
  check_collapse_keeps_previous<float>(true);
}

// The subspace lives in the call's own buffers: an operator that grows and
// overwrites the device workspace on every apply changes nothing.
TEST(Davidson, OperatorMayTakeTheDeviceWorkspace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  Problem<double> p{res};
  p.op.scribble = true;
  const auto options = Problem<double>::converging_options();
  const auto info = p.solve(20, options);
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_GT(res.device_workspace(0).num_elements(), std::size_t{1} << 20U);
  p.expect_lowest_pairs(options);
}

// Out of budget after one step: MaxIterationsExceeded, with that step's Ritz
// values, the guess's span being the first 4 coordinates, written out.
TEST(Davidson, MaxIterationsIsAnErrorWithTheLastRitzValues) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  Problem<double> p{res};
  fb::DavidsonOptions options;
  options.max_iterations = 1;
  const auto info = p.solve(20, options);
  ASSERT_FALSE(info);
  EXPECT_EQ(info.error().kind, fb::ErrorKind::MaxIterationsExceeded);
  const auto expected = leading_spectrum(res, p.a, Problem<double>::kN, Problem<double>::kRoots);
  const auto got = to_host(res, p.values.data(), Problem<double>::kRoots);
  for (std::int32_t r = 0; r < Problem<double>::kRoots; ++r) {
    EXPECT_NEAR(got[r], expected[r], 1e-12) << r;
  }
}

// Zero corrections leave nothing to expand by: Divergence at the first step.
TEST(Davidson, ZeroCorrectionsAreDivergence) {
  FBF_REQUIRE_GPU();
  for (const bool canonical : {false, true}) {
    SCOPED_TRACE(std::format("canonical_orthonormalization {}", canonical));
    fb::DeviceResources res{0};
    Problem<double> p{res};
    p.precondition.zero = true;
    fb::DavidsonOptions options;
    options.canonical_orthonormalization = canonical;
    const auto info = p.solve(20, options);
    ASSERT_FALSE(info);
    EXPECT_EQ(info.error().kind, fb::ErrorKind::Divergence);
    EXPECT_EQ(p.op.applies, 1);
  }
}

TEST(Davidson, RejectsABadShape) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  Problem<double> p{res};
  const auto bad = [&](std::int32_t n_roots, std::int32_t max_subspace, std::int32_t guess_count,
                       std::int32_t max_iterations) {
    fb::DavidsonOptions options;
    options.max_iterations = max_iterations;
    const auto info = fb::davidson_solve<double>(res, p.op, p.precondition, Problem<double>::kN,
                                                 n_roots, max_subspace, p.guess.data(), guess_count,
                                                 p.values.data(), p.vectors.data(), options);
    return !info && info.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(bad(0, 8, 4, 10)); // no roots
  EXPECT_TRUE(bad(4, 7, 4, 10)); // max_subspace < 2 * n_roots
  EXPECT_TRUE(bad(4, 8, 3, 10)); // guess_count < n_roots
  EXPECT_TRUE(bad(4, 8, 9, 10)); // guess_count > max_subspace
  EXPECT_TRUE(bad(4, 8, 4, 0));  // no iterations
  EXPECT_EQ(p.op.applies, 0);
}

} // namespace
