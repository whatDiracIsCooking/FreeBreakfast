// GPU cases for recipes.cg_unitary, on the Brockett criterion of the paper's
// section 5.1: J(W) = Re tr(W^H R W N), N = diag(1..n), over unitary W
// (orthogonal for a real T). Its maximum puts R's eigenvalues on the diagonal
// of W^H R W in ascending order, the largest paired with the largest weight;
// its minimum puts them descending. R = Q diag(lambda) Q^H with Q a dense
// unitary and lambda chosen, so the optimum is known in closed form and needs
// no reference eigensolver. Further cases cover both line searches, what
// conjugacy buys over steepest descent, a cost that takes the device workspace
// on every call, the error returns, and U(1) / O(1).

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

constexpr std::int32_t kN = 6;

template <typename T>
using Real = wwr::ComplexToRealType<T>;

// The host reference is complex double throughout, real T included.
using Cd = std::complex<double>;
// Column-major n x n on the host, element (i, j) at i + j*n.
using HostMatrix = std::vector<Cd>;

HostMatrix identity(std::int32_t n) {
  HostMatrix a(static_cast<std::size_t>(n) * n);
  for (std::int32_t i = 0; i < n; ++i) {
    a[i + (i * n)] = 1.0;
  }
  return a;
}

// a b, or a^H b.
HostMatrix multiply(std::int32_t n, const HostMatrix &a, const HostMatrix &b, bool adjoint_a) {
  HostMatrix c(a.size());
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t l = 0; l < n; ++l) {
      for (std::int32_t i = 0; i < n; ++i) {
        const Cd ail = adjoint_a ? std::conj(a[l + (i * n)]) : a[i + (l * n)];
        c[i + (j * n)] += ail * b[l + (j * n)];
      }
    }
  }
  return c;
}

// I - 2 v v^H / (v^H v): Hermitian and unitary.
HostMatrix householder(std::int32_t n, const std::vector<Cd> &v) {
  double vv = 0.0;
  for (const Cd x : v) {
    vv += std::norm(x);
  }
  HostMatrix p = identity(n);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = 0; i < n; ++i) {
      p[i + (j * n)] -= 2.0 * v[i] * std::conj(v[j]) / vv;
    }
  }
  return p;
}

// A dense unitary, real for a real T: two reflectors, so no entry of Q is
// structurally zero and W_0 = I is far from every optimum.
template <typename T>
HostMatrix dense_unitary(std::int32_t n) {
  std::vector<Cd> v1(n);
  std::vector<Cd> v2(n);
  for (std::int32_t i = 0; i < n; ++i) {
    const double x = static_cast<double>(i);
    const double im = wwr::complex_fp<T> ? 1.0 : 0.0;
    v1[i] = {std::cos((0.7 * x) + 0.1), im * std::sin(0.4 * x)};
    v2[i] = {std::sin((1.3 * x) + 0.4) + 0.5, im * std::cos(0.9 * x)};
  }
  return multiply(n, householder(n, v1), householder(n, v2), false);
}

// Distinct, unevenly spaced and in no particular order, so the optimum's
// sorting is the solver's doing.
std::vector<double> spectrum(std::int32_t n) {
  std::vector<double> lambda(n);
  for (std::int32_t i = 0; i < n; ++i) {
    lambda[i] = std::sin((2.3 * i) + 0.5) + (0.1 * i);
  }
  return lambda;
}

template <typename T>
T narrow(const Cd z) {
  if constexpr (wwr::complex_fp<T>) {
    return T{static_cast<Real<T>>(z.real()), static_cast<Real<T>>(z.imag())};
  } else {
    return static_cast<T>(z.real());
  }
}

// A device element as complex double. bit_cast, not .x/.y: hipFloatComplex is
// a class, and both vendors' complex types are two packed components.
template <typename T>
Cd widen(const T x) {
  if constexpr (wwr::complex_fp<T>) {
    const auto z = std::bit_cast<std::complex<Real<T>>>(x);
    return {static_cast<double>(z.real()), static_cast<double>(z.imag())};
  } else {
    return static_cast<double>(x);
  }
}

template <typename T>
fb::Buffers::device<T> upload(fb::DeviceResources &res, const HostMatrix &host) {
  std::vector<T> narrowed;
  for (const Cd z : host) {
    narrowed.push_back(narrow<T>(z));
  }
  fb::Buffers::device<T> d(narrowed.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), narrowed.data(), narrowed.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return d;
}

template <typename T>
HostMatrix download(fb::DeviceResources &res, const fb::Buffers::device<T> &d) {
  std::vector<T> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  HostMatrix out;
  for (const T x : host) {
    out.push_back(widen(x));
  }
  return out;
}

// J(W) = tr(W^H R W N) as a cg_unitary cost: Psi = R W N, J = Re dotc(W, Psi).
// The R W product takes the device workspace. With scribble set, each call
// first grows the workspace and fills it with NaN bytes, as a cost taking it
// for scratch of its own would. Notes any call made outside host pointer mode.
template <typename T>
struct Brockett {
  static constexpr std::int32_t kOrder = 2;

  const T *r;
  const T *weights; // N's diagonal, n entries
  bool scribble = false;
  mutable std::int32_t calls = 0;
  mutable bool host_mode = true;

  fb::Status prologue(fb::DeviceResources &res) const {
    ++calls;
    wwr::wwrblasPointerMode_t mode{};
    host_mode =
        host_mode &&
        wwr::wwrblasGetPointerMode(res.blas().get(), &mode) == wwr::WWRBLAS_STATUS_SUCCESS &&
        mode == wwr::WWRBLAS_POINTER_MODE_HOST;
    if (!scribble) {
      return {};
    }
    auto &work = res.device_workspace(static_cast<std::size_t>(calls) << 12U);
    return fb::gpu_status(
        wwr::wwrMemsetAsync(work.data(), 0xFF, work.num_elements(), res.stream()));
  }

  fb::Status euclidean_gradient(fb::DeviceResources &res, std::int32_t n, const T *w,
                                T *psi) const {
    if (auto status = prologue(res); !status) {
      return status;
    }
    const wwr::wwrblasHandle_t blas = res.blas().get();
    auto rw = wwr::extension::reinterpret_buffer_view<T>(
        res.device_workspace(static_cast<std::size_t>(n) * n * sizeof(T)));
    if (auto status = fb::gpu_status(
            wwr::gemm<T, std::int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n,
                                       &fb::kOne<T>, r, n, w, n, &fb::kZero<T>, rw.data(), n));
        !status) {
      return status;
    }
    return fb::gpu_status(wwr::dgmm<T, std::int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, n, n, rw.data(),
                                                     n, weights, 1, psi, n));
  }

  fb::Status value(fb::DeviceResources &res, std::int32_t n, const T *w, const T *psi,
                   T *out) const {
    if (auto status = prologue(res); !status) {
      return status;
    }
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const std::int64_t nn = static_cast<std::int64_t>(n) * n;
    const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                            fb::CreatePolicy<wwr::wwrblasStatus_t>>
        device_result{blas, wwr::WWRBLAS_POINTER_MODE_DEVICE};
    if constexpr (wwr::complex_fp<T>) {
      return fb::gpu_status(wwr::dotc<T, std::int64_t>(blas, nn, w, 1, psi, 1, out));
    } else {
      return fb::gpu_status(wwr::dot<T, std::int64_t>(blas, nn, w, 1, psi, 1, out));
    }
  }
};

static_assert(fb::UnitaryCostFunction<Brockett<double>, double>);
static_assert(fb::UnitaryCostFunction<Brockett<wwr::wwrFloatComplex>, wwr::wwrFloatComplex>);

// One Brockett problem for one T, staged on the device, W_0 = I.
template <typename T>
struct BrockettProblem {
  fb::DeviceResources &res;
  std::int32_t n;
  std::vector<double> lambda = spectrum(n);
  HostMatrix q = dense_unitary<T>(n);
  HostMatrix r_host = multiply(n, multiply(n, q, diagonal(), false), adjoint(q), false);
  fb::Buffers::device<T> r = upload<T>(res, r_host);
  fb::Buffers::device<T> weights = upload<T>(res, weight_vector());
  fb::Buffers::device<T> w = upload<T>(res, identity(n));
  Brockett<T> cost{r.data(), weights.data()};

  HostMatrix diagonal() const {
    HostMatrix d(static_cast<std::size_t>(n) * n);
    for (std::int32_t i = 0; i < n; ++i) {
      d[i + (i * n)] = lambda[i];
    }
    return d;
  }
  HostMatrix adjoint(const HostMatrix &a) const {
    HostMatrix t(a.size());
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        t[j + (i * n)] = std::conj(a[i + (j * n)]);
      }
    }
    return t;
  }
  HostMatrix weight_vector() const {
    HostMatrix v(n);
    for (std::int32_t i = 0; i < n; ++i) {
      v[i] = static_cast<double>(i + 1);
    }
    return v;
  }

  fb::Result<fb::CgInfo> solve(fb::CgDirection direction, const fb::CgOptions<Real<T>> &options) {
    return fb::cg_unitary<T>(res, n, w.data(), cost, direction, options);
  }

  // The largest departure of W^H W from I, and of W^H R W from the optimum's
  // diagonal: lambda ascending for a maximum, descending for a minimum.
  struct Errors {
    double unitarity = 0.0;
    double optimum = 0.0;
  };
  Errors errors(fb::CgDirection direction) {
    const HostMatrix got = download(res, w);
    const HostMatrix wtw = multiply(n, got, got, true);
    const HostMatrix m = multiply(n, got, multiply(n, r_host, got, false), true);
    std::vector<double> sorted = lambda;
    std::ranges::sort(sorted);
    if (direction == fb::CgDirection::Minimize) {
      std::ranges::reverse(sorted);
    }
    Errors e;
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i + (j * n));
        const Cd expected = i == j ? Cd{sorted[i]} : Cd{};
        e.unitarity = std::max(e.unitarity, std::abs(wtw[k] - (i == j ? 1.0 : 0.0)));
        e.optimum = std::max(e.optimum, std::abs(m[k] - expected));
      }
    }
    return e;
  }
};

// The optimum's accuracy is the gradient tolerance's: <G, G> < 1e-20 leaves
// ||G|| ~ 1e-10 in double, 1e-10 leaves ~ 1e-5 in float, each over the
// spectral gaps (>= 0.1 here). Unitarity is the exponential's, O(n eps) per
// step and no worse for the steps taken.
template <typename T>
double optimum_tolerance() {
  return std::same_as<Real<T>, float> ? 5e-3 : 1e-7;
}
template <typename T>
double unitarity_tolerance() {
  return std::same_as<Real<T>, float> ? 1e-4 : 1e-12;
}

template <typename T>
class CgUnitary : public ::testing::Test {};

using CgTypes = ::testing::Types<float, double, wwr::wwrFloatComplex, wwr::wwrDoubleComplex>;
TYPED_TEST_SUITE(CgUnitary, CgTypes);

TYPED_TEST(CgUnitary, PolynomialSearchMaximizesBrockett) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  BrockettProblem<T> p{res, kN};
  const auto info = p.solve(fb::CgDirection::Maximize, {});
  ASSERT_TRUE(info) << info.error().message;
  const auto e = p.errors(fb::CgDirection::Maximize);
  EXPECT_LE(e.unitarity, unitarity_tolerance<T>());
  EXPECT_LE(e.optimum, optimum_tolerance<T>());
  EXPECT_LT(info->gradient_norm_sq,
            static_cast<double>(fb::CgOptions<Real<T>>{}.gradient_tolerance));
  EXPECT_TRUE(p.cost.host_mode);
}

TYPED_TEST(CgUnitary, PolynomialSearchMinimizesBrockett) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  BrockettProblem<T> p{res, kN};
  const auto info = p.solve(fb::CgDirection::Minimize, {});
  ASSERT_TRUE(info) << info.error().message;
  const auto e = p.errors(fb::CgDirection::Minimize);
  EXPECT_LE(e.unitarity, unitarity_tolerance<T>());
  EXPECT_LE(e.optimum, optimum_tolerance<T>());
}

TYPED_TEST(CgUnitary, DftSearchMaximizesBrockett) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  BrockettProblem<T> p{res, kN};
  fb::CgOptions<Real<T>> options;
  options.method = fb::LineSearchMethod::Dft;
  const auto info = p.solve(fb::CgDirection::Maximize, options);
  ASSERT_TRUE(info) << info.error().message;
  const auto e = p.errors(fb::CgDirection::Maximize);
  EXPECT_LE(e.unitarity, unitarity_tolerance<T>());
  EXPECT_LE(e.optimum, optimum_tolerance<T>());
  EXPECT_TRUE(p.cost.host_mode);
}

// U(1) and O(1) have nothing to optimize: the Riemannian gradient of any J is
// zero there, so the start is the answer.
TYPED_TEST(CgUnitary, OrderOneIsConvergedAtTheStart) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  BrockettProblem<T> p{res, 1};
  const auto info = p.solve(fb::CgDirection::Maximize, {});
  ASSERT_TRUE(info) << info.error().message;
  EXPECT_EQ(info->iterations, 0);
  EXPECT_EQ(download(res, p.w), identity(1));
}

// reset_period = 1 discards the conjugate direction every step: Riemannian
// steepest descent, which turns 90 degrees each step and so takes longer.
TEST(CgUnitaryConjugacy, BeatsSteepestDescent) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();

  fb::DeviceResources res{0};
  BrockettProblem<wwr::wwrDoubleComplex> cg{res, kN};
  const auto conjugate = cg.solve(fb::CgDirection::Maximize, {});
  ASSERT_TRUE(conjugate) << conjugate.error().message;

  BrockettProblem<wwr::wwrDoubleComplex> sd{res, kN};
  fb::CgOptions<double> options;
  options.reset_period = 1;
  options.max_iterations = 5000;
  const auto steepest = sd.solve(fb::CgDirection::Maximize, options);
  ASSERT_TRUE(steepest) << steepest.error().message;
  EXPECT_LT(conjugate->iterations, steepest->iterations);
}

// The iterate and the samples live in the call's own buffers: a cost that
// grows and overwrites the device workspace on every call changes nothing.
TEST(CgUnitaryWorkspace, CostMayTakeTheDeviceWorkspace) {
  FBF_REQUIRE_GPU();
  using T = wwr::wwrDoubleComplex;

  fb::DeviceResources res{0};
  for (const auto method : {fb::LineSearchMethod::Polynomial, fb::LineSearchMethod::Dft}) {
    BrockettProblem<T> p{res, kN};
    p.cost.scribble = true;
    fb::CgOptions<double> options;
    options.method = method;
    const auto info = p.solve(fb::CgDirection::Maximize, options);
    ASSERT_TRUE(info) << info.error().message;
    const auto e = p.errors(fb::CgDirection::Maximize);
    EXPECT_LE(e.unitarity, unitarity_tolerance<T>());
    EXPECT_LE(e.optimum, optimum_tolerance<T>());
    EXPECT_GT(p.cost.calls, 0);
  }
}

TEST(CgUnitaryErrors, OutOfIterations) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  BrockettProblem<double> p{res, kN};
  fb::CgOptions<double> options;
  options.max_iterations = 2;
  const auto info = p.solve(fb::CgDirection::Maximize, options);
  ASSERT_FALSE(info);
  EXPECT_EQ(info.error().kind, fb::ErrorKind::MaxIterationsExceeded);
  // Two steps were taken, so W moved, and stayed orthogonal.
  EXPECT_NE(download(res, p.w), identity(kN));
  EXPECT_LE(p.errors(fb::CgDirection::Maximize).unitarity, unitarity_tolerance<double>());
}

TEST(CgUnitaryErrors, RejectsBadArguments) {
  FBF_REQUIRE_GPU();

  fb::DeviceResources res{0};
  BrockettProblem<double> p{res, kN};
  const auto rejected = [&](std::int32_t n, double *w, const fb::CgOptions<double> &options) {
    const auto info = fb::cg_unitary<double>(res, n, w, p.cost, fb::CgDirection::Maximize, options);
    return !info && info.error().kind == fb::ErrorKind::InvalidConfig;
  };
  const auto with = [](auto edit) {
    fb::CgOptions<double> options;
    edit(options);
    return options;
  };
  EXPECT_TRUE(rejected(0, p.w.data(), {}));
  EXPECT_TRUE(rejected(kN, nullptr, {}));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.max_iterations = 0; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.gradient_tolerance = -1.0; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.gradient_tolerance = std::nan(""); })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.polynomial_order = 0; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.polynomial_order = 6; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.dft_samples_per_period = 1; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.dft_periods = 0; })));
  EXPECT_TRUE(rejected(kN, p.w.data(), with([](auto &o) { o.reset_period = -1; })));
  EXPECT_EQ(p.cost.calls, 0);
}

} // namespace
