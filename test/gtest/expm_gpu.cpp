// GPU cases for recipes.expm: skew_eig and skew_expm against a skew-Hermitian
// H whose exponential is known in closed form. H = Q B Q^H, with Q a dense
// unitary (a product of two Householder reflectors) and B a canonical form
// whose exponential is cos and sin alone: 2 x 2 rotation generators for a real
// T, the diagonal i phi for a complex one. So exp(tH) = Q exp(tB) Q^H and the
// spectrum of iH are exact on the host, with no series and no reference expm.
// One theta / phi is repeated, so a degenerate eigenspace -- whose basis heevd
// may pick freely -- is exercised too.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

// Odd, so the real case has the lone zero eigenvalue as well as the +-theta
// pairs.
constexpr std::int32_t kN = 37;

template <typename T>
using Real = wwr::ComplexToRealType<T>;
template <typename T>
using Eigvec = wwr::RealToComplexType<Real<T>>;

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

// a b, or a b^H.
HostMatrix multiply(std::int32_t n, const HostMatrix &a, const HostMatrix &b, bool adjoint_b) {
  HostMatrix c(a.size());
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t l = 0; l < n; ++l) {
      const Cd blj = adjoint_b ? std::conj(b[j + (l * n)]) : b[l + (j * n)];
      for (std::int32_t i = 0; i < n; ++i) {
        c[i + (j * n)] += a[i + (l * n)] * blj;
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

// A dense unitary, real for a real T: the product of two reflectors, so no
// entry of Q is structurally zero.
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

// The canonical form's frequencies: theta per 2 x 2 block for a real T,
// phi per diagonal entry for a complex one. Entry 1 repeats entry 0.
template <typename T>
std::vector<double> frequencies(std::int32_t n) {
  const std::int32_t count = wwr::complex_fp<T> ? n : n / 2;
  std::vector<double> f(count);
  for (std::int32_t k = 0; k < count; ++k) {
    f[k] = wwr::complex_fp<T> ? -1.7 + (0.31 * k) : 0.2 + (0.45 * k);
  }
  f[1] = f[0];
  return f;
}

// exp(tB): the phases e^{i phi t}, or the rotations by theta t (and a 1).
template <typename T>
HostMatrix canonical_exp(std::int32_t n, const std::vector<double> &f, double t) {
  HostMatrix b(static_cast<std::size_t>(n) * n);
  if constexpr (wwr::complex_fp<T>) {
    for (std::int32_t k = 0; k < n; ++k) {
      b[k + (k * n)] = std::polar(1.0, f[k] * t);
    }
  } else {
    // [[0, -theta], [theta, 0]] exponentiates to the rotation by theta t.
    for (std::size_t k = 0; k < f.size(); ++k) {
      const auto i = static_cast<std::int32_t>(2 * k);
      const double c = std::cos(f[k] * t);
      const double s = std::sin(f[k] * t);
      b[i + (i * n)] = c;
      b[i + 1 + (i * n)] = s;
      b[i + ((i + 1) * n)] = -s;
      b[i + 1 + ((i + 1) * n)] = c;
    }
    if (n % 2 == 1) {
      b[(n - 1) + ((n - 1) * n)] = 1.0;
    }
  }
  return b;
}

// B: diag(i phi), or the [[0, -theta], [theta, 0]] blocks (and a zero).
template <typename T>
HostMatrix canonical_generator(std::int32_t n, const std::vector<double> &f) {
  HostMatrix b(static_cast<std::size_t>(n) * n);
  if constexpr (wwr::complex_fp<T>) {
    for (std::int32_t k = 0; k < n; ++k) {
      b[k + (k * n)] = {0.0, f[k]};
    }
  } else {
    for (std::size_t k = 0; k < f.size(); ++k) {
      const auto i = static_cast<std::int32_t>(2 * k);
      b[i + 1 + (i * n)] = f[k];
      b[i + ((i + 1) * n)] = -f[k];
    }
  }
  return b;
}

// The eigenvalues of iB (so of iH), ascending: -phi, or +-theta and a zero.
template <typename T>
std::vector<double> spectrum(std::int32_t n, const std::vector<double> &f) {
  std::vector<double> w;
  for (const double x : f) {
    if constexpr (wwr::complex_fp<T>) {
      w.push_back(-x);
    } else {
      w.push_back(x);
      w.push_back(-x);
    }
  }
  if (!wwr::complex_fp<T> && n % 2 == 1) {
    w.push_back(0.0);
  }
  std::ranges::sort(w);
  return w;
}

// Q M Q^H.
HostMatrix conjugate(std::int32_t n, const HostMatrix &q, const HostMatrix &m) {
  return multiply(n, multiply(n, q, m, false), q, true);
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
void upload(fb::DeviceResources &res, fb::Buffers::device<T> &d, const std::vector<T> &host) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

template <typename T>
std::vector<T> download(fb::DeviceResources &res, const fb::Buffers::device<T> &d) {
  std::vector<T> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// The closed-form problem for one T, staged on the device and decomposed.
template <typename T>
struct SkewProblem {
  fb::DeviceResources &res;
  std::int32_t n;
  std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> f = frequencies<T>(n);
  HostMatrix q = dense_unitary<T>(n);
  HostMatrix h_host = conjugate(n, q, canonical_generator<T>(n, f));
  std::vector<T> h_narrow;
  fb::Buffers::device<T> h{nn, res.device()};
  fb::Buffers::device<Eigvec<T>> u{nn, res.device()};
  fb::Buffers::device<Real<T>> w{static_cast<std::size_t>(n), res.device()};
  fb::Buffers::device<T> e{nn, res.device()};

  void stage() {
    for (const Cd z : h_host) {
      h_narrow.push_back(narrow<T>(z));
    }
    upload(res, h, h_narrow);
  }

  // The largest |exp(tH)_ij - (Q exp(tB) Q^H)_ij| over every element.
  double expm_error(double t) {
    const HostMatrix expected = conjugate(n, q, canonical_exp<T>(n, f, t));
    const std::vector<T> got = download(res, e);
    double err = 0.0;
    for (std::size_t k = 0; k < nn; ++k) {
      err = std::max(err, std::abs(widen(got[k]) - expected[k]));
    }
    return err;
  }
};

// Every entry of exp(tH) is at most 1 in modulus. Its error is the
// decomposition's, O(n eps), plus the phase error t * dw, which grows with
// |t| times the largest frequency.
template <typename T>
double tolerance(std::int32_t n, double t, double max_frequency) {
  const double eps = std::numeric_limits<Real<T>>::epsilon();
  return 16.0 * n * eps * (1.0 + (std::abs(t) * max_frequency));
}

template <typename T>
class SkewExpm : public ::testing::Test {};

using SkewTypes = ::testing::Types<float, double, wwr::wwrFloatComplex, wwr::wwrDoubleComplex>;
TYPED_TEST_SUITE(SkewExpm, SkewTypes);

TYPED_TEST(SkewExpm, EigenvaluesAreTheClosedFormSpectrum) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  SkewProblem<T> p{res, kN};
  p.stage();
  ASSERT_TRUE(fb::skew_eig<T>(res, p.n, p.h.data(), p.u.data(), p.w.data()));

  const std::vector<double> expected = spectrum<T>(p.n, p.f);
  const std::vector<Real<T>> got = download(res, p.w);
  ASSERT_EQ(got.size(), expected.size());
  // An eigenvalue's error is O(n eps ||H||), ||H|| being the largest frequency.
  const double max_frequency =
      std::abs(std::ranges::max(p.f, {}, [](double x) { return std::abs(x); }));
  const double tol = tolerance<T>(p.n, 1.0, max_frequency);
  for (std::size_t k = 0; k < got.size(); ++k) {
    EXPECT_NEAR(static_cast<double>(got[k]), expected[k], tol) << "eigenvalue " << k;
  }
  // h is the caller's, and is read, never written.
  EXPECT_EQ(std::memcmp(download(res, p.h).data(), p.h_narrow.data(), p.nn * sizeof(T)), 0);
}

// t = 0 is the identity; a negative t is the inverse direction; the large t
// winds every phase through many turns.
TYPED_TEST(SkewExpm, ExponentialMatchesTheClosedFormAtEveryStep) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  SkewProblem<T> p{res, kN};
  p.stage();
  ASSERT_TRUE(fb::skew_eig<T>(res, p.n, p.h.data(), p.u.data(), p.w.data()));

  const double max_frequency =
      std::abs(std::ranges::max(p.f, {}, [](double x) { return std::abs(x); }));
  for (const double t : {0.0, 0.37, -1.1, 25.0}) {
    ASSERT_TRUE(
        fb::skew_expm<T>(res, p.n, p.u.data(), p.w.data(), static_cast<Real<T>>(t), p.e.data()));
    EXPECT_LE(p.expm_error(t), tolerance<T>(p.n, t, max_frequency)) << "t = " << t;
  }
}

TYPED_TEST(SkewExpm, EmptyOperandSucceedsWithoutTouchingAnything) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  EXPECT_TRUE(fb::skew_eig<T>(res, 0, nullptr, nullptr, nullptr));
  EXPECT_TRUE(fb::skew_expm<T>(res, 0, nullptr, nullptr, Real<T>{1}, nullptr));
  EXPECT_EQ(res.device_workspace(0).num_elements(), 0U);
}

} // namespace
