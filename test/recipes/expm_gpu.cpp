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
// H = 0 at the PSB3 case's 124 orbitals, #992's input: rocSOLVER's heevd
// returned NaN eigenvectors with info 0 for it at any order above 65. The
// split-tridiagonal operands of #999 are at the same order.
constexpr std::int32_t kZeroN = 124;

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

// H = 0, a zero orbital step: iH = I diag(0) I^H and exp(t 0) = I, each
// exactly, whatever the vendor's heevd makes of it.
TYPED_TEST(SkewExpm, ZeroOperandIsExactlyTheIdentity) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  fb::DeviceResources res{0};
  const auto nn = static_cast<std::size_t>(kZeroN) * kZeroN;
  fb::Buffers::device<T> h{nn, res.device()};
  fb::Buffers::device<Eigvec<T>> u{nn, res.device()};
  fb::Buffers::device<Real<T>> w{static_cast<std::size_t>(kZeroN), res.device()};
  fb::Buffers::device<T> e{nn, res.device()};
  upload(res, h, std::vector<T>(nn, narrow<T>(0.0)));
  ASSERT_TRUE(fb::skew_eig<T>(res, kZeroN, h.data(), u.data(), w.data()));

  const HostMatrix one = identity(kZeroN);
  const std::vector<Eigvec<T>> got_u = download(res, u);
  for (std::size_t k = 0; k < nn; ++k) {
    ASSERT_EQ(widen(got_u[k]), one[k]) << "U element " << k;
  }
  for (const Real<T> x : download(res, w)) {
    ASSERT_EQ(x, Real<T>{0});
  }
  for (const double t : {0.37, -1.1}) {
    ASSERT_TRUE(
        fb::skew_expm<T>(res, kZeroN, u.data(), w.data(), static_cast<Real<T>>(t), e.data()));
    const std::vector<T> got_e = download(res, e);
    for (std::size_t k = 0; k < nn; ++k) {
      ASSERT_EQ(widen(got_e[k]), one[k]) << "t = " << t << ", element " << k;
    }
  }
}

// One #999 operand: H, and its 2-norm in closed form.
struct SplitOperand {
  const char *name;
  HostMatrix h;
  double norm;
};

double largest_frequency(const std::vector<double> &f) {
  return std::abs(std::ranges::max(f, {}, [](double x) { return std::abs(x); }));
}

// Nonzero skew-Hermitian H whose iH reduces to a tridiagonal that splits, of
// the kinds rocSOLVER's heevd (ROCm 7.2.4) returned info 0 and NaN
// eigenvectors for at order 124 (#999's probes): the canonical form itself (a real
// diagonal iH for a complex T, 2 x 2 blocks for a real one); one nonzero
// pair; a dense leading half beside a zero one; and a dense H scaled down to
// 1e-200 (1e-30 for a float T, inside its normal range).
//
// That last one passes only because skew_eig scales iH by a power of two.
// Unscaled, both vendors miss this bar: rocSOLVER returns U = I, w = 0
// (#999), and cuSOLVER's heevd leaves U^H U - I at 8e-5 for a real double T
// (#1004).
template <typename T>
std::vector<SplitOperand> split_operands(std::int32_t n) {
  std::vector<SplitOperand> cases;
  const std::vector<double> f = frequencies<T>(n);
  cases.push_back({"canonical form", canonical_generator<T>(n, f), largest_frequency(f)});

  HostMatrix pair(static_cast<std::size_t>(n) * n);
  pair[1] = 1.0;
  pair[n] = -1.0;
  cases.push_back({"one pair", pair, 1.0});

  const std::int32_t m = n / 2;
  const std::vector<double> fm = frequencies<T>(m);
  const HostMatrix block = conjugate(m, dense_unitary<T>(m), canonical_generator<T>(m, fm));
  HostMatrix half(static_cast<std::size_t>(n) * n);
  for (std::int32_t j = 0; j < m; ++j) {
    for (std::int32_t i = 0; i < m; ++i) {
      half[i + (j * n)] = block[i + (j * m)];
    }
  }
  cases.push_back({"zero half", half, largest_frequency(fm)});

  const double scale = std::is_same_v<Real<T>, float> ? 1e-30 : 1e-200;
  HostMatrix tiny = conjugate(n, dense_unitary<T>(n), canonical_generator<T>(n, f));
  for (Cd &z : tiny) {
    z *= scale;
  }
  cases.push_back({"scaled to the bottom of the range", tiny, scale * largest_frequency(f)});
  return cases;
}

// The bar is heevd's backward error, n eps ||H||_2 (rounding analysis, not a
// fit to a run): U^H U = I to n eps, and iH U = U diag(w) to n eps ||H||_2,
// each per element, which the matrix 2-norm bounds. Finiteness is its own
// check: std::max would pass a NaN over.
TYPED_TEST(SkewExpm, SplitTridiagonalOperandsDecomposeToRounding) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;
  constexpr std::int32_t n = kZeroN;
  const auto nn = static_cast<std::size_t>(n) * n;
  const double eps = std::numeric_limits<Real<T>>::epsilon();

  fb::DeviceResources res{0};
  fb::Buffers::device<T> h{nn, res.device()};
  fb::Buffers::device<Eigvec<T>> u{nn, res.device()};
  fb::Buffers::device<Real<T>> w{static_cast<std::size_t>(n), res.device()};
  for (const SplitOperand &c : split_operands<T>(n)) {
    SCOPED_TRACE(c.name);
    std::vector<T> h_narrow;
    for (const Cd z : c.h) {
      h_narrow.push_back(narrow<T>(z));
    }
    upload(res, h, h_narrow);
    ASSERT_TRUE(fb::skew_eig<T>(res, n, h.data(), u.data(), w.data()));

    HostMatrix uh;
    for (const Eigvec<T> x : download(res, u)) {
      uh.push_back(widen(x));
    }
    const std::vector<Real<T>> got_w = download(res, w);
    const auto bad_u = std::ranges::count_if(
        uh, [](Cd z) { return !std::isfinite(z.real()) || !std::isfinite(z.imag()); });
    const auto bad_w = std::ranges::count_if(got_w, [](Real<T> x) { return !std::isfinite(x); });
    EXPECT_EQ(bad_u, 0) << "non-finite elements of U";
    EXPECT_EQ(bad_w, 0) << "non-finite eigenvalues";
    if (bad_u != 0 || bad_w != 0) {
      continue;
    }

    // The Hermitian iH heevd sees: h's lower triangle, as passed.
    HostMatrix ih(nn);
    for (std::int32_t j = 0; j < n; ++j) {
      ih[j + (j * n)] = (Cd{0.0, 1.0} * widen(h_narrow[j + (j * n)])).real();
      for (std::int32_t i = j + 1; i < n; ++i) {
        ih[i + (j * n)] = Cd{0.0, 1.0} * widen(h_narrow[i + (j * n)]);
        ih[j + (i * n)] = std::conj(ih[i + (j * n)]);
      }
    }
    const HostMatrix hu = multiply(n, ih, uh, false);
    double orth = 0.0;
    double resid = 0.0;
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < n; ++i) {
        Cd gram = i == j ? -1.0 : 0.0;
        for (std::int32_t k = 0; k < n; ++k) {
          gram += std::conj(uh[k + (i * n)]) * uh[k + (j * n)];
        }
        orth = std::max(orth, std::abs(gram));
        const std::size_t ij = i + (static_cast<std::size_t>(j) * n);
        resid = std::max(resid, std::abs(hu[ij] - (uh[ij] * static_cast<double>(got_w[j]))));
      }
    }
    EXPECT_LE(orth, n * eps);
    EXPECT_LE(resid, n * eps * c.norm);
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
