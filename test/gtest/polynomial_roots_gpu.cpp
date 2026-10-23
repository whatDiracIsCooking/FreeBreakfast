// GPU cases for recipes.polynomial_roots: Aberth-Ehrlich on a batch of
// polynomials, one warp each, over all four coefficient types.
//
// There is no reference solver here (no LAPACK), so random batches are
// certified a posteriori, in double, against the coefficients as given:
//   - backward error |p(z)| / sum |a_k| |z|^k, conditioning-free, to O(n eps);
//   - the Weierstrass inclusion disks |z - z_i| <= n |W_i|, with
//     W_i = p(z_i) / (a_n prod_{j != i} (z_i - z_j)). Every root lies in their
//     union, and a disk disjoint from the rest holds exactly one, so pairwise
//     disjoint disks prove each root was found once, and each radius bounds
//     that root's forward error.
// Exact cases (roots of unity, a zero root) pin the forward error directly.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

using cd = std::complex<double>;

// Host conversions between each coefficient type and std::complex<double>,
// plus its root type CT and real type R.
template <typename T>
struct Elem;

template <>
struct Elem<float> {
  using CT = wwr::wwrFloatComplex;
  using R = float;
  static constexpr bool kComplex = false;
  static float from(const cd v) { return static_cast<float>(v.real()); }
  static cd to(const float v) { return {v, 0.0}; }
};

template <>
struct Elem<double> {
  using CT = wwr::wwrDoubleComplex;
  using R = double;
  static constexpr bool kComplex = false;
  static double from(const cd v) { return v.real(); }
  static cd to(const double v) { return {v, 0.0}; }
};

template <>
struct Elem<wwr::wwrFloatComplex> {
  using CT = wwr::wwrFloatComplex;
  using R = float;
  static constexpr bool kComplex = true;
  static CT from(const cd v) {
    return CT{static_cast<float>(v.real()), static_cast<float>(v.imag())};
  }
  static cd to(const CT v) { return {v.x, v.y}; }
};

template <>
struct Elem<wwr::wwrDoubleComplex> {
  using CT = wwr::wwrDoubleComplex;
  using R = double;
  static constexpr bool kComplex = true;
  static CT from(const cd v) { return CT{v.real(), v.imag()}; }
  static cd to(const CT v) { return {v.x, v.y}; }
};

template <typename T>
using Root = typename Elem<T>::CT;

template <typename T>
constexpr double kEps = std::numeric_limits<typename Elem<T>::R>::epsilon();

template <typename T>
void upload(fb::DeviceResources &res, fb::Buffers::device<T> &d, const std::vector<T> &host) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

template <typename T>
std::vector<T> download(fb::DeviceResources &res, const fb::Buffers::device<T> &d,
                        std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), count * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

struct Solved {
  std::vector<std::vector<cd>> roots; // per polynomial
  std::vector<std::int32_t> iters;
};

// polynomial_roots over polys (each n+1 ascending coefficients, already in T).
template <typename T>
Solved solve(const std::vector<std::vector<T>> &polys, const typename Elem<T>::R tol,
             const std::int32_t max_iter) {
  using CT = Root<T>;
  const auto batch = static_cast<std::int32_t>(polys.size());
  const auto n = static_cast<std::int32_t>(polys.front().size()) - 1;
  std::vector<T> flat;
  for (const auto &p : polys) {
    flat.insert(flat.end(), p.begin(), p.end());
  }

  fb::DeviceResources res{0};
  fb::Buffers::device<T> d_coeffs(flat.size(), res.device());
  fb::Buffers::device<CT> d_roots(static_cast<std::size_t>(batch) * n, res.device());
  fb::Buffers::device<std::int32_t> d_iters(static_cast<std::size_t>(batch), res.device());
  upload(res, d_coeffs, flat);

  const fb::Status status = fb::polynomial_roots(res, n, batch, d_coeffs.data(), d_roots.data(),
                                                 d_iters.data(), tol, max_iter);
  EXPECT_TRUE(status) << status.error().message;

  const auto roots = download(res, d_roots, static_cast<std::size_t>(batch) * n);
  Solved out;
  out.iters = download(res, d_iters, static_cast<std::size_t>(batch));
  for (std::int32_t b = 0; b < batch; ++b) {
    std::vector<cd> rb;
    for (std::int32_t i = 0; i < n; ++i) {
      rb.push_back(Elem<CT>::to(roots[(static_cast<std::size_t>(b) * n) + i]));
    }
    out.roots.push_back(std::move(rb));
  }
  return out;
}

template <typename T>
std::vector<cd> widen(const std::vector<T> &p) {
  std::vector<cd> a;
  for (const T &c : p) {
    a.push_back(Elem<T>::to(c));
  }
  return a;
}

// p(z), by Horner in double.
cd evaluate(const std::vector<cd> &a, const cd z) {
  cd p = 0.0;
  for (std::size_t k = a.size(); k-- > 0;) {
    p = (p * z) + a[k];
  }
  return p;
}

// |p(z)| / sum |a_k| |z|^k, in double.
double backward_error(const std::vector<cd> &a, const cd z) {
  double bound = 0.0;
  for (std::size_t k = a.size(); k-- > 0;) {
    bound = (bound * std::abs(z)) + std::abs(a[k]);
  }
  return bound == 0.0 ? 0.0 : std::abs(evaluate(a, z)) / bound;
}

// The Weierstrass inclusion radii n |W_i|. Infinite where two estimates
// coincide, so a duplicated root can never pass the disjointness check.
std::vector<double> inclusion_radii(const std::vector<cd> &a, const std::vector<cd> &z) {
  const std::size_t n = z.size();
  std::vector<double> radii(n);
  for (std::size_t i = 0; i < n; ++i) {
    cd w = evaluate(a, z[i]) / a.back();
    for (std::size_t j = 0; j < n; ++j) {
      if (j != i) {
        w /= z[i] - z[j];
      }
    }
    radii[i] = static_cast<double>(n) * std::abs(w);
  }
  return radii;
}

// Greedy nearest-unused matching; the largest distance, relative to
// max(1, |ref|).
double max_match_error(const std::vector<cd> &got, const std::vector<cd> &ref) {
  std::vector<bool> used(ref.size(), false);
  double worst = 0.0;
  for (const cd z : got) {
    std::size_t best = ref.size();
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < ref.size(); ++j) {
      if (!used[j] && std::abs(z - ref[j]) < best_d) {
        best_d = std::abs(z - ref[j]);
        best = j;
      }
    }
    if (best == ref.size()) {
      return std::numeric_limits<double>::infinity(); // a NaN root matched nothing
    }
    used[best] = true;
    worst = std::max(worst, best_d / std::max(1.0, std::abs(ref[best])));
  }
  return worst;
}

// Gaussian coefficients (complex for a complex T), rounded into T.
template <typename T>
std::vector<std::vector<T>> random_polys(const std::int32_t n, const std::int32_t batch,
                                         const std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::normal_distribution<double> nd;
  std::vector<std::vector<T>> polys(batch, std::vector<T>(n + 1));
  for (auto &p : polys) {
    for (auto &a : p) {
      a = Elem<T>::from(Elem<T>::kComplex ? cd{nd(gen), nd(gen)} : cd{nd(gen), 0.0});
    }
  }
  return polys;
}

template <typename T>
class PolynomialRoots : public ::testing::Test {};

using CoefficientTypes =
    ::testing::Types<float, double, wwr::wwrFloatComplex, wwr::wwrDoubleComplex>;
TYPED_TEST_SUITE(PolynomialRoots, CoefficientTypes);

// Random batches up to the full warp (32 or 64 roots), each of 9 polynomials:
// several blocks, the last one partial, and idle lanes below the full width.
TYPED_TEST(PolynomialRoots, RandomBatchesAreCertified) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;
  const double eps = kEps<T>;

  constexpr auto kFull = static_cast<std::int32_t>(fb::kWarpSize);
  for (const std::int32_t n : {1, 2, 3, 7, 16, kFull - 1, kFull}) {
    const auto polys = random_polys<T>(n, 9, 1234U + static_cast<std::uint32_t>(n));
    const auto got = solve<T>(polys, static_cast<typename Elem<T>::R>(eps), 500);
    for (std::size_t b = 0; b < polys.size(); ++b) {
      const auto a = widen(polys[b]);
      const auto &z = got.roots[b];
      const auto ctx = "n=" + std::to_string(n) + " b=" + std::to_string(b);
      EXPECT_GE(got.iters[b], 1) << ctx << ": did not converge";

      double be = 0.0;
      for (const cd zi : z) {
        be = std::max(be, backward_error(a, zi));
      }
      EXPECT_LE(be, 8.0 * n * eps) << ctx << ": backward error";

      const auto radii = inclusion_radii(a, z);
      for (std::size_t i = 0; i < z.size(); ++i) {
        EXPECT_LE(radii[i], 100.0 * n * eps * std::max(1.0, std::abs(z[i])))
            << ctx << " i=" << i << ": inclusion radius";
        for (std::size_t j = i + 1; j < z.size(); ++j) {
          EXPECT_GT(std::abs(z[i] - z[j]), radii[i] + radii[j])
              << ctx << " i=" << i << " j=" << j << ": inclusion disks overlap";
        }
      }
    }
  }
}

// z^n - 1 at the full warp: the n-th roots of unity, exactly.
TYPED_TEST(PolynomialRoots, RootsOfUnity) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;
  constexpr auto kN = static_cast<std::int32_t>(fb::kWarpSize);

  std::vector<T> p(kN + 1, Elem<T>::from(0.0));
  p[0] = Elem<T>::from(-1.0);
  p[kN] = Elem<T>::from(1.0);
  const auto got = solve<T>({p}, static_cast<typename Elem<T>::R>(kEps<T>), 500);
  EXPECT_GE(got.iters[0], 1);

  std::vector<cd> exact;
  for (std::int32_t k = 0; k < kN; ++k) {
    exact.push_back(std::polar(1.0, 2.0 * std::numbers::pi * k / kN));
  }
  EXPECT_LE(max_match_error(got.roots[0], exact), 16.0 * kEps<T>);
}

// z^3 - z: roots 0 and +-1, the zero root exercising the relative-correction
// test where |z| -> 0.
TYPED_TEST(PolynomialRoots, ZeroRoot) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  const std::vector<T> p = {Elem<T>::from(0.0), Elem<T>::from(-1.0), Elem<T>::from(0.0),
                            Elem<T>::from(1.0)};
  const auto got = solve<T>({p}, static_cast<typename Elem<T>::R>(kEps<T>), 500);
  EXPECT_GE(got.iters[0], 1);
  EXPECT_LE(max_match_error(got.roots[0], {0.0, 1.0, -1.0}), 4.0 * kEps<T>);
}

// max_iter too small to settle, and a NaN coefficient: both report -1 in the
// same batch as a polynomial that does settle, and the call still succeeds.
TYPED_TEST(PolynomialRoots, ReportsNonConvergencePerPolynomial) {
  FBF_REQUIRE_GPU();
  using T = TypeParam;

  const auto polys = random_polys<T>(20, 2, 99U);
  const auto capped = solve<T>(polys, static_cast<typename Elem<T>::R>(kEps<T>), 1);
  EXPECT_EQ(capped.iters[0], -1);
  EXPECT_EQ(capped.iters[1], -1);

  auto mixed = polys;
  mixed[1][5] = Elem<T>::from(cd{std::numeric_limits<double>::quiet_NaN(), 0.0});
  const auto got = solve<T>(mixed, static_cast<typename Elem<T>::R>(kEps<T>), 500);
  EXPECT_GE(got.iters[0], 1);
  EXPECT_EQ(got.iters[1], -1);
}

// n == 0 or batch == 0 launches nothing: null pointers would fault otherwise.
TEST(PolynomialRootsArguments, EmptyLaunchesNothing) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const double *coeffs = nullptr;
  wwr::wwrDoubleComplex *roots = nullptr;
  EXPECT_TRUE(fb::polynomial_roots(res, 0, 4, coeffs, roots, nullptr, 1e-14, 10));
  EXPECT_TRUE(fb::polynomial_roots(res, 5, 0, coeffs, roots, nullptr, 1e-14, 10));
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

TEST(PolynomialRootsArguments, RejectsInvalidArguments) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> d_c(8, res.device());
  fb::Buffers::device<wwr::wwrDoubleComplex> d_r(8, res.device());
  fb::Buffers::device<std::int32_t> d_i(1, res.device());
  const auto kind = [&](const std::int32_t n, const std::int32_t batch, const double *c,
                        const double tol, const std::int32_t max_iter) {
    const fb::Status s =
        fb::polynomial_roots(res, n, batch, c, d_r.data(), d_i.data(), tol, max_iter);
    return s ? std::optional<fb::ErrorKind>{} : std::optional{s.error().kind};
  };
  const std::optional bad{fb::ErrorKind::InvalidConfig};
  EXPECT_EQ(kind(-1, 1, d_c.data(), 1e-14, 10), bad);
  EXPECT_EQ(kind(3, -1, d_c.data(), 1e-14, 10), bad);
  EXPECT_EQ(kind(3, 1, d_c.data(), -1.0, 10), bad);
  EXPECT_EQ(kind(3, 1, d_c.data(), std::numeric_limits<double>::quiet_NaN(), 10), bad);
  EXPECT_EQ(kind(3, 1, d_c.data(), 1e-14, -1), bad);
  EXPECT_EQ(kind(3, 1, nullptr, 1e-14, 10), bad);
  // One root per lane: a degree past the warp has no lane for its last root.
  EXPECT_EQ(kind(static_cast<std::int32_t>(fb::kWarpSize) + 1, 1, d_c.data(), 1e-14, 10), bad);
}

} // namespace
