/// @file
/// @brief Closed-form Cartesian-to-spherical transform for one shell's angular
/// momentum @c l -- the per-l block @c BasisSet::cartesian_to_spherical builds
/// from, giving the 2l+1 real solid harmonics as combinations of the ncart(l)
/// Cartesian monomials in @ref free_breakfast::cartesian_powers order.
///
/// **The closed form.** A degree-@c l real solid harmonic is a known explicit
/// polynomial in @c x,y,z; the coefficient of @c x^lx y^ly z^lz in the real
/// solid harmonic of magnetic quantum number @c m is the Schlegel-Frisch
/// expansion (IJQC 54, 83 (1995); Helgaker, "Molecular Electronic-Structure
/// Theory", Eqs. 6.4.47-6.4.49) -- a finite triple sum of binomial
/// coefficients, computed directly below with no basis construction, no
/// orthogonalization and no linear solve. Columns are ordered by @c m ascending,
/// @c -l..+l, the PySCF/libcint convention (pinned in
/// @c test/gtest/basis_host.cpp); for @c l<=1 the block is the identity (PySCF
/// presents @c p as the bare Cartesian @c (x,y,z), and @c s is trivial),
/// matching the identity block
/// @c cartesian_to_spherical already applies to a @c Cartesian shell.
///
/// **G-orthonormal, so the per-shell scale is a single scalar.** Each column is
/// normalized to unit length under the Gaussian-moment inner product
/// @code
///   <x^a1 y^b1 z^c1, x^a2 y^b2 z^c2> = (a1+a2-1)!! (b1+b2-1)!! (c1+c2-1)!!
/// @endcode
/// (zero if any exponent sum is odd) -- the angular part of the Gaussian
/// self-overlap moment (@ref detail::monomial_gram). The real solid harmonics
/// are mutually orthogonal under this metric by construction, so after that
/// per-column normalization the block @c B satisfies @c B^T G B = I. This is
/// exactly the property @c cartesian_to_spherical relies on: multiplying by its
/// single per-shell scale @c sqrt((2l-1)!!) then reconciles this repo's
/// Cartesian normalization with a unit-normalized spherical AO, yielding a
/// uniform @c x*I diagonal with every cross-@c m term exactly zero.
///
/// **This is a purely angular, normalization-agnostic construction.** Every
/// Cartesian component of a @ref Shell shares the identical radial contraction
/// (the shell stores ONE primitive list, used for every @c (a,b,c)), so the
/// table's coefficients combine Cartesian AOs into genuine real solid harmonics
/// regardless of how those AOs are normalized -- the shared radial factor just
/// rides along. Reconciling this repo's Cartesian normalization convention with
/// a unit-normalized spherical AO is therefore a single overall per-l scalar,
/// applied where the transform is used (@c BasisSet::cartesian_to_spherical),
/// not here.
///
/// **Storage convention.** The returned block is row-major, ncart(l) x (2l+1):
/// element (row, col) lives at @c data[row*(2l+1) + col]; rows run over
/// @ref cartesian_powers(l), columns over @c m = -l..+l.
module;

#include "basis.h"

export module free_breakfast.basis:harmonics;

import :ao;
import std;

export namespace free_breakfast {

/// @brief The closed-form Cartesian-to-spherical transform block for angular
/// momentum @p l (>= 0): ncart(l) x (2l+1), row-major, rows over
/// @ref cartesian_powers(l) and columns the 2l+1 real solid harmonics in
/// @c m = -l..+l (PySCF/libcint) order.
///
/// G-orthonormal under the Gaussian-moment inner product (@c B^T G B = I), so
/// the single @c sqrt((2l-1)!!) scale @c cartesian_to_spherical applies
/// reproduces a uniform diagonal with zero cross-m coupling. For @c l<=1 the
/// block is the identity.
[[nodiscard]] std::vector<double> solid_harmonic_c2s(int l);

} // namespace free_breakfast

namespace free_breakfast::detail {

// (n-1)!! for odd n >= 1 (i.e. n = a1+a2-1 with a1+a2 even and >= 0); n < 0
// means the empty product, 1.0, matching the a1=a2=0 case.
[[nodiscard]] inline double double_factorial_odd(int n) {
  double result = 1.0;
  for (int k = n; k > 1; k -= 2) {
    result *= static_cast<double>(k);
  }
  return result;
}

// The angular Gaussian-moment inner product between two same-degree Cartesian
// monomials -- see this file's doc comment. Zero unless every per-axis exponent
// SUM is even (an odd 1D moment of a centered Gaussian vanishes by parity).
[[nodiscard]] inline double monomial_gram(const std::array<int, 3> &p, const std::array<int, 3> &q) {
  const int sx = p[0] + q[0];
  const int sy = p[1] + q[1];
  const int sz = p[2] + q[2];
  if ((sx % 2) != 0 || (sy % 2) != 0 || (sz % 2) != 0) {
    return 0.0;
  }
  return double_factorial_odd(sx - 1) * double_factorial_odd(sy - 1) * double_factorial_odd(sz - 1);
}

// n! as a double, exact for the small n a basis's l ever reaches.
[[nodiscard]] inline double factorial(int n) {
  double result = 1.0;
  for (int k = 2; k <= n; ++k) {
    result *= static_cast<double>(k);
  }
  return result;
}

// Binomial coefficient C(n, k) as a double, exact for the small n this ever
// sees (n <= l; a basis's l is tiny). Zero outside 0 <= k <= n.
[[nodiscard]] inline double binomial(int n, int k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  double result = 1.0;
  for (int i = 0; i < k; ++i) {
    result = result * static_cast<double>(n - i) / static_cast<double>(i + 1);
  }
  return result;
}

// Coefficients of the (unnormalized) real solid harmonic (l, m) over the
// Cartesian monomials x^lx y^ly z^lz of degree l, indexed by (lx, ly, lz). The
// Schlegel-Frisch closed form (Helgaker Eqs. 6.4.47-6.4.49): a finite triple
// sum over t, u and a magnetic index v (half-integer-offset for m < 0, which is
// why v is tracked as 2*v internally). The absolute prefactor is irrelevant --
// the caller G-normalizes each column -- but it is kept so the bare directions
// already match PySCF's cart2sph to floating point, not merely up to scale.
[[nodiscard]] inline std::map<std::array<int, 3>, double> solid_harmonic_coeffs(int l, int m) {
  const int ma = std::abs(m);
  std::map<std::array<int, 3>, double> coeffs;

  const double prefactor =
      (1.0 / (std::pow(2.0, l) * factorial(l))) *
      std::sqrt(2.0 * factorial(l + ma) * factorial(l - ma) / (m == 0 ? 2.0 : 1.0));

  // two_v0 is 2*v's starting value: 0 for m >= 0, 1 for m < 0 (the half-integer
  // offset of the real-harmonic sine branch).
  const int two_v0 = (m >= 0) ? 0 : 1;
  for (int t = 0; t <= (l - ma) / 2; ++t) {
    for (int u = 0; u <= t; ++u) {
      for (int two_v = two_v0; two_v <= ma; two_v += 2) {
        const int lx = (2 * t) + ma - (2 * u) - two_v;
        const int ly = (2 * u) + two_v;
        const int lz = l - (2 * t) - ma;
        if (lx < 0 || ly < 0 || lz < 0) {
          continue;
        }
        const int sign_power = t + ((two_v - two_v0) / 2);
        const double term = (((sign_power % 2) == 0) ? 1.0 : -1.0) * std::pow(0.25, t) *
                            binomial(l, t) * binomial(l - t, ma + t) * binomial(t, u) *
                            binomial(ma, two_v);
        coeffs[std::array<int, 3>{lx, ly, lz}] += prefactor * term;
      }
    }
  }
  return coeffs;
}

} // namespace free_breakfast::detail

export namespace free_breakfast {

[[nodiscard]] inline std::vector<double> solid_harmonic_c2s(int l) {
  const int ncart = static_cast<int>(cartesian_function_count(l));
  const int nsph = (2 * l) + 1;

  // s and p: PySCF presents these as the bare Cartesian set (s trivially, p as
  // (x, y, z)), so the block is the identity -- the same block
  // cartesian_to_spherical applies to a Cartesian shell.
  if (l <= 1) {
    std::vector<double> identity(static_cast<std::size_t>(ncart) * static_cast<std::size_t>(nsph), 0.0);
    for (int i = 0; i < ncart; ++i) {
      identity[(static_cast<std::size_t>(i) * nsph) + i] = 1.0;
    }
    return identity;
  }

  const std::vector<std::array<int, 3>> powers = cartesian_powers(l);
  std::map<std::array<int, 3>, int> row_of;
  for (int i = 0; i < ncart; ++i) {
    row_of[powers[static_cast<std::size_t>(i)]] = i;
  }

  std::vector<double> block(static_cast<std::size_t>(ncart) * static_cast<std::size_t>(nsph), 0.0);
  for (int col = 0; col < nsph; ++col) {
    const int m = col - l;
    for (const auto &[key, value] : detail::solid_harmonic_coeffs(l, m)) {
      block[(static_cast<std::size_t>(row_of.at(key)) * nsph) + col] = value;
    }
  }

  // G-normalize each column under the Gaussian-moment inner product, so
  // block^T G block = I and cartesian_to_spherical's single sqrt((2l-1)!!)
  // scale yields a uniform diagonal.
  for (int col = 0; col < nsph; ++col) {
    double norm2 = 0.0;
    for (int a = 0; a < ncart; ++a) {
      const double ca = block[(static_cast<std::size_t>(a) * nsph) + col];
      if (ca == 0.0) {
        continue;
      }
      for (int b = 0; b < ncart; ++b) {
        const double cb = block[(static_cast<std::size_t>(b) * nsph) + col];
        if (cb == 0.0) {
          continue;
        }
        norm2 += ca * cb *
                 detail::monomial_gram(powers[static_cast<std::size_t>(a)], powers[static_cast<std::size_t>(b)]);
      }
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (int a = 0; a < ncart; ++a) {
      block[(static_cast<std::size_t>(a) * nsph) + col] *= inv;
    }
  }
  return block;
}

} // namespace free_breakfast
