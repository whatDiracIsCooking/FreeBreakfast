/// @file
/// @brief Closed-form Cartesian-to-spherical transform for one shell's angular
/// momentum @c l -- the per-l block @c BasisSet::spherical_transform_matrix builds
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
/// presents @c p as the bare Cartesian @c (x,y,z), and @c s is trivial).
///
/// **G-orthonormal, so the per-shell scale is a single scalar.** Each column is
/// normalized to unit length under the Gaussian-moment inner product
/// @code
///   <x^a1 y^b1 z^c1, x^a2 y^b2 z^c2> = (a1+a2-1)!! (b1+b2-1)!! (c1+c2-1)!!
/// @endcode
/// (zero if any exponent sum is odd) -- the angular part of the Gaussian
/// self-overlap moment (@ref monomial_gram). The real solid harmonics
/// are mutually orthogonal under this metric by construction, so after that
/// per-column normalization the block @c B satisfies @c B^T G B = I. This is
/// exactly the property @c spherical_transform_matrix relies on: multiplying by its
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
/// applied where the transform is used (@c BasisSet::spherical_transform_matrix),
/// not here.
///
/// **Storage convention.** The returned block is column-major, ncart(l) x
/// (2l+1): element (row, col) lives at @c data[row + col*ncart(l)]; rows run over
/// @ref cartesian_powers(l), columns over @c m = -l..+l.
module;

#include "basis.h"

export module free_breakfast.basis:harmonics;

import :ao;
import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;

export namespace free_breakfast {

/// @brief The closed-form Cartesian-to-spherical transform block for angular
/// momentum @p l (>= 0): ncart(l) x (2l+1), column-major, rows over
/// @ref cartesian_powers(l) and columns the 2l+1 real solid harmonics in
/// @c m = -l..+l (PySCF/libcint) order.
///
/// G-orthonormal under the Gaussian-moment inner product (@c B^T G B = I), so
/// the single @c sqrt((2l-1)!!) scale @c spherical_transform_matrix applies
/// reproduces a uniform diagonal with zero cross-m coupling. For @c l<=1 the
/// block is the identity.
///
/// **Unscaled.** This is the purely angular block, without that
/// @c sqrt((2l-1)!!) factor, so it is not a shell's block of @c U: transform
/// a matrix with @c spherical_transform_matrix, never with this.
[[nodiscard]] std::vector<double> solid_harmonic_block(int32_t l);

} // namespace free_breakfast

// Not exported: helpers of the inline solid_harmonic_block below.
namespace free_breakfast {

// The angular Gaussian-moment inner product between two same-degree Cartesian
// monomials -- see this file's doc comment. Zero unless every per-axis exponent
// SUM is even (an odd 1D moment of a centered Gaussian vanishes by parity).
[[nodiscard]] inline double monomial_gram(const std::array<int32_t, 3> &p,
                                          const std::array<int32_t, 3> &q) {
  const int32_t sx = p[0] + q[0];
  const int32_t sy = p[1] + q[1];
  const int32_t sz = p[2] + q[2];
  if ((sx % 2) != 0 || (sy % 2) != 0 || (sz % 2) != 0) {
    return 0.0;
  }
  return math::double_factorial_odd(sx - 1) * math::double_factorial_odd(sy - 1) *
         math::double_factorial_odd(sz - 1);
}

// Coefficients of the (unnormalized) real solid harmonic (l, m) over the
// Cartesian monomials x^lx y^ly z^lz of degree l, indexed by (lx, ly, lz). The
// Schlegel-Frisch closed form (Helgaker Eqs. 6.4.47-6.4.49): a finite triple
// sum over t, u and a magnetic index v (half-integer-offset for m < 0, which is
// why v is tracked as 2*v internally). The absolute prefactor is irrelevant --
// the caller G-normalizes each column -- but it is kept so the bare directions
// already match PySCF's cart2sph to floating point, not merely up to scale.
[[nodiscard]] inline std::map<std::array<int32_t, 3>, double> solid_harmonic_coeffs(int32_t l,
                                                                                    int32_t m) {
  const int32_t ma = std::abs(m);
  std::map<std::array<int32_t, 3>, double> coeffs;

  const double prefactor =
      (1.0 / (std::pow(2.0, l) * math::factorial(l))) *
      std::sqrt(2.0 * math::factorial(l + ma) * math::factorial(l - ma) / (m == 0 ? 2.0 : 1.0));

  // two_v0 is 2*v's starting value: 0 for m >= 0, 1 for m < 0 (the half-integer
  // offset of the real-harmonic sine branch).
  const int32_t two_v0 = (m >= 0) ? 0 : 1;
  for (int32_t t = 0; t <= (l - ma) / 2; ++t) {
    for (int32_t u = 0; u <= t; ++u) {
      for (int32_t two_v = two_v0; two_v <= ma; two_v += 2) {
        const int32_t lx = (2 * t) + ma - (2 * u) - two_v;
        const int32_t ly = (2 * u) + two_v;
        const int32_t lz = l - (2 * t) - ma;
        if (lx < 0 || ly < 0 || lz < 0) {
          continue;
        }
        const int32_t sign_power = t + ((two_v - two_v0) / 2);
        const double term = (((sign_power % 2) == 0) ? 1.0 : -1.0) * std::pow(0.25, t) *
                            math::binomial(l, t) * math::binomial(l - t, ma + t) *
                            math::binomial(t, u) * math::binomial(ma, two_v);
        coeffs[std::array<int32_t, 3>{lx, ly, lz}] += prefactor * term;
      }
    }
  }
  return coeffs;
}

} // namespace free_breakfast

export namespace free_breakfast {

[[nodiscard]] inline std::vector<double> solid_harmonic_block(int32_t l) {
  const size_t ncart = function_count_cart(l);
  const size_t nsph = function_count_sphe(l);

  // s and p: PySCF presents these as the bare Cartesian set (s trivially, p as
  // (x, y, z)), so the block is the identity -- the same block
  // spherical_transform_matrix applies to a Cartesian shell.
  if (l <= 1) {
    std::vector<double> identity(ncart * nsph, 0.0);
    for (size_t i = 0; i < ncart; ++i) {
      identity[i + (i * ncart)] = 1.0;
    }
    return identity;
  }

  const std::vector<std::array<int32_t, 3>> powers = cartesian_powers(l);
  std::map<std::array<int32_t, 3>, size_t> row_of;
  for (size_t i = 0; i < ncart; ++i) {
    row_of[powers[i]] = i;
  }

  std::vector<double> block(ncart * nsph, 0.0);
  for (size_t col = 0; col < nsph; ++col) {
    // The one signed quantity: columns run m = -l..+l.
    const int32_t m = static_cast<int32_t>(col) - l;
    for (const auto &[key, value] : solid_harmonic_coeffs(l, m)) {
      block[row_of.at(key) + (col * ncart)] = value;
    }
  }

  // G-normalize each column under the Gaussian-moment inner product, so
  // block^T G block = I and spherical_transform_matrix's single sqrt((2l-1)!!)
  // scale yields a uniform diagonal.
  for (size_t col = 0; col < nsph; ++col) {
    double norm2 = 0.0;
    for (size_t a = 0; a < ncart; ++a) {
      const double ca = block[a + (col * ncart)];
      if (ca == 0.0) {
        continue;
      }
      for (size_t b = 0; b < ncart; ++b) {
        const double cb = block[b + (col * ncart)];
        if (cb == 0.0) {
          continue;
        }
        norm2 += ca * cb * monomial_gram(powers[a], powers[b]);
      }
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (size_t a = 0; a < ncart; ++a) {
      block[a + (col * ncart)] *= inv;
    }
  }
  return block;
}

} // namespace free_breakfast
