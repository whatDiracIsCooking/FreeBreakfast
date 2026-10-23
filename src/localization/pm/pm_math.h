/// @file
/// @brief The Pipek-Mezey arithmetic, one element at a time, written once for
/// the kernels and the host tests.
///
/// Every numerical step of the localizer is a small function here, and
/// pm_kernels.cu only maps thread indices onto them. The math:: functions
/// come from primitives.h and FBF_PM_HOST_DEVICE is its FBF_HOST_DEVICE under a
/// name of its own (primitives.h #undefs its macro), so in a host compile this
/// is plain C++: test/gtest/pipek_mezey_host.cpp drives the same arithmetic with no device, and
/// checks the algebra (the functional rising, the rotation angle, the schedule)
/// there. The GPU test then only confirms the kernels index it correctly.
///
/// Being a header a .cu includes, everything here is public (docs/style.md,
/// "Headers shared with device code"); each name is part of the localizer's
/// documented arithmetic, not a helper.
///
/// ## The functional
///
/// Pipek-Mezey maximizes, over orthogonal transformations of the occupied
/// space,
///
///     L_p = sum_i sum_A (Q^A_ii)^p,
///     Q^A_ij = 1/2 sum_{mu in A} (C_{mu i} W_{mu j} + C_{mu j} W_{mu i}),  W = S C,
///
/// where Q^A is atom A's symmetrized Mulliken population matrix and p is 2
/// (Pipek and Mezey, J. Chem. Phys. 90, 4916 (1989)) or 4 (Knizia, J. Chem.
/// Theory Comput. 9, 4834 (2013), eq. 4 and Appendix D). The populations of an
/// S-orthonormal orbital sum to one over the atoms, so raising them to a power
/// and summing rewards putting each orbital's weight on few atoms.

#pragma once

// Standard headers, then primitives.h, which orders its own ahead of the vendor
// runtime header (under HIP a libc++ header must not follow it).
#include <type_traits>

#include "primitives.h"

// __host__ __device__ __forceinline__ in a device pass, where primitives.h has
// brought in the runtime header that defines __forceinline__; nothing in a host
// compile. #undef'd at the end: this header is included as text.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define FBF_PM_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define FBF_PM_HOST_DEVICE
#endif

namespace free_breakfast {

// ---------------------------------------------------------------------------
// Storage layout
// ---------------------------------------------------------------------------

/// @brief Offset of Q^A_ij in the population tensor.
///
/// Atom-fastest, ((i * n_occ) + j) * n_atoms + A, so a warp walking the atom
/// axis reads contiguous memory. Every kernel puts the atom on threadIdx.x for
/// that reason: the build writes coalesced, and the pair reduction and the
/// rotation read coalesced.
///
/// constexpr for its implied inline: in a host compile FBF_PM_HOST_DEVICE is
/// empty, and a plain function defined in a header would then be defined once
/// per includer.
[[nodiscard]] constexpr FBF_PM_HOST_DEVICE size_t pm_pop_index(const int32_t n_atoms,
                                                               const int32_t n_occ, const int32_t i,
                                                               const int32_t j, const int32_t a) {
  return (((static_cast<size_t>(i) * static_cast<size_t>(n_occ)) + static_cast<size_t>(j)) *
          static_cast<size_t>(n_atoms)) +
         static_cast<size_t>(a);
}

// ---------------------------------------------------------------------------
// The schedule
// ---------------------------------------------------------------------------

/// @brief Fill a round-robin ("circle method") tournament over @p m indices.
///
/// A sweep visits each of the m(m-1)/2 orbital pairs once, and keeps the pairs
/// of each round disjoint, so a whole round applies as one block-diagonal
/// rotation. That is a round-robin tournament: m-1 rounds of m/2 disjoint
/// pairs, index m-1 the fixed pivot and the rest rotating past it.
///
/// @p m is even and at least 2. An odd occupied count is padded with the dummy
/// index n_occ, which is then the pivot, so in each round exactly one real
/// orbital draws it and sits the round out.
///
/// @p pairs holds (m - 1) * m entries: round r starts at pairs + r * m and is
/// m/2 consecutive (first, second) pairs. Host only: the driver builds the
/// schedule once and uploads it.
inline void pm_fill_round_robin(const int32_t m, int32_t *const pairs) {
  const int32_t pivot = m - 1;
  for (int32_t r = 0; r < pivot; ++r) {
    int32_t *slot = pairs + (static_cast<size_t>(r) * static_cast<size_t>(m));
    // Pair 0 draws the pivot against the round number; the rest fan out
    // around it, one step further apart per pair.
    slot[0] = pivot;
    slot[1] = r;
    for (int32_t i = 1; i < m / 2; ++i) {
      slot += 2;
      slot[0] = (r + i) % pivot;
      slot[1] = (((r - i) % pivot) + pivot) % pivot;
    }
  }
}

// ---------------------------------------------------------------------------
// The Mulliken populations
// ---------------------------------------------------------------------------

/// @brief Q^A_ij, from the AOs [ao_begin, ao_end) atom A owns.
///
/// @p c and @p w = S C are column-major with leading dimension @p ld. An atom's
/// AOs are one contiguous range: a basis placed over a molecule lays out its
/// shells atom by atom. Symmetric in (i, j); the kernel computes the i <= j
/// triangle and mirrors it, so the stored tensor is symmetric to the last bit
/// rather than to a summation-order rounding.
template <typename T>
FBF_PM_HOST_DEVICE T pm_population_element(const int32_t i, const int32_t j, const int32_t ld,
                                           const T *const c, const T *const w,
                                           const int32_t ao_begin, const int32_t ao_end) {
  const size_t col_i = static_cast<size_t>(i) * static_cast<size_t>(ld);
  const size_t col_j = static_cast<size_t>(j) * static_cast<size_t>(ld);
  T acc{0};
  for (int32_t mu = ao_begin; mu < ao_end; ++mu) {
    const auto m = static_cast<size_t>(mu);
    acc += (c[col_i + m] * w[col_j + m]) + (c[col_j + m] * w[col_i + m]);
  }
  return T{0.5} * acc;
}

// ---------------------------------------------------------------------------
// The p = 2 rotation of one pair
// ---------------------------------------------------------------------------

/// @brief One atom's terms of a pair's p = 2 Jacobi coefficients.
template <typename T>
struct PmPairStats {
  T a{0}; ///< Atom A's term of A_pq
  T b{0}; ///< Atom A's term of B_pq
};

/// @brief Atom A's terms of A_pq and B_pq.
///
/// Rotating orbitals p and q by gamma (p' = cos p + sin q, q' = -sin p + cos q)
/// changes L_2 by exactly
///
///     dL(gamma) = A_pq (1 - cos 4gamma) + B_pq sin 4gamma,
///     A_pq = sum_A [ Q^A_pq^2 - 1/4 (Q^A_pp - Q^A_qq)^2 ],
///     B_pq = sum_A   Q^A_pq (Q^A_pp - Q^A_qq).
///
/// The kernel sums these over the atoms.
template <typename T>
FBF_PM_HOST_DEVICE PmPairStats<T> pm_pair_atom_stats(const T q_pp, const T q_qq, const T q_pq) {
  const T d = q_pp - q_qq;
  return PmPairStats<T>{(q_pq * q_pq) - (T{0.25} * d * d), q_pq * d};
}

/// @brief The rotation applied to one pair, and what it gains.
template <typename T>
struct PmRotation {
  T cos_gamma{1}; ///< cos(gamma)
  T sin_gamma{0}; ///< sin(gamma)
  T gain{0};      ///< The change in L at gamma; 0 for a pair left alone
};

/// @brief The p = 2 rotation of one pair, from its summed A and B.
///
/// dL is largest at gamma = atan2(B, -A) / 4, in [-pi/4, pi/4], where it is
/// A + sqrt(A^2 + B^2) >= 0: an accepted rotation never lowers L.
///
/// A pair whose gain does not exceed @p gain_threshold is left alone. The test
/// is a negated `>`, so a NaN, which finite populations cannot produce but
/// which would wreck the whole occupied space if rotated by, also falls through
/// to the identity.
template <typename T>
FBF_PM_HOST_DEVICE PmRotation<T> pm_rotation(const T a, const T b, const T gain_threshold) {
  const T gain = a + math::sqrt((a * a) + (b * b));
  if (!(gain > gain_threshold)) {
    return PmRotation<T>{T{1}, T{0}, T{0}};
  }
  const T gamma = T{0.25} * math::atan2(b, -a);
  return PmRotation<T>{math::cos(gamma), math::sin(gamma), gain};
}

// ---------------------------------------------------------------------------
// The p = 4 rotation of one pair
// ---------------------------------------------------------------------------
//
// With S = (Qpp + Qqq)/2, D = (Qpp - Qqq)/2 and R^2 = D^2 + Qpq^2 per atom,
// rotating the pair by gamma gives
//
//     (Qpp')^4 + (Qqq')^4 = c0 + c4 cos(4gamma - 2phi) + c8 cos(8gamma - 4phi),
//     c4 = R^2 (6 S^2 + R^2),   c8 = R^4 / 4.
//
// Unlike p = 2 this has an 8gamma harmonic as well as a 4gamma one, so there is
// no closed-form maximum. Knizia's angle drops the 8gamma term and takes the
// maximum of the rest, gamma = atan2(B4, A4) / 4. The sign differs from p = 2's
// atan2(B, -A) because L_4's gamma-dependent part is written here as
// A4 cos 4gamma + B4 sin 4gamma, with no (1 - cos) flip; the host test checks
// the angle against a brute-force scan of the exact quartic.
//
// That angle is only approximate, and for an occasional pair the exact gain
// there is a small loss. So the gain is evaluated exactly at the chosen angle,
// from two more sums the same per-atom pass yields, and a pair that would lose
// is left alone. That keeps every round, and so every sweep, monotonic.

/// @brief One atom's terms of a pair's four p = 4 sums.
template <typename T>
struct PmPairStats4 {
  T a4{0}; ///< Atom A's term of the cos 4gamma coefficient
  T b4{0}; ///< Atom A's term of the sin 4gamma coefficient
  T a8{0}; ///< Atom A's term of the cos 8gamma coefficient
  T b8{0}; ///< Atom A's term of the sin 8gamma coefficient
};

/// @brief Atom A's terms of the four p = 4 sums for the pair (p, q).
///
/// With u = D^2 - Qpq^2 and v = 2 D Qpq (u^2 + v^2 = R^4, so they stand in for
/// R^2 cos 2phi and R^2 sin 2phi with no trigonometry), the exact change in
/// atom A's part of L_4 is
///
///     dL4_A(gamma) = A4 (cos 4gamma - 1) + B4 sin 4gamma
///                  + A8 (cos 8gamma - 1) + B8 sin 8gamma,
///     A4 = (6 S^2 + R^2) u,   B4 = (6 S^2 + R^2) v,
///     A8 = u^2 / 2 - R^4 / 4,   B8 = u v / 2.
template <typename T>
FBF_PM_HOST_DEVICE PmPairStats4<T> pm_pair_atom_stats4(const T q_pp, const T q_qq, const T q_pq) {
  const T s = T{0.5} * (q_pp + q_qq);
  const T d = T{0.5} * (q_pp - q_qq);
  const T r2 = (d * d) + (q_pq * q_pq);
  const T u = (d * d) - (q_pq * q_pq);
  const T v = T{2} * d * q_pq;
  const T w = (T{6} * s * s) + r2;
  return PmPairStats4<T>{w * u, w * v, (T{0.5} * u * u) - (T{0.25} * r2 * r2), T{0.5} * u * v};
}

/// @brief The exact change in L_4 from rotating one pair by @p gamma, from its
/// summed A4, B4, A8 and B8. Nothing is dropped: this is the guard's measure.
template <typename T>
FBF_PM_HOST_DEVICE T pm_gain4(const T gamma, const T a4, const T b4, const T a8, const T b8) {
  const T c4 = math::cos(T{4} * gamma);
  const T s4 = math::sin(T{4} * gamma);
  const T c8 = math::cos(T{8} * gamma);
  const T s8 = math::sin(T{8} * gamma);
  return (a4 * (c4 - T{1})) + (b4 * s4) + (a8 * (c8 - T{1})) + (b8 * s8);
}

/// @brief The p = 4 rotation of one pair: Knizia's angle, kept only if its
/// exact gain exceeds @p gain_threshold. The same negated `>` as pm_rotation
/// turns a NaN into the identity.
template <typename T>
FBF_PM_HOST_DEVICE PmRotation<T> pm_rotation4(const T a4, const T b4, const T a8, const T b8,
                                              const T gain_threshold) {
  const T gamma = T{0.25} * math::atan2(b4, a4);
  const T gain = pm_gain4(gamma, a4, b4, a8, b8);
  if (!(gain > gain_threshold)) {
    return PmRotation<T>{T{1}, T{0}, T{0}};
  }
  return PmRotation<T>{math::cos(gamma), math::sin(gamma), gain};
}

/// @brief One diagonal population's term of L_@p kExponent: its square or its
/// fourth power.
template <int32_t kExponent, typename T>
  requires(kExponent == 2 || kExponent == 4)
FBF_PM_HOST_DEVICE T pm_functional_term(const T value) {
  const T sq = value * value;
  if constexpr (kExponent == 2) {
    return sq;
  } else {
    return sq * sq;
  }
}

// ---------------------------------------------------------------------------
// Applying a round
// ---------------------------------------------------------------------------

/// @brief One element of Q' = R^T Q R for a round's block-diagonal R.
///
/// A round's rotations act on disjoint pairs, so R has 2 x 2 blocks and column
/// i has two nonzeros: R[i,i] = cos_i and R[partner(i),i] = sin_i. A pair (p, q)
/// carries cos_p = cos_q = cos gamma, sin_p = sin gamma, sin_q = -sin gamma; an
/// orbital sitting the round out has cos 1, sin 0 and is its own partner, for
/// which the same expression is the identity. Each element of Q' so reads four
/// of Q and nothing else, which makes a round one out-of-place kernel.
template <typename T>
FBF_PM_HOST_DEVICE T pm_rotate_population(const T cos_i, const T sin_i, const T cos_j,
                                          const T sin_j, const T q_ij, const T q_i_pj,
                                          const T q_pi_j, const T q_pi_pj) {
  return (cos_i * cos_j * q_ij) + (cos_i * sin_j * q_i_pj) + (sin_i * cos_j * q_pi_j) +
         (sin_i * sin_j * q_pi_pj);
}

/// @brief One element of M' = M R, the same round applied to a column-major
/// matrix's columns (the accumulated rotation U).
template <typename T>
FBF_PM_HOST_DEVICE T pm_rotate_column(const T m_col, const T m_partner_col, const T cos_c,
                                      const T sin_c) {
  return (m_col * cos_c) + (m_partner_col * sin_c);
}

} // namespace free_breakfast

#undef FBF_PM_HOST_DEVICE
