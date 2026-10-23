// The Pipek-Mezey kernels, their launchers (pipek_mezey_bridge.h), and the
// arithmetic they map thread indices onto: every numerical step of the
// localizer is a small __device__ function below, and each kernel only picks
// its elements. The population tensor is atom-fastest and every kernel puts
// the atom on threadIdx.x, so a warp walks contiguous memory. One source for
// both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Pipek-Mezey maximizes, over orthogonal transformations of the occupied
// space,
//
//     L_p = sum_i sum_A (Q^A_ii)^p,
//     Q^A_ij = 1/2 sum_{mu in A} (C_{mu i} W_{mu j} + C_{mu j} W_{mu i}),  W = S C,
//
// where Q^A is atom A's symmetrized Mulliken population matrix and p is 2
// (Pipek and Mezey, J. Chem. Phys. 90, 4916 (1989)) or 4 (Knizia, J. Chem.
// Theory Comput. 9, 4834 (2013), eq. 4 and Appendix D). The populations of an
// S-orthonormal orbital sum to one over the atoms, so raising them to a power
// and summing rewards putting each orbital's weight on few atoms.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++. It brings
// primitives.h's math:: with it.
#include "block_reduce.cuh"

#include "pipek_mezey_bridge.h"

namespace free_breakfast::device {

namespace {

namespace cg = cooperative_groups;

// The element-wise kernels' block: the atom (or row) rides threadIdx.x.
constexpr uint32_t kElementBlock = kFourWarps;

// A pair's reduction over the atoms. Four warps: a pair's atom count is the
// molecule's, and a block per pair keeps a round's pairs in parallel.
constexpr uint32_t kPairWarps = 4;
constexpr uint32_t kPairThreads = kPairWarps * kWarpSize;

// The functional's one block, the whole of it, for a deterministic fold.
constexpr uint32_t kFunctionalWarps = kMaxThreads / kWarpSize;

struct PmAdd {
  __device__ double operator()(const double a, const double b) const { return a + b; }
};

// ---------------------------------------------------------------------------
// Storage layout
// ---------------------------------------------------------------------------

// Offset of Q^A_ij in the population tensor.
//
// Atom-fastest, ((i * n_occ) + j) * n_atoms + A, so a warp walking the atom
// axis reads contiguous memory. Every kernel puts the atom on threadIdx.x for
// that reason: the build writes coalesced, and the pair reduction and the
// rotation read coalesced.
[[nodiscard]] __device__ constexpr size_t pm_pop_index(const int32_t n_atoms, const int32_t n_occ,
                                                       const int32_t i, const int32_t j,
                                                       const int32_t a) {
  return (((static_cast<size_t>(i) * static_cast<size_t>(n_occ)) + static_cast<size_t>(j)) *
          static_cast<size_t>(n_atoms)) +
         static_cast<size_t>(a);
}

// ---------------------------------------------------------------------------
// The Mulliken populations
// ---------------------------------------------------------------------------

// Q^A_ij, from the AOs [ao_begin, ao_end) atom A owns.
//
// c and w = S C are column-major with leading dimension ld. An atom's
// AOs are one contiguous range: a basis placed over a molecule lays out its
// shells atom by atom. Symmetric in (i, j); the kernel computes the i <= j
// triangle and mirrors it, so the stored tensor is symmetric to the last bit
// rather than to a summation-order rounding.
__device__ __forceinline__ double
pm_population_element(const int32_t i, const int32_t j, const int32_t ld, const double *const c,
                      const double *const w, const int32_t ao_begin, const int32_t ao_end) {
  const size_t col_i = static_cast<size_t>(i) * static_cast<size_t>(ld);
  const size_t col_j = static_cast<size_t>(j) * static_cast<size_t>(ld);
  double acc{0};
  for (int32_t mu = ao_begin; mu < ao_end; ++mu) {
    const auto m = static_cast<size_t>(mu);
    acc += (c[col_i + m] * w[col_j + m]) + (c[col_j + m] * w[col_i + m]);
  }
  return 0.5 * acc;
}

// ---------------------------------------------------------------------------
// The p = 2 rotation of one pair
// ---------------------------------------------------------------------------

// One atom's terms of a pair's p = 2 Jacobi coefficients.
struct PmPairStats {
  double a{0}; // Atom A's term of A_pq
  double b{0}; // Atom A's term of B_pq
};

// Atom A's terms of A_pq and B_pq.
//
// Rotating orbitals p and q by gamma (p' = cos p + sin q, q' = -sin p + cos q)
// changes L_2 by exactly
//
//     dL(gamma) = A_pq (1 - cos 4gamma) + B_pq sin 4gamma,
//     A_pq = sum_A [ Q^A_pq^2 - 1/4 (Q^A_pp - Q^A_qq)^2 ],
//     B_pq = sum_A   Q^A_pq (Q^A_pp - Q^A_qq).
//
// The kernel sums these over the atoms.
__device__ __forceinline__ PmPairStats pm_pair_atom_stats(const double q_pp, const double q_qq,
                                                          const double q_pq) {
  const double d = q_pp - q_qq;
  return PmPairStats{(q_pq * q_pq) - (0.25 * d * d), q_pq * d};
}

// The rotation applied to one pair, and what it gains.
struct PmRotation {
  double cos_gamma{1}; // cos(gamma)
  double sin_gamma{0}; // sin(gamma)
  double gain{0};      // The change in L at gamma; 0 for a pair left alone
};

// The p = 2 rotation of one pair, from its summed A and B.
//
// dL is largest at gamma = atan2(B, -A) / 4, in [-pi/4, pi/4], where it is
// A + sqrt(A^2 + B^2) >= 0: an accepted rotation never lowers L.
//
// A pair whose gain does not exceed gain_threshold is left alone. The test
// is a negated `>`, so a NaN, which finite populations cannot produce but
// which would wreck the whole occupied space if rotated by, also falls through
// to the identity.
__device__ __forceinline__ PmRotation pm_rotation(const double a, const double b,
                                                  const double gain_threshold) {
  const double gain = a + math::sqrt((a * a) + (b * b));
  if (!(gain > gain_threshold)) {
    return PmRotation{1.0, 0.0, 0.0};
  }
  const double gamma = 0.25 * math::atan2(b, -a);
  return PmRotation{math::cos(gamma), math::sin(gamma), gain};
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
// no closed-form maximum. The angle is Knizia's (Appendix D),
// gamma = atan2(B, -A) / 4 over
//
//     A = sum_A [ -Qpp^4 - Qqq^4 + 6 (Qpp^2 + Qqq^2) Qpq^2 + Qpp^3 Qqq + Qpp Qqq^3 ],
//     B = sum_A   4 Qpq (Qpp^3 - Qqq^3),
//
// which "neglects some high-order terms". In this file's sums (below) that is
// B = 2 (B4 + 2 B8) and -A = 2 (A4 + 4 A8): B is half the exact gain's slope
// at gamma = 0 and -A an eighth of its negated curvature there, so for a small
// angle it is a Newton step on the exact quartic. What matters is the slope:
// the angle is zero only where the gain is stationary, so a converged sweep
// is a stationary point of L_4.
//
// The obvious alternative is not. Dropping the 8gamma harmonic and taking the
// maximum of the rest, atan2(B4, A4) / 4, lands nearer each pair's own maximum
// more often, but it is zero where B4 vanishes, not where the slope
// 4 B4 + 8 B8 does, and a sweep driven by it stalls short of the maximum with
// gradients ~1e-3 left on the table.
//
// The angle is only approximate, and for an occasional pair the exact gain
// there is a small loss. So the gain is evaluated exactly at the chosen angle,
// from the same four sums, and a pair that would lose is left alone. That
// keeps every round, and so every sweep, monotonic.

// One atom's terms of a pair's four p = 4 sums.
struct PmPairStats4 {
  double a4{0}; // Atom A's term of the cos 4gamma coefficient
  double b4{0}; // Atom A's term of the sin 4gamma coefficient
  double a8{0}; // Atom A's term of the cos 8gamma coefficient
  double b8{0}; // Atom A's term of the sin 8gamma coefficient
};

// Atom A's terms of the four p = 4 sums for the pair (p, q).
//
// With u = D^2 - Qpq^2 and v = 2 D Qpq (u^2 + v^2 = R^4, so they stand in for
// R^2 cos 2phi and R^2 sin 2phi with no trigonometry), the exact change in
// atom A's part of L_4 is
//
//     dL4_A(gamma) = A4 (cos 4gamma - 1) + B4 sin 4gamma
//                  + A8 (cos 8gamma - 1) + B8 sin 8gamma,
//     A4 = (6 S^2 + R^2) u,   B4 = (6 S^2 + R^2) v,
//     A8 = u^2 / 2 - R^4 / 4,   B8 = u v / 2.
__device__ __forceinline__ PmPairStats4 pm_pair_atom_stats4(const double q_pp, const double q_qq,
                                                            const double q_pq) {
  const double s = 0.5 * (q_pp + q_qq);
  const double d = 0.5 * (q_pp - q_qq);
  const double r2 = (d * d) + (q_pq * q_pq);
  const double u = (d * d) - (q_pq * q_pq);
  const double v = 2.0 * d * q_pq;
  const double w = (6.0 * s * s) + r2;
  return PmPairStats4{w * u, w * v, (0.5 * u * u) - (0.25 * r2 * r2), 0.5 * u * v};
}

// The exact change in L_4 from rotating one pair by gamma, from its
// summed A4, B4, A8 and B8. Nothing is dropped: this is the guard's measure.
__device__ __forceinline__ double pm_gain4(const double gamma, const double a4, const double b4,
                                           const double a8, const double b8) {
  const double c4 = math::cos(4.0 * gamma);
  const double s4 = math::sin(4.0 * gamma);
  const double c8 = math::cos(8.0 * gamma);
  const double s8 = math::sin(8.0 * gamma);
  return (a4 * (c4 - 1.0)) + (b4 * s4) + (a8 * (c8 - 1.0)) + (b8 * s8);
}

// The p = 4 rotation of one pair: Knizia's angle, kept only if its
// exact gain exceeds gain_threshold. The same negated `>` as pm_rotation
// turns a NaN into the identity.
__device__ __forceinline__ PmRotation pm_rotation4(const double a4, const double b4,
                                                   const double a8, const double b8,
                                                   const double gain_threshold) {
  // atan2(B, -A) with Knizia's B and -A halved: the factor cancels.
  const double gamma = 0.25 * math::atan2(b4 + (2.0 * b8), a4 + (4.0 * a8));
  const double gain = pm_gain4(gamma, a4, b4, a8, b8);
  if (!(gain > gain_threshold)) {
    return PmRotation{1.0, 0.0, 0.0};
  }
  return PmRotation{math::cos(gamma), math::sin(gamma), gain};
}

// One diagonal population's term of L_kExponent: its square or its
// fourth power.
template <int32_t kExponent>
  requires(kExponent == 2 || kExponent == 4)
__device__ __forceinline__ double pm_functional_term(const double value) {
  const double sq = value * value;
  if constexpr (kExponent == 2) {
    return sq;
  } else {
    return sq * sq;
  }
}

// ---------------------------------------------------------------------------
// Applying a round
// ---------------------------------------------------------------------------

// One element of Q' = R^T Q R for a round's block-diagonal R.
//
// A round's rotations act on disjoint pairs, so R has 2 x 2 blocks and column
// i has two nonzeros: R[i,i] = cos_i and R[partner(i),i] = sin_i. A pair (p, q)
// carries cos_p = cos_q = cos gamma, sin_p = sin gamma, sin_q = -sin gamma; an
// orbital sitting the round out has cos 1, sin 0 and is its own partner, for
// which the same expression is the identity. Each element of Q' so reads four
// of Q and nothing else, which makes a round one out-of-place kernel.
__device__ __forceinline__ double pm_rotate_population(const double cos_i, const double sin_i,
                                                       const double cos_j, const double sin_j,
                                                       const double q_ij, const double q_i_pj,
                                                       const double q_pi_j, const double q_pi_pj) {
  return (cos_i * cos_j * q_ij) + (cos_i * sin_j * q_i_pj) + (sin_i * cos_j * q_pi_j) +
         (sin_i * sin_j * q_pi_pj);
}

// One element of M' = M R, the same round applied to a column-major
// matrix's columns (the accumulated rotation U).
__device__ __forceinline__ double pm_rotate_column(const double m_col, const double m_partner_col,
                                                   const double cos_c, const double sin_c) {
  return (m_col * cos_c) + (m_partner_col * sin_c);
}

// Q^A_ij over a (atom tiles, n_occ, n_occ) grid. The orbital pair rides y and
// z, so the triangle test is a whole-block early exit and nothing is
// un-ranked. The reads of c and w are not coalesced (neighbouring threads are
// neighbouring atoms, owning disjoint AO ranges); this runs once, and keeping
// the atom on x is what coalesces the kernels that run every round.
__global__ void populations_kernel(const int32_t n_atoms, const int32_t n_occ, const int32_t ld,
                                   const double *const c, const double *const w,
                                   const int32_t *const atom_ao_offsets, double *const pops) {
  const auto i = static_cast<int32_t>(blockIdx.y);
  const auto j = static_cast<int32_t>(blockIdx.z);
  const auto a = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (j < i || a >= n_atoms) {
    return;
  }
  const double value =
      pm_population_element(i, j, ld, c, w, atom_ao_offsets[a], atom_ao_offsets[a + 1]);
  pops[pm_pop_index(n_atoms, n_occ, i, j, a)] = value;
  if (i != j) {
    pops[pm_pop_index(n_atoms, n_occ, j, i, a)] = value;
  }
}

// The identity rotation for an orbital drawn against the padding index.
__device__ void sit_out(const int32_t p, double *const cos_out, double *const sin_out,
                        int32_t *const partner_out) {
  cos_out[p] = 1.0;
  sin_out[p] = 0.0;
  partner_out[p] = p;
}

// One block per pair: the block reduces the pair's sums over the atoms and
// thread 0 turns them into a rotation. Every thread makes every block_reduce
// call (they are collectives), each into its own partials array, so no barrier
// is needed between them.
template <int32_t kExponent>
__global__ void __launch_bounds__(kPairThreads)
    pair_angle_kernel(const int32_t n_atoms, const int32_t n_occ, const double *const pops,
                      const int32_t *const pairs, const double gain_threshold,
                      double *const cos_out, double *const sin_out, int32_t *const partner_out) {
  const size_t base = 2 * static_cast<size_t>(blockIdx.x);
  const int32_t p = pairs[base];
  const int32_t q = pairs[base + 1];
  const auto tid = static_cast<int32_t>(threadIdx.x);

  if (p >= n_occ || q >= n_occ) {
    // The bye: block-uniform, so the early return skips no collective.
    if (tid == 0) {
      sit_out(p < n_occ ? p : q, cos_out, sin_out, partner_out);
    }
    return;
  }

  const double *const diag_p = pops + pm_pop_index(n_atoms, n_occ, p, p, 0);
  const double *const diag_q = pops + pm_pop_index(n_atoms, n_occ, q, q, 0);
  const double *const off_pq = pops + pm_pop_index(n_atoms, n_occ, p, q, 0);
  const auto block = cg::this_thread_block();
  const PmAdd add{};

  PmRotation rot{};
  if constexpr (kExponent == 2) {
    __shared__ double partials_a[kPairWarps];
    __shared__ double partials_b[kPairWarps];
    double sum_a{0};
    double sum_b{0};
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kPairThreads)) {
      const PmPairStats stats = pm_pair_atom_stats(diag_p[a], diag_q[a], off_pq[a]);
      sum_a += stats.a;
      sum_b += stats.b;
    }
    sum_a = block_reduce<kPairWarps>(block, partials_a, sum_a, add);
    sum_b = block_reduce<kPairWarps>(block, partials_b, sum_b, add);
    rot = pm_rotation(sum_a, sum_b, gain_threshold);
  } else {
    __shared__ double partials_a4[kPairWarps];
    __shared__ double partials_b4[kPairWarps];
    __shared__ double partials_a8[kPairWarps];
    __shared__ double partials_b8[kPairWarps];
    double sum_a4{0};
    double sum_b4{0};
    double sum_a8{0};
    double sum_b8{0};
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kPairThreads)) {
      const PmPairStats4 stats = pm_pair_atom_stats4(diag_p[a], diag_q[a], off_pq[a]);
      sum_a4 += stats.a4;
      sum_b4 += stats.b4;
      sum_a8 += stats.a8;
      sum_b8 += stats.b8;
    }
    sum_a4 = block_reduce<kPairWarps>(block, partials_a4, sum_a4, add);
    sum_b4 = block_reduce<kPairWarps>(block, partials_b4, sum_b4, add);
    sum_a8 = block_reduce<kPairWarps>(block, partials_a8, sum_a8, add);
    sum_b8 = block_reduce<kPairWarps>(block, partials_b8, sum_b8, add);
    rot = pm_rotation4(sum_a4, sum_b4, sum_a8, sum_b8, gain_threshold);
  }

  if (tid == 0) {
    cos_out[p] = rot.cos_gamma;
    cos_out[q] = rot.cos_gamma;
    sin_out[p] = rot.sin_gamma;
    sin_out[q] = -rot.sin_gamma;
    partner_out[p] = q;
    partner_out[q] = p;
  }
}

// pops_out = R^T pops_in R, over the population build's grid and triangle, the
// mirrored write keeping the tensor exactly symmetric across a sweep.
__global__ void rotate_populations_kernel(const int32_t n_atoms, const int32_t n_occ,
                                          const double *const pops_in, const double *const cos_v,
                                          const double *const sin_v, const int32_t *const partner,
                                          double *const pops_out) {
  const auto i = static_cast<int32_t>(blockIdx.y);
  const auto j = static_cast<int32_t>(blockIdx.z);
  const auto a = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (j < i || a >= n_atoms) {
    return;
  }
  const int32_t pi = partner[i];
  const int32_t pj = partner[j];
  const double value = pm_rotate_population(cos_v[i], sin_v[i], cos_v[j], sin_v[j],
                                            pops_in[pm_pop_index(n_atoms, n_occ, i, j, a)],
                                            pops_in[pm_pop_index(n_atoms, n_occ, i, pj, a)],
                                            pops_in[pm_pop_index(n_atoms, n_occ, pi, j, a)],
                                            pops_in[pm_pop_index(n_atoms, n_occ, pi, pj, a)]);
  pops_out[pm_pop_index(n_atoms, n_occ, i, j, a)] = value;
  if (i != j) {
    pops_out[pm_pop_index(n_atoms, n_occ, j, i, a)] = value;
  }
}

// out = in R on an n_occ x n_occ matrix: a thread per element, the column on y.
__global__ void rotate_columns_kernel(const int32_t n_occ, const double *const in,
                                      const double *const cos_v, const double *const sin_v,
                                      const int32_t *const partner, double *const out) {
  const auto col = static_cast<int32_t>(blockIdx.y);
  const auto row = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (row >= n_occ) {
    return;
  }
  const auto ld = static_cast<size_t>(n_occ);
  const auto r = static_cast<size_t>(row);
  out[(static_cast<size_t>(col) * ld) + r] =
      pm_rotate_column(in[(static_cast<size_t>(col) * ld) + r],
                       in[(static_cast<size_t>(partner[col]) * ld) + r], cos_v[col], sin_v[col]);
}

// L into out[0], one block: the set is n_occ * n_atoms elements, and a single
// thread-ordered fold makes L, and so the convergence test, bit-reproducible.
// A multi-block sum closed by an atomic would not be.
template <int32_t kExponent>
__global__ void __launch_bounds__(kMaxThreads)
    functional_kernel(const int32_t n_atoms, const int32_t n_occ, const double *const pops,
                      double *const out) {
  __shared__ double partials[kFunctionalWarps];
  const auto tid = static_cast<int32_t>(threadIdx.x);
  double acc{0};
  for (int32_t i = 0; i < n_occ; ++i) {
    const double *const diag = pops + pm_pop_index(n_atoms, n_occ, i, i, 0);
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kMaxThreads)) {
      acc += pm_functional_term<kExponent>(diag[a]);
    }
  }
  acc = block_reduce<kFunctionalWarps>(cg::this_thread_block(), partials, acc, PmAdd{});
  if (tid == 0) {
    out[0] = acc;
  }
}

__global__ void identity_kernel(const int32_t n, double *const matrix) {
  const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
  if (idx >= total) {
    return;
  }
  // Column-major (r, c) sits at c * n + r: the diagonal is every (n + 1)th.
  matrix[idx] = (idx % (static_cast<size_t>(n) + 1) == 0) ? 1.0 : 0.0;
}

// The population build's and the rotation's grid: atom tiles by n_occ by n_occ.
dim3 triangle_grid(const int32_t n_atoms, const int32_t n_occ) {
  return {math::idivup(static_cast<uint32_t>(n_atoms), kElementBlock), static_cast<uint32_t>(n_occ),
          static_cast<uint32_t>(n_occ)};
}

} // namespace

wwr::wwrError_t pm_populations(const wwr::wwrStream_t stream, const int32_t n_atoms,
                               const int32_t n_occ, const int32_t ld, const double *const c,
                               const double *const w, const int32_t *const atom_ao_offsets,
                               double *const pops) {
  populations_kernel<<<triangle_grid(n_atoms, n_occ), kElementBlock, 0, stream>>>(
      n_atoms, n_occ, ld, c, w, atom_ao_offsets, pops);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pm_pair_angles(const wwr::wwrStream_t stream, const int32_t exponent,
                               const int32_t n_atoms, const int32_t n_occ, const double *const pops,
                               const int32_t *const pairs, const int32_t n_pairs,
                               const double gain_threshold, double *const cos_out,
                               double *const sin_out, int32_t *const partner_out) {
  const auto blocks = static_cast<uint32_t>(n_pairs);
  if (exponent == 4) {
    pair_angle_kernel<4><<<blocks, kPairThreads, 0, stream>>>(
        n_atoms, n_occ, pops, pairs, gain_threshold, cos_out, sin_out, partner_out);
  } else {
    pair_angle_kernel<2><<<blocks, kPairThreads, 0, stream>>>(
        n_atoms, n_occ, pops, pairs, gain_threshold, cos_out, sin_out, partner_out);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pm_rotate_populations(const wwr::wwrStream_t stream, const int32_t n_atoms,
                                      const int32_t n_occ, const double *const pops_in,
                                      const double *const cos_v, const double *const sin_v,
                                      const int32_t *const partner, double *const pops_out) {
  rotate_populations_kernel<<<triangle_grid(n_atoms, n_occ), kElementBlock, 0, stream>>>(
      n_atoms, n_occ, pops_in, cos_v, sin_v, partner, pops_out);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pm_rotate_columns(const wwr::wwrStream_t stream, const int32_t n_occ,
                                  const double *const in, const double *const cos_v,
                                  const double *const sin_v, const int32_t *const partner,
                                  double *const out) {
  const dim3 grid{math::idivup(static_cast<uint32_t>(n_occ), kElementBlock),
                  static_cast<uint32_t>(n_occ)};
  rotate_columns_kernel<<<grid, kElementBlock, 0, stream>>>(n_occ, in, cos_v, sin_v, partner, out);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pm_functional(const wwr::wwrStream_t stream, const int32_t exponent,
                              const int32_t n_atoms, const int32_t n_occ, const double *const pops,
                              double *const out) {
  if (exponent == 4) {
    functional_kernel<4><<<1, kMaxThreads, 0, stream>>>(n_atoms, n_occ, pops, out);
  } else {
    functional_kernel<2><<<1, kMaxThreads, 0, stream>>>(n_atoms, n_occ, pops, out);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pm_identity(const wwr::wwrStream_t stream, const int32_t n, double *const matrix) {
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
  const auto blocks = static_cast<uint32_t>(math::idivup(total, size_t{kElementBlock}));
  identity_kernel<<<blocks, kElementBlock, 0, stream>>>(n, matrix);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
