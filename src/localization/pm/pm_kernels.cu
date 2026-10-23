// The Pipek-Mezey kernels and their launchers (pm_bridge.h). The arithmetic is
// pm_math.h's; each kernel maps thread indices onto it. The population tensor
// is atom-fastest and every kernel puts the atom on threadIdx.x, so a warp
// walks contiguous memory. One source for both backends -- nvcc under CUDA,
// clang -x hip under HIP.

// block_reduce.cuh first: it orders its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++.
#include "block_reduce.cuh"

#include "pm_math.h"

#include "pm_bridge.h"

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
  template <typename T>
  __device__ T operator()(const T a, const T b) const {
    return a + b;
  }
};

// Q^A_ij over a (atom tiles, n_occ, n_occ) grid. The orbital pair rides y and
// z, so the triangle test is a whole-block early exit and nothing is
// un-ranked. The reads of c and w are not coalesced (neighbouring threads are
// neighbouring atoms, owning disjoint AO ranges); this runs once, and keeping
// the atom on x is what coalesces the kernels that run every round.
template <typename T>
__global__ void populations_kernel(const int32_t n_atoms, const int32_t n_occ, const int32_t ld,
                                   const T *const c, const T *const w,
                                   const int32_t *const atom_ao_offsets, T *const pops) {
  const auto i = static_cast<int32_t>(blockIdx.y);
  const auto j = static_cast<int32_t>(blockIdx.z);
  const auto a = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (j < i || a >= n_atoms) {
    return;
  }
  const T value = pm_population_element(i, j, ld, c, w, atom_ao_offsets[a], atom_ao_offsets[a + 1]);
  pops[pm_pop_index(n_atoms, n_occ, i, j, a)] = value;
  if (i != j) {
    pops[pm_pop_index(n_atoms, n_occ, j, i, a)] = value;
  }
}

// The identity rotation for an orbital drawn against the padding index.
template <typename T>
__device__ void sit_out(const int32_t p, T *const cos_out, T *const sin_out,
                        int32_t *const partner_out) {
  cos_out[p] = T{1};
  sin_out[p] = T{0};
  partner_out[p] = p;
}

// One block per pair: the block reduces the pair's sums over the atoms and
// thread 0 turns them into a rotation. Every thread makes every block_reduce
// call (they are collectives), each into its own partials array, so no barrier
// is needed between them.
template <int32_t kExponent, typename T>
__global__ void __launch_bounds__(kPairThreads)
    pair_angle_kernel(const int32_t n_atoms, const int32_t n_occ, const T *const pops,
                      const int32_t *const pairs, const T gain_threshold, T *const cos_out,
                      T *const sin_out, int32_t *const partner_out) {
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

  const T *const diag_p = pops + pm_pop_index(n_atoms, n_occ, p, p, 0);
  const T *const diag_q = pops + pm_pop_index(n_atoms, n_occ, q, q, 0);
  const T *const off_pq = pops + pm_pop_index(n_atoms, n_occ, p, q, 0);
  const auto block = cg::this_thread_block();
  const PmAdd add{};

  PmRotation<T> rot{};
  if constexpr (kExponent == 2) {
    __shared__ T partials_a[kPairWarps];
    __shared__ T partials_b[kPairWarps];
    T sum_a{0};
    T sum_b{0};
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kPairThreads)) {
      const PmPairStats<T> stats = pm_pair_atom_stats(diag_p[a], diag_q[a], off_pq[a]);
      sum_a += stats.a;
      sum_b += stats.b;
    }
    sum_a = block_reduce<kPairWarps>(block, partials_a, sum_a, add);
    sum_b = block_reduce<kPairWarps>(block, partials_b, sum_b, add);
    rot = pm_rotation(sum_a, sum_b, gain_threshold);
  } else {
    __shared__ T partials_a4[kPairWarps];
    __shared__ T partials_b4[kPairWarps];
    __shared__ T partials_a8[kPairWarps];
    __shared__ T partials_b8[kPairWarps];
    T sum_a4{0};
    T sum_b4{0};
    T sum_a8{0};
    T sum_b8{0};
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kPairThreads)) {
      const PmPairStats4<T> stats = pm_pair_atom_stats4(diag_p[a], diag_q[a], off_pq[a]);
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
template <typename T>
__global__ void rotate_populations_kernel(const int32_t n_atoms, const int32_t n_occ,
                                          const T *const pops_in, const T *const cos_v,
                                          const T *const sin_v, const int32_t *const partner,
                                          T *const pops_out) {
  const auto i = static_cast<int32_t>(blockIdx.y);
  const auto j = static_cast<int32_t>(blockIdx.z);
  const auto a = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (j < i || a >= n_atoms) {
    return;
  }
  const int32_t pi = partner[i];
  const int32_t pj = partner[j];
  const T value = pm_rotate_population(cos_v[i], sin_v[i], cos_v[j], sin_v[j],
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
template <typename T>
__global__ void rotate_columns_kernel(const int32_t n_occ, const T *const in, const T *const cos_v,
                                      const T *const sin_v, const int32_t *const partner,
                                      T *const out) {
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
template <int32_t kExponent, typename T>
__global__ void __launch_bounds__(kMaxThreads)
    functional_kernel(const int32_t n_atoms, const int32_t n_occ, const T *const pops,
                      T *const out) {
  __shared__ T partials[kFunctionalWarps];
  const auto tid = static_cast<int32_t>(threadIdx.x);
  T acc{0};
  for (int32_t i = 0; i < n_occ; ++i) {
    const T *const diag = pops + pm_pop_index(n_atoms, n_occ, i, i, 0);
    for (int32_t a = tid; a < n_atoms; a += static_cast<int32_t>(kMaxThreads)) {
      acc += pm_functional_term<kExponent>(diag[a]);
    }
  }
  acc = block_reduce<kFunctionalWarps>(cg::this_thread_block(), partials, acc, PmAdd{});
  if (tid == 0) {
    out[0] = acc;
  }
}

template <typename T>
__global__ void identity_kernel(const int32_t n, T *const matrix) {
  const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
  if (idx >= total) {
    return;
  }
  // Column-major (r, c) sits at c * n + r: the diagonal is every (n + 1)th.
  matrix[idx] = (idx % (static_cast<size_t>(n) + 1) == 0) ? T{1} : T{0};
}

// The population build's and the rotation's grid: atom tiles by n_occ by n_occ.
dim3 triangle_grid(const int32_t n_atoms, const int32_t n_occ) {
  return {math::idivup(static_cast<uint32_t>(n_atoms), kElementBlock), static_cast<uint32_t>(n_occ),
          static_cast<uint32_t>(n_occ)};
}

} // namespace

template <typename T>
wwr::wwrError_t pm_populations(const wwr::wwrStream_t stream, const int32_t n_atoms,
                               const int32_t n_occ, const int32_t ld, const T *const c,
                               const T *const w, const int32_t *const atom_ao_offsets,
                               T *const pops) {
  populations_kernel<T><<<triangle_grid(n_atoms, n_occ), kElementBlock, 0, stream>>>(
      n_atoms, n_occ, ld, c, w, atom_ao_offsets, pops);
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t pm_pair_angles(const wwr::wwrStream_t stream, const int32_t exponent,
                               const int32_t n_atoms, const int32_t n_occ, const T *const pops,
                               const int32_t *const pairs, const int32_t n_pairs,
                               const T gain_threshold, T *const cos_out, T *const sin_out,
                               int32_t *const partner_out) {
  const auto blocks = static_cast<uint32_t>(n_pairs);
  if (exponent == 4) {
    pair_angle_kernel<4, T><<<blocks, kPairThreads, 0, stream>>>(
        n_atoms, n_occ, pops, pairs, gain_threshold, cos_out, sin_out, partner_out);
  } else {
    pair_angle_kernel<2, T><<<blocks, kPairThreads, 0, stream>>>(
        n_atoms, n_occ, pops, pairs, gain_threshold, cos_out, sin_out, partner_out);
  }
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t pm_rotate_populations(const wwr::wwrStream_t stream, const int32_t n_atoms,
                                      const int32_t n_occ, const T *const pops_in,
                                      const T *const cos_v, const T *const sin_v,
                                      const int32_t *const partner, T *const pops_out) {
  rotate_populations_kernel<T><<<triangle_grid(n_atoms, n_occ), kElementBlock, 0, stream>>>(
      n_atoms, n_occ, pops_in, cos_v, sin_v, partner, pops_out);
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t pm_rotate_columns(const wwr::wwrStream_t stream, const int32_t n_occ,
                                  const T *const in, const T *const cos_v, const T *const sin_v,
                                  const int32_t *const partner, T *const out) {
  const dim3 grid{math::idivup(static_cast<uint32_t>(n_occ), kElementBlock),
                  static_cast<uint32_t>(n_occ)};
  rotate_columns_kernel<T>
      <<<grid, kElementBlock, 0, stream>>>(n_occ, in, cos_v, sin_v, partner, out);
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t pm_functional(const wwr::wwrStream_t stream, const int32_t exponent,
                              const int32_t n_atoms, const int32_t n_occ, const T *const pops,
                              T *const out) {
  if (exponent == 4) {
    functional_kernel<4, T><<<1, kMaxThreads, 0, stream>>>(n_atoms, n_occ, pops, out);
  } else {
    functional_kernel<2, T><<<1, kMaxThreads, 0, stream>>>(n_atoms, n_occ, pops, out);
  }
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t pm_identity(const wwr::wwrStream_t stream, const int32_t n, T *const matrix) {
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
  const auto blocks = static_cast<uint32_t>(math::idivup(total, size_t{kElementBlock}));
  identity_kernel<T><<<blocks, kElementBlock, 0, stream>>>(n, matrix);
  return wwr::wwrGetLastError();
}

template wwr::wwrError_t pm_populations<float>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                               const float *, const float *, const int32_t *,
                                               float *);
template wwr::wwrError_t pm_populations<double>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                                const double *, const double *, const int32_t *,
                                                double *);
template wwr::wwrError_t pm_pair_angles<float>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                               const float *, const int32_t *, int32_t, float,
                                               float *, float *, int32_t *);
template wwr::wwrError_t pm_pair_angles<double>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                                const double *, const int32_t *, int32_t, double,
                                                double *, double *, int32_t *);
template wwr::wwrError_t pm_rotate_populations<float>(wwr::wwrStream_t, int32_t, int32_t,
                                                      const float *, const float *, const float *,
                                                      const int32_t *, float *);
template wwr::wwrError_t pm_rotate_populations<double>(wwr::wwrStream_t, int32_t, int32_t,
                                                       const double *, const double *,
                                                       const double *, const int32_t *, double *);
template wwr::wwrError_t pm_rotate_columns<float>(wwr::wwrStream_t, int32_t, const float *,
                                                  const float *, const float *, const int32_t *,
                                                  float *);
template wwr::wwrError_t pm_rotate_columns<double>(wwr::wwrStream_t, int32_t, const double *,
                                                   const double *, const double *, const int32_t *,
                                                   double *);
template wwr::wwrError_t pm_functional<float>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                              const float *, float *);
template wwr::wwrError_t pm_functional<double>(wwr::wwrStream_t, int32_t, int32_t, int32_t,
                                               const double *, double *);
template wwr::wwrError_t pm_identity<float>(wwr::wwrStream_t, int32_t, float *);
template wwr::wwrError_t pm_identity<double>(wwr::wwrStream_t, int32_t, double *);

} // namespace free_breakfast::device
