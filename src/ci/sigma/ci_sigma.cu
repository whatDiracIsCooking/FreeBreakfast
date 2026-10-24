// The device half of the GPU sigma: the gather of D from c, the gather of σ
// from G, the diagonal, and the 1-RDM's unpacking. The contraction G = D W between the two gathers is
// one gemm, on the module's side. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.
//
// Both gathers are one thread per row, walking the row's α links (its β
// string held) and then its β links (its α string held). Adjacent threads are
// adjacent determinants, mostly sharing an α string, so the α walk reads c (or
// G) contiguously across a warp and every write to a column of D is
// coalesced. A thread owns its row, so nothing is atomic and the summation
// order is fixed.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"

#include "ci_sigma_bridge.h"

namespace free_breakfast::device {

namespace {

constexpr uint32_t kCiBlock = kFourWarps;

// The fields of one link (see the bridge header).
constexpr int32_t kLinkWidth = 3;

// One row's place: which vector, and which α and β strings.
struct CiRow {
  size_t vector_base; // j * n_det, the row's vector's first element
  int32_t alpha;      // I_α
  int32_t beta;       // I_β
};

__device__ CiRow ci_row(const size_t row, const int32_t n_det, const int32_t n_beta) {
  const auto det = static_cast<int32_t>(row % static_cast<size_t>(n_det));
  return CiRow{row - static_cast<size_t>(det), det / n_beta, det % n_beta};
}

// For each of the row's links, f(element, pair, sign): element is the linked
// determinant's offset in the row's vector.
template <typename F>
__device__ void ci_walk_links(const CiRow row, const int32_t n_beta,
                              const int32_t *const alpha_links, const int32_t alpha_per,
                              const int32_t *const beta_links, const int32_t beta_per, F &&f) {
  const int32_t *a = alpha_links + (static_cast<size_t>(row.alpha) * alpha_per * kLinkWidth);
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    f(row.vector_base + (static_cast<size_t>(a[0]) * static_cast<size_t>(n_beta)) +
          static_cast<size_t>(row.beta),
      a[1], a[2]);
  }
  const int32_t *b = beta_links + (static_cast<size_t>(row.beta) * beta_per * kLinkWidth);
  const size_t alpha_base =
      row.vector_base + (static_cast<size_t>(row.alpha) * static_cast<size_t>(n_beta));
  for (int32_t e = 0; e < beta_per; ++e, b += kLinkWidth) {
    f(alpha_base + static_cast<size_t>(b[0]), b[1], b[2]);
  }
}

// D of T: double for apply, float for apply_fp32.
template <typename T>
__global__ void ci_gather_d_kernel(const int32_t row_begin, const int32_t rows, const int32_t n_det,
                                   const int32_t n_beta, const int32_t *const alpha_links,
                                   const int32_t alpha_per, const int32_t *const beta_links,
                                   const int32_t beta_per, const double *const c, T *const d) {
  const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (r >= rows) {
    return;
  }
  // α and β links can name the same pair, so the row accumulates: at most one
  // link of each spin per pair, since E_vw and E_wv cannot both act on a
  // string.
  T *const d_row = d + r;
  const auto ld = static_cast<size_t>(rows);
  ci_walk_links(ci_row(static_cast<size_t>(row_begin + r), n_det, n_beta), n_beta, alpha_links,
                alpha_per, beta_links, beta_per,
                [&](const size_t element, const int32_t pair, const int32_t sign) {
                  d_row[static_cast<size_t>(pair) * ld] +=
                      static_cast<T>(static_cast<double>(sign) * c[element]);
                });
}

// G of T, summed in double.
template <typename T>
__global__ void ci_sigma_from_g_kernel(const int32_t rows, const int32_t n_det,
                                       const int32_t n_beta, const int32_t n_pairs,
                                       const int32_t *const alpha_links, const int32_t alpha_per,
                                       const int32_t *const beta_links, const int32_t beta_per,
                                       const double e_core, const double *const c, const T *const g,
                                       double *const sigma) {
  const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const auto row = static_cast<size_t>(r);
  const auto ld = static_cast<size_t>(rows);
  // <I|E_qp|J> = <J|E_pq|I> = s for each of I's own links (p, q, J, s), and G
  // is symmetric in its pair, so I's links reach every G_J[tu] σ_I takes.
  double sum =
      (e_core * c[row]) + static_cast<double>(g[(static_cast<size_t>(n_pairs) * ld) + row]);
  ci_walk_links(ci_row(row, n_det, n_beta), n_beta, alpha_links, alpha_per, beta_links, beta_per,
                [&](const size_t element, const int32_t pair, const int32_t sign) {
                  sum += static_cast<double>(sign) *
                         static_cast<double>(g[(static_cast<size_t>(pair) * ld) + element]);
                });
  sigma[row] = sum;
}

// Whether spin orbital s (orbital s % n, α for s < n) is occupied.
__device__ bool ci_occupied(const uint64_t alpha, const uint64_t beta, const int32_t n,
                            const int32_t s) {
  return s < n ? ((alpha >> s) & 1U) != 0 : ((beta >> (s - n)) & 1U) != 0;
}

__global__ void ci_diagonal_kernel(const int32_t n, const int32_t n_alpha, const int32_t n_beta,
                                   const uint64_t *const strings_alpha,
                                   const uint64_t *const strings_beta, const double e_core,
                                   const double *const terms, double *const diagonal) {
  const int64_t det = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (det >= static_cast<int64_t>(n_alpha) * n_beta) {
    return;
  }
  const uint64_t sa = strings_alpha[det / n_beta];
  const uint64_t sb = strings_beta[det % n_beta];
  const auto m = static_cast<size_t>(n);
  const double *const coulomb = terms + m;
  const double *const exchange = terms + m + (m * m);
  // host_diagonal's order: the one-electron terms over the occupied spin
  // orbitals ascending, then the pairs x < y in the same order.
  double sum = e_core;
  for (int32_t x = 0; x < 2 * n; ++x) {
    if (ci_occupied(sa, sb, n, x)) {
      sum += terms[x % n];
    }
  }
  for (int32_t x = 0; x < 2 * n; ++x) {
    if (!ci_occupied(sa, sb, n, x)) {
      continue;
    }
    const auto i = static_cast<size_t>(x % n);
    for (int32_t y = x + 1; y < 2 * n; ++y) {
      if (!ci_occupied(sa, sb, n, y)) {
        continue;
      }
      const auto j = static_cast<size_t>(y % n);
      sum += coulomb[(i * m) + j];
      if ((x < n) == (y < n)) {
        sum -= exchange[(i * m) + j];
      }
    }
  }
  diagonal[det] = sum;
}

__global__ void ci_unpack_rdm1_kernel(const int32_t n, const double *const packed,
                                      double *const gamma) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(n) * n) {
    return;
  }
  const auto t = static_cast<int32_t>(at / n);
  const auto u = static_cast<int32_t>(at % n);
  const int32_t hi = t > u ? t : u;
  const int32_t lo = t > u ? u : t;
  // An off-diagonal packed entry is γ_tu + γ_ut = 2 γ_tu.
  const double x = packed[(static_cast<size_t>(hi) * (hi + 1) / 2) + lo];
  gamma[at] = t == u ? x : 0.5 * x;
}

uint32_t ci_blocks(const int64_t items) {
  return static_cast<uint32_t>((items + kCiBlock - 1) / kCiBlock);
}

template <typename T>
wwr::wwrError_t ci_gather_d_as(const wwr::wwrStream_t stream, const int32_t row_begin,
                               const int32_t rows, const int32_t n_det, const int32_t n_beta,
                               const int32_t *const alpha_links, const int32_t alpha_per,
                               const int32_t *const beta_links, const int32_t beta_per,
                               const double *const c, T *const d) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  ci_gather_d_kernel<T><<<ci_blocks(rows), kCiBlock, 0, stream>>>(
      row_begin, rows, n_det, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, d);
  return wwr::wwrGetLastError();
}

template <typename T>
wwr::wwrError_t ci_sigma_from_g_as(const wwr::wwrStream_t stream, const int32_t rows,
                                   const int32_t n_det, const int32_t n_beta, const int32_t n_pairs,
                                   const int32_t *const alpha_links, const int32_t alpha_per,
                                   const int32_t *const beta_links, const int32_t beta_per,
                                   const double e_core, const double *const c, const T *const g,
                                   double *const sigma) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  ci_sigma_from_g_kernel<T><<<ci_blocks(rows), kCiBlock, 0, stream>>>(
      rows, n_det, n_beta, n_pairs, alpha_links, alpha_per, beta_links, beta_per, e_core, c, g,
      sigma);
  return wwr::wwrGetLastError();
}

} // namespace

wwr::wwrError_t ci_gather_d(const wwr::wwrStream_t stream, const int32_t row_begin,
                            const int32_t rows, const int32_t n_det, const int32_t n_beta,
                            const int32_t *const alpha_links, const int32_t alpha_per,
                            const int32_t *const beta_links, const int32_t beta_per,
                            const double *const c, double *const d) {
  return ci_gather_d_as(stream, row_begin, rows, n_det, n_beta, alpha_links, alpha_per, beta_links,
                        beta_per, c, d);
}

wwr::wwrError_t ci_gather_d(const wwr::wwrStream_t stream, const int32_t row_begin,
                            const int32_t rows, const int32_t n_det, const int32_t n_beta,
                            const int32_t *const alpha_links, const int32_t alpha_per,
                            const int32_t *const beta_links, const int32_t beta_per,
                            const double *const c, float *const d) {
  return ci_gather_d_as(stream, row_begin, rows, n_det, n_beta, alpha_links, alpha_per, beta_links,
                        beta_per, c, d);
}

wwr::wwrError_t ci_sigma_from_g(const wwr::wwrStream_t stream, const int32_t rows,
                                const int32_t n_det, const int32_t n_beta, const int32_t n_pairs,
                                const int32_t *const alpha_links, const int32_t alpha_per,
                                const int32_t *const beta_links, const int32_t beta_per,
                                const double e_core, const double *const c, const double *const g,
                                double *const sigma) {
  return ci_sigma_from_g_as(stream, rows, n_det, n_beta, n_pairs, alpha_links, alpha_per,
                            beta_links, beta_per, e_core, c, g, sigma);
}

wwr::wwrError_t ci_sigma_from_g(const wwr::wwrStream_t stream, const int32_t rows,
                                const int32_t n_det, const int32_t n_beta, const int32_t n_pairs,
                                const int32_t *const alpha_links, const int32_t alpha_per,
                                const int32_t *const beta_links, const int32_t beta_per,
                                const double e_core, const double *const c, const float *const g,
                                double *const sigma) {
  return ci_sigma_from_g_as(stream, rows, n_det, n_beta, n_pairs, alpha_links, alpha_per,
                            beta_links, beta_per, e_core, c, g, sigma);
}

wwr::wwrError_t ci_diagonal(const wwr::wwrStream_t stream, const int32_t n, const int32_t n_alpha,
                            const int32_t n_beta, const uint64_t *const strings_alpha,
                            const uint64_t *const strings_beta, const double e_core,
                            const double *const terms, double *const diagonal) {
  const int64_t n_det = static_cast<int64_t>(n_alpha) * n_beta;
  if (n_det == 0) {
    return wwr::wwrSuccess;
  }
  ci_diagonal_kernel<<<ci_blocks(n_det), kCiBlock, 0, stream>>>(
      n, n_alpha, n_beta, strings_alpha, strings_beta, e_core, terms, diagonal);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_unpack_rdm1(const wwr::wwrStream_t stream, const int32_t n,
                               const double *const packed, double *const gamma) {
  const int64_t elements = static_cast<int64_t>(n) * n;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  ci_unpack_rdm1_kernel<<<ci_blocks(elements), kCiBlock, 0, stream>>>(n, packed, gamma);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
