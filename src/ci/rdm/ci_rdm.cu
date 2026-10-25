// The device half of CiRdm::rdm12: the gather of D_{pq,K} = <c| E_pq |K> over
// each determinant's own replacement lists, and Γ's assembly from M and γ. The
// products Dᵀ D (a syrk) and Dᵀ c (a gemm) between the two are on the module's
// side. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.
//
// One thread per determinant K, walking K's α links (K_β held) and then its β
// links (K_α held): a†_p a_q |K> = s |J> is <J| E_pq |K> = s, so K's own links
// reach every c_J that D_K takes. Adjacent threads are adjacent determinants,
// so every write to a column of D is coalesced. A thread owns its row, so
// nothing is atomic and the summation order is fixed.
//
// In CiLayout::symmetric a thread is a triangle determinant of the packed c,
// by ci_symmetric.h's triangle walk, the one the sigma's gathers take.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "ci_rdm_bridge.h"
#include "ci_symmetric.h"

namespace free_breakfast::device {

namespace {

// The fields of one link (see the bridge header).
constexpr int32_t kLinkWidth = 3;

// In the symmetric layout a row is triangle element det = (a, b), a ≥ b, of
// the packed c, and a link to a determinant below the triangle reads its
// mirror, times 1/√2 off the diagonal (see ci_symmetric.h); an off-diagonal
// row is then scaled by √2, so that the products over the triangle weight it
// 2, for itself and its mirror.
template <bool Symmetric>
__global__ void ci_rdm_gather_d_kernel(const int32_t begin, const int32_t rows,
                                       const int32_t n_beta, const int32_t *const alpha_links,
                                       const int32_t alpha_per, const int32_t *const beta_links,
                                       const int32_t beta_per, const double *const c,
                                       double *const d) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= rows) {
    return;
  }
  const int32_t det = begin + static_cast<int32_t>(i);
  int32_t ka = 0;
  int32_t kb = 0;
  double scale = 1.0;
  if constexpr (Symmetric) {
    ci_triangle_strings(det, ka, kb);
    scale = ka == kb ? 1.0 : kCiSqrt2;
  } else {
    ka = det / n_beta;
    kb = det % n_beta;
  }
  // c(a, b) of the unpacked vector.
  const auto at = [&](const int32_t a, const int32_t b) {
    if constexpr (Symmetric) {
      return a == b ? c[ci_triangle(a, b)] : kCiInvSqrt2 * c[ci_triangle(a, b)];
    } else {
      return c[(static_cast<size_t>(a) * static_cast<size_t>(n_beta)) + static_cast<size_t>(b)];
    }
  };
  const auto ld = static_cast<size_t>(rows);
  // α and β links can name the same pair (the diagonal p == q), so the row
  // accumulates.
  double *const d_row = d + i;
  const int32_t *a = alpha_links + (static_cast<size_t>(ka) * alpha_per * kLinkWidth);
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    d_row[static_cast<size_t>(a[1]) * ld] += static_cast<double>(a[2]) * (scale * at(a[0], kb));
  }
  const int32_t *b = beta_links + (static_cast<size_t>(kb) * beta_per * kLinkWidth);
  for (int32_t e = 0; e < beta_per; ++e, b += kLinkWidth) {
    d_row[static_cast<size_t>(b[1]) * ld] += static_cast<double>(b[2]) * (scale * at(ka, b[0]));
  }
}

__global__ void ci_rdm_assemble_kernel(const int32_t n, const double *const m,
                                       const double *const gamma, double *const gamma2) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto nn = static_cast<int64_t>(n) * n;
  if (at >= nn * nn) {
    return;
  }
  // at = (t u) · n² + (v w), each pair row-major.
  const int64_t tu = at / nn;
  const int64_t vw = at % nn;
  const int64_t t = tu / n;
  const int64_t u = tu % n;
  const int64_t v = vw / n;
  const int64_t w = vw % n;
  // M_{tu,wv}, read from the lower triangle M is symmetric about.
  const int64_t wv = (w * n) + v;
  const int64_t hi = tu > wv ? tu : wv;
  const int64_t lo = tu > wv ? wv : tu;
  const double delta = u == v ? gamma[(t * n) + w] : 0.0;
  gamma2[at] = m[hi + (lo * nn)] - delta;
}

} // namespace

wwr::wwrError_t ci_rdm_assemble(const wwr::wwrStream_t stream, const int32_t n,
                                const double *const m, const double *const gamma,
                                double *const gamma2) {
  const int64_t elements = static_cast<int64_t>(n) * n * n * n;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = static_cast<uint32_t>(math::idivup(elements, int64_t{kFourWarps}));
  ci_rdm_assemble_kernel<<<blocks, kFourWarps, 0, stream>>>(n, m, gamma, gamma2);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_rdm_gather_d(const wwr::wwrStream_t stream, const bool symmetric,
                                const int32_t begin, const int32_t rows, const int32_t n_beta,
                                const int32_t *const alpha_links, const int32_t alpha_per,
                                const int32_t *const beta_links, const int32_t beta_per,
                                const double *const c, double *const d) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = math::idivup(static_cast<uint32_t>(rows), kFourWarps);
  if (symmetric) {
    ci_rdm_gather_d_kernel<true><<<blocks, kFourWarps, 0, stream>>>(
        begin, rows, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, d);
  } else {
    ci_rdm_gather_d_kernel<false><<<blocks, kFourWarps, 0, stream>>>(
        begin, rows, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, d);
  }
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
