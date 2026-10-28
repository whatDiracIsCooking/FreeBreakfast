// The device half of ci::Rdm::rdm12: the gather of D_{pq,K} = <c| E_pq |K> over
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
// In ci::Layout::symmetric a thread is a triangle determinant of the packed c,
// by ci_symmetric.h's triangle walk, the one the sigma's gathers take.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "ci_rdm_bridge.h"
#include "ci_symmetric.h"

namespace free_breakfast::ci::device {

namespace {

// In the symmetric layout a row is triangle element det = (a, b), a ≥ b, of
// the packed c, and a link to a determinant below the triangle reads its
// mirror, times 1/√2 off the diagonal (see ci_symmetric.h); an off-diagonal
// row is then scaled by √2, so that the products over the triangle weight it
// 2, for itself and its mirror.
template <bool Symmetric>
__global__ void ci_rdm_gather_d_kernel(const int32_t begin, const int32_t rows, const int32_t n,
                                       const int32_t n_beta, const Replacement *const alpha_links,
                                       const int32_t alpha_per, const Replacement *const beta_links,
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
    triangle_strings(det, ka, kb);
    scale = ka == kb ? 1.0 : kSqrt2<double>;
  } else {
    ka = det / n_beta;
    kb = det % n_beta;
  }
  // c(a, b) of the unpacked vector.
  const auto at = [&](const int32_t a, const int32_t b) {
    if constexpr (Symmetric) {
      return a == b ? c[triangle(a, b)] : kInvSqrt2<double> * c[triangle(a, b)];
    } else {
      return c[(static_cast<size_t>(a) * static_cast<size_t>(n_beta)) + static_cast<size_t>(b)];
    }
  };
  const auto ld = static_cast<size_t>(rows);
  // α and β links can name the same pair (the diagonal p == q), so the row
  // accumulates. A link's column is its unpacked pair p · n + q.
  double *const d_row = d + i;
  const auto column = [&](const Replacement l) {
    return (static_cast<size_t>(l.p) * static_cast<size_t>(n)) + static_cast<size_t>(l.q);
  };
  const Replacement *const a = alpha_links + (static_cast<size_t>(ka) * alpha_per);
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    d_row[column(l) * ld] += static_cast<double>(l.sign) * (scale * at(l.target, kb));
  }
  const Replacement *const b = beta_links + (static_cast<size_t>(kb) * beta_per);
  for (int32_t e = 0; e < beta_per; ++e) {
    const Replacement l = b[e];
    d_row[column(l) * ld] += static_cast<double>(l.sign) * (scale * at(ka, l.target));
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

wwr::wwrError_t rdm_assemble(const wwr::wwrStream_t stream, const int32_t n, const double *const m,
                             const double *const gamma, double *const gamma2) {
  const int64_t elements = static_cast<int64_t>(n) * n * n * n;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = static_cast<uint32_t>(math::idivup(elements, int64_t{kFourWarps}));
  ci_rdm_assemble_kernel<<<blocks, kFourWarps, 0, stream>>>(n, m, gamma, gamma2);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_gather_d(const wwr::wwrStream_t stream, const bool symmetric,
                             const int32_t begin, const int32_t rows, const int32_t n,
                             const int32_t n_beta, const Replacement *const alpha_links,
                             const int32_t alpha_per, const Replacement *const beta_links,
                             const int32_t beta_per, const double *const c, double *const d) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = math::idivup(static_cast<uint32_t>(rows), kFourWarps);
  if (symmetric) {
    ci_rdm_gather_d_kernel<true><<<blocks, kFourWarps, 0, stream>>>(
        begin, rows, n, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, d);
  } else {
    ci_rdm_gather_d_kernel<false><<<blocks, kFourWarps, 0, stream>>>(
        begin, rows, n, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, d);
  }
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::ci::device
