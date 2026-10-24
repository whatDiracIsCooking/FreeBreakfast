// The device half of rdm12: the gather of D_{pq,K} = <c| E_pq |K> over each
// determinant's own replacement lists. The products D Dᵀ and D c that give the
// 2- and 1-RDM are gemms, on the module's side. One source for both backends
// -- nvcc under CUDA, clang -x hip under HIP.
//
// One thread per determinant K, walking K's α links (K_β held) and then its β
// links (K_α held): a†_p a_q |K> = s |J> is <J| E_pq |K> = s, so K's own links
// reach every c_J that D_K takes. Adjacent threads are adjacent determinants,
// so every write to a column of D is coalesced. A thread owns its row, so
// nothing is atomic and the summation order is fixed.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"

#include "ci_rdm_bridge.h"

namespace free_breakfast::device {

namespace {

constexpr uint32_t kCiRdmBlock = kFourWarps;

// The fields of one link (see the bridge header).
constexpr int32_t kLinkWidth = 3;

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
  const int32_t ka = det / n_beta;
  const int32_t kb = det % n_beta;
  const auto ld = static_cast<size_t>(rows);
  // α and β links can name the same pair (the diagonal p == q), so the row
  // accumulates.
  double *const d_row = d + i;
  const int32_t *a = alpha_links + (static_cast<size_t>(ka) * alpha_per * kLinkWidth);
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    const size_t j =
        (static_cast<size_t>(a[0]) * static_cast<size_t>(n_beta)) + static_cast<size_t>(kb);
    d_row[static_cast<size_t>(a[1]) * ld] += static_cast<double>(a[2]) * c[j];
  }
  const size_t alpha_base = static_cast<size_t>(ka) * static_cast<size_t>(n_beta);
  const int32_t *b = beta_links + (static_cast<size_t>(kb) * beta_per * kLinkWidth);
  for (int32_t e = 0; e < beta_per; ++e, b += kLinkWidth) {
    d_row[static_cast<size_t>(b[1]) * ld] +=
        static_cast<double>(b[2]) * c[alpha_base + static_cast<size_t>(b[0])];
  }
}

} // namespace

wwr::wwrError_t ci_rdm_gather_d(const wwr::wwrStream_t stream, const int32_t begin,
                                const int32_t rows, const int32_t n_beta,
                                const int32_t *const alpha_links, const int32_t alpha_per,
                                const int32_t *const beta_links, const int32_t beta_per,
                                const double *const c, double *const d) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks =
      static_cast<uint32_t>((static_cast<int64_t>(rows) + kCiRdmBlock - 1) / kCiRdmBlock);
  ci_rdm_gather_d_kernel<<<blocks, kCiRdmBlock, 0, stream>>>(begin, rows, n_beta, alpha_links,
                                                             alpha_per, beta_links, beta_per, c, d);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
