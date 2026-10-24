// The device half of ci_solve's per-iteration work: S² on a block of CI
// vectors, which the spin penalty adds into sigma's output, and the diagonal
// preconditioner. One source for both backends -- nvcc under CUDA, clang -x
// hip under HIP.
//
// S² is a gather, one thread per row: (Σ_pq E^β_pq E^α_qp x)_J walks J's α
// links a†_p a_q and, for each, the β partner a†_q a_p of J's β string, read
// from that string's slot table. S² is real symmetric, so J's own links reach
// every K with <J|S²|K> nonzero. A thread owns its row, so nothing is atomic
// and the summation order is fixed.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"

#include "ci_solver_bridge.h"

namespace free_breakfast::device {

namespace {

constexpr uint32_t kCiBlock = kFourWarps;

// The fields of one α link (see the bridge header).
constexpr int32_t kLinkWidth = 3;

// The preconditioner's floor on |theta − diag|: a root concentrated on one
// determinant has theta ≈ its diagonal.
constexpr double kDenominatorFloor = 1e-3;

__global__ void ci_spin_square_kernel(const int32_t rows, const int32_t n_det, const int32_t n_beta,
                                      const int32_t n, const int32_t *const alpha_links,
                                      const int32_t alpha_per, const int32_t *const beta_slots,
                                      const double constant, const double shift, const double scale,
                                      const bool accumulate, const double *const x,
                                      double *const y) {
  const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const auto row = static_cast<size_t>(r);
  const auto det = static_cast<int32_t>(row % static_cast<size_t>(n_det));
  const size_t vector_base = row - static_cast<size_t>(det);
  const int32_t ja = det / n_beta;
  const int32_t jb = det % n_beta;

  double sum = (constant - shift) * x[row];
  const int32_t *a = alpha_links + (static_cast<size_t>(ja) * alpha_per * kLinkWidth);
  const int32_t *const slots =
      beta_slots + (static_cast<size_t>(jb) * static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    const int32_t slot = slots[a[1]];
    if (slot == 0) {
      continue;
    }
    const int32_t kb = (slot < 0 ? -slot : slot) - 1;
    const double sign = static_cast<double>(slot < 0 ? -a[2] : a[2]);
    sum -= sign * x[vector_base + (static_cast<size_t>(a[0]) * static_cast<size_t>(n_beta)) +
                    static_cast<size_t>(kb)];
  }
  y[row] = (accumulate ? y[row] : 0.0) + (scale * sum);
}

__global__ void ci_precondition_kernel(const int32_t n, const double theta,
                                       const double *const shifted, const double *const residual,
                                       double *const correction) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n) {
    return;
  }
  double denominator = theta - shifted[i];
  if (denominator < kDenominatorFloor && denominator > -kDenominatorFloor) {
    denominator = denominator < 0.0 ? -kDenominatorFloor : kDenominatorFloor;
  }
  correction[i] = residual[i] / denominator;
}

uint32_t ci_blocks(const int64_t items) {
  return static_cast<uint32_t>((items + kCiBlock - 1) / kCiBlock);
}

} // namespace

wwr::wwrError_t ci_spin_square(const wwr::wwrStream_t stream, const int32_t rows,
                               const int32_t n_det, const int32_t n_beta, const int32_t n,
                               const int32_t *const alpha_links, const int32_t alpha_per,
                               const int32_t *const beta_slots, const double constant,
                               const double shift, const double scale, const bool accumulate,
                               const double *const x, double *const y) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  ci_spin_square_kernel<<<ci_blocks(rows), kCiBlock, 0, stream>>>(
      rows, n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale,
      accumulate, x, y);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_precondition(const wwr::wwrStream_t stream, const int32_t n, const double theta,
                                const double *const shifted, const double *const residual,
                                double *const correction) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  ci_precondition_kernel<<<ci_blocks(n), kCiBlock, 0, stream>>>(n, theta, shifted, residual,
                                                                correction);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
