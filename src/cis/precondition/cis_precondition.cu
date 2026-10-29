// The device half of cis.precondition: the orbital-energy diagonal Δε of the
// CIS A matrix, and the diagonal correction r / (θ − Δε) over a whole block
// of residuals in one launch. One source for both backends -- nvcc under CUDA,
// clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "cis_precondition_bridge.h"

namespace free_breakfast::cis::device {

namespace {

// The floor on |θ − Δε|: a root concentrated on one excitation has θ ≈ its
// Δε, and θ can land on one exactly. The CI solver's floor, for the same
// reason (src/ci/solver/ci_solver.cu).
constexpr double kDenominatorFloor = 1e-3;

__global__ void cis_diagonal_kernel(const int32_t n_occ, const int32_t n_virt,
                                    const double *const orbital_energies, double *const delta) {
  const int64_t e = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (e >= static_cast<int64_t>(n_occ) * n_virt) {
    return;
  }
  const int64_t i = e / n_virt;
  const int64_t a = e - (i * n_virt);
  delta[e] = orbital_energies[n_occ + a] - orbital_energies[i];
}

__global__ void cis_precondition_kernel(const int32_t n, const int32_t columns,
                                        const double *const theta, const double *const delta,
                                        const double *const residual, double *const correction) {
  const int64_t e = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (e >= static_cast<int64_t>(n) * columns) {
    return;
  }
  const int64_t c = e / n;
  double denominator = theta[c] - delta[e - (c * n)];
  if (denominator < kDenominatorFloor && denominator > -kDenominatorFloor) {
    denominator = denominator < 0.0 ? -kDenominatorFloor : kDenominatorFloor;
  }
  correction[e] = residual[e] / denominator;
}

} // namespace

wwr::wwrError_t diagonal(const wwr::wwrStream_t stream, const int32_t n_occ, const int32_t n_virt,
                         const double *const orbital_energies, double *const delta) {
  const int64_t items = static_cast<int64_t>(n_occ) * n_virt;
  if (items <= 0) {
    return wwr::wwrSuccess;
  }
  cis_diagonal_kernel<<<static_cast<uint32_t>(math::idivup(items, int64_t{kFourWarps})), kFourWarps,
                        0, stream>>>(n_occ, n_virt, orbital_energies, delta);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t precondition(const wwr::wwrStream_t stream, const int32_t n, const int32_t columns,
                             const double *const theta, const double *const delta,
                             const double *const residual, double *const correction) {
  const int64_t items = static_cast<int64_t>(n) * columns;
  if (items <= 0) {
    return wwr::wwrSuccess;
  }
  cis_precondition_kernel<<<static_cast<uint32_t>(math::idivup(items, int64_t{kFourWarps})),
                            kFourWarps, 0, stream>>>(n, columns, theta, delta, residual,
                                                     correction);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::cis::device
