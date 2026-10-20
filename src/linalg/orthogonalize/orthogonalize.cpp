/// @file
/// @brief orthogonalize, and its float / double instantiations.

module;

// The kernel launcher, defined in inverse_sqrt.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "inverse_sqrt_bridge.h"

module free_breakfast.linalg.orthogonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.primitives;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.linalg.blas;

#include "diagnostics.h"

namespace free_breakfast {

// cond(S) above this is logged at warn. GPU4PySCF's check_linear_dependency
// warns at the same value; it is the line reference codes treat as "the basis
// has gone near-linearly dependent", adopted rather than re-derived.
template <real_fp T>
constexpr T kOrthogonalizeConditionWarn = T{1e10};

// syevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree.
constexpr auto kOrthogonalizeEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kOrthogonalizeFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;

// Log cond(S) = lambda_max / lambda_min: at debug always, at warn past the
// threshold or for a non-positive lambda_min. A near-singular basis is a
// numerical-health signal every reference code surfaces.
template <real_fp T>
void log_orthogonalize_condition(const T lambda_min, const T lambda_max, const T floor) {
  if (lambda_min <= T{0}) {
    FBF_LOG_WARN("orthogonalize: S has a non-positive eigenvalue ({:.3e}); the basis is "
                 "near-linearly dependent, and modes at or below the floor ({:.3e}) are dropped",
                 static_cast<double>(lambda_min), static_cast<double>(floor));
    return;
  }
  const T condition = lambda_max / lambda_min;
  FBF_LOG_DEBUG("orthogonalize: cond(S) = {:.3e} (lambda_min = {:.3e}, lambda_max = {:.3e})",
                static_cast<double>(condition), static_cast<double>(lambda_min),
                static_cast<double>(lambda_max));
  if (condition > kOrthogonalizeConditionWarn<T>) {
    FBF_LOG_WARN("orthogonalize: cond(S) = {:.3e} exceeds {:.0e}; the basis is near-linearly "
                 "dependent, and modes at or below the floor ({:.3e}) are dropped",
                 static_cast<double>(condition),
                 static_cast<double>(kOrthogonalizeConditionWarn<T>), static_cast<double>(floor));
  }
}

template <real_fp T>
Status orthogonalize(DeviceResources &res, const int32_t n, const T *s, T *x,
                     const T eigenvalue_floor) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);

  // syevd's workspace size for n. The legacy API ignores the matrix and
  // eigenvalue pointers in a size query.
  int32_t lwork = 0;
  FBF_TRY(
      gpu_status(wwr::syevd_bufferSize<T>(res.solver().get(), kOrthogonalizeEigMode,
                                          kOrthogonalizeFillMode, n, nullptr, n, nullptr, &lwork)));

  // Every intermediate is carved from res's workspace, requested once, in full,
  // so no later request can grow it and free a region still in use:
  //
  //   [ U: n*n | eigenvalues: n | syevd's work: lwork, then M: n*n ]
  //
  // U and the eigenvalues are syevd's outputs and live to the end. M is first
  // written by gmmwich, after syevd is done with its work on the same stream,
  // so the two share a region. Each region starts on a kWorkspaceAlignment
  // boundary, as a fresh allocation would.
  constexpr size_t kAlign = DeviceResources::kWorkspaceAlignmentIn<T>;
  const size_t u_len = math::align_up(nn, kAlign);
  const size_t eigenvalues_len = math::align_up(static_cast<size_t>(n), kAlign);
  const size_t work_len = std::max(static_cast<size_t>(lwork), nn);
  auto workspace = wwr::extension::reinterpret_buffer_view<T>(
      res.device_workspace((u_len + eigenvalues_len + work_len) * sizeof(T)));
  T *const u = workspace.data();
  T *const eigenvalues = u + u_len;
  T *const work = eigenvalues + eigenvalues_len;
  T *const m = work;

  // 1. S = U Lambda U^T. syevd overwrites its input with the eigenvectors, so it
  //    runs on a copy of S in U, leaving the caller's S untouched.
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(u, s, nn * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(
      gpu_status(wwr::syevd<T>(res.solver().get(), kOrthogonalizeEigMode, kOrthogonalizeFillMode, n,
                               u, n, eigenvalues, work, lwork, res.info().data())));

  // The one host round-trip: the extreme eigenvalues for the condition log
  // (ascending order, so first and last), read here because step 2 overwrites
  // them, then the convergence flag, whose fetch synchronizes the stream and so
  // completes all three. Host locals for the eigenvalues: a pageable
  // destination only makes the copy synchronous.
  T lambda_min = T{0};
  T lambda_max = T{0};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_min, eigenvalues, sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_max, eigenvalues + (n - 1), sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  const int32_t info_host = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug here, not in S); info > 0 is
  // non-convergence, which is S's doing.
  if (info_host < 0) {
    return err_gpu("orthogonalize: syevd rejected argument {}", -info_host);
  }
  if (info_host > 0) {
    return err_numerical("orthogonalize: syevd did not converge on S (info = {})", info_host);
  }
  log_orthogonalize_condition(lambda_min, lambda_max, eigenvalue_floor);

  // 2. Lambda <- Lambda^{-1/2} in place, dropping modes at or below the floor.
  //    Per element in the kernel, so steps 2-3 never synchronize.
  FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
      device::inverse_sqrt<T>(stream, n, eigenvalues, eigenvalue_floor))));

  // 3. X = U diag(Lambda^{-1/2}) U^T, symmetric by construction: gmmwich's
  //    op_x = T form, a dgmm (M = U diag(Lambda^{-1/2}), in O(n^2) rather than a
  //    gemm against a dense diagonal) into M, then X = M U^T.
  return gpu_status(gmmwich(res.blas(), wwr::WWRBLAS_OP_T, n, u, eigenvalues, m, x));
}

template Status orthogonalize<float>(DeviceResources &, int32_t, const float *, float *, float);
template Status orthogonalize<double>(DeviceResources &, int32_t, const double *, double *, double);

} // namespace free_breakfast
