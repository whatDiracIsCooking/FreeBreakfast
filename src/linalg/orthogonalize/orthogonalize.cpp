/// @file
/// @brief orthogonalize and its workspace query, and their float / double
/// instantiations.

module;

// The kernel launcher, defined in inverse_sqrt.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "inverse_sqrt_bridge.h"

module free_breakfast.linalg.orthogonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

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
Result<int32_t> orthogonalize_workspace_size(DeviceResources &res, const int32_t n) {
  if (n <= 0) {
    return 0;
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  // The legacy API ignores the matrix and eigenvalue pointers in a size query.
  int32_t lwork = 0;
  FBF_TRY(
      gpu_status(wwr::syevd_bufferSize<T>(res.solver().get(), kOrthogonalizeEigMode,
                                          kOrthogonalizeFillMode, n, nullptr, n, nullptr, &lwork)));
  return lwork;
}

template <real_fp T>
Status orthogonalize(DeviceResources &res, const int32_t n, const T *s, T *x, T *u_scratch,
                     T *m_scratch, T *eigenvalues, T *syevd_work, const int32_t syevd_lwork,
                     int32_t *info, const T eigenvalue_floor) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const size_t matrix_bytes = sizeof(T) * static_cast<size_t>(n) * static_cast<size_t>(n);

  // 1. S = U Lambda U^T. syevd overwrites its input with the eigenvectors, so it
  //    runs on a copy of S in U, leaving the caller's S untouched.
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(u_scratch, s, matrix_bytes, wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(
      gpu_status(wwr::syevd<T>(res.solver().get(), kOrthogonalizeEigMode, kOrthogonalizeFillMode, n,
                               u_scratch, n, eigenvalues, syevd_work, syevd_lwork, info)));

  // The one host round-trip: the convergence flag, plus the extreme eigenvalues
  // for the condition log (ascending order, so first and last). They are read
  // here because step 2 overwrites them. Host locals, not caller scratch: a
  // pageable destination only makes the copy synchronous, and the stream is
  // synchronized next anyway.
  int32_t info_host = 0;
  T lambda_min = T{0};
  T lambda_max = T{0};
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(&info_host, info, sizeof(int32_t), wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_min, eigenvalues, sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_max, eigenvalues + (n - 1), sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));
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
  //    Per element in the kernel, so steps 2-4 never synchronize.
  FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
      device::inverse_sqrt<T>(stream, n, eigenvalues, eigenvalue_floor))));

  // 3. M = U diag(Lambda^{-1/2}): scale column j of U by lambda_j^{-1/2}, in
  //    O(n^2) rather than a gemm against a dense diagonal.
  FBF_TRY(gpu_status(wwr::dgmm<T, int32_t>(res.blas().get(), wwr::WWRBLAS_SIDE_RIGHT, n, n,
                                           u_scratch, n, eigenvalues, 1, m_scratch, n)));

  // 4. X = M U^T, symmetric by construction.
  return gpu_status(wwr::gemm<T, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n,
                                          n, n, &kOne<T>, m_scratch, n, u_scratch, n, &kZero<T>, x,
                                          n));
}

template Result<int32_t> orthogonalize_workspace_size<float>(DeviceResources &, int32_t);
template Result<int32_t> orthogonalize_workspace_size<double>(DeviceResources &, int32_t);

template Status orthogonalize<float>(DeviceResources &, int32_t, const float *, float *, float *,
                                     float *, float *, float *, int32_t, int32_t *, float);
template Status orthogonalize<double>(DeviceResources &, int32_t, const double *, double *,
                                      double *, double *, double *, double *, int32_t, int32_t *,
                                      double);

} // namespace free_breakfast
