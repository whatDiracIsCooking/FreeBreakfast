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
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.linalg.blas;
import free_breakfast.linalg.diagonalize;

#include "diagnostics.h"

namespace free_breakfast {

// cond(S) above this is logged at warn. GPU4PySCF's check_linear_dependency
// warns at the same value; it is the line reference codes treat as "the basis
// has gone near-linearly dependent", adopted rather than re-derived.
template <real_fp T>
constexpr T kOrthogonalizeConditionWarn = T{1e10};

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

  // U and the eigenvalues are scratch held across diagonalize, which takes the
  // workspace for syevd's work, so they are the routine's own pool buffers.
  // Their frees are ordered on res.stream() after the gmmwich that reads them.
  Buffers::device<T> u(nn, res.device());
  Buffers::device<T> eigenvalues(static_cast<size_t>(n), res.device());

  // 1. S = U Lambda U^T. diagonalize overwrites its input with the
  //    eigenvectors, so it runs on a copy of S in U, leaving the caller's S
  //    untouched. It synchronizes the stream to read syevd's flag.
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(u.data(), s, nn * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(diagonalize<T>(res, n, u.data(), eigenvalues.data()));

  // The extreme eigenvalues for the condition log (ascending order, so first
  // and last), read here because step 2 overwrites them. The stream is idle
  // after diagonalize's sync, so these wait on nothing but the copies. Host
  // locals: a pageable destination only makes the copy synchronous.
  T lambda_min = T{0};
  T lambda_max = T{0};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_min, eigenvalues.data(), sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&lambda_max, eigenvalues.data() + (n - 1), sizeof(T),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  log_orthogonalize_condition(lambda_min, lambda_max, eigenvalue_floor);

  // 2. Lambda <- Lambda^{-1/2} in place, dropping modes at or below the floor.
  //    Per element in the kernel, so steps 2-3 never synchronize.
  FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
      device::inverse_sqrt<T>(stream, n, eigenvalues.data(), eigenvalue_floor))));

  // 3. X = U diag(Lambda^{-1/2}) U^T, symmetric by construction: gmmwich's
  //    OpX = T form, a dgmm (M = U diag(Lambda^{-1/2}), in O(n^2) rather than a
  //    gemm against a dense diagonal) into M, then X = M U^T. M is the
  //    workspace's: diagonalize is done with it, and nothing else is called
  //    while M is live. syevd's work is usually larger than n*n, so this
  //    request rarely grows it.
  auto m = wwr::extension::reinterpret_buffer_view<T>(res.device_workspace(nn * sizeof(T)));
  return gpu_status(
      gmmwich<wwr::WWRBLAS_OP_T>(res.blas(), n, u.data(), eigenvalues.data(), m.data(), x));
}

template Status orthogonalize<float>(DeviceResources &, int32_t, const float *, float *, float);
template Status orthogonalize<double>(DeviceResources &, int32_t, const double *, double *, double);

} // namespace free_breakfast
