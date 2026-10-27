/// @file
/// @brief orthogonalize and orthonormalize, and their float / double
/// instantiations.

module;

// The kernel launcher, defined in inverse_sqrt.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "inverse_sqrt_bridge.h"

module free_breakfast.recipes.orthogonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.diagonalize;

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
  FBF_TRY(gpu_status(device::inverse_sqrt<T>(stream, n, eigenvalues.data(), eigenvalue_floor)));

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

template <real_fp T>
Result<int32_t> orthonormalize(DeviceResources &res, const int32_t n, const int32_t k, T *const v,
                               const T floor) {
  if (k < 0 || (k > 0 && n < 1)) {
    return err_config("orthonormalize: needs k >= 0, and n >= 1 when k > 0 (n = {}, k = {})", n, k);
  }
  if (k == 0) {
    return 0;
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};
  const auto nz = static_cast<size_t>(n);
  const auto kz = static_cast<size_t>(k);

  // G (then its eigenvectors U), the eigenvalues and D are held across
  // diagonalize, which takes the workspace for syevd, so they are the call's
  // own: one pool buffer, freed on the stream after the gemm that reads U.
  Buffers::device<T> small(kz * kz + (2 * kz), res.device());
  T *const g = small.data();
  T *const lambda = g + (kz * kz);
  T *const d = lambda + kz;
  std::vector<T> lambda_host(kz);

  // G = V^T V over v's first m columns, a gemv per column of G. Not a gemm:
  // cuBLAS runs an m x m output over a reduction of n ~ 1e5 rows on one
  // output tile, about 200 us on the 3080 for m = 3 against 4 us a gemv, and
  // syrk, which tiles only the output, took 10-20 ms.
  const auto gram = [&](const int32_t m) -> Status {
    for (int32_t j = 0; j < m; ++j) {
      FBF_TRY(gpu_status(wwr::gemv<T, int32_t>(
          blas, wwr::WWRBLAS_OP_T, n, m, &kOne<T>, v, n, v + (static_cast<size_t>(j) * nz), 1,
          &kZero<T>, g + (static_cast<size_t>(j) * static_cast<size_t>(m)), 1)));
    }
    return {};
  };

  int32_t m = k; // the columns this pass orthonormalizes
  for (int32_t pass = 0; pass < 2 && m > 0; ++pass) {
    const auto mz = static_cast<size_t>(m);
    // 1. G = V^T V.
    FBF_TRY(gram(m));
    // 2. D = diag(G)^{-1/2}, a column of norm <= floor scaled to zero; then
    //    G' = D G D, in place (each output element reads only its own input).
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(d, sizeof(T), g, (mz + 1) * sizeof(T), sizeof(T), mz,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(device::inverse_sqrt<T>(stream, m, d, floor * floor)));
    FBF_TRY(
        gpu_status(wwr::dgmm<T, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, m, m, g, m, d, 1, g, m)));
    FBF_TRY(
        gpu_status(wwr::dgmm<T, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, m, m, g, m, d, 1, g, m)));
    // 3. G' = U Lambda U^T (a sync), and Lambda read back to count the kept
    //    modes: the top of the ascending spectrum. The stream is idle, so the
    //    copy waits on nothing else.
    FBF_TRY(diagonalize<T>(res, m, g, lambda));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(lambda_host.data(), lambda, mz * sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    const T cut = std::max(floor * floor, static_cast<T>(m) * std::numeric_limits<T>::epsilon() *
                                              lambda_host[mz - 1]);
    const auto kept = static_cast<int32_t>(std::count_if(
        lambda_host.begin(), lambda_host.begin() + m, [cut](T l) { return l > cut; }));
    if (kept == 0) {
      return 0;
    }
    // 4. U_kept <- D U_kept Lambda_kept^{-1/2}, then V U_kept into the
    //    workspace and back over V's leading columns.
    T *const u_kept = g + (static_cast<size_t>(m - kept) * mz);
    T *const lambda_kept = lambda + (m - kept);
    FBF_TRY(gpu_status(device::inverse_sqrt<T>(stream, kept, lambda_kept, T{0})));
    FBF_TRY(gpu_status(
        wwr::dgmm<T, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, m, kept, u_kept, m, d, 1, u_kept, m)));
    FBF_TRY(gpu_status(wwr::dgmm<T, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, m, kept, u_kept, m,
                                             lambda_kept, 1, u_kept, m)));
    const auto [product] = res.device_regions<T>(nz * static_cast<size_t>(kept));
    FBF_TRY(gpu_status(gemm(blas, n, kept, m, &kOne<T>, v, u_kept, &kZero<T>, product)));
    const size_t kept_bytes = nz * static_cast<size_t>(kept) * sizeof(T);
    FBF_TRY(gpu_status(
        wwr::wwrMemcpyAsync(v, product, kept_bytes, wwr::wwrMemcpyDeviceToDevice, stream)));
    m = kept;

    // 5. The second pass. This one leaves Q^T Q = I + E, ||E|| ~ kappa
    //    epsilon, kappa = lambda_max / lambda_min over the kept modes. When
    //    that is below sqrt(epsilon) with room to spare, no mode can drop
    //    again, and one Newton-Schulz step, Q <- Q (3 I - Q^T Q) / 2, leaves
    //    O(E^2), rounding: gemvs and a gemm, no sync or syevd, where a canonical
    //    pass costs both (7% of ci::solve's wall time at CAS(10,10) on the
    //    3080). Davidson's correction blocks have a median kappa near 16.
    const T eps = std::numeric_limits<T>::epsilon();
    const T kappa = lambda_host[mz - 1] / lambda_host[mz - static_cast<size_t>(kept)];
    if (pass == 0 && kappa * eps <= T{1e-2} * std::sqrt(eps)) {
      const T neg_half{-0.5};
      const T three_halves{1.5};
      FBF_TRY(gram(m));
      // product still holds Q: product <- 3/2 Q - 1/2 Q (Q^T Q).
      FBF_TRY(gpu_status(gemm(blas, n, m, m, &neg_half, v, g, &three_halves, product)));
      FBF_TRY(gpu_status(
          wwr::wwrMemcpyAsync(v, product, kept_bytes, wwr::wwrMemcpyDeviceToDevice, stream)));
      return m;
    }
  }
  return m;
}

template Status orthogonalize<float>(DeviceResources &, int32_t, const float *, float *, float);
template Status orthogonalize<double>(DeviceResources &, int32_t, const double *, double *, double);
template Result<int32_t> orthonormalize<float>(DeviceResources &, int32_t, int32_t, float *, float);
template Result<int32_t> orthonormalize<double>(DeviceResources &, int32_t, int32_t, double *,
                                                double);

} // namespace free_breakfast
