/// @file
/// @brief skew_eig and skew_expm, and their four instantiations.

module;

// The phase_scale launcher, defined in expm.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "expm_bridge.h"

module free_breakfast.recipes.expm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// heevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree.
constexpr auto kSkewEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kSkewFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;

// Every scalar these calls pass is a host constant.
using HostScalars =
    wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>;

template <wwr::usual_fp T>
Status skew_eig(DeviceResources &res, const int32_t n, const T *h,
                wwr::RealToComplexType<wwr::ComplexToRealType<T>> *u,
                wwr::ComplexToRealType<T> *w) {
  using R = wwr::ComplexToRealType<T>;
  using Z = wwr::RealToComplexType<R>;
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const HostScalars host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};
  // 64-bit: n * n passes 2^31 at n = 46341.
  const int64_t nn = static_cast<int64_t>(n) * n;

  // u = iH. A real h is the imaginary plane of iH: zero u, then copy h into
  // every second R of it. A complex h is copied whole and scaled by i.
  if constexpr (wwr::real_fp<T>) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(u, 0, static_cast<size_t>(nn) * sizeof(Z), res.stream())));
    FBF_TRY(gpu_status(wwr::copy<R, int64_t>(blas, nn, h, 1, reinterpret_cast<R *>(u) + 1, 2)));
  } else {
    FBF_TRY(gpu_status(wwr::copy<Z, int64_t>(blas, nn, h, 1, u, 1)));
    FBF_TRY(gpu_status(wwr::scal<Z, int64_t>(blas, nn, &kImaginaryUnit<Z>, u, 1)));
  }

  // heevd's workspace size for n, queried per call: it is a host computation.
  int32_t lwork = 0;
  FBF_TRY(gpu_status(wwr::heevd_bufferSize<Z>(res.solver().get(), kSkewEigMode, kSkewFillMode, n, u,
                                              n, w, &lwork)));
  auto work = wwr::extension::reinterpret_buffer_view<Z>(
      res.device_workspace(static_cast<size_t>(lwork) * sizeof(Z)));
  FBF_TRY(gpu_status(wwr::heevd<Z>(res.solver().get(), kSkewEigMode, kSkewFillMode, n, u, n, w,
                                   work.data(), lwork, res.info().data())));

  const int32_t info = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug in the caller, not in H); info > 0 is
  // non-convergence, which is H's doing.
  if (info < 0) {
    return err_gpu("skew_eig: heevd rejected argument {}", -info);
  }
  if (info > 0) {
    return err_numerical("skew_eig: heevd did not converge (info = {})", info);
  }
  return {};
}

template <wwr::usual_fp T>
Status skew_expm(DeviceResources &res, const int32_t n,
                 const wwr::RealToComplexType<wwr::ComplexToRealType<T>> *u,
                 const wwr::ComplexToRealType<T> *w, const wwr::ComplexToRealType<T> t, T *e) {
  using R = wwr::ComplexToRealType<T>;
  using Z = wwr::RealToComplexType<R>;
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const HostScalars host_scalars{blas, wwr::WWRBLAS_POINTER_MODE_HOST};
  const int64_t nn = static_cast<int64_t>(n) * n;

  // v = U diag(e^{-i t w}), then a real T's complex product after it.
  constexpr int64_t kBlocks = wwr::real_fp<T> ? 2 : 1;
  auto work = wwr::extension::reinterpret_buffer_view<Z>(
      res.device_workspace(static_cast<size_t>(kBlocks * nn) * sizeof(Z)));
  Z *v = work.data();
  FBF_TRY(gpu_status(device::phase_scale(res.stream(), n, u, w, t, v)));

  // e = v U^H: straight into e for a complex T; for a real T into the second
  // block, whose real plane -- every second R -- is then copied out.
  Z *product = nullptr;
  if constexpr (wwr::real_fp<T>) {
    product = v + nn;
  } else {
    product = e;
  }
  FBF_TRY(gpu_status(wwr::gemm<Z, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_C, n, n, n,
                                           &kOne<Z>, v, n, u, n, &kZero<Z>, product, n)));
  if constexpr (wwr::real_fp<T>) {
    FBF_TRY(
        gpu_status(wwr::copy<R, int64_t>(blas, nn, reinterpret_cast<const R *>(product), 2, e, 1)));
  }
  return {};
}

template Status skew_eig<float>(DeviceResources &, int32_t, const float *, wwr::wwrFloatComplex *,
                                float *);
template Status skew_eig<double>(DeviceResources &, int32_t, const double *,
                                 wwr::wwrDoubleComplex *, double *);
template Status skew_eig<wwr::wwrFloatComplex>(DeviceResources &, int32_t,
                                               const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *,
                                               float *);
template Status skew_eig<wwr::wwrDoubleComplex>(DeviceResources &, int32_t,
                                                const wwr::wwrDoubleComplex *,
                                                wwr::wwrDoubleComplex *, double *);

template Status skew_expm<float>(DeviceResources &, int32_t, const wwr::wwrFloatComplex *,
                                 const float *, float, float *);
template Status skew_expm<double>(DeviceResources &, int32_t, const wwr::wwrDoubleComplex *,
                                  const double *, double, double *);
template Status skew_expm<wwr::wwrFloatComplex>(DeviceResources &, int32_t,
                                                const wwr::wwrFloatComplex *, const float *, float,
                                                wwr::wwrFloatComplex *);
template Status skew_expm<wwr::wwrDoubleComplex>(DeviceResources &, int32_t,
                                                 const wwr::wwrDoubleComplex *, const double *,
                                                 double, wwr::wwrDoubleComplex *);

} // namespace free_breakfast
