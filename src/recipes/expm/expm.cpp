/// @file
/// @brief skew_eig and skew_expm, and their four instantiations.

module;

// The phase_scale launcher, defined in expm.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "expm_bridge.h"
// WWR_SELECTED_HIP, for skew_eig's choice of eigensolver. Macros only.
#include "selected_backend.h"

module free_breakfast.recipes.expm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.footprint;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast {

// heevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree.
constexpr auto kSkewEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kSkewFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;
// The frexp exponent iH's largest element is scaled to before the eigensolver:
// see skew_eig.
constexpr int32_t kSkewTopExponent = 10;
#if defined(WWR_SELECTED_HIP)
// heevdx's range: every eigenpair, so no vl/vu or il/iu.
constexpr auto kSkewRange = wwr::WWRSOLVER_EIG_RANGE_ALL;
#endif

// Every scalar these calls pass is a host constant.
using HostScalars =
    wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>;

// heevd's (heevdx's under HIP) work for order n, in Z, shared by skew_eig and
// skew_eig_lwork so the two cannot disagree. The query reads neither the
// matrix nor the eigenvalues, so the lwork query passes null for both.
template <typename Z>
Result<int32_t> skew_eig_query(const wwr::wwrsolverDnHandle_t solver, const int32_t n, Z *const u,
                               wwr::ComplexToRealType<Z> *const w) {
  using R = wwr::ComplexToRealType<Z>;
  int32_t lwork = 0;
#if defined(WWR_SELECTED_HIP)
  // heevdx over the whole range, not heevd: rocSOLVER's heevd (ROCm 7.2.4)
  // returns info 0 and NaN eigenvectors above order 65 whenever iH's
  // tridiagonal form splits -- a diagonal or block-diagonal H, one nonzero
  // pair, a zero half, entries near 1e-200 (#999). heevdx gives every one a
  // finite U. cuSOLVER's heevd is correct on them, so CUDA keeps it.
  int32_t found = 0;
  FBF_TRY(gpu_status(wwr::heevdx_bufferSize<Z>(solver, kSkewEigMode, kSkewRange, kSkewFillMode, n,
                                               u, n, R{0}, R{0}, 0, 0, &found, w, &lwork)));
#else
  FBF_TRY(gpu_status(
      wwr::heevd_bufferSize<Z>(solver, kSkewEigMode, kSkewFillMode, n, u, n, w, &lwork)));
#endif
  return lwork;
}

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

  // H = 0: iH = I diag(0) I^H, returned as that with no heevd. It is the
  // exact decomposition at H = 0 and nowhere else (a nonzero H has a nonzero
  // eigenvalue, which w = 0 misses by its whole size), so the test is
  // equality: iamax's largest |re| + |im| is 0 exactly when every element is.
  // rocSOLVER's heevd gives this input info 0 and NaN eigenvectors above
  // order 65 (#992), and a zero orbital step is this input.
  int64_t top_at = 0;
  FBF_TRY(gpu_status(wwr::iamax<Z, int64_t>(blas, nn, u, 1, &top_at)));
  Z top{};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&top, u + (top_at - 1), sizeof(Z),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  const auto [re, im] = std::bit_cast<std::array<R, 2>>(top);
  if (re == R{0} && im == R{0}) {
    // u is iH, so already 0: the ones go onto its diagonal, stride n + 1.
    const std::vector<Z> ones(static_cast<size_t>(n), kOne<Z>);
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
        u, (static_cast<size_t>(n) + 1) * sizeof(Z), ones.data(), sizeof(Z), sizeof(Z),
        static_cast<size_t>(n), wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(w, 0, static_cast<size_t>(n) * sizeof(R), res.stream())));
    return res.sync();
  }

  // Both vendors' solvers lose accuracy at a fixed absolute size of iH, not
  // at one relative to it, so iH goes in scaled by the power of two that puts
  // its largest element's frexp exponent at kSkewTopExponent, and w comes out
  // scaled back: exact both ways (short of the subnormal range), U untouched.
  // The element is the zero guard's, so this costs two scals and no sync. The
  // exponent is clamped so neither factor overflows.
  //
  // rocSOLVER (#999): the residual iH U - U diag(w) stops shrinking with H
  // (2e-8 relative at ||H|| = 1e-8 in double), below about 1e-160 both its
  // solvers return U = I, w = 0, and complex-float heevdx loses 50x its n eps
  // bar on #999's zero-half input once the element drops under 1. cuSOLVER
  // (#1004): on #999's operands for a real T, heevd holds its floor down to
  // exponent -475 (double) and -46 (float), and one binade lower U^H U - I is
  // 1e4 n eps and 2 n eps; on the complex ones it holds it over the whole
  // normal range. Both are measured, not derived: rocSOLVER's heevdx is at its
  // floor from +1 (float) and -4 (double) up, cuSOLVER's heevd from -46 and
  // -475 up to +20, the top probed, so the one +10 serves both, far from
  // overflow.
  int32_t exponent = 0;
  std::frexp(std::max(std::abs(re), std::abs(im)), &exponent);
  constexpr int32_t kMaxExponent = std::numeric_limits<R>::max_exponent - 1;
  const int32_t shift = std::clamp(kSkewTopExponent - exponent, -kMaxExponent, kMaxExponent);
  const R scale = std::ldexp(R{1}, shift);
  const R unscale = std::ldexp(R{1}, -shift);
  FBF_TRY(gpu_status(wwr::scal<R, int64_t>(blas, 2 * nn, &scale, reinterpret_cast<R *>(u), 1)));

  // The solver's workspace size for n, queried per call: it is a host
  // computation.
  const wwr::wwrsolverDnHandle_t solver = res.solver().get();
  const int32_t lwork = FBF_TRY(skew_eig_query<Z>(solver, n, u, w));
#if defined(WWR_SELECTED_HIP)
  int32_t found = 0;
  const auto [work] = FBF_TRY(res.device_regions<Z>("skew_eig", static_cast<size_t>(lwork)));
  FBF_TRY(gpu_status(wwr::heevdx<Z>(solver, kSkewEigMode, kSkewRange, kSkewFillMode, n, u, n, R{0},
                                    R{0}, 0, 0, &found, w, work, lwork, res.info().data())));
#else
  const auto [work] = FBF_TRY(res.device_regions<Z>("skew_eig", static_cast<size_t>(lwork)));
  FBF_TRY(gpu_status(wwr::heevd<Z>(solver, kSkewEigMode, kSkewFillMode, n, u, n, w, work, lwork,
                                   res.info().data())));
#endif

  const int32_t info = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug in the caller, not in H); info > 0 is
  // non-convergence, which is H's doing.
  if (info < 0) {
    return err_gpu("skew_eig: the eigensolver rejected argument {}", -info);
  }
  if (info > 0) {
    return err_numerical("skew_eig: the eigensolver did not converge (info = {})", info);
  }
#if defined(WWR_SELECTED_HIP)
  // The whole range is all n pairs; fewer would leave columns of u unset.
  if (found != n) {
    return err_numerical("skew_eig: heevdx found {} of {} eigenpairs", found, n);
  }
#endif
  return gpu_status(wwr::scal<R, int64_t>(blas, n, &unscale, w, 1));
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
  Z *const v =
      std::get<0>(FBF_TRY(res.device_regions<Z>("skew_expm", static_cast<size_t>(kBlocks * nn))));
  FBF_TRY(gpu_status(device::phase_scale(res.stream(), n, u, w, t, v)));

  // e = v U^H: straight into e for a complex T; for a real T into the second
  // block, whose real plane -- every second R -- is then copied out.
  Z *product = nullptr;
  if constexpr (wwr::real_fp<T>) {
    product = v + nn;
  } else {
    product = e;
  }
  FBF_TRY(gpu_status(
      gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_C>(blas, n, &kOne<Z>, v, u, &kZero<Z>, product)));
  if constexpr (wwr::real_fp<T>) {
    FBF_TRY(
        gpu_status(wwr::copy<R, int64_t>(blas, nn, reinterpret_cast<const R *>(product), 2, e, 1)));
  }
  return {};
}

template <wwr::usual_fp T>
Result<size_t> skew_eig_lwork(DeviceResources &res, const int32_t n) {
  using Z = wwr::RealToComplexType<wwr::ComplexToRealType<T>>;
  if (n <= 0) {
    return size_t{0};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t lwork = FBF_TRY(skew_eig_query<Z>(res.solver().get(), n, nullptr, nullptr));
  return static_cast<size_t>(lwork);
}

template <wwr::usual_fp T>
MemoryFootprint skew_eig_footprint(const size_t lwork) {
  using Z = wwr::RealToComplexType<wwr::ComplexToRealType<T>>;
  MemoryFootprint out;
  out.device.workspace = math::align_up(lwork * sizeof(Z), DeviceResources::kWorkspaceAlignment);
  return out;
}

template <wwr::usual_fp T>
MemoryFootprint skew_expm_footprint(const int32_t n) {
  using Z = wwr::RealToComplexType<wwr::ComplexToRealType<T>>;
  MemoryFootprint out;
  if (n > 0) {
    constexpr size_t kBlocks = wwr::real_fp<T> ? 2 : 1;
    const auto order = static_cast<size_t>(n);
    out.device.workspace =
        math::align_up(kBlocks * order * order * sizeof(Z), DeviceResources::kWorkspaceAlignment);
  }
  return out;
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

template Result<size_t> skew_eig_lwork<float>(DeviceResources &, int32_t);
template Result<size_t> skew_eig_lwork<double>(DeviceResources &, int32_t);
template Result<size_t> skew_eig_lwork<wwr::wwrFloatComplex>(DeviceResources &, int32_t);
template Result<size_t> skew_eig_lwork<wwr::wwrDoubleComplex>(DeviceResources &, int32_t);
template MemoryFootprint skew_eig_footprint<float>(size_t);
template MemoryFootprint skew_eig_footprint<double>(size_t);
template MemoryFootprint skew_eig_footprint<wwr::wwrFloatComplex>(size_t);
template MemoryFootprint skew_eig_footprint<wwr::wwrDoubleComplex>(size_t);
template MemoryFootprint skew_expm_footprint<float>(int32_t);
template MemoryFootprint skew_expm_footprint<double>(int32_t);
template MemoryFootprint skew_expm_footprint<wwr::wwrFloatComplex>(int32_t);
template MemoryFootprint skew_expm_footprint<wwr::wwrDoubleComplex>(int32_t);

} // namespace free_breakfast
