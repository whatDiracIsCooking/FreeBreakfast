/// @file
/// @brief norm, and its Norm x float / double instantiations.

module;

// The abs_at launcher, defined in abs_at.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "abs_at_bridge.h"

module free_breakfast.recipes.norm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

template <Norm which, real_fp T>
Status norm(DeviceResources &res, const int32_t n, const T *x, const int32_t incx, T *result) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  wwr::wwrblasPointerMode_t mode{};
  FBF_TRY(gpu_status(wwr::wwrblasGetPointerMode(blas, &mode)));
  const bool on_device = mode == wwr::WWRBLAS_POINTER_MODE_DEVICE;

  if (n <= 0) {
    if (!on_device) {
      *result = T{0};
      return {};
    }
    // All-zero bits are +0.0 in both precisions.
    return gpu_status(wwr::wwrMemsetAsync(result, 0, sizeof(T), res.stream()));
  }

  if constexpr (which == Norm::l1) {
    return gpu_status(wwr::asum<T, int32_t>(blas, n, x, incx, result));
  } else if constexpr (which == Norm::l2) {
    return gpu_status(wwr::nrm2<T, int32_t>(blas, n, x, incx, result));
  } else {
    if (on_device) {
      // iamax's 1-based index lands on the device, and abs_at reads the element
      // it names there, so nothing waits. The index's free is ordered on
      // res.stream() after that kernel.
      Buffers::device<int32_t> at(1, res.device());
      FBF_TRY(gpu_status(wwr::iamax<T, int32_t>(blas, n, x, incx, at.data())));
      return gpu_status(device::abs_at(res.stream(), x, incx, at.data(), result));
    }
    // Host mode: iamax has waited and its index is on the host. Copy that one
    // element back and wait again, so @p result holds the norm on return, as
    // asum's and nrm2's do in this mode.
    int32_t at = 0;
    FBF_TRY(gpu_status(wwr::iamax<T, int32_t>(blas, n, x, incx, &at)));
    T top{};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&top, x + (static_cast<int64_t>(at - 1) * incx),
                                           sizeof(T), wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    *result = std::abs(top);
    return {};
  }
}

template Status norm<Norm::l1, float>(DeviceResources &, int32_t, const float *, int32_t, float *);
template Status norm<Norm::l2, float>(DeviceResources &, int32_t, const float *, int32_t, float *);
template Status norm<Norm::inf, float>(DeviceResources &, int32_t, const float *, int32_t, float *);
template Status norm<Norm::l1, double>(DeviceResources &, int32_t, const double *, int32_t,
                                       double *);
template Status norm<Norm::l2, double>(DeviceResources &, int32_t, const double *, int32_t,
                                       double *);
template Status norm<Norm::inf, double>(DeviceResources &, int32_t, const double *, int32_t,
                                        double *);

} // namespace free_breakfast
