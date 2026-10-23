/// @file
/// @brief polynomial_roots, and its four coefficient-type instantiations.

module;

// The kernel launcher, defined in aberth.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "aberth_bridge.h"

module free_breakfast.recipes.polynomial_roots;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.kernel; // kWarpSize: the degree bound
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

namespace free_breakfast {

template <wwr::usual_fp T, wwr::complex_fp CT>
  requires std::same_as<wwr::ComplexToRealType<CT>, wwr::ComplexToRealType<T>>
Status polynomial_roots(DeviceResources &res, const int32_t n, const int32_t batch, const T *coeffs,
                        CT *roots, int32_t *iters, const wwr::ComplexToRealType<T> tol,
                        const int32_t max_iter) {
  // !(tol >= 0) also rejects a NaN tol, which no correction would ever meet.
  if (n < 0 || batch < 0 || max_iter < 0 || !(tol >= 0)) {
    return err_config("polynomial_roots: n ({}), batch ({}), max_iter ({}) and tol ({}) must "
                      "all be >= 0",
                      n, batch, max_iter, tol);
  }
  if (n == 0 || batch == 0) {
    return {};
  }
  if (static_cast<uint32_t>(n) > kWarpSize) {
    return err_config("polynomial_roots: degree {} is over the warp width {}: one lane per root", n,
                      kWarpSize);
  }
  if (coeffs == nullptr || roots == nullptr || iters == nullptr) {
    return err_config("polynomial_roots: a null device pointer with {} polynomials to solve",
                      batch);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::aberth(res.stream(), n, batch, coeffs, roots, iters, tol, max_iter));
}

template Status polynomial_roots<float, wwr::wwrFloatComplex>(DeviceResources &, int32_t, int32_t,
                                                              const float *, wwr::wwrFloatComplex *,
                                                              int32_t *, float, int32_t);
template Status polynomial_roots<double, wwr::wwrDoubleComplex>(DeviceResources &, int32_t, int32_t,
                                                                const double *,
                                                                wwr::wwrDoubleComplex *, int32_t *,
                                                                double, int32_t);
template Status polynomial_roots<wwr::wwrFloatComplex, wwr::wwrFloatComplex>(
    DeviceResources &, int32_t, int32_t, const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *,
    int32_t *, float, int32_t);
template Status polynomial_roots<wwr::wwrDoubleComplex, wwr::wwrDoubleComplex>(
    DeviceResources &, int32_t, int32_t, const wwr::wwrDoubleComplex *, wwr::wwrDoubleComplex *,
    int32_t *, double, int32_t);

} // namespace free_breakfast
