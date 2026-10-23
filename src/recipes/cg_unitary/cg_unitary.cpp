/// @file
/// @brief The line searches' scalar stages, polynomial_step and dft_step, and
/// their four instantiations: every device launch the module makes.

module;

// The launchers, defined in line_search.cu and select_root.cu: declared in the
// GMF so they keep external linkage (see the header).
#include "cg_unitary_bridge.h"

module free_breakfast.recipes.cg_unitary;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.polynomial_roots;

#include "diagnostics.h"

namespace free_breakfast {

// Aberth's settling test: a relative correction at unit roundoff, or a residual
// at Horner's floor. 500 iterations is far past what a degree <= 30
// polynomial needs; a cap reached means the roots will not settle, and the
// selectors then report no step.
template <typename R>
constexpr R kAberthTolerance = std::numeric_limits<R>::epsilon();
constexpr int32_t kAberthMaxIterations = 500;

// The steps' scratch, kCgMaxSamples elements of the widest type per slot. Each
// slot is a whole number of workspace alignment steps, so every region starts
// aligned for any of the types it may hold.
template <typename Z>
constexpr size_t kSlotBytes = kCgMaxSamples * sizeof(Z);

template <wwr::usual_fp T>
Result<wwr::ComplexToRealType<T>>
polynomial_step(DeviceResources &res, const T *const dots, const int32_t order,
                const wwr::ComplexToRealType<T> h, const wwr::ComplexToRealType<T> sample_scale) {
  using R = wwr::ComplexToRealType<T>;
  using Z = wwr::RealToComplexType<R>;
  constexpr size_t kSlot = kSlotBytes<Z>;
  static_assert(kSlot % DeviceResources::kWorkspaceAlignment == 0);

  // coeffs | roots | iters | the root.
  std::byte *const base = res.device_workspace(4 * kSlot).data();
  auto *const coeffs = reinterpret_cast<R *>(base);
  auto *const roots = reinterpret_cast<Z *>(base + kSlot);
  auto *const iters = reinterpret_cast<int32_t *>(base + (2 * kSlot));
  auto *const root = reinterpret_cast<R *>(base + (3 * kSlot));

  FBF_TRY(gpu_status(device::table1_coefficients(res.stream(), dots, order, sample_scale, coeffs)));
  // Parenthesized: the template argument list's comma would split the macro.
  FBF_TRY((polynomial_roots<R, Z>(res, order, 1, coeffs, roots, iters, kAberthTolerance<R>,
                                  kAberthMaxIterations)));
  // The interpolant is in s = mu / h, so the interval (0, T_mu] is (0, order].
  FBF_TRY(gpu_status(device::first_real_root(res.stream(), coeffs, order, roots, iters,
                                             static_cast<R>(order), root)));
  R s{};
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(&s, root, sizeof(R), wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return s * h;
}

template <wwr::usual_fp T>
Result<wwr::ComplexToRealType<T>>
dft_step(DeviceResources &res, const T *const dots, const T *const cost_dots,
         const int32_t num_samples, const wwr::ComplexToRealType<T> sample_scale,
         const wwr::ComplexToRealType<T> t_dft, const bool maximizing) {
  using R = wwr::ComplexToRealType<T>;
  using Z = wwr::RealToComplexType<R>;
  constexpr size_t kSlot = kSlotBytes<Z>;
  static_assert(kSlot % DeviceResources::kWorkspaceAlignment == 0);

  // coeffs | roots | iters | args | num_args | the step.
  std::byte *const base = res.device_workspace(6 * kSlot).data();
  auto *const coeffs = reinterpret_cast<Z *>(base);
  auto *const roots = reinterpret_cast<Z *>(base + kSlot);
  auto *const iters = reinterpret_cast<int32_t *>(base + (2 * kSlot));
  auto *const args = reinterpret_cast<R *>(base + (3 * kSlot));
  auto *const num_args = reinterpret_cast<int32_t *>(base + (4 * kSlot));
  auto *const mu = reinterpret_cast<R *>(base + (5 * kSlot));

  FBF_TRY(gpu_status(device::hann_dft(res.stream(), dots, num_samples, sample_scale, coeffs)));
  // Degree num_samples - 1 <= kCgMaxSamples - 2, inside the warp.
  FBF_TRY((polynomial_roots<Z, Z>(res, num_samples - 1, 1, coeffs, roots, iters,
                                  kAberthTolerance<R>, kAberthMaxIterations)));
  FBF_TRY(gpu_status(device::unit_circle_arguments(res.stream(), coeffs, num_samples, roots, iters,
                                                   maximizing, args, num_args)));
  FBF_TRY(gpu_status(device::select_dft_step(res.stream(), args, num_args, cost_dots, num_samples,
                                             t_dft, maximizing, mu)));
  R step{};
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(&step, mu, sizeof(R), wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return step;
}

template Result<float> polynomial_step<float>(DeviceResources &, const float *, int32_t, float,
                                              float);
template Result<double> polynomial_step<double>(DeviceResources &, const double *, int32_t, double,
                                                double);
template Result<float> polynomial_step<wwr::wwrFloatComplex>(DeviceResources &,
                                                             const wwr::wwrFloatComplex *, int32_t,
                                                             float, float);
template Result<double> polynomial_step<wwr::wwrDoubleComplex>(DeviceResources &,
                                                               const wwr::wwrDoubleComplex *,
                                                               int32_t, double, double);

template Result<float> dft_step<float>(DeviceResources &, const float *, const float *, int32_t,
                                       float, float, bool);
template Result<double> dft_step<double>(DeviceResources &, const double *, const double *, int32_t,
                                         double, double, bool);
template Result<float> dft_step<wwr::wwrFloatComplex>(DeviceResources &,
                                                      const wwr::wwrFloatComplex *,
                                                      const wwr::wwrFloatComplex *, int32_t, float,
                                                      float, bool);
template Result<double> dft_step<wwr::wwrDoubleComplex>(DeviceResources &,
                                                        const wwr::wwrDoubleComplex *,
                                                        const wwr::wwrDoubleComplex *, int32_t,
                                                        double, double, bool);

} // namespace free_breakfast
