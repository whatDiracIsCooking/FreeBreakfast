// The device half of free_breakfast.integrals.eri.density_fitting.gradient:
// the launchers over the generated gmetric<lP, lQ> and geri3<la, lb, lP>, and
// the check of their tables against the energy kernels' limits
// (gradient_bridge.h). One source for both backends -- nvcc under CUDA, clang
// -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `gmetric` or
// `geri3`. kernels/gmetric.h and kernels/geri3.h declare the templates, every
// specialization and their tables, nullptr where a class is not emitted, so
// nothing here names a class: a class a table names but no unit defines fails
// to link. Launching a specialization another TU defines wants relocatable
// device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' headers bring the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "gradient_bridge.h"

#include "kernels/geri3.h"
#include "kernels/gmetric.h"

static_assert(std::extent_v<decltype(gmetric_kernels), 0> ==
                      free_breakfast::device::kMetricMaxMomentum + 1 &&
                  std::extent_v<decltype(gmetric_kernels), 1> ==
                      free_breakfast::device::kMetricMaxMomentum + 1,
              "kernels/'s gmetric [lP][lQ] table and kMetricMaxMomentum disagree: rerun "
              "tools/emit/df_gradient_kernels.py");

static_assert(std::extent_v<decltype(geri3_kernels), 0> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1 &&
                  std::extent_v<decltype(geri3_kernels), 1> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1 &&
                  std::extent_v<decltype(geri3_kernels), 2> ==
                      free_breakfast::device::kThreeCenterAuxMaxMomentum + 1,
              "kernels/'s geri3 [la][lb][lP] table and the three-center limits disagree: "
              "rerun tools/emit/df_gradient_kernels.py");

namespace free_breakfast::device {

wwr::wwrError_t metric_gradient_class(const wwr::wwrStream_t stream, const int32_t lp,
                                      const int32_t lq, const int32_t n_rec, const double *records,
                                      const int32_t n_pair, const int32_t *pairs,
                                      const int32_t n_aux, const int32_t n_atom, const double *w,
                                      double *gradient) {
  if (lp < 0 || lq < lp || lq > kMetricMaxMomentum || n_rec < 1 || n_pair < 1 || n_aux < 1 ||
      n_atom < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto rec = [&](const int32_t field) {
    return records + (static_cast<size_t>(field) * static_cast<size_t>(n_rec));
  };
  const auto seg = [&](const int32_t field) {
    return pairs + (static_cast<size_t>(field) * static_cast<size_t>(n_pair));
  };
  const auto kernel = gmetric_kernels[lp][lq];
  // One thread per aux shell pair, kFourWarps to a block.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(n_pair), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(n_pair, n_rec, n_aux, n_atom, rec(0), rec(1), rec(2),
                                            rec(3), rec(4), seg(0), seg(1), seg(2), seg(3), seg(4),
                                            seg(5), seg(6), seg(7), w, gradient);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t
three_center_gradient_class(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                            const int32_t lp, const int32_t n_bra_rec, const double *bra,
                            const int32_t n_aux_rec, const double *aux, const int32_t n_cell,
                            const int32_t *cells, const int32_t n_ao, const int32_t n_aux,
                            const int32_t n_atom, const double *gamma, double *gradient) {
  if (la < 0 || lb < la || lb > kThreeCenterMaxMomentum || lp < 0 ||
      lp > kThreeCenterAuxMaxMomentum || n_bra_rec < 1 || n_aux_rec < 1 || n_cell < 1 || n_ao < 1 ||
      n_aux < 1 || n_atom < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto b = [&](const int32_t field) {
    return bra + (static_cast<size_t>(field) * static_cast<size_t>(n_bra_rec));
  };
  const auto a = [&](const int32_t field) {
    return aux + (static_cast<size_t>(field) * static_cast<size_t>(n_aux_rec));
  };
  const auto c = [&](const int32_t field) {
    return cells + (static_cast<size_t>(field) * static_cast<size_t>(n_cell));
  };
  const auto kernel = geri3_kernels[la][lb][lp];
  // One thread per cell, kFourWarps to a block.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(n_cell), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(
      n_cell, n_bra_rec, n_aux_rec, n_ao, n_aux, n_atom, b(0), b(1), b(2), b(3), b(4), b(5), b(6),
      b(7), b(8), b(9), b(10), b(11), b(12), a(0), a(1), a(2), a(3), a(4), c(0), c(1), c(2), c(3),
      c(4), c(5), c(6), c(7), c(8), c(9), gamma, gradient);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
