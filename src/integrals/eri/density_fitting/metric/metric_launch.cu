// The device half of free_breakfast.integrals.eri.density_fitting.metric: the
// launcher over the generated metric<lP, lQ>, and the check of their table
// against the metric's limit (metric_bridge.h). One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `metric`.
// kernels/metric.h declares the template, every specialization and
// metric_kernels, their [lP][lQ] table, nullptr where lP > lQ (only lP <= lQ is
// emitted), so nothing here names a class: a class the table names but no unit
// defines fails to link. Launching a specialization another TU defines wants
// relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "metric_bridge.h"

#include "kernels/metric.h"

static_assert(std::extent_v<decltype(metric_kernels), 0> ==
                      free_breakfast::device::kMetricMaxMomentum + 1 &&
                  std::extent_v<decltype(metric_kernels), 1> ==
                      free_breakfast::device::kMetricMaxMomentum + 1,
              "kernels/'s [lP][lQ] table and kMetricMaxMomentum disagree: rerun "
              "tools/emit/df_kernels.py");

namespace free_breakfast::device {

wwr::wwrError_t metric_class(const wwr::wwrStream_t stream, const int32_t lp, const int32_t lq,
                             const int32_t n_rec, const double *records, const int32_t n_pair,
                             const int32_t *pairs, const int32_t n_aux, double *metric) {
  if (lp < 0 || lq < lp || lq > kMetricMaxMomentum || n_rec < 1 || n_pair < 1 || n_aux < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto rec = [&](const int32_t field) {
    return records + static_cast<size_t>(field) * static_cast<size_t>(n_rec);
  };
  const auto seg = [&](const int32_t field) {
    return pairs + static_cast<size_t>(field) * static_cast<size_t>(n_pair);
  };
  const auto kernel = metric_kernels[lp][lq];
  // One thread per aux shell pair, kFourWarps to a block.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(n_pair), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(n_pair, n_rec, n_aux, rec(0), rec(1), rec(2), rec(3),
                                            rec(4), seg(0), seg(1), seg(2), seg(3), seg(4), seg(5),
                                            metric);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
