// The generated metric<lP, lQ> table, checked against the metric's limit
// (metric_bridge.h) at compile time. No launcher yet: this TU defines nothing.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// kernels/metric.h declares the template, every specialization and
// metric_kernels, their [lP][lQ] table, nullptr where lP > lQ (only lP <= lQ
// is emitted).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). The bridge includes
// integral_types.h first; the kernels' header brings the runtime in through
// felizjueves/runtime.h.
#include <type_traits>

#include "metric_bridge.h"

#include "kernels/metric.h"

static_assert(std::extent_v<decltype(metric_kernels), 0> ==
                      free_breakfast::device::kMetricMaxMomentum + 1 &&
                  std::extent_v<decltype(metric_kernels), 1> ==
                      free_breakfast::device::kMetricMaxMomentum + 1,
              "kernels/'s [lP][lQ] table and kMetricMaxMomentum disagree: rerun "
              "tools/emit_df_kernels.py");
