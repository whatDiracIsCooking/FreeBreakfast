// The DF Coulomb metric's launch wrapper over the generated metric<lP, lQ>
// kernels. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.
//
// Forked from src/integrals/eri/density_fitting/metric/metric_launch.cu,
// rewritten as metric.h's extern "C", enqueue-only launch wrapper
// (rust/docs/gpu_layer.md, "Layering"), its refusals split into
// FB_LAUNCH_INVALID_ARGUMENT and FB_LAUNCH_UNSUPPORTED ("Statuses"), and the
// 32-bit extent the kernels index by refused rather than assumed. The
// generated kernels are sources of this library of their own (kernels/), each
// one explicit specialization of the kernel template `metric`.
// kernels/metric.h declares the template, every specialization and
// metric_kernels, their [lP][lQ] table, nullptr where lP > lQ (only lP <= lQ
// is emitted), so nothing here names a class: a class the table names but no
// unit defines fails to link. Launching a specialization another TU defines
// wants relocatable device code under nvcc (CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why). The kernels' header brings the
// runtime in through felizjueves/runtime.h.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "eri/density_fitting/metric/metric.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "eri/density_fitting/metric/kernels/metric.h"

static_assert(std::extent_v<decltype(metric_kernels), 0> == FB_METRIC_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(metric_kernels), 1> == FB_METRIC_MAX_MOMENTUM + 1,
              "kernels/'s [lP][lQ] table and FB_METRIC_MAX_MOMENTUM disagree: rerun "
              "tools/emit/df_kernels.py");

extern "C" fb_launch_status fb_metric_class(const wwr_stream stream, const int32_t lp,
                                            const int32_t lq, const int32_t n_rec,
                                            const double *records, const int32_t n_pair,
                                            const int32_t *pairs, const int32_t n_aux,
                                            double *metric) {
  if (records == nullptr || pairs == nullptr || metric == nullptr || n_rec < 1 || n_pair < 1 ||
      n_aux < 1 || int64_t{n_aux} * n_aux > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  if (lp < 0 || lq < lp || lq > FB_METRIC_MAX_MOMENTUM) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto rec = [&](const int32_t field) {
    return records + (static_cast<size_t>(field) * static_cast<size_t>(n_rec));
  };
  const auto seg = [&](const int32_t field) {
    return pairs + (static_cast<size_t>(field) * static_cast<size_t>(n_pair));
  };
  const auto kernel = metric_kernels[lp][lq];
  // One thread per aux shell pair, kFourWarps to a block.
  const uint32_t blocks =
      free_breakfast::math::idivup(static_cast<uint32_t>(n_pair), free_breakfast::kFourWarps);
  // The C ABI's stream is the runtime's, behind a void *.
  kernel<<<blocks, free_breakfast::kFourWarps, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      n_pair, n_rec, n_aux, rec(0), rec(1), rec(2), rec(3), rec(4), seg(0), seg(1), seg(2), seg(3),
      seg(4), seg(5), metric);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
