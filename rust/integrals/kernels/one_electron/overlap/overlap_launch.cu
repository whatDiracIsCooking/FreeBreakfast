// The overlap's launch wrapper over the generated overlap<la, lb> kernels. One
// source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Forked from src/integrals/one_electron/overlap/overlap_launch.cu, rewritten
// as overlap.h's extern "C", enqueue-only launch wrapper
// (rust/docs/gpu_layer.md, "Layering"). The generated kernels are sources of
// this library of their own (kernels/), each one explicit specialization of
// the kernel template `overlap`. kernels/overlap.h declares the template,
// every specialization and overlap_kernels, their [la][lb] table, so nothing
// here names a class: a class the table names but no unit defines fails to
// link. Launching a specialization another TU defines wants relocatable
// device code under nvcc (CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why). The kernels' header brings the
// runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "one_electron/overlap/overlap.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "one_electron/overlap/kernels/overlap.h"

namespace {

// The largest momentum the emitted table covers, its own limit rather than
// the tree's kMaxAngularMomentum, so this engine moves to a higher l when its
// kernels are emitted for one; the header exports it.
constexpr int32_t kMaxL = static_cast<int32_t>(std::extent_v<decltype(overlap_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(overlap_kernels), 1> == kMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/one_electron_kernels.py --out-root");
static_assert(kMaxL == FB_OVERLAP_MAX_MOMENTUM,
              "kernels/overlap.h's table and FB_OVERLAP_MAX_MOMENTUM disagree");

} // namespace

extern "C" fb_launch_status fb_overlap_class(const wwr_stream stream,
                                             const fb_primitive_pair_class_view *cls,
                                             const int32_t n_ao, double *s) {
  if (cls == nullptr || s == nullptr || cls->n_rec < 1 || n_ao < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  if (cls->la < 0 || cls->lb < cls->la || cls->lb > kMaxL) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto kernel = overlap_kernels[cls->la][cls->lb];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks =
      free_breakfast::math::idivup(static_cast<uint32_t>(cls->n_rec), free_breakfast::kFourWarps);
  // The C ABI's stream is the runtime's, behind a void *.
  kernel<<<blocks, free_breakfast::kFourWarps, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      cls->n_rec, n_ao, cls->zeta, cls->pa_x, cls->pa_y, cls->pa_z, cls->pb_x, cls->pb_y, cls->pb_z,
      cls->s, cls->off_a, cls->off_b, s);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
