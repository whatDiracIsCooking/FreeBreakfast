// The device half of free_breakfast.integrals.one_electron.overlap: the
// launcher over the generated overlap<la, lb> kernels. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, so the launches below see their definitions in the same TU: no
// cross-TU kernel declaration to keep in step with the generator. Each is one
// explicit specialization of the kernel template `overlap`, and kernels/units.h
// includes every one beside the class list, so nothing here names a class: the
// table is built from the template (kernel_table.h), and a class the list names
// but no unit defines fails to link. They sit in an anonymous namespace, so no
// name of theirs reaches the device link.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h,
// kernel_table.h and primitives.h include theirs first, as the bridge does
// integral_types.h. The runtime itself comes in through felizjueves/runtime.h
// before the namespace opens, as a first inclusion inside one would declare the
// runtime there; each unit's own include of it is then a no-op (#pragma once).
#include "block_params.h"
#include "kernel_table.h"
#include "primitives.h"

#include "overlap_bridge.h"

#include "felizjueves/runtime.h"

namespace free_breakfast::device {

namespace {

#include "kernels/units.h"

static_assert(lists_every_class(overlap_classes),
              "kernels/ is not every la <= lb <= kMaxAngularMomentum: rerun "
              "tools/emit_one_electron_kernels.py");

// The generated kernels share one signature (FelizJueves's Target.S parameter
// block). Taken from a kernel, not transcribed, so it follows whatever the
// generator emits.
using OverlapKernel = decltype(&overlap<0, 0>);

// The family, as kernel_table names it per class.
struct OverlapFamily {
  template <int32_t la, int32_t lb>
  static constexpr OverlapKernel kernel() {
    return &overlap<la, lb>;
  }
};

constexpr KernelTable<OverlapKernel> kOverlapKernels = kernel_table<OverlapKernel, OverlapFamily>();

} // namespace

int32_t overlap_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                      const int32_t n_ao, double *s) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const OverlapKernel kernel = kOverlapKernels[cls.la][cls.lb];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, cls.off_a,
                                            cls.off_b, s);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
