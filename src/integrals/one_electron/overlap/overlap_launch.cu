// The device half of free_breakfast.integrals.one_electron.overlap: the
// launcher over the generated overlap<la, lb> kernels. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `overlap`.
// kernels/overlap.h declares the template, every specialization and
// overlap_kernels, their [la][lb] table, so nothing here names a class: a class
// the table names but no unit defines fails to link. Launching a
// specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "overlap_bridge.h"

#include "kernels/overlap.h"

namespace free_breakfast::device {

static_assert(std::extent_v<decltype(overlap_kernels), 0> == kMaxAngularMomentum + 1 &&
                  std::extent_v<decltype(overlap_kernels), 1> == kMaxAngularMomentum + 1,
              "kernels/ was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_one_electron_kernels.py");

int32_t overlap_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                      const int32_t n_ao, double *s) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const auto kernel = overlap_kernels[cls.la][cls.lb];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, cls.off_a,
                                            cls.off_b, s);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
