// The device half of free_breakfast.integrals.one_electron.overlap_mixed: the
// launcher over the generated overlap_mixed<la, lb> kernels. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `overlap_mixed`, one
// per ORDERED class. kernels/overlap_mixed.h declares the template, every
// specialization and overlap_mixed_kernels, their full [la][lb] table.
// Launching a specialization another TU defines wants relocatable device code
// under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "overlap_mixed_bridge.h"

#include "kernels/overlap_mixed.h"

namespace free_breakfast::device {

// The largest momentum the emitted table covers, its own limit rather than
// the tree's kMaxAngularMomentum, so this engine moves to a higher l when its
// kernels are emitted for one.
constexpr int32_t kMaxL =
    static_cast<int32_t>(std::extent_v<decltype(overlap_mixed_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(overlap_mixed_kernels), 1> == kMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/one_electron_kernels.py");

wwr::wwrError_t overlap_mixed_class(const wwr::wwrStream_t stream,
                                    const PrimitivePairClassView &cls, const int32_t n_rows,
                                    const int32_t n_cols, double *s) {
  if (cls.la < 0 || cls.la > kMaxL || cls.lb < 0 || cls.lb > kMaxL || cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  // The class swapped (the bridge has the why): the columns' shell is the
  // kernel's row, so its row-major S is column-major in the rows.
  const auto kernel = overlap_mixed_kernels[cls.lb][cls.la];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_cols, n_rows, cls.zeta, cls.pb_x, cls.pb_y,
                                            cls.pb_z, cls.pa_x, cls.pa_y, cls.pa_z, cls.s,
                                            cls.off_b, cls.off_a, s);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
