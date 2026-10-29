// The device half of free_breakfast.integrals.one_electron.dipole: the
// launcher over the generated dipole<la, lb> kernels. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `dipole`.
// kernels/dipole.h declares the template, every specialization and
// dipole_kernels, their [la][lb] table, so nothing here names a class: a class
// the table names but no unit defines fails to link. Launching a
// specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h,
// constants.h and primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "dipole_bridge.h"

#include "kernels/dipole.h"

namespace free_breakfast::device {

// The largest momentum the emitted table covers: d, below the other families'
// f (tools/emit/one_electron_kernels.py's MU_MAX_L). The module's kDipoleMaxL
// refuses an f class before anything is launched; this bound is what keeps a
// launch inside the table regardless.
constexpr int32_t kMaxL = static_cast<int32_t>(std::extent_v<decltype(dipole_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(dipole_kernels), 1> == kMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/one_electron_kernels.py");

wwr::wwrError_t dipole_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                             const int32_t n_ao, double *mu) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxL || cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = dipole_kernels[cls.la][cls.lb];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step. The kernels take the origin C and the charge q
  // at run time; this dipole is <a| r |b> about the origin: C = 0, q = 1.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.p_x, cls.p_y, cls.p_z,
                                            cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
                                            cls.pb_z, cls.s, cls.off_a, cls.off_b, kZero<double>,
                                            kZero<double>, kZero<double>, kOne<double>, mu);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
