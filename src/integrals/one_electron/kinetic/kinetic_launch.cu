// The device half of free_breakfast.integrals.one_electron.kinetic: the
// launcher over the generated t_{la}{lb} kernels. One source for both backends
// -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, as overlap_launch.cu does: the launches below see their definitions in
// the same TU.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "kinetic_bridge.h"

#include "kernels/t_00.cu"
#include "kernels/t_01.cu"
#include "kernels/t_02.cu"
#include "kernels/t_11.cu"
#include "kernels/t_12.cu"
#include "kernels/t_22.cu"

namespace free_breakfast::device {

namespace {

// The generated kernels share one signature (FelizJueves's Target.T parameter
// block: Target.S's plus eb_ket). Taken from a kernel, not transcribed, so it
// follows whatever the generator emits.
using KineticKernel = decltype(&t_00);

// Indexed [la][lb]; the la > lb half is empty because the host swaps every
// shell pair into la <= lb (free_breakfast.integrals.mmd).
constexpr KineticKernel kKineticKernels[kMaxAngularMomentum + 1][kMaxAngularMomentum + 1] = {
    {t_00, t_01, t_02},
    {nullptr, t_11, t_12},
    {nullptr, nullptr, t_22},
};

} // namespace

int32_t kinetic_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                      const int32_t n_ao, double *t) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const KineticKernel kernel = kKineticKernels[cls.la][cls.lb];
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, cls.eb, cls.off_a,
                                            cls.off_b, t);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
