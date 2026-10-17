// The device half of free_breakfast.integrals.one_electron.overlap: the
// launcher over the generated s_{la}{lb} kernels. One source for both backends
// -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, so the launches below see their definitions in the same TU: no
// cross-TU kernel declaration to keep in step with the generator, and their
// extern "C" names reach no other TU of this library.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). The bridge includes
// integral_types.h first.
#include "overlap_bridge.h"

#include "kernels/s_00.cu"
#include "kernels/s_01.cu"
#include "kernels/s_02.cu"
#include "kernels/s_11.cu"
#include "kernels/s_12.cu"
#include "kernels/s_22.cu"

namespace free_breakfast::device {

namespace {

// One thread per record, so any size works; a multiple of the warp and the
// wavefront keeps every block's lanes in step.
constexpr int32_t kOverlapBlockSize = 128;

// The generated kernels share one signature (FelizJueves's Target.S parameter
// block), which is what lets one table cover every class. They spell their
// integers `int`, which int32_t is on every target this tree builds for
// (docs/style.md "Integer types"), so the kernels convert to this type exactly.
using OverlapKernel = void (*)(int32_t, int32_t, const double *, const double *, const double *,
                               const double *, const int32_t *, const int32_t *, double *);

// Indexed [la][lb]; the la > lb half is empty because the host swaps every
// shell pair into la <= lb (free_breakfast.integrals.mmd).
constexpr OverlapKernel kOverlapKernels[kMaxAngularMomentum + 1][kMaxAngularMomentum + 1] = {
    {s_00, s_01, s_02},
    {nullptr, s_11, s_12},
    {nullptr, nullptr, s_22},
};

} // namespace

int32_t overlap_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                      const int32_t n_ao, double *s) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const OverlapKernel kernel = kOverlapKernels[cls.la][cls.lb];
  const int32_t blocks = (cls.n_rec + kOverlapBlockSize - 1) / kOverlapBlockSize;
  kernel<<<blocks, kOverlapBlockSize, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.xpa, cls.xpb,
                                                   cls.s, cls.off_a, cls.off_b, s);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
