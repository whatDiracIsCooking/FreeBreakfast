// The device half of the overlap in free_breakfast.integrals.one_electron: the
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
#include "one_electron_bridge.h"

#include "s_overlap/s_00.cu"
#include "s_overlap/s_01.cu"
#include "s_overlap/s_02.cu"
#include "s_overlap/s_11.cu"
#include "s_overlap/s_12.cu"
#include "s_overlap/s_22.cu"

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
constexpr OverlapKernel kOverlapKernels[kOneElectronMaxAngularMomentum + 1]
                                       [kOneElectronMaxAngularMomentum + 1] = {
                                           {s_00, s_01, s_02},
                                           {nullptr, s_11, s_12},
                                           {nullptr, nullptr, s_22},
};

} // namespace

int32_t overlap_class(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                      const int32_t n_rec, const int32_t n_ao, const double *zeta,
                      const double *xpa, const double *xpb, const double *s_rec,
                      const int32_t *off_a, const int32_t *off_b, double *s) {
  if (la < 0 || lb < la || lb > kOneElectronMaxAngularMomentum || n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const OverlapKernel kernel = kOverlapKernels[la][lb];
  const int32_t blocks = (n_rec + kOverlapBlockSize - 1) / kOverlapBlockSize;
  kernel<<<blocks, kOverlapBlockSize, 0, stream>>>(n_rec, n_ao, zeta, xpa, xpb, s_rec, off_a, off_b,
                                                   s);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
