// The device half of the kinetic energy in free_breakfast.integrals.one_electron:
// the launcher over the generated t_{la}{lb} kernels. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, as overlap_launch.cu does: the launches below see their definitions in
// the same TU.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). The bridge includes
// integral_types.h first.
#include "one_electron_bridge.h"

#include "t_kinetic/t_00.cu"
#include "t_kinetic/t_01.cu"
#include "t_kinetic/t_02.cu"
#include "t_kinetic/t_11.cu"
#include "t_kinetic/t_12.cu"
#include "t_kinetic/t_22.cu"

namespace free_breakfast::device {

namespace {

// One thread per record, as the overlap's.
constexpr int32_t kKineticBlockSize = 128;

// The generated kernels share one signature (FelizJueves's Target.T parameter
// block: Target.S's plus eb_ket), spelled `int` for int32_t as the overlap's.
using KineticKernel = void (*)(int32_t, int32_t, const double *, const double *, const double *,
                               const double *, const double *, const int32_t *, const int32_t *,
                               double *);

// Indexed [la][lb]; the la > lb half is empty because the host swaps every
// shell pair into la <= lb (free_breakfast.integrals.mmd).
constexpr KineticKernel kKineticKernels[kOneElectronMaxAngularMomentum + 1]
                                       [kOneElectronMaxAngularMomentum + 1] = {
                                           {t_00, t_01, t_02},
                                           {nullptr, t_11, t_12},
                                           {nullptr, nullptr, t_22},
};

} // namespace

int32_t kinetic_class(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                      const int32_t n_rec, const int32_t n_ao, const double *zeta,
                      const double *xpa, const double *xpb, const double *s_rec, const double *eb,
                      const int32_t *off_a, const int32_t *off_b, double *t) {
  if (la < 0 || lb < la || lb > kOneElectronMaxAngularMomentum || n_rec < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const KineticKernel kernel = kKineticKernels[la][lb];
  const int32_t blocks = (n_rec + kKineticBlockSize - 1) / kKineticBlockSize;
  kernel<<<blocks, kKineticBlockSize, 0, stream>>>(n_rec, n_ao, zeta, xpa, xpb, s_rec, eb, off_a,
                                                   off_b, t);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
