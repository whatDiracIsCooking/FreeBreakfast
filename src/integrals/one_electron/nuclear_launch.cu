// The device half of the nuclear attraction in
// free_breakfast.integrals.one_electron: the launcher over the generated
// v_{la}{lb} kernels. One source for both backends -- nvcc under CUDA, clang
// -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, as overlap_launch.cu does, but each inside a namespace of its own: every
// v_*.cu defines the same non-static `__device__ double fj_boys(int, double)`
// (FelizJueves's wide Boys function), so six of them at namespace scope in one
// TU would define it six times. A namespace keeps the emitted text untouched;
// the kernels' extern "C" names are distinct already, so it changes nothing
// about them. Each file's own `#include <runtime.h>` is a no-op here, the
// bridge having included it (it is #pragma once) before the first namespace
// opens.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). The bridge includes
// integral_types.h first.
#include "one_electron_bridge.h"

namespace free_breakfast::device::nuclear_00 {
#include "potential_en/kernels/v_00.cu"
} // namespace free_breakfast::device::nuclear_00
namespace free_breakfast::device::nuclear_01 {
#include "potential_en/kernels/v_01.cu"
} // namespace free_breakfast::device::nuclear_01
namespace free_breakfast::device::nuclear_02 {
#include "potential_en/kernels/v_02.cu"
} // namespace free_breakfast::device::nuclear_02
namespace free_breakfast::device::nuclear_11 {
#include "potential_en/kernels/v_11.cu"
} // namespace free_breakfast::device::nuclear_11
namespace free_breakfast::device::nuclear_12 {
#include "potential_en/kernels/v_12.cu"
} // namespace free_breakfast::device::nuclear_12
namespace free_breakfast::device::nuclear_22 {
#include "potential_en/kernels/v_22.cu"
} // namespace free_breakfast::device::nuclear_22

namespace free_breakfast::device {

namespace {

// One thread per record, as the overlap's. The nuclei are serial inside it.
constexpr int32_t kNuclearBlockSize = 128;

// The generated kernels share one signature (FelizJueves's Target.V parameter
// block), spelled `int` for int32_t as the overlap's.
using NuclearKernel = void (*)(int32_t, int32_t, int32_t, const double *, const double *,
                               const double *, const double *, const double *, const int32_t *,
                               const int32_t *, const double *, const double *, double *);

// Indexed [la][lb]; the la > lb half is empty because the host swaps every
// shell pair into la <= lb (free_breakfast.integrals.mmd).
constexpr NuclearKernel kNuclearKernels[kMaxAngularMomentum + 1][kMaxAngularMomentum + 1] = {
    {nuclear_00::v_00, nuclear_01::v_01, nuclear_02::v_02},
    {nullptr, nuclear_11::v_11, nuclear_12::v_12},
    {nullptr, nullptr, nuclear_22::v_22},
};

} // namespace

int32_t nuclear_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                      const NucleiView &nuclei, const int32_t n_ao, double *v) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1 ||
      nuclei.n_nuc < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const NuclearKernel kernel = kNuclearKernels[cls.la][cls.lb];
  const int32_t blocks = (cls.n_rec + kNuclearBlockSize - 1) / kNuclearBlockSize;
  kernel<<<blocks, kNuclearBlockSize, 0, stream>>>(cls.n_rec, n_ao, nuclei.n_nuc, cls.zeta, cls.p,
                                                   cls.xpa, cls.xpb, cls.s, cls.off_a, cls.off_b,
                                                   nuclei.z, nuclei.c, v);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
