// The device half of free_breakfast.integrals.one_electron.potential_en: the
// launcher over the generated v_{la}{lb} kernels. One source for both backends
// -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are #included rather than compiled as sources of their
// own, as overlap_launch.cu does, but each inside a namespace of its own: every
// v_*.cu defines the same non-static `__device__ double fj_boys(int, double)`
// (FelizJueves's wide Boys function), so six of them at namespace scope in one
// TU would define it six times. A namespace keeps the emitted text untouched;
// the kernels' extern "C" names are distinct already, so it changes nothing
// about them. Each file's own `#include "felizjueves/runtime.h"` is a no-op
// here, this file having included it (it is #pragma once) before the first
// namespace opens -- which it must: that header includes the vendor runtime
// under HIP, and a first inclusion inside a namespace would declare the runtime
// there.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "potential_en_bridge.h"

#include "felizjueves/runtime.h"

namespace free_breakfast::device::potential_en_00 {
#include "kernels/v_00.cu"
} // namespace free_breakfast::device::potential_en_00
namespace free_breakfast::device::potential_en_01 {
#include "kernels/v_01.cu"
} // namespace free_breakfast::device::potential_en_01
namespace free_breakfast::device::potential_en_02 {
#include "kernels/v_02.cu"
} // namespace free_breakfast::device::potential_en_02
namespace free_breakfast::device::potential_en_11 {
#include "kernels/v_11.cu"
} // namespace free_breakfast::device::potential_en_11
namespace free_breakfast::device::potential_en_12 {
#include "kernels/v_12.cu"
} // namespace free_breakfast::device::potential_en_12
namespace free_breakfast::device::potential_en_22 {
#include "kernels/v_22.cu"
} // namespace free_breakfast::device::potential_en_22

namespace free_breakfast::device {

namespace {

// The generated kernels share one signature (FelizJueves's Target.V parameter
// block, its record arrays one PrimPairSoA, P among them), int32_t throughout
// as the overlap's.
using PotentialEnKernel = void (*)(int32_t, int32_t, int32_t, PrimPairSoA, const double *,
                                   const double *, double *);

// Indexed [la][lb]; the la > lb half is empty because the host swaps every
// shell pair into la <= lb (free_breakfast.integrals.mmd).
constexpr PotentialEnKernel kPotentialEnKernels[kMaxAngularMomentum + 1][kMaxAngularMomentum + 1] =
    {
        {potential_en_00::v_00, potential_en_01::v_01, potential_en_02::v_02},
        {nullptr, potential_en_11::v_11, potential_en_12::v_12},
        {nullptr, nullptr, potential_en_22::v_22},
};

} // namespace

int32_t potential_en_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                           const NucleiView &nuclei, const int32_t n_ao, double *v) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1 ||
      nuclei.n_nuc < 1) {
    return static_cast<int32_t>(wwr::wwrErrorInvalidValue);
  }
  const PotentialEnKernel kernel = kPotentialEnKernels[cls.la][cls.lb];
  // One thread per record, the nuclei serial inside it, kFourWarps to a block; a
  // whole number of warps keeps every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, nuclei.n_nuc, cls.recs, nuclei.z,
                                            nuclei.c, v);
  return static_cast<int32_t>(wwr::wwrGetLastError());
}

} // namespace free_breakfast::device
