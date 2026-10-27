// The device half of free_breakfast.integrals.eri.integral_direct.coulomb.dq:
// the launcher over the generated dq<la, lb> kernels, and the small kernel that
// makes the weights they leave to their caller. One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `dq`. kernels/dq.h
// declares the template, every specialization and dq_kernels, their [la][lb]
// table, so nothing here names a class: a class the table names but no unit
// defines fails to link. Launching a specialization another TU defines wants
// relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "dq_bridge.h"
#include "pair_class_bridge.h"

#include "kernels/dq.h"

namespace {

// A record's weight in the sum over unique shell pairs: a shell against itself
// (its two offsets equal, as mmd records it) is counted once, every other pair
// once for (c, d) and once for (d, c).
__global__ void dq_weights_kernel(const int32_t n_rec, const int32_t *off_a, const int32_t *off_b,
                                  double *w) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    w[k] = off_a[k] == off_b[k] ? 1.0 : 2.0;
  }
}

} // namespace

namespace free_breakfast::device {

// The largest momentum the emitted table covers: J's limit.
constexpr int32_t kMaxL = static_cast<int32_t>(std::extent_v<decltype(dq_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(dq_kernels), 1> == kMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/coulomb_kernels.py");

static_assert(kMaxL == kCoulombMaxMomentum,
              "kernels/'s table and kCoulombMaxMomentum disagree: rerun "
              "tools/emit/coulomb_kernels.py");

wwr::wwrError_t dq_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                         const int32_t n_ao, const double *density, double *w, double *dq,
                         double *dmax) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxL || cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  dq_weights_kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, cls.off_a, cls.off_b, w);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  const auto kernel = dq_kernels[cls.la][cls.lb];
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, w, cls.off_a,
                                            cls.off_b, density, dq, dmax);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
