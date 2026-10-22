// The device half of
// free_breakfast.integrals.eri.integral_direct.coulomb.j_coulomb: the launcher
// over the generated jab<la, lb> kernels, and the small kernel that makes the
// row map they leave to their caller. One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `jab`.
// kernels/jab.h declares the template, every specialization and jab_kernels,
// their [la][lb] table, so nothing here names a class: a class the table
// names but no unit defines fails to link. Launching a specialization another
// TU defines wants relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "j_coulomb_bridge.h"

#include "kernels/jab.h"

namespace {

// rows[k] = k: see j_coulomb_rows.
__global__ void j_coulomb_rows_kernel(const int32_t n, int32_t *rows) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n) {
    rows[k] = k;
  }
}

} // namespace

namespace free_breakfast::device {

static_assert(std::extent_v<decltype(jab_kernels), 0> == kMaxAngularMomentum + 1 &&
                  std::extent_v<decltype(jab_kernels), 1> == kMaxAngularMomentum + 1,
              "kernels/ was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");

namespace {

// One thread per record, kFourWarps to a block; a whole number of warps keeps
// every block's lanes in step.
uint32_t record_blocks(const int32_t n_rec) {
  return math::idivup(static_cast<uint32_t>(n_rec), kFourWarps);
}

} // namespace

wwr::wwrError_t j_coulomb_rows(const wwr::wwrStream_t stream, const int32_t n, int32_t *rows) {
  if (n < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  j_coulomb_rows_kernel<<<record_blocks(n), kFourWarps, 0, stream>>>(n, rows);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t j_coulomb_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                const int32_t n_ao, const int32_t *rows, const double *jp,
                                double *j) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum || cls.n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = jab_kernels[cls.la][cls.lb];
  kernel<<<record_blocks(cls.n_rec), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, cls.n_rec, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
      cls.pb_z, cls.s, cls.off_a, cls.off_b, rows, jp, j);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
