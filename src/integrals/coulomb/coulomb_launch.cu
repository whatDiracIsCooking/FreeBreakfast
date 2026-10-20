// The device half of free_breakfast.integrals.coulomb: the launchers over the
// generated dq<la, lb>, e_hermite<lp, lq> and jab<la, lb> kernels, and the two
// small kernels that make their inputs a build without screening has no values
// for. One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of its family's template. Each family's
// header declares the template, every specialization and its table, so nothing
// here names a class: a class the table names but no unit defines fails to
// link. Launching a specialization another TU defines wants relocatable device
// code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' headers bring the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "coulomb_bridge.h"

#include "kernels/dq/dq.h"
#include "kernels/e_hermite/e_hermite.h"
#include "kernels/jab/jab.h"

namespace {

// ones[k] = 1, rows[k] = k: see coulomb_constants.
__global__ void coulomb_constants_kernel(const int32_t n, double *ones, int32_t *rows) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n) {
    ones[k] = 1.0;
    rows[k] = k;
  }
}

// A record's weight in the sum over unique shell pairs: a shell against itself
// (its two offsets equal, as mmd records it) is counted once, every other pair
// once for (c, d) and once for (d, c).
__global__ void coulomb_weights_kernel(const int32_t n_rec, const int32_t *off_a,
                                       const int32_t *off_b, double *w) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    w[k] = off_a[k] == off_b[k] ? 1.0 : 2.0;
  }
}

} // namespace

namespace free_breakfast::device {

// One pair class per la <= lb <= kMaxAngularMomentum, and one Hermite class per
// lp, lq <= 2 kMaxAngularMomentum: a d-shell pair's E reaches |t| = 4.
constexpr auto kPairExtent = static_cast<size_t>(kMaxAngularMomentum + 1);
constexpr auto kHermiteExtent = static_cast<size_t>(2 * kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(dq_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(dq_kernels), 1> == kPairExtent,
              "kernels/dq was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(jab_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(jab_kernels), 1> == kPairExtent,
              "kernels/jab was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(e_hermite_kernels), 0> == kHermiteExtent &&
                  std::extent_v<decltype(e_hermite_kernels), 1> == kHermiteExtent,
              "kernels/e_hermite was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");

// One warp per bra record: blockDim = (1, warp width), the block e_hermite's
// Jsh[n_p * FJ_WAVEFRONT] was sized for. FJ_WAVEFRONT comes from WWR_WARP_SIZE
// (deps/CMakeLists.txt), so this holds unless the two have been split.
static_assert(e_hermite_block_dim[0] == 1 && e_hermite_block_dim[1] == kWarpSize,
              "e_hermite's block is not (1, kWarpSize): FJ_WAVEFRONT and "
              "WWR_WARP_SIZE disagree");

namespace {

bool has_pair_kernel(const PrimitivePairClassView &cls) {
  return cls.la >= 0 && cls.lb >= cls.la && cls.lb <= kMaxAngularMomentum && cls.n_rec >= 1;
}

// One thread per record, kFourWarps to a block; a whole number of warps keeps
// every block's lanes in step.
uint32_t record_blocks(const int32_t n_rec) {
  return math::idivup(static_cast<uint32_t>(n_rec), kFourWarps);
}

} // namespace

wwr::wwrError_t coulomb_constants(const wwr::wwrStream_t stream, const int32_t n, double *ones,
                                  int32_t *rows) {
  if (n < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  coulomb_constants_kernel<<<record_blocks(n), kFourWarps, 0, stream>>>(n, ones, rows);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_dq(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                           const int32_t n_ao, const double *density, double *w, double *dq,
                           double *dmax) {
  if (!has_pair_kernel(cls)) {
    return wwr::wwrErrorInvalidValue;
  }
  const uint32_t blocks = record_blocks(cls.n_rec);
  coulomb_weights_kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, cls.off_a, cls.off_b, w);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  const auto kernel = dq_kernels[cls.la][cls.lb];
  kernel<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z,
                                            cls.pb_x, cls.pb_y, cls.pb_z, cls.s, w, cls.off_a,
                                            cls.off_b, density, dq, dmax);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_e_hermite(const wwr::wwrStream_t stream, const PrimitivePairClassView &bra,
                                  const PrimitivePairClassView &ket, const double *ones,
                                  const double *dq, const double *dmax, double *jp) {
  if (!has_pair_kernel(bra) || !has_pair_kernel(ket)) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = e_hermite_kernels[bra.la + bra.lb][ket.la + ket.lb];
  // One warp per bra record, the kets strip-mined across its lanes; no dynamic
  // shared memory, J_p's accumulator is sized for this block in the text.
  const dim3 block(e_hermite_block_dim[0], e_hermite_block_dim[1]);
  constexpr double kTau = 0.0; // unscreened
  kernel<<<static_cast<uint32_t>(bra.n_rec), block, 0, stream>>>(
      bra.n_rec, ket.n_rec, kTau, bra.p_x, bra.p_y, bra.p_z, bra.zeta, ones, ket.p_x, ket.p_y,
      ket.p_z, ket.zeta, ones, dmax, dq, jp);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t coulomb_jab(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                            const int32_t n_ao, const int32_t *rows, const double *jp, double *j) {
  if (!has_pair_kernel(cls)) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = jab_kernels[cls.la][cls.lb];
  kernel<<<record_blocks(cls.n_rec), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, cls.n_rec, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
      cls.pb_z, cls.s, cls.off_a, cls.off_b, rows, jp, j);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
