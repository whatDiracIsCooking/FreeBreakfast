// The device half of free_breakfast.integrals.coulomb.e_hermite: the launcher
// over the generated e_hermite<lp, lq> kernels. One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `e_hermite`.
// kernels/e_hermite.h declares the template, every specialization and
// e_hermite_kernels, their [lp][lq] table, so nothing here names a class: a
// class the table names but no unit defines fails to link. Launching a
// specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h includes
// its own first, as the bridge does integral_types.h; the kernels' header
// brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"

#include "e_hermite_bridge.h"

#include "kernels/e_hermite.h"

namespace free_breakfast::device {

// One Hermite class per lp, lq <= 2 kMaxAngularMomentum: a d-shell pair's E
// reaches |t| = 4.
constexpr auto kHermiteExtent = static_cast<size_t>(2 * kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(e_hermite_kernels), 0> == kHermiteExtent &&
                  std::extent_v<decltype(e_hermite_kernels), 1> == kHermiteExtent,
              "kernels/ was not emitted for kMaxAngularMomentum: rerun "
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

} // namespace

wwr::wwrError_t e_hermite_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &bra,
                                const PrimitivePairClassView &ket, const double tau,
                                const double *ip, const double *iq, const double *dq,
                                const double *dmax, double *jp) {
  if (!has_pair_kernel(bra) || !has_pair_kernel(ket)) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto kernel = e_hermite_kernels[bra.la + bra.lb][ket.la + ket.lb];
  // One warp per bra record, the kets strip-mined across its lanes; no dynamic
  // shared memory, J_p's accumulator is sized for this block in the text.
  const dim3 block(e_hermite_block_dim[0], e_hermite_block_dim[1]);
  kernel<<<static_cast<uint32_t>(bra.n_rec), block, 0, stream>>>(
      bra.n_rec, ket.n_rec, tau, bra.p_x, bra.p_y, bra.p_z, bra.zeta, ip, ket.p_x, ket.p_y, ket.p_z,
      ket.zeta, iq, dmax, dq, jp);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
