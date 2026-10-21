// The device half of free_breakfast.integrals.exchange: the launcher over the
// generated kb<la, lc, lb, ld>, and the small kernels that make the per-record
// inputs it leaves to its caller (the rank-to-AO table, the bra's Ip with
// max|D| folded in, and its Dcd rows). One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/kb/),
// each one explicit specialization of the template kernels/kb/kb.h declares with
// its table, so nothing here names a class: a class the table names but no
// unit defines fails to link. Launching a specialization another TU defines
// wants relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "angular_momentum.h"
#include "block_params.h"
#include "primitives.h"

#include "exchange_bridge.h"

#include "kernels/kb/kb.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::size_t;
using fb::uint32_t;

// off[rank_b[r]] = off_b[r]: see exchange_rank_offsets.
__global__ void rank_offsets_kernel(const int32_t n_rec, const int32_t *rank_b,
                                    const int32_t *off_b, int32_t *off) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < n_rec) {
    off[rank_b[r]] = off_b[r];
  }
}

// ip[r] = bound[r] * *dmax: see exchange_bra_bound.
__global__ void bra_bound_kernel(const int32_t n_rec, const double *bound, const double *dmax,
                                 double *ip) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < n_rec) {
    ip[r] = bound[r] * *dmax;
  }
}

// dc[r] = rank_b[r] * n_d: see exchange_bra_rows.
__global__ void bra_rows_kernel(const int32_t n_rec, const int32_t *rank_b, const int32_t n_d,
                                int32_t *dc) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < n_rec) {
    dc[r] = rank_b[r] * n_d;
  }
}

constexpr size_t kExtent = static_cast<size_t>(fb::kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(kb_kernels), 0> == kExtent &&
                  std::extent_v<decltype(kb_kernels), 1> == kExtent &&
                  std::extent_v<decltype(kb_kernels), 2> == kExtent &&
                  std::extent_v<decltype(kb_kernels), 3> == kExtent,
              "kernels/kb was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_exchange_kernels.py");

// One warp per K_ab block: blockDim = (1, warp width), the block kb's Ksh was
// sized for. FJ_WAVEFRONT comes from WWR_WARP_SIZE (deps/CMakeLists.txt), so
// this holds unless the two have been split.
static_assert(kb_block_dim[0] == 1 && kb_block_dim[1] == fb::kWarpSize,
              "kb's block is not (1, kWarpSize): FJ_WAVEFRONT and WWR_WARP_SIZE disagree");

// kb's shared memory is static, `Ksh[(n_p + n_a n_b) * blockDim.y]` doubles,
// n_p = (lp + 1)(lp + 2)(lp + 3) / 6 Hermite functions of lp = la + lc and n_a
// n_b the class's Cartesian block; it is launched with no dynamic bytes. So
// the budget is checked on that formula at the largest class, (d d | d d):
// 71 slots, 18,176 B at 32 lanes, 36,352 B at 64. A class over 48 KiB would
// fail its launch (or, emitted, its compile) on a device without an opt-in.
constexpr size_t kSharedBudget = 48 * 1024;

constexpr size_t k_shared_bytes(const int32_t la, const int32_t lc, const int32_t lb) {
  const auto lp = static_cast<size_t>(la + lc);
  const auto na = static_cast<size_t>((la + 1) * (la + 2) / 2);
  const auto nb = static_cast<size_t>((lb + 1) * (lb + 2) / 2);
  return ((lp + 1) * (lp + 2) * (lp + 3) / 6 + na * nb) * kb_block_dim[1] * sizeof(double);
}

static_assert(k_shared_bytes(fb::kMaxAngularMomentum, fb::kMaxAngularMomentum,
                             fb::kMaxAngularMomentum) <= kSharedBudget,
              "kb's static shared memory at kMaxAngularMomentum is over 48 KiB");

bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= fb::kMaxAngularMomentum;
}

bool has_records(const fb::OrderedPairClassView &cls) {
  return valid_momentum(cls.la) && valid_momentum(cls.lb) && cls.n_rec >= 1;
}

// One thread per record, kFourWarps to a block.
uint32_t record_blocks(const int32_t n_rec) {
  return fb::math::idivup(static_cast<uint32_t>(n_rec), fb::kFourWarps);
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_rank_offsets(const wwr::wwrStream_t stream,
                                      const OrderedPairClassView &cls, int32_t *off) {
  if (!has_records(cls)) {
    return wwr::wwrErrorInvalidValue;
  }
  rank_offsets_kernel<<<record_blocks(cls.n_rec), kFourWarps, 0, stream>>>(cls.n_rec, cls.rank_b,
                                                                           cls.off_b, off);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_bra_bound(const wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                                   const double *dmax, double *ip) {
  if (!has_records(bra)) {
    return wwr::wwrErrorInvalidValue;
  }
  bra_bound_kernel<<<record_blocks(bra.n_rec), kFourWarps, 0, stream>>>(bra.n_rec, bra.bound, dmax,
                                                                        ip);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_bra_rows(const wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                                  const int32_t n_d, int32_t *dc) {
  if (!has_records(bra) || n_d < 0) {
    return wwr::wwrErrorInvalidValue;
  }
  bra_rows_kernel<<<record_blocks(bra.n_rec), kFourWarps, 0, stream>>>(bra.n_rec, bra.rank_b, n_d,
                                                                       dc);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_k(const wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                           const double *ip, const int32_t *dc, const OrderedPairClassView &ket,
                           const int32_t n_dc, const double tau, const double *dcd,
                           const int32_t n_blk, const int32_t *blocks, const int32_t n_ao,
                           double *k) {
  if (!has_records(bra) || !has_records(ket) || n_blk < 1 || n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  // The table is indexed as the template's parameters mean: bra (a, c), ket
  // (b, d).
  const auto kernel = kb_kernels[bra.la][bra.lb][ket.la][ket.lb];
  const auto seg = [&](const int32_t field) { return blocks + static_cast<size_t>(field) * n_blk; };
  // One warp per block; no dynamic shared memory, Ksh is sized for this block
  // in the text. maxD = 1: max|D| is in ip (exchange_bra_bound).
  const dim3 grid(static_cast<uint32_t>(n_blk));
  const dim3 lanes(kb_block_dim[0], kb_block_dim[1]);
  kernel<<<grid, lanes, 0, stream>>>(
      n_blk, bra.n_rec, ket.n_rec, n_dc, n_ao, tau, 1.0, seg(0), seg(1), seg(2), seg(3), seg(4),
      seg(5), bra.p_x, bra.p_y, bra.p_z, bra.zeta, bra.pa_x, bra.pa_y, bra.pa_z, bra.pb_x, bra.pb_y,
      bra.pb_z, bra.s, ip, dc, ket.p_x, ket.p_y, ket.p_z, ket.zeta, ket.pa_x, ket.pa_y, ket.pa_z,
      ket.pb_x, ket.pb_y, ket.pb_z, ket.s, ket.bound, ket.rank_b, dcd, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
