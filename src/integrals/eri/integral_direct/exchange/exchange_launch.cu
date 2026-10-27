// The device half of free_breakfast.integrals.eri.integral_direct.exchange: the
// launcher over the generated kb<la, lc, lb, ld> (every shell through d) and
// kbr<la, lc, lb, ld> (any shell f), and the small kernel that makes the
// rank-to-AO table it leaves to its caller (the bra's Dcd rows are the ordered
// classes', pairs_launch.cu). One source for both backends -- nvcc under CUDA,
// clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/kb/,
// kernels/kbr/), each one explicit specialization of the template its family's
// header declares with its table, so nothing here names a class: a class the
// table names but no unit defines fails to link. Launching a specialization
// another TU defines wants relocatable device code under nvcc (CMakeLists.txt).
// kbr's class tables (kernels/kbr/kbr_tables.h) are device arrays this TU
// alone defines.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "exchange_bridge.h"
#include "exchange_helpers_bridge.h"

#include "kernels/kb/kb.h"
#include "kernels/kbr/kbr.h"
#include "kernels/kbr/kbr_tables.h"

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

// The largest momentum each emitted table covers, its own limit rather than the
// tree's kMaxAngularMomentum: kbr's is K's, kb's where kbr takes over. The
// host's copies (exchange_helpers_bridge.h) are checked against them here.
constexpr size_t kExtent = std::extent_v<decltype(kbr_kernels), 0>;
constexpr int32_t kMaxL = static_cast<int32_t>(kExtent) - 1;
constexpr size_t kKbExtent = std::extent_v<decltype(kb_kernels), 0>;
constexpr int32_t kKbMaxL = static_cast<int32_t>(kKbExtent) - 1;

static_assert(std::extent_v<decltype(kbr_kernels), 1> == kExtent &&
                  std::extent_v<decltype(kbr_kernels), 2> == kExtent &&
                  std::extent_v<decltype(kbr_kernels), 3> == kExtent,
              "kernels/kbr's [la][lc][lb][ld] table is not square: rerun "
              "tools/emit/exchange_kernels.py");
static_assert(std::extent_v<decltype(kb_kernels), 1> == kKbExtent &&
                  std::extent_v<decltype(kb_kernels), 2> == kKbExtent &&
                  std::extent_v<decltype(kb_kernels), 3> == kKbExtent,
              "kernels/kb's [la][lc][lb][ld] table is not square: rerun "
              "tools/emit/exchange_kernels.py");
static_assert(kMaxL == fb::device::kExchangeMaxMomentum,
              "kernels/kbr's table and kExchangeMaxMomentum disagree");
static_assert(kKbMaxL == fb::device::kExchangeKbMaxMomentum,
              "kernels/kb's table and kExchangeKbMaxMomentum disagree");

// One warp per K_ab block: blockDim = (1, warp width), the block kb's and
// kbr's Ksh was sized for. FJ_WAVEFRONT comes from WWR_WARP_SIZE
// (deps/CMakeLists.txt), so this holds unless the two have been split.
static_assert(kb_block_dim[0] == 1 && kb_block_dim[1] == fb::kWarpSize,
              "kb's block is not (1, kWarpSize): FJ_WAVEFRONT and WWR_WARP_SIZE disagree");
static_assert(kbr_block_dim[0] == kb_block_dim[0] && kbr_block_dim[1] == kb_block_dim[1],
              "kbr's block is not kb's");

// kb's and kbr's shared memory is static, `Ksh[(n_p + n_a n_b) * blockDim.y]`
// doubles, n_p = (lp + 1)(lp + 2)(lp + 3) / 6 Hermite functions of lp = la + lc
// and n_a n_b the class's Cartesian block; it is launched with no dynamic
// bytes. So the budget is checked on that formula at the tables' largest
// class, (f f | f f): 184 slots, 47,104 B at 32 lanes, just under. At 64 lanes
// it is 94,208 B, which no device gives a static array: a wave64 build stops
// here. A class over 48 KiB would fail its launch (or, emitted, its compile)
// on a device without an opt-in.
constexpr size_t kSharedBudget = 48 * 1024;

constexpr size_t k_shared_bytes(const int32_t la, const int32_t lc, const int32_t lb) {
  const auto lp = static_cast<size_t>(la + lc);
  const auto na = static_cast<size_t>((la + 1) * (la + 2) / 2);
  const auto nb = static_cast<size_t>((lb + 1) * (lb + 2) / 2);
  return ((lp + 1) * (lp + 2) * (lp + 3) / 6 + na * nb) * kb_block_dim[1] * sizeof(double);
}

static_assert(k_shared_bytes(kMaxL, kMaxL, kMaxL) <= kSharedBudget,
              "kbr's static shared memory at its table's largest class is over 48 KiB");

bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= kMaxL;
}

bool has_records(const fb::OrderedPairClassView &cls) {
  return valid_momentum(cls.la) && valid_momentum(cls.lb) && cls.n_rec >= 1;
}

// Whether the class pair is kbr's: any shell past kb's table.
bool rolled(const fb::OrderedPairClassView &bra, const fb::OrderedPairClassView &ket) {
  return bra.la > kKbMaxL || bra.lb > kKbMaxL || ket.la > kKbMaxL || ket.lb > kKbMaxL;
}

// The device address of one of kbr_tables.h's arrays, as a launch argument.
wwr::wwrError_t table_address(const void *symbol, const int32_t *&at) {
  void *p = nullptr;
  const wwr::wwrError_t code = wwr::wwrGetSymbolAddress(&p, symbol);
  at = static_cast<const int32_t *>(p);
  return code;
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

wwr::wwrError_t exchange_k(const wwr::wwrStream_t stream, const OrderedPairClassView &bra,
                           const int32_t *dc, const OrderedPairClassView &ket, const int32_t n_dc,
                           const double tau, const double *dcd, const int32_t n_blk,
                           const int32_t *blocks, const double *factor, const int32_t n_ao,
                           double *k) {
  if (!has_records(bra) || !has_records(ket) || n_blk < 1 || n_ao < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto seg = [&](const int32_t field) { return blocks + static_cast<size_t>(field) * n_blk; };
  // One warp per block; no dynamic shared memory, Ksh is sized for this block
  // in the text. Ip and Iq are the records' own dressed bounds, the density in
  // factor. The tables are indexed as the templates' parameters mean: bra
  // (a, c), ket (b, d).
  const dim3 grid(static_cast<uint32_t>(n_blk));
  const dim3 lanes(kb_block_dim[0], kb_block_dim[1]);
  if (!rolled(bra, ket)) {
    const auto kernel = kb_kernels[bra.la][bra.lb][ket.la][ket.lb];
    kernel<<<grid, lanes, 0, stream>>>(
        n_blk, bra.n_rec, ket.n_rec, n_dc, n_ao, tau, seg(0), seg(1), seg(2), seg(3), seg(4),
        seg(5), factor, bra.p_x, bra.p_y, bra.p_z, bra.zeta, bra.pa_x, bra.pa_y, bra.pa_z, bra.pb_x,
        bra.pb_y, bra.pb_z, bra.s, bra.record_bound, dc, ket.p_x, ket.p_y, ket.p_z, ket.zeta,
        ket.pa_x, ket.pa_y, ket.pa_z, ket.pb_x, ket.pb_y, ket.pb_z, ket.s, ket.record_bound,
        ket.rank_b, dcd, k);
    return wwr::wwrGetLastError();
  }
  // kbr reads E, the bra's as Eac and the ket's signed as Ebd, and its class
  // tables: the bra pair's E pattern as ac_*, the ket pair's as bd_*, and hpq
  // of (lp, lq).
  if (bra.e_bra == nullptr || ket.e_ket == nullptr) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t *e_ptr = nullptr;
  const int32_t *e_col = nullptr;
  const int32_t *hpq = nullptr;
  if (const wwr::wwrError_t code = table_address(kbr_e_ptr, e_ptr); code != wwr::wwrSuccess) {
    return code;
  }
  if (const wwr::wwrError_t code = table_address(kbr_e_col, e_col); code != wwr::wwrSuccess) {
    return code;
  }
  if (const wwr::wwrError_t code = table_address(kbr_hpq, hpq); code != wwr::wwrSuccess) {
    return code;
  }
  const auto kernel = kbr_kernels[bra.la][bra.lb][ket.la][ket.lb];
  kernel<<<grid, lanes, 0, stream>>>(
      n_blk, bra.n_rec, ket.n_rec, n_dc, n_ao, tau, seg(0), seg(1), seg(2), seg(3), seg(4), seg(5),
      factor, bra.p_x, bra.p_y, bra.p_z, bra.zeta, bra.s, bra.record_bound, dc, ket.p_x, ket.p_y,
      ket.p_z, ket.zeta, ket.s, ket.record_bound, ket.rank_b, bra.e_bra, ket.e_ket,
      e_ptr + kbr_e_ptr_at[bra.la][bra.lb], e_col + kbr_e_col_at[bra.la][bra.lb],
      e_ptr + kbr_e_ptr_at[ket.la][ket.lb], e_col + kbr_e_col_at[ket.la][ket.lb],
      hpq + kbr_hpq_at[bra.la + bra.lb][ket.la + ket.lb], dcd, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
