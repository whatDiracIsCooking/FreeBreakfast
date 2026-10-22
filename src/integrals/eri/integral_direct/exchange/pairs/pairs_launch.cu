// The device half of
// free_breakfast.integrals.eri.integral_direct.exchange.pairs: an ordered
// class's order (each group sorted by mmd's dressed bound of the orientation)
// and its gather, straight from the mmd class it is drawn from, once per basis.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "merge_sort_bridge.h"
#include "pairs_bridge.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::size_t;
using fb::uint32_t;

// Which orientations of an mmd record an ordered class holds, and so how a
// slot names one (pairs_bridge.h).
enum class Orientation : int32_t {
  Stored,  // slot r: record r as stored
  Swapped, // slot r: record r swapped
  Both,    // slot 2r: record r as stored; slot 2r + 1: it swapped
};

// One slot's record, and whether it is swapped.
struct Slot {
  int32_t rec;
  bool swapped;
};

__device__ Slot slot_of(const Orientation o, const uint32_t s) {
  if (o == Orientation::Both) {
    return {static_cast<int32_t>(s / 2), (s & 1U) != 0};
  }
  return {static_cast<int32_t>(s), o == Orientation::Swapped};
}

// The orientations of the ordered (la, lb) of from, or false if la is neither
// of from's momenta.
bool orientation_of(const fb::PrimitivePairClassView &from, const int32_t la, Orientation &o) {
  if (from.la == from.lb && la == from.la) {
    o = Orientation::Both;
  } else if (la == from.la) {
    o = Orientation::Stored;
  } else if (la == from.lb) {
    o = Orientation::Swapped;
  } else {
    return false;
  }
  return true;
}

uint32_t slot_count(const fb::PrimitivePairClassView &from, const Orientation o) {
  return static_cast<uint32_t>(from.n_rec) * (o == Orientation::Both ? 2U : 1U);
}

// key[s] = -bound of slot s and perm[s] = s: the ascending sort of key is the
// descending order of bound, carrying each slot. A diagonal record's swapped
// slot gets one too; the group sort moves it past every record.
__global__ void bound_key_kernel(const fb::PrimitivePairClassView from, const Orientation o,
                                 const uint32_t n_slots, double *key, uint32_t *perm) {
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < n_slots) {
    const Slot at = slot_of(o, s);
    key[s] = -(at.swapped ? from.bound_ba : from.bound_ab)[at.rec];
    perm[s] = s;
  }
}

// key[k] = the rank of slot perm[k]'s first shell, its group; n_groups for the
// swapped slot of a diagonal record, which does not exist.
__global__ void group_key_kernel(const fb::PrimitivePairClassView from, const Orientation o,
                                 const uint32_t n_slots, const int32_t n_groups,
                                 const uint32_t *perm, double *key) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n_slots) {
    const Slot at = slot_of(o, perm[k]);
    const bool absent = at.swapped && from.off_a[at.rec] == from.off_b[at.rec];
    key[k] = static_cast<double>(absent       ? n_groups
                                 : at.swapped ? from.rank_b[at.rec]
                                              : from.rank_a[at.rec]);
  }
}

// group_start[g] = the first k whose sorted group key is g or above, by
// bisection over the n_slots keys: g's first record, and for g == n_groups
// the record count, the absent slots all sorting past it.
__global__ void group_start_kernel(const uint32_t n_slots, const int32_t n_groups,
                                   const double *key, int32_t *group_start) {
  const auto g = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (g <= n_groups) {
    uint32_t lo = 0;
    uint32_t hi = n_slots;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (key[mid] < static_cast<double>(g)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    group_start[g] = static_cast<int32_t>(lo);
  }
}

// The ordered class's fields, as the gather writes them (pairs_bridge.h's
// order), passed by value.
struct OrderedFields {
  double *doubles[12];
  int32_t *ints[3];
};

// Record k of the ordered class is slot perm[k]: one thread per record, the
// stores coalesced.
__global__ void gather_kernel(const fb::PrimitivePairClassView from, const Orientation o,
                              const int32_t n_rec, const uint32_t *perm, const OrderedFields to) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k >= n_rec) {
    return;
  }
  const Slot at = slot_of(o, perm[k]);
  const int32_t r = at.rec;
  const bool sw = at.swapped;
  to.doubles[0][k] = from.zeta[r];
  to.doubles[1][k] = from.p_x[r];
  to.doubles[2][k] = from.p_y[r];
  to.doubles[3][k] = from.p_z[r];
  // The first shell's P - A, offset and the second's rank are mmd's ket's,
  // and its bound the swapped one, when swapped.
  to.doubles[4][k] = (sw ? from.pb_x : from.pa_x)[r];
  to.doubles[5][k] = (sw ? from.pb_y : from.pa_y)[r];
  to.doubles[6][k] = (sw ? from.pb_z : from.pa_z)[r];
  to.doubles[7][k] = (sw ? from.pa_x : from.pb_x)[r];
  to.doubles[8][k] = (sw ? from.pa_y : from.pb_y)[r];
  to.doubles[9][k] = (sw ? from.pa_z : from.pb_z)[r];
  to.doubles[10][k] = from.s[r];
  to.doubles[11][k] = (sw ? from.bound_ba : from.bound_ab)[r];
  to.ints[0][k] = (sw ? from.off_b : from.off_a)[r];
  to.ints[1][k] = (sw ? from.off_a : from.off_b)[r];
  to.ints[2][k] = (sw ? from.rank_a : from.rank_b)[r];
}

// One thread per item, kFourWarps to a block.
uint32_t blocks_of(const uint32_t n) {
  return fb::math::idivup(n, fb::kFourWarps);
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t ordered_pair_order(const wwr::wwrStream_t stream,
                                   const PrimitivePairClassView &from, const int32_t la,
                                   const int32_t n_groups, double *key, double *key_alt,
                                   uint32_t *perm, uint32_t *perm_alt, int32_t *group_start) {
  Orientation o{};
  if (from.n_rec < 1 || n_groups < 1 || !orientation_of(from, la, o)) {
    return wwr::wwrErrorInvalidValue;
  }
  const uint32_t n = slot_count(from, o);
  bound_key_kernel<<<blocks_of(n), kFourWarps, 0, stream>>>(from, o, n, key, perm);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  if (const auto code =
          static_cast<wwr::wwrError_t>(merge_sort_passes(stream, n, key, key_alt, perm, perm_alt));
      code != wwr::wwrSuccess) {
    return code;
  }
  group_key_kernel<<<blocks_of(n), kFourWarps, 0, stream>>>(from, o, n, n_groups, perm, key);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  if (const auto code =
          static_cast<wwr::wwrError_t>(merge_sort_passes(stream, n, key, key_alt, perm, perm_alt));
      code != wwr::wwrSuccess) {
    return code;
  }
  group_start_kernel<<<blocks_of(static_cast<uint32_t>(n_groups) + 1U), kFourWarps, 0, stream>>>(
      n, n_groups, key, group_start);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ordered_pair_gather(const wwr::wwrStream_t stream,
                                    const PrimitivePairClassView &from, const int32_t la,
                                    const int32_t n_rec, const uint32_t *perm,
                                    double *const doubles[12], int32_t *const ints[3]) {
  Orientation o{};
  if (n_rec < 1 || !orientation_of(from, la, o)) {
    return wwr::wwrErrorInvalidValue;
  }
  OrderedFields to{};
  for (size_t f = 0; f < 12; ++f) {
    to.doubles[f] = doubles[f];
  }
  for (size_t f = 0; f < 3; ++f) {
    to.ints[f] = ints[f];
  }
  gather_kernel<<<blocks_of(static_cast<uint32_t>(n_rec)), kFourWarps, 0, stream>>>(from, o, n_rec,
                                                                                    perm, to);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
