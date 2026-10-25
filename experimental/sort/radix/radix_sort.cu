// The device half of free_breakfast.experimental.sort.radix: the three kernels
// of one LSD radix pass, and the launcher that runs every pass. One source for
// both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The keys are doubles, but the kernels never see one: they move each key's 64
// bits as a uint64_t and take digits from ordered_bits() of it, a bijection
// whose unsigned order is the doubles' IEEE-754 total order. Remapping on every
// read rather than in place costs two ALU ops a key and saves an encode and a
// decode pass over memory.
//
// A pass over the digit (ordered_bits(key) >> shift) & (kRadixBuckets - 1):
//
//   1. count   -- each block counts its tile's keys per digit with
//                 block_reduce, into counts[digit * num_tiles + tile].
//   2. scan    -- one block turns counts, in that digit-major order, into an
//                 exclusive prefix sum: each (digit, tile) slot becomes where
//                 that tile's keys of that digit start in the output.
//   3. scatter -- each block ranks its keys among the tile's keys of the same
//                 digit (an exclusive sum of "is my digit" flags per digit) and
//                 writes each to its slot's start plus its rank.
//
// The rank preserves input order within a digit, so every pass is stable,
// which is what makes LSD correct.

// Standard headers, then block_reduce.cuh, which orders its own standard
// headers ahead of the vendor runtime header: under HIP that header must not
// precede libc++.
#include <utility>

#include "block_reduce.cuh"

#include "block_scan.cuh"
#include "radix_sort_bridge.h"

namespace free_breakfast::device {

namespace {

constexpr uint32_t kTileWarps = kRadixSortTile / kWarpSize;
// The scan runs as one block, as wide as a block can be.
constexpr uint32_t kScanWarps = 1024 / kWarpSize;
constexpr uint32_t kScanThreads = kScanWarps * kWarpSize;
constexpr uint32_t kDigitMask = kRadixBuckets - 1;
constexpr uint64_t kSignBit = uint64_t{1} << 63;

struct AddOp {
  __device__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a + b; }
};

// A double's bits, remapped so that unsigned order is IEEE-754 total order. A
// non-negative gets its sign bit set, which puts it above every negative and
// keeps its own magnitude order. A negative has every bit flipped, which puts it
// below and reverses its magnitude order, since a larger magnitude is the
// smaller number. NaNs land past the infinities on their sign's side.
__device__ __forceinline__ uint64_t ordered_bits(const uint64_t bits) {
  return bits ^ ((bits & kSignBit) != 0 ? ~uint64_t{0} : kSignBit);
}

// This thread's key digit, or kRadixBuckets -- which matches no digit -- past
// the end of the keys.
__device__ __forceinline__ uint32_t tile_digit(const uint64_t *keys, const uint32_t n,
                                               const uint32_t shift) {
  const uint32_t i = blockIdx.x * kRadixSortTile + threadIdx.x;
  return i < n ? static_cast<uint32_t>(ordered_bits(keys[i]) >> shift) & kDigitMask : kRadixBuckets;
}

__global__ void radix_count_kernel(const uint64_t *keys, const uint32_t n, const uint32_t shift,
                                   uint32_t *counts) {
  __shared__ uint32_t partials[kTileWarps];
  const cg::thread_block block = cg::this_thread_block();
  const uint32_t tile_start = blockIdx.x * kRadixSortTile;
  // The last tile can be partial; nactive keeps the threads past n out of the
  // fold, so their value does not matter.
  const uint32_t nactive = n - tile_start < kRadixSortTile ? n - tile_start : kRadixSortTile;
  const uint32_t digit = tile_digit(keys, n, shift);
  for (uint32_t d = 0; d < kRadixBuckets; ++d) {
    const uint32_t count =
        block_reduce<kTileWarps>(block, partials, digit == d ? 1U : 0U, AddOp{}, nactive);
    if (threadIdx.x == 0) {
      counts[d * gridDim.x + blockIdx.x] = count;
    }
    // The next block_reduce rewrites partials.
    block.sync();
  }
}

// One block; walks the counts in kScanThreads-wide chunks, carrying the running
// total between them.
__global__ void radix_scan_kernel(uint32_t *counts, const uint32_t len) {
  __shared__ uint32_t warp_totals[kScanWarps];
  const cg::thread_block block = cg::this_thread_block();
  uint32_t carry = 0;
  for (uint32_t base = 0; base < len; base += kScanThreads) {
    const uint32_t i = base + threadIdx.x;
    const uint32_t v = i < len ? counts[i] : 0;
    uint32_t chunk_total = 0;
    const uint32_t prefix = block_exclusive_sum<kScanWarps>(block, warp_totals, v, chunk_total);
    if (i < len) {
      counts[i] = carry + prefix;
    }
    carry += chunk_total;
    block.sync();
  }
}

__global__ void radix_scatter_kernel(const uint64_t *keys_in, const uint32_t *values_in,
                                     const uint32_t n, const uint32_t shift,
                                     const uint32_t *offsets, uint64_t *keys_out,
                                     uint32_t *values_out) {
  __shared__ uint32_t warp_totals[kTileWarps];
  const cg::thread_block block = cg::this_thread_block();
  const uint32_t digit = tile_digit(keys_in, n, shift);
  // Every thread joins every digit's scan (they are collectives) and keeps the
  // rank for its own. A thread past n has digit kRadixBuckets, so it flags 0 in
  // every scan and ranks nowhere.
  uint32_t rank = 0;
  for (uint32_t d = 0; d < kRadixBuckets; ++d) {
    uint32_t digit_total = 0;
    const uint32_t prefix =
        block_exclusive_sum<kTileWarps>(block, warp_totals, digit == d ? 1U : 0U, digit_total);
    if (digit == d) {
      rank = prefix;
    }
    block.sync();
  }
  if (digit < kRadixBuckets) {
    const uint32_t i = blockIdx.x * kRadixSortTile + threadIdx.x;
    const uint32_t dst = offsets[digit * gridDim.x + blockIdx.x] + rank;
    keys_out[dst] = keys_in[i];
    if (values_in != nullptr) {
      values_out[dst] = values_in[i];
    }
  }
}

// ---------------------------------------------------------------------------
// The small path, up to kSmallSortMaxKeys keys: tile sort, then merge.
// ---------------------------------------------------------------------------
// Below ~16k keys the multi-kernel path above is all launch overhead (48
// launches), and the vendor sort's is several. This path is one launch up to
// kTileKeys keys, and one more per merge round above that (at most 4).
//
// The tile is 1024 keys (256 threads of 4) because that measured fastest: a
// 2048-key tile took 24 us where two 1024-key tiles and a merge took ~20 (RTX
// 3080), since a pass's fixed cost of zeroing, scanning and syncing the
// histogram grows with the warps in the block. See the README.
//
//
//   1. radix_sort_tile_kernel: each block sorts a tile of up to kTileKeys keys
//      on chip. The keys (and values) stay in registers through all 8 passes
//      of 8-bit digits; each pass ranks them stably and exchanges them through
//      shared memory, so global memory is read once and written once. A pass
//      whose digit is the same for every key in the tile is skipped.
//   2. merge_runs_kernel: stable merge-path merges of sorted runs, doubling the
//      run length each round until one run covers n.
//
// Ranking is a warp multisplit: kSmallBits ballots give each lane the mask of
// lanes holding its digit, so a key's rank among its warp's keys of that digit
// is a running per-(digit, warp) count plus the popcount of the lower lanes in
// that mask. No atomics, so the result is deterministic. One block-wide
// exclusive scan of the per-(digit, warp) counts in digit-major order then
// gives each warp's start per digit.
//
// In a tile, warp w owns slots [w * kWarpSize * kTileItems, ...); lane l's item
// k is slot w * kWarpSize * kTileItems + k * kWarpSize + l. So (warp, item,
// lane) order is input order, which is the order ranks are handed out in: the
// scan orders warps, the running count orders items, the popcount lanes. Slots
// past the tile's last key hold kPadKey, which sorts after every real key
// (stably, so after a real all-ones key too) and is never written out.

constexpr uint32_t kSmallBits = 8;
constexpr uint32_t kSmallBuckets = 1U << kSmallBits;
constexpr uint32_t kSmallDigitMask = kSmallBuckets - 1;
constexpr uint32_t kTileThreads = 256;
constexpr uint32_t kTileMaxWarps = kTileThreads / kWarpSize;
constexpr uint32_t kTileItems = 4;
constexpr uint32_t kTileKeys = kTileThreads * kTileItems;
constexpr uint32_t kSmallSortMaxTiles = 16;
constexpr uint32_t kSmallSortMaxKeys = kTileKeys * kSmallSortMaxTiles;
// Slots of the (digit, warp) histogram each thread scans, whatever the block
// size: 8, or 4 on wave64.
constexpr uint32_t kSmallSlotsPerThread = kSmallBuckets / kWarpSize;
constexpr uint32_t kMergeItems = 8;
constexpr uint32_t kMergeThreads = 128;
constexpr uint64_t kPadKey = ~uint64_t{0};

static_assert(kRadixKeyBits % kSmallBits == 0);
static_assert(kSmallBuckets % kWarpSize == 0);
static_assert(kTileKeys % kMergeItems == 0, "a merge thread's outputs never straddle two runs");

// ordered_bits' inverse: a set top bit marks a non-negative, whose sign bit is
// cleared again; a clear one marks a negative, whose bits are flipped back.
__device__ __forceinline__ uint64_t unordered_bits(const uint64_t ordered) {
  return ordered ^ ((ordered & kSignBit) != 0 ? kSignBit : ~uint64_t{0});
}

// The number of set lanes in a ballot mask, which is 32 bits under CUDA and 64
// under HIP.
template <typename LaneMask>
__device__ __forceinline__ uint32_t lane_count(const LaneMask mask) {
  if constexpr (sizeof(LaneMask) == sizeof(uint32_t)) {
    return static_cast<uint32_t>(__popc(mask));
  } else {
    return static_cast<uint32_t>(__popcll(mask));
  }
}

// Slot (digit, warp) of a histogram whose digit rows are padded by one slot.
// Unpadded, the lanes of a warp ranking different digits would all hit the
// bank of column `warp`; padded, slot (d, w) sits in bank (d + w) % 32.
template <uint32_t kWarps>
__device__ __forceinline__ uint32_t small_slot(const uint32_t digit, const uint32_t warp) {
  return digit * (kWarps + 1) + warp;
}

// Sorts each block's tile of in[blockIdx.x * kKeys, ...) into the same range of
// out, which may be in. In place is safe: every key is in registers before any
// is written.
template <uint32_t kWarps, bool kWithValues>
__global__ void __launch_bounds__(kWarps *kWarpSize)
    radix_sort_tile_kernel(const uint64_t *in, const uint32_t *values_in, uint64_t *out,
                           uint32_t *values_out, const uint32_t n) {
  constexpr uint32_t kThreads = kWarps * kWarpSize;
  constexpr uint32_t kKeys = kThreads * kTileItems;
  // hist[small_slot(digit, warp)]: during ranking, that warp's running count
  // of the digit; after the scan, where that warp's keys of the digit start.
  __shared__ uint32_t hist[kSmallBuckets * (kWarps + 1)];
  __shared__ uint32_t warp_totals[kWarps];
  __shared__ uint64_t exchange_keys[kKeys];
  __shared__ uint32_t exchange_values[kWithValues ? kKeys : 1];
  const cg::thread_block block = cg::this_thread_block();
  auto warp_tile = cg::tiled_partition<kWarpSize>(block);
  using LaneMask = decltype(warp_tile.ballot(true));
  const uint32_t lane = warp_tile.thread_rank();
  const uint32_t warp = warp_tile.meta_group_rank();
  const LaneMask lanes_below = (LaneMask{1} << lane) - 1;
  const uint32_t first = warp * kWarpSize * kTileItems + lane;
  const uint32_t tile_base = blockIdx.x * kKeys;
  const uint32_t tile_n = n - tile_base < kKeys ? n - tile_base : kKeys;

  uint64_t key[kTileItems];
  uint32_t value[kTileItems];
#pragma unroll
  for (uint32_t k = 0; k < kTileItems; ++k) {
    const uint32_t slot = first + k * kWarpSize;
    key[k] = slot < tile_n ? ordered_bits(in[tile_base + slot]) : kPadKey;
    if constexpr (kWithValues) {
      value[k] = slot < tile_n ? values_in[tile_base + slot] : 0;
    }
  }

  for (uint32_t shift = 0; shift < kRadixKeyBits; shift += kSmallBits) {
    for (uint32_t s = threadIdx.x; s < kSmallBuckets * (kWarps + 1); s += kThreads) {
      hist[s] = 0;
    }
    block.sync();

    // Rank: rank << kSmallBits | digit per item. Each warp touches only its
    // own hist column, so warp syncs suffice here.
    uint32_t packed[kTileItems];
#pragma unroll
    for (uint32_t k = 0; k < kTileItems; ++k) {
      const uint32_t digit = static_cast<uint32_t>(key[k] >> shift) & kSmallDigitMask;
      // The lanes whose digit agrees with this lane's on every bit. Seeded
      // from a ballot, not ~0: under HIP the mask is 64 bits even for a
      // 32-lane tile, and the bits past the tile must start clear.
      LaneMask same = warp_tile.ballot(true);
#pragma unroll
      for (uint32_t b = 0; b < kSmallBits; ++b) {
        const bool bit = ((digit >> b) & 1U) != 0;
        const LaneMask ones = warp_tile.ballot(bit);
        same &= bit ? ones : static_cast<LaneMask>(~ones);
      }
      uint32_t *const slot = &hist[small_slot<kWarps>(digit, warp)];
      const uint32_t below = lane_count(static_cast<LaneMask>(same & lanes_below));
      const uint32_t rank = *slot + below;
      // Every lane reads the running count before the group's lowest lane
      // advances it past the whole group.
      warp_tile.sync();
      if (below == 0) {
        *slot += lane_count(same);
      }
      warp_tile.sync();
      packed[k] = rank << kSmallBits | digit;
    }
    block.sync();

    // Scan: each thread a run of consecutive (digit, warp) slots in
    // digit-major order, then one block scan of the run totals.
    const uint32_t base = threadIdx.x * kSmallSlotsPerThread;
    uint32_t run[kSmallSlotsPerThread];
    uint32_t run_total = 0;
#pragma unroll
    for (uint32_t j = 0; j < kSmallSlotsPerThread; ++j) {
      run[j] = hist[small_slot<kWarps>((base + j) / kWarps, (base + j) % kWarps)];
      run_total += run[j];
    }
    uint32_t block_total = 0;
    uint32_t offset = block_exclusive_sum<kWarps>(block, warp_totals, run_total, block_total);
#pragma unroll
    for (uint32_t j = 0; j < kSmallSlotsPerThread; ++j) {
      hist[small_slot<kWarps>((base + j) / kWarps, (base + j) % kWarps)] = offset;
      offset += run[j];
    }
    block.sync();

    // Every key in one digit: this pass's stable order is the current one, so
    // nothing moves. __syncthreads_or is also the barrier that ends every
    // thread's read of hist before the next pass zeroes it.
    bool uniform = false;
    for (uint32_t d = threadIdx.x; d < kSmallBuckets; d += kThreads) {
      const uint32_t begin = hist[small_slot<kWarps>(d, 0)];
      const uint32_t end = d + 1 < kSmallBuckets ? hist[small_slot<kWarps>(d + 1, 0)] : kKeys;
      uniform = uniform || end - begin == kKeys;
    }
    if (__syncthreads_or(uniform) != 0) {
      continue;
    }

    // Exchange through shared memory into the pass's order, then read back
    // in the same (warp, item, lane) layout. The next write to the exchange
    // buffers is a pass away, past two block barriers.
#pragma unroll
    for (uint32_t k = 0; k < kTileItems; ++k) {
      const uint32_t digit = packed[k] & kSmallDigitMask;
      const uint32_t to = hist[small_slot<kWarps>(digit, warp)] + (packed[k] >> kSmallBits);
      exchange_keys[to] = key[k];
      if constexpr (kWithValues) {
        exchange_values[to] = value[k];
      }
    }
    block.sync();
#pragma unroll
    for (uint32_t k = 0; k < kTileItems; ++k) {
      key[k] = exchange_keys[first + k * kWarpSize];
      if constexpr (kWithValues) {
        value[k] = exchange_values[first + k * kWarpSize];
      }
    }
  }

#pragma unroll
  for (uint32_t k = 0; k < kTileItems; ++k) {
    const uint32_t slot = first + k * kWarpSize;
    if (slot < tile_n) {
      out[tile_base + slot] = unordered_bits(key[k]);
      if constexpr (kWithValues) {
        values_out[tile_base + slot] = value[k];
      }
    }
  }
}

// Merges each pair of adjacent sorted runs of length `run` in src into dst
// (the last pair may be short, or a lone run). Thread t writes outputs
// [t * kMergeItems, ...): it binary-searches the merge path for where they
// start, then merges kMergeItems. Ties go to the left run, so the merge is
// stable, and the order is ordered_bits', the sort's.
__global__ void merge_runs_kernel(const uint64_t *src, const uint32_t *src_values, uint64_t *dst,
                                  uint32_t *dst_values, const uint32_t n, const uint32_t run) {
  const uint32_t out_begin = (blockIdx.x * kMergeThreads + threadIdx.x) * kMergeItems;
  if (out_begin >= n) {
    return;
  }
  const uint32_t pair_base = out_begin / (2 * run) * (2 * run);
  const uint32_t a_len = n - pair_base < run ? n - pair_base : run;
  const uint32_t rest = n - pair_base - a_len;
  const uint32_t b_len = rest < run ? rest : run;
  const uint64_t *const a = src + pair_base;
  const uint64_t *const b = a + a_len;
  const uint32_t diag = out_begin - pair_base;

  // The number of left-run keys among the first `diag` outputs.
  uint32_t lo = diag > b_len ? diag - b_len : 0;
  uint32_t hi = diag < a_len ? diag : a_len;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (ordered_bits(a[mid]) <= ordered_bits(b[diag - 1 - mid])) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  uint32_t i = lo;
  uint32_t j = diag - lo;

  const uint32_t pair_end = pair_base + a_len + b_len;
  const uint32_t out_end = out_begin + kMergeItems < pair_end ? out_begin + kMergeItems : pair_end;
  for (uint32_t o = out_begin; o < out_end; ++o) {
    const bool take_a = j >= b_len || (i < a_len && ordered_bits(a[i]) <= ordered_bits(b[j]));
    const uint32_t from = take_a ? pair_base + i : pair_base + a_len + j;
    dst[o] = src[from];
    if (src_values != nullptr) {
      dst_values[o] = src_values[from];
    }
    i += take_a ? 1 : 0;
    j += take_a ? 0 : 1;
  }
}

template <uint32_t kWarps>
wwr::wwrError_t launch_radix_sort_tiles(const wwr::wwrStream_t stream, const uint32_t num_tiles,
                                        const uint64_t *in, const uint32_t *values_in,
                                        uint64_t *out, uint32_t *values_out, const uint32_t n) {
  if (values_in != nullptr) {
    radix_sort_tile_kernel<kWarps, true>
        <<<num_tiles, kWarps * kWarpSize, 0, stream>>>(in, values_in, out, values_out, n);
  } else {
    radix_sort_tile_kernel<kWarps, false>
        <<<num_tiles, kWarps * kWarpSize, 0, stream>>>(in, nullptr, out, nullptr, n);
  }
  return wwr::wwrGetLastError();
}

// One tile, in place, in the fewest warps that hold n: the per-pass cost of
// zeroing, scanning and syncing the histogram scales with the warps, not n.
template <uint32_t kWarps>
wwr::wwrError_t sort_one_tile(const wwr::wwrStream_t stream, uint64_t *keys, uint32_t *values,
                              const uint32_t n) {
  if constexpr (kWarps < kTileMaxWarps) {
    if (n > kWarps * kWarpSize * kTileItems) {
      return sort_one_tile<kWarps * 2>(stream, keys, values, n);
    }
  }
  return launch_radix_sort_tiles<kWarps>(stream, 1, keys, values, keys, values, n);
}

wwr::wwrError_t radix_sort_small(const wwr::wwrStream_t stream, uint64_t *keys, uint64_t *keys_alt,
                                 uint32_t *values, uint32_t *values_alt, const uint32_t n) {
  if (n <= kTileKeys) {
    return sort_one_tile<1>(stream, keys, values, n);
  }
  // Tile-sort into whichever half leaves the last merge round writing the
  // caller's arrays: rounds alternate halves, and there are ceil(log2(tiles)).
  const uint32_t num_tiles = math::idivup(n, kTileKeys);
  uint32_t rounds = 0;
  for (uint32_t run = kTileKeys; run < n; run *= 2) {
    ++rounds;
  }
  uint64_t *src = rounds % 2 == 0 ? keys : keys_alt;
  uint64_t *dst = rounds % 2 == 0 ? keys_alt : keys;
  uint32_t *src_values = values == nullptr ? nullptr : rounds % 2 == 0 ? values : values_alt;
  uint32_t *dst_values = values == nullptr ? nullptr : rounds % 2 == 0 ? values_alt : values;
  wwr::wwrError_t err =
      launch_radix_sort_tiles<kTileMaxWarps>(stream, num_tiles, keys, values, src, src_values, n);
  const uint32_t merge_blocks =
      (n + kMergeThreads * kMergeItems - 1) / (kMergeThreads * kMergeItems);
  for (uint32_t run = kTileKeys; run < n && err == wwr::wwrSuccess; run *= 2) {
    merge_runs_kernel<<<merge_blocks, kMergeThreads, 0, stream>>>(src, src_values, dst, dst_values,
                                                                  n, run);
    err = wwr::wwrGetLastError();
    uint64_t *const merged = dst;
    dst = src;
    src = merged;
    uint32_t *const merged_values = dst_values;
    dst_values = src_values;
    src_values = merged_values;
  }
  return err;
}

} // namespace

wwr::wwrError_t radix_sort_passes(const wwr::wwrStream_t stream, const uint32_t n, double *keys,
                                  double *keys_alt, uint32_t *values, uint32_t *values_alt,
                                  uint32_t *counts) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  // The kernels only load and store the keys, as bits; no double is ever read
  // through these.
  auto *bits = reinterpret_cast<uint64_t *>(keys);
  auto *bits_alt = reinterpret_cast<uint64_t *>(keys_alt);

  // Small inputs: a tile sort, then at most four merge rounds.
  if (n <= kSmallSortMaxKeys) {
    return radix_sort_small(stream, bits, bits_alt, values, values_alt, n);
  }

  const uint32_t num_tiles = math::idivup(n, kRadixSortTile);
  const uint32_t count_len = num_tiles * kRadixBuckets;
  uint32_t *values_dst = values != nullptr ? values_alt : nullptr;
  for (uint32_t shift = 0; shift < kRadixKeyBits; shift += kRadixBits) {
    radix_count_kernel<<<num_tiles, kRadixSortTile, 0, stream>>>(bits, n, shift, counts);
    if (const wwr::wwrError_t err = wwr::wwrGetLastError(); err != wwr::wwrSuccess) {
      return err;
    }
    radix_scan_kernel<<<1, kScanThreads, 0, stream>>>(counts, count_len);
    if (const wwr::wwrError_t err = wwr::wwrGetLastError(); err != wwr::wwrSuccess) {
      return err;
    }
    radix_scatter_kernel<<<num_tiles, kRadixSortTile, 0, stream>>>(bits, values, n, shift, counts,
                                                                   bits_alt, values_dst);
    if (const wwr::wwrError_t err = wwr::wwrGetLastError(); err != wwr::wwrSuccess) {
      return err;
    }
    // The output of this pass is the input of the next. The pass count is
    // even, so the last swap leaves bits and values pointing at the caller's.
    std::swap(bits, bits_alt);
    std::swap(values, values_dst);
  }
  return wwr::wwrSuccess;
}

} // namespace free_breakfast::device
