// A stable merge sort of double keys, optionally carrying uint32_t values: a
// block sort of each tile on chip, then merge passes over global memory. One
// source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// mmd_launch.cu orders each class's records by zeta with it, and
// coulomb/sort_bounds' sort_bounds_launch.cu each ket class by Iq * Dmax. It
// is also the device half of free_breakfast.experimental.sort.merge, whose
// README has the algorithm's timings against radix_sort and the vendor sorts,
// and whose tests (test/experimental/merge_sort_gpu.cpp) hold it to
// std::strong_order.
//
// As in experimental/sort/radix's radix_sort.cu, the kernels move each key's 64
// bits as a uint64_t and compare ordered_bits() of them, whose unsigned order
// is IEEE-754 total order. Keys are remapped as they are loaded and mapped back
// as they are stored.
//
//   1. block_sort_kernel: each block sorts a tile of kTile keys. Each thread
//      sorts its kItems keys in registers (odd-even transposition, which swaps
//      only strictly-out-of-order neighbours, so it is stable), then the
//      threads' runs are merged in shared memory, doubling each round until
//      one run covers the tile.
//   2. merge_pass_kernel: merges adjacent sorted runs of length `run`,
//      doubling it each pass until one run covers n. Each block owns kTile
//      consecutive outputs. Two warps find where they start and end on the
//      merge path, and the block stages that slice of both runs in shared
//      memory and merges it there.
//
// Every merge in shared memory is a merge-path merge: each thread
// binary-searches the merge path for where its kItems outputs start, then
// merges them serially. Every merge takes from the left run on a tie, so the
// sort is stable.
//
// What did not work: merging by rank, where every key binary-searches the
// other run for its own output slot. Its kItems searches per thread are
// independent, so their loads overlap, but they are ~kItems times the shared
// memory loads of one search and a serial merge, and the kernels ran about
// twice as slow.
//
// Slots past the tile's last key hold kPadKey, which sorts after every real key
// (stably, so after a real all-ones key too) and is never written out.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"
#include "primitives.h"

#include "merge_sort_bridge.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <cooperative_groups.h>

#include <runtime.h>

namespace free_breakfast::device {

namespace {

namespace cg = cooperative_groups;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxWarps = kThreads / kWarpSize;
// Odd, so neighbouring threads' runs of kItems keys in shared memory start on
// different banks.
constexpr uint32_t kItems = 11;
constexpr uint32_t kTile = kThreads * kItems;
constexpr uint64_t kSignBit = uint64_t{1} << 63;
constexpr uint64_t kPadKey = ~uint64_t{0};

static_assert(kThreads >= 2 * kWarpSize, "a merge pass searches with two warps");

// A double's bits, remapped so that unsigned order is IEEE-754 total order
// (radix_sort.cu's ordered_bits, which has the reasoning).
__device__ __forceinline__ uint64_t ordered_bits(const uint64_t bits) {
  return bits ^ ((bits & kSignBit) != 0 ? ~uint64_t{0} : kSignBit);
}

// ordered_bits' inverse.
__device__ __forceinline__ uint64_t unordered_bits(const uint64_t ordered) {
  return ordered ^ ((ordered & kSignBit) != 0 ? kSignBit : ~uint64_t{0});
}

template <typename LaneMask>
__device__ __forceinline__ uint32_t lane_count(const LaneMask mask) {
  if constexpr (sizeof(LaneMask) == sizeof(uint32_t)) {
    return static_cast<uint32_t>(__popc(mask));
  } else {
    return static_cast<uint32_t>(__popcll(mask));
  }
}

// The number of keys of a (a_len of them) among the first `diag` outputs of
// the stable merge of a and b, where a and b are both in s, b right after a.
__device__ __forceinline__ uint32_t merge_path(const uint64_t *s, const uint32_t a_len,
                                               const uint32_t b_len, const uint32_t diag) {
  uint32_t lo = diag > b_len ? diag - b_len : 0;
  uint32_t hi = diag < a_len ? diag : a_len;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (s[mid] <= s[a_len + diag - 1 - mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The number of keys of a (a_len of them) among the first `diag` outputs of
// the stable merge of a and b, by a whole warp over global memory: each step
// probes kWarpSize evenly spaced points at once, so the range shrinks
// kWarpSize-fold per dependent load rather than two-fold. Every lane returns
// the answer.
template <typename Warp>
__device__ __forceinline__ uint32_t warp_merge_path(const Warp &warp, const uint64_t *a,
                                                    const uint32_t a_len, const uint64_t *b,
                                                    const uint32_t b_len, const uint32_t diag) {
  const uint32_t lane = warp.thread_rank();
  uint32_t lo = diag > b_len ? diag - b_len : 0;
  uint32_t hi = diag < a_len ? diag : a_len;
  while (lo < hi) {
    const uint32_t stride = math::idivup(hi - lo, kWarpSize);
    const uint32_t p = lo + lane * stride;
    const bool takes_a = p < hi && ordered_bits(a[p]) <= ordered_bits(b[diag - 1 - p]);
    // The probes that take from a are a prefix of the lanes: the merge path's
    // predicate is true up to the answer and false from it on.
    const uint32_t taken = lane_count(warp.ballot(takes_a));
    if (taken == 0) {
      hi = lo;
    } else {
      const uint32_t last = lo + (taken - 1) * stride;
      lo = last + 1;
      hi = last + stride < hi ? last + stride : hi;
    }
  }
  return lo;
}

// Merges up to kItems outputs, from output `diag` on, of a (a_len keys at s)
// and b (b_len keys right after it) into key, and where in s each came from
// into from. Outputs past a_len + b_len are left unset.
__device__ __forceinline__ void merge_items(const uint64_t *s, const uint32_t a_len,
                                            const uint32_t b_len, const uint32_t diag,
                                            uint64_t (&key)[kItems], uint32_t (&from)[kItems]) {
  uint32_t i = merge_path(s, a_len, b_len, diag);
  uint32_t j = diag - i;
  uint64_t next_a = i < a_len ? s[i] : 0;
  uint64_t next_b = j < b_len ? s[a_len + j] : 0;
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const bool take_a = j >= b_len || (i < a_len && next_a <= next_b);
    key[k] = take_a ? next_a : next_b;
    from[k] = take_a ? i : a_len + j;
    if (take_a) {
      ++i;
      if (i < a_len) {
        next_a = s[i];
      }
    } else {
      ++j;
      if (j < b_len) {
        next_b = s[a_len + j];
      }
    }
  }
}

// Sorts each block's tile of in[blockIdx.x * kKeys, ...) into the same range of
// out, which may be in: every key is in shared memory before any is written.
// A lone tile smaller than kTile runs in fewer warps, and so fewer rounds.
template <uint32_t kWarps, bool kWithValues>
__global__ void __launch_bounds__(kWarps *kWarpSize)
    block_sort_kernel(const uint64_t *in, const uint32_t *values_in, uint64_t *out,
                      uint32_t *values_out, const uint32_t n) {
  constexpr uint32_t kBlock = kWarps * kWarpSize;
  constexpr uint32_t kKeys = kBlock * kItems;
  __shared__ uint64_t s_keys[kKeys];
  __shared__ uint32_t s_values[kWithValues ? kKeys : 1];
  const uint32_t t = threadIdx.x;
  const uint32_t tile_base = blockIdx.x * kKeys;
  const uint32_t tile_n = n - tile_base < kKeys ? n - tile_base : kKeys;

  // Striped from global memory, so the loads coalesce; read back blocked, so
  // each thread holds kItems consecutive keys.
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const uint32_t slot = k * kBlock + t;
    s_keys[slot] = slot < tile_n ? ordered_bits(in[tile_base + slot]) : kPadKey;
    if constexpr (kWithValues) {
      s_values[slot] = slot < tile_n ? values_in[tile_base + slot] : 0;
    }
  }
  __syncthreads();
  uint64_t key[kItems];
  uint32_t value[kItems];
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    key[k] = s_keys[t * kItems + k];
    if constexpr (kWithValues) {
      value[k] = s_values[t * kItems + k];
    }
  }

  // kItems rounds of odd-even transposition sort the thread's run.
#pragma unroll
  for (uint32_t r = 0; r < kItems; ++r) {
#pragma unroll
    for (uint32_t k = r % 2; k + 1 < kItems; k += 2) {
      if (key[k] > key[k + 1]) {
        const uint64_t kt = key[k];
        key[k] = key[k + 1];
        key[k + 1] = kt;
        if constexpr (kWithValues) {
          const uint32_t vt = value[k];
          value[k] = value[k + 1];
          value[k + 1] = vt;
        }
      }
    }
  }

  // Each round merges pairs of `width`-thread runs, by 2 * width threads, which
  // read only their pair's slots. While a pair fits in a warp, a warp barrier
  // is enough: the write before the merge must wait for the previous round's
  // reads, by `width` threads, and the merge for this round's writes, by 2 *
  // width.
  const auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  for (uint32_t width = 1; width < kBlock; width *= 2) {
    if (width <= kWarpSize) {
      warp.sync();
    } else {
      __syncthreads();
    }
#pragma unroll
    for (uint32_t k = 0; k < kItems; ++k) {
      s_keys[t * kItems + k] = key[k];
      if constexpr (kWithValues) {
        s_values[t * kItems + k] = value[k];
      }
    }
    if (2 * width <= kWarpSize) {
      warp.sync();
    } else {
      __syncthreads();
    }
    const uint32_t run = width * kItems;
    const uint32_t pair_base = t / (2 * width) * (2 * run);
    uint32_t from[kItems];
    merge_items(s_keys + pair_base, run, run, t % (2 * width) * kItems, key, from);
    if constexpr (kWithValues) {
#pragma unroll
      for (uint32_t k = 0; k < kItems; ++k) {
        value[k] = s_values[pair_base + from[k]];
      }
    }
  }

  // Blocked into shared memory, striped out, so the stores coalesce.
  __syncthreads();
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    s_keys[t * kItems + k] = key[k];
    if constexpr (kWithValues) {
      s_values[t * kItems + k] = value[k];
    }
  }
  __syncthreads();
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const uint32_t slot = k * kBlock + t;
    if (slot < tile_n) {
      out[tile_base + slot] = unordered_bits(s_keys[slot]);
      if constexpr (kWithValues) {
        values_out[tile_base + slot] = s_values[slot];
      }
    }
  }
}

// Merges each pair of adjacent sorted runs of length `run` in src into dst
// (the last pair may be short, or a lone run). run is a multiple of kTile, so
// no block's outputs straddle two pairs.
template <bool kWithValues>
__global__ void __launch_bounds__(kThreads)
    merge_pass_kernel(const uint64_t *src, const uint32_t *src_values, uint64_t *dst,
                      uint32_t *dst_values, const uint32_t n, const uint32_t run) {
  __shared__ uint64_t s_keys[kTile];
  __shared__ uint32_t s_values[kWithValues ? kTile : 1];
  __shared__ uint32_t s_split[2];
  const uint32_t t = threadIdx.x;
  const uint32_t out_begin = blockIdx.x * kTile;
  const uint32_t pair_base = out_begin / (2 * run) * (2 * run);
  const uint32_t a_len = n - pair_base < run ? n - pair_base : run;
  const uint32_t rest = n - pair_base - a_len;
  const uint32_t b_len = rest < run ? rest : run;
  const uint32_t pair_end = pair_base + a_len + b_len;
  const uint32_t out_end = out_begin + kTile < pair_end ? out_begin + kTile : pair_end;
  const uint64_t *const a = src + pair_base;
  const uint64_t *const b = a + a_len;
  const uint32_t diag_begin = out_begin - pair_base;
  const uint32_t diag_end = out_end - pair_base;

  // Warp 0 finds where this block's slice starts on the merge path, warp 1
  // where it ends.
  const auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  if (warp.meta_group_rank() < 2) {
    const uint32_t diag = warp.meta_group_rank() == 0 ? diag_begin : diag_end;
    const uint32_t split = warp_merge_path(warp, a, a_len, b, b_len, diag);
    if (warp.thread_rank() == 0) {
      s_split[warp.meta_group_rank()] = split;
    }
  }
  __syncthreads();
  const uint32_t a_begin = s_split[0];
  const uint32_t a_count = s_split[1] - a_begin;
  const uint32_t b_begin = diag_begin - a_begin;
  const uint32_t total = diag_end - diag_begin;
  const uint32_t b_count = total - a_count;

  // The slice's keys of a, then of b, striped into shared memory.
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const uint32_t slot = k * kThreads + t;
    if (slot < total) {
      const uint32_t at = slot < a_count ? pair_base + a_begin + slot
                                         : pair_base + a_len + b_begin + (slot - a_count);
      s_keys[slot] = ordered_bits(src[at]);
      if constexpr (kWithValues) {
        s_values[slot] = src_values[at];
      }
    }
  }
  __syncthreads();

  const uint32_t diag = t * kItems < total ? t * kItems : total;
  uint64_t key[kItems];
  uint32_t from[kItems];
  merge_items(s_keys, a_count, b_count, diag, key, from);
  uint32_t value[kItems];
  if constexpr (kWithValues) {
#pragma unroll
    for (uint32_t k = 0; k < kItems; ++k) {
      value[k] = diag + k < total ? s_values[from[k]] : 0;
    }
  }
  __syncthreads();
#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    if (diag + k < total) {
      s_keys[diag + k] = key[k];
      if constexpr (kWithValues) {
        s_values[diag + k] = value[k];
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t k = 0; k < kItems; ++k) {
    const uint32_t slot = k * kThreads + t;
    if (slot < total) {
      dst[out_begin + slot] = unordered_bits(s_keys[slot]);
      if constexpr (kWithValues) {
        dst_values[out_begin + slot] = s_values[slot];
      }
    }
  }
}

template <uint32_t kWarps, bool kWithValues>
wwr::wwrError_t launch_block_sort(const wwr::wwrStream_t stream, const uint32_t num_tiles,
                                  const uint64_t *in, const uint32_t *values_in, uint64_t *out,
                                  uint32_t *values_out, const uint32_t n) {
  block_sort_kernel<kWarps, kWithValues>
      <<<num_tiles, kWarps * kWarpSize, 0, stream>>>(in, values_in, out, values_out, n);
  return wwr::wwrGetLastError();
}

// One tile, in place, in the fewest warps that hold n: each doubling of the
// block costs a merge round.
template <uint32_t kWarps, bool kWithValues>
wwr::wwrError_t sort_one_tile(const wwr::wwrStream_t stream, uint64_t *keys, uint32_t *values,
                              const uint32_t n) {
  if constexpr (kWarps < kMaxWarps) {
    if (n > kWarps * kWarpSize * kItems) {
      return sort_one_tile<kWarps * 2, kWithValues>(stream, keys, values, n);
    }
  }
  return launch_block_sort<kWarps, kWithValues>(stream, 1, keys, values, keys, values, n);
}

template <bool kWithValues>
wwr::wwrError_t merge_sort_launches(const wwr::wwrStream_t stream, uint64_t *keys,
                                    uint64_t *keys_alt, uint32_t *values, uint32_t *values_alt,
                                    const uint32_t n) {
  if (n <= kTile) {
    return sort_one_tile<1, kWithValues>(stream, keys, values, n);
  }
  const uint32_t num_tiles = math::idivup(n, kTile);
  // Block-sort into whichever half leaves the last pass writing the caller's
  // arrays: passes alternate halves, and there are ceil(log2(num_tiles)).
  uint32_t passes = 0;
  for (uint64_t run = kTile; run < n; run *= 2) {
    ++passes;
  }
  uint64_t *src = passes % 2 == 0 ? keys : keys_alt;
  uint64_t *dst = passes % 2 == 0 ? keys_alt : keys;
  uint32_t *src_values = passes % 2 == 0 ? values : values_alt;
  uint32_t *dst_values = passes % 2 == 0 ? values_alt : values;
  wwr::wwrError_t err = launch_block_sort<kMaxWarps, kWithValues>(stream, num_tiles, keys, values,
                                                                  src, src_values, n);
  for (uint64_t run = kTile; run < n && err == wwr::wwrSuccess; run *= 2) {
    merge_pass_kernel<kWithValues><<<num_tiles, kThreads, 0, stream>>>(
        src, src_values, dst, dst_values, n, static_cast<uint32_t>(run));
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

wwr::wwrError_t merge_sort_passes(const wwr::wwrStream_t stream, const uint32_t n, double *keys,
                                  double *keys_alt, uint32_t *values, uint32_t *values_alt) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  // The kernels only load and store the keys, as bits.
  auto *bits = reinterpret_cast<uint64_t *>(keys);
  auto *bits_alt = reinterpret_cast<uint64_t *>(keys_alt);
  return values != nullptr
             ? merge_sort_launches<true>(stream, bits, bits_alt, values, values_alt, n)
             : merge_sort_launches<false>(stream, bits, bits_alt, nullptr, nullptr, n);
}

} // namespace free_breakfast::device
