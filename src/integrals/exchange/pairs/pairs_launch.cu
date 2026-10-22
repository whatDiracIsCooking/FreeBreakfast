// The device half of free_breakfast.integrals.exchange.pairs: each record's
// dressed Schwarz bound, through mmd's launcher of the generated
// record_bound<la, lb> (mmd_bridge.h), and the order that sorts each group by
// it, once per basis. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "merge_sort_bridge.h"
#include "mmd_bridge.h"
#include "pairs_bridge.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::size_t;
using fb::uint32_t;

// key[k] = -bound[k] and perm[k] = k: the ascending sort of key is the
// descending order of bound, carrying each record's index.
__global__ void bound_key_kernel(const int32_t n_rec, const double *bound, double *key,
                                 uint32_t *perm) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    key[k] = -bound[k];
    perm[k] = static_cast<uint32_t>(k);
  }
}

// key[k] = the index of the group record perm[k] is in: the last g with
// group_start[g] <= perm[k], by bisection over the n_groups + 1 bounds.
__global__ void group_key_kernel(const int32_t n_rec, const int32_t n_groups,
                                 const int32_t *group_start, const uint32_t *perm, double *key) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    const auto r = static_cast<int32_t>(perm[k]);
    int32_t lo = 0;
    int32_t hi = n_groups;
    while (hi - lo > 1) {
      const int32_t mid = lo + (hi - lo) / 2;
      if (group_start[mid] <= r) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    key[k] = static_cast<double>(lo);
  }
}

// field[k] = from[perm[k]]: one thread per record, the stores coalesced.
template <typename T>
__global__ void gather_kernel(const int32_t n_rec, const uint32_t *perm, const T *from, T *field) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    field[k] = from[perm[k]];
  }
}

// One thread per record, kFourWarps to a block.
uint32_t record_blocks(const int32_t n_rec) {
  return fb::math::idivup(static_cast<uint32_t>(n_rec), fb::kFourWarps);
}

template <typename T>
wwr::wwrError_t permute_field(const wwr::wwrStream_t stream, const int32_t n_rec,
                              const uint32_t *perm, T *field, T *tmp) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  if (const wwr::wwrError_t code = wwr::wwrMemcpyAsync(
          tmp, field, sizeof(T) * static_cast<size_t>(n_rec), wwr::wwrMemcpyDeviceToDevice, stream);
      code != wwr::wwrSuccess) {
    return code;
  }
  gather_kernel<T><<<record_blocks(n_rec), fb::kFourWarps, 0, stream>>>(n_rec, perm, tmp, field);
  return wwr::wwrGetLastError();
}

} // namespace

namespace free_breakfast::device {

// mmd's launcher over the fields record_bound reads, which the ordered class
// names as mmd's view does, "a" its first shell.
wwr::wwrError_t ordered_pair_bound(const wwr::wwrStream_t stream, const OrderedPairClassView &cls,
                                   double *bound) {
  return ordered_pair_bound(stream,
                            PrimitivePairClassView{
                                .la = cls.la,
                                .lb = cls.lb,
                                .n_rec = cls.n_rec,
                                .zeta = cls.zeta,
                                .pa_x = cls.pa_x,
                                .pa_y = cls.pa_y,
                                .pa_z = cls.pa_z,
                                .pb_x = cls.pb_x,
                                .pb_y = cls.pb_y,
                                .pb_z = cls.pb_z,
                                .s = cls.s,
                            },
                            bound);
}

wwr::wwrError_t ordered_pair_order(const wwr::wwrStream_t stream, const int32_t n_rec,
                                   const int32_t n_groups, const int32_t *group_start,
                                   const double *bound, double *key, double *key_alt,
                                   uint32_t *perm, uint32_t *perm_alt) {
  if (n_rec < 1 || n_groups < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto n = static_cast<uint32_t>(n_rec);
  bound_key_kernel<<<record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, bound, key, perm);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  if (const auto code =
          static_cast<wwr::wwrError_t>(merge_sort_passes(stream, n, key, key_alt, perm, perm_alt));
      code != wwr::wwrSuccess) {
    return code;
  }
  group_key_kernel<<<record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, n_groups, group_start,
                                                                    perm, key);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  return static_cast<wwr::wwrError_t>(merge_sort_passes(stream, n, key, key_alt, perm, perm_alt));
}

wwr::wwrError_t ordered_pair_permute(const wwr::wwrStream_t stream, const int32_t n_rec,
                                     const uint32_t *perm, double *field, double *tmp) {
  return permute_field(stream, n_rec, perm, field, tmp);
}

wwr::wwrError_t ordered_pair_permute(const wwr::wwrStream_t stream, const int32_t n_rec,
                                     const uint32_t *perm, int32_t *field, int32_t *tmp) {
  return permute_field(stream, n_rec, perm, field, tmp);
}

} // namespace free_breakfast::device
