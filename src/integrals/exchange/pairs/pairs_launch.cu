// The device half of free_breakfast.integrals.exchange.pairs: each record's
// dressed Schwarz bound through the generated record_bound<la, lb>
// (kernels/record_bound/, written by tools/emit_exchange_kernels.py, each class
// its own unit of this library: launching a specialization another TU defines
// wants relocatable device code under nvcc), and the order that sorts each
// group by it, once per basis. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "angular_momentum.h"
#include "merge_sort_bridge.h"
#include "pairs_bridge.h"

#include "kernels/record_bound/record_bound.h"

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

constexpr auto kExtent = static_cast<size_t>(fb::kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(record_bound_kernels), 0> == kExtent &&
                  std::extent_v<decltype(record_bound_kernels), 1> == kExtent,
              "kernels/record_bound was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_exchange_kernels.py");

bool valid_momentum(const int32_t l) {
  return l >= 0 && l <= fb::kMaxAngularMomentum;
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

wwr::wwrError_t ordered_pair_bound(const wwr::wwrStream_t stream, const OrderedPairClassView &cls,
                                   double *bound) {
  if (cls.n_rec < 1 || !valid_momentum(cls.la) || !valid_momentum(cls.lb)) {
    return wwr::wwrErrorInvalidValue;
  }
  // One thread per record in 32-thread blocks, as FelizJueves's launch_shape
  // states it; the mapping is not part of the numerical contract.
  constexpr uint32_t kThreads = 32;
  record_bound_kernels[cls.la][cls.lb]<<<math::idivup(static_cast<uint32_t>(cls.n_rec), kThreads),
                                         kThreads, 0, stream>>>(cls.n_rec, cls.zeta, cls.pa_x,
                                                                cls.pa_y, cls.pa_z, cls.pb_x,
                                                                cls.pb_y, cls.pb_z, cls.s, bound);
  return wwr::wwrGetLastError();
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
