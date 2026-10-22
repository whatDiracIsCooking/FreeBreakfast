// The device half of free_breakfast.integrals.mmd: each class's records put in
// descending zeta order, once per basis, after their copy to the device, then
// each record's Schwarz bound through the generated pair_bound<la, lb>
// (kernels/pair_bound/, written by tools/emit_one_electron_kernels.py) and its
// dressed bound, as stored and swapped, through the generated
// record_bound<la, lb> (kernels/record_bound/, written by
// tools/emit_exchange_kernels.py), each class its own unit of this library:
// launching a specialization another TU defines wants relocatable device code
// under nvcc. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "angular_momentum.h"
#include "merge_sort_bridge.h"
#include "mmd_bridge.h"

#include "kernels/pair_bound/pair_bound.h"
#include "kernels/record_bound/record_bound.h"

namespace {

// key[k] = -zeta[k] and perm[k] = k: the ascending sort of key is the
// descending order of zeta, carrying each record's index.
__global__ void mmd_order_key_kernel(const int32_t n_rec, const double *zeta, double *key,
                                     uint32_t *perm) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    key[k] = -zeta[k];
    perm[k] = static_cast<uint32_t>(k);
  }
}

// field[k] = from[perm[k]]: one thread per record, the stores coalesced.
template <typename T>
__global__ void mmd_gather_kernel(const int32_t n_rec, const uint32_t *perm, const T *from,
                                  T *field) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    field[k] = from[perm[k]];
  }
}

} // namespace

namespace free_breakfast::device {

// One pair class per la <= lb <= kMaxAngularMomentum.
constexpr auto kPairExtent = static_cast<size_t>(kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(pair_bound_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(pair_bound_kernels), 1> == kPairExtent,
              "kernels/pair_bound was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_one_electron_kernels.py");

static_assert(std::extent_v<decltype(record_bound_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(record_bound_kernels), 1> == kPairExtent,
              "kernels/record_bound was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_exchange_kernels.py");

namespace {

uint32_t mmd_record_blocks(const int32_t n_rec) {
  return math::idivup(static_cast<uint32_t>(n_rec), kFourWarps);
}

template <typename T>
wwr::wwrError_t mmd_permute_field(const wwr::wwrStream_t stream, const int32_t n_rec,
                                  const uint32_t *perm, T *field, T *tmp) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  if (const wwr::wwrError_t code = wwr::wwrMemcpyAsync(
          tmp, field, sizeof(T) * static_cast<size_t>(n_rec), wwr::wwrMemcpyDeviceToDevice, stream);
      code != wwr::wwrSuccess) {
    return code;
  }
  mmd_gather_kernel<T>
      <<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, perm, tmp, field);
  return wwr::wwrGetLastError();
}

} // namespace

wwr::wwrError_t mmd_order(const wwr::wwrStream_t stream, const int32_t n_rec, const double *zeta,
                          double *key, double *key_alt, uint32_t *perm, uint32_t *perm_alt) {
  if (n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  mmd_order_key_kernel<<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, zeta, key, perm);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  return static_cast<wwr::wwrError_t>(
      merge_sort_passes(stream, static_cast<uint32_t>(n_rec), key, key_alt, perm, perm_alt));
}

wwr::wwrError_t mmd_permute(const wwr::wwrStream_t stream, const int32_t n_rec,
                            const uint32_t *perm, double *field, double *tmp) {
  return mmd_permute_field(stream, n_rec, perm, field, tmp);
}

wwr::wwrError_t mmd_permute(const wwr::wwrStream_t stream, const int32_t n_rec,
                            const uint32_t *perm, int32_t *field, int32_t *tmp) {
  return mmd_permute_field(stream, n_rec, perm, field, tmp);
}

wwr::wwrError_t mmd_bound(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                          const int32_t n_rec, const double *zeta, double *bound) {
  if (n_rec < 1 || la < 0 || lb < la || lb > kMaxAngularMomentum) {
    return wwr::wwrErrorInvalidValue;
  }
  pair_bound_kernels[la][lb]<<<mmd_record_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, zeta,
                                                                                  bound);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ordered_pair_bound(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                   double *bound) {
  if (cls.n_rec < 1 || cls.la < 0 || cls.la > kMaxAngularMomentum || cls.lb < 0 ||
      cls.lb > kMaxAngularMomentum) {
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

} // namespace free_breakfast::device
