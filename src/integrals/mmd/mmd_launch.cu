// The device half of free_breakfast.integrals.mmd: each class's records put in
// descending zeta order, once per basis, after their copy to the device. One
// source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "merge_sort_bridge.h"
#include "mmd_bridge.h"

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

} // namespace free_breakfast::device
