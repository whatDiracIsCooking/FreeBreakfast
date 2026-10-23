// The sweep's completion (exchange_helpers_bridge.h): each AO's shell start,
// and the mirror that copies K's upper shell blocks, times a sign, onto its
// lower ones. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.

// exchange_helpers.cuh first: it orders its standard headers ahead of the
// vendor runtime header, which under HIP must not precede libc++.
#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::int64_t;

// One thread per AO of the momentum's shells, by rank.
__global__ void shell_starts_kernel(const int32_t n, const int32_t nx, const int32_t *off,
                                    int32_t *start) {
  const auto e = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < n) {
    const int32_t o = off[e / nx];
    start[o + e % nx] = o;
  }
}

// One thread per element of K; those of a lower shell block read their
// transpose, which is in an upper one, so no element is both read and written.
__global__ void mirror_kernel(const int64_t n, const int64_t n_ao, const int32_t *start,
                              const double sign, double *k) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= n) {
    return;
  }
  const int64_t m = e / n_ao;
  const int64_t q = e % n_ao;
  if (q < start[m]) {
    k[e] = sign * k[q * n_ao + m];
  }
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_shell_starts(const wwr::wwrStream_t stream, const int32_t l,
                                      const int32_t n_s, const int32_t *off, int32_t *start) {
  if (!exchange::valid_momentum(l) || n_s < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t nx = exchange::cartesian_count(l);
  const int64_t n = static_cast<int64_t>(n_s) * nx;
  if (n > exchange::kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  shell_starts_kernel<<<exchange::element_blocks(n), kFourWarps, 0, stream>>>(
      static_cast<int32_t>(n), nx, off, start);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t exchange_mirror(const wwr::wwrStream_t stream, const int32_t n_ao,
                                const int32_t *start, const double sign, double *k) {
  if (n_ao < 1 || (sign != 1.0 && sign != -1.0)) {
    return wwr::wwrErrorInvalidValue;
  }
  const int64_t n = static_cast<int64_t>(n_ao) * n_ao;
  mirror_kernel<<<exchange::element_blocks(n), kFourWarps, 0, stream>>>(n, n_ao, start, sign, k);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
