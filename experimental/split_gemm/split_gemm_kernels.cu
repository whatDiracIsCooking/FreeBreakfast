// The device half of free_breakfast.experimental.split_gemm: the hi/lo split of
// an fp64 operand into two fp32 planes, and the fp64 accumulation of one K
// chunk's fp32 product. One source for both backends -- nvcc under CUDA, clang
// -x hip under HIP.
//
// Both are elementwise and bandwidth-bound; neither is the thing being
// measured. They exist so the three Sgemm calls can be the vendor's own, which
// is the whole point of this probe: the demo's hand-written int8 WMMA digest
// lost to rocBLAS on kernel quality, so this one writes no GEMM at all.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "split_gemm_bridge.h"

namespace free_breakfast::device {

namespace {

// One thread per element. hi is the fp64 value rounded to fp32; lo is the
// remainder, itself rounded to fp32. The subtraction is exact in fp64
// (Sterbenz), so hi + lo loses only what lo's own 24 bits cannot hold.
__global__ void split_hi_lo_kernel(const double *const x, float *const hi, float *const lo,
                                   const int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const double value = x[i];
  const auto high = static_cast<float>(value);
  hi[i] = high;
  lo[i] = static_cast<float>(value - static_cast<double>(high));
}

// One thread per element.
__global__ void accumulate_f32_kernel(const float *const src, double *const dst,
                                      const int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  dst[i] += static_cast<double>(src[i]);
}

// The grid for one thread per element, at kFourWarps per block.
dim3 elementwise_grid(const int64_t count) {
  return dim3{static_cast<uint32_t>(math::idivup(count, static_cast<int64_t>(kFourWarps)))};
}

} // namespace

wwr::wwrError_t split_hi_lo(const wwr::wwrStream_t stream, const double *const x, float *const hi,
                            float *const lo, const int64_t count) {
  if (count <= 0) {
    return wwr::wwrSuccess;
  }
  split_hi_lo_kernel<<<elementwise_grid(count), kFourWarps, 0, stream>>>(x, hi, lo, count);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t accumulate_f32(const wwr::wwrStream_t stream, const float *const src,
                               double *const dst, const int64_t count) {
  if (count <= 0) {
    return wwr::wwrSuccess;
  }
  accumulate_f32_kernel<<<elementwise_grid(count), kFourWarps, 0, stream>>>(src, dst, count);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
