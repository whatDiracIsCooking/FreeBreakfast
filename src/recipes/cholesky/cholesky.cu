// The device half of free_breakfast.recipes.cholesky: the diagonal-block
// factorization. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"

#include "cholesky_bridge.h"

namespace free_breakfast::device {

namespace {

// Right-looking and unblocked over the block in shared memory, column k per
// step: every thread reads the pivot, a barrier, column k scaled by its
// square root, a barrier, the rank-1 update of the trailing triangle, a
// barrier. `block[j][i]` is element (i, j), so a column is contiguous.
//
// The barrier between reading the pivot and writing its square root back is
// the one rocSOLVER's potf2_kernel_small (ROCm 7.2) lacks: there a thread that
// reaches the pivot after thread 0 has overwritten it divides its rows by
// sqrt(l_kk) instead of l_kk, which made the DF metric "not positive definite"
// on HIP, or its factor silently wrong, whenever waves of the block drifted
// apart (#404, #379).
__global__ void cholesky_diagonal_block_kernel(const int32_t jb, const int32_t first,
                                               double *__restrict__ a, const int64_t lda,
                                               int32_t *__restrict__ info) {
  __shared__ double block[kCholeskyBlock][kCholeskyBlock];
  // Uniform across the block: every thread reads the flag an earlier launch
  // wrote, before this one can write it.
  if (*info != 0) {
    return;
  }
  const auto tid = static_cast<int32_t>(threadIdx.x);
  const auto stride = static_cast<int32_t>(blockDim.x);
  const int32_t elements = jb * jb;

  for (int32_t e = tid; e < elements; e += stride) {
    const int32_t i = e % jb;
    const int32_t j = e / jb;
    if (i >= j) {
      block[j][i] = a[i + (j * lda)];
    }
  }
  __syncthreads();

  int32_t failed = 0;
  for (int32_t k = 0; k < jb; ++k) {
    const double pivot = block[k][k];
    __syncthreads();
    // Every thread read the same pivot, so every thread leaves together.
    if (!(pivot > 0.0)) {
      failed = k + 1;
      break;
    }
    const double l = sqrt(pivot);
    for (int32_t i = k + tid; i < jb; i += stride) {
      block[k][i] = i == k ? l : block[k][i] / l;
    }
    __syncthreads();
    // The trailing lower triangle, from column k, which this step does not
    // write.
    const int32_t rest = jb - k - 1;
    for (int32_t e = tid; e < rest * rest; e += stride) {
      const int32_t i = k + 1 + (e % rest);
      const int32_t j = k + 1 + (e / rest);
      if (i >= j) {
        block[j][i] -= block[k][i] * block[k][j];
      }
    }
    __syncthreads();
  }

  for (int32_t e = tid; e < elements; e += stride) {
    const int32_t i = e % jb;
    const int32_t j = e / jb;
    if (i >= j) {
      a[i + (j * lda)] = block[j][i];
    }
  }
  if (failed != 0 && tid == 0) {
    *info = first + failed;
  }
}

} // namespace

wwr::wwrError_t cholesky_diagonal_block(const wwr::wwrStream_t stream, const int32_t jb,
                                        const int32_t first, double *a, const int64_t lda,
                                        int32_t *info) {
  if (jb < 1 || jb > kCholeskyBlock || first < 0 || lda < jb) {
    return wwr::wwrErrorInvalidValue;
  }
  cholesky_diagonal_block_kernel<<<1, kFourWarps, 0, stream>>>(jb, first, a, lda, info);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
