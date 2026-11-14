// The diagonal tile of the blocked Cholesky factorization, in shared memory.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Forked from src/recipes/cholesky/cholesky.cu. The kernel's factorization is
// unchanged: per column, read the pivot, barrier, scale the column, barrier,
// rank-1 update of the trailing triangle, barrier. Two things differ, both at
// the edges of it. It also writes the factored tile's lower triangle, packed,
// into l11: the panel's trsm takes L11 as an operand while it writes the
// panel below it in the same columns, and a safe call cannot hand one
// allocation out as both, so the caller keeps the copy in a scratch row of
// its own (rust/recipes/cholesky has the loop). And the launcher is rewritten
// as cholesky.h's extern "C", enqueue-only launch wrapper
// (rust/docs/gpu_layer.md, "From bridge to C ABI"): lda is the module's
// int32_t, widened here, the null pointers are refused as well as the C++
// launcher's arguments, and the status is the C ABI's wwr_get_last_error(), so
// no vendor code crosses.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"
#include "primitives.h"

#include "cholesky/cholesky.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <runtime.h>

namespace free_breakfast::device {

namespace {

// The tile's order, the header's constant: the shared array is sized by it.
constexpr int32_t kCholeskyTile = FB_CHOLESKY_TILE;
static_assert(kCholeskyTile * kCholeskyTile * sizeof(double) <= 48 * 1024,
              "a tile within CUDA's 48 KiB of static shared memory");

// Right-looking and unblocked over the tile in shared memory, column k per
// step: every thread reads the pivot, a barrier, column k scaled by its
// square root, a barrier, the rank-1 update of the trailing triangle, a
// barrier. `tile[j][i]` is element (i, j), so a column is contiguous.
//
// The barrier between reading the pivot and writing its square root back is
// the one rocSOLVER's potf2_kernel_small (ROCm 7.2) lacks: there a thread that
// reaches the pivot after thread 0 has overwritten it divides its rows by
// sqrt(l_kk) instead of l_kk, which made the DF metric "not positive definite"
// on HIP, or its factor silently wrong, whenever waves of the block drifted
// apart (#404, #379).
__global__ void cholesky_diagonal_kernel(const int32_t jb, const int32_t first,
                                         double *__restrict__ a, const int64_t lda,
                                         double *__restrict__ l11, int32_t *__restrict__ info) {
  __shared__ double tile[kCholeskyTile][kCholeskyTile];
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
      tile[j][i] = a[i + (j * lda)];
    }
  }
  __syncthreads();

  int32_t failed = 0;
  for (int32_t k = 0; k < jb; ++k) {
    const double pivot = tile[k][k];
    __syncthreads();
    // Every thread read the same pivot, so every thread leaves together.
    if (!(pivot > 0.0)) {
      failed = k + 1;
      break;
    }
    const double l = sqrt(pivot);
    for (int32_t i = k + tid; i < jb; i += stride) {
      tile[k][i] = i == k ? l : tile[k][i] / l;
    }
    __syncthreads();
    // The trailing lower triangle, from column k, which this step does not
    // write.
    const int32_t rest = jb - k - 1;
    for (int32_t e = tid; e < rest * rest; e += stride) {
      const int32_t i = k + 1 + (e % rest);
      const int32_t j = k + 1 + (e / rest);
      if (i >= j) {
        tile[j][i] -= tile[k][i] * tile[k][j];
      }
    }
    __syncthreads();
  }

  for (int32_t e = tid; e < elements; e += stride) {
    const int32_t i = e % jb;
    const int32_t j = e / jb;
    if (i >= j) {
      a[i + (j * lda)] = tile[j][i];
      l11[i + (j * jb)] = tile[j][i];
    }
  }
  if (failed != 0 && tid == 0) {
    *info = first + failed;
  }
}

} // namespace

} // namespace free_breakfast::device

extern "C" fb_launch_status fb_cholesky_diagonal(const wwr_stream stream, const int32_t jb,
                                                 const int32_t first, double *a, const int32_t lda,
                                                 double *l11, int32_t *info) {
  using namespace free_breakfast;
  using namespace free_breakfast::device;
  if (jb < 1 || jb > kCholeskyTile || first < 0 || lda < jb || a == nullptr || l11 == nullptr ||
      info == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The C ABI's stream is the runtime's, behind a void *. One block: the
  // tile is in its shared memory.
  cholesky_diagonal_kernel<<<1, kFourWarps, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      jb, first, a, int64_t{lda}, l11, info);
  // The launch just enqueued, as a status: the C ABI's own wwr_get_last_error(),
  // whose wwr_status values are fb_launch_status's positive range.
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
