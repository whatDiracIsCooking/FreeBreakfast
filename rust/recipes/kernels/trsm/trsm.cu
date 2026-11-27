// The diagonal block of the blocked triangular solve, in shared memory.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Forked from src/recipes/trsm/trsm.cu. The kernel is unchanged. The launcher
// is rewritten as trsm.h's extern "C", enqueue-only launch wrapper
// (rust/docs/gpu_layer.md, "From bridge to C ABI"): doubles alone, the one
// element type a caller solves in, the leading dimensions the module's
// int32_t, widened here, the null pointers and the short leading dimensions
// refused as well as the C++ launcher's arguments, and the status the C ABI's
// wwr_get_last_error(), so no vendor code crosses.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"
#include "primitives.h"

#include "trsm/trsm.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <runtime.h>

namespace free_breakfast::device {

namespace {

// The block's order, the header's constant: the shared arrays are sized by it.
constexpr int32_t kTrsmBlock = FB_TRSM_TILE;

// The right-hand sides one thread block takes: its tile is kTrsmBlock x
// kTrsmColumns, 24 KiB of doubles. With the packed triangle (16.25 KiB) and
// its diagonal's reciprocals the kernel holds 40.75 KiB of shared memory,
// under CUDA's 48 KiB static limit and the 64 KiB of LDS under HIP.
constexpr int32_t kTrsmColumns = 48;

// The threads that share one right-hand side, each taking every
// kTrsmLanes-th row of an update.
constexpr int32_t kTrsmLanes = 4;

// The thread block: kTrsmLanes threads to each of its kTrsmColumns columns,
// 192, whole warps at a warp of 32 or 64.
constexpr uint32_t kTrsmThreads = kTrsmColumns * kTrsmLanes;

// One thread block per kTrsmColumns right-hand sides, kTrsmLanes threads to a
// column. The triangle, packed, and the block's tile of B go to shared memory,
// and the solve is forward substitution over them with one barrier a row.
//
// The triangle is first scaled to a unit one, M = M' D: column j below the
// diagonal times 1 / M(j, j). Then M' Y = s B needs no division and no pivot
// step: once row k is final, each column's threads take their multiple of it
// from the rows below, a barrier, and row k + 1 is final. X = D^-1 Y is
// applied on the way out.
//
// An upper M is solved as a lower one by reversing its rows and columns, and
// the tile's rows with them: M(p, q) for p <= q goes to (jb-1-p, jb-1-q),
// which is on or below the diagonal. So one loop serves both triangles, and
// only the loads and stores know which it is.
//
// `x[i][c]` is the tile's (i, c): a warp's threads, consecutive in c, write
// consecutive words of a row and read one word of the triangle.
constexpr int32_t kTrsmPacked = kTrsmBlock * (kTrsmBlock + 1) / 2;

static_assert(kTrsmThreads % kWarpSize == 0 && kTrsmThreads <= kMaxThreads,
              "the diagonal-block kernel's thread block is whole warps");
static_assert((kTrsmPacked + kTrsmBlock + kTrsmBlock * kTrsmColumns) * sizeof(double) <= 48 * 1024,
              "the kernel's arrays within CUDA's 48 KiB of static shared memory");

// Where column j of the packed lower triangle starts: (i, j), i >= j, is at
// packed_column(j) + i - j.
__device__ constexpr int32_t packed_column(const int32_t j) {
  return (j * kTrsmBlock) - (j * (j - 1) / 2);
}

template <typename T>
__global__ void trsm_diagonal_block_kernel(const int32_t jb, const int32_t ncols, const T s,
                                           const T *__restrict__ a, const int64_t lda,
                                           const bool a_trans, const bool upper, const bool unit,
                                           T *__restrict__ b, const int64_t ldb,
                                           const bool b_trans) {
  __shared__ T tri[kTrsmPacked];
  __shared__ T reciprocal[kTrsmBlock];
  __shared__ T x[kTrsmBlock][kTrsmColumns];

  const auto tid = static_cast<int32_t>(threadIdx.x);
  const auto stride = static_cast<int32_t>(blockDim.x);
  const int32_t c0 = static_cast<int32_t>(blockIdx.x) * kTrsmColumns;
  const int32_t nc = min(kTrsmColumns, ncols - c0);
  const int32_t elements = jb * nc;
  const auto flip = [&](const int32_t r) { return upper ? jb - 1 - r : r; };
  // B(r, c0 + c), element by element: down a column for B, along a row for
  // B^T, so a warp's accesses are consecutive in memory either way.
  const auto b_index = [&](const int32_t e, int32_t &r, int32_t &c) -> int64_t {
    if (b_trans) {
      r = e / nc;
      c = e % nc;
      return (c0 + c) + (r * ldb);
    }
    r = e % jb;
    c = e / jb;
    return r + ((c0 + c) * ldb);
  };

  // alpha = 0: BLAS sets X to zero and reads neither A nor B, so a NaN in
  // either does not reach it.
  if (s == T{0}) {
    for (int32_t e = tid; e < elements; e += stride) {
      int32_t r = 0;
      int32_t c = 0;
      b[b_index(e, r, c)] = T{0};
    }
    return;
  }

  for (int32_t e = tid; e < jb * jb; e += stride) {
    const int32_t i = e % jb;
    const int32_t j = e / jb;
    if (i < j) {
      continue;
    }
    if (i == j && unit) {
      tri[packed_column(j)] = T{1};
      continue;
    }
    const int64_t p = flip(i);
    const int64_t q = flip(j);
    tri[packed_column(j) + i - j] = a_trans ? a[q + (p * lda)] : a[p + (q * lda)];
  }
  for (int32_t e = tid; e < elements; e += stride) {
    int32_t r = 0;
    int32_t c = 0;
    const int64_t at = b_index(e, r, c);
    x[flip(r)][c] = s * b[at];
  }
  __syncthreads();
  if (tid < jb) {
    reciprocal[tid] = T{1} / tri[packed_column(tid)];
  }
  __syncthreads();
  for (int32_t e = tid; e < jb * jb; e += stride) {
    const int32_t i = e % jb;
    const int32_t j = e / jb;
    if (i > j) {
      tri[packed_column(j) + i - j] *= reciprocal[j];
    }
  }
  __syncthreads();

  // This thread's column of the tile, and which of its rows it takes.
  const int32_t c = tid % kTrsmColumns;
  const int32_t lane = tid / kTrsmColumns;
  for (int32_t k = 0; k + 1 < jb; ++k) {
    if (c < nc) {
      const T xk = x[k][c];
      const T *const column = tri + packed_column(k) - k;
      for (int32_t i = k + 1 + lane; i < jb; i += kTrsmLanes) {
        x[i][c] -= column[i] * xk;
      }
    }
    __syncthreads();
  }

  for (int32_t e = tid; e < elements; e += stride) {
    int32_t r = 0;
    int32_t col = 0;
    const int64_t at = b_index(e, r, col);
    const int32_t row = flip(r);
    b[at] = x[row][col] * reciprocal[row];
  }
}

template <typename T>
fb_launch_status launch(const wwr_stream stream, const int32_t jb, const int32_t ncols, const T s,
                        const T *a, const int32_t lda, const bool a_trans, const bool upper,
                        const bool unit, T *b, const int32_t ldb, const bool b_trans) {
  if (jb < 1 || jb > kTrsmBlock || ncols < 1 || lda < jb || ldb < (b_trans ? ncols : jb) ||
      a == nullptr || b == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const auto blocks = static_cast<uint32_t>(math::idivup(ncols, kTrsmColumns));
  // The C ABI's stream is the runtime's, behind a void *.
  trsm_diagonal_block_kernel<T><<<blocks, kTrsmThreads, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      jb, ncols, s, a, int64_t{lda}, a_trans, upper, unit, b, int64_t{ldb}, b_trans);
  // The launch just enqueued, as a status: the C ABI's own wwr_get_last_error(),
  // whose wwr_status values are fb_launch_status's positive range.
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

} // namespace

} // namespace free_breakfast::device

extern "C" fb_launch_status
fb_trsm_diagonal_f64(const wwr_stream stream, const int32_t jb, const int32_t ncols, const double s,
                     const double *a, const int32_t lda, const bool a_trans, const bool upper,
                     const bool unit, double *b, const int32_t ldb, const bool b_trans) {
  return free_breakfast::device::launch(stream, jb, ncols, s, a, lda, a_trans, upper, unit, b, ldb,
                                        b_trans);
}
