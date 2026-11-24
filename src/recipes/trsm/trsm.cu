// The device half of free_breakfast.recipes.trsm: the diagonal-block solve.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "trsm_bridge.h"

namespace free_breakfast::device {

namespace {

// One thread block per kTrsmColumns right-hand sides. The triangle and the
// block's tile of B go to shared memory, and the solve is forward substitution
// over them, row k per step: row k divided by the pivot, a barrier, the rows
// below it lose their multiple of row k, a barrier.
//
// An upper M is solved as a lower one by reversing its rows and columns, and
// the tile's rows with them: M(p, q) for p <= q goes to tri at
// (jb-1-p, jb-1-q), which is on or below the diagonal. So one loop serves both
// triangles, and only the loads and stores know which it is.
//
// `tri[j][i]` is the flipped M's (i, j) and `x[c][i]` the tile's (i, c), so
// the update's threads, consecutive in i, read consecutive words of both.
template <typename T>
__global__ void trsm_diagonal_block_kernel(const int32_t jb, const int32_t ncols, const T s,
                                           const T *__restrict__ a, const int64_t lda,
                                           const bool a_trans, const bool upper, const bool unit,
                                           T *__restrict__ b, const int64_t ldb,
                                           const bool b_trans) {
  __shared__ T tri[kTrsmBlock][kTrsmBlock];
  __shared__ T x[kTrsmColumns][kTrsmBlock];

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
      tri[j][i] = T{1};
      continue;
    }
    const int64_t p = flip(i);
    const int64_t q = flip(j);
    tri[j][i] = a_trans ? a[q + (p * lda)] : a[p + (q * lda)];
  }
  for (int32_t e = tid; e < elements; e += stride) {
    int32_t r = 0;
    int32_t c = 0;
    const int64_t at = b_index(e, r, c);
    x[c][flip(r)] = s * b[at];
  }
  __syncthreads();

  for (int32_t k = 0; k < jb; ++k) {
    if (tid < nc) {
      x[tid][k] /= tri[k][k];
    }
    __syncthreads();
    const int32_t rest = jb - k - 1;
    for (int32_t e = tid; e < rest * nc; e += stride) {
      const int32_t i = k + 1 + (e % rest);
      const int32_t c = e / rest;
      x[c][i] -= tri[k][i] * x[c][k];
    }
    __syncthreads();
  }

  for (int32_t e = tid; e < elements; e += stride) {
    int32_t r = 0;
    int32_t c = 0;
    const int64_t at = b_index(e, r, c);
    b[at] = x[c][flip(r)];
  }
}

template <typename T>
wwr::wwrError_t launch(const wwr::wwrStream_t stream, const int32_t jb, const int32_t ncols,
                       const T s, const T *a, const int64_t lda, const bool a_trans,
                       const bool upper, const bool unit, T *b, const int64_t ldb,
                       const bool b_trans) {
  if (jb < 1 || jb > kTrsmBlock || ncols < 1 || lda < 1 || ldb < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto blocks = static_cast<uint32_t>(math::idivup(ncols, kTrsmColumns));
  trsm_diagonal_block_kernel<T><<<blocks, kFourWarps, 0, stream>>>(jb, ncols, s, a, lda, a_trans,
                                                                   upper, unit, b, ldb, b_trans);
  return wwr::wwrGetLastError();
}

} // namespace

wwr::wwrError_t trsm_diagonal_block(const wwr::wwrStream_t stream, const int32_t jb,
                                    const int32_t ncols, const float s, const float *a,
                                    const int64_t lda, const bool a_trans, const bool upper,
                                    const bool unit, float *b, const int64_t ldb,
                                    const bool b_trans) {
  return launch(stream, jb, ncols, s, a, lda, a_trans, upper, unit, b, ldb, b_trans);
}

wwr::wwrError_t trsm_diagonal_block(const wwr::wwrStream_t stream, const int32_t jb,
                                    const int32_t ncols, const double s, const double *a,
                                    const int64_t lda, const bool a_trans, const bool upper,
                                    const bool unit, double *b, const int64_t ldb,
                                    const bool b_trans) {
  return launch(stream, jb, ncols, s, a, lda, a_trans, upper, unit, b, ldb, b_trans);
}

} // namespace free_breakfast::device
