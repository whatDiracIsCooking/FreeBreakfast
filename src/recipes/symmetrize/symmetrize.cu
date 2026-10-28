// The device half of free_breakfast.recipes.symmetrize: the in-place average
// with the transpose, and the triangle mirror. One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "symmetrize_bridge.h"

namespace free_breakfast::device {

namespace {

// A tile is kTile x kTile elements, fixed rather than kWarpSize: the average
// holds two of them, 2 x 32 x 33 doubles (16.5 KiB), where a 64-wide tile on a
// wave64 build would be 66 KiB, past CDNA's 64 KiB of LDS. The extra column
// keeps a read along a tile's row (stride kTile + 1) off a single bank.
//
// kTileRows threads down each column, each taking every kTileRows-th column of
// the tile: eight warps per block, so a block's loads are in flight together.
// One warp per block would leave the SM bound by its resident-block limit (16
// on sm_86), at a third of its threads.
constexpr int32_t kTile = 32;
constexpr int32_t kTileRows = 8;
static_assert((kTile * kTileRows) % kWarpSize == 0, "a block must be whole warps");
static_assert(kTile * kTileRows <= kMaxThreads);

// Block (bi, bj) of a T x T grid of tiles owns lower tile (bi, bj) and, off
// the diagonal, its mirror (bj, bi); the blocks above the diagonal leave at
// once. `lower[c][r]` is element (i0 + r, j0 + c), so a thread's x is a row
// and a warp's loads and stores run down a column.
//
// (a + b) / 2 is the same bits as (b + a) / 2: IEEE addition is commutative,
// and a sum times a constant has no product to contract into an FMA. So the
// two threads that write (i, j) and (j, i), adding the same pair in opposite
// orders, write the same value.
__global__ void symmetrize_kernel(const int32_t n, double *__restrict__ a, const int64_t lda) {
  const uint32_t bi = blockIdx.x;
  const uint32_t bj = blockIdx.y;
  if (bi < bj) {
    return;
  }
  __shared__ double lower[kTile][kTile + 1];
  __shared__ double upper[kTile][kTile + 1];
  const bool diagonal = bi == bj;
  const int64_t i0 = int64_t{bi} * kTile;
  const int64_t j0 = int64_t{bj} * kTile;
  const auto r = static_cast<int32_t>(threadIdx.x);

  for (auto c = static_cast<int32_t>(threadIdx.y); c < kTile; c += kTileRows) {
    if (i0 + r < n && j0 + c < n) {
      lower[c][r] = a[(i0 + r) + ((j0 + c) * lda)];
    }
    if (!diagonal && j0 + r < n && i0 + c < n) {
      upper[c][r] = a[(j0 + r) + ((i0 + c) * lda)];
    }
  }
  __syncthreads();

  // On the diagonal the mirror of (r, c) is the same tile's (c, r); off it,
  // the other tile's.
  for (auto c = static_cast<int32_t>(threadIdx.y); c < kTile; c += kTileRows) {
    if (diagonal) {
      if (i0 + r < n && j0 + c < n) {
        a[(i0 + r) + ((j0 + c) * lda)] = (lower[c][r] + lower[r][c]) * kHalf<double>;
      }
      continue;
    }
    if (i0 + r < n && j0 + c < n) {
      a[(i0 + r) + ((j0 + c) * lda)] = (lower[c][r] + upper[r][c]) * kHalf<double>;
    }
    if (j0 + r < n && i0 + c < n) {
      a[(j0 + r) + ((i0 + c) * lda)] = (upper[c][r] + lower[r][c]) * kHalf<double>;
    }
  }
}

// The same grid; block (bi, bj) copies source tile (bi, bj) when FromLower,
// (bj, bi) when not, transposed onto its mirror. The source is read whole, and
// only the strict other triangle is written: on a diagonal tile, the half of
// it across the diagonal; off it, the whole destination, which no other block
// reads or writes.
template <bool FromLower>
__global__ void mirror_triangle_kernel(const int32_t n, double *__restrict__ a, const int64_t lda) {
  const uint32_t bi = blockIdx.x;
  const uint32_t bj = blockIdx.y;
  if (bi < bj) {
    return;
  }
  __shared__ double tile[kTile][kTile + 1];
  const int64_t row0 = int64_t{FromLower ? bi : bj} * kTile;
  const int64_t col0 = int64_t{FromLower ? bj : bi} * kTile;
  const auto r = static_cast<int32_t>(threadIdx.x);

  for (auto c = static_cast<int32_t>(threadIdx.y); c < kTile; c += kTileRows) {
    if (row0 + r < n && col0 + c < n) {
      tile[c][r] = a[(row0 + r) + ((col0 + c) * lda)];
    }
  }
  __syncthreads();

  // Destination (col0 + r, row0 + c) is source (row0 + c, col0 + r).
  for (auto c = static_cast<int32_t>(threadIdx.y); c < kTile; c += kTileRows) {
    const int64_t i = col0 + r;
    const int64_t j = row0 + c;
    if (i < n && j < n && (FromLower ? i < j : i > j)) {
      a[i + (j * lda)] = tile[r][c];
    }
  }
}

// The T x T grid over an n x n matrix, or 0 tiles when T passes the grid's
// y limit (2^16 - 1, the tighter of the two on both backends): a matrix of
// order 2,097,121 or more, 35 TB of doubles.
uint32_t tiles(const int32_t n) {
  constexpr int32_t kMaxGridY = 65535;
  const int32_t t = math::idivup(n, kTile);
  return t > kMaxGridY ? 0 : static_cast<uint32_t>(t);
}

} // namespace

wwr::wwrError_t symmetrize(const wwr::wwrStream_t stream, const int32_t n, double *a,
                           const int64_t lda) {
  const uint32_t t = n < 1 ? 0 : tiles(n);
  if (t == 0 || lda < n) {
    return wwr::wwrErrorInvalidValue;
  }
  symmetrize_kernel<<<dim3{t, t}, dim3{kTile, kTileRows}, 0, stream>>>(n, a, lda);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t mirror_triangle(const wwr::wwrStream_t stream, const bool from_lower,
                                const int32_t n, double *a, const int64_t lda) {
  const uint32_t t = n < 1 ? 0 : tiles(n);
  if (t == 0 || lda < n) {
    return wwr::wwrErrorInvalidValue;
  }
  if (from_lower) {
    mirror_triangle_kernel<true><<<dim3{t, t}, dim3{kTile, kTileRows}, 0, stream>>>(n, a, lda);
  } else {
    mirror_triangle_kernel<false><<<dim3{t, t}, dim3{kTile, kTileRows}, 0, stream>>>(n, a, lda);
  }
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
