/// @file
/// @brief gemm_split_fp32: the two splits, then per K chunk the `terms` Sgemm
/// calls and the fp64 fold.

module;

// The launchers, defined in split_gemm_kernels.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "split_gemm_bridge.h"

module free_breakfast.experimental.split_gemm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

int64_t split_scratch_floats(const int64_t m, const int64_t n, const int64_t k) noexcept {
  return (2 * ((m * k) + (k * n))) + (m * n);
}

// Where one K chunk starts in an operand, and what its packed leading
// dimension is. When K is the operand's LEADING extent a chunk is a pointer
// offset of k0; when it is the trailing one a chunk is k0 whole columns, so
// the offset is k0 leading dimensions.
//
// Which case an operand is in depends on the operand as well as its op, and
// the two are mirrored: A is m x k under OP_N (K trailing) and k x m under
// OP_T (K leading), while B is k x n under OP_N (K LEADING) and n x k under
// OP_T. So A leads with K when transposed and B when it is not.
struct ChunkStride {
  int64_t offset_per_k;
  int64_t packed_ld;
};

ChunkStride chunk_stride(const bool k_leads, const int64_t k_extent, const int64_t other_extent,
                         const int64_t ld) {
  if (k_leads) {
    return {1, k_extent};
  }
  return {ld, other_extent};
}

Status gemm_split_fp32(DeviceResources &res, const wwr::wwrblasOperation_t op_a,
                       const wwr::wwrblasOperation_t op_b, const int64_t m, const int64_t n,
                       const int64_t k, const double *const a, const int64_t lda,
                       const double *const b, const int64_t ldb, double *const c, const int64_t ldc,
                       const int64_t k_chunk, const int32_t terms, const SplitScratch &scratch) {
  if (terms < 1 || terms > 3) {
    return err_unsupported("gemm_split_fp32: terms must be 1, 2 or 3, not {}", terms);
  }
  if (m <= 0 || n <= 0 || k <= 0 || k_chunk <= 0) {
    return err_unsupported("gemm_split_fp32: m, n, k and k_chunk must be positive, not "
                           "{}, {}, {} and {}",
                           m, n, k, k_chunk);
  }
  const ChunkStride a_stride = chunk_stride(op_a != wwr::WWRBLAS_OP_N, k, m, lda);
  const ChunkStride b_stride = chunk_stride(op_b == wwr::WWRBLAS_OP_N, k, n, ldb);
  if (lda != a_stride.packed_ld || ldb != b_stride.packed_ld) {
    return err_unsupported("gemm_split_fp32: the operands must be packed -- lda {} and ldb {} "
                           "against the packed {} and {}; the K chunking slices them in place",
                           lda, ldb, a_stride.packed_ld, b_stride.packed_ld);
  }

  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(device::split_hi_lo(stream, a, scratch.a_hi, scratch.a_lo, m * k)));
  FBF_TRY(gpu_status(device::split_hi_lo(stream, b, scratch.b_hi, scratch.b_lo, k * n)));

  // Borrowed for the calls, as recipes' gemmwich does: alpha and beta are on
  // the host.
  const wwr::wwrblasHandle_t handle = res.blas().get();
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{handle, wwr::WWRBLAS_POINTER_MODE_HOST};

  constexpr float kOneF = 1.0F;
  constexpr float kZeroF = 0.0F;
  for (int64_t k0 = 0; k0 < k; k0 += k_chunk) {
    const int64_t extent = std::min(k_chunk, k - k0);
    const int64_t a_skip = k0 * a_stride.offset_per_k;
    const int64_t b_skip = k0 * b_stride.offset_per_k;

    // The three products, each into the same fp32 accumulator: the first
    // overwrites it, the corrections add. A_lo B_lo is dropped (see the
    // interface).
    const std::array<std::pair<const float *, const float *>, 3> products{
        {{scratch.a_hi + a_skip, scratch.b_hi + b_skip},
         {scratch.a_hi + a_skip, scratch.b_lo + b_skip},
         {scratch.a_lo + a_skip, scratch.b_hi + b_skip}}};
    for (int32_t term = 0; term < terms; ++term) {
      const auto [a_plane, b_plane] = products[static_cast<size_t>(term)];
      FBF_TRY(gpu_status(wwr::gemm<float, int64_t>(handle, op_a, op_b, m, n, extent, &kOneF,
                                                   a_plane, lda, b_plane, ldb,
                                                   term == 0 ? &kZeroF : &kOneF, scratch.c32, m)));
    }

    // c32 is m x n packed; C may be strided, so fold column by column when it
    // is. One launch covers the packed case, which is the digest's.
    if (ldc == m) {
      FBF_TRY(gpu_status(device::accumulate_f32(stream, scratch.c32, c, m * n)));
    } else {
      for (int64_t column = 0; column < n; ++column) {
        FBF_TRY(gpu_status(
            device::accumulate_f32(stream, scratch.c32 + (column * m), c + (column * ldc), m)));
      }
    }
  }
  return {};
}

} // namespace free_breakfast
