/// @file
/// @brief Half-precision GEMM through the vendor BLAS: does cuBLAS / hipBLAS
/// multiply __half matrices on this card, and how well.
///
/// Both accumulators go through wwr.blas's neutral wwrblasHgemm /
/// wwrblasGemmEx; wwr::gemm<T> is still S/D/C/Z only.

export module free_breakfast.experimental.fp16_gemm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
// wwrHalf: not re-exported by free_breakfast.core.wwr, and this is not src/.
import wwr.fp16;

export namespace free_breakfast {

/// @brief What the products are summed in. The inputs and C are __half either way.
enum class Fp16Accumulate : uint8_t {
  /// The classic Hgemm: alpha, beta and the sum in __half (compute type 16F).
  kHalf,
  /// GemmEx with __half A, B and C, alpha, beta and the sum in float
  /// (compute type 32F): the tensor-core path's natural accumulator.
  kFloat,
};

/// @brief C = A B for column-major, untransposed m x k A, k x n B and m x n C,
/// every one __half and packed (lda = m, ldb = k, ldc = m)
///
/// Queued on res.stream() through res.blas(), with alpha = 1 and beta = 0 read
/// from host memory; nothing synchronizes. Returns the vendor's status: a
/// backend that does not support the call says so here, not by aborting.
[[nodiscard]] wwr::wwrblasStatus_t gemm_fp16(DeviceResources &res, Fp16Accumulate accumulate,
                                             int32_t m, int32_t n, int32_t k, const wwr::wwrHalf *a,
                                             const wwr::wwrHalf *b, wwr::wwrHalf *c);

} // namespace free_breakfast
