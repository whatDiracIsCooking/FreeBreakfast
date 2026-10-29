/// @file
/// @brief fp64 GEMM through the cuBLASLt / hipBLASLt heuristic path, on the
/// shape `nevpt2::rdms`' digest runs: does the Lt surface pick a better
/// algorithm than plain Dgemm, and does hipBLASLt offer an fp64 one at all.
///
/// The three digest GEMMs are most of an SC-NEVPT2 run, and their shape is a
/// short m (n_act^2), a long n (n_act^4) and a long k (the tile width). cuBLAS
/// runs that shape well below the m = 3 n_act^2 one a fused digest would give,
/// so the library's algorithm choice -- not the card -- may be the limit.
/// CUDA has fp64 emulation as its lever; ROCm has none, which is what makes
/// the Lt heuristic worth ruling in or out there.

export module free_breakfast.experimental.blas_lt;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
// The Lt surface: not re-exported by free_breakfast.core.wwr, and this is not src/.
import wwr.extension.blaslt;

export namespace free_breakfast {

/// @brief An Lt library handle. It owns no stream: wwrblasLtMatmul takes one
/// per call, unlike BlasHandle's stream-bound handle.
using BlasLtHandle = wwr::extension::BlasLtHandleWrapper<CreatePolicy<wwr::wwrblasLtStatus_t>,
                                                         DestroyPolicy<wwr::wwrblasLtStatus_t>>;

/// @brief What one @ref gemm_lt call found.
struct LtResult {
  /// @brief How many algorithms the heuristic offered for the shape. **0 is an
  /// answer, not a failure**: the backend has no fp64 algorithm for it, which
  /// is the question this component exists to ask of hipBLASLt.
  int32_t algorithms;
  /// @brief The matmul's status, or the heuristic's when @ref algorithms is 0.
  wwr::wwrblasLtStatus_t status;
};

/// @brief C += op(A) op(B) in fp64 through the Lt heuristic, column-major, with
/// `wwr::gemm`'s argument order and the digest's alpha = beta = 1
///
/// Queued on res.stream(); nothing synchronizes. The heuristic is asked for one
/// algorithm per call, so the search is in the timing exactly as a caller
/// doing no caching would pay it. `workspace` may be null with
/// `workspace_bytes` 0, which restricts the heuristic to algorithms needing
/// none.
///
/// Returns the vendor's status rather than aborting, as `gemm_fp16` does: a
/// backend that cannot run the call says so here.
[[nodiscard]] LtResult gemm_lt(DeviceResources &res, BlasLtHandle &lt, wwr::wwrblasOperation_t op_a,
                               wwr::wwrblasOperation_t op_b, int64_t m, int64_t n, int64_t k,
                               const double *a, int64_t lda, const double *b, int64_t ldb,
                               double *c, int64_t ldc, void *workspace, size_t workspace_bytes);

} // namespace free_breakfast
