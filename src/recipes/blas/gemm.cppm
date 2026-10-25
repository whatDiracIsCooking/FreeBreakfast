/// @file
/// @brief gemm: wwr's gemm for packed operands, rectangular or square

export module free_breakfast.recipes.blas:gemm;

import :shared; // raw_blas_handle, BlasOp
import free_breakfast.core.types;
import free_breakfast.core.wwr; // usual_fp, gemm<T, IntT>, blas_handle, wwrblasOperation_t

export namespace free_breakfast {

/// @brief C = alpha * OpA(A) * OpB(B) + beta * C, for m x k OpA(A), k x n OpB(B)
/// and m x n C.
///
/// wwr::gemm with each leading dimension the row count of its matrix as stored:
/// every matrix is column-major and packed, so A is m x k under OP_N and k x m
/// otherwise (lda m or k), B is k x n or n x k (ldb k or n), and ldc is m.
/// `OpA` / `OpB` are WWRBLAS_OP_N, _T or _C (_C is the conjugate transpose; on
/// a real T it is the same as _T), fixed at compile time, and each one not
/// given is WWRBLAS_OP_N: `gemm(...)` is A B, `gemm<WWRBLAS_OP_T>(...)` is
/// A^T B. For a padded leading dimension (a block of a larger matrix) or an op
/// only known at run time, call wwr::gemm.
///
/// `handle` is a BlasHandle or a raw wwrblasHandle_t. The work is enqueued on
/// the stream the handle is bound to, on the calling thread's current device.
/// `alpha` and `beta` are read as the handle's pointer mode says: host memory by
/// default (`&kOne<T>`), device memory under WWRBLAS_POINTER_MODE_DEVICE.
/// With `beta` zero, C need not be initialised. C must not alias A or B.
///
/// Returns the library's status, as wwr::gemm does, so the caller picks the
/// tier: FBF_GPU_MUST, or gpu_status where a Status is wanted.
template <wwr::wwrblasOperation_t OpA = wwr::WWRBLAS_OP_N,
          wwr::wwrblasOperation_t OpB = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires BlasOp<OpA> && BlasOp<OpB>
[[nodiscard]] wwr::wwrblasStatus_t gemm(const H &handle, int32_t m, int32_t n, int32_t k,
                                        const T *alpha, const T *a, const T *b, const T *beta,
                                        T *c) {
  constexpr bool a_is_n = OpA == wwr::WWRBLAS_OP_N;
  constexpr bool b_is_n = OpB == wwr::WWRBLAS_OP_N;
  return wwr::gemm<T, int32_t>(raw_blas_handle(handle), OpA, OpB, m, n, k, alpha, a, a_is_n ? m : k,
                               b, b_is_n ? k : n, beta, c, m);
}

/// @brief C = alpha * OpA(A) * OpB(B) + beta * C, for n x n A, B and C.
///
/// The packed gemm above with m = n = k = `n`, so every leading dimension is
/// `n`; the ops, handle, scalars and status are as there.
template <wwr::wwrblasOperation_t OpA = wwr::WWRBLAS_OP_N,
          wwr::wwrblasOperation_t OpB = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires BlasOp<OpA> && BlasOp<OpB>
[[nodiscard]] wwr::wwrblasStatus_t gemm(const H &handle, int32_t n, const T *alpha, const T *a,
                                        const T *b, const T *beta, T *c) {
  return gemm<OpA, OpB>(handle, n, n, n, alpha, a, b, beta, c);
}

} // namespace free_breakfast
