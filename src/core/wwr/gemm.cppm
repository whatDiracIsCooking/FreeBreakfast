/// @file
/// @brief gemm: wwr's gemm for square, contiguous operands

export module free_breakfast.core.wwr:gemm;

import std;
import free_breakfast.core.types;
import wwr.wrappers.common; // usual_fp
import wwr.wrappers.blas;   // gemm<T, IntT>
import wwr.extension.blas;  // blas_handle, wwrblasHandle_t, wwrblasOperation_t

namespace free_breakfast {

// Not exported. wwr's own unwrapping (ScopedPointerMode::raw_handle) is
// private, so the two arms of blas_handle are resolved here.
template <wwr::extension::blas_handle H>
wwr::wwrblasHandle_t raw_blas_handle(const H &handle) noexcept {
  if constexpr (std::same_as<H, wwr::wwrblasHandle_t>) {
    return handle;
  } else {
    return handle.get();
  }
}

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief C = alpha * op_a(A) * op_b(B) + beta * C, for n x n A, B and C.
///
/// wwr::gemm with m = n = k = `n` and every leading dimension `n`: each matrix
/// is column-major and contiguous. `op_a` / `op_b` are WWRBLAS_OP_N, _T or _C
/// (_C is the conjugate transpose; on a real T it is the same as _T). For a
/// rectangular shape or a padded leading dimension, call wwr::gemm.
///
/// `handle` is a BlasHandle or a raw wwrblasHandle_t. The work is enqueued on
/// the stream the handle is bound to, on the calling thread's current device.
/// `alpha` and `beta` are read as the handle's pointer mode says: host memory by
/// default (`&kOne<T>`), device memory under WWRBLAS_POINTER_MODE_DEVICE.
/// With `beta` zero, C need not be initialised. C must not alias A or B.
///
/// Returns the library's status, as wwr::gemm does, so the caller picks the
/// tier: FBF_GPU_MUST, or gpu_status where a Status is wanted.
template <wwr::usual_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t
gemm(const H &handle, const wwr::wwrblasOperation_t op_a, const wwr::wwrblasOperation_t op_b,
     int32_t n, const T *alpha, const T *a, const T *b, const T *beta, T *c) {
  return wwr::gemm<T, int32_t>(raw_blas_handle(handle), op_a, op_b, n, n, n, alpha, a, n, b, n,
                               beta, c, n);
}

/// @brief C = alpha * op(A) * op(B) + beta * C, for n x n A, B and C, with the
/// same op on both.
///
/// The overload above with op_a = op_b = `op_both`; everything it says about the
/// handle, the scalars, C and the return applies here. With WWRBLAS_OP_T this
/// is C = alpha * A^T B^T + beta * C, i.e. alpha * (B A)^T: the transpose of the
/// product taken in the other order, not of A B.
template <wwr::usual_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t gemm(const H &handle, const wwr::wwrblasOperation_t op_both,
                                        int32_t n, const T *alpha, const T *a, const T *b,
                                        const T *beta, T *c) {
  // Qualified, so argument-dependent lookup cannot weigh in on which gemm.
  return free_breakfast::gemm(handle, op_both, op_both, n, alpha, a, b, beta, c);
}

/// @brief C = alpha * A * B + beta * C, for n x n A, B and C, none transposed.
///
/// The overload above with op_both = WWRBLAS_OP_N.
template <wwr::usual_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t gemm(const H &handle, int32_t n, const T *alpha, const T *a,
                                        const T *b, const T *beta, T *c) {
  // Qualified, as above.
  return free_breakfast::gemm(handle, wwr::WWRBLAS_OP_N, n, alpha, a, b, beta, c);
}

} // namespace free_breakfast
