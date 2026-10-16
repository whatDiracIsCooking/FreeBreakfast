/// @file
/// @brief gemmwich: the sandwich product C = op_x(X)^T op_f(F) op_x(X), as two gemms

export module free_breakfast.linalg.blas:gemmwich;

import std;
import :gemm; // raw_blas_handle
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.wwr;       // CreatePolicy, real_fp, gemm<T, IntT>, ScopedPointerMode

namespace free_breakfast {

// Not exported. The two ops gemmwich accepts: _C is left out, since on a real T
// it is _T under another name.
constexpr bool is_n_or_t(const wwr::wwrblasOperation_t op) noexcept {
  return op == wwr::WWRBLAS_OP_N || op == wwr::WWRBLAS_OP_T;
}

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief C = op_x(X)^T op_f(F) op_x(X), for n x m op_x(X), n x n F and m x m C,
/// through an n x m `work`.
///
/// `op_x` and `op_f` are each WWRBLAS_OP_N or WWRBLAS_OP_T; anything else
/// returns WWRBLAS_STATUS_INVALID_VALUE before anything is enqueued. Every
/// matrix is column-major and contiguous, so its leading dimension is its row
/// count as stored:
///
/// | Matrix | Stored as | Leading dimension |
/// | --- | --- | --- |
/// | X, op_x = N | n x m | n |
/// | X, op_x = T | m x n | m |
/// | F | n x n | n |
/// | work | n x m | n |
/// | C | m x m | m |
///
/// So op_x = N with X stored n x m is X^T op_f(F) X, the projection of F onto
/// X's m columns. Two gemms, with no host synchronization between them:
///
///   work = op_f(F) op_x(X)
///   C    = op_x(X)^T work
///
/// `work` is scratch: its contents on entry are ignored and on return are
/// op_f(F) op_x(X). `work` must not alias X, F or C, and C must not alias X or
/// F; F and C need not be symmetric, and C is overwritten, not accumulated into.
///
/// The two gemms read their scalars (one and zero) from host memory, so the
/// handle is held in host pointer mode for the call by wwr's ScopedPointerMode
/// and handed back in whatever mode it had. Reading or setting the mode goes
/// through CreatePolicy: a failure there (a bad handle) aborts, as
/// ScopedDeviceIndex does.
///
/// `handle` is a BlasHandle or a raw wwrblasHandle_t; the work is enqueued on its
/// stream, on the calling thread's current device. Returns the first failing
/// gemm status, or success; a failure leaves C (and possibly `work`)
/// unspecified.
template <wwr::real_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t gemmwich(const H &handle, const wwr::wwrblasOperation_t op_x,
                                            const wwr::wwrblasOperation_t op_f, int32_t m,
                                            int32_t n, const T *x, const T *f, T *work, T *c) {
  if (!is_n_or_t(op_x) || !is_n_or_t(op_f)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  // op_x(X)^T, as the op on X in the second gemm: T for op_x = N, N for op_x = T.
  const auto op_x_transposed = op_x == wwr::WWRBLAS_OP_N ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
  const int32_t ldx = op_x == wwr::WWRBLAS_OP_N ? n : m;

  const wwr::wwrblasHandle_t raw = raw_blas_handle(handle);
  // The bare-handle constructor borrows rather than co-owns: fine, since the
  // guard lives only for this call and the caller's handle outlives it.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{raw, wwr::WWRBLAS_POINTER_MODE_HOST};

  // work (n x m) = op_f(F) (n x n) * op_x(X) (n x m)
  if (const auto status = wwr::gemm<T, int32_t>(raw, op_f, op_x, n, m, n, &kOne<T>, f, n, x, ldx,
                                                &kZero<T>, work, n);
      status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }
  // C (m x m) = op_x(X)^T (m x n) * work (n x m)
  return wwr::gemm<T, int32_t>(raw, op_x_transposed, wwr::WWRBLAS_OP_N, m, m, n, &kOne<T>, x, ldx,
                               work, n, &kZero<T>, c, m);
}

/// @brief C = op(X)^T F op(X), for n x n X, F and C, through an n x n `work`.
///
/// The overload above with op_x = `op`, op_f = WWRBLAS_OP_N and m = n:
/// WWRBLAS_OP_N for X^T F X (the overload below), WWRBLAS_OP_T for X F X^T.
/// `work` is left holding F op(X). Everything else the overload above says
/// applies here.
template <wwr::real_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t gemmwich(const H &handle, const wwr::wwrblasOperation_t op,
                                            int32_t n, const T *x, const T *f, T *work, T *c) {
  // Qualified, so argument-dependent lookup cannot weigh in on which gemmwich.
  return free_breakfast::gemmwich(handle, op, wwr::WWRBLAS_OP_N, n, n, x, f, work, c);
}

/// @brief C = X^T F X, for n x n X, F and C, through an n x n `work`.
///
/// The overload above with op = WWRBLAS_OP_N; everything it says applies here.
template <wwr::real_fp T, wwr::extension::blas_handle H>
[[nodiscard]] wwr::wwrblasStatus_t gemmwich(const H &handle, int32_t n, const T *x, const T *f,
                                            T *work, T *c) {
  // Qualified, as above.
  return free_breakfast::gemmwich(handle, wwr::WWRBLAS_OP_N, n, x, f, work, c);
}

} // namespace free_breakfast
