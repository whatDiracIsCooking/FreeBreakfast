/// @file
/// @brief gmmwich: the diagonal sandwich C = OpX(X)^T diag(d) OpX(X), as a dgmm
/// and a gemm

export module free_breakfast.linalg.blas:gmmwich;

import :shared; // raw_blas_handle, RealOp
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.wwr;       // CreatePolicy, real_fp, gemm, dgmm, ScopedPointerMode

export namespace free_breakfast {

/// @brief C = OpX(X)^T diag(d) OpX(X), for n x m OpX(X), length-n d and m x m
/// C, through a `work` the size of X.
///
/// gemmwich with F = diag(d), without forming F: the diagonal is applied to X by
/// a dgmm in O(nm) rather than by a gemm against a dense diagonal in O(n^2 m).
/// `OpX` is WWRBLAS_OP_N or WWRBLAS_OP_T (RealOp), fixed at compile time, and
/// WWRBLAS_OP_N when not given. Every matrix is column-major and contiguous,
/// and d is contiguous (stride 1):
///
/// | OpX | X stored as | work = | C = |
/// | --- | --- | --- | --- |
/// | N | n x m, ld n | diag(d) X (n x m) | X^T work |
/// | T | m x n, ld m | X diag(d) (m x n) | work X^T |
///
/// So `work` always has X's stored shape and leading dimension, and the diagonal
/// lands on whichever side of X needs no transpose. OpX = N is X^T diag(d) X;
/// OpX = T is X diag(d) X^T, which for an eigenvector matrix X = U is the
/// spectral recombination U f(Lambda) U^T.
///
/// `work` is scratch: its contents on entry are ignored and on return are as in
/// the table. `work` must not alias X, d or C, and C must not alias X or d; C is
/// overwritten, not accumulated into, and is symmetric up to rounding.
///
/// The dgmm takes no scalars; the gemm reads one and zero from host memory, so
/// the handle is held in host pointer mode for the call by wwr's
/// ScopedPointerMode and handed back in whatever mode it had, as in gemmwich.
///
/// `handle` is a BlasHandle or a raw wwrblasHandle_t; the work is enqueued on its
/// stream, on the calling thread's current device, with no host synchronization
/// between the two calls. Returns the first failing status, or success; a
/// failure leaves C (and possibly `work`) unspecified.
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N, wwr::real_fp T,
          wwr::extension::blas_handle H>
  requires RealOp<OpX>
[[nodiscard]] wwr::wwrblasStatus_t gmmwich(const H &handle, int32_t m, int32_t n, const T *x,
                                           const T *d, T *work, T *c) {
  constexpr bool x_is_n = OpX == wwr::WWRBLAS_OP_N;
  // X's (and so work's) stored shape: n x m for N, m x n for T.
  const int32_t rows = x_is_n ? n : m;
  const int32_t cols = x_is_n ? m : n;

  const wwr::wwrblasHandle_t raw = raw_blas_handle(handle);
  // Borrowed, as in gemmwich: the guard lives only for this call.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{raw, wwr::WWRBLAS_POINTER_MODE_HOST};

  // work = diag(d) X (N) or X diag(d) (T): d always multiplies X's length-n side.
  constexpr auto side = x_is_n ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT;
  if (const auto status = wwr::dgmm<T, int32_t>(raw, side, rows, cols, x, rows, d, 1, work, rows);
      status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }
  // C (m x m) = X^T work (N) or work X^T (T), contracting over n.
  constexpr auto op_a = x_is_n ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
  constexpr auto op_b = x_is_n ? wwr::WWRBLAS_OP_N : wwr::WWRBLAS_OP_T;
  const T *a = x_is_n ? x : work;
  const T *b = x_is_n ? work : x;
  return wwr::gemm<T, int32_t>(raw, op_a, op_b, m, m, n, &kOne<T>, a, rows, b, rows, &kZero<T>, c,
                               m);
}

/// @brief C = OpX(X)^T diag(d) OpX(X), for n x n X and C and length-n d, through
/// an n x n `work`.
///
/// The overload above with m = n: `gmmwich(...)` is X^T diag(d) X and
/// `gmmwich<WWRBLAS_OP_T>(...)` is X diag(d) X^T. Everything the overload above
/// says applies here.
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N, wwr::real_fp T,
          wwr::extension::blas_handle H>
  requires RealOp<OpX>
[[nodiscard]] wwr::wwrblasStatus_t gmmwich(const H &handle, int32_t n, const T *x, const T *d,
                                           T *work, T *c) {
  // Qualified, so argument-dependent lookup cannot weigh in on which gmmwich.
  return free_breakfast::gmmwich<OpX>(handle, n, n, x, d, work, c);
}

} // namespace free_breakfast
