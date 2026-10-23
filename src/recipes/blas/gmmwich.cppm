/// @file
/// @brief gmmwich: the diagonal sandwich C = OpX(X)^H diag(d) OpX(X), as a dgmm
/// and a gemm

export module free_breakfast.recipes.blas:gmmwich;

import :shared; // raw_blas_handle, SandwichOp, kAdjointOp
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.wwr;       // CreatePolicy, usual_fp, gemm, dgmm, ScopedPointerMode

export namespace free_breakfast {

/// @brief C = OpX(X)^H diag(d) OpX(X), for n x m OpX(X), length-n d and m x m
/// C, through a `work` the size of X.
///
/// gemmwich with F = diag(d), without forming F: the diagonal is applied to X by
/// a dgmm in O(nm) rather than by a gemm against a dense diagonal in O(n^2 m).
/// ^H is T's adjoint, as in gemmwich: the transpose on a real T, the conjugate
/// transpose on a complex one. `OpX` is WWRBLAS_OP_N or that adjoint
/// (SandwichOp: WWRBLAS_OP_T on a real T, WWRBLAS_OP_C on a complex one), fixed
/// at compile time, and WWRBLAS_OP_N when not given. Every matrix is
/// column-major and contiguous, and d is contiguous (stride 1) and of type T, so
/// on a complex T a real diagonal is stored with zero imaginary parts:
///
/// | OpX | X stored as | work = | C = |
/// | --- | --- | --- | --- |
/// | N | n x m, ld n | diag(d) X (n x m) | X^H work |
/// | T / C | m x n, ld m | X diag(d) (m x n) | work X^H |
///
/// So `work` always has X's stored shape and leading dimension, and the diagonal
/// lands on whichever side of X needs no adjoint. OpX = N is X^H diag(d) X;
/// OpX = the adjoint is X diag(d) X^H, which for an eigenvector matrix X = U is
/// the spectral recombination U f(Lambda) U^H.
///
/// `work` is scratch: its contents on entry are ignored and on return are as in
/// the table. `work` must not alias X, d or C, and C must not alias X or d; C is
/// overwritten, not accumulated into, and for a real d is Hermitian (symmetric,
/// on a real T) up to rounding.
///
/// The dgmm takes no scalars; the gemm reads one and zero from host memory, so
/// the handle is held in host pointer mode for the call by wwr's
/// ScopedPointerMode and handed back in whatever mode it had, as in gemmwich.
///
/// `handle` is a BlasHandle or a raw wwrblasHandle_t; the work is enqueued on its
/// stream, on the calling thread's current device, with no host synchronization
/// between the two calls. Returns the first failing status, or success; a
/// failure leaves C (and possibly `work`) unspecified.
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires SandwichOp<OpX, T>
[[nodiscard]] wwr::wwrblasStatus_t gmmwich(const H &handle, int32_t m, int32_t n, const T *x,
                                           const T *d, T *work, T *c) {
  constexpr bool x_is_n = OpX == wwr::WWRBLAS_OP_N;
  // X's (and so work's) stored shape: n x m for N, m x n for the adjoint.
  const int32_t rows = x_is_n ? n : m;
  const int32_t cols = x_is_n ? m : n;

  const wwr::wwrblasHandle_t raw = raw_blas_handle(handle);
  // Borrowed, as in gemmwich: the guard lives only for this call.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{raw, wwr::WWRBLAS_POINTER_MODE_HOST};

  // work = diag(d) X (N) or X diag(d) (adjoint): d always multiplies X's
  // length-n side.
  constexpr auto side = x_is_n ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT;
  if (const auto status = wwr::dgmm<T, int32_t>(raw, side, rows, cols, x, rows, d, 1, work, rows);
      status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }
  // C (m x m) = X^H work (N) or work X^H (adjoint), contracting over n.
  constexpr auto op_a = x_is_n ? kAdjointOp<T> : wwr::WWRBLAS_OP_N;
  constexpr auto op_b = x_is_n ? wwr::WWRBLAS_OP_N : kAdjointOp<T>;
  const T *a = x_is_n ? x : work;
  const T *b = x_is_n ? work : x;
  return wwr::gemm<T, int32_t>(raw, op_a, op_b, m, m, n, &kOne<T>, a, rows, b, rows, &kZero<T>, c,
                               m);
}

/// @brief C = OpX(X)^H diag(d) OpX(X), for n x n X and C and length-n d, through
/// an n x n `work`.
///
/// The overload above with m = n: `gmmwich(...)` is X^H diag(d) X, and OpX = the
/// adjoint (`gmmwich<WWRBLAS_OP_T>(...)` on a real T, `<WWRBLAS_OP_C>` on a
/// complex one) is X diag(d) X^H. Everything the overload above says applies
/// here.
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires SandwichOp<OpX, T>
[[nodiscard]] wwr::wwrblasStatus_t gmmwich(const H &handle, int32_t n, const T *x, const T *d,
                                           T *work, T *c) {
  // Qualified, so argument-dependent lookup cannot weigh in on which gmmwich.
  return free_breakfast::gmmwich<OpX>(handle, n, n, x, d, work, c);
}

} // namespace free_breakfast
