/// @file
/// @brief gemmwich: the sandwich product C = OpX(X)^H OpF(F) OpX(X), as two gemms

export module free_breakfast.linalg.blas:gemmwich;

import :shared; // raw_blas_handle, SandwichOp, kAdjointOp
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.wwr;       // CreatePolicy, usual_fp, gemm<T, IntT>, ScopedPointerMode

export namespace free_breakfast {

/// @brief C = OpX(X)^H OpF(F) OpX(X), for n x m OpX(X), n x n F and m x m C,
/// through an n x m `work`.
///
/// ^H is T's adjoint: the transpose on a real T, the conjugate transpose on a
/// complex one, so on a real T this is OpX(X)^T OpF(F) OpX(X). `OpX` and `OpF`
/// are each WWRBLAS_OP_N or that adjoint (SandwichOp): WWRBLAS_OP_T on a real
/// T, WWRBLAS_OP_C on a complex one. They are fixed at compile time, and each
/// one not given is WWRBLAS_OP_N. Every matrix is column-major and contiguous,
/// so its leading dimension is its row count as stored:
///
/// | Matrix | Stored as | Leading dimension |
/// | --- | --- | --- |
/// | X, OpX = N | n x m | n |
/// | X, OpX = T / C | m x n | m |
/// | F | n x n | n |
/// | work | n x m | n |
/// | C | m x m | m |
///
/// So OpX = N with X stored n x m is X^H OpF(F) X, the projection of F onto
/// X's m columns. Two gemms, with no host synchronization between them:
///
///   work = OpF(F) OpX(X)
///   C    = OpX(X)^H work
///
/// `work` is scratch: its contents on entry are ignored and on return are
/// OpF(F) OpX(X). `work` must not alias X, F or C, and C must not alias X or
/// F; F and C need not be symmetric (or Hermitian), and C is overwritten, not
/// accumulated into.
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
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N,
          wwr::wwrblasOperation_t OpF = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires SandwichOp<OpX, T> && SandwichOp<OpF, T>
[[nodiscard]] wwr::wwrblasStatus_t gemmwich(const H &handle, int32_t m, int32_t n, const T *x,
                                            const T *f, T *work, T *c) {
  // OpX(X)^H, as the op on X in the second gemm: the adjoint for OpX = N, N for
  // OpX = the adjoint (SandwichOp admits nothing else).
  constexpr auto op_x_adjoint = OpX == wwr::WWRBLAS_OP_N ? kAdjointOp<T> : wwr::WWRBLAS_OP_N;
  const int32_t ldx = OpX == wwr::WWRBLAS_OP_N ? n : m;

  const wwr::wwrblasHandle_t raw = raw_blas_handle(handle);
  // The bare-handle constructor borrows rather than co-owns: fine, since the
  // guard lives only for this call and the caller's handle outlives it.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t, CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{raw, wwr::WWRBLAS_POINTER_MODE_HOST};

  // work (n x m) = OpF(F) (n x n) * OpX(X) (n x m)
  if (const auto status =
          wwr::gemm<T, int32_t>(raw, OpF, OpX, n, m, n, &kOne<T>, f, n, x, ldx, &kZero<T>, work, n);
      status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }
  // C (m x m) = OpX(X)^H (m x n) * work (n x m)
  return wwr::gemm<T, int32_t>(raw, op_x_adjoint, wwr::WWRBLAS_OP_N, m, m, n, &kOne<T>, x, ldx,
                               work, n, &kZero<T>, c, m);
}

/// @brief C = OpX(X)^H OpF(F) OpX(X), for n x n X, F and C, through an n x n
/// `work`.
///
/// The overload above with m = n: `gemmwich(...)` is X^H F X, and OpX = the
/// adjoint (`gemmwich<WWRBLAS_OP_T>(...)` on a real T, `<WWRBLAS_OP_C>` on a
/// complex one) is X F X^H. `work` is left holding OpF(F) OpX(X). Everything
/// else the overload above says applies here.
template <wwr::wwrblasOperation_t OpX = wwr::WWRBLAS_OP_N,
          wwr::wwrblasOperation_t OpF = wwr::WWRBLAS_OP_N, wwr::usual_fp T,
          wwr::extension::blas_handle H>
  requires SandwichOp<OpX, T> && SandwichOp<OpF, T>
[[nodiscard]] wwr::wwrblasStatus_t gemmwich(const H &handle, int32_t n, const T *x, const T *f,
                                            T *work, T *c) {
  // Qualified, so argument-dependent lookup cannot weigh in on which gemmwich.
  return free_breakfast::gemmwich<OpX, OpF>(handle, n, n, x, f, work, c);
}

} // namespace free_breakfast
