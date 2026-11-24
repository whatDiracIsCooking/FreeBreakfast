/// @file
/// @brief The Cholesky factorization A = L L^T of a symmetric positive
/// definite matrix on the GPU, in place: blocked, its diagonal blocks in shared
/// memory and its panels over trsm and syrk; and the solve A X = B over its
/// factor, two of recipes/trsm's solves.
///
/// The README has why this is not the dense solver's potrf.

export module free_breakfast.recipes.cholesky;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Factor the symmetric @p a in place, A = L L^T, L in the lower
/// triangle: LAPACK's potrf with uplo = L, and the same flag.
///
/// @p a is n x n, column-major and packed (element (i, j) at i + n j). Only
/// its lower triangle is read, and on success that triangle holds L; the strict
/// upper triangle is never written. On a leading minor that is not positive the
/// factorization stops there, the returned flag names it, and the lower
/// triangle is left unspecified.
///
/// Everything is queued on res.stream(), with res.blas() in host pointer mode
/// (its default), and res.info() takes the flag; the call synchronizes the
/// stream once, to read it. Nothing is drawn from res.device_workspace(), so
/// @p a may point into it. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @param res  Supplies the stream, the BLAS handle and the flag.
/// @param n    Matrix order. n = 0 is empty and succeeds.
/// @param a    In: the symmetric operand. Out: L in the lower triangle.
/// @return 0 when A is positive definite, else the order of the first leading
///         minor that is not positive (1-based, potrf's info); InvalidConfig
///         for a negative @p n; or the GpuRuntime error of the first runtime,
///         launch or BLAS call that failed.
[[nodiscard]] Result<int32_t> cholesky(DeviceResources &res, int32_t n, double *a);

/// @brief Solve A X = B in place over cholesky's factor: B <- L^-T L^-1 B,
/// LAPACK's potrs with uplo = L.
///
/// @p l is n x n, column-major and packed, with L in its lower triangle, as
/// cholesky leaves it; only that triangle is read. @p b is n x @p nrhs,
/// column-major and packed. They must not overlap. A factor cholesky flagged
/// is not detected: as in potrs, a zero pivot gives Inf or NaN.
///
/// Two recipe trsm calls (L Y = B, then L^T X = Y), queued on res.stream()
/// with res.blas() in host pointer mode (its default); nothing is synchronized.
/// No solver handle, no flag, and neither the library's trsm workspace nor
/// res.device_workspace() is touched, so @p l or @p b may point into the
/// latter. The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @param res   Supplies the stream and the BLAS handle.
/// @param n     Matrix order.
/// @param nrhs  Number of right-hand sides, B's columns.
/// @param l     The factor, L in the lower triangle.
/// @param b     In: B. Out: X.
/// @return Success, with nothing launched when n or nrhs is 0; InvalidConfig
///         for a negative @p n or @p nrhs; or the GpuRuntime error of the first
///         launch or BLAS call that failed.
[[nodiscard]] Status cholesky_solve(DeviceResources &res, int32_t n, int32_t nrhs, const double *l,
                                    double *b);

} // namespace free_breakfast
