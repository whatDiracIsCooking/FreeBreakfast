/// @file
/// @brief The Cholesky factorization A = L L^T of a symmetric positive
/// definite matrix on the GPU, in place: blocked, its diagonal blocks in shared
/// memory and its panels over trsm and syrk.
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

} // namespace free_breakfast
