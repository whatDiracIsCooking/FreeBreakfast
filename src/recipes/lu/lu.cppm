/// @file
/// @brief A general square system A X = B solved on the GPU by the LU
/// factorization with partial pivoting, P A = L U: wwr's dense-solver getrf
/// and getrs, their scratch and flag taken from DeviceResources.
///
/// The README has why the legacy pair, and the concurrent-process check its
/// HIP path was given.

export module free_breakfast.recipes.lu;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Solve A X = B in place: @p a factored, P A = L U (LAPACK's getrf),
/// and, when that leaves U nonsingular, @p b overwritten by X (getrs).
///
/// @p a is n x n and @p b n x @p nrhs, both column-major and packed (leading
/// dimension n). A need not be symmetric, nor definite. On return @p a holds
/// the factors L (unit diagonal, below) and U (on and above), whatever the
/// flag. When U(i, i) is exactly zero for some i the solve is not attempted,
/// @p b is left as it was, and the returned flag is getrf's: the first such
/// i, 1-based.
///
/// The work and the pivots are res.device_workspace()'s, grown to getrf's size
/// for n if it is smaller: the one allocation this can make. Neither @p a nor
/// @p b may point into it. res.info() takes the flags. Everything is queued on
/// res.stream(), and the call synchronizes it twice, to read getrf's flag and
/// then getrs's. The device res.dev_idx() names is made current for the call
/// and the previous one restored.
///
/// @param res   Supplies the stream, the solver handle, the scratch and the
///              flag.
/// @param n     Matrix order. n = 0 is empty and succeeds.
/// @param nrhs  Columns of @p b.
/// @param a     In: A. Out: its LU factors.
/// @param b     In: B. Out: X, when the returned flag is 0.
/// @return 0 when U is nonsingular and @p b holds X, else getrf's flag;
///         InvalidConfig for a negative @p n or @p nrhs; or a GpuRuntime
///         error: the first runtime or solver call that failed, or an argument
///         getrf or getrs rejected.
[[nodiscard]] Result<int32_t> lu_solve(DeviceResources &res, int32_t n, int32_t nrhs, double *a,
                                       double *b);

} // namespace free_breakfast
