/// @file
/// @brief A general square system A X = B solved on the GPU by the LU
/// factorization with partial pivoting, P A = L U: wwr's dense-solver getrf,
/// its scratch and flag taken from DeviceResources, then a solve of the tree's
/// own (the pivots' row interchanges and two trsm recipe calls).
///
/// The README has why the solve is not the library's getrs, and the
/// concurrent-process check getrf's HIP path was given.

export module free_breakfast.recipes.lu;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;

export namespace free_breakfast {

/// @brief Solve A X = B in place: @p a factored, P A = L U (LAPACK's getrf),
/// and, when that leaves U nonsingular, @p b overwritten by X: B <- P B, then
/// L^-1 B and U^-1 B by the trsm recipe.
///
/// @p a is n x n and @p b n x @p nrhs, both column-major and packed (leading
/// dimension n). A need not be symmetric, nor definite. On return @p a holds
/// the factors L (unit diagonal, below) and U (on and above), whatever the
/// flag. When U(i, i) is exactly zero for some i the solve is not attempted,
/// @p b is left as it was, and the returned flag is getrf's: the first such
/// i, 1-based.
///
/// The work and the pivots are res.device_workspace()'s, grown to getrf's size
/// for n if it is smaller: the one allocation this can make (the solve takes
/// nothing beyond the gemm the BLAS handle's fixed workspace serves). Neither
/// @p a nor @p b may point into it. res.info() takes getrf's flag. Everything
/// is queued on res.stream(), and the call synchronizes it once, to read that
/// flag; the solve after it is queued and not waited for. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @param res   Supplies the stream, the solver and BLAS handles, the scratch
///              and the flag.
/// @param n     Matrix order. n = 0 is empty and succeeds.
/// @param nrhs  Columns of @p b.
/// @param a     In: A. Out: its LU factors.
/// @param b     In: B. Out: X, when the returned flag is 0.
/// @return 0 when U is nonsingular and @p b holds X, else getrf's flag;
///         InvalidConfig for a negative @p n or @p nrhs; or a GpuRuntime
///         error: the first runtime, solver, launch or BLAS call that failed,
///         or an argument getrf rejected.
[[nodiscard]] Result<int32_t> lu_solve(DeviceResources &res, int32_t n, int32_t nrhs, double *a,
                                       double *b);

/// @brief getrf's work for order @p n, in doubles, as @ref lu_solve sizes it:
/// the solver's own query through res.solver(), a host computation. 0 for
/// n <= 0.
///
/// @return The size, or the GpuRuntime error of the query.
[[nodiscard]] Result<size_t> getrf_lwork(DeviceResources &res, int32_t n);

/// @brief What @ref lu_solve takes at order @p n, whose @ref getrf_lwork is
/// @p lwork: getrf's work and the n pivots, the device workspace. Empty for
/// n <= 0.
///
/// It takes the size rather than a handle so that a footprint computed before
/// the run (the pipeline's FootprintDimensions::getrf_lwork) holds none.
[[nodiscard]] MemoryFootprint lu_solve_footprint(int32_t n, size_t lwork);

} // namespace free_breakfast
