/// @file
/// @brief Symmetric (Loewdin) orthogonalizer X = S^{-1/2}, on the GPU, over
/// wwr's dense-solver and BLAS dispatch.
///
/// Given a real-symmetric, positive-(semi)definite operand S, this builds
///
///     X = S^{-1/2} = U * Lambda^{-1/2} * U^T
///
/// where S = U Lambda U^T is the symmetric eigendecomposition (the columns of U
/// are eigenvectors, Lambda the ascending eigenvalues). For the SCF overlap this
/// turns the generalized eigenproblem F C = S C eps into a standard one: with
/// F' = X^T F X and C = X C', F' C' = C' eps. Nothing here is specific to SCF; a
/// density-fitting Coulomb metric or a localization overlap reduces to the same
/// eigendecomposition, eigenvalue map and recombination.
///
/// The README has the computation step by step, the conditioning trade-off and
/// why every intermediate is caller-owned.

export module free_breakfast.linalg.orthogonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Default eigenvalue floor for orthogonalize: an eigenvalue at or
/// below it drops its mode.
template <real_fp T>
inline constexpr T kOrthogonalizeEigenvalueFloor = T{1e-10};

/// @brief The size, in elements of T, of the syevd workspace orthogonalize
/// needs for an n x n operand.
///
/// Query it once per n and keep the workspace. The query fixes the same
/// eigenvector mode and triangle orthogonalize passes to syevd, so the two
/// cannot disagree. Runs on the device res.dev_idx() names.
///
/// @param res  Supplies the dense-solver handle.
/// @param n    Matrix order. n <= 0 needs no workspace and returns 0.
/// @return The element count, or a GpuRuntime error if the query failed.
template <real_fp T>
[[nodiscard]] Result<int32_t> orthogonalize_workspace_size(DeviceResources &res, int32_t n);

/// @brief Build the symmetric (Loewdin) orthogonalizer X = S^{-1/2} on the GPU.
///
/// Diagonalizes @p s with syevd, maps the eigenvalues to Lambda^{-1/2} with a
/// kernel, and assembles X = U Lambda^{-1/2} U^T with gmmwich (a dgmm for the
/// column scaling, then a gemm). On success @p x holds the n x n orthogonalizer, which satisfies
/// X^T S X == I to rounding for a well-conditioned S. @p s is not modified.
///
/// Every matrix is a column-major device pointer with packed leading dimension
/// n (element (i, j) at i + j*n). Only the lower triangle of @p s is read.
///
/// Allocates nothing; every intermediate is caller-owned scratch, so a caller
/// that runs this repeatedly allocates once:
///   - @p u_scratch, @p m_scratch: n*n elements each, for U and
///     M = U diag(Lambda^{-1/2});
///   - @p eigenvalues: n elements, overwritten with Lambda and then
///     Lambda^{-1/2};
///   - @p syevd_work: @p syevd_lwork elements, as
///     orthogonalize_workspace_size<T> reports for this n;
///   - @p info: 1 element, syevd's convergence flag.
///
/// Everything is queued on res.stream(), and the call synchronizes that stream
/// once, to read syevd's convergence flag and the extreme eigenvalues. The
/// device res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @param res               Supplies the stream and the BLAS and solver handles.
/// @param n                 Matrix order. n <= 0 is an empty X and succeeds.
/// @param s                 The symmetric operand (n x n).
/// @param x                 Out: the orthogonalizer (n x n).
/// @param u_scratch         Scratch for U (n x n).
/// @param m_scratch         Scratch for M (n x n).
/// @param eigenvalues       Scratch for the eigenvalues (n).
/// @param syevd_work        syevd's device workspace.
/// @param syevd_lwork       Size of @p syevd_work, in elements.
/// @param info              syevd's device convergence flag (1).
/// @param eigenvalue_floor  Eigenvalues at or below this drop their mode
///                          (contribute 0). Should be non-negative.
/// @return Success; a Numerical error if syevd did not converge on @p s; or
///         the GpuRuntime error of the first runtime, solver, BLAS or launch
///         call that failed. A mode under the floor is dropped, not an error.
template <real_fp T>
[[nodiscard]] Status orthogonalize(DeviceResources &res, int32_t n, const T *s, T *x, T *u_scratch,
                                   T *m_scratch, T *eigenvalues, T *syevd_work, int32_t syevd_lwork,
                                   int32_t *info,
                                   T eigenvalue_floor = kOrthogonalizeEigenvalueFloor<T>);

} // namespace free_breakfast
