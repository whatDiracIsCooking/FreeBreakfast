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
/// who owns which intermediate.

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
/// Every intermediate -- U, M = U diag(Lambda^{-1/2}), the eigenvalues and
/// syevd's workspace -- is carved from res.device_workspace(), which grows to
/// n*n + n + max(syevd's size, n*n) elements of T (each region rounded up) if
/// it is smaller: the one allocation this can make. res.info() takes syevd's flag. Neither @p s nor @p x may
/// point into res.device_workspace().
///
/// Everything is queued on res.stream(), and the call synchronizes that stream
/// once, to read syevd's convergence flag and the extreme eigenvalues. The
/// device res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @param res               Supplies the stream, the BLAS and solver handles,
///                          the scratch workspace and syevd's convergence flag.
/// @param n                 Matrix order. n <= 0 is an empty X and succeeds.
/// @param s                 The symmetric operand (n x n).
/// @param x                 Out: the orthogonalizer (n x n).
/// @param eigenvalue_floor  Eigenvalues at or below this drop their mode
///                          (contribute 0). Should be non-negative.
/// @return Success; a Numerical error if syevd did not converge on @p s; or
///         the GpuRuntime error of the first runtime, solver, BLAS or launch
///         call that failed. A mode under the floor is dropped, not an error.
template <real_fp T>
[[nodiscard]] Status orthogonalize(DeviceResources &res, int32_t n, const T *s, T *x,
                                   T eigenvalue_floor = kOrthogonalizeEigenvalueFloor<T>);

} // namespace free_breakfast
