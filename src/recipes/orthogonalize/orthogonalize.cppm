/// @file
/// @brief Symmetric (Loewdin) orthogonalizer X = S^{-1/2}, and the rank-dropping
/// orthonormalization of a block of vectors, on the GPU, over wwr's
/// dense-solver and BLAS dispatch.
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

export module free_breakfast.recipes.orthogonalize;

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
/// Diagonalizes @p s (diagonalize, a syevd), maps the eigenvalues to Lambda^{-1/2} with a
/// kernel, and assembles X = U Lambda^{-1/2} U^T with gmmwich (a dgmm for the
/// column scaling, then a gemm). On success @p x holds the n x n orthogonalizer, which satisfies
/// X^T S X == I to rounding for a well-conditioned S. @p s is not modified.
///
/// Every matrix is a column-major device pointer with packed leading dimension
/// n (element (i, j) at i + j*n). Only the lower triangle of @p s is read.
///
/// U and the eigenvalues are pool buffers on res.device(), since they live
/// across diagonalize's use of res.device_workspace(). syevd's work and then
/// M = U diag(Lambda^{-1/2}) are the workspace's, which grows if it is smaller
/// than either; res.info() takes syevd's flag. Neither @p s nor @p x may point
/// into res.device_workspace().
///
/// Everything is queued on res.stream(). The call waits on it twice: in
/// diagonalize, for syevd's convergence flag, and then for the extreme
/// eigenvalues, by which point the stream is idle. The
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
/// @return Success; diagonalize's Numerical error if syevd did not converge on
///         @p s; or
///         the GpuRuntime error of the first runtime, solver, BLAS or launch
///         call that failed. A mode under the floor is dropped, not an error.
template <real_fp T>
[[nodiscard]] Status orthogonalize(DeviceResources &res, int32_t n, const T *s, T *x,
                                   T eigenvalue_floor = kOrthogonalizeEigenvalueFloor<T>);

/// @brief Orthonormalize the columns of the n x k device block @p v in place,
/// dropping those that add no direction: canonical orthogonalization, then a
/// second pass.
///
/// One pass forms G = V^T V (a gemv per column), scales it to a unit diagonal, G' = D G D
/// with D = diag(G)^{-1/2}, diagonalizes G' = U Lambda U^T, and keeps the
/// modes above the cut: V <- V D U_kept Lambda_kept^{-1/2} (dgmms and a gemm
/// into scratch, copied back to the front of @p v). One pass leaves an error
/// of order kappa epsilon, kappa = cond of G' over the kept modes, so a second
/// follows: one Newton-Schulz step Q <- Q (3 I - Q^T Q) / 2 when kappa
/// epsilon <= 1e-2 sqrt(epsilon) (gemvs and a gemm, no sync), else the same canonical
/// pass over the kept columns. The unit diagonal is why a short column next to a long
/// one keeps its direction: G's own eigenvalues would bury it below
/// epsilon ||G||.
///
/// What is dropped: a column of norm at or below @p floor, outright (D's zero);
/// then a mode of the unit-normalized columns whose squared norm lambda is at
/// or below floor^2, or below k epsilon lambda_max, the rounding a Gram matrix
/// cannot see past. For columns of comparable norm the floor is a norm, as
/// Gram-Schmidt's would be.
///
/// The kept columns are combinations of the input's, in no particular order;
/// span(Q) = span(V) up to the dropped modes. Columns past the returned count
/// are left with scratch values.
///
/// G, its eigenvalues and D are pool buffers on res.device(), held across
/// diagonalize; the n x k gemm product is res.device_workspace()'s.
/// Everything is queued on res.stream() with res.blas() held in host pointer
/// mode (the previous mode restored). Each canonical pass synchronizes the
/// stream in diagonalize and reads the eigenvalues back; the Newton-Schulz
/// step does not synchronize. The device res.dev_idx() names
/// is made current for the call. @p v may not point into
/// res.device_workspace().
///
/// @param n      Rows of @p v, at least 1 when @p k > 0.
/// @param k      Columns of @p v. k = 0 keeps nothing and succeeds.
/// @param v      In: the n x k block, leading dimension n. Out: its first
///               (returned count) columns orthonormal.
/// @param floor  The norm at or below which a direction is dropped.
/// @return The number of columns kept; InvalidConfig for n < 1 with k > 0, or
///         k < 0; diagonalize's Numerical error; or the first failing GPU
///         call's.
template <real_fp T>
[[nodiscard]] Result<int32_t> orthonormalize(DeviceResources &res, int32_t n, int32_t k, T *v,
                                             T floor);

} // namespace free_breakfast
