/// @file
/// @brief The symmetric eigendecomposition A = U diag(w) U^T on the GPU, over
/// wwr's dense-solver syevd, its scratch and flag taken from DeviceResources.
///
/// The README has who owns which buffer, and why the legacy syevd.

export module free_breakfast.recipes.diagonalize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Diagonalize the real-symmetric @p a in place on the GPU:
/// A = U diag(w) U^T.
///
/// On success @p a holds the eigenvectors U (column j belongs to w[j]) and
/// @p w the eigenvalues in ascending order. Only the lower triangle of @p a is
/// read. @p a is a column-major device pointer with packed leading dimension n
/// (element (i, j) at i + j*n), @p w a device pointer to n elements.
///
/// syevd's work is res.device_workspace(), grown to syevd's size for n if it is
/// smaller: the one allocation this can make. res.info() takes syevd's flag.
/// Neither @p a nor @p w may point into res.device_workspace().
///
/// Everything is queued on res.stream(), and the call synchronizes that stream
/// once, to read the flag. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @param res  Supplies the stream, the solver handle, the scratch workspace
///             and syevd's convergence flag.
/// @param n    Matrix order. n <= 0 is empty and succeeds.
/// @param a    In: the symmetric operand (n x n). Out: its eigenvectors.
/// @param w    Out: the n ascending eigenvalues.
/// @return Success; a Numerical error if syevd did not converge; or a
///         GpuRuntime error: the first runtime or solver call that failed, or
///         an argument syevd rejected.
template <real_fp T>
[[nodiscard]] Status diagonalize(DeviceResources &res, int32_t n, T *a, T *w);

} // namespace free_breakfast
