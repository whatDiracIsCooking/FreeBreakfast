/// @file
/// @brief The exponential of a skew-Hermitian (or real skew-symmetric) matrix
/// on the GPU, by the Hermitian eigendecomposition of iH.
///
/// exp(tH) for H^H = -H is unitary (orthogonal for a real H), and is the
/// geodesic of U(n) (O(n)) through the identity in direction H. A Riemannian
/// solver on that group evaluates it at many t for one H, so the work is split
/// in two: skew_eig diagonalizes once, skew_expm evaluates at each t. The
/// README has the derivation and why this is not a general expm.

export module free_breakfast.recipes.expm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Diagonalize the skew-Hermitian @p h through the Hermitian iH:
/// iH = U diag(w) U^H, so H = -i U diag(w) U^H.
///
/// U is complex even for a real @p h: a real skew-symmetric matrix has
/// imaginary eigenvalues and complex eigenvectors, so @p u's element type is
/// T's complex counterpart (T itself when T is complex).
///
/// On success @p u holds the eigenvectors U (column j belongs to w[j]) and
/// @p w the n real eigenvalues of iH in ascending order. A real skew-symmetric
/// @p h has them in pairs +-w, with one 0 for odd n. Only the lower triangle
/// of @p h is read; that h is skew-Hermitian is the caller's promise, not
/// checked. All three are column-major device pointers, @p h and @p u with
/// packed leading dimension n, and none may overlap another.
///
/// heevd's work is res.device_workspace(), grown to heevd's size for n if it
/// is smaller: the one allocation this can make. res.info() takes heevd's
/// flag. None of @p h, @p u or @p w may point into res.device_workspace().
///
/// Everything is queued on res.stream() with res.blas() held in host pointer
/// mode, and the call synchronizes that stream once, to read the flag. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored. Instantiated for float, double, wwrFloatComplex and
/// wwrDoubleComplex.
///
/// @param res  Supplies the stream, the BLAS and solver handles, the scratch
///             workspace and heevd's convergence flag.
/// @param n    Matrix order. n <= 0 is empty and succeeds.
/// @param h    The skew-Hermitian operand (n x n); not modified.
/// @param u    Out: the eigenvectors of iH (n x n).
/// @param w    Out: the n ascending eigenvalues of iH.
/// @return Success; a Numerical error if heevd did not converge; or a
///         GpuRuntime error: the first runtime, BLAS or solver call that
///         failed, or an argument heevd rejected.
template <wwr::usual_fp T>
[[nodiscard]] Status skew_eig(DeviceResources &res, int32_t n, const T *h,
                              wwr::RealToComplexType<wwr::ComplexToRealType<T>> *u,
                              wwr::ComplexToRealType<T> *w);

/// @brief exp(tH) = U diag(e^{-i t w}) U^H, from skew_eig's @p u and @p w.
///
/// Exact up to the eigendecomposition's accuracy at any t: the phases are
/// formed directly, with no scaling and squaring and no Pade degree, so a large
/// t costs nothing extra. For a real T the product is real in exact
/// arithmetic, and @p e receives its real part.
///
/// The intermediates (U diag(e^{-i t w}), and for a real T the complex product)
/// take res.device_workspace(), grown to n * n -- or 2 n * n for a real T --
/// complex elements if it is smaller. None of @p u, @p w or @p e may
/// point into it. @p e is a column-major device pointer with packed leading
/// dimension n.
///
/// Queued on res.stream() with res.blas() held in host pointer mode, without
/// synchronizing. The device res.dev_idx() names is made current for the call
/// and the previous one restored. Instantiated as skew_eig is.
///
/// @param res  Supplies the stream, the BLAS handle and the scratch workspace.
/// @param n    Matrix order. n <= 0 is empty and succeeds.
/// @param u    skew_eig's eigenvectors (n x n).
/// @param w    skew_eig's n eigenvalues.
/// @param t    The step along the geodesic.
/// @param e    Out: exp(tH) (n x n), unitary (orthogonal for a real T).
/// @return Success; or a GpuRuntime error: the first launch or BLAS call that
///         failed.
template <wwr::usual_fp T>
[[nodiscard]] Status skew_expm(DeviceResources &res, int32_t n,
                               const wwr::RealToComplexType<wwr::ComplexToRealType<T>> *u,
                               const wwr::ComplexToRealType<T> *w, wwr::ComplexToRealType<T> t,
                               T *e);

} // namespace free_breakfast
