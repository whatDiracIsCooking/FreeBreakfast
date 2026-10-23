/// @file
/// @brief Every root of a batch of polynomials, by the Aberth-Ehrlich
/// iteration, on the GPU.
///
/// One warp per polynomial, all of one degree, and one lane per root.
/// Coefficients are real or complex; roots are always complex, in the same
/// precision.

export module free_breakfast.recipes.polynomial_roots;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Every root of @p batch degree-@p n polynomials.
///
/// Polynomial b is coeffs[b*(n+1) + k] = a_k, ascending, with a_n != 0. Its
/// roots land in roots[b*n, b*n + n), in no particular order, and in iters[b]
/// the iterations it took, or -1 if @p max_iter passed first (a NaN
/// coefficient never converges). A root settles when its relative correction is
/// <= @p tol or |p(z)| is at Horner's rounding floor. All pointers are device
/// memory.
///
/// Queued on res.stream(), without synchronizing. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
/// Instantiated for float, double, wwrFloatComplex and wwrDoubleComplex
/// coefficients; CT is the complex type of the same precision.
///
/// @param n Degree, at most kWarpSize (free_breakfast.core.kernel): one lane
///          per root, so 32 on NVIDIA and RDNA, 64 on CDNA
/// @return Success, with nothing launched when @p n or @p batch is 0;
///         InvalidConfig for a negative @p n, @p batch or @p max_iter, a
///         negative or NaN @p tol, an @p n over kWarpSize, or a null pointer
///         with work to do; or the GpuRuntime error of a failed launch.
template <wwr::usual_fp T, wwr::complex_fp CT>
  requires std::same_as<wwr::ComplexToRealType<CT>, wwr::ComplexToRealType<T>>
[[nodiscard]] Status polynomial_roots(DeviceResources &res, int32_t n, int32_t batch,
                                      const T *coeffs, CT *roots, int32_t *iters,
                                      wwr::ComplexToRealType<T> tol, int32_t max_iter);

} // namespace free_breakfast
