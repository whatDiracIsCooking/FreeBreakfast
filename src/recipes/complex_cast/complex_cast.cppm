/// @file
/// @brief Split a complex device matrix into real and imaginary planes, and
/// merge two planes back, on the GPU.
///
/// The packing glue a complex routine needs when its operands arrive, or its
/// results must leave, as separate real arrays -- for instance a complex
/// product formed from real gemms. One elementwise kernel each, over wwr's
/// parallel_for.

export module free_breakfast.recipes.complex_cast;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Split the complex matrix @p a into its real and imaginary planes.
///
/// re(i,j) <- Re(a(i,j)), im(i,j) <- Im(a(i,j)) over the column-major
/// @p rows x @p cols block. @p a has leading dimension @p lda, both planes
/// @p ldp (each >= @p rows); padding rows are neither read nor written. All
/// pointers are device memory, and @p re and @p im must not overlap @p a.
///
/// Queued on res.stream(), without synchronizing. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
/// Instantiated for wwrFloatComplex and wwrDoubleComplex.
///
/// @return Success, with nothing launched when @p rows or @p cols is <= 0; or
///         the GpuRuntime error of a failed launch.
template <wwr::complex_fp Z>
[[nodiscard]] Status split_planes(DeviceResources &res, int32_t rows, int32_t cols, const Z *a,
                                  int32_t lda, wwr::ComplexToRealType<Z> *re,
                                  wwr::ComplexToRealType<Z> *im, int32_t ldp);

/// @brief Merge the real and imaginary planes @p re and @p im into the complex
/// matrix @p c; split_planes' inverse.
///
/// c(i,j) <- (re(i,j), im(i,j)) over the column-major @p rows x @p cols block.
/// Both planes have leading dimension @p ldp, @p c has @p ldc (each >= @p rows);
/// padding rows are neither read nor written. All pointers are device memory,
/// and @p c must not overlap either plane.
///
/// Queued on res.stream(), without synchronizing, on res.dev_idx() as
/// split_planes is. Instantiated for wwrFloatComplex and wwrDoubleComplex.
///
/// @return Success, with nothing launched when @p rows or @p cols is <= 0; or
///         the GpuRuntime error of a failed launch.
template <wwr::complex_fp Z>
[[nodiscard]] Status
merge_planes(DeviceResources &res, int32_t rows, int32_t cols, const wwr::ComplexToRealType<Z> *re,
             const wwr::ComplexToRealType<Z> *im, int32_t ldp, Z *c, int32_t ldc);

} // namespace free_breakfast
