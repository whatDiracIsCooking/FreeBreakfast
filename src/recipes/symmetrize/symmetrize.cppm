/// @file
/// @brief Make a square device matrix symmetric in place: by averaging it with
/// its transpose, or by mirroring one triangle onto the other.
///
/// What a geam cannot do: it may not write over an operand it transposes, so
/// A <- (A + A^T) / 2 by geam needs a second matrix and a copy back, and BLAS
/// has no call that fills the triangle a syrk leaves.

export module free_breakfast.recipes.symmetrize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief A <- (A + A^T) / 2 in place; (i, j) and (j, i) come out the same
/// bits.
///
/// @p a is the column-major @p n x @p n block with leading dimension @p lda;
/// rows past @p n are neither read nor written. Each pair is summed once, in
/// either order, and halved, so the result is exactly symmetric whatever the
/// input's asymmetry, and a symmetric input comes back unchanged.
///
/// One launch on res.stream(), without synchronizing; nothing is drawn from
/// res.device_workspace(). The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return Success, with nothing launched when @p n is 0; InvalidConfig for a
///         negative @p n or an @p lda below @p n; or the GpuRuntime error of a
///         failed launch.
[[nodiscard]] Status symmetrize(DeviceResources &res, int32_t n, double *a, int32_t lda);

/// @brief Overwrite the triangle opposite @p fill with the transpose of
/// @p fill's, in place: what fills the other half of a syrk's result.
///
/// With WWRBLAS_FILL_MODE_LOWER, a(i, j) <- a(j, i) for every i < j; with
/// WWRBLAS_FILL_MODE_UPPER, for every i > j. The @p fill triangle and the
/// diagonal are not written, and no value of the opposite triangle reaches the
/// result, so it may hold anything, NaN included. Layout, stream and device as
/// the averaging overload's.
///
/// @return Success, with nothing launched when @p n is 0; InvalidConfig for a
///         negative @p n, an @p lda below @p n or a @p fill that is
///         neither; or the GpuRuntime error of a failed launch.
[[nodiscard]] Status symmetrize(DeviceResources &res, wwr::wwrblasFillMode_t fill, int32_t n,
                                double *a, int32_t lda);

} // namespace free_breakfast
