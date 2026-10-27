/// @file
/// @brief Stable merge sort of double keys on the GPU, in IEEE-754 total
/// order, optionally carrying uint32_t values, and the column gather that
/// applies its permutation to a matrix.
///
/// See README.md.

export module free_breakfast.recipes.sort;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Sort the first @p n of @p keys ascending on the GPU, stably, and
/// permute @p values with them unless it is null
///
/// IEEE-754 total order (std::strong_order on double): equal keys keep their
/// input order, -0.0 sorts before +0.0, and NaNs sort past the infinities on
/// their sign's side. The keys are moved as raw bits. Both arrays are device
/// pointers sorted in place, and neither may point into
/// res.device_workspace(). Pass 0 .. n-1 as @p values to get the sorting
/// permutation.
///
/// The scratch -- a second copy of the keys, and of the values when given -- is
/// carved from res.device_workspace(), which grows to fit if it is smaller: the
/// one allocation this can make.
///
/// Everything is queued on res.stream(), and nothing synchronizes it. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored.
///
/// @param n       Keys to sort. n <= 0 sorts nothing and succeeds.
/// @param keys    In and out: the keys (n elements).
/// @param values  In and out: the payload (n elements), or null.
/// @return Success, or the GpuRuntime error of the first launch that failed.
[[nodiscard]] Status merge_sort(DeviceResources &res, int32_t n, double *keys,
                                uint32_t *values = nullptr);

/// @brief Column j of @p dst = column perm[j] of @p src, for @p rows x @p cols
/// column-major matrices with leading dimension @p rows
///
/// merge_sort's permutation, applied: with @p perm the values a sort carried
/// from 0 .. cols-1, @p dst holds @p src's columns in the sorted keys' order.
/// Every pointer is a device pointer, @p dst must not overlap @p src, and
/// every perm[j] must be below @p cols.
///
/// Queued on res.stream(), without synchronizing. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
///
/// @return Success, with nothing launched when @p rows or @p cols is 0;
///         InvalidConfig for a negative @p rows or @p cols, or a null pointer
///         with work to do; or the GpuRuntime error of a failed launch.
[[nodiscard]] Status gather_columns(DeviceResources &res, int32_t rows, int32_t cols,
                                    const uint32_t *perm, const double *src, double *dst);

} // namespace free_breakfast
