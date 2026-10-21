/// @file
/// @brief Stable merge sort of double keys on the GPU, in IEEE-754 total
/// order, optionally carrying uint32_t values.
///
/// Experimental: built only under FREE_BREAKFAST_BUILD_EXPERIMENTAL. The
/// README has the algorithm and its timings against radix_sort.

export module free_breakfast.experimental.sort.merge;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Sort the first @p n of @p keys ascending on the GPU, stably, and
/// permute @p values with them unless it is null
///
/// radix_sort's contract exactly: IEEE-754 total order (std::strong_order on
/// double), equal keys keep their input order, the keys are moved as raw bits,
/// both arrays are device pointers sorted in place and neither may point into
/// res.device_workspace(). The two sorts give bit-identical results.
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

/// @brief merge_sort's contract over the vendor's stable DeviceMergeSort (CUB
/// on CUDA, hipCUB on HIP): the reference merge_sort is benchmarked and checked
/// against
///
/// Same arguments, same in-place result, same use of res; its scratch is the
/// vendor's temporary storage. The vendor is handed a total-order comparator,
/// so unlike cub_radix_sort it agrees with merge_sort everywhere, zeros
/// included.
///
/// @return Success, or the GpuRuntime error of the first vendor call that failed.
[[nodiscard]] Status cub_merge_sort(DeviceResources &res, int32_t n, double *keys,
                                    uint32_t *values = nullptr);

} // namespace free_breakfast
