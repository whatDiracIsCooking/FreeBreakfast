/// @file
/// @brief The vendor reference merge_sort is benchmarked and checked against,
/// and merge_sort itself, re-exported from free_breakfast.recipes.sort.
///
/// Experimental: built only under FREE_BREAKFAST_BUILD_EXPERIMENTAL. The
/// README has the algorithm and its timings against radix_sort.

export module free_breakfast.experimental.sort.merge;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.recipes.sort; // merge_sort

export namespace free_breakfast {

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
