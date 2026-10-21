/// @file
/// @brief Stable LSD radix sort of double keys on the GPU, in IEEE-754 total
/// order, optionally carrying uint32_t values, over core/kernel's block_reduce.
///
/// Experimental: built only under FREE_BREAKFAST_BUILD_EXPERIMENTAL. The
/// README has the algorithm and its known costs.

export module free_breakfast.experimental.sort.radix;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief Sort the first @p n of @p keys ascending on the GPU, stably, and
/// permute @p values with them unless it is null
///
/// The order is IEEE-754 total order, which is std::strong_order on double:
/// -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN. Every key, NaN
/// included, has a place, and -0.0 sorts before +0.0 rather than equal to it.
/// The keys are moved as raw bits, so each comes out bit-for-bit as it went in.
///
/// Equal keys (equal bits) keep their input order, so passing the indices
/// 0 .. n-1 as @p values yields the stable sorting permutation. Both arrays
/// are device pointers, sorted in place; neither may point into
/// res.device_workspace().
///
/// The scratch -- a second copy of the keys (and of the values, when given)
/// plus one digit count per (bucket, tile) -- is carved from
/// res.device_workspace(), which grows to fit if it is smaller: the one
/// allocation this can make.
///
/// Everything is queued on res.stream(), and nothing synchronizes it. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored.
///
/// @param n       Keys to sort. n <= 0 sorts nothing and succeeds.
/// @param keys    In and out: the keys (n elements).
/// @param values  In and out: the payload (n elements), or null.
/// @return Success, or the GpuRuntime error of the first launch that failed.
[[nodiscard]] Status radix_sort(DeviceResources &res, int32_t n, double *keys,
                                uint32_t *values = nullptr);

/// @brief radix_sort's contract over the vendor's DeviceRadixSort (CUB on CUDA,
/// hipCUB on HIP): the reference radix_sort is benchmarked and checked against
///
/// Same arguments, same in-place result, same use of res; its scratch is the
/// vendor's temporary storage in place of the digit counts. One difference in
/// order: the vendor treats -0.0 and +0.0 as equal keys (kept in input order)
/// where radix_sort puts -0.0 first. Everything else, NaNs included, sorts
/// identically.
///
/// @return Success, or the GpuRuntime error of the first vendor call that failed.
[[nodiscard]] Status cub_radix_sort(DeviceResources &res, int32_t n, double *keys,
                                    uint32_t *values = nullptr);

} // namespace free_breakfast
