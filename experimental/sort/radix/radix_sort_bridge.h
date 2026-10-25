/// @file
/// @brief The radix-sort launcher and its scratch sizing, declared once for the
/// module that calls it and the device TU that defines it.
///
/// Included by radix_sort.cpp in its GLOBAL MODULE FRAGMENT, and by
/// radix_sort.cu directly: a declaration in a module's purview would get module
/// linkage and never bind to the definition in the .cu (see
/// src/recipes/orthogonalize/inverse_sqrt_bridge.h). Everything here is public
/// to both, so it is only what the host side needs to size the scratch.
/// wwrError_t comes from a different wwr header in each pass, as
/// src/integrals/one_electron/overlap/overlap_bridge.h explains.

#pragma once

// Standard (via block_params.h and integral_types.h) before runtime.h: under
// HIP a libc++ header must not follow the vendor runtime header (see
// primitives.h).
#include "block_params.h"
#include "integral_types.h"
#include "primitives.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief Bits in a key: a double's IEEE-754 representation
inline constexpr uint32_t kRadixKeyBits = 64;

/// @brief Key bits sorted per pass; kRadixKeyBits / kRadixBits passes sort a key
inline constexpr uint32_t kRadixBits = 4;

/// @brief Buckets per pass: one per value of a kRadixBits digit
inline constexpr uint32_t kRadixBuckets = 1U << kRadixBits;

/// @brief Keys per block in the count and scatter kernels: one per thread
inline constexpr uint32_t kRadixSortTile = kFourWarps;

static_assert(kRadixKeyBits % kRadixBits == 0 && (kRadixKeyBits / kRadixBits) % 2 == 0,
              "an even number of passes leaves the sorted keys in the caller's buffer");

/// @brief uint32_t slots of digit counts radix_sort_passes needs for @p n keys:
/// one per (bucket, tile)
inline constexpr size_t radix_sort_count_slots(const size_t n) {
  return kRadixBuckets * math::idivup(n, size_t{kRadixSortTile});
}

/// @brief Sort the first @p n of @p keys ascending in IEEE-754 total order,
/// stably, on @p stream, and apply the same permutation to @p values unless it
/// is null
///
/// Total order is -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN. The
/// keys are moved as raw bits and never computed with, so -0.0 and every NaN
/// payload come out bit-for-bit as they went in.
///
/// LSD radix sort, kRadixBits per pass. Each pass moves the data between the
/// caller's arrays and the @p keys_alt / @p values_alt scratch; the pass count
/// is even, so the result ends in @p keys and @p values. Every pointer is a
/// device pointer, and none of the arrays may overlap.
///
/// @param keys_alt    Scratch, @p n elements
/// @param values      Null for a keys-only sort
/// @param values_alt  Scratch, @p n elements; ignored when @p values is null
/// @param counts      Scratch, radix_sort_count_slots(n) elements
/// @return The status of the first failed launch, or success. Launch errors
///         only: a fault inside a kernel surfaces at the next synchronization.
[[nodiscard]] wwr::wwrError_t radix_sort_passes(wwr::wwrStream_t stream, uint32_t n, double *keys,
                                                double *keys_alt, uint32_t *values,
                                                uint32_t *values_alt, uint32_t *counts);

/// @brief The bytes of temporary storage cub_radix_sort_vendor needs for @p n
/// keys, into @p bytes: the vendor's size query, which launches nothing
///
/// @return The query's status
[[nodiscard]] wwr::wwrError_t cub_radix_sort_temp_bytes(uint32_t n, bool with_values,
                                                        size_t *bytes);

/// @brief radix_sort_passes' contract over the vendor's DeviceRadixSort (CUB on
/// CUDA, hipCUB on HIP), the reference the experimental sort is measured
/// against
///
/// Same arrays, same in-place result: the vendor's DoubleBuffer may leave the
/// sorted data in the scratch half, and it is then copied back on @p stream.
/// The vendor's order is not quite total order; see the README.
///
/// @param temp        Scratch, @p temp_bytes from cub_radix_sort_temp_bytes
/// @return The status of the first failed call, or success
[[nodiscard]] wwr::wwrError_t cub_radix_sort_vendor(wwr::wwrStream_t stream, uint32_t n,
                                                    double *keys, double *keys_alt,
                                                    uint32_t *values, uint32_t *values_alt,
                                                    void *temp, size_t temp_bytes);

} // namespace free_breakfast::device
