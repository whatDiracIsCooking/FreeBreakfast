// The vendor reference for free_breakfast.experimental.sort.merge: the same
// sort through DeviceMergeSort -- CUB on CUDA, hipCUB (rocPRIM) on HIP,
// reached as wwr::wwrcub so this file has no backend #if. It is what
// merge_sort.cu is benchmarked and checked against, not a second
// implementation to maintain.
//
// Unlike DeviceRadixSort, the vendor merge sort takes a comparator, so it is
// given total order itself: the keys go in as their uint64_t bits and are
// compared as ordered_bits. Its stable variants then promise exactly
// merge_sort's result, -0.0 before +0.0 included.

// Standard-only headers first, then cub.h, which includes <mdspan> ahead of the
// vendor headers under HIP; then the bridge, whose runtime.h comes after them.
// Under HIP no libc++ header may follow the vendor runtime header (primitives.h
// has why).
#include "integral_types.h"

#include <cub.h>

#include "merge_sort_bridge.h"

namespace free_breakfast::device {

namespace {

namespace vendor = wwr::wwrcub;

constexpr uint64_t kSignBit = uint64_t{1} << 63;

// IEEE-754 total order on a double's bits: merge_sort.cu's ordered_bits,
// compared unsigned.
struct TotalLess {
  __host__ __device__ static uint64_t ordered_bits(const uint64_t bits) {
    return bits ^ ((bits & kSignBit) != 0 ? ~uint64_t{0} : kSignBit);
  }
  __host__ __device__ bool operator()(const uint64_t a, const uint64_t b) const {
    return ordered_bits(a) < ordered_bits(b);
  }
};

} // namespace

int32_t cub_merge_sort_temp_bytes(const uint32_t n, const bool with_values, size_t *bytes) {
  // A null temp pointer makes the call a size query; the arrays are not read.
  const auto num_items = static_cast<int32_t>(n);
  uint64_t *const keys = nullptr;
  if (with_values) {
    uint32_t *const values = nullptr;
    return static_cast<int32_t>(vendor::DeviceMergeSort::StableSortPairs(
        nullptr, *bytes, keys, values, num_items, TotalLess{}));
  }
  return static_cast<int32_t>(
      vendor::DeviceMergeSort::StableSortKeys(nullptr, *bytes, keys, num_items, TotalLess{}));
}

int32_t cub_merge_sort_vendor(const wwr::wwrStream_t stream, const uint32_t n, double *keys,
                              uint32_t *values, void *temp, size_t temp_bytes) {
  // The vendor only moves and compares the keys, as bits.
  auto *const bits = reinterpret_cast<uint64_t *>(keys);
  const auto num_items = static_cast<int32_t>(n);
  if (values != nullptr) {
    return static_cast<int32_t>(vendor::DeviceMergeSort::StableSortPairs(
        temp, temp_bytes, bits, values, num_items, TotalLess{}, stream));
  }
  return static_cast<int32_t>(vendor::DeviceMergeSort::StableSortKeys(
      temp, temp_bytes, bits, num_items, TotalLess{}, stream));
}

} // namespace free_breakfast::device
