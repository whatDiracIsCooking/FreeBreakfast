// The vendor reference for free_breakfast.experimental.sort.radix: the same
// sort through DeviceRadixSort -- CUB on CUDA, hipCUB on HIP, reached as
// wwr::wwrcub so this file has no backend #if. It is what radix_sort.cu is
// benchmarked and checked against, not a second implementation to maintain.

// Standard-only headers first, then cub.h, which includes <mdspan> ahead of the
// vendor headers under HIP; then the bridge, whose runtime.h comes after them.
// Under HIP no libc++ header may follow the vendor runtime header (primitives.h
// has why).
#include "block_params.h"
#include "integral_types.h"

#include <cub.h>

#include "radix_sort_bridge.h"

namespace free_breakfast::device {

namespace {

namespace vendor = wwr::wwrcub;

// Every bit of the key: the vendor default, spelled out because the stream
// argument comes after it.
constexpr int32_t kBeginBit = 0;
constexpr int32_t kEndBit = static_cast<int32_t>(kRadixKeyBits);

// The vendor may finish in either half of a DoubleBuffer; copy back to the
// caller's array when it finished in the scratch one. Non-const: CUB's
// Current() is not a const member.
template <typename T>
wwr::wwrError_t copy_back(vendor::DoubleBuffer<T> &buffer, T *caller, const uint32_t n,
                          const wwr::wwrStream_t stream) {
  if (buffer.Current() == caller) {
    return wwr::wwrSuccess;
  }
  return wwr::wwrMemcpyAsync(caller, buffer.Current(), static_cast<size_t>(n) * sizeof(T),
                             wwr::wwrMemcpyDeviceToDevice, stream);
}

} // namespace

int32_t cub_radix_sort_temp_bytes(const uint32_t n, const bool with_values, size_t *bytes) {
  // A null temp pointer makes the call a size query; the buffers are not read.
  vendor::DoubleBuffer<double> keys{nullptr, nullptr};
  const auto num_items = static_cast<int32_t>(n);
  if (with_values) {
    vendor::DoubleBuffer<uint32_t> values{nullptr, nullptr};
    return static_cast<int32_t>(
        vendor::DeviceRadixSort::SortPairs(nullptr, *bytes, keys, values, num_items));
  }
  return static_cast<int32_t>(vendor::DeviceRadixSort::SortKeys(nullptr, *bytes, keys, num_items));
}

int32_t cub_radix_sort_vendor(const wwr::wwrStream_t stream, const uint32_t n, double *keys,
                              double *keys_alt, uint32_t *values, uint32_t *values_alt, void *temp,
                              size_t temp_bytes) {
  vendor::DoubleBuffer<double> key_buffer{keys, keys_alt};
  const auto num_items = static_cast<int32_t>(n);
  wwr::wwrError_t err = wwr::wwrSuccess;
  if (values != nullptr) {
    vendor::DoubleBuffer<uint32_t> value_buffer{values, values_alt};
    err = vendor::DeviceRadixSort::SortPairs(temp, temp_bytes, key_buffer, value_buffer, num_items,
                                             kBeginBit, kEndBit, stream);
    if (err == wwr::wwrSuccess) {
      err = copy_back(value_buffer, values, n, stream);
    }
  } else {
    err = vendor::DeviceRadixSort::SortKeys(temp, temp_bytes, key_buffer, num_items, kBeginBit,
                                            kEndBit, stream);
  }
  if (err == wwr::wwrSuccess) {
    err = copy_back(key_buffer, keys, n, stream);
  }
  return static_cast<int32_t>(err);
}

} // namespace free_breakfast::device
