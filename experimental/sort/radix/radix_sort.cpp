/// @file
/// @brief radix_sort and cub_radix_sort: each carves its scratch and hands the
/// sort to the device.

module;

// The launchers, defined in radix_sort.cu and cub_radix_sort.cu: declared in
// the GMF so they keep external linkage (see the header).
#include "radix_sort_bridge.h"

module free_breakfast.experimental.sort.radix;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// Both sorts' scratch: a second half for the keys, one for the values when
// there are any, and a tail of sort-specific bytes (digit counts, or the
// vendor's temporary storage).
struct RadixSortScratch {
  double *keys_alt;
  uint32_t *values_alt;
  std::byte *tail;
};

// Carved from res's workspace in bytes, the element types differing: [ keys_alt
// | values_alt, only with values | tail ], each region on a kWorkspaceAlignment
// boundary, as a fresh allocation would be.
RadixSortScratch carve_radix_sort_scratch(DeviceResources &res, const size_t count,
                                          const bool with_values, const size_t tail_bytes) {
  constexpr size_t kAlign = DeviceResources::kWorkspaceAlignment;
  const size_t keys_bytes = math::align_up(count * sizeof(double), kAlign);
  const size_t values_bytes = with_values ? math::align_up(count * sizeof(uint32_t), kAlign) : 0;
  std::byte *const base = res.device_workspace(keys_bytes + values_bytes + tail_bytes).data();
  return {reinterpret_cast<double *>(base), reinterpret_cast<uint32_t *>(base + keys_bytes),
          base + keys_bytes + values_bytes};
}

Status radix_sort(DeviceResources &res, const int32_t n, double *keys, uint32_t *values) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t count = static_cast<size_t>(n);
  const RadixSortScratch scratch = carve_radix_sort_scratch(
      res, count, values != nullptr, device::radix_sort_count_slots(count) * sizeof(uint32_t));
  return gpu_status(static_cast<wwr::wwrError_t>(device::radix_sort_passes(
      res.stream(), static_cast<uint32_t>(n), keys, scratch.keys_alt, values, scratch.values_alt,
      reinterpret_cast<uint32_t *>(scratch.tail))));
}

Status cub_radix_sort(DeviceResources &res, const int32_t n, double *keys, uint32_t *values) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto un = static_cast<uint32_t>(n);
  size_t temp_bytes = 0;
  FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
      device::cub_radix_sort_temp_bytes(un, values != nullptr, &temp_bytes))));
  const RadixSortScratch scratch =
      carve_radix_sort_scratch(res, static_cast<size_t>(n), values != nullptr, temp_bytes);
  return gpu_status(static_cast<wwr::wwrError_t>(
      device::cub_radix_sort_vendor(res.stream(), un, keys, scratch.keys_alt, values,
                                    scratch.values_alt, scratch.tail, temp_bytes)));
}

} // namespace free_breakfast
