/// @file
/// @brief merge_sort and cub_merge_sort: each carves its scratch and hands the
/// sort to the device.

module;

// The launchers, defined in merge_sort.cu (src/integrals/coulomb/sort_bounds)
// and cub_merge_sort.cu: declared in the GMF so they keep external linkage (see
// the headers).
#include "cub_merge_sort_bridge.h"
#include "merge_sort_bridge.h"

module free_breakfast.experimental.sort.merge;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

Status merge_sort(DeviceResources &res, const int32_t n, double *keys, uint32_t *values) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  // Carved in bytes, the element types differing: [ keys_alt | values_alt,
  // only with values ], the second on a kWorkspaceAlignment boundary.
  const size_t count = static_cast<size_t>(n);
  const size_t keys_bytes =
      math::align_up(count * sizeof(double), DeviceResources::kWorkspaceAlignment);
  const size_t values_bytes = values != nullptr ? count * sizeof(uint32_t) : 0;
  std::byte *const base = res.device_workspace(keys_bytes + values_bytes).data();
  return gpu_status(static_cast<wwr::wwrError_t>(device::merge_sort_passes(
      res.stream(), static_cast<uint32_t>(n), keys, reinterpret_cast<double *>(base), values,
      reinterpret_cast<uint32_t *>(base + keys_bytes))));
}

Status cub_merge_sort(DeviceResources &res, const int32_t n, double *keys, uint32_t *values) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto un = static_cast<uint32_t>(n);
  size_t temp_bytes = 0;
  FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
      device::cub_merge_sort_temp_bytes(un, values != nullptr, &temp_bytes))));
  std::byte *const temp = res.device_workspace(temp_bytes).data();
  return gpu_status(static_cast<wwr::wwrError_t>(
      device::cub_merge_sort_vendor(res.stream(), un, keys, values, temp, temp_bytes)));
}

} // namespace free_breakfast
