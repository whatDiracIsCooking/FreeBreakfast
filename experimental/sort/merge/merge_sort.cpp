/// @file
/// @brief cub_merge_sort: sizes and carves the vendor's scratch and hands the
/// sort to the device.

module;

// The launchers, defined in cub_merge_sort.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "cub_merge_sort_bridge.h"

module free_breakfast.experimental.sort.merge;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

Status cub_merge_sort(DeviceResources &res, const int32_t n, double *keys, uint32_t *values) {
  if (n <= 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto un = static_cast<uint32_t>(n);
  size_t temp_bytes = 0;
  FBF_TRY(gpu_status(device::cub_merge_sort_temp_bytes(un, values != nullptr, &temp_bytes)));
  std::byte *const temp = res.device_workspace(temp_bytes).data();
  return gpu_status(
      device::cub_merge_sort_vendor(res.stream(), un, keys, values, temp, temp_bytes));
}

} // namespace free_breakfast
