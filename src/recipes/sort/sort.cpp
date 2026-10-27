/// @file
/// @brief merge_sort, which carves its scratch and hands the sort to the
/// device, and gather_columns.

module;

// The launchers, defined in merge_sort.cu and gather_columns.cu: declared in
// the GMF so they keep external linkage (see the headers).
#include "gather_columns_bridge.h"
#include "merge_sort_bridge.h"

module free_breakfast.recipes.sort;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

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
  return gpu_status(device::merge_sort_passes(res.stream(), static_cast<uint32_t>(n), keys,
                                              reinterpret_cast<double *>(base), values,
                                              reinterpret_cast<uint32_t *>(base + keys_bytes)));
}

Status gather_columns(DeviceResources &res, const int32_t rows, const int32_t cols,
                      const uint32_t *perm, const double *src, double *dst) {
  if (rows < 0 || cols < 0) {
    return err_config("gather_columns: rows ({}) and cols ({}) must be >= 0", rows, cols);
  }
  if (rows == 0 || cols == 0) {
    return {};
  }
  if (perm == nullptr || src == nullptr || dst == nullptr) {
    return err_config("gather_columns: a null device pointer with a {} x {} matrix to gather", rows,
                      cols);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::gather_columns(res.stream(), rows, cols, perm, src, dst));
}

} // namespace free_breakfast
