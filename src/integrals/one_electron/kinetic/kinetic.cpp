/// @file
/// @brief The T build.

module;

// The kernel launcher, defined in kinetic_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "kinetic_bridge.h"

module free_breakfast.integrals.one_electron.kinetic;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Status kinetic(DeviceResources &res, const std::span<const DevicePrimitivePairClass> classes,
               CartesianMatrixAO &t) {
  FBF_TRY(check_one_electron_build(res, classes, t, "kinetic"));
  if (t.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(t.data(), 0, sizeof(double) * t.num_elements(), stream)));
  for (const DevicePrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
        device::kinetic_class(stream, cls.view(), static_cast<int32_t>(t.n()), t.data()))));
  }
  return {};
}

} // namespace free_breakfast
