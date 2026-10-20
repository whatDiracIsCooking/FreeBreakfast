/// @file
/// @brief The S build.

module;

// The kernel launcher, defined in overlap_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "overlap_bridge.h"

module free_breakfast.integrals.one_electron.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Status overlap(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
               CartesianMatrixAO &s) {
  FBF_TRY(check_one_electron_build(res, classes, s, "overlap"));
  if (s.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(s.data(), 0, sizeof(double) * s.num_elements(), stream)));
  const auto n_ao = static_cast<int32_t>(s.n());
  return gpu_status(launch_each(
      classes, stream, [&](const wwr::wwrStream_t cls_stream, const PrimitivePairClass &cls) {
        return device::overlap_class(cls_stream, cls.view(), n_ao, s.data());
      }));
}

} // namespace free_breakfast
