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
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.symmetrize;

#include "diagnostics.h"

namespace free_breakfast {

Status kinetic(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
               CartesianMatrixAO &t) {
  FBF_TRY(check_one_electron_build(res, classes, t, "kinetic"));
  if (t.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(t.data(), 0, sizeof(double) * t.num_elements(), stream)));
  const auto n_ao = static_cast<int32_t>(t.n());
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return device::kinetic_class(cls.device_stream()->stream(), cls.view(), n_ao, t.data());
  })));
  // The kernels write the lower triangle only (row-major, row >= col); the
  // upper is mirrored from it once every class has joined. Row-major lower is
  // column-major upper, the recipe's vocabulary.
  return symmetrize(res, wwr::WWRBLAS_FILL_MODE_UPPER, n_ao, t.data(), n_ao);
}

} // namespace free_breakfast
