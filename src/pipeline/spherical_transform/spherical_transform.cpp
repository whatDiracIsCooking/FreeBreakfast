/// @file
/// @brief SphericalTransformStep::run: U uploaded, then written as .npy if the
/// step names a file.

module free_breakfast.pipeline.spherical_transform;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.npy;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<SphericalTransformOutput>
SphericalTransformStep::run(View<BasisSetOutput> in, DeviceResources &res) const {
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  SphericalTransformOutput out{FBF_TRY(upload_spherical_transform_matrix(res, basis))};
  if (!npy.empty()) {
    // The host build, not a download: it needs no device sync, and it writes
    // the identity where the upload gave nullptr.
    FBF_TRY(write_npy(npy, NpyArray{.shape = {basis.nao_cart(), basis.nao_sphe()},
                                    .data = spherical_transform_matrix(basis)}));
    FBF_LOG_INFO("spherical_transform: wrote {}", npy.string());
  }
  return out;
}

} // namespace free_breakfast
