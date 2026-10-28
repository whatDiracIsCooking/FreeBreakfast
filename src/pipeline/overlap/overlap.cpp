/// @file
/// @brief OverlapStep::run: S built and carried onto the spherical layout,
/// then written as .npy if the step names a file.

module free_breakfast.pipeline.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.overlap;
import free_breakfast.io.npy;

#include "diagnostics.h"

namespace free_breakfast {

// S to @p path as .npy: downloaded on res.stream(), synchronized once, and
// written in its own column-major layout.
[[nodiscard]] Status write_overlap_npy(DeviceResources &res, const std::filesystem::path &path,
                                       const SphericalMatrixAO &s) {
  NpyArray host{.shape = {s.n(), s.n()}, .data = std::vector<double>(s.num_elements())};
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(host.data.data(), s.data(), host.data.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return write_npy(path, host);
}

[[nodiscard]] Result<OverlapOutput>
OverlapStep::run(View<BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput> in,
                 DeviceResources &res) const {
  MatrixAOPair s(res, in.get<BasisSetOutput>().basis, in.get<SphericalTransformOutput>().u);
  FBF_TRY(overlap(res, in.get<PrimitivePairClassesOutput>().classes, s.cartesian()));
  FBF_TRY(s.cartesian_to_spherical(res));
  if (!npy.empty()) {
    FBF_TRY(write_overlap_npy(res, npy, s.spherical()));
    FBF_LOG_INFO("overlap: wrote {}", npy.string());
  }
  return OverlapOutput{std::move(s.spherical())};
}

} // namespace free_breakfast
