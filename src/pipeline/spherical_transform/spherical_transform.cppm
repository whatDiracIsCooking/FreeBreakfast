/// @file
/// @brief The `SphericalTransform` step: the basis's Cartesian-to-spherical
///        transform U, uploaded once for every AO-matrix step to share.
///
/// A thin wrapper over recipes.matrix_ao's upload_spherical_transform_matrix.
/// Device-shaped; see README.md for the one device contract it does not keep.

export module free_breakfast.pipeline.spherical_transform;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief U for one @ref BasisSet, on one device: what every @ref MatrixAOPair
/// over that basis converts through.
///
/// @ref u is @c nullptr when no shell is above p, since U is then the identity
/// (@ref upload_spherical_transform_matrix). Never written after the upload,
/// so steps on different streams may share it.
struct SphericalTransformOutput {
  /// @brief U, @c nao_cart x @c nao_sphe column-major; or @c nullptr.
  std::shared_ptr<const Buffers::device<double>> u;
};

/// @brief Uploads the @ref BasisSet's U to @p res's device.
struct SphericalTransformStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "spherical_transform";

  /// @brief Build U on the host and upload it. A failed upload is
  /// ErrorKind::GpuRuntime.
  [[nodiscard]] Result<SphericalTransformOutput> run(View<BasisSetOutput> in,
                                                     DeviceResources &res) const {
    return SphericalTransformOutput{
        FBF_TRY(upload_spherical_transform_matrix(res, in.get<BasisSetOutput>().basis))};
  }
};

} // namespace free_breakfast
