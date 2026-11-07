/// @file
/// @brief The `SphericalTransform` step: the basis's Cartesian-to-spherical
///        transform U, uploaded once for every AO-matrix step to share.
///
/// A thin wrapper over recipes.matrix_ao's upload_spherical_transform_matrix,
/// with U written as .npy if asked. Device-shaped; see README.md for the one
/// device contract it does not keep.

export module free_breakfast.pipeline.spherical_transform;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;

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

  /// @brief Where U is written as a float64 .npy file, @c nao_cart x
  /// @c nao_sphe in Fortran order, if not empty. Written from the host build
  /// of U, so an explicit identity when the product's @ref
  /// SphericalTransformOutput::u is @c nullptr. Arrives resolved
  /// (pipeline/config sets it to spherical_transform.npy in the run's output
  /// directory when [spherical_transform] write_npy is set).
  std::filesystem::path npy{};

  /// @brief Build U on the host and upload it. A failed upload is
  /// ErrorKind::GpuRuntime; an @ref npy file that cannot be written is
  /// @c write_npy's error.
  [[nodiscard]] Result<SphericalTransformOutput> run(View<BasisSetOutput> in,
                                                     DeviceResources &res) const;
};

} // namespace free_breakfast
