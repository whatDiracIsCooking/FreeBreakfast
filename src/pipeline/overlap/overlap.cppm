/// @file
/// @brief The `Overlap` step: the spherical AO overlap matrix S.
///
/// A thin wrapper over integrals.one_electron's overlap and recipes.matrix_ao's
/// cartesian_to_spherical: S is built Cartesian in a transient MatrixAOPair and
/// only its spherical form is published, and written as .npy if asked.

export module free_breakfast.pipeline.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.pipeline.spherical_transform;
export import free_breakfast.pipeline.primitive_pair_classes;

export namespace free_breakfast {

/// @brief The spherical AO overlap S_mn = <m|n>, @c nao_sphe square.
struct OverlapOutput {
  SphericalMatrixAO s;
};

/// @brief Builds the @ref BasisSet's overlap on @p res's device.
struct OverlapStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "overlap";

  /// @brief Where S is written as a float64 .npy file, @c nao_sphe square in
  /// Fortran order, if not empty. Arrives resolved (pipeline/config sets it to
  /// overlap.npy in the run's output directory when [overlap] write_npy is
  /// set).
  std::filesystem::path npy{};

  /// @brief S, built Cartesian and carried onto the spherical layout. A
  /// failed launch, copy or BLAS call is ErrorKind::GpuRuntime; an @ref npy
  /// file that cannot be written is @c write_npy's error.
  [[nodiscard]] Result<OverlapOutput>
  run(View<BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
