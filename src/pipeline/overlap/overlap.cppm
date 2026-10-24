/// @file
/// @brief The `Overlap` step: the spherical AO overlap matrix S.
///
/// A thin wrapper over integrals.one_electron's overlap and recipes.matrix_ao's
/// cartesian_to_spherical: S is built Cartesian in a transient MatrixAOPair and
/// only its spherical form is published.

export module free_breakfast.pipeline.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.integrals.one_electron.overlap;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.pipeline.spherical_transform;
export import free_breakfast.pipeline.primitive_pair_classes;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The spherical AO overlap S_mn = <m|n>, @c nao_sphe square.
struct OverlapOutput {
  SphericalMatrixAO s;
};

/// @brief Builds the @ref BasisSet's overlap on @p res's device.
struct OverlapStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "overlap";

  /// @brief S, built Cartesian and carried onto the spherical layout. A
  /// failed launch or BLAS call is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<OverlapOutput>
  run(View<BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const {
    MatrixAOPair s(res, in.get<BasisSetOutput>().basis, in.get<SphericalTransformOutput>().u);
    FBF_TRY(overlap(res, in.get<PrimitivePairClassesOutput>().classes, s.cartesian()));
    FBF_TRY(s.cartesian_to_spherical(res));
    return OverlapOutput{std::move(s.spherical())};
  }
};

} // namespace free_breakfast
