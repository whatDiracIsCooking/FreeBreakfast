/// @file
/// @brief The `Overlap` step: the spherical AO overlap matrix S.
///
/// A thin wrapper over integrals.one_electron's overlap and linalg.matrix_ao's
/// cartesian_to_spherical: S is built Cartesian in a transient MatrixAOPair and
/// only its spherical form is published.

export module free_breakfast.pipeline.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.one_electron.overlap;
import free_breakfast.pipeline.step;
export import free_breakfast.linalg.matrix_ao;
export import free_breakfast.pipeline.spherical_transform;
export import free_breakfast.pipeline.primitive_pair_classes;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The spherical AO overlap S_mn = <m|n>, @c nao_sphe square.
struct Overlap {
  SphericalMatrixAO s;
};

/// @brief Builds the @ref BasisSet's overlap on @p res's device.
struct OverlapStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "overlap";

  /// @brief S, built Cartesian and carried onto the spherical layout. A
  /// failed launch or BLAS call is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<Overlap> run(View<BasisSet, SphericalTransform, PrimitivePairClasses> in,
                                    DeviceResources &res) const {
    MatrixAOPair s(res, in.get<BasisSet>(), in.get<SphericalTransform>().u);
    FBF_TRY(overlap(res, in.get<PrimitivePairClasses>().classes, s.cartesian()));
    FBF_TRY(s.cartesian_to_spherical(res));
    return Overlap{std::move(s.spherical())};
  }
};

} // namespace free_breakfast
