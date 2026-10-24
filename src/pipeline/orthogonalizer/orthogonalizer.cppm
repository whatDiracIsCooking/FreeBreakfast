/// @file
/// @brief The `Orthogonalizer` step: the symmetric (Loewdin) orthogonalizer
///        X = S^{-1/2} of the spherical overlap.
///
/// A thin wrapper over recipes.orthogonalize. Device-shaped; see README.md for
/// the one device contract it does not keep.

export module free_breakfast.pipeline.orthogonalizer;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.pipeline.overlap;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief X = S^{-1/2} over the spherical AOs, @c nao_sphe square: X^T S X = I
/// on the modes above the eigenvalue floor, which turns F C = S C eps into a
/// standard eigenproblem.
struct OrthogonalizerOutput {
  SphericalMatrixAO x;
};

/// @brief Builds the @ref OverlapOutput's orthogonalizer on @p res's device.
struct OrthogonalizerStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "orthogonalizer";

  /// @brief An overlap eigenvalue at or below this drops its mode
  /// (contributes 0 to X). Should be non-negative.
  double eigenvalue_floor = kOrthogonalizeEigenvalueFloor<double>;

  /// @brief X from S. A non-converged eigensolve is ErrorKind::Numerical; a
  /// failed runtime, solver or BLAS call is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<OrthogonalizerOutput> run(View<BasisSetOutput, OverlapOutput> in,
                                                 DeviceResources &res) const {
    const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
    SphericalMatrixAO x(res, in.get<BasisSetOutput>().basis);
    FBF_TRY(orthogonalize(res, static_cast<int32_t>(s.n()), s.data(), x.data(), eigenvalue_floor));
    return OrthogonalizerOutput{std::move(x)};
  }
};

} // namespace free_breakfast
