/// @file
/// @brief The `PrimitivePairClasses` step: the basis's McMurchie-Davidson
///        primitive-pair records, on the device, for every integral step.
///
/// A thin wrapper over integrals.mmd's primitive_pair_classes. Device-shaped;
/// see README.md for the streams the records are built on, and why they are
/// finished when it returns.

export module free_breakfast.pipeline.primitive_pair_classes;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.step;
export import free_breakfast.integrals.mmd;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief Every shell-pair class of one @ref BasisSet, in ascending (la, lb)
/// order, each on its own work stream (@ref primitive_pair_classes).
///
/// What the one-electron and two-electron builds launch over. Built once per
/// geometry and only read after: a build reads the device side, on the class's
/// own stream.
struct PrimitivePairClassesOutput {
  /// @brief The classes; none for an empty basis.
  std::vector<PrimitivePairClass> classes;
};

/// @brief Builds the @ref BasisSet's primitive-pair records on @p res's device.
struct PrimitivePairClassesStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "primitive_pair_classes";

  /// @brief Form, order and bound every class's records on the device.
  /// primitive_pair_classes waits for each class's stream before it returns,
  /// so a consumer on any stream reads them finished. A failed copy, launch or
  /// wait is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<PrimitivePairClassesOutput> run(View<BasisSetOutput> in,
                                                       DeviceResources &res) const {
    return PrimitivePairClassesOutput{
        FBF_TRY(primitive_pair_classes(res, in.get<BasisSetOutput>().basis))};
  }
};

} // namespace free_breakfast
