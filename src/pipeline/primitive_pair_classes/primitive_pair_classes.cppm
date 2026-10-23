/// @file
/// @brief The `PrimitivePairClasses` step: the basis's McMurchie-Davidson
///        primitive-pair records, on the device, for every integral step.
///
/// A thin wrapper over integrals.mmd's primitive_pair_classes. Device-shaped;
/// see README.md for the streams the records are built on, and why it waits
/// on them.

export module free_breakfast.pipeline.primitive_pair_classes;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.basis;
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
struct PrimitivePairClasses {
  /// @brief The classes; none for an empty basis.
  std::vector<PrimitivePairClass> classes;
};

/// @brief Builds the @ref BasisSet's primitive-pair records on @p res's device.
struct PrimitivePairClassesStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "primitive_pair_classes";

  /// @brief Write, upload, order and bound every class's records, then wait
  /// for each class's stream, so a consumer on any stream reads them
  /// finished. A failed copy or wait is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<PrimitivePairClasses> run(View<BasisSet> in, DeviceResources &res) const {
    PrimitivePairClasses out{FBF_TRY(primitive_pair_classes(res, in.get<BasisSet>()))};
    for (const PrimitivePairClass &cls : out.classes) {
      FBF_TRY(gpu_status(cls.device_stream()->fork_stream().sync()));
    }
    return out;
  }
};

} // namespace free_breakfast
