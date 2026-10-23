/// @file
/// @brief The `PipekMezey` step: the SCF's occupied orbitals localized by
/// Pipek-Mezey.
///
/// A thin wrapper over localization.pm. Device-shaped.

export module free_breakfast.pipeline.pipek_mezey;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.pipeline.step;
export import free_breakfast.core.wwr; // Buffers, in the product
export import free_breakfast.localization.pm;
export import free_breakfast.pipeline.scf;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The localized occupied orbitals of a closed-shell RHF wavefunction,
/// on the device the step ran on.
struct PipekMezey {
  /// @brief The occupied orbitals localized: nao_sphe x @ref n_occ,
  /// column-major, C_local = C_occ U.
  Buffers::device<double> c;
  /// @brief The n_occ x n_occ orthogonal U, column-major.
  Buffers::device<double> u;
  /// @brief The occupied count, electron_count() / 2: the columns of @ref c.
  int32_t n_occ;
  /// @brief The sweeps taken and the functional before and after.
  PipekMezeyInfo info;
};

/// @brief Localizes the @ref Scf's occupied orbitals on @p res's device.
struct PipekMezeyStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "pipek_mezey";

  /// @brief The sweeps, the tolerances and the exponent.
  PipekMezeyOptions options;

  /// @brief The localized orbitals. The Scf has already refused an open
  /// shell, so the occupied count is half the electrons. Errors are
  /// pipek_mezey_localize's: InvalidConfig for an option out of range,
  /// MaxIterationsExceeded when the sweeps do not converge, GpuRuntime for a
  /// failed launch, copy or BLAS call.
  [[nodiscard]] Result<PipekMezey> run(View<Molecule, BasisSet, Overlap, Scf> in,
                                       DeviceResources &res) const {
    const BasisSet &basis = in.get<BasisSet>();
    const int32_t n_occ = in.get<Molecule>().electron_count() / 2;
    const size_t n_ao = basis.nao_sphe();
    const auto occ = static_cast<size_t>(n_occ);
    PipekMezey out{
        .c = Buffers::device<double>(n_ao * occ, res.device()),
        .u = Buffers::device<double>(occ * occ, res.device()),
        .n_occ = n_occ,
        .info = {},
    };
    // C is AO by MO with the occupied orbitals first, so its first n_occ
    // columns are the occupied block, packed.
    out.info = FBF_TRY(pipek_mezey_localize<double>(res, basis, n_occ, in.get<Overlap>().s.data(),
                                                    in.get<Scf>().c.data(), out.c.data(),
                                                    out.u.data(), options));
    return out;
  }
};

} // namespace free_breakfast
