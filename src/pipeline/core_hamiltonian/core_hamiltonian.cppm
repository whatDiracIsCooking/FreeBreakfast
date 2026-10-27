/// @file
/// @brief The `CoreHamiltonian` step: the spherical core Hamiltonian H = T + V.
///
/// A thin wrapper over integrals.one_electron's core_hamiltonian and
/// recipes.matrix_ao's cartesian_to_spherical: H is built Cartesian in a
/// transient MatrixAOPair and only its spherical form is published.

export module free_breakfast.pipeline.core_hamiltonian;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.integrals.one_electron.core_hamiltonian;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.pipeline.spherical_transform;
export import free_breakfast.pipeline.primitive_pair_classes;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The spherical core Hamiltonian H = T + V, @c nao_sphe square: the
/// kinetic energy plus the attraction to the @ref Molecule's nuclei.
struct CoreHamiltonianOutput {
  SphericalMatrixAO h;
};

/// @brief Builds the core Hamiltonian on @p res's device.
struct CoreHamiltonianStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "core_hamiltonian";

  /// @brief H, built Cartesian and carried onto the spherical layout. A class
  /// above f is ErrorKind::InvalidConfig; a failed launch or BLAS call is
  /// ErrorKind::GpuRuntime.
  [[nodiscard]] Result<CoreHamiltonianOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const {
    MatrixAOPair h(res, in.get<BasisSetOutput>().basis, in.get<SphericalTransformOutput>().u);
    FBF_TRY(core_hamiltonian(res, in.get<PrimitivePairClassesOutput>().classes,
                             in.get<MoleculeOutput>().molecule, h.cartesian()));
    FBF_TRY(h.cartesian_to_spherical(res));
    return CoreHamiltonianOutput{std::move(h.spherical())};
  }
};

} // namespace free_breakfast
