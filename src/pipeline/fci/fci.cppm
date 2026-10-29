/// @file
/// @brief The `Fci` step: the lowest roots of the molecule's spin of the full
/// CI over every one of the @ref ScfOutput's orbitals, refused up front when
/// the determinant space would not fit on the device.
///
/// Device-shaped. CASCI with no core and every orbital active: `run` makes
/// CasciStep's calls (`integrals.active_space`, `ci.solver` over `ci.sigma`'s
/// `ci::Sigma`, whose `rdm1` gives the 1-RDMs), all private to the
/// implementation unit. See README.md.

export module free_breakfast.pipeline.fci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.davidson; // DavidsonInfo, in the product
export import free_breakfast.pipeline.scf;

export namespace free_breakfast {

/// @brief The lowest roots of the full CI over a set of RHF orbitals, of the
/// molecule's spin S, ascending in energy.
struct FciOutput {
  /// @brief The orbitals, all of them correlated: the ScfOutput's, in its order.
  int32_t nmo;
  /// @brief The electrons: N_alpha = (n_electrons + 2S) / 2 and
  /// N_beta = (n_electrons - 2S) / 2, 2S + 1 the molecule's multiplicity.
  int32_t n_electrons;
  /// @brief The total energy per root, nuclear repulsion included (hartree).
  std::vector<double> energies;
  /// @brief <S^2> per root.
  std::vector<double> spin_squares;
  /// @brief The spin-summed 1-RDM per root over the ScfOutput's orbitals, root r at
  /// [r nmo^2, (r + 1) nmo^2), nmo x nmo, row-major.
  std::vector<double> rdm1;
  /// @brief What the Davidson solve did.
  DavidsonInfo davidson;
};

/// @brief FCI over the @ref ScfOutput's canonical orbitals, on @p res's device.
struct FciStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "fci";

  /// @brief The roots wanted, at least 1.
  int32_t nroots = 1;

  /// @brief The @ref nroots lowest roots of spin S, 2S + 1 the molecule's
  /// multiplicity, by CasciStep's path with ncas the ScfOutput's orbital count
  /// and nelecas every electron, so no core and the ScfOutput's orbitals in
  /// their own order. First, before anything is allocated,
  /// the device memory the determinant space needs (the CI vectors, the
  /// Davidson subspace, the sigma's scratch, the n^4 integrals) is estimated
  /// against what the device has free. `ci::solve` runs at a 1e-11 relative
  /// residual and 400 iterations, not its defaults (README.md has why).
  /// Synchronizes res.stream(): the integrals and every Davidson iteration
  /// are read back to the host.
  ///
  /// @return The roots; InvalidConfig for @ref nroots below 1; Unsupported,
  ///         before anything is allocated, for an open-shell ScfOutput
  ///         (n_alpha != n_beta), whose n_alpha + n_beta is otherwise the
  ///         electron count; InvalidConfig for a multiplicity that does
  ///         not fit the electrons (2S above N, of the other parity, or
  ///         N_alpha above nmo); Allocation when
  ///         the estimate exceeds the device's free memory; GpuRuntime for a
  ///         failed memory query or copy; or the error of the first module
  ///         call that failed: `active_space_hamiltonian`'s, `ci::Sigma`'s, or
  ///         `ci::solve`'s.
  [[nodiscard]] Result<FciOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
           CoreHamiltonianOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
