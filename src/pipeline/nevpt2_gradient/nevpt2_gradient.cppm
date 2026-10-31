/// @file
/// @brief The `Nevpt2Gradient` step: the analytic nuclear gradient of the
/// SC-NEVPT2 total energy of the single-state DF-CASSCF in the
/// @ref CasscfOutput (@c nevpt2::sc_gradient over the density-fitted
/// EriOutput), every term of it and the Lagrangian's diagnostics, written as
/// `nevpt2_gradient.json`.
///
/// Device-shaped. The numerics are free_breakfast.nevpt2.gradient's; `run`
/// only refuses what it cannot take and wires the pipeline's products into
/// it. See README.md.

export module free_breakfast.pipeline.nevpt2_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.nevpt2.gradient; // nevpt2::ScGradient, in the product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief Which NEVPT2 a @ref Nevpt2GradientStep differentiates.
enum class Nevpt2Variant : uint8_t {
  Sc, ///< Strongly contracted, @c nevpt2::sc_gradient.
  Pc, ///< Partially contracted: refused until its gradient lands (#564).
};

/// @brief The nuclear gradient of one root's NEVPT2 total energy, and its
/// terms.
struct Nevpt2GradientOutput {
  /// @brief dE/dR (hartree / bohr) of E_cas + E_corr in
  /// @c gradient.terms.total, n_atom x 3 row-major, element 3 A + k, atoms in
  /// the Molecule's order; the five terms it sums beside it, the class
  /// energies of E_corr, z^c with max |z^c| and the smallest inactive gap,
  /// the Z-vector with its PCG iterations and residual, and X^L's residues.
  nevpt2::ScGradient gradient;
};

/// @brief The SC-NEVPT2 nuclear gradient on @p res's device.
struct Nevpt2GradientStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nevpt2_gradient";

  /// @brief The root whose energy is differentiated, in
  /// @ref CasscfOutput::energies' order: 0, since the CASSCF must have one.
  int32_t root = 0;
  /// @brief SC or PC. PC is refused, in `run` as when the config is read,
  /// until its gradient lands.
  Nevpt2Variant variant = Nevpt2Variant::Sc;
  /// @brief Where the gradient is written as JSON, if not empty: the root,
  /// the energies, the total and each term as n_atom rows of three, and the
  /// diagnostics. Arrives resolved (pipeline/config sets it to
  /// nevpt2_gradient.json in the run's output directory).
  std::filesystem::path json_path{};

  /// @brief @c nevpt2::sc_gradient at @ref root over a density-fitted
  /// EriOutput's B, L and aux basis, the core Hamiltonian, and the
  /// CasscfOutput's orbitals (core and virtuals canonical), CI vector and
  /// RDMs, each class's slabs in one batch. The gradient is the energy's
  /// only as far as the CASSCF converged, so a config that wants it to 1e-6
  /// tightens `[casscf]` conv_tol_grad.
  ///
  /// @return The gradient; Unsupported, before anything runs, for
  ///         @ref variant PC, for an open-shell ScfOutput (n_alpha !=
  ///         n_beta), a molecule of multiplicity other than 1, an
  ///         integral-direct EriOutput (naming `[eri]`) or a state average
  ///         (a CasscfOutput of more than one root), each naming the reason;
  ///         InvalidConfig for a @ref root the CASSCF does not have or a
  ///         CasscfOutput whose 1-RDMs are not one ncas^2 block per root;
  ///         otherwise @c sc_gradient's error, or @c write_json's for a
  ///         @ref json_path that cannot be written.
  [[nodiscard]] Result<Nevpt2GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
