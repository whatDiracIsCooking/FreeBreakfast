/// @file
/// @brief The `Gradient` step: the closed-shell RHF nuclear gradient at the
/// ScfOutput's orbitals, over either kind of EriOutput, every term of it
/// (@c gradient::df_rhf density-fitted, @c gradient::rhf integral-direct),
/// written as `gradient.json`.
///
/// Device-shaped. The numerics are free_breakfast.gradient's; `run` only
/// refuses what it cannot take and wires the pipeline's products into it. See
/// README.md.

export module free_breakfast.pipeline.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.gradient; // gradient::RhfTerms, in the product
export import free_breakfast.pipeline.scf;

export namespace free_breakfast {

/// @brief The nuclear gradient of the ScfOutput's energy, and its terms.
struct GradientOutput {
  /// @brief dE/dR (hartree / bohr) in @c gradient.total, each n_atom x 3
  /// row-major, element 3 A + k, atoms in the Molecule's order; the six terms
  /// it sums beside it.
  gradient::RhfTerms gradient;
};

/// @brief The RHF nuclear gradient on @p res's device.
struct GradientStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "gradient";

  /// @brief Where the gradient is written as JSON, if not empty: the SCF
  /// energy, the total and each term as n_atom rows of three. Arrives resolved
  /// (pipeline/config sets it to gradient.json in the run's output directory).
  std::filesystem::path json_path{};

  /// @brief @c gradient::df_rhf over a density-fitted EriOutput's B, L and
  /// aux basis, or @c gradient::rhf over an integral-direct one's K classes,
  /// and the ScfOutput's C, orbital energies and n_alpha occupied orbitals.
  /// The gradient is the SCF energy's only as far as the SCF converged, so a
  /// config that wants it to 1e-8 tightens `[scf]`.
  ///
  /// @return The gradient; Unsupported, before anything runs, for an ROHF
  ///         ScfOutput (n_alpha != n_beta), naming the reason; otherwise
  ///         @c df_rhf's or @c rhf's error, or @c write_json's for a
  ///         @ref json_path that cannot be written.
  [[nodiscard]] Result<GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
