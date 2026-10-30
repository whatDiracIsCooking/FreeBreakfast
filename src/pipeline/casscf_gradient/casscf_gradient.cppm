/// @file
/// @brief The `CasscfGradient` step: the analytic nuclear gradient of one root
/// of the @ref CasscfOutput, the DF-CASSCF's, every term of it
/// (@c mcscf::df_gradient over the density-fitted EriOutput), written as
/// `casscf_gradient.json`.
///
/// Device-shaped. The numerics are free_breakfast.mcscf.gradient's; `run`
/// only refuses what it cannot take and wires the pipeline's products into
/// it. See README.md.

export module free_breakfast.pipeline.casscf_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.mcscf.gradient; // mcscf::GradientTerms, in the product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief The nuclear gradient of one root of the CASSCF, and its terms.
struct CasscfGradientOutput {
  /// @brief dE/dR (hartree / bohr) in @c gradient.total, each n_atom x 3
  /// row-major, element 3 A + k, atoms in the Molecule's order; the five
  /// terms it sums beside it.
  mcscf::GradientTerms gradient;
};

/// @brief The DF-CASSCF nuclear gradient on @p res's device.
struct CasscfGradientStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "casscf_gradient";

  /// @brief The root whose energy is differentiated, in
  /// @ref CasscfOutput::energies' order.
  int32_t root = 0;
  /// @brief Where the gradient is written as JSON, if not empty: the root,
  /// its energy, the total and each term as n_atom rows of three. Arrives
  /// resolved (pipeline/config sets it to casscf_gradient.json in the run's
  /// output directory).
  std::filesystem::path json_path{};

  /// @brief @c mcscf::df_gradient at @ref root over a density-fitted
  /// EriOutput's B, L and aux basis, the core Hamiltonian, and the
  /// CasscfOutput's orbitals, CI vectors and weighted RDMs. The gradient is
  /// the CASSCF energy's only as far as the CASSCF converged, so a config
  /// that wants it to 1e-8 tightens `[casscf]` conv_tol_grad.
  ///
  /// @return The gradient; Unsupported, before anything runs, for an
  ///         integral-direct EriOutput (the CASSCF over it is not the
  ///         DF-CASSCF this is the gradient of), naming the reason;
  ///         InvalidConfig for a CasscfOutput whose 1-RDMs are not one ncas^2
  ///         block per weight; otherwise @c df_gradient's error (Unsupported
  ///         for a state average of more than one root, InvalidConfig for a
  ///         @ref root it does not have), or @c write_json's for a
  ///         @ref json_path that cannot be written.
  [[nodiscard]] Result<CasscfGradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, CasscfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
