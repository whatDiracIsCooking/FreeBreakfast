/// @file
/// @brief The `Nac` step: the nonadiabatic couplings between pairs of roots of
/// the @ref CasscfOutput, a DF-SA-CASSCF at equal weights
/// (@c mcscf::df_nac over the density-fitted EriOutput), each with its
/// numerator and CSF term, written as `nac.json`.
///
/// Device-shaped. The numerics are free_breakfast.mcscf.nac's; `run` only
/// refuses what it cannot take and wires the pipeline's products into it. See
/// README.md.

export module free_breakfast.pipeline.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.mcscf.nac; // mcscf::Nac and NacOptions, in the step and product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief The couplings between pairs of roots of the CASSCF.
struct NacOutput {
  /// @brief (I, J), the bra and the ket, of each coupling, in
  /// @ref NacStep::pairs' order.
  std::vector<std::array<int32_t, 2>> pairs;
  /// @brief d_IJ = <I|dJ/dR>, h_IJ and d^CSF_IJ per pair, each n_atom x 3
  /// row-major, element 3 A + k, atoms in the Molecule's order; beside
  /// @ref pairs.
  std::vector<mcscf::Nac> couplings;
};

/// @brief The DF-SA-CASSCF nonadiabatic couplings on @p res's device.
struct NacStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nac";
  /// @brief The CASSCF's weights are equal when they span no more than this,
  /// absolute: df_nac's own tolerance on the weighted gamma against the
  /// roots' plain average. pipeline/config refuses a `[casscf]` past it too.
  static constexpr double kEqualWeights = 1e-10;

  /// @brief The (bra, ket) pairs, roots in @ref CasscfOutput::energies'
  /// order, each coupling <bra|d ket/dR>. Empty: every pair I < J, ordered by
  /// I then J.
  std::vector<std::array<int32_t, 2>> pairs{};
  /// @brief What each coupling's @c total is (mcscf::NacOptions); the
  /// numerator and the CSF term are written unscaled whatever it says.
  mcscf::NacOptions options{};
  /// @brief Where the couplings are written as JSON, if not empty: the
  /// options, then per pair the roots, the gap, h, csf and total as n_atom
  /// rows of three, and the Z-vector's products and residual. Arrives
  /// resolved (pipeline/config sets it to nac.json in the run's output
  /// directory).
  std::filesystem::path json_path{};

  /// @brief @c mcscf::df_nac for each of @ref pairs over a density-fitted
  /// EriOutput's B, L and aux basis, the core Hamiltonian, and the
  /// CasscfOutput's orbitals, CI vectors and RDMs. The couplings are the
  /// CASSCF's only as far as it converged, so a config that wants them to
  /// 1e-8 tightens `[casscf]` conv_tol_grad.
  ///
  /// @return The couplings; before anything runs on the device, Unsupported
  ///         for an integral-direct EriOutput (the CASSCF over it is not the
  ///         DF-CASSCF these are the couplings of) or a CASSCF at unequal
  ///         weights, each naming the reason, and InvalidConfig for a CASSCF
  ///         of fewer than two roots, a pair whose roots are equal or not
  ///         both the CASSCF's, or a CasscfOutput whose 1-RDMs are not one
  ///         ncas^2 block per weight; otherwise @c df_nac's error
  ///         (Unsupported for a gap below 1e-10 hartree without
  ///         mult_ediff), or @c write_json's for a @ref json_path that cannot
  ///         be written.
  [[nodiscard]] Result<NacOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, CasscfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
