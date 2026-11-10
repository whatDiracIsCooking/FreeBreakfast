/// @file
/// @brief The `Nevpt2Nac` step: the SC- or PC-QD-NEVPT2 derivative couplings
/// between pairs of QD states of the DF-SA-CASSCF in the @ref CasscfOutput
/// (@c nevpt2::qd_sc_nac or @c nevpt2::qd_pc_nac over the density-fitted
/// EriOutput), each with its pieces and diagnostics, and the effective
/// Hamiltonian's eigenvalues and R, written as `nevpt2_nac.json`.
///
/// Device-shaped. The numerics are free_breakfast.nevpt2.nac's; `run` only
/// refuses what it cannot take and wires the pipeline's products into it. See
/// README.md.

export module free_breakfast.pipeline.nevpt2_nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.footprint;
import free_breakfast.mcscf.nac;                // mcscf::NacOptions, in the step
import free_breakfast.pipeline.nevpt2_gradient; // Nevpt2Variant, in the step
export import free_breakfast.nevpt2.nac;        // nevpt2::QdNac, in the product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief The QD-NEVPT2 couplings between pairs of QD states, and the
/// effective Hamiltonian they are of.
struct Nevpt2NacOutput {
  /// @brief H_eff's eigenvalues E_K, ascending: the QD states, SC's or PC's
  /// as @ref Nevpt2NacStep::variant says (hartree).
  std::vector<double> energies;
  /// @brief H_eff's eigenvectors R over the CASSCF roots, n x n, state K's
  /// column at [K n, (K + 1) n), each signed so that its largest-magnitude
  /// component is positive (the couplings' convention).
  std::vector<double> vectors;
  /// @brief (P, Q), the bra and the ket, of each coupling, in
  /// @ref Nevpt2NacStep::pairs' order.
  std::vector<std::array<int32_t, 2>> pairs;
  /// @brief d_PQ = <Psi_P|dPsi_Q/dR> and its pieces and diagnostics per
  /// pair, each n_atom x 3 row-major, element 3 A + k, atoms in the
  /// Molecule's order; beside @ref pairs.
  std::vector<nevpt2::QdNac> couplings;
};

/// @brief The SC- or PC-QD-NEVPT2 derivative couplings on @p res's device.
struct Nevpt2NacStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nevpt2_nac";

  /// @brief The (bra, ket) pairs of QD states, indices into H_eff's
  /// eigenvalues, ascending, each coupling <bra|d ket/dR>. Empty: every
  /// pair P < Q, ordered by P then Q.
  std::vector<std::array<int32_t, 2>> pairs{};
  /// @brief SC or PC: which of @c nevpt2::qd_sc_nac and
  /// @c nevpt2::qd_pc_nac `run` calls, and whose H_eff the states are.
  Nevpt2Variant variant = Nevpt2Variant::Sc;
  /// @brief What each coupling's @c total is (mcscf::NacOptions); the
  /// numerator and the other pieces are written unscaled whatever it says.
  mcscf::NacOptions options{};
  /// @brief Where the couplings are written as JSON, if not empty: the
  /// options, H_eff's eigenvalues and R, then per pair the states, the gap,
  /// h, cas, pt2, csf and total as n_atom rows of three, and the
  /// diagnostics. Arrives resolved (pipeline/config sets it to
  /// nevpt2_nac.json in the run's output directory).
  std::filesystem::path json_path{};

  /// @brief H_eff and R of @ref variant over @c nevpt2::qd_inputs, then
  /// @c nevpt2::qd_sc_nac, or @c nevpt2::qd_pc_nac for PC, for each of
  /// @ref pairs over a density-fitted EriOutput's B, L and aux basis, the
  /// core Hamiltonian, and the CasscfOutput's orbitals, CI vectors and RDMs,
  /// each class's slabs in one batch. The couplings are the CASSCF's only as
  /// far as it converged, so a config that wants them to 1e-8 tightens
  /// `[casscf]` conv_tol_grad.
  ///
  /// @return The couplings; before anything runs on the device, Unsupported
  ///         for an open-shell ScfOutput (n_alpha != n_beta), a molecule of
  ///         multiplicity other than 1 or an integral-direct EriOutput,
  ///         naming the reason, and InvalidConfig for a CASSCF of fewer
  ///         than two roots, a pair whose states are equal or not both
  ///         H_eff's, or a CasscfOutput whose 1-RDMs are not one ncas^2
  ///         block per weight; otherwise the effective Hamiltonian's error
  ///         (PC's refusal, a metric with no positive eigenvalue or a
  ///         non-positive denominator, a Numerical error naming the class, among them),
  ///         the coupling's (QD states or coupled CASSCF roots within
  ///         mcscf::kRootGapFloor), or @c write_json's for a @ref json_path
  ///         that cannot be written.
  [[nodiscard]] Result<Nevpt2NacOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;

  /// @brief What @ref run holds at @p dims, all of it freed on return (the
  /// couplings are on the host): the CASSCF's orbitals and CI vectors back on
  /// the device; H_eff over @c nevpt2::qd_inputs (the effective
  /// Hamiltonian's walk over every pair) and its eigensolve; then one pair's
  /// @c nevpt2::qd_sc_nac_footprint or @c qd_pc_nac_footprint (each pair's
  /// is freed before the next), over the planned CASSCF's roots and weights
  /// (dims.casscf_nroots and dims.casscf_unequal_weights), PC's classes at
  /// the rank that holds the most. Empty for what @ref run refuses before
  /// allocating: an open shell, no density fitting, or under two roots.
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions &dims) const;
};

} // namespace free_breakfast
