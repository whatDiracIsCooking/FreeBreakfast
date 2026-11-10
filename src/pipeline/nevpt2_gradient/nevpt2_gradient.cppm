/// @file
/// @brief The `Nevpt2Gradient` step: the analytic nuclear gradient of the
/// SC- or PC-NEVPT2 total energy of the single-state DF-CASSCF in the
/// @ref CasscfOutput (@c nevpt2::sc_gradient or @c nevpt2::pc_gradient over
/// the density-fitted EriOutput), or with @ref
/// Nevpt2GradientStep::quasi_degenerate of one SC- or PC-QD-NEVPT2 state of
/// the DF-SA-CASSCF (@c nevpt2::qd_sc_gradient or @c nevpt2::qd_pc_gradient),
/// every term of it and the Lagrangian's diagnostics, written as
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
import free_breakfast.pipeline.footprint;
export import free_breakfast.nevpt2.gradient; // nevpt2::ScGradient and PcGradient, in the product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief Which NEVPT2 a @ref Nevpt2GradientStep differentiates.
enum class Nevpt2Variant : uint8_t {
  Sc, ///< Strongly contracted, @c nevpt2::sc_gradient (@c qd_sc_gradient).
  Pc, ///< Partially contracted, @c nevpt2::pc_gradient (@c qd_pc_gradient).
};

/// @brief The nuclear gradient of one root's NEVPT2 total energy, or of one
/// QD-NEVPT2 state's, and its terms.
struct Nevpt2GradientOutput {
  /// @brief State-specific (ScGradient, PcGradient): dE/dR (hartree / bohr)
  /// of E_cas + E_corr in @c gradient.terms.total, n_atom x 3 row-major,
  /// element 3 A + k, atoms in the Molecule's order; the five terms it sums
  /// beside it, the class energies of E_corr, z^c with max |z^c| and the
  /// smallest inactive gap, the Z-vector with its PCG iterations and
  /// residual, and X^L's residues. Quasi-degenerate (QdScGradient,
  /// QdPcGradient): dE_K/dR in @c gradient.total, the same layout, E_K
  /// H_eff's eigenvalue @ref Nevpt2GradientStep::state, with its two
  /// derivative passes and its Lagrangian. The alternative is the step's
  /// @ref Nevpt2GradientStep::variant and @ref
  /// Nevpt2GradientStep::quasi_degenerate.
  std::variant<nevpt2::ScGradient, nevpt2::PcGradient, nevpt2::QdScGradient, nevpt2::QdPcGradient>
      gradient;
  /// @brief The core orbitals left uncorrelated, @ref
  /// Nevpt2GradientStep::frozen_core resolved ("auto" counted out).
  int32_t frozen_core = 0;
};

/// @brief The SC- or PC-NEVPT2 nuclear gradient, state-specific or of one
/// QD-NEVPT2 state, on @p res's device.
struct Nevpt2GradientStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nevpt2_gradient";

  /// @brief The root whose energy is differentiated, in
  /// @ref CasscfOutput::energies' order: 0, since the CASSCF must have one.
  /// Unread with @ref quasi_degenerate, where @ref state names what is
  /// differentiated.
  int32_t root = 0;
  /// @brief SC or PC: which of @c nevpt2::sc_gradient and
  /// @c nevpt2::pc_gradient `run` calls.
  Nevpt2Variant variant = Nevpt2Variant::Sc;
  /// @brief Differentiate one QD-NEVPT2 state of the state-averaged CASSCF,
  /// @c nevpt2::qd_sc_gradient or @c nevpt2::qd_pc_gradient, rather than
  /// one root's NEVPT2. Needs two roots or more.
  bool quasi_degenerate = false;
  /// @brief With @ref quasi_degenerate, the QD state K: an index into the
  /// effective Hamiltonian's eigenvalues, ascending. 0 otherwise.
  int32_t state = 0;
  /// @brief `[nevpt2]`'s @c frozen_core, read the same way: a count, or
  /// Nevpt2Step::kFrozenCoreAuto. The lowest canonical core orbitals that
  /// leave every class with a core index and stay in F^I; their rotations
  /// against the correlated core enter through z^c (src/nevpt2/gradient/
  /// README.md "The frozen core"). 0, the default, correlates every core
  /// orbital.
  int32_t frozen_core = 0;
  /// @brief The single-state gradient's Z-vector tolerance, @c
  /// nevpt2::sc_gradient's and @c pc_gradient's @c zvector_tolerance (their
  /// default): its true residual is held to it. No TOML key: `[hessian]`'s
  /// ladder tightens it to measure what it leaves in the gradient. Unread
  /// with @ref quasi_degenerate.
  double zvector_tolerance = 1e-10;
  /// @brief PC's conditioning floor, @c nevpt2::pc_gradient's and
  /// @c qd_pc_gradient's @c pc_floor (EnergyInputs::pc_floor): 0, the
  /// round-off cut. No TOML key: a PC `[hessian]` sets it on the chain it
  /// differentiates (HessianStep::pc_floor, #923), and a caller may set it on
  /// a QD step it builds (Nevpt2Psb3.PcQdActiveActiveGateAtTheTwist, #1020).
  /// Unread by SC.
  double pc_floor = 0.0;
  /// @brief Where the gradient is written as JSON, if not empty: the root,
  /// the energies, the total and each term as n_atom rows of three, and the
  /// diagnostics. Arrives resolved (pipeline/config sets it to
  /// nevpt2_gradient.json in the run's output directory).
  std::filesystem::path json_path{};

  /// @brief @c nevpt2::sc_gradient, or @c nevpt2::pc_gradient for
  /// @ref variant PC, at @ref root over a density-fitted
  /// EriOutput's B, L and aux basis, the core Hamiltonian, and the
  /// CasscfOutput's orbitals (core and virtuals canonical), CI vector and
  /// RDMs, each class's slabs in one batch. The gradient is the energy's
  /// only as far as the CASSCF converged, so a config that wants it to 1e-6
  /// tightens `[casscf]` conv_tol_grad.
  ///
  /// With @ref quasi_degenerate, @c nevpt2::qd_sc_gradient or
  /// @c nevpt2::qd_pc_gradient of @ref state over the same products, γ the
  /// weighted average of the roots' 1-RDMs. H_eff and u_K are the
  /// gradient's own, from the same CASSCF: the step does not read a
  /// `[nevpt2]` step's. With @ref frozen_core, each call's @c nfrozen.
  ///
  /// @return The gradient; Unsupported, before anything runs, for an
  ///         open-shell ScfOutput (n_alpha != n_beta), a molecule of
  ///         multiplicity other than 1, an integral-direct EriOutput (naming
  ///         `[eri]`) or, without @ref quasi_degenerate, a state average (a
  ///         CasscfOutput of more than one root), each naming the reason;
  ///         InvalidConfig for a @ref root the CASSCF does not have, for
  ///         @ref quasi_degenerate over fewer than two roots, a @ref state
  ///         past the roots, or a @ref state without @ref quasi_degenerate,
  ///         or a CasscfOutput whose 1-RDMs are not one ncas^2 block per
  ///         root, or a @ref frozen_core (resolved) past the CASSCF's core
  ///         (Nevpt2Step::resolve_frozen_core); otherwise the gradient's error (PC's refusals, a metric
  ///         with no positive eigenvalue or a pole -- an intruder, |t| > 1,
  ///         or a denominator at round-off -- among them, the
  ///         energy's Numerical error unchanged), or @c write_json's for a
  ///         @ref json_path that cannot be written.
  [[nodiscard]] Result<Nevpt2GradientOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
           CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
      DeviceResources &res) const;

  /// @brief What @ref run holds at @p dims: the CASSCF's orbitals and CI
  /// vectors back on the device as the gradient's mcscf::Solution, freed on
  /// return, and the gradient's own footprint (@c nevpt2::sc_gradient_footprint,
  /// @c pc_gradient_footprint, or with @ref quasi_degenerate @c
  /// qd_sc_gradient_footprint or @c qd_pc_gradient_footprint over the planned
  /// CASSCF's roots and weights, dims.casscf_nroots and
  /// dims.casscf_unequal_weights), its tiles the least they run with and PC's
  /// classes at the rank that holds the most. Persistent: the gradient's
  /// device arrays, which the product keeps (z^c and the Z-vector, and QD's
  /// Lagrangian). Sized over the active space at @p dims, ncore from the
  /// electrons, and @ref frozen_core over dims.atoms. Empty for what @ref run
  /// refuses before allocating.
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions &dims) const;
};

} // namespace free_breakfast
