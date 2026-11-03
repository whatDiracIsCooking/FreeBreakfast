/// @file
/// @brief The `Optimize` step: a geometry optimization, @c geometry::optimize
/// over the energy and gradient of the chain from the molecule to
/// `[gradient]` (RHF), to `[casscf_gradient]` (a DF-CASSCF root) or to
/// `[nevpt2_gradient]` (its SC- or PC-NEVPT2 total energy, or one SC- or
/// PC-QD-NEVPT2 state's, followed from geometry to geometry), rerun as a
/// nested Pipeline at every geometry.
///
/// Node-shaped, because each evaluation runs a Pipeline of its own. Its
/// product, `OptimizeOutput`, is declared in pipeline.molecule, whose
/// `MoleculeStep::load` reads it: a config with `[optimize]` loads the
/// molecule from the final.xyz this step writes, so every step that reads the
/// molecule runs at the optimum. See README.md.

export module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.geometry; // geometry::Options, an option
// Every step the three chains rerun, MoleculeOutput and OptimizeOutput among
// the products they re-export.
export import free_breakfast.pipeline.gradient;
export import free_breakfast.pipeline.casscf_gradient;
export import free_breakfast.pipeline.nevpt2_gradient;

export namespace free_breakfast {

/// @brief Optimizes the geometry of @ref molecule over the energy and
/// gradient of the chain the other members configure, on @p node's device 0.
struct OptimizeStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "optimize";

  /// @brief The convergence thresholds (hartree / bohr, bohr) and the step
  /// cap, and the trust radius and starting Hessian, @c geometry::optimize's.
  geometry::Options options{};

  /// @brief The chain rerun at every geometry, configured as the config's own
  /// tables configure it (pipeline/config copies them in). @ref molecule's xyz
  /// is the starting geometry. The steps with no options -- the spherical
  /// transform, the pair classes, the overlap, the core Hamiltonian and the
  /// gradient -- are run at their defaults, and no output path is kept: no
  /// step of the chain writes a file.
  MoleculeStep molecule{};
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};
  /// @brief The CASSCF the chain runs after the SCF, read only when
  /// @ref casscf_gradient or @ref nevpt2_gradient is set. Its active space is picked once, at the
  /// first geometry; every later one starts from the previous evaluation's
  /// converged orbitals, re-orthonormalized in the new overlap.
  CasscfStep casscf{};
  /// @brief Set: the chain ends in @ref casscf and this gradient of one of
  /// its roots, over density fitting only, and the energy optimized is that
  /// root's. Empty, with @ref nevpt2_gradient empty too: it ends in
  /// `[gradient]`, the RHF's.
  std::optional<CasscfGradientStep> casscf_gradient{};
  /// @brief Set: the chain ends in @ref casscf and this NEVPT2 gradient of
  /// its root, and the energy optimized is that root's NEVPT2 total energy,
  /// E_cas + E_corr; with its quasi_degenerate, the energy E_K of its QD
  /// state K, followed from each geometry to the next by @ref
  /// qd_state_overlap. At most one of it and @ref casscf_gradient is set.
  std::optional<Nevpt2GradientStep> nevpt2_gradient{};

  /// @brief Where every evaluated geometry is written, in order, as a
  /// multi-frame XYZ (Angstrom), if not empty; each frame's comment line holds
  /// its energy, and whether the step to it was accepted.
  std::filesystem::path trajectory{};
  /// @brief Where the optimized geometry is written as XYZ, if not empty.
  /// pipeline/config sets it, and loads the molecule from it after this step.
  std::filesystem::path final_xyz{};
  /// @brief Where the history is written as JSON, if not empty: the final
  /// energy, then per evaluation the energy, the four norms, the trust radius
  /// and whether it was accepted, and over a QD state its matched overlap
  /// and H_eff's smallest gap.
  std::filesystem::path history{};

  /// @brief @c geometry::optimize from @ref molecule's geometry, each
  /// evaluation one Pipeline run of the chain, its energy and gradient the
  /// ScfOutput's energy and the GradientOutput's total; under
  /// @ref casscf_gradient, the CasscfOutput's energy of its root and the
  /// CasscfGradientOutput's total; under @ref nevpt2_gradient, the root's
  /// E_cas + E_corr and the Nevpt2GradientOutput's total, or with its
  /// quasi_degenerate H_eff's eigenvalue E_K and dE_K/dR. Every step the
  /// chain holds keeps its own refusals, and the first one stops the
  /// optimization as it is, never stepped through: for an open shell,
  /// `[gradient]`'s; under @ref nevpt2_gradient, its every refusal (an SC
  /// degenerate inactive pair, a degenerate H_eff pair among them) at
  /// whichever geometry it is met.
  ///
  /// A QD state K is followed by identity: at every evaluation after the
  /// first, the CASSCF roots are matched to the last evaluation's by
  /// @c mcscf::state_overlaps and u_K, H_eff's eigenvector K, to the last
  /// u_K in that matched root basis, by @ref qd_state_overlap, whose
  /// overlap is logged. The index K differentiated never changes: H_eff's
  /// eigenvalues are ascending and continuous, so the state at K keeps its
  /// identity until two of them nearly cross, which is where the overlap
  /// falls, and the optimization is refused there rather than carried onto
  /// the other surface.
  ///
  /// @return The optimized Molecule, on device 0, and its energy; with no
  ///         device, ErrorKind::GpuRuntime; with both @ref casscf_gradient
  ///         and @ref nevpt2_gradient set, InvalidConfig, and under either
  ///         with an integral-direct @ref eri, Unsupported, before anything
  ///         runs; over a QD state, @ref qd_state_overlap's Numerical error
  ///         at the evaluation where the state is lost, or
  ///         @c mcscf::state_overlaps' error;
  ///         otherwise the first error of an
  ///         evaluation's Pipeline, @c geometry::optimize's own
  ///         (MaxIterationsExceeded past @c options.max_steps), or the error
  ///         writing an output file.
  [[nodiscard]] Result<OptimizeOutput> run(View<> in, NodeResources &node) const;
};

/// @brief The overlap below which @ref qd_state_overlap refuses: a QD state
/// whose u has turned by more than about 26 degrees in one step is losing
/// its identity to another, well before the two mix equally (1/sqrt 2), and
/// an optimizer step away from a near crossing turns it by far less.
inline constexpr double kQdStateOverlapFloor = 0.9;

/// @brief The QD state an @ref OptimizeStep follows, matched across one
/// step: |sum_I u_I s_I u'_pi(I)|, @p previous's u (one entry per root of
/// the last CASSCF) carried into @p current's (one per root of this one) by
/// the root matching pi and the signs s that @p root_overlaps gives.
///
/// @p root_overlaps is @c mcscf::state_overlaps' n x n row-major, element
/// I n + J = <I(last)|J(this)>. Each last root I is matched to the root J
/// of largest |S_IJ|, which must be a different J for every I, and s_I is
/// that S_IJ's sign. u's own sign is arbitrary, hence the magnitude.
///
/// @return The overlap; InvalidConfig if the three sizes do not agree (n,
///         n and n^2, n at least 1); Numerical if the roots do not match one
///         to one, or the overlap is below @ref kQdStateOverlapFloor (the
///         state is lost, a crossing), naming it.
[[nodiscard]] Result<double> qd_state_overlap(std::span<const double> root_overlaps,
                                              std::span<const double> previous,
                                              std::span<const double> current);

} // namespace free_breakfast
