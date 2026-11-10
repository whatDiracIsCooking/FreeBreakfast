/// @file
/// @brief The `Optimize` step: a geometry optimization, @c geometry::optimize
/// over the energy and gradient of the chain from the molecule to
/// `[gradient]` (RHF), to `[casscf_gradient]` (a DF-CASSCF root, followed
/// from geometry to geometry over a state average) or to
/// `[nevpt2_gradient]` (its SC- or PC-NEVPT2 total energy, or one SC- or
/// PC-QD-NEVPT2 state's, followed from geometry to geometry), rerun as a
/// nested Pipeline at every geometry; or, with `meci`, the minimum-energy
/// conical intersection of two roots of a DF-SA-CASSCF, @c
/// geometry::optimize_meci over one @c mcscf::df_intersection per geometry,
/// or with a quasi-degenerate `[nevpt2_gradient]` beside it of two SC- or
/// PC-QD-NEVPT2 states, over one @c nevpt2::qd_sc_intersection (or
/// qd_pc_intersection) per geometry.
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
import free_breakfast.pipeline.footprint;
import free_breakfast.io.json;         // json, recorded_optimize's entries
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
  /// root's; over a state average, its root is the start's, followed from
  /// each geometry to the next by @ref match_roots. Empty, with
  /// @ref nevpt2_gradient empty too: it ends in `[gradient]`, the RHF's.
  std::optional<CasscfGradientStep> casscf_gradient{};
  /// @brief Set: the chain ends in @ref casscf and this NEVPT2 gradient of
  /// its root, and the energy optimized is that root's NEVPT2 total energy,
  /// E_cas + E_corr; with its quasi_degenerate, the energy E_K of its QD
  /// state K, followed from each geometry to the next by @ref
  /// qd_state_overlap. At most one of it and @ref casscf_gradient is set.
  /// Beside @ref meci, quasi-degenerate, it names the QD-NEVPT2 whose states
  /// cross (its variant; its state is unread, and must be 0).
  std::optional<Nevpt2GradientStep> nevpt2_gradient{};
  /// @brief Set, {I, J}: the step finds the minimum-energy conical
  /// intersection of roots I and J of @ref casscf, a state average in which
  /// the two carry equal weights, over density fitting only, by
  /// @c geometry::optimize_meci (whose `max_gap` @ref options holds) over
  /// @c mcscf::df_intersection's g_I, g_J and h_IJ at every geometry; the
  /// energy reported is their mean. The pair is followed by
  /// @ref pair_overlap. With @ref nevpt2_gradient set, quasi-degenerate,
  /// {P, Q} are instead QD-NEVPT2 states (H_eff's eigenvalues ascending),
  /// SC or PC as its variant, at any weights, their g_P, g_Q and h_PQ
  /// @c nevpt2::qd_sc_intersection's (or qd_pc_intersection's), and the
  /// pair followed by @ref pair_overlap over H_eff's eigenvectors.
  /// @ref casscf_gradient may not be set beside it.
  std::optional<std::array<int32_t, 2>> meci{};

  /// @brief Where every evaluated geometry is written, in order, as a
  /// multi-frame XYZ (Angstrom), if not empty; each frame's comment line holds
  /// its energy, and whether the step to it was accepted.
  std::filesystem::path trajectory{};
  /// @brief Where the optimized geometry is written as XYZ, if not empty.
  /// pipeline/config sets it, and loads the molecule from it after this step.
  std::filesystem::path final_xyz{};
  /// @brief Where the history is written as JSON, if not empty: the final
  /// energy, then per evaluation the energy, the four norms, the trust radius
  /// and whether it was accepted; over a state-averaged root, the root
  /// followed and its matched overlap; over a QD state, its matched overlap
  /// and H_eff's smallest gap. Under @ref meci the energy is the pair's mean,
  /// beside it the lower and upper energies and the gap (at the end too),
  /// and per evaluation the branching plane's dimension, the
  /// @ref pair_overlap in place of whether it was accepted, and the CASSCF's
  /// ||g|| and largest CI residual; over QD states also H_eff's gap of the
  /// pair, the third-state margin E_{Q+1} - E_Q (Q the higher of the pair;
  /// null when there is none) and the CASSCF's smallest root gap.
  ///
  /// The trajectory and the history are written as the walk goes, by
  /// @ref recorded_optimize: a walk that fails leaves the evaluations before
  /// the failure in both, the history with the failure's message as "error"
  /// and no final energy, and no final_xyz.
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
  /// A state-averaged root is followed by identity: at every evaluation
  /// after the first, the CASSCF's roots are matched to the last
  /// evaluation's by @c mcscf::state_overlaps and @ref match_roots, and the
  /// gradient is taken of the root the last one matched, not of the index,
  /// its overlap logged.
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
  ///         runs, as are @ref check_meci's refusals; under @ref meci,
  ///         @ref pair_overlap's Numerical error at the evaluation where the
  ///         pair is lost, or @c mcscf::df_intersection's (over QD states,
  ///         @c nevpt2::qd_sc_intersection's or qd_pc_intersection's); over a
  ///         state-averaged root, @ref match_roots'
  ///         Numerical error at the evaluation where the root is lost; over
  ///         a QD state, @ref qd_state_overlap's; or
  ///         @c mcscf::state_overlaps' error;
  ///         otherwise the first error of an
  ///         evaluation's Pipeline, @c geometry::optimize's own
  ///         (MaxIterationsExceeded past @c options.max_steps), or the error
  ///         writing an output file. Every error after the walk starts
  ///         leaves the trajectory and history @ref recorded_optimize keeps.
  [[nodiscard]] Result<OptimizeOutput> run(View<> in, NodeResources &node) const;

  /// @brief What @ref run holds at @p dims, over the RHF chain: the start's
  /// Molecule, held through the walk; at each evaluation the chain's plan,
  /// folded as pre-flight folds a plan (plan_footprint over its steps'
  /// footprints, in the order the evaluation runs them) and freed whole
  /// before the next; then the optimum's Molecule, the product. The
  /// chain's steps share @p node's DeviceResources, so its workspace is
  /// this step's. Every geometry has the same dimensions, so one fold holds
  /// for all of them. The history and the trajectory are pageable host
  /// memory, not modelled.
  ///
  /// None over a CASSCF chain (@ref casscf_gradient, @ref nevpt2_gradient,
  /// @ref meci), whose carried orbitals and library calls are not all
  /// footprinted yet: pre-flight refuses it under a cap. Transitional, as
  /// step_footprint's optional is (#1122).
  [[nodiscard]] std::optional<MemoryFootprint> footprint(const FootprintDimensions &dims) const;

  /// @brief Under @ref meci, what @ref run refuses before anything runs, and
  /// pipeline/config before it plans: @ref casscf_gradient set beside it, or
  /// a @ref nevpt2_gradient that is not quasi-degenerate (one state, none
  /// to cross) or names a state other than 0 (InvalidConfig); @ref casscf of
  /// one root, I = J, or a root (or QD state) outside its average
  /// (InvalidConfig); over CASSCF roots, unequal weights on I and J
  /// (Unsupported: @c mcscf::df_intersection's rotation between them would
  /// divide by E_I - E_J); over QD states, a molecule of multiplicity other
  /// than 1 (Unsupported); an integral-direct @ref eri (Unsupported).
  /// Without @ref meci, nothing.
  [[nodiscard]] Status check_meci() const;
};

/// @brief @c geometry::optimize of @p start's atoms (their positions, bohr,
/// are x0) over @p energy_and_gradient under @p step's options, keeping its
/// trajectory, final_xyz and history (each skipped when empty) as the walk
/// goes: the one write path @ref OptimizeStep::run's every walk shares.
///
/// Before the walk, a final_xyz left by an earlier run is removed. After
/// every evaluation geometry::optimize records, the trajectory is rewritten
/// with every recorded frame, and the history with every recorded entry
/// under "iterations" and nothing else, so a walk killed partway leaves its
/// trail. On the walk's exit: converged, the history's head gets "energy" and
/// final_xyz is written; failed, at whichever evaluation, the history gets
/// "error", the failure's message, and final_xyz is not written. Each entry
/// is the Iteration's energy, four norms, trust radius and accepted, and
/// what @p annotate(k, entry) adds to the k-th.
///
/// @return The solution; geometry::optimize's error, after the files carry
///         it, or the error writing one of them.
[[nodiscard]] Result<geometry::Solution> recorded_optimize(
    const OptimizeStep &step, std::span<const Atom> start,
    const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
        &energy_and_gradient,
    const std::function<void(size_t k, json &entry)> &annotate = {});

/// @brief @ref recorded_optimize's walk by @c geometry::optimize_meci: each
/// entry the MECISolution::Iteration's energies (the mean as "energy",
/// "lower", "upper" and "gap"), four norms, trust radius and branching
/// dimension, and @p annotate's; converged, the head's "energy" is the mean,
/// beside "lower", "upper" and "gap", and what @p annotate_head adds.
[[nodiscard]] Result<geometry::MECISolution>
recorded_optimize_meci(const OptimizeStep &step, std::span<const Atom> start,
                       const std::function<Result<std::array<double, 2>>(
                           std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                           std::span<double> h_ij)> &energies_and_gradients,
                       const std::function<void(size_t k, json &entry)> &annotate = {},
                       const std::function<void(json &head)> &annotate_head = {});

/// @brief The overlap below which @ref match_roots and @ref qd_state_overlap
/// refuse: a followed root, or QD state, whose state has turned by more than
/// about 26 degrees in one step is losing its identity to another, well
/// before the two mix equally (1/sqrt 2), and an optimizer step away from a
/// near crossing turns it by far less.
inline constexpr double kStateOverlapFloor = 0.9;

/// @brief One last root's match in this geometry's CASSCF, as
/// @ref match_roots makes it.
struct RootMatch {
  /// @brief This geometry's root J the last root I is matched to.
  int32_t root;
  /// @brief S_IJ over the 2-norm of row I of S, signed: the cosine between
  /// J and I's part in this geometry's roots, and its sign the phase that
  /// carries I's CI into J's.
  double overlap;
};

/// @brief @p root_overlaps, @c mcscf::state_overlaps' n x n row-major
/// S_IJ = <I(last)|J(this)>, with each row I divided by its 2-norm: element
/// I n + J the cosine between J and root I's part in this geometry's roots.
///
/// The whole determinant's overlap, the core's among it, falls with the
/// step's length for every root alike (to about 0.76 at 0.1 bohr on
/// formaldehyde's C-O), which is not a root turning; this is what
/// @ref match_roots holds to the floor, and a block of it (a pair's 2 x 2)
/// what a check of several roots at once reads.
///
/// @return The normalized n x n; InvalidConfig if @p root_overlaps is not
///         n^2 for an n at least 1; Numerical for a row of zeros, naming it.
[[nodiscard]] Result<std::vector<double>>
normalized_overlaps(std::span<const double> root_overlaps);

/// @brief The CASSCF roots of the last geometry matched to this one's: each
/// last root I to the root J of largest |S_IJ|, which must be a different J
/// for every I, with that S_IJ over row I's norm.
///
/// @p root_overlaps is @c mcscf::state_overlaps' n x n row-major, element
/// I n + J = <I(last)|J(this)>, matched over @ref normalized_overlaps. Each
/// last root in @p followed must keep its normalized overlap at least
/// @ref kStateOverlapFloor: below, the root is lost (a crossing), not
/// followed onto another surface.
///
/// @return One match per last root, in order; InvalidConfig if
///         @p root_overlaps is not n^2 for an n at least 1, or a
///         @p followed root is not one of the n; Numerical if the roots do
///         not match one to one (or @ref normalized_overlaps' row of
///         zeros), or a followed
///         root's normalized overlap is below the floor, naming it.
[[nodiscard]] Result<std::vector<RootMatch>> match_roots(std::span<const double> root_overlaps,
                                                         std::span<const int32_t> followed = {});

/// @brief The pair an @ref OptimizeStep's `meci` follows, matched across one
/// step: the smaller singular value of the 2 x 2 block of rows and columns
/// @p i and @p j of @ref normalized_overlaps(@p root_overlaps). The block is
/// the two roots' space carried from the last geometry into this one's, so
/// I and J may swap order, or flip phase, freely; a singular value falls
/// only when that space turns into another root's.
///
/// @return The overlap; InvalidConfig if @p root_overlaps is not n^2 for an
///         n at least 1, or @p i and @p j are not two distinct roots of the
///         n; @ref normalized_overlaps' Numerical for a row of zeros;
///         Numerical if the overlap is below @ref kStateOverlapFloor (the
///         pair is lost to another root), naming it.
[[nodiscard]] Result<double> pair_overlap(std::span<const double> root_overlaps, int32_t i,
                                          int32_t j);

/// @brief The QD pair an @ref OptimizeStep's `meci` follows over
/// quasi-degenerate states, matched across one step: the smaller singular
/// value of the 2 x 2 block M_ab = u_a(last)^T S u_b(this), a and b each
/// @p p or @p q, S @ref normalized_overlaps(@p root_overlaps) and u_K
/// H_eff's eigenvector K, @p previous's from the last geometry and
/// @p current's from this one (n x n each, state K's at [K n, (K + 1) n)).
/// As the CASSCF pair's, P and Q may swap order, flip phase or mix freely,
/// and so may the CASSCF roots under them: only the two states' space is
/// followed, through the roots' overlaps, never a matching of roots.
///
/// @return The overlap; InvalidConfig if @p root_overlaps is not n^2 for an
///         n at least 1, @p previous or @p current is not n^2, or @p p and
///         @p q are not two distinct states of the n;
///         @ref normalized_overlaps' Numerical for a row of zeros; Numerical
///         if the overlap is below @ref kStateOverlapFloor (the pair is lost
///         to another state), naming it.
[[nodiscard]] Result<double> pair_overlap(std::span<const double> root_overlaps,
                                          std::span<const double> previous,
                                          std::span<const double> current, int32_t p, int32_t q);

/// @brief The QD state an @ref OptimizeStep follows, matched across one
/// step: |sum_I u_I s_I u'_pi(I)|, @p previous's u (one entry per root of
/// the last CASSCF) carried into @p current's (one per root of this one) by
/// the root matching pi and the signs s that @ref match_roots makes of
/// @p root_overlaps. u's own sign is arbitrary, hence the magnitude.
///
/// @return The overlap; InvalidConfig if the three sizes do not agree (n,
///         n and n^2, n at least 1); @ref match_roots' Numerical if the roots
///         do not match one to one; Numerical if the overlap is below
///         @ref kStateOverlapFloor (the state is lost, a crossing), naming
///         it.
[[nodiscard]] Result<double> qd_state_overlap(std::span<const double> root_overlaps,
                                              std::span<const double> previous,
                                              std::span<const double> current);

} // namespace free_breakfast
