/// @file
/// @brief The `Hessian` step: the Cartesian Hessian at the pipeline's
/// molecule -- the closed-shell RHF's, or with a `[casscf_gradient]` that
/// root's of a DF-CASSCF (state-specific or state-averaged), or with a
/// `[nevpt2_gradient]` the SC- or PC-NEVPT2 energy's on a single-state
/// DF-CASSCF, or numerically one SC- or PC-QD-NEVPT2 state's on a
/// DF-SA-CASSCF -- and its harmonic frequencies and normal modes. By default
/// analytic, @c hessian::df_rhf, @c hessian::df_casscf,
/// @c hessian::df_sacasscf, @c hessian::df_sc_nevpt2 or
/// @c hessian::df_pc_nevpt2 over one nested Pipeline run of the chain from
/// the molecule to `[scf]` or `[casscf]`; or by central differences of the analytic gradient of the
/// chain to `[gradient]`, `[casscf_gradient]` or `[nevpt2_gradient]`, rerun
/// at each of the 6N
/// displaced geometries, each displaced CASSCF started from the reference's
/// orbitals and CI and checked against it, and a QD-NEVPT2 state tracked by
/// its eigenvector's overlap with the reference's.
///
/// Node-shaped, because each method runs Pipelines of its own, as
/// `[optimize]`'s evaluations do. It reads the MoleculeOutput, so beside an
/// `[optimize]` it runs at the optimum. See README.md.

export module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.geometry; // geometry::Vibrations, in the product
// Every step the three chains rerun, and MoleculeOutput among the products
// they re-export.
export import free_breakfast.pipeline.gradient;
export import free_breakfast.pipeline.casscf_gradient;
export import free_breakfast.pipeline.nevpt2_gradient;

export namespace free_breakfast {

/// @brief How `[hessian]` computes the Hessian.
enum class HessianMethod : uint8_t {
  /// @c hessian::df_rhf at the SCF of one chain run, or the CASSCF root's
  /// (@c hessian::df_casscf, @c hessian::df_sacasscf), or its SC- or
  /// PC-NEVPT2 energy's (@c hessian::df_sc_nevpt2, @c hessian::df_pc_nevpt2):
  /// density fitting only.
  Analytic,
  /// Central differences of the analytic gradient, the RHF's, a DF-CASSCF
  /// root's or the SC- or PC-NEVPT2's: 6N chain reruns.
  Numerical,
};

/// @brief The Cartesian Hessian of the SCF energy, or of a CASSCF root's or
/// its NEVPT2 energy's, at the MoleculeOutput's
/// geometry, how far it is from the two identities it should keep, and the
/// harmonic vibrations it gives.
struct HessianOutput {
  /// @brief d²E/dx dy (hartree / bohr²), 3 n_atom square, row-major, x =
  /// 3 A + k, atoms in the Molecule's order, symmetrized, (H + Hᵀ) / 2.
  std::vector<double> hessian;
  /// @brief max |H - Hᵀ| before symmetrizing: the central differences', or
  /// the analytic terms' sum's (their rounding).
  double asymmetry{0.0};
  /// @brief max over x and l of |Σ_B H[x][3 B + l]| of @ref hessian: zero
  /// for a Hessian invariant under translation.
  double translational_sum{0.0};
  /// @brief @c geometry::harmonic_analysis of @ref hessian at the
  /// MoleculeOutput's atoms: the frequencies (cm⁻¹, ascending, an imaginary
  /// one negative) and the Cartesian normal modes.
  geometry::Vibrations vibrations;
  /// @brief dmu_c/dx (e), 3 n_atom x 3 row-major, element (x, c) at 3 x + c:
  /// @c hessian::rhf_dipole_derivative over the analytic Hessian's CPHF U^x,
  /// the origin at 0. Empty under HessianMethod::Numerical or a CASSCF, and
  /// for a basis with a shell above d (the dipole integrals' @c kDipoleMaxL).
  std::vector<double> dipole_derivative;
  /// @brief The IR intensities (km/mol), one per vibrations.frequencies:
  /// @c hessian::infrared_intensities of @ref dipole_derivative over the
  /// normal modes. Empty when @ref dipole_derivative is.
  std::vector<double> intensities;
};

/// @brief The RHF's, a DF-CASSCF root's or its NEVPT2 energy's Hessian at
/// the MoleculeOutput's geometry and its harmonic vibrations, on @p node's
/// device 0.
struct HessianStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "hessian";

  /// @brief How the Hessian is computed.
  HessianMethod method{HessianMethod::Analytic};
  /// @brief HessianMethod::Numerical's step h (bohr): column j is
  /// (g(R + h q_j) - g(R - h q_j)) / 2h, q_j coordinate j (@ref coordinates).
  /// Its error is O(h²) in the gradient's third derivative plus the SCF's
  /// noise over 2h, so the SCF in @ref scf must be converged well past the
  /// step. Under @ref ladder, the ladder's coarsest step instead. Not read by
  /// the analytic method.
  double displacement{1e-3};
  /// @brief HessianMethod::Numerical's displacement coordinates, or the
  /// analytic NEVPT2 Hessian's: n rows of n (n = 3 n_atom), row-major, row j
  /// coordinate j as a Cartesian displacement, orthonormal; the Hessian is
  /// differenced along each, or taken in them by @c hessian::df_sc_nevpt2 or
  /// @c df_pc_nevpt2, and reported in Cartesians, Qᵀ H Q. Empty: the
  /// Cartesian ones. The config's, never detected: symmetry-adapted ones
  /// name a failing block's irreps (README.md, "The ladder and the
  /// bilinearity check"), and keep SC's symmetric block apart from its
  /// broken one (README.md, "The NEVPT2 chain").
  std::vector<double> coordinates{};
  /// @brief HessianMethod::Numerical over the CASSCF chain: per coordinate,
  /// the fourth-order central stencil, (g(-2s) - 8 g(-s) + 8 g(s) - g(2s)) /
  /// 12s, at s = h, h/2 and h/4 (h @ref displacement; eight gradients, not
  /// two), each element's ratio of successive differences judged against
  /// 16, the gradient's noise measured by restarts, and the Hessian the
  /// finest step's. Needed by @ref pairs. pipeline/config sets it under
  /// @ref nevpt2_gradient unless the config says otherwise.
  bool ladder{false};
  /// @brief The coordinate pairs (i, j) the bilinearity check, Eq.
  /// (bilinear) of docs/nevpt2/nevpt2.tex, is run on under @ref ladder: the
  /// ladder along (q_i + q_j) / sqrt 2 against (H_ii + H_jj + H_ij + H_ji) /
  /// 2. A pair failing it above the ladders' bars withholds the frequencies.
  /// Empty: none. std::nullopt: every pair, up to
  /// @ref kMaxEveryPairCoordinates coordinates, and none (logged) above;
  /// pipeline/config's default under @ref nevpt2_gradient.
  std::optional<std::vector<std::array<int32_t, 2>>> pairs{std::in_place};
  /// @brief The most coordinates @ref pairs' std::nullopt checks every pair
  /// of: four atoms, 66 pairs, 528 gradients beside the ladder's 96.
  static constexpr size_t kMaxEveryPairCoordinates = 12;
  /// @brief @ref pc_floor's default: PSB3's second-order ladder is clean at
  /// a relative cut of 1e-10 and not at the round-off cut (#917), at a cost
  /// of under 6e-9 Eh in PC E_corr there (#923). README.md, "The PC floor".
  static constexpr double kPcFloor = 1e-10;
  /// @brief Under a PC @ref nevpt2_gradient, PC's conditioning floor
  /// (EnergyInputs::pc_floor) for the whole chain the Hessian differentiates:
  /// @c hessian::df_pc_nevpt2's, or each displaced gradient's
  /// (Nevpt2GradientStep::pc_floor). Non-negative; 0 is the library's
  /// round-off cut. Unread otherwise; a `[nevpt2_gradient]` with no
  /// `[hessian]` stays at 0.
  double pc_floor{kPcFloor};

  /// @brief The chain each method runs, configured as the config's own
  /// tables configure it (pipeline/config copies them in). The steps with no
  /// options -- the spherical transform, the pair classes, the overlap, the
  /// core Hamiltonian and the gradient -- are run at their defaults, and no
  /// step of the chain writes a file.
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};
  /// @brief The CASSCF the chain runs after the SCF, read only when
  /// @ref casscf_gradient or @ref nevpt2_gradient is set. Its active space is picked once, at the
  /// reference geometry; every displaced solve starts from the reference's
  /// orbitals, Loewdin-orthonormalized in the displaced overlap, and from its
  /// CI (CasscfStep::ci_guess).
  CasscfStep casscf{};
  /// @brief Set: the Hessian is this root's of @ref casscf, over density
  /// fitting only -- HessianMethod::Analytic's @c hessian::df_casscf (one
  /// root) or @c hessian::df_sacasscf (a state average, at its weights), or
  /// HessianMethod::Numerical's differences of this gradient. Empty: the
  /// RHF's.
  std::optional<CasscfGradientStep> casscf_gradient{};
  /// @brief Set: the Hessian is this SC- or PC-NEVPT2 energy's of the
  /// single-state @ref casscf's root, all-electron, over density fitting
  /// only -- HessianMethod::Analytic's @c hessian::df_sc_nevpt2 or
  /// @c hessian::df_pc_nevpt2 by @ref Nevpt2GradientStep::variant, or
  /// HessianMethod::Numerical's differences of this gradient over the
  /// CASSCF chain's displaced solves, each followed by this step. With
  /// @ref Nevpt2GradientStep::quasi_degenerate, numerically only, QD state
  /// @ref Nevpt2GradientStep::state's of the state-averaged @ref casscf: at
  /// each displacement the state whose H_eff eigenvector overlaps the
  /// reference's most (@ref tracked_qd_state), not the one of the same rank.
  /// Exclusive with @ref casscf_gradient.
  std::optional<Nevpt2GradientStep> nevpt2_gradient{};

  /// @brief Where the Hessian is written as JSON, if not empty: the method,
  /// the Hessian as 3 n_atom rows of 3 n_atom, the asymmetry, the
  /// translational sum, the frequencies and the normal modes, and when the
  /// product has them the dipole derivative and the IR intensities; under
  /// @ref ladder its rungs, ratios, bars and noise, and each checked pair's
  /// verdict (README.md has the keys). Written before a failing pair's
  /// error, without the frequencies and modes. Arrives
  /// resolved (pipeline/config sets it to hessian.json in the run's output
  /// directory).
  std::filesystem::path json_path{};

  /// @brief The Hessian at the MoleculeOutput's geometry: under
  /// HessianMethod::Analytic one Pipeline run of the chain ending in ScfStep
  /// and @c hessian::df_rhf over its products (with @ref casscf_gradient,
  /// the chain ending in @ref casscf and @c hessian::df_casscf or
  /// @c hessian::df_sacasscf at its solution; with @ref nevpt2_gradient,
  /// @c hessian::df_sc_nevpt2 or @c hessian::df_pc_nevpt2 there, in
  /// @ref coordinates); under
  /// HessianMethod::Numerical the 6N gradients at the geometry displaced by
  /// ±@ref displacement along each coordinate, each one Pipeline run of the
  /// chain ending in GradientStep, and their central differences; under
  /// @ref casscf_gradient, numerically, one chain to @ref casscf at the geometry, the
  /// reference, then the chain ending in it at each displaced geometry,
  /// started from the reference's orbitals orthonormalized in the displaced
  /// overlap, C (Cᵀ S C)^-1/2, and from its CI vectors, and held to it by
  /// @ref check_displaced_casscf; under @ref nevpt2_gradient the same, its
  /// gradient the NEVPT2's, each displacement's max |z^c|, smallest
  /// inactive gap and Z-vector residual logged. Under a quasi_degenerate
  /// @ref nevpt2_gradient, the QD state's gradient at the reference too,
  /// for its eigenvector R_P; at each displacement the state
  /// @ref tracked_qd_state picks, its gradient retaken when that is not the
  /// state of the same rank, and the QD gap, the smallest CASSCF root gap,
  /// the ensemble's edge E_{n+1} - E_n (a CASCI of n + 1 roots over the
  /// solve's orbitals) and max |zeta| logged.
  /// Under @ref ladder, eight displaced gradients per coordinate, four
  /// restarts and one re-solve 100 times tighter at each of four of them for
  /// the noise, and eight per pair of @ref pairs. Then symmetrized, checked, and its harmonic vibrations;
  /// under the analytic RHF, the dipole derivative from the same CPHF and the
  /// IR intensities, unless a shell is above d (logged as a warning).
  ///
  /// @return The Hessian; with no device, ErrorKind::GpuRuntime; under the
  ///         analytic method, Unsupported, before anything runs, for an
  ///         integral-direct @ref eri, and after the SCF for an ROHF one;
  ///         otherwise the first error of a Pipeline (`[gradient]`'s
  ///         Unsupported for an ROHF SCF among them), of @c hessian::df_rhf,
  ///         @c hessian::rhf_dipole_derivative or
  ///         @c geometry::harmonic_analysis, or the error writing
  ///         @ref json_path. Under @ref casscf_gradient: Unsupported, before
  ///         anything runs, for an integral-direct @ref eri; analytic,
  ///         InvalidConfig for a root the CASSCF does not have, and the
  ///         CASSCF's or the analytic Hessian's error (Divergence at a saddle,
  ///         Numerical at a degeneracy); numerical, a CASSCF's or the
  ///         gradient's error (MaxIterationsExceeded for one that does not
  ///         converge), naming the geometry, and @ref check_displaced_casscf's
  ///         Numerical for a displaced solve whose active space or roots are
  ///         not the reference's. Under @ref nevpt2_gradient, before
  ///         anything runs: InvalidConfig beside @ref casscf_gradient or for
  ///         a @ref Nevpt2GradientStep::frozen_core other than 0;
  ///         Unsupported for an integral-direct @ref eri, a state-averaged
  ///         @ref casscf (nroots above 1) without
  ///         @ref Nevpt2GradientStep::quasi_degenerate, or quasi_degenerate
  ///         under the analytic method, each naming the reason;
  ///         InvalidConfig for quasi_degenerate over fewer than two roots or
  ///         a @ref Nevpt2GradientStep::state past them; numerical QD, a
  ///         displacement's @ref tracked_qd_state refusal (Numerical),
  ///         naming the geometry, its gradient never differenced;
  ///         analytic, then the CASSCF's error and the NEVPT2
  ///         Hessian's as it is (Numerical naming SC's refused coordinate
  ///         pairs, or PC's refused classes, lifts and degenerate pairs);
  ///         numerical, the CASSCF chain's errors, and the NEVPT2
  ///         gradient's (PC's refusals among them), naming the geometry.
  ///         Before anything runs: InvalidConfig for @ref ladder or
  ///         @ref pairs under the analytic method, @ref coordinates under
  ///         it without @ref nevpt2_gradient, pairs
  ///         without the ladder, coordinates not n orthonormal rows of n, or
  ///         a pair outside n or named twice; Unsupported for the ladder
  ///         over the RHF chain. Last, Numerical naming every pair that is
  ///         not bilinear, the frequencies withheld.
  [[nodiscard]] Result<HessianOutput> run(View<MoleculeOutput> in, NodeResources &node) const;
};

/// @brief The numerical CASSCF Hessian's guard on one displaced solve: that
/// it converged to the reference's solution moved, not to another one. Each
/// converged active orbital t must lie more in the start's active space than
/// out of it, Σ_u ⟨start_u|S|active_t⟩² > 1/2 (an orbital swapped in from
/// the core or the virtuals has about 0), and each root r must be more the
/// reference's root r than anything else, ⟨reference_r|c_r⟩² > 1/2 (a root
/// that changed places has about 0). Both are 1 - O(h²) for a solve that
/// followed the reference.
///
/// @p s is the displaced overlap, @p nao square; @p start_active and
/// @p active the start's and the converged active orbitals, @p nao rows by
/// ncas columns, column-major, the start orthonormal in @p s; @p reference_ci
/// and @p ci @p nroots unit CI vectors each, root r at [r dim, (r + 1) dim).
/// Host only.
///
/// @return Success; Numerical naming the first active orbital or root that
///         left, and its weight; InvalidConfig for spans of mismatched sizes.
[[nodiscard]] Status check_displaced_casscf(size_t nao, std::span<const double> s,
                                            std::span<const double> start_active,
                                            std::span<const double> active,
                                            std::span<const double> reference_ci,
                                            std::span<const double> ci, size_t nroots);

/// @brief The numerical QD-NEVPT2 Hessian's state tracking at one displaced
/// solve: which of its H_eff eigenstates is the reference's state P. Each
/// displaced eigenvector u_k (H_eff's, over the displaced CASSCF roots) is
/// taken into the reference's root basis by @p signs, s_r = ±1 the sign of
/// ⟨reference root r|displaced root r⟩ (the roots matched by index, as
/// @ref check_displaced_casscf holds them), and overlapped with
/// @p reference, R_P: o_k = Σ_r R_P,r s_r u_k,r. The state is the k of
/// largest |o_k|, whatever its rank. A solve that followed the reference has
/// 1 - |o_k| = O(h²); a state that swapped with another has about 0.
///
/// @p reference holds n numbers, @p vectors n² (state k at [k n, (k + 1) n)),
/// @p signs n. Host only.
///
/// @return {k, |o_k|}; Numerical when two states tie, the second |o| within
///         1e-3 of the largest, or when 1 - |o_k| is 1e-3 or more (the bar
///         `test/gtest/data/qd_nevpt2_hessian/`'s references were sampled
///         under), naming the overlaps; InvalidConfig for spans of
///         mismatched sizes.
[[nodiscard]] Result<std::pair<size_t, double>> tracked_qd_state(std::span<const double> reference,
                                                                 std::span<const double> vectors,
                                                                 std::span<const double> signs);

/// @brief The numerical ladder's tightened CASSCF re-solve (its σ_solves):
/// @p conv_tol and @p conv_tol_grad each 100 times tighter, but never below
/// what fp64 resolves at the solve's weighted energy @p energy over
/// @p pairs non-redundant rotations, and never looser than given.
///
/// mcscf::solve converges on |ΔE| < conv_tol, ΔE the difference of two
/// computed energies, each good to about ε |E| (ε = 2.2e-16, one ulp at
/// most): its floor is 2 ε |E|. And on ‖g‖ < conv_tol_grad over @p pairs
/// elements, each g_pq = 2 (F_pq − F_qp) the difference of two Fock
/// elements built from what builds E, so each good to about 2 ε |E| and,
/// independent, ‖g‖'s floor 2 ε |E| √pairs. Host only.
///
/// @return {conv_tol, conv_tol_grad}: min(t, max(t / 100, its floor)) each.
[[nodiscard]] std::pair<double, double>
tightened_casscf_tolerances(double conv_tol, double conv_tol_grad, double energy, size_t pairs);

} // namespace free_breakfast
