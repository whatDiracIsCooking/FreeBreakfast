/// @file
/// @brief free_breakfast.tools.qd_nevpt2_gradient: the energy path of the
///        DF-SA-CASSCF + QD-NEVPT2 chain at displaced geometries, and from it
///        the numerical nuclear derivative of the effective Hamiltonian.
///
/// `free-breakfast-qd-nevpt2-gradient` (main.cpp) writes the
/// `qd_nevpt2_gradient/` and `qd_nevpt2_nac/coupling/` references over
/// @ref nevpt2::ChainSampler, and a GPU test or a consumer such as #785's MECI
/// check calls @ref nevpt2::numerical_heff in process. None of it calls the
/// derivative code (Z-vectors, `df_gradient`, `qd_*_gradient`): every number
/// is a difference of the energy path, which `qd_nevpt2/`'s goldens pin to
/// independent code, and that independence is what makes it an oracle.
///
/// Everything here is in `free_breakfast::nevpt2`, the layer of the quantity
/// it differences, with names no src/nevpt2 module uses.

export module free_breakfast.tools.qd_nevpt2_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.atom;
// The chain's products (CasscfOutput, Nevpt2Output, EriOutput, ...) and
// mcscf::Solution, in the sampler's accessors.
import free_breakfast.pipeline.nevpt2;

export namespace free_breakfast::nevpt2 {

/// @brief The continuity guard's bound on a CASSCF root's energy off its
/// first-order prediction, over |dR|^2 (hartree/bohr^2): another stationary
/// point or another state misses it by orders of magnitude.
inline constexpr double kPredictionBound = 2.0;
/// @brief The guard's bound on 1 - |overlap| of a matched CASSCF root with
/// the reference's (and, in the tool's energy modes, of a matched H_eff
/// eigenvector).
inline constexpr double kCiTol = 1e-3;
/// @brief The polish's target: a polished point is rerun from its own
/// orbitals until the CASSCF's ||g|| falls under it, at most a fixed number
/// of times. An overlap or an off-diagonal of H_eff feels a residual ||g||
/// to first order, an eigenvalue only through the roots' energies.
inline constexpr double kPolishTarget = 3e-12;
/// @brief The bound on any error bar of the coupling mode, and of
/// @ref numerical_heff: a consumer's bar built from them is bounded. Not
/// #744's 2e-7: the weighted case's PC overlap reaches 3.9e-6 over its
/// small steps (qd_nevpt2_nac/README.md has every family's). A bound, not a
/// target.
inline constexpr double kCouplingErrorBarTol = 5e-6;

/// @brief A molecule and its chain: DF `[eri]` + DF-SA-CASSCF +
/// `[nevpt2] quasi_degenerate` (one root: the per-root SC- and PC-NEVPT2).
struct NumericalChain {
  /// @brief The reference geometry (bohr), charge 0, singlet.
  std::vector<Atom> atoms;
  /// @brief The orbital basis, a compiled-in name or a path.
  std::string basis;
  /// @brief The fitting set.
  std::string aux_basis{"cc-pVDZ-JKFIT"};
  /// @brief The active space.
  int32_t ncas = 0;
  /// @brief The active electrons.
  int32_t nelecas = 0;
  /// @brief The reference's active orbitals (0-based); empty, the RHF's
  /// frontier ones.
  std::vector<int32_t> active;
  /// @brief The state average's weights, one per root.
  std::vector<double> weights;
  /// @brief PC beside SC: PC refused at any point fails the run.
  bool pc = true;
};

/// @brief The chain at one geometry, in the reference's root basis: each
/// displaced CASSCF root matched to the reference's by
/// `mcscf::state_overlaps` and signed so that that overlap is positive.
struct ChainPoint {
  /// @brief The CASSCF roots' energies, in the reference's order.
  std::vector<double> casscf;
  /// @brief QD: SC's H_eff eigenvalues, ascending (not matched to the
  /// reference's states). One root: its SC-NEVPT2 total.
  std::vector<double> sc;
  /// @brief As @ref sc, for PC; empty without NumericalChain::pc.
  std::vector<double> pc;
  /// @brief QD: SC's H_eff eigenvectors in the reference's root basis, state
  /// k (as @ref sc) at [k n, (k + 1) n).
  std::vector<double> sc_vectors;
  /// @brief As @ref sc_vectors, for PC.
  std::vector<double> pc_vectors;
  /// @brief n x n, s[M n + N] = <M(R0)|N(R)>.
  std::vector<double> s;
  /// @brief QD: SC's symmetrized H_eff, n x n.
  std::vector<double> sc_heff;
  /// @brief QD: PC's symmetrized H_eff, n x n; empty without
  /// NumericalChain::pc.
  std::vector<double> pc_heff;
  /// @brief The CASSCF's ||g||.
  double gradient_norm = 0.0;
  /// @brief 1 - the smallest singular value of the active space's overlap
  /// with the reference's.
  double subspace = 0.0;
  /// @brief The worst 1 - |overlap| of a matched root.
  double ci = 0.0;
  /// @brief The smallest gap between two canonical core energies.
  double core_gap = std::numeric_limits<double>::infinity();
  /// @brief The smallest gap between two canonical virtual energies.
  double virtual_gap = std::numeric_limits<double>::infinity();
  /// @brief The converged orbitals (AO by MO, column-major), for a restart.
  std::vector<double> c;
  /// @brief Their energies.
  std::vector<double> mo_energies;
};

/// @brief The chain's reference point and its displaced points.
///
/// The reference is RHF (SAD guess), then the DF-SA-CASSCF over its frontier
/// orbitals (or NumericalChain::active) to ||g|| < 1e-10 and |ΔE| < 1e-13,
/// then the NEVPT2. Each displaced point re-converges the DF-SA-CASSCF from
/// the reference's orbitals, Löwdin-orthonormalized in the displaced overlap,
/// at the same bars, and reruns the NEVPT2, as one nested `Pipeline`. With
/// polish, every point, the reference's too, is then rerun from its own
/// orbitals until ||g|| < @ref kPolishTarget (at most ten times, eight macro
/// iterations each, the lowest ||g|| kept).
class ChainSampler {
public:
  /// @brief Converges the reference, and checks it: density-fitted, the
  /// CASSCF under its bar, QD-NEVPT2 present (and PC-QD with
  /// NumericalChain::pc), or one root's PC not refused.
  /// @return The sampler, or the first step's error, Numerical for a check
  ///         that failed, InvalidConfig for a chain with no root.
  [[nodiscard]] static Result<ChainSampler> create(NodeResources &node, const NumericalChain &chain,
                                                   bool polish);

  /// @brief Moves the reference and its products.
  ChainSampler(ChainSampler &&) noexcept;
  /// @brief Moves the reference and its products.
  ChainSampler &operator=(ChainSampler &&) noexcept;
  ~ChainSampler();

  /// @brief The chain at the reference displaced by @p displacement (3N,
  /// bohr, atom-major), from the reference's orbitals.
  ///
  /// The CASSCF roots' guard is applied here: the CASSCF under its bar,
  /// every reference root matched to a displaced root alone, each matched
  /// overlap within @ref kCiTol of 1, the active space's within 1e-3, the QD
  /// energies present, and PC's cut the reference's modes (as a multiset).
  /// H_eff's eigenstates are not matched: that, and the energy prediction,
  /// are the caller's.
  /// @return The point, or Numerical naming the guard that refused it.
  [[nodiscard]] Result<ChainPoint> at(std::span<const double> displacement);
  /// @brief As the other overload, from @p from's orbitals.
  [[nodiscard]] Result<ChainPoint> at(std::span<const double> displacement, const ChainPoint &from);

  /// @brief The reference as a point: its own roots (s the identity), its
  /// eigenvectors, gaps and orbitals.
  [[nodiscard]] const ChainPoint &reference() const;
  /// @brief The reference's CASSCF.
  [[nodiscard]] const CasscfOutput &casscf() const;
  /// @brief The reference's NEVPT2.
  [[nodiscard]] const Nevpt2Output &nevpt2() const;
  /// @brief The reference's molecule.
  [[nodiscard]] const Molecule &molecule() const;
  /// @brief The reference's orbital basis.
  [[nodiscard]] const BasisSet &basis() const;
  /// @brief The reference's primitive-pair classes.
  [[nodiscard]] const std::vector<PrimitivePairClass> &classes() const;
  /// @brief The reference's fitted integrals.
  [[nodiscard]] const DensityFitted &fit() const;
  /// @brief The reference's core Hamiltonian.
  [[nodiscard]] const SphericalMatrixAO &core_hamiltonian() const;
  /// @brief The reference's CASSCF as an `mcscf::Solution`.
  [[nodiscard]] const mcscf::Solution &solution() const;
  /// @brief How many polishing passes have run, the reference's included.
  [[nodiscard]] size_t polish_passes() const;

private:
  struct State;
  explicit ChainSampler(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

/// @brief A Richardson value and its error bar.
struct Richardson {
  /// @brief (4 D(h/4) - D(h/2)) / 3, over the two finest steps.
  double value = 0.0;
  /// @brief |value - the same over the two coarser| + noise / (2 h/4).
  double error_bar = 0.0;
};

/// @brief #696's convention, as the tool applies it: Richardson over
/// central differences @p coarse, @p middle and @p fine at halving steps,
/// and its error bar, the distance from the coarser pair's plus the
/// per-point @p noise over twice @p finest_step.
[[nodiscard]] Richardson richardson(double coarse, double middle, double fine, double noise,
                                    double finest_step);

/// @brief A differenced quantity per component.
struct Differenced {
  /// @brief Richardson's value per component.
  std::vector<double> value;
  /// @brief Its error bar per component.
  std::vector<double> error_bar;
};

/// @brief One scheme's ∇H_eff at the reference, along the caller's
/// components, in the reference's root basis.
struct HeffDerivative {
  /// @brief The root count n.
  size_t n = 0;
  /// @brief The central-difference steps (bohr), halving.
  std::array<double, 3> steps{};
  /// @brief The per-point noise every error bar carries (hartree).
  double noise = 0.0;
  /// @brief H_eff at the reference, n x n.
  std::vector<double> heff;
  /// @brief Its eigenvalues, ascending.
  std::vector<double> energies;
  /// @brief R, its eigenvectors, state K at [K n, (K + 1) n).
  std::vector<double> vectors;
  /// @brief Per component and step, (H_eff(+h) - H_eff(-h)) / 2h, n x n.
  std::vector<std::array<std::vector<double>, 3>> central;
  /// @brief Per element [I n + J], its derivative.
  std::vector<Differenced> elements;
  /// @brief Per state K, g_K = R_K^T (∇H_eff) R_K.
  std::vector<Differenced> g;
  /// @brief Per pair P < Q, in the order (0, 1), (0, 2), ..., (1, 2), ...,
  /// h_PQ = R_P^T (∇H_eff) R_Q.
  std::vector<Differenced> h;
};

/// @brief SC's, and PC's, ∇H_eff at one reference.
struct NumericalHeff {
  /// @brief The reference's CASSCF root energies.
  std::vector<double> casscf_energies;
  /// @brief The reference CASSCF's ||g||, after the polish.
  double gradient_norm = 0.0;
  /// @brief SC's.
  HeffDerivative sc;
  /// @brief PC's, with NumericalChain::pc.
  std::optional<HeffDerivative> pc;
  /// @brief The worst CASSCF guard values over every displaced point:
  /// 1 - overlap of the active space and of a matched root, and the energy
  /// prediction's (hartree/bohr^2).
  double subspace = 0.0;
  /// @brief As @ref subspace, for the matched roots' CI.
  double ci = 0.0;
  /// @brief As @ref subspace, for the energy's prediction.
  double prediction = 0.0;
  /// @brief The displaced points run.
  size_t points = 0;
};

/// @brief The numerical ∇H_eff of the chain at its reference geometry,
/// along the caller's @p components (each 3N, bohr, atom-major; not
/// necessarily unit), by central differences at @p steps (halving) and
/// Richardson, with every point polished.
///
/// Only the CASSCF roots are matched (`ChainSampler::at`), never H_eff's
/// eigenstates: at a QD degeneracy a step rotates u_P and u_Q within their
/// pair by O(1), while H_eff stays smooth. The CASSCF's continuity guard,
/// without the derivative code: per component and step, each root's
/// |E_r(+h) + E_r(-h) - 2 E_r(0)| / 2|dR|^2, the even part of its
/// first-order prediction's miss, under @ref kPredictionBound.
///
/// Every error bar is @ref richardson's with @p noise, the per-point noise
/// of H_eff's elements and of any contraction of them (hartree): the case's
/// energy noise, as `qd_nevpt2_gradient/README.md` sets it.
/// @return The derivatives; InvalidConfig for fewer than two roots, a
///         component of the wrong length, or a step not positive; Numerical
///         for a refused point or guard, or any element's, g_K's or h_PQ's
///         error bar above @ref kCouplingErrorBarTol; or the chain's error.
[[nodiscard]] Result<NumericalHeff> numerical_heff(NodeResources &node, const NumericalChain &chain,
                                                   std::span<const std::vector<double>> components,
                                                   std::array<double, 3> steps, double noise);

/// @brief tr(W ∇H_eff) per component, W n x n with w[I n + J] its (I, J)
/// element, differenced as the elements are: its Richardson value and error
/// bar from the contracted central differences.
[[nodiscard]] Differenced contract(const HeffDerivative &d, std::span<const double> w);

/// @brief A pair's branching-plane quantities, #785's C2.
struct HeffPair {
  /// @brief ḡ = (g_P + g_Q) / 2.
  Differenced mean;
  /// @brief x = g_Q - g_P.
  Differenced x;
  /// @brief h_PQ.
  Differenced h;
};

/// @brief ḡ, x and h of states @p p and @p q over @p vectors (as
/// HeffDerivative::vectors; the reference's, or a rotation of them within
/// the pair). ḡ and span{x, h} are invariant to such a rotation.
[[nodiscard]] HeffPair heff_pair(const HeffDerivative &d, std::span<const double> vectors, size_t p,
                                 size_t q);

} // namespace free_breakfast::nevpt2
