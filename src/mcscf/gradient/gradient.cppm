/// @file
/// @brief The analytic nuclear gradient of a converged DF-CASSCF on the GPU,
/// state-specific, of one root of a state average or of a weighted mix of its
/// roots, term by term,
///
///     dE/dR = Sum D dT + Sum D dV - Sum W dS + dE_2/dR + dV_nn,
///
/// the two-electron part one Gam_mnP against d(mn|P) and one W_PQ against
/// d(P|Q). A state average's root is over its Z-vector's multipliers, or the
/// caller's, and a caller may add an MO adjoint of its own to the same
/// densities. At a transition density the core's own energy is switched off
/// (@ref df_transition_gradient). And the multipliers that hold the roots
/// apart for a Lagrangian not invariant to their rotation
/// (@ref root_pair_multipliers, @ref zeta_terms). See README.md for each
/// term's derivation.

export module free_breakfast.mcscf.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.solver;
import free_breakfast.ci.rdm;
import free_breakfast.mcscf.solve;
import free_breakfast.response.zvector;

export namespace free_breakfast::mcscf {

/// @brief A CASSCF nuclear gradient and the terms it sums, each flat and
/// row-major as @c (n_atom, 3), as @c gradient::RhfTerms: element 3 A + k is
/// dE/dR_{A,k} (hartree / bohr), atoms in the molecule's order.
struct GradientTerms {
  /// @brief Sum_mn D_mn dT_mn/dR, D the AO 1-RDM.
  std::vector<double> kinetic;
  /// @brief Sum_mn D_mn dV_mn/dR, the operator's part included.
  std::vector<double> potential;
  /// @brief -Sum_mn W_mn dS_mn/dR, W = C sym(F) C^T, F the generalized Fock.
  std::vector<double> overlap;
  /// @brief The fitted two-electron energy's derivative: J, the core and
  /// core-active K and the active 2-RDM term, through the three-center
  /// integrals' and the metric's derivatives.
  std::vector<double> two_electron;
  /// @brief dV_nn/dR.
  std::vector<double> nuclear_repulsion;
  /// @brief The sum of the five.
  std::vector<double> total;
};

/// @brief The MO-basis adjoint of a term a caller adds to the Lagrangian
/// (@c docs/nevpt2/nevpt2.tex Eq. grad): with Phi(h, B) that term, B^L_pq the
/// fitted pairs (pq|rs) = Sum_L B^L_pq B^L_rs over the orbitals C,
/// h_bar = dPhi/dh, Lambda^L = dPhi/dB^L, and X = 2 h h_bar +
/// 2 Sum_L B^L Lambda^L, so that dPhi = Sum_xp theta_xp X_xp along
/// C -> C (1 + theta). Every pointer is on the call's device, nmo = nao.
/// h_bar and Lambda are symmetric in (p, q), as Phi's are; X is read only
/// through sym(X).
struct MoAdjoint {
  /// @brief h_bar, nmo x nmo column-major.
  const double *h_bar = nullptr;
  /// @brief Lambda, naux x nmo^2: element (L, p + nmo q) at
  /// L + naux (p + nmo q), as @c b_mo leaves B.
  const double *lambda = nullptr;
  /// @brief X, nmo x nmo column-major: X_xp at x + nmo p.
  const double *x = nullptr;
};

/// @brief The nuclear gradient of the DF-CASSCF energy of root @p root at
/// @p solution, a converged solve of one state or a state average, every term
/// of it.
///
/// D = 2 C_c C_c^T + C_a gamma C_a^T and W = C sym(F) C^T, F the generalized
/// Fock of @c response::fock over @p b, against the one-electron derivatives
/// (@c gradient::one_electron_terms). The two-electron energy over the fit
/// is E_2 = 1/2 J[D_t].D_t - 1/2 J[D_a].D_a - Sum_i Sum_pq G_pq (ip|iq) +
/// 1/2 Sum Gamma_tuvw (tu|vw), D_t = D, G = diag(1_core, gamma) over the
/// occupied orbitals, i core; its derivative is one Gam_mnP against
/// @c three_center_gradient and one W_PQ against @c metric_gradient, every
/// piece of both built by gemms over the fitted pairs c_P,pq = (L^-T B_pq)_P
/// (@c b_mo, then a trsm). The README has each piece.
///
/// For a state average the terms are taken at root I's Lagrangian
/// (README.md): the Z-vector (response::zvector over the CI-relaxed Hessian at
/// the solution), then the same terms at gamma_I + gamma_T and Gamma_I +
/// Gamma_T (gamma_T z_c's transition RDMs) plus their derivative along
/// C -> C (1 + kappa(z_o)) at the averaged RDMs, all still in one Gam and
/// one W. The weights are @c sa_gradient_weights': the solution's own, or
/// recovered from its RDMs when it carries none; the active electrons are
/// trace(gamma).
///
/// @p multipliers, when given, are (z_o, z_c) in place of the Z-vector, for
/// one root as for a state average: one root with them is taken at the same
/// Lagrangian, at weight 1. They are in @c zvector's coordinates: z_o over
/// the packed pairs, z_c dim x nroots, root r's CI row
/// (H_0 - E_r) z_c,r + ... in PySCF's gen_g_hop convention, unweighted (the
/// call weights the transition RDMs by w_r), and theta the rotations among
/// the roots (ZVector::theta), or empty for none. @p extra, when given, adds
/// P = C h_bar C^T to the one-electron D, Gam = L^-T C Lambda C^T,
/// W_PQ += -1/2 (L^-T G^aux L^-1)_PQ with G^aux_LL' = Sum_pq Lambda^L_pq
/// B^L'_pq, and 1/2 C sym(X) C^T to the overlap's W, so the derivative
/// integrals are still evaluated once.
///
/// The formula is the gradient of the energy @ref solve converges to under
/// DensityFitting::Full over the same @p b, and only at its convergence: W
/// stands in for the orbitals' response, and the CI is variational. A
/// solution from another mode cannot be told apart, and its gradient is not
/// this.
///
/// @p molecule is the molecule both bases were placed on (by @c place), and
/// @p classes are @c primitive_pair_classes of @p basis; @p h is the core
/// Hamiltonian over @p basis. @p b is B over aux's naux = aux.nao_sphe()
/// functions as @c density_fit leaves it (naux x nao^2, element
/// (Q, n + nao m) at Q + naux (n + nao m)), and @p l its L, the naux x naux
/// lower triangle, column-major (the strict upper triangle is not read).
/// @p ncore and @p ncas are the solve's; @p root picks the state.
///
/// **Gam is held in full**, naux nao^2 doubles beside B, as in
/// @c gradient::df_rhf; with the fitted pairs and their back-transforms the
/// call's own device memory is about 8 naux (nao^2 + naux +
/// 2 ncore nocc + max(ncore, ncas) nao + 2 ncas^2) bytes, nocc =
/// ncore + ncas, and an extra adjoint's 8 naux (nao^2 + naux) more. Past the
/// memory budget (res.memory_budget()) it is Unsupported, naming the bytes,
/// before anything is allocated.
///
/// Queued on res.stream(); each derivative call synchronizes it, so the call
/// returns with everything done. Scratch for @c b_mo, the Fock and the
/// derivative calls comes from res.device_workspace(), so none of @p b and
/// @p l may point into it. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return The gradient and its terms; Unsupported for memory past the
///         memory budget, for naux nao^2 past int32_t, or for a state
///         average whose weights are recovered and whose roots' 1-RDMs are
///         linearly dependent (they cannot be); InvalidConfig if solution.c is not
///         basis.nao_sphe() square or not on @p res's device, if ncore is
///         negative, ncas below 1 or ncore + ncas past nao, if @p root is not
///         one of the solution's, if its RDMs are not ncas^2 and ncas^4 long,
///         if @p b is not naux nao^2 doubles over naux >= 1, if an @p extra
///         pointer is null, for a state average if root_gammas is not
///         ncas^2 per root or the weighted gamma is not their average, or,
///         for a state average or @p multipliers, if trace(gamma) is no
///         electron count of the molecule's multiplicity, the CI vectors are
///         not the roots', or the multipliers are not the packed pairs' and
///         dim x nroots long with no rotations or one per pair of roots;
///         the first error of @c casscf_integrals, @c fock, the CI objects,
///         the Hessian, @c zvector or a derivative call; or the GpuRuntime
///         error of the first copy or BLAS call that failed.
[[nodiscard]] Result<GradientTerms>
df_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const Solution &solution, int32_t ncore, int32_t ncas, int32_t root,
            const response::ZVector *multipliers = nullptr, const MoAdjoint *extra = nullptr);

/// @brief @ref df_gradient at a weighted mix of the solution's roots,
/// Sum_I v_I dE_I/dR, in place of one root: @p mix holds v_I, one per root,
/// each >= 0 and summing to 1, or of any sign and summing to 0 (not the
/// solution's own weights).
///
/// The terms are taken at the v-weighted RDMs Sum_I v_I (gamma_I, Gamma_I),
/// and the Z-vector's orbital right-hand side is the v-weighted root
/// gradient, so by linearity the result is Sum_I v_I times the per-root
/// gradients (to the Z-vector solves' convergence, as the mix's and each
/// root's are separate solves), still over one derivative-integral pass. @p multipliers and
/// @p extra compose with it as with a root. A one-hot @p mix is the root
/// call: the same operations, in the same order.
///
/// A mix summing to 0 weights the core's own energy and the nuclear
/// repulsion by 0: it is @ref df_transition_gradient at the mix's RDMs, its
/// Z-vector (when not given) solved over the transition orbital gradient
/// there. The mix's RDMs are two @c ci::Rdm::rdm12 calls, at its positive
/// and its negative part each scaled to sum 1.
///
/// @return @ref df_gradient's, with InvalidConfig in place of the root's
///         check if @p mix is not one weight per root, a weight is not
///         finite, they sum to neither 1 nor 0 within 1e-12, or one is
///         negative in a mix summing to 1.
[[nodiscard]] Result<GradientTerms>
df_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const Solution &solution, int32_t ncore, int32_t ncas, std::span<const double> mix,
            const response::ZVector *multipliers = nullptr, const MoAdjoint *extra = nullptr);

/// @brief The terms of @ref df_gradient at the caller's active-space RDMs
/// @p gamma and @p gamma2 with the core's own energy switched off: the
/// Hellmann-Feynman gradient of a coupling <I|H|J> between orthogonal roots
/// (PySCF's @c grad_elec_active), whose core-core part multiplies <I|J> = 0.
///
/// The energy differentiated is E(gamma, Gamma) less E_nuc and the core's
/// own E_cc = Sum_i 2 h_ii + Sum_ij [2 (ii|jj) - (ij|ij)]: Sum h D_a +
/// J[D_c].D_a - 1/2 K[D_c].D_a + 1/2 Sum Gamma_tuvw (tu|vw), D_a =
/// C_a gamma C_a^T, D_c = 2 C_c C_c^T. So the one-electron D is D_a, J is
/// the core-active cross term alone, K's G has no core block, and F is the
/// generalized Fock of @c response::fock at (gamma, Gamma) less 2 F^I in its
/// core columns (2 F^A there); @c nuclear_repulsion is zero. The orbital
/// gradient of the same energy, the right-hand side of a Z-vector, is
/// @c response::fock's gradient at (gamma, Gamma) less its gradient at zero
/// RDMs (README.md "A transition density").
///
/// @p gamma and @p gamma2 are row-major as @c ci::Rdm::rdm12's (ncas^2 and
/// ncas^4), with its symmetries: gamma symmetric and Gamma_tuvw = Gamma_vwtu
/// = Gamma_utwv, as @c ci::Rdm::trans_rdm12 leaves the symmetrized transition
/// RDMs. They are not checked for it; the formula reads them as such.
///
/// @p multipliers, when given, add the Lagrangian's terms on top exactly as
/// @ref df_gradient takes them for a root (the solution's weights, z_c's
/// transition RDMs added to @p gamma and @p gamma2, the orbital response at
/// the solution's weighted RDMs, core included, as the state-averaged
/// gradient's derivative); without them nothing responds and no Z-vector is
/// solved, at any number of roots. @p extra is @ref df_gradient's. Every other
/// argument, the memory and the stream are @ref df_gradient's.
///
/// @return The terms, nuclear repulsion zero; InvalidConfig if @p gamma or
///         @p gamma2 is not ncas^2 or ncas^4 long; otherwise @ref df_gradient's
///         errors, whose messages name mcscf::df_gradient (they share its body).
[[nodiscard]] Result<GradientTerms> df_transition_gradient(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const Solution &solution, int32_t ncore,
    int32_t ncas, std::span<const double> gamma, std::span<const double> gamma2,
    const response::ZVector *multipliers = nullptr, const MoAdjoint *extra = nullptr);

/// @brief Which of @ref df_gradient's and @ref df_transition_gradient's
/// calls a @ref df_gradient_footprint is of.
enum class GradientForm : uint8_t {
  /// @brief @ref df_gradient at a root, or at a mix of the roots summing to 1.
  Mix,
  /// @brief @ref df_gradient at a mix summing to 0 (the core's energy off).
  ZeroMix,
  /// @brief @ref df_transition_gradient at the caller's RDMs.
  Transition,
};

/// @brief What @p form's call holds over @p basis and @p aux, with @p ncore
/// core and @p ncas active orbitals holding @p nelecas electrons at
/// @p multiplicity, and a solution of @p nroots roots, all of it freed by the
/// call's return (the terms are on the host): scratch and the device
/// workspace.
///
/// In the call's order, each phase beside what is held before it: the fitted
/// integrals and the solution's Fock, kept; kappa; for a call that responds
/// (a state average, or @p multipliers given), the Lagrangian -- the CI
/// strings, the RDMs, and when no @p multipliers are given the sigma, the
/// Hessian, the CI-relaxed Hessian, the right-hand side's Fock (a mix
/// summing to 0: the transition orbital gradient), the sigma's apply scratch
/// grown to Layout::full and @c response::zvector's, then z_c and c +- z_c --
/// and its Fock (a transition's too), kept; [C | C~] and the five response
/// matrices; the orbital response; D, D_a, the one-electron D, -W (and a
/// transition's D_c) and the one-electron terms; Gam, W and the fitted
/// densities, an @p extra adjoint's half transform and G^aux; Y, G, and the
/// fitted core and active pairs with their V and Z (three of each when it
/// responds). The device workspace is the largest of every call's: the
/// integrals', the Focks', the Hessians' and their product's, the RDMs',
/// the two-factor df_exchange's, b_mo's and the two derivative
/// contractions'. @p rotating says the rotations among the roots are not
/// zero (unequal weights, or @p multipliers with a nonzero theta): z_c then
/// takes their nroots^2 matrix, and zvector its own. Empty where the call
/// refuses before allocating. B, L, C, the CI vectors and the multipliers
/// are the caller's.
[[nodiscard]] MemoryFootprint df_gradient_footprint(const BasisSet &basis, const BasisSet &aux,
                                                    int32_t ncore, int32_t ncas, int32_t nelecas,
                                                    int32_t multiplicity, int32_t nroots,
                                                    bool rotating,
                                                    GradientForm form = GradientForm::Mix,
                                                    bool multipliers = false, bool extra = false);

/// @brief The weights w_r a state-averaged @p solution is over:
/// @c Solution::weights as carried, when the weighted gamma is their sum of
/// @c Solution::root_gammas to 1e-10 (every solution @c solve returns).
/// Otherwise, a Solution carrying none or its gamma mixed at others, they
/// are recovered: the least-squares solution of Sum_r w_r gamma_r = gamma
/// with Sum_r w_r = 1 over the roots' and the weighted gamma (ncas^2 each,
/// @p ncas given), clamped at zero and renormalized. @ref df_gradient,
/// @c df_intersection and @c df_nac at unequal weights take theirs by it.
///
/// @return The weights, one per root; InvalidConfig when the solution
///         carries weights but not one per root; for a recovery, Unsupported
///         when the roots' 1-RDMs are linearly dependent (they do not
///         determine the weights), InvalidConfig when gamma is not their
///         weighted sum to 1e-10 or a weight is below -1e-10. The messages
///         name mcscf::df_gradient.
[[nodiscard]] Result<std::vector<double>> sa_gradient_weights(const Solution &solution,
                                                              size_t ncas);

/// @brief The multipliers on the conditions <c_s|H|c_r> = 0, r < s, that a
/// Lagrangian not invariant to rotating the roots among themselves carries
/// (README.md "Holding the roots apart"), each pair r < s in (r, s)
/// lexicographic order (ZVector::theta's), nroots (nroots - 1) / 2 of them.
struct RootPairMultipliers {
  /// @brief tau_sr = c_s . y_r - c_r . y_s, the Lagrangian's derivative
  /// along the rotation c_r gaining Theta_sr c_s, for every pair.
  std::vector<double> tau;
  /// @brief zeta_sr = -tau_sr / (E_s - E_r) for a pair of equal weights,
  /// zero for the rest: the closed-form multiplier zvector does not solve
  /// for.
  std::vector<double> zeta;
  /// @brief t_sr = tau_sr / (2 (w_r - w_s)) for a pair of unequal weights,
  /// zero for the rest: zvector's @p theta_rhs.
  std::vector<double> theta_rhs;
};

/// @brief tau from a caller's raw CI adjoint, split by @c rotating_pairs
/// (zvector's own weight test) into the equal-weight pairs' closed-form
/// zeta and the unequal pairs' rotation right-hand side t.
///
/// @p y_bar is dE/dc_r for every root, the CI adjoint before any projection
/// or weighting, on the device as dim x nroots column-major (zvector's
/// @p ci_rhs layout), dim the roots' vector length. @p weights are the
/// average's w_r, one per root (@ref sa_gradient_weights'). tau is one gemm
/// C^T y_bar, read back to the host. Queued on res.stream() with res.blas()
/// (host pointer mode), and waited for. One root has no pairs: three empty
/// vectors.
///
/// @p limit_pair, if set, names one equal-weight pair whose zeta is taken
/// at its limit, 0, whatever its gap (README.md "A pair at its limit"): it
/// is never refused under the floor, and its tau is still reported, but it
/// is refused when |tau| is above @c kLimitTauGate. The caller is what knows
/// the limit is 0 (nevpt2::qd_pc_intersection's mean gradient, #902).
///
/// @return The multipliers; InvalidConfig if @p roots holds no root or not
///         a whole number of vectors, @p weights is not one per root,
///         @p y_bar is null, or @p limit_pair is not two distinct roots of
///         equal weight; Numerical, naming the pair and its gap, when two
///         roots other than @p limit_pair are closer than @c kRootGapFloor
///         hartree (zeta divides by the gap, and zvector divides t by it), or
///         naming @p limit_pair and its tau above @c kLimitTauGate; or the
///         GpuRuntime error of the gemm or the copy.
[[nodiscard]] Result<RootPairMultipliers>
root_pair_multipliers(DeviceResources &res, const ci::Roots &roots, std::span<const double> weights,
                      const double *y_bar,
                      std::optional<std::array<int32_t, 2>> limit_pair = std::nullopt);

/// @brief root_pair_multipliers' gate on the tau of a pair whose zeta is
/// taken at its limit 0 (hartree): far above tau's round-off (8.9e-16,
/// README.md "What is bitwise") and below #892's PC tau_K. It cannot prove
/// the limit is 0, since tau / gap is 0 / 0 whatever the gate, but it
/// catches a W, a method or a pair the caller's argument does not cover.
constexpr double kLimitTauGate = 1e-11;

/// @brief What @ref root_pair_multipliers holds over @p nroots roots: the
/// nroots^2 overlaps, scratch. Empty for one root, which has no pairs.
[[nodiscard]] MemoryFootprint root_pair_multipliers_footprint(int32_t nroots);

/// @brief A zeta's two consumers, over its pairs' transition RDMs.
struct ZetaTerms {
  /// @brief zeta . g = Sum_{r<s} zeta_sr g^(sr), g^(sr)
  /// @c transition_orbital_gradient's, on the device over the packed pairs
  /// of @c response::pack_orbital_pairs: a Z-vector's orbital right-hand
  /// side.
  Buffers::device<double> orbital;
  /// @brief gamma^zeta = Sum_{r<s} zeta_sr gamma^(sr), ncas^2 row-major on
  /// the host.
  std::vector<double> gamma;
  /// @brief Gamma^zeta = Sum_{r<s} zeta_sr Gamma^(sr), ncas^4 row-major on
  /// the host. With @ref gamma, what one @ref df_transition_gradient takes
  /// for zeta's nuclear term.
  std::vector<double> gamma2;
};

/// @brief zeta . g and the folded transition RDMs (gamma^zeta, Gamma^zeta)
/// of @p zeta, one value per pair r < s in @ref RootPairMultipliers' order
/// (its zeta, or any other).
///
/// For each pair with zeta_sr nonzero, in that order: @p rdm's
/// @c trans_rdm12(c_s, c_r) of the solution's roots, read back; g^(sr) from
/// @c transition_orbital_gradient over @p ppaa and @p papa
/// (@c casscf_integrals' over solution.c, shared by every pair); then
/// zeta_sr g^(sr) added into the orbital part by an axpy, and zeta_sr times
/// the RDMs into the folded ones on the host, each from zero. A pair at
/// zero costs nothing, so the unequal pairs of a split are free. Bitwise
/// from call to call, as g^(sr) is.
///
/// @p rdm is over the roots' strings; @p molecule, @p h, @p b, @p naux,
/// @p ncore and @p ncas are @c transition_orbital_gradient's, solution.c its
/// orbitals. Queued on res.stream() with res.blas() (host pointer mode),
/// synchronized per pair. Beyond each call's own, ncas^2 + ncas^4 device
/// doubles for the RDMs.
///
/// @return The terms; InvalidConfig if @p zeta is not one per pair of the
///         solution's roots, ncore and ncas do not fit solution.c, or the
///         CI vectors are not nroots of @p rdm's size; or the first error of
///         @c trans_rdm12, @c transition_orbital_gradient, the axpy or a
///         copy.
[[nodiscard]] Result<ZetaTerms>
zeta_terms(DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
           const Buffers::device<double> &b, int32_t naux, const Solution &solution, int32_t ncore,
           int32_t ncas, const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
           ci::Rdm &rdm, std::span<const double> zeta);

/// @brief What @ref zeta_terms holds over @p nmo orbitals, @p naux fitting
/// functions, @p ncore core and @p ncas active, with CI vectors of @p dim
/// determinants (the Rdm's size()), when at least one zeta is nonzero: the
/// orbital part over the packed pairs, persistent; the RDMs' device copy and
/// one pair's @c transition_orbital_gradient at a time, scratch; the larger
/// of @c trans_rdm12's and the gradient's device workspace. A zeta of all
/// zeros allocates only the orbital part and the RDMs. Empty where the call
/// refuses.
[[nodiscard]] MemoryFootprint zeta_terms_footprint(int32_t nmo, int32_t naux, int32_t ncore,
                                                   int32_t ncas, int32_t dim);

} // namespace free_breakfast::mcscf
