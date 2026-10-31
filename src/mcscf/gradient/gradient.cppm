/// @file
/// @brief The analytic nuclear gradient of a converged DF-CASSCF on the GPU,
/// state-specific or of one root of a state average, term by term,
///
///     dE/dR = Sum D dT + Sum D dV - Sum W dS + dE_2/dR + dV_nn,
///
/// the two-electron part one Gam_mnP against d(mn|P) and one W_PQ against
/// d(P|Q). A state average's root is over its Z-vector's multipliers, or the
/// caller's, and a caller may add an MO adjoint of its own to the same
/// densities. See README.md for each term's derivation.

export module free_breakfast.mcscf.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.zvector;

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
/// (@c nuclear_gradient.tex Eqs. P-W): with Phi(h, B) that term, B^L_pq the
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
/// (README.md): the Z-vector (mcscf::zvector over the CI-relaxed Hessian at
/// the solution), then the same terms at gamma_I + gamma_T and Gamma_I +
/// Gamma_T (gamma_T z_c's transition RDMs) plus their derivative along
/// C -> C (1 + kappa(z_o)) at the averaged RDMs, all still in one Gam and
/// one W. The weights are recovered from @c Solution::root_gammas and the
/// weighted gamma, and the active electrons are trace(gamma).
///
/// @p multipliers, when given, are (z_o, z_c) in place of the Z-vector, for
/// one root as for a state average: one root with them is taken at the same
/// Lagrangian, at weight 1. They are in @c zvector's coordinates: z_o over
/// the packed pairs, z_c dim x nroots, root r's CI row
/// (H_0 - E_r) z_c,r + ... in PySCF's gen_g_hop convention, unweighted (the
/// call weights the transition RDMs by w_r). @p extra, when given, adds
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
/// device's free memory it is Unsupported, naming the bytes, before anything
/// is allocated.
///
/// Queued on res.stream(); each derivative call synchronizes it, so the call
/// returns with everything done. Scratch for @c b_mo, the Fock and the
/// derivative calls comes from res.device_workspace(), so none of @p b and
/// @p l may point into it. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return The gradient and its terms; Unsupported for memory past the
///         device's free bytes, for naux nao^2 past int32_t, or for a state
///         average whose roots' 1-RDMs are linearly dependent (its weights
///         cannot be recovered); InvalidConfig if solution.c is not
///         basis.nao_sphe() square or not on @p res's device, if ncore is
///         negative, ncas below 1 or ncore + ncas past nao, if @p root is not
///         one of the solution's, if its RDMs are not ncas^2 and ncas^4 long,
///         if @p b is not naux nao^2 doubles over naux >= 1, if an @p extra
///         pointer is null, for a state average if root_gammas is not
///         ncas^2 per root or the weighted gamma is not their average, or,
///         for a state average or @p multipliers, if trace(gamma) is no
///         electron count of the molecule's multiplicity, the CI vectors are
///         not the roots', or the multipliers are not the packed pairs' and
///         dim x nroots long;
///         the first error of @c casscf_integrals, @c fock, the CI objects,
///         the Hessian, @c zvector or a derivative call; or the GpuRuntime
///         error of the first copy or BLAS call that failed.
[[nodiscard]] Result<GradientTerms>
df_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const Solution &solution, int32_t ncore, int32_t ncas, int32_t root,
            const ZVector *multipliers = nullptr, const MoAdjoint *extra = nullptr);

} // namespace free_breakfast::mcscf
