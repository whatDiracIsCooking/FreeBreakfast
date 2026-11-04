/// @file
/// @brief What a QD-NEVPT2 minimum-energy conical intersection search needs
/// of one QD pair (P, Q), SC or PC, on a DF-SA-CASSCF: the two states'
/// gradients g_P and g_Q and their interstate coupling h_PQ, all at the one
/// set of H_eff eigenvectors R, with no refusal of a degenerate pair; and the
/// pair's gap sensitivities, which bound how far the gap moves for an
/// unconverged SA-CASSCF.
///
/// Built on nevpt2::qd_sc_gradient's and qd_pc_gradient's W forms: H_eff and
/// R once, then G(W) at u_P u_P^T, u_Q u_Q^T and
/// W^PQ = 1/2 (u_P u_Q^T + u_Q u_P^T), by one several-W call (the default)
/// or three W-form calls. The README has the derivation of the
/// sensitivities, their pairing with the CASSCF's residuals, and the timing
/// that sets the default.

export module free_breakfast.nevpt2.intersection;

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

export namespace free_breakfast::nevpt2 {

/// @brief How qd_sc_intersection and qd_pc_intersection reach their result.
struct QdIntersectionOptions {
  /// @brief One several-W qd_*_gradient call over the pair's three W's (one
  /// set of the adjoint's forward arrays, one CI-relaxed Hessian, one
  /// mcscf::zvectors solve of the three columns), rather than three W-form
  /// calls. The derivative passes stay three. README.md "Sharing the three
  /// calls" has the measurement that sets the default: shared was faster on
  /// every case measured.
  bool shared = true;
  /// @brief Each W's Z-vector tolerance: qd_*_gradient's
  /// @c zvector_tolerance, to which each solve's true residual is held (one
  /// parameter for the solve and its check). The CI response inside H~ is
  /// converged to mcscf::response_tolerance_for it
  /// (../gradient/README.md "The Z-vector's floor").
  double zvector_tolerance = 1e-10;
};

/// @brief qd_sc_intersection's (or qd_pc_intersection's) result: the three
/// vectors of a QD pair's branching space at one R, and its gap's
/// sensitivities to the SA-CASSCF's residuals.
struct QdIntersection {
  /// @brief dE_P/dR = G(u_P u_P^T), n_atom x 3 row-major (hartree / bohr).
  std::vector<double> g_p;
  /// @brief dE_Q/dR = G(u_Q u_Q^T), n_atom x 3 row-major.
  std::vector<double> g_q;
  /// @brief h_PQ = u_P^T (dH_eff/dR) u_Q = G(W^PQ), n_atom x 3 row-major.
  std::vector<double> h_pq;
  /// @brief H_eff's eigenvalues, ascending (hartree): the QD energies.
  std::vector<double> eigenvalues;
  /// @brief H_eff's eigenvectors R, n x n, state K's column at
  /// [K n, (K + 1) n), each signed so that its largest-magnitude component is
  /// positive (QdScLagrangian::eigenvectors'): the u's every W was built
  /// from.
  std::vector<double> eigenvectors;
  /// @brief |E_Q - E_P|, reported and never gated: at a degeneracy u_P and
  /// u_Q are any orthonormal pair of the two-state space, and the branching
  /// plane span{g_Q - g_P, h_PQ} the same for each.
  double heff_gap = 0.0;
  /// @brief The smallest |E_r - E_s| over the CASSCF roots (hartree).
  double root_gap = std::numeric_limits<double>::infinity();
  /// @brief s_o = sqrt(||z_o^Q - z_o^P||^2 + 4 ||z_o^PQ||^2), z_o^W the
  /// orbital multipliers of G(W): the gap moves by at most s_o ||g|| for an
  /// orbital gradient g, mcscf::Solution::gradient_norm's vector (factor 1).
  double gap_sensitivity_orbital = 0.0;
  /// @brief s_c, s_o's formula over the CI multipliers with each root's
  /// block scaled by 2 w_r (z^c_r 2 w_r): the gap moves by at most
  /// s_c sqrt(n_roots) rho for Davidson residuals of norm at most rho
  /// (ci::Roots::davidson.max_residual_norm).
  double gap_sensitivity_ci = 0.0;
};

/// @brief The SC-QD-NEVPT2 branching space of QD states @p p and @p q
/// (H_eff's eigenvalues ascending) on the DF-SA-CASSCF @p solution, and the
/// pair's gap sensitivities.
///
/// H_eff = diag(E_r) + 1/2 (C + C^T) over nevpt2::qd_inputs and
/// sc_effective_hamiltonian, diagonalized once; then nevpt2::qd_sc_gradient
/// at u_P u_P^T, u_Q u_Q^T and W^PQ, one several-W call or (with
/// @p options.shared false) three W-form calls, each Z-vector's true
/// residual (ZVector::residual, one more product at the solution) held to
/// @p options.zvector_tolerance, and s_o and s_c from their multipliers. Unlike
/// qd_sc_coupling there is no refusal of a degenerate QD pair: nothing here
/// divides by E_Q - E_P. Unequal SA weights are
/// allowed; the only division by a gap is root_pair_multipliers' by the
/// CASSCF roots', which keeps its refusal.
///
/// The arguments are qd_sc_coupling's, without its fold, and @p options.
///
/// @return The branching space and sensitivities; InvalidConfig for p = q,
///         a state not one of the roots' or a tolerance not positive;
///         Numerical, naming the W, when a Z-vector's true residual is above
///         the tolerance; or the first error of
///         qd_inputs, sc_effective_hamiltonian, the diagonalization or a
///         W-form call (a root of zero weight, the gate, the Z-vector, a
///         derivative pass).
[[nodiscard]] Result<QdIntersection>
qd_sc_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t p,
                   int32_t q, int32_t batch, const QdIntersectionOptions &options = {});

/// @brief The PC-QD-NEVPT2 branching space and gap sensitivities:
/// qd_sc_intersection over pc_effective_hamiltonian and qd_pc_gradient's W
/// form. The arguments are qd_sc_intersection's.
///
/// @return As qd_sc_intersection; a refused PC class's Numerical error,
///         pc_effective_hamiltonian's unchanged (qd_pc_coupling's), never a
///         result with the class zeroed.
[[nodiscard]] Result<QdIntersection>
qd_pc_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t p,
                   int32_t q, int32_t batch, const QdIntersectionOptions &options = {});

} // namespace free_breakfast::nevpt2
