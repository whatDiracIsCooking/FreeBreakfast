/// @file
/// @brief The SC-NEVPT2 correlation energy of one root as a function of the
/// MO integrals (h, B) and the CI vector c, and its adjoint: h_bar = dE/dh,
/// Lambda = dE/dB, y = dE/dc, and the orbital source term X = 2 h h_bar +
/// 2 sum_L B^L Lambda^L with A = X - X^T.
///
/// nuclear_gradient.tex section 5: the per-root input builders (the inactive
/// Fock's blocks, the orbital energies from (h, B, gamma), h2e, (tu|vw), and
/// the RDMs) run forward and then backwards through nevpt2.energy's,
/// nevpt2.rdm's and nevpt2.density_fitting's adjoints. Then the
/// inactive-rotation multipliers z^c of the Lagrangian E + <z^c, f> from any
/// such A, and the term Phi = <z^c, f> they add with its adjoints. Then the
/// SC-NEVPT2 nuclear gradient of a single-state DF-CASSCF over them: the
/// Z-vector, the Lagrangian's symmetry gate, and one mcscf::df_gradient pass.
/// The PC-NEVPT2 energy, adjoint and gradient are the same code with the PC
/// class energies (and their reverse pass) in place of SC's. The SC-QD-NEVPT2
/// effective Hamiltonian of several roots over shared orbitals is the same
/// builders around sc_effective_hamiltonian, and its adjoint along any W;
/// the PC-QD one the same around pc_effective_hamiltonian. And the SC-QD
/// Lagrangian's multipliers on a DF-SA-CASSCF (qd_sc_lagrangian): z^c, the
/// roots' rotation multipliers and the SA Z-vector, gated, at a state's
/// W = u_K u_K^T or at any symmetric W; the SC-QD
/// nuclear gradient over it (qd_sc_gradient), two derivative passes; and the
/// PC-QD one (qd_pc_gradient), the same over qd_pc_adjoint.
/// The README has the layouts and every term.

export module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.zvector;
import free_breakfast.mcscf.gradient;
import free_breakfast.nevpt2.energy;

export namespace free_breakfast::nevpt2 {

/// @brief The SC-NEVPT2 class energies of the root @p vector over the
/// orbitals @p c, taken as they are: every input class_energies reads is built
/// from (h, B, c) here, the orbital energies as the diagonal of
/// f = F^I + F^A(gamma), and nothing is canonicalized.
///
/// @p b is B as density_fit leaves it, naux x nao^2 ([Q][n][m], symmetric in
/// m and n); @p c is nao x @p nmo, column-major, columns [0, ncore) the core,
/// then @p ncas active, then the virtuals; @p h is the AO core Hamiltonian,
/// nao x nao. The vector is a unit vector over @p alpha and @p beta, alpha
/// major, as nevpt2::rdms takes it; its active electrons are the two sets'.
/// dm1 and dm2 are traces of nevpt2::rdms' dm3 (make_dm123's convention).
/// Each class walks its slabs in batches of at most @p batch, as in
/// class_energies. nevpt2::rdms gets half the device's free memory at entry.
///
/// @return The class energies; InvalidConfig for counts that do not fit
///         (ncas not the strings' orbitals, ncore + ncas past nmo); or the
///         error of a builder, nevpt2::rdms or class_energies.
[[nodiscard]] Result<ClassEnergies>
sc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
          int32_t naux, int32_t nao, int32_t nmo, const double *b, const double *c, const double *h,
          int32_t ncore, int32_t ncas, const double *vector, int32_t batch);

/// @brief sc_adjoint's result, every array on res's device.
struct ScAdjoint {
  /// @brief The class energies, as sc_energy computes them.
  ClassEnergies energies;
  /// @brief h_bar = dE/dh in the MO basis of c, nmo x nmo column-major,
  /// symmetric: dE = <h_bar, C^T dh C>.
  Buffers::device<double> h_bar;
  /// @brief Lambda = dE/dB^L in the MO basis, naux x nmo^2, element
  /// (L, p, q) at L + naux (p + nmo q) as b_mo leaves B, symmetric in (p, q).
  Buffers::device<double> lambda;
  /// @brief y = dE/dc, one per determinant, at fixed (h, B): the raw
  /// derivative of E as a function of an unnormalized c.
  Buffers::device<double> y;
  /// @brief X = 2 h h_bar + 2 sum_L B^L Lambda^L, nmo x nmo column-major,
  /// X_xp at x + nmo p: dE = sum_xp theta_xp X_xp along C -> C (1 + theta).
  Buffers::device<double> x;
  /// @brief A = X - X^T: dE = sum_{p > q} kappa_pq A_pq along
  /// C -> C exp(kappa), kappa antisymmetric.
  Buffers::device<double> a;
  /// @brief The orbital energies class_energies read, f's diagonal at each
  /// orbital's own index, nmo long: core and virtual set, the active entries
  /// 0. What zc_multipliers divides by.
  Buffers::device<double> epsilon;
};

/// @brief The adjoint of sc_energy at (@p h, @p b, @p c, @p vector): the
/// arguments, the forward and its batch are sc_energy's.
///
/// One reverse pass. class_energies_adjoint takes E back to every
/// EnergyInputs field and, through Lambda, to the slabs; the orbital
/// energies' and the Fock blocks' bars go back through their builders onto
/// h, B and gamma (B_cc read off its diagonal included); gamma's and dm2's
/// bars fold into dm3's by the electron-count trace; nevpt2::rdms_adjoint
/// takes dm3's, f3ca's and f3ac's onto y and (tu|vw); and the bars on h2e,
/// (tu|vw) and the MO blocks go onto Lambda through nevpt2.density_fitting's
/// adjoints. h_bar and Lambda are symmetrized last, and X is built from them
/// as an AO half-transform: 2 C^T B_AO^L (C Lambda^L), one strided-batched
/// gemm and two gemms, plus 2 h h_bar.
///
/// @return The energies and the adjoints; sc_energy's errors, or those of
///         class_energies_adjoint, rdms_adjoint and the density-fitted
///         adjoints.
[[nodiscard]] Result<ScAdjoint> sc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                           const ci::DeviceStringSet &beta, int32_t naux,
                                           int32_t nao, int32_t nmo, const double *b,
                                           const double *c, const double *h, int32_t ncore,
                                           int32_t ncas, const double *vector, int32_t batch);

/// @brief The PC-NEVPT2 class energies of the root @p vector over the
/// orbitals @p c: sc_energy's inputs, built the same way from the same
/// arguments, through pc_class_energies instead of class_energies.
///
/// @return The class energies; sc_energy's InvalidConfig for counts that do
///         not fit; pc_class_energies' Numerical error naming a refused
///         class; or the error of a builder or nevpt2::rdms.
[[nodiscard]] Result<PcClassEnergies>
pc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
          int32_t naux, int32_t nao, int32_t nmo, const double *b, const double *c, const double *h,
          int32_t ncore, int32_t ncas, const double *vector, int32_t batch);

/// @brief pc_adjoint's result: ScAdjoint's arrays, in its conventions, of
/// the PC correlation energy.
struct PcAdjoint {
  /// @brief The class energies, as pc_energy computes them.
  PcClassEnergies energies;
  /// @brief dE/dh in the MO basis, as ScAdjoint::h_bar.
  Buffers::device<double> h_bar;
  /// @brief dE/dB^L in the MO basis, as ScAdjoint::lambda.
  Buffers::device<double> lambda;
  /// @brief dE/dc at fixed (h, B), raw, as ScAdjoint::y.
  Buffers::device<double> y;
  /// @brief The orbital source term X, as ScAdjoint::x.
  Buffers::device<double> x;
  /// @brief A = X - X^T, as ScAdjoint::a. PC is invariant to an inactive
  /// rotation only between degenerate orbital energies, so its core-core and
  /// virtual-virtual blocks are not zero in general: zc_multipliers takes
  /// them as it takes SC's.
  Buffers::device<double> a;
  /// @brief The orbital energies pc_class_energies read, as
  /// ScAdjoint::epsilon.
  Buffers::device<double> epsilon;
};

/// @brief The adjoint of pc_energy at (@p h, @p b, @p c, @p vector): the
/// arguments are pc_energy's. sc_adjoint's reverse pass with
/// pc_class_energies_adjoint in place of class_energies_adjoint: the
/// builders, rdms_adjoint, the density-fitted adjoints, the symmetrization
/// and the source term are the same code.
///
/// @return The energies and the adjoints; a refused class's Numerical error,
///         the energy's own (pc_class_energies_adjoint's guarantee), never a
///         zero; or sc_adjoint's other errors.
[[nodiscard]] Result<PcAdjoint> pc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                           const ci::DeviceStringSet &beta, int32_t naux,
                                           int32_t nao, int32_t nmo, const double *b,
                                           const double *c, const double *h, int32_t ncore,
                                           int32_t ncas, const double *vector, int32_t batch);

/// @brief The SC-QD-NEVPT2 effective Hamiltonian's correlation elements of
/// the roots @p vectors over the shared orbitals @p c, every input built
/// from (h, B, c) and the roots: element [M n + N] is
/// sc_effective_hamiltonian's (M the bra, N the ket), its total
/// <M|V|Psi_N^(1)>.
///
/// The arguments but the roots are sc_energy's. @p vectors holds
/// weights.size() unit vectors laid out as ci::Roots::vectors (root r's at
/// [r ndet, (r + 1) ndet), ndet = alpha.size() beta.size()). One set of MO
/// blocks, F^I and h2e serves every pair; the orbital energies are the
/// diagonal of the state-averaged Fock, F^I + F^A(gamma_SA), gamma_SA =
/// sum_r w_r gamma_r, rebuilt from the roots (nothing canonicalized); each
/// ordered pair's dm3 is nevpt2::rdms' (the bra/ket overload off the
/// diagonal, the one-vector one with f3ca and f3ac on it), and its dm2 and
/// dm1 dm3's traces, as sc_energy's are. At one root, weight 1, the element
/// is sc_energy's.
///
/// @return The n^2 elements; sc_energy's InvalidConfig for counts that do
///         not fit, or zc_adjoint's for weights it refuses; or the error of
///         a builder, nevpt2::rdms or sc_effective_hamiltonian.
[[nodiscard]] Result<std::vector<ClassEnergies>>
qd_sc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
             const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
             const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
             std::span<const double> weights, const double *vectors, int32_t batch);

/// @brief qd_sc_adjoint's result, every array on res's device, each in
/// ScAdjoint's convention.
struct QdScAdjoint {
  /// @brief The n^2 elements, as qd_sc_energy computes them.
  std::vector<ClassEnergies> elements;
  /// @brief dE/dh in the MO basis, nmo x nmo, symmetric, as ScAdjoint::h_bar.
  Buffers::device<double> h_bar;
  /// @brief dE/dB^L in the MO basis, naux x nmo^2, as ScAdjoint::lambda.
  Buffers::device<double> lambda;
  /// @brief y_r = dE/dc_r at fixed (h, B) for every root r, root r's at
  /// [r ndet, (r + 1) ndet) as the vectors are laid out: raw, the derivative
  /// of E as a function of unnormalized c_r, not projected.
  Buffers::device<double> y;
  /// @brief The orbital source term X, as ScAdjoint::x.
  Buffers::device<double> x;
  /// @brief A = X - X^T, as ScAdjoint::a.
  Buffers::device<double> a;
  /// @brief The state-averaged Fock's core and virtual diagonals at their own
  /// indices, nmo long, the active entries 0: what the elements read, and
  /// what zc_multipliers divides by.
  Buffers::device<double> epsilon;
};

/// @brief The adjoint of E[W] = sum_MN W_MN H_MN, H_MN qd_sc_energy's
/// element (M, N) total, at (@p h, @p b, @p c), the roots @p vectors at
/// @p weights, and fixed @p w (W, n x n row-major, [M n + N] on element
/// (M, N)): the arguments but w are qd_sc_energy's.
///
/// W is any n x n matrix. The QD-NEVPT2 gradient's is u_K u_K^T, u_K state
/// K's eigenvector of the symmetrized H_eff, so that W's symmetric part on
/// the unsymmetrized elements is the eigenvalue's derivative.
///
/// One reverse pass. sc_effective_hamiltonian_adjoint takes E to every
/// shared input, to each root's own RDMs and, pair at a time, to each
/// transition's dm1, dm2 and dm3, whose bars fold into dm3's by the traces'
/// adjoint and go through the bra/ket nevpt2::rdms_adjoint (its f3 bars
/// null) onto y of the bra and the ket before the next pair. Then
/// sc_adjoint's builders backwards; gamma_SA's bar goes onto each root's own
/// dm1 bar at its weight, so every root, its W_rr 0 or not, takes the
/// orbital energies' share; each root's own bars then fold and go through
/// the one-vector rdms_adjoint onto its y and (tu|vw)'s bar; and h_bar,
/// Lambda, X and A are formed as sc_adjoint forms them. At one root, weight
/// 1 and W = 1, the result is sc_adjoint's to round-off.
///
/// @return The elements and the adjoints; InvalidConfig for @p w not
///         weights.size()^2 long, or qd_sc_energy's errors; or those of
///         sc_effective_hamiltonian_adjoint, rdms_adjoint and the
///         density-fitted adjoints.
[[nodiscard]] Result<QdScAdjoint>
qd_sc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
              const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
              const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
              std::span<const double> weights, const double *vectors, std::span<const double> w,
              int32_t batch);

/// @brief The PC-QD-NEVPT2 effective Hamiltonian's correlation elements of
/// the roots @p vectors over the shared orbitals @p c: qd_sc_energy's inputs,
/// built the same way from the same arguments, through
/// pc_effective_hamiltonian instead of sc_effective_hamiltonian. Element
/// [M n + N] is ⟨M|V|Ψ_N⁽¹⁾⟩ per class, ket N's own solve; at one root,
/// weight 1, it is pc_energy's.
///
/// @return The n^2 elements; qd_sc_energy's InvalidConfig; a refused class's
///         Numerical error naming the class and the state, as
///         pc_effective_hamiltonian refuses it; or the error of a builder or
///         nevpt2::rdms.
[[nodiscard]] Result<std::vector<PcClassEnergies>>
qd_pc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
             const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
             const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
             std::span<const double> weights, const double *vectors, int32_t batch);

/// @brief qd_pc_adjoint's result: QdScAdjoint's arrays, in its conventions,
/// of the PC-QD functional.
struct QdPcAdjoint {
  /// @brief The n^2 elements, as qd_pc_energy computes them.
  std::vector<PcClassEnergies> elements;
  /// @brief dE/dh in the MO basis, as QdScAdjoint::h_bar.
  Buffers::device<double> h_bar;
  /// @brief dE/dB^L in the MO basis, as QdScAdjoint::lambda.
  Buffers::device<double> lambda;
  /// @brief y_r = dE/dc_r per root, raw, as QdScAdjoint::y.
  Buffers::device<double> y;
  /// @brief The orbital source term X, as QdScAdjoint::x.
  Buffers::device<double> x;
  /// @brief A = X - X^T, as QdScAdjoint::a; on the inactive blocks it is
  /// PcAdjoint::a's kind, zero on a degenerate pair only.
  Buffers::device<double> a;
  /// @brief The state-averaged Fock's core and virtual diagonals, as
  /// QdScAdjoint::epsilon.
  Buffers::device<double> epsilon;
};

/// @brief The adjoint of E[W] = sum_MN W_MN H_MN, H_MN qd_pc_energy's
/// element (M, N) total, at fixed @p w: qd_sc_adjoint's arguments and
/// reverse pass with pc_effective_hamiltonian_adjoint at H̄ = W in place of
/// sc_effective_hamiltonian_adjoint, as pc_adjoint is sc_adjoint's. The
/// transition pairs' bars, handed over after the walk in row-major pair
/// order, go through the bra/ket rdms_adjoint as SC-QD's do; the shared
/// builders, gamma_SA's bar onto every root and h_bar, Lambda, X, A and ε are
/// the same code. At one root, weight 1 and W = 1, the result is
/// pc_adjoint's to round-off.
///
/// PC's finish drops the restricted inverse's rank-projector terms, exact
/// when each b_{M←N} lies in S_N's kept span, which holds at eigenstates
/// (S_N's dropped modes annihilate |N⟩); the ladders along each c_r check it.
///
/// @return The elements and the adjoints; InvalidConfig as qd_sc_adjoint's;
///         a refused class's Numerical error, qd_pc_energy's, never a zero;
///         or the errors of rdms_adjoint and the density-fitted adjoints.
[[nodiscard]] Result<QdPcAdjoint>
qd_pc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
              const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
              const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
              std::span<const double> weights, const double *vectors, std::span<const double> w,
              int32_t batch);

/// @brief zc_multipliers' result.
struct ZcMultipliers {
  /// @brief z^c, nmo x nmo column-major and symmetric, on res's device:
  /// z_pq = -A_pq / (2 (eps_p - eps_q)) for p != q both core or both
  /// virtual, 0 everywhere else, the diagonal included, and on a degenerate
  /// pair whose A is round-off.
  Buffers::device<double> z;
  /// @brief max |z_pq|.
  double max_z = 0.0;
  /// @brief The smallest |eps_p - eps_q| over the core-core and
  /// virtual-virtual pairs p != q, degenerate ones included; infinity when
  /// neither space has two orbitals.
  double min_gap = std::numeric_limits<double>::infinity();
};

/// @brief The inactive-rotation multipliers z^c of a correlation energy
/// whose orbital source term is @p a (A = X - X^T, nmo x nmo column-major on
/// the device, dE = sum_{p > q} kappa_pq A_pq along C exp(kappa)), over the
/// orbital energies @p epsilon (nmo on the device, the core's and virtuals'
/// read). Park Eqs. 56-57 in this tree's A: the Lagrangian E + <z, f>, f the
/// generalized Fock zc_adjoint builds, is stationary along every core-core
/// and virtual-virtual rotation when f is diagonal there, the pseudocanonical
/// condition, so each pair is one division. Nothing in it is SC's: any A
/// over the same orbitals takes it.
///
/// A pair closer than 1e-6 Eh is not determined by the state: its z is 0
/// when |A_pq| <= 1e-8, the round-off an abelian subgroup leaves on a
/// degenerate pair split across irreps, and otherwise refused.
///
/// @return The multipliers; InvalidConfig for counts that do not fit; a
///         Numerical error naming the pair, its gap and |A| for a degenerate
///         pair with A past round-off; or a copy's GpuRuntime error.
[[nodiscard]] Result<ZcMultipliers> zc_multipliers(DeviceResources &res, int32_t nmo, int32_t ncore,
                                                   int32_t ncas, const double *a,
                                                   const double *epsilon);

/// @brief zc_adjoint's result, every array on res's device. Each adjoint is
/// in sc_adjoint's convention, so the caller adds them field by field.
struct ZcAdjoint {
  /// @brief Phi = <z, f> = sum_pq z_pq f_pq, f the generalized Fock of the
  /// weighted density.
  double phi = 0.0;
  /// @brief dPhi/dh in the MO basis, nmo x nmo: z itself.
  Buffers::device<double> h_bar;
  /// @brief dPhi/dB^L in the MO basis, naux x nmo^2 as ScAdjoint::lambda,
  /// symmetric in (p, q).
  Buffers::device<double> lambda;
  /// @brief y^c_r = dPhi/dc_r at fixed (h, B) for every root r, one per
  /// determinant each, root r's at [r ndet, (r + 1) ndet) as the vectors are
  /// laid out: the raw derivative of Phi as a function of an unnormalized
  /// c_r, as ScAdjoint::y, its root's weight included.
  Buffers::device<double> y;
  /// @brief X = 2 h h_bar + 2 sum_L B^L Lambda^L, nmo x nmo, as
  /// ScAdjoint::x.
  Buffers::device<double> x;
  /// @brief A = X - X^T: dPhi = sum_{p > q} kappa_pq A_pq along
  /// C -> C exp(kappa).
  Buffers::device<double> a;
};

/// @brief Phi = <@p z, f(gamma_SA)> at (@p h, @p b, @p c) and the roots
/// @p vectors at @p weights, and its adjoints at fixed @p z (nmo x nmo
/// column-major on the device, symmetric): the arguments but z, the weights
/// and the vectors are sc_energy's.
///
/// f_pq = h_pq + sum_rs D_rs [(pq|rs) - 1/2 (pr|sq)], D the reference's MO
/// density (2 on the core's diagonal, gamma_SA = sum_r w_r gamma_r on the
/// active block): the state-averaged Fock, over which a state average's core
/// and virtuals are canonical, and at one root (weight 1) the root's own, so
/// its core and virtual diagonals are sc_adjoint's epsilon. B goes to the MO
/// basis whole, naux nmo^2; each gamma_r is ci::Sigma's rdm1 of its root, and
/// y^c one batched sigma over the one-body (W + W^T)_tu, W_tu =
/// dPhi/dgamma_SA,tu, each root's column scaled by its weight.
///
/// @p vectors holds weights.size() unit vectors over @p alpha and @p beta,
/// root r's at [r ndet, (r + 1) ndet), ndet = alpha.size() beta.size(), as
/// ci::Roots::vectors lays them out. One root at weight 1 is the
/// single-state Phi, through the same operations.
///
/// @return Phi and its adjoints; InvalidConfig for counts that do not fit
///         (ncas not the strings' orbitals, ncore + ncas past nmo), or for
///         no weights, a negative (or NaN) weight, or weights not summing to
///         1 within 1e-12; or the error of b_mo, ci::Sigma, an einsum or a
///         BLAS call.
[[nodiscard]] Result<ZcAdjoint> zc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                           const ci::DeviceStringSet &beta, int32_t naux,
                                           int32_t nao, int32_t nmo, const double *b,
                                           const double *c, const double *h, int32_t ncore,
                                           int32_t ncas, std::span<const double> weights,
                                           const double *vectors, const double *z);

/// @brief lagrangian_gradient's and sc_gradient's result.
struct ScGradient {
  /// @brief dE/dR of E_cas + E_corr, every term as mcscf::df_gradient
  /// returns them (n_atom x 3, row-major).
  mcscf::GradientTerms terms;
  /// @brief The class energies of E_corr, as sc_energy computes them.
  ClassEnergies energies;
  /// @brief z^c, with max |z^c| and the smallest inactive gap.
  ZcMultipliers zc;
  /// @brief (Z, z) in mcscf::zvector's coordinates: Z over the packed pairs,
  /// z the CI part, off the root; its PCG iterations and residual.
  mcscf::ZVector multipliers;
  /// @brief max |X^L_pq - X^L_qp| over p > q per block of the Lagrangian's
  /// orbital source term: core-core, active-active, virtual-virtual,
  /// active-core, virtual-core, virtual-active.
  std::array<double, 6> asymmetry{};
};

/// @brief The nuclear gradient of the Lagrangian of E_cas + E_corr at the
/// single-state DF-CASSCF @p solution, for a correlation energy whose
/// adjoint is @p corr, inactive multipliers @p zc and their term's adjoint
/// @p phi (zc_adjoint's at zc.z): nuclear_gradient.tex Algorithm 1 from the
/// Z-vector on.
///
/// @p corr is consumed: @p phi's h_bar, Lambda, y, X and A are added onto it
/// in place. The Z-vector is mcscf::zvector over the CI-relaxed Hessian at
/// the solution, its orbital right-hand side g + A (g the CASSCF's orbital
/// gradient, A = A^corr + A^{z^c} over the packed pairs) and its CI one
/// 1/2 (y + y^c), the tex's J_cc = 2 P (H - E) P put in zvector's
/// convention (zvector projects it off the root). The Lagrangian's orbital
/// source term must then be symmetric in every block: its antisymmetric part
/// A^L = A^cas(gamma + gamma_T) + A^Z + A is formed from the generalized Fock
/// at the root's RDMs plus z's transition RDMs, and from that Fock's
/// derivative along Z, and a block past 1e-8 is refused. Then one
/// mcscf::df_gradient with (Z, z) as its multipliers and (h_bar, Lambda, X)
/// as its extra adjoint.
///
/// The arguments but these are mcscf::df_gradient's; the solution's
/// orbitals are the ones @p corr and @p phi were taken at (columns core,
/// active, virtual, the core and virtuals canonical), and its root is c.
///
/// @return The gradient, the multipliers and the gate's residues;
///         Unsupported for a state average (more than one root);
///         InvalidConfig for counts that do not fit or adjoints of the wrong
///         length; Numerical, naming the block and its residue, when the
///         gate fails; or the first error of casscf_integrals, the Fock, the
///         CI objects, the Hessian, mcscf::zvector or mcscf::df_gradient.
[[nodiscard]] Result<ScGradient>
lagrangian_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, ScAdjoint corr,
                    ZcMultipliers zc, const ZcAdjoint &phi);

/// @brief The SC-NEVPT2 nuclear gradient of the single-state DF-CASSCF
/// @p solution: sc_adjoint at its orbitals and root, zc_multipliers and
/// zc_adjoint over its A and epsilon, then lagrangian_gradient. The
/// arguments are mcscf::df_gradient's and @p batch sc_energy's; @p b and
/// @p h are the AO B and core Hamiltonian both calls read.
///
/// @return The gradient; Unsupported for a state average; or the first
///         error of the calls it makes.
[[nodiscard]] Result<ScGradient>
sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t batch);

/// @brief pc_gradient's result: ScGradient's fields, with PC's class
/// energies.
struct PcGradient {
  /// @brief dE/dR of E_cas + E_corr^PC, as ScGradient::terms.
  mcscf::GradientTerms terms;
  /// @brief The class energies of E_corr^PC, as pc_energy computes them.
  PcClassEnergies energies;
  /// @brief z^c of PC's A and epsilon, as ScGradient::zc.
  ZcMultipliers zc;
  /// @brief (Z, z), as ScGradient::multipliers.
  mcscf::ZVector multipliers;
  /// @brief The gate's residues per block, as ScGradient::asymmetry.
  std::array<double, 6> asymmetry{};
};

/// @brief The PC-NEVPT2 nuclear gradient of the single-state DF-CASSCF
/// @p solution: sc_gradient with pc_adjoint in place of sc_adjoint.
/// zc_multipliers and zc_adjoint over PC's A and epsilon, under the same
/// degenerate-pair rule, then lagrangian_gradient. The arguments are
/// sc_gradient's.
///
/// @return The gradient; Unsupported for a state average; the energy's
///         Numerical error when PC refuses a class (no gap at the cut, a
///         non-positive denominator); or the first error of the calls it
///         makes.
[[nodiscard]] Result<PcGradient>
pc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t batch);

/// @brief qd_sc_lagrangian's result: what the SC-QD-NEVPT2 gradient's
/// derivative passes take, and what it reports. Every device array is on
/// res's device.
struct QdScLagrangian {
  /// @brief H_eff's eigenvalues, ascending: E_r delta_rs + 1/2 (C_rs + C_sr),
  /// E_r the solution's root energies, C the correlation elements. Empty
  /// from the W form, which builds no H_eff.
  std::vector<double> eigenvalues;
  /// @brief The seed W, n x n row-major: u_K u_K^T, u_K state K's
  /// eigenvector of H_eff, or the W form's own.
  std::vector<double> w;
  /// @brief rho_r = W_rr (u_rK^2 for a state), summing to tr W: the
  /// weighting of the root energies, mcscf::df_gradient's mix.
  std::vector<double> rho;
  /// @brief The state average's weights w_r, mcscf::sa_gradient_weights',
  /// every one positive.
  std::vector<double> weights;
  /// @brief qd_sc_adjoint at W with Phi^SA's adjoint (zc_adjoint at zc.z)
  /// added field by field: h_bar, lambda and x are df_gradient's extra, y is
  /// y_bar_r = y_bar^corr_r + y^c_r per root, raw, and a is A^corr + A^{z^c}.
  /// elements and epsilon are qd_sc_adjoint's. In qd_pc_gradient's, the same
  /// of qd_pc_adjoint, and elements is empty (QdPcGradient::elements).
  QdScAdjoint corr;
  /// @brief z^c over qd_sc_adjoint's A and the SA Fock's epsilon.
  ZcMultipliers zc;
  /// @brief tau from corr.y, split into zeta (equal-weight pairs) and t
  /// (unequal pairs, zvector's theta_rhs).
  mcscf::RootPairMultipliers pairs;
  /// @brief zeta . g, and the folded (gamma^zeta, Gamma^zeta) that one
  /// mcscf::df_transition_gradient takes for zeta's nuclear term.
  mcscf::ZetaTerms zeta;
  /// @brief (Z, z, Theta) in mcscf::zvector's coordinates, its PCG
  /// iterations and residual: df_gradient's multipliers.
  mcscf::ZVector multipliers;
  /// @brief max |X^L_pq - X^L_qp| per block, as ScGradient::asymmetry.
  std::array<double, 6> asymmetry{};
};

/// @brief The SC-QD-NEVPT2 Lagrangian of state @p state (H_eff's eigenvalues
/// ascending) on the DF-SA-CASSCF @p solution, and its multipliers:
/// nuclear_gradient.tex Algorithm qdgrad up to the derivative passes.
///
///     L = E[W] + Z . g_SA + sum_r 2 w_r z_r . (H - E_r) c_r
///         + sum_{r<s} (2 (w_r - w_s) Theta_sr + zeta_sr) <c_s|H|c_r>
///         + <z^c, f^SA>,
///
/// E[W] = sum_r rho_r E_r + sum_MN W_MN C_MN at W = u_K u_K^T. In order:
/// qd_sc_energy and H_eff's eigenvectors (diagonalize), then the W form
/// below at W = u_K u_K^T, one code path: qd_sc_adjoint at W;
/// zc_multipliers over its A and epsilon and zc_adjoint at the weights,
/// added onto it; mcscf::root_pair_multipliers on the summed y_bar, and
/// mcscf::zeta_terms on its zeta; then mcscf::zvector over the CI-relaxed
/// Hessian at the solution and its weights, the orbital right-hand side
/// g^rho + zeta . g + A over the packed pairs (g^rho response::fock's
/// gradient at the rho-weighted RDMs, its core part at tr W), the CI one y_bar_r / (2 w_r) (zvector
/// projects it off the roots) and the rotation one t. Last the gate: X^L
/// symmetric in every block, its CAS part at the rho-weighted RDMs plus z's
/// (with Theta) and zeta's transition RDMs, a block past 1e-8 refused.
///
/// The arguments are mcscf::df_gradient's but the derivative integrals' and
/// @p batch, sc_energy's. The solution's core and virtuals are canonical
/// over the state-averaged Fock, as the Casscf step leaves them; the
/// weights are recovered by mcscf::sa_gradient_weights and used everywhere.
///
/// @return The Lagrangian's pieces; InvalidConfig for counts that do not
///         fit, @p state not one of the roots', or a root of zero weight (it
///         has no SA CI row: below kRotationWeightTolerance); Numerical,
///         naming both states, when H_eff's eigenvalue @p state is within
///         mcscf::kRootGapFloor of another (u_K's derivative is not defined
///         there); Numerical, naming the block, when the gate fails; or the
///         first error of the calls it makes (root_pair_multipliers' refusal
///         of a near-degenerate pair of roots among them).
[[nodiscard]] Result<QdScLagrangian>
qd_sc_lagrangian(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t state,
                 int32_t batch);

/// @brief The SC-QD-NEVPT2 Lagrangian of E[W] at any symmetric weight
/// matrix @p w (n x n row-major, n the solution's roots), and its
/// multipliers: the state form's every step from qd_sc_adjoint at W on,
/// with rho = diag W, and no H_eff. The state form is this at W = u_K u_K^T.
/// Everything in L is linear in W but the core's energy, which E[W] carries
/// at tr W: g^rho's zero-RDM part and the gate's are taken at that weight,
/// so tr W need not be 1 (0 for an interstate coupling's W^PQ).
///
/// The other arguments are the state form's. The result's eigenvalues is
/// empty.
///
/// @return The Lagrangian's pieces; InvalidConfig for a W not n^2 long, or
///         one not symmetric to 1e-12 or not finite (naming the pair), and
///         the state form's other refusals (counts, a root of zero weight,
///         the gate, root_pair_multipliers'); never H_eff's degenerate-pair
///         refusal, which is the state's alone.
[[nodiscard]] Result<QdScLagrangian>
qd_sc_lagrangian(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                 const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
                 const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
                 std::span<const double> w, int32_t batch);

/// @brief qd_sc_gradient's result: the gradient, its two derivative passes,
/// the Lagrangian they were taken at, and the gaps it divides by.
struct QdScGradient {
  /// @brief dE_K/dR, n_atom x 3 row-major: terms.total + transition.total.
  std::vector<double> total;
  /// @brief The mcscf::df_gradient pass: E_cas at the mix rho, the
  /// multipliers (Z, z, Theta) and the correlation + Phi arrays as its extra.
  mcscf::GradientTerms terms;
  /// @brief The mcscf::df_transition_gradient pass at zeta's folded RDMs,
  /// zeta's nuclear term; every term zero (n_atom x 3) when every zeta is 0.
  mcscf::GradientTerms transition;
  /// @brief The Lagrangian: eigenvalues, rho, z^c (max |z^c|, the smallest
  /// inactive gap), tau/zeta/t, (Z, z, Theta) with the PCG iterations and
  /// residual, and the gate's residues.
  QdScLagrangian lagrangian;
  /// @brief max |zeta_sr|.
  double max_zeta = 0.0;
  /// @brief The smallest |E_r - E_s| over the CASSCF roots.
  double root_gap = std::numeric_limits<double>::infinity();
  /// @brief The smallest |lambda_K - lambda_L| of H_eff, L != K; infinite
  /// from the W form.
  double heff_gap = std::numeric_limits<double>::infinity();
};

/// @brief The SC-QD-NEVPT2 nuclear gradient of QD state @p state (H_eff's
/// eigenvalues ascending) on the DF-SA-CASSCF @p solution: qd_sc_lagrangian,
/// then its two derivative passes (nuclear_gradient.tex Algorithm qdgrad,
/// step 5): one mcscf::df_gradient at the mix rho with (Z, z, Theta) as its
/// multipliers and the correlation + Phi arrays (h_bar, Lambda, X) as its
/// extra, and one mcscf::df_transition_gradient at zeta's folded
/// (gamma^zeta, Gamma^zeta) with neither, skipped when every zeta is zero.
///
/// The arguments are qd_sc_lagrangian's with mcscf::df_gradient's @p classes
/// and @p l.
///
/// @return The gradient; qd_sc_lagrangian's errors unchanged (a root of zero
///         weight, a degenerate H_eff pair, the gate), or the first error of
///         a derivative pass.
[[nodiscard]] Result<QdScGradient> qd_sc_gradient(DeviceResources &res, const Molecule &molecule,
                                                  const BasisSet &basis,
                                                  std::span<const PrimitivePairClass> classes,
                                                  const BasisSet &aux, const SphericalMatrixAO &h,
                                                  const Buffers::device<double> &b, const double *l,
                                                  const mcscf::Solution &solution, int32_t ncore,
                                                  int32_t ncas, int32_t state, int32_t batch);

/// @brief The nuclear gradient G(W) of E[W] at any symmetric weight matrix
/// @p w: the W form of qd_sc_lagrangian, then the state form's two
/// derivative passes. The first is one mcscf::df_gradient at the mix rho
/// when rho sums to 0, or to 1 with no negative entry; otherwise two, at
/// rho = tr W p + d: p >= 0 rho's positive part summing to 1 (uniform when
/// it has none), with zero multipliers, scaled by tr W, and d = rho - tr W p
/// summing to 0 (the core off) with the multipliers and the arrays. G is
/// linear in W; the state form, dE_K/dR, is G(u_K u_K^T).
///
/// The arguments are qd_sc_gradient's with @p w for @p state.
///
/// @return The gradient, heff_gap infinite; the W form of
///         qd_sc_lagrangian's errors unchanged, or the first error of a
///         derivative pass.
[[nodiscard]] Result<QdScGradient>
qd_sc_gradient(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
               std::span<const double> w, int32_t batch);

/// @brief qd_pc_gradient's result: QdScGradient's fields, with PC's
/// elements.
struct QdPcGradient {
  /// @brief dE_K/dR, as QdScGradient::total.
  std::vector<double> total;
  /// @brief The mcscf::df_gradient pass, as QdScGradient::terms.
  mcscf::GradientTerms terms;
  /// @brief zeta's mcscf::df_transition_gradient pass, as
  /// QdScGradient::transition.
  mcscf::GradientTerms transition;
  /// @brief The PC-QD Lagrangian in QdScLagrangian's fields: H_eff's
  /// eigenvalues are PC's, and corr is qd_pc_adjoint's arrays plus Phi^SA's,
  /// its elements empty.
  QdScLagrangian lagrangian;
  /// @brief The n^2 elements, as qd_pc_energy computes them.
  std::vector<PcClassEnergies> elements;
  /// @brief max |zeta_sr|.
  double max_zeta = 0.0;
  /// @brief The smallest |E_r - E_s| over the CASSCF roots.
  double root_gap = std::numeric_limits<double>::infinity();
  /// @brief The smallest |lambda_K - lambda_L| of PC's H_eff, L != K.
  double heff_gap = std::numeric_limits<double>::infinity();
};

/// @brief The PC-QD-NEVPT2 nuclear gradient of QD state @p state (PC's H_eff
/// eigenvalues ascending) on the DF-SA-CASSCF @p solution: qd_sc_gradient
/// with qd_pc_energy and qd_pc_adjoint in place of qd_sc_energy and
/// qd_sc_adjoint. The Lagrangian's assembly (z^c the same division over
/// PC's A, under the same degenerate-pair rule; tau split into zeta and t;
/// the CI right-hand side y_bar_r / (2 w_r); the gate) and the two
/// derivative passes are the same code. The arguments are qd_sc_gradient's.
///
/// @return The gradient; a refused class's Numerical error, qd_pc_energy's
///         unchanged, never a gradient with the class zeroed;
///         qd_sc_gradient's other errors (a root of zero weight, a
///         degenerate pair of PC's H_eff, the gate), named for
///         nevpt2::qd_pc_gradient; or the first error of the calls it makes.
[[nodiscard]] Result<QdPcGradient> qd_pc_gradient(DeviceResources &res, const Molecule &molecule,
                                                  const BasisSet &basis,
                                                  std::span<const PrimitivePairClass> classes,
                                                  const BasisSet &aux, const SphericalMatrixAO &h,
                                                  const Buffers::device<double> &b, const double *l,
                                                  const mcscf::Solution &solution, int32_t ncore,
                                                  int32_t ncas, int32_t state, int32_t batch);

} // namespace free_breakfast::nevpt2
