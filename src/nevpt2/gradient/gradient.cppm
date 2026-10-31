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
/// class energies (and their reverse pass) in place of SC's. The README has
/// the layouts and every term.

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
  /// @brief Phi = <z, f> = sum_pq z_pq f_pq, f the generalized Fock.
  double phi = 0.0;
  /// @brief dPhi/dh in the MO basis, nmo x nmo: z itself.
  Buffers::device<double> h_bar;
  /// @brief dPhi/dB^L in the MO basis, naux x nmo^2 as ScAdjoint::lambda,
  /// symmetric in (p, q).
  Buffers::device<double> lambda;
  /// @brief y^c = dPhi/dc at fixed (h, B), one per determinant: the raw
  /// derivative of Phi as a function of an unnormalized c, as ScAdjoint::y.
  Buffers::device<double> y;
  /// @brief X = 2 h h_bar + 2 sum_L B^L Lambda^L, nmo x nmo, as
  /// ScAdjoint::x.
  Buffers::device<double> x;
  /// @brief A = X - X^T: dPhi = sum_{p > q} kappa_pq A_pq along
  /// C -> C exp(kappa).
  Buffers::device<double> a;
};

/// @brief Phi = <@p z, f> at (@p h, @p b, @p c, @p vector), and its adjoints
/// at fixed @p z (nmo x nmo column-major on the device, symmetric): the
/// arguments but z are sc_energy's.
///
/// f_pq = h_pq + sum_rs D_rs [(pq|rs) - 1/2 (pr|sq)], D the reference's MO
/// density (2 on the core's diagonal, gamma on the active block), so its
/// core and virtual diagonals are sc_adjoint's epsilon. B goes to the MO
/// basis whole, naux nmo^2; gamma is ci::Sigma's rdm1 of @p vector, and y^c
/// one sigma over the one-body (W + W^T)_tu, W_tu = dPhi/dgamma_tu.
///
/// @return Phi and its adjoints; InvalidConfig for counts that do not fit
///         (ncas not the strings' orbitals, ncore + ncas past nmo); or the
///         error of b_mo, ci::Sigma, an einsum or a BLAS call.
[[nodiscard]] Result<ZcAdjoint> zc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                           const ci::DeviceStringSet &beta, int32_t naux,
                                           int32_t nao, int32_t nmo, const double *b,
                                           const double *c, const double *h, int32_t ncore,
                                           int32_t ncas, const double *vector, const double *z);

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

} // namespace free_breakfast::nevpt2
