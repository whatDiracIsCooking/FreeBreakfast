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
/// such A, and the term Phi = <z^c, f> they add with its adjoints. The README
/// has the layouts and every term.

export module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
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

} // namespace free_breakfast::nevpt2
