/// @file
/// @brief The active-space Hamiltonian of a set of orbitals, (E_core, h_eff,
/// (tu|vw)), built integral-direct from @c coulomb and @c exchange: no AO
/// two-electron integral is stored.
///
/// What a CASCI (and, once per macro-iteration, a CASSCF) diagonalizes over:
/// FCI is ncore = 0 and ncas = nmo. Classes through d only, since J and K
/// refuse higher.

export module free_breakfast.integrals.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

export namespace free_breakfast {

/// @brief The Hamiltonian of an active space: its constant, its one-electron
/// part dressed by the core, and its two-electron integrals, on the device it
/// was built on.
struct ActiveSpaceHamiltonian {
  /// @brief E_core = E_nuc + sum_i (2 h_ii + 2 J_ii - K_ii) over the core
  /// orbitals i (hartree): the core's energy, nuclear repulsion included.
  double e_core;
  /// @brief The number of active orbitals, n_act: the order of @ref h_eff and
  /// each of @ref eri's four indices.
  int32_t ncas;
  /// @brief h_eff = C_act^T (h + 2 J[P] - K[P]) C_act, P = C_core C_core^T,
  /// n_act x n_act, column-major.
  Buffers::device<double> h_eff;
  /// @brief (tu|vw) in chemists' notation, every one of the n_act^4 elements,
  /// (t, u, v, w) at t + n_act (u + n_act (v + n_act w)).
  Buffers::device<double> eri;
};

/// @brief The active-space Hamiltonian of the orbitals @p c: columns
/// [0, ncore) the doubly occupied core, [ncore, ncore + ncas) the active space.
///
/// Every matrix is over @p basis, built and carried as the Scf step does:
/// @p h the spherical core Hamiltonian, @p c the spherical MO coefficients (AO
/// by MO, nmo = c.n()), @p u @c upload_spherical_transform_matrix's U (null
/// when no shell is above p), @p j_classes @c coulomb_pair_classes and
/// @p k_classes @c ordered_pair_classes of it. The classes are built once per
/// basis, so a CASSCF loop builds none.
///
/// E_core and h_eff come from one @c coulomb (at @p tau_j) and one
/// @c exchange (at @p tau_k) of the symmetric core density P, skipped for
/// ncore = 0. (tu|vw) takes n_act (n_act + 1) / 2 @c coulomb calls, one per
/// pair v <= w, each of the symmetrized pair density
/// D_vw = (C_v C_w^T + C_w C_v^T) / 2: J needs a symmetric density, and an
/// antisymmetric one has J = 0, so its symmetric part is the whole of
/// J[C_v C_w^T]. Each J is contracted as C_act^T J C_act into the (v, w) and
/// (w, v) blocks.
///
/// **The prune cap.** @c coulomb_pair_classes prunes assuming |D_cd| <= 2, and
/// a pair density can exceed that. With m_v = max_mu |(U C)_mu v| over the
/// Cartesian rows, |D_vw| <= m_v m_w, so each @c coulomb runs on s D_vw,
/// s = min(1, 2 / (m_v m_w)), screened at s @p tau_j, and its J is scaled by
/// 1 / s: J's absolute screen at @p tau_j holds for D_vw itself.
///
/// @p tau_k's default is @c exchange's own; @p tau_j's is 1e-14, not
/// @c coulomb's 1e-10, at which (tu|vw) = (vw|tu) breaks by 1e-10 (H2O/cc-pVDZ),
/// past 1e-12 max|(tu|vw)|; at 1e-13 N2/cc-pVDZ's still breaks by up to
/// 9e-13, past it on some runs. Queued on
/// res.stream(); it synchronizes it for each of E_core's two dots (none for
/// ncore = 0) and once to read the Cartesian active orbitals back for the m_v. Its scratch is
/// res.device_workspace()'s, besides the three Cartesian matrices @c coulomb
/// and @c exchange take, allocated per call. The device res.dev_idx() names is
/// made current for the call and the previous one restored.
///
/// @return The Hamiltonian; InvalidConfig if ncas is below 1, ncore is
///         negative, ncore + ncas exceeds nmo, a tau is negative or NaN, @p h
///         or @p c is not of @p basis's spherical order or not on res's device;
///         @c coulomb's and
///         @c exchange's own InvalidConfig (a class above d, or classes of
///         another basis); or the GpuRuntime error of the first BLAS, copy or
///         launch that failed.
[[nodiscard]] Result<ActiveSpaceHamiltonian>
active_space_hamiltonian(DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
                         const SphericalMatrixAO &h,
                         const std::shared_ptr<const Buffers::device<double>> &u,
                         std::span<CoulombPairClass> j_classes,
                         std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
                         int32_t ncore, int32_t ncas, double tau_j = 1e-14, double tau_k = 1e-13);

} // namespace free_breakfast
