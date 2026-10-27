/// @file
/// @brief The energy, the inactive, active and generalized Fock matrices and
/// the orbital gradient of a (state-averaged) CASSCF wavefunction at fixed
/// orbitals and RDMs, on the device; and the packed order of the
/// non-redundant orbital pairs that the gradient, the Hessian and the
/// optimizer share.
///
/// F^I and F^A are one @c coulomb and one @c exchange each; the generalized
/// Fock's two-electron part is one gemm over @c casscf_integrals' (pq|uv). No
/// AO two-electron integral is stored. See README.md.

export module free_breakfast.response.fock;

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

export namespace free_breakfast::response {

/// @brief A CASSCF wavefunction's energy and first derivatives at fixed
/// orbitals C and RDMs, on the device it was built on.
///
/// Every matrix is over the nmo orbitals, column-major, element (p, q) at
/// p + nmo q: columns [0, ncore) of C the core, [ncore, nocc) the active
/// space, nocc = ncore + ncas, the rest virtual.
struct Fock {
  /// @brief E = E_core + sum_tu h_eff,tu gamma_tu + 1/2 sum_tuvw (tu|vw) Gamma_tuvw
  /// (hartree), nuclear repulsion included: the weighted energy of the states
  /// the RDMs average.
  double energy;
  /// @brief F^I = C^T (h + 2 J[P_c] - K[P_c]) C, P_c = C_core C_core^T,
  /// nmo x nmo. Its core-core trace with C^T h C's is E_core less E_nuc, and
  /// its active block is @c active_space_hamiltonian's h_eff.
  Buffers::device<double> fock_inactive;
  /// @brief F^A = C^T (J[P_a] - 1/2 K[P_a]) C, P_a = C_act gamma C_act^T,
  /// nmo x nmo; zero for ncas = 0.
  Buffers::device<double> fock_active;
  /// @brief The generalized Fock F, nmo x nmo, column q for orbital q:
  /// 2 (F^I + F^A)_pq for q core; sum_v F^I_pv gamma_vq +
  /// sum_uvw (pu|vw) Gamma_quvw for q active; 0 for q virtual.
  Buffers::device<double> fock;
  /// @brief g_pq = 2 (F_pq - F_qp), dE/dx_pq for C exp(kappa(x)), over the
  /// non-redundant pairs p > q in @ref pack_orbital_pairs' order.
  Buffers::device<double> gradient;
  /// @brief The weighted gamma as given, ncas x ncas, column-major:
  /// gamma_tu at t + ncas u.
  Buffers::device<double> gamma;
  /// @brief The weighted Gamma as given, ncas^4, Gamma_tuvw at
  /// t + ncas (u + ncas (v + ncas w)): (tu|vw)'s order in
  /// @c ActiveSpaceHamiltonian::eri, not @c ci::Rdm::rdm12's row-major one.
  Buffers::device<double> gamma2;
};

/// @brief The energy, F^I, F^A, the generalized Fock and the orbital gradient
/// of the orbitals @p c (columns [0, ncore) the core, [ncore, ncore + ncas)
/// the active space) and the weighted RDMs @p gamma and @p gamma2.
///
/// @p basis, @p molecule, @p h, @p u and the two kinds of classes are what
/// @c active_space_hamiltonian takes, built once per basis; @p ppaa is
/// @c casscf_integrals' (pq|uv) over the same @p c, nmo^2 ncas^2 doubles
/// (empty for ncas = 0). @p gamma and @p gamma2 are @c ci::Rdm::rdm12's, read
/// back to the host, in its layout: gamma n x n row-major, Gamma n^4 row-major (t, u, v, w). Both
/// are uploaded into the result.
///
/// F^I takes one @c coulomb (at @p tau_j) and one @c exchange (at @p tau_k)
/// of P_c, skipped for ncore = 0; F^A one of each of P_a, skipped for
/// ncas = 0, built of s P_a at s tau, s = min(1, 2 / sum_tu |gamma_tu| m_t m_u),
/// m_t = max_mu |(U C)_mu t|, and scaled by 1 / s: the |D| <= 2 prune cap of
/// @c coulomb_pair_classes, held as @c casscf_integrals holds it for each pair
/// density. The generalized Fock's sum_uvw (pu|vw) Gamma_tuvw is one gemm,
/// over (pu|vw) gathered from @p ppaa's active columns. E_core and h_eff are
/// read off F^I (E_core = E_nuc + sum_i (h_ii + F^I_ii), h_eff = F^I's active
/// block), the definitions @c active_space_hamiltonian builds them by, and
/// (tu|vw) off @p ppaa: no J or K past F^I's and F^A's.
///
/// ncas = 0 is the RHF limit: F = 2 F^I in its core columns, the gradient is
/// 4 F^I_ai over virtual a and occupied i, and the energy the RHF one.
///
/// Queued on res.stream() with res.blas(), whose pointer mode must be host
/// (its default); it synchronizes res.stream() once to read the active
/// orbitals back for the m_t (not for ncas = 0) and once at the end, for the
/// energy. Its scratch is res.device_workspace()'s, besides the three
/// Cartesian matrices @c coulomb and @c exchange take, allocated per call.
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The energy and matrices; InvalidConfig if ncore or ncas is
///         negative, ncore + ncas is 0 or exceeds nmo, a tau is negative or
///         NaN, @p h or @p c is not of @p basis's spherical order or not on
///         res's device, @p ppaa is not nmo^2 ncas^2 long, or @p gamma or
///         @p gamma2 is not ncas^2 or ncas^4 long; @c coulomb's and
///         @c exchange's own InvalidConfig; or the GpuRuntime error of the
///         first BLAS, copy or launch that failed.
[[nodiscard]] Result<Fock> fock(DeviceResources &res, const BasisSet &basis,
                                const Molecule &molecule, const SphericalMatrixAO &h,
                                const std::shared_ptr<const Buffers::device<double>> &u,
                                std::span<CoulombPairClass> j_classes,
                                std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
                                int32_t ncore, int32_t ncas, const Buffers::device<double> &ppaa,
                                std::span<const double> gamma, std::span<const double> gamma2,
                                double tau_j = 1e-14, double tau_k = 1e-14);

/// @brief The non-redundant pairs of the nmo x nmo column-major device matrix
/// @p full into the device vector @p packed, in the packed order: element
/// (p, q), p > q, for q core and p active or virtual, then q active and p
/// virtual, column by column (q outer), p ascending within a column. So
/// @p packed is two column-major blocks back to back: rows [ncore, nmo) of
/// columns [0, ncore), then rows [nocc, nmo) of columns [ncore, nocc),
/// ncore (nmo - ncore) + ncas (nmo - nocc) doubles in all.
///
/// Two 2D copies on res.stream(), not synchronized.
///
/// @return InvalidConfig if ncore or ncas is negative or ncore + ncas
///         exceeds nmo; or a copy's GpuRuntime error.
[[nodiscard]] Status pack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                        int32_t ncas, const double *full, double *packed);

/// @brief The antisymmetric kappa of the packed @p packed into the nmo x nmo
/// column-major device matrix @p kappa: kappa_pq = x_pq and kappa_qp = -x_pq
/// for each packed pair p > q, every other element 0. C exp(kappa) is then
/// the rotation along x that @ref Fock::gradient differentiates.
/// @p kappa and @p packed must not overlap.
///
/// A memset, two 2D copies and two transposing geams on res.stream() with
/// res.blas() (host pointer mode), not synchronized.
///
/// @return InvalidConfig as @ref pack_orbital_pairs; or the GpuRuntime error
///         of a copy or a geam.
[[nodiscard]] Status unpack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                          int32_t ncas, const double *packed, double *kappa);

} // namespace free_breakfast::response
