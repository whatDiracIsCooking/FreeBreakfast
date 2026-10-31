/// @file
/// @brief Analytic nuclear Hessians on the GPU: the closed-shell DF-RHF one,
/// df_rhf, the sum of its explicit part (the one-electron, nuclear-repulsion
/// and fitted two-electron second derivatives at fixed orbitals) and its
/// response part, everything the orbitals' first-order response adds, in the
/// symmetric form
///
///     -4 B^T U - 2 (Sigma + Sigma^T) + 2 Omega + Theta,
///
/// over the CPHF right-hand sides B^x and solutions U^x. Each part is its own
/// entry point too. Beside it, the dipole derivative the same U^x gives, and
/// the IR intensities. The README has each term's derivation and the memory it
/// holds.

export module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.response.cphf;

export namespace free_breakfast::hessian {

/// @brief The response part of a closed-shell RHF nuclear Hessian, and the
/// orbital response it was built from.
struct RhfResponse {
  /// @brief d^2E/dx dy less its explicit part, 3 n_atom x 3 n_atom, x = 3 A + k
  /// the coordinate R_{A,k} (hartree / bohr^2). Symmetric to rounding, so it
  /// reads the same row- or column-major.
  std::vector<double> hessian;
  /// @brief U^x, the CPHF solution per coordinate, column x of
  /// (n_vir n_occ) x 3 n_atom with U^x_ai at a + n_vir i, and its residuals:
  /// what a dipole derivative reuses.
  response::Cphf cphf;
};

/// @brief The response part of the closed-shell DF-RHF nuclear Hessian at
/// converged orbitals: the total less the second-derivative integrals
/// contracted with D, W and D (x) D (and V_nn's), in the symmetric form.
///
/// Per coordinate x, with S^x = dS/dx and the skeleton Fock derivative
/// F^(x) = T^x + V^x + J^(x) - 1/2 K^(x) at fixed D (@c overlap_derivative,
/// @c kinetic_derivative, @c potential_derivative, @c df_jk_derivative),
/// their MO blocks X_pi = C_p^T X C_i over every MO p and occupied i,
/// D_S^x = 2 C_o S^x_oo C_o^T and G = J - K/2 over the fit,
///
///     B^x_ai = -F^(x)_ai + eps_i S^x_ai + G(D_S^x)_ai,   M U^x = B^x
///     H_xy   = -2 (B^x.U^y + B^y.U^x) - 2 (Sigma_xy + Sigma_yx)
///              + 2 Omega_xy + Theta_xy
///     Sigma_xy = Sum_ij S^x_ij F^(y)_ij
///     Omega_xy = Sum_ij S^x_ij S^y_ij (eps_i + eps_j)
///     Theta_xy = tr[D_S^x G(D_S^y)] = 2 Sum_ij S^x_ij G(D_S^y)_ij
///
/// the CPHF by @c response::cphf, every column in lockstep. B^x.U^y is
/// B^T M^-1 B, symmetric only once the CPHF is solved exactly; its
/// symmetrized half is taken, so H is symmetric to rounding and the part
/// discarded is the CPHF residual's.
///
/// @p molecule is the molecule both bases were placed on (by @c place), and
/// @p classes are @c primitive_pair_classes of @p basis. @p b is B as
/// @c density_fit leaves it, naux x nao^2 with element (Q, n + nao m) at
/// Q + naux (n + nao m), and @p l is its L (column-major, the lower triangle
/// read), naux = aux.nao_sphe(). @p c is the nao x nmo RHF orbitals,
/// column-major, nmo = mo_energies.size(), the first @p nocc occupied, with
/// @p mo_energies their canonical orbital energies on the host. @p tolerance
/// and @p max_iterations are @c response::cphf's.
///
/// Holds three MO stacks (nmo nocc 3 n_atom doubles each), the right-hand
/// sides, and while the skeleton Fock derivatives are built three 3 n_atom
/// nao^2 stacks of AO matrices (F^(x), J^(x), K^(x)) besides
/// @c df_jk_derivative's own scratch; past the device's free memory that is
/// Unsupported, naming the bytes, before anything is allocated.
///
/// Queued on res.stream(); the derivative calls and the CPHF synchronize it,
/// and the call returns with everything done. Scratch for the J and K builds
/// comes from res.device_workspace(), so none of @p b, @p l and @p c may point
/// into it. The device res.dev_idx() names is made current for the call and
/// the previous one restored.
///
/// @return The response part and U; InvalidConfig if @p molecule is not of
///         basis.natoms() atoms, if @p nocc is not in [1, nmo), if @p c is not
///         nao nmo or @p b not naux nao^2 long, or for any input the
///         derivative calls or the CPHF refuse; Unsupported for memory past
///         the device's free bytes or a stack past int32_t; the CPHF's
///         MaxIterationsExceeded or Divergence; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<RhfResponse>
df_rhf_response(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const Buffers::device<double> &b, const double *l, const Buffers::device<double> &c,
                std::span<const double> mo_energies, int32_t nocc, double tolerance = 1e-10,
                int32_t max_iterations = 100);

/// @brief The two-electron explicit part of a closed-shell DF-RHF nuclear
/// Hessian: the fitted Coulomb and exchange energies differentiated twice at
/// fixed orbitals, each 3 n_atom x 3 n_atom, x = 3 A + k the coordinate
/// R_{A,k} (hartree / bohr^2), symmetric to rounding.
struct RhfTwoElectron {
  /// @brief d^2 E_J / dx dy, E_J = 1/2 u^T M^-1 u, u = T^T vec D.
  std::vector<double> coulomb;
  /// @brief d^2 E_x / dx dy, E_x = -Sum_ij T_ij^T M^-1 T_ij over the
  /// occupied pairs.
  std::vector<double> exchange;
};

/// @brief The two-electron explicit part of the closed-shell DF-RHF nuclear
/// Hessian at fixed orbitals: the second-derivative integrals contracted with
/// the fitted densities, and the fit's cross terms (rhf_hessian.tex eqs. dfj,
/// dfk). With T_mn,P = (mn|P), M = (P|Q) = L L^T, d = M^-1 T^T vec D,
/// c_ij = M^-1 T_ij over occupied i, j, and R^x_ij = T^x_ij - M^x c_ij,
///
///     E_J^xy = u^xy^T d - 1/2 d^T M^xy d + g^x^T M^-1 g^y,  g^x = 2 Sum_i R^x_ii
///     E_x^xy = -Sum_ij (2 T^xy_ij^T c_ij - c_ij^T M^xy c_ij
///                       + 2 R^x_ij^T M^-1 R^y_ij)
///
/// the first two of each by @c three_center_hessian and @c metric_hessian
/// (at D (x) d and -1/2 d d^T, and at @c gradient::df_rhf's Gam^K and W^K),
/// the cross terms as Gram matrices of L^-1 R^x. These are PySCF's
/// @c df.hessian.rhf partial_hess_elec's ej and -ek.
///
/// **The exchange cross term is never held whole** (3 n_atom nocc^2 naux
/// doubles): R^x for the three coordinates of one atom, one atom block, is
/// built by one walk over the aux atoms (@c three_center_derivative's blocks
/// against @c metric_derivative's M1), and the Hessian is contracted one pair
/// of atom blocks at a time, so n_atom (n_atom + 1) / 2 blocks are built. The
/// Coulomb vectors L^-1 g^x come from the same blocks.
///
/// @p basis and @p aux are made by @c place over one molecule, and @p classes
/// are @c primitive_pair_classes of @p basis. @p b is B as @c density_fit
/// leaves it, naux x nao^2 at Q + naux (n + nao m), and @p l is its L
/// (column-major, the lower triangle read). @p c is the nao x nmo orbitals,
/// column-major, of which only the first @p nocc are read.
///
/// The call holds the fitted pairs c (naux nocc^2), M1 (3 naux^2), one aux
/// atom's derivative block (3 nao^2 nblk) and two atom blocks
/// (3 naux nocc^2 each); before that, the fitted densities of the explicit
/// part (naux nao^2, like @c gradient::df_rhf's). Past the device's free
/// memory that is Unsupported, naming the bytes, before anything is
/// allocated.
///
/// Queued on res.stream() and synchronized; the call returns with everything
/// done. None of @p b, @p l and @p c may point into res.device_workspace().
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The Coulomb and exchange parts; InvalidConfig if the bases do not
///         share their atoms, if @p nocc is not in [1, nao], or if @p c is
///         shorter than nao nocc or @p b not naux nao^2 long, or for any
///         input the integral calls refuse; Unsupported for memory past the
///         device's free bytes or an index past int32_t; or the GpuRuntime
///         error of the first copy, BLAS call or launch that failed.
[[nodiscard]] Result<RhfTwoElectron>
df_rhf_two_electron(DeviceResources &res, const BasisSet &basis,
                    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                    const Buffers::device<double> &b, const double *l,
                    const Buffers::device<double> &c, int32_t nocc);

/// @brief The two-electron explicit part of a DF-CASSCF nuclear Hessian
/// (casscf_hessian.tex eq. df2) and its pieces, each 3 n_atom x 3 n_atom,
/// x = 3 A + k the coordinate R_{A,k} (hartree / bohr^2), symmetric to
/// rounding.
struct CasscfTwoElectron {
  /// @brief Sum_aP Gam~_aP T^xy_aP, Gam~ = Gamma c.
  std::vector<double> three_center;
  /// @brief -1/2 Sum_PQ W^_PQ M^xy_PQ, W^ = c^T Gamma c.
  std::vector<double> metric;
  /// @brief The cross term's J piece, Gamma = D (x) D - D_a (x) D_a:
  /// g^x^T M^-1 g^y - g_a^x^T M^-1 g_a^y, g^x = u^x - M^x d.
  std::vector<double> cross_coulomb;
  /// @brief Its K piece, -2 Sum_i Sum_pq G_pq R^x_ip^T M^-1 R^y_iq, i core,
  /// G = diag(1_core, gamma) over the occupied orbitals.
  std::vector<double> cross_exchange;
  /// @brief Its active piece, Sum Gamma_tuvw R^x_tu^T M^-1 R^y_vw.
  std::vector<double> cross_active;
  /// @brief The five summed: d^2 E_2 / dx dy at fixed orbitals and RDMs.
  std::vector<double> total;
};

/// @brief The two-electron explicit part of the DF-CASSCF nuclear Hessian at
/// fixed orbitals and RDMs (casscf_hessian.tex eq. df2): with
/// T_a,P = (a|P) over AO pairs a, M = (P|Q) = L L^T, the fitted pairs
/// c = T M^-1 and R^x = T^x - c M^x, for the pair-space density Gamma_ab of
/// E_2 = 1/2 Sum Gamma_ab (T M^-1 T^T)_ab,
///
///     E_2^xy = Sum Gam~ T^xy - 1/2 Sum W^ M^xy + Sum_ab Gamma_ab [R^x M^-1 R^y^T]_ab
///
/// the first two by @c three_center_hessian and @c metric_hessian at
/// @c mcscf::df_gradient's Gam and W (at these RDMs, no Lagrangian), the
/// cross term split as the design's section 6 so no pair-space Gamma is
/// formed: J (Gamma = D (x) D - D_a (x) D_a, two Gram matrices of L^-1 g^x),
/// K (core with core and with active, -Sum_i Sum_pq G_pq (ip|iq)) and the
/// active 2-RDM's (Sum Gamma_tuvw R^x_tu^T M^-1 R^y_vw).
///
/// **The cross term is never held whole.** As in @c df_rhf_two_electron,
/// L^-1 R^x over the occupied pairs for the three coordinates of one atom
/// (3 naux nocc^2 doubles, an atom block) is built by one walk over the aux
/// atoms, and the Hessian contracted one pair of atom blocks at a time, so
/// n_atom (n_atom + 1) / 2 blocks are built; J's vectors, K's core rows
/// (3 naux ncore nocc after G) and the active pairs all come from each block.
///
/// @p basis and @p aux are made by @c place over one molecule, and @p classes
/// are @c primitive_pair_classes of @p basis. @p b is B as @c density_fit
/// leaves it, naux x nao^2 at Q + naux (n + nao m), and @p l is its L
/// (column-major, the lower triangle read). @p c is the nao x nmo orbitals,
/// column-major, core then active, of which only the first ncore + ncas are
/// read. @p gamma and @p gamma2 are the active 1- and 2-RDM on the host,
/// row-major as @c ci::Rdm::rdm12 writes them (E = Sum h D + 1/2 Sum (pq|rs)
/// Gamma_pqrs, PySCF's make_rdm12): a state's own, or any other densities the
/// caller wants the explicit part at.
///
/// The call holds the fitted occupied pairs (naux nocc^2) and Gamma
/// (ncas^4); then the explicit part's Gam (naux nao^2, like
/// @c mcscf::df_gradient's), or the cross term's M1 (3 naux^2), one aux
/// atom's derivative block (3 nao^2 nblk), two atom blocks and K's
/// half-transform. Past the device's free memory that is Unsupported, naming
/// the bytes, before anything is allocated.
///
/// Queued on res.stream() and synchronized; the call returns with everything
/// done. None of @p b, @p l and @p c may point into res.device_workspace().
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The pieces and their sum; InvalidConfig if the bases do not share
///         their atoms, if ncore < 0, ncas < 1 or ncore + ncas > nao, if @p c
///         is shorter than nao (ncore + ncas) or @p b not naux nao^2 long, if
///         the RDMs are not ncas^2 and ncas^4 long, or for any input the
///         integral calls refuse; Unsupported for memory past the device's
///         free bytes or an index past int32_t; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<CasscfTwoElectron>
df_casscf_two_electron(DeviceResources &res, const BasisSet &basis,
                       std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                       const Buffers::device<double> &b, const double *l,
                       const Buffers::device<double> &c, int32_t ncore, int32_t ncas,
                       std::span<const double> gamma, std::span<const double> gamma2);

/// @brief The closed-shell DF-RHF nuclear Hessian and every term of it, each
/// 3 n_atom x 3 n_atom, x = 3 A + k the coordinate R_{A,k}
/// (hartree / bohr^2), as @c gradient::RhfTerms holds the gradient's.
struct RhfTerms {
  /// @brief d^2E/dx dy, the sum of the terms below symmetrized,
  /// (H + H^T) / 2: exactly symmetric, so it reads the same row- or
  /// column-major.
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the terms' sum before it was symmetrized: each
  /// term is symmetric to rounding, so this is their rounding. What the
  /// response part's own symmetrization discards is the CPHF residual's,
  /// which response.cphf.residuals reports.
  double asymmetry{0.0};
  /// @brief The one-electron explicit part: T's and V's second-derivative
  /// integrals contracted with D, S's with -W, W = 2 Sum_i eps_i C_i C_i^T
  /// (PySCF's partial_hess_elec e1).
  std::vector<double> one_electron;
  /// @brief d^2 V_nn / dx dy, @c nuclear_repulsion_hessian.
  std::vector<double> nuclear_repulsion;
  /// @brief The fitted Coulomb and exchange explicit parts,
  /// @ref df_rhf_two_electron's.
  RhfTwoElectron two_electron;
  /// @brief The response part and the CPHF it was built from,
  /// @ref df_rhf_response's.
  RhfResponse response;
};

/// @brief The closed-shell DF-RHF analytic nuclear Hessian at converged
/// orbitals: @c overlap_hessian at -W, @c kinetic_hessian and
/// @c potential_hessian at D = 2 C_o C_o^T, @c nuclear_repulsion_hessian,
/// @ref df_rhf_two_electron and @ref df_rhf_response, summed and symmetrized.
/// PySCF's @c pyscf.df.hessian.rhf over the same fitting set (auxbasis
/// response 2).
///
/// The inputs are @ref df_rhf_response's, and are checked as it checks them
/// before anything runs: @p molecule the one both bases were placed on,
/// @p classes @c primitive_pair_classes of @p basis, @p b and @p l
/// @c density_fit's B and L, @p c the nao x nmo orbitals (nmo =
/// mo_energies.size(), the first @p nocc occupied) and @p mo_energies their
/// canonical energies; @p tolerance and @p max_iterations are the CPHF's.
///
/// The parts run one after another, each freeing what it held before the
/// next, so the call's peak memory is the larger of the two two-electron
/// calls' (each refuses its own, Unsupported, naming the bytes). D and -W
/// (nao^2 each) are held only for the one-electron part.
///
/// Queued on res.stream(); every part synchronizes it, and the call returns
/// with everything done. None of @p b, @p l and @p c may point into
/// res.device_workspace(). The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return The Hessian and its terms; InvalidConfig if @p molecule is not of
///         basis.natoms() atoms, if @p nocc is not in [1, nmo), if @p c is not
///         nao nmo or @p b not naux nao^2 long, or for any input a part
///         refuses; or the first error of a part (Unsupported for memory, the
///         CPHF's MaxIterationsExceeded or Divergence, a GpuRuntime error).
[[nodiscard]] Result<RhfTerms> df_rhf(DeviceResources &res, const Molecule &molecule,
                                      const BasisSet &basis,
                                      std::span<const PrimitivePairClass> classes,
                                      const BasisSet &aux, const Buffers::device<double> &b,
                                      const double *l, const Buffers::device<double> &c,
                                      std::span<const double> mo_energies, int32_t nocc,
                                      double tolerance = 1e-10, int32_t max_iterations = 100);

/// @brief The closed-shell RHF dipole derivative, dmu_c/dR_{A,k}, from the
/// orbitals' first-order response: for every coordinate x = 3 A + k,
///
///     dmu/dx = -tr(D mu^x) - 4 Sum_ai U^x_ai mu_ai + 2 Sum_ij S^x_ij mu_ij
///              + Z_A e_k,
///
/// mu_c = <m| r_c |n> about the origin (@c dipole), electrons of charge -1,
/// D = 2 C_o C_o^T, mu^x the integrals' nuclear derivative at fixed D
/// (@c dipole_gradient), S^x the overlap's (@c overlap_derivative) and mu_pq
/// = C_p^T mu C_q (rhf_hessian.tex section 10, step 6). U^x is the CPHF
/// solution a Hessian solves anyway, @c df_rhf's response.cphf.solution, so
/// the derivative costs no response solve of its own. For a neutral molecule
/// it does not depend on the origin, and its sum over the atoms vanishes to
/// the CPHF's tolerance.
///
/// @p molecule is the molecule @p basis was placed on, and @p classes are
/// @c primitive_pair_classes of @p basis. @p c is the nao x nmo orbitals,
/// column-major (nmo = c.num_elements() / nao), the first @p nocc occupied, and
/// @p u is U^x as @c response::cphf returns it, (n_vir n_occ) x 3 n_atom with
/// U^x_ai at a + n_vir i of column x. The dipole integrals stop at d
/// (@c kDipoleMaxL), so a basis with an f shell is refused.
///
/// Holds S^x (3 n_atom nao^2 doubles) and the three dipole matrices with their
/// MO blocks; past the device's free memory that is Unsupported, naming the
/// bytes, before anything is allocated. Queued on res.stream() and
/// synchronized: the call returns with everything done. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return dmu_c/dx (e, the dipole in e bohr per bohr), 3 n_atom x 3 row-major:
///         element (x, c) at 3 x + c; InvalidConfig if @p molecule is not of
///         basis.natoms() atoms, if @p c is not a whole number of nao columns,
///         if @p nocc is not in [1, nmo), if @p u is not (nmo - nocc) nocc
///         3 n_atom long, or for any input the integral calls refuse (an f
///         shell among them); Unsupported for memory past the device's free
///         bytes or 3 n_atom nao^2 past int32_t; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<std::vector<double>>
rhf_dipole_derivative(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                      std::span<const PrimitivePairClass> classes, const Buffers::device<double> &c,
                      int32_t nocc, const Buffers::device<double> &u);

/// @brief The harmonic IR intensities, I_q = K |Sum_x L_qx dmu/dx|^2 in km/mol,
/// K = N_A / (12 eps_0 c^2) e^2 / u = 974.88 km/mol per e^2 / amu.
///
/// @p modes are the Cartesian normal modes as @c geometry::harmonic_analysis
/// returns them, n_modes x 3 n_atom row-major, each its unit mass-weighted
/// eigenvector over sqrt(m_A / amu); @p dipole_derivative is
/// @ref rhf_dipole_derivative's 3 n_atom x 3. A mode's sign does not matter.
///
/// @return One intensity per mode, in the modes' order; InvalidConfig if
///         @p dipole_derivative is not 3 n_atom x 3 or @p modes not a whole
///         number of 3 n_atom rows.
[[nodiscard]] Result<std::vector<double>>
infrared_intensities(std::span<const double> modes, std::span<const double> dipole_derivative);

} // namespace free_breakfast::hessian
