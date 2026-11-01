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
import free_breakfast.recipes.matrix_ao;
import free_breakfast.response.cphf;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.zvector;

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
/// **A rotation.** With @p rotation, an antisymmetric nmo x nmo K on the
/// device (column-major, MO indices C's columns), the call returns instead the
/// same terms at the K-rotated densities, K D - D K and K on each index of
/// Gamma: their derivative along C -> C (1 + e K) at fixed MO-basis RDMs, a
/// state average's z_o part. @p c must then hold every orbital (nao x nmo).
/// It is the product rule on every occupied C: the explicit part's Gam and W
/// gain each intermediate's derivative (mcscf::df_gradient's orbital
/// response), and the cross term's atom blocks are built over [C_o | C K_o]
/// (twice the occupied orbitals, four times the block) and cut into r and
/// its derivative r', each pair of atoms contracted as r' with r and r with
/// r'. Without it (the default) nothing of this runs.
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
///         is shorter than nao (ncore + ncas) (with a rotation, not nao nmo
///         with nmo >= ncore + ncas) or @p b not naux nao^2 long, if
///         the RDMs are not ncas^2 and ncas^4 long, or for any input the
///         integral calls refuse; Unsupported for memory past the device's
///         free bytes or an index past int32_t; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<CasscfTwoElectron> df_casscf_two_electron(
    DeviceResources &res, const BasisSet &basis, std::span<const PrimitivePairClass> classes,
    const BasisSet &aux, const Buffers::device<double> &b, const double *l,
    const Buffers::device<double> &c, int32_t ncore, int32_t ncas, std::span<const double> gamma,
    std::span<const double> gamma2, const double *rotation = nullptr);

/// @brief E_xy, the fixed-parameter second derivative of a DF-CASSCF energy
/// along the symmetric (Lowdin) connection (casscf_hessian.tex eq. exy), and
/// its pieces, each 3 n_atom x 3 n_atom, x = 3 A + k the coordinate R_{A,k}
/// (hartree / bohr^2), column-major and symmetric to rounding.
struct CasscfExy {
  /// @brief Sum D^AO h^xy: T's and V's second-derivative integrals against
  /// the 1-RDM in the AO basis. S^xy is the connection's.
  std::vector<double> one_electron;
  /// @brief d^2 V_nn / dx dy, @c nuclear_repulsion_hessian.
  std::vector<double> nuclear_repulsion;
  /// @brief The fitted two-electron explicit part, eq. df2,
  /// @ref df_casscf_two_electron's at the same RDMs.
  CasscfTwoElectron two_electron;
  /// @brief 2 Sum_pq F_pq tau^xy_pq, tau^xy = -1/2 S^xy + 3/8 (S^x S^y +
  /// S^y S^x): the connection's second derivative against F, eq. fock.
  std::vector<double> connection_second;
  /// @brief 2 Sum_pq (F^[x]_pq tau^y_pq + F^[y]_pq tau^x_pq), tau^x =
  /// -1/2 S^x: the derivative-integral Fock against the first-order
  /// connection.
  std::vector<double> connection_fock;
  /// @brief E^(2)[tau^x, tau^y]: the energy at the integrals one-index
  /// transformed by tau^x on one index and tau^y on another, every pair of
  /// positions (the orbital Hessian's contraction, with symmetric generators).
  std::vector<double> connection_twice;
  /// @brief The six summed: E_xy.
  std::vector<double> exy;
};

/// @brief E_xy of a DF-CASSCF at fixed orbitals and RDMs (casscf_hessian.tex
/// eq. exy): the explicit part (Sum D^AO h^xy, V_nn^xy and the fitted
/// eq. df2) plus the connection C T(x), T = (C^T S(x) C)^-1/2, to second
/// order:
///
///     E_xy = explicit + 2 Sum F tau^xy + 2 Sum (F^[x] tau^y + F^[y] tau^x)
///            + E^(2)[tau^x, tau^y],
///     tau^x = -1/2 S^x,  tau^xy = -1/2 S^xy + 3/8 (S^x S^y + S^y S^x),
///
/// S^x and S^xy in the MO basis of @p c. The 3/8 terms are what make the
/// connection orthonormal at second order; without them E_xy is wrong by
/// about its own size (README.md).
///
/// **Any densities.** @p gamma and @p gamma2 are active 1- and 2-RDMs with
/// the closed-shell core of @p ncore orbitals, on the host, row-major as
/// @c ci::Rdm::rdm12 writes them (E = Sum h D + 1/2 Sum (pq|rs) Gamma_pqrs):
/// a state's own, root I's of a state average, or any other, and E_xy is
/// linear in them at a fixed core. So the transition RDMs of a state
/// average's z_c enter added to root I's, as @c mcscf::df_gradient takes
/// them. @p fock_x is F^[x] at the same RDMs, nmo x nmo per coordinate,
/// (p, q, x) at p + nmo (q + nmo x): @c mcscf::Derivatives::fock, from
/// @c mcscf::df_derivatives with weights one-hot on the root.
///
/// The connection terms are gemms over B in the MO basis (every pair, naux
/// nmo^2 doubles) and the stack of tau^x: F from B and G = B_oo Gamma over the
/// occupied orbitals; 2 Sum F tau^xy as @c overlap_hessian at -C sym(F) C^T
/// and the Gram matrix of sym(F) tau^x and tau^y; the F^[x] term the Gram
/// matrix of F^[x] and tau^y; and E^(2) from, per x, B_Q tau^x_occ (one
/// batched gemm), as two more Gram matrices. README.md has the algebra.
///
/// @p molecule is the one both bases were placed on; @p classes are
/// @c primitive_pair_classes of @p basis; @p h the spherical core Hamiltonian;
/// @p b and @p l @c density_fit's B (naux x nao^2) and L. @p c is the nao x
/// nmo orbitals, column-major, core, active, then the rest; every column is
/// read (tau^x reaches the virtuals).
///
/// **A rotation: L_xy's z_o part.** With @p rotation, an antisymmetric
/// nmo x nmo K on the device (column-major, C's columns), the call returns E_xy
/// and each piece at the K-rotated densities K D - D K and K on each index of
/// Gamma, which have a virtual index: d/de of E_xy along C -> C (1 + e K) at
/// fixed MO-basis RDMs, a state average's z_o part of L_xy at K = K(z_o) and
/// the averaged RDMs. @p fock_x is then F^[x] at the same rotated densities
/// (@c mcscf::Derivatives::fock_rotated at that K). The product rule gives each
/// piece from the unrotated machinery: the explicit part at the derivative
/// of D^AO and @ref df_casscf_two_electron's rotation; the two F terms at
/// F_L = F K^T + F at the inner indices transformed by K (as
/// @c mcscf::df_gradient's W takes its z_o) and at @p fock_x; and E^(2) with
/// each of h, B, tau^x and tau^y in turn its commutator with K. V_nn's piece
/// is zero. Without it (the default) nothing of this runs.
///
/// Memory: held for the connection terms, two stacks of 3 n_atom nmo^2
/// doubles, B (naux nmo^2), two of 3 n_atom naux nocc^2 and one of 3 n_atom
/// nmo nocc, beside nocc^4 and the calls' own; past the device's free memory
/// that is Unsupported, before it is allocated. The explicit parts run first
/// and free theirs (@ref df_casscf_two_electron refuses its own).
///
/// Queued on res.stream() and synchronized; the call returns with everything
/// done. None of @p b, @p l, @p c, @p h and @p fock_x may point into
/// res.device_workspace(). The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return The pieces and their sum; InvalidConfig if the molecule and the
///         bases do not share their atoms, if ncore < 0, ncas < 1, @p c is not
///         nao nmo long or ncore + ncas > nmo, if @p h is not nao x nao or
///         @p b not naux nao^2 long, if the RDMs are not ncas^2 and ncas^4
///         long or @p fock_x not 3 n_atom nmo^2, or for any input the integral
///         calls refuse; Unsupported for memory past the device's free bytes or
///         an index past int32_t; or the GpuRuntime error of the first copy,
///         BLAS call or launch that failed.
[[nodiscard]] Result<CasscfExy>
df_casscf_exy(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const Buffers::device<double> &c, int32_t ncore, int32_t ncas,
              std::span<const double> gamma, std::span<const double> gamma2,
              const Buffers::device<double> &fock_x, const double *rotation = nullptr);

/// @brief The state-specific DF-CASSCF nuclear Hessian and its terms, each
/// 3 n_atom x 3 n_atom, x = 3 A + k the coordinate R_{A,k}
/// (hartree / bohr^2), column-major.
struct CasscfTerms {
  /// @brief d^2E/dx dy = E_xy + E_x lambda lambda^y, the two below summed and
  /// symmetrized, (H + H^T) / 2: exactly symmetric.
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the sum before it was symmetrized: E_xy's
  /// rounding and what the CP solve's residual leaves in the response.
  double asymmetry{0.0};
  /// @brief E_xy and its pieces, @ref df_casscf_exy's at the root's RDMs.
  CasscfExy exy;
  /// @brief The response term of eq. ss, Lambda^T E_lambda.: element (x, y)
  /// the right-hand side of x against the solution of y,
  /// E_x lambda . lambda^y = -lambda^x^T E_lambda lambda lambda^y. Not
  /// symmetrized.
  std::vector<double> response;
  /// @brief lambda^x, E_lambda lambda lambda^x = -E_lambda x, in
  /// @c mcscf::zvectors' coordinates (z_o over the packed pairs, z_c off the
  /// root), with the solver's own iterations and residuals.
  mcscf::ZVectors cp;
  /// @brief Per coordinate, the norm of both blocks of
  /// E_lambda lambda lambda^x + E_lambda x, by one product of the unreduced
  /// operator at the solution (the CI rows halved, as zvector's): what the
  /// inner CI solves of the Schur complement cannot hide.
  std::vector<double> residuals;
};

/// @brief The state-specific DF-CASSCF analytic nuclear Hessian at a
/// converged solve (casscf_hessian.tex eq. ss, Algorithm 1):
///
///     d^2E/dx dy = E_xy + E_x lambda lambda^y,   E_lambda lambda lambda^y = -E_lambda y,
///
/// E_xy by @ref df_casscf_exy at the root's RDMs and F^[x];
/// E_lambda x by @c mcscf::df_derivatives (weight 1); lambda^x by
/// @c mcscf::zvectors (Schur) over the CI-relaxed Hessian at the solution,
/// all 3 n_atom columns at once, the CI right-hand sides halved into its
/// convention. The response is assembled as Lambda^T E_lambda. (the
/// solutions against the right-hand sides, orbital and CI rows), not as
/// -Lambda^T E_lambda lambda Lambda, so the operator's scaling cancels.
///
/// **A saddle is refused, not summed.** At a state-specific excited root
/// (or any solution that is not a minimum in the parameters) E_lambda lambda
/// is indefinite and the CP solve is not guaranteed to converge; the
/// Schur complement's inner CI solves stop there without an error. So the
/// solution is checked by one product of the unreduced operator
/// (@c RelaxedHessian::apply_coupled): a column whose residual is above
/// 1e-6 is a Divergence, as is a Divergence of the solve itself.
///
/// The inputs are @c mcscf::df_gradient's: @p molecule the one both bases
/// were placed on, @p classes @c primitive_pair_classes of @p basis, @p h the
/// core Hamiltonian, @p b and @p l @c density_fit's B (naux x nao^2) and L,
/// and @p solution a converged @c mcscf::solve of **one** root over the same
/// fit (DensityFitting::Full): its c (nao x nao, core, active, virtual), its
/// root's vector and energy, and gamma and gamma2 (the active electrons are
/// trace(gamma), of the molecule's multiplicity). Stationarity is assumed,
/// not checked. @p tolerance and @p max_iterations are @c zvectors'.
///
/// The parts run one after another; the peak is the largest of
/// @c df_derivatives' stacks, @ref df_casscf_exy's and the CP solve's
/// (H~'s scratch over 3 n_atom columns), each refused on its own as
/// Unsupported past the device's free memory.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The Hessian and its terms; InvalidConfig if the solution has
///         other than one root, its C is not nao square on @p res's device,
///         ncore < 0, ncas < 1 or ncore + ncas > nao, its RDMs are not
///         ncas^2 and ncas^4 long, trace(gamma) is no electron count of the
///         multiplicity, its vector is not the strings' length, or @p b is not
///         naux nao^2 long, or for any input a part refuses; Divergence when
///         the CP solve meets negative curvature or leaves a residual above
///         1e-6; or the first error of a part (Unsupported for memory, the
///         solve's MaxIterationsExceeded, a GpuRuntime error).
[[nodiscard]] Result<CasscfTerms>
df_casscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
          std::span<const PrimitivePairClass> classes, const BasisSet &aux,
          const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
          const mcscf::Solution &solution, int32_t ncore, int32_t ncas, double tolerance = 1e-10,
          int32_t max_iterations = 200);

/// @brief The z terms of a state-averaged DF-CASSCF Hessian at one root I's
/// multipliers z (casscf_hessian.tex eq. sa, bullets 3 and 4).
struct SaCasscfZTerms {
  /// @brief L_xy's z part, z . d_xy g_SA: E_xy at z's densities (the
  /// z_o-rotated averaged densities and the transition RDMs of z_c), piece by
  /// piece as @ref df_casscf_exy returns them, 3 n_atom x 3 n_atom
  /// column-major, symmetric to rounding; nuclear_repulsion is zero.
  CasscfExy lxy;
  /// @brief L_xlambda, root I's CP right-hand side d_x d_lambda E_I plus
  /// I^x z, a covector per coordinate: size x 3 n_atom column-major on the
  /// device, column x in @c mcscf::ThirdDerivative's packed layout (the
  /// orbital pairs, d_r per root, Theta_sr for r < s at unequal weights), the
  /// CI rows the energy's own. Its dot product with lambda^y in the same
  /// layout is L_xlambda lambda^y, element (x, y).
  Buffers::device<double> lxlambda;
};

/// @brief The z terms of a state-averaged DF-CASSCF analytic Hessian for
/// root @p root, at its Lagrangian multipliers @p z, every piece from a
/// merged part (README.md, "The state-averaged z terms"):
///
///     L_xy z part = df_casscf_exy along K(z_o) at the averaged RDMs
///                   + df_casscf_exy at the transition RDMs of z_c
///     L_xlambda   = root I's df_derivatives right-hand side (one-hot weights)
///                   + I^x z (ThirdDerivative::apply_integrals over
///                     df_integral_derivatives)
///
/// the transition RDMs, and their F^[x], as half the difference of those of
/// the roots c_r + dc_r and c_r - dc_r, dc_r = d_r + Sum_s Theta_sr c_s.
///
/// The inputs are @c mcscf::df_derivatives': @p molecule the one both bases
/// were placed on, @p classes @c primitive_pair_classes of @p basis, @p h the
/// core Hamiltonian, @p b and @p l @c density_fit's B (naux x nao^2) and L,
/// @p c the nao x nao orbitals (core, active, virtual), @p roots the
/// weights.size() >= 2 unit roots of @p nalpha and @p nbeta electrons in the
/// ncas orbitals (root r at r n_det), eigenvectors of the active-space H, at
/// @p weights, all equal or all pairwise different. @p z is root I's
/// multipliers in @c mcscf::ThirdDerivative's layout and the energy's
/// convention (the fixture's z; not @c mcscf::zvector's unscaled CI rows):
/// pairs + n_det nroots doubles, plus nroots (nroots - 1) / 2 at unequal
/// weights.
///
/// The parts run one after another, each freeing what it held: four
/// @c df_derivatives calls, three @c df_casscf_exy calls, then the derivative
/// integrals (3 n_atom nmo^2 (naux + 1) doubles) and the operator. Each part
/// refuses its own memory, Unsupported, before it allocates.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The two terms; InvalidConfig if @p c is not nao square on @p res's
///         device, the molecule and the bases do not share their atoms,
///         ncore < 0, ncas < 1 or ncore + ncas > nao, fewer than two weights,
///         @p root not among them, weights equal for some pairs and not
///         others, @p roots or @p z not of their lengths, or for any input a
///         part refuses; or the first error of a part (Unsupported for
///         memory, a GpuRuntime error).
[[nodiscard]] Result<SaCasscfZTerms>
df_sacasscf_z(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha,
              int32_t nbeta, const Buffers::device<double> &roots, std::span<const double> weights,
              int32_t root, const Buffers::device<double> &z);

/// @brief A state-averaged DF-CASSCF root's analytic nuclear Hessian and its
/// terms (casscf_hessian.tex eq. sa), each 3 n_atom x 3 n_atom, x = 3 A + k
/// the coordinate R_{A,k} (hartree / bohr^2), column-major.
struct SaCasscfTerms {
  /// @brief d^2E_I/dx dy, the sum of the terms below symmetrized,
  /// (H + H^T) / 2: exactly symmetric.
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the sum before it was symmetrized: rounding and
  /// what the solves' residuals leave.
  double asymmetry{0.0};
  /// @brief E_I,xy and its pieces, @ref df_casscf_exy's at root I's RDMs.
  CasscfExy exy;
  /// @brief L_xy's z part and L_xlambda, @ref df_sacasscf_z's at the
  /// gradient's own z.
  SaCasscfZTerms z_terms;
  /// @brief L_xlambda lambda^y, element (x, y); the sum takes it and its
  /// transpose. Not symmetric.
  std::vector<double> lxlambda_lambda;
  /// @brief lambda^x^T (d^2 E_I / d lambda d lambda) lambda^y: root I's own
  /// parameter Hessian between the responses.
  std::vector<double> root_hessian;
  /// @brief lambda^x^T (z . d^2 g_SA / d lambda d lambda) lambda^y: the
  /// third-derivative action (@c mcscf::ThirdDerivative).
  std::vector<double> third;
  /// @brief Eq. rot, the rotations among the roots at equal weights
  /// (@c mcscf::root_rotation's, root I); zero at unequal weights, where the
  /// rotations are in lambda and the sum is complete without it.
  std::vector<double> rotation;
  /// @brief z, root I's gradient multipliers (@c mcscf::zvector over the SA
  /// Hessian, g_I its right-hand side): z_o, z_c and, at unequal weights, the
  /// rotations Theta. The same vector as the energy's (only the equations'
  /// rows are scaled), so it is @ref df_sacasscf_z's z as it comes.
  mcscf::ZVector z;
  /// @brief lambda^x, A lambda^x = -d_x g_SA (@c mcscf::zvectors, Schur), with
  /// the rotations at unequal weights in @c theta, and the solver's own
  /// iterations and residuals.
  mcscf::ZVectors cp;
  /// @brief Per coordinate, the norm of every block of A lambda^x + d_x g_SA
  /// (the CI rows over 2 w_r, the rotation rows over 2 (w_r - w_s), as
  /// zvector's) by one product of the unreduced operator.
  std::vector<double> residuals;
};

/// @brief Root @p root's state-averaged DF-CASSCF analytic nuclear Hessian
/// at a converged average (casscf_hessian.tex eq. sa):
///
///     d^2E_I/dx dy = L_xy + L_xlambda lambda^y + lambda^x^T L_lambday
///                    + lambda^x^T L_lambdalambda lambda^y  (+ eq. rot at equal weights),
///
/// L_xy = E_I,xy (@ref df_casscf_exy at root I's RDMs) + its z part and
/// L_xlambda (@ref df_sacasscf_z, at z from @c mcscf::zvector);
/// L_lambdalambda = root I's parameter Hessian (@c RelaxedHessian's
/// unreduced product at weights one-hot on I, the CI rows scaled back to the
/// energy's) + the third-derivative action (@c mcscf::ThirdDerivative at z);
/// lambda^x by @c mcscf::zvectors (Schur) over the SA Hessian on
/// @c mcscf::df_derivatives' SA right-hand sides, every coordinate at once.
/// At equal weights the rotations among the roots are not parameters and
/// @c mcscf::root_rotation's eq. rot (factor 2 included) is added; at
/// unequal weights they are in z and lambda (Theta) and it is not.
///
/// **A degeneracy is refused.** Two roots under 1e-8 hartree apart are
/// refused before anything runs; at equal weights @c mcscf::root_rotation
/// refuses a mixing rate max |theta^x_sr| above 10 per bohr, and at unequal
/// weights the same bound is put on lambda^x's rotations. **A saddle is
/// refused** as in @ref df_casscf: a lambda^x column whose unreduced residual
/// is above 1e-6 is a Divergence.
///
/// The inputs are @ref df_casscf's, @p solution a converged
/// @c mcscf::solve of two roots or more over the same fit at @p weights
/// (positive, all equal or all pairwise different, one per root; a Solution
/// does not carry them), @p root the state. Stationarity is assumed, not
/// checked. @p tolerance and @p max_iterations are the solves'.
///
/// The parts run one after another, each freeing what it held; the peak is
/// @ref df_sacasscf_z's derivative integrals (3 n_atom nmo^2 (naux + 1)
/// doubles) or the CP solve's scratch, each refused on its own as
/// Unsupported past the device's free memory.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The Hessian and its terms; InvalidConfig if the solution has
///         fewer than two roots, @p weights are not one positive weight per
///         root, @p root is not one of them, C is not nao square on @p res's
///         device, ncore < 0, ncas < 1 or ncore + ncas > nao, the RDMs are
///         not ncas^2 and ncas^4 long, trace(gamma) is no electron count of
///         the multiplicity, the vectors are not the roots', @p b is not
///         naux nao^2 long, or a part refuses (weights equal for some pairs
///         only); Numerical at a degeneracy; Divergence when a solve meets
///         negative curvature or leaves a residual above 1e-6; or the first
///         error of a part.
[[nodiscard]] Result<SaCasscfTerms>
df_sacasscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, std::span<const double> weights, int32_t root,
            int32_t ncore, int32_t ncas, double tolerance = 1e-10, int32_t max_iterations = 200);

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
