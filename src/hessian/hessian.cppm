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
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.response.cphf;
import free_breakfast.mcscf.solve;
import free_breakfast.response.zvector;

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
/// @c df_jk_derivative's own scratch (@ref df_rhf_response_footprint has all
/// of it); past the memory budget (res.memory_budget()) that is Unsupported,
/// naming the bytes, before anything is allocated.
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
///         the memory budget or a stack past int32_t; the CPHF's
///         MaxIterationsExceeded or Divergence; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<RhfResponse>
df_rhf_response(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const Buffers::device<double> &b, const double *l, const Buffers::device<double> &c,
                std::span<const double> mo_energies, int32_t nocc, double tolerance = 1e-10,
                int32_t max_iterations = 100);

/// @brief What @ref df_rhf_response holds over @p basis and @p aux with @p nmo
/// orbitals, @p nocc of them occupied:
///
/// - persistent: U^x, @c response::cphf's solution, returned;
/// - scratch: the three MO stacks with one coordinate's J/K buffers, held
///   throughout; beside them, in turn, S^x (@c overlap_derivative), then
///   T^x and V^x and J^(x), K^(x), and the right-hand sides beside the
///   CPHF's own;
/// - workspace: the largest of the MO carry's X C_o, @c df_jk_derivative's,
///   one coordinate's @c df_coulomb and two-factor @c df_exchange, the
///   CPHF's and the closing products' carves.
///
/// Nothing pinned. Empty for nocc not in [1, nmo), or no AOs or fitting
/// functions.
[[nodiscard]] MemoryFootprint df_rhf_response_footprint(const BasisSet &basis, const BasisSet &aux,
                                                        int32_t nmo, int32_t nocc);

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
/// the fitted densities, and the fit's cross terms (docs/hessian/hessian.tex
/// eqs. dfj, dfk). With T_mn,P = (mn|P), M = (P|Q) = L L^T, d = M^-1 T^T vec D,
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
/// **The exchange cross term is held whole only when the memory holds it**
/// (3 n_atom nocc^2 naux doubles): R^x for the three coordinates of one
/// atom is one atom block, and one walk over the aux atoms
/// (@c three_center_derivative's blocks against @c metric_derivative's M1)
/// builds a batch of them, every atom's when they fit the budget (one walk),
/// else two batches of half as many as fit, contracted one pair of batches
/// at a time: n_b (n_b + 1) / 2 walks for n_b batches, n_atom (n_atom + 1) / 2
/// at one atom a batch. The Coulomb vectors L^-1 g^x come from the same
/// blocks.
///
/// @p basis and @p aux are made by @c place over one molecule, and @p classes
/// are @c primitive_pair_classes of @p basis. @p b is B as @c density_fit
/// leaves it, naux x nao^2 at Q + naux (n + nao m), and @p l is its L
/// (column-major, the lower triangle read). @p c is the nao x nmo orbitals,
/// column-major, of which only the first @p nocc are read.
///
/// The call holds the fitted pairs c (naux nocc^2), M1 (3 naux^2), one aux
/// atom's derivative block (3 nao^2 nblk), two 3 n_atom x 3 n_atom products
/// and at least two atom blocks with their Coulomb vectors (3 naux
/// (nocc^2 + 1) each); before that, the fitted densities of the explicit
/// part (naux nao^2, like @c gradient::df_rhf's). The budget is the memory
/// budget (res.memory_budget()), or @p max_bytes when that is less (0: no
/// cap); past it, at two atom blocks, the call is Unsupported, naming the
/// bytes, before anything is allocated. The batch, the call's last
/// allocation, takes what the memory budget leaves after the explicit part
/// (never past @p max_bytes), less the workspace's growth still to come and
/// @c kPoolHeadroomBytes, up to every atom; batching changes the result only
/// by rounding.
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
///         budget or an index past int32_t; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<RhfTwoElectron>
df_rhf_two_electron(DeviceResources &res, const BasisSet &basis,
                    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                    const Buffers::device<double> &b, const double *l,
                    const Buffers::device<double> &c, int32_t nocc, size_t max_bytes = 0);

/// @brief What @ref df_rhf_two_electron holds over @p basis and @p aux with
/// @p nocc occupied orbitals, its batch at its least (two atom blocks, one
/// for a single atom):
///
/// - scratch: the fitted pairs and the twos, beside the explicit part's
///   buffer and then the cross terms' (M1, one aux atom's derivative block,
///   the two products and the batch), the larger;
/// - workspace: the largest of @c b_mo's, @c three_center_hessian's,
///   @c metric_hessian's, @c metric_derivative's and, over the aux atoms,
///   @c three_center_derivative's and its half-transform's carves.
///
/// The batch grows with the memory budget, so this is the least the call runs
/// with. Nothing persistent (the parts are on the host) and nothing pinned.
/// Empty if the bases do not share their atoms or @p nocc is not in [1, nao].
[[nodiscard]] MemoryFootprint df_rhf_two_electron_footprint(const BasisSet &basis,
                                                            const BasisSet &aux, int32_t nocc);

/// @brief The two-electron explicit part of a DF-CASSCF nuclear Hessian
/// (docs/hessian/hessian.tex eq. df2) and its pieces, each 3 n_atom x
/// 3 n_atom, x = 3 A + k the coordinate R_{A,k} (hartree / bohr^2), symmetric
/// to rounding.
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
/// fixed orbitals and RDMs (docs/hessian/hessian.tex eq. df2): with
/// T_a,P = (a|P) over AO pairs a, M = (P|Q) = L L^T, the fitted pairs
/// c = T M^-1 and R^x = T^x - c M^x, for the pair-space density Gamma_ab of
/// E_2 = 1/2 Sum Gamma_ab (T M^-1 T^T)_ab,
///
///     E_2^xy = Sum Gam~ T^xy - 1/2 Sum W^ M^xy + Sum_ab Gamma_ab [R^x M^-1 R^y^T]_ab
///
/// the first two by @c three_center_hessian and @c metric_hessian at
/// @c mcscf::df_gradient's Gam and W (at these RDMs, no Lagrangian), the
/// cross term split as the design's section df so no pair-space Gamma is
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
/// half-transform (@ref df_casscf_two_electron_footprint has all of it). Past
/// the memory budget (res.memory_budget()) that is Unsupported, naming the
/// bytes, before anything is allocated.
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
///         integral calls refuse; Unsupported for memory past the memory
///         budget or an index past int32_t; or the GpuRuntime error of the
///         first copy, BLAS call or launch that failed.
[[nodiscard]] Result<CasscfTwoElectron> df_casscf_two_electron(
    DeviceResources &res, const BasisSet &basis, std::span<const PrimitivePairClass> classes,
    const BasisSet &aux, const Buffers::device<double> &b, const double *l,
    const Buffers::device<double> &c, int32_t ncore, int32_t ncas, std::span<const double> gamma,
    std::span<const double> gamma2, const double *rotation = nullptr);

/// @brief What @ref df_casscf_two_electron holds over @p basis and @p aux with
/// @p ncore core and @p ncas active orbitals, with a rotation (@p rotated) or
/// without, all of it scratch or the workspace:
///
/// - scratch: the fitted occupied pairs, the weights and Gamma (with a
///   rotation [C_o | C~_o] and its fitted pairs too; one double without),
///   beside the explicit part's buffer and then the cross terms' (M1, one aux
///   atom's derivative block, two atom blocks, K's half-transform, the active
///   blocks, the Coulomb vectors and the products), the larger;
/// - workspace: the largest of @c b_mo's (over 2 nocc orbitals with a
///   rotation), @c three_center_hessian's, @c metric_hessian's,
///   @c metric_derivative's and, over the aux atoms,
///   @c three_center_derivative's and its half-transform's carves.
///
/// Nothing persistent (the pieces are on the host) and nothing pinned. Empty
/// for the counts it refuses.
[[nodiscard]] MemoryFootprint df_casscf_two_electron_footprint(const BasisSet &basis,
                                                               const BasisSet &aux, int32_t ncore,
                                                               int32_t ncas, bool rotated = false);

/// @brief E_xy, the fixed-parameter second derivative of a DF-CASSCF energy
/// along the symmetric (Lowdin) connection (docs/hessian/hessian.tex eq.
/// exy), and its pieces, each 3 n_atom x 3 n_atom, x = 3 A + k the coordinate
/// R_{A,k} (hartree / bohr^2), column-major and symmetric to rounding.
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

/// @brief E_xy of a DF-CASSCF at fixed orbitals and RDMs
/// (docs/hessian/hessian.tex eq. exy): the explicit part (Sum D^AO h^xy,
/// V_nn^xy and the fitted eq. df2) plus the connection C T(x),
/// T = (C^T S(x) C)^-1/2, to second order:
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
/// nmo nocc, beside nocc^4 and the calls' own (@ref df_casscf_exy_footprint
/// has all of it); past the memory budget (res.memory_budget()) that is
/// Unsupported, before it is allocated. The explicit parts run first
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
///         calls refuse; Unsupported for memory past the memory budget or an
///         index past int32_t; or the GpuRuntime error of the first copy,
///         BLAS call or launch that failed.
[[nodiscard]] Result<CasscfExy>
df_casscf_exy(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const Buffers::device<double> &c, int32_t ncore, int32_t ncas,
              std::span<const double> gamma, std::span<const double> gamma2,
              const Buffers::device<double> &fock_x, const double *rotation = nullptr);

/// @brief What @ref df_casscf_exy holds over @p basis and @p aux with @p ncore
/// core and @p ncas active orbitals, nmo = nao, with a rotation (@p rotated)
/// or without, all of it scratch or the workspace: the occupied D and Gamma,
/// held throughout; beside them in turn D^AO with
/// @c one_electron_hessian_footprint's scratch, then
/// @ref df_casscf_two_electron_footprint, then the connection terms' held
/// buffers (with S^x, @c one_electron_derivative_footprint, until tau is
/// made, and then -C sym(F) C^T with @c overlap_hessian's); the workspace the
/// largest of their carves and @c b_mo's over every pair. Nothing pinned.
/// Empty for the counts it refuses.
[[nodiscard]] MemoryFootprint df_casscf_exy_footprint(const BasisSet &basis, const BasisSet &aux,
                                                      int32_t ncore, int32_t ncas,
                                                      bool rotated = false);

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
  /// @c response::zvectors' coordinates (z_o over the packed pairs, z_c off the
  /// root), with the solver's own iterations and residuals.
  response::ZVectors cp;
  /// @brief Per coordinate, the norm of both blocks of
  /// E_lambda lambda lambda^x + E_lambda x, by one product of the unreduced
  /// operator at the solution (the CI rows halved, as zvector's): what the
  /// inner CI solves of the Schur complement cannot hide.
  std::vector<double> residuals;
};

/// @brief The state-specific DF-CASSCF analytic nuclear Hessian at a
/// converged solve (docs/hessian/hessian.tex eq. ss, Algorithm ss):
///
///     d^2E/dx dy = E_xy + E_x lambda lambda^y,   E_lambda lambda lambda^y = -E_lambda y,
///
/// E_xy by @ref df_casscf_exy at the root's RDMs and F^[x];
/// E_lambda x by @c mcscf::df_derivatives (weight 1); lambda^x by
/// @c response::zvectors (Schur) over the CI-relaxed Hessian at the solution,
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
/// Unsupported past the memory budget (@ref df_casscf_footprint has the
/// whole).
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
///         the CP solve meets negative curvature, runs out of products, finds
///         H~ singular by its probe column (README.md "A singular H~") or
///         leaves a residual above 1e-6; or the first error of a part
///         (Unsupported for memory, a GpuRuntime error).
[[nodiscard]] Result<CasscfTerms>
df_casscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
          std::span<const PrimitivePairClass> classes, const BasisSet &aux,
          const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
          const mcscf::Solution &solution, int32_t ncore, int32_t ncas, double tolerance = 1e-10,
          int32_t max_iterations = 200);

/// @brief What @ref df_casscf holds over @p basis and @p aux with @p ncore
/// core and @p ncas active orbitals over the determinants of @p nalpha and
/// @p nbeta active electrons, nmo = nao; @p syevd_lwork is @c syevd_lwork at
/// order ncas (@c mcscf::df_derivatives_footprint's):
///
/// - persistent: lambda^x, CasscfTerms::cp, returned;
/// - scratch: the CP step's (the strings, @c mcscf::df_derivatives_footprint's
///   F^[x] and right-hand sides held to the end, the integrals and the Fock,
///   the sigma with its apply scratch grown to Layout::full, H and H~ over
///   3 n_atom + 1 block columns, the right-hand sides, the probed
///   @c response::zvectors and the check's block product), then C's copy
///   beside @ref df_casscf_exy_footprint, then the response's n_x^2;
/// - workspace: the largest of the calls'.
///
/// H~'s block products build their J and K at the least width
/// (@c response::hessian_apply_footprint); the width grows with the memory
/// budget, so this is the least the call runs with. Nothing pinned. Empty for
/// the counts it refuses.
[[nodiscard]] MemoryFootprint df_casscf_footprint(const BasisSet &basis, const BasisSet &aux,
                                                  int32_t ncore, int32_t ncas, int32_t nalpha,
                                                  int32_t nbeta, size_t syevd_lwork);

/// @brief lambda^x alone: @ref df_casscf's CP-CASSCF step, E_lambda lambda
/// lambda^x = -E_lambda x for every coordinate, the same solve and the same
/// check on the unreduced equations, without E_xy or the response term.
/// What @ref df_nevpt2_tangents reads; @ref df_casscf's CasscfTerms::cp is
/// the same solution.
///
/// The inputs, the refusals and the memory are @ref df_casscf's, its
/// messages naming hessian::df_casscf_cp.
///
/// @return lambda^x in @c response::zvectors' coordinates (z_o over the packed
///         pairs, z_c, the CI response itself, off the root), with the
///         solver's own iterations and residuals; or @ref df_casscf's errors.
[[nodiscard]] Result<response::ZVectors>
df_casscf_cp(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, int32_t ncore, int32_t ncas, double tolerance = 1e-10,
             int32_t max_iterations = 200);

/// @brief A core or virtual pair below the NEVPT2 gradient's degenerate-pair
/// rule along one coordinate, which @ref df_nevpt2_tangents gave kappa_oo 0:
/// the first-order change of the pair's 2 x 2 block of f along it.
struct DegeneratePair {
  /// @brief The coordinate x = 3 A + k.
  int32_t coordinate{0};
  /// @brief The pair's orbitals, p > q, both core or both virtual (C's
  /// columns).
  int32_t p{0};
  int32_t q{0};
  /// @brief eps_p - eps_q (hartree), below 1e-6 in magnitude.
  double gap{0.0};
  /// @brief f^(x)_pq, the change of f the rest of t_x makes.
  double f{0.0};
  /// @brief (f^(x)_pp - f^(x)_qq) / 2, the change of the pair's split.
  double split{0.0};
};

/// @brief t_x, the first-order change of every input of a single-state
/// NEVPT2 energy along each nuclear coordinate x = 3 A + k at a converged
/// DF-CASSCF reference (docs/nevpt2/nevpt2.tex Eqs. koo, tx), handed to
/// @p consume a chunk of coordinates at a time:
///
///     h_dot^x = h1^x + kappa^xT h_MO + h_MO kappa^x
///     B_dot^x = W^x + kappa^xT B_Q + B_Q kappa^x     (each fitted index Q)
///     c_dot^x = c^x
///     kappa^x = kappa_nr^x + kappa_oo^x,  kappa_oo^x_pq = -f^(x)_pq / (eps_p - eps_q),
///
/// (h1^x, W^x) the derivative integrals with the connection
/// (@c mcscf::df_integral_derivatives), (kappa_nr^x, c^x) the CP-CASSCF
/// response lambda^x of @p cp, and kappa_oo^x the core-core and
/// virtual-virtual rotations that keep the core and virtuals canonical
/// (eq. koo): f^(x) the change of f's two inactive blocks along everything
/// above but kappa_oo (@c nevpt2::builders_tangent), eps f's diagonal. So t_x
/// is (h_MO, B_MO, c) of the re-solved, re-canonicalized CASSCF
/// differentiated along x, its active orbitals turned by no active-active
/// rotation. C -> C (1 + kappa), kappa antisymmetric.
///
/// **A degenerate pair.** Below the gradient's degenerate-pair rule (z^c's:
/// |eps_p - eps_q| < 1e-6) kappa_oo^x_pq is not a division the state
/// determines. For SC (@p pc false) it is 0 when |f^(x)_pq| <= 1e-8, the
/// pair split across irreps, and past that Numerical, naming the pair and
/// the coordinate. For PC, which is invariant to the pair's rotation, every
/// such pair and coordinate is recorded, kappa_oo^x_pq left 0, and the call
/// goes on: @ref df_pc_nevpt2 takes the pair's off-diagonal f itself.
///
/// @p consume(x0, count, h_dot, b_dot, c_dot, kappa) sees the coordinates
/// [x0, x0 + count), in order, each once, on the device and valid until it
/// returns, all in the MO basis of the solution's C, in the layout
/// @c nevpt2::builders_tangent reads: @p h_dot count nmo x nmo symmetric
/// (slab j at j nmo^2), @p b_dot count naux x nmo^2 ((Q, p, q) at
/// Q + naux (p + nmo q), as @c b_mo leaves B, slab j at j naux nmo^2),
/// @p c_dot count n_det, and @p kappa count nmo x nmo antisymmetric, the
/// total rotation. Work it queues on res.stream() is ordered before the
/// next chunk; its error is returned as it stands and ends the call.
///
/// The inputs are @ref df_casscf's, @p solution's core and virtuals
/// canonical in f (as the Casscf step leaves them), and @p cp lambda^x at
/// the same solution (@ref df_casscf_cp, or @ref df_casscf's
/// CasscfTerms::cp). All-electron: the core NEVPT2 correlates is the
/// whole one.
///
/// W^x is built for as many coordinates at once as a third of the budget
/// holds: of the memory budget (res.memory_budget()) less the pool's
/// fragmentation headroom, never what one coordinate's part needs (below),
/// never less than one coordinate's own; or of @p max_bytes when given, with
/// @c mcscf::df_integral_derivatives' own refusal past it. Each chunk is then
/// taken in parts as wide as the memory budget at its call holds (B_dot, as
/// large as W^x, and @c nevpt2::builders_tangent's arrays per coordinate),
/// Unsupported when one coordinate's does not fit. Held throughout: h_MO and
/// B_MO (naux nmo^2). The workspace is first grown to
/// @ref df_nevpt2_tangents_footprint's. Queued on res.stream() with res.blas() (host
/// pointer mode); it synchronizes many times. None of @p b and @p l may
/// point into res.device_workspace(). The device res.dev_idx() names is
/// made current for the call and the previous one restored.
///
/// @return The pairs PC recorded, in coordinate order (empty for SC);
///         InvalidConfig for @ref df_casscf's input refusals, a null
///         @p consume, @p cp not of 3 n_atom columns over the packed pairs
///         and the determinants, or core or virtuals not canonical (an
///         off-diagonal |f_pq| above 1e-8); Numerical for SC at a degenerate
///         pair; Unsupported for naux nmo past int32_t or memory past the
///         budget; or the first error of a part or of @p consume.
[[nodiscard]] Result<std::vector<DegeneratePair>> df_nevpt2_tangents(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    int32_t ncore, int32_t ncas, const response::ZVectors &cp, bool pc,
    const std::function<Status(int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
                               const double *c_dot, const double *kappa)> &consume,
    size_t max_bytes = 0);

/// @brief What @ref df_nevpt2_tangents holds over @p basis and @p aux with
/// @p ncore core and @p ncas active orbitals over the determinants of
/// @p nalpha and @p nbeta electrons, at its least, all of it scratch beside
/// the workspace: the strings, h_MO and the MO factor, held throughout;
/// beside them @c mcscf::df_integral_derivatives_footprint, and beside what
/// that holds at its first chunk one coordinate's part (h_dot, B_dot and
/// kappa, @c nevpt2::builders_tangent_footprint's, kappa_oo). The parts and
/// W^x's chunk grow with the budget past this. The consumer's memory is its
/// own. Empty for bases that are not one placement or counts it refuses.
[[nodiscard]] MemoryFootprint df_nevpt2_tangents_footprint(const BasisSet &basis,
                                                           const BasisSet &aux, int32_t ncore,
                                                           int32_t ncas, int32_t nalpha,
                                                           int32_t nbeta);

/// @brief Sum_pq h_bar_pq h^xy_pq + Sum_L,pq Lambda_L,pq B^xy_L,pq for every
/// pair of nuclear coordinates x = 3 A + k, y: the second derivative of the
/// functional <h_bar, h_MO> + <Lambda, B_MO> of the density-fitted MO
/// integrals at a fixed symmetric adjoint (h_bar, Lambda), the E_corr + Phi
/// share of an NEVPT2 Hessian's skeleton L_AB without its integral curvature
/// (docs/nevpt2/nevpt2.tex eq. Nab, first term).
///
/// Along the path (README.md "The DF second derivative at an MO adjoint")
///
///     C(R) = C (C^T S(R) C)^-1/2,   B_MO(R) = (L^-1 M(R) L^-T)^-1/2 L^-1 (P|pq)(R)
///
/// over the orbitals C(R) and h_MO(R) = C(R)^T h(R) C(R): the Lowdin
/// connection to second order (tau^xy = -1/2 S^xy + 3/8 (S^x S^y + S^y S^x),
/// as @ref df_casscf_exy) and the fitted index's symmetric gauge to second
/// order, whose first order is mcscf::df_integral_derivatives' W^x. The
/// second-derivative integrals are contracted once each at the adjoint's AO
/// forms (C h_bar C^T, L^-T C Lambda C^T, -1/2 L^-T sym(G) L^-1, the
/// overlap's -1/2 C sym(X) C^T), the rest is Gram matrices of
/// per-coordinate stacks over df_integral_derivatives' (h1^x, tau^x, W^x),
/// the products against mu^x taken by M^x's atom blocks.
///
/// The inputs are @ref df_casscf_exy's (the molecule both bases were placed
/// on, @p classes of @p basis, the AO core Hamiltonian @p h, @p b and @p l as
/// @c density_fit leaves them) with @p c any orthonormal orbitals over
/// @p basis, nmo = nao. @p h_bar is nmo x nmo column-major and @p lambda
/// naux x nmo^2, (L, p, q) at L + naux (p + nmo q) (nevpt2::ScAdjoint's
/// layouts), both on the device and symmetric in (p, q); neither is checked
/// for it.
///
/// Held for the call: B and Lambda over every MO pair and one more such
/// scratch (3 naux nmo^2), two 3 n_atom nmo^2 stacks and a few naux^2 and
/// nmo^2 matrices (none per coordinate);
/// W^x is df_integral_derivatives' chunks, as many coordinates at once as
/// the rest of the budget holds (the memory budget, res.memory_budget(), at
/// the call, or @p max_bytes when that is less). Past the budget with one coordinate's
/// W^x and df_integral_derivatives' own buffers the call is Unsupported,
/// naming the bytes, before anything is allocated. Queued on res.stream()
/// with res.blas(); the derivative calls synchronize it. None of @p b,
/// @p l, @p h_bar and @p lambda may point into res.device_workspace().
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The 3 n_atom x 3 n_atom matrix, element x + 3 n_atom y (symmetric
///         to rounding); InvalidConfig if the molecule and bases are not one
///         placement, @p c or @p h is not nao x nao, @p b not naux nao^2,
///         or @p l, @p h_bar or @p lambda is null; Unsupported for memory
///         past the budget or naux nmo^2, naux^2 or 3 n_atom nmo^2
///         past int32_t; or the first error of a derivative call, copy or
///         BLAS call.
[[nodiscard]] Result<std::vector<double>>
df_adjoint_xy(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const SphericalMatrixAO &c, const double *h_bar, const double *lambda,
              size_t max_bytes = 0);

/// @brief What @ref df_adjoint_xy holds over @p basis and @p aux at its least,
/// all of it scratch beside the workspace: the held buffers and the AO
/// matrix throughout; beside them in turn @c metric_derivative_footprint's,
/// @c mcscf::df_integral_derivatives_footprint, three
/// @c one_electron_hessian_footprint's, @c metric_hessian_footprint's and
/// @c three_center_hessian_footprint's. W^x's chunk grows with the budget
/// past this. Empty for bases that are not one placement.
[[nodiscard]] MemoryFootprint df_adjoint_xy_footprint(const BasisSet &basis, const BasisSet &aux);

/// @brief The z terms of a state-averaged DF-CASSCF Lagrangian's Hessian at
/// its multipliers z (docs/hessian/hessian.tex eq. sa, bullets 3 and 4; for
/// a QD-NEVPT2 state, docs/nevpt2/nevpt2.tex Eq. qd-Lxx's third derivative).
struct SaCasscfZTerms {
  /// @brief L_xy's z part, z . d_xy g_SA: E_xy at z's densities (the
  /// z_o-rotated averaged densities and the transition RDMs of z_c), piece by
  /// piece as @ref df_casscf_exy returns them, 3 n_atom x 3 n_atom
  /// column-major, symmetric to rounding; nuclear_repulsion is zero.
  CasscfExy lxy;
  /// @brief L_xlambda, the CP right-hand side d_x d_lambda E at the call's
  /// right-hand side weights (none without them) plus I^x z, a covector per
  /// coordinate: size x 3 n_atom column-major on the device, column x in
  /// @c mcscf::ThirdDerivative's packed layout (the orbital pairs, d_r per
  /// root, Theta_sr for every r < s unless all weights are equal), the CI
  /// rows the energy's own. Its dot product with lambda^y in the same layout
  /// is L_xlambda lambda^y, element (x, y).
  Buffers::device<double> lxlambda;
  /// @brief Lambda^T (z . d2 g_SA / dlambda dlambda) Lambda over the call's k
  /// columns Lambda, @c mcscf::ThirdDerivative's action: k x k column-major
  /// on the host, (i, j) = T(lambda_i, lambda_j, z); empty for k = 0.
  std::vector<double> third;
};

/// @brief The z terms of a state-averaged DF-CASSCF Lagrangian's analytic
/// Hessian at its multipliers @p z = (Z, {z_r}, Theta): one root's
/// (@ref df_sacasscf's), or a QD-NEVPT2 Lagrangian's (nevpt2::qd_sc_lagrangian
/// and qd_pc_gradient's multipliers, docs/nevpt2/nevpt2.tex Eq. qd-Lxx's
/// third derivative, its CI curvature the root set's, Eq. qd-curv), every
/// piece from a merged part (README.md, "The state-averaged z terms"):
///
///     L_xy z part = df_casscf_exy along K(z_o) at the averaged RDMs
///                   + df_casscf_exy at the transition RDMs of z_c
///     L_xlambda   = the df_derivatives right-hand side at @p rhs_weights
///                   (one-hot for root I's E_I, rho = diag W for a QD
///                   Lagrangian's E_cas; none when empty)
///                   + I^x z (mcscf::df_integral_derivatives, through
///                     ThirdDerivative::apply_integrals)
///     third       = Lambda^T ThirdDerivative(z).apply(Lambda), over @p k
///                   columns @p lambda (none for k = 0)
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
/// @p weights (equal for some pairs only is taken). @p rhs_weights are the
/// right-hand side's, one per root as @c mcscf::df_derivatives takes them
/// (each >= 0, summing to 1), or empty for none. @p z is the multipliers in
/// @c mcscf::ThirdDerivative's layout and the energy's convention, which is
/// @c response::zvector's (z_o, z_c, Theta) as it comes (zvector scales its
/// equations' CI rows, not its unknowns): pairs + n_det nroots doubles, plus
/// nroots (nroots - 1) / 2 unless every weight is equal (an equal pair's
/// Theta zero, as zvector leaves it). At all-equal weights a z that carries
/// them too makes every rotation a parameter (ThirdDerivative's
/// equal_rotations): a QD-NEVPT2 x_A moves the equal-weight rotations (Eq.
/// qd-theta), so its L_xlambda's rotation rows and third derivative need
/// them. Its CI part is taken off every root here. @p lambda is @p k
/// columns in z's layout (size x k column-major on the device), null for
/// k = 0.
///
/// The parts run one after another, each freeing what it held: four
/// @c df_derivatives calls, three @c df_casscf_exy calls, then the derivative
/// integrals (nmo^2 naux doubles per coordinate, for as many coordinates at
/// once as the memory budget holds) and the operator. Each part refuses its
/// own memory, Unsupported, before it allocates.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The two terms; InvalidConfig if @p c is not nao square on @p res's
///         device, the molecule and the bases do not share their atoms,
///         ncore < 0, ncas < 1 or ncore + ncas > nao, fewer than two weights,
///         @p rhs_weights neither empty nor one per root, @p roots or @p z
///         not of their lengths, a negative @p k or a null @p lambda with
///         k > 0, or for any input a part refuses
///         (df_derivatives' of @p rhs_weights); or the first error of a part
///         (Unsupported for memory, a GpuRuntime error).
[[nodiscard]] Result<SaCasscfZTerms>
df_sacasscf_z(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha,
              int32_t nbeta, const Buffers::device<double> &roots, std::span<const double> weights,
              std::span<const double> rhs_weights, const Buffers::device<double> &z, int32_t k = 0,
              const double *lambda = nullptr);

/// @brief What @ref df_sacasscf_z holds as @ref df_sacasscf calls it, with a
/// right-hand side's weights and k = 0, over @p basis and @p aux with
/// @p ncore core and @p ncas active orbitals and @p nroots roots over the
/// determinants of @p nalpha and @p nbeta electrons, nmo = nao; @p rotating
/// whether the weights rotate (z then carries every rotation), @p syevd_lwork
/// as @ref df_casscf_footprint's:
///
/// - persistent: L_xlambda, returned;
/// - scratch: the strings and C's copy, held throughout; beside them in turn
///   the right-hand side's @c mcscf::df_derivatives_footprint, L_xy's z_o
///   part (K, the RDMs, F^[x] at the rotated densities and
///   @ref df_casscf_exy_footprint along K), its z_c part (c +- dc and dc, the
///   RDMs and E_xy at each, F^[x] beside it), and the third-derivative
///   operator (@c mcscf::third_derivative_footprint) with Y beside
///   @c mcscf::df_integral_derivatives_footprint;
/// - workspace: the largest of the calls'.
///
/// The derivative integrals' chunk grows with the memory budget, so this is
/// the least the call runs with. Nothing pinned. Empty for the counts it
/// refuses.
[[nodiscard]] MemoryFootprint df_sacasscf_z_footprint(const BasisSet &basis, const BasisSet &aux,
                                                      int32_t ncore, int32_t ncas, int32_t nalpha,
                                                      int32_t nbeta, int32_t nroots, bool rotating,
                                                      size_t syevd_lwork);

/// @brief The multiplier terms of a single-state Lagrangian's Hessian at
/// given multipliers (Z, z) (docs/nevpt2/nevpt2.tex eq. hess): the three places
/// Z . g, g the CASSCF energy's gradient in its parameters, enters
/// L_AB, L_Ax and L_xx.
struct CasscfZTerms {
  /// @brief Z . d2 g / dR dR: E_xy at Z's densities (the Z-rotated root's
  /// densities and the transition RDMs of z), piece by piece as
  /// @ref df_casscf_exy returns them, 3 n_atom x 3 n_atom column-major,
  /// symmetric to rounding; nuclear_repulsion is zero.
  CasscfExy lxy;
  /// @brief Z . d2 g / dR dx, I^x Z alone (no d_x d_lambda E: that is
  /// @ref df_casscf's), a covector per coordinate: size x 3 n_atom
  /// column-major on the device, column x in @c mcscf::ThirdDerivative's
  /// packed layout (the orbital pairs, then the root's CI). Its dot product
  /// with x^y in the same layout is element (x, y) of L_Ax x^y's Z part.
  Buffers::device<double> lxlambda;
  /// @brief Lambda^T (Z . d2 g / dx dx) Lambda over the call's k columns
  /// Lambda: k x k column-major on the host, (i, j) =
  /// T(lambda_i, lambda_j, Z); empty for k = 0.
  std::vector<double> third;
};

/// @brief The Z terms of a single-state DF-CASSCF reference's Lagrangian
/// Hessian at caller-supplied multipliers @p z (a NEVPT2 gradient's (Z, z),
/// say): @ref df_sacasscf_z's two terms at one root of weight 1, without
/// the root's own right-hand side, and the third derivative's action over a
/// block of response vectors (README.md, "The single-state Z terms"):
///
///     lxy       = df_casscf_exy along K(Z) + at the transition RDMs of z
///     lxlambda  = I^x Z (mcscf::df_integral_derivatives through
///                 ThirdDerivative::apply_integrals)
///     third     = Lambda^T ThirdDerivative(Z).apply(Lambda)
///
/// Computes no response: Lambda is the caller's (@ref df_casscf_cp's
/// lambda^x, say).
///
/// The inputs are @ref df_casscf's: @p solution a one-root DF-CASSCF over
/// the same fit, its root an eigenvector of the active-space H at its C.
/// @p z is pairs + n_det doubles on the device in @c response::zvector's
/// coordinates (z_o over the packed pairs, then z, c -> c + z), its CI part
/// taken off the root here. @p lambda is @p k such columns (size x k,
/// column-major, device), null when @p k is 0.
///
/// The parts run one after another as @ref df_sacasscf_z's do, less its
/// one-hot @c df_derivatives call; Y (size x k) is held for the action.
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The three terms; InvalidConfig for @ref df_casscf's input
///         refusals, @p z not of its length, a negative @p k, or a null
///         @p lambda with k > 0, or for any input a part refuses; or the
///         first error of a part (Unsupported for memory, a GpuRuntime
///         error).
[[nodiscard]] Result<CasscfZTerms>
df_casscf_z(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
            const Buffers::device<double> &z, int32_t k = 0, const double *lambda = nullptr);

/// @brief The SC-NEVPT2 analytic nuclear Hessian and its parts, each k x k
/// in the call's k coordinates (hartree / bohr^2), column-major.
struct ScNevpt2Terms {
  /// @brief d^2E/dq_i dq_j of E_cas + E_corr, the parts below summed and
  /// symmetrized, (H + H^T) / 2: exactly symmetric.
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the sum before it was symmetrized.
  double asymmetry{0.0};
  /// @brief The translational sum rule, max |Q^T H Q T| over the three
  /// rigid translations T, on a full set of coordinates (k = 3 n_atom);
  /// NaN on fewer.
  double translation{0.0};
  /// @brief E_cas's Hessian, @ref df_casscf's, in the coordinates.
  std::vector<double> e_cas;
  /// @brief Z . d2g/dR dR, @ref df_casscf_z's lxy summed.
  std::vector<double> z_lxy;
  /// @brief L_Ax x_B + x_A L_xB of Z . g over x = (kappa_nr, c): I^x Z
  /// against lambda^y, both orders, symmetric.
  std::vector<double> z_lx;
  /// @brief x_A (Z . d2g/dx dx) x_B over x = (kappa_nr, c).
  std::vector<double> z_third;
  /// @brief The Z terms' kappa_oo rows,
  /// 1/2 <kappa_oo^B, [X_N, kappa_nr^A]> + (A <-> B) (README "The kappa_oo
  /// rows"). E_cas's are zero.
  std::vector<double> z_koo;
  /// @brief Sum h_bar h^AB + Sum Lambda B^AB at E_corr + Phi's adjoint,
  /// @ref df_adjoint_xy's.
  std::vector<double> skeleton;
  /// @brief <t_A, HVP t_B>, nevpt2::sc_hessian_product's products along
  /// t_B paired with t_A (a pair that both populate an empty perturber with
  /// each as the other's partner); not symmetrized.
  std::vector<double> hvp;
  /// @brief The rotation's cross term with the integrals' first derivative,
  /// <N_I, O(I^A, kappa^B) + O(I^B, kappa^A)>.
  std::vector<double> rotation_tau;
  /// @brief The rotation's second order, <N_I, 1/2 O(O(I, kappa^A),
  /// kappa^B)> + (A <-> B).
  std::vector<double> rotation_kappa;
  /// @brief @ref df_casscf's terms, Cartesian: E_cas, lambda^x and its CP
  /// residuals.
  CasscfTerms casscf;
  /// @brief The SC-NEVPT2 correlation energy (hartree).
  double e_corr{0.0};
  /// @brief The gradient's hazards: max |z^c| and the smallest core-core or
  /// virtual-virtual orbital-energy gap (hartree).
  double max_zc{0.0};
  double min_gap{0.0};
  /// @brief The gradient's Z-vector's true residual.
  double zvector_residual{0.0};
  /// @brief The largest weighted rho_tau of an accepted pair (0 when no
  /// pair of coordinates populates an empty perturber).
  double defect{0.0};
  /// @brief The smallest t_mix = 1 / 2 |kappa_oo_pq| over the coordinates
  /// and the canonical inactive pairs (bohr; infinite when none mixes).
  double mix_step{std::numeric_limits<double>::infinity()};
  /// @brief nevpt2::sc_hessian_product's rounding bound, the worst kept
  /// perturber's sigma / |N|: past 1e-7 the call is refused (README "The SC
  /// conditioning refusal").
  double rounding{0.0};
};

/// @brief The SC-NEVPT2 analytic nuclear Hessian of the single-state
/// DF-CASSCF @p solution (docs/nevpt2/nevpt2.tex eq. hess), every part from a
/// merged one, no integrals of its own (README.md "The SC-NEVPT2 Hessian"):
///
///     H = E_cas                           (df_casscf)
///       + Z . d2g/dRdR + L_Ax x_B + x_A L_xB + x_A L_xx x_B   (df_casscf_z at (Z, z))
///       + the Z terms' kappa_oo rows      (closed form)
///       + Sum h_bar h^AB + Lambda B^AB    (df_adjoint_xy at E_corr + Phi's adjoint)
///       + <t_A, HVP t_B>                  (nevpt2::sc_hessian_product over df_nevpt2_tangents' t)
///       + the rotation cross terms        (O(I^A, kappa^B), O(O(I, kappa^A), kappa^B))
///
/// in k coordinates q_i = Sum_x Q_ix R_x, @p coordinates Q's rows (k x
/// 3 n_atom row-major, orthonormal), the Cartesian ones when empty. The HVP
/// is taken along each coordinate's t, not transformed: an empty perturber
/// a direction populates makes it non-linear in the direction (K6). A pair
/// of coordinates that both populate one is taken again with each as the
/// other's partner, and refused when its energy is not a quadratic form on
/// the pair (nevpt2::class_energies_hvp's weighted rho_tau past 1e-6). A
/// coordinate breaks the symmetry when it populates an empty perturber
/// against the largest population scale over the coordinates (a rigid
/// translation's own is round-off). SC is not invariant to rotations within
/// the core or the virtuals, so a coordinate that mixes a canonical inactive
/// pair within 1e-3 bohr (t_mix = 1 / 2 |kappa_oo_pq|, the avoided
/// crossing's half-width; #817's branch-switch convention) is refused: its
/// element would be the curvature of a region narrower than a vibration
/// (README "Mixing inactive pairs").
///
/// The inputs are @ref df_casscf's (a converged one-root DF-CASSCF, its core
/// and virtuals canonical) and @p batch nevpt2::sc_energy's. All-electron:
/// it takes no frozen core. Stationarity is assumed, not checked.
///
/// The parts run one after another, each freeing what it held. Held
/// throughout: t_i's (h, B, c) and kappa (k (naux nmo^2 + 2 nmo^2 + n_det)),
/// kappa_nr, and E_corr + Phi's adjoint (naux nmo^2).
///
/// @return The Hessian and its parts; InvalidConfig for @ref df_casscf's
///         input refusals (a state average among them) or coordinates that
///         are not orthonormal rows of 3 n_atom; Numerical naming every
///         refused pair of coordinates, or every coordinate that mixes an
///         inactive pair within 1e-3 bohr (its pair, gap and t_mix; "use
///         PC-NEVPT2"), or df_nevpt2_tangents' degenerate
///         inactive pair, or the gradient's z^c or gate, or a kept
///         perturber whose rounding bound sigma / |N| is past 1e-7 (its
///         class, perturber, N and bound; README "The SC conditioning
///         refusal"); Divergence for the CP
///         solve's saddle; Unsupported for naux nmo^2 past int32_t or memory;
///         or the first error of a part.
[[nodiscard]] Result<ScNevpt2Terms>
df_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t batch = 0,
             std::span<const double> coordinates = {});

/// @brief What @ref df_sc_nevpt2 holds over @p basis and @p aux with @p ncore
/// core and @p ncas active orbitals over the determinants of @p nalpha and
/// @p nbeta electrons, in @p k coordinates at @p batch, at its least;
/// @p syevd_lwork per order (at order ncas the CP solve's):
///
/// - persistent: CasscfTerms::cp, returned;
/// - scratch: the strings and @ref df_casscf_footprint's; then, held to the
///   end, t_i's (h, B, c, kappa) and kappa_nr; beside them in turn
///   @ref df_nevpt2_tangents_footprint; h_bar and Lambda, held, with
///   @c nevpt2::sc_gradient_footprint's calls (its strings the call's own),
///   z^c and the multipliers kept; the products along one coordinate
///   (@c nevpt2::sc_hessian_product_footprint), then a pair's beside its
///   directions and partners; the Z terms (L_xlambda kept) beside x^x;
///   @ref df_adjoint_xy_footprint; the rotation terms' stacks;
/// - workspace: the largest of the calls'.
///
/// The products' chunk, W^x's and the tangents' parts grow with the memory
/// budget past this. Nothing pinned. Empty for the counts it refuses.
///
/// A pair is taken again only when both its coordinates populate an empty
/// perturber, which the dimensions do not say, so it is counted; @p pairs
/// false leaves it out, for a caller that knows none is (a test's bar).
[[nodiscard]] MemoryFootprint df_sc_nevpt2_footprint(const BasisSet &basis, const BasisSet &aux,
                                                     int32_t ncore, int32_t ncas, int32_t nalpha,
                                                     int32_t nbeta, int32_t k, int32_t batch,
                                                     std::span<const size_t> syevd_lwork,
                                                     bool pairs = true);

/// @brief The PC-NEVPT2 analytic nuclear Hessian and its parts, each k x k
/// in the call's k coordinates (hartree / bohr^2), column-major: @ref
/// ScNevpt2Terms' parts over PC's adjoint and product.
struct PcNevpt2Terms {
  /// @brief d^2E/dq_i dq_j of E_cas + E_corr^PC, the parts below summed and
  /// symmetrized, (H + H^T) / 2: exactly symmetric.
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the sum before it was symmetrized.
  double asymmetry{0.0};
  /// @brief The translational sum rule, as ScNevpt2Terms::translation.
  double translation{0.0};
  /// @brief E_cas's Hessian, @ref df_casscf's, in the coordinates.
  std::vector<double> e_cas;
  /// @brief Z . d2g/dR dR at PC's multipliers, as ScNevpt2Terms::z_lxy.
  std::vector<double> z_lxy;
  /// @brief The L_Ax pair of Z . g, as ScNevpt2Terms::z_lx.
  std::vector<double> z_lx;
  /// @brief x_A (Z . d2g/dx dx) x_B, as ScNevpt2Terms::z_third.
  std::vector<double> z_third;
  /// @brief The Z terms' kappa_oo rows, as ScNevpt2Terms::z_koo.
  std::vector<double> z_koo;
  /// @brief The skeleton at E_corr^PC + Phi's adjoint, @ref df_adjoint_xy's.
  std::vector<double> skeleton;
  /// @brief <t_A, HVP t_B>, nevpt2::pc_hessian_product's product along t_B
  /// paired with t_A; not symmetrized.
  std::vector<double> hvp;
  /// @brief The rotation's cross term with the integrals' first derivative,
  /// as ScNevpt2Terms::rotation_tau.
  std::vector<double> rotation_tau;
  /// @brief The rotation's second order, as ScNevpt2Terms::rotation_kappa.
  std::vector<double> rotation_kappa;
  /// @brief The degenerate pairs' off-diagonal f terms (README "A degenerate
  /// inactive pair"); zero without such a pair.
  std::vector<double> degenerate;
  /// @brief @ref df_casscf's terms, Cartesian: E_cas, lambda^x and its CP
  /// residuals.
  CasscfTerms casscf;
  /// @brief The PC-NEVPT2 correlation energy (hartree).
  double e_corr{0.0};
  /// @brief Per class (PySCF's order), the smallest metric eigenvalue the cut
  /// kept over the largest (nevpt2::PcClassEnergies::kept); 0 for Sijrs and
  /// a class with no tuple.
  std::array<double, 8> kept{};
  /// @brief The gradient's max |z^c|.
  double max_zc{0.0};
  /// @brief The smallest core-core or virtual-virtual orbital-energy gap
  /// (hartree).
  double min_gap{0.0};
  /// @brief The gradient's Z-vector's true residual.
  double zvector_residual{0.0};
  /// @brief [i][k]: class k's Schur-lift step t* along coordinate i (bohr),
  /// nevpt2::PcHessianProduct::lift_steps along t_i, its CI part the CP
  /// response c^i; infinite where nothing rises.
  std::vector<std::array<double, 8>> lift_steps;
  /// @brief The core or virtual pairs (p, q) under the degenerate-pair rule
  /// (1e-6 hartree) that @ref degenerate takes.
  std::vector<std::array<int32_t, 2>> degenerate_pairs;
  /// @brief The largest |E_corr| change of turning those pairs' orbitals in
  /// the bases @ref degenerate reads them in: PC's invariance, to O(gap).
  double rotation_shift{0.0};
};

/// @brief The PC-NEVPT2 analytic nuclear Hessian of the single-state
/// DF-CASSCF @p solution: @ref df_sc_nevpt2's assembly, every part the same,
/// over PC's adjoint (nevpt2::pc_adjoint), the gradient's multipliers at it
/// (nevpt2::pc_gradient's) and nevpt2::pc_hessian_product along each
/// coordinate's t (README "The PC-NEVPT2 Hessian"). @p coordinates are
/// @ref df_sc_nevpt2's; @p pc_floor is EnergyInputs::pc_floor, the energy's
/// and every part's.
///
/// PC has no empty perturbers, so no pair is taken again. A core or virtual
/// pair under 1e-6 hartree, which @ref df_nevpt2_tangents leaves unrotated,
/// is taken through its off-diagonal f, read as a split of the diagonal in
/// turned bases (PcNevpt2Terms::degenerate): PC is invariant to the pair's
/// rotation, so its energy is smooth there. Refused, Numerical, where one
/// analytic Hessian does not represent the PC surface:
/// - a class whose smallest kept metric eigenvalue is below 1e-10 of its
///   largest (#917), named with it: @p pc_floor 1e-10 or more cuts it;
/// - a coordinate along which a class's Schur lift reaches the cut within
///   t* < 1e-3 bohr (the surface switches branch inside a vibrational or
///   finite-difference step), every such coordinate named with its class and
///   t*.
///
/// Unsupported for an orbital under 1e-6 hartree from two others (a cluster
/// of three or more): only disjoint pairs are taken.
///
/// The inputs, memory and other refusals are @ref df_sc_nevpt2's, the
/// energy's own refusals of a class (no positive metric eigenvalue, a pole)
/// among them.
[[nodiscard]] Result<PcNevpt2Terms>
df_pc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t batch = 0,
             std::span<const double> coordinates = {}, double pc_floor = 0.0);

/// @brief What @ref df_pc_nevpt2 holds: @ref df_sc_nevpt2_footprint over PC's
/// gradient (@c nevpt2::pc_gradient_footprint) and product
/// (@c nevpt2::pc_hessian_product_footprint), no pair taken again.
/// @p dropped is theirs: the state's PcClassEnergies::dropped, or empty for
/// the rank that holds the most.
[[nodiscard]] MemoryFootprint
df_pc_nevpt2_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore, int32_t ncas,
                       int32_t nalpha, int32_t nbeta, int32_t k, int32_t batch,
                       std::span<const size_t> syevd_lwork,
                       std::span<const std::array<int64_t, 8>> dropped = {});

/// @brief A state-averaged DF-CASSCF root's analytic nuclear Hessian and its
/// terms (docs/hessian/hessian.tex eq. sa), each 3 n_atom x 3 n_atom,
/// x = 3 A + k the coordinate R_{A,k} (hartree / bohr^2), column-major.
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
  /// gradient's own z, its right-hand side weights one-hot on I.
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
  /// @brief z, root I's gradient multipliers (@c response::zvector over the SA
  /// Hessian, g_I its right-hand side): z_o, z_c and, at unequal weights, the
  /// rotations Theta. The same vector as the energy's (only the equations'
  /// rows are scaled), so it is @ref df_sacasscf_z's z as it comes.
  response::ZVector z;
  /// @brief lambda^x, A lambda^x = -d_x g_SA (@c response::zvectors, Schur), with
  /// the rotations at unequal weights in @c theta, and the solver's own
  /// iterations and residuals.
  response::ZVectors cp;
  /// @brief Per coordinate, the norm of every block of A lambda^x + d_x g_SA
  /// (the CI rows over 2 w_r, the rotation rows over 2 (w_r - w_s), as
  /// zvector's) by one product of the unreduced operator.
  std::vector<double> residuals;
};

/// @brief Root @p root's state-averaged DF-CASSCF analytic nuclear Hessian
/// at a converged average (docs/hessian/hessian.tex eq. sa):
///
///     d^2E_I/dx dy = L_xy + L_xlambda lambda^y + lambda^x^T L_lambday
///                    + lambda^x^T L_lambdalambda lambda^y  (+ eq. rot at equal weights),
///
/// L_xy = E_I,xy (@ref df_casscf_exy at root I's RDMs) + its z part and
/// L_xlambda (@ref df_sacasscf_z, at z from @c response::zvector);
/// L_lambdalambda = root I's parameter Hessian (@c RelaxedHessian's
/// unreduced product at weights one-hot on I, the CI rows scaled back to the
/// energy's) + the third-derivative action (@c mcscf::ThirdDerivative at z);
/// lambda^x by @c response::zvectors (Schur) over the SA Hessian on
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
/// is above 1e-6 is a Divergence. So is a singular SA Hessian, found by
/// @ref df_casscf's probe column (README.md "A singular H~").
///
/// The inputs are @ref df_casscf's, @p solution a converged
/// @c mcscf::solve of two roots or more over the same fit at @p weights
/// (positive, all equal or all pairwise different, one per root; a Solution
/// does not carry them), @p root the state. Stationarity is assumed, not
/// checked. @p tolerance and @p max_iterations are the solves'.
///
/// The parts run one after another, each freeing what it held; the peak is
/// @ref df_sacasscf_z's derivative integrals (as many coordinates' nmo^2 naux
/// at once as fit) or the CP solve's scratch, each refused on its own as
/// Unsupported past the memory budget (@ref df_sacasscf_footprint has the
/// whole).
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
///         naux nao^2 long, or the weights are equal for some pairs only
///         (eq. rot and lambda^x's rotations would both be needed);
///         Numerical at a degeneracy; Divergence when a solve meets
///         negative curvature, the CP solve runs out of products or finds
///         the SA Hessian singular by its probe column, or a residual is
///         left above 1e-6; or the first error of a part.
[[nodiscard]] Result<SaCasscfTerms>
df_sacasscf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            std::span<const PrimitivePairClass> classes, const BasisSet &aux,
            const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
            const mcscf::Solution &solution, std::span<const double> weights, int32_t root,
            int32_t ncore, int32_t ncas, double tolerance = 1e-10, int32_t max_iterations = 200);

/// @brief What @ref df_sacasscf holds over @p basis and @p aux with @p ncore
/// core and @p ncas active orbitals and @p nroots roots over the determinants
/// of @p nalpha and @p nbeta electrons, nmo = nao; @p rotating whether two of
/// the weights differ (@c response::rotating_pairs), @p syevd_lwork as
/// @ref df_casscf_footprint's:
///
/// - persistent: z, lambda^x and the z terms' L_xlambda, returned;
/// - scratch: the strings, the integrals, the SA Fock, the sigma with its
///   apply scratch, the SA H and H~ (3 n_atom + 1 block columns) and
///   @c mcscf::df_derivatives_footprint's SA right-hand sides, held to the
///   end; beside them in turn the probed @c response::zvectors and its check,
///   root I's Fock with @c response::zvector_footprint's z, z packed, the
///   third-derivative operator (@c mcscf::third_derivative_footprint), z cut
///   and Lambda, held to the end; then the third derivative's action, root
///   I's H and H~ with their block product, @ref df_sacasscf_z_footprint,
///   F^[x] at one-hot weights beside @ref df_casscf_exy_footprint, and at
///   equal weights @c mcscf::root_rotation_footprint;
/// - workspace: the largest of the calls'.
///
/// The block products' J and K width and the derivative integrals' chunk
/// grow with the memory budget, so this is the least the call runs with.
/// Nothing pinned. Empty for the counts it refuses.
[[nodiscard]] MemoryFootprint df_sacasscf_footprint(const BasisSet &basis, const BasisSet &aux,
                                                    int32_t ncore, int32_t ncas, int32_t nalpha,
                                                    int32_t nbeta, int32_t nroots, bool rotating,
                                                    size_t syevd_lwork);

/// @brief @ref df_sacasscf_cp's lambda^x and the check it passed.
struct SaCasscfCp {
  /// @brief lambda^x, as SaCasscfTerms::cp: A lambda^x = -d_x g_SA over every
  /// coordinate, z_o over the packed pairs, z_c (xi_r per root, off every
  /// root) and in @c theta the rotations of the pairs of unequal weights (zero
  /// for an equal pair).
  response::ZVectors cp;
  /// @brief Per coordinate, as SaCasscfTerms::residuals: the norm of every
  /// block of A lambda^x + d_x g_SA by one product of the unreduced
  /// operator, the rotation rows those of the pairs of unequal weights.
  std::vector<double> residuals;
};

/// @brief lambda^x alone: @ref df_sacasscf's CP-SA-CASSCF step,
/// A lambda^x = -d_x g_SA for every coordinate, the same solve, the same
/// check on the unreduced equations and the same bound on the rotations,
/// without z or any term of the Hessian. It does not depend on a root, so one
/// solve serves every state built on the average (@ref df_qd_nevpt2_tangents);
/// @ref df_sacasscf's SaCasscfTerms::cp is the same solution, and df_sacasscf
/// runs this step as its own.
///
/// The weights may be equal for some pairs only (0.4, 0.4, 0.2): each pair of
/// unequal weights carries its rotation Theta_sr in the solve and in the
/// check, each equal pair none (its rotation is eq. qd-theta's, closed form;
/// @ref df_qd_nevpt2_tangents).
///
/// The inputs, the refusals and the memory are @ref df_sacasscf's less the
/// root, its messages naming hessian::df_sacasscf_cp.
///
/// @return lambda^x and its residuals; or @ref df_sacasscf's errors (the
///         root's aside).
[[nodiscard]] Result<SaCasscfCp>
df_sacasscf_cp(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const mcscf::Solution &solution, std::span<const double> weights, int32_t ncore,
               int32_t ncas, double tolerance = 1e-10, int32_t max_iterations = 200);

/// @brief What @ref df_qd_nevpt2_tangents reports beside the tangents it hands
/// over.
struct QdNevpt2Tangents {
  /// @brief The rotations among the roots per coordinate, nroots (nroots - 1)
  /// / 2 a coordinate in (r, s) lexicographic order (ZVector::theta's), c_r
  /// gaining theta_sr c_s: the CP solve's at a pair of unequal weights,
  /// eq. qd-theta's at an equal one. Element (pair, x) at pair + npairs x.
  std::vector<double> theta;
  /// @brief max |theta^x_sr| over every coordinate and pair.
  double max_theta{0.0};
  /// @brief max |kappa_oo^x_pq| over every coordinate and pair.
  double max_kappa_oo{0.0};
  /// @brief The pairs PC recorded under the degenerate-pair rule, every
  /// pair under it along every coordinate, in coordinate order (empty for
  /// SC).
  std::vector<DegeneratePair> recorded;
};

/// @brief t_x, the first-order change of every input of a QD-NEVPT2
/// effective Hamiltonian along each nuclear coordinate x = 3 A + k at a
/// converged DF-SA-CASSCF reference, and the parameter response x_A it comes
/// from (docs/nevpt2/nevpt2.tex, the QD Hessian's parameter response: Eqs.
/// qd-K, qd-theta, qd-gsa and koo at f^SA), handed to @p consume a chunk of
/// coordinates at a time:
///
///     h_dot^x   = h1^x + kappa^xT h_MO + h_MO kappa^x
///     B_dot^x   = W^x + kappa^xT B_Q + B_Q kappa^x     (each fitted index Q)
///     c_dot^x_r = xi^x_r + sum_s theta^x_sr c_s        (theta_rs = -theta_sr)
///     kappa^x   = kappa_nr^x + kappa_oo^x,
///
/// in three steps, each closed once the one before it is done:
///
/// 1. **CP-SA-CASSCF**, @p cp (@ref df_sacasscf_cp's, the caller's, so one
///    solve serves every QD state): kappa_nr^x, xi^x_r per root and the
///    rotations theta^x_sr of the pairs of unequal weights.
/// 2. **The equal-weight rotations**, Eq. qd-theta:
///    theta^x_sr = H^x_sr / (E_r - E_s), H^x_sr the first-order change of
///    <c_s|H|c_r> along everything above (the derivative integrals with the
///    connection, and kappa_nr^x; xi adds nothing, being off every root): the
///    active Hamiltonian's tangent (h1eff and (tu|vw) of
///    @c nevpt2::qd_builders_tangent) against the roots' transition RDMs.
/// 3. **kappa_oo at f^SA**, Eq. koo with f -> f^SA:
///    kappa_oo^x_pq = -f^(x)_pq / (eps_p - eps_q) over the core and over the
///    virtuals, f^(x) the change of the state-averaged Fock's blocks along
///    steps 1 and 2 (@c nevpt2::qd_builders_tangent; an equal-weight rotation
///    does not move it, Eq. qd-gsa), eps its diagonal, under
///    @ref df_nevpt2_tangents' degenerate-pair rule: for SC (@p pc false) a
///    pair under 1e-6 Eh with |f^(x)_pq| > 1e-8 is Numerical, naming the
///    pair and the coordinate; for PC every pair under 1e-6 Eh is recorded
///    along every coordinate, kappa_oo^x_pq left 0, and @ref df_qd_pc_nevpt2
///    takes its off-diagonal f (#1277).
///
/// So t_x is (h_MO, B_MO, {c_r}) of the re-solved SA-CASSCF differentiated
/// along x, its roots diagonalizing H, its core and virtuals canonical in
/// f^SA, its active orbitals turned by no active-active rotation.
///
/// **Near a degeneracy it refuses.** Two roots under
/// @c mcscf::kRootGapFloor (1e-8 Eh) apart are Numerical, naming them,
/// before anything runs; a rotation |theta^x_sr| above 10 per bohr
/// (@c mcscf::root_rotation's bound) is Numerical, naming the pair and the
/// coordinate.
///
/// @p consume(x0, count, h_dot, b_dot, c_dot, kappa) is
/// @ref df_nevpt2_tangents', but @p c_dot holds every root's: count blocks of
/// nroots n_det, root r's at + r n_det in a block (the vectors' layout).
///
/// The inputs are @ref df_sacasscf_cp's, @p solution's core and virtuals
/// canonical in f^SA (as the Casscf step leaves them; an off-diagonal
/// |f_pq| above 1e-8 is refused), and @p cp at the same solution and
/// weights. All-electron.
///
/// The memory is @ref df_nevpt2_tangents', with
/// @c nevpt2::qd_builders_tangent's arrays (every ordered pair's RDMs per
/// coordinate) in place of builders_tangent's. Queued on res.stream() with
/// res.blas() (host pointer mode); it synchronizes many times. None of @p b
/// and @p l may point into res.device_workspace(). The device res.dev_idx()
/// names is made current for the call and the previous one restored.
///
/// @return The rotations, the largest rotation and kappa_oo, and PC's
///         records; InvalidConfig for @ref df_sacasscf_cp's input refusals, a
///         null @p consume, @p cp not of 3 n_atom columns over the packed
///         pairs, the roots' determinants and the pairs of roots, or core or
///         virtuals not canonical; Numerical at a degeneracy, a rotation past
///         the bound, or for SC at a degenerate pair; Unsupported for naux nmo
///         past int32_t or memory past the budget; or the first error of a
///         part or of @p consume.
[[nodiscard]] Result<QdNevpt2Tangents> df_qd_nevpt2_tangents(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    std::span<const double> weights, int32_t ncore, int32_t ncas, const response::ZVectors &cp,
    bool pc,
    const std::function<Status(int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
                               const double *c_dot, const double *kappa)> &consume,
    size_t max_bytes = 0);

/// @brief What @ref df_qd_nevpt2_tangents holds, as
/// @ref df_nevpt2_tangents_footprint's over @p nroots roots: a part's
/// QD builders' tangent (@c nevpt2::qd_builders_tangent_footprint) and c_dot
/// over every root, and beside the first part's tangent the roots'
/// transition RDMs.
[[nodiscard]] MemoryFootprint df_qd_nevpt2_tangents_footprint(const BasisSet &basis,
                                                              const BasisSet &aux, int32_t ncore,
                                                              int32_t ncas, int32_t nalpha,
                                                              int32_t nbeta, int32_t nroots);

/// @brief The zeta term of a QD-NEVPT2 Lagrangian's Hessian, T = Sum_{r<s}
/// zeta_sr <c_s|H|c_r> (docs/nevpt2/nevpt2.tex Eq. qd-Lxx's last-but-one
/// term and its mixed and skeleton shares): its three places, each the
/// second derivative of T itself, which is not stationary.
struct QdZetaTerms {
  /// @brief T's share of L_AB, d2 T / dR_x dR_y at fixed parameters along
  /// the orthonormal connection: E_xy at the folded transition RDMs less its
  /// zero-RDM value (no core energy), piece by piece as @ref df_casscf_exy
  /// returns them, 3 n_atom x 3 n_atom column-major, symmetric to rounding;
  /// nuclear_repulsion is zero.
  CasscfExy lxy;
  /// @brief T's share of L_Ax, d2 T / dR_x dx, a covector per coordinate:
  /// size x 3 n_atom column-major on the device, column x in this call's
  /// layout (the packed orbital pairs, d_r per root off every root, then
  /// Theta_sr for every r < s). Its dot product with x^y in the same layout
  /// is element (x, y).
  Buffers::device<double> lxlambda;
  /// @brief Lambda^T (d2 T / dx dx) Lambda over the call's k columns Lambda:
  /// k x k column-major on the host, symmetric to rounding; empty for k = 0.
  std::vector<double> lxx;
};

/// @brief The three shares of the zeta term T = Sum_{r<s} zeta_sr
/// <c_s|H|c_r> of a QD-NEVPT2 Lagrangian's analytic Hessian, at a converged
/// DF-SA-CASSCF @p solution and caller-supplied @p zeta (the Lagrangian's
/// equal-weight pairs', @c mcscf::RootPairMultipliers::zeta; nevpt2.tex Eqs.
/// qd-Lxx and the mixed and skeleton terms), README.md "The zeta term":
///
///     lxy      = 1/2 (E_xy(c + C S) - E_xy(c - C S)),   S = (n / 4) Z
///     lxlambda = the same half difference of the CP right-hand sides
///                (orbital, CI), and (Z M^x)_rs - (Z M^x)_sr per pair,
///                M^x = <c_r|H~^x|c_s>
///     lxx      = Lambda^T T_xx Lambda,
///     T_uv     = kappa_u^T (H(gamma^zeta) - H(0)) kappa_v         (orbital)
///              + g(R_u) . kappa_v + g(R_v) . kappa_u            (orbital-CI)
///              + <X_u, H X_v Z> - 1/2 <X_v, K_u H C Z>
///              - 1/2 <X_u, K_v H C Z>                           (CI)
///
/// with Z the symmetric n x n of the zeta (zero diagonal), C the roots,
/// K_u the root set's generator of u (Eq. qd-K) and X_u = K_u C, H(.)
/// @c response::Hessian at the RDMs given, R_u = Sum_q T(X_u,q, (C Z)_q) the
/// symmetrized transition RDMs and g(.) @c mcscf::transition_orbital_gradient
/// at them. Every core-energy part multiplies the change of <c_s|c_r> = 0
/// and drops. The roots are any orthonormal set: nothing assumes they are
/// eigenvectors.
///
/// **The parameters**: one column of size = pairs + n_det nroots +
/// nroots (nroots - 1) / 2 doubles, @c mcscf::ThirdDerivative's layout with
/// every pair's rotation Theta_sr (c_r gaining Theta_sr c_s), at equal
/// weights too: T moves along an equal-weight rotation, which E_SA does not.
/// A column's CI part is taken off every root here.
///
/// The inputs are @ref df_sacasscf_cp's less the weights, which T does not
/// read: @p zeta one per pair r < s in (r, s) lexicographic order, @p lambda
/// @p k columns of size (size x k column-major on the device), null for k = 0.
///
/// The parts run one after another, each freeing what it held: three
/// @c mcscf::df_derivatives calls (at c + C S, c - C S and the roots) and two
/// @ref df_casscf_exy calls, then for k > 0 the active-space integrals,
/// @c mcscf::zeta_terms' folded RDMs, two orbital Hessians, one sigma per
/// root of every column and one @c transition_orbital_gradient per column.
/// Each part refuses its own memory, Unsupported, before it allocates.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. None of @p b and @p l may point into
/// res.device_workspace(). The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return The three shares; InvalidConfig for @ref df_sacasscf_cp's input
///         refusals, @p zeta not one per pair, a negative @p k or a null
///         @p lambda with k > 0, or for any input a part refuses; Numerical
///         for two roots under @c mcscf::kRootGapFloor apart; or the first
///         error of a part (Unsupported for memory, a GpuRuntime error).
[[nodiscard]] Result<QdZetaTerms>
df_qd_zeta(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
           std::span<const PrimitivePairClass> classes, const BasisSet &aux,
           const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
           const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
           std::span<const double> zeta, int32_t k = 0, const double *lambda = nullptr);

/// @brief @ref df_qd_eigenvector_response's result: term (ii) of every QD
/// state's Hessian, the couplings it is built from, and dR_P/dR_x. Every
/// array is on the host; x = 3 A + k, nx = 3 n_atom.
struct QdEigenvectorResponse {
  /// @brief H_eff's eigenvalues E_P, ascending.
  std::vector<double> eigenvalues;
  /// @brief H_eff's eigenvectors R, n x n, state P's column at [P n,
  /// (P + 1) n), each signed so that its largest-magnitude component is
  /// positive (nevpt2::qd_sc_coupling's convention), in the solution's roots'
  /// phases.
  std::vector<double> eigenvectors;
  /// @brief h_PQ = G(W^PQ) for every pair P < Q in (P, Q) lexicographic
  /// order, nx a pair: component x of pair k at [k nx + x]; h_QP = h_PQ.
  std::vector<double> couplings;
  /// @brief Term (ii) per state, Eq. qd-hess: 2 Sum_{Q != P} h_PQ,x h_PQ,y /
  /// (E_P - E_Q), nx x nx column-major, state P's at [P nx^2], exactly
  /// symmetric.
  std::vector<double> term;
  /// @brief dR_P/dR_x per state, Eq. qd-dR in the gauge R_P^T dR_P = 0:
  /// Sum_{Q != P} R_Q h_QP,x / (E_P - E_Q), n x nx column-major, state P's at
  /// [P n nx], component i along x at [P n nx + i + n x].
  std::vector<double> dr;
  /// @brief max |term (ii)_P| per state: the numerator of its share of the
  /// Hessian (the whole Hessian's assembler divides by its own largest).
  std::vector<double> largest;
  /// @brief max |Sum_P term (ii)_P|, Eq. qd-sum's term (ii) part, zero by
  /// construction (each pair enters P and Q with opposite signs): a
  /// diagnostic, never gated.
  double sum_rule = 0.0;
  /// @brief The smallest |E_P - E_Q|, P != Q; infinite for one root.
  double qd_gap = std::numeric_limits<double>::infinity();
  /// @brief The largest true residual of the shared Z-vector solve's
  /// columns, each held to the call's tolerance.
  double max_residual = 0.0;
};

/// @brief Term (ii) of the QD-NEVPT2 nuclear Hessian (docs/nevpt2/nevpt2.tex
/// Eq. qd-hess), the QD eigenvector response, for every state at once, from
/// first derivatives alone, at a converged DF-SA-CASSCF @p solution: SC's
/// (@p pc false) or PC's.
///
///     term (ii)_P,xy = 2 Sum_{Q != P} h_PQ,x h_PQ,y / (E_P - E_Q)
///     dR_P/dR_x      = Sum_{Q != P} R_Q h_QP,x / (E_P - E_Q)       (Eq. qd-dR)
///
/// with h_PQ = G(W^PQ), W^PQ = 1/2 (R_P R_Q^T + R_Q R_P^T), the interstate
/// coupling of nevpt2::qd_sc_coupling (or qd_pc_coupling). In order:
///
/// 1. H_eff = diag(E_r) + 1/2 (C + C^T) over nevpt2::qd_inputs and the
///    method's effective Hamiltonian, its eigenvalues and R;
/// 2. **the refusal**, before any adjoint is built: a pair of QD states under
///    @c mcscf::kRootGapFloor apart is Numerical, naming both (term (ii)
///    divides by their gap, and a symmetry-degenerate E_P has no Hessian of
///    this form; README.md "Term (ii)");
/// 3. every n (n - 1) / 2 trace-zero W^PQ through one call of
///    nevpt2::qd_sc_gradient's (qd_pc_gradient's) several-W form: one shared
///    Z-vector solve, each column's true residual held to
///    @p zvector_tolerance, and each W's single core-off df_gradient pass;
/// 4. the closed forms above on the host.
///
/// One root returns its eigenvalue, R = 1 and empty sums (zero term, zero
/// dR), with no gradient call. The inputs are nevpt2::qd_sc_coupling's less
/// the pair, all-electron; @p batch is the effective Hamiltonian's, and
/// @p pc_floor PC's conditioning floor (EnergyInputs::pc_floor), read by
/// H_eff and every coupling alike (unread for SC).
///
/// With @p pair, a crossing pair (P, Q) (Eqs. qd-pair and qd-lowdin): its
/// gap is neither refused nor divided by, every other pair's still is; h_PQ
/// is still returned (in the call's basis of the eigenspace), and P's and Q's
/// term and dR sum over S not in {P, Q} alone, the outside states.
///
/// @return Term (ii) and its parts; InvalidConfig for a tolerance not
///         positive or a @p pair not two distinct states; Numerical for a
///         degenerate QD pair; or the first error of nevpt2::qd_inputs, the
///         effective Hamiltonian (a refused PC class among them) or the
///         several-W gradient, unchanged.
[[nodiscard]] Result<QdEigenvectorResponse> df_qd_eigenvector_response(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    int32_t ncore, int32_t ncas, bool pc, int32_t batch = 0, double zvector_tolerance = 1e-10,
    double pc_floor = 0.0, std::optional<std::array<int32_t, 2>> pair = std::nullopt);

/// @brief The SC-QD-NEVPT2 analytic nuclear Hessian and its parts, each k x k
/// in the call's k coordinates (hartree / bohr^2), column-major, with every
/// gap it divides by.
struct QdScNevpt2Terms {
  /// @brief The parts below summed and symmetrized, (H + H^T) / 2, exactly
  /// symmetric: d^2 E_P / dq_i dq_j for a state (term (i) + term (ii)),
  /// term (i) alone, d^2 E[W] at fixed W, for a weight matrix, or at a
  /// crossing term (i) at W = sum_XY c_XY R_X R_Y^T plus the outside-state
  /// sum (@ref df_qd_sc_nevpt2_crossing).
  std::vector<double> hessian;
  /// @brief max |H - H^T| of the sum before it was symmetrized.
  double asymmetry{0.0};
  /// @brief The translational sum rule, as ScNevpt2Terms::translation.
  double translation{0.0};
  /// @brief E_rho = sum_r rho_r E_r's share, rho = diag W, of every place
  /// it enters: E_xy at the rho-weighted RDMs, its L_Ax x_B + x_A L_xB (the CP
  /// right-hand side at rho) and x_A (d^2 E_rho / dx dx) x_B (the unreduced
  /// CI-relaxed Hessian at rho, the rotations among the roots included).
  std::vector<double> e_sa;
  /// @brief (Z, z, Theta) . d^2 g_SA / dR dR, the multipliers' L_xy part.
  std::vector<double> z_lxy;
  /// @brief I^x z against x^y, both orders, symmetric.
  std::vector<double> z_lx;
  /// @brief x_A ((Z, z, Theta) . d^2 g_SA / dx dx) x_B, mcscf::ThirdDerivative's.
  std::vector<double> z_third;
  /// @brief The kappa_oo rows, as ScNevpt2Terms::z_koo, at E^corr[W] + Phi^SA's
  /// orbital source.
  std::vector<double> z_koo;
  /// @brief The zeta term's three shares, @ref df_qd_zeta's: L_AB, the L_Ax
  /// pair, and x_A T_xx x_B.
  std::vector<double> zeta;
  /// @brief The skeleton at E^corr[W] + Phi^SA's adjoint, @ref df_adjoint_xy's.
  std::vector<double> skeleton;
  /// @brief <t_A, HVP t_B>, nevpt2::qd_sc_hessian_product's product along t_B
  /// paired with t_A (a pair that both populate an empty tuple with each as
  /// the other's partner; @ref df_qd_pc_nevpt2's, qd_pc_hessian_product's);
  /// not symmetrized.
  std::vector<double> hvp;
  /// @brief The rotation cross terms, as ScNevpt2Terms::rotation_tau and
  /// rotation_kappa.
  std::vector<double> rotation_tau;
  std::vector<double> rotation_kappa;
  /// @brief Term (ii), @ref df_qd_eigenvector_response's for the state, in the
  /// coordinates; at a crossing the outside-state sum over S not in {P, Q}
  /// in its place; empty for a weight matrix.
  std::vector<double> term_ii;
  /// @brief max |term (ii)| over max |hessian|; 0 for a weight matrix.
  double term_ii_share{0.0};
  /// @brief H_eff's eigenvalues, ascending; the seed W (n x n row-major,
  /// R_P R_P^T for a state) and rho = diag W.
  std::vector<double> eigenvalues;
  std::vector<double> w;
  std::vector<double> rho;
  /// @brief Table qd-divisions' gaps (hartree): the smallest |E_Q - E_P|, Q
  /// != P (infinite for a weight matrix; at a crossing the smallest between
  /// the pair and an outside state, the pair's own gap not among them), the
  /// smallest CASSCF root gap, the
  /// ensemble's edge E_{n+1} - E_n (infinite when the active space has no
  /// further root), and the smallest core-core or virtual-virtual SA-Fock gap.
  double qd_gap{std::numeric_limits<double>::infinity()};
  double root_gap{std::numeric_limits<double>::infinity()};
  double edge_gap{std::numeric_limits<double>::infinity()};
  double min_gap{std::numeric_limits<double>::infinity()};
  /// @brief The smallest |Delta_t + H_t / N_t| over every state's kept
  /// perturbers (d_t / N_t, d_t = N_t Delta_t + H_t; hartree): the
  /// near-intruder the finish divides by.
  double min_denominator{std::numeric_limits<double>::infinity()};
  /// @brief max |theta^x_sr|, max |kappa_oo^x_pq| (per bohr, Cartesian), max
  /// |zeta_sr| and max |z^c|.
  double max_theta{0.0};
  double max_kappa_oo{0.0};
  double max_zeta{0.0};
  double max_zc{0.0};
  /// @brief The CP solve's worst unreduced residual and the Lagrangian's
  /// Z-vector's true residual.
  double cp_residual{0.0};
  double zvector_residual{0.0};
  /// @brief The largest weighted rho_tau of an accepted pair, ket or
  /// coupling (0 when no pair of coordinates populates an empty tuple).
  double defect{0.0};
  /// @brief The smallest t_mix = 1 / 2 |kappa_oo_pq| over the coordinates
  /// and the canonical inactive pairs (bohr), as ScNevpt2Terms::mix_step.
  double mix_step{std::numeric_limits<double>::infinity()};
  /// @brief nevpt2::qd_sc_hessian_product's rounding bound, the worst kept
  /// tuple's sigma / |N| over the reference roots W weighs, and where it is
  /// ({root, class, tuple}; the root -1 when none): past 1e-7 the call is
  /// refused (README "The SC conditioning refusal").
  double rounding{0.0};
  std::array<int64_t, 3> rounding_at{-1, 0, 0};
};

/// @brief The SC-QD-NEVPT2 analytic nuclear Hessian of QD state @p state
/// (H_eff's eigenvalues ascending) on the DF-SA-CASSCF @p solution
/// (docs/nevpt2/nevpt2.tex Eq. qd-hess, Algorithm qdhess): term (i) at
/// W^P = R_P R_P^T (the W form below) plus term (ii),
/// @ref df_qd_eigenvector_response's, in the same coordinates (README.md
/// "The SC-QD-NEVPT2 Hessian").
///
/// **Refused before anything runs** (Numerical, naming what): a QD gap
/// |E_Q - E_P| under mcscf::kRootGapFloor, two CASSCF roots under it, or the
/// ensemble's edge E_{n+1} - E_n under it (the n + 1-th root of the active
/// space at the solution's orbitals); @p nfrozen other than 0 (Unsupported:
/// all-electron only). Then, after the CP solve, as @ref df_sc_nevpt2
/// refuses: an inactive pair under 1e-6 Eh that a coordinate splits
/// (@ref df_qd_nevpt2_tangents'), a coordinate that mixes one within 1e-3
/// bohr, a kept tuple of a reference root W weighs whose rounding bound
/// sigma / |N| is past 1e-7 (README "The SC conditioning refusal", #1102:
/// its Hessian would spread past the bars run to run), and the pairs of
/// coordinates nevpt2::qd_sc_hessian_product refuses on any model state's
/// empty tuples (#962's per-block refusal), named together.
///
/// The inputs are @ref df_sc_nevpt2's, @p solution a converged
/// DF-SA-CASSCF of two roots or more, its core and virtuals canonical in
/// f^SA, its weights mcscf::sa_gradient_weights'; @p batch is the class
/// walks'. The memory is the W form's and @ref df_qd_eigenvector_response's,
/// one after the other.
///
/// @return The Hessian, its parts and the gaps; InvalidConfig as
///         @ref df_sacasscf_cp's inputs, @p state not a QD state, or
///         coordinates that are not orthonormal rows of 3 n_atom; the
///         refusals above; or the first error of a part.
[[nodiscard]] Result<QdScNevpt2Terms>
df_qd_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t state,
                int32_t batch = 0, std::span<const double> coordinates = {}, int32_t nfrozen = 0);

/// @brief Term (i) of the SC-QD-NEVPT2 nuclear Hessian at a fixed symmetric
/// weight matrix @p w (n x n row-major), d^2 E[W] / dq_i dq_j at fixed W
/// (docs/nevpt2/nevpt2.tex Eq. qd-hessW): Part I's assembly
/// (@ref df_sc_nevpt2's, part for part, in the same parametrization and
/// coordinates) over the state average's Lagrangian at W,
///
///     L = E_rho + Z . g_SA + sum_r 2 w_r z_r . (H - E_r) c_r
///         + sum_{r<s} (2 (w_r - w_s) Theta_sr + zeta_sr) <c_s|H|c_r>
///         + E^corr[W] + Phi^SA,
///
/// the response x_A = (kappa_nr, kappa_oo, xi_r, theta_sr) of
/// @ref df_sacasscf_cp and @ref df_qd_nevpt2_tangents (every pair's rotation,
/// Eq. qd-theta's at an equal pair), and the multipliers of
/// nevpt2::qd_sc_lagrangian's W form:
///
///     H = E_rho's (E_xy at rho, its L_Ax pair, x_A H~_rho x_B)
///       + the multipliers' (df_sacasscf_z's L_xy and I^x z, ThirdDerivative's
///         x_A T x_B, every rotation a parameter)
///       + the zeta term (df_qd_zeta) + the kappa_oo rows (closed form)
///       + Sum h_bar h^AB + Lambda B^AB    (df_adjoint_xy)
///       + <t_A, HVP t_B>                  (nevpt2::qd_sc_hessian_product)
///       + the rotation cross terms        (as df_sc_nevpt2's)
///
/// W is any symmetric matrix: rho = diag W need not be a mixture. E_rho's
/// share, whose pieces take the core at weight 1, is taken at rho = a p+ -
/// b p- by linearity, p+ and p- the mixtures of rho's positive and negative
/// parts (a second pass only when rho has a negative entry, as a seam's W
/// of Eq. qd-lowdin, trace 0, does); every other part is linear in W and
/// takes it as it stands. The refusals and the reports are the state form's
/// but the QD gap (infinite here) and term (ii) (empty).
///
/// The parts run one after another, each freeing what it held. Held
/// throughout: t_i's (h, B, {c_r}) and kappa (k (naux nmo^2 + 2 nmo^2 + n
/// n_det)), kappa_nr, x_A in the coordinates, and E^corr[W] + Phi^SA's
/// adjoint (naux nmo^2).
///
/// @return The Hessian and its parts; InvalidConfig for a W not n^2 long,
///         not finite or not symmetric to 1e-12, and the state form's input
///         refusals; its other refusals; or the first error of a part.
[[nodiscard]] Result<QdScNevpt2Terms>
df_qd_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
                std::span<const double> w, int32_t batch = 0,
                std::span<const double> coordinates = {}, int32_t nfrozen = 0);

/// @brief The PC-QD-NEVPT2 analytic nuclear Hessian and its parts: @ref
/// df_qd_sc_nevpt2's assembly over PC's pieces, in its fields, and PC's
/// conditioning on every model state.
struct QdPcNevpt2Terms {
  /// @brief The Hessian, its parts, the gaps and the reports, as
  /// df_qd_sc_nevpt2 returns them: hvp is nevpt2::qd_pc_hessian_product's
  /// (no partners), the multipliers and E^corr[W] + Phi^SA's adjoint
  /// nevpt2::qd_pc_gradient's, term (ii) PC's, all at the call's pc_floor.
  /// SC's min_denominator, defect, mix_step and rounding are not PC's and
  /// stay at their defaults.
  QdScNevpt2Terms terms;
  /// @brief Per model state N, each class's smallest kept metric eigenvalue
  /// over its largest (nevpt2::PcClassEnergies::kept, ket N's) at the floor;
  /// 0 for Sijrs and a class with no tuple.
  std::vector<std::array<double, 8>> kept;
  /// @brief [i][N][class]: t* (bohr) along coordinate i at which ket N's
  /// class's Schur lift reaches the cut (nevpt2::QdPcHessianProduct's);
  /// infinite where nothing rises.
  std::vector<std::vector<std::array<double, 8>>> lift_steps;
  /// @brief The degenerate inactive pairs' off-diagonal f terms of term (i)
  /// at W, k x k, in the Hessian (README "The PC-QD-NEVPT2 Hessian", "At a
  /// degenerate inactive pair", #1277); zero without such a pair, and for SC.
  std::vector<double> degenerate;
  /// @brief The core or virtual pairs (p, q) under the degenerate-pair rule
  /// (1e-6 hartree, at f^SA) that @ref degenerate takes.
  std::vector<std::array<int32_t, 2>> degenerate_pairs;
  /// @brief The largest |E^corr[W]| change of turning those pairs' orbitals
  /// in the bases @ref degenerate reads them in: the elements' invariance,
  /// to O(gap).
  double rotation_shift{0.0};
};

/// @brief The PC-QD-NEVPT2 analytic nuclear Hessian of QD state @p state
/// (PC's H_eff eigenvalues ascending) on the DF-SA-CASSCF @p solution
/// (docs/nevpt2/nevpt2.tex Eq. qd-hess, Algorithm qdhess): @ref
/// df_qd_sc_nevpt2's state form, part for part, with PC's pieces --
/// nevpt2::qd_pc_energy's H_eff, nevpt2::qd_pc_gradient's Lagrangian at
/// W^P, nevpt2::qd_pc_hessian_product's product and PC's term (ii) -- every
/// one at @p pc_floor (EnergyInputs::pc_floor), so all of them differentiate
/// one energy (README.md "The PC-QD-NEVPT2 Hessian").
///
/// **Refused before anything runs** (Numerical, naming what): the SC form's
/// gaps (a QD gap, two CASSCF roots, the ensemble's edge, each under
/// mcscf::kRootGapFloor), and a class of any model state kept below 1e-10 of
/// its metric's largest eigenvalue (naming the state, the class, its kept and
/// @p pc_floor: PC's second derivative is not attainable there, #917);
/// @p nfrozen other than 0 (Unsupported). **A degenerate inactive pair**
/// (under 1e-6 hartree at f^SA, @ref df_qd_nevpt2_tangents' records) is
/// taken as @ref df_pc_nevpt2 takes one, in term (i) at W
/// (QdPcNevpt2Terms::degenerate): every H_eff element is invariant to the
/// pair's rotation; an orbital under 1e-6 hartree from two others (a cluster
/// of three or more) is Unsupported. **After the CP solve**, as
/// @ref df_pc_nevpt2 refuses, on every model state: the pairs
/// of coordinates (i, j) along both of which a ket's cut metric modes lift
/// with a first-order overlap to another state (nevpt2::qd_pc_hessian_product's
/// refusal, Eq. qd-pcfail), with each one's t*; and a coordinate along which
/// any state's Schur lift reaches the cut within 1e-3 bohr; all named
/// together.
///
/// The inputs are @ref df_qd_sc_nevpt2's, and @p pc_floor @ref df_pc_nevpt2's.
///
/// @return The Hessian, its parts, the gaps and PC's conditioning;
///         InvalidConfig as @ref df_qd_sc_nevpt2's; the refusals above; or
///         the first error of a part (a class qd_pc_energy refuses among
///         them).
[[nodiscard]] Result<QdPcNevpt2Terms>
df_qd_pc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t state,
                int32_t batch = 0, std::span<const double> coordinates = {}, double pc_floor = 0.0,
                int32_t nfrozen = 0);

/// @brief Term (i) of the PC-QD-NEVPT2 nuclear Hessian at a fixed symmetric
/// weight matrix @p w (n x n row-major), d^2 E[W] / dq_i dq_j at fixed W
/// (Eq. qd-hessW): @ref df_qd_sc_nevpt2's W form with the state form's PC
/// pieces above, at any symmetric W. The refusals are the state form's but
/// the QD gap; @p pc_floor is its.
[[nodiscard]] Result<QdPcNevpt2Terms> df_qd_pc_nevpt2(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    int32_t ncore, int32_t ncas, std::span<const double> w, int32_t batch = 0,
    std::span<const double> coordinates = {}, double pc_floor = 0.0, int32_t nfrozen = 0);

/// @brief The SC-QD-NEVPT2 Hessian at a crossing of QD states @p p and @p q
/// (docs/nevpt2/nevpt2.tex Eqs. qd-pair and qd-lowdin), where E_P and E_Q
/// have none (a cone) but every other state stays gapped: for a symmetric
/// 2 x 2 @p c, row-major, over a basis V_0, V_1 of the pair's eigenspace,
///
///     H[c] = term (i) at W = sum_ab c_ab V_a V_b^T             (the W form)
///          + sum_XY c~_XY sum_{S not in {P, Q}} (h_XS,A h_SY,B + h_XS,B h_SY,A)
///                          (1 / (E_X - E_S) + 1 / (E_Y - E_S)) / 2,
///
/// X, Y over {p, q}, R and h_XS = G(W^XS) @ref df_qd_eigenvector_response's
/// with the pair excluded, V = R M and c~ = M c M^T c in R's basis.
/// **V is set by the eigenspace's projector alone**: the eigensolver's R_P,
/// R_Q are any basis of it (a 1e-14 rounding of H_eff turns them by 1e-7
/// rad at a 1e-7 gap), so M = A (A^T A)^(-1/2), A_ab = R_{X_a}[r_b], makes V
/// the Loewdin-orthonormalized projections of CASSCF roots r_0 < r_1 (the
/// pair of roots whose |det A| is largest) onto it, the same from one call
/// to the next. c = 1 is the pair's sum d^2 (E_P + E_Q), Eq. qd-pair, in any
/// basis, and defined away from a crossing too (there the sum of the two
/// states' Hessians). c = sym(e_a e_b^T) is element (a, b) of the seam's
/// second-order effective 2 x 2, Eq. qd-lowdin (at E_X = E_Y = E_0; c =
/// [[0, 1/2], [1/2, 0]] its off-diagonal, term (i) at the seam's W = 1/2
/// (V_0 V_1^T + V_1 V_0^T)): its elements are V's, its eigenvalues are
/// basis-free. term_ii holds the outside-state sum; qd_gap the smallest
/// |E_X - E_S|; w the W.
///
/// **Refused before anything runs**, beside the state form's gaps but the
/// pair's own: an outside state within mcscf::kRootGapFloor of P or Q
/// (Numerical, naming it and the pair's state; the pair's eigenspace is not
/// isolated there), and two outside states within it of each other (@ref
/// df_qd_eigenvector_response's refusal). The other refusals are the W
/// form's. The couplings run first, then term (i).
///
/// @return H[c] and its parts; InvalidConfig for (p, q) not two distinct QD
///         states or @p c not 4 finite doubles symmetric to 1e-12, and the W
///         form's; the refusals above; or the first error of a part.
[[nodiscard]] Result<QdScNevpt2Terms> df_qd_sc_nevpt2_crossing(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    int32_t ncore, int32_t ncas, int32_t p, int32_t q, std::span<const double> c, int32_t batch = 0,
    std::span<const double> coordinates = {}, int32_t nfrozen = 0);

/// @brief The PC-QD-NEVPT2 Hessian at a crossing of QD states @p p and @p q:
/// @ref df_qd_sc_nevpt2_crossing's form over PC's pieces, at @p pc_floor,
/// with @ref df_qd_pc_nevpt2's refusals but the pair's own gap.
[[nodiscard]] Result<QdPcNevpt2Terms> df_qd_pc_nevpt2_crossing(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, const double *l, const mcscf::Solution &solution,
    int32_t ncore, int32_t ncas, int32_t p, int32_t q, std::span<const double> c, int32_t batch = 0,
    std::span<const double> coordinates = {}, double pc_floor = 0.0, int32_t nfrozen = 0);

/// @brief Which of the QD Hessians' forms a footprint is of: one state's
/// (term (i) at W^P, then term (ii)), a weight matrix's (term (i) at W) or a
/// crossing's (the couplings first, then term (i) at the pair's W).
enum class QdNevpt2Form : int32_t { state, weights, crossing };

/// @brief What @ref df_qd_sc_nevpt2 (and @ref df_qd_sc_nevpt2_crossing, by
/// @p form) holds over @p basis and @p aux with @p ncore core and @p ncas
/// active orbitals, @p nroots roots over the determinants of @p nalpha and
/// @p nbeta electrons (@p rotating: two weights differ), in @p k coordinates
/// at @p batch, at its least; @p syevd_lwork per order (at ncas the CP
/// solve's, at nroots H_eff's). Nothing persistent; in turn, beside the
/// strings, Q^T and from the CP solve on lambda^x, t_i's and x_A in the
/// coordinates: the ensemble's edge (the CI solve over one more root);
/// H_eff's elements and eigenpairs; at a crossing the couplings; the CP
/// solve; @ref df_qd_nevpt2_tangents_footprint; the Lagrangian at W, kept to
/// the end (@c nevpt2::qd_sc_w_gradient_footprint); the products along one
/// coordinate, then a pair's beside its directions and partners (unless
/// @p pairs is false, for a caller that knows none is taken again); E_rho's
/// share; the multipliers' z terms; the zeta term; @ref
/// df_adjoint_xy_footprint; the rotation terms; for a state, term (ii)
/// (@ref df_qd_eigenvector_response's, its several-W gradient bounded by
/// n (n - 1) / 2 trace-zero W's, exact at two roots). The workspace is the
/// largest of the calls'. The products' chunk, W^x's and the tangents'
/// parts grow with the memory budget past this. Empty for the counts it
/// refuses.
[[nodiscard]] MemoryFootprint
df_qd_sc_nevpt2_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore, int32_t ncas,
                          int32_t nalpha, int32_t nbeta, int32_t nroots, bool rotating, int32_t k,
                          int32_t batch, std::span<const size_t> syevd_lwork,
                          QdNevpt2Form form = QdNevpt2Form::state, bool pairs = true);

/// @brief @ref df_qd_sc_nevpt2_footprint over PC's pieces (@ref df_qd_pc_nevpt2
/// and @ref df_qd_pc_nevpt2_crossing): qd_pc_energy's H_eff, qd_pc_gradient's
/// W form with its passes, qd_pc_hessian_product, PC's term (ii); @p pairs a
/// direction the product refuses taken again with its partner. @p dropped as
/// theirs: per state, its diagonal PcClassEnergies::dropped, or empty for
/// the rank that holds the most.
[[nodiscard]] MemoryFootprint
df_qd_pc_nevpt2_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore, int32_t ncas,
                          int32_t nalpha, int32_t nbeta, int32_t nroots, bool rotating, int32_t k,
                          int32_t batch, std::span<const size_t> syevd_lwork,
                          QdNevpt2Form form = QdNevpt2Form::state,
                          std::span<const std::array<int64_t, 8>> dropped = {}, bool pairs = true);

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

/// @brief What @ref df_rhf holds over @p basis and @p aux with @p nmo orbitals,
/// @p nocc of them occupied: D, -W, gmmwich's diagonal and work beside
/// @c one_electron_hessian_footprint's scratch, freed; then
/// @ref df_rhf_two_electron_footprint, at its least; then
/// @ref df_rhf_response_footprint, whose U^x (response.cphf.solution) is
/// returned and so persistent. Empty for nocc not in [1, nmo) or no AOs.
[[nodiscard]] MemoryFootprint df_rhf_footprint(const BasisSet &basis, const BasisSet &aux,
                                               int32_t nmo, int32_t nocc);

/// @brief The closed-shell RHF dipole derivative, dmu_c/dR_{A,k}, from the
/// orbitals' first-order response: for every coordinate x = 3 A + k,
///
///     dmu/dx = -tr(D mu^x) - 4 Sum_ai U^x_ai mu_ai + 2 Sum_ij S^x_ij mu_ij
///              + Z_A e_k,
///
/// mu_c = <m| r_c |n> about the origin (@c dipole), electrons of charge -1,
/// D = 2 C_o C_o^T, mu^x the integrals' nuclear derivative at fixed D
/// (@c dipole_gradient), S^x the overlap's (@c overlap_derivative) and mu_pq
/// = C_p^T mu C_q (docs/hessian/hessian.tex section rhf-plan, step 6). U^x is
/// the CPHF solution a Hessian solves anyway, @c df_rhf's
/// response.cphf.solution, so the derivative costs no response solve of its
/// own. For a neutral molecule it does not depend on the origin, and its sum
/// over the atoms vanishes to the CPHF's tolerance.
///
/// @p molecule is the molecule @p basis was placed on, and @p classes are
/// @c primitive_pair_classes of @p basis. @p c is the nao x nmo orbitals,
/// column-major (nmo = c.num_elements() / nao), the first @p nocc occupied, and
/// @p u is U^x as @c response::cphf returns it, (n_vir n_occ) x 3 n_atom with
/// U^x_ai at a + n_vir i of column x. The dipole integrals stop at d
/// (@c kDipoleMaxL), so a basis with an f shell is refused.
///
/// Holds S^x (3 n_atom nao^2 doubles) and the three dipole matrices with their
/// MO blocks (@ref rhf_dipole_derivative_footprint has all of it); past the
/// memory budget (res.memory_budget()) that is Unsupported, naming the bytes,
/// before anything is allocated. Queued on res.stream() and
/// synchronized: the call returns with everything done. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return dmu_c/dx (e, the dipole in e bohr per bohr), 3 n_atom x 3 row-major:
///         element (x, c) at 3 x + c; InvalidConfig if @p molecule is not of
///         basis.natoms() atoms, if @p c is not a whole number of nao columns,
///         if @p nocc is not in [1, nmo), if @p u is not (nmo - nocc) nocc
///         3 n_atom long, or for any input the integral calls refuse (an f
///         shell among them); Unsupported for memory past the memory budget
///         or 3 n_atom nao^2 past int32_t; or the GpuRuntime error of the first
///         copy, BLAS call or launch that failed.
[[nodiscard]] Result<std::vector<double>>
rhf_dipole_derivative(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                      std::span<const PrimitivePairClass> classes, const Buffers::device<double> &c,
                      int32_t nocc, const Buffers::device<double> &u);

/// @brief What @ref rhf_dipole_derivative holds over @p basis with @p nmo
/// orbitals, @p nocc of them occupied, all of it scratch or the workspace: D
/// beside @c dipole_gradient_footprint's scratch, freed; then the MO blocks,
/// the three P^c, C_o's half product and the device result, held to the end,
/// beside in turn U and the three dipole @c MatrixAOPair with
/// @c dipole_footprint's workspace, and S^x (@c overlap_derivative). Nothing
/// pinned. Empty for nocc not in [1, nmo) or no AOs.
[[nodiscard]] MemoryFootprint rhf_dipole_derivative_footprint(const BasisSet &basis, int32_t nmo,
                                                              int32_t nocc);

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
