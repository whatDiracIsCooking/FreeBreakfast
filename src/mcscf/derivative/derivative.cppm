/// @file
/// @brief The per-coordinate derivative quantities of a DF-CASSCF at fixed
/// parameters, on the GPU: the derivative generalized Fock F^[x] and the
/// CP-CASSCF right-hand sides (casscf_hessian.tex eqs. fock, rhs-orb and
/// rhs-ci), every nuclear coordinate at once, at the RDMs of any weighting of
/// any roots. See README.md for the algebra and the layouts.

export module free_breakfast.mcscf.derivative;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast::mcscf {

/// @brief The derivatives of a CASSCF energy along every nuclear coordinate
/// x = 3 A + k at fixed parameters (the connection C T(x), rotation and CI
/// held), each stacked over x on its last index. MO indices are C's columns,
/// core, active, virtual. README.md has each layout and sign.
struct Derivatives {
  /// @brief F^[x], eq. fock at the derivative integrals (h^x and the fitted
  /// (pq|rs)^x, metric included) and the unperturbed RDMs, no connection:
  /// nmo x nmo column-major per x, element (p, q, x) at p + nmo (q + nmo x).
  /// Zero in the virtual columns.
  Buffers::device<double> fock;
  /// @brief The orbital right-hand side, eq. rhs-orb: G^x_pq = dE_x/dkappa_pq
  /// = 2 (F~^x_pq - F~^x_qp) for the non-redundant pairs p > q, G^x_qp =
  /// -G^x_pq, zero on the core-core, active-active and virtual-virtual
  /// blocks, F~^x = F^[x] + F[tau^x] (tau^x = -1/2 S^x_MO). Layout as fock.
  Buffers::device<double> rhs_orbital;
  /// @brief The CI right-hand side, eq. rhs-ci: column r of slab x is
  /// 2 w_r Q H~^x c_r, Q = 1 - sum_s c_s c_s^T over every root given,
  /// H~^x = H^[x] + H[tau^x]: n_det x nroots column-major per x, element
  /// (I, r, x) at I + n_det (r + nroots x).
  Buffers::device<double> rhs_ci;
  /// @brief gamma^x_sr = dE_x/dTheta_sr = 2 (w_r - w_s) <c_s|H~^x|c_r>, the
  /// right-hand side of the rotations among the roots (a parameter at
  /// unequal weights only): host, element (s, r, x) at
  /// s + nroots (r + nroots x).
  std::vector<double> rhs_theta;
  /// @brief <c_r|H~^x|c_s>, symmetrized, its core energy included and V_nn^x
  /// not: host, element (r, s, x) at r + nroots (s + nroots x). Its diagonal
  /// is root r's fixed-parameter gradient less V_nn^x.
  std::vector<double> hamiltonian;
  /// @brief With a rotation K given, F^[x] at the K-rotated densities
  /// K D - D K and K on each index of Gamma (the unperturbed D and Gamma, so a
  /// virtual index appears): d/de of F^[x] along C -> C (1 + e K) at fixed
  /// MO-basis RDMs, its outer index held, = F^[x] K^T + F^[x] at the
  /// integrals' inner three indices one-index transformed by K. Layout as
  /// fock, every column filled. Empty without a rotation.
  std::optional<Buffers::device<double>> fock_rotated;
};

/// @brief F^[x] and the CP-CASSCF right-hand sides of a DF-CASSCF, for every
/// nuclear coordinate, at the orbitals @p c and the weighted RDMs of the
/// @p roots at @p weights.
///
/// The energy is E = sum_r w_r <c_r|H|c_r> over the fit B, with the
/// connection first and the rotation second (casscf_hessian.tex eq. order):
/// one root at weight 1 is a state-specific CASSCF's (E_lambda x), the roots of
/// a state average at its weights are the SA right-hand sides at the averaged
/// densities, and one-hot weights over the same roots are root I's own
/// right-hand side, projected off every root. Nothing requires stationarity.
///
/// F^[x] is h^x (@c kinetic_derivative + @c potential_derivative), the
/// derivative J and K of the core density and of the active one
/// (@c df_jk_derivative, the active density through its natural orbitals),
/// and Sum_uvw (pu|vw)^x Gamma_tuvw over the fitted (pu|vw)^x, built by
/// walking the aux atoms (@c three_center_derivative, @c metric_derivative).
/// F[tau^x] and H[tau^x] are the same expressions at the integrals one-index
/// transformed by tau^x: the derivative of the undifferentiated F and H along
/// C -> C (1 + e tau^x), J and K of the two response densities batched over
/// the coordinates (the block @c df_coulomb and @c df_exchange), and
/// ppaa and papa (@c casscf_integrals). The CI part is one @c ci::Sigma
/// product per coordinate over H~^x. README.md has every term.
///
/// With @p rotation, an antisymmetric nao x nao K on the device (column-major,
/// MO indices C's columns), also F^[x] at the K-rotated densities
/// (Derivatives::fock_rotated): the z_o part of a state-averaged Hessian's
/// connection term, by the product rule on every inner C. Its J^x and K^x of
/// the rotated core and active densities are four more @c df_jk_derivative
/// calls, each indefinite density taken as the difference of two
/// closed-shell ones, and its active 2-RDM term one more walk over the aux
/// atoms. Without it (the default) nothing of this runs.
///
/// @p molecule is the molecule both bases were placed on (by @c place);
/// @p classes are @c primitive_pair_classes of @p basis; @p h the core
/// Hamiltonian; @p b and @p l the fit as @c density_fit leaves it (B
/// naux x nao^2, L the naux x naux lower triangle). @p c is nao x nao, columns
/// [0, ncore) core and [ncore, ncore + ncas) active. @p roots are
/// weights.size() unit CI vectors over the determinants of @p nalpha and
/// @p nbeta electrons in the ncas orbitals (@c ci::Sigma's Layout::full),
/// on the device, root r at r n_det; @p weights are w_r, each >= 0, summing
/// to 1 (@c ci::Rdm::rdm12's rule). Gamma is used 8-fold symmetrized.
///
/// Memory: about 20 stacks of 3 n_atom nao^2 doubles, besides
/// (3 n_atom)(nao ncas^3 + 2 naux ncas^2) and the derivative calls' own
/// scratch; past the device's free memory it is Unsupported, before anything
/// is allocated.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times. The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The derivatives; InvalidConfig if @p c is not nao x nao on
///         @p res's device, ncore is negative, ncas below 1 or ncore + ncas
///         past nao, @p b is not naux nao^2 long, the molecule is not the
///         bases', @p weights is empty, or the RDM's natural occupations are
///         negative past 1e-8; Unsupported for memory past the device's free
///         bytes or an index past int32_t; the first error of the strings,
///         @c ci::Rdm, @c casscf_integrals, @c response::fock, a derivative
///         call, a J/K build or the sigma; or the GpuRuntime error of the
///         first copy or BLAS call that failed.
[[nodiscard]] Result<Derivatives>
df_derivatives(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
               std::span<const PrimitivePairClass> classes, const BasisSet &aux,
               const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
               const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha,
               int32_t nbeta, const double *roots, std::span<const double> weights,
               const double *rotation = nullptr);

} // namespace free_breakfast::mcscf
