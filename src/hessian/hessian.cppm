/// @file
/// @brief Analytic nuclear Hessians on the GPU: so far the response part of
/// the closed-shell DF-RHF one, everything the orbitals' first-order response
/// adds to the explicit second-derivative integrals, in the symmetric form
///
///     -4 B^T U - 2 (Sigma + Sigma^T) + 2 Omega + Theta,
///
/// over the CPHF right-hand sides B^x and solutions U^x. The README has each
/// term's derivation and the memory it holds.

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

} // namespace free_breakfast::hessian
