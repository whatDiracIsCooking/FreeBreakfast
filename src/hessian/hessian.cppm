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

} // namespace free_breakfast::hessian
