/// @file
/// @brief The closed-shell coupled-perturbed Hartree-Fock equations
/// M U = B for a block of right-hand sides at once, by preconditioned
/// conjugate gradients run in lockstep over the density-fitted J and K.
///
/// See README.md.

export module free_breakfast.response.cphf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast::response {

/// @brief The solution of one block of CPHF equations, on the device they
/// were solved on.
struct Cphf {
  /// @brief U, the (n_vir n_occ) x ncol block, column-major: column x's
  /// U_ai at a + n_vir i, a over the virtuals and i over the occupied.
  Buffers::device<double> solution;
  /// @brief ||M U_x - B_x|| per column, from one more product at the
  /// solution.
  std::vector<double> residuals;
  /// @brief Lockstep sweeps: the products the slowest column took (the
  /// residuals' not counted).
  int32_t iterations;
};

/// @brief U for M U_x = B_x, every column x of @p rhs at once: closed-shell
/// RHF's CPHF, by preconditioned conjugate gradients from zero, run in
/// lockstep with each column stopped once ||M U_x - B_x|| <= @p tolerance.
///
/// M is the real singlet orbital Hessian of a closed-shell RHF over its
/// virtual-occupied pairs,
///   M_ai,bj = (eps_a - eps_i) delta_ab delta_ij + 4 (ai|bj) - (ab|ij) - (aj|ib),
/// never formed: a product is
///   (M U)_ai = (eps_a - eps_i) U_ai + C_a^T G(D_U) C_i,
///   D_U = 2 (X + X^T), X = C_v U C_o^T, G = J - K / 2,
/// D_U = 2 L R^T with L = [C_v U, C_o] and R = [C_o, C_v U]: J and K of a
/// block of columns' D_U by one batched @c df_coulomb and one batched
/// two-factor @c df_exchange, as many columns at once as half the device's
/// free memory holds. The preconditioner is (eps_a - eps_i)^-1, a pair whose
/// difference is below 1e-8 left unscaled.
///
/// @p b is B over @p naux fitting functions and C's nao AOs, as
/// @c df_coulomb reads it: naux nao^2 doubles. @p c is the nao x nmo
/// orbitals, column-major, the first @p nocc occupied; @p energies their nmo
/// orbital energies, on the host, the virtuals' above the occupied ones'. @p rhs
/// is B's right-hand sides, (n_vir n_occ) x ncol in the solution's layout,
/// n_vir = nmo - nocc; ncol is its length over n_vir n_occ.
///
/// Queued on res.stream() with res.blas() (host pointer mode); each column's
/// step synchronizes. Its scratch is its own, allocated per call:
/// @c df_coulomb and @c df_exchange take res.device_workspace(), which grows
/// to a batch's X and Y. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return The solution; InvalidConfig if nocc is not in [1, nmo), @p c is
///         not a whole number of nmo-long columns, @p b is not naux nao^2
///         long, @p rhs is empty or not a whole number of n_vir n_occ
///         columns, @p tolerance is not positive or @p max_iterations is
///         below 1; MaxIterationsExceeded when a column's residual is still
///         above @p tolerance after @p max_iterations sweeps; Divergence when
///         the conjugate gradients meet a direction M is not positive along,
///         which is the RHF reference being unstable to a real singlet
///         rotation; or the first error of a J or K build, BLAS call or copy.
[[nodiscard]] Result<Cphf> cphf(DeviceResources &res, const Buffers::device<double> &b,
                                int32_t naux, const Buffers::device<double> &c,
                                std::span<const double> energies, int32_t nocc,
                                const Buffers::device<double> &rhs, double tolerance = 1e-10,
                                int32_t max_iterations = 100);

} // namespace free_breakfast::response
