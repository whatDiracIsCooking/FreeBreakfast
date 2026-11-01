/// @file
/// @brief The Lagrange multipliers of one root's state-averaged CASSCF
/// gradient: z_o over the packed pairs by preconditioned conjugate gradients
/// over the CI-relaxed Hessian, and z_c, the CI's response to it.
///
/// Internal to the MCSCF layer: free_breakfast.mcscf does not re-export it.
/// See README.md.

export module free_breakfast.mcscf.zvector;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.solver;
import free_breakfast.response.hessian;
import free_breakfast.mcscf.relaxed_hessian;

export namespace free_breakfast::mcscf {

/// @brief One root's multipliers, on the device they were solved on.
struct ZVector {
  /// @brief z_o, H~ z_o = -g_I + H_oc (H_0 - E)^-1 P y, over the packed pairs
  /// (H~'s size()); y = 0 without a CI right-hand side.
  Buffers::device<double> orbital;
  /// @brief z_c, dim x nroots column-major: root r's CI response to z_o and
  /// the right-hand side, -(H_0 - E_r)^-1 P (H'(z_o) c_r + y_r), with
  /// P = 1 - sum_s c_s c_s^T; off every root.
  Buffers::device<double> ci;
  /// @brief The rotations among the roots, on the host: Theta_sr for r < s
  /// in (r, s) lexicographic order, nroots (nroots - 1) / 2 of them (none
  /// for one root), c_r gaining Theta_sr c_s (README.md "The rotations among
  /// the roots"). Zero for a pair of equal weights, where they are redundant.
  std::vector<double> theta;
  /// @brief H~ products the conjugate gradients took (the residual's not
  /// counted).
  int32_t iterations;
  /// @brief ||H~ z_o + g_I - H_oc (H_0 - E)^-1 P y||, from one more product
  /// at the solution.
  double residual;
};

/// @brief The Z-vector of root I of a state-averaged CASSCF: H~ z_o = -g_I by
/// conjugate gradients from zero, preconditioned by diag(H)^-1, to
/// ||r|| <= @p tolerance, and z_c from @ref RelaxedHessian::respond at z_o
/// with every root projected out.
///
/// @p relaxed is H~ at the state-averaged orbitals and roots; @p hessian and
/// @p roots are the ones it was created over (H's diagonal is the
/// preconditioner, a pair whose diagonal is below 1e-8 left unscaled).
/// @p gradient is g_I on the device, @c response::fock's gradient at root
/// I's own RDMs and the same orbitals, in the packed order.
///
/// @p ci_rhs is the CI right-hand side y on the device (dim x nroots,
/// column-major), or null for none: a CASSCF gradient's is zero, every root
/// being an eigenvector of the same H_0. The equations solved are
///   H z_o + H_oc z_c = -g_I,
///   (H_0 - E_r) z_c,r + P H'(z_o) c_r = -P y_r   for every root r,
/// the CI rows unweighted, in the coordinates of PySCF's gen_g_hop (README.md
/// "The CI right-hand side"); y must be in the roots' spin, and is projected
/// off every root here. z_c is eliminated, so z_o alone is solved for and z_c
/// follows from it.
///
/// For every pair of roots r < s at unequal weights the rotation Theta_sr
/// is a parameter too, its row the CI row's component along c_s:
///   (E_s - E_r) Theta_sr + c_s . H'(z_o) c_r = -t_sr,
/// eliminated with z_c (README.md "The rotations among the roots").
/// @p theta_rhs is t on the host, as ZVector::theta is laid out, or empty
/// for none (a CASSCF gradient's is zero: E_I is stationary in them); t_sr is
/// dL/dTheta_sr over 2 (w_r - w_s), so a derivative's 2 (w_r - w_s) H~_sr
/// gives H~_sr. A pair of equal weights is projected, its t ignored.
///
/// Queued on res.stream() with res.blas() (host pointer mode); every product
/// synchronizes. The device res.dev_idx() names is made current for the call
/// and the previous one restored.
///
/// @return The multipliers; InvalidConfig if @p hessian is not of
///         @p relaxed's size, @p roots holds no root or not a whole number of
///         vectors, @p tolerance is not positive or @p max_iterations below 1;
///         MaxIterationsExceeded when the residual is still above
///         @p tolerance after @p max_iterations products; Divergence when the
///         conjugate gradients meet a direction H~ is not positive along (not
///         a minimum); or the first error of a product, BLAS call or copy.
///         InvalidConfig too for a @p theta_rhs neither empty nor
///         nroots (nroots - 1) / 2 long.
[[nodiscard]] Result<ZVector> zvector(DeviceResources &res, RelaxedHessian &relaxed,
                                      const response::Hessian &hessian, const ci::Roots &roots,
                                      const double *gradient, const double *ci_rhs = nullptr,
                                      double tolerance = 1e-10, int32_t max_iterations = 200,
                                      std::span<const double> theta_rhs = {});

/// @brief How @ref zvectors solves its columns (README.md "Many right-hand
/// sides").
enum class CpSolver : uint8_t {
  /// @brief @ref zvector on each column in turn.
  PerColumn,
  /// @brief The Schur complement in lockstep: conjugate gradients over H~ on
  /// every column at once, one block product per sweep (H over the active
  /// columns in one call, their CI responses' conjugate gradients in one).
  Schur,
  /// @brief The unreduced equations in (z_o, z_c): conjugate gradients on
  /// the coupled operator, its CI rows scaled by 2 w_r to make it symmetric,
  /// preconditioned by diag(H) and 2 w_r |diag(H_0) - E_r|. No CI response
  /// is solved inside a product.
  Coupled,
};

/// @brief The multipliers of @ref zvectors' columns, on the device they were
/// solved on.
struct ZVectors {
  /// @brief z_o per column, size() x ncol column-major.
  Buffers::device<double> orbital;
  /// @brief z_c per column, one dim x nroots block (as ZVector::ci) each,
  /// off every root.
  Buffers::device<double> ci;
  /// @brief The rotations among the roots per column, on the host: one
  /// ZVector::theta (nroots (nroots - 1) / 2) each.
  std::vector<double> theta;
  /// @brief Products each column took (the residual's not counted): H~'s,
  /// or the coupled operator's.
  std::vector<int32_t> iterations;
  /// @brief Each column's true residual from one more product: PerColumn and
  /// Schur ||H~ z_o + g - H_oc (H_0 - E)^-1 P y||, ZVector::residual's (the
  /// rotations eliminated with z_c); Coupled the norm of every block of the
  /// equations, the CI and rotation rows unscaled.
  std::vector<double> residuals;
};

/// @brief @ref zvector on @p ncol right-hand sides at once: the CP-CASSCF
/// equations E_ll l^x = -E_lx of a Hessian's nuclear coordinates, or any
/// other set over the same H~.
///
/// @p gradients is size() x @p ncol on the device, one orbital right-hand
/// side per column; @p ci_rhs is null, or ncol blocks of dim x nroots (each
/// as zvector's @p ci_rhs); @p theta_rhs empty, or ncol of zvector's
/// @p theta_rhs on the host. Every method carries the rotations among the
/// roots at unequal weights. Each column is solved as @ref zvector solves it
/// alone, to ||r|| <= @p tolerance per column, the CI rows in zvector's
/// convention (unweighted, unscaled). @p solver picks the method, Schur by
/// default (the fastest measured; Coupled is slower on a larger molecule, so
/// it is not); the others reach the same solution to the CI response's
/// precision (README.md "Which is the default" has the measurements).
///
/// Coupled needs every weight positive (its CI rows are scaled by 2 w_r),
/// and H_0 - E_r positive off the roots along every direction its
/// preconditioner reaches: the roots must be the lowest of their spin. With
/// rotations it needs (w_r - w_s)(E_s - E_r) > 0 for every pair of unequal
/// weights (their rows are scaled by 2 (w_r - w_s)): the weights falling as
/// the energies rise.
///
/// @return The multipliers; InvalidConfig as @ref zvector's, for @p ncol
///         below 1, or for Coupled with a weight not positive or a pair
///         whose weights rise with its energies;
///         MaxIterationsExceeded when a column is still above @p tolerance
///         after @p max_iterations products; Divergence as zvector's, for the
///         operator a method iterates on; or the first error of a product,
///         BLAS call or copy.
[[nodiscard]] Result<ZVectors> zvectors(DeviceResources &res, RelaxedHessian &relaxed,
                                        const response::Hessian &hessian, const ci::Roots &roots,
                                        int32_t ncol, const double *gradients,
                                        const double *ci_rhs = nullptr,
                                        CpSolver solver = CpSolver::Schur, double tolerance = 1e-10,
                                        int32_t max_iterations = 200,
                                        std::span<const double> theta_rhs = {});

} // namespace free_breakfast::mcscf
