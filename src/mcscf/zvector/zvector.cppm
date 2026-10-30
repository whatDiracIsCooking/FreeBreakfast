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
  /// @brief z_o, H~ z_o = -g_I, over the packed pairs (H~'s size()).
  Buffers::device<double> orbital;
  /// @brief z_c, dim x nroots column-major: root r's CI response to z_o,
  /// -(H_0 - E_r)^-1 (1 - sum_s c_s c_s^T) H'(z_o) c_r, off every root.
  Buffers::device<double> ci;
  /// @brief H~ products the conjugate gradients took (the residual's not
  /// counted).
  int32_t iterations;
  /// @brief ||H~ z_o + g_I||, from one more product at the solution.
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
/// I's own RDMs and the same orbitals, in the packed order. Its CI part is
/// zero, every root being an eigenvector of the same H_0, so z_o alone is
/// solved for and z_c follows from it.
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
[[nodiscard]] Result<ZVector> zvector(DeviceResources &res, RelaxedHessian &relaxed,
                                      const response::Hessian &hessian, const ci::Roots &roots,
                                      const double *gradient, double tolerance = 1e-10,
                                      int32_t max_iterations = 200);

} // namespace free_breakfast::mcscf
