/// @file
/// @brief A Cartesian quasi-Newton minimizer: BFGS under a trust radius, with
/// the rigid-body translations and rotations projected out, over a
/// caller-supplied energy and gradient; its counterpart for the minimum-energy
/// crossing of two states; and the harmonic vibrational analysis of a
/// Cartesian Hessian.
///
/// optimize and optimize_meci are host-side: the state is a 3N vector and a
/// 3N x 3N Hessian, and they know nothing about molecules, the SCF or the
/// pipeline.
/// harmonic_analysis takes the atoms, for their masses, and diagonalizes on
/// the GPU. See README.md for the methods and their defaults.

export module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.atom;

export namespace free_breakfast::geometry {

/// @brief The knobs of one optimize call. Units are the callback's: bohr and
/// hartree for a molecule.
struct Options {
  /// @brief Converged when the largest |component| of the projected gradient
  /// is at or below this (hartree / bohr; Gaussian's default) ...
  double max_force = 4.5e-4;
  /// @brief ... its root mean square over the coordinates at or below this ...
  double rms_force = 3.0e-4;
  /// @brief ... the largest |component| of the step that reached the geometry
  /// at or below this (bohr) ...
  double max_displacement = 1.8e-3;
  /// @brief ... and that step's root mean square at or below this.
  double rms_displacement = 1.2e-3;
  /// @brief Energy-and-gradient evaluations after the first, rejected ones
  /// included, before MaxIterationsExceeded. At least 0.
  int32_t max_steps = 100;
  /// @brief The initial trust radius: the 2-norm bound of a step (bohr).
  double trust_radius = 0.3;
  /// @brief The largest the trust radius grows to.
  double max_trust_radius = 1.0;
  /// @brief The starting Hessian is this times the identity (hartree / bohr^2).
  double initial_hessian = 0.5;
  /// @brief Project the 3 translations and up to 3 rotations of the points
  /// about their centroid out of the gradient and the step, so the centroid
  /// stays where it is and no step rotates the points. Needs a coordinate
  /// count divisible by 3. Off for a function of coordinates that are not
  /// points in space.
  bool project_rigid_body = true;
  /// @brief optimize_meci's fifth criterion: converged only when the gap
  /// between the two energies is also at or below this (hartree). optimize
  /// ignores it.
  double max_gap = 1e-5;
};

/// @brief One energy-and-gradient evaluation, for the log.
struct Iteration {
  /// @brief The energy at this geometry.
  double energy;
  /// @brief The largest |component| and the root mean square of the projected
  /// gradient at this geometry.
  double max_force;
  double rms_force;
  /// @brief The same two of the step that reached this geometry; 0 for the
  /// first.
  double max_displacement;
  double rms_displacement;
  /// @brief The trust radius the next step is taken under.
  double trust_radius;
  /// @brief false when the energy rose over the last accepted geometry's, so
  /// this geometry was dropped and the step retried from the accepted one
  /// under a quarter of its length.
  bool accepted;
};

/// @brief A converged optimization.
struct Solution {
  /// @brief The optimized coordinates, in the order @p x0 gave them.
  std::vector<double> x;
  /// @brief The energy at @ref x.
  double energy;
  /// @brief One entry per evaluation, the first at @p x0.
  std::vector<Iteration> iterations;
};

/// @brief The local minimum of the energy @p energy_and_gradient computes,
/// from @p x0.
///
/// @p energy_and_gradient(x, gradient) returns the energy at @p x and writes
/// its gradient, x.size() values, into @p gradient; an error it returns ends
/// the optimization with that error.
///
/// Each step: the gradient g projected (when options.project_rigid_body),
/// the quasi-Newton step p = -B^-1 g projected again and scaled back to the
/// trust radius when longer, and the energy at x + p. A rise over the energy
/// at x rejects it, and the trust radius becomes |p| / 4. Otherwise the step
/// is taken, B gets the BFGS update over (p, the change in projected gradient)
/// when their product is positive, and the trust radius is |p| / 4 when the
/// energy change is under a quarter of the quadratic model's, doubles (up to
/// max_trust_radius) when it is over three quarters and p reached the radius,
/// and stays otherwise. Converged at an accepted geometry whose four norms
/// are all at or below their options; the first evaluation has no step, so
/// it converges on the forces alone.
///
/// @return The solution; InvalidConfig if @p x0 is empty or not finite, its
///         size is not divisible by 3 under options.project_rigid_body, a
///         threshold or the initial Hessian is not above 0, the trust radius
///         is not above 0 or above max_trust_radius, or max_steps is
///         negative; Numerical if an energy or gradient is not finite;
///         MaxIterationsExceeded once max_steps runs out; or the first error
///         of @p energy_and_gradient.
[[nodiscard]] Result<Solution>
optimize(std::span<const double> x0,
         const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
             &energy_and_gradient,
         const Options &options = {});

/// @brief A converged minimum-energy crossing of two states.
struct MECISolution {
  /// @brief One evaluation of the two states, for the log.
  struct Iteration {
    /// @brief The lower and the upper energy at this geometry, and their
    /// difference.
    double lower;
    double upper;
    double gap;
    /// @brief The largest |component| and the root mean square of the
    /// projected mean gradient P (g_I + g_J) / 2 at this geometry.
    double max_force;
    double rms_force;
    /// @brief The same two of the step that reached this geometry; 0 for the
    /// first.
    double max_displacement;
    double rms_displacement;
    /// @brief The trust radius the next step is taken under.
    double trust_radius;
    /// @brief 2 when the branching plane at this geometry is {x, y}; 1 when y,
    /// the coupling, had nothing left after x and was dropped.
    int32_t branching_dimension;
  };
  /// @brief The coordinates of the crossing, in the order @p x0 gave them.
  std::vector<double> x;
  /// @brief The lower and the upper energy at @ref x.
  double lower;
  double upper;
  /// @brief One entry per evaluation, the first at @p x0.
  std::vector<Iteration> iterations;
};

/// @brief The minimum-energy crossing of two states the callback computes,
/// from @p x0, by gradient projection (Bearpark, Robb and Schlegel; Sicilia,
/// Bearpark, Blancafort and Robb, JCTC 4, 257 (2008)).
///
/// @p energies_and_gradients(x, g_i, g_j, h_ij) returns the two energies at
/// @p x, and writes the gradient of each into @p g_i and @p g_j and their
/// coupling vector into @p h_ij, x.size() values each; an error it returns
/// ends the optimization with that error. The energies may come in either
/// order: the lower one, with its gradient, is taken as I. Only the direction
/// of h_ij is read, so its scale and sign are free.
///
/// Each step: the rigid-body motions (when options.project_rigid_body) are
/// projected off x = g_J - g_I and y = h_IJ, which are orthonormalized into
/// the branching plane; a y with nothing left after x (under 1e-8 of the
/// larger of |x| and |y|) is dropped, which leaves a seam of one more
/// dimension. The step is the Newton step on the gap along x,
/// -(E_J - E_I) / |x| x^, plus the BFGS step on the mean energy in the
/// intersection space, -(P B P)^+ P g, with g = (g_I + g_J) / 2 and P the
/// projector off the rigid motions and the plane; the sum is scaled back to
/// the trust radius when longer. No step is rejected: the composite step is
/// not the descent of any one function. Instead the trust radius follows the
/// ratio of the actual to the modelled change in the mean energy plus the gap:
/// outside [1/4, 7/4] it becomes |p| / 4, within [3/4, 5/4] with the step at
/// the radius it doubles, up to max_trust_radius. B gets the BFGS update over
/// the step and the change in P g, both in the old intersection space, when
/// their product is positive. Converged at a geometry whose gap is at or below
/// options.max_gap and whose four norms, of -P g and of the step, are all at
/// or below their options; the first evaluation converges on the gap and the
/// forces alone.
///
/// @return The solution; InvalidConfig as optimize, or max_gap not above 0;
///         Numerical if an energy or vector is not finite, or the gradient
///         difference vanishes after the projection; MaxIterationsExceeded
///         once max_steps runs out; or the first error of
///         @p energies_and_gradients.
[[nodiscard]] Result<MECISolution>
optimize_meci(std::span<const double> x0,
              const std::function<Result<std::array<double, 2>>(
                  std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                  std::span<double> h_ij)> &energies_and_gradients,
              const Options &options = {});

/// @brief The harmonic vibrations of a molecule: the internal modes alone,
/// the rigid-body translations and rotations removed.
struct Vibrations {
  /// @brief The harmonic frequencies (cm^-1), ascending; an imaginary one is
  /// reported as minus its magnitude.
  std::vector<double> frequencies;
  /// @brief The Cartesian normal modes, frequencies.size() x 3N row-major:
  /// mode j's displacement of atom A along axis k at modes[3N j + 3A + k].
  /// Each is its unit mass-weighted eigenvector divided by sqrt(m_A) (amu),
  /// PySCF's `norm_mode`; the sign is arbitrary.
  std::vector<double> modes;
};

/// @brief The harmonic frequencies and normal modes of the Cartesian Hessian
/// @p hessian (hartree / bohr^2) of the molecule @p atoms (bohr).
///
/// The Hessian is mass-weighted with the isotope-averaged masses (PySCF's
/// `elements.MASSES`, the standard atomic weights), the rigid-body motions
/// (three translations, and three rotations, two for a linear molecule, none
/// for one atom) are projected out by diagonalizing in the orthonormal
/// complement of their span, and each eigenvalue lambda becomes the frequency
/// sign(lambda) sqrt(|lambda|) in cm^-1. 3N - 6 modes (3N - 5 linear) are
/// returned. Both eigensolves are recipes' diagonalize on res's stream.
///
/// @param res      Supplies the stream, the BLAS and solver handles and the
///                 scratch workspace.
/// @param atoms    The molecule; only each atom's element and position are
///                 read.
/// @param hessian  3N x 3N, element (3A + k, 3B + l) at
///                 hessian[3N (3A + k) + 3B + l]; symmetric.
/// @return The vibrations; InvalidConfig if @p atoms is empty, an element is
///         not 1 to 118, a position or Hessian element is not finite, or the
///         Hessian is not 3N x 3N; Numerical if an eigensolve did not
///         converge; or a GpuRuntime error.
[[nodiscard]] Result<Vibrations> harmonic_analysis(DeviceResources &res,
                                                   std::span<const Atom> atoms,
                                                   std::span<const double> hessian);

} // namespace free_breakfast::geometry
