/// @file
/// @brief A Cartesian quasi-Newton minimizer: BFGS under a trust radius, with
/// the rigid-body translations and rotations projected out, over a
/// caller-supplied energy and gradient; its counterpart for the minimum-energy
/// crossing of two states; the intrinsic reaction coordinate from a
/// transition state; and the harmonic vibrational analysis of a Cartesian
/// Hessian.
///
/// optimize, optimize_meci and irc are host-side: the state is a 3N vector
/// and a 3N x 3N Hessian, and they know nothing about the SCF or the
/// pipeline (irc reads only the atoms' elements, for their masses).
/// harmonic_analysis takes the atoms, for their masses, and diagonalizes on
/// the GPU. See README.md for the methods and their defaults.

export module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.atom;

export namespace free_breakfast::geometry {

/// @brief One interatomic distance optimize holds fixed: points atoms[0] and
/// atoms[1] (0-based, 3 coordinates each) at distance (the callback's length
/// unit, bohr for a molecule).
struct DistanceConstraint {
  std::array<int32_t, 2> atoms;
  double distance;
};

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
  /// @brief Distances optimize holds fixed while it minimizes over the rest:
  /// the start is moved onto them, and every step is projected off their
  /// gradients and moved back onto them, so the four force norms are of the
  /// gradient projected off them too. Needs a coordinate count divisible by
  /// 3. optimize_meci refuses any.
  std::vector<DistanceConstraint> constraints{};
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
  /// @brief One per options.constraints, in order: its distance at this
  /// geometry, and its multiplier lambda_k, the least-squares fit of the
  /// gradient g = sum_k lambda_k dd_k/dx + (the projected gradient). At a
  /// converged geometry lambda_k is dE/dd_k, the energy's slope along the
  /// constrained distance with the rest relaxed (hartree / bohr). Empty with
  /// no constraints.
  std::vector<double> distances{};
  std::vector<double> multipliers{};
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
/// Under options.constraints (SHAKE/RATTLE-like): x0 is first moved onto
/// them; the gradient is also projected off the constraints' gradients at x
/// (orthonormalized after the rigid motions), and p is -(P B P)^+ g, P the
/// projector off all of them, by a Cholesky solve of P B P + (I - P) as in
/// optimize_meci; x + p is moved back onto them along those same gradients,
/// by Newton's method on the distances, to 1e-12 of each (relative, and
/// absolute under 1). The step's two norms are of that corrected step; B is
/// updated over its projection P s and y the change in the Lagrangian's
/// gradient g - sum_k lambda_k dd_k/dx at the new multipliers, projected by
/// P. And an energy change under 1000 ulps of the energies, as in
/// optimize_meci, neither rejects the step nor moves the trust radius.
///
/// @return The solution; InvalidConfig if @p x0 is empty or not finite, its
///         size is not divisible by 3 under options.project_rigid_body, a
///         threshold or the initial Hessian is not above 0, the trust radius
///         is not above 0 or above max_trust_radius, max_steps is
///         negative, or a constraint names a point out of range or one point
///         twice, repeats another's pair, or has a distance not finite and
///         above 0 (or there are constraints and the size is not divisible
///         by 3); Numerical if an energy or gradient is not finite, the
///         constraints' gradients are linearly dependent at a geometry, or
///         Newton's method cannot move a point back onto them;
///         MaxIterationsExceeded once max_steps runs out; or the first error
///         of @p energy_and_gradient, or of @p on_iteration.
///
/// @p on_iteration, if set, is called with each evaluation's Iteration as it
/// is recorded, the first at @p x0, so a caller keeps the walk's log on an
/// error that loses the Solution; an error it returns ends the optimization
/// with that error.
[[nodiscard]] Result<Solution>
optimize(std::span<const double> x0,
         const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
             &energy_and_gradient,
         const Options &options = {},
         const std::function<Status(const Iteration &)> &on_iteration = {});

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
    /// @brief true when the callback gave the mean gradient alone here, and
    /// the plane (with |x|) is carried from the last evaluation that had one.
    bool plane_carried;
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
/// of h_ij is read, so its scale and sign are free. A callback with no
/// branching plane at @p x (nevpt2::qd_pc_intersection inside a degenerate
/// CASSCF pair, README.md) writes the mean gradient into both @p g_i and
/// @p g_j and zeros into @p h_ij: the plane, x^, y^ and |x|, is then carried
/// from the last evaluation, x^ and y^ re-orthonormalized against this
/// geometry's rigid motions, and only P g is new.
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
/// @return The solution; InvalidConfig as optimize, max_gap not above 0, or
///         any options.constraints;
///         Numerical if an energy or vector is not finite, the gradient
///         difference vanishes after the projection (but for a mean gradient
///         alone), or the first evaluation is a mean gradient alone, with no
///         plane to carry; MaxIterationsExceeded
///         once max_steps runs out; or the first error of
///         @p energies_and_gradients, or of @p on_iteration.
///
/// @p on_iteration, if set, is called with each evaluation's Iteration as it
/// is recorded, as optimize's is.
[[nodiscard]] Result<MECISolution>
optimize_meci(std::span<const double> x0,
              const std::function<Result<std::array<double, 2>>(
                  std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                  std::span<double> h_ij)> &energies_and_gradients,
              const Options &options = {},
              const std::function<Status(const MECISolution::Iteration &)> &on_iteration = {});

/// @brief The knobs of one irc call. Mass-weighted units are amu^1/2 bohr for
/// a coordinate and hartree / (amu^1/2 bohr) for a gradient.
struct IrcOptions {
  /// @brief The step s along the path (amu^1/2 bohr, mass-weighted): point
  /// k + 1 is on the hypersphere of radius s / 2 about the pivot s / 2 down
  /// the steepest descent from point k, so it is k + 1 steps of s along the
  /// path from the start. Above 0.
  double step_size = 0.1;
  /// @brief The points after the start before the walk stops
  /// (IrcStop::MaxPoints). At least 1.
  int32_t max_points = 50;
  /// @brief A point whose Cartesian gradient, projected off the rigid
  /// motions, has its largest |component| at or below this (hartree / bohr;
  /// Options::max_force's default) is a minimum, and ends the walk
  /// (IrcStop::Minimum). Above 0.
  double max_force = 4.5e-4;
  /// @brief The hypersphere optimization of a point converges when the part
  /// of the mass-weighted gradient tangent to the sphere has 2-norm at or
  /// below this (hartree / (amu^1/2 bohr)) ... Above 0.
  double sphere_gradient = 1e-4;
  /// @brief ... and the next correction the quadratic model predicts on the
  /// sphere has 2-norm at or below this (amu^1/2 bohr). Above 0.
  double sphere_displacement = 1e-3;
  /// @brief The energy-and-gradient evaluations one point may take before
  /// the walk stops unconverged (IrcStop::SphereNotConverged). At least 1.
  int32_t max_sphere_iterations = 20;
  /// @brief Project the rigid-body motions out of the gradient, the step and
  /// the Hessian, as Options::project_rigid_body, in mass-weighted
  /// coordinates. Off for a function of coordinates that are not points in
  /// space.
  bool project_rigid_body = true;
};

/// @brief Why an irc walk stopped.
enum class IrcStop : uint8_t {
  /// The last point's force is at or below max_force.
  Minimum,
  /// The model puts the minimum along the last point's steepest descent
  /// within half a step, inside the next hypersphere: the path's minimum is
  /// within a step of the last point.
  MinimumWithinStep,
  /// The next point's energy rose over the last's, or it lies behind its
  /// pivot: the path passed its minimum, and that point is not kept.
  EnergyRose,
  /// max_points points were taken.
  MaxPoints,
  /// A point's hypersphere optimization ran out of max_sphere_iterations;
  /// that point is not kept.
  SphereNotConverged,
};

/// @brief The name of @p stop as the history writes it: "minimum",
/// "minimum_within_step", "energy_rose", "max_points" or
/// "sphere_not_converged".
[[nodiscard]] std::string_view irc_stop_name(IrcStop stop);

/// @brief One point of the path.
struct IrcPoint {
  /// @brief The arc length from the start (amu^1/2 bohr): k step_size at
  /// point k, the start's 0.
  double arc_length;
  /// @brief The energy here.
  double energy;
  /// @brief The coordinates (bohr), in the order the start gave them.
  std::vector<double> x;
  /// @brief The 2-norm and the largest |component| of the Cartesian gradient
  /// projected off the rigid motions (hartree / bohr).
  double gradient_norm;
  double max_force;
  /// @brief The evaluations this point's hypersphere optimization took; 0 at
  /// the start.
  int32_t sphere_iterations;
  /// @brief The 2-norm of the mass-weighted gradient's part tangent to the
  /// sphere here, at convergence; 0 at the start.
  double tangential_gradient;
};

/// @brief One direction's walk: the start, then each point taken.
struct IrcPath {
  std::vector<IrcPoint> points;
  IrcStop stop;
};

/// @brief The intrinsic reaction coordinate from the transition state
/// @p start down one side, by Gonzalez and Schlegel's second-order method
/// (GS2; J. Chem. Phys. 90, 2154 (1989)) in mass-weighted Cartesians, the
/// masses harmonic_analysis's.
///
/// @p energy_and_gradient(x, gradient) is optimize's callback: the energy at
/// @p x (bohr, 3 per atom) and its gradient written into @p gradient. Its
/// first call is at @p start, the path's point 0.
///
/// Each step, from point k at q_k (mass-weighted) with the gradient g_k: the
/// pivot q* = q_k + (s / 2) u, u the unit steepest descent -g_k / |g_k| (at
/// the start, @p direction's unit mass-weighted displacement instead, the
/// TS's gradient being 0 there); then point k + 1, the minimum of the energy
/// on the hypersphere |q - q*| = s / 2, where the gradient is parallel to
/// q - q*. That minimum is found by a constrained Newton iteration on a
/// quadratic model: from the trial p = q - q* with gradient g and the model
/// Hessian H (diagonalized over the internal coordinates, H v_i = b_i v_i),
/// the next p' has components (b_i p_i - g_i) / (b_i - lambda), lambda below
/// every b_i and fixed by |p'| = s / 2 (the trust-region secular equation,
/// solved by bisection; the hard case, nothing along the lowest v, takes
/// lambda = b_0 and makes up the length along v_0). Each trial is one
/// evaluation, and H gets Bofill's update (SR1 and Powell's symmetric
/// Broyden mixed by the SR1 denominator's cosine; it stays indefinite near
/// the TS, where BFGS would not) over every two evaluations in a row. H
/// starts as @p hessian mass-weighted. The first trial comes from the model
/// at q_k itself, which lies on the sphere, so a quadratic surface costs one
/// evaluation per point. Converged when the tangential gradient and the next
/// correction are at or below options.sphere_gradient and
/// options.sphere_displacement.
///
/// The walk stops at a point whose force is at or below options.max_force;
/// before a step whose pivot would pass the model's minimum along the
/// steepest descent, |g| / (u H u) at or under s / 2 (GS2's sphere then
/// holds the minimum, and its own minimum flips between the sphere's two
/// sides); at a point whose energy rose over the last's or that lies behind
/// its pivot (not kept); at options.max_points; or when a point's sphere does
/// not converge in options.max_sphere_iterations (not kept). The first three
/// leave the last point within about a step of the path's minimum.
///
/// @param start      The transition state: its positions are x0, its
///                   elements give the masses.
/// @param hessian    The Cartesian Hessian at @p start (hartree / bohr^2),
///                   3N x 3N row-major, harmonic_analysis's layout.
/// @param direction  The Cartesian displacement the walk leaves along, 3N:
///                   the imaginary mode as harmonic_analysis gives it, or
///                   its negative for the other side. Only its direction is
///                   read.
/// @return The path; InvalidConfig if @p start is empty, an element is not 1
///         to 118, a position, Hessian element or @p direction is not finite,
///         the Hessian is not 3N x 3N, @p direction is not 3N or has nothing
///         left once the rigid motions are projected off it, or an option is
///         out of its range; Numerical if an energy or gradient is not
///         finite; or the first error of @p energy_and_gradient, or of
///         @p on_point.
///
/// @p on_point, if set, is called with each point as it is kept, the start
/// first, so a caller keeps the walk's log on an error that loses the path;
/// an error it returns ends the walk with that error.
[[nodiscard]] Result<IrcPath>
irc(std::span<const Atom> start, std::span<const double> hessian, std::span<const double> direction,
    const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
        &energy_and_gradient,
    const IrcOptions &options = {}, const std::function<Status(const IrcPoint &)> &on_point = {});

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

/// @brief What @ref harmonic_analysis takes for @p atoms, @p syevd_lwork
/// being syevd's work by order (@ref syevd_lwork at order k its element k),
/// through at least 3N: the projector, H_mw, the m x m projected Hessian and
/// the two spectra, scratch, m the internal modes (3N - 6, 3N - 5 linear);
/// the device workspace, the largest of syevd's work at 3N and at m and the
/// n x m products carved from it. Empty with no atoms or no internal mode,
/// where it allocates nothing.
[[nodiscard]] MemoryFootprint harmonic_analysis_footprint(std::span<const Atom> atoms,
                                                          std::span<const size_t> syevd_lwork);

} // namespace free_breakfast::geometry
