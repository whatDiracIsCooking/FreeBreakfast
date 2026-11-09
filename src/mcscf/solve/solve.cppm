/// @file
/// @brief The two-step, state-averaged CASSCF loop on the device: per macro
/// iteration the integrals over the current orbitals, the CI, the weighted RDMs, the energy and gradient, and an
/// augmented-Hessian orbital step under a trust radius.
///
/// See README.md for the loop, its defaults and what each one settles.

export module free_breakfast.mcscf.solve;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
// Re-exported: Options holds ci::SolverOptions and the solution ci::Roots.
export import free_breakfast.ci.solver;

export namespace free_breakfast::mcscf {

/// @brief Which of the loop's builds are density-fitted, over the B
/// @ref solve is given.
enum class DensityFitting : uint8_t {
  Direct,        ///< None: every J and K integral-direct, B unused
  ApproxHessian, ///< The orbital Hessian's alone (PySCF's approx_hessian); the
                 ///< integrals, energy and gradient stay integral-direct, so
                 ///< the loop converges to the exact CASSCF
  Full,          ///< Every one: the integrals, the Fock, the gradient and
                 ///< the Hessian, so the loop converges to the DF-CASSCF
                 ///< and no J or K is integral-direct
};

/// @brief The knobs of one solve call.
struct Options {
  /// @brief Converged when |Delta E| of the weighted energy over the last
  /// accepted macro iteration is below this (hartree) ...
  double conv_tol = 1e-10;
  /// @brief ... and the orbital gradient's 2-norm ||g|| is below this.
  double conv_tol_grad = 1e-6;
  /// @brief Macro iterations, rejected ones included, before
  /// MaxIterationsExceeded. At least 1.
  int32_t max_cycle = 50;
  /// @brief The initial and largest trust radius: the 2-norm bound of a step
  /// x in the packed pairs, radians for a single pair.
  double trust_radius = 0.5;
  /// @brief Augmented-Hessian solves per macro iteration on its frozen
  /// integrals (the quadratic model), each from the model gradient the last
  /// one left. At least 1.
  int32_t max_micro = 4;
  /// @brief The loop's tuning, for tests and experiments, none a TOML key:
  /// each default is the value the loop was measured at (README.md
  /// "Defaults, and what each settles"). A rise of the weighted energy past
  /// energy_rise_tolerance (hartree) rejects a step. At least 0.
  double energy_rise_tolerance = 1e-11;
  /// @brief The CI's residual tolerance schedule, min(ci_loosest,
  /// ci_per_gradient ||g||^2) of the last iteration's ||g||, held at or above
  /// @ref ci_tolerance_for's floor. Both above 0.
  double ci_loosest = 1e-8;
  double ci_per_gradient = 1e-4;
  /// @brief The augmented Hessian's subspace solve: an absolute residual of
  /// ah_per_gradient ||g_model|| (above 0); at most ah_max_subspace columns
  /// before a restart, and ah_max_iterations added per micro iteration (both
  /// at least 1); and H~ shifted by ah_level_shift (PySCF's
  /// ah_level_shift, at least 0).
  double ah_per_gradient = 1e-2;
  int32_t ah_max_subspace = 40;
  int32_t ah_max_iterations = 50;
  double ah_level_shift = 1e-8;
  /// @brief Micro iterations stop once the model gradient is this fraction
  /// of ||g||. At least 0.
  double micro_gradient_fraction = 0.1;
  /// @brief The trust radius doubles when an accepted step's energy change is
  /// within this fraction of the model's. Above 0.
  double trust_agreement = 0.25;
  /// @brief A step along negative curvature the model has not earned is cut
  /// to negative_curvature_fraction of the radius (in (0, 1]); the gradient
  /// earns it when |g.x| is at least 1 / curvature_dominance of
  /// |x.H~x| / 2 (above 0).
  double negative_curvature_fraction = 0.25;
  double curvature_dominance = 10.0;
  /// @brief The CI's: its spin penalty, and in davidson.residual_tolerance the
  /// loosest floor the schedule reaches near convergence: the floor is
  /// @ref ci_tolerance_for, which tightens it when conv_tol_grad asks.
  ci::SolverOptions ci;
  /// @brief The screening of every integral-direct J and K build (none in
  /// DensityFitting::Full).
  double tau_j = 1e-14;
  double tau_k = 1e-14;
  /// @brief Which builds take B rather than J and K integral-direct.
  DensityFitting density_fitting = DensityFitting::Direct;
};

/// @brief The floor of the CI's relative residual tolerance near
/// convergence: @p tolerance (options.ci's), tightened so the absolute
/// residual, about that times max(1, |@p energy|), is at most 1e-2 of
/// @p conv_tol_grad, but never below 1e-14. A CI converged only to its
/// absolute residual r leaves ||g|| free to drift up to ~3.5 r: at 1e-12
/// relative, PSB3's r is 2.5e-10 (|E| ~ 248), and ||g|| stalled at 2e-10 to
/// 8e-10 (#867, README.md "The CI's floor under ||g||").
[[nodiscard]] double ci_tolerance_for(double conv_tol_grad, double energy, double tolerance);

/// @brief What one macro iteration did, for the log.
struct Iteration {
  /// @brief The weighted energy at this iteration's orbitals (hartree).
  double energy;
  /// @brief energy less the last accepted iteration's; 0 for the first.
  double energy_change;
  /// @brief ||g|| at this iteration's orbitals.
  double gradient_norm;
  /// @brief The trust radius the next step was taken under.
  double trust_radius;
  /// @brief Davidson iterations of this iteration's CI: its final,
  /// double-precision stage (ci::Roots::davidson), the final CI's added
  /// when it took one.
  int32_t ci_iterations;
  /// @brief Iterations of the single-precision stage before it
  /// (ci::Roots::fp32_iterations), 0 when there was none; the final CI's
  /// added likewise.
  int32_t ci_fp32_iterations;
  /// @brief Augmented-Hessian solves of the step taken from here (0 when
  /// converged or rejected).
  int32_t micro_iterations;
  /// @brief Orbital-Hessian columns applied for that step, each two J and two
  /// K builds when there is a core, and the CI's response to it (sigmas and
  /// two RDMs, no J or K): the step's cost.
  int32_t hessian_products;
  /// @brief false when the energy rose over the last accepted iteration, so
  /// these orbitals were dropped and the step retried at half the length.
  bool accepted;
};

/// @brief A converged solve.
struct Solution {
  /// @brief The optimized orbitals, AO by MO: core, active, virtual, as given.
  SphericalMatrixAO c;
  /// @brief The CI at @ref c, to @ref ci_tolerance_for's residual tolerance
  /// (options.ci's, or tighter when conv_tol_grad asks): each
  /// root's energy (hartree) and <S^2>, and the unit CI vectors on the
  /// device, root r at [r dim, (r + 1) dim).
  ci::Roots roots;
  /// @brief sum_r w_r E_r (hartree).
  double energy;
  /// @brief ||g|| at @ref c.
  double gradient_norm;
  /// @brief The weighted gamma, ncas x ncas, on the host (symmetric).
  std::vector<double> gamma;
  /// @brief The weighted Gamma, ncas^4 row-major (t, u, v, w), on the host:
  /// @c ci::Rdm::rdm12's layout.
  std::vector<double> gamma2;
  /// @brief Each root's gamma, root r at [r ncas^2, (r + 1) ncas^2).
  std::vector<double> root_gammas;
  /// @brief One entry per macro iteration run.
  std::vector<Iteration> iterations;
  /// @brief The weights w_r the average was solved at, one per root (equal
  /// ones as 1 / nroots, exactly). Empty in a Solution built without them,
  /// whose weights @c sa_gradient_weights then recovers from its RDMs.
  std::vector<double> weights;
};

/// @brief The state-averaged CASSCF of the @p nroots lowest roots of spin
/// S, 2S + 1 = @p molecule's multiplicity, so N_alpha - N_beta = 2S of the
/// @p nelecas, from the orbitals @p c
/// (columns [0, ncore) the core, [ncore, ncore + ncas) the active space).
///
/// @p basis, @p molecule, @p h, @p u and the two kinds of classes are what
/// @c fock takes; the classes must stay alive for the call, though
/// DensityFitting::Full reads neither them nor @p u. @p weights
/// are w_r, empty for equal ones. @p b is B over @p naux fitting functions,
/// as the density-fitted @c Hessian::create reads it
/// (@c DensityFitted::b, naux nmo^2), for the builds
/// options.density_fitting fits; null for none, which only
/// DensityFitting::Direct accepts. Its storage must outlive the call.
/// @p ci_guess is empty, or nroots dim device doubles on res's device, root r
/// at [r dim, (r + 1) dim), as @ref Solution::roots holds them: the first
/// macro iteration's CI starts from it rather than from ci::solve's own
/// guesses (a reference solution's CI, for a solve from its orbitals moved).
/// Empty, the solve is the one without it, to the bit.
///
/// Each macro iteration (the numbers are Options' defaults):
/// @c casscf_integrals over the current C; @c ci::solve
/// over the GPU sigma, warm-started from the last iteration's roots (the first
/// from @p ci_guess, or from its own guesses), its residual
/// tolerance min(1e-8, 1e-4 ||g||^2) of the last iteration, held at or above
/// options.ci's; @c ci::Rdm::rdm12 at @p weights and @c fock. Converged
/// when |Delta E| < conv_tol and ||g|| < conv_tol_grad; converged on a CI
/// looser than options.ci's, the CI is solved again at those orbitals to
/// options.ci's tolerance, warm, and the RDMs, energy and gradient rebuilt
/// over it, so the solution describes one (C, c), and the loop goes on if
/// they are no longer converged. A rise of the energy
/// over the last accepted iteration past 1e-11 hartree rejects the orbitals:
/// the step from the accepted ones is retried at half its length and the
/// trust radius halved. Otherwise up to max_micro augmented-Hessian solves of
/// the quadratic model at frozen integrals, the lowest root of [[0, g^T], [g,
/// H~]], H~ = H - H_oc (H_cc - E)^-1 H_co the Hessian with the CI relaxed
/// (@c Hessian H, and the CI's response to each column by conjugate
/// gradients over the sigma), give the step x, at most the trust radius
/// long; the solves share one subspace, grown by
/// diagonally preconditioned residuals, so a later one pays only for what the
/// earlier ones' products lack; and
/// C <- C exp(kappa(x)).
/// The radius doubles (up to options.trust_radius) when an accepted step's
/// energy change is within 25% of the model's. A step along negative
/// curvature (x.H~x < 0) is cut to a quarter of the radius unless the last
/// accepted step agreed so and the gradient drives it (|g.x| at least a tenth
/// of |x.H~x| / 2); the README's "The trust region" has the measurements.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes many times per iteration. The device res.dev_idx() names is
/// made current for the call and the previous one restored.
///
/// @return The solution; InvalidConfig if ncore is negative, ncas is below 1,
///         ncore + ncas exceeds nmo, nelecas is outside [0, 2 ncas], the
///         multiplicity does not fit it (2S above nelecas, of the other
///         parity, or N_alpha above ncas), nroots
///         is below 1 or above the determinant count, a weight is negative,
///         the weights do not sum to 1 within 1e-12 or their count is not
///         nroots, an option is out of range, @p c is not of @p basis's
///         spherical order, options.density_fitting needs B and @p b is
///         null, @p b is given but not naux nmo^2 doubles over naux >= 1, or
///         @p ci_guess is neither empty nor nroots dim doubles;
///         MaxIterationsExceeded once max_cycle runs out; or
///         the first error of an integral, CI, RDM, Fock, Hessian or rotation
///         call.
[[nodiscard]] Result<Solution>
solve(DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
      const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
      std::span<CoulombPairClass> j_classes, std::span<OrderedPairClass> k_classes,
      const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nelecas, int32_t nroots,
      std::span<const double> weights = {}, const Options &options = {},
      const Buffers::device<double> *b = nullptr, int32_t naux = 0,
      std::span<const double> ci_guess = {});

/// @brief What @ref solve holds over @p basis (nmo = its spherical AOs) with
/// @p ncore core orbitals and @p nelecas electrons in @p ncas active ones,
/// spin @p multiplicity, @p nroots roots, @p options, and, for a
/// density-fitted options.density_fitting, @p naux fitting functions:
///
/// - persistent: the Solution's C and CI vectors.
/// - scratch: at its most, what the loop holds across its macro iterations
///   (the strings, the sigma's tables and its apply scratch at Layout::full,
///   which the CI response's products grow it to; C twice, the diagonal, the
///   RDMs, the step, kappa and its eigenpairs and exponential, the augmented
///   Hessian's subspace and the last roots), with one macro iteration's
///   largest phase over it: @c casscf_integrals'; or the CI's, warm from the
///   last roots (cold from its own guesses in the first); or the Fock's, the
///   Hessian's create and the relaxed Hessian's and one of its products,
///   each on the integrals and the ones before it.
/// - workspace: the largest of every routine's it calls: the integrals', the
///   CI's, the RDMs' and transition RDMs', the Fock's, the Hessian's create
///   and product, the 1-RDM's, the augmented Hessian's syevd at every order
///   of its subspace (the most it takes, reached when the subspace fills),
///   and skew_eig's and skew_expm's at nmo.
///
/// @p syevd_lwork is syevd's work by order (FootprintDimensions::syevd_lwork)
/// through at least the larger of ci::solve_footprint's and the augmented
/// Hessian's ah_max_subspace + 1, and @p skew_eig_lwork skew_eig's at nmo
/// (@c skew_eig_lwork). The loop's steady state is counted: a macro
/// iteration after a product grew the sigma's scratch, and a product after
/// the first of its iteration; a solve that converges before either does not
/// reach it. Empty for a problem @ref solve refuses before it allocates.
[[nodiscard]] MemoryFootprint solve_footprint(const BasisSet &basis, int32_t ncore, int32_t ncas,
                                              int32_t nelecas, int32_t multiplicity, int32_t nroots,
                                              const Options &options, int32_t naux,
                                              std::span<const size_t> syevd_lwork,
                                              size_t skew_eig_lwork);

} // namespace free_breakfast::mcscf
