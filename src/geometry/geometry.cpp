/// @file
/// @brief optimize: BFGS under a trust radius over a caller's energy and
/// gradient, with the rigid-body motions projected out, and distance
/// constraints held by projection and a SHAKE-like correction.

module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import :rigid_body;
import :quasi_newton;

#include "diagnostics.h"

namespace free_breakfast::geometry {

// The distance between points a and b of x.
double point_distance(std::span<const double> x, int32_t a, int32_t b) {
  const auto ia = 3 * static_cast<size_t>(a);
  const auto ib = 3 * static_cast<size_t>(b);
  const double dx = x[ia] - x[ib];
  const double dy = x[ia + 1] - x[ib + 1];
  const double dz = x[ia + 2] - x[ib + 2];
  return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

// Row k the gradient of constraint k's distance at x: the unit vector from
// point b to point a at a, and its negative at b.
std::vector<std::vector<double>> constraint_rows(std::span<const double> x,
                                                 std::span<const DistanceConstraint> constraints) {
  std::vector<std::vector<double>> rows;
  for (const DistanceConstraint &c : constraints) {
    std::vector<double> row(x.size(), 0.0);
    const auto ia = 3 * static_cast<size_t>(c.atoms[0]);
    const auto ib = 3 * static_cast<size_t>(c.atoms[1]);
    const double r = point_distance(x, c.atoms[0], c.atoms[1]);
    for (size_t k = 0; k < 3; ++k) {
      const double e = (x[ia + k] - x[ib + k]) / r;
      row[ia + k] = e;
      row[ib + k] = -e;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

// z with M z = r, M k x k row-major, by Gaussian elimination with partial
// pivoting; nullopt when a pivot is under 1e-12 of M's largest |element|.
std::optional<std::vector<double>> solve_small(std::vector<double> m, std::vector<double> r) {
  const size_t k = r.size();
  double scale = 0.0;
  for (const double value : m) {
    scale = std::max(scale, std::abs(value));
  }
  for (size_t j = 0; j < k; ++j) {
    size_t pivot = j;
    for (size_t i = j + 1; i < k; ++i) {
      if (std::abs(m[i * k + j]) > std::abs(m[pivot * k + j])) {
        pivot = i;
      }
    }
    if (!(std::abs(m[pivot * k + j]) > 1e-12 * scale)) {
      return std::nullopt;
    }
    if (pivot != j) {
      for (size_t c = 0; c < k; ++c) {
        std::swap(m[j * k + c], m[pivot * k + c]);
      }
      std::swap(r[j], r[pivot]);
    }
    for (size_t i = j + 1; i < k; ++i) {
      const double f = m[i * k + j] / m[j * k + j];
      for (size_t c = j; c < k; ++c) {
        m[i * k + c] -= f * m[j * k + c];
      }
      r[i] -= f * r[j];
    }
  }
  for (size_t i = k; i-- > 0;) {
    double value = r[i];
    for (size_t c = i + 1; c < k; ++c) {
      value -= m[i * k + c] * r[c];
    }
    r[i] = value / m[i * k + i];
  }
  return r;
}

// Each constraint's tolerance: 1e-12 of its distance, and absolute under 1.
// Far below any test of it (1e-8 Angstrom is 1.9e-8 bohr) and far above the
// rounding of a distance of coordinates of order 10.
constexpr double kConstraintTolerance = 1e-12;

// x moved onto the constraints along the rows (their gradients at the
// geometry the step was taken from, as SHAKE moves along the bonds of the
// last step): x + sum_j mu_j rows_j, mu by Newton's method on the k
// distances, whose Jacobian is rows(x)_i . rows_j.
Status restore_constraints(std::span<double> x, const std::vector<std::vector<double>> &rows,
                           std::span<const DistanceConstraint> constraints) {
  const size_t k = constraints.size();
  constexpr int32_t kMaxIterations = 50;
  double worst = 0.0;
  for (int32_t iteration = 0; iteration <= kMaxIterations; ++iteration) {
    std::vector<double> residual(k);
    worst = 0.0;
    bool met = true;
    for (size_t i = 0; i < k; ++i) {
      const DistanceConstraint &c = constraints[i];
      residual[i] = point_distance(x, c.atoms[0], c.atoms[1]) - c.distance;
      worst = std::max(worst, std::abs(residual[i]));
      met = met && std::abs(residual[i]) <= kConstraintTolerance * std::max(1.0, c.distance);
    }
    if (met) {
      return {};
    }
    if (iteration == kMaxIterations) {
      break;
    }
    const std::vector<std::vector<double>> now = constraint_rows(x, constraints);
    std::vector<double> jacobian(k * k);
    for (size_t i = 0; i < k; ++i) {
      for (size_t j = 0; j < k; ++j) {
        jacobian[i * k + j] =
            std::inner_product(now[i].begin(), now[i].end(), rows[j].begin(), 0.0);
      }
      residual[i] = -residual[i];
    }
    const std::optional<std::vector<double>> mu = solve_small(std::move(jacobian), residual);
    if (!mu) {
      break;
    }
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] += (*mu)[j] * rows[j][i];
      }
    }
  }
  return err_numerical("geometry::optimize: the distance constraints cannot be restored after "
                       "a step (largest error {:.3e})",
                       worst);
}

// P B P + (I - P), P = I - Q Q^T over the orthonormal basis Q, n x n
// row-major: B on the free coordinates and the identity on Q's span, which
// keeps it positive definite with B.
std::vector<double> free_model(std::span<const double> hessian,
                               const std::vector<std::vector<double>> &basis) {
  const size_t n = basis.empty() ? 0 : basis.front().size();
  std::vector<double> projector(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    projector[i * n + i] = 1.0;
  }
  for (const std::vector<double> &q : basis) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        projector[i * n + j] -= q[i] * q[j];
      }
    }
  }
  std::vector<double> half(n * n, 0.0);
  std::vector<double> model(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        half[i * n + j] += projector[i * n + k] * hessian[k * n + j];
      }
    }
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        model[i * n + j] += half[i * n + k] * projector[k * n + j];
      }
    }
    for (size_t j = 0; j < n; ++j) {
      model[i * n + j] += (i == j ? 1.0 : 0.0) - projector[i * n + j];
    }
  }
  return model;
}

// One evaluated geometry beyond its energy: the basis the gradient and the
// step are projected off (the rigid motions, then the constraint rows
// orthonormalized after them), the constraint rows the next step's
// correction moves along, the multipliers, and under constraints the
// gradient before its projection, which the BFGS update's Lagrangian
// gradient is made of.
struct ProjectedPoint {
  std::vector<std::vector<double>> basis;
  std::vector<std::vector<double>> rows;
  std::vector<double> multipliers;
  std::vector<double> raw;
};

// The energy and the projected gradient at x, into gradient; the rest of the
// geometry's projection into point.
Result<double>
evaluate(const std::function<Result<double>(std::span<const double>, std::span<double>)> &f,
         std::span<const double> x, const Options &options, ProjectedPoint &point,
         std::span<double> gradient) {
  const double energy = FBF_TRY(f(x, gradient));
  if (!std::isfinite(energy) ||
      !std::ranges::all_of(gradient, [](double value) { return std::isfinite(value); })) {
    return err_numerical("geometry::optimize: the energy or gradient at an evaluated point is "
                         "not finite (energy {})",
                         energy);
  }
  point.basis.clear();
  if (options.project_rigid_body) {
    point.basis = rigid_body_basis(x, {});
  }
  if (!options.constraints.empty()) {
    const size_t k = options.constraints.size();
    point.raw.assign(gradient.begin(), gradient.end());
    point.rows = constraint_rows(x, options.constraints);
    // lambda = (A A^T)^-1 A g, the least-squares fit g ~ A^T lambda.
    std::vector<double> normal(k * k);
    std::vector<double> rhs(k);
    for (size_t i = 0; i < k; ++i) {
      for (size_t j = 0; j < k; ++j) {
        normal[i * k + j] = std::inner_product(point.rows[i].begin(), point.rows[i].end(),
                                               point.rows[j].begin(), 0.0);
      }
      rhs[i] =
          std::inner_product(point.rows[i].begin(), point.rows[i].end(), gradient.begin(), 0.0);
    }
    std::optional<std::vector<double>> lambda = solve_small(std::move(normal), std::move(rhs));
    // A row with nothing left after the rigid motions and the rows before it
    // (1e-8 of its own length, sqrt 2) is a constraint the others already
    // fix: the projection, and the multipliers, have no unique answer.
    bool independent = lambda.has_value();
    for (const std::vector<double> &row : point.rows) {
      std::vector<double> v = row;
      project_out(point.basis, v);
      const double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
      if (!(norm > 1e-8 * std::numbers::sqrt2)) {
        independent = false;
        break;
      }
      for (double &value : v) {
        value /= norm;
      }
      point.basis.push_back(std::move(v));
    }
    if (!independent) {
      return err_numerical("geometry::optimize: the {} distance constraints' gradients are "
                           "linearly dependent at this geometry, so some of them fix the same "
                           "motion",
                           k);
    }
    point.multipliers = std::move(*lambda);
  }
  project_out(point.basis, gradient);
  return energy;
}

// InvalidConfig for a constraint naming a point not among the n / 3, one
// point twice, the pair of an earlier one, or a distance not finite and
// above 0.
Status check_constraints(size_t n, std::span<const DistanceConstraint> constraints) {
  if (constraints.empty()) {
    return {};
  }
  if (n % 3 != 0) {
    return err_config("geometry::optimize: distance constraints need points in space, 3 "
                      "coordinates each, and {} is not divisible by 3",
                      n);
  }
  const auto points = static_cast<int64_t>(n / 3);
  for (size_t k = 0; k < constraints.size(); ++k) {
    const auto [a, b] = constraints[k].atoms;
    if (a < 0 || b < 0 || a >= points || b >= points) {
      return err_config("geometry::optimize: constraint {} names points [{}, {}], and there are "
                        "{} (0-based)",
                        k, a, b, points);
    }
    if (a == b) {
      return err_config("geometry::optimize: constraint {} names point {} twice", k, a);
    }
    if (!std::isfinite(constraints[k].distance) || !(constraints[k].distance > 0.0)) {
      return err_config("geometry::optimize: constraint {}'s distance is {}, and must be finite "
                        "and above 0",
                        k, constraints[k].distance);
    }
    for (size_t j = 0; j < k; ++j) {
      const auto [c, d] = constraints[j].atoms;
      if ((a == c && b == d) || (a == d && b == c)) {
        return err_config("geometry::optimize: constraints {} and {} both name points [{}, {}]", j,
                          k, a, b);
      }
    }
  }
  return {};
}

Result<Solution>
optimize(std::span<const double> x0,
         const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
             &energy_and_gradient,
         const Options &options, const std::function<Status(const Iteration &)> &on_iteration) {
  const size_t n = x0.size();
  if (n == 0 || !std::ranges::all_of(x0, [](double value) { return std::isfinite(value); })) {
    return err_config("geometry::optimize: x0 must be non-empty and finite, got {} coordinates", n);
  }
  if (options.project_rigid_body && n % 3 != 0) {
    return err_config("geometry::optimize: project_rigid_body needs points in space, 3 "
                      "coordinates each, and {} is not divisible by 3",
                      n);
  }
  // Negated comparisons, so a NaN option is refused too.
  if (!(options.max_force > 0.0) || !(options.rms_force > 0.0) ||
      !(options.max_displacement > 0.0) || !(options.rms_displacement > 0.0) ||
      !(options.initial_hessian > 0.0)) {
    return err_config("geometry::optimize: the four thresholds and initial_hessian must be above "
                      "0");
  }
  if (!(options.trust_radius > 0.0) || !(options.trust_radius <= options.max_trust_radius) ||
      options.max_steps < 0) {
    return err_config("geometry::optimize: needs 0 < trust_radius <= max_trust_radius and "
                      "max_steps >= 0, got {}, {} and {}",
                      options.trust_radius, options.max_trust_radius, options.max_steps);
  }
  FBF_TRY(check_constraints(n, options.constraints));
  const bool constrained = !options.constraints.empty();

  const auto converged = [&options](std::span<const double> g, std::span<const double> s) {
    const auto [g_max, g_rms] = max_and_rms(g);
    const auto [s_max, s_rms] = max_and_rms(s);
    return g_max <= options.max_force && g_rms <= options.rms_force &&
           s_max <= options.max_displacement && s_rms <= options.rms_displacement;
  };
  Solution solution{std::vector<double>(x0.begin(), x0.end()), 0.0, {}};
  const auto record = [&solution, &on_iteration, &options](
                          double energy, std::span<const double> x, std::span<const double> g,
                          std::span<const double> s, const ProjectedPoint &point,
                          double trust_radius, bool accepted) -> Status {
    const auto [g_max, g_rms] = max_and_rms(g);
    const auto [s_max, s_rms] = max_and_rms(s);
    Iteration it{energy, g_max, g_rms, s_max, s_rms, trust_radius, accepted, {}, {}};
    for (const DistanceConstraint &c : options.constraints) {
      it.distances.push_back(point_distance(x, c.atoms[0], c.atoms[1]));
    }
    it.multipliers = point.multipliers;
    solution.iterations.push_back(std::move(it));
    if (on_iteration) {
      FBF_TRY(on_iteration(solution.iterations.back()));
    }
    return {};
  };

  std::vector<double> &x = solution.x;
  if (constrained) {
    FBF_TRY(restore_constraints(x, constraint_rows(x, options.constraints), options.constraints));
  }
  std::vector<double> g(n);
  ProjectedPoint point;
  solution.energy = FBF_TRY(evaluate(energy_and_gradient, x, options, point, g));
  double trust_radius = options.trust_radius;
  const std::vector<double> no_step(n, 0.0);
  FBF_TRY(record(solution.energy, x, g, no_step, point, trust_radius, true));
  if (converged(g, no_step)) {
    return solution;
  }

  std::vector<double> hessian(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    hessian[i * n + i] = options.initial_hessian;
  }
  std::vector<double> x_trial(n);
  std::vector<double> g_trial(n);
  ProjectedPoint point_trial;
  std::vector<double> hs(n);
  std::vector<double> y(n);
  std::vector<double> s(n);
  std::vector<double> s_free(n);

  for (int32_t step = 0; step < options.max_steps; ++step) {
    // Under constraints the model is optimize_meci's, P B P + (I - P), so the
    // Cholesky solve gives -(P B P)^+ g, B on the free coordinates alone: the
    // projection after an unprojected solve, -P B^-1 g, would let B's
    // coupling to the constrained directions turn the free step.
    std::vector<double> p = FBF_TRY(newton_step(
        "geometry::optimize", constrained ? free_model(hessian, point.basis) : hessian, g));
    project_out(point.basis, p);
    const double length = std::sqrt(std::inner_product(p.begin(), p.end(), p.begin(), 0.0));
    if (length > trust_radius) {
      for (double &value : p) {
        value *= trust_radius / length;
      }
    }
    const double s_norm = std::min(length, trust_radius);

    // The quadratic model's change, g.p + p.B.p / 2.
    for (size_t i = 0; i < n; ++i) {
      hs[i] = std::inner_product(p.begin(), p.end(), hessian.begin() + i * n, 0.0);
    }
    double php = std::inner_product(p.begin(), p.end(), hs.begin(), 0.0);
    const double predicted = std::inner_product(g.begin(), g.end(), p.begin(), 0.0) + 0.5 * php;

    for (size_t i = 0; i < n; ++i) {
      x_trial[i] = x[i] + p[i];
    }
    // The step taken: p, or under constraints p and its correction back onto
    // them, which B is then updated over.
    if (constrained) {
      FBF_TRY(restore_constraints(x_trial, point.rows, options.constraints));
      for (size_t i = 0; i < n; ++i) {
        s[i] = x_trial[i] - x[i];
      }
      // The update is over the free coordinates at x, as optimize_meci's.
      s_free = s;
      project_out(point.basis, s_free);
      for (size_t i = 0; i < n; ++i) {
        hs[i] = std::inner_product(s_free.begin(), s_free.end(), hessian.begin() + i * n, 0.0);
      }
      php = std::inner_product(s_free.begin(), s_free.end(), hs.begin(), 0.0);
    } else {
      s = p;
      s_free = p;
    }
    const double energy =
        FBF_TRY(evaluate(energy_and_gradient, x_trial, options, point_trial, g_trial));

    // Under constraints, optimize_meci's rounding floor: a change under 1000
    // ulps of the energies neither rejects the step nor moves the trust
    // radius. Near a tight convergence a step's energy change, f^2 / 2k, is
    // under the energy's rounding, and without the floor that noise rejects
    // steps and collapses the radius, so the walk stalls short of its
    // thresholds.
    const double noise = constrained ? 1e3 * std::numeric_limits<double>::epsilon() *
                                           std::max(std::abs(energy), std::abs(solution.energy))
                                     : 0.0;
    if (energy > solution.energy + noise) {
      trust_radius = 0.25 * s_norm;
      FBF_TRY(record(energy, x_trial, g_trial, s, point_trial, trust_radius, false));
      continue;
    }

    // The update's agreement bounds: Nocedal and Wright's Algorithm 4.1.
    if (!constrained || std::abs(predicted) > noise) {
      const double ratio = predicted < 0.0 ? (energy - solution.energy) / predicted : 0.0;
      if (ratio < 0.25) {
        trust_radius = 0.25 * s_norm;
      } else if (ratio > 0.75 && s_norm >= 0.99 * trust_radius) {
        trust_radius = std::min(2.0 * trust_radius, options.max_trust_radius);
      }
    }

    // BFGS: B + y y^T / (y.s) - (B s)(B s)^T / (s.B.s). Skipped unless the
    // curvature y.s is positive past rounding, which is what keeps B
    // positive definite. Under constraints y is the change in the
    // Lagrangian's gradient g - sum_k lambda_k dd_k/dx at the new
    // multipliers, as SQP's is: the new projected gradient less the old raw
    // one with the new multipliers' constraint forces taken off. That is
    // what carries the constraints' curvature, lambda_k d2d_k/dx2, into B;
    // the difference of the two projected gradients would mix in the
    // multipliers' own change along the old constraint gradients.
    for (size_t i = 0; i < n; ++i) {
      y[i] = g_trial[i] - g[i];
    }
    if (constrained) {
      for (size_t i = 0; i < n; ++i) {
        double old = point.raw[i];
        for (size_t k = 0; k < options.constraints.size(); ++k) {
          old -= point_trial.multipliers[k] * point.rows[k][i];
        }
        y[i] = g_trial[i] - old;
      }
      project_out(point.basis, y);
    }
    const double ys = std::inner_product(y.begin(), y.end(), s_free.begin(), 0.0);
    const double y_norm = std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0));
    const double step_norm =
        std::sqrt(std::inner_product(s_free.begin(), s_free.end(), s_free.begin(), 0.0));
    if (ys > std::numeric_limits<double>::epsilon() * y_norm * step_norm && php > 0.0) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          hessian[i * n + j] += y[i] * y[j] / ys - hs[i] * hs[j] / php;
        }
      }
    }

    std::swap(x, x_trial);
    std::swap(g, g_trial);
    std::swap(point, point_trial);
    solution.energy = energy;
    FBF_TRY(record(energy, x, g, s, point, trust_radius, true));
    if (converged(g, s)) {
      return solution;
    }
  }
  return err_max_iterations_exceeded(
      "geometry::optimize: not converged after {} steps (max force {}, rms force {})",
      options.max_steps, solution.iterations.back().max_force,
      solution.iterations.back().rms_force);
}

} // namespace free_breakfast::geometry
