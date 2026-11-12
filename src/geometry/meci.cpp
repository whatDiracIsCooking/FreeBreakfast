/// @file
/// @brief optimize_meci: gradient projection onto the minimum-energy crossing
/// of two states, a Newton step on the gap and BFGS on the mean energy in the
/// intersection space.

module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import :rigid_body;
import :quasi_newton;

#include "diagnostics.h"

namespace free_breakfast::geometry {

using MeciCallback = std::function<Result<std::array<double, 2>>(
    std::span<const double>, std::span<double>, std::span<double>, std::span<double>)>;

// What one evaluation leaves for the step from it.
struct MeciPoint {
  double lower;
  double upper;
  // The rigid motions, then x^, then y^ when the plane kept it: P projects
  // off all of them.
  std::vector<std::vector<double>> basis;
  // x = g_J - g_I, rigid-projected, and its length.
  std::vector<double> gap_gradient;
  double gap_gradient_norm;
  bool coupled;
  // (g_I + g_J) / 2, and P applied to it.
  std::vector<double> mean_gradient;
  std::vector<double> projected;
  // The callback gave the mean gradient alone, and the plane is @p plane's,
  // the last evaluation that had one.
  bool carried = false;
};

// The callback at @p x. One that writes g_I = g_J and h = 0 has no branching
// plane there (a mean gradient alone): x^, y^ and |x| are then @p plane's,
// x^ and y^ re-orthonormalized against this geometry's rigid motions, and
// without a @p plane (the first evaluation) it is refused.
Result<MeciPoint> evaluate_meci(const MeciCallback &f, std::span<const double> x, bool project,
                                const MeciPoint *const plane) {
  const size_t n = x.size();
  std::vector<double> g_i(n);
  std::vector<double> g_j(n);
  std::vector<double> h(n);
  std::array<double, 2> energies = FBF_TRY(f(x, g_i, g_j, h));
  const auto finite = [](std::span<const double> v) {
    return std::ranges::all_of(v, [](double value) { return std::isfinite(value); });
  };
  if (!finite(energies) || !finite(g_i) || !finite(g_j) || !finite(h)) {
    return err_numerical("geometry::optimize_meci: an energy, gradient or coupling at an evaluated "
                         "point is not finite (energies {} and {})",
                         energies[0], energies[1]);
  }
  if (energies[0] > energies[1]) {
    std::swap(energies[0], energies[1]);
    std::swap(g_i, g_j);
  }

  MeciPoint point{
      energies[0], energies[1], {}, std::vector<double>(n), 0.0, false, std::vector<double>(n), {}};
  if (project) {
    point.basis = rigid_body_basis(x, {});
  }
  for (size_t i = 0; i < n; ++i) {
    point.gap_gradient[i] = g_j[i] - g_i[i];
    point.mean_gradient[i] = 0.5 * (g_i[i] + g_j[i]);
  }
  const auto length = [](std::span<const double> v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
  };
  point.carried = std::ranges::equal(g_i, g_j) &&
                  std::ranges::all_of(h, [](double value) { return value == 0.0; });
  if (point.carried) {
    if (plane == nullptr) {
      return err_numerical("geometry::optimize_meci: the first evaluation gave a mean gradient "
                           "alone (g_I = g_J, h = 0), and there is no earlier branching plane to "
                           "carry");
    }
    // x^ then y^, after the plane's rigid motions: each loses this
    // geometry's, and y^ x^ again, then is renormalized.
    const size_t rigid = plane->basis.size() - (plane->coupled ? 2 : 1);
    const size_t x_at = point.basis.size();
    point.coupled = plane->coupled;
    point.gap_gradient_norm = plane->gap_gradient_norm;
    for (size_t k = rigid; k < plane->basis.size(); ++k) {
      std::vector<double> v = plane->basis[k];
      project_out(point.basis, v);
      const double norm = length(v);
      for (double &value : v) {
        value /= norm;
      }
      point.basis.push_back(std::move(v));
    }
    for (size_t i = 0; i < n; ++i) {
      point.gap_gradient[i] = point.gap_gradient_norm * point.basis[x_at][i];
    }
    point.projected = point.mean_gradient;
    project_out(point.basis, point.projected);
    return point;
  }
  project_out(point.basis, point.gap_gradient);
  point.gap_gradient_norm = length(point.gap_gradient);
  if (!(point.gap_gradient_norm > 0.0)) {
    return err_numerical("geometry::optimize_meci: the gradient difference g_J - g_I vanishes "
                         "after the rigid-body projection, so the branching plane has no x");
  }

  std::vector<double> x_hat = point.gap_gradient;
  for (double &value : x_hat) {
    value /= point.gap_gradient_norm;
  }
  project_out(point.basis, h);
  const double before = length(h);
  point.basis.push_back(std::move(x_hat));
  project_out({point.basis.back()}, h);
  const double after = length(h);
  // Relative to the larger of |y| and |x|, so a coupling that is rounding
  // noise beside x (a symmetry-zero h) is dropped rather than turned into a
  // random direction of the plane.
  point.coupled = after > 1e-8 * std::max(before, point.gap_gradient_norm);
  if (point.coupled) {
    for (double &value : h) {
      value /= after;
    }
    point.basis.push_back(std::move(h));
  }

  point.projected = point.mean_gradient;
  project_out(point.basis, point.projected);
  return point;
}

Result<MECISolution>
optimize_meci(std::span<const double> x0,
              const std::function<Result<std::array<double, 2>>(
                  std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                  std::span<double> h_ij)> &energies_and_gradients,
              const Options &options,
              const std::function<Status(const MECISolution::Iteration &)> &on_iteration) {
  const size_t n = x0.size();
  if (n == 0 || !std::ranges::all_of(x0, [](double value) { return std::isfinite(value); })) {
    return err_config("geometry::optimize_meci: x0 must be non-empty and finite, got {} "
                      "coordinates",
                      n);
  }
  if (options.project_rigid_body && n % 3 != 0) {
    return err_config("geometry::optimize_meci: project_rigid_body needs points in space, 3 "
                      "coordinates each, and {} is not divisible by 3",
                      n);
  }
  // Negated comparisons, so a NaN option is refused too.
  if (!(options.max_force > 0.0) || !(options.rms_force > 0.0) ||
      !(options.max_displacement > 0.0) || !(options.rms_displacement > 0.0) ||
      !(options.max_gap > 0.0) || !(options.initial_hessian > 0.0)) {
    return err_config("geometry::optimize_meci: the five thresholds and initial_hessian must be "
                      "above 0");
  }
  if (!(options.trust_radius > 0.0) || !(options.trust_radius <= options.max_trust_radius) ||
      options.max_steps < 0) {
    return err_config("geometry::optimize_meci: needs 0 < trust_radius <= max_trust_radius and "
                      "max_steps >= 0, got {}, {} and {}",
                      options.trust_radius, options.max_trust_radius, options.max_steps);
  }
  if (!options.constraints.empty()) {
    return err_config("geometry::optimize_meci: distance constraints are optimize's alone, and "
                      "{} are set",
                      options.constraints.size());
  }

  MECISolution solution{std::vector<double>(x0.begin(), x0.end()), 0.0, 0.0, {}};
  double trust_radius = options.trust_radius;
  // Logs the evaluation, hands it to on_iteration, and says whether it has
  // converged.
  const auto record = [&](const MeciPoint &point, std::span<const double> s) -> Result<bool> {
    const auto [g_max, g_rms] = max_and_rms(point.projected);
    const auto [s_max, s_rms] = max_and_rms(s);
    const double gap = point.upper - point.lower;
    solution.lower = point.lower;
    solution.upper = point.upper;
    solution.iterations.push_back(MECISolution::Iteration{point.lower, point.upper, gap, g_max,
                                                          g_rms, s_max, s_rms, trust_radius,
                                                          point.coupled ? 2 : 1, point.carried});
    if (on_iteration) {
      FBF_TRY(on_iteration(solution.iterations.back()));
    }
    return gap <= options.max_gap && g_max <= options.max_force && g_rms <= options.rms_force &&
           s_max <= options.max_displacement && s_rms <= options.rms_displacement;
  };

  std::vector<double> &x = solution.x;
  MeciPoint point =
      FBF_TRY(evaluate_meci(energies_and_gradients, x, options.project_rigid_body, nullptr));
  if (FBF_TRY(record(point, std::vector<double>(n, 0.0)))) {
    return solution;
  }

  std::vector<double> hessian(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    hessian[i * n + i] = options.initial_hessian;
  }
  std::vector<double> projector(n * n);
  std::vector<double> half(n * n);
  std::vector<double> model(n * n);
  std::vector<double> x_trial(n);
  std::vector<double> hs(n);
  std::vector<double> y(n);
  const auto dot = [](std::span<const double> a, std::span<const double> b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  };

  for (int32_t step = 0; step < options.max_steps; ++step) {
    // P = I - Q Q^T over the basis Q, and the model P B P + Q Q^T: B on the
    // intersection space, the identity on the rest, so the Cholesky solve
    // against -P g gives -(P B P)^+ P g while B stays positive definite.
    std::ranges::fill(projector, 0.0);
    for (size_t i = 0; i < n; ++i) {
      projector[i * n + i] = 1.0;
    }
    for (const std::vector<double> &q : point.basis) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          projector[i * n + j] -= q[i] * q[j];
        }
      }
    }
    std::ranges::fill(half, 0.0);
    std::ranges::fill(model, 0.0);
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
    std::vector<double> p = FBF_TRY(newton_step("geometry::optimize_meci", model, point.projected));
    project_out(point.basis, p);

    // The Newton step on the gap, along x^ (the basis entry after the rigid
    // motions).
    const double gap = point.upper - point.lower;
    const std::vector<double> &x_hat = point.basis[point.basis.size() - (point.coupled ? 2 : 1)];
    for (size_t i = 0; i < n; ++i) {
      p[i] -= gap / point.gap_gradient_norm * x_hat[i];
    }
    const double length = std::sqrt(dot(p, p));
    if (length > trust_radius) {
      for (double &value : p) {
        value *= trust_radius / length;
      }
    }
    const double s_norm = std::min(length, trust_radius);

    // The model's change in the mean energy plus the gap: g.p + p'.B.p' / 2,
    // p' = P p, for the mean, and x.p, the gap's linear model.
    std::vector<double> p_in = p;
    project_out(point.basis, p_in);
    for (size_t i = 0; i < n; ++i) {
      hs[i] = dot(p_in, std::span<const double>(hessian).subspan(i * n, n));
    }
    const double php = dot(p_in, hs);
    const double mean_model = dot(point.mean_gradient, p) + 0.5 * php;

    for (size_t i = 0; i < n; ++i) {
      x_trial[i] = x[i] + p[i];
    }
    MeciPoint trial =
        FBF_TRY(evaluate_meci(energies_and_gradients, x_trial, options.project_rigid_body, &point));

    // The agreement test of Nocedal and Wright's Algorithm 4.1, made two-sided
    // since the modelled change may be a rise (a gap step up a tilted cone),
    // and skipped when the modelled change is under the energies' rounding.
    // A carried plane at either end leaves the gap unmodelled (its x^ is a
    // nearby point's, and around a cone x^ turns with the direction to the
    // seam), so the test is then the mean energy's alone.
    const bool gap_modelled = !point.carried && !trial.carried;
    const double predicted = mean_model + (gap_modelled ? dot(point.gap_gradient, p) : 0.0);
    const double actual = 0.5 * (trial.lower + trial.upper) - 0.5 * (point.lower + point.upper) +
                          (gap_modelled ? (trial.upper - trial.lower) - gap : 0.0);
    const double noise = 1e3 * std::numeric_limits<double>::epsilon() *
                         std::max({std::abs(point.lower), std::abs(point.upper),
                                   std::abs(trial.lower), std::abs(trial.upper)});
    if (std::abs(predicted) > noise) {
      const double ratio = actual / predicted;
      if (ratio < 0.25 || ratio > 1.75) {
        trust_radius = 0.25 * s_norm;
      } else if (ratio > 0.75 && ratio < 1.25 && s_norm >= 0.99 * trust_radius) {
        trust_radius = std::min(2.0 * trust_radius, options.max_trust_radius);
      }
    }

    // BFGS over the old intersection space: s = P p, and y = P (the change
    // in P g). The new point's P carries the plane's turn, which is how the
    // seam's curvature reaches B. Skipped unless y.s is positive past
    // rounding, which keeps B positive definite.
    for (size_t i = 0; i < n; ++i) {
      y[i] = trial.projected[i] - point.projected[i];
    }
    project_out(point.basis, y);
    const double ys = dot(y, p_in);
    const double y_norm = std::sqrt(dot(y, y));
    const double s_in_norm = std::sqrt(dot(p_in, p_in));
    if (ys > std::numeric_limits<double>::epsilon() * y_norm * s_in_norm && php > 0.0) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          hessian[i * n + j] += y[i] * y[j] / ys - hs[i] * hs[j] / php;
        }
      }
    }

    std::swap(x, x_trial);
    point = std::move(trial);
    if (FBF_TRY(record(point, p))) {
      return solution;
    }
  }
  const MECISolution::Iteration &last = solution.iterations.back();
  return err_max_iterations_exceeded("geometry::optimize_meci: not converged after {} steps (gap "
                                     "{}, max force {}, rms force {})",
                                     options.max_steps, last.gap, last.max_force, last.rms_force);
}

} // namespace free_breakfast::geometry
