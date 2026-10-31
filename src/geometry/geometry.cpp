/// @file
/// @brief optimize: BFGS under a trust radius over a caller's energy and
/// gradient, with the rigid-body motions projected out.

module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import :rigid_body;

#include "diagnostics.h"

namespace free_breakfast::geometry {

// v less its components along the orthonormal basis, in place.
void project_out(const std::vector<std::vector<double>> &basis, std::span<double> v) {
  for (const std::vector<double> &u : basis) {
    const double overlap = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    for (size_t j = 0; j < v.size(); ++j) {
      v[j] -= overlap * u[j];
    }
  }
}

// The largest |component| and the root mean square of v.
std::pair<double, double> max_and_rms(std::span<const double> v) {
  double largest = 0.0;
  double squares = 0.0;
  for (const double value : v) {
    largest = std::max(largest, std::abs(value));
    squares += value * value;
  }
  return {largest, std::sqrt(squares / static_cast<double>(v.size()))};
}

// p = -B^-1 g, by a Cholesky factorization of the n x n symmetric B
// (row-major). B is the BFGS Hessian, kept positive definite by updating it
// only over a positive curvature, so a pivot at or below zero means rounding
// has broken that and is reported rather than stepped over.
Result<std::vector<double>> newton_step(std::span<const double> hessian,
                                        std::span<const double> g) {
  const size_t n = g.size();
  std::vector<double> l(hessian.begin(), hessian.end());
  for (size_t j = 0; j < n; ++j) {
    double pivot = l[j * n + j];
    for (size_t k = 0; k < j; ++k) {
      pivot -= l[j * n + k] * l[j * n + k];
    }
    if (!(pivot > 0.0)) {
      return err_numerical("geometry::optimize: the BFGS Hessian lost positive definiteness at "
                           "pivot {} of {}",
                           j, n);
    }
    l[j * n + j] = std::sqrt(pivot);
    for (size_t i = j + 1; i < n; ++i) {
      double value = l[i * n + j];
      for (size_t k = 0; k < j; ++k) {
        value -= l[i * n + k] * l[j * n + k];
      }
      l[i * n + j] = value / l[j * n + j];
    }
  }
  // L z = -g, then L^T p = z.
  std::vector<double> p(n);
  for (size_t i = 0; i < n; ++i) {
    double value = -g[i];
    for (size_t k = 0; k < i; ++k) {
      value -= l[i * n + k] * p[k];
    }
    p[i] = value / l[i * n + i];
  }
  for (size_t i = n; i-- > 0;) {
    double value = p[i];
    for (size_t k = i + 1; k < n; ++k) {
      value -= l[k * n + i] * p[k];
    }
    p[i] = value / l[i * n + i];
  }
  return p;
}

// The energy and the projected gradient at x, into gradient; the rigid-body
// basis at x into basis.
Result<double>
evaluate(const std::function<Result<double>(std::span<const double>, std::span<double>)> &f,
         std::span<const double> x, bool project, std::vector<std::vector<double>> &basis,
         std::span<double> gradient) {
  const double energy = FBF_TRY(f(x, gradient));
  if (!std::isfinite(energy) ||
      !std::ranges::all_of(gradient, [](double value) { return std::isfinite(value); })) {
    return err_numerical("geometry::optimize: the energy or gradient at an evaluated point is "
                         "not finite (energy {})",
                         energy);
  }
  if (project) {
    basis = rigid_body_basis(x, {});
    project_out(basis, gradient);
  }
  return energy;
}

Result<Solution>
optimize(std::span<const double> x0,
         const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
             &energy_and_gradient,
         const Options &options) {
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

  const auto converged = [&options](std::span<const double> g, std::span<const double> s) {
    const auto [g_max, g_rms] = max_and_rms(g);
    const auto [s_max, s_rms] = max_and_rms(s);
    return g_max <= options.max_force && g_rms <= options.rms_force &&
           s_max <= options.max_displacement && s_rms <= options.rms_displacement;
  };
  Solution solution{std::vector<double>(x0.begin(), x0.end()), 0.0, {}};
  const auto record = [&solution](double energy, std::span<const double> g,
                                  std::span<const double> s, double trust_radius, bool accepted) {
    const auto [g_max, g_rms] = max_and_rms(g);
    const auto [s_max, s_rms] = max_and_rms(s);
    solution.iterations.push_back(
        Iteration{energy, g_max, g_rms, s_max, s_rms, trust_radius, accepted});
  };

  std::vector<double> &x = solution.x;
  std::vector<double> g(n);
  std::vector<std::vector<double>> basis;
  solution.energy = FBF_TRY(evaluate(energy_and_gradient, x, options.project_rigid_body, basis, g));
  double trust_radius = options.trust_radius;
  const std::vector<double> no_step(n, 0.0);
  record(solution.energy, g, no_step, trust_radius, true);
  if (converged(g, no_step)) {
    return solution;
  }

  std::vector<double> hessian(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    hessian[i * n + i] = options.initial_hessian;
  }
  std::vector<double> x_trial(n);
  std::vector<double> g_trial(n);
  std::vector<std::vector<double>> basis_trial;
  std::vector<double> hs(n);
  std::vector<double> y(n);

  for (int32_t step = 0; step < options.max_steps; ++step) {
    std::vector<double> p = FBF_TRY(newton_step(hessian, g));
    if (options.project_rigid_body) {
      project_out(basis, p);
    }
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
    const double php = std::inner_product(p.begin(), p.end(), hs.begin(), 0.0);
    const double predicted = std::inner_product(g.begin(), g.end(), p.begin(), 0.0) + 0.5 * php;

    for (size_t i = 0; i < n; ++i) {
      x_trial[i] = x[i] + p[i];
    }
    const double energy = FBF_TRY(
        evaluate(energy_and_gradient, x_trial, options.project_rigid_body, basis_trial, g_trial));

    if (energy > solution.energy) {
      trust_radius = 0.25 * s_norm;
      record(energy, g_trial, p, trust_radius, false);
      continue;
    }

    // The update's agreement bounds: Nocedal and Wright's Algorithm 4.1.
    const double ratio = predicted < 0.0 ? (energy - solution.energy) / predicted : 0.0;
    if (ratio < 0.25) {
      trust_radius = 0.25 * s_norm;
    } else if (ratio > 0.75 && s_norm >= 0.99 * trust_radius) {
      trust_radius = std::min(2.0 * trust_radius, options.max_trust_radius);
    }

    // BFGS: B + y y^T / (y.s) - (B s)(B s)^T / (s.B.s), s = p. Skipped unless
    // the curvature y.s is positive past rounding, which is what keeps B
    // positive definite.
    for (size_t i = 0; i < n; ++i) {
      y[i] = g_trial[i] - g[i];
    }
    const double ys = std::inner_product(y.begin(), y.end(), p.begin(), 0.0);
    const double y_norm = std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0));
    if (ys > std::numeric_limits<double>::epsilon() * y_norm * s_norm && php > 0.0) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          hessian[i * n + j] += y[i] * y[j] / ys - hs[i] * hs[j] / php;
        }
      }
    }

    std::swap(x, x_trial);
    std::swap(g, g_trial);
    std::swap(basis, basis_trial);
    solution.energy = energy;
    record(energy, g, p, trust_radius, true);
    if (converged(g, p)) {
      return solution;
    }
  }
  return err_max_iterations_exceeded(
      "geometry::optimize: not converged after {} steps (max force {}, rms force {})",
      options.max_steps, solution.iterations.back().max_force,
      solution.iterations.back().rms_force);
}

} // namespace free_breakfast::geometry
