/// @file
/// @brief irc: the intrinsic reaction coordinate from a transition state by
/// Gonzalez and Schlegel's second-order method, in mass-weighted Cartesians,
/// over a caller's energy and gradient.

module free_breakfast.geometry;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.atom;
import :rigid_body;
import :quasi_newton;
import :masses;

#include "diagnostics.h"

namespace free_breakfast::geometry {

std::string_view irc_stop_name(IrcStop stop) {
  switch (stop) {
  case IrcStop::Minimum:
    return "minimum";
  case IrcStop::MinimumWithinStep:
    return "minimum_within_step";
  case IrcStop::EnergyRose:
    return "energy_rose";
  case IrcStop::MaxPoints:
    return "max_points";
  case IrcStop::SphereNotConverged:
    return "sphere_not_converged";
  }
  return "unknown";
}

// a . b.
double irc_dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// One evaluated geometry of an irc walk.
struct IrcEvaluation {
  // The mass-weighted coordinates, sqrt(m) x.
  std::vector<double> q;
  // The mass-weighted gradient, g / sqrt(m), projected off the rigid motions.
  std::vector<double> g;
  double energy;
  // The orthonormal rigid motions in mass-weighted coordinates at q; empty
  // without the projection.
  std::vector<std::vector<double>> rigid;
  // The 2-norm and the largest |component| of the Cartesian gradient,
  // projected off the (unweighted) rigid motions: what the walk reports and
  // stops on, in optimize's units.
  double gradient_norm;
  double max_force;
};

// An orthonormal basis of the complement of the orthonormal @p rigid in
// R^n, by diagonalizing the projector I - Q Q^T, whose eigenvalues are 0 on
// the rigid span and 1 on the rest, as harmonic_analysis takes its internal
// basis; R^n's unit vectors without the projection.
std::vector<std::vector<double>> irc_internal_basis(const std::vector<std::vector<double>> &rigid,
                                                    size_t n) {
  std::vector<double> projector(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double value = i == j ? 1.0 : 0.0;
      for (const std::vector<double> &u : rigid) {
        value -= u[i] * u[j];
      }
      projector[i * n + j] = value;
    }
  }
  if (rigid.empty()) {
    std::vector<std::vector<double>> basis(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
      basis[i][i] = 1.0;
    }
    return basis;
  }
  const auto [values, vectors] = symmetric_eigen(std::move(projector), n);
  std::vector<std::vector<double>> basis;
  for (size_t j = 0; j < n; ++j) {
    if (values[j] > 0.5) {
      std::vector<double> v(n);
      for (size_t i = 0; i < n; ++i) {
        v[i] = vectors[i * n + j];
      }
      basis.push_back(std::move(v));
    }
  }
  return basis;
}

// The point p' on the sphere |p'| = radius that the quadratic model with the
// n x n row-major Hessian @p hessian predicts from the trial p, where the
// gradient is g: the minimum of g . (p' - p) + 1/2 (p' - p) H (p' - p) on the
// sphere, over the span of @p basis (orthonormal, m vectors of n). In H's
// eigenbasis over it, H v_i = b_i v_i ascending, the stationarity
// g + H (p' - p) = lambda p' gives p'_i = c_i / (b_i - lambda),
// c_i = b_i p_i - g_i, and the minimum is the root of |p'(lambda)| = radius
// below b_0, where |p'| rises monotonically from 0. Bisection finds it to
// the last bit. When even lambda at b_0 leaves |p'| short (the hard case, c
// nothing along v_0), the shortfall is made up along v_0, on the side the
// trial already leaned, and @p hard, when given, is set.
std::vector<double> irc_sphere_step(std::span<const double> hessian,
                                    const std::vector<std::vector<double>> &basis,
                                    std::span<const double> p, std::span<const double> g,
                                    double radius, bool *hard = nullptr) {
  const size_t n = p.size();
  const size_t m = basis.size();
  // H_int = B^T H B, m x m.
  std::vector<std::vector<double>> hb(m, std::vector<double>(n, 0.0));
  for (size_t j = 0; j < m; ++j) {
    for (size_t r = 0; r < n; ++r) {
      hb[j][r] = irc_dot(hessian.subspan(r * n, n), basis[j]);
    }
  }
  std::vector<double> h_int(m * m);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      h_int[i * m + j] = 0.5 * (irc_dot(basis[i], hb[j]) + irc_dot(basis[j], hb[i]));
    }
  }
  const auto [b, v] = symmetric_eigen(std::move(h_int), m);
  // p and g in the eigenbasis, and c.
  std::vector<double> p_int(m);
  std::vector<double> g_int(m);
  for (size_t j = 0; j < m; ++j) {
    p_int[j] = irc_dot(basis[j], p);
    g_int[j] = irc_dot(basis[j], g);
  }
  std::vector<double> pe(m, 0.0);
  std::vector<double> c(m, 0.0);
  for (size_t i = 0; i < m; ++i) {
    double pi = 0.0;
    double gi = 0.0;
    for (size_t j = 0; j < m; ++j) {
      pi += v[j * m + i] * p_int[j];
      gi += v[j * m + i] * g_int[j];
    }
    pe[i] = pi;
    c[i] = b[i] * pi - gi;
  }
  const double r2 = radius * radius;
  const auto length2 = [&](double lambda) {
    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
      const double t = c[i] / (b[i] - lambda);
      sum += t * t;
    }
    return sum;
  };
  // At lo every |c_i / (b_i - lo)| is at most |c_i| radius / |c|, so the
  // length is at most radius: the root is in [lo, b_0).
  const double c_norm = std::sqrt(irc_dot(c, c));
  double lo = b[0] - c_norm / radius - 1.0;
  double hi = b[0];
  for (int32_t iteration = 0; iteration < 400; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) {
      break;
    }
    (length2(mid) > r2 ? hi : lo) = mid;
  }
  std::vector<double> step(m);
  for (size_t i = 0; i < m; ++i) {
    step[i] = c[i] / (b[i] - lo);
  }
  const double short_by = r2 - irc_dot(step, step);
  if (hard != nullptr) {
    *hard = short_by > 1e-10 * r2;
  }
  if (short_by > 1e-10 * r2) {
    const double lean = step[0] != 0.0 ? step[0] : pe[0];
    step[0] += (lean < 0.0 ? -1.0 : 1.0) * std::sqrt(short_by);
  }
  // Back to the full space: p' = B V step.
  std::vector<double> out(n, 0.0);
  for (size_t j = 0; j < m; ++j) {
    double coefficient = 0.0;
    for (size_t i = 0; i < m; ++i) {
      coefficient += v[j * m + i] * step[i];
    }
    for (size_t r = 0; r < n; ++r) {
      out[r] += coefficient * basis[j][r];
    }
  }
  return out;
}

// The first sphere's step, under #1285's hold, from a trial p ahead of the
// pivot whose model step would land behind it: along the sphere's great
// circle down the tangential gradient g_T = g - (g . p) p / radius^2, by the
// Newton step |g_T| / kappa on E's curvature along that circle, kappa =
// u H u - (g . p) / radius^2 (u = -g_T / |g_T|, the circle's own curvature
// folded in), at most a quarter turn, and halved until the point is ahead of
// the pivot (p . down >= 0). The trial itself when g_T is zero.
std::vector<double> irc_sphere_descent(std::span<const double> hessian, std::span<const double> p,
                                       std::span<const double> g, std::span<const double> down,
                                       double radius) {
  const size_t n = p.size();
  const double along = irc_dot(g, p) / (radius * radius);
  std::vector<double> u(n);
  for (size_t i = 0; i < n; ++i) {
    u[i] = (along * p[i]) - g[i];
  }
  const double u_norm = std::sqrt(irc_dot(u, u));
  std::vector<double> out(p.begin(), p.end());
  if (!(u_norm > 0.0)) {
    return out;
  }
  for (double &value : u) {
    value /= u_norm;
  }
  double curvature = -along;
  for (size_t i = 0; i < n; ++i) {
    curvature += u[i] * irc_dot(hessian.subspan(i * n, n), u);
  }
  const double quarter = 0.5 * std::numbers::pi;
  double angle = curvature > 0.0 ? std::min(u_norm / (curvature * radius), quarter) : quarter;
  for (int32_t halving = 0; halving < 60; ++halving, angle *= 0.5) {
    for (size_t i = 0; i < n; ++i) {
      out[i] = (std::cos(angle) * p[i]) + (radius * std::sin(angle) * u[i]);
    }
    if (irc_dot(out, down) >= 0.0) {
      break;
    }
  }
  return out;
}

// Bofill's update of the n x n row-major @p hessian over the step s and the
// change in gradient y: with xi = y - H s, phi = (xi . s)^2 / (|xi|^2 |s|^2)
// of the SR1 update H + xi xi^T / (xi . s) plus 1 - phi of Powell's
// symmetric Broyden H + (xi s^T + s xi^T) / |s|^2 - (xi . s) s s^T / |s|^4.
// phi / (xi . s) is written (xi . s) / (|xi|^2 |s|^2), so a vanishing SR1
// denominator weighs its term to zero rather than dividing by it. Skipped
// for a step or a model error at rounding.
void irc_bofill_update(std::span<double> hessian, std::span<const double> s,
                       std::span<const double> y) {
  const size_t n = s.size();
  const double ss = irc_dot(s, s);
  std::vector<double> xi(y.begin(), y.end());
  for (size_t i = 0; i < n; ++i) {
    xi[i] -= irc_dot(hessian.subspan(i * n, n), s);
  }
  const double xx = irc_dot(xi, xi);
  const double xs = irc_dot(xi, s);
  if (!(ss > 1e-24) || !(xx > 1e-30 * (1.0 + irc_dot(y, y)))) {
    return;
  }
  const double phi = (xs * xs) / (xx * ss);
  const double sr1 = xs / (xx * ss);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const double psb = ((xi[i] * s[j]) + (s[i] * xi[j])) / ss - (xs * s[i] * s[j]) / (ss * ss);
      hessian[i * n + j] += (sr1 * xi[i] * xi[j]) + ((1.0 - phi) * psb);
    }
  }
}

Result<IrcPath>
irc(std::span<const Atom> start, std::span<const double> hessian, std::span<const double> direction,
    const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
        &energy_and_gradient,
    const IrcOptions &options, const std::function<Status(const IrcPoint &)> &on_point) {
  const size_t points = start.size();
  const size_t n = 3 * points;
  if (points == 0 || hessian.size() != n * n || direction.size() != n) {
    return err_config("geometry::irc: needs at least one atom, a 3N x 3N Hessian and a 3N "
                      "direction, got {} atoms, {} Hessian elements and {} direction elements",
                      points, hessian.size(), direction.size());
  }
  if (!(options.step_size > 0.0) || !std::isfinite(options.step_size) ||
      !(options.max_force > 0.0) || !(options.sphere_gradient > 0.0) ||
      !(options.sphere_displacement > 0.0)) {
    return err_config("geometry::irc: step_size, max_force, sphere_gradient and "
                      "sphere_displacement must be finite and positive");
  }
  if (options.max_points < 1 || options.max_sphere_iterations < 1) {
    return err_config("geometry::irc: max_points and max_sphere_iterations must be at least 1, "
                      "got {} and {}",
                      options.max_points, options.max_sphere_iterations);
  }
  const auto finite = [](std::span<const double> v) {
    return std::ranges::all_of(v, [](double value) { return std::isfinite(value); });
  };
  if (!finite(hessian) || !finite(direction)) {
    return err_config("geometry::irc: the Hessian or the direction is not finite");
  }
  // sqrt(m) per atom, and per coordinate.
  std::vector<double> sqrt_mass(points);
  std::vector<double> scale(n);
  std::vector<double> x0(n);
  for (size_t a = 0; a < points; ++a) {
    const auto z = static_cast<size_t>(std::to_underlying(start[a].element));
    if (z < 1 || z > kAverageMasses.size()) {
      return err_config("geometry::irc: atom {} has Z = {}, not 1 to {}", a, z,
                        kAverageMasses.size());
    }
    sqrt_mass[a] = std::sqrt(kAverageMasses[z - 1]);
    x0[3 * a] = start[a].position.x;
    x0[3 * a + 1] = start[a].position.y;
    x0[3 * a + 2] = start[a].position.z;
    for (size_t k = 0; k < 3; ++k) {
      scale[3 * a + k] = sqrt_mass[a];
    }
  }
  if (!finite(x0)) {
    return err_config("geometry::irc: an atom's position is not finite");
  }
  const bool project = options.project_rigid_body;
  const auto rigid_at = [&](std::span<const double> x) {
    return project ? rigid_body_basis(x, sqrt_mass) : std::vector<std::vector<double>>{};
  };

  // The way out: the direction mass-weighted, off the rigid motions, unit.
  std::vector<double> leave(n);
  for (size_t i = 0; i < n; ++i) {
    leave[i] = direction[i] * scale[i];
  }
  project_out(rigid_at(x0), leave);
  const double leave_norm = std::sqrt(irc_dot(leave, leave));
  if (!(leave_norm > 1e-10 * std::max(1.0, *std::ranges::max_element(scale)))) {
    return err_config("geometry::irc: the direction is a rigid-body motion, or zero");
  }
  for (double &value : leave) {
    value /= leave_norm;
  }

  // The model Hessian, mass-weighted and symmetrized.
  std::vector<double> model(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      model[i * n + j] = 0.5 * (hessian[i * n + j] + hessian[j * n + i]) / (scale[i] * scale[j]);
    }
  }

  std::vector<double> gradient(n);
  const auto evaluate = [&](std::span<const double> x) -> Result<IrcEvaluation> {
    const double energy = FBF_TRY(energy_and_gradient(x, gradient));
    if (!std::isfinite(energy) || !finite(gradient)) {
      return err_numerical("geometry::irc: the energy or gradient is not finite");
    }
    IrcEvaluation out{.q = std::vector<double>(n),
                      .g = std::vector<double>(n),
                      .energy = energy,
                      .rigid = rigid_at(x),
                      .gradient_norm = 0.0,
                      .max_force = 0.0};
    for (size_t i = 0; i < n; ++i) {
      out.q[i] = x[i] * scale[i];
      out.g[i] = gradient[i] / scale[i];
    }
    project_out(out.rigid, out.g);
    std::vector<double> cartesian(gradient.begin(), gradient.end());
    if (project) {
      project_out(rigid_body_basis(x, {}), cartesian);
    }
    out.gradient_norm = std::sqrt(irc_dot(cartesian, cartesian));
    out.max_force = max_and_rms(cartesian).first;
    return out;
  };
  IrcPath path{.points = {}, .stop = IrcStop::MaxPoints};
  const auto keep = [&](const IrcEvaluation &at, int32_t k, int32_t iterations,
                        double tangential) -> Status {
    IrcPoint point{.arc_length = static_cast<double>(k) * options.step_size,
                   .energy = at.energy,
                   .x = std::vector<double>(n),
                   .gradient_norm = at.gradient_norm,
                   .max_force = at.max_force,
                   .sphere_iterations = iterations,
                   .tangential_gradient = tangential};
    for (size_t i = 0; i < n; ++i) {
      point.x[i] = at.q[i] / scale[i];
    }
    path.points.push_back(std::move(point));
    if (on_point) {
      FBF_TRY(on_point(path.points.back()));
    }
    return {};
  };

  IrcEvaluation point = FBF_TRY(evaluate(x0));
  FBF_TRY(keep(point, 0, 0, 0.0));
  const double radius = 0.5 * options.step_size;
  // A sphere searched in at most this many evaluations, the model's step and
  // two corrections, leaves the model's step the next sphere's first trial.
  constexpr int32_t kModelSphereEvaluations = 3;
  // The previous point's mass-weighted |g|.
  double previous_norm = 0.0;
  // The last point's displacement on its sphere from that sphere's far pole,
  // p_k - (s / 2) u: how far the path turned off the straight line over the
  // last step.
  std::vector<double> turn;
  for (int32_t k = 0; k < options.max_points; ++k) {
    // The pivot, half a step down: along the way out at the TS, along the
    // steepest descent after.
    const double g_norm = std::sqrt(irc_dot(point.g, point.g));
    std::vector<double> down = leave;
    if (k > 0) {
      for (size_t i = 0; i < n; ++i) {
        down[i] = -point.g[i] / g_norm;
      }
      // The path's minimum within a step, read off the walk: along the path
      // dE/dsigma = -|g|, so |g| falling linearly from the last point to this
      // one reaches 0 within one more step s once it has halved. The next
      // sphere would then hold the minimum, and its own minimum has no
      // continuation of the path to find (it flips between the sphere's two
      // sides). Not the model's line minimum |g| / (u H u) (#1278): a point
      // off the path by its sphere's tolerance has a gradient across a stiff
      // valley, and that puts the line's minimum within a step while the
      // path still runs along the valley; |g| moves only at second order.
      if (2.0 * g_norm <= previous_norm) {
        path.stop = IrcStop::MinimumWithinStep;
        return path;
      }
    }
    previous_norm = g_norm;
    std::vector<double> pivot(n);
    for (size_t i = 0; i < n; ++i) {
      pivot[i] = point.q[i] + (radius * down[i]);
    }
    // The hypersphere. Point k is on it, so the first trial is the model's
    // step from there. On a smooth valley the Bofill model has learnt the
    // surface over the walk, and that step is the sphere's minimum to its
    // tolerance in one evaluation (#1295). On a stiff, sharply bent valley
    // the model is not good over the sphere's diameter (#1288): its soft
    // curvature is a small difference of stiff elements that the turning
    // floor changes by more than it over a step, its sphere minimum lands in
    // the hard case or behind the pivot, and a search from there wanders.
    // Then, and after a sphere that took more than kModelSphereEvaluations,
    // the first trial is the far pole turned as the last step turned,
    // (s / 2) u + turn put back on the sphere, which asks nothing of the
    // model. It is not the first trial throughout: it carries point k's
    // error on its sphere into the next start, twice over (in turn, and in
    // u, tilted by that error's gradient across the floor, which grows
    // against |g| as |g| falls along a soft valley), where the model's step
    // starts from point k's own gradient. With hold_ahead, a trial ahead of
    // the pivot is not followed by one behind it: the model's step is
    // replaced by irc_sphere_descent's.
    std::vector<double> first_p;
    if (k > 0) {
      std::vector<double> p_k(n);
      for (size_t i = 0; i < n; ++i) {
        p_k[i] = -radius * down[i];
      }
      bool hard = false;
      first_p =
          irc_sphere_step(model, irc_internal_basis(point.rigid, n), p_k, point.g, radius, &hard);
      if (hard || irc_dot(first_p, down) < 0.0 ||
          path.points.back().sphere_iterations > kModelSphereEvaluations) {
        for (size_t i = 0; i < n; ++i) {
          first_p[i] = (radius * down[i]) + turn[i];
        }
        const double first_norm = std::sqrt(irc_dot(first_p, first_p));
        for (size_t i = 0; i < n; ++i) {
          first_p[i] = first_norm > 0.0 ? radius * first_p[i] / first_norm : radius * down[i];
        }
      }
    }
    IrcEvaluation trial = point;
    int32_t evaluations = 0;
    double tangential = 0.0;
    const auto search = [&](bool hold_ahead) -> Result<bool> {
      trial = point;
      int32_t taken = 0;
      for (;;) {
        std::vector<double> p(n);
        for (size_t i = 0; i < n; ++i) {
          p[i] = trial.q[i] - pivot[i];
        }
        std::vector<double> next_p =
            taken == 0 && !first_p.empty()
                ? first_p
                : irc_sphere_step(model, irc_internal_basis(trial.rigid, n), p, trial.g, radius);
        double correction = 0.0;
        for (size_t i = 0; i < n; ++i) {
          correction += (next_p[i] - p[i]) * (next_p[i] - p[i]);
        }
        correction = std::sqrt(correction);
        if (taken > 0) {
          // The gradient's part tangent to the sphere at the trial.
          const double along = irc_dot(trial.g, p) / irc_dot(p, p);
          double t2 = 0.0;
          for (size_t i = 0; i < n; ++i) {
            const double t = trial.g[i] - along * p[i];
            t2 += t * t;
          }
          tangential = std::sqrt(t2);
          if (tangential <= options.sphere_gradient && correction <= options.sphere_displacement) {
            return true;
          }
          if (taken == options.max_sphere_iterations) {
            return false;
          }
        }
        if (hold_ahead && irc_dot(p, down) > 0.0 && irc_dot(next_p, down) < 0.0) {
          next_p = irc_sphere_descent(model, p, trial.g, down, radius);
        }
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
          x[i] = (pivot[i] + next_p[i]) / scale[i];
        }
        IrcEvaluation next = FBF_TRY(evaluate(x));
        ++taken;
        ++evaluations;
        std::vector<double> s(n);
        std::vector<double> y(n);
        for (size_t i = 0; i < n; ++i) {
          s[i] = next.q[i] - trial.q[i];
          y[i] = next.g[i] - trial.g[i];
        }
        project_out(next.rigid, s);
        project_out(next.rigid, y);
        irc_bofill_update(model, s, y);
        trial = std::move(next);
      }
    };
    const auto behind_pivot = [&] {
      double ahead = 0.0;
      for (size_t i = 0; i < n; ++i) {
        ahead += (trial.q[i] - pivot[i]) * down[i];
      }
      return ahead < 0.0;
    };
    // The first sphere passes through the TS, whose gradient is zero, so the
    // TS is a stationary point of E on it, and a minimum: E's curvature on
    // the sphere there is the Hessian's over the tangent space, the positive
    // modes. (Point k on a later sphere is stationary too, but its curvature
    // there is lowered by |g_k| / (s / 2), the Lagrange multiplier.) On a
    // stiff, sharply curved valley the model's step from the mode-displaced
    // point can jump the sphere's diameter back to the TS, and the search
    // converges there (#1285). That search is run again from the start's
    // model, holding its trials ahead of the pivot once one is. Each search
    // has max_sphere_iterations evaluations; the point counts both.
    std::vector<double> start_model;
    if (k == 0) {
      start_model = model;
    }
    bool converged = FBF_TRY(search(false));
    if (k == 0 && converged && behind_pivot()) {
      model = std::move(start_model);
      converged = FBF_TRY(search(true));
    }
    if (!converged) {
      path.stop = IrcStop::SphereNotConverged;
      return path;
    }
    // A rise, or a point behind the pivot (the step turned back by more
    // than a right angle): past the path's minimum either way.
    const bool behind = behind_pivot();
    if (trial.energy > point.energy || behind) {
      path.stop = IrcStop::EnergyRose;
      return path;
    }
    turn.resize(n);
    for (size_t i = 0; i < n; ++i) {
      turn[i] = trial.q[i] - pivot[i] - (radius * down[i]);
    }
    point = std::move(trial);
    FBF_TRY(keep(point, k + 1, evaluations, tangential));
    // A minimum: the force at or below max_force, and |g| halved over the
    // step, so the path's minimum within one more (the test each step opens
    // with, MinimumWithinStep). The force alone is not (#1294): in a soft
    // valley it
    // falls under max_force with the minimum still many steps on (the Cope
    // PC-NEVPT2 IRC's, 26), and the walk goes on.
    if (point.max_force <= options.max_force &&
        2.0 * std::sqrt(irc_dot(point.g, point.g)) <= previous_norm) {
      path.stop = IrcStop::Minimum;
      return path;
    }
  }
  path.stop = IrcStop::MaxPoints;
  return path;
}

} // namespace free_breakfast::geometry
