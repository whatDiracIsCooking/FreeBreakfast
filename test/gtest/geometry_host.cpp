// Host-only cases for geometry::optimize: a quadratic well whose minimum it
// reaches exactly, Rosenbrock, a four-point Lennard-Jones cluster whose
// centroid and orientation the projection must hold, the rejection of an
// uphill step, and the refusals.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.geometry;

namespace {

namespace fb = free_breakfast;
namespace geo = fb::geometry;

// Tight enough that a converged point is the minimum to rounding, not just to
// Gaussian's defaults.
geo::Options tight_options() {
  geo::Options options;
  options.max_force = 1e-10;
  options.rms_force = 1e-10;
  options.max_displacement = 1e-10;
  options.rms_displacement = 1e-10;
  options.project_rigid_body = false;
  return options;
}

// E = sum_i k_i (x_i - m_i)^2 / 2: a well with its minimum at m and curvature
// k_i along coordinate i.
constexpr std::array<double, 6> kWellMinimum{1.0, -2.0, 0.5, 3.0, 1.0, -1.0};

fb::Result<double> well(std::span<const double> curvatures, std::span<const double> x,
                        std::span<double> gradient) {
  double energy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double d = x[i] - kWellMinimum[i];
    energy += 0.5 * curvatures[i] * d * d;
    gradient[i] = curvatures[i] * d;
  }
  return energy;
}

// The extended Rosenbrock function, sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2,
// its minimum 0 at every x_i = 1.
fb::Result<double> rosenbrock(std::span<const double> x, std::span<double> gradient) {
  std::ranges::fill(gradient, 0.0);
  double energy = 0.0;
  for (std::size_t i = 0; i + 1 < x.size(); ++i) {
    const double a = x[i + 1] - x[i] * x[i];
    const double b = 1.0 - x[i];
    energy += 100.0 * a * a + b * b;
    gradient[i] += -400.0 * x[i] * a - 2.0 * b;
    gradient[i + 1] += 200.0 * a;
  }
  return energy;
}

// A Lennard-Jones cluster, epsilon = sigma = 1: E = sum_{i<j} 4 (r^-12 - r^-6).
// Four points' global minimum is the regular tetrahedron of edge 2^(1/6), at
// E = -6.
fb::Result<double> lennard_jones(std::span<const double> x, std::span<double> gradient) {
  std::ranges::fill(gradient, 0.0);
  double energy = 0.0;
  const std::size_t points = x.size() / 3;
  for (std::size_t i = 0; i < points; ++i) {
    for (std::size_t j = i + 1; j < points; ++j) {
      std::array<double, 3> d{};
      double r2 = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        d[k] = x[3 * i + k] - x[3 * j + k];
        r2 += d[k] * d[k];
      }
      const double inv6 = 1.0 / (r2 * r2 * r2);
      energy += 4.0 * (inv6 * inv6 - inv6);
      const double scale = 4.0 * (-12.0 * inv6 * inv6 + 6.0 * inv6) / r2;
      for (std::size_t k = 0; k < 3; ++k) {
        gradient[3 * i + k] += scale * d[k];
        gradient[3 * j + k] -= scale * d[k];
      }
    }
  }
  return energy;
}

std::array<double, 3> centroid(std::span<const double> x) {
  std::array<double, 3> c{};
  const std::size_t points = x.size() / 3;
  for (std::size_t i = 0; i < points; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      c[k] += x[3 * i + k] / static_cast<double>(points);
    }
  }
  return c;
}

TEST(GeometryOptimize, QuadraticWellIsotropicInOneStep) {
  // With the starting Hessian equal to the well's, the first step is the exact
  // Newton step and lands on the minimum; the second evaluation, of the zero
  // step from there, is what has a displacement under the thresholds.
  const std::array<double, 6> curvatures{3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
  geo::Options options = tight_options();
  options.initial_hessian = 3.0;
  options.trust_radius = 100.0;
  options.max_trust_radius = 100.0;
  const std::vector<double> x0(6, 0.0);
  const auto result = geo::optimize(
      x0, [&](std::span<const double> x, std::span<double> g) { return well(curvatures, x, g); },
      options);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_EQ(result->iterations.size(), 3U);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    EXPECT_DOUBLE_EQ(result->x[i], kWellMinimum[i]);
  }
  EXPECT_EQ(result->energy, 0.0);
}

TEST(GeometryOptimize, QuadraticWellAnisotropic) {
  // The curvatures span a factor of 6 and the starting Hessian matches none of
  // them, so the minimum is reached by the BFGS updates; 18 evaluations when
  // this was written.
  const std::array<double, 6> curvatures{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<double> x0(6, 0.0);
  const auto result = geo::optimize(
      x0, [&](std::span<const double> x, std::span<double> g) { return well(curvatures, x, g); },
      tight_options());
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_LE(result->iterations.size(), 25U);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    EXPECT_NEAR(result->x[i], kWellMinimum[i], 1e-10);
  }
  EXPECT_LE(result->energy, 1e-20);
}

TEST(GeometryOptimize, Rosenbrock) {
  // From the classic start along its curved valley; 56 evaluations, 3 of them
  // rejected, when this was written.
  geo::Options options = tight_options();
  options.max_steps = 200;
  const std::vector<double> x0{-1.2, 1.0, 1.0};
  const auto result = geo::optimize(x0, rosenbrock, options);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  for (const double value : result->x) {
    EXPECT_NEAR(value, 1.0, 1e-8);
  }
  EXPECT_LE(result->energy, 1e-16);
}

TEST(GeometryOptimize, LennardJonesClusterHoldsCentroidAndOrientation) {
  // A distorted, non-planar four-point cluster away from the origin. Every
  // evaluated point records where the optimizer went, so each step can be
  // checked against the accepted geometry it was taken from.
  std::vector<double> x0{0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.5, 1.0, 0.2, 0.4, 0.5, 1.1};
  for (std::size_t i = 0; i < x0.size(); ++i) {
    x0[i] += std::array<double, 3>{3.0, -1.0, 2.0}[i % 3];
  }
  std::vector<std::vector<double>> visited;
  geo::Options options = tight_options();
  options.project_rigid_body = true;
  const auto result = geo::optimize(
      x0,
      [&](std::span<const double> x, std::span<double> g) {
        visited.emplace_back(x.begin(), x.end());
        return lennard_jones(x, g);
      },
      options);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  ASSERT_EQ(visited.size(), result->iterations.size());

  EXPECT_NEAR(result->energy, -6.0, 1e-12);
  const double edge = std::pow(2.0, 1.0 / 6.0);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = i + 1; j < 4; ++j) {
      double r2 = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        const double d = result->x[3 * i + k] - result->x[3 * j + k];
        r2 += d * d;
      }
      EXPECT_NEAR(std::sqrt(r2), edge, 1e-9) << "points " << i << ", " << j;
    }
  }

  // The centroid does not move: the translations are out of every step.
  const auto c0 = centroid(x0);
  const auto c = centroid(result->x);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(c[k], c0[k], 1e-12);
  }

  // No step rotates the points: sum_i (r_i - c) x s_i = 0 about the geometry
  // each step left, rejected steps included.
  std::size_t base = 0;
  for (std::size_t e = 1; e < visited.size(); ++e) {
    const std::vector<double> &from = visited[base];
    const auto cf = centroid(from);
    std::array<double, 3> torque{};
    for (std::size_t i = 0; i < 4; ++i) {
      std::array<double, 3> d{};
      std::array<double, 3> s{};
      for (std::size_t k = 0; k < 3; ++k) {
        d[k] = from[3 * i + k] - cf[k];
        s[k] = visited[e][3 * i + k] - from[3 * i + k];
      }
      torque[0] += d[1] * s[2] - d[2] * s[1];
      torque[1] += d[2] * s[0] - d[0] * s[2];
      torque[2] += d[0] * s[1] - d[1] * s[0];
    }
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(torque[k], 0.0, 1e-12) << "evaluation " << e << ", axis " << k;
    }
    if (result->iterations[e].accepted) {
      base = e;
    }
  }
}

TEST(GeometryOptimize, LennardJonesClusterAtDefaults) {
  // Gaussian's thresholds, as the pipeline will run it: converged well short
  // of the cap, near the minimum.
  const std::vector<double> x0{0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.5, 1.0, 0.2, 0.4, 0.5, 1.1};
  const auto result = geo::optimize(x0, lennard_jones);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_LE(result->iterations.size(), 30U);
  EXPECT_NEAR(result->energy, -6.0, 1e-6);
  const geo::Iteration &last = result->iterations.back();
  EXPECT_TRUE(last.accepted);
  EXPECT_LE(last.max_force, 4.5e-4);
  EXPECT_LE(last.rms_force, 3.0e-4);
  EXPECT_LE(last.max_displacement, 1.8e-3);
  EXPECT_LE(last.rms_displacement, 1.2e-3);
}

TEST(GeometryOptimize, UphillStepIsRejectedAndShrinksTheTrustRadius) {
  // E = x^2 from x = 0.3 with a starting Hessian 20 times too soft: the
  // Newton step is -6, cut to the trust radius of 1, and lands at -0.7, above
  // the start. It is rejected, the radius becomes a quarter of it, and the
  // retry is taken from the start again.
  std::vector<std::vector<double>> visited;
  const auto parabola = [&](std::span<const double> x, std::span<double> g) -> fb::Result<double> {
    visited.emplace_back(x.begin(), x.end());
    std::ranges::fill(g, 0.0);
    g[0] = 2.0 * x[0];
    return x[0] * x[0];
  };
  geo::Options options = tight_options();
  options.initial_hessian = 0.1;
  options.trust_radius = 1.0;
  const std::vector<double> x0{0.3, 0.0, 0.0};
  const auto result = geo::optimize(x0, parabola, options);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  ASSERT_GE(result->iterations.size(), 3U);

  const geo::Iteration &start = result->iterations[0];
  const geo::Iteration &uphill = result->iterations[1];
  EXPECT_TRUE(start.accepted);
  EXPECT_DOUBLE_EQ(start.trust_radius, 1.0);
  EXPECT_FALSE(uphill.accepted);
  EXPECT_GT(uphill.energy, start.energy);
  EXPECT_DOUBLE_EQ(uphill.energy, 0.49);
  EXPECT_DOUBLE_EQ(uphill.max_displacement, 1.0);
  EXPECT_DOUBLE_EQ(uphill.trust_radius, 0.25);
  // The retry: from x0, not from the rejected point, under the new radius.
  EXPECT_DOUBLE_EQ(visited[2][0], 0.05);
  EXPECT_NEAR(result->x[0], 0.0, 1e-10);
}

TEST(GeometryOptimize, StartAtMinimumConvergesOnTheForces) {
  std::int32_t calls = 0;
  const auto flat = [&](std::span<const double>, std::span<double> g) -> fb::Result<double> {
    ++calls;
    std::ranges::fill(g, 0.0);
    return -1.0;
  };
  const std::vector<double> x0{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const auto result = geo::optimize(x0, flat);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(result->iterations.size(), 1U);
  EXPECT_EQ(result->energy, -1.0);
}

TEST(GeometryOptimize, Refusals) {
  const auto unused = [](std::span<const double>, std::span<double> g) -> fb::Result<double> {
    std::ranges::fill(g, 0.0);
    return 0.0;
  };
  const std::vector<double> x0{0.0, 0.0, 0.0};
  const auto refused = [&](std::span<const double> x, const geo::Options &options) {
    const auto result = geo::optimize(x, unused, options);
    return !result.has_value() && result.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused({}, {}));
  EXPECT_TRUE(refused(std::vector<double>{0.0, std::nan("")}, tight_options()));
  EXPECT_TRUE(refused(std::vector<double>{0.0, 1.0}, {}));
  geo::Options options;
  options.max_force = 0.0;
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.initial_hessian = -1.0;
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.trust_radius = 2.0;
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.max_steps = -1;
  EXPECT_TRUE(refused(x0, options));
}

TEST(GeometryOptimize, MaxStepsAndCallbackErrors) {
  geo::Options options = tight_options();
  options.max_steps = 2;
  const auto capped = geo::optimize(std::vector<double>{-1.2, 1.0, 1.0}, rosenbrock, options);
  ASSERT_FALSE(capped.has_value());
  EXPECT_EQ(capped.error().kind, fb::ErrorKind::MaxIterationsExceeded);

  std::int32_t calls = 0;
  const auto failing = [&](std::span<const double> x, std::span<double> g) -> fb::Result<double> {
    if (++calls == 2) {
      return fb::err_divergence("scf did not converge");
    }
    return rosenbrock(x, g);
  };
  const auto failed = geo::optimize(std::vector<double>{-1.2, 1.0, 1.0}, failing, options);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().kind, fb::ErrorKind::Divergence);

  const auto not_finite = [](std::span<const double>, std::span<double> g) -> fb::Result<double> {
    std::ranges::fill(g, 0.0);
    return std::numeric_limits<double>::quiet_NaN();
  };
  const auto nan = geo::optimize(std::vector<double>{0.0, 0.0, 0.0}, not_finite, options);
  ASSERT_FALSE(nan.has_value());
  EXPECT_EQ(nan.error().kind, fb::ErrorKind::Numerical);
}

} // namespace
