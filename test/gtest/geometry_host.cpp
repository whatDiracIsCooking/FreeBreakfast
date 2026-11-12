// Host-only cases for geometry::optimize: a quadratic well whose minimum it
// reaches exactly, Rosenbrock, a four-point Lennard-Jones cluster whose
// centroid and orientation the projection must hold, the rejection of an
// uphill step, and the refusals; distance constraints on that cluster, held at
// every evaluation, with the multiplier the constrained minimum's slope, and
// their refusals. And for geometry::optimize_meci: two-state
// model surfaces over four points' distances, a peaked and a tilted cone and
// an uncoupled crossing, each with a known minimum-energy crossing, the trust
// radius's shrink, the refusals, and a plane carried under a floor where the
// callback gives the mean gradient alone (#902). And for geometry::irc (#1269):
// a curved valley whose every point is held to GS2's hypersphere condition
// and whose two sides reach the two minima, stiff and bent valleys, a long
// soft valley modelled on the Cope IRC's (#1294, #1295), a quadratic saddle
// walked in one evaluation per point along its straight path, two points
// whose centre of mass and axis the projection holds, the stops and the
// refusals.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.geometry;
import free_breakfast.atom;

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

// --- Distance constraints (#1260) -------------------------------------------

double separation(std::span<const double> x, std::size_t a, std::size_t b) {
  double r2 = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    const double d = x[3 * a + k] - x[3 * b + k];
    r2 += d * d;
  }
  return std::sqrt(r2);
}

// The four-point cluster LennardJonesClusterHoldsCentroidAndOrientation starts
// from, its 0-1 distance 1.204.
const std::vector<double> kCluster{0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.5, 1.0, 0.2, 0.4, 0.5, 1.1};

// tight_options' thresholds, with the rigid motions projected. Near them a
// step's energy change, f^2 / 2k, is under the energy's rounding, which is
// what optimize's rounding floor under constraints is for.
constexpr double kConstrainedForce = 1e-10;

geo::Options constrained_options(std::vector<geo::DistanceConstraint> constraints) {
  geo::Options options;
  options.max_force = kConstrainedForce;
  options.rms_force = kConstrainedForce;
  options.max_displacement = 1e-8;
  options.rms_displacement = 1e-8;
  options.constraints = std::move(constraints);
  return options;
}

// The cluster with points 0 and 1 held at 1.3, off both the start's 1.204 and
// the free minimum's 2^(1/6): every evaluated geometry, the start's
// included, is on the constraint to 1e-12 relative, and the centroid never
// moves (the correction along a bond moves its two ends oppositely). At the
// optimum the Lagrange condition holds on the raw gradient: g - lambda dd/dx
// is under the force threshold, which is the projected gradient and the
// multiplier checked together. And lambda is the constrained minimum's slope
// dE*/dd, by a central difference of two more optimizations at 1.3 +- 1e-5:
// its truncation h^2 E*''' / 6 is 7e-10 (E*''' = 44, measured as the 7.3e-8
// such a difference is off at h = 1e-4), its rounding about an ulp of E over
// 2h, 1e-10, and E*'s own error at these forces f^2 / 2k is far under both,
// so the bar is 2e-9.
TEST(GeometryOptimizeConstrained, LennardJonesHoldsTheDistanceAndItsMultiplierIsTheSlope) {
  constexpr double kHeld = 1.3;
  std::vector<std::vector<double>> visited;
  const auto result = geo::optimize(
      kCluster,
      [&](std::span<const double> x, std::span<double> g) {
        visited.emplace_back(x.begin(), x.end());
        return lennard_jones(x, g);
      },
      constrained_options({{.atoms = {0, 1}, .distance = kHeld}}));
  ASSERT_TRUE(result.has_value()) << result.error().message;
  ASSERT_EQ(visited.size(), result->iterations.size());
  for (std::size_t e = 0; e < visited.size(); ++e) {
    EXPECT_NEAR(separation(visited[e], 0, 1), kHeld, 1e-12 * kHeld) << "evaluation " << e;
    ASSERT_EQ(result->iterations[e].distances.size(), 1U);
    ASSERT_EQ(result->iterations[e].multipliers.size(), 1U);
    EXPECT_NEAR(result->iterations[e].distances[0], separation(visited[e], 0, 1), 1e-14);
  }
  const auto c0 = centroid(kCluster);
  const auto c = centroid(result->x);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(c[k], c0[k], 1e-12);
  }

  const double lambda = result->iterations.back().multipliers[0];
  std::vector<double> g(result->x.size());
  ASSERT_TRUE(lennard_jones(result->x, g).has_value());
  const double r = separation(result->x, 0, 1);
  for (std::size_t k = 0; k < 3; ++k) {
    const double e = (result->x[k] - result->x[3 + k]) / r;
    g[k] -= lambda * e;
    g[3 + k] += lambda * e;
  }
  EXPECT_LE(std::abs(std::ranges::max(g, {}, [](double v) { return std::abs(v); })),
            kConstrainedForce);

  constexpr double kStep = 1e-5;
  std::array<double, 2> ends{};
  for (std::size_t i = 0; i < 2; ++i) {
    const double held = kHeld + (i == 0 ? kStep : -kStep);
    const auto moved =
        geo::optimize(result->x, lennard_jones, constrained_options({{{0, 1}, held}}));
    ASSERT_TRUE(moved.has_value()) << moved.error().message;
    ends[i] = moved->energy;
  }
  const double slope = (ends[0] - ends[1]) / (2.0 * kStep);
  EXPECT_NEAR(lambda, slope, 2e-9);
  std::cout << std::format("[ constrained ] LJ4 r01 = {}: {} evaluations, E {:.12f}, lambda "
                           "{:.10f}, dE*/dd {:.10f}\n",
                           kHeld, result->iterations.size(), result->energy, lambda, slope);
}

// Two constraints sharing point 0, both off at the start (1.204 and 1.136):
// coupled, so the correction is a 2 x 2 Newton solve. Both hold at every
// evaluation, and the walk converges.
TEST(GeometryOptimizeConstrained, TwoCoupledDistancesHold) {
  std::vector<std::vector<double>> visited;
  const auto result = geo::optimize(
      kCluster,
      [&](std::span<const double> x, std::span<double> g) {
        visited.emplace_back(x.begin(), x.end());
        return lennard_jones(x, g);
      },
      constrained_options({{{0, 1}, 1.3}, {{2, 0}, 1.0}}));
  ASSERT_TRUE(result.has_value()) << result.error().message;
  for (const std::vector<double> &x : visited) {
    EXPECT_NEAR(separation(x, 0, 1), 1.3, 1.3e-12);
    EXPECT_NEAR(separation(x, 2, 0), 1.0, 1e-12);
  }
  EXPECT_EQ(result->iterations.back().multipliers.size(), 2U);
  EXPECT_LE(result->iterations.back().max_force, kConstrainedForce);
}

// Each malformed constraint is InvalidConfig, and so is one beside
// optimize_meci; three points on a line held at distances 1, 1 and 2 have
// three constraint gradients in one direction, Numerical.
TEST(GeometryOptimizeConstrained, Refusals) {
  const auto refused = [](std::vector<geo::DistanceConstraint> constraints) {
    const auto result = geo::optimize(kCluster, lennard_jones, constrained_options(constraints));
    return !result.has_value() && result.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused({{{0, 4}, 1.0}}));
  EXPECT_TRUE(refused({{{-1, 2}, 1.0}}));
  EXPECT_TRUE(refused({{{1, 1}, 1.0}}));
  EXPECT_TRUE(refused({{{0, 1}, 1.0}, {{1, 0}, 1.1}}));
  EXPECT_TRUE(refused({{{0, 1}, 0.0}}));
  EXPECT_TRUE(refused({{{0, 1}, std::nan("")}}));
  geo::Options flat = tight_options();
  flat.constraints = {{{0, 1}, 1.0}};
  const auto unaligned = geo::optimize(std::vector<double>{0.0, 0.0, 0.0, 1.0}, rosenbrock, flat);
  ASSERT_FALSE(unaligned.has_value());
  EXPECT_EQ(unaligned.error().kind, fb::ErrorKind::InvalidConfig);

  const auto meci = geo::optimize_meci(
      kCluster,
      [](std::span<const double>, std::span<double>, std::span<double>,
         std::span<double>) -> fb::Result<std::array<double, 2>> { return std::array{0.0, 0.0}; },
      constrained_options({{{0, 1}, 1.0}}));
  ASSERT_FALSE(meci.has_value());
  EXPECT_EQ(meci.error().kind, fb::ErrorKind::InvalidConfig);

  const std::vector<double> line{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const auto dependent = geo::optimize(
      line, lennard_jones, constrained_options({{{0, 1}, 1.0}, {{1, 2}, 1.0}, {{0, 2}, 2.0}}));
  ASSERT_FALSE(dependent.has_value());
  EXPECT_EQ(dependent.error().kind, fb::ErrorKind::Numerical) << dependent.error().message;
}

// ---------------------------------------------------------------------------
// optimize_meci
// ---------------------------------------------------------------------------

// The model surfaces' points: a non-planar four-point arrangement (bohr),
// whose six distances are the crossing's.
const std::vector<double> kMeciPoints{0.0, 0.0, 0.0, 2.4, 0.0, 0.0, 0.6, 2.2, 0.0, 0.9, 0.8, 2.0};
constexpr std::array<std::array<std::size_t, 2>, 6> kPairs{
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

std::array<double, 6> distances(std::span<const double> x) {
  std::array<double, 6> r{};
  for (std::size_t k = 0; k < 6; ++k) {
    double r2 = 0.0;
    for (std::size_t c = 0; c < 3; ++c) {
      const double d = x[3 * kPairs[k][0] + c] - x[3 * kPairs[k][1] + c];
      r2 += d * d;
    }
    r[k] = std::sqrt(r2);
  }
  return r;
}

// A 2 x 2 diabatic Hamiltonian over u_k = r_k - r_k(kMeciPoints), the six
// distances' displacements, so every rigid motion is an exact zero mode:
// H_11 = S + d, H_22 = S - d, H_12 = v, with
//   S = sum_k K_k u_k^2 / 2 + t_0 u_0 + t_1 u_1,
//   d = a_0 u_0 + a_1 u_1 + u_0 (p_d . u) + u_1 (q_d . u),
//   v = b_0 u_0 + b_1 u_1 + u_0 (p_v . u) + u_1 (q_v . u).
// Every quadratic coupling carries a factor u_0 or u_1, so d = v = 0 on
// u_0 = u_1 = 0 (or on u_0 = 0 alone when v = 0 and a_1 = 0), and S has its
// least value there at u = 0 when the tilt t is along u_0 and u_1: the known
// crossing is kMeciPoints, up to a rigid motion. There the seam's Lagrangian
// is K over the seam's distances, with no constraint-curvature term (the
// gradient of the Lagrangian in u vanishes and d, v are linear in u_0, u_1
// along the seam), while the quadratic couplings still turn the branching
// plane along it and the distances curve it in Cartesians.
struct Model {
  std::array<double, 6> curvature;
  std::array<double, 2> tilt;
  std::array<double, 2> a;
  std::array<double, 6> p_d;
  std::array<double, 6> q_d;
  std::array<double, 2> b;
  std::array<double, 6> p_v;
  std::array<double, 6> q_v;
};

// The peaked cone. At the crossing the seam's least curvature (its Cartesian
// Lagrangian Hessian on the four seam directions) is 0.394 Eh/bohr^2; the
// plane's pair x = 2 grad d, 2h = 2 grad v has |x| = 0.240 and |2h| = 0.176,
// and the plane's smallest singular value is 0.138 Eh/bohr.
constexpr Model kPeaked{{0.6, 0.5, 0.4, 0.45, 0.35, 0.5},
                        {0.0, 0.0},
                        {0.08, 0.02},
                        {0.04, 0.0, 0.0, 0.05, 0.0, 0.0},
                        {0.0, 0.0, 0.03, 0.0, 0.0, 0.0},
                        {0.01, 0.06},
                        {0.0, 0.0, 0.0, 0.0, 0.0, -0.03},
                        {0.0, 0.02, 0.0, 0.0, 0.04, 0.0}};

// The same cone tilted in its branching plane: the mean energy's gradient at
// the crossing is 0.05 along u_0 and -0.04 along u_1. The tilt leaves the
// seam's curvature and the plane as kPeaked's.
constexpr Model kTilted{kPeaked.curvature, {0.05, -0.04}, kPeaked.a,   kPeaked.p_d,
                        kPeaked.q_d,       kPeaked.b,     kPeaked.p_v, kPeaked.q_v};

// An uncoupled crossing, v = 0: h = 0 everywhere, the seam is d = 0, u_0 = 0,
// of dimension 3N - 7 = 5, and tilted along u_0 by 0.03. Its least seam
// curvature is 0.370 Eh/bohr^2 and |x| = 0.226 Eh/bohr.
constexpr Model kUncoupled{
    kPeaked.curvature, {0.03, 0.0}, {0.08, 0.0}, kPeaked.p_d, {}, {}, {}, {}};

// The value and the u-gradient of l_0 u_0 + l_1 u_1 + u_0 (p . u) + u_1 (q . u).
double coupling(const std::array<double, 2> &l, const std::array<double, 6> &p,
                const std::array<double, 6> &q, const std::array<double, 6> &u,
                std::array<double, 6> &du) {
  double pu = 0.0;
  double qu = 0.0;
  for (std::size_t k = 0; k < 6; ++k) {
    pu += p[k] * u[k];
    qu += q[k] * u[k];
  }
  for (std::size_t k = 0; k < 6; ++k) {
    du[k] = u[0] * p[k] + u[1] * q[k];
  }
  du[0] += l[0] + pu;
  du[1] += l[1] + qu;
  return l[0] * u[0] + l[1] * u[1] + u[0] * pu + u[1] * qu;
}

// The adiabatic pair of the model at x, lower first: E = S -+ rho,
// rho = sqrt(d^2 + v^2), their gradients, and h = (v grad d - d grad v) / rho,
// the coupling <1| grad H |2> of the two eigenvectors (on the seam, rho = 0,
// the pair at mixing angle 0).
fb::Result<std::array<double, 2>> model_pair(const Model &m, std::span<const double> x,
                                             std::span<double> g_i, std::span<double> g_j,
                                             std::span<double> h) {
  const std::array<double, 6> r = distances(x);
  const std::array<double, 6> r0 = distances(kMeciPoints);
  std::array<double, 6> u{};
  std::array<double, 6> ds{};
  double s = 0.0;
  for (std::size_t k = 0; k < 6; ++k) {
    u[k] = r[k] - r0[k];
    s += 0.5 * m.curvature[k] * u[k] * u[k];
    ds[k] = m.curvature[k] * u[k];
  }
  s += m.tilt[0] * u[0] + m.tilt[1] * u[1];
  ds[0] += m.tilt[0];
  ds[1] += m.tilt[1];
  std::array<double, 6> dd{};
  std::array<double, 6> dv{};
  const double d = coupling(m.a, m.p_d, m.q_d, u, dd);
  const double v = coupling(m.b, m.p_v, m.q_v, u, dv);
  const double rho = std::sqrt(d * d + v * v);
  const double cd = rho > 0.0 ? d / rho : 1.0;
  const double cv = rho > 0.0 ? v / rho : 0.0;

  std::ranges::fill(g_i, 0.0);
  std::ranges::fill(g_j, 0.0);
  std::ranges::fill(h, 0.0);
  for (std::size_t k = 0; k < 6; ++k) {
    const double gap_k = cd * dd[k] + cv * dv[k];
    const double h_k = cv * dd[k] - cd * dv[k];
    const std::size_t a = kPairs[k][0];
    const std::size_t b = kPairs[k][1];
    for (std::size_t c = 0; c < 3; ++c) {
      const double e = (x[3 * a + c] - x[3 * b + c]) / r[k];
      g_i[3 * a + c] += (ds[k] - gap_k) * e;
      g_i[3 * b + c] -= (ds[k] - gap_k) * e;
      g_j[3 * a + c] += (ds[k] + gap_k) * e;
      g_j[3 * b + c] -= (ds[k] + gap_k) * e;
      h[3 * a + c] += h_k * e;
      h[3 * b + c] -= h_k * e;
    }
  }
  return std::array<double, 2>{s - rho, s + rho};
}

// The largest distance of a point of x from its place in ref after the best
// proper superposition: centroids matched, then the rotation that is the
// polar factor of the 3 x 3 cross-covariance, by Newton's iteration
// R <- (R + R^-T) / 2 (Higham), which converges to it quadratically.
double aligned_error(std::span<const double> x, std::span<const double> ref) {
  const auto cx = centroid(x);
  const auto cr = centroid(ref);
  const std::size_t points = x.size() / 3;
  std::array<std::array<double, 3>, 3> rot{};
  for (std::size_t i = 0; i < points; ++i) {
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = 0; b < 3; ++b) {
        rot[a][b] += (ref[3 * i + a] - cr[a]) * (x[3 * i + b] - cx[b]);
      }
    }
  }
  for (std::int32_t it = 0; it < 60; ++it) {
    // R^-T is the cofactor matrix over the determinant.
    std::array<std::array<double, 3>, 3> cof{};
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = 0; b < 3; ++b) {
        const std::size_t a1 = (a + 1) % 3;
        const std::size_t a2 = (a + 2) % 3;
        const std::size_t b1 = (b + 1) % 3;
        const std::size_t b2 = (b + 2) % 3;
        cof[a][b] = rot[a1][b1] * rot[a2][b2] - rot[a1][b2] * rot[a2][b1];
      }
    }
    const double det = rot[0][0] * cof[0][0] + rot[0][1] * cof[0][1] + rot[0][2] * cof[0][2];
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = 0; b < 3; ++b) {
        rot[a][b] = 0.5 * (rot[a][b] + cof[a][b] / det);
      }
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < points; ++i) {
    double e2 = 0.0;
    for (std::size_t a = 0; a < 3; ++a) {
      double placed = cr[a];
      for (std::size_t b = 0; b < 3; ++b) {
        placed += rot[a][b] * (x[3 * i + b] - cx[b]);
      }
      e2 += (placed - ref[3 * i + a]) * (placed - ref[3 * i + a]);
    }
    worst = std::max(worst, std::sqrt(e2));
  }
  return worst;
}

// The thresholds the bar is derived from: gap 1e-12 Eh, forces 1e-10 Eh/bohr,
// displacements 1e-9 bohr. A converged point is then within
// force / curvature + gap / (the plane's least slope) of the crossing, under
// 1e-10 / 0.37 + 1e-12 / 0.138 = 2.8e-10 bohr for every model here, inside
// the 1e-8 bohr bar.
geo::Options meci_options() {
  geo::Options options;
  options.max_gap = 1e-12;
  options.max_force = 1e-10;
  options.rms_force = 1e-10;
  options.max_displacement = 1e-9;
  options.rms_displacement = 1e-9;
  return options;
}

constexpr double kMeciBar = 1e-8;

// Off the seam: every point moved by about 0.1 bohr.
std::vector<double> off_seam_start() {
  constexpr std::array<double, 12> kShift{0.1,  -0.05, 0.08, -0.12, 0.07, 0.03,
                                          0.05, -0.09, 0.11, -0.06, 0.1,  -0.08};
  std::vector<double> x = kMeciPoints;
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] += kShift[i];
  }
  return x;
}

// On the seam away from its minimum: only the fourth point moves, so the
// first two distances, u_0 and u_1, stay exactly 0 and the gap is 0.
std::vector<double> on_seam_start() {
  std::vector<double> x = kMeciPoints;
  x[9] += 0.3;
  x[10] -= 0.2;
  x[11] += 0.25;
  return x;
}

// optimize_meci over the model from x0, held to the bar: converged within the
// thresholds, at the known crossing to kMeciBar after alignment, its
// centroid fixed and every step free of rotation, and the plane's dimension
// the model's at every evaluation.
fb::Result<geo::MECISolution> expect_meci(const std::string &name, const Model &model,
                                          const std::vector<double> &x0,
                                          const geo::Options &options, std::int32_t dimension) {
  std::vector<std::vector<double>> visited;
  auto result = geo::optimize_meci(
      x0,
      [&](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
          std::span<double> h) {
        visited.emplace_back(x.begin(), x.end());
        return model_pair(model, x, g_i, g_j, h);
      },
      options);
  EXPECT_TRUE(result.has_value()) << name << ": " << result.error().message;
  if (!result.has_value()) {
    return result;
  }
  EXPECT_EQ(visited.size(), result->iterations.size()) << name;

  const double error = aligned_error(result->x, kMeciPoints);
  ::testing::Test::RecordProperty(name + "_position_error", std::format("{:.3e}", error));
  ::testing::Test::RecordProperty(name + "_evaluations", std::to_string(visited.size()));
  EXPECT_LE(error, kMeciBar) << name;

  const geo::MECISolution::Iteration &last = result->iterations.back();
  EXPECT_LE(last.gap, options.max_gap) << name;
  EXPECT_LE(last.max_force, options.max_force) << name;
  EXPECT_LE(last.rms_force, options.rms_force) << name;
  EXPECT_LE(last.max_displacement, options.max_displacement) << name;
  EXPECT_LE(last.rms_displacement, options.rms_displacement) << name;
  EXPECT_EQ(result->lower, last.lower) << name;
  EXPECT_EQ(result->upper, last.upper) << name;
  for (const geo::MECISolution::Iteration &it : result->iterations) {
    EXPECT_EQ(it.branching_dimension, dimension) << name;
  }

  const auto c0 = centroid(x0);
  const auto c = centroid(result->x);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(c[k], c0[k], 1e-12) << name;
  }
  // No step is rejected, so each is taken from the evaluation before it.
  for (std::size_t e = 1; e < visited.size(); ++e) {
    const std::vector<double> &from = visited[e - 1];
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
      EXPECT_NEAR(torque[k], 0.0, 1e-12) << name << ", evaluation " << e << ", axis " << k;
    }
  }
  return result;
}

TEST(GeometryOptimizeMeci, PeakedConeFromOffAndOnTheSeam) {
  expect_meci("peaked_off_seam", kPeaked, off_seam_start(), meci_options(), 2);
  expect_meci("peaked_on_seam", kPeaked, on_seam_start(), meci_options(), 2);
}

TEST(GeometryOptimizeMeci, TiltedConeFromOffAndOnTheSeam) {
  expect_meci("tilted_off_seam", kTilted, off_seam_start(), meci_options(), 2);
  expect_meci("tilted_on_seam", kTilted, on_seam_start(), meci_options(), 2);
}

TEST(GeometryOptimizeMeci, UncoupledCrossingDropsTheCoupling) {
  // h = 0 at every geometry, so y is dropped at every step (dimension 1) and
  // the seam is 3N - 7.
  expect_meci("uncoupled_off_seam", kUncoupled, off_seam_start(), meci_options(), 1);
  expect_meci("uncoupled_on_seam", kUncoupled, on_seam_start(), meci_options(), 1);
}

TEST(GeometryOptimizeMeci, OvershootShrinksTheTrustRadius) {
  // On the peaked seam, the fourth point 0.44 bohr from its place, with a
  // starting Hessian about 20 times softer than the seam: the first step is
  // cut to the trust radius of 0.5, and the model predicts the mean energy
  // plus the gap to fall by 0.164 Eh where it falls by 0.021 (ratio 0.13,
  // when this was written), so the radius drops to a quarter of the step.
  // The run still converges to the bar.
  geo::Options options = meci_options();
  options.initial_hessian = 0.02;
  options.trust_radius = 0.5;
  options.max_trust_radius = 0.5;
  const auto result = expect_meci("overshoot", kPeaked, on_seam_start(), options, 2);
  ASSERT_TRUE(result.has_value());
  ASSERT_GE(result->iterations.size(), 2U);
  EXPECT_DOUBLE_EQ(result->iterations[0].trust_radius, 0.5);
  // The first step's length, from its rms over the 12 coordinates.
  EXPECT_NEAR(result->iterations[1].rms_displacement * std::sqrt(12.0), 0.5, 1e-12);
  EXPECT_NEAR(result->iterations[1].trust_radius, 0.125, 1e-12);
}

TEST(GeometryOptimizeMeci, Refusals) {
  const auto unused = [](std::span<const double>, std::span<double> g_i, std::span<double> g_j,
                         std::span<double> h) -> fb::Result<std::array<double, 2>> {
    std::ranges::fill(g_i, 0.0);
    std::ranges::fill(g_j, 1.0);
    std::ranges::fill(h, 0.0);
    return std::array<double, 2>{0.0, 1.0};
  };
  const std::vector<double> x0{0.0, 0.0, 0.0};
  const auto refused = [&](std::span<const double> x, const geo::Options &options) {
    const auto result = geo::optimize_meci(x, unused, options);
    return !result.has_value() && result.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused({}, {}));
  EXPECT_TRUE(refused(std::vector<double>{0.0, 0.0, std::nan("")}, {}));
  EXPECT_TRUE(refused(std::vector<double>{0.0, 1.0}, {}));
  geo::Options options;
  options.max_gap = 0.0;
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.rms_force = std::nan("");
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.trust_radius = 2.0;
  EXPECT_TRUE(refused(x0, options));
  options = {};
  options.max_steps = -1;
  EXPECT_TRUE(refused(x0, options));
}

TEST(GeometryOptimizeMeci, MaxStepsAndCallbackErrors) {
  geo::Options options = meci_options();
  options.max_steps = 2;
  const auto peaked = [](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                         std::span<double> h) { return model_pair(kPeaked, x, g_i, g_j, h); };
  const auto capped = geo::optimize_meci(off_seam_start(), peaked, options);
  ASSERT_FALSE(capped.has_value());
  EXPECT_EQ(capped.error().kind, fb::ErrorKind::MaxIterationsExceeded);

  std::int32_t calls = 0;
  const auto failing = [&](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                           std::span<double> h) -> fb::Result<std::array<double, 2>> {
    if (++calls == 2) {
      return fb::err_divergence("scf did not converge");
    }
    return model_pair(kPeaked, x, g_i, g_j, h);
  };
  const auto failed = geo::optimize_meci(off_seam_start(), failing, options);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().kind, fb::ErrorKind::Divergence);

  const auto not_finite = [](std::span<const double>, std::span<double> g_i, std::span<double> g_j,
                             std::span<double> h) -> fb::Result<std::array<double, 2>> {
    std::ranges::fill(g_i, 0.0);
    std::ranges::fill(g_j, 1.0);
    std::ranges::fill(h, std::numeric_limits<double>::quiet_NaN());
    return std::array<double, 2>{0.0, 1.0};
  };
  const auto nan = geo::optimize_meci(off_seam_start(), not_finite, options);
  ASSERT_FALSE(nan.has_value());
  EXPECT_EQ(nan.error().kind, fb::ErrorKind::Numerical);

  // Equal gradients: no x to step the gap along.
  const auto parallel = [](std::span<const double>, std::span<double> g_i, std::span<double> g_j,
                           std::span<double> h) -> fb::Result<std::array<double, 2>> {
    std::ranges::fill(g_i, 0.0);
    std::ranges::fill(g_j, 0.0);
    std::ranges::fill(h, 0.0);
    return std::array<double, 2>{0.0, 1.0};
  };
  const auto flat = geo::optimize_meci(off_seam_start(), parallel, options);
  ASSERT_FALSE(flat.has_value());
  EXPECT_EQ(flat.error().kind, fb::ErrorKind::Numerical);
}

TEST(GeometryOptimizeMeci, EnergiesInEitherOrder) {
  // The callback hands the upper state first: the result is the same
  // crossing, the lower energy reported as lower.
  const auto swapped = [](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                          std::span<double> h) -> fb::Result<std::array<double, 2>> {
    const auto e = model_pair(kTilted, x, g_j, g_i, h);
    return std::array<double, 2>{(*e)[1], (*e)[0]};
  };
  const auto result = geo::optimize_meci(off_seam_start(), swapped, meci_options());
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_LE(aligned_error(result->x, kMeciPoints), kMeciBar);
  EXPECT_LE(result->lower, result->upper);
}

// A callback with no plane near the seam (#902's QD-PC walk inside a
// degenerate CASSCF pair): under a gap of kFloor it writes the mean gradient
// into both g_I and g_J and zeros into h, and the plane is carried. As in
// #902, the floor sits under max_gap, so a carried point already meets the
// gap criterion, and the walk's remaining work is P g. The peaked cone has
// no tilt, so the mean gradient vanishes at the crossing and a carried
// plane misplaces P g by second order only: the walk still reaches the
// crossing, within force / curvature + max_gap / (the plane's least slope),
// 1e-10 / 0.37 + 1e-8 / 0.138 = 7.3e-8 bohr, under kCarriedBar; the plane
// carried from the first evaluation under the floor on, and never at the
// start.
TEST(GeometryOptimizeMeci, CarriesThePlaneUnderAFloor) {
  constexpr double kFloor = 1e-9;
  constexpr double kCarriedBar = 1e-7;
  geo::Options options = meci_options();
  options.max_gap = 1e-8;
  const auto mean_only = [](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                            std::span<double> h) -> fb::Result<std::array<double, 2>> {
    const auto e = model_pair(kPeaked, x, g_i, g_j, h);
    if (e && (*e)[1] - (*e)[0] < kFloor) {
      for (std::size_t k = 0; k < g_i.size(); ++k) {
        g_i[k] = 0.5 * (g_i[k] + g_j[k]);
      }
      std::ranges::copy(g_i, g_j.begin());
      std::ranges::fill(h, 0.0);
    }
    return e;
  };
  const auto result = geo::optimize_meci(off_seam_start(), mean_only, options);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  const double error = aligned_error(result->x, kMeciPoints);
  ::testing::Test::RecordProperty("carried_position_error", std::format("{:.3e}", error));
  ::testing::Test::RecordProperty("carried_evaluations", std::to_string(result->iterations.size()));
  EXPECT_LE(error, kCarriedBar);
  std::size_t carried = 0;
  for (std::size_t k = 0; k < result->iterations.size(); ++k) {
    const geo::MECISolution::Iteration &it = result->iterations[k];
    EXPECT_EQ(it.plane_carried, it.gap < kFloor) << "evaluation " << k;
    EXPECT_EQ(it.branching_dimension, 2) << "evaluation " << k;
    carried += it.plane_carried ? 1U : 0U;
  }
  EXPECT_FALSE(result->iterations.front().plane_carried);
  EXPECT_GE(carried, 1U);
  EXPECT_TRUE(result->iterations.back().plane_carried);
  std::cout << std::format("[ meci ] carried: {} evaluations, {} carried, position error {:.3e}\n",
                           result->iterations.size(), carried, error);

  // On the seam from the start: no plane to carry, refused.
  const auto start = geo::optimize_meci(on_seam_start(), mean_only, options);
  ASSERT_FALSE(start.has_value());
  EXPECT_EQ(start.error().kind, fb::ErrorKind::Numerical);
  EXPECT_NE(start.error().message.find("no earlier branching plane"), std::string::npos)
      << start.error().message;
}

// --- irc (#1269) ------------------------------------------------------------

// One hydrogen at the origin, the coordinates its own (no rigid motions).
std::vector<fb::Atom> one_hydrogen() {
  return {fb::Atom{{0.0, 0.0, 0.0}, fb::Element::H}};
}

// The curved valley E = (x^2 - 1)^2 + 2 (y - x^2 / 2)^2 + z^2 / 2: a saddle at
// the origin, its Hessian diag(-4, 4, 1), and minima at (+-1, 1/2, 0), the
// path between them bending through y = x^2 / 2.
fb::Result<double> valley(std::span<const double> x, std::span<double> gradient) {
  const double a = (x[0] * x[0]) - 1.0;
  const double b = x[1] - (0.5 * x[0] * x[0]);
  gradient[0] = (4.0 * x[0] * a) - (4.0 * b * x[0]);
  gradient[1] = 4.0 * b;
  gradient[2] = x[2];
  return (a * a) + (2.0 * b * b) + (0.5 * x[2] * x[2]);
}

constexpr std::array<double, 9> kValleyHessian{-4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0};

// The hydrogen mass harmonic_analysis and irc weight with.
constexpr double kHydrogenMass = 1.008;

geo::IrcOptions valley_options() {
  geo::IrcOptions options;
  options.step_size = 0.05;
  options.max_points = 200;
  options.max_force = 1e-6;
  options.sphere_gradient = 1e-10;
  options.sphere_displacement = 1e-8;
  options.project_rigid_body = false;
  return options;
}

// Each side of the valley: every point k + 1 on the sphere of radius s / 2
// about point k's pivot (to 1e-12), its gradient's part tangent to that
// sphere under sphere_gradient (recomputed here), the energy falling at every
// point, and the walk stopped with |g| halved over its last step, so within
// one step (mass-weighted) of the side's minimum. This is the case the stop
// was added for (#1269): without it, the next sphere holds the minimum near
// its pivot, and its iteration flips between the two sides and never
// converges (SphereNotConverged, at both sides).
TEST(GeometryIrc, CurvedValleyHoldsTheHypersphereAndReachesBothMinima) {
  const geo::IrcOptions options = valley_options();
  const double sqrt_m = std::sqrt(kHydrogenMass);
  for (const double side : {1.0, -1.0}) {
    const std::array<double, 3> direction{side, 0.0, 0.0};
    const auto path = geo::irc(one_hydrogen(), kValleyHessian, direction, valley, options);
    ASSERT_TRUE(path.has_value()) << path.error().message;
    const auto &points = path->points;
    ASSERT_GE(points.size(), 3U);
    EXPECT_EQ(path->stop, geo::IrcStop::MinimumWithinStep) << geo::irc_stop_name(path->stop);
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
      EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
      EXPECT_DOUBLE_EQ(points[k + 1].arc_length, static_cast<double>(k + 1) * options.step_size);
      // The pivot: half a step from point k along the way out, or down.
      std::array<double, 3> g{};
      (void)valley(points[k].x, g);
      std::array<double, 3> down{side, 0.0, 0.0};
      if (k > 0) {
        const double norm = std::hypot(g[0], g[1], g[2]) / sqrt_m;
        for (int32_t i = 0; i < 3; ++i) {
          down[i] = -(g[i] / sqrt_m) / norm;
        }
      }
      std::array<double, 3> p{};
      for (int32_t i = 0; i < 3; ++i) {
        p[i] = sqrt_m * points[k + 1].x[i] -
               (sqrt_m * points[k].x[i] + 0.5 * options.step_size * down[i]);
      }
      EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 0.5 * options.step_size, 1e-12) << k;
      std::array<double, 3> g1{};
      (void)valley(points[k + 1].x, g1);
      for (double &v : g1) {
        v /= sqrt_m;
      }
      const double along =
          (g1[0] * p[0] + g1[1] * p[1] + g1[2] * p[2]) / (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const double tangential =
          std::hypot(g1[0] - along * p[0], g1[1] - along * p[1], g1[2] - along * p[2]);
      EXPECT_LE(tangential, options.sphere_gradient) << k;
      // Downhill: the gradient points back at the pivot.
      EXPECT_LT(along, 0.0) << k;
    }
    const auto &end = points.back().x;
    const double miss = sqrt_m * std::hypot(end[0] - side, end[1] - 0.5, end[2]);
    EXPECT_LE(miss, options.step_size) << "side " << side;
    // It left along the way out.
    EXPECT_GT(side * points[1].x[0], 0.0);
  }
}

// #1278: a soft valley bending on an arc of radius R = 10 about (0, R), with
// a stiff floor across it, E = 1e-3 u^2 + 3 (r - R)^2 + z^2 / 200, r the
// distance from (0, R, 0), u = theta^2 / theta_m^2 - 1, theta the angle
// along the arc from the TS at the origin, theta_m = 0.3: minima at
// (+-R sin theta_m, R (1 - cos theta_m), 0).
fb::Result<double> arc_valley(std::span<const double> x, std::span<double> gradient) {
  constexpr double kRadius = 10.0;
  constexpr double kThetaM2 = 0.09;
  const double dx = x[0];
  const double dy = x[1] - kRadius;
  const double r2 = (dx * dx) + (dy * dy);
  const double r = std::sqrt(r2);
  const double theta = std::atan2(dx, -dy);
  const double u = (theta * theta / kThetaM2) - 1.0;
  const double de_dtheta = 1e-3 * 4.0 * u * theta / kThetaM2;
  const double de_dr = 6.0 * (r - kRadius);
  gradient[0] = (de_dr * dx / r) - (de_dtheta * dy / r2);
  gradient[1] = (de_dr * dy / r) + (de_dtheta * dx / r2);
  gradient[2] = 1e-2 * x[2];
  return (1e-3 * u * u) + (3.0 * (r - kRadius) * (r - kRadius)) + (5e-3 * x[2] * x[2]);
}

// At the default sphere thresholds each point of the arc valley is off its
// floor by up to sphere_gradient / 6, so its gradient carries a part across
// the floor, and that puts the minimum along the steepest descent,
// |g| / (u H u), within half a step 10 steps short of the minimum: the model's
// line test, the stop before #1278, ended each side there (point 20 of 30).
// |g| sees that part at second order, so the walk runs on to within a step
// of the minimum, its energy falling at every point. Its last points' |g| is
// below sphere_gradient, where a sphere's point is not resolved, so which of
// the two stops that leave it within a step ends the walk is not either: the
// |g| halving, or the next point's energy rising (since #1288's first trial).
TEST(GeometryIrc, SoftCurvedValleyRunsToItsMinimum) {
  // d2E/dtheta2 = -4e-3 / theta_m^2 at the TS, over R^2 along x.
  const std::array<double, 9> hessian{
      -4e-3 / (0.09 * 100.0), 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1e-2};
  geo::IrcOptions options;
  options.max_points = 200;
  options.max_force = 1e-10;
  options.project_rigid_body = false;
  const double sqrt_m = std::sqrt(kHydrogenMass);
  for (const double side : {1.0, -1.0}) {
    const std::array<double, 3> direction{side, 0.0, 0.0};
    const auto path = geo::irc(one_hydrogen(), hessian, direction, arc_valley, options);
    ASSERT_TRUE(path.has_value()) << path.error().message;
    const auto &points = path->points;
    EXPECT_TRUE(path->stop == geo::IrcStop::MinimumWithinStep ||
                path->stop == geo::IrcStop::EnergyRose)
        << geo::irc_stop_name(path->stop);
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
      EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
    }
    const auto &end = points.back().x;
    const double miss = sqrt_m * std::hypot(end[0] - (side * 10.0 * std::sin(0.3)),
                                            end[1] - (10.0 * (1.0 - std::cos(0.3))), end[2]);
    ::testing::Test::RecordProperty(side > 0.0 ? "forward_miss" : "backward_miss",
                                    std::format("{:.3e}", miss));
    std::cout << std::format("[ irc ] soft arc valley side {:+.0f}: {} points, {:.4f} amu^1/2 "
                             "bohr from the minimum (bar {})\n",
                             side, points.size() - 1, miss, options.step_size);
    EXPECT_LE(miss, options.step_size) << "side " << side;
  }
}

// #1285: a stiff valley bent sharply, E = (x^2 - 1)^2 + k (y - 2 x^2)^2 +
// z^2 / 2, its TS at the origin (Hessian diag(-4, 2k, 1)) and minima at
// (+-1, 2, 0). The first sphere passes through the TS, where E on it has a
// minimum; the model's step from the mode-displaced point jumped the sphere's
// diameter back to it, the search converged there, and each side stopped
// EnergyRose with the TS alone, at k = 200 and s = 0.05 (the issue's case),
// k = 400 and s = 0.05, and k = 200 and s = 0.1. The retry holding the trials
// ahead of the pivot leaves the TS, puts point 1 on its sphere with its
// tangential gradient under sphere_gradient, and runs each side to within a
// step of its minimum, the energy falling at every point.
TEST(GeometryIrc, StiffSharplyCurvedValleyLeavesTheTransitionState) {
  const double sqrt_m = std::sqrt(kHydrogenMass);
  for (const auto [stiffness, step] : {std::pair{200.0, 0.05}, {400.0, 0.05}, {200.0, 0.1}}) {
    const auto surface = [stiffness](std::span<const double> x,
                                     std::span<double> gradient) -> fb::Result<double> {
      const double a = (x[0] * x[0]) - 1.0;
      const double b = x[1] - (2.0 * x[0] * x[0]);
      gradient[0] = (4.0 * x[0] * a) - (8.0 * stiffness * b * x[0]);
      gradient[1] = 2.0 * stiffness * b;
      gradient[2] = x[2];
      return (a * a) + (stiffness * b * b) + (0.5 * x[2] * x[2]);
    };
    const std::array<double, 9> hessian{-4.0, 0.0, 0.0, 0.0, 2.0 * stiffness, 0.0, 0.0, 0.0, 1.0};
    geo::IrcOptions options;
    options.step_size = step;
    options.max_points = 200;
    options.project_rigid_body = false;
    for (const double side : {1.0, -1.0}) {
      const std::array<double, 3> direction{side, 0.0, 0.0};
      const auto path = geo::irc(one_hydrogen(), hessian, direction, surface, options);
      ASSERT_TRUE(path.has_value()) << path.error().message;
      const auto &points = path->points;
      ASSERT_GE(points.size(), 3U) << "k " << stiffness << " s " << step << " side " << side;
      EXPECT_EQ(path->stop, geo::IrcStop::MinimumWithinStep) << geo::irc_stop_name(path->stop);
      for (std::size_t k = 0; k + 1 < points.size(); ++k) {
        EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
      }
      // Point 1: ahead, on the sphere of radius s / 2 about the pivot s / 2
      // along the way out, its gradient's part tangent to it converged.
      const auto &first = points[1].x;
      EXPECT_GT(side * first[0], 0.0);
      const std::array<double, 3> p{(sqrt_m * first[0]) - (0.5 * step * side), sqrt_m * first[1],
                                    sqrt_m * first[2]};
      EXPECT_NEAR(std::hypot(p[0], p[1], p[2]), 0.5 * step, 1e-12);
      EXPECT_LE(points[1].tangential_gradient, options.sphere_gradient);
      const auto &end = points.back().x;
      const double miss = sqrt_m * std::hypot(end[0] - side, end[1] - 2.0, end[2]);
      std::cout << std::format("[ irc ] stiff valley k {} s {} side {:+.0f}: {} points, point 1 "
                               "in {} evaluations, {:.4f} amu^1/2 bohr from the minimum\n",
                               stiffness, step, side, points.size() - 1,
                               points[1].sphere_iterations, miss);
      EXPECT_LE(miss, step) << "k " << stiffness << " s " << step << " side " << side;
    }
  }
}

// The stiff valley bent by b, E = (x^2 - 1)^2 + k (y - b x^2)^2 + z^2 / 2,
// minima at (+-1, b, 0), as a callback.
std::function<fb::Result<double>(std::span<const double>, std::span<double>)>
bent_valley(double stiffness, double bend) {
  return [stiffness, bend](std::span<const double> x,
                           std::span<double> gradient) -> fb::Result<double> {
    const double a = (x[0] * x[0]) - 1.0;
    const double b = x[1] - (bend * x[0] * x[0]);
    gradient[0] = (4.0 * x[0] * a) - (4.0 * bend * stiffness * b * x[0]);
    gradient[1] = 2.0 * stiffness * b;
    gradient[2] = x[2];
    return (a * a) + (stiffness * b * b) + (0.5 * x[2] * x[2]);
  };
}

// #1288: the valley above stiffer or bent harder, at the default sphere
// thresholds and max_sphere_iterations. A later sphere's first trial was the
// Bofill model's step from point k, across the sphere's diameter, and on these
// valleys the model is not good over that distance: a mid-valley sphere used
// up its 20 evaluations, and each side stopped SphereNotConverged part-way
// (50 of 84 points at k = 200, b = 4, s = 0.05; 14 of 23 at k = 1000, b = 2,
// s = 0.1). From the far pole turned as the last step turned (the first
// trial wherever the model's step lands in the hard case or behind the pivot,
// or follows a sphere of more than 3, #1295), each side runs to within a step
// of its minimum, its energy falling at every point. Each side's points are
// the ones 200 evaluations per sphere reached before.
// k = 1000, b = 4, s = 0.1 is the limit left (src/geometry/README.md): its
// first sphere, which the fix does not change, stops at the default budget
// with the TS alone, and runs to its minimum at 40.
TEST(GeometryIrc, StiffBentValleyRunsToItsMinimaAtTheDefaultBudget) {
  const double sqrt_m = std::sqrt(kHydrogenMass);
  struct Valley {
    double stiffness;
    double bend;
    double step;
  };
  const auto walk = [&](const Valley &v, const geo::IrcOptions &options, double side) {
    const std::array<double, 9> hessian{-4.0, 0.0, 0.0, 0.0, 2.0 * v.stiffness, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 3> direction{side, 0.0, 0.0};
    return geo::irc(one_hydrogen(), hessian, direction, bent_valley(v.stiffness, v.bend), options);
  };
  const auto options_for = [](const Valley &v) {
    geo::IrcOptions options;
    options.step_size = v.step;
    options.max_points = 200;
    options.project_rigid_body = false;
    return options;
  };
  for (const Valley v :
       {Valley{200.0, 4.0, 0.05}, Valley{200.0, 4.0, 0.1}, Valley{400.0, 2.0, 0.1},
        Valley{1000.0, 2.0, 0.05}, Valley{1000.0, 2.0, 0.1}, Valley{1000.0, 1.0, 0.05},
        Valley{1000.0, 1.0, 0.1}, Valley{1000.0, 4.0, 0.05}}) {
    const geo::IrcOptions options = options_for(v);
    for (const double side : {1.0, -1.0}) {
      const auto path = walk(v, options, side);
      ASSERT_TRUE(path.has_value()) << path.error().message;
      const auto &points = path->points;
      EXPECT_EQ(path->stop, geo::IrcStop::MinimumWithinStep)
          << "k " << v.stiffness << " b " << v.bend << " s " << v.step << " side " << side << ": "
          << geo::irc_stop_name(path->stop) << " after " << points.size() - 1;
      int32_t evaluations = 0;
      int32_t worst = 0;
      for (std::size_t k = 0; k + 1 < points.size(); ++k) {
        EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
        evaluations += points[k + 1].sphere_iterations;
        if (k > 0) {
          worst = std::max(worst, points[k + 1].sphere_iterations);
        }
      }
      const auto &end = points.back().x;
      const double miss = sqrt_m * std::hypot(end[0] - side, end[1] - v.bend, end[2]);
      std::cout << std::format("[ irc ] bent valley k {} b {} s {} side {:+.0f}: {} points, {} "
                               "evaluations, a later sphere's most {}, {:.4f} amu^1/2 bohr from "
                               "the minimum\n",
                               v.stiffness, v.bend, v.step, side, points.size() - 1, evaluations,
                               worst, miss);
      EXPECT_LE(miss, v.step) << "k " << v.stiffness << " b " << v.bend << " s " << v.step
                              << " side " << side;
    }
  }
  // The limit: the first sphere's two searches.
  const Valley limit{1000.0, 4.0, 0.1};
  geo::IrcOptions options = options_for(limit);
  const auto stopped = walk(limit, options, 1.0);
  ASSERT_TRUE(stopped.has_value()) << stopped.error().message;
  EXPECT_EQ(stopped->stop, geo::IrcStop::SphereNotConverged) << geo::irc_stop_name(stopped->stop);
  EXPECT_EQ(stopped->points.size(), 1U);
  options.max_sphere_iterations = 40;
  const auto raised = walk(limit, options, 1.0);
  ASSERT_TRUE(raised.has_value()) << raised.error().message;
  EXPECT_EQ(raised->stop, geo::IrcStop::MinimumWithinStep) << geo::irc_stop_name(raised->stop);
  const auto &end = raised->points.back().x;
  EXPECT_LE(sqrt_m * std::hypot(end[0] - 1.0, end[1] - limit.bend, end[2]), limit.step);
}

// #1295: a model of the Cope PC-NEVPT2 IRC's surface, for two hydrogens'
// six coordinates (no rigid motions). A long valley bent gently on an arc of
// radius R = 30 about (0, R) in (x0, x1), its stiff floor 0.15 (r - R)^2, and
// the profile along the arc length sigma = R theta from the TS at the origin,
// E_p = D (sigma^2 / sigma_m^2 - 1)^2 / (1 + sigma^2 / c^2)^(3/2), D = 0.049,
// sigma_m = 11, c = 4.3: a stiff TS (curvature -9.6e-3, the Cope's ~500i
// cm^-1 mass-weighted), |g| peaking at 1.2e-2 two units out and falling over
// the next nine by ~40, as the Cope's mass-weighted |g| does, to a soft
// basin. Four transverse coordinates x2..x5 of curvature 4e-4 to 5e-2 (the
// Cope's soft torsions up), each with its floor rising as a_i sigma^2 / 2
// (a = 0, 0.05, 0.02, 0.01), so the path bends through them too. Minima at
// sigma = +-sigma_m on the arc, x_(2+i) = a_i sigma_m^2 / 2.
fb::Result<double> long_valley(std::span<const double> x, std::span<double> gradient) {
  constexpr double kRadius = 30.0;
  constexpr double kDepth = 0.049;
  constexpr double kSigmaM2 = 121.0;
  constexpr double kC2 = 4.3 * 4.3;
  constexpr double kFloor = 0.15;
  constexpr std::array<double, 4> kStiffness{2e-4, 1e-3, 5e-3, 2.5e-2};
  constexpr std::array<double, 4> kRise{0.0, 0.05, 0.02, 0.01};
  const double dx = x[0];
  const double dy = x[1] - kRadius;
  const double r2 = (dx * dx) + (dy * dy);
  const double r = std::sqrt(r2);
  const double sigma = kRadius * std::atan2(dx, -dy);
  const double a = (sigma * sigma / kSigmaM2) - 1.0;
  const double b = 1.0 + (sigma * sigma / kC2);
  double energy = kDepth * a * a / std::pow(b, 1.5);
  double de_dsigma = kDepth * ((4.0 * a * sigma / kSigmaM2 / std::pow(b, 1.5)) -
                               (3.0 * a * a * sigma / kC2 / std::pow(b, 2.5)));
  for (std::size_t i = 0; i < kStiffness.size(); ++i) {
    const double d = x[2 + i] - (0.5 * kRise[i] * sigma * sigma);
    energy += kStiffness[i] * d * d;
    de_dsigma -= 2.0 * kStiffness[i] * d * kRise[i] * sigma;
    gradient[2 + i] = 2.0 * kStiffness[i] * d;
  }
  energy += kFloor * (r - kRadius) * (r - kRadius);
  const double de_dr = 2.0 * kFloor * (r - kRadius);
  gradient[0] = (de_dr * dx / r) - (de_dsigma * kRadius * dy / r2);
  gradient[1] = (de_dr * dy / r) + (de_dsigma * kRadius * dx / r2);
  return energy;
}

// The long valley's TS Hessian, diagonal: E_p'' = -2 D (2 / sigma_m^2 +
// 3 / (2 c^2)) along x0, the floor's and the transverse coordinates' 2 k.
std::array<double, 36> long_valley_hessian() {
  std::array<double, 36> hessian{};
  const std::array<double, 6> diagonal{
      -2.0 * 0.049 * ((2.0 / 121.0) + (1.5 / (4.3 * 4.3))), 0.3, 4e-4, 2e-3, 1e-2, 5e-2};
  for (std::size_t i = 0; i < 6; ++i) {
    hessian[i * 6 + i] = diagonal[i];
  }
  return hessian;
}

std::vector<fb::Atom> two_hydrogens() {
  return {fb::Atom{{0.0, 0.0, 0.0}, fb::Element::H}, fb::Atom{{0.0, 0.0, 0.0}, fb::Element::H}};
}

// #1295: on the long valley at every default but max_force (held out of the
// way, so the walk stops by the |g| halving), the model's step from point k
// is the sphere's minimum to its tolerance at most points, as on the Cope
// (69% there before #1288). With #1288's far pole as every later sphere's
// first trial, which carries each point's error on its sphere into the next
// start, a side took 215 evaluations, 1.69x, and 24 of 117 later spheres in
// one (the Cope's: 1.64x, 15%). kPreFarPole is a side's count before #1288,
// the old irc.cpp built against this test: 127 over 117 points, 106 of 116
// in one. The first trial now the model's step unless it is in the hard
// case, behind the pivot or after a sphere of more than 3, a side walks
// those spheres exactly as then up to point 114, and is held to the count
// within 10%: the last few points' |g| is below sphere_gradient, where a
// sphere's point is not resolved, and they take 2 or 3 evaluations either
// way (132 over 118 points, 104 of 117 in one).
TEST(GeometryIrc, SoftLongValleyTakesTheModelsStep) {
  constexpr int32_t kPreFarPole = 127;
  geo::IrcOptions options;
  options.max_points = 400;
  options.max_force = 1e-10;
  options.project_rigid_body = false;
  const std::array<double, 36> hessian = long_valley_hessian();
  for (const double side : {1.0, -1.0}) {
    const std::array<double, 6> direction{side, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto path = geo::irc(two_hydrogens(), hessian, direction, long_valley, options);
    ASSERT_TRUE(path.has_value()) << path.error().message;
    const auto &points = path->points;
    int32_t evaluations = 0;
    int32_t first = 0;
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
      EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
      evaluations += points[k + 1].sphere_iterations;
      first += k > 0 && points[k + 1].sphere_iterations == 1 ? 1 : 0;
    }
    std::cout << std::format("[ irc ] long valley side {:+.0f}: {} points, {}, {} evaluations, "
                             "{} of {} later spheres in one\n",
                             side, points.size() - 1, geo::irc_stop_name(path->stop), evaluations,
                             first, points.size() - 2);
    EXPECT_EQ(path->stop, geo::IrcStop::MinimumWithinStep) << geo::irc_stop_name(path->stop);
    EXPECT_LE(evaluations, kPreFarPole + (kPreFarPole / 10));
    // Most later spheres in one: 21% with the far pole, 91% before it.
    EXPECT_GE(10 * first, 8 * static_cast<int32_t>(points.size() - 2));
  }
}

// #1294: the long valley at every default. Its force falls under max_force
// at point 92, 2.615 amu^1/2 bohr short of the minimum and 5.8e-4 above it,
// where the Minimum stop ended the walk until #1294 (the Cope's: 2.63, 26
// steps, +0.227 kcal/mol). A minimum is now a force under max_force and |g|
// halved over the last step, so each side runs on, its energy falling at
// every point, and stops Minimum within a step of the minimum (118 points,
// 0.082 from it).
TEST(GeometryIrc, SoftLongValleyRunsPastItsForceToItsMinimum) {
  geo::IrcOptions options;
  options.max_points = 400;
  options.project_rigid_body = false;
  const double sqrt_m = std::sqrt(kHydrogenMass);
  const double theta = 11.0 / 30.0;
  const std::array<double, 6> minimum{30.0 * std::sin(theta),
                                      30.0 * (1.0 - std::cos(theta)),
                                      0.0,
                                      0.05 * 60.5,
                                      0.02 * 60.5,
                                      0.01 * 60.5};
  const auto miss = [&](std::span<const double> x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
      sum += (x[i] - minimum[i]) * (x[i] - minimum[i]);
    }
    return sqrt_m * std::sqrt(sum);
  };
  const std::array<double, 36> hessian = long_valley_hessian();
  for (const double side : {1.0, -1.0}) {
    const std::array<double, 6> direction{side, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto path = geo::irc(two_hydrogens(), hessian, direction, long_valley, options);
    ASSERT_TRUE(path.has_value()) << path.error().message;
    const auto &points = path->points;
    std::size_t under = 0;
    while (under < points.size() && !(under > 0 && points[under].max_force <= options.max_force)) {
      ++under;
    }
    ASSERT_LT(under, points.size());
    std::vector<double> mirrored(points.back().x);
    mirrored[0] *= side;
    std::vector<double> at_under(points[under].x);
    at_under[0] *= side;
    std::cout << std::format("[ irc ] long valley, default max_force, side {:+.0f}: {} points, "
                             "{}, force under max_force at point {} ({:.3f} from the minimum), "
                             "{:.4f} from it at the end, E {:.3e} above it\n",
                             side, points.size() - 1, geo::irc_stop_name(path->stop), under,
                             miss(at_under), miss(mirrored), points.back().energy);
    EXPECT_EQ(path->stop, geo::IrcStop::Minimum) << geo::irc_stop_name(path->stop);
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
      EXPECT_LT(points[k + 1].energy, points[k].energy) << k;
    }
    EXPECT_GT(points.size() - 1, under);
    EXPECT_GT(miss(at_under), 10.0 * options.step_size);
    EXPECT_LE(miss(mirrored), options.step_size);
  }
}

// E = -x^2 + 3 y^2 / 2 + z^2: a quadratic saddle whose path is the x axis.
// The model is the surface, so every point takes one evaluation, stays on the
// axis, and lies k steps out; with no minimum the walk runs to max_points.
TEST(GeometryIrc, QuadraticSaddleTakesOneEvaluationPerPointAlongItsAxis) {
  const auto saddle = [](std::span<const double> x,
                         std::span<double> gradient) -> fb::Result<double> {
    gradient[0] = -2.0 * x[0];
    gradient[1] = 3.0 * x[1];
    gradient[2] = 2.0 * x[2];
    return -(x[0] * x[0]) + (1.5 * x[1] * x[1]) + (x[2] * x[2]);
  };
  const std::array<double, 9> hessian{-2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0};
  geo::IrcOptions options;
  options.step_size = 0.2;
  options.max_points = 8;
  options.project_rigid_body = false;
  const std::array<double, 3> direction{-3.0, 0.0, 0.0};
  std::size_t kept = 0;
  const auto path = geo::irc(one_hydrogen(), hessian, direction, saddle, options,
                             [&](const geo::IrcPoint &) -> fb::Status {
                               ++kept;
                               return {};
                             });
  ASSERT_TRUE(path.has_value()) << path.error().message;
  EXPECT_EQ(path->stop, geo::IrcStop::MaxPoints);
  ASSERT_EQ(path->points.size(), 9U);
  EXPECT_EQ(kept, 9U);
  for (std::size_t k = 1; k < path->points.size(); ++k) {
    const geo::IrcPoint &point = path->points[k];
    EXPECT_EQ(point.sphere_iterations, 1) << k;
    EXPECT_NEAR(point.x[0], -static_cast<double>(k) * 0.2 / std::sqrt(kHydrogenMass), 1e-12);
    EXPECT_NEAR(point.x[1], 0.0, 1e-14);
    EXPECT_NEAR(point.x[2], 0.0, 1e-14);
  }
}

// Two hydrogens on a skew axis bound by E = ((r - 2)^2 - 1)^2: a maximum at
// r = 2, the TS, and minima at r = 1 and 3. Under the projection the walk
// keeps the centre of mass and the axis to 1e-12 at every point, and the two
// sides reach r = 1 and r = 3 within a step.
TEST(GeometryIrc, TwoPointsKeepTheirCentreAndAxis) {
  const std::array<double, 3> e{0.48, -0.6, 0.64}; // unit
  const std::array<double, 3> centre{0.3, -0.2, 1.1};
  std::vector<fb::Atom> atoms(2);
  for (std::size_t a = 0; a < 2; ++a) {
    const double sign = a == 0 ? -1.0 : 1.0;
    atoms[a] = fb::Atom{{centre[0] + sign * e[0], centre[1] + sign * e[1], centre[2] + sign * e[2]},
                        fb::Element::H};
  }
  const auto bond = [](std::span<const double> x,
                       std::span<double> gradient) -> fb::Result<double> {
    std::array<double, 3> d{x[3] - x[0], x[4] - x[1], x[5] - x[2]};
    const double r = std::hypot(d[0], d[1], d[2]);
    const double u = ((r - 2.0) * (r - 2.0)) - 1.0;
    const double de_dr = 4.0 * u * (r - 2.0);
    for (int32_t k = 0; k < 3; ++k) {
      gradient[3 + k] = de_dr * d[k] / r;
      gradient[k] = -gradient[3 + k];
    }
    return u * u;
  };
  // d2E/dr2 = -4 at r = 2: -4 [[e e^T, -e e^T], [-e e^T, e e^T]].
  std::array<double, 36> hessian{};
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      const double sign = (i < 3) == (j < 3) ? 1.0 : -1.0;
      hessian[i * 6 + j] = -4.0 * sign * e[i % 3] * e[j % 3];
    }
  }
  geo::IrcOptions options;
  options.step_size = 0.05;
  options.max_points = 100;
  options.max_force = 1e-8;
  options.sphere_gradient = 1e-10;
  options.sphere_displacement = 1e-8;
  for (const double side : {1.0, -1.0}) {
    // Forward stretches the bond.
    const std::array<double, 6> direction{-side * e[0], -side * e[1], -side * e[2],
                                          side * e[0],  side * e[1],  side * e[2]};
    const auto path = geo::irc(atoms, hessian, direction, bond, options);
    ASSERT_TRUE(path.has_value()) << path.error().message;
    for (const geo::IrcPoint &point : path->points) {
      std::array<double, 3> d{};
      for (int32_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(0.5 * (point.x[k] + point.x[3 + k]), centre[k], 1e-12);
        d[k] = point.x[3 + k] - point.x[k];
      }
      const double r = std::hypot(d[0], d[1], d[2]);
      for (int32_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(d[k] / r, e[k], 1e-12);
      }
    }
    const auto &end = path->points.back().x;
    const double r = std::hypot(end[3] - end[0], end[4] - end[1], end[5] - end[2]);
    // One step s mass-weighted moves r by s / sqrt(m / 2).
    EXPECT_NEAR(r, 2.0 + side, options.step_size / std::sqrt(0.5 * kHydrogenMass));
    EXPECT_EQ(side > 0.0, path->points[1].x[3] - path->points[1].x[0] > 2.0 * e[0]);
  }
}

// A start at a minimum: the sphere's minimum is the start itself, behind
// the pivot, and the walk stops with the start alone. A sphere given one
// evaluation and thresholds it cannot meet stops unconverged. And a callback's
// error, and a non-finite energy, end it.
TEST(GeometryIrc, StopsAndCallbackErrors) {
  const auto bowl = [](std::span<const double> x,
                       std::span<double> gradient) -> fb::Result<double> {
    for (std::size_t i = 0; i < 3; ++i) {
      gradient[i] = x[i];
    }
    return 0.5 * ((x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]));
  };
  const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> direction{0.0, 1.0, 0.0};
  geo::IrcOptions options;
  options.project_rigid_body = false;
  const auto rose = geo::irc(one_hydrogen(), identity, direction, bowl, options);
  ASSERT_TRUE(rose.has_value()) << rose.error().message;
  EXPECT_EQ(rose->stop, geo::IrcStop::EnergyRose);
  EXPECT_EQ(rose->points.size(), 1U);

  options.max_sphere_iterations = 1;
  options.sphere_gradient = 1e-300;
  options.sphere_displacement = 1e-300;
  const auto unconverged =
      geo::irc(one_hydrogen(), kValleyHessian, std::array{1.0, 0.0, 0.0}, valley, options);
  ASSERT_TRUE(unconverged.has_value()) << unconverged.error().message;
  EXPECT_EQ(unconverged->stop, geo::IrcStop::SphereNotConverged);
  EXPECT_EQ(unconverged->points.size(), 1U);

  const auto failing = [](std::span<const double>, std::span<double>) -> fb::Result<double> {
    return fb::err_numerical("the SCF did not converge");
  };
  const auto failed = geo::irc(one_hydrogen(), identity, direction, failing, options);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().message, "the SCF did not converge");

  const auto nan = [](std::span<const double>, std::span<double> g) -> fb::Result<double> {
    std::ranges::fill(g, 0.0);
    return std::numeric_limits<double>::quiet_NaN();
  };
  const auto not_finite = geo::irc(one_hydrogen(), identity, direction, nan, options);
  ASSERT_FALSE(not_finite.has_value());
  EXPECT_EQ(not_finite.error().kind, fb::ErrorKind::Numerical);
}

TEST(GeometryIrc, Refusals) {
  const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> direction{1.0, 0.0, 0.0};
  const auto expect_config = [](const fb::Result<geo::IrcPath> &out, std::string_view needle) {
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig) << out.error().message;
    EXPECT_NE(out.error().message.find(needle), std::string::npos) << out.error().message;
  };
  geo::IrcOptions free;
  free.project_rigid_body = false;
  expect_config(geo::irc({}, identity, direction, valley, free), "at least one atom");
  expect_config(geo::irc(one_hydrogen(), std::span(identity).first(8), direction, valley, free),
                "3N x 3N");
  expect_config(geo::irc(one_hydrogen(), identity, std::array{1.0, 0.0}, valley, free),
                "3N direction");
  for (const auto &tweak : std::vector<std::function<void(geo::IrcOptions &)>>{
           [](geo::IrcOptions &o) { o.step_size = 0.0; },
           [](geo::IrcOptions &o) { o.max_force = -1.0; },
           [](geo::IrcOptions &o) { o.sphere_gradient = 0.0; },
           [](geo::IrcOptions &o) { o.sphere_displacement = 0.0; }}) {
    geo::IrcOptions options = free;
    tweak(options);
    expect_config(geo::irc(one_hydrogen(), identity, direction, valley, options), "positive");
  }
  for (const auto &tweak : std::vector<std::function<void(geo::IrcOptions &)>>{
           [](geo::IrcOptions &o) { o.max_points = 0; },
           [](geo::IrcOptions &o) { o.max_sphere_iterations = 0; }}) {
    geo::IrcOptions options = free;
    tweak(options);
    expect_config(geo::irc(one_hydrogen(), identity, direction, valley, options), "at least 1");
  }
  std::array<double, 9> nan_hessian = identity;
  nan_hessian[4] = std::numeric_limits<double>::infinity();
  expect_config(geo::irc(one_hydrogen(), nan_hessian, direction, valley, free), "not finite");
  std::vector<fb::Atom> ghost = one_hydrogen();
  ghost[0].element = fb::Element{0};
  expect_config(geo::irc(ghost, identity, direction, valley, free), "Z = 0");
  // One atom has nothing but rigid motions, so every direction is one.
  expect_config(geo::irc(one_hydrogen(), identity, direction, valley, geo::IrcOptions{}),
                "rigid-body motion");
}

} // namespace
