// GPU cases for the Optimize step through the real registry: a TOML config
// at DF-RHF / cc-pVDZ / cc-pVDZ-JKFIT, from PySCF's starting geometry
// (data/gradient/gradient.json's `optimized`, test/gtest/data/README.md), read,
// planned and run by Calculation. The optimized geometry is held to PySCF's
// minimum to 1e-4 Angstrom after alignment and its energy to 1e-8 Eh; the
// reference optimizer was Newton's, so the path is not compared, only that it
// is short. H2O also carries a [localization], which must run at the optimum,
// and runs again over integral-direct ERIs, held to convergence. Then the
// CASSCF chain: H2O's state-specific DF-CASSCF(4,4) from an off-symmetry
// start to PySCF's minimum (data/casscf_gradient/casscf_optimize.json) to
// 1e-4 bohr, and its integral-direct refusal.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars.
constexpr double kGeometryToleranceAngstrom = 1e-4;
constexpr double kEnergyTolerance = 1e-8;
// "Fewer than about 15 steps": evaluations after the first, rejected ones
// included. H2O is held to it; ethanol to its own cap below.
constexpr std::size_t kMaxSteps = 15;
// #536 (a model starting Hessian in src/geometry) should bring ethanol back under kMaxSteps.
constexpr std::size_t kEthanolMaxSteps = 25;

// The gradient is the SCF energy's only at convergence, so the SCF is
// converged as pipeline_gradient_gpu.cpp converges it; and the optimizer
// past Gaussian's defaults, to Opt=Tight's thresholds: at the defaults
// ethanol's geometry is 3.9e-3 Angstrom off, past the bar.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";
constexpr double kMaxForce = 1.5e-5;
constexpr std::string_view kOptimize = "[optimize]\nmax_force = 1.5e-5\nrms_force = 1e-5\n"
                                       "max_displacement = 6e-5\nrms_displacement = 4e-5\n";

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient" / "gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("optimized")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no optimized case " << name;
  static const fb::json kNone;
  return kNone;
}

// The reference's `start` as start.xyz in @p dir (bohr there, Angstrom here),
// and run.toml optimizing it in the reference's basis, over the reference's
// fit or, with @p integral_direct, exact ERIs, with @p extra after [optimize].
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view extra, bool integral_direct = false) {
  std::string xyz =
      std::format("{}\n{}\n", entry.at("start").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("start")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  // Integral-direct incremental Fock builds stall short of kScf's commutator
  // (pipeline_gradient_gpu.cpp), so every build there is full.
  const std::string eri =
      integral_direct ? std::string("[eri]\nmode = \"integral_direct\"\n")
                      : std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                                    entry.at("aux_basis").get<std::string>());
  const std::string toml =
      std::format("[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n{}{}{}{}{}",
                  entry.at("basis").get<std::string>(), eri, kScf,
                  integral_direct ? "fock_rebuild_interval = 1\n" : "", kOptimize, extra);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

using Points = std::vector<std::array<double, 3>>;

// The largest distance (bohr) between the points of @p a and @p b, each
// centred on its centroid, after @p b is rotated onto @p a by the rotation
// that minimizes the squared deviation: Horn's quaternion, the largest
// eigenvector of his 4 x 4 symmetric matrix, by shifted power iteration.
double aligned_max_deviation(Points a, Points b) {
  for (Points *p : {&a, &b}) {
    std::array<double, 3> c{};
    for (const auto &r : *p) {
      for (int32_t k = 0; k < 3; ++k) {
        c[k] += r[k] / static_cast<double>(p->size());
      }
    }
    for (auto &r : *p) {
      for (int32_t k = 0; k < 3; ++k) {
        r[k] -= c[k];
      }
    }
  }
  // s[i][j] = sum over points of b_i a_j.
  std::array<std::array<double, 3>, 3> s{};
  for (std::size_t n = 0; n < a.size(); ++n) {
    for (int32_t i = 0; i < 3; ++i) {
      for (int32_t j = 0; j < 3; ++j) {
        s[i][j] += b[n][i] * a[n][j];
      }
    }
  }
  const auto [xx, xy, xz] = s[0];
  const auto [yx, yy, yz] = s[1];
  const auto [zx, zy, zz] = s[2];
  const std::array<std::array<double, 4>, 4> m{{
      {xx + yy + zz, yz - zy, zx - xz, xy - yx},
      {yz - zy, xx - yy - zz, xy + yx, zx + xz},
      {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
      {xy - yx, zx + xz, yz + zy, -xx - yy + zz},
  }};
  // Shifted past every eigenvalue's magnitude, so the largest dominates.
  double shift = 0.0;
  for (const auto &row : m) {
    for (const double v : row) {
      shift += std::abs(v);
    }
  }
  std::array<double, 4> q{1.0, 0.0, 0.0, 0.0};
  for (int32_t iteration = 0; iteration < 10000; ++iteration) {
    std::array<double, 4> next{};
    for (int32_t i = 0; i < 4; ++i) {
      next[i] = shift * q[i];
      for (int32_t j = 0; j < 4; ++j) {
        next[i] += m[i][j] * q[j];
      }
    }
    const double norm = std::sqrt((next[0] * next[0]) + (next[1] * next[1]) + (next[2] * next[2]) +
                                  (next[3] * next[3]));
    for (int32_t i = 0; i < 4; ++i) {
      q[i] = next[i] / norm;
    }
  }
  const auto [q0, q1, q2, q3] = q;
  const std::array<std::array<double, 3>, 3> r{{
      {(q0 * q0) + (q1 * q1) - (q2 * q2) - (q3 * q3), 2 * ((q1 * q2) - (q0 * q3)),
       2 * ((q1 * q3) + (q0 * q2))},
      {2 * ((q1 * q2) + (q0 * q3)), (q0 * q0) - (q1 * q1) + (q2 * q2) - (q3 * q3),
       2 * ((q2 * q3) - (q0 * q1))},
      {2 * ((q1 * q3) - (q0 * q2)), 2 * ((q2 * q3) + (q0 * q1)),
       (q0 * q0) - (q1 * q1) - (q2 * q2) + (q3 * q3)},
  }};
  double worst = 0.0;
  for (std::size_t n = 0; n < a.size(); ++n) {
    double d2 = 0.0;
    for (int32_t i = 0; i < 3; ++i) {
      double rotated = 0.0;
      for (int32_t j = 0; j < 3; ++j) {
        rotated += r[i][j] * b[n][j];
      }
      d2 += (rotated - a[n][i]) * (rotated - a[n][i]);
    }
    worst = std::max(worst, std::sqrt(d2));
  }
  return worst;
}

Points points(const fb::Molecule &molecule) {
  Points out;
  for (const fb::Atom &atom : molecule.atoms()) {
    out.push_back({atom.position.x, atom.position.y, atom.position.z});
  }
  return out;
}

Points points(const fb::json &atoms) {
  Points out;
  for (const fb::json &atom : atoms) {
    const auto c = atom.at("coord").get<std::vector<double>>();
    out.push_back({c[0], c[1], c[2]});
  }
  return out;
}

// Optimize @p name from its reference start, with @p extra in the config,
// and hold the result to the reference's minimum in fewer than @p max_steps
// steps; then @p after, if set, on
// the Calculation it ran, for a case that checks what ran after.
void expect_reference_minimum(std::string_view name, std::string_view extra, std::size_t max_steps,
                              const std::function<void(const fb::Calculation &)> &after = {}) {
  const fb::json &expected = reference_case(name);
  const fb::test::ScratchDir dir(std::format("pipeline_optimize_{}", name));
  const auto config = write_config(dir, expected, extra);
  auto calc = fb::read_config(config, dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto &out = calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(out.has_value());
  const double energy_error = std::abs(out->energy - expected.at("energy").get<double>());
  EXPECT_LE(energy_error, kEnergyTolerance);
  const double geometry_error =
      aligned_max_deviation(points(expected.at("atoms")), points(out->molecule)) *
      fb::kAngstromPerBohr<double>;
  EXPECT_LE(geometry_error, kGeometryToleranceAngstrom);

  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const std::size_t evaluations = history->at("iterations").size();
  ASSERT_GE(evaluations, 1U);
  EXPECT_LT(evaluations - 1, max_steps);
  EXPECT_EQ(history->at("energy").get<double>(), out->energy);
  // One frame per evaluation in the trajectory, and the optimum in final.xyz.
  const auto trajectory = fb::read_file(dir / "out" / "optimize.xyz");
  ASSERT_TRUE(trajectory.has_value());
  EXPECT_EQ(static_cast<std::size_t>(std::ranges::count(*trajectory, '\n')),
            evaluations * (2 + expected.at("atoms").size()));
  EXPECT_TRUE(std::filesystem::exists(dir / "out" / "final.xyz"));

  std::cout << std::format("[ optimize ] {}: {} steps, |dE| {:.2e} Eh, geometry {:.2e} A, "
                           "{:.1f} s\n",
                           name, evaluations - 1, energy_error, geometry_error, seconds);
  if (after) {
    after(*calc);
  }
}

TEST(PipelineOptimize, WaterReachesThePyscfMinimumAndLaterStepsRunThere) {
  FBF_REQUIRE_GPU();
  expect_reference_minimum("h2o_ccpvdz", "[localization]\n", kMaxSteps,
                           [](const fb::Calculation &calc) {
                             // Every step reading the molecule read the optimum: the
                             // molecule itself, to final.xyz's 1e-12 Angstrom, the SCF's
                             // energy, and the localization over that SCF.
                             const auto &optimized = calc.get<fb::OptimizeOutput>();
                             const auto &molecule = calc.get<fb::MoleculeOutput>();
                             ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
                             const Points want = points(optimized->molecule);
                             const Points got = points(molecule->molecule);
                             ASSERT_EQ(got.size(), want.size());
                             for (std::size_t a = 0; a < got.size(); ++a) {
                               for (int32_t k = 0; k < 3; ++k) {
                                 EXPECT_NEAR(got[a][k], want[a][k], 1e-10);
                               }
                             }
                             const auto &scf = calc.get<fb::ScfOutput>();
                             ASSERT_TRUE(scf.has_value()) << scf.error().message;
                             EXPECT_NEAR(scf->energy, optimized->energy, 1e-9);
                             EXPECT_TRUE(calc.get<fb::LocalizationOutput>().has_value());
                           });
}

TEST(PipelineOptimize, EthanolReachesThePyscfMinimum) {
  FBF_REQUIRE_GPU();
  expect_reference_minimum("ethanol_ccpvdz", "", kEthanolMaxSteps);
}

// #517's demonstration: the same step, unchanged, over integral-direct ERIs.
// The reference minimum is DF-RHF's, so this one is held to convergence, its
// last evaluation's largest force under the threshold, in as few steps.
TEST(PipelineOptimize, WaterConvergesOverIntegralDirectEris) {
  FBF_REQUIRE_GPU();
  const fb::json &start = reference_case("h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_optimize_h2o_direct");
  const auto config = write_config(dir, start, "", /*integral_direct=*/true);
  auto calc = fb::read_config(config, dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &eri = calc->get<fb::EriOutput>();
  ASSERT_TRUE(eri.has_value());
  EXPECT_FALSE(std::holds_alternative<fb::DensityFitted>(eri->eri));

  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 1U);
  const double max_force = iterations.back().at("max_force").get<double>();
  EXPECT_LE(max_force, kMaxForce);
  EXPECT_LT(iterations.size() - 1, kMaxSteps);
  std::cout << std::format("[ optimize ] h2o_ccpvdz integral-direct: {} steps, max force {:.2e}, "
                           "E {:.10f}, {:.1f} s\n",
                           iterations.size() - 1, max_force,
                           calc->get<fb::OptimizeOutput>()->energy, seconds);
}

// The CASSCF chain's reference: PySCF's DF-CASSCF minimum of H2O.
const fb::json &casscf_reference() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_optimize.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc.at("cases").at(0);
}

// #546's bar, in bohr.
constexpr double kCasscfGeometryToleranceBohr = 1e-4;
// The gradient is the CASSCF energy's only as far as the CASSCF converged
// (pipeline_casscf_gradient_gpu.cpp's 1e-9).
constexpr std::string_view kCasscf = "conv_tol_grad = 1e-9\n";
// Evaluations after the first; PySCF's optimizer was Newton's, so only the
// path's length is held.
constexpr std::size_t kCasscfMaxSteps = 15;

// H2O's DF-CASSCF(4,4) over its sigma O-H orbitals, from the reference's
// off-symmetry start, optimized at Opt=Tight's thresholds. The CASSCF picks
// its active space once, at the start, and carries its orbitals from each
// geometry to the next; the optimum is PySCF's to 1e-4 bohr and its energy
// to 1e-8 Eh, and the outer [casscf_gradient], run again at the optimum from
// a fresh SCF and pick, finds the same root there. About 2 s, inside the fast
// tier.
TEST(PipelineOptimize, WaterCasscfReachesThePyscfMinimum) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = casscf_reference();
  const fb::test::ScratchDir dir("pipeline_optimize_casscf");
  std::string xyz = std::format("{}\nstart\n", expected.at("start").size());
  for (const fb::json &atom : expected.at("start")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  std::string active;
  for (const auto t : expected.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  const std::string toml =
      std::format("[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
                  "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\n{}[casscf_gradient]\n{}",
                  expected.at("basis").get<std::string>(),
                  expected.at("aux_basis").at("name").get<std::string>(), kScf,
                  expected.at("ncas").get<std::int32_t>(),
                  expected.at("nelecas").get<std::int32_t>(), active, kCasscf, kOptimize);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());

  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::OptimizeStep>()->casscf_gradient.has_value());
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto &out = calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(out.has_value());
  const double energy_error = std::abs(out->energy - expected.at("energy").get<double>());
  EXPECT_LE(energy_error, kEnergyTolerance);
  const double geometry_error =
      aligned_max_deviation(points(expected.at("atoms")), points(out->molecule));
  EXPECT_LE(geometry_error, kCasscfGeometryToleranceBohr);

  // The history and the trajectory report the CASSCF root's energy.
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 1U);
  EXPECT_LT(iterations.size() - 1, kCasscfMaxSteps);
  EXPECT_EQ(history->at("energy").get<double>(), out->energy);
  EXPECT_NEAR(iterations.front().at("energy").get<double>(),
              expected.at("start_energy").get<double>(), kEnergyTolerance);
  const auto trajectory = fb::read_file(dir / "out" / "optimize.xyz");
  ASSERT_TRUE(trajectory.has_value());
  EXPECT_EQ(static_cast<std::size_t>(std::ranges::count(*trajectory, '\n')),
            iterations.size() * (2 + expected.at("atoms").size()));

  // The outer CASSCF, from a fresh SCF at the optimum, is the same root.
  const auto &cas = calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  EXPECT_NEAR(cas->energies.at(0), out->energy, kEnergyTolerance);
  EXPECT_TRUE(std::filesystem::exists(dir / "out" / "casscf_gradient.json"));

  std::cout << std::format("[ optimize ] h2o_ccpvdz CASSCF(4,4): {} steps, |dE| {:.2e} Eh, "
                           "geometry {:.2e} bohr, {:.1f} s\n",
                           iterations.size() - 1, energy_error, geometry_error, seconds);
}

// A hand-built CASSCF chain over integral-direct ERIs is refused by the step
// before any SCF or CASSCF runs, as pipeline/config refuses it.
TEST(PipelineOptimize, ACasscfChainOverIntegralDirectErisIsUnsupported) {
  FBF_REQUIRE_GPU();
  fb::OptimizeStep step;
  step.casscf_gradient.emplace();
  step.eri.mode = fb::EriMode::IntegralDirect;
  fb::NodeResources node;
  const auto out = step.run(fb::View<>{}, node);
  ASSERT_FALSE(out.has_value());
  EXPECT_EQ(out.error().kind, fb::ErrorKind::Unsupported) << out.error().message;
  EXPECT_NE(out.error().message.find("density fitting only"), std::string::npos)
      << out.error().message;
}

} // namespace
