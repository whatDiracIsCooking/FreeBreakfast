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
// 1e-4 bohr, and its integral-direct refusal. Then the NEVPT2 chain: the
// same water's SC-NEVPT2 over that DF-CASSCF(4,4), from the skew start
// (data/nevpt2_gradient/nevpt2_gradient.json) to a stationary point whose
// energy is PySCF's DF-NEVPT2 single point there
// (data/nevpt2_gradient/nevpt2_optimize.json) to 1e-9, and a refusal of the
// gradient step ending the optimization with its message. Then a
// state-averaged root: skew formaldehyde's S1 over its DF-SA2-CASSCF(4,3),
// followed to PySCF's pyramidal minimum
// (data/casscf_gradient/sa_casscf_optimize.json) within bars derived from
// that file's force and least curvature. Last the QD
// chain: formaldehyde's SC-QD-NEVPT2 state 1 over its DF-SA2-CASSCF(4,3)
// (data/qd_nevpt2_gradient/h2co_ccpvdz_sa2.json), followed by overlap to a
// stationary point that a fresh [nevpt2_gradient] confirms, and skew
// formaldehyde's state 2 over its DF-SA3-CASSCF(4,3)
// (data/qd_nevpt2_gradient/h2co_skew_ccpvdz_sa3.json), where the followed
// state's overlap falls below 1 along the path.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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

// --- A state-averaged CASSCF root -------------------------------------------

// The SA root chain's reference: PySCF's DF-SA2-CASSCF(4,3) minimum of
// formaldehyde's root 1 from the skew start, with the reference's own last
// force (max_force) and the least curvature of its projected Hessian (k_min).
const fb::json &sa_casscf_reference() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "sa_casscf_optimize.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc.at("cases").at(0);
}

// @p b's points mirrored through the yz plane.
Points mirrored(Points b) {
  for (auto &r : b) {
    r[0] = -r[0];
  }
  return b;
}

// Formaldehyde's S1 (n -> pi*), root 1 of its DF-SA2-CASSCF(4,3), cc-pVDZ /
// cc-pVDZ-JKFIT, optimized from the pyramidalized, twisted start of
// data/casscf_nac/'s h2co_skew_ccpvdz_sa3 at Opt=Tight's thresholds, the
// root followed from each geometry to the next (#772). S1's minimum is
// pyramidal; a planar start would keep C2v and stop at the planar saddle,
// where PySCF would agree with it, so the start is not planar, and the
// minimum comes as two mirror images, so the alignment allows a reflection.
// The bars are derived from the fixture: the geometry within
// delta = (F_test + F_ref) / k_min, F_test this test's max_force and F_ref the
// reference's last force, and the energy within (F_test + F_ref) delta, the
// first-order bound for that displacement. Every evaluation after the first
// logs a matched overlap at or above the floor, and the root followed is 1
// at every one. Fewer than 15 steps, a regression guard.
TEST(PipelineOptimize, SkewFormaldehydeSaRootOneReachesThePyscfMinimum) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = sa_casscf_reference();
  const auto state = expected.at("state").get<std::int32_t>();
  constexpr std::int32_t kFollowedRoot = 1;
  ASSERT_EQ(state, kFollowedRoot);
  const double f_ref = expected.at("max_force").get<double>();
  const double k_min = expected.at("k_min").get<double>();
  const double delta = (kMaxForce + f_ref) / k_min;
  // The issue's rule: over 1e-3 bohr, the force threshold is tightened, never
  // the bar.
  ASSERT_LE(delta, 1e-3) << "tighten kMaxForce";
  const double energy_bar = (kMaxForce + f_ref) * delta;

  // At the pyramidal optimum the outer RHF, run afresh after the
  // optimization, converges slowly (|[F,D]| 3.3e-10 at its 100th
  // iteration): kScf's bars, with more iterations.
  constexpr std::string_view kPyramidalScf = "[scf]\nenergy_tolerance = 1e-12\n"
                                             "commutator_tolerance = 1e-10\n"
                                             "max_iterations = 400\n";
  const fb::test::ScratchDir dir("pipeline_optimize_sa_casscf");
  std::string xyz = std::format("{}\nstart\n", expected.at("start").size());
  for (const fb::json &atom : expected.at("start")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\n{}[casscf_gradient]\nroot = {}\n{}",
      expected.at("basis").get<std::string>(),
      expected.at("aux_basis").at("name").get<std::string>(), kPyramidalScf,
      expected.at("ncas").get<std::int32_t>(), expected.at("nelecas").get<std::int32_t>(),
      expected.at("weights").size(), kCasscf, state, kOptimize);
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
  const Points want = points(expected.at("atoms"));
  const Points got = points(out->molecule);
  const double geometry_error =
      std::min(aligned_max_deviation(want, got), aligned_max_deviation(want, mirrored(got)));
  EXPECT_LE(geometry_error, delta);
  const double energy_error = std::abs(out->energy - expected.at("energy").get<double>());
  EXPECT_LE(energy_error, energy_bar);

  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 2U);
  // A regression guard, not a correctness check: 18 steps when it was set
  // (#772), the BFGS started from the identity Hessian (one step rejected on
  // the way down the 40-degree pyramidalization), which a model starting
  // Hessian (#536) should shorten.
  constexpr std::size_t kSaRootMaxSteps = 25;
  EXPECT_LT(iterations.size() - 1, kSaRootMaxSteps);
  EXPECT_EQ(history->at("energy").get<double>(), out->energy);
  EXPECT_TRUE(iterations.front().at("state_overlap").is_null());
  double worst_overlap = 1.0;
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    const fb::json &it = iterations[i];
    ASSERT_TRUE(it.contains("root") && it.contains("state_overlap")) << i;
    EXPECT_EQ(it.at("root").get<std::int32_t>(), kFollowedRoot) << i;
    if (i > 0) {
      const double overlap = it.at("state_overlap").get<double>();
      EXPECT_GE(overlap, fb::kStateOverlapFloor) << i;
      worst_overlap = std::min(worst_overlap, overlap);
    }
    std::cout << std::format(
        "[ optimize ] SA root evaluation {:>2}: E {:.10f}, max force {:.2e}, root {}, matched "
        "overlap {}\n",
        i, it.at("energy").get<double>(), it.at("max_force").get<double>(),
        it.at("root").get<std::int32_t>(),
        i == 0 ? std::string("-") : std::format("{:.10f}", it.at("state_overlap").get<double>()));
  }
  // Diagnostics, not bars: the start against PySCF's, and the outer CASSCF
  // at the optimum, from a fresh SCF and pick.
  const double start_error =
      std::abs(iterations.front().at("energy").get<double>() -
               expected.at("start_energies").at(static_cast<std::size_t>(state)).get<double>());
  const auto &cas = calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  std::cout << std::format(
      "[ optimize ] h2co_skew SA2-CASSCF(4,3) root {}: {} steps, geometry {:.2e} bohr (delta "
      "{:.2e}), |dE| {:.2e} Eh (bar {:.2e}), worst matched overlap {:.10f}; start |dE| {:.2e}, "
      "fresh outer root {} E - optimized {:.2e}; F_test {:.1e}, F_ref {:.2e}, k_min {:.6f}; "
      "{:.1f} s\n",
      state, iterations.size() - 1, geometry_error, delta, energy_error, energy_bar, worst_overlap,
      start_error, state, cas->energies.at(static_cast<std::size_t>(state)) - out->energy,
      kMaxForce, f_ref, k_min, seconds);
}

// The NEVPT2 chain's start, data/nevpt2_gradient/nevpt2_gradient.json's
// skew H2O, and its reference, PySCF's DF-NEVPT2 at this step's optimum.
const fb::json &nevpt2_start() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_gradient" /
                              "nevpt2_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == "h2o_skew_ccpvdz") {
      return entry;
    }
  }
  ADD_FAILURE() << "no case h2o_skew_ccpvdz in nevpt2_gradient.json";
  static const fb::json kNone;
  return kNone;
}

const fb::json &nevpt2_optimum() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_gradient" /
                              "nevpt2_optimize.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

// The issue's bar: the optimum's energy against PySCF's at that geometry.
constexpr double kNevpt2EnergyTolerance = 1e-9;
// The optimum against the geometry PySCF's single point was taken at. At a
// stationary point an offset d moves the energy by g . d + d^T H d / 2: the
// reference's own gradient there is under kMaxForce per component (its
// numerical gradient, in the file), so 1e-5 bohr on each of nine coordinates
// is at most 3 * 1.5e-5 * 1e-5 = 4.5e-10 plus a quadratic term under 1e-10,
// inside kNevpt2EnergyTolerance.
constexpr double kNevpt2GeometryToleranceBohr = 1e-5;
// As the CASSCF chain's.
constexpr std::size_t kNevpt2MaxSteps = 15;

// skew H2O's start as start.xyz in @p dir and run.toml optimizing its
// SC-NEVPT2 over DF-CASSCF(4,4) [2, 3, 5, 6] at Opt=Tight's thresholds, with
// @p casscf_extra in [casscf].
std::filesystem::path write_nevpt2_config(const fb::test::ScratchDir &dir,
                                          std::string_view casscf_extra) {
  const fb::json &entry = nevpt2_start();
  std::string xyz = std::format("{}\nstart\n", entry.at("atoms").size());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  std::string active;
  for (const auto t : entry.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\n{}{}[nevpt2_gradient]\n{}",
      entry.at("basis").get<std::string>(), entry.at("aux_basis").at("name").get<std::string>(),
      kScf, entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(), active,
      kCasscf, casscf_extra, kOptimize);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// skew H2O's SC-NEVPT2 over its DF-CASSCF(4,4), optimized from the skew
// start with the CASSCF chain's carried orbitals: it converges, its last
// evaluation's largest force under the threshold, to the geometry
// nevpt2_optimize.json records, where PySCF's DF-NEVPT2 single point is the
// optimized energy to 1e-9 and PySCF's numerical gradient is under the
// threshold too. The first evaluation is the start's PySCF energy
// (nevpt2_gradient.json) and the outer [nevpt2_gradient], rerun at the
// optimum from a fresh SCF and pick, the optimized energy. 13 steps in about
// 4 s on the 3080, inside the fast tier.
TEST(PipelineOptimize, WaterNevpt2ReachesAStationaryPointAtPyscfsEnergy) {
  FBF_REQUIRE_GPU();
  const fb::json &start = nevpt2_start();
  const fb::json &expected = nevpt2_optimum();
  const fb::test::ScratchDir dir("pipeline_optimize_nevpt2");
  auto calc = fb::read_config(write_nevpt2_config(dir, ""), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::OptimizeStep>()->nevpt2_gradient.has_value());
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(out.has_value());
  // The optimum, to round-trip, for the reference tool's input.
  std::string optimum;
  for (const fb::Atom &atom : out->molecule.atoms()) {
    optimum += std::format(" [{:.17g}, {:.17g}, {:.17g}]", atom.position.x, atom.position.y,
                           atom.position.z);
  }

  // Converged: the last evaluation's largest force under the threshold.
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 1U);
  const double max_force = iterations.back().at("max_force").get<double>();
  EXPECT_LE(max_force, kMaxForce);
  EXPECT_LT(iterations.size() - 1, kNevpt2MaxSteps);
  EXPECT_EQ(history->at("energy").get<double>(), out->energy);
  EXPECT_NEAR(iterations.front().at("energy").get<double>(),
              start.at("energies").at("sc").get<double>(), kNevpt2EnergyTolerance);

  // At PySCF's single point.
  const double geometry_error =
      aligned_max_deviation(points(expected.at("atoms")), points(out->molecule));
  EXPECT_LE(geometry_error, kNevpt2GeometryToleranceBohr);
  const double energy_error =
      std::abs(out->energy - expected.at("energies").at("sc").get<double>());
  EXPECT_LE(energy_error, kNevpt2EnergyTolerance);
  EXPECT_LE(expected.at("max_gradient").get<double>(), kMaxForce);

  // The outer [nevpt2_gradient], from a fresh SCF and pick at the optimum.
  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_NEAR(written->at("energy").get<double>(), out->energy, kNevpt2EnergyTolerance);

  std::cout << std::format("[ optimize ] h2o_skew_ccpvdz SC-NEVPT2 CAS(4,4): {} steps, max force "
                           "{:.2e}, |dE| {:.2e} Eh, geometry {:.2e} bohr, E {:.12f}, {:.1f} s\n"
                           "[ optimize ] optimum (bohr):{}\n",
                           iterations.size() - 1, max_force, energy_error, geometry_error,
                           out->energy, seconds, optimum);
}

// A gradient the step refuses at run ends the optimization with that
// refusal, not a step through it: the chain's CASSCF a state average, which
// [nevpt2_gradient] refuses once the first SCF and CASSCF have run (the
// config refuses it before; the step is copied out and edited by hand to
// reach the step's own refusal). Nothing is written. A degenerate SC pair or
// a PC rank change, met at any geometry, ends it the same way:
// geometry::optimize returns its callback's first error.
TEST(PipelineOptimize, ARefusedNevpt2GradientEndsTheOptimizationWithItsMessage) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_optimize_nevpt2_refused");
  auto calc = fb::read_config(write_nevpt2_config(dir, ""), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::OptimizeStep step = *calc->step<fb::OptimizeStep>();
  ASSERT_TRUE(step.nevpt2_gradient.has_value());
  step.casscf.nroots = 2;
  fb::NodeResources node;
  const auto out = step.run(fb::View<>{}, node);
  ASSERT_FALSE(out.has_value());
  EXPECT_EQ(out.error().kind, fb::ErrorKind::Unsupported) << out.error().message;
  EXPECT_EQ(out.error().message, "[nevpt2_gradient] differentiates a single-state CASSCF, and "
                                 "[casscf] has 2 roots: set [casscf] nroots = 1");
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "final.xyz"));
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "optimize.json"));
}

// Both CASSCF gradients set by hand is refused before anything runs, as
// pipeline/config refuses a config naming both, and so is the NEVPT2 chain
// over integral-direct ERIs.
TEST(PipelineOptimize, AHandBuiltNevpt2ChainIsRefusedBeforeItRuns) {
  FBF_REQUIRE_GPU();
  fb::NodeResources node;
  fb::OptimizeStep both;
  both.casscf_gradient.emplace();
  both.nevpt2_gradient.emplace();
  both.eri.mode = fb::EriMode::DensityFitting;
  const auto two = both.run(fb::View<>{}, node);
  ASSERT_FALSE(two.has_value());
  EXPECT_EQ(two.error().kind, fb::ErrorKind::InvalidConfig) << two.error().message;
  EXPECT_NE(two.error().message.find("[casscf_gradient] or of [nevpt2_gradient]"),
            std::string::npos)
      << two.error().message;

  fb::OptimizeStep direct;
  direct.nevpt2_gradient.emplace();
  direct.eri.mode = fb::EriMode::IntegralDirect;
  const auto refused = direct.run(fb::View<>{}, node);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Unsupported) << refused.error().message;
  EXPECT_NE(refused.error().message.find("reruns [nevpt2_gradient], which runs over density "
                                         "fitting only"),
            std::string::npos)
      << refused.error().message;
}

// --- The QD-NEVPT2 state ----------------------------------------------------

// A case of data/qd_nevpt2_gradient/: its geometry, basis and aux basis, its
// CAS and its state average, with its DF-SA-CASSCF at the generator's
// convergence (that directory's README.md).
fb::json qd_case(std::string_view name) {
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2_gradient" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return read.has_value() ? std::move(*read) : fb::json{};
}

// kOptimize's rms_force, beside kMaxForce: the optimizer's two force
// criteria, which a fresh gradient at the optimum is held to (#707).
constexpr double kRmsForce = 1e-5;
// As the NEVPT2 chain's.
constexpr std::size_t kQdMaxSteps = 15;

// What a QD optimization's test reads back: the steps taken, the worst
// matched overlap and smallest H_eff gap along the path, and the fresh
// single point's largest component, rms and 2-norm at the optimum.
struct QdPath {
  std::size_t steps{0};
  double worst_overlap{1.0};
  double smallest_gap{std::numeric_limits<double>::infinity()};
  double fresh_max{0.0};
  double fresh_rms{0.0};
  double fresh_norm{0.0};
};

// SC-QD-NEVPT2 state @p state of qd_case(@p name), optimized at Opt=Tight's
// thresholds from the case's start under @p scf, the outer [nevpt2_gradient]
// a fresh single point from a fresh SCF and pick at the optimum. Holds it to
// converging in fewer than @p max_steps, its last evaluation's largest force
// under max_force; every step's matched overlap, kept in optimize.json (null
// at the start, which has no last state), at or above kStateOverlapFloor;
// and the fresh gradient's largest component under max_force and its rms
// under rms_force, the optimizer's own criteria. Prints every evaluation's
// matched overlap and H_eff gap, and the summary.
QdPath optimize_qd_state(std::string_view name, std::int32_t state, std::string_view scf,
                         std::size_t max_steps) {
  QdPath path;
  const fb::json start = qd_case(name);
  const fb::test::ScratchDir dir(std::format("pipeline_optimize_qd_{}_{}", name, state));
  std::string xyz = std::format("{}\nstart\n", start.at("atoms").size());
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nconv_tol_grad = 1e-10\n"
      "conv_tol = 1e-13\nmax_cycle = 200\n"
      "[nevpt2_gradient]\nquasi_degenerate = true\nstate = {}\n{}",
      start.at("basis").get<std::string>(), start.at("aux_basis").get<std::string>(), scf,
      start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
      start.at("weights").size(), state, kOptimize);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return path;
  }
  EXPECT_TRUE(calc->step<fb::OptimizeStep>()->nevpt2_gradient.has_value() &&
              calc->step<fb::OptimizeStep>()->nevpt2_gradient->quasi_degenerate);
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return path;
  }
  const auto &out = calc->get<fb::OptimizeOutput>();
  EXPECT_TRUE(out.has_value());

  const auto history = fb::read_json(dir / "out" / "optimize.json");
  if (!history.has_value()) {
    ADD_FAILURE() << history.error().message;
    return path;
  }
  const fb::json &iterations = history->at("iterations");
  EXPECT_GE(iterations.size(), 2U);
  path.steps = iterations.size() - 1;
  const double max_force = iterations.back().at("max_force").get<double>();
  EXPECT_LE(max_force, kMaxForce);
  EXPECT_LT(path.steps, max_steps);
  EXPECT_TRUE(iterations.front().at("state_overlap").is_null());
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    const fb::json &it = iterations[i];
    EXPECT_TRUE(it.contains("state_overlap") && it.contains("heff_gap")) << i;
    path.smallest_gap = std::min(path.smallest_gap, it.at("heff_gap").get<double>());
    if (i > 0) {
      const double overlap = it.at("state_overlap").get<double>();
      EXPECT_GE(overlap, fb::kStateOverlapFloor) << i;
      path.worst_overlap = std::min(path.worst_overlap, overlap);
    }
    std::cout << std::format(
        "[ optimize ] {} QD state {} evaluation {:>2}: E {:.10f}, max "
        "force {:.2e}, matched overlap {}, H_eff gap {:.6f}\n",
        name, state, i, it.at("energy").get<double>(), it.at("max_force").get<double>(),
        i == 0 ? std::string("-") : std::format("{:.10f}", it.at("state_overlap").get<double>()),
        it.at("heff_gap").get<double>());
  }

  // The fresh single point at the optimum: the outer [nevpt2_gradient].
  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  if (!written.has_value()) {
    ADD_FAILURE() << written.error().message;
    return path;
  }
  EXPECT_TRUE(written->at("quasi_degenerate").get<bool>());
  EXPECT_EQ(written->at("state").get<std::int32_t>(), state);
  double fresh_sq = 0.0;
  std::size_t components = 0;
  for (const auto &row : written->at("total").get<std::vector<std::vector<double>>>()) {
    for (const double g : row) {
      path.fresh_max = std::max(path.fresh_max, std::abs(g));
      fresh_sq += g * g;
      ++components;
    }
  }
  EXPECT_EQ(components, 3 * start.at("atoms").size());
  path.fresh_rms = std::sqrt(fresh_sq / static_cast<double>(std::max<std::size_t>(components, 1)));
  path.fresh_norm = std::sqrt(fresh_sq);
  EXPECT_LE(path.fresh_max, kMaxForce);
  EXPECT_LE(path.fresh_rms, kRmsForce);

  std::cout << std::format(
      "[ optimize ] {} SC-QD-NEVPT2 state {}: {} steps, max force {:.2e}, E {:.12f}, worst "
      "matched overlap {:.10f}, smallest H_eff gap {:.6f}; fresh gradient max {:.2e} rms {:.2e} "
      "2-norm {:.2e} (max_force {:.1e}, rms_force {:.1e}), E_K {:.12f}; {:.1f} s\n",
      name, state, path.steps, max_force, out.has_value() ? out->energy : 0.0, path.worst_overlap,
      path.smallest_gap, path.fresh_max, path.fresh_rms, path.fresh_norm, kMaxForce, kRmsForce,
      written->at("energy").get<double>(), seconds);
  return path;
}

// QD state K = 1 (the n -> pi* state, not the lowest, so the state is
// followed by its identity rather than taken by index) of planar
// formaldehyde's SC-QD-NEVPT2 over its DF-SA2-CASSCF(4,3), h2co_ccpvdz_sa2,
// held to optimize_qd_state's bars: no crossing along the path. The point
// is planar (C2v), so its two roots are of different irreps and every
// overlap is 1 to rounding; the next case is one where they are not. 10
// steps in about 3 s on the 3080, inside the fast tier.
TEST(PipelineOptimize, FormaldehydeQdStateOneReachesAStationaryPoint) {
  FBF_REQUIRE_GPU();
  optimize_qd_state("h2co_ccpvdz_sa2", 1, kScf, kQdMaxSteps);
}

// The skew formaldehyde's start (h2co_skew_ccpvdz_sa3, C1, three roots
// averaged) takes steps to the stationary point of its SC-QD-NEVPT2 state
// K = 2, the highest, over which the state is followed (#768): the roots
// are of no symmetry, H_eff mixes them, and its eigenvector turns between
// steps. The overlap falls to 0.9932 at step 2 (0.9952, 0.9932, 0.9990 and
// 0.9991 over steps 1-4) and is back above 0.9999 from step 5 on, never near
// the floor; H_eff's gap goes 0.149 -> 0.082 at its smallest (step 14) and
// is 0.086 at the optimum, no crossing. Held to optimize_qd_state's bars,
// and to below 0.999 at some step, so the following is not trivial; that
// bar and the floor are fixed, never moved to make a run pass. 24 steps.
// The optimum's C-O is stretched to 1.67 Angstrom, where the fresh RHF
// oscillates: it converges in about 100 iterations with a DIIS history of
// 4, so the SCF is given that and 400 iterations; the bars are kScf's. About
// 30 s on the gfx1200, so in the slow tier.
TEST(PipelineOptimize, SkewFormaldehydeQdStateTwoIsFollowedThroughTurningOverlaps) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr std::string_view kStretchedScf = "[scf]\nenergy_tolerance = 1e-12\n"
                                             "commutator_tolerance = 1e-10\n"
                                             "max_iterations = 400\ndiis_history = 4\n";
  // Below which a step's state has turned: the overlap of a state the
  // symmetry keeps apart is 1 to 1e-10, as the planar case's.
  constexpr double kTurnedOverlap = 0.999;
  constexpr std::size_t kSkewMaxSteps = 30;
  const QdPath path = optimize_qd_state("h2co_skew_ccpvdz_sa3", 2, kStretchedScf, kSkewMaxSteps);
  EXPECT_LT(path.worst_overlap, kTurnedOverlap);
}

} // namespace
