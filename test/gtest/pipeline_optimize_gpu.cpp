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
// state's overlap falls below 1 along the path. Last the MECI: ethylene's
// S0/S1 crossing over its DF-SA2-CASSCF(2,2) from the twisted,
// pyramidalized start of data/casscf_meci/, a fresh single point at the
// result holding it to the thresholds, and the geometry and mean energy held
// to the fixture's within bars derived from it. Last the QD-NEVPT2 MECI: the
// same ethylene's SC- and PC-QD-NEVPT2 states 0 and 1 over its
// DF-SA3-CASSCF(2,2), from that fixture's SA2 MECI, held to #785's C1 (the
// fresh point's gap), C2 (seam stationarity by the numerical dH_eff of
// tools/qd_nevpt2_gradient) and C3 (the third state's margin).

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;
import free_breakfast.nevpt2;
import free_breakfast.tools.qd_nevpt2_gradient;

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
// reach the step's own refusal). The refusal is met at the walk's first
// evaluation, so the history is its message as "error" over no entries, and
// there is no final.xyz. A degenerate SC pair or a PC rank change, met at
// any geometry, ends it the same way: geometry::optimize returns its
// callback's first error.
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
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  EXPECT_EQ(history->at("error").get<std::string>(), out.error().message);
  EXPECT_FALSE(history->contains("energy"));
  EXPECT_TRUE(history->at("iterations").empty());
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

// --- The MECI ----------------------------------------------------------------

// data/casscf_meci/casscf_meci.json: the S0/S1 MECI of ethylene's
// DF-SA2-CASSCF(2,2), cc-pVDZ / cc-pVDZ-JKFIT, its start, and the
// quantities #774's bars are derived from (that directory's README.md).
const fb::json &meci_reference() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_meci" /
                              "casscf_meci.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

// The issue's gap threshold (hartree); the four others are kOptimize's,
// Opt=Tight's.
constexpr double kMeciGap = 1e-7;

// An n_atom x 3 array of rows, flat.
std::vector<double> flat_rows(const fb::json &rows) {
  std::vector<double> out;
  for (const auto &row : rows.get<std::vector<std::vector<double>>>()) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

// The two roots' energies, gradients and coupling at one geometry.
struct PairPoint {
  std::array<double, 2> energies{};
  std::vector<double> g0;
  std::vector<double> g1;
  std::vector<double> h;
};

// One mcscf::df_intersection of roots 0 and 1 over @p calc's own products,
// after it ran: its [casscf], from its own RHF orbitals and active-space
// pick, never an optimization's carried ones. The Solution is built as
// pipeline/casscf_gradient builds it.
PairPoint fresh_intersection(const fb::Calculation &calc, fb::DeviceResources &res) {
  PairPoint point;
  const auto &cas = *calc.get<fb::CasscfOutput>();
  const fb::BasisSet &basis = calc.get<fb::BasisSetOutput>()->basis;
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t e = 0; e < n2; ++e) {
      gamma[e] += cas.weights[r] * cas.rdm1[(r * n2) + e];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  const auto *const fit = std::get_if<fb::DensityFitted>(&calc.get<fb::EriOutput>()->eri);
  if (fit == nullptr) {
    ADD_FAILURE() << "the ERIs are not density-fitted";
    return point;
  }
  auto out = fb::mcscf::df_intersection(res, calc.get<fb::MoleculeOutput>()->molecule, basis,
                                        calc.get<fb::PrimitivePairClassesOutput>()->classes,
                                        fit->aux, calc.get<fb::CoreHamiltonianOutput>()->h, fit->b,
                                        fit->l.data(), solution, cas.ncore, cas.ncas, 0, 1);
  if (!out.has_value()) {
    ADD_FAILURE() << out.error().message;
    return point;
  }
  point.energies = {cas.energies.at(0), cas.energies.at(1)};
  point.g0 = std::move(out->g_i);
  point.g1 = std::move(out->g_j);
  point.h = std::move(out->h_ij);
  return point;
}

// The largest |component| of P (g0 + g1) / 2 at @p atoms (bohr), P the
// projector off the three translations and three rotations about the
// centroid and then the branching plane {x = g1 - g0, y = h}, each
// orthonormalized in that order after them: #771's projector, written here
// apart from it, the fixture's definition of the projected force.
double max_projected_force(const Points &atoms, const PairPoint &point) {
  const std::size_t n = 3 * atoms.size();
  std::array<double, 3> centroid{};
  for (const auto &r : atoms) {
    for (std::size_t k = 0; k < 3; ++k) {
      centroid[k] += r[k] / static_cast<double>(atoms.size());
    }
  }
  std::vector<std::vector<double>> directions;
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<double> t(n, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      t[(3 * a) + k] = 1.0;
    }
    directions.push_back(std::move(t));
  }
  for (std::size_t k = 0; k < 3; ++k) {
    // e_k x (r - centroid).
    std::vector<double> rot(n, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      std::array<double, 3> d{};
      for (std::size_t m = 0; m < 3; ++m) {
        d[m] = atoms[a][m] - centroid[m];
      }
      const std::size_t k1 = (k + 1) % 3;
      const std::size_t k2 = (k + 2) % 3;
      rot[(3 * a) + k2] = d[k1];
      rot[(3 * a) + k1] = -d[k2];
    }
    directions.push_back(std::move(rot));
  }
  std::vector<double> x(n);
  for (std::size_t c = 0; c < n; ++c) {
    x[c] = point.g1[c] - point.g0[c];
  }
  directions.push_back(std::move(x));
  directions.push_back(point.h);
  std::vector<std::vector<double>> basis;
  for (std::vector<double> v : directions) {
    const double before = std::sqrt(std::ranges::fold_left(
        v, 0.0, [](double acc, double value) { return acc + (value * value); }));
    for (int32_t pass = 0; pass < 2; ++pass) {
      for (const auto &u : basis) {
        double dot = 0.0;
        for (std::size_t c = 0; c < n; ++c) {
          dot += u[c] * v[c];
        }
        for (std::size_t c = 0; c < n; ++c) {
          v[c] -= dot * u[c];
        }
      }
    }
    const double norm = std::sqrt(std::ranges::fold_left(
        v, 0.0, [](double acc, double value) { return acc + (value * value); }));
    if (norm > 1e-8 * before) {
      for (double &value : v) {
        value /= norm;
      }
      basis.push_back(std::move(v));
    }
  }
  std::vector<double> g(n);
  for (std::size_t c = 0; c < n; ++c) {
    g[c] = 0.5 * (point.g0[c] + point.g1[c]);
  }
  for (const auto &u : basis) {
    double dot = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      dot += u[c] * g[c];
    }
    for (std::size_t c = 0; c < n; ++c) {
      g[c] -= dot * u[c];
    }
  }
  double largest = 0.0;
  for (const double value : g) {
    largest = std::max(largest, std::abs(value));
  }
  return largest;
}

// The gap's smaller slope in the branching plane (hartree / bohr): twice the
// smaller singular value of the 2 x 3N matrix [x / 2; h], x = g1 - g0 and h
// the coupling, from the 2 x 2 Gram matrix. The fixture's invariant (its
// README's 0.1413), where x_norm and h_norm alone are not reproducible.
double smaller_gap_slope(std::span<const double> g0, std::span<const double> g1,
                         std::span<const double> h) {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  for (std::size_t k = 0; k < g0.size(); ++k) {
    const double half_x = 0.5 * (g1[k] - g0[k]);
    a += half_x * half_x;
    b += half_x * h[k];
    c += h[k] * h[k];
  }
  const double smaller = (0.5 * (a + c)) - std::hypot(0.5 * (a - c), b);
  return 2.0 * std::sqrt(std::max(0.0, smaller));
}

// The fixture's atoms as an XYZ (Angstrom) at @p path.
void write_xyz(const std::filesystem::path &path, const fb::json &atoms) {
  std::string xyz = std::format("{}\nstart\n", atoms.size());
  for (const fb::json &atom : atoms) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(path, xyz).has_value());
}

// Ethylene's S0/S1 MECI, roots 0 and 1 of its DF-SA2-CASSCF(2,2), cc-pVDZ /
// cc-pVDZ-JKFIT, from the fixture's start (the second CH2 twisted 80 degrees
// and pyramidalized, no symmetry: from the Cs start the search reaches a
// saddle on the seam), at Opt=Tight's four thresholds and a gap of 1e-7. The
// claims, each run once against a bar derived from the fixture, with F this
// test's max_force plus the reference's 1e-6 and Delta this test's gap plus
// the reference's 1e-8, and s_min the gap's smaller slope in the branching
// plane (smaller_gap_slope of the fixture's MECI vectors, 0.1413), not
// min(x_norm, 2 h_norm), which is not reproducible and never smaller:
// - it is an MECI: a fresh single point at final.xyz (the outer [casscf],
//   from a fresh RHF and pick, then one df_intersection) has a gap within
//   kMeciGap and a largest projected force within max_force;
// - it is the reference's MECI: the geometry, after alignment (a reflection
//   allowed), within delta = F / k_min + Delta / s_min, and the mean energy
//   within gbar_plane Delta / s_min + F^2 / k_min of the fixture's;
// - every optimize.json pair overlap after the first at or above the floor.
// Logged, not gating: the first evaluation's energies, and a fresh single
// point's g_0, g_1 and h_01 at the start (h up to its sign) against the
// fixture's, and the step count. 42 steps in about 15 s on the 3080: past
// ten seconds, but the family's cheapest, so in the fast tier.
TEST(PipelineOptimize, EthyleneMeciReachesTheReferenceMeci) {
  FBF_REQUIRE_GPU();
  const fb::json &reference = meci_reference();
  const fb::json &meci = reference.at("meci");
  const double f_bar = kMaxForce + reference.at("optimizer").at("force_tolerance").get<double>();
  const double gap_bar = kMeciGap + reference.at("optimizer").at("gap_tolerance").get<double>();
  const double k_min = meci.at("k_min").get<double>();
  const double gbar_plane = meci.at("gbar_plane").get<double>();
  const std::vector<double> ref_g0 = flat_rows(meci.at("gradients").at(0));
  const std::vector<double> ref_g1 = flat_rows(meci.at("gradients").at(1));
  const std::vector<double> ref_h = flat_rows(meci.at("h"));
  const double s_min = smaller_gap_slope(ref_g0, ref_g1, ref_h);
  // The derivation against the README's recorded value, not a claim.
  ASSERT_NEAR(s_min, 0.1413, 5e-5);
  const double delta = (f_bar / k_min) + (gap_bar / s_min);
  // The issue's rule: over 1e-3 bohr, the thresholds are tightened, never
  // the bar.
  ASSERT_LE(delta, 1e-3) << "tighten max_force or kMeciGap";
  const double energy_bar = (gbar_plane * gap_bar / s_min) + (f_bar * f_bar / k_min);

  const fb::test::ScratchDir dir("pipeline_optimize_meci");
  write_xyz(dir / "start.xyz", reference.at("start").at("atoms"));
  const std::string chain = std::format(
      "[basis_set]\nbasis = \"{}\"\n[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[scf]\nmax_iterations = 400\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\n{}",
      reference.at("basis").get<std::string>(),
      reference.at("aux_basis").at("name").get<std::string>(),
      reference.at("ncas").get<std::int32_t>(), reference.at("nelecas").get<std::int32_t>(),
      reference.at("weights").size(), kCasscf);
  fb::NodeResources node;

  // The start's diagnostics: a single point of its own.
  ASSERT_TRUE(
      fb::write_file(dir / "start.toml", std::format("[molecule]\nxyz = \"start.xyz\"\n{}", chain))
          .has_value());
  auto start_calc = fb::read_config(dir / "start.toml", dir.path(), dir / "start");
  ASSERT_TRUE(start_calc.has_value()) << start_calc.error().message;
  const fb::Status start_ran = start_calc->run(node);
  ASSERT_TRUE(start_ran.has_value()) << start_ran.error().message;
  const PairPoint start = fresh_intersection(*start_calc, node.device(0));
  const fb::json &ref_start = reference.at("start");
  const auto max_diff = [](std::span<const double> a, std::span<const double> b, double sign) {
    double m = 0.0;
    for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
      m = std::max(m, std::abs(a[k] - (sign * b[k])));
    }
    return m;
  };
  const std::vector<double> start_h = flat_rows(ref_start.at("h"));
  const double start_g0 = max_diff(start.g0, flat_rows(ref_start.at("gradients").at(0)), 1.0);
  const double start_g1 = max_diff(start.g1, flat_rows(ref_start.at("gradients").at(1)), 1.0);
  const double start_h_err =
      std::min(max_diff(start.h, start_h, 1.0), max_diff(start.h, start_h, -1.0));

  // The optimization.
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             std::format("[molecule]\nxyz = \"start.xyz\"\n{}{}meci = [0, 1]\n"
                                         "max_gap = {}\n",
                                         chain, kOptimize, kMeciGap))
                  .has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::OptimizeStep>()->meci.has_value());
  ASSERT_EQ(calc->step<fb::OptimizeStep>()->options.max_gap, kMeciGap);
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(out.has_value());

  // It is an MECI: the fresh single point at final.xyz, the outer [casscf].
  const Points got = points(out->molecule);
  const PairPoint fresh = fresh_intersection(*calc, node.device(0));
  ASSERT_EQ(fresh.g0.size(), 3 * got.size());
  const double fresh_gap = std::abs(fresh.energies[1] - fresh.energies[0]);
  const double fresh_force = max_projected_force(got, fresh);
  EXPECT_LE(fresh_gap, kMeciGap);
  EXPECT_LE(fresh_force, kMaxForce);

  // It is the reference's MECI.
  const Points want = points(meci.at("atoms"));
  const double direct = aligned_max_deviation(want, got);
  const double reflected = aligned_max_deviation(want, mirrored(got));
  const double geometry_error = std::min(direct, reflected);
  EXPECT_LE(geometry_error, delta);
  const double energy_error = std::abs(out->energy - meci.at("mean_energy").get<double>());
  EXPECT_LE(energy_error, energy_bar);

  // The path: the pair kept at every evaluation.
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 1U);
  EXPECT_EQ(history->at("energy").get<double>(), out->energy);
  EXPECT_TRUE(iterations.front().at("pair_overlap").is_null());
  double worst_overlap = 1.0;
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    const fb::json &it = iterations[i];
    if (i > 0) {
      const double overlap = it.at("pair_overlap").get<double>();
      EXPECT_GE(overlap, fb::kStateOverlapFloor) << i;
      worst_overlap = std::min(worst_overlap, overlap);
    }
    std::cout << std::format(
        "[ optimize ] MECI evaluation {:>2}: E {:.10f} {:.10f} gap {:.3e}, max force {:.2e}, "
        "plane {}, pair overlap {}\n",
        i, it.at("lower").get<double>(), it.at("upper").get<double>(), it.at("gap").get<double>(),
        it.at("max_force").get<double>(), it.at("branching_dimension").get<std::int32_t>(),
        i == 0 ? std::string("-") : std::format("{:.10f}", it.at("pair_overlap").get<double>()));
  }
  const auto ref_start_energies = ref_start.at("energies").get<std::vector<double>>();
  std::cout << std::format(
      "[ optimize ] ethylene SA2-CASSCF(2,2) MECI: {} steps, {:.1f} s\n"
      "[ optimize ]   fresh single point: gap {:.2e} (bar {:.1e}), max projected force {:.2e} "
      "(bar {:.1e})\n"
      "[ optimize ]   geometry {:.2e} bohr ({}; direct {:.2e}, reflected {:.2e}), delta {:.2e} = "
      "F/k_min {:.2e} + Delta/s_min {:.2e}\n"
      "[ optimize ]   mean energy {:.2e} Eh, bar {:.2e} = gbar Delta/s_min {:.2e} + F^2/k_min "
      "{:.2e}\n"
      "[ optimize ]   F {:.2e}, Delta {:.2e}, k_min {:.6f}, s_min {:.6f}, gbar_plane {:.6f}; "
      "worst pair overlap {:.10f}\n"
      "[ optimize ]   start (diagnostic): |dE_0| {:.2e} |dE_1| {:.2e}, max |dg_0| {:.2e} "
      "max |dg_1| {:.2e} max |dh| {:.2e} (up to sign), max |h| {:.2e}\n",
      iterations.size() - 1, seconds, fresh_gap, kMeciGap, fresh_force, kMaxForce, geometry_error,
      direct <= reflected ? "direct" : "reflected", direct, reflected, delta, f_bar / k_min,
      gap_bar / s_min, energy_error, energy_bar, gbar_plane * gap_bar / s_min,
      f_bar * f_bar / k_min, f_bar, gap_bar, k_min, s_min, gbar_plane, worst_overlap,
      std::abs(iterations.front().at("lower").get<double>() - ref_start_energies.at(0)),
      std::abs(iterations.front().at("upper").get<double>() - ref_start_energies.at(1)), start_g0,
      start_g1, start_h_err,
      std::abs(std::ranges::max(start_h, {}, [](double v) { return std::abs(v); })));
}

// --- The QD-NEVPT2 MECI ------------------------------------------------------

// The case's state average: every singlet of the CAS(2,2), equal weights, so
// the third state is there for the margin and the oracle's plane.
constexpr std::int32_t kQdMeciRoots = 3;
// The issue's thresholds: the gap, and kOptimize's Opt=Tight four.
constexpr double kQdMeciGap = 1e-7;
// The chain's [casscf]: conv_tol_grad 1e-10, so s_o ||g|| sits far under the
// gap (the default 1e-6 would put it near s_o 1e-6, above Delta).
constexpr std::string_view kQdMeciCasscf = "conv_tol_grad = 1e-10\nconv_tol = 1e-13\n"
                                           "max_cycle = 200\n";
// C1: the convergence term's allowance, a tenth of the gap.
constexpr double kQdConvergenceShare = 0.1;
// C3: data/casscf_meci's S2_MARGIN (meci_reference.py), hartree.
constexpr double kQdThirdStateMargin = 1e-2;
// C2's oracle: the energy path of the chain (#783) at the fresh point,
// central differences at three halving steps and Richardson, with the
// cc-pVDZ cases' per-point noise (qd_nevpt2_gradient/README.md).
constexpr std::array<double, 3> kOracleSteps = {1e-3, 5e-4, 2.5e-4};
constexpr double kOracleNoise = 1e-11;

// One QD-NEVPT2 single point's quantities at a CASSCF: the QD energies, the
// branching vectors and the gap sensitivities, and the CASSCF's residuals.
struct QdPoint {
  std::vector<double> energies;
  std::vector<double> g0;
  std::vector<double> g1;
  std::vector<double> h;
  double s_o{0.0};
  double s_c{0.0};
  double gradient_norm{0.0};
  double ci_residual{0.0};
};

// One qd_{sc,pc}_intersection of states 0 and 1 over @p calc's own products
// after it ran: its outer [casscf], from its own RHF orbitals and pick, never
// an optimization's carried ones (fresh_intersection's Solution).
QdPoint fresh_qd_intersection(const fb::Calculation &calc, fb::DeviceResources &res, bool pc) {
  QdPoint point;
  const auto &cas = *calc.get<fb::CasscfOutput>();
  const fb::BasisSet &basis = calc.get<fb::BasisSetOutput>()->basis;
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t e = 0; e < n2; ++e) {
      gamma[e] += cas.weights[r] * cas.rdm1[(r * n2) + e];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  const auto *const fit = std::get_if<fb::DensityFitted>(&calc.get<fb::EriOutput>()->eri);
  if (fit == nullptr) {
    ADD_FAILURE() << "the ERIs are not density-fitted";
    return point;
  }
  const fb::Molecule &molecule = calc.get<fb::MoleculeOutput>()->molecule;
  const auto &classes = calc.get<fb::PrimitivePairClassesOutput>()->classes;
  const fb::SphericalMatrixAO &h = calc.get<fb::CoreHamiltonianOutput>()->h;
  auto out =
      pc ? fb::nevpt2::qd_pc_intersection(res, molecule, basis, classes, fit->aux, h, fit->b,
                                          fit->l.data(), solution, cas.ncore, cas.ncas, 0, 1, 0)
         : fb::nevpt2::qd_sc_intersection(res, molecule, basis, classes, fit->aux, h, fit->b,
                                          fit->l.data(), solution, cas.ncore, cas.ncas, 0, 1, 0);
  if (!out.has_value()) {
    ADD_FAILURE() << out.error().message;
    return point;
  }
  point.energies = out->eigenvalues;
  point.g0 = std::move(out->g_p);
  point.g1 = std::move(out->g_q);
  point.h = std::move(out->h_pq);
  point.s_o = out->gap_sensitivity_orbital;
  point.s_c = out->gap_sensitivity_ci;
  point.gradient_norm = cas.gradient_norm;
  point.ci_residual = cas.ci_residual_norm;
  return point;
}

// An orthonormal basis of the 3N - 6 internal directions at @p atoms (bohr):
// the Cartesian unit vectors orthogonalized, twice, against the translations
// and the rotations about the centroid (max_projected_force's) and each
// other, the ones with anything left kept.
std::vector<std::vector<double>> internal_directions(const Points &atoms) {
  const std::size_t n = 3 * atoms.size();
  std::array<double, 3> centroid{};
  for (const auto &r : atoms) {
    for (std::size_t k = 0; k < 3; ++k) {
      centroid[k] += r[k] / static_cast<double>(atoms.size());
    }
  }
  std::vector<std::vector<double>> candidates;
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<double> t(n, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      t[(3 * a) + k] = 1.0;
    }
    candidates.push_back(std::move(t));
  }
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<double> rot(n, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      const std::size_t k1 = (k + 1) % 3;
      const std::size_t k2 = (k + 2) % 3;
      rot[(3 * a) + k2] = atoms[a][k1] - centroid[k1];
      rot[(3 * a) + k1] = -(atoms[a][k2] - centroid[k2]);
    }
    candidates.push_back(std::move(rot));
  }
  for (std::size_t c = 0; c < n; ++c) {
    std::vector<double> e(n, 0.0);
    e[c] = 1.0;
    candidates.push_back(std::move(e));
  }
  std::vector<std::vector<double>> basis;
  std::size_t rigid = 0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    std::vector<double> v = candidates[i];
    for (int32_t pass = 0; pass < 2; ++pass) {
      for (const auto &u : basis) {
        const double dot = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
        for (std::size_t c = 0; c < n; ++c) {
          v[c] -= dot * u[c];
        }
      }
    }
    const double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    if (norm > 1e-6) {
      for (double &value : v) {
        value /= norm;
      }
      basis.push_back(std::move(v));
      rigid += i < 6 ? 1 : 0;
    }
  }
  return {basis.begin() + static_cast<std::ptrdiff_t>(rigid), basis.end()};
}

// Sum_i c_i u_i: a vector of components along @p directions, in Cartesians.
std::vector<double> cartesian(std::span<const double> c,
                              const std::vector<std::vector<double>> &directions) {
  std::vector<double> out(directions.front().size(), 0.0);
  for (std::size_t i = 0; i < directions.size(); ++i) {
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] += c[i] * directions[i][k];
    }
  }
  return out;
}

double norm2(std::span<const double> v) {
  return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

double largest_bar(const fb::nevpt2::Differenced &d) {
  return std::ranges::max(d.error_bar);
}

// The issue's C2 at @p atoms: the oracle's ḡ, x and h along the internal
// directions, P ḡ back in Cartesians (max_projected_force's projector), and
// the bar F + 2 e (1 + 2 |ḡ| / min(|x|, 2 |h|)). Returns whether it passed.
struct OracleCheck {
  double force{0.0};
  double bar{0.0};
  double oracle_term{0.0};
  double e{0.0};
  double gbar_norm{0.0};
  double plane_norm{0.0};
  std::size_t points{0};
  double seconds{0.0};
  std::vector<double> gbar;
  std::vector<double> x;
  std::vector<double> h;
  std::vector<std::vector<double>> directions;
};

std::optional<OracleCheck> oracle_at(fb::NodeResources &node, const Points &at,
                                     const std::vector<fb::Atom> &atoms, bool pc) {
  OracleCheck check;
  check.directions = internal_directions(at);
  EXPECT_EQ(check.directions.size(), (3 * at.size()) - 6);
  fb::nevpt2::NumericalChain chain{
      .atoms = atoms,
      .basis = meci_reference().at("basis").get<std::string>(),
      .aux_basis = meci_reference().at("aux_basis").at("name").get<std::string>(),
      .ncas = meci_reference().at("ncas").get<std::int32_t>(),
      .nelecas = meci_reference().at("nelecas").get<std::int32_t>(),
      .active = {},
      .weights = std::vector<double>(kQdMeciRoots, 1.0 / kQdMeciRoots),
      .pc = pc};
  const auto t0 = std::chrono::steady_clock::now();
  auto heff = fb::nevpt2::numerical_heff(node, chain, check.directions, kOracleSteps, kOracleNoise);
  check.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!heff.has_value()) {
    ADD_FAILURE() << "the oracle: " << heff.error().message;
    return std::nullopt;
  }
  check.points = heff->points;
  const fb::nevpt2::HeffDerivative &d = pc ? *heff->pc : heff->sc;
  const fb::nevpt2::HeffPair pair = fb::nevpt2::heff_pair(d, d.vectors, 0, 1);
  check.e = std::max({largest_bar(pair.mean), largest_bar(pair.x), largest_bar(pair.h)});
  check.gbar = cartesian(pair.mean.value, check.directions);
  check.x = cartesian(pair.x.value, check.directions);
  check.h = cartesian(pair.h.value, check.directions);
  check.gbar_norm = norm2(check.gbar);
  check.plane_norm = std::min(norm2(check.x), 2.0 * norm2(check.h));
  check.oracle_term = 2.0 * check.e * (1.0 + (2.0 * check.gbar_norm / check.plane_norm));
  check.bar = kMaxForce + check.oracle_term;
  PairPoint plane;
  plane.g0.resize(check.gbar.size());
  plane.g1.resize(check.gbar.size());
  for (std::size_t k = 0; k < check.gbar.size(); ++k) {
    plane.g0[k] = check.gbar[k] - (0.5 * check.x[k]);
    plane.g1[k] = check.gbar[k] + (0.5 * check.x[k]);
  }
  plane.h = check.h;
  check.force = max_projected_force(at, plane);
  return check;
}

// Ethylene's SC- (or PC-) QD-NEVPT2 MECI of states 0 and 1 over its
// DF-SA3-CASSCF(2,2), cc-pVDZ / cc-pVDZ-JKFIT, equal weights, from
// data/casscf_meci's `meci.atoms` (the SA2 CASSCF MECI, twisted off Cs), at
// Opt=Tight's four thresholds and a gap of 1e-7. The issue's claims, each
// run once against its derived bar:
// - C1, the gap: a fresh single point at final.xyz (the outer [casscf] from
//   a fresh RHF and pick, then one qd_*_intersection) has QD states 0 and 1
//   within Delta + sum over the optimizer's last point and the fresh one of
//   s_o ||g|| + s_c sqrt(3) rho, ||g|| and rho the CASSCFs' reported
//   gradient_norm and ci_residual_norm, s_o and s_c the fresh point's; that
//   convergence term at or below Delta / 10;
// - C2, seam stationarity: the oracle's P ḡ at the fresh
//   point (oracle_at) within F + 2 e (1 + 2 |ḡ| / min(|x|, 2 |h|)), the
//   oracle's term at or below F; on a failure, the start's analytic ḡ, x and
//   h against the oracle there, logged to localize it;
// - C3: E_2 - E_1 at the fresh point at least kQdThirdStateMargin.
// Logged, not gating: every optimize.json entry's pair overlap, third-state
// margin and CASSCF residuals, the step count, the final mean energy, the
// RMS displacement from the start, and the wall times.
void expect_qd_meci(bool pc) {
  const std::string_view scheme = pc ? "PC" : "SC";
  const fb::json &reference = meci_reference();
  const fb::test::ScratchDir dir(std::format("pipeline_optimize_qd_meci_{}", pc ? "pc" : "sc"));
  write_xyz(dir / "start.xyz", reference.at("meci").at("atoms"));
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n[eri]\nmode = "
      "\"density_fitting\"\naux_basis = \"{}\"\n[scf]\nmax_iterations = 400\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\n{}"
      "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"{}\"\n{}meci = [0, 1]\n"
      "max_gap = {}\n",
      reference.at("basis").get<std::string>(),
      reference.at("aux_basis").at("name").get<std::string>(),
      reference.at("ncas").get<std::int32_t>(), reference.at("nelecas").get<std::int32_t>(),
      kQdMeciRoots, kQdMeciCasscf, pc ? "pc" : "sc", kOptimize, kQdMeciGap);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  ASSERT_TRUE(calc->step<fb::OptimizeStep>()->meci.has_value());
  ASSERT_TRUE(calc->step<fb::OptimizeStep>()->nevpt2_gradient.has_value());
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(out.has_value());

  // The path.
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 1U);
  EXPECT_TRUE(iterations.front().at("pair_overlap").is_null());
  double worst_overlap = 1.0;
  double least_margin = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    const fb::json &it = iterations[i];
    if (i > 0) {
      worst_overlap = std::min(worst_overlap, it.at("pair_overlap").get<double>());
    }
    least_margin = std::min(least_margin, it.at("third_state_margin").get<double>());
    std::cout << std::format(
        "[ optimize ] {}-QD MECI evaluation {:>2}: E {:.10f} {:.10f} gap {:.3e} (H_eff {:.3e}), "
        "max force {:.2e}, pair overlap {}, third-state margin {:.6f}, CASSCF root gap {:.6f}, "
        "|g| {:.2e}, rho {:.2e}\n",
        scheme, i, it.at("lower").get<double>(), it.at("upper").get<double>(),
        it.at("gap").get<double>(), it.at("heff_gap").get<double>(),
        it.at("max_force").get<double>(),
        i == 0 ? std::string("-") : std::format("{:.10f}", it.at("pair_overlap").get<double>()),
        it.at("third_state_margin").get<double>(), it.at("root_gap").get<double>(),
        it.at("gradient_norm").get<double>(), it.at("ci_residual_norm").get<double>());
  }
  const fb::json &last = iterations.back();
  const double opt_g = last.at("gradient_norm").get<double>();
  const double opt_rho = last.at("ci_residual_norm").get<double>();

  // C1 and C3 at the fresh single point.
  const Points got = points(out->molecule);
  const QdPoint fresh = fresh_qd_intersection(*calc, node.device(0), pc);
  ASSERT_EQ(fresh.energies.size(), static_cast<std::size_t>(kQdMeciRoots));
  const double fresh_gap = fresh.energies[1] - fresh.energies[0];
  const double root_n = std::sqrt(static_cast<double>(kQdMeciRoots));
  const double opt_term = (fresh.s_o * opt_g) + (fresh.s_c * root_n * opt_rho);
  const double fresh_term =
      (fresh.s_o * fresh.gradient_norm) + (fresh.s_c * root_n * fresh.ci_residual);
  const double convergence = opt_term + fresh_term;
  const double gap_bar = kQdMeciGap + convergence;
  EXPECT_LE(convergence, kQdConvergenceShare * kQdMeciGap)
      << "the convergence term is over Delta / 10: tighten [casscf] conv_tol_grad, never the bar";
  EXPECT_LE(fresh_gap, gap_bar);
  const double margin = fresh.energies[2] - fresh.energies[1];
  EXPECT_GE(margin, kQdThirdStateMargin);

  double rms = 0.0;
  const Points from = points(reference.at("meci").at("atoms"));
  for (std::size_t a = 0; a < got.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      rms += (got[a][k] - from[a][k]) * (got[a][k] - from[a][k]);
    }
  }
  rms = std::sqrt(rms / static_cast<double>(3 * got.size()));
  std::cout << std::format(
      "[ optimize ] ethylene {}-QD-NEVPT2 SA3-CASSCF(2,2) MECI: {} steps, {:.1f} s, mean E "
      "{:.10f}, RMS displacement from the start {:.4f} bohr; worst pair overlap {:.10f}, least "
      "third-state margin {:.6f}\n"
      "[ optimize ]   C1: fresh gap {:.3e} <= {:.3e} = Delta {:.1e} + convergence {:.3e} (opt "
      "s_o|g| + s_c sqrt3 rho {:.2e}, fresh {:.2e}; Delta/10 {:.1e}); s_o {:.4e} s_c {:.4e}, "
      "opt |g| {:.2e} rho {:.2e}, fresh |g| {:.2e} rho {:.2e}\n"
      "[ optimize ]   C3: E_2 - E_1 {:.6f} >= {:.0e}\n",
      scheme, iterations.size() - 1, seconds, out->energy, rms, worst_overlap, least_margin,
      fresh_gap, gap_bar, kQdMeciGap, convergence, opt_term, fresh_term,
      kQdConvergenceShare * kQdMeciGap, fresh.s_o, fresh.s_c, opt_g, opt_rho, fresh.gradient_norm,
      fresh.ci_residual, margin, kQdThirdStateMargin);
  // C2, by the oracle at the fresh point.
  const std::vector<fb::Atom> atoms = out->molecule.atoms();
  const auto check = oracle_at(node, got, atoms, pc);
  ASSERT_TRUE(check.has_value());
  EXPECT_LE(check->oracle_term, kMaxForce)
      << "the oracle's term is over F: smaller steps or more rungs, never the looser bar";
  EXPECT_LE(check->force, check->bar);
  std::cout << std::format(
      "[ optimize ]   C2: max |P gbar| {:.3e} <= {:.3e} = F {:.1e} + oracle {:.3e} (e {:.2e}, "
      "|gbar| {:.3e}, min(|x|, 2|h|) {:.3e}); {} points, {:.1f} s\n",
      check->force, check->bar, kMaxForce, check->oracle_term, check->e, check->gbar_norm,
      check->plane_norm, check->points, check->seconds);
  if (check->force <= check->bar) {
    return;
  }
  // Only on a failure: the start's analytic g and h against the oracle there.
  const fb::test::ScratchDir start_dir(
      std::format("pipeline_optimize_qd_meci_start_{}", pc ? "pc" : "sc"));
  write_xyz(start_dir / "start.xyz", reference.at("meci").at("atoms"));
  const std::string single = toml.substr(0, toml.find("[nevpt2_gradient]"));
  ASSERT_TRUE(fb::write_file(start_dir / "run.toml", single).has_value());
  auto start_calc = fb::read_config(start_dir / "run.toml", start_dir.path(), start_dir / "out");
  ASSERT_TRUE(start_calc.has_value()) << start_calc.error().message;
  ASSERT_TRUE(start_calc->run(node).has_value());
  const QdPoint analytic = fresh_qd_intersection(*start_calc, node.device(0), pc);
  const auto start_check =
      oracle_at(node, from, start_calc->get<fb::MoleculeOutput>()->molecule.atoms(), pc);
  ASSERT_TRUE(start_check.has_value());
  double dg = 0.0;
  double dx = 0.0;
  double dh_plus = 0.0;
  double dh_minus = 0.0;
  for (const auto &u : start_check->directions) {
    const auto along = [&](std::span<const double> v) {
      return std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
    };
    const double g = 0.5 * (along(analytic.g0) + along(analytic.g1));
    const double x = along(analytic.g1) - along(analytic.g0);
    const double h = along(analytic.h);
    dg = std::max(dg, std::abs(g - along(start_check->gbar)));
    dx = std::max(dx, std::abs(x - along(start_check->x)));
    dh_plus = std::max(dh_plus, std::abs(h - along(start_check->h)));
    dh_minus = std::max(dh_minus, std::abs(h + along(start_check->h)));
  }
  std::cout << std::format("[ optimize ]   C2 failed; at the start, analytic against the oracle "
                           "(e {:.2e}): max |d gbar| {:.3e}, max |d x| {:.3e}, max |d h| {:.3e} "
                           "(up to sign)\n",
                           start_check->e, dg, dx, std::min(dh_plus, dh_minus));
}

// SC: C1, C2 (the oracle's 72 single points) and C3. The optimization alone
// is 31 steps in about 25 s on the 3080, past slow_test.h's ten seconds and
// dearer than the CASSCF MECI above, so the slow tier.
TEST(PipelineOptimize, EthyleneScQdMeciIsAnMeciAndSeamStationary) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_qd_meci(false);
}

// PC, the same case: C1, C2 and C3, in the slow tier.
TEST(PipelineOptimize, EthylenePcQdMeciIsAnMeciAndSeamStationary) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_qd_meci(true);
}

// The QD MECI's optimize.json carries the branching space and the gap
// sensitivities (#844): skew formaldehyde's SC-QD-NEVPT2 states 0 and 1
// over its DF-SA3-CASSCF(4,3) (h2co_skew_ccpvdz_sa3), every threshold loose
// so the start is the crossing and the one evaluation the final geometry.
// Every entry holds gap_sensitivity_orbital and gap_sensitivity_ci; the head
// holds g_p, g_q and h_pq, each n_atom rows of three, and nroots. The start
// is written at final.xyz's twelve decimals, so the outer [casscf] runs at
// the evaluation's own geometry from the same RHF, and one
// qd_sc_intersection over it is the product the file wrote: the three
// vectors and the two sensitivities are held to it at kQdProductBar of the
// vectors' largest magnitude and of s_o + s_c, the reproducibility of
// separately converged Z-vectors (../../src/nevpt2/intersection/README.md,
// Tests: 1e-9 of the largest summand, of the largest norm summed), not a
// convergence allowance. The pair is apart here (the H_eff gap is printed),
// so u_P and u_Q, and the vectors with them, are fixed but for h_PQ's sign,
// the pair's relative phase. s_o ||g|| + s_c sqrt(nroots) rho is formed from the
// file alone and printed.
TEST(PipelineOptimize, SkewFormaldehydeQdMeciWritesItsBranchingSpace) {
  FBF_REQUIRE_GPU();
  constexpr double kQdProductBar = 1e-9;
  const fb::json start = qd_case("h2co_skew_ccpvdz_sa3");
  const fb::test::ScratchDir dir("pipeline_optimize_qd_meci_branching");
  std::string xyz = std::format("{}\nstart\n", start.at("atoms").size());
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.12f} {:.12f} {:.12f}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "start.xyz", xyz).has_value());
  const std::size_t nroots = start.at("weights").size();
  const std::string toml =
      std::format("[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
                  "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\n{}"
                  "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"sc\"\n"
                  "[optimize]\nmeci = [0, 1]\nmax_gap = 1.0\nmax_force = 1.0\nrms_force = 1.0\n"
                  "max_displacement = 1.0\nrms_displacement = 1.0\n",
                  start.at("basis").get<std::string>(), start.at("aux_basis").get<std::string>(),
                  kScf, start.at("ncas").get<std::int32_t>(),
                  start.at("nelecas").get<std::int32_t>(), nroots, kQdMeciCasscf);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_EQ(iterations.size(), 1U);
  for (const fb::json &it : iterations) {
    EXPECT_TRUE(it.at("gap_sensitivity_orbital").is_number());
    EXPECT_TRUE(it.at("gap_sensitivity_ci").is_number());
  }
  const fb::json &last = iterations.back();
  EXPECT_EQ(history->at("nroots").get<std::size_t>(), nroots);
  const std::size_t natom = start.at("atoms").size();
  for (const std::string key : {"g_p", "g_q", "h_pq"}) {
    const fb::json &rows = history->at(key);
    ASSERT_TRUE(rows.is_array()) << key;
    ASSERT_EQ(rows.size(), natom) << key;
    for (const fb::json &row : rows) {
      ASSERT_TRUE(row.is_array() && row.size() == 3) << key;
    }
  }

  const QdPoint product = fresh_qd_intersection(*calc, node.device(0), false);
  ASSERT_EQ(product.energies.size(), nroots);
  // Each against the product at kQdProductBar of a common scale: the three
  // vectors' largest magnitude (a lower bound on the derivative passes'
  // largest summand), and the two sensitivities summed.
  const auto largest = [](const std::vector<double> &v) {
    return std::ranges::max(v, {}, [](double e) { return std::abs(e); });
  };
  const double vector_scale = std::max(
      {std::abs(largest(product.g0)), std::abs(largest(product.g1)), std::abs(largest(product.h))});
  const double sensitivity_scale = product.s_o + product.s_c;
  const auto expect_near = [&](std::string_view key, const std::vector<double> &got,
                               const std::vector<double> &want, double scale) {
    ASSERT_EQ(got.size(), want.size()) << key;
    double worst = 0.0;
    for (std::size_t k = 0; k < got.size(); ++k) {
      worst = std::max(worst, std::abs(got[k] - want[k]));
    }
    EXPECT_LE(worst, kQdProductBar * scale) << key;
    std::cout << std::format("[ optimize ] QD MECI {}: max |file - product| {:.3e} (bar {:.3e})\n",
                             key, worst, kQdProductBar * scale);
  };
  expect_near("g_p", flat_rows(history->at("g_p")), product.g0, vector_scale);
  expect_near("g_q", flat_rows(history->at("g_q")), product.g1, vector_scale);
  // h_PQ's sign is the pair's relative phase, which the CASSCF's CI vectors'
  // own signs set and a separate run need not repeat (optimize_meci reads
  // only its direction): held to the product up to that one sign.
  const std::vector<double> h_file = flat_rows(history->at("h_pq"));
  std::vector<double> h_product = product.h;
  if (std::inner_product(h_file.begin(), h_file.end(), h_product.begin(), 0.0) < 0.0) {
    std::ranges::transform(h_product, h_product.begin(), std::negate<>{});
  }
  expect_near("h_pq", h_file, h_product, vector_scale);
  expect_near("gap_sensitivity_orbital", {last.at("gap_sensitivity_orbital").get<double>()},
              {product.s_o}, sensitivity_scale);
  expect_near("gap_sensitivity_ci", {last.at("gap_sensitivity_ci").get<double>()}, {product.s_c},
              sensitivity_scale);

  const double bound =
      (last.at("gap_sensitivity_orbital").get<double>() * last.at("gradient_norm").get<double>()) +
      (last.at("gap_sensitivity_ci").get<double>() *
       std::sqrt(history->at("nroots").get<double>()) * last.at("ci_residual_norm").get<double>());
  EXPECT_TRUE(std::isfinite(bound));
  std::cout << std::format(
      "[ optimize ] QD MECI from the file: H_eff gap {:.6f}, s_o {:.4e} s_c "
      "{:.4e}, |g| {:.2e} rho {:.2e}, s_o|g| + s_c sqrt(n) rho {:.3e}\n",
      last.at("heff_gap").get<double>(), last.at("gap_sensitivity_orbital").get<double>(),
      last.at("gap_sensitivity_ci").get<double>(), last.at("gradient_norm").get<double>(),
      last.at("ci_residual_norm").get<double>(), bound);
}

} // namespace
