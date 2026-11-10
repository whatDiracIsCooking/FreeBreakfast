// The numerical [hessian] over a quasi_degenerate [nevpt2_gradient] chain
// (#959), against #953's and #1058's semi-numerical QD references,
// data/qd_nevpt2_hessian/ (its README has the cases):
//
// - The C1 case (slow-gated, SC and PC), h2co_skew_ccpvdz_sa3, equal
//   weights, state 1: the step's ladder at the reference's h for each
//   variant (SC 3.125e-4, PC 2.5e-3), PC at the reference's round-off
//   floor (pc_floor = 0), no pairs. Every element of the finest rung's
//   Hessian is within 2 x the reference's bar of the reference's, the
//   ladder settles, and the state was tracked at every displacement. The
//   outer CASSCF and QD energies are first held to the reference's, so the
//   differences are taken around its solution.
// - The C2v A1/B2 case (slow-gated, SC), h2nb_sto3g_sa2, in its
//   symmetry-adapted coordinates at its h (4e-5), every pair of its six
//   vibrations checked, each state: the bilinearity check flags exactly the
//   reference's expected SC refusals, none, so the frequencies are
//   published. PC's is not gated (the README: its computed surface jumps
//   at the cut, and its refusal is analytic).
// - The refusals (fast): the step's own QD ones, run directly by either
//   method (a frozen core, integral-direct ERIs, one root, a state past
//   them), each before any displaced solve. The swap refusal itself is tracked_qd_state's
//   (test/pipeline/hessian_host.cpp), at_geometry naming the point.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kSkew = "h2co_skew_ccpvdz_sa3";
constexpr std::string_view kCoupled = "h2nb_sto3g_sa2";

// The outer CASSCF and QD energies against the reference's: the same
// solution (the reference's polished to ||g|| < 3e-12, this one to 1e-10).
constexpr double kEnergyTolerance = 1e-8;
// The finest rung's Hessian against the reference's: within this many times
// its bar (the issue's acceptance).
constexpr double kBarFactor = 2.0;
// The C1 case's state: K >= 1, the first excited QD state.
constexpr std::int32_t kSkewState = 1;
// [casscf] conv_tol_grad, both cases': the NEVPT2 chain's 1e-10. The
// ladder's tightened re-solve then asks 1e-12 (#1090: the solve's subspace
// once refused its first column there, mcscf/solve/README.md).
constexpr double kConvTolGrad = 1e-10;

const fb::json &reference(std::string_view name) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto found = docs.find(name); found != docs.end()) {
    return found->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2_hessian" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{name}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

// The reference @p name's molecule.xyz in @p dir (bohr there, Angstrom here)
// and run.toml: its basis, fit, CAS and weights, the SCF and CASSCF converged
// tight (its gradient to @p conv_tol_grad), a quasi_degenerate
// [nevpt2_gradient] of @p variant at @p state and a [hessian] of @p hessian.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, std::string_view name,
                                   std::string_view variant, std::int32_t state,
                                   std::string_view hessian, double conv_tol_grad) {
  const fb::json &doc = reference(name);
  std::string xyz = std::format("{}\n{}\n", doc.at("atoms").size(), name);
  for (const fb::json &atom : doc.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  EXPECT_TRUE(doc.at("active").is_null()); // the frontier orbitals
  std::string weights;
  for (const double w : doc.at("weights").get<std::vector<double>>()) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
                  "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
                  "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\n"
                  "conv_tol = 1e-12\nconv_tol_grad = {:g}\n"
                  "[nevpt2_gradient]\nvariant = \"{}\"\nquasi_degenerate = true\nstate = {}\n"
                  "[hessian]\n{}",
                  doc.at("basis").at("name").get<std::string>(),
                  doc.at("aux_basis").at("name").get<std::string>(),
                  doc.at("ncas").get<std::int32_t>(), doc.at("nelecas").get<std::int32_t>(),
                  doc.at("weights").size(), weights, conv_tol_grad, variant, state, hessian);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The run of @p name's @p variant at @p state with the [hessian] body
// @p hessian, its outer CASSCF and QD energies held to the reference's;
// @return the run's status and its hessian.json.
std::pair<fb::Status, fb::json> run_hessian(std::string_view name, std::string_view variant,
                                            std::int32_t state, std::string_view hessian,
                                            double conv_tol_grad) {
  const fb::json &doc = reference(name);
  const fb::test::ScratchDir dir(
      std::format("pipeline_qd_nevpt2_hessian_{}_{}_{}", name, variant, state));
  auto read = fb::read_config(write_config(dir, name, variant, state, hessian, conv_tol_grad),
                              dir.path(), dir / "out");
  EXPECT_TRUE(read.has_value()) << read.error().message;
  if (!read.has_value()) {
    return {fb::Status{}, fb::json{}};
  }
  fb::Calculation calc = std::move(*read);
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const auto t0 = std::chrono::steady_clock::now();
  fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto &cas = calc.get<fb::CasscfOutput>();
  const auto &gradient = calc.get<fb::Nevpt2GradientOutput>();
  EXPECT_TRUE(cas.has_value() && gradient.has_value());
  if (cas.has_value() && gradient.has_value()) {
    const auto want = doc.at("reference").at("casscf_energies").get<std::vector<double>>();
    EXPECT_EQ(cas->energies.size(), want.size());
    for (std::size_t r = 0; r < std::min(want.size(), cas->energies.size()); ++r) {
      EXPECT_NEAR(cas->energies[r], want[r], kEnergyTolerance) << "root " << r;
    }
    const double want_qd = doc.at("reference")
                               .at(variant)
                               .at("energies")
                               .at(static_cast<std::size_t>(state))
                               .get<double>();
    std::visit(
        [&]<class G>(const G &g) {
          if constexpr (std::same_as<G, fb::nevpt2::QdScGradient> ||
                        std::same_as<G, fb::nevpt2::QdPcGradient>) {
            EXPECT_NEAR(g.lagrangian.eigenvalues.at(static_cast<std::size_t>(state)), want_qd,
                        kEnergyTolerance);
          } else {
            ADD_FAILURE() << "a state-specific gradient";
          }
        },
        gradient->gradient);
  }
  auto written = fb::read_json(dir / "out" / "hessian.json");
  EXPECT_TRUE(written.has_value()) << written.error().message;
  fb::json out = written.has_value() ? std::move(*written) : fb::json{};
  if (!out.is_null()) {
    const fb::json &noise = out.at("ladder").at("noise");
    std::cout << std::format("[ qd nevpt2 hessian ] {} {} state {}: {:.0f} s, sigma_g {:.2e} "
                             "from the {}, floor {:.2e}, failing {}, unsettled {}, qd {}\n",
                             name, variant, state, seconds, noise.at("sigma_g").get<double>(),
                             noise.at("from").get<std::string>(), noise.at("floor").get<double>(),
                             out.at("failing_pairs").dump(),
                             out.at("ladder").at("unsettled").dump(), out.at("qd").dump());
  }
  return {std::move(ran), std::move(out)};
}

// The C1 case: the step's Hessian of state kSkewState is the reference's to
// 2 x its bar, element by element.
void expect_skew(std::string_view variant) {
  const fb::json &doc = reference(kSkew);
  ASSERT_FALSE(doc.is_null());
  const double h = doc.at("steps").at(variant).at("h").get<double>();
  const std::string hessian =
      std::format("method = \"numerical\"\ndisplacement = {:.17g}\nladder = true\npairs = []\n{}",
                  h, variant == "pc" ? "pc_floor = 0.0\n" : "");
  const auto [ran, out] = run_hessian(kSkew, variant, kSkewState, hessian, kConvTolGrad);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_FALSE(out.is_null()) << (ran.has_value() ? std::string{} : ran.error().message);
  EXPECT_EQ(out.at("variant").get<std::string>(), variant);
  EXPECT_EQ(out.at("ladder").at("steps").at(0).get<double>(), h);
  EXPECT_EQ(out.at("qd").at("state").get<std::int32_t>(), kSkewState);
  EXPECT_EQ(out.at("qd").at("reranked").get<std::int32_t>(), 0);
  EXPECT_TRUE(out.at("ladder").at("unsettled").empty()) << out.at("ladder").at("unsettled").dump();
  const fb::json *want_state = nullptr;
  for (const fb::json &s : doc.at(variant).at("states")) {
    if (s.at("state").get<std::int32_t>() == kSkewState) {
      want_state = &s;
    }
  }
  ASSERT_NE(want_state, nullptr);
  // C1, Cartesian: the reference's matrices are per Cartesian element.
  const auto got = out.at("hessian").get<std::vector<std::vector<double>>>();
  const auto want = want_state->at("hessian").get<std::vector<std::vector<double>>>();
  const auto bar = want_state->at("bar").get<std::vector<std::vector<double>>>();
  const std::size_t n = 3 * doc.at("atoms").size();
  ASSERT_EQ(got.size(), n);
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double allowed = kBarFactor * bar[i][j];
      EXPECT_NEAR(got[i][j], want[i][j], allowed) << variant << " (" << i << ", " << j << ")";
      worst = std::max(worst, std::abs(got[i][j] - want[i][j]) / allowed);
    }
  }
  std::cout << std::format("[ qd nevpt2 hessian ] {} {} state {}: worst |H - reference| / (2 "
                           "bar) {:.3f}\n",
                           kSkew, variant, kSkewState, worst);
}

// The A1/B2 case, SC: no pair of vibrations fails (the reference's expected
// SC refusals are none), so the frequencies are published.
void expect_coupled(std::int32_t state) {
  const fb::json &doc = reference(kCoupled);
  ASSERT_FALSE(doc.is_null());
  ASSERT_TRUE(doc.at("expected_refusals").at("sc").empty());
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  const auto vectors = doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>();
  // Every pair of vibrations: a translation or rotation couples nothing.
  std::vector<std::size_t> vibrations;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!labels[i].starts_with("T_") && !labels[i].starts_with("R_")) {
      vibrations.push_back(i);
    }
  }
  ASSERT_EQ(vibrations.size(), 6U);
  std::string pairs;
  for (std::size_t a = 0; a < vibrations.size(); ++a) {
    for (std::size_t b = a + 1; b < vibrations.size(); ++b) {
      pairs += std::format("{}[{}, {}]", pairs.empty() ? "" : ", ", vibrations[a], vibrations[b]);
    }
  }
  std::string hessian = std::format("method = \"numerical\"\ndisplacement = {:.17g}\nladder = "
                                    "true\npairs = [{}]\ncoordinates = [\n",
                                    doc.at("steps").at("sc").at("h").get<double>(), pairs);
  for (const auto &row : vectors) {
    hessian += "  [";
    for (std::size_t x = 0; x < row.size(); ++x) {
      hessian += std::format("{}{:.17g}", x == 0 ? "" : ", ", row[x]);
    }
    hessian += "],\n";
  }
  hessian += "]\n";
  const auto [ran, out] = run_hessian(kCoupled, "sc", state, hessian, kConvTolGrad);
  ASSERT_FALSE(out.is_null()) << (ran.has_value() ? std::string{} : ran.error().message);
  EXPECT_TRUE(out.at("failing_pairs").empty()) << out.at("failing_pairs").dump();
  EXPECT_EQ(out.at("bilinearity").size(), 15U);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  EXPECT_TRUE(out.contains("frequencies"));
  EXPECT_EQ(out.at("qd").at("reranked").get<std::int32_t>(), 0);
  // The coupled pair's verdict, by name.
  const auto coupled = doc.at("coupled").get<std::vector<std::string>>();
  const auto index = [&](const std::string &label) {
    return static_cast<std::size_t>(std::ranges::find(labels, label) - labels.begin());
  };
  const std::array<std::size_t, 2> pair{index(coupled[0]), index(coupled[1])};
  bool seen = false;
  for (const fb::json &p : out.at("bilinearity")) {
    if (p.at("pair").get<std::array<std::size_t, 2>>() == pair) {
      seen = true;
      std::cout << std::format("[ qd nevpt2 hessian ] {} sc state {}: ({}, {}) residual {:+.2e} "
                               "allowed {:.2e}\n",
                               kCoupled, state, coupled[0], coupled[1],
                               p.at("residual").get<double>(), p.at("allowed").get<double>());
    }
  }
  EXPECT_TRUE(seen);
}

TEST(PipelineQdNevpt2Hessian, ScSkewStateOneMatchesTheReference) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("sc");
}

TEST(PipelineQdNevpt2Hessian, PcSkewStateOneMatchesTheReference) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("pc");
}

TEST(PipelineQdNevpt2Hessian, ScCoupledA1B2FlagsNoPair) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_coupled(0);
  expect_coupled(1);
}

// The step's own QD refusals, run directly on a planned step changed after
// planning: each returned at once, before the reference CASSCF, naming its
// reason.
TEST(PipelineQdNevpt2Hessian, StepRefusesBeforeAnyDisplacedSolve) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_qd_nevpt2_hessian_step");
  auto read = fb::read_config(
      write_config(dir, kCoupled, "sc", 1, "method = \"numerical\"\n", kConvTolGrad), dir.path(),
      dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::HessianStep planned = *read->step<fb::HessianStep>();
  planned.json_path.clear();
  ASSERT_TRUE(planned.nevpt2_gradient && planned.nevpt2_gradient->quasi_degenerate);
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  fb::DeviceResources &res = node.device(0);
  auto molecule = fb::MoleculeStep{.xyz = dir / "molecule.xyz"}.run(fb::View<>{}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

  const auto refused = [&](const fb::HessianStep &step, fb::ErrorKind kind, std::string_view says) {
    const auto out = step.run(fb::View<fb::MoleculeOutput>{*molecule}, node);
    ASSERT_FALSE(out.has_value()) << says;
    EXPECT_EQ(out.error().kind, kind) << out.error().message;
    EXPECT_NE(out.error().message.find(says), std::string::npos) << out.error().message;
    EXPECT_FALSE(out.error().message.starts_with("[hessian] at ")) << out.error().message;
  };
  // By either method: the analytic one (#968) refuses the same.
  for (const fb::HessianMethod method :
       {fb::HessianMethod::Numerical, fb::HessianMethod::Analytic}) {
    fb::HessianStep base = planned;
    base.method = method;
    if (method == fb::HessianMethod::Analytic) {
      base.ladder = false;
      base.pairs.emplace();
    }
    {
      fb::HessianStep step = base;
      step.nevpt2_gradient->frozen_core = 1;
      refused(step, fb::ErrorKind::InvalidConfig, "frozen_core is 1");
    }
    {
      fb::HessianStep step = base;
      step.eri.mode = fb::EriMode::IntegralDirect;
      refused(step, fb::ErrorKind::Unsupported, "'integral_direct'");
    }
    {
      fb::HessianStep step = base;
      step.casscf.nroots = 1;
      step.casscf.weights = {1.0};
      refused(step, fb::ErrorKind::InvalidConfig, "QD needs two roots or more");
    }
    {
      fb::HessianStep step = base;
      step.nevpt2_gradient->state = 2;
      refused(step, fb::ErrorKind::InvalidConfig, "is QD state 2's");
    }
  }
}

} // namespace
