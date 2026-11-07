// The numerical [hessian] over the [nevpt2_gradient] chain (#804), and its
// ladder and bilinearity check (#812), against #803's semi-numerical
// references, data/nevpt2_hessian/ (STO-3G / cc-pVDZ-JKFIT; its README has
// the cases):
//
// - The skew case (slow-gated, SC and PC), h2co_skew_sto3g, every block
//   defined: the config's ladder at the fixture's h over every pair. No pair
//   fails, the frequencies are published, some ratios of the Hessian's ladder
//   are judged and every one lies in the window about 16, and the finest
//   rung's Hessian is
//   within 2 x (the fixture's bar + its own) of the fixture's. Each
//   Calculation's outer CASSCF and NEVPT2 energies are first held to the
//   fixture's, so the differences are taken around its solution.
// - The symmetric case (slow-gated, SC and PC), nh3_cs_sto3g, in the
//   fixture's symmetry-adapted coordinates, given in the config: SC fails the
//   fixture's three pairs, and only pairs inside its A'' x A'' block (the
//   amended acceptance, #812: the analytic gradient finds more of that
//   block than the fixture's energy stencil could); PC fails none. SC's step
//   then returns Numerical naming them, and hessian.json carries the ladders
//   and no frequencies.
// - The refusals (fast): the step's own (the ladder under the analytic
//   method among them), run directly, each returned before any displaced
//   solve. The config's (quasi_degenerate, a frozen core, a state average,
//   integral-direct ERIs, a [casscf_gradient] beside it) touch no device, and
//   are test/pipeline/hessian_host.cpp's. The analytic method itself is
//   pipeline_nevpt2_analytic_hessian_gpu.cpp's (#818).

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

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kSkew = "h2co_skew_sto3g";
constexpr std::string_view kSymmetric = "nh3_cs_sto3g";

// The outer CASSCF and NEVPT2 energies against the fixture's: the same
// solution (PySCF's, polished to 1e-12, against a CASSCF to 1e-10).
constexpr double kEnergyTolerance = 1e-8;
// The ratio window: 16 +- 20%, the fixture's and the step's.
constexpr double kRatioLow = 12.8;
constexpr double kRatioHigh = 19.2;
// The finest rung's Hessian against the fixture's: within this many times
// the sum of the two bars, each its ladder's own error estimate.
constexpr double kBarFactor = 2.0;

// The SCF and CASSCF converged tight: each gradient's noise is divided by h.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

const fb::json &fixture(std::string_view name = kSkew) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto found = docs.find(name); found != docs.end()) {
    return found->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{name}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

// The fixture @p name's molecule.xyz in @p dir (bohr there, Angstrom here),
// and run.toml over its basis, fit and active space: the SCF @p scf, the
// CASSCF, [nevpt2_gradient] of @p variant with @p extra keys and, unless
// @p hessian is empty, a [hessian] table of that body.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, std::string_view variant,
                                   std::string_view hessian, std::string_view scf = kScf,
                                   std::string_view extra = "", std::string_view name = kSkew) {
  const fb::json &doc = fixture(name);
  std::string xyz = std::format("{}\n{}\n", doc.at("atoms").size(), name);
  for (const fb::json &atom : doc.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string active;
  for (const auto t : doc.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\nconv_tol = 1e-12\nconv_tol_grad = 1e-10\n"
      "[nevpt2_gradient]\nvariant = \"{}\"\n{}",
      doc.at("charge").get<std::int32_t>(), doc.at("basis").at("name").get<std::string>(),
      doc.at("aux_basis").at("name").get<std::string>(), scf, doc.at("ncas").get<std::int32_t>(),
      doc.at("nelecas").get<std::int32_t>(), active, variant, extra);
  if (!hessian.empty()) {
    toml += std::format("[hessian]\n{}", hessian);
  }
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// @p calc's NEVPT2 total energy at its molecule.
double nevpt2_energy(const fb::Calculation &calc) {
  const auto &cas = calc.get<fb::CasscfOutput>();
  const auto &out = calc.get<fb::Nevpt2GradientOutput>();
  EXPECT_TRUE(cas.has_value() && out.has_value());
  if (!cas.has_value() || !out.has_value()) {
    return 0.0;
  }
  return std::visit(
      [&]<class G>(const G &g) -> double {
        if constexpr (std::same_as<G, fb::nevpt2::ScGradient> ||
                      std::same_as<G, fb::nevpt2::PcGradient>) {
          return cas->energies[0] + g.energies.total;
        } else {
          ADD_FAILURE() << "a QD gradient";
          return 0.0;
        }
      },
      out->gradient);
}

// The [hessian] table of the ladder at the fixture @p doc's h, over every
// pair by default, in its coordinates (when they are not the identity).
std::string ladder_table(const fb::json &doc) {
  std::string table = std::format("method = \"numerical\"\ndisplacement = {}\n",
                                  doc.at("steps").at("h").get<double>());
  const auto vectors = doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>();
  bool identity = true;
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    for (std::size_t j = 0; j < vectors.size(); ++j) {
      identity = identity && vectors[i][j] == (i == j ? 1.0 : 0.0);
    }
  }
  if (!identity) {
    table += "coordinates = [\n";
    for (const auto &row : vectors) {
      table += "  [";
      for (std::size_t x = 0; x < row.size(); ++x) {
        table += std::format("{}{:.17g}", x == 0 ? "" : ", ", row[x]);
      }
      table += "],\n";
    }
    table += "]\n";
  }
  return table;
}

// The run of the fixture @p name's @p variant through the binary's path, its
// outer energies held to the fixture's; @return the run's status and its
// hessian.json.
std::pair<fb::Status, fb::json> run_ladder(std::string_view name, std::string_view variant) {
  const fb::json &doc = fixture(name);
  const fb::test::ScratchDir dir(std::format("pipeline_nevpt2_hessian_{}_{}", name, variant));
  auto read = fb::read_config(write_config(dir, variant, ladder_table(doc), kScf, "", name),
                              dir.path(), dir / "out");
  EXPECT_TRUE(read.has_value()) << read.error().message;
  if (!read.has_value()) {
    return {fb::Status{}, fb::json{}};
  }
  fb::Calculation calc = std::move(*read);
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto &cas = calc.get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (cas.has_value()) {
    EXPECT_NEAR(cas->energy, doc.at("energies").at("casscf").get<double>(), kEnergyTolerance);
    EXPECT_NEAR(nevpt2_energy(calc), doc.at("energies").at(variant).get<double>(),
                kEnergyTolerance);
  }
  auto written = fb::read_json(dir / "out" / "hessian.json");
  EXPECT_TRUE(written.has_value()) << written.error().message;
  fb::json out = written.has_value() ? std::move(*written) : fb::json{};
  if (!out.is_null()) {
    // The measured solve noise lies under its first-order bound.
    EXPECT_LE(out.at("ladder").at("noise").at("sigma_solves").get<double>(),
              out.at("ladder").at("noise").at("solve_bound").get<double>());
    // Each checked pair's numbers, the worst first.
    std::vector<fb::json> pairs(out.at("bilinearity").begin(), out.at("bilinearity").end());
    std::ranges::sort(pairs, std::greater<>{}, [](const fb::json &p) {
      return std::abs(p.at("residual").get<double>()) / p.at("allowed").get<double>();
    });
    std::string worst;
    for (std::size_t k = 0; k < std::min<std::size_t>(pairs.size(), 8); ++k) {
      const fb::json &p = pairs[k];
      const auto ladder = p.at("ladder").get<std::vector<double>>();
      worst += std::format(
          "\n  {} residual {:+.2e} allowed {:.2e} (x{:.2f}) ratio {} upper {:+.2e} "
          "lower {:+.2e}",
          p.at("pair").dump(), p.at("residual").get<double>(), p.at("allowed").get<double>(),
          std::abs(p.at("residual").get<double>()) / p.at("allowed").get<double>(),
          p.at("ratio").dump(), ladder[0] - ladder[1], ladder[1] - ladder[2]);
    }
    const fb::json &noise = out.at("ladder").at("noise");
    std::cout << std::format(
        "[ nevpt2 hessian ] {} {}: {:.0f} s, sigma_g {:.2e} from the {} "
        "(restarts {:.2e}, solves {:.2e}, bound {:.2e} = residual {:.2e} x "
        "response {:.2e}), floor {:.2e}, failing {}, unsettled {}; worst "
        "pairs:{}\n",
        name, variant, seconds, noise.at("sigma_g").get<double>(),
        noise.at("from").get<std::string>(), noise.at("sigma_restarts").get<double>(),
        noise.at("sigma_solves").get<double>(), noise.at("solve_bound").get<double>(),
        noise.at("solve_residual").get<double>(), noise.at("response").get<double>(),
        noise.at("floor").get<double>(), out.at("failing_pairs").dump(),
        out.at("ladder").at("unsettled").dump(), worst);
  }
  return {std::move(ran), std::move(out)};
}

// The skew case: no pair fails, the frequencies are published, the judged
// ratios settle, and the Hessian is the fixture's to the bars.
void expect_skew(std::string_view variant) {
  const fb::json &doc = fixture(kSkew);
  ASSERT_FALSE(doc.is_null());
  const auto [ran, out] = run_ladder(kSkew, variant);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_FALSE(out.is_null());
  const std::size_t n = 3 * doc.at("atoms").size();
  EXPECT_EQ(out.at("variant").get<std::string>(), variant);
  EXPECT_TRUE(out.at("failing_pairs").empty()) << out.at("failing_pairs").dump();
  EXPECT_EQ(out.at("bilinearity").size(), n * (n - 1) / 2);
  EXPECT_EQ(out.at("frequencies").size(), n - 6);
  // C1, Cartesian: the fixture's matrices are per Cartesian element.
  for (const auto &irrep : doc.at("coordinates").at("irreps")) {
    ASSERT_EQ(irrep.get<std::string>(), "A");
  }
  const fb::json &ladder = out.at("ladder");
  const auto ratios = ladder.at("ratios");
  const auto bar = ladder.at("bar").get<std::vector<std::vector<double>>>();
  const auto got = out.at("hessian").get<std::vector<std::vector<double>>>();
  const auto want = doc.at(variant).at("hessian").get<std::vector<std::vector<double>>>();
  const auto want_bar = doc.at(variant).at("bar").get<std::vector<std::vector<double>>>();
  std::size_t judged = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (!ratios[i][j].is_null()) {
        ++judged;
        EXPECT_GT(ratios[i][j].get<double>(), kRatioLow)
            << variant << " (" << i << ", " << j << ")";
        EXPECT_LT(ratios[i][j].get<double>(), kRatioHigh)
            << variant << " (" << i << ", " << j << ")";
      }
      const double allowed = kBarFactor * (want_bar[i][j] + std::max(bar[i][j], bar[j][i]));
      EXPECT_NEAR(got[i][j], want[i][j], allowed) << variant << " (" << i << ", " << j << ")";
      worst = std::max(worst, std::abs(got[i][j] - want[i][j]) / allowed);
    }
  }
  EXPECT_TRUE(ladder.at("unsettled").empty()) << ladder.at("unsettled").dump();
  // A ladder that judges nothing checks nothing.
  EXPECT_GT(judged, 0U);
  std::cout << std::format("[ nevpt2 hessian ] {} {}: {} ratios judged; worst |H - fixture| / "
                           "allowed {:.3f}\n",
                           kSkew, variant, judged, worst);
}

// The symmetric case: the fixture's failing pairs fail, and only pairs in
// its failed blocks; a failure withholds the frequencies, as Numerical
// naming the pairs.
void expect_symmetric(std::string_view variant) {
  const fb::json &doc = fixture(kSymmetric);
  ASSERT_FALSE(doc.is_null());
  const auto [ran, out] = run_ladder(kSymmetric, variant);
  ASSERT_FALSE(out.is_null());
  const auto want =
      doc.at(variant).at("failing_pairs").get<std::vector<std::array<std::size_t, 2>>>();
  const auto got = out.at("failing_pairs").get<std::vector<std::array<std::size_t, 2>>>();
  // The fixture's failing pairs fail here too...
  for (const auto &pair : want) {
    EXPECT_TRUE(std::ranges::contains(got, pair))
        << variant << " (" << pair[0] << ", " << pair[1] << ") passed; failing "
        << out.at("failing_pairs").dump();
  }
  // ... and every pair that fails lies in a block the fixture recorded as
  // failed (SC's A'' x A''): every other pair passes. The analytic gradient's
  // bars are far below the fixture's there, so more of that block's pairs
  // fail than the fixture's three (README.md, "The ladder and the
  // bilinearity check").
  const auto irreps = doc.at("coordinates").at("irreps").get<std::vector<std::string>>();
  const auto blocks =
      doc.at(variant).at("failed_blocks").get<std::vector<std::vector<std::string>>>();
  for (const auto &[a, b] : got) {
    std::vector<std::string> block{irreps[a], irreps[b]};
    std::ranges::sort(block);
    EXPECT_TRUE(std::ranges::contains(blocks, block))
        << variant << " (" << a << ", " << b << ") fails outside the fixture's failed blocks";
  }
  EXPECT_EQ(out.at("coordinates"), doc.at("coordinates").at("vectors"));
  if (want.empty()) {
    EXPECT_TRUE(got.empty()) << out.at("failing_pairs").dump();
    ASSERT_TRUE(ran.has_value()) << ran.error().message;
    EXPECT_TRUE(out.contains("frequencies"));
    return;
  }
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::Numerical) << ran.error().message;
  for (const auto &[a, b] : got) {
    EXPECT_NE(ran.error().message.find(std::format("[{}, {}]", a, b)), std::string::npos)
        << ran.error().message;
  }
  EXPECT_FALSE(out.contains("frequencies"));
  EXPECT_FALSE(out.contains("modes"));
  EXPECT_TRUE(out.contains("hessian"));
}

TEST(PipelineNevpt2Hessian, ScSkewLadderIsBilinearAndMatchesFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("sc");
}

TEST(PipelineNevpt2Hessian, PcSkewLadderIsBilinearAndMatchesFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("pc");
}

TEST(PipelineNevpt2Hessian, ScSymmetricFailsOnlyItsAPrimePrimeBlock) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_symmetric("sc");
}

TEST(PipelineNevpt2Hessian, PcSymmetricIsBilinear) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_symmetric("pc");
}

// The step's own refusals, run directly on a planned step changed after
// planning: each returned at once, before the reference CASSCF or any
// displaced solve, naming its reason.
TEST(PipelineNevpt2Hessian, StepRefusesBeforeAnyDisplacedSolve) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_nevpt2_hessian_step");
  auto read =
      fb::read_config(write_config(dir, "sc", "method = \"numerical\"\n"), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::HessianStep planned = *read->step<fb::HessianStep>();
  planned.json_path.clear();
  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  auto molecule = fb::MoleculeStep{.xyz = dir / "molecule.xyz"}.run(fb::View<>{}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

  const auto refused = [&](const fb::HessianStep &step, fb::ErrorKind kind, std::string_view says) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto out = step.run(fb::View<fb::MoleculeOutput>{*molecule}, node);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ASSERT_FALSE(out.has_value()) << says;
    EXPECT_EQ(out.error().kind, kind) << out.error().message;
    EXPECT_NE(out.error().message.find(says), std::string::npos) << out.error().message;
    // Nothing ran: no geometry is named.
    EXPECT_FALSE(out.error().message.starts_with("[hessian] at ")) << out.error().message;
    std::cout << std::format("[ nevpt2 hessian ] refused in {:.1e} s: {}\n", seconds,
                             out.error().message);
  };
  {
    fb::HessianStep step = planned;
    step.eri.mode = fb::EriMode::IntegralDirect;
    refused(step, fb::ErrorKind::Unsupported, "integral_direct");
  }
  {
    fb::HessianStep step = planned;
    step.casscf.nroots = 2;
    step.casscf.weights = {0.5, 0.5};
    refused(step, fb::ErrorKind::Unsupported, "nroots is 2");
  }
  {
    fb::HessianStep step = planned;
    step.nevpt2_gradient->quasi_degenerate = true;
    refused(step, fb::ErrorKind::Unsupported, "quasi_degenerate");
  }
  {
    fb::HessianStep step = planned;
    step.nevpt2_gradient->frozen_core = 2;
    refused(step, fb::ErrorKind::InvalidConfig, "frozen_core is 2");
  }
  {
    fb::HessianStep step = planned;
    // The planned numerical step's ladder, under the analytic method.
    step.method = fb::HessianMethod::Analytic;
    refused(step, fb::ErrorKind::InvalidConfig, "ladder and pairs are the numerical method's");
  }
  {
    fb::HessianStep step = planned;
    step.casscf_gradient.emplace();
    refused(step, fb::ErrorKind::InvalidConfig, "both");
  }
}

} // namespace
