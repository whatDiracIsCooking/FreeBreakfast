// The numerical [hessian] over the [nevpt2_gradient] chain (#804): central
// differences of nevpt2::sc_gradient / pc_gradient over the 6N displaced
// DF-CASSCF solves, against #803's semi-numerical reference,
// data/nevpt2_hessian/h2co_skew_sto3g.json (skew H2CO, STO-3G /
// cc-pVDZ-JKFIT, CAS(4,4), every block defined; its README has the setup).
//
// - The ladder (slow-gated, SC and PC): the config's Calculation at h, then
//   the planned step again at h/2 and h/4. Per element the ratio of
//   successive differences, (D(h) - D(h/2)) / (D(h/2) - D(h/4)), settles
//   near 4 (the central difference's h²) wherever the finer difference is
//   above the noise floor (below: printed, never judged); the Richardson
//   extrapolant of the two finer steps, R = (4 D(h/4) - D(h/2)) / 3, within
//   2 x (the fixture's bar + R's own estimate |R - R(h, h/2)|) of the
//   fixture's Hessian. Each Calculation's outer CASSCF and NEVPT2 energies
//   are first held to the fixture's, so the differences are taken around its
//   solution.
// - The refusals (fast): the config's (the analytic method, quasi_degenerate,
//   a frozen core, a state average, integral-direct ERIs, a [casscf_gradient]
//   beside it) and the step's own, run directly, each returned before any
//   displaced solve.

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

const std::filesystem::path kFixture =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" / "h2co_skew_sto3g.json";

// The ladder's coarsest step (bohr): the fixture's h. The fixture differences
// to 2h with a fourth-order stencil and its README found SC smooth there;
// the central difference here reaches h, h/2 and h/4 only. Its h² error is
// ~1e-4 at h and its h⁴ remainder after Richardson ~1e-9, under the bars.
constexpr double kStep = 0.03;
// The outer CASSCF and NEVPT2 energies against the fixture's: the same
// solution (PySCF's, polished to 1e-12, against a CASSCF to 1e-10).
constexpr double kEnergyTolerance = 1e-8;
// The ratio window: 4 +- 20%, the fixture's own (16 +- 20%) for the
// second-order stencil.
constexpr double kRatioLow = 3.2;
constexpr double kRatioHigh = 4.8;
// A finer difference is judged only above this many times its noise, so the
// noise moves the ratio by under 4 / kFloorFactor = 0.2 (the window's half
// width is 0.8): the fixture's floor_factor.
constexpr double kFloorFactor = 20.0;
// The issue's bar: within 2 x (the fixture's bar + the estimate).
constexpr double kBarFactor = 2.0;

// The SCF and CASSCF converged tight: each gradient's noise is divided by 2h.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";
constexpr std::string_view kLooseScf = "[scf]\nenergy_tolerance = 1e-6\n";

const fb::json &fixture() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(kFixture);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

// The fixture's molecule.xyz in @p dir (bohr there, Angstrom here), and
// run.toml over its basis, fit and active space: the SCF @p scf, the
// CASSCF(4,4), [nevpt2_gradient] of @p variant with @p extra keys and,
// unless @p hessian is empty, a [hessian] table of that body.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, std::string_view variant,
                                   std::string_view hessian, std::string_view scf = kScf,
                                   std::string_view extra = "") {
  const fb::json &doc = fixture();
  std::string xyz = std::format("{}\n{}\n", doc.at("atoms").size(), "h2co_skew_sto3g");
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

// @p calc's NEVPT2 total energy and gradient at its molecule.
std::pair<double, std::vector<double>> nevpt2_at(const fb::Calculation &calc) {
  const auto &cas = calc.get<fb::CasscfOutput>();
  const auto &out = calc.get<fb::Nevpt2GradientOutput>();
  EXPECT_TRUE(cas.has_value() && out.has_value());
  if (!cas.has_value() || !out.has_value()) {
    return {};
  }
  return std::visit(
      [&]<class G>(const G &g) -> std::pair<double, std::vector<double>> {
        if constexpr (std::same_as<G, fb::nevpt2::ScGradient> ||
                      std::same_as<G, fb::nevpt2::PcGradient>) {
          return {cas->energies[0] + g.energies.total, g.terms.total};
        } else {
          ADD_FAILURE() << "a QD gradient";
          return {};
        }
      },
      out->gradient);
}

// The fixture's Hessian of @p scheme, its bar, in Cartesians: Qᵀ M Q, Q's
// rows the fixture's coordinates (the identity for this C1 case).
std::vector<double> cartesian(const fb::json &matrix, const fb::json &vectors) {
  const auto m = matrix.get<std::vector<std::vector<double>>>();
  const auto q = vectors.get<std::vector<std::vector<double>>>();
  const std::size_t n = m.size();
  std::vector<double> out(n * n, 0.0);
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          out[(x * n) + y] += q[i][x] * m[i][j] * q[j][y];
        }
      }
    }
  }
  return out;
}

// The step's Hessian of @p variant ("sc", "pc") at h, h/2 and h/4 on the
// fixture, against its Hessian of that scheme (the issue's acceptance).
void expect_ladder(std::string_view variant) {
  const fb::json &doc = fixture();
  ASSERT_FALSE(doc.is_null());
  const std::size_t n = 3 * doc.at("atoms").size();
  const fb::json &scheme = doc.at(variant);
  const std::vector<double> want =
      cartesian(scheme.at("hessian"), doc.at("coordinates").at("vectors"));
  // A bar is not a matrix to rotate: per element, it is the identity's here.
  ASSERT_EQ(doc.at("coordinates").at("labels").size(), n);
  for (const auto &row : doc.at("coordinates").at("irreps")) {
    ASSERT_EQ(row.get<std::string>(), "A") << "the bar is per Cartesian element only in C1";
  }
  std::vector<double> bar(n * n);
  const auto bar_rows = scheme.at("bar").get<std::vector<std::vector<double>>>();
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      bar[(x * n) + y] = bar_rows[x][y];
    }
  }

  const fb::test::ScratchDir dir(std::format("pipeline_nevpt2_hessian_{}", variant));
  auto read = fb::read_config(
      write_config(dir, variant, std::format("method = \"numerical\"\ndisplacement = {}\n", kStep)),
      dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  fb::NodeResources node;
  auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc.run(node);
  std::vector<double> seconds{
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()};
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  // At the fixture's solution.
  const auto &cas = calc.get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  EXPECT_NEAR(cas->energy, doc.at("energies").at("casscf").get<double>(), kEnergyTolerance);
  const auto [energy, gradient] = nevpt2_at(calc);
  EXPECT_NEAR(energy, doc.at("energies").at(variant).get<double>(), kEnergyTolerance);

  // The gradient's noise sigma_g, measured: the same chain from the SCF
  // converged to 1e-6, a CASSCF started elsewhere and converged to the same
  // conv_tol_grad; max |g - g'| over the components.
  double sigma = 0.0;
  {
    const fb::test::ScratchDir loose(std::format("pipeline_nevpt2_hessian_{}_noise", variant));
    auto other =
        fb::read_config(write_config(loose, variant, "", kLooseScf), loose.path(), loose / "out");
    ASSERT_TRUE(other.has_value()) << other.error().message;
    const fb::Status other_ran = other->run(node);
    ASSERT_TRUE(other_ran.has_value()) << other_ran.error().message;
    const auto [e_other, g_other] = nevpt2_at(*other);
    ASSERT_EQ(g_other.size(), gradient.size());
    for (std::size_t i = 0; i < gradient.size(); ++i) {
      sigma = std::max(sigma, std::abs(gradient[i] - g_other[i]));
    }
  }

  // D(h) from the Calculation, D(h/2) and D(h/4) from the planned step
  // again at its molecule, writing nothing.
  std::vector<std::vector<double>> ladder;
  std::vector<double> asymmetry;
  const auto &first = calc.get<fb::HessianOutput>();
  ASSERT_TRUE(first.has_value()) << first.error().message;
  ASSERT_EQ(first->hessian.size(), n * n);
  ladder.push_back(first->hessian);
  asymmetry.push_back(first->asymmetry);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), "numerical");
  EXPECT_EQ(written->at("variant").get<std::string>(), variant);
  EXPECT_EQ(written->at("displacement").get<double>(), kStep);
  const auto &molecule = calc.get<fb::MoleculeOutput>();
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  for (const double h : {kStep / 2.0, kStep / 4.0}) {
    fb::HessianStep step = *calc.step<fb::HessianStep>();
    step.displacement = h;
    step.json_path.clear();
    t0 = std::chrono::steady_clock::now();
    const auto out = step.run(fb::View<fb::MoleculeOutput>{*molecule}, node);
    seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    ASSERT_TRUE(out.has_value()) << out.error().message;
    ladder.push_back(out->hessian);
    asymmetry.push_back(out->asymmetry);
  }

  // Per element: the ratio where the finer difference is above its noise,
  // sqrt(10) sigma_g / h (D(t)'s noise is sigma_g / (sqrt(2) t), at t = h/2
  // and h/4); the extrapolant and its estimate against the fixture.
  const double floor = kFloorFactor * std::sqrt(10.0) * sigma / kStep;
  std::size_t judged = 0;
  double ratio_low = std::numeric_limits<double>::infinity();
  double ratio_high = -std::numeric_limits<double>::infinity();
  std::string unsettled;
  double worst = 0.0; // |R - fixture| / (bar + estimate), the largest: under 2
  std::size_t worst_at = 0;
  double worst_estimate = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      const std::size_t xy = (x * n) + y;
      const double coarse = ladder[0][xy] - ladder[1][xy];
      const double fine = ladder[1][xy] - ladder[2][xy];
      if (std::abs(fine) > floor) {
        const double ratio = coarse / fine;
        ++judged;
        ratio_low = std::min(ratio_low, ratio);
        ratio_high = std::max(ratio_high, ratio);
        EXPECT_GT(ratio, kRatioLow) << variant << " (" << x << ", " << y << ")";
        EXPECT_LT(ratio, kRatioHigh) << variant << " (" << x << ", " << y << ")";
      } else if (x <= y) {
        unsettled += std::format(" ({}, {}) {:.1e}/{:.1e}", x, y, coarse, fine);
      }
      const double extrapolant = ((4.0 * ladder[2][xy]) - ladder[1][xy]) / 3.0;
      const double estimate =
          std::abs(extrapolant - (((4.0 * ladder[1][xy]) - ladder[0][xy]) / 3.0));
      const double allowed = kBarFactor * (bar[xy] + estimate);
      EXPECT_NEAR(extrapolant, want[xy], allowed) << variant << " (" << x << ", " << y << ")";
      worst_estimate = std::max(worst_estimate, estimate);
      if (const double ratio = std::abs(extrapolant - want[xy]) / (bar[xy] + estimate);
          ratio > worst) {
        worst = ratio;
        worst_at = xy;
      }
    }
  }
  std::cout << std::format(
      "[ nevpt2 hessian ] {}: ladder h = {}, {}, {} in {::.1f} s; sigma_g {:.1e}, floor "
      "{:.1e}; {} of {} ratios judged, in [{:.3f}, {:.3f}]; worst |R - fixture| / (bar + "
      "estimate) {:.3f} (bar 2) at ({}, {}); largest estimate {:.1e}; asymmetry {::.1e}\n",
      variant, kStep, kStep / 2, kStep / 4, seconds, sigma, floor, judged, n * n, ratio_low,
      ratio_high, worst, worst_at / n, worst_at % n, worst_estimate, asymmetry);
  if (!unsettled.empty()) {
    std::cout << std::format("[ nevpt2 hessian ] {}: below the floor, not judged (D(h) - D(h/2) "
                             "/ D(h/2) - D(h/4)):{}\n",
                             variant, unsettled);
  }
}

TEST(PipelineNevpt2Hessian, ScLadderMatchesFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_ladder("sc");
}

TEST(PipelineNevpt2Hessian, PcLadderMatchesFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_ladder("pc");
}

// The config's refusals, planned and not run: each before anything touches a
// device.
TEST(PipelineNevpt2Hessian, ConfigRefusesWhatTheHessianDoesNotCover) {
  const fb::test::ScratchDir dir("pipeline_nevpt2_hessian_config");
  const auto planned = [&](std::string_view hessian, std::string_view extra = "",
                           std::string_view eri = "density_fitting", std::int32_t nroots = 1) {
    const std::filesystem::path path = write_config(dir, "pc", hessian, kScf, extra);
    auto text = fb::read_file(path);
    EXPECT_TRUE(text.has_value());
    std::string toml = text.has_value() ? *text : std::string{};
    const auto replace = [&](std::string_view from, std::string_view to) {
      const std::size_t at = toml.find(from);
      ASSERT_NE(at, std::string::npos) << from;
      toml.replace(at, from.size(), to);
    };
    if (eri != "density_fitting") {
      // An aux basis is refused beside the other mode.
      if (const std::size_t at = toml.find("aux_basis = "); at != std::string::npos) {
        toml.erase(at, toml.find('\n', at) + 1 - at);
      }
      replace("density_fitting", eri);
    }
    if (nroots != 1) {
      replace("conv_tol_grad", std::format("nroots = {}\nconv_tol_grad", nroots));
    }
    return fb::parse_config(toml, dir.path(), dir / "out");
  };
  constexpr std::string_view kNumerical = "method = \"numerical\"\n";

  // The chain copied in: the CASSCF and the gradient, the variant kept.
  const auto ok = planned(kNumerical);
  ASSERT_TRUE(ok.has_value()) << ok.error().message;
  const auto &step = ok->step<fb::HessianStep>();
  ASSERT_TRUE(step.has_value());
  ASSERT_TRUE(step->nevpt2_gradient.has_value());
  EXPECT_EQ(step->nevpt2_gradient->variant, fb::Nevpt2Variant::Pc);
  EXPECT_FALSE(step->casscf_gradient.has_value());
  EXPECT_EQ(step->casscf.ncas, 4);
  EXPECT_EQ(step->method, fb::HessianMethod::Numerical);

  const auto refused = [](const fb::Result<fb::Calculation> &calc, fb::ErrorKind kind,
                          std::string_view says) {
    ASSERT_FALSE(calc.has_value());
    EXPECT_EQ(calc.error().kind, kind) << calc.error().message;
    EXPECT_NE(calc.error().message.find(says), std::string::npos) << calc.error().message;
  };
  // The analytic method, by default (an empty table) and by name.
  refused(planned("\n"), fb::ErrorKind::Unsupported, "numerical only");
  refused(planned("method = \"analytic\"\n"), fb::ErrorKind::Unsupported, "numerical only");
  refused(planned(kNumerical, "quasi_degenerate = true\n", "density_fitting", 2),
          fb::ErrorKind::Unsupported, "quasi_degenerate");
  refused(planned(kNumerical, "frozen_core = 1\n"), fb::ErrorKind::InvalidConfig, "frozen_core");
  refused(planned(kNumerical, "frozen_core = \"auto\"\n"), fb::ErrorKind::InvalidConfig,
          "frozen_core");
  // A state average: [nevpt2_gradient]'s own refusal, before anything runs.
  refused(planned(kNumerical, "", "density_fitting", 2), fb::ErrorKind::Unsupported, "nroots");
  refused(planned(kNumerical, "", "integral_direct"), fb::ErrorKind::Unsupported,
          "integral_direct");
  refused(planned(kNumerical, "[casscf_gradient]\n"), fb::ErrorKind::InvalidConfig, "names both");
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
    step.method = fb::HessianMethod::Analytic;
    refused(step, fb::ErrorKind::Unsupported, "numerical only");
  }
  {
    fb::HessianStep step = planned;
    step.casscf_gradient.emplace();
    refused(step, fb::ErrorKind::InvalidConfig, "both");
  }
}

} // namespace
