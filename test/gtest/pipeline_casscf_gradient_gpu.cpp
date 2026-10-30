// GPU cases for the CasscfGradient step through the real registry: TOML
// configs read, planned and run by Calculation -- the DF-RHF, the DF-CASSCF
// from its orbitals and the gradient of its root -- their
// casscf_gradient.json against PySCF's df.grad.casscf
// (data/casscf_gradient/casscf_gradient.json), the root's energy, the total
// and every term to 1e-8, for each state-specific case (H2O and the skewed
// H2O, cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4)); and a root the CASSCF does not
// have, refused with nothing written. One [eri] serves the whole pipeline, so
// the CASSCF starts from the DF-RHF's orbitals where the reference's starts
// from the exact RHF's: the same active orbitals, and the converged DF-CASSCF
// is its start's only through which stationary point it finds, as in
// pipeline_casscf_gpu.cpp's density-fitted cases. No state-averaged root:
// mcscf::df_gradient does not take one yet.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar, as mcscf_gradient_gpu.cpp holds mcscf::df_gradient.
constexpr double kTermTolerance = 1e-8;
// The root's energy against PySCF's, as pipeline_casscf_gpu.cpp holds it.
constexpr double kEnergyTolerance = 1e-8;
// The CASSCF's convergence. The formula takes W for the orbitals' response,
// so its error is first order in ||g||: at 1e-9, as mcscf_gradient_gpu.cpp
// converges it, ~1e-9 in the gradient.
constexpr double kConvTolGrad = 1e-9;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_gradient.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule as molecule.xyz in @p dir (bohr there, Angstrom
// here), and run.toml naming it in the reference's basis, over its fitting
// set, with its active space in [casscf] and @p casscf_gradient after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view casscf_gradient) {
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string active;
  for (const auto t : entry.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\nconv_tol_grad = {:g}\n{}",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(),
      entry.at("aux_basis").at("name").get<std::string>(), entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), active, kConvTolGrad, casscf_gradient);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// @p ours, casscf_gradient.json's rows, against @p ref's to kTermTolerance;
// the worst error, for the record a PR reports.
double compare(const std::string &what, const fb::json &ours, const fb::json &ref) {
  const auto got = ours.get<std::vector<std::vector<double>>>();
  const auto want = ref.get<std::vector<std::vector<double>>>();
  EXPECT_EQ(got.size(), want.size()) << what;
  if (got.size() != want.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t a = 0; a < want.size(); ++a) {
    EXPECT_EQ(got[a].size(), 3U) << what << " [" << a << "]";
    for (std::size_t k = 0; k < 3 && k < got[a].size(); ++k) {
      EXPECT_NEAR(got[a][k], want[a][k], kTermTolerance) << what << " [" << a << "][" << k << "]";
      worst = std::max(worst, std::abs(got[a][k] - want[a][k]));
    }
  }
  return worst;
}

// The case's config run by Calculation: its casscf_gradient.json against the
// reference's root 0, the energy, the total and every part to 1e-8; the worst
// error printed, for the record a PR reports.
void expect_matches(std::string_view name) {
  const fb::json &entry = reference_case(name);
  ASSERT_EQ(entry.at("weights").size(), 1U) << "a state-specific case";
  const fb::json &expected = entry.at("roots").at(0);
  const fb::test::ScratchDir dir("pipeline_casscf_gradient");
  auto calc =
      fb::read_config(write_config(dir, entry, "[casscf_gradient]\n"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto written = fb::read_json(dir / "out" / "casscf_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("root").get<std::int32_t>(), 0);
  EXPECT_NEAR(written->at("energy").get<double>(), expected.at("energy").get<double>(),
              kEnergyTolerance);
  const double total = compare("total", written->at("total"), expected.at("total"));
  std::string report = std::format("[ casscf_gradient ] {}: total {:.2e}", name, total);
  double worst = total;
  for (const auto &item : expected.at("parts").items()) {
    const std::string &key = item.key();
    const double err = compare(key, written->at("terms").at(key), item.value());
    report += std::format(" {} {:.2e}", key, err);
    worst = std::max(worst, err);
  }
  EXPECT_EQ(written->at("terms").size(), expected.at("parts").size());
  std::cout << report << std::format(", max {:.2e}\n", worst);

  // The product is what the file holds.
  const auto &out = calc->get<fb::CasscfGradientOutput>();
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->gradient.total.size(), 3 * entry.at("atoms").size());
}

class PipelineCasscfGradient : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineCasscfGradient, JsonMatchesPyscf) {
  FBF_REQUIRE_GPU();
  expect_matches(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasscfGradient,
                         ::testing::Values("h2o_ccpvdz", "h2o_skew_ccpvdz"),
                         [](const auto &info) { return std::string(info.param); });

// A root the single-root CASSCF does not have: df_gradient's InvalidConfig is
// the step's, and the run's, with no casscf_gradient.json written.
TEST(PipelineCasscfGradient, ARootTheCasscfLacksIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_casscf_gradient_root");
  auto calc = fb::read_config(
      write_config(dir, reference_case("h2o_ccpvdz"), "[casscf_gradient]\nroot = 1\n"), dir.path(),
      dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::InvalidConfig) << ran.error().message;
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "casscf_gradient.json"));
}

} // namespace
