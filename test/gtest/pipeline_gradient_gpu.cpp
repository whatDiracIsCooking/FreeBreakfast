// GPU cases for the Gradient step through the real registry: TOML configs
// for H2O in cc-pVDZ read, planned and run by Calculation, their gradient.json
// against PySCF's (data/gradient/gradient.json, test/gtest/data/README.md),
// the total and every term to 1e-8: density-fitted over cc-pVDZ-JKFIT against
// df.grad.rhf (df_rhf's h2o_ccpvdz), and integral-direct against
// scf.grad.rhf (rhf's h2o_ccpvdz); and the ROHF refusal, naming its reason.

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

// The issue's bar, as df_rhf_gradient_gpu.cpp holds gradient::df_rhf.
constexpr double kTermTolerance = 1e-8;
// The SCF energy against PySCF's, as pipeline_df_scf_gpu.cpp holds it.
constexpr double kEnergyTolerance = 1e-8;

// W stands in for the orbitals' response only at convergence, so the SCF is
// converged well past its defaults, as df_rhf_gradient_gpu.cpp converges it.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

// @p name in gradient.json's @p section: df_rhf or rhf.
const fb::json &reference_case(std::string_view section, std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient" / "gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at(section)) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no " << section << " case " << name;
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule as molecule.xyz in @p dir (bohr there, Angstrom
// here), and run.toml naming it, with @p molecule_extra in its table, in the
// reference's basis, with @p eri and @p scf before [gradient].
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view molecule_extra, std::string_view eri,
                                   std::string_view scf) {
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
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n{}[basis_set]\nbasis = \"{}\"\n{}{}"
                  "[gradient]\n",
                  molecule_extra, entry.at("basis").get<std::string>(), eri, scf);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

std::string df_eri(const fb::json &entry) {
  return std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                     entry.at("aux_basis").get<std::string>());
}

// @p ours, gradient.json's rows, against @p ref's to kTermTolerance; the
// worst error, for the record a PR reports.
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

// @p config, run by Calculation in @p dir: its gradient.json against
// @p expected, the energy and every row to 1e-8; the worst error printed
// under @p tag, for the record a PR reports.
void expect_gradient_json(const fb::test::ScratchDir &dir, const std::filesystem::path &config,
                          const fb::json &expected, std::string_view tag) {
  auto calc = fb::read_config(config, dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto written = fb::read_json(dir / "out" / "gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_NEAR(written->at("energy").get<double>(), expected.at("energy").get<double>(),
              kEnergyTolerance);
  std::string report = std::format("[ gradient ] {} gradient.json:", tag);
  double worst = compare("total", written->at("total"), expected.at("total"));
  report += std::format(" total {:.2e}", worst);
  for (const auto &item : expected.at("terms").items()) {
    const std::string &key = item.key();
    const double err = compare(key, written->at("terms").at(key), item.value());
    report += std::format(" {} {:.2e}", key, err);
    worst = std::max(worst, err);
  }
  std::cout << report << std::format(", max {:.2e}\n", worst);

  // The product is what the file holds.
  const auto &out = calc->get<fb::GradientOutput>();
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->gradient.total.size(), 3 * expected.at("atoms").size());
}

TEST(PipelineGradient, DfRhfGradientJsonMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("df_rhf", "h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_gradient");
  expect_gradient_json(dir, write_config(dir, expected, "", df_eri(expected), kScf), expected,
                       "df_rhf h2o_ccpvdz");
}

TEST(PipelineGradient, RhfGradientJsonMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("rhf", "h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_gradient_direct");
  // A missing [eri] is filled in integral-direct. Every Fock build is full,
  // so the SCF reaches kScf's commutator: the incremental builds stall short
  // of it.
  const std::string scf = std::format("{}fock_rebuild_interval = 1\n", kScf);
  expect_gradient_json(dir, write_config(dir, expected, "", "", scf), expected, "rhf h2o_ccpvdz");
}

// A refusal needs no tight SCF, so [scf] is at its defaults.
void expect_refused(std::string_view tag, std::string_view molecule_extra, std::string_view eri,
                    std::string_view needle) {
  const fb::json &expected = reference_case("df_rhf", "h2o_ccpvdz");
  const fb::test::ScratchDir dir(tag);
  const auto config = write_config(dir, expected, molecule_extra, eri, "");
  auto calc = fb::read_config(config, dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::Unsupported) << ran.error().message;
  EXPECT_NE(ran.error().message.find(needle), std::string::npos) << ran.error().message;
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "gradient.json"));
}

TEST(PipelineGradient, RefusesRohf) {
  FBF_REQUIRE_GPU();
  // H2O+, a doublet: ROHF over the same fit.
  expect_refused("pipeline_gradient_rohf", "charge = 1\nmultiplicity = 2\n",
                 df_eri(reference_case("df_rhf", "h2o_ccpvdz")), "ROHF");
}

} // namespace
