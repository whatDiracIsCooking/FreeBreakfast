// GPU case for the Hessian step through the real registry: a TOML config for
// the skew water (C1) in cc-pVDZ / cc-pVDZ-JKFIT, read, planned and run by
// Calculation, its hessian.json -- 18 DF-RHF SCF + gradient reruns
// differenced -- against PySCF's analytic DF-RHF Hessian over the same aux
// set (data/hessian_numerical/hessian_numerical.json,
// test/gtest/data/README.md) to 1e-5, and its asymmetry before symmetrizing
// and translational sum checked, each printed for the record a PR reports.

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

// The issue's bar against PySCF (hartree / bohr²).
constexpr double kHessianTolerance = 1e-5;
// The central differences' asymmetry and translational sum: each is the
// h² error and the SCF's noise over 2h, as the agreement is, so at the same
// bar.
constexpr double kInvarianceTolerance = 1e-5;

// The gradient is the SCF energy's only at convergence, and its noise is
// divided by 2h, so the SCF is converged as pipeline_gradient_gpu.cpp
// converges it.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "hessian_numerical" /
                              "hessian_numerical.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name;
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule as molecule.xyz in @p dir (bohr there, Angstrom
// here), and run.toml naming it, over the reference's basis and fit, with
// [hessian] at its default displacement.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry) {
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
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}[hessian]\nmethod = "
      "\"numerical\"\n",
      entry.at("basis").get<std::string>(), entry.at("aux_basis").get<std::string>(), kScf);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

TEST(PipelineHessian, NumericalDfRhfHessianMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_hessian");
  auto calc = fb::read_config(write_config(dir, expected), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), "numerical");
  const auto got = written->at("hessian").get<std::vector<std::vector<double>>>();
  const auto want = expected.at("hessian").get<std::vector<std::vector<double>>>();
  const std::size_t n = 3 * expected.at("atoms").size();
  ASSERT_EQ(got.size(), n);
  ASSERT_EQ(want.size(), n);
  double worst = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    ASSERT_EQ(got[x].size(), n) << "row " << x;
    for (std::size_t y = 0; y < n; ++y) {
      EXPECT_NEAR(got[x][y], want[x][y], kHessianTolerance) << "[" << x << "][" << y << "]";
      worst = std::max(worst, std::abs(got[x][y] - want[x][y]));
    }
  }

  // The product is what the file holds, and keeps the two identities.
  const auto &out = calc->get<fb::HessianOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;
  ASSERT_EQ(out->hessian.size(), n * n);
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      EXPECT_EQ(out->hessian[(x * n) + y], got[x][y]);
      EXPECT_EQ(out->hessian[(x * n) + y], out->hessian[(y * n) + x]);
    }
  }
  EXPECT_LT(out->asymmetry, kInvarianceTolerance);
  EXPECT_LT(out->translational_sum, kInvarianceTolerance);
  EXPECT_EQ(written->at("asymmetry").get<double>(), out->asymmetry);
  EXPECT_EQ(written->at("translational_sum").get<double>(), out->translational_sum);
  std::cout << std::format("[ hessian ] h2o_ccpvdz numerical vs pyscf.df.hessian.rhf: max {:.2e}, "
                           "asymmetry {:.2e}, translational sum {:.2e}\n",
                           worst, out->asymmetry, out->translational_sum);
}

} // namespace
