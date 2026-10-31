// GPU cases for the Hessian step through the real registry, TOML configs in
// cc-pVDZ / cc-pVDZ-JKFIT read, planned and run by Calculation:
//
// - the skew water (C1), by both methods: the numerical hessian.json -- 18
//   DF-RHF SCF + gradient reruns differenced -- against PySCF's analytic
//   DF-RHF Hessian over the same aux set
//   (data/hessian_numerical/hessian_numerical.json, test/gtest/data/README.md)
//   to 1e-5, the analytic one against it to 1e-7 and against the numerical
//   one to the central differences' error, and the two methods' frequencies
//   against each other; each asymmetry and translational sum checked;
// - the water optimized from data/gradient/gradient.json's start, then the
//   analytic [hessian] at the optimum: its harmonic frequencies against
//   PySCF's thermo.harmonic_analysis at PySCF's DF-RHF minimum
//   (data/rhf_hessian/rhf_hessian.json's h2o_ccpvdz_opt) to 0.1 cm^-1.
//
// Each prints its numbers for the record a PR reports.

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

// The numerical method's bar against PySCF (hartree / bohr²).
constexpr double kHessianTolerance = 1e-5;
// The analytic method's against PySCF, hessian::df_rhf's own bar.
constexpr double kAnalyticTolerance = 1e-7;
// The analytic against the numerical Hessian: the central differences' error,
// their bar against PySCF.
constexpr double kMethodsTolerance = kHessianTolerance;
// The two methods' frequencies at the skew water, and the optimized water's
// against PySCF's at its minimum (cm^-1): the issue's bar.
constexpr double kFrequencyTolerance = 0.1;
// The central differences' asymmetry and translational sum: each is the
// h² error and the SCF's noise over 2h, as the agreement is, so at the same
// bar.
constexpr double kInvarianceTolerance = 1e-5;

// The gradient is the SCF energy's only at convergence, and its noise is
// divided by 2h, so the SCF is converged as pipeline_gradient_gpu.cpp
// converges it.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

// The case @p name of @p list in @p file under the test data.
const fb::json &reference_case(const std::filesystem::path &file, std::string_view list,
                               std::string_view name) {
  static std::map<std::filesystem::path, fb::json> docs;
  auto [doc, inserted] = docs.try_emplace(file);
  if (inserted) {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / file);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    if (read.has_value()) {
      doc->second = std::move(*read);
    }
  }
  if (doc->second.contains(list)) {
    for (const fb::json &entry : doc->second.at(list)) {
      if (entry.at("name").get<std::string>() == name) {
        return entry;
      }
    }
  }
  ADD_FAILURE() << "no case " << name << " in " << file;
  static const fb::json kNone;
  return kNone;
}

const std::filesystem::path kNumericalFile = "hessian_numerical/hessian_numerical.json";

// @p atoms (bohr) as an XYZ file's text (Angstrom), @p name its comment.
std::string xyz_text(const fb::json &atoms, std::string_view name) {
  std::string xyz = std::format("{}\n{}\n", atoms.size(), name);
  for (const fb::json &atom : atoms) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  return xyz;
}

// The reference's molecule as molecule.xyz in @p dir, and run.toml naming
// it, over the reference's basis and fit, with @p hessian after the chain.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view hessian) {
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz",
                             xyz_text(entry.at("atoms"), entry.at("name").get<std::string>()))
                  .has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}{}",
                  entry.at("basis").get<std::string>(), entry.at("aux_basis").get<std::string>(),
                  kScf, hessian);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The run of @p config, its output directory @p dir / "out"; its
// HessianOutput, checked against what hessian.json holds, @p method's. The
// node outlives the Calculation, whose products are on its device.
struct Ran {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  double seconds{0.0};
};

void run_config(const fb::test::ScratchDir &dir, const std::filesystem::path &config,
                std::string_view method, Ran &ran) {
  auto read = fb::read_config(config, dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation &calc = ran.calc.emplace(std::move(*read));
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status status = calc.run(ran.node);
  ran.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(status.has_value()) << status.error().message;
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), method);
  EXPECT_EQ(written->contains("displacement"), method == "numerical");

  // The product is what the file holds, and is symmetric.
  const auto &out = calc.get<fb::HessianOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;
  const auto rows = written->at("hessian").get<std::vector<std::vector<double>>>();
  const std::size_t n = rows.size();
  ASSERT_EQ(out->hessian.size(), n * n);
  for (std::size_t x = 0; x < n; ++x) {
    ASSERT_EQ(rows[x].size(), n) << "row " << x;
    for (std::size_t y = 0; y < n; ++y) {
      EXPECT_EQ(out->hessian[(x * n) + y], rows[x][y]);
      EXPECT_EQ(out->hessian[(x * n) + y], out->hessian[(y * n) + x]);
    }
  }
  EXPECT_EQ(written->at("asymmetry").get<double>(), out->asymmetry);
  EXPECT_EQ(written->at("translational_sum").get<double>(), out->translational_sum);
  // 3N - 6 vibrations for a nonlinear molecule, each a mode of 3N.
  EXPECT_EQ(out->vibrations.frequencies.size(), n - 6);
  EXPECT_EQ(written->at("frequencies").get<std::vector<double>>(), out->vibrations.frequencies);
  const auto modes = written->at("modes").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(modes.size(), out->vibrations.frequencies.size());
  for (std::size_t j = 0; j < modes.size(); ++j) {
    ASSERT_EQ(modes[j].size(), n);
    for (std::size_t x = 0; x < n; ++x) {
      EXPECT_EQ(modes[j][x], out->vibrations.modes[(j * n) + x]);
    }
  }
}

// max |a - b| of two equal-length vectors, each element held to @p tolerance.
double expect_near_all(const std::vector<double> &a, const std::vector<double> &b, double tolerance,
                       std::string_view what) {
  EXPECT_EQ(a.size(), b.size()) << what;
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    EXPECT_NEAR(a[i], b[i], tolerance) << what << " [" << i << "]";
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// Both methods at the skew water: each against PySCF, the analytic against
// the numerical to the central differences' error, and their frequencies
// against each other.
TEST(PipelineHessian, AnalyticAndNumericalDfRhfHessiansMatchPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case(kNumericalFile, "cases", "h2o_ccpvdz");
  const std::size_t n = 3 * expected.at("atoms").size();
  std::vector<double> want;
  for (const auto &row : expected.at("hessian").get<std::vector<std::vector<double>>>()) {
    want.insert(want.end(), row.begin(), row.end());
  }
  ASSERT_EQ(want.size(), n * n);

  const fb::test::ScratchDir numerical_dir("pipeline_hessian_numerical");
  Ran numerical;
  run_config(numerical_dir,
             write_config(numerical_dir, expected, "[hessian]\nmethod = \"numerical\"\n"),
             "numerical", numerical);
  ASSERT_FALSE(HasFatalFailure());
  const fb::test::ScratchDir analytic_dir("pipeline_hessian_analytic");
  Ran analytic;
  run_config(analytic_dir, write_config(analytic_dir, expected, "[hessian]\n"), "analytic",
             analytic);
  ASSERT_FALSE(HasFatalFailure());

  const auto &fd = *numerical.calc->get<fb::HessianOutput>();
  const auto &an = *analytic.calc->get<fb::HessianOutput>();
  const double fd_error = expect_near_all(fd.hessian, want, kHessianTolerance, "numerical");
  const double an_error = expect_near_all(an.hessian, want, kAnalyticTolerance, "analytic");
  const double methods = expect_near_all(an.hessian, fd.hessian, kMethodsTolerance, "methods");
  // The central differences' asymmetry and translational sum are their h²
  // error and the SCF's noise over 2h, as the agreement is, so at its bar;
  // the analytic Hessian's are its terms' rounding.
  EXPECT_LT(fd.asymmetry, kHessianTolerance);
  EXPECT_LT(fd.translational_sum, kHessianTolerance);
  EXPECT_LT(an.asymmetry, 1e-10);
  EXPECT_LT(an.translational_sum, 1e-8);
  const double frequencies = expect_near_all(an.vibrations.frequencies, fd.vibrations.frequencies,
                                             kFrequencyTolerance, "frequencies");

  std::cout << std::format("[ hessian ] h2o_ccpvdz numerical vs pyscf.df.hessian.rhf: max {:.2e}, "
                           "asymmetry {:.2e}, translational sum {:.2e}, {:.1f} s\n",
                           fd_error, fd.asymmetry, fd.translational_sum, numerical.seconds);
  std::cout << std::format("[ hessian ] h2o_ccpvdz analytic vs pyscf: max {:.2e}, vs numerical "
                           "{:.2e}, asymmetry {:.2e}, translational sum {:.2e}, {:.1f} s; "
                           "frequencies {:.2f} {:.2f} {:.2f} cm^-1, vs numerical {:.2e}\n",
                           an_error, methods, an.asymmetry, an.translational_sum, analytic.seconds,
                           an.vibrations.frequencies[0], an.vibrations.frequencies[1],
                           an.vibrations.frequencies[2], frequencies);
}

// [optimize] then the analytic [hessian]: the Hessian is the optimum's, and
// its frequencies are PySCF's at its own DF-RHF minimum.
TEST(PipelineHessian, OptimizedWaterFrequenciesMatchPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &start = reference_case("gradient/gradient.json", "optimized", "h2o_ccpvdz");
  const fb::json &minimum =
      reference_case("rhf_hessian/rhf_hessian.json", "cases", "h2o_ccpvdz_opt");
  // The two references are one minimum (PySCF's DF-RHF / cc-pVDZ / JKFIT):
  // the same energy, oriented differently.
  ASSERT_NEAR(start.at("energy").get<double>(), minimum.at("energy").get<double>(), 1e-10);
  const fb::test::ScratchDir dir("pipeline_hessian_optimized");
  EXPECT_TRUE(
      fb::write_file(dir / "start.xyz", xyz_text(start.at("start"), "h2o_ccpvdz")).has_value());
  // pipeline_optimize_gpu.cpp's thresholds, Opt=Tight's.
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[optimize]\nmax_force = 1.5e-5\nrms_force = 1e-5\nmax_displacement = 6e-5\n"
      "rms_displacement = 4e-5\n[hessian]\n",
      start.at("basis").get<std::string>(), start.at("aux_basis").get<std::string>(), kScf);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  Ran ran;
  run_config(dir, dir / "run.toml", "analytic", ran);
  ASSERT_FALSE(HasFatalFailure());

  // The Hessian's molecule is the optimum.
  const auto &optimized = ran.calc->get<fb::OptimizeOutput>();
  ASSERT_TRUE(optimized.has_value()) << optimized.error().message;
  const auto &molecule = ran.calc->get<fb::MoleculeOutput>();
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  ASSERT_EQ(molecule->molecule.atoms().size(), optimized->molecule.atoms().size());
  for (std::size_t a = 0; a < molecule->molecule.atoms().size(); ++a) {
    const auto &got = molecule->molecule.atoms()[a].position;
    const auto &want = optimized->molecule.atoms()[a].position;
    EXPECT_NEAR(got.x, want.x, 1e-10);
    EXPECT_NEAR(got.y, want.y, 1e-10);
    EXPECT_NEAR(got.z, want.z, 1e-10);
  }
  const auto &out = *ran.calc->get<fb::HessianOutput>();
  EXPECT_LT(out.translational_sum, 1e-8);
  const double worst = expect_near_all(out.vibrations.frequencies,
                                       minimum.at("frequencies").get<std::vector<double>>(),
                                       kFrequencyTolerance, "frequencies");
  std::cout << std::format("[ hessian ] h2o_ccpvdz optimized, then analytic: frequencies {:.3f} "
                           "{:.3f} {:.3f} cm^-1, vs pyscf {:.2e}, optimize + hessian {:.1f} s\n",
                           out.vibrations.frequencies[0], out.vibrations.frequencies[1],
                           out.vibrations.frequencies[2], worst, ran.seconds);
}

} // namespace
