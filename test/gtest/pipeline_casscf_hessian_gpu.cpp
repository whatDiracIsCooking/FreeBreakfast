// GPU cases for the [hessian] step over the CASSCF chain, analytic and
// numerical, TOML configs read, planned and run by Calculation, against
// data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py, over the same
// cc-pVDZ-JKFIT fit): the design's stretched water, CAS(4,4) over its two O-H
// sigma/sigma* pairs. Each config's own [casscf] -- the outer one, from the
// SCF and the pick -- is first held to the fixture's energy, so the reference
// the step works at is the fixture's solution (a minimum of E_SA, not the
// saddle PySCF's own solve finds for three of these cases, nor another
// minimum); then the Hessian against the fixture's relaxed_fd, a
// fourth-order difference of the analytic relaxed gradient, its asymmetry and
// translational sum checked and printed.
//
// Numerical, to the issue's 1e-5:
// - h2o_631g, state-specific in 6-31G: the fast case, the family's cheapest;
// - h2o_631g_sa2 (0.5, 0.5) and h2o_631g_sa2_unequal (0.7, 0.3), root 0,
//   and h2o_ccpvdz, state-specific with d shells: slow-gated.
// Analytic (hessian::df_casscf, hessian::df_sacasscf), to 1e-7: h2o_631g,
// h2o_ccpvdz, and root 1 of both two-root averages, all fast.
// Analytic against numerical in the same pipeline: root 1 of h2o_631g_sa2,
// the step run again numerically at the Calculation's molecule
// (slow-gated); and the CASSCF water of data/casscf_gradient/ optimized,
// then the analytic [hessian] at the optimum, its frequencies reported, and
// the numerical one beside it.
//
// Beside them, an unconverged CASSCF refused, naming the geometry. The guard
// on a displaced solve that swapped an orbital or a root is forced in
// test/pipeline/hessian_host.cpp, over check_displaced_casscf itself.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The numerical Hessian against relaxed_fd or the analytic one
// (hartree / bohr²): the issue's bar. The central difference's h² error and
// the CASSCF's noise over 2h are below it.
constexpr double kHessianTolerance = 1e-5;
// The analytic Hessian against relaxed_fd: the fixture's own
// analytic-vs-difference gap is 2.4e-8 at worst, and the CASSCF here is
// converged to conv_tol_grad = 1e-9 rather than read from the fixture.
constexpr double kAnalyticTolerance = 1e-7;
// The harmonic frequencies of the two methods' Hessians (cm⁻¹).
constexpr double kFrequencyTolerance = 0.1;
// The reference CASSCF's energy against the fixture's: the same solution.
constexpr double kEnergyTolerance = 1e-8;

// The RHF orbitals the CASSCF starts from. The (0.7, 0.3) average has three
// minima within reach of a pick by index -- E_SA -75.80024 from these,
// -75.81939 from [2, 4, 5, 6], and the fixture's -75.81993 from
// [3, 4, 5, 6] or [2, 3, 5, 7] -- so each case's pick is the one that
// reaches the fixture's, and the energy check below says so.
constexpr std::string_view kSigmaPairs = "2, 3, 5, 6";

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

// The SCF and the CASSCF converged tight: each gradient's noise is divided
// by 2h.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = [] {
    auto read = fb::read_json(kDataDir / "casscf_hessian.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  if (doc.contains("cases")) {
    for (const fb::json &entry : doc.at("cases")) {
      if (entry.at("name").get<std::string>() == name) {
        return entry;
      }
    }
  }
  ADD_FAILURE() << "no case " << name;
  static const fb::json kNone;
  return kNone;
}

// @p atoms (bohr) as an XYZ file's text (Angstrom).
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

// The case's molecule.xyz in @p dir and run.toml over its basis and fit, the
// CASSCF(4,4) over the RHF orbitals @p active at its weights, then the
// gradient of @p root and the [hessian] by @p method.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view active, std::string_view method = "numerical",
                                   std::int32_t root = 0) {
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz",
                             xyz_text(entry.at("atoms"), entry.at("name").get<std::string>()))
                  .has_value());
  const auto weights = entry.at("weights").get<std::vector<double>>();
  std::string weight_list;
  for (const double w : weights) {
    weight_list += std::format("{}{:.17g}", weight_list.empty() ? "" : ", ", w);
  }
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
                  "[casscf]\nncas = 4\nnelecas = 4\nactive = [{}]\nnroots = {}\nweights = [{}]\n"
                  "conv_tol = 1e-12\nconv_tol_grad = 1e-9\n"
                  "[casscf_gradient]\nroot = {}\n[hessian]\nmethod = \"{}\"\n",
                  entry.at("basis").get<std::string>(), entry.at("aux_basis").get<std::string>(),
                  kScf, active, weights.size(), weight_list, root, method);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// max |a - b| over two n x n matrices, each element held to @p tolerance.
double expect_near_matrix(const std::vector<double> &a, const std::vector<double> &b, std::size_t n,
                          double tolerance, std::string_view what) {
  double worst = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      EXPECT_NEAR(a[(x * n) + y], b[(x * n) + y], tolerance)
          << what << " (" << x << ", " << y << ")";
      worst = std::max(worst, std::abs(a[(x * n) + y] - b[(x * n) + y]));
    }
  }
  return worst;
}

// The step's frequencies against @p want's, ascending both, to
// kFrequencyTolerance; the largest gap.
double expect_near_frequencies(const fb::HessianOutput &got, const fb::HessianOutput &want) {
  EXPECT_EQ(got.vibrations.frequencies.size(), want.vibrations.frequencies.size());
  double worst = 0.0;
  for (std::size_t j = 0;
       j < std::min(got.vibrations.frequencies.size(), want.vibrations.frequencies.size()); ++j) {
    EXPECT_NEAR(got.vibrations.frequencies[j], want.vibrations.frequencies[j], kFrequencyTolerance)
        << "mode " << j;
    worst =
        std::max(worst, std::abs(got.vibrations.frequencies[j] - want.vibrations.frequencies[j]));
  }
  return worst;
}

// The planned [hessian] of @p calc run again by @p method at its molecule,
// writing nothing.
fb::Result<fb::HessianOutput> rerun_hessian(const fb::Calculation &calc, fb::HessianMethod method,
                                            fb::NodeResources &node) {
  fb::HessianStep step = *calc.step<fb::HessianStep>();
  step.method = method;
  step.json_path.clear();
  const auto &molecule = calc.get<fb::MoleculeOutput>();
  if (!molecule.has_value()) {
    return std::unexpected(molecule.error());
  }
  return step.run(fb::View<fb::MoleculeOutput>{*molecule}, node);
}

// The case @p name end to end, the CASSCF picked from the RHF orbitals
// @p active: the outer CASSCF at the fixture's energy, then the step's
// Hessian of @p root by @p method, the product as hessian.json holds it,
// against relaxed_fd to @p tolerance. The Calculation is left in @p keep,
// when given, for a rerun by the other method.
void expect_relaxed_fd(std::string_view name, std::string_view active,
                       std::string_view method = "numerical", std::int32_t root = 0,
                       double tolerance = kHessianTolerance,
                       std::optional<fb::Calculation> *keep = nullptr) {
  const fb::json &entry = reference_case(name);
  ASSERT_FALSE(entry.is_null());
  const std::size_t n = 3 * entry.at("atoms").size();
  const auto relaxed = fb::read_npy(kDataDir / std::format("{}_root{}_relaxed_fd.npy", name, root));
  ASSERT_TRUE(relaxed.has_value()) << relaxed.error().message;
  ASSERT_EQ(relaxed->shape, (std::vector<std::size_t>{n, n}));
  // Fortran order: element (y, x), dg_y/dx_x, at y + n x, so x-major here
  // as the step's raw rows are. Symmetrized, as the step's is.
  std::vector<double> want(n * n);
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      want[(x * n) + y] = 0.5 * (relaxed->data[(x * n) + y] + relaxed->data[(y * n) + x]);
    }
  }

  const fb::test::ScratchDir dir(
      std::format("pipeline_casscf_hessian_{}_{}_{}", name, method, root));
  auto read =
      fb::read_config(write_config(dir, entry, active, method, root), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::NodeResources node;
  fb::Calculation calc = std::move(*read);
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status status = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(status.has_value()) << status.error().message;

  // The pick converged to the fixture's solution, which the step's own
  // reference solve, the same chain, is too.
  const auto &cas = calc.get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  EXPECT_NEAR(cas->energy, entry.at("e_sa").get<double>(), kEnergyTolerance);

  const auto &out = calc.get<fb::HessianOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;
  ASSERT_EQ(out->hessian.size(), n * n);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), method);
  EXPECT_EQ(written->at("root").get<std::int32_t>(), root);
  const auto rows = written->at("hessian").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(rows.size(), n);
  for (std::size_t x = 0; x < n; ++x) {
    EXPECT_EQ(rows[x],
              std::vector<double>(out->hessian.begin() + static_cast<std::ptrdiff_t>(x * n),
                                  out->hessian.begin() + static_cast<std::ptrdiff_t>((x + 1) * n)));
  }
  EXPECT_EQ(written->at("frequencies").get<std::vector<double>>(), out->vibrations.frequencies);
  // No dipole derivative for a CASSCF: no IR intensities.
  EXPECT_TRUE(out->dipole_derivative.empty());
  EXPECT_TRUE(out->intensities.empty());
  const double worst = expect_near_matrix(out->hessian, want, n, tolerance, name);
  // The central differences' asymmetry and translational sum are their h²
  // error and the CASSCF's noise over 2h, as the agreement is; the analytic
  // terms' are their rounding and the CP solve's residual.
  EXPECT_LT(out->asymmetry, tolerance);
  EXPECT_LT(out->translational_sum, tolerance);
  std::cout << std::format("[ hessian ] {} root {} {} DF-CASSCF vs relaxed_fd: max {:.2e}, "
                           "asymmetry {:.2e}, translational sum {:.2e}, {:.1f} s\n",
                           name, root, method, worst, out->asymmetry, out->translational_sum,
                           seconds);
  if (keep != nullptr) {
    keep->emplace(std::move(calc));
  }
}

TEST(PipelineCasscfHessian, StateSpecific631gMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  expect_relaxed_fd("h2o_631g", kSigmaPairs);
}

TEST(PipelineCasscfHessian, StateAveragedEqualWeightsMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_relaxed_fd("h2o_631g_sa2", kSigmaPairs);
}

TEST(PipelineCasscfHessian, StateAveragedUnequalWeightsMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_relaxed_fd("h2o_631g_sa2_unequal", "3, 4, 5, 6");
}

TEST(PipelineCasscfHessian, StateSpecificCcpvdzMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_relaxed_fd("h2o_ccpvdz", kSigmaPairs);
}

TEST(PipelineCasscfHessian, AnalyticStateSpecific631gMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  expect_relaxed_fd("h2o_631g", kSigmaPairs, "analytic", 0, kAnalyticTolerance);
}

TEST(PipelineCasscfHessian, AnalyticStateSpecificCcpvdzMatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  expect_relaxed_fd("h2o_ccpvdz", kSigmaPairs, "analytic", 0, kAnalyticTolerance);
}

TEST(PipelineCasscfHessian, AnalyticStateAveragedEqualWeightsRoot1MatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  expect_relaxed_fd("h2o_631g_sa2", kSigmaPairs, "analytic", 1, kAnalyticTolerance);
}

TEST(PipelineCasscfHessian, AnalyticStateAveragedUnequalWeightsRoot1MatchesRelaxedFd) {
  FBF_REQUIRE_GPU();
  expect_relaxed_fd("h2o_631g_sa2_unequal", "3, 4, 5, 6", "analytic", 1, kAnalyticTolerance);
}

// One root of a state average by both methods in the same pipeline: the
// analytic [hessian] the config plans, then the step again, numerically, at
// the Calculation's molecule. Root 1 at equal weights, where the analytic
// sum carries eq. rot. Each is against relaxed_fd above; here against each
// other, to the central differences' error, and their frequencies.
TEST(PipelineCasscfHessian, StateAveragedRootAnalyticMatchesNumerical) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  std::optional<fb::Calculation> calc;
  expect_relaxed_fd("h2o_631g_sa2", kSigmaPairs, "analytic", 1, kAnalyticTolerance, &calc);
  ASSERT_TRUE(calc.has_value());
  const auto &analytic = calc->get<fb::HessianOutput>();
  ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
  fb::NodeResources node;
  const auto numerical = rerun_hessian(*calc, fb::HessianMethod::Numerical, node);
  ASSERT_TRUE(numerical.has_value()) << numerical.error().message;
  const std::size_t n = 3 * reference_case("h2o_631g_sa2").at("atoms").size();
  ASSERT_EQ(numerical->hessian.size(), n * n);
  const double worst = expect_near_matrix(numerical->hessian, analytic->hessian, n,
                                          kHessianTolerance, "h2o_631g_sa2 root 1");
  const double frequencies = expect_near_frequencies(*numerical, *analytic);
  std::cout << std::format("[ hessian ] h2o_631g_sa2 root 1 numerical vs analytic: max {:.2e}, "
                           "frequencies {:.2e} cm^-1\n",
                           worst, frequencies);
}

// The CASSCF water of data/casscf_gradient/casscf_optimize.json (PySCF's
// DF-CASSCF(4,4) minimum in cc-pVDZ), optimized from its off-symmetry start
// at Opt=Tight's thresholds as pipeline_optimize_gpu.cpp's
// WaterCasscfReachesThePyscfMinimum does, then the analytic [hessian] at the
// optimum: the frequencies reported, in the product and hessian.json, three
// real ones at a minimum. Then the step again, numerically, at the same
// molecule: the two Hessians to the central differences' error and their
// frequencies to 0.1 cm⁻¹.
TEST(PipelineCasscfHessian, OptimizedWaterAnalyticFrequenciesMatchNumerical) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                                 "casscf_optimize.json");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  const fb::json &entry = doc->at("cases").at(0);
  const fb::test::ScratchDir dir("pipeline_casscf_hessian_optimized");
  ASSERT_TRUE(fb::write_file(dir / "start.xyz", xyz_text(entry.at("start"), "start")).has_value());
  std::string active;
  for (const auto t : entry.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", t);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n{}"
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\nconv_tol = 1e-12\nconv_tol_grad = 1e-9\n"
      "[casscf_gradient]\n"
      "[optimize]\nmax_force = 1.5e-5\nrms_force = 1e-5\nmax_displacement = 6e-5\n"
      "rms_displacement = 4e-5\n[hessian]\n",
      entry.at("basis").get<std::string>(), entry.at("aux_basis").at("name").get<std::string>(),
      kScf, entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(), active);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());

  auto read = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  ASSERT_EQ(calc.step<fb::HessianStep>()->method, fb::HessianMethod::Analytic);
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  // At PySCF's minimum, and the Hessian there.
  const auto &optimum = calc.get<fb::OptimizeOutput>();
  ASSERT_TRUE(optimum.has_value()) << optimum.error().message;
  EXPECT_NEAR(optimum->energy, entry.at("energy").get<double>(), kEnergyTolerance);
  // The outer CASSCF, picked afresh at the optimum as the step's own is, is
  // the optimized root.
  const auto &cas = calc.get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value()) << cas.error().message;
  EXPECT_NEAR(cas->energy, optimum->energy, kEnergyTolerance);
  const auto &analytic = calc.get<fb::HessianOutput>();
  ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
  const std::size_t n = 3 * entry.at("atoms").size();
  ASSERT_EQ(analytic->hessian.size(), n * n);
  ASSERT_EQ(analytic->vibrations.frequencies.size(), n - 6);
  for (const double frequency : analytic->vibrations.frequencies) {
    EXPECT_GT(frequency, 0.0);
  }
  EXPECT_LT(analytic->translational_sum, kAnalyticTolerance);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), "analytic");
  EXPECT_EQ(written->at("root").get<std::int32_t>(), 0);
  EXPECT_EQ(written->at("frequencies").get<std::vector<double>>(),
            analytic->vibrations.frequencies);

  const auto t1 = std::chrono::steady_clock::now();
  const auto numerical = rerun_hessian(calc, fb::HessianMethod::Numerical, node);
  const double numerical_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  ASSERT_TRUE(numerical.has_value()) << numerical.error().message;
  const double worst = expect_near_matrix(numerical->hessian, analytic->hessian, n,
                                          kHessianTolerance, "optimized water");
  const double frequencies = expect_near_frequencies(*numerical, *analytic);
  std::cout << std::format("[ hessian ] h2o_ccpvdz CASSCF(4,4) at the optimum: frequencies "
                           "{::.3f} cm^-1; numerical vs analytic max {:.2e}, frequencies "
                           "{:.2e} cm^-1; optimize + analytic {:.1f} s, numerical {:.1f} s\n",
                           analytic->vibrations.frequencies, worst, frequencies, seconds,
                           numerical_seconds);
}

// A CASSCF that does not converge is the run's error, never differenced,
// and the message says at which geometry: here the reference's, which two
// macro iterations cannot converge.
TEST(PipelineCasscfHessian, AnUnconvergedCasscfIsRefused) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_631g");
  ASSERT_FALSE(entry.is_null());
  const fb::test::ScratchDir dir("pipeline_casscf_hessian_unconverged");
  // The outer [casscf] reads the same table and would fail first, so the
  // planned step is run alone, its own copy of the chain cut to two macro
  // iterations.
  auto read = fb::read_config(write_config(dir, entry, kSigmaPairs), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::HessianStep step = *read->step<fb::HessianStep>();
  step.casscf.max_cycle = 2;
  step.json_path.clear();
  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  auto molecule = fb::MoleculeStep{.xyz = dir / "molecule.xyz"}.run(fb::View<>{}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  const auto result = step.run(fb::View<fb::MoleculeOutput>{*molecule}, node);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().kind, fb::ErrorKind::MaxIterationsExceeded);
  EXPECT_TRUE(result.error().message.starts_with("[hessian] at the reference geometry: "))
      << result.error().message;
}

} // namespace
