// GPU cases for the numerical [hessian] over the CASSCF chain, TOML configs
// read, planned and run by Calculation, against data/casscf_hessian/
// (tools/emit/casscf_hessian_reference.py, over the same cc-pVDZ-JKFIT fit):
// the design's stretched water, CAS(4,4) over its two O-H sigma/sigma*
// pairs. Each config's own [casscf] -- the outer one, from the SCF and the
// pick -- is first held to the fixture's energy, so the reference the step
// differences around is the fixture's solution (a minimum of E_SA, not the
// saddle PySCF's own solve finds for three of these cases, nor another
// minimum); then the Hessian
// against the fixture's relaxed_fd, a fourth-order difference of the
// analytic relaxed gradient, to the issue's 1e-5, its asymmetry and
// translational sum checked and printed.
//
// - h2o_631g, state-specific in 6-31G: the fast case, the family's cheapest;
// - h2o_631g_sa2 (0.5, 0.5) and h2o_631g_sa2_unequal (0.7, 0.3), root 0,
//   and h2o_ccpvdz, state-specific with d shells: slow-gated.
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

// The Hessian against relaxed_fd (hartree / bohr²): the issue's bar. The
// central difference's h² error and the CASSCF's noise over 2h are below it.
constexpr double kHessianTolerance = 1e-5;
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
// gradient of root 0 and the numerical [hessian].
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view active) {
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
                  "[casscf_gradient]\nroot = 0\n[hessian]\nmethod = \"numerical\"\n",
                  entry.at("basis").get<std::string>(), entry.at("aux_basis").get<std::string>(),
                  kScf, active, weights.size(), weight_list);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The case @p name end to end, the CASSCF picked from the RHF orbitals
// @p active: the outer CASSCF at the fixture's energy, then the step's
// Hessian, the product as hessian.json holds it, against relaxed_fd.
void expect_relaxed_fd(std::string_view name, std::string_view active) {
  const fb::json &entry = reference_case(name);
  ASSERT_FALSE(entry.is_null());
  const std::size_t n = 3 * entry.at("atoms").size();
  const auto relaxed = fb::read_npy(kDataDir / std::format("{}_root0_relaxed_fd.npy", name));
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

  const fb::test::ScratchDir dir(std::format("pipeline_casscf_hessian_{}", name));
  auto read = fb::read_config(write_config(dir, entry, active), dir.path(), dir / "out");
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
  ASSERT_NEAR(cas->energy, entry.at("e_sa").get<double>(), kEnergyTolerance);

  const auto &out = calc.get<fb::HessianOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;
  ASSERT_EQ(out->hessian.size(), n * n);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("method").get<std::string>(), "numerical");
  EXPECT_EQ(written->at("root").get<std::int32_t>(), 0);
  const auto rows = written->at("hessian").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(rows.size(), n);
  double worst = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      EXPECT_EQ(rows[x][y], out->hessian[(x * n) + y]);
      EXPECT_NEAR(out->hessian[(x * n) + y], want[(x * n) + y], kHessianTolerance)
          << name << " (" << x << ", " << y << ")";
      worst = std::max(worst, std::abs(out->hessian[(x * n) + y] - want[(x * n) + y]));
    }
  }
  // The central differences' asymmetry and translational sum are their h²
  // error and the CASSCF's noise over 2h, as the agreement is.
  EXPECT_LT(out->asymmetry, kHessianTolerance);
  EXPECT_LT(out->translational_sum, kHessianTolerance);
  std::cout << std::format("[ hessian ] {} numerical DF-CASSCF vs relaxed_fd: max {:.2e}, "
                           "asymmetry {:.2e}, translational sum {:.2e}, {:.1f} s\n",
                           name, worst, out->asymmetry, out->translational_sum, seconds);
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
