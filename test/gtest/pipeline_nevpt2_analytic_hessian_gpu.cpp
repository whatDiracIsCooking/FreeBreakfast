// The analytic [hessian] over the [nevpt2_gradient] chain (#818): the
// step's wiring of hessian::df_sc_nevpt2 (#816) and df_pc_nevpt2 (#817),
// on #803's fixtures, data/nevpt2_hessian/ (STO-3G / cc-pVDZ-JKFIT). The
// Hessians themselves are those calls' own tests'; here:
//
// - The skew case, h2co_skew_sto3g, per variant: the config run by
//   Calculation, its hessian.json and product against a direct call of the
//   variant's Hessian on the same inputs, the Calculation's outer chain's
//   products (the same chain the step reruns), to 1e-10 of the largest
//   element; the frequencies published. PC's chain runs at the step's
//   default floor, HessianStep::kPcFloor = 1e-10 (#923): the direct call is
//   made at it, hessian.json records it, and the run's own [nevpt2_gradient]
//   carries it.
// - The symmetric case, nh3_cs_sto3g, SC, in the fixture's
//   symmetry-adapted coordinates: df_sc_nevpt2 refuses the whole A'' x A''
//   off-diagonal block (#816's amended acceptance), and the step returns that
//   refusal as it is, Numerical, naming the ten pairs, and writes no
//   hessian.json.
//
// The config's refusals (a state average, integral-direct ERIs, a frozen
// core, quasi_degenerate) are test/pipeline/hessian_host.cpp's: they need no
// device.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.mcscf;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The step against the direct call, relative to the largest element: the
// wiring, which reruns the same chain the Calculation ran once.
constexpr double kWiring = 1e-10;

const fb::json &fixture(std::string_view name) {
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

// The fixture @p doc's molecule.xyz in @p dir (bohr there, Angstrom here),
// and run.toml over its basis, fit and active space, the CASSCF converged
// tight, [nevpt2_gradient] of @p variant and the [hessian] table @p hessian.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &doc,
                                   std::string_view variant, std::string_view hessian) {
  std::string xyz = std::format("{}\nfixture\n", doc.at("atoms").size());
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
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\nconv_tol = 1e-12\nconv_tol_grad = 1e-10\n"
      "[nevpt2_gradient]\nvariant = \"{}\"\n[hessian]\n{}",
      doc.at("charge").get<std::int32_t>(), doc.at("basis").at("name").get<std::string>(),
      doc.at("aux_basis").at("name").get<std::string>(), doc.at("ncas").get<std::int32_t>(),
      doc.at("nelecas").get<std::int32_t>(), active, variant, hessian);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The variant's Hessian called directly on @p calc's outer products, the
// single-state CASSCF's Solution built as the step builds it, PC's at the
// step's default floor; k x k column-major, as returned.
fb::Result<std::vector<double>> direct_hessian(const fb::Calculation &calc, fb::NodeResources &node,
                                               std::string_view variant) {
  fb::DeviceResources &res = node.device(0);
  const fb::CasscfOutput &cas = *calc.get<fb::CasscfOutput>();
  const fb::BasisSet &basis = calc.get<fb::BasisSetOutput>()->basis;
  const auto &fit = std::get<fb::DensityFitted>(calc.get<fb::EriOutput>()->eri);
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = cas.rdm1,
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
      .weights = cas.weights};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  const fb::Molecule &molecule = calc.get<fb::MoleculeOutput>()->molecule;
  const auto &classes = calc.get<fb::PrimitivePairClassesOutput>()->classes;
  const fb::SphericalMatrixAO &h = calc.get<fb::CoreHamiltonianOutput>()->h;
  if (variant == "pc") {
    auto terms =
        fb::hessian::df_pc_nevpt2(res, molecule, basis, classes, fit.aux, h, fit.b, fit.l.data(),
                                  solution, cas.ncore, cas.ncas, 0, {}, fb::HessianStep::kPcFloor);
    if (!terms.has_value()) {
      return std::unexpected(std::move(terms.error()));
    }
    return std::move(terms->hessian);
  }
  auto terms = fb::hessian::df_sc_nevpt2(res, molecule, basis, classes, fit.aux, h, fit.b,
                                         fit.l.data(), solution, cas.ncore, cas.ncas);
  if (!terms.has_value()) {
    return std::unexpected(std::move(terms.error()));
  }
  return std::move(terms->hessian);
}

// The skew case: the step's Hessian is the direct call's, and has
// frequencies.
void expect_skew(std::string_view variant) {
  const fb::json &doc = fixture("h2co_skew_sto3g");
  ASSERT_FALSE(doc.is_null());
  const fb::test::ScratchDir dir(std::format("pipeline_nevpt2_analytic_hessian_{}", variant));
  auto read = fb::read_config(write_config(dir, doc, variant, ""), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  ASSERT_EQ(calc.step<fb::HessianStep>()->method, fb::HessianMethod::Analytic);
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  const std::vector<double> want = [&] {
    auto direct = direct_hessian(calc, node, variant);
    EXPECT_TRUE(direct.has_value()) << direct.error().message;
    return direct.has_value() ? std::move(*direct) : std::vector<double>{};
  }();
  const std::size_t n = 3 * doc.at("atoms").size();
  ASSERT_EQ(want.size(), n * n);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::json &out = *written;
  EXPECT_EQ(out.at("method").get<std::string>(), "analytic");
  EXPECT_EQ(out.at("variant").get<std::string>(), variant);
  if (variant == "pc") {
    ASSERT_TRUE(out.contains("pc_floor"));
    EXPECT_EQ(out.at("pc_floor").get<double>(), 1e-10);
    EXPECT_EQ(calc.step<fb::HessianStep>()->pc_floor, 1e-10);
    EXPECT_EQ(calc.step<fb::Nevpt2GradientStep>()->pc_floor, 1e-10);
  } else {
    EXPECT_FALSE(out.contains("pc_floor"));
  }
  EXPECT_FALSE(out.contains("ladder"));
  EXPECT_FALSE(out.contains("displacement"));
  const auto got = out.at("hessian").get<std::vector<std::vector<double>>>();
  const fb::HessianOutput &product = *calc.get<fb::HessianOutput>();
  ASSERT_EQ(got.size(), n);
  double largest = 0.0;
  for (const double v : want) {
    largest = std::max(largest, std::abs(v));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(got[i].size(), n);
    for (std::size_t j = 0; j < n; ++j) {
      // The direct call's is column-major and exactly symmetric.
      const double expected = want[i + (n * j)];
      EXPECT_NEAR(got[i][j], expected, kWiring * largest)
          << variant << " (" << i << ", " << j << ")";
      EXPECT_EQ(product.hessian[(i * n) + j], got[i][j]);
      worst = std::max(worst, std::abs(got[i][j] - expected));
    }
  }
  // Every vibration published: the skew geometry is C1, no pair refused.
  EXPECT_EQ(out.at("frequencies").size(), n - 6);
  EXPECT_EQ(product.vibrations.frequencies.size(), n - 6);
  EXPECT_TRUE(product.dipole_derivative.empty());
  std::cout << std::format("[ nevpt2 analytic hessian ] {}: {:.1f} s (outer chain, the step and "
                           "its nested chain), max |H - direct| {:.2e} = {:.2e} of the largest, "
                           "asymmetry {:.2e}, translational sum {:.2e}\n",
                           variant, seconds, worst, worst / largest, product.asymmetry,
                           product.translational_sum);
}

TEST(PipelineNevpt2AnalyticHessian, ScSkewMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("sc");
}

TEST(PipelineNevpt2AnalyticHessian, PcSkewMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("pc");
}

// The symmetric case, SC in the fixture's coordinates: the refusal of the
// A'' x A'' off-diagonal block reaches the step's output, naming each pair.
TEST(PipelineNevpt2AnalyticHessian, ScSymmetricRefusalNamesTheAPrimePrimeBlock) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = fixture("nh3_cs_sto3g");
  ASSERT_FALSE(doc.is_null());
  std::string table = "coordinates = [\n";
  for (const auto &row :
       doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>()) {
    table += "  [";
    for (std::size_t x = 0; x < row.size(); ++x) {
      table += std::format("{}{:.17g}", x == 0 ? "" : ", ", row[x]);
    }
    table += "],\n";
  }
  table += "]\n";
  const fb::test::ScratchDir dir("pipeline_nevpt2_analytic_hessian_symmetric");
  auto read = fb::read_config(write_config(dir, doc, "sc", table), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const fb::Status ran = calc.run(node);
  ASSERT_FALSE(ran.has_value()) << "no pair refused";
  ASSERT_EQ(ran.error().kind, fb::ErrorKind::Numerical) << ran.error().message;
  const std::string &message = ran.error().message;
  std::cout << std::format("[ nevpt2 analytic hessian ] nh3 sc: {}\n", message);

  // The pairs it names: df_sc_nevpt2's list, "coordinate pairs (i, j), ...:".
  const std::string lead = "coordinate pairs ";
  const std::size_t from = message.find(lead);
  ASSERT_NE(from, std::string::npos) << message;
  const std::string list =
      message.substr(from + lead.size(), message.find(':', from) - from - lead.size());
  std::set<std::pair<std::size_t, std::size_t>> named;
  const std::regex pair_re(R"(\((\d+), (\d+)\))");
  for (auto it = std::sregex_iterator(list.begin(), list.end(), pair_re);
       it != std::sregex_iterator(); ++it) {
    named.emplace(std::stoul((*it)[1].str()), std::stoul((*it)[2].str()));
  }
  // Exactly the A'' x A'' off-diagonal block: its ten pairs, i < j.
  const auto irreps = doc.at("coordinates").at("irreps").get<std::vector<std::string>>();
  const std::string symmetric = doc.at("coordinates").at("irreps_order")[0].get<std::string>();
  std::set<std::pair<std::size_t, std::size_t>> block;
  for (std::size_t i = 0; i < irreps.size(); ++i) {
    for (std::size_t j = i + 1; j < irreps.size(); ++j) {
      if (irreps[i] != symmetric && irreps[j] != symmetric) {
        block.emplace(i, j);
      }
    }
  }
  ASSERT_EQ(block.size(), 10U);
  EXPECT_EQ(named, block) << list;
  // No Hessian, so nothing written.
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "hessian.json"));
}

} // namespace
