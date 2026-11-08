// The analytic [hessian] over a quasi_degenerate [nevpt2_gradient] chain
// (#968): the step's wiring of hessian::df_qd_sc_nevpt2 (#965) and
// df_qd_pc_nevpt2 (#966), on data/qd_nevpt2_hessian/'s cases. The Hessians
// themselves are those calls' own tests'; here:
//
// - The C1 case, h2co_skew_ccpvdz_sa3, equal weights, QD state 1, per
//   variant (slow-gated): the config run by Calculation, its hessian.json
//   and product against a direct call of the variant's QD Hessian on the
//   same inputs, the Calculation's outer chain's products (the same chain
//   the step reruns), to 1e-10 of the largest element; the frequencies
//   published, and hessian.json's "qd" block the call's state, eigenvalues,
//   term (ii)'s share and gaps. PC at pc_floor = 1e-3 (kPcWiringFloor),
//   the direct call made at it, so the step is also seen passing a
//   non-default floor through: at the default 1e-10, root 1 keeps Sijr, Srs
//   and Sir modes down to 6e-8 of their metric's largest eigenvalue, which
//   amplifies bit-level input differences -- the step's Hessian missed the
//   direct call's by 3.3e-10 of the largest element, and two direct calls
//   on identical inputs differed by 5.1e-11 (RTX 3080). #964 cut skew at
//   1e-3 for the same reason.
// - The C2v A1/B2 case, h2nb_sto3g_sa2, PC, state 1, in its
//   symmetry-adapted coordinates at the references' round-off cut
//   (pc_floor = 0, df_qd_pc_nevpt2_hessian_gpu's ground): df_qd_pc_nevpt2
//   refuses exactly the fixture's expected PC blocks, (B2 1, B2 1),
//   (B2 1, B2 2) and (B2 2, B2 2), and the step returns that refusal as it
//   is, Numerical, naming them, and writes no hessian.json. SC refuses
//   nothing there (the fixture's expected SC refusals are none).
//
// The config's refusals (a frozen core, integral-direct ERIs) are
// test/pipeline/hessian_host.cpp's: they need no device.

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

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The step against the direct call, relative to the largest element: the
// wiring, which reruns the same chain the Calculation ran once.
constexpr double kWiring = 1e-10;
// K >= 1: the first excited QD state.
constexpr std::int32_t kState = 1;
// PC's floor on the C1 case (the header says why not the default).
constexpr double kPcWiringFloor = 1e-3;

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

// The case @p doc's molecule.xyz in @p dir (bohr there, Angstrom here) and
// run.toml: its basis, fit, CAS and weights, the SCF and CASSCF converged
// tight, a quasi_degenerate [nevpt2_gradient] of @p variant at kState and
// the [hessian] table @p hessian.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &doc,
                                   std::string_view variant, std::string_view hessian) {
  std::string xyz = std::format("{}\nreference\n", doc.at("atoms").size());
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
                  "conv_tol = 1e-12\nconv_tol_grad = 1e-10\n"
                  "[nevpt2_gradient]\nvariant = \"{}\"\nquasi_degenerate = true\nstate = {}\n"
                  "[hessian]\n{}",
                  doc.at("basis").at("name").get<std::string>(),
                  doc.at("aux_basis").at("name").get<std::string>(),
                  doc.at("ncas").get<std::int32_t>(), doc.at("nelecas").get<std::int32_t>(),
                  doc.at("weights").size(), weights, variant, kState, hessian);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The variant's QD Hessian of kState called directly on @p calc's outer
// products, the state-averaged CASSCF's Solution built as the step builds
// it (gamma the weighted 1-RDM), PC's at kPcWiringFloor; with the QD
// block's numbers.
fb::Result<fb::hessian::QdScNevpt2Terms>
direct_hessian(const fb::Calculation &calc, fb::NodeResources &node, std::string_view variant) {
  fb::DeviceResources &res = node.device(0);
  const fb::CasscfOutput &cas = *calc.get<fb::CasscfOutput>();
  const fb::BasisSet &basis = calc.get<fb::BasisSetOutput>()->basis;
  const auto &fit = std::get<fb::DensityFitted>(calc.get<fb::EriOutput>()->eri);
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
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
        fb::hessian::df_qd_pc_nevpt2(res, molecule, basis, classes, fit.aux, h, fit.b, fit.l.data(),
                                     solution, cas.ncore, cas.ncas, kState, 0, {}, kPcWiringFloor);
    if (!terms.has_value()) {
      return std::unexpected(std::move(terms.error()));
    }
    return std::move(terms->terms);
  }
  return fb::hessian::df_qd_sc_nevpt2(res, molecule, basis, classes, fit.aux, h, fit.b,
                                      fit.l.data(), solution, cas.ncore, cas.ncas, kState);
}

// The C1 case: the step's Hessian is the direct call's, with frequencies,
// and hessian.json's "qd" the call's report.
void expect_skew(std::string_view variant) {
  const fb::json &doc = reference("h2co_skew_ccpvdz_sa3");
  ASSERT_FALSE(doc.is_null());
  const fb::test::ScratchDir dir(std::format("pipeline_qd_nevpt2_analytic_hessian_{}", variant));
  auto read = fb::read_config(write_config(dir, doc, variant,
                                           variant == "pc"
                                               ? std::format("pc_floor = {:g}\n", kPcWiringFloor)
                                               : std::string{}),
                              dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  ASSERT_EQ(calc.step<fb::HessianStep>()->method, fb::HessianMethod::Analytic);
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  auto direct = direct_hessian(calc, node, variant);
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  const fb::hessian::QdScNevpt2Terms &want = *direct;
  const std::size_t n = 3 * doc.at("atoms").size();
  ASSERT_EQ(want.hessian.size(), n * n);
  auto written = fb::read_json(dir / "out" / "hessian.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  const fb::json &out = *written;
  EXPECT_EQ(out.at("method").get<std::string>(), "analytic");
  EXPECT_EQ(out.at("variant").get<std::string>(), variant);
  if (variant == "pc") {
    EXPECT_EQ(out.at("pc_floor").get<double>(), kPcWiringFloor);
    EXPECT_EQ(calc.step<fb::HessianStep>()->pc_floor, kPcWiringFloor);
    EXPECT_EQ(calc.step<fb::Nevpt2GradientStep>()->pc_floor, kPcWiringFloor);
  } else {
    EXPECT_FALSE(out.contains("pc_floor"));
  }
  EXPECT_FALSE(out.contains("ladder"));
  EXPECT_FALSE(out.contains("displacement"));
  const auto got = out.at("hessian").get<std::vector<std::vector<double>>>();
  const fb::HessianOutput &product = *calc.get<fb::HessianOutput>();
  ASSERT_EQ(got.size(), n);
  double largest = 0.0;
  for (const double v : want.hessian) {
    largest = std::max(largest, std::abs(v));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(got[i].size(), n);
    for (std::size_t j = 0; j < n; ++j) {
      // The direct call's is column-major and exactly symmetric.
      const double expected = want.hessian[i + (n * j)];
      EXPECT_NEAR(got[i][j], expected, kWiring * largest)
          << variant << " (" << i << ", " << j << ")";
      EXPECT_EQ(product.hessian[(i * n) + j], got[i][j]);
      worst = std::max(worst, std::abs(got[i][j] - expected));
    }
  }
  // The QD block: the call's own report, at the same solution.
  const fb::json &qd = out.at("qd");
  EXPECT_EQ(qd.at("state").get<std::int32_t>(), kState);
  const auto eigenvalues = qd.at("eigenvalues").get<std::vector<double>>();
  ASSERT_EQ(eigenvalues.size(), want.eigenvalues.size());
  for (std::size_t k = 0; k < eigenvalues.size(); ++k) {
    EXPECT_NEAR(eigenvalues[k], want.eigenvalues[k], 1e-10) << k;
  }
  EXPECT_EQ(qd.at("energy").get<double>(), eigenvalues[static_cast<std::size_t>(kState)]);
  EXPECT_GT(qd.at("term_ii_share").get<double>(), 0.0);
  EXPECT_NEAR(qd.at("heff_gap").get<double>(), want.qd_gap, 1e-10);
  EXPECT_NEAR(qd.at("root_gap").get<double>(), want.root_gap, 1e-10);
  for (const std::string_view key : {"edge", "inactive_gap", "max_theta", "max_kappa_oo",
                                     "max_zeta", "max_zc", "cp_residual", "zvector_residual"}) {
    EXPECT_TRUE(qd.contains(key)) << key;
  }
  EXPECT_EQ(qd.contains("kept"), variant == "pc");
  EXPECT_EQ(qd.contains("min_denominator"), variant == "sc");
  // Every vibration published: C1, no pair refused.
  EXPECT_EQ(out.at("frequencies").size(), n - 6);
  EXPECT_EQ(product.vibrations.frequencies.size(), n - 6);
  EXPECT_TRUE(product.dipole_derivative.empty());
  std::cout << std::format("[ qd nevpt2 analytic hessian ] {} state {}: {:.1f} s (outer chain, "
                           "the step and its nested chain), max |H - direct| {:.2e} = {:.2e} of "
                           "the largest, asymmetry {:.2e}, translational sum {:.2e}, qd {}\n",
                           variant, kState, seconds, worst, worst / largest, product.asymmetry,
                           product.translational_sum, qd.dump());
}

TEST(PipelineQdNevpt2AnalyticHessian, ScSkewStateOneMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("sc");
}

TEST(PipelineQdNevpt2AnalyticHessian, PcSkewStateOneMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_skew("pc");
}

// The A1/B2 case, PC in the case's coordinates: the refusal of the lifted
// B2 blocks reaches the step's output, naming each block.
TEST(PipelineQdNevpt2AnalyticHessian, PcCoupledA1B2RefusalNamesTheB2Blocks) {
  FBF_REQUIRE_GPU();
  const fb::json &doc = reference("h2nb_sto3g_sa2");
  ASSERT_FALSE(doc.is_null());
  std::set<std::pair<std::string, std::string>> expected;
  for (const fb::json &block : doc.at("expected_refusals").at("pc")) {
    const auto c = block.at("coordinates").get<std::vector<std::string>>();
    expected.emplace(std::min(c[0], c[1]), std::max(c[0], c[1]));
  }
  ASSERT_EQ(expected.size(), 3U);
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  std::string table = "pc_floor = 0.0\ncoordinates = [\n";
  for (const auto &row :
       doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>()) {
    table += "  [";
    for (std::size_t x = 0; x < row.size(); ++x) {
      table += std::format("{}{:.17g}", x == 0 ? "" : ", ", row[x]);
    }
    table += "],\n";
  }
  table += "]\n";
  const fb::test::ScratchDir dir("pipeline_qd_nevpt2_analytic_hessian_a1b2");
  auto read = fb::read_config(write_config(dir, doc, "pc", table), dir.path(), dir / "out");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  fb::Calculation calc = std::move(*read);
  fb::NodeResources node;
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc.run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_FALSE(ran.has_value()) << "no block refused";
  ASSERT_EQ(ran.error().kind, fb::ErrorKind::Numerical) << ran.error().message;
  const std::string &message = ran.error().message;
  std::cout << std::format("[ qd nevpt2 analytic hessian ] h2nb pc state {}: {:.1f} s: {}\n",
                           kState, seconds, message);

  // The blocks it names: df_qd_pc_nevpt2's list, "coordinate pairs (i, j),
  // ...:", by the case's labels.
  const std::string_view lead = "PC-QD-NEVPT2 has no Hessian on the coordinate pairs ";
  const std::size_t from = message.find(lead);
  ASSERT_NE(from, std::string::npos) << message;
  const std::string list =
      message.substr(from + lead.size(), message.find(':', from) - from - lead.size());
  std::set<std::pair<std::string, std::string>> named;
  const std::regex pair_re(R"(\((\d+), (\d+)\))");
  for (auto it = std::sregex_iterator(list.begin(), list.end(), pair_re);
       it != std::sregex_iterator(); ++it) {
    const std::string &u = labels.at(std::stoul((*it)[1].str()));
    const std::string &v = labels.at(std::stoul((*it)[2].str()));
    named.emplace(std::min(u, v), std::max(u, v));
  }
  EXPECT_EQ(named, expected) << list;
  // No Hessian, so nothing written.
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "hessian.json"));
}

} // namespace
