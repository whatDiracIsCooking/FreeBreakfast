// GPU case for the Nevpt2Gradient step through the real registry: a TOML
// config read, planned and run by Calculation -- the DF-RHF, the DF-CASSCF
// from its orbitals and the SC- or PC-NEVPT2 gradient of its root -- for
// data/nevpt2_gradient/nevpt2_gradient.json's skewed H2O (cc-pVDZ /
// cc-pVDZ-JKFIT, CAS(4,4), every component nonzero). Its
// nevpt2_gradient.json against nevpt2::sc_gradient or nevpt2::pc_gradient
// called directly over the same Calculation's products (every term, to
// round-off), and against the reference's numerical gradient of that variant
// (Richardson over re-converged energies) per component to 1e-6, as
// nevpt2_gradient_gpu.cpp holds sc_gradient and pc_gradient.
//
// With quasi_degenerate: data/qd_nevpt2_gradient/'s h6_631g_sa2 (6-31G /
// cc-pVDZ-JKFIT, CAS(2,3) averaged over two roots) at QD state 1, SC and PC,
// against #696's numerical g . u of that state within twice each number's
// error bar; and a PC-QD refusal reported as the [nevpt2] step reports it.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::string_view kCase = "h2o_skew_ccpvdz";
// nevpt2_gradient_gpu.cpp's bars: each component within 1e-6 Eh/a0 of the
// numerical gradient, itself a difference; E_corr against the reference's
// to 1e-9, and the CASSCF energy to mcscf_gradient_gpu.cpp's 1e-8, at the
// CASSCF's convergence kConvTolGrad.
constexpr double kNumericalTolerance = 1e-6;
constexpr double kConvTolGrad = 1e-9;
constexpr double kCorrTolerance = 1e-9;
constexpr double kCasscfEnergyTolerance = 1e-8;
// The step against sc_gradient called directly over the same products: the
// same calls on the same inputs, apart only by the einsums' atomics, which
// reassociate every sum.
constexpr double kSameTolerance = 1e-10;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_gradient" /
                              "nevpt2_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in nevpt2_gradient.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule as molecule.xyz in @p dir (bohr there, Angstrom
// here), and run.toml naming it in the reference's basis, over its fitting
// set, with its active space in [casscf] and [nevpt2_gradient] of
// @p variant after it.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view variant) {
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
      "[casscf]\nncas = {}\nnelecas = {}\nactive = [{}]\nconv_tol_grad = {:g}\n"
      "[nevpt2_gradient]\nvariant = \"{}\"\n",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(),
      entry.at("aux_basis").at("name").get<std::string>(), entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), active, kConvTolGrad, variant);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The mcscf::Solution of @p calc's CasscfOutput, built as the step builds it.
fb::mcscf::Solution direct_solution(fb::Calculation &calc, fb::DeviceResources &res) {
  const auto &cas = *calc.get<fb::CasscfOutput>();
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, calc.get<fb::BasisSetOutput>()->basis),
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
      .iterations = cas.iterations};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  return solution;
}

// nevpt2::sc_gradient over @p calc's products.
fb::Result<nevpt2::ScGradient> direct_gradient(fb::Calculation &calc, fb::DeviceResources &res) {
  const auto &cas = *calc.get<fb::CasscfOutput>();
  const auto &df = std::get<fb::DensityFitted>(calc.get<fb::EriOutput>()->eri);
  const fb::mcscf::Solution solution = direct_solution(calc, res);
  return nevpt2::sc_gradient(res, calc.get<fb::MoleculeOutput>()->molecule,
                             calc.get<fb::BasisSetOutput>()->basis,
                             calc.get<fb::PrimitivePairClassesOutput>()->classes, df.aux,
                             calc.get<fb::CoreHamiltonianOutput>()->h, df.b, df.l.data(), solution,
                             cas.ncore, cas.ncas, 0);
}

// nevpt2::pc_gradient over @p calc's products.
fb::Result<nevpt2::PcGradient> direct_pc_gradient(fb::Calculation &calc, fb::DeviceResources &res) {
  const auto &cas = *calc.get<fb::CasscfOutput>();
  const auto &df = std::get<fb::DensityFitted>(calc.get<fb::EriOutput>()->eri);
  const fb::mcscf::Solution solution = direct_solution(calc, res);
  return nevpt2::pc_gradient(res, calc.get<fb::MoleculeOutput>()->molecule,
                             calc.get<fb::BasisSetOutput>()->basis,
                             calc.get<fb::PrimitivePairClassesOutput>()->classes, df.aux,
                             calc.get<fb::CoreHamiltonianOutput>()->h, df.b, df.l.data(), solution,
                             cas.ncore, cas.ncas, 0);
}

// The worst |file - direct| over @p rows against @p g, n_atom x 3.
double worst_against(const fb::json &rows, const std::vector<double> &g) {
  const auto got = rows.get<std::vector<std::vector<double>>>();
  EXPECT_EQ(3 * got.size(), g.size());
  double worst = 0.0;
  for (std::size_t a = 0; a < got.size() && 3 * a + 2 < g.size(); ++a) {
    EXPECT_EQ(got[a].size(), 3U);
    for (std::size_t k = 0; k < 3 && k < got[a].size(); ++k) {
      worst = std::max(worst, std::abs(got[a][k] - g[(3 * a) + k]));
    }
  }
  return worst;
}

TEST(PipelineNevpt2Gradient, JsonMatchesScGradientAndTheNumericalGradient) {
  FBF_REQUIRE_GPU();
  const auto start = std::chrono::steady_clock::now();
  const fb::json &entry = reference_case(kCase);
  ASSERT_EQ(entry.at("weights").size(), 1U) << "a state-specific case";
  const fb::test::ScratchDir dir("pipeline_nevpt2_gradient");
  auto calc = fb::read_config(write_config(dir, entry, "sc"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;

  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("root").get<std::int32_t>(), 0);
  EXPECT_EQ(written->at("variant").get<std::string>(), "sc");
  const double e_corr = written->at("e_corr").get<double>();
  const double e_casscf = written->at("e_casscf").get<double>();
  EXPECT_NEAR(e_corr, entry.at("e_corr").at("sc").get<double>(), kCorrTolerance);
  EXPECT_NEAR(e_casscf, entry.at("energies").at("casscf").get<double>(), kCasscfEnergyTolerance);
  EXPECT_DOUBLE_EQ(written->at("energy").get<double>(), e_casscf + e_corr);

  // Against the reference's numerical gradient, per component.
  const auto total = written->at("total").get<std::vector<std::vector<double>>>();
  const fb::json &rows = entry.at("gradients").at("sc").at("richardson");
  ASSERT_EQ(total.size(), rows.size());
  double numerical = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    ASSERT_EQ(total[a].size(), 3U);
    for (std::size_t k = 0; k < 3; ++k) {
      const double ref = rows[a][k].get<double>();
      EXPECT_NEAR(total[a][k], ref, kNumericalTolerance) << "[" << a << "][" << k << "]";
      numerical = std::max(numerical, std::abs(total[a][k] - ref));
    }
  }

  // Against sc_gradient called directly over the same products: the total,
  // every term, E_corr and the diagnostics.
  const auto direct = direct_gradient(*calc, node.device(0));
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  const fb::mcscf::GradientTerms &t = direct->terms;
  double same = worst_against(written->at("total"), t.total);
  const fb::json &terms = written->at("terms");
  EXPECT_EQ(terms.size(), 5U);
  for (const auto &[key, g] :
       {std::pair{"kinetic", &t.kinetic}, std::pair{"potential", &t.potential},
        std::pair{"overlap", &t.overlap}, std::pair{"two_electron", &t.two_electron},
        std::pair{"nuclear_repulsion", &t.nuclear_repulsion}}) {
    const double err = worst_against(terms.at(key), *g);
    EXPECT_LT(err, kSameTolerance) << key;
    same = std::max(same, err);
  }
  EXPECT_LT(same, kSameTolerance);
  EXPECT_NEAR(e_corr, direct->energies.total, kSameTolerance);
  EXPECT_NEAR(written->at("max_zc").get<double>(), direct->zc.max_z, kSameTolerance);
  EXPECT_NEAR(written->at("min_inactive_gap").get<double>(), direct->zc.min_gap, kSameTolerance);
  EXPECT_EQ(written->at("pcg_iterations").get<std::int32_t>(), direct->multipliers.iterations);
  EXPECT_EQ(written->at("asymmetry").size(), 6U);
  for (const auto &item : written->at("asymmetry").items()) {
    EXPECT_LT(item.value().get<double>(), 1e-8) << item.key();
  }

  // The product is what the file holds.
  const auto &out = calc->get<fb::Nevpt2GradientOutput>();
  ASSERT_TRUE(out.has_value());
  ASSERT_TRUE(std::holds_alternative<nevpt2::ScGradient>(out->gradient));
  EXPECT_EQ(std::get<nevpt2::ScGradient>(out->gradient).terms.total.size(),
            3 * entry.at("atoms").size());
  std::cout << std::format(
      "[ nevpt2_gradient ] {} (run {:.2f} s): numerical {:.2e}, direct "
      "{:.2e}, E_corr {:+.2e}, max |z^c| {:.3e}, PCG {} iterations\n",
      kCase, seconds.count(), numerical, same, e_corr - entry.at("e_corr").at("sc").get<double>(),
      written->at("max_zc").get<double>(), written->at("pcg_iterations").get<std::int32_t>());
}

// The same run with variant = "pc": its nevpt2_gradient.json against
// nevpt2::pc_gradient called directly (round-off), and against the
// reference's numerical PC gradient per component to 1e-6; E_corr to 1e-9
// against e_corr.pc, and the metric modes each class cut, PC's one addition
// to the file.
TEST(PipelineNevpt2Gradient, PcJsonMatchesPcGradientAndTheNumericalGradient) {
  FBF_REQUIRE_GPU();
  const auto start = std::chrono::steady_clock::now();
  const fb::json &entry = reference_case(kCase);
  ASSERT_EQ(entry.at("weights").size(), 1U) << "a state-specific case";
  const fb::test::ScratchDir dir("pipeline_nevpt2_gradient_pc");
  auto calc = fb::read_config(write_config(dir, entry, "pc"), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;

  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("root").get<std::int32_t>(), 0);
  EXPECT_EQ(written->at("variant").get<std::string>(), "pc");
  const double e_corr = written->at("e_corr").get<double>();
  const double e_casscf = written->at("e_casscf").get<double>();
  EXPECT_NEAR(e_corr, entry.at("e_corr").at("pc").get<double>(), kCorrTolerance);
  EXPECT_NEAR(e_casscf, entry.at("energies").at("casscf").get<double>(), kCasscfEnergyTolerance);
  EXPECT_DOUBLE_EQ(written->at("energy").get<double>(), e_casscf + e_corr);

  // Against the reference's numerical PC gradient, per component.
  const auto total = written->at("total").get<std::vector<std::vector<double>>>();
  const fb::json &rows = entry.at("gradients").at("pc").at("richardson");
  ASSERT_EQ(total.size(), rows.size());
  double numerical = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    ASSERT_EQ(total[a].size(), 3U);
    for (std::size_t k = 0; k < 3; ++k) {
      const double ref = rows[a][k].get<double>();
      EXPECT_NEAR(total[a][k], ref, kNumericalTolerance) << "[" << a << "][" << k << "]";
      numerical = std::max(numerical, std::abs(total[a][k] - ref));
    }
  }

  // Against pc_gradient called directly over the same products: the total,
  // every term, E_corr, the diagnostics and the dropped modes.
  const auto direct = direct_pc_gradient(*calc, node.device(0));
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  const fb::mcscf::GradientTerms &t = direct->terms;
  double same = worst_against(written->at("total"), t.total);
  const fb::json &terms = written->at("terms");
  EXPECT_EQ(terms.size(), 5U);
  for (const auto &[key, g] :
       {std::pair{"kinetic", &t.kinetic}, std::pair{"potential", &t.potential},
        std::pair{"overlap", &t.overlap}, std::pair{"two_electron", &t.two_electron},
        std::pair{"nuclear_repulsion", &t.nuclear_repulsion}}) {
    const double err = worst_against(terms.at(key), *g);
    EXPECT_LT(err, kSameTolerance) << key;
    same = std::max(same, err);
  }
  EXPECT_LT(same, kSameTolerance);
  EXPECT_NEAR(e_corr, direct->energies.total, kSameTolerance);
  EXPECT_NEAR(written->at("max_zc").get<double>(), direct->zc.max_z, kSameTolerance);
  EXPECT_NEAR(written->at("min_inactive_gap").get<double>(), direct->zc.min_gap, kSameTolerance);
  EXPECT_EQ(written->at("pcg_iterations").get<std::int32_t>(), direct->multipliers.iterations);
  EXPECT_EQ(written->at("asymmetry").size(), 6U);
  for (const auto &item : written->at("asymmetry").items()) {
    EXPECT_LT(item.value().get<double>(), 1e-8) << item.key();
  }
  EXPECT_EQ(
      written->at("dropped").get<std::vector<std::int64_t>>(),
      std::vector<std::int64_t>(direct->energies.dropped.begin(), direct->energies.dropped.end()));

  // The product is what the file holds.
  const auto &out = calc->get<fb::Nevpt2GradientOutput>();
  ASSERT_TRUE(out.has_value());
  ASSERT_TRUE(std::holds_alternative<nevpt2::PcGradient>(out->gradient));
  EXPECT_EQ(std::get<nevpt2::PcGradient>(out->gradient).terms.total.size(),
            3 * entry.at("atoms").size());
  std::cout << std::format(
      "[ nevpt2_gradient ] {} PC (run {:.2f} s): numerical {:.2e}, direct "
      "{:.2e}, E_corr {:+.2e}, max |z^c| {:.3e}, PCG {} iterations\n",
      kCase, seconds.count(), numerical, same, e_corr - entry.at("e_corr").at("pc").get<double>(),
      written->at("max_zc").get<double>(), written->at("pcg_iterations").get<std::int32_t>());
}

// --- Quasi-degenerate -------------------------------------------------------

// nevpt2_gradient_gpu.cpp's bar against #696's numbers: twice the committed
// error bar of each one; and the generator's DF-SA-CASSCF convergence
// (data/qd_nevpt2_gradient/README.md, "What each point is").
constexpr double kQdBarFactor = 2.0;
constexpr double kQdConvTolGrad = 1e-10;
constexpr double kQdConvTol = 1e-13;
constexpr std::int32_t kQdMaxCycle = 200;

const fb::json &qd_gradient_data(std::string_view name) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto it = docs.find(name); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2_gradient" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string(name), read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

// The case @p data's own molecule, basis, fitting set, active space, roots
// and weights as run.toml in @p dir, its DF-SA-CASSCF at the generator's
// bars, and [nevpt2_gradient] quasi_degenerate at @p state of @p variant.
std::filesystem::path write_qd_config(const fb::test::ScratchDir &dir, const fb::json &data,
                                      std::int32_t state, std::string_view variant) {
  std::string xyz =
      std::format("{}\n{}\n", data.at("atoms").size(), data.at("name").get<std::string>());
  for (const fb::json &atom : data.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : data.at("weights").get<std::vector<double>>()) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  // A case with no active list takes the frontier orbitals, as the
  // generator does.
  std::string active;
  if (data.contains("active") && !data.at("active").is_null()) {
    for (const auto t : data.at("active").get<std::vector<std::int32_t>>()) {
      active += std::format("{}{}", active.empty() ? "" : ", ", t);
    }
    active = std::format("active = [{}]\n", active);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\n{}nroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n"
      "conv_tol = {:g}\nmax_cycle = {}\n"
      "[nevpt2_gradient]\nquasi_degenerate = true\nstate = {}\nvariant = \"{}\"\n",
      data.at("basis").get<std::string>(), data.at("aux_basis").get<std::string>(),
      data.at("ncas").get<std::int32_t>(), data.at("nelecas").get<std::int32_t>(), active,
      data.at("weights").size(), weights, kQdConvTolGrad, kQdConvTol, kQdMaxCycle, state, variant);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// Run the case @p name at QD state @p state of @p variant through
// Calculation, and hold its nevpt2_gradient.json's total along each of #696's
// directions to twice that number's error bar.
void expect_qd_state(std::string_view name, std::int32_t state, std::string_view variant) {
  const auto start = std::chrono::steady_clock::now();
  const fb::json &data = qd_gradient_data(name);
  ASSERT_EQ(data.at("name").get<std::string>(), name);
  const auto k = static_cast<std::size_t>(state);
  ASSERT_LT(k, data.at("weights").size());
  const fb::test::ScratchDir dir(std::format("pipeline_nevpt2_gradient_qd_{}", variant));
  auto calc = fb::read_config(write_qd_config(dir, data, state, variant), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;

  const auto written = fb::read_json(dir / "out" / "nevpt2_gradient.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_TRUE(written->at("quasi_degenerate").get<bool>());
  EXPECT_EQ(written->at("state").get<std::int32_t>(), state);
  EXPECT_EQ(written->at("variant").get<std::string>(), variant);
  const auto qd_energies = written->at("qd_energies").get<std::vector<double>>();
  ASSERT_EQ(qd_energies.size(), data.at("weights").size());
  EXPECT_EQ(written->at("energy").get<double>(), qd_energies[k]);

  // The product is what the file holds, the variant's QD alternative.
  const auto &out = calc->get<fb::Nevpt2GradientOutput>();
  ASSERT_TRUE(out.has_value());
  const std::vector<double> &g = variant == "pc"
                                     ? std::get<nevpt2::QdPcGradient>(out->gradient).total
                                     : std::get<nevpt2::QdScGradient>(out->gradient).total;
  ASSERT_EQ(g.size(), 3 * data.at("atoms").size());
  const auto rows = written->at("total").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(3 * rows.size(), g.size());
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t c = 0; c < 3; ++c) {
      EXPECT_EQ(rows[a].at(c), g[(3 * a) + c]) << a << ", " << c;
    }
  }

  // Against #696's numbers of state K along each stored direction.
  const fb::json &directional = data.at("directional").at(variant).at(k);
  const fb::json &directions = data.at("directions");
  double worst = 0.0;
  for (std::size_t d = 0; d < directions.size(); ++d) {
    const auto u = directions.at(d).get<std::vector<double>>();
    ASSERT_EQ(u.size(), g.size());
    double ours = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      ours += g[i] * u[i];
    }
    const double ref = directional.at("richardson").at(d).get<double>();
    const double bar = directional.at("error_bar").at(d).get<double>();
    EXPECT_LE(std::abs(ours - ref), kQdBarFactor * bar) << variant << " state " << k << " u" << d;
    worst = std::max(worst, std::abs(ours - ref) / bar);
    std::cout << std::format("[ nevpt2_gradient ] {} QD {} state {} u{}: analytic {:+.10f} "
                             "numerical {:+.10f} diff {:+.2e} ({:.3f} bars of {:.1e})\n",
                             name, variant, k, d, ours, ref, ours - ref, std::abs(ours - ref) / bar,
                             bar);
  }
  std::cout << std::format(
      "[ nevpt2_gradient ] {} QD {} state {} (run {:.2f} s): worst {:.3f} bars, E_K {:.10f} "
      "({:+.2e} against #696's)\n",
      name, variant, k, seconds.count(), worst, qd_energies[k],
      qd_energies[k] -
          data.at("reference").at(std::format("{}_energies", variant)).at(k).get<double>());
}

// State 1, not 0, so an index or ordering error between H_eff's eigenvalues
// and the state wired through shows.
TEST(PipelineNevpt2Gradient, QdScStateOneMatchesTheNumericalGradient) {
  FBF_REQUIRE_GPU();
  expect_qd_state("h6_631g_sa2", 1, "sc");
}

TEST(PipelineNevpt2Gradient, QdPcStateOneMatchesTheNumericalGradient) {
  FBF_REQUIRE_GPU();
  expect_qd_state("h6_631g_sa2", 1, "pc");
}

// The class named before "'s metric" in @p message, or empty.
std::string refused_class(const std::string &message) {
  const std::size_t at = message.find("'s metric");
  if (at == std::string::npos) {
    return {};
  }
  const std::size_t from = message.rfind(' ', at);
  return message.substr(from == std::string::npos ? 0 : from + 1,
                        at - (from == std::string::npos ? 0 : from + 1));
}

// A PC-QD refusal is reported as the [nevpt2] step reports it: the
// pipeline_nevpt2_gpu.cpp PcRefusalKeepsScQd case (HF CAS(10,7) averaged over
// three roots, the F 1s active), with [nevpt2] quasi_degenerate beside
// [nevpt2_gradient] quasi_degenerate, variant = "pc". The energy step keeps
// SC and records PC's refusal; the gradient, which has no SC to keep, fails
// with the energy's Numerical error naming the same class, and writes
// nothing.
TEST(PipelineNevpt2Gradient, QdPcRefusalIsTheEnergyStepsRefusal) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_nevpt2_gradient_qd_refused");
  ASSERT_TRUE(
      fb::write_file(dir / "hf.xyz", "2\nhydrogen fluoride\nH 0 0 0\nF 0 0 0.917\n").has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"hf.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pVDZ-JKFIT\"\n"
                  "[casscf]\nncas = 7\nnelecas = 10\nnroots = 3\nconv_tol_grad = {:g}\n"
                  "[nevpt2]\nquasi_degenerate = true\n"
                  "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"pc\"\n",
                  kConvTolGrad);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::Numerical) << ran.error().message;
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "nevpt2_gradient.json"));

  // The energy step ran beside it: SC-QD kept, PC-QD refused.
  const auto &nevpt2 = calc->get<fb::Nevpt2Output>();
  ASSERT_TRUE(nevpt2.has_value());
  EXPECT_EQ(nevpt2->qd_energies.size(), 3U);
  ASSERT_FALSE(nevpt2->pc_qd_refusal.empty());
  EXPECT_NE(nevpt2->pc_qd_refusal.find("metric has no gap"), std::string::npos)
      << nevpt2->pc_qd_refusal;
  EXPECT_NE(ran.error().message.find("metric has no gap"), std::string::npos)
      << ran.error().message;
  const std::string energy_class = refused_class(nevpt2->pc_qd_refusal);
  EXPECT_FALSE(energy_class.empty()) << nevpt2->pc_qd_refusal;
  EXPECT_EQ(refused_class(ran.error().message), energy_class) << ran.error().message;
  std::cout << std::format("[ nevpt2_gradient ] HF CAS(10,7) SA-3 PC-QD: gradient '{}'; [nevpt2] "
                           "'{}'\n",
                           ran.error().message, nevpt2->pc_qd_refusal);
}

} // namespace
