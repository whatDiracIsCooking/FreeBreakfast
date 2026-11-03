// GPU case for the Nevpt2Nac step through the real registry: a TOML config
// read, planned and run by Calculation -- the DF-RHF, the DF-SA-CASSCF from
// its orbitals and the QD-NEVPT2 coupling of one pair of QD states -- for
// data/qd_nevpt2/'s h8_ring_sto3g_sa3 (STO-3G / cc-pVDZ-JKFIT, CAS(6,4)
// averaged over three roots, converged as #696's generator converges it).
// Pair (1, 2), both indices past 0, so an index or ordering error between
// H_eff's states and the pair wired through shows; once SC and once PC.
//
// Its nevpt2_nac.json and product against nevpt2::qd_sc_nac or qd_pc_nac
// called directly over the same Calculation's products: csf to 1e-10 of its
// own largest element (the milestone's path agreement: the same calls on the
// same inputs, apart only by the atomics' reassociation); h, cas, pt2 and
// total, which each pass through a Z-vector solve the two runs converge
// independently, to 1e-9 of the largest part summed into d (#775); and
// H_eff's gap at the pair against the call's own, to 1e-10 of the largest
// eigenvalue. The step's refusals of a malformed pair
// in run, over the same products, before anything runs.

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

constexpr std::string_view kCase = "h8_ring_sto3g_sa3";
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";
// #696's generator's bars, as nevpt2_nac_gpu.cpp converges the case.
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-13;
constexpr std::int32_t kMaxCycle = 200;
// The step against the direct call (#750's "How results are compared": path
// agreement, relative to the largest element of the quantity compared).
constexpr double kPathTolerance = 1e-10;
// Fields through a Z-vector solve: the step and the call converge theirs
// independently, so they agree to the solver, not round-off (#775,
// https://github.com/whatDiracIsCooking/FreeBreakfast/issues/767#issuecomment-6608119792).
constexpr double kSolveTolerance = 1e-9;
constexpr std::int32_t kBra = 1;
constexpr std::int32_t kKet = 2;

const fb::json &qd_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / "qd_nevpt2.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in qd_nevpt2/qd_nevpt2.json";
  static const fb::json kNone;
  return kNone;
}

// The case's molecule as molecule.xyz in @p dir (bohr there, Angstrom here),
// and run.toml over its basis, fitting set and state average at the
// generator's bars, with [nevpt2_nac] at pair (kBra, kKet) of @p variant.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view variant) {
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : entry.at("weights").get<std::vector<double>>()) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n"
      "conv_tol = {:g}\nmax_cycle = {}\n"
      "[nevpt2_nac]\npairs = [[{}, {}]]\nvariant = \"{}\"\n",
      entry.at("basis").get<std::string>(), kDfAuxBasis, entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), entry.at("weights").size(), weights, kConvTolGrad,
      kConvTol, kMaxCycle, kBra, kKet, variant);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The mcscf::Solution of @p calc's CasscfOutput, built as the step builds it
// (gamma the weighted average of the roots' 1-RDMs).
fb::mcscf::Solution direct_solution(const fb::Calculation &calc, fb::DeviceResources &res) {
  const auto &cas = *calc.get<fb::CasscfOutput>();
  const auto n2 = static_cast<std::size_t>(cas.ncas) * cas.ncas;
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, calc.get<fb::BasisSetOutput>()->basis),
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
      .iterations = cas.iterations};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return solution;
}

double largest_abs(std::span<const double> v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

// max |ours - want| / scale (max |want| unless given), EXPECTed within
// @p tolerance.
double relative(std::span<const double> ours, std::span<const double> want, std::string_view what,
                const double tolerance = kPathTolerance, double scale = 0.0) {
  EXPECT_EQ(ours.size(), want.size()) << what;
  scale = scale > 0.0 ? scale : largest_abs(want);
  EXPECT_GT(scale, 0.0) << what;
  double m = 0.0;
  for (std::size_t i = 0; i < std::min(ours.size(), want.size()); ++i) {
    m = std::max(m, std::abs(ours[i] - want[i]));
  }
  const double rel = scale > 0.0 ? m / scale : m;
  EXPECT_LE(rel, tolerance) << what << ": " << m << " of " << scale;
  return rel;
}

// An n_atom x 3 array of rows, flat.
std::vector<double> flat(const fb::json &rows) {
  std::vector<double> out;
  for (const auto &row : rows.get<std::vector<std::vector<double>>>()) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

// Run the case's pair (kBra, kKet) of @p variant through Calculation, and
// hold it to the direct call over the same products.
void expect_step_matches_direct(std::string_view variant) {
  const auto start = std::chrono::steady_clock::now();
  const fb::json &entry = qd_case(kCase);
  ASSERT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir(std::format("pipeline_nevpt2_nac_{}", variant));
  auto calc = fb::read_config(write_config(dir, entry, variant), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const std::chrono::duration<double> step_seconds = std::chrono::steady_clock::now() - start;
  ASSERT_LT(calc->get<fb::CasscfOutput>()->gradient_norm, kConvTolGrad);

  const auto &out = calc->get<fb::Nevpt2NacOutput>();
  ASSERT_TRUE(out.has_value());
  const std::size_t n = entry.at("weights").size();
  ASSERT_EQ(out->energies.size(), n);
  ASSERT_EQ(out->vectors.size(), n * n);
  ASSERT_EQ(out->pairs.size(), 1U);
  EXPECT_EQ(out->pairs[0], (std::array<std::int32_t, 2>{kBra, kKet}));
  ASSERT_EQ(out->couplings.size(), 1U);
  const nevpt2::QdNac &ours = out->couplings[0];

  // The file is the product.
  const auto written = fb::read_json(dir / "out" / "nevpt2_nac.json");
  ASSERT_TRUE(written.has_value()) << written.error().message;
  EXPECT_EQ(written->at("variant").get<std::string>(), variant);
  EXPECT_FALSE(written->at("mult_ediff").get<bool>());
  EXPECT_FALSE(written->at("use_etfs").get<bool>());
  EXPECT_EQ(written->at("e_casscf").get<std::vector<double>>(),
            calc->get<fb::CasscfOutput>()->energies);
  EXPECT_EQ(written->at("qd_energies").get<std::vector<double>>(), out->energies);
  const auto r = written->at("r").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(r.size(), n);
  for (std::size_t k = 0; k < n; ++k) {
    ASSERT_EQ(r[k].size(), n);
    for (std::size_t m = 0; m < n; ++m) {
      EXPECT_EQ(r[k][m], out->vectors[(k * n) + m]) << k << ", " << m;
    }
    // The sign rule: each column's largest-magnitude component positive.
    EXPECT_GT(*std::ranges::max_element(r[k], {}, [](double x) { return std::abs(x); }), 0.0);
  }
  ASSERT_EQ(written->at("couplings").size(), 1U);
  const fb::json &pair = written->at("couplings").at(0);
  EXPECT_EQ(pair.at("bra").get<std::int32_t>(), kBra);
  EXPECT_EQ(pair.at("ket").get<std::int32_t>(), kKet);
  EXPECT_EQ(pair.at("energy_gap").get<double>(), out->energies[kKet] - out->energies[kBra]);
  EXPECT_EQ(flat(pair.at("h")), ours.numerator);
  EXPECT_EQ(flat(pair.at("cas")), ours.cas);
  EXPECT_EQ(flat(pair.at("pt2")), ours.pt2);
  EXPECT_EQ(flat(pair.at("csf")), ours.csf);
  EXPECT_EQ(flat(pair.at("total")), ours.total);
  EXPECT_EQ(pair.at("heff_gap").get<double>(), ours.heff_gap);
  EXPECT_EQ(pair.at("max_zeta").get<double>(), ours.max_zeta);
  using Iterations = std::array<std::int32_t, 3>;
  EXPECT_EQ(pair.at("zvector").at("iterations").get<Iterations>(), ours.iterations);

  // The direct call over the same products.
  fb::DeviceResources &res = node.device(0);
  const auto &cas = *calc->get<fb::CasscfOutput>();
  const auto &df = std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  const fb::mcscf::Solution solution = direct_solution(*calc, res);
  const auto direct_start = std::chrono::steady_clock::now();
  auto direct = variant == "pc"
                    ? nevpt2::qd_pc_nac(res, calc->get<fb::MoleculeOutput>()->molecule,
                                        calc->get<fb::BasisSetOutput>()->basis,
                                        calc->get<fb::PrimitivePairClassesOutput>()->classes,
                                        df.aux, calc->get<fb::CoreHamiltonianOutput>()->h, df.b,
                                        df.l.data(), solution, cas.ncore, cas.ncas, kBra, kKet, 0)
                    : nevpt2::qd_sc_nac(res, calc->get<fb::MoleculeOutput>()->molecule,
                                        calc->get<fb::BasisSetOutput>()->basis,
                                        calc->get<fb::PrimitivePairClassesOutput>()->classes,
                                        df.aux, calc->get<fb::CoreHamiltonianOutput>()->h, df.b,
                                        df.l.data(), solution, cas.ncore, cas.ncas, kBra, kKet, 0);
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  const std::chrono::duration<double> direct_seconds =
      std::chrono::steady_clock::now() - direct_start;
  // The largest part summed into d: h over the gap, cas, pt2, csf; h is
  // compared on the scale times the gap.
  const double gap_pq = direct->heff_gap;
  const double d_scale =
      std::max({largest_abs(direct->numerator) / gap_pq, largest_abs(direct->cas),
                largest_abs(direct->pt2), largest_abs(direct->csf)});
  const double h =
      relative(ours.numerator, direct->numerator, "h", kSolveTolerance, d_scale * gap_pq);
  const double d_cas = relative(ours.cas, direct->cas, "cas", kSolveTolerance, d_scale);
  const double d_pt2 = relative(ours.pt2, direct->pt2, "pt2", kSolveTolerance, d_scale);
  const double csf = relative(ours.csf, direct->csf, "csf");
  const double total = relative(ours.total, direct->total, "total", kSolveTolerance, d_scale);
  // H_eff's gap at the pair, the step's own H_eff against the call's.
  const double gap = std::abs((out->energies[kKet] - out->energies[kBra]) - direct->heff_gap) /
                     largest_abs(out->energies);
  EXPECT_LE(gap, kPathTolerance);

  std::cout << std::format("[ nevpt2_nac ] {} {}-QD ({}, {}): step against direct, relative: h "
                           "{:.2e}, cas {:.2e}, pt2 {:.2e}, csf {:.2e}, total {:.2e}, H_eff gap "
                           "{:.2e}; max |d| {:.4e} (step run {:.2f} s, direct {:.2f} s)\n",
                           kCase, variant, kBra, kKet, h, d_cas, d_pt2, csf, total, gap,
                           largest_abs(direct->total), step_seconds.count(),
                           direct_seconds.count());

  // The refusals in run, over the same products, before anything runs: a
  // pair of equal states, and one past H_eff's.
  const fb::StepInput<fb::Nevpt2NacStep> in{*calc->get<fb::MoleculeOutput>(),
                                            *calc->get<fb::BasisSetOutput>(),
                                            *calc->get<fb::PrimitivePairClassesOutput>(),
                                            *calc->get<fb::EriOutput>(),
                                            *calc->get<fb::CoreHamiltonianOutput>(),
                                            *calc->get<fb::ScfOutput>(),
                                            *calc->get<fb::CasscfOutput>()};
  const std::vector<std::array<std::int32_t, 2>> bad_pairs{{1, 1}, {0, 3}};
  for (const std::array<std::int32_t, 2> &bad : bad_pairs) {
    const auto refused = fb::Nevpt2NacStep{.pairs = {bad}}.run(in, res);
    ASSERT_FALSE(refused.has_value()) << bad[0] << ", " << bad[1];
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
    EXPECT_NE(refused.error().message.find("is not two distinct QD states"), std::string::npos)
        << refused.error().message;
  }
}

TEST(PipelineNevpt2Nac, ScStepMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  expect_step_matches_direct("sc");
}

TEST(PipelineNevpt2Nac, PcStepMatchesTheDirectCall) {
  FBF_REQUIRE_GPU();
  expect_step_matches_direct("pc");
}

} // namespace
