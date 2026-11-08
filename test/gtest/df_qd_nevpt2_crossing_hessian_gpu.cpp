// GPU cases for hessian::df_qd_sc_nevpt2_crossing and df_qd_pc_nevpt2_crossing,
// the QD-NEVPT2 Hessian at a crossing (docs/nevpt2/nevpt2.tex Eqs. qd-pair and
// qd-lowdin), and the W forms at a seam's W, on this tree's own DF-SA-CASSCF
// of data/qd_nevpt2_hessian/'s cases through the registry:
//
// - the pair's sum d^2 (E_0 + E_1), c = 1, at the ethylene SA3 MECIs against
//   #953's G(W^0 + W^1) ladders within 2x their bars, SC and PC, with the
//   outside-state sum (the third state's) above 10 bars; and away from any
//   crossing on h2co_skew_ccpvdz_sa3's pair (0, 1), SC and PC;
// - the seam's 2 x 2 (Eq. qd-lowdin) at the MECIs, on its own: each element
//   symmetric to the solves, the sum of its diagonal the pair's ladder, and
//   in a basis of the eigenspace turned by 0.3 rad its elements the turn of
//   these, and so its determinant (its eigenvalues) the same, within 2x the
//   fixture's seam bar;
// - term (i) at the fixture's seam W = 1/2 (R_0 R_1^T + R_1 R_0^T), R its
//   recorded eigenvectors signed into this solve's root phases, through the
//   W form, within 2x the seam ladder's bars, SC and PC;
// - the refusals: a third state within kRootGapFloor of the pair (H_eff made
//   rank-one degenerate by the roots' energies alone), a pair that is not
//   two states, and a c that is not symmetric.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The reference SA-CASSCF's ||g|| and |Delta E| (the QD Hessian tests').
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;
// The issue's gate: every element within this many of the fixture's bars.
constexpr double kBars = 2.0;
// The visibility rule: the outside-state sum has an element above this many
// bars.
constexpr double kVisibleBars = 10.0;
// The asymmetry is the solves' error: first order in their residuals (1e-10)
// times the largest element of any part.
constexpr double kSolveTolerance = 1e-10;
// The angle (rad) the eigenspace's basis is turned by for the invariance.
constexpr double kTurn = 0.3;

double max_abs(std::span<const double> v) {
  return std::ranges::fold_left(
      v, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double solve_bar(const fb::hessian::QdScNevpt2Terms &h) {
  double largest = 0.0;
  for (const std::vector<double> *part :
       {&h.e_sa, &h.z_lxy, &h.z_lx, &h.z_third, &h.z_koo, &h.zeta, &h.skeleton, &h.hvp,
        &h.rotation_tau, &h.rotation_kappa, &h.term_ii}) {
    largest = std::max(largest, max_abs(*part));
  }
  return kSolveTolerance * largest;
}

const fb::json &document(const std::filesystem::path &relative) {
  static std::map<std::string, fb::json, std::less<>> docs;
  const std::string key = relative.string();
  if (const auto it = docs.find(key); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / relative);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(key, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &fixture(const std::string_view name) {
  return document(std::filesystem::path{"qd_nevpt2_hessian"} / std::format("{}.json", name));
}

std::vector<std::vector<double>> matrix(const fb::json &m) {
  return m.get<std::vector<std::vector<double>>>();
}

// One case: the registry's products and the Solution of its SA-CASSCF.
struct Reference {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;

  [[nodiscard]] fb::DeviceResources &res() const { return node->device(0); }
  [[nodiscard]] const fb::Molecule &molecule() const {
    return calc->get<fb::MoleculeOutput>()->molecule;
  }
  [[nodiscard]] const fb::BasisSet &basis() const { return calc->get<fb::BasisSetOutput>()->basis; }
  [[nodiscard]] std::span<const fb::PrimitivePairClass> classes() const {
    return calc->get<fb::PrimitivePairClassesOutput>()->classes;
  }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  }
  [[nodiscard]] const fb::SphericalMatrixAO &h() const {
    return calc->get<fb::CoreHamiltonianOutput>()->h;
  }

  // The crossing form of pair (p, q) at c, SC's or PC's, as SC's terms.
  [[nodiscard]] fb::Result<fb::hessian::QdScNevpt2Terms>
  crossing(const fb::mcscf::Solution &s, const bool pc, const std::int32_t p, const std::int32_t q,
           std::span<const double> c) const {
    if (pc) {
      auto t = fb::hessian::df_qd_pc_nevpt2_crossing(res(), molecule(), basis(), classes(),
                                                     fit().aux, h(), fit().b, fit().l.data(), s,
                                                     ncore, ncas, p, q, c);
      if (!t) {
        return std::unexpected(std::move(t.error()));
      }
      return std::move(t->terms);
    }
    return fb::hessian::df_qd_sc_nevpt2_crossing(res(), molecule(), basis(), classes(), fit().aux,
                                                 h(), fit().b, fit().l.data(), s, ncore, ncas, p, q,
                                                 c);
  }

  // The W form at @p w, SC's or PC's.
  [[nodiscard]] fb::Result<fb::hessian::QdScNevpt2Terms> w_form(const bool pc,
                                                                std::span<const double> w) const {
    if (pc) {
      auto t = fb::hessian::df_qd_pc_nevpt2(res(), molecule(), basis(), classes(), fit().aux, h(),
                                            fit().b, fit().l.data(), *solution, ncore, ncas, w);
      if (!t) {
        return std::unexpected(std::move(t.error()));
      }
      return std::move(t->terms);
    }
    return fb::hessian::df_qd_sc_nevpt2(res(), molecule(), basis(), classes(), fit().aux, h(),
                                        fit().b, fit().l.data(), *solution, ncore, ncas, w);
  }

  // A copy of the solution with other root energies (the vectors and orbitals
  // the same).
  [[nodiscard]] fb::mcscf::Solution with_energies(std::vector<double> energies) const {
    fb::DeviceResources &r = res();
    fb::mcscf::Solution s{.c = fb::SphericalMatrixAO(r, basis()),
                          .roots = {.energies = std::move(energies),
                                    .spin_squares = solution->roots.spin_squares,
                                    .vectors = fb::Buffers::device<double>(
                                        solution->roots.vectors.num_elements(), r.device()),
                                    .davidson = {},
                                    .fp32_iterations = 0},
                          .energy = solution->energy,
                          .gradient_norm = solution->gradient_norm,
                          .gamma = solution->gamma,
                          .gamma2 = solution->gamma2,
                          .root_gammas = solution->root_gammas,
                          .iterations = {},
                          .weights = solution->weights};
    EXPECT_EQ(wwr::wwrMemcpyAsync(s.c.data(), solution->c.data(),
                                  solution->c.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrMemcpyAsync(s.roots.vectors.data(), solution->roots.vectors.data(),
                                  s.roots.vectors.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(r.sync().has_value());
    return s;
  }

  // H_eff = diag(E_r) + (C + C^T) / 2, n x n row-major, SC's or PC's.
  [[nodiscard]] std::vector<double> heff(const bool pc) const {
    std::vector<double> totals;
    const fb::Status got = fb::nevpt2::qd_inputs(
        res(), molecule(), basis(), fit().aux, h(), fit().b, *solution, ncore, ncas,
        [&](const fb::nevpt2::MoBlocks &blocks,
            const std::span<const fb::nevpt2::EnergyInputs> pairs) -> fb::Status {
          if (pc) {
            const auto e = fb::nevpt2::pc_effective_hamiltonian(res(), blocks, pairs, 0);
            if (!e) {
              return std::unexpected(e.error());
            }
            std::ranges::transform(*e, std::back_inserter(totals),
                                   &fb::nevpt2::PcClassEnergies::total);
            return {};
          }
          const auto e = fb::nevpt2::sc_effective_hamiltonian(res(), blocks, pairs, 0);
          if (!e) {
            return std::unexpected(e.error());
          }
          std::ranges::transform(*e, std::back_inserter(totals), &fb::nevpt2::ClassEnergies::total);
          return {};
        });
    EXPECT_TRUE(got.has_value()) << got.error().message;
    const std::size_t n = solution->roots.energies.size();
    std::vector<double> out(n * n, 0.0);
    if (totals.size() != n * n) {
      ADD_FAILURE() << totals.size() << " elements for " << n << " roots";
      return out;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        out[(i * n) + j] = (i == j ? solution->roots.energies[i] : 0.0) +
                           (0.5 * (totals[(i * n) + j] + totals[(j * n) + i]));
      }
    }
    return out;
  }
};

std::unique_ptr<Reference> make_reference(const fb::json &doc, const std::string &name) {
  auto ref = std::make_unique<Reference>();
  ref->node = std::make_unique<fb::NodeResources>();
  const fb::test::ScratchDir dir("df_qd_nevpt2_crossing_hessian");
  const auto &atoms = doc.at("atoms");
  std::string xyz = std::format("{}\n{}\n", atoms.size(), name);
  for (const fb::json &atom : atoms) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const auto w = doc.at("weights").get<std::vector<double>>();
  std::string weights;
  for (const double x : w) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", x);
  }
  const fb::json &basis = doc.at("basis");
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol = {:g}\n"
      "conv_tol_grad = {:g}\n",
      basis.is_string() ? basis.get<std::string>() : basis.at("name").get<std::string>(),
      doc.at("aux_basis").at("name").get<std::string>(), doc.at("ncas").get<std::int32_t>(),
      doc.at("nelecas").get<std::int32_t>(), w.size(), weights, kConvTol, kConvTolGrad);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return ref;
  }
  const fb::Status ran = calc->run(*ref->node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return ref;
  }
  ref->calc.emplace(std::move(*calc));
  fb::DeviceResources &res = ref->res();
  const auto &cas = ref->calc->get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (!cas.has_value()) {
    return ref;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  ref->ncore = cas->ncore;
  ref->ncas = cas->ncas;
  const auto n2 = static_cast<std::size_t>(cas->ncas) * static_cast<std::size_t>(cas->ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas->weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas->weights[r] * cas->rdm1[(r * n2) + i];
    }
  }
  std::cout << std::format("[ casscf   ] {}: E {}, ||g|| {:.1e}\n", name, cas->energies,
                           cas->gradient_norm);
  ref->solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, ref->basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref->solution->roots.vectors.data(), cas->vectors.data(),
                                cas->vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref->solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return ref;
}

// One reference per case for the whole binary: the SA-CASSCF runs once.
const Reference &reference(const std::string &name) {
  static std::map<std::string, std::unique_ptr<Reference>, std::less<>> cache;
  auto &slot = cache[name];
  if (!slot) {
    slot = make_reference(fixture(name), name);
  }
  return *slot;
}

// The fixture's entry @p kind ("pairs", "seams") of @p scheme for (p, q).
const fb::json *entry(const fb::json &doc, const std::string_view scheme,
                      const std::string_view kind, const std::int32_t p, const std::int32_t q) {
  for (const fb::json &e : doc.at(std::string{scheme}).at(std::string{kind})) {
    const auto pq = e.at(kind == "seams" ? "seam" : "pair").get<std::vector<std::int32_t>>();
    if (pq == std::vector<std::int32_t>{p, q}) {
      return &e;
    }
  }
  return nullptr;
}

// The largest |part| in bars of the fixture's elements (k x k column-major).
double in_bars(std::span<const double> part, const std::vector<std::vector<double>> &bar) {
  double worst = 0.0;
  const std::size_t k = bar.size();
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      worst = std::max(worst, std::abs(part[i + (k * j)]) / bar[i][j]);
    }
  }
  return worst;
}

// Every element of @p got within kBars of @p want's hessian; the worst miss.
double expect_within(std::span<const double> got, const fb::json &want, const std::string &label) {
  const auto hessian = matrix(want.at("hessian"));
  const auto bar = matrix(want.at("bar"));
  const std::size_t k = hessian.size();
  EXPECT_EQ(got.size(), k * k) << label;
  if (got.size() != k * k) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const double value = got[i + (k * j)];
      const double miss = std::abs(value - hessian[i][j]) / bar[i][j];
      worst = std::max(worst, miss);
      EXPECT_LE(miss, kBars) << label << ": element (" << i << ", " << j << ") " << value
                             << " against " << hessian[i][j] << ", bar " << bar[i][j];
    }
  }
  return worst;
}

const std::vector<double> kIdentity{1.0, 0.0, 0.0, 1.0};

// The pair's sum d^2 (E_P + E_Q) of fixture @p name's pair (p, q) against its
// G(W^P + W^Q) ladder; at a crossing, the outside-state sum visible.
void expect_pair_sum(const std::string &name, const bool pc, const std::int32_t p,
                     const std::int32_t q, const bool at_crossing) {
  const fb::json &doc = fixture(name);
  const Reference &ref = reference(name);
  ASSERT_TRUE(ref.solution.has_value());
  const std::string_view scheme = pc ? "pc" : "sc";
  const fb::json *want = entry(doc, scheme, "pairs", p, q);
  ASSERT_NE(want, nullptr) << name << " has no " << scheme << " pair (" << p << ", " << q << ")";
  const std::string label = std::format("{} {} pair ({}, {})", name, scheme, p, q);
  const auto start = std::chrono::steady_clock::now();
  const auto got = ref.crossing(*ref.solution, pc, p, q, kIdentity);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(got.has_value()) << label << ": " << got.error().message;
  const double worst = expect_within(got->hessian, *want, label);
  const auto bar = matrix(want->at("bar"));
  const double outside = in_bars(got->term_ii, bar);
  std::cout << std::format(
      "[ qd pair  ] {}: worst element {:.3f} bars (allowed {:g}), largest |H| {:.3e}; outside "
      "sum {:.3e} bars (rule {:g}), share {:.3e}; E {}, gap to the outside {:.4f}, asymmetry "
      "{:.1e} (bar {:.1e}), sum rule {:.1e}; {:.1f} s\n",
      label, worst, kBars, max_abs(got->hessian), outside, kVisibleBars, got->term_ii_share,
      got->eigenvalues, got->qd_gap, got->asymmetry, solve_bar(*got), got->translation, seconds);
  EXPECT_LE(got->asymmetry, solve_bar(*got)) << label;
  if (at_crossing) {
    EXPECT_GT(outside, kVisibleBars) << label;
  }
}

TEST(HessianDfQdNevpt2Crossing, ScPairSumAtTheMeci) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_pair_sum("ethylene_ccpvdz_sa3_sc_meci", false, 0, 1, true);
}

TEST(HessianDfQdNevpt2Crossing, PcPairSumAtTheMeci) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_pair_sum("ethylene_ccpvdz_sa3_pc_meci", true, 0, 1, true);
}

TEST(HessianDfQdNevpt2Crossing, ScPairSumAwayFromACrossing) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_pair_sum("h2co_skew_ccpvdz_sa3", false, 0, 1, false);
}

TEST(HessianDfQdNevpt2Crossing, PcPairSumAwayFromACrossing) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_pair_sum("h2co_skew_ccpvdz_sa3", true, 0, 1, false);
}

// c for element (X, Y) of the 2 x 2 in the basis (cos t R_P + sin t R_Q,
// -sin t R_P + cos t R_Q): sym(u_X u_Y^T).
std::vector<double> element(const std::size_t x, const std::size_t y, const double t) {
  const std::array<std::array<double, 2>, 2> u{
      {{std::cos(t), std::sin(t)}, {-std::sin(t), std::cos(t)}}};
  std::vector<double> c(4);
  for (std::size_t a = 0; a < 2; ++a) {
    for (std::size_t b = 0; b < 2; ++b) {
      c[(a * 2) + b] = 0.5 * ((u[x][a] * u[y][b]) + (u[y][a] * u[x][b]));
    }
  }
  return c;
}

// The seam's 2 x 2 at the MECI, on its own: each element symmetric, its
// diagonal summing to the pair's ladder, and in a basis of the eigenspace
// turned by kTurn its elements the turn of these (so its eigenvalues the
// same), within kBars of the fixture's seam bar; the seam's term (i) at the
// fixture's W printed against its ladder.
void expect_seam(const std::string &name, const bool pc) {
  const fb::json &doc = fixture(name);
  const Reference &ref = reference(name);
  ASSERT_TRUE(ref.solution.has_value());
  const std::string_view scheme = pc ? "pc" : "sc";
  const fb::json *seam = entry(doc, scheme, "seams", 0, 1);
  const fb::json *pair = entry(doc, scheme, "pairs", 0, 1);
  ASSERT_NE(seam, nullptr) << name << " has no seam";
  ASSERT_NE(pair, nullptr) << name << " has no pair";
  const auto bar = matrix(seam->at("bar"));
  const std::size_t k = bar.size();
  const std::string label = std::format("{} {} seam", name, scheme);

  // M_00, M_01, M_11, then M'_00 and M'_01 in the turned basis.
  const std::array<std::tuple<std::size_t, std::size_t, double>, 5> calls{
      {{0, 0, 0.0}, {0, 1, 0.0}, {1, 1, 0.0}, {0, 0, kTurn}, {0, 1, kTurn}}};
  std::array<std::vector<double>, 5> m;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t e = 0; e < calls.size(); ++e) {
    const auto [x, y, t] = calls[e];
    const auto got = ref.crossing(*ref.solution, pc, 0, 1, element(x, y, t));
    ASSERT_TRUE(got.has_value()) << label << ": " << got.error().message;
    EXPECT_LE(got->asymmetry, solve_bar(*got)) << label << " call " << e;
    m[e] = got->hessian;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const auto pair_hessian = matrix(pair->at("hessian"));
  const auto pair_bar = matrix(pair->at("bar"));
  const double c = std::cos(kTurn);
  const double sn = std::sin(kTurn);
  double trace_miss = 0.0;
  double turned = 0.0;
  double turned_det = 0.0;
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t at = i + (k * j);
      const double b = bar[i][j];
      const double m00 = m[0][at];
      const double m01 = m[1][at];
      const double m11 = m[2][at];
      trace_miss = std::max(trace_miss, std::abs(m00 + m11 - pair_hessian[i][j]) / pair_bar[i][j]);
      // u = (c, s), v = (-s, c): M'_00 = u^T M u, M'_01 = u^T M v.
      const double want00 = (c * c * m00) + (2.0 * c * sn * m01) + (sn * sn * m11);
      const double want01 = (c * sn * (m11 - m00)) + (((c * c) - (sn * sn)) * m01);
      turned = std::max({turned, std::abs(m[3][at] - want00) / b, std::abs(m[4][at] - want01) / b});
      // The determinant from the turned elements, M'_11 = tr M - M'_00.
      const double t11 = m00 + m11 - m[3][at];
      const double det = (m00 * m11) - (m01 * m01);
      const double det_t = (m[3][at] * t11) - (m[4][at] * m[4][at]);
      const double scale = std::max({std::abs(m00), std::abs(m01), std::abs(m11), b});
      turned_det = std::max(turned_det, std::abs(det_t - det) / (scale * b));
    }
  }
  std::cout << std::format("[ qd seam  ] {}: M_00 + M_11 against the pair's ladder {:.3e} of its "
                           "bars; turned by {:g} rad, elements {:.3e} and determinant {:.3e} "
                           "seam bars (allowed {:g}); max |M_01| {:.3e}; {:.1f} s for five\n",
                           label, trace_miss, kTurn, turned, turned_det, kBars, max_abs(m[1]),
                           seconds);
  EXPECT_LE(trace_miss, kBars) << label;
  EXPECT_LE(turned, kBars) << label;
  EXPECT_LE(turned_det, kBars) << label;

  // Term (i) at the fixture's seam W, R its recorded eigenvectors
  // (reference.<scheme>.vectors, the R the seam ladder was taken at) signed
  // into this solve's root phases by H_eff's off-diagonal signs, through the
  // W form: never this solve's own R, whose basis of the eigenspace is
  // another.
  const std::size_t n = ref.solution->roots.energies.size();
  const auto theirs = doc.at("reference").at(std::string{scheme}).at("heff");
  const auto r = doc.at("reference").at(std::string{scheme}).at("vectors");
  const std::vector<double> ours = ref.heff(pc);
  std::vector<double> phase(n, 1.0);
  for (std::size_t s = 1; s < n; ++s) {
    phase[s] = (ours[s] * theirs.at(s).get<double>()) < 0.0 ? -1.0 : 1.0;
  }
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double r0i = r.at(i).get<double>();
      const double r1i = r.at(n + i).get<double>();
      const double r0j = r.at(j).get<double>();
      const double r1j = r.at(n + j).get<double>();
      w[(i * n) + j] = 0.5 * phase[i] * phase[j] * ((r0i * r1j) + (r1i * r0j));
    }
  }
  const auto fixed = ref.w_form(pc, w);
  ASSERT_TRUE(fixed.has_value()) << label << ": " << fixed.error().message;
  const auto want = matrix(seam->at("hessian"));
  const double worst = expect_within(fixed->hessian, *seam, label + " term (i) at W");
  std::cout << std::format("[ qd seam  ] {}: term (i) at the fixture's W (phases {}), worst "
                           "element {:.3f} bars (allowed {:g}); asymmetry {:.1e} (bar {:.1e})\n",
                           label, phase, worst, kBars, fixed->asymmetry, solve_bar(*fixed));
  EXPECT_LE(fixed->asymmetry, solve_bar(*fixed)) << label;
  // The phases agree on the third off-diagonal.
  EXPECT_GT(ours[5] * theirs.at(5).get<double>() * phase[1] * phase[2], 0.0) << label;
}

TEST(HessianDfQdNevpt2Crossing, ScSeamTwoByTwo) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_seam("ethylene_ccpvdz_sa3_sc_meci", false);
}

TEST(HessianDfQdNevpt2Crossing, PcSeamTwoByTwo) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_seam("ethylene_ccpvdz_sa3_pc_meci", true);
}

// The refusals. A third state within kRootGapFloor of the pair: the roots'
// energies moved so that SC's H_eff is lambda + sigma v v^T on the diagonal
// and off it (sigma v_i v_j = H_ij), rank one about lambda, so two states
// are exactly degenerate and the third apart; the crossing then asks for a
// pair of the third and one of the two.
TEST(HessianDfQdNevpt2Crossing, RefusesBeforeAnythingRuns) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const std::string name = "ethylene_ccpvdz_sa3_sc_meci";
  const Reference &ref = reference(name);
  ASSERT_TRUE(ref.solution.has_value());
  const auto expect_refused = [](const auto &r, const fb::ErrorKind kind,
                                 const std::string_view what, const std::string_view names) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, kind) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_qd_sc_nevpt2_crossing"), std::string::npos)
        << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find(names), std::string::npos)
        << what << ": " << r.error().message;
    std::cout << std::format("[ qd cross ] refuses {}: {}\n", what, r.error().message);
  };
  expect_refused(ref.crossing(*ref.solution, false, 1, 1, kIdentity), fb::ErrorKind::InvalidConfig,
                 "a pair of one state", "is not a pair");
  expect_refused(ref.crossing(*ref.solution, false, 0, 3, kIdentity), fb::ErrorKind::InvalidConfig,
                 "a state past the roots", "is not a pair");
  expect_refused(ref.crossing(*ref.solution, false, 0, 1, std::vector<double>{1.0, 0.5, 0.0, 1.0}),
                 fb::ErrorKind::InvalidConfig, "c asymmetric", "c is not symmetric");

  const std::vector<double> h = ref.heff(false);
  ASSERT_EQ(h.size(), 9U);
  const double c01 = h[1];
  const double c02 = h[2];
  const double c12 = h[5];
  const double sigma = (c01 * c02 * c12) > 0.0 ? 1.0 : -1.0;
  const std::array<double, 3> v2{sigma * c01 * c02 / c12, sigma * c01 * c12 / c02,
                                 sigma * c02 * c12 / c01};
  const double lambda = h[0] - (sigma * v2[0]);
  std::vector<double> e = ref.solution->roots.energies;
  for (std::size_t i = 0; i < 3; ++i) {
    e[i] += lambda + (sigma * v2[i]) - h[i * 4];
  }
  // sigma > 0: states 0 and 1 at lambda, the third above; else 1 and 2.
  const std::int32_t lone = sigma > 0.0 ? 2 : 0;
  const std::int32_t partner = 1;
  const auto start = std::chrono::steady_clock::now();
  const auto r = ref.crossing(ref.with_energies(e), false, std::min(lone, partner),
                              std::max(lone, partner), kIdentity);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  expect_refused(r, fb::ErrorKind::Numerical, "a third state on the pair",
                 std::format("QD state {} is", sigma > 0.0 ? 0 : 2));
  if (!r.has_value()) {
    EXPECT_NE(r.error().message.find("the crossing pair's state 1"), std::string::npos)
        << r.error().message;
  }
  std::cout << std::format("[ qd cross ] the third state's refusal took {:.2f} s\n", seconds);
}

} // namespace
