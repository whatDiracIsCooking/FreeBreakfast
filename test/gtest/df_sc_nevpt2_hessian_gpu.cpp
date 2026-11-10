// GPU cases for hessian::df_sc_nevpt2, the SC-NEVPT2 analytic nuclear
// Hessian, on this tree's own DF-CASSCF of data/nevpt2_hessian/'s fixtures
// (#803) through the registry, against their semi-numerical SC Hessians:
//
// - h2co_skew_sto3g (C1, every block defined): every element within 2x the
//   fixture's bar, each part's largest element in bars (the visibility
//   rule), and the diagnostics;
// - nh3_cs_sto3g (Cs, symmetry-adapted coordinates), #816's amended gate:
//   the refused pairs (Numerical, naming them) contain the fixture's failing
//   pairs and lie inside the A'' x A'' off-diagonal block, #894's kink; every
//   element not refused within 2x the bar, from runs over the A' block with
//   one A'' coordinate (or a pair not refused) added;
// - examples/nevpt2.toml's water (C2v, cc-pVDZ) (#952): refused, Numerical,
//   naming the coordinate whose canonical inactive pair mixes within 1e-3
//   bohr (O y in Cartesians, the B2 vibration in symmetry-adapted ones) and
//   pointing to PC; without the B2 vibration, its symmetry-adapted
//   coordinates are not refused (the rigid translations, round-off
//   directions, populate no empty perturber);
// - a C3v ammonia whose virtual e pairs are degenerate: df_nevpt2_tangents'
//   refusal, passed on;
// - the input refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"
#include "symmetric_water.h"

namespace {

namespace fb = free_breakfast;

// The reference CASSCF's ||g|| (df_nevpt2_tangents_gpu.cpp's).
constexpr double kConvTolGrad = 1e-10;
// The issue's gate: every element within this many of the fixture's bars.
constexpr double kBars = 2.0;
// The visibility rule: each part has an element above this many bars.
constexpr double kVisibleBars = 10.0;
// Diagnostics: the CP residuals are df_casscf's (a converged solve sits near
// 1e-9), the asymmetry and the sum rule a fraction of the smallest bar.
constexpr double kCpResidual = 1e-8;
constexpr double kAsymmetry = 1e-9;
constexpr double kSumRule = 1e-8;

const fb::json &fixture(std::string_view name) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto it = docs.find(name); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{name}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *d) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(d, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

// This tree's DF-CASSCF of a fixture's start (atoms, charge, basis, fitting
// set, active space) through the real registry in @p node.
std::optional<fb::Calculation> run_casscf(const fb::json &start, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("df_sc_nevpt2_hessian");
  std::string xyz =
      std::format("{}\n{}\n", start.at("atoms").size(), start.at("name").get<std::string>());
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string active;
  for (const std::int32_t a : start.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
      start.at("charge").get<std::int32_t>(), start.at("basis").at("name").get<std::string>(),
      kConvTolGrad, start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
      active, start.at("aux_basis").at("name").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return std::nullopt;
  }
  const fb::Status ran = calc->run(node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return std::nullopt;
  }
  return std::move(*calc);
}

// One case: the registry's products and the one-root Solution of its
// canonical CASSCF.
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
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  }

  [[nodiscard]] fb::Result<fb::hessian::ScNevpt2Terms>
  hessian(std::span<const double> coordinates = {}) const {
    return hessian_of(*solution, coordinates);
  }
  [[nodiscard]] fb::Result<fb::hessian::ScNevpt2Terms>
  hessian_of(const fb::mcscf::Solution &s, std::span<const double> coordinates = {}) const {
    return fb::hessian::df_sc_nevpt2(res(), molecule(), basis(),
                                     calc->get<fb::PrimitivePairClassesOutput>()->classes,
                                     fit().aux, calc->get<fb::CoreHamiltonianOutput>()->h, fit().b,
                                     fit().l.data(), s, ncore, ncas, 0, coordinates);
  }
};

Reference reference(const fb::json &start) {
  Reference ref{.node = std::make_unique<fb::NodeResources>(
                    fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces}),
                .calc = std::nullopt,
                .solution = std::nullopt};
  std::optional<fb::Calculation> calc = run_casscf(start, *ref.node);
  if (!calc.has_value()) {
    return ref;
  }
  ref.calc.emplace(std::move(*calc));
  fb::DeviceResources &res = ref.res();
  const auto &cas = ref.calc->get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (!cas.has_value()) {
    return ref;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  ref.ncore = cas->ncore;
  ref.ncas = cas->ncas;
  std::cout << std::format("[ casscf   ] {}: E {:.10f}, ||g|| {:.1e}\n",
                           start.at("name").get<std::string>(), cas->energy, cas->gradient_norm);
  ref.solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = cas->rdm1,
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  upload(res, cas->vectors, ref.solution->roots.vectors.data());
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref.solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return ref;
}

// A fixture's n x n matrix, [i][j], as its rows.
std::vector<std::vector<double>> matrix(const fb::json &m) {
  return m.get<std::vector<std::vector<double>>>();
}

// The rows @p keep of the fixture's coordinates, flattened row-major.
std::vector<double> coordinate_rows(const fb::json &doc, std::span<const std::size_t> keep) {
  const auto vectors = matrix(doc.at("coordinates").at("vectors"));
  std::vector<double> q;
  for (const std::size_t i : keep) {
    q.insert(q.end(), vectors[i].begin(), vectors[i].end());
  }
  return q;
}

// The largest |part| in bars of the fixture's elements, over @p keep's
// coordinates (part k x k column-major over them).
double in_bars(std::span<const double> part, const std::vector<std::vector<double>> &bar,
               std::span<const std::size_t> keep) {
  double worst = 0.0;
  const std::size_t k = keep.size();
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      worst = std::max(worst, std::abs(part[i + (k * j)]) / bar[keep[i]][keep[j]]);
    }
  }
  return worst;
}

// Every element of @p h over @p keep's coordinates against the fixture's,
// in bars; the worst miss.
double expect_within_bars(const fb::hessian::ScNevpt2Terms &h, const fb::json &scheme,
                          std::span<const std::size_t> keep, std::string_view what) {
  const auto want = matrix(scheme.at("hessian"));
  const auto bar = matrix(scheme.at("bar"));
  const std::size_t k = keep.size();
  EXPECT_EQ(h.hessian.size(), k * k) << what;
  double worst = 0.0;
  for (std::size_t j = 0; j < k && h.hessian.size() == k * k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t a = keep[i];
      const std::size_t b = keep[j];
      const double miss = std::abs(h.hessian[i + (k * j)] - want[a][b]) / bar[a][b];
      worst = std::max(worst, miss);
      EXPECT_LE(miss, kBars) << what << ": element (" << a << ", " << b << ") "
                             << h.hessian[i + (k * j)] << " against " << want[a][b] << ", bar "
                             << bar[a][b];
    }
  }
  return worst;
}

std::vector<std::size_t> iota(std::size_t n) {
  std::vector<std::size_t> v(n);
  std::iota(v.begin(), v.end(), std::size_t{0});
  return v;
}

TEST(HessianDfScNevpt2, MatchesTheSkewFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &doc = fixture("h2co_skew_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const auto keep = iota(3 * ref.basis().natoms());
  const auto start = std::chrono::steady_clock::now();
  auto h = ref.hessian(coordinate_rows(doc, keep));
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const double worst = expect_within_bars(*h, doc.at("sc"), keep, "skew");
  const auto bar = matrix(doc.at("sc").at("bar"));
  std::cout << std::format("[ skew     ] worst element {:.3f} bars (allowed {:g}); {:.1f} s\n",
                           worst, kBars, seconds);

  // Visibility: each part's largest element in bars.
  for (const auto &[name, part] :
       {std::pair{"E_cas", &h->e_cas}, std::pair{"Z d2g/dRdR", &h->z_lxy},
        std::pair{"Z L_Ax x_B", &h->z_lx}, std::pair{"Z third", &h->z_third},
        std::pair{"Z kappa_oo rows", &h->z_koo}, std::pair{"skeleton", &h->skeleton},
        std::pair{"<t, HVP t>", &h->hvp}, std::pair{"tau-kappa", &h->rotation_tau},
        std::pair{"kappa-kappa", &h->rotation_kappa}}) {
    const double bars = in_bars(*part, bar, keep);
    std::cout << std::format("[ visible  ] {:16} {:10.3e} bars (rule {:g})\n", name, bars,
                             kVisibleBars);
  }
  double cp = 0.0;
  for (const double r : h->casscf.residuals) {
    cp = std::max(cp, r);
  }
  std::cout << std::format("[ skew     ] E_corr {:.10f} (fixture {:.10f}), CP residual {:.1e}, "
                           "Z-vector {:.1e}, asymmetry {:.1e}, sum rule {:.1e}, max |z^c| "
                           "{:.3e}, min gap {:.3e}, defect {:.1e}, t_mix {:.3e} bohr, rounding "
                           "{:.2e}\n",
                           h->e_corr, doc.at("e_corr").at("sc").get<double>(), cp,
                           h->zvector_residual, h->asymmetry, h->translation, h->max_zc, h->min_gap,
                           h->defect, h->mix_step, h->rounding);
  EXPECT_LE(cp, kCpResidual);
  EXPECT_LE(h->asymmetry, kAsymmetry);
  EXPECT_LE(h->translation, kSumRule);
  EXPECT_EQ(h->defect, 0.0); // no SC perturber is empty here
}

TEST(HessianDfScNevpt2, RefusesTheSymmetricFixturesKinkBlock) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &doc = fixture("nh3_cs_sto3g");
  const Reference ref = reference(doc);
  ASSERT_TRUE(ref.solution.has_value());
  const fb::json &sc = doc.at("sc");
  const std::size_t n = 3 * ref.basis().natoms();
  const auto irreps = doc.at("coordinates").at("irreps").get<std::vector<std::string>>();
  const std::string symmetric = doc.at("coordinates").at("irreps_order")[0].get<std::string>();
  std::vector<std::size_t> a1;
  std::vector<std::size_t> a2;
  for (std::size_t i = 0; i < n; ++i) {
    (irreps[i] == symmetric ? a1 : a2).push_back(i);
  }

  // Every coordinate: refused, Numerical, naming the pairs.
  auto refused = ref.hessian(coordinate_rows(doc, iota(n)));
  ASSERT_FALSE(refused.has_value()) << "no pair refused";
  ASSERT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
  const std::string &message = refused.error().message;
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
  std::cout << std::format("[ nh3      ] refused {} pairs: {}\n", named.size(), list);
  // (a) the fixture's failing pairs among them; (b) all inside the A'' x A''
  // off-diagonal block.
  for (const auto &[i, j] : sc.at("failing_pairs").get<std::vector<std::array<std::size_t, 2>>>()) {
    EXPECT_TRUE(named.contains({std::min(i, j), std::max(i, j)}))
        << "the fixture's failing pair (" << i << ", " << j << ") not refused";
  }
  for (const auto &[i, j] : named) {
    EXPECT_TRUE(i != j && irreps[i] != symmetric && irreps[j] != symmetric)
        << "(" << i << ", " << j << ") is outside the A'' x A'' off-diagonal block";
  }

  // (c) every element not refused: the A' block with each A'' coordinate,
  // and with each A'' pair not refused.
  std::vector<std::vector<std::size_t>> runs;
  for (const std::size_t i : a2) {
    runs.push_back(a1);
    runs.back().push_back(i);
  }
  for (std::size_t x = 0; x < a2.size(); ++x) {
    for (std::size_t y = x + 1; y < a2.size(); ++y) {
      if (!named.contains({a2[x], a2[y]})) {
        runs.push_back(a1);
        runs.back().push_back(a2[x]);
        runs.back().push_back(a2[y]);
      }
    }
  }
  double worst = 0.0;
  double mix_step = std::numeric_limits<double>::infinity();
  for (const auto &keep : runs) {
    auto h = ref.hessian(coordinate_rows(doc, keep));
    ASSERT_TRUE(h.has_value()) << h.error().message;
    worst = std::max(worst, expect_within_bars(*h, sc, keep, "nh3"));
    EXPECT_LE(h->asymmetry, kAsymmetry);
    mix_step = std::min(mix_step, h->mix_step);
  }
  std::cout << std::format("[ nh3      ] {} runs: worst element not refused {:.3f} bars (allowed "
                           "{:g}); t_mix {:.3e} bohr\n",
                           runs.size(), worst, kBars, mix_step);
}

// The coordinates a df_sc_nevpt2 mixing refusal names: "along coordinates
// i, j, ...:".
std::set<std::size_t> mixing_coordinates(const std::string &message) {
  const std::string lead = "along coordinates ";
  const std::size_t from = message.find(lead);
  EXPECT_NE(from, std::string::npos) << message;
  std::set<std::size_t> named;
  if (from == std::string::npos) {
    return named;
  }
  const std::string list =
      message.substr(from + lead.size(), message.find(':', from) - from - lead.size());
  const std::regex index_re(R"(\d+)");
  for (auto it = std::sregex_iterator(list.begin(), list.end(), index_re);
       it != std::sregex_iterator(); ++it) {
    named.insert(std::stoul(it->str()));
  }
  return named;
}

// examples/nevpt2.toml's water, Cartesian: its a1 and b2 virtuals 17 and 18
// are 1.4e-3 Eh apart, and O y mixes them within 7.5e-4 bohr. The B2 element
// was SC's t -> 0 limit, 15.7 Eh/bohr^2 against PC's 1.29 (#952).
TEST(HessianDfScNevpt2, RefusesTheSymmetricWater) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  const Reference ref = reference(fb::test::symmetric_water());
  ASSERT_TRUE(ref.solution.has_value());
  auto refused = ref.hessian();
  ASSERT_FALSE(refused.has_value()) << "SC's B2 element published";
  ASSERT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
  const std::string &message = refused.error().message;
  std::cout << std::format("[ water    ] {}\n", message);
  EXPECT_NE(message.find("use PC-NEVPT2"), std::string::npos) << message;
  EXPECT_NE(message.find("orbitals (17, 18)"), std::string::npos) << message;
  // Atom-major (x, y, z), the molecule in the yz plane: O y (1) is B2, O z
  // (2) totally symmetric.
  const std::set<std::size_t> named = mixing_coordinates(message);
  EXPECT_TRUE(named.contains(1)) << "O y not named";
  EXPECT_FALSE(named.contains(2)) << "O z, totally symmetric, named";
}

// The same water in symmetry-adapted coordinates: refused on the B2
// vibration alone. Without it, the rest (A1, and the rigid motions) is a
// Hessian: the translations' tangents are round-off, whose populations of
// the empty perturbers are read against the largest scale over the
// coordinates, not their own (#952: before, the pair (Tz, Tx) was refused).
TEST(HessianDfScNevpt2, SymmetricWaterRefusesOnlyItsB2Vibration) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference ref = reference(fb::test::symmetric_water());
  ASSERT_TRUE(ref.solution.has_value());
  const fb::test::WaterCoordinates c = fb::test::symmetric_water_coordinates();
  ASSERT_EQ(c.rows.size(), 9U);
  ASSERT_EQ(c.labels[5], "B2 vibration");
  auto refused = ref.hessian(fb::test::symmetric_water_rows(iota(9)));
  ASSERT_FALSE(refused.has_value()) << "SC's B2 element published";
  ASSERT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
  std::cout << std::format("[ water    ] {}\n", refused.error().message);
  EXPECT_EQ(mixing_coordinates(refused.error().message), std::set<std::size_t>{5});

  const std::vector<std::size_t> keep{0, 1, 2, 3, 4, 6, 7, 8};
  auto h = ref.hessian(fb::test::symmetric_water_rows(keep));
  ASSERT_TRUE(h.has_value()) << h.error().message;
  // The translations' rows (0 Tz, 3 Ty, 6 Tx at 0, 3, 5 here) vanish.
  double rigid = 0.0;
  for (const std::size_t i : {std::size_t{0}, std::size_t{3}, std::size_t{5}}) {
    for (std::size_t j = 0; j < keep.size(); ++j) {
      rigid = std::max(rigid, std::abs(h->hessian[i + (keep.size() * j)]));
    }
  }
  std::cout << std::format("[ water    ] without the B2 vibration: translations' rows {:.1e}, "
                           "asymmetry {:.1e}, defect {:.1e}, t_mix {:.3e} bohr, rounding {:.2e}\n",
                           rigid, h->asymmetry, h->defect, h->mix_step, h->rounding);
  EXPECT_LE(rigid, kSumRule);
  EXPECT_LE(h->asymmetry, kAsymmetry);
}

// A C3v ammonia in 6-31G, its full valence CAS(8,7) (df_nevpt2_tangents_gpu.cpp's):
// its virtual e pairs are degenerate, and E displacements move f within them.
fb::json c3v_ammonia() {
  const double rho = std::sqrt((1.9 * 1.9) - (0.7 * 0.7));
  fb::json atoms = fb::json::array();
  atoms.push_back({{"symbol", "N"}, {"coord", {0.0, 0.0, 0.0}}});
  for (std::int32_t k = 0; k < 3; ++k) {
    const double phi = 2.0 * std::numbers::pi * k / 3.0;
    atoms.push_back({{"symbol", "H"}, {"coord", {rho * std::cos(phi), rho * std::sin(phi), -0.7}}});
  }
  return {{"name", "nh3_c3v_631g"},
          {"atoms", atoms},
          {"charge", 0},
          {"basis", {{"name", "6-31G"}}},
          {"aux_basis", {{"name", "cc-pVDZ-JKFIT"}}},
          {"ncas", 7},
          {"nelecas", 8},
          {"active", {1, 2, 3, 4, 5, 6, 7}}};
}

TEST(HessianDfScNevpt2, DegenerateInactivePairIsRefused) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.solution.has_value());
  const auto h = ref.hessian();
  ASSERT_FALSE(h.has_value()) << "a degenerate pair's kappa_oo was taken";
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << h.error().message;
  EXPECT_NE(h.error().message.find("hessian::df_nevpt2_tangents"), std::string::npos)
      << h.error().message;
  EXPECT_NE(h.error().message.find("along coordinate "), std::string::npos) << h.error().message;
  std::cout << std::format("[ c3v      ] {}\n", h.error().message);
}

TEST(HessianDfScNevpt2, Refusals) {
  FBF_REQUIRE_GPU();
  const Reference ref = reference(fixture("h2co_skew_sto3g"));
  ASSERT_TRUE(ref.solution.has_value());
  fb::DeviceResources &res = ref.res();
  const auto expect_refused = [](const auto &r, const char *what) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_sc_nevpt2"), std::string::npos)
        << what << ": " << r.error().message;
  };
  const std::size_t n = 3 * ref.basis().natoms();
  // Not whole rows, and two rows that are not orthonormal.
  expect_refused(ref.hessian(std::vector<double>(n + 1, 0.0)), "a row and one more");
  std::vector<double> two(2 * n, 0.0);
  two[0] = 1.0;
  two[n] = 1.0;
  expect_refused(ref.hessian(two), "two rows along one coordinate");
  // A state average (QD's reference): two roots.
  fb::mcscf::Solution averaged{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = {ref.solution->energy, ref.solution->energy + 0.1},
                .spin_squares = {0.0, 0.0},
                .vectors = fb::Buffers::device<double>(
                    2 * ref.solution->roots.vectors.num_elements(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = ref.solution->energy,
      .gradient_norm = ref.solution->gradient_norm,
      .gamma = ref.solution->gamma,
      .gamma2 = ref.solution->gamma2,
      .root_gammas = ref.solution->root_gammas,
      .iterations = {},
      .weights = {0.5, 0.5}};
  expect_refused(ref.hessian_of(averaged), "a state average");
}

} // namespace
