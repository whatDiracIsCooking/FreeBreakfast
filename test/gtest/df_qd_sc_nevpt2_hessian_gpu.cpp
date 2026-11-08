// GPU cases for hessian::df_qd_sc_nevpt2, the SC-QD-NEVPT2 analytic nuclear
// Hessian (docs/nevpt2/nevpt2.tex Eqs. qd-hess and qd-hessW), on this tree's
// own DF-SA-CASSCF of data/qd_nevpt2_hessian/'s cases through the registry,
// against their semi-numerical SC-QD Hessians (#953):
//
// - h2co_skew_ccpvdz_sa3 (C1, equal weights), states K = 1 and K = 2 each on
//   its own (root 2's SC bars sit near the einsums' ~1e-8 relative floor,
//   #1055), and h2co_skew_ccpvdz_sa3_w424 (0.4/0.2/0.4), state K = 1: every
//   element within 2x the fixture's bar; each part's largest element in bars
//   (the visibility rule), theta's share at equal weights, and the reports;
// - h2nb_sto3g_sa2 (C2v A1/B2, its symmetry-adapted coordinates): #953's SC
//   expected-refusal set is empty, so no block is refused on state 0 (the
//   rigid translations, round-off directions, break no symmetry); no element
//   couples two irreps; the rotation block is the gradient's, (R_z, R_z) =
//   v^T L G from the state's own G (no reference data); and term (ii)'s
//   h_01 along the b2 vibrations is held to the fixture's C_01 ladders.
//   State 1 is refused by SC's conditioning rule (#1102): root 1's Si
//   tuples keep N of 1e-10 and 3e-11, whose Hessian spreads run to run past
//   every bar;
// - the refusals before anything runs, each on a constructed input: a QD
//   gap, a CASSCF root gap and the ensemble's edge under kRootGapFloor, a
//   frozen core; and an inactive pair under 1e-6 Eh that a coordinate
//   splits (df_qd_nevpt2_tangents' C3v ammonia).

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

// The reference SA-CASSCF's ||g|| and |Delta E| (df_qd_nevpt2_tangents_gpu.cpp's:
// Eq. qd-theta reads H c_r = E_r c_r).
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;
// The issue's gate: every element within this many of the fixture's bars.
constexpr double kBars = 2.0;
// The visibility rule: each part has an element above this many bars.
constexpr double kVisibleBars = 10.0;
// The CP solve's unreduced residual (df_sacasscf_cp's columns sit near 1e-10).
constexpr double kCpResidual = 1e-9;
// What no part determines, the asymmetry and an element C2v forbids, is the
// solves' error: first order in their residuals (the CP's and the
// Z-vector's, 1e-10), each times the size of what it multiplies, so at most
// this times the largest element of any part (the skeleton and the HVP,
// which cancel to the Hessian's size).
constexpr double kSolveTolerance = 1e-10;

double max_abs(std::span<const double> v) {
  return std::ranges::fold_left(
      v, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// kSolveTolerance times the largest element of any part.
double solve_bar(const fb::hessian::QdScNevpt2Terms &h) {
  double largest = 0.0;
  for (const std::vector<double> *part :
       {&h.e_sa, &h.z_lxy, &h.z_lx, &h.z_third, &h.z_koo, &h.zeta, &h.skeleton, &h.hvp,
        &h.rotation_tau, &h.rotation_kappa, &h.term_ii}) {
    largest = std::max(largest, max_abs(*part));
  }
  return kSolveTolerance * largest;
}
// The A1/B2 case's coupling against the fixture's C_01 ladders: within 2x
// each direction's bar (3.4e-9, its noise floor), as the Hessian's elements.
constexpr double kCouplingBars = 2.0;

// A state average's start: atoms (bohr), basis, fitting set, active space,
// weights.
struct Start {
  std::string name;
  std::vector<std::pair<std::string, std::array<double, 3>>> atoms;
  std::string basis;
  std::string aux_basis;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<double> weights;
  std::vector<std::int32_t> active;
};

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

Start start_of(const fb::json &entry, const std::string &name) {
  Start s{.name = name,
          .atoms = {},
          .basis = entry.at("basis").is_string() ? entry.at("basis").get<std::string>()
                                                 : entry.at("basis").at("name").get<std::string>(),
          .aux_basis = entry.at("aux_basis").at("name").get<std::string>(),
          .ncas = entry.at("ncas").get<std::int32_t>(),
          .nelecas = entry.at("nelecas").get<std::int32_t>(),
          .weights = entry.at("weights").get<std::vector<double>>(),
          .active = {}};
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    s.atoms.push_back({atom.at(0).get<std::string>(), {coord[0], coord[1], coord[2]}});
  }
  return s;
}

// data/qd_nevpt2/qd_nevpt2.json's case @p name.
Start qd_case(const std::string &name) {
  for (const fb::json &entry : document("qd_nevpt2/qd_nevpt2.json").at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return start_of(entry, name);
    }
  }
  ADD_FAILURE() << "no case " << name;
  return {};
}

// df_qd_nevpt2_tangents_gpu.cpp's C3v ammonia in 6-31G, SA-2 CAS(8,7) over
// its full valence, both roots A1: its virtual e pairs stay degenerate.
Start c3v_ammonia() {
  const double rho = std::sqrt((1.9 * 1.9) - (0.7 * 0.7));
  Start s{.name = "nh3_c3v_631g_sa2",
          .atoms = {{"N", {0.0, 0.0, 0.0}}},
          .basis = "6-31G",
          .aux_basis = "cc-pVDZ-JKFIT",
          .ncas = 7,
          .nelecas = 8,
          .weights = {0.5, 0.5},
          .active = {1, 2, 3, 4, 5, 6, 7}};
  for (std::int32_t k = 0; k < 3; ++k) {
    const double phi = 2.0 * std::numbers::pi * k / 3.0;
    s.atoms.push_back({"H", {rho * std::cos(phi), rho * std::sin(phi), -0.7}});
  }
  return s;
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

  [[nodiscard]] fb::Result<fb::hessian::QdScNevpt2Terms>
  hessian(const fb::mcscf::Solution &s, const std::int32_t state,
          std::span<const double> coordinates = {}, const std::int32_t nfrozen = 0) const {
    return fb::hessian::df_qd_sc_nevpt2(res(), molecule(), basis(), classes(), fit().aux, h(),
                                        fit().b, fit().l.data(), s, ncore, ncas, state, 0,
                                        coordinates, nfrozen);
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

  // H_eff's correlation elements C, n x n [M n + N].
  [[nodiscard]] std::vector<double> totals() const {
    std::vector<double> out;
    const fb::Status got = fb::nevpt2::qd_inputs(
        res(), molecule(), basis(), fit().aux, h(), fit().b, *solution, ncore, ncas,
        [&](const fb::nevpt2::MoBlocks &blocks,
            const std::span<const fb::nevpt2::EnergyInputs> pairs) -> fb::Status {
          const auto e = fb::nevpt2::sc_effective_hamiltonian(res(), blocks, pairs, 0);
          if (!e) {
            return std::unexpected(e.error());
          }
          std::ranges::transform(*e, std::back_inserter(out), &fb::nevpt2::ClassEnergies::total);
          return {};
        });
    EXPECT_TRUE(got.has_value()) << got.error().message;
    return out;
  }

  // The active space's n + 1-th root at the solution's orbitals.
  [[nodiscard]] double next_root() const {
    fb::DeviceResources &r = res();
    const auto ints = fb::casscf_integrals(r, basis(), molecule(), h(), fit().b,
                                           static_cast<std::int32_t>(fit().aux.nao_sphe()),
                                           solution->c, ncore, ncas);
    EXPECT_TRUE(ints.has_value()) << ints.error().message;
    const auto n = static_cast<std::size_t>(ncas);
    std::vector<double> h_eff(n * n);
    std::vector<double> eri(n * n * n * n);
    EXPECT_EQ(wwr::wwrMemcpyAsync(h_eff.data(), ints->hamiltonian.h_eff.data(),
                                  h_eff.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                  r.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrMemcpyAsync(eri.data(), ints->hamiltonian.eri.data(),
                                  eri.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                  r.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(r.sync().has_value());
    const std::int32_t nelecas = static_cast<std::int32_t>(std::lround(std::ranges::fold_left(
        std::views::iota(std::size_t{0}, n), 0.0,
        [&](const double t, const std::size_t i) { return t + solution->gamma[i * (n + 1)]; })));
    const auto alpha = fb::ci::DeviceStringSet::create(r, ncas, nelecas / 2);
    EXPECT_TRUE(alpha.has_value());
    auto sigma = fb::ci::Sigma::create(r, *alpha, *alpha);
    EXPECT_TRUE(sigma.has_value());
    EXPECT_TRUE(sigma->set_integrals(r, ints->hamiltonian.e_core, h_eff, eri).has_value());
    const auto dim = static_cast<std::size_t>(sigma->size());
    fb::Buffers::device<double> diagonal(dim, r.device());
    EXPECT_TRUE(sigma->diagonal(r, diagonal.data()).has_value());
    const auto nroots = static_cast<std::int32_t>(solution->roots.energies.size());
    const auto more = fb::ci::solve(r, *sigma, *alpha, *alpha, diagonal.data(), nroots + 1);
    EXPECT_TRUE(more.has_value()) << more.error().message;
    return more->energies.back();
  }
};

std::unique_ptr<Reference> make_reference(const Start &start) {
  auto ref = std::make_unique<Reference>();
  ref->node = std::make_unique<fb::NodeResources>();
  const fb::test::ScratchDir dir("df_qd_sc_nevpt2_hessian");
  std::string xyz = std::format("{}\n{}\n", start.atoms.size(), start.name);
  for (const auto &[symbol, coord] : start.atoms) {
    xyz += std::format(
        "{} {:.17g} {:.17g} {:.17g}\n", symbol, coord[0] * fb::kAngstromPerBohr<double>,
        coord[1] * fb::kAngstromPerBohr<double>, coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : start.weights) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  std::string active;
  for (const std::int32_t a : start.active) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol = {:g}\n"
      "conv_tol_grad = {:g}\n{}",
      start.basis, start.aux_basis, start.ncas, start.nelecas, start.weights.size(), weights,
      kConvTol, kConvTolGrad, active.empty() ? "" : std::format("active = [{}]\n", active));
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
  std::cout << std::format("[ casscf   ] {}: E {}, ||g|| {:.1e}\n", start.name, cas->energies,
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
const Reference &reference(const Start &start) {
  static std::map<std::string, std::unique_ptr<Reference>, std::less<>> cache;
  auto &slot = cache[start.name];
  if (!slot) {
    slot = make_reference(start);
  }
  return *slot;
}

std::vector<std::vector<double>> matrix(const fb::json &m) {
  return m.get<std::vector<std::vector<double>>>();
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

// theta^A's contribution at equal weights, where it is Eq. qd-theta's closed
// form and in no solve: E_rho's curvature along the rotations alone,
// sum_{r<s} 2 (rho_r - rho_s) (E_s - E_r) theta^A_sr theta^B_sr, one block
// of e_sa that exists only through theta, in the fixture's bars.
void expect_theta_visible(const Reference &ref, const fb::hessian::QdScNevpt2Terms &h,
                          const std::vector<std::vector<double>> &bar, const std::string &label) {
  const std::vector<double> &weights = ref.solution->weights;
  const auto cp = fb::hessian::df_sacasscf_cp(
      ref.res(), ref.molecule(), ref.basis(), ref.classes(), ref.fit().aux, ref.h(), ref.fit().b,
      ref.fit().l.data(), *ref.solution, weights, ref.ncore, ref.ncas);
  ASSERT_TRUE(cp.has_value()) << cp.error().message;
  const auto tangents = fb::hessian::df_qd_nevpt2_tangents(
      ref.res(), ref.molecule(), ref.basis(), ref.classes(), ref.fit().aux, ref.h(), ref.fit().b,
      ref.fit().l.data(), *ref.solution, weights, ref.ncore, ref.ncas, cp->cp, false,
      [](std::int32_t, std::int32_t, const double *, const double *, const double *,
         const double *) -> fb::Status { return {}; });
  ASSERT_TRUE(tangents.has_value()) << tangents.error().message;
  const std::size_t n = weights.size();
  const std::size_t pairs = n * (n - 1) / 2;
  const std::size_t k = 3 * ref.basis().natoms();
  const std::vector<double> &e = ref.solution->roots.energies;
  std::vector<double> block(k * k, 0.0);
  for (std::size_t b = 0; b < k; ++b) {
    for (std::size_t a = 0; a < k; ++a) {
      std::size_t at = 0;
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = r + 1; s < n; ++s, ++at) {
          block[a + (k * b)] += 2.0 * (h.rho[r] - h.rho[s]) * (e[s] - e[r]) *
                                tangents->theta[at + (pairs * a)] *
                                tangents->theta[at + (pairs * b)];
        }
      }
    }
  }
  const double bars = in_bars(block, bar);
  std::cout << std::format("[ visible  ] {} theta (E_rho's theta-theta block) {:.3e} bars (rule "
                           "{:g}), max |theta| {:.3f}\n",
                           label, bars, kVisibleBars, tangents->max_theta);
  EXPECT_GT(bars, kVisibleBars) << label;
}

// State @p state's analytic Hessian of fixture @p name against its SC
// reference: every element within kBars, each part visible, the reports.
void expect_matches(const std::string &name, const std::int32_t state) {
  const fb::json &doc = fixture(name);
  const Reference &ref = reference(start_of(doc, name));
  ASSERT_TRUE(ref.solution.has_value());
  const fb::json *want = nullptr;
  for (const fb::json &s : doc.at("sc").at("states")) {
    if (s.at("state").get<std::int32_t>() == state) {
      want = &s;
    }
  }
  ASSERT_NE(want, nullptr) << name << " has no state " << state;
  const auto start = std::chrono::steady_clock::now();
  const auto h = ref.hessian(*ref.solution, state);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(h.has_value()) << h.error().message;
  const auto hessian = matrix(want->at("hessian"));
  const auto bar = matrix(want->at("bar"));
  const std::size_t k = hessian.size();
  ASSERT_EQ(h->hessian.size(), k * k);
  const std::string label = std::format("{} K={}", name, state);
  double worst = 0.0;
  double largest = 0.0;
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const double got = h->hessian[i + (k * j)];
      const double miss = std::abs(got - hessian[i][j]) / bar[i][j];
      worst = std::max(worst, miss);
      largest = std::max(largest, std::abs(hessian[i][j]));
      EXPECT_LE(miss, kBars) << label << ": element (" << i << ", " << j << ") " << got
                             << " against " << hessian[i][j] << ", bar " << bar[i][j];
    }
  }
  std::cout << std::format("[ qd sc    ] {}: worst element {:.3f} bars (allowed {:g}), largest "
                           "|H| {:.3e}; {:.1f} s\n",
                           label, worst, kBars, largest, seconds);

  const bool equal = std::ranges::all_of(
      ref.solution->weights, [&](const double w) { return w == ref.solution->weights.front(); });
  // Visibility: each part's largest element in bars.
  for (const auto &[part_name, part] :
       {std::pair{"E_SA (E_rho)", &h->e_sa}, std::pair{"z d2g/dRdR", &h->z_lxy},
        std::pair{"z L_Ax x_B", &h->z_lx}, std::pair{"z third", &h->z_third},
        std::pair{"kappa_oo rows", &h->z_koo}, std::pair{"zeta", &h->zeta},
        std::pair{"skeleton", &h->skeleton}, std::pair{"<t, HVP t>", &h->hvp},
        std::pair{"tau-kappa", &h->rotation_tau}, std::pair{"kappa-kappa", &h->rotation_kappa},
        std::pair{"term (ii)", &h->term_ii}}) {
    const double bars = in_bars(*part, bar);
    std::cout << std::format("[ visible  ] {} {:14} {:10.3e} bars (rule {:g})\n", label, part_name,
                             bars, kVisibleBars);
    // zeta lives on the equal-weight pairs alone: at 0.4/0.2/0.4 its one
    // pair's multiplier is round-off small, and it is gated at equal weights.
    if (part != &h->zeta || equal) {
      EXPECT_GT(bars, kVisibleBars) << label << ": " << part_name;
    }
  }
  std::cout << std::format(
      "[ qd sc    ] {}: E {}, QD gap {:.4f}, root gap {:.4f}, edge {:.4f}, inactive gap {:.3e}, "
      "smallest |Delta + H/N| {:.3e}, max |theta| {:.3f}, max |kappa_oo| {:.2f}, max |zeta| "
      "{:.3e}, max |z^c| {:.3e}, term (ii)'s share {:.3e}; CP residual {:.1e}, Z-vector {:.1e}, "
      "asymmetry {:.1e} (bar {:.1e}), sum rule {:.1e}, defect {:.1e}, t_mix {:.3e} bohr, "
      "rounding {:.2e} at {}\n",
      label, h->eigenvalues, h->qd_gap, h->root_gap, h->edge_gap, h->min_gap, h->min_denominator,
      h->max_theta, h->max_kappa_oo, h->max_zeta, h->max_zc, h->term_ii_share, h->cp_residual,
      h->zvector_residual, h->asymmetry, solve_bar(*h), h->translation, h->defect, h->mix_step,
      h->rounding, h->rounding_at);
  EXPECT_LE(h->cp_residual, kCpResidual) << label;
  EXPECT_LE(h->asymmetry, solve_bar(*h)) << label;
  EXPECT_GT(h->term_ii_share, 0.0) << label;
  if (equal) {
    expect_theta_visible(ref, *h, bar, label);
  }
}

TEST(HessianDfQdScNevpt2, MatchesTheSkewFixtureStateOne) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_matches("h2co_skew_ccpvdz_sa3", 1);
}

TEST(HessianDfQdScNevpt2, MatchesTheSkewFixtureStateTwo) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_matches("h2co_skew_ccpvdz_sa3", 2);
}

TEST(HessianDfQdScNevpt2, MatchesTheUnequalWeightFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_matches("h2co_skew_ccpvdz_sa3_w424", 1);
}

// The A1/B2 coupling case in its symmetry-adapted coordinates. State 1 is
// refused by SC's conditioning rule, naming root 1's Si (#1102: its Hessian
// spread past the bars run to run, 1.4e-5 at (A1 1, A1 1) against a bar of
// 1.6e-8). State 0, which #953's SC expected-refusal set leaves unrefused,
// couples no two irreps, and its rotation block is the gradient's: a
// rotation-invariant energy has (R_z, R_z) = -(1/n) Sum_A G_A . (z x v_A),
// v_A = z x (x_A - c) / n the coordinate's row, from the state's own G and
// held to the solves' bar. Then the coupling h_01 along the two b2
// vibrations (term (ii)'s input) against the C_01 ladders.
TEST(HessianDfQdScNevpt2, CoupledA1B2RefusesStateOne) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const std::string name = "h2nb_sto3g_sa2";
  const fb::json &doc = fixture(name);
  ASSERT_TRUE(doc.at("expected_refusals").at("sc").empty());
  const Reference &ref = reference(start_of(doc, name));
  ASSERT_TRUE(ref.solution.has_value());
  const auto vectors = matrix(doc.at("coordinates").at("vectors"));
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  const auto irreps = doc.at("coordinates").at("irreps").get<std::vector<std::string>>();
  std::vector<double> q;
  for (const auto &row : vectors) {
    q.insert(q.end(), row.begin(), row.end());
  }
  const std::size_t k = vectors.size();
  const std::size_t n_x = 3 * ref.basis().natoms();

  // State 1: refused, naming the class.
  {
    const auto h = ref.hessian(*ref.solution, 1, q);
    ASSERT_FALSE(h.has_value()) << "state 1 was not refused";
    EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << h.error().message;
    EXPECT_NE(h.error().message.find("SC-NEVPT2 has no attainable Hessian"), std::string::npos)
        << h.error().message;
    EXPECT_NE(h.error().message.find("reference root 1's class Si"), std::string::npos)
        << h.error().message;
    std::cout << std::format("[ qd a1b2  ] state 1 refused: {}\n", h.error().message);
  }

  const auto h = ref.hessian(*ref.solution, 0, q);
  ASSERT_TRUE(h.has_value()) << "state 0: " << h.error().message;
  ASSERT_EQ(h->hessian.size(), k * k);
  std::cout << std::format("[ qd a1b2  ] state 0: E {}, QD gap {:.4f}, edge {:.4f}, defect "
                           "{:.1e}, t_mix {:.3e}, term (ii)'s share {:.3e}, asymmetry {:.1e}, "
                           "rounding {:.2e} at {}\n",
                           h->eigenvalues, h->qd_gap, h->edge_gap, h->defect, h->mix_step,
                           h->term_ii_share, h->asymmetry, h->rounding, h->rounding_at);
  // No element couples two irreps: C2v's zeros, to the solves' error.
  double across = 0.0;
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      if (irreps[i] != irreps[j]) {
        across = std::max(across, std::abs(h->hessian[i + (k * j)]));
      }
    }
  }
  std::cout << std::format("[ qd a1b2  ] state 0: largest element across irreps {:.1e}, the "
                           "solves' bar {:.1e}\n",
                           across, solve_bar(*h));
  EXPECT_LE(across, solve_bar(*h));
  EXPECT_LE(h->asymmetry, solve_bar(*h));
  for (std::size_t i = 0; i < k; ++i) {
    std::string row;
    for (std::size_t j = 0; j < k; ++j) {
      row += std::format(" {:11.4e}", h->hessian[i + (k * j)]);
    }
    std::cout << std::format("[ qd a1b2  ] {:5}{}\n", labels[i], row);
  }

  // The rotation block from the gradient alone. With z x (x_A - c) = n v_A,
  // n and z x c = (d_x, d_y, 0) are fitted over the atoms by least squares
  // (exact for a rotation row); Sum_A G_A = 0 drops c from the sum.
  {
    const auto rz =
        static_cast<std::size_t>(std::ranges::find(labels, std::string{"R_z"}) - labels.begin());
    ASSERT_LT(rz, k);
    const std::vector<double> &v = vectors[rz];
    const fb::json &atoms = doc.at("atoms");
    // Normal equations over the unknowns (n, d_x, d_y): n v_Ax + d_x = -y_A,
    // n v_Ay + d_y = x_A.
    std::array<std::array<double, 4>, 3> a{};
    for (std::size_t at = 0; at < n_x / 3; ++at) {
      const auto x = atoms.at(at).at(1).get<std::vector<double>>();
      for (const auto &[row, rhs] : {std::pair{std::array{v[3 * at], 1.0, 0.0}, -x[1]},
                                     std::pair{std::array{v[(3 * at) + 1], 0.0, 1.0}, x[0]}}) {
        for (std::size_t r = 0; r < 3; ++r) {
          for (std::size_t c = 0; c < 3; ++c) {
            a[r][c] += row[r] * row[c];
          }
          a[r][3] += row[r] * rhs;
        }
      }
    }
    for (std::size_t p = 0; p < 3; ++p) {
      for (std::size_t r = 0; r < 3; ++r) {
        if (r != p) {
          const double f = a[r][p] / a[p][p];
          for (std::size_t c = 0; c < 4; ++c) {
            a[r][c] -= f * a[p][c];
          }
        }
      }
    }
    const double n = a[0][3] / a[0][0];
    const auto g = fb::nevpt2::qd_sc_gradient(
        ref.res(), ref.molecule(), ref.basis(), ref.classes(), ref.fit().aux, ref.h(), ref.fit().b,
        ref.fit().l.data(), *ref.solution, ref.ncore, ref.ncas, 0, 0);
    ASSERT_TRUE(g.has_value()) << g.error().message;
    ASSERT_EQ(g->total.size(), n_x);
    // z x v_A = (-v_Ay, v_Ax, 0).
    double want = 0.0;
    for (std::size_t at = 0; at < n_x / 3; ++at) {
      want -= ((g->total[3 * at] * -v[(3 * at) + 1]) + (g->total[(3 * at) + 1] * v[3 * at])) / n;
    }
    const double got = h->hessian[rz * (k + 1)];
    std::cout << std::format("[ qd a1b2  ] state 0: (R_z, R_z) {:.12e} against v^T L G "
                             "{:.12e} (n {:.6f}), miss {:.1e}, the solves' bar {:.1e}\n",
                             got, want, n, std::abs(got - want), solve_bar(*h));
    EXPECT_LE(std::abs(got - want), kBars * solve_bar(*h));
  }

  // h_01 along each b2 vibration: the couplings in the coordinates, against
  // the fixture's D at the finest rung (signs follow the roots' phases).
  const auto response = fb::hessian::df_qd_eigenvector_response(
      ref.res(), ref.molecule(), ref.basis(), ref.classes(), ref.fit().aux, ref.h(), ref.fit().b,
      ref.fit().l.data(), *ref.solution, ref.ncore, ref.ncas, false);
  ASSERT_TRUE(response.has_value()) << response.error().message;
  for (const fb::json &direction : doc.at("sc").at(0).at("directions")) {
    const auto dv = direction.at("vector").get<std::vector<double>>();
    ASSERT_EQ(dv.size(), n_x);
    double got = 0.0;
    for (std::size_t x = 0; x < n_x; ++x) {
      got += dv[x] * response->couplings[x];
    }
    const double want = direction.at("ladder").back().get<double>();
    const double bar = direction.at("bar").get<double>();
    std::cout << std::format("[ qd a1b2  ] h_01 along {}: {:.9e} against |D| {:.9e}, bar {:.1e}\n",
                             direction.at("direction").get<std::string>(), got, want, bar);
    EXPECT_LE(std::abs(std::abs(got) - std::abs(want)), kCouplingBars * bar)
        << direction.at("direction").get<std::string>();
  }
}

// The refusals before anything runs, each on a constructed input.
TEST(HessianDfQdScNevpt2, RefusesBeforeAnythingRuns) {
  FBF_REQUIRE_GPU();
  const Reference &ref = reference(qd_case("h2co_ccpvdz_sa2"));
  ASSERT_TRUE(ref.solution.has_value());
  const auto expect_refused = [](const auto &r, const fb::ErrorKind kind,
                                 const std::string_view what, const std::string_view names) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, kind) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_qd_sc_nevpt2"), std::string::npos)
        << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find(names), std::string::npos)
        << what << ": " << r.error().message;
    std::cout << std::format("[ qd sc    ] refuses {}: {}\n", what, r.error().message);
  };
  const std::vector<double> energies = ref.solution->roots.energies;
  const auto timed = [](const auto &call) {
    const auto start = std::chrono::steady_clock::now();
    auto r = call();
    return std::pair{
        std::move(r),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
  };

  // A frozen core.
  expect_refused(ref.hessian(*ref.solution, 1, {}, 1), fb::ErrorKind::Unsupported, "frozen core",
                 "frozen");
  // Two CASSCF roots within the gap floor.
  {
    std::vector<double> e = energies;
    e[1] = e[0] + 5e-9;
    expect_refused(ref.hessian(ref.with_energies(e), 1), fb::ErrorKind::Numerical, "root gap",
                   "roots 0 and 1");
  }
  // The ensemble's edge: root n - 1 moved to within 5e-9 of root n.
  {
    std::vector<double> e = energies;
    e.back() = ref.next_root() - 5e-9;
    const auto [r, seconds] = timed([&] { return ref.hessian(ref.with_energies(e), 1); });
    expect_refused(r, fb::ErrorKind::Numerical, "the ensemble's edge", "edge");
    std::cout << std::format("[ qd sc    ] the edge's refusal took {:.2f} s\n", seconds);
  }
  // A QD gap: root 1's energy moved so that H_eff's diagonal is equal (its
  // off-diagonal vanishes by C2v symmetry), df_qd_eigenvector_response_gpu's
  // construction.
  {
    const std::vector<double> totals = ref.totals();
    ASSERT_EQ(totals.size(), 4U);
    std::vector<double> e = energies;
    e[1] = energies[0] + totals[0] - totals[3];
    const auto [r, seconds] = timed([&] { return ref.hessian(ref.with_energies(e), 1); });
    expect_refused(r, fb::ErrorKind::Numerical, "a QD gap", "QD states 0 and 1");
    std::cout << std::format("[ qd sc    ] the QD gap's refusal took {:.2f} s\n", seconds);
  }
  // A state that is not one, and W's checks.
  expect_refused(ref.hessian(*ref.solution, 2), fb::ErrorKind::InvalidConfig, "state 2",
                 "not one of");
  const auto w_form = [&](std::span<const double> w) {
    return fb::hessian::df_qd_sc_nevpt2(ref.res(), ref.molecule(), ref.basis(), ref.classes(),
                                        ref.fit().aux, ref.h(), ref.fit().b, ref.fit().l.data(),
                                        *ref.solution, ref.ncore, ref.ncas, w);
  };
  expect_refused(w_form(std::vector<double>{1.0, 0.0, 0.0}), fb::ErrorKind::InvalidConfig,
                 "W short", "W of 3");
  expect_refused(w_form(std::vector<double>{0.5, 0.1, 0.2, 0.5}), fb::ErrorKind::InvalidConfig,
                 "W asymmetric", "not symmetric");
  expect_refused(
      w_form(std::vector<double>{0.5, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()}),
      fb::ErrorKind::InvalidConfig, "W not finite", "not finite");
}

// An inactive pair under 1e-6 Eh that a coordinate splits: the C3v ammonia's
// virtual e pairs, df_qd_nevpt2_tangents' refusal passed on.
TEST(HessianDfQdScNevpt2, RefusesADegenerateInactivePair) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference &ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.solution.has_value());
  const auto r = ref.hessian(*ref.solution, 1);
  ASSERT_FALSE(r.has_value()) << "a degenerate inactive pair was taken";
  EXPECT_EQ(r.error().kind, fb::ErrorKind::Numerical) << r.error().message;
  EXPECT_NE(r.error().message.find("hessian::df_qd_nevpt2_tangents"), std::string::npos)
      << r.error().message;
  EXPECT_NE(r.error().message.find("along coordinate "), std::string::npos) << r.error().message;
  std::cout << std::format("[ qd sc    ] nh3 refuses: {}\n", r.error().message);
}

} // namespace
