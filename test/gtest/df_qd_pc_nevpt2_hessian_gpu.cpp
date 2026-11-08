// GPU cases for hessian::df_qd_pc_nevpt2, the PC-QD-NEVPT2 analytic nuclear
// Hessian (docs/nevpt2/nevpt2.tex Eqs. qd-hess and qd-hessW), on this tree's
// own DF-SA-CASSCF of data/qd_nevpt2_hessian/'s cases through the registry,
// against their semi-numerical PC-QD Hessians (#953), at the default cut the
// fixtures were generated at:
//
// - h2co_skew_ccpvdz_sa3 (C1, equal weights), states K = 1 and K = 2, and
//   h2co_skew_ccpvdz_sa3_weighted (0.4/0.4/0.2), state K = 1: every element
//   within 2x the fixture's PC bar; each part's largest element in bars (the
//   visibility rule), theta's share at equal weights, and the reports;
// - h2nb_sto3g_sa2 (C2v A1/B2, its symmetry-adapted coordinates): exactly
//   #953's PC expected-refusal blocks, (B2 1, B2 1), (B2 1, B2 2) and
//   (B2 2, B2 2), refused on both states, each with its t* under 1e-3 bohr,
//   and nothing else; and, in the coordinates h2nb_sto3g_sa2_hessian's PC
//   ladders (every one but the two b2 vibrations, along which its sampler
//   refuses PC, #1093), every element of both states within 2x its bar;
// - the refusals before anything runs, each on a constructed input: a QD
//   gap, a CASSCF root gap and the ensemble's edge under kRootGapFloor, a
//   frozen core, and a class kept under 1e-10 (df_pc_nevpt2_hessian_gpu's
//   C2v formaldehyde CAS(6,6), here SA2), with pc_floor named; and a
//   degenerate inactive pair a coordinate splits (df_qd_nevpt2_tangents'
//   C3v ammonia at a pc_floor of 1e-10, PC's record of it).

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
// df_qd_pc_nevpt2's Schur-lift step (bohr): a refused block's t* is under it.
constexpr double kLiftStep = 1e-3;
// df_qd_pc_nevpt2's kept floor: a class kept under it is refused.
constexpr double kKept = 1e-10;

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

// df_pc_nevpt2_hessian_gpu.cpp's C2v formaldehyde in STO-3G with CAS(6,6)
// over orbitals 5 to 10, here an SA2: Sr's and Si's metrics have real modes
// between the default cut and 1e-10 of their largest.
Start c2v_formaldehyde() {
  return {
      .name = "h2co_c2v_sto3g_cas66_sa2",
      .atoms = {{"C", {0.0, 0.0, 0.0}},
                {"O", {0.0, 0.0, 1.21 / fb::kAngstromPerBohr<double>}},
                {"H",
                 {0.0, 0.94 / fb::kAngstromPerBohr<double>, -0.59 / fb::kAngstromPerBohr<double>}},
                {"H",
                 {0.0, -0.94 / fb::kAngstromPerBohr<double>,
                  -0.59 / fb::kAngstromPerBohr<double>}}},
      .basis = "sto-3g",
      .aux_basis = "cc-pVDZ-JKFIT",
      .ncas = 6,
      .nelecas = 6,
      .weights = {0.5, 0.5},
      .active = {5, 6, 7, 8, 9, 10}};
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

  [[nodiscard]] fb::Result<fb::hessian::QdPcNevpt2Terms>
  hessian(const fb::mcscf::Solution &s, const std::int32_t state,
          std::span<const double> coordinates = {}, const double pc_floor = 0.0,
          const std::int32_t nfrozen = 0) const {
    return fb::hessian::df_qd_pc_nevpt2(res(), molecule(), basis(), classes(), fit().aux, h(),
                                        fit().b, fit().l.data(), s, ncore, ncas, state, 0,
                                        coordinates, pc_floor, nfrozen);
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

  // PC's H_eff correlation elements C, n x n [M n + N].
  [[nodiscard]] std::vector<double> totals() const {
    std::vector<double> out;
    const fb::Status got = fb::nevpt2::qd_inputs(
        res(), molecule(), basis(), fit().aux, h(), fit().b, *solution, ncore, ncas,
        [&](const fb::nevpt2::MoBlocks &blocks,
            const std::span<const fb::nevpt2::EnergyInputs> pairs) -> fb::Status {
          const auto e = fb::nevpt2::pc_effective_hamiltonian(res(), blocks, pairs, 0);
          if (!e) {
            return std::unexpected(e.error());
          }
          std::ranges::transform(*e, std::back_inserter(out), &fb::nevpt2::PcClassEnergies::total);
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
  const fb::test::ScratchDir dir("df_qd_pc_nevpt2_hessian");
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
      ref.fit().l.data(), *ref.solution, weights, ref.ncore, ref.ncas, cp->cp, true,
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

// State @p state's analytic Hessian of fixture @p name against its PC
// reference: every element within kBars, each part visible, the reports.
void expect_matches(const std::string &name, const std::int32_t state) {
  const fb::json &doc = fixture(name);
  const Reference &ref = reference(start_of(doc, name));
  ASSERT_TRUE(ref.solution.has_value());
  const fb::json *want = nullptr;
  for (const fb::json &s : doc.at("pc").at("states")) {
    if (s.at("state").get<std::int32_t>() == state) {
      want = &s;
    }
  }
  ASSERT_NE(want, nullptr) << name << " has no state " << state;
  const auto start = std::chrono::steady_clock::now();
  const auto got = ref.hessian(*ref.solution, state);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(got.has_value()) << got.error().message;
  const fb::hessian::QdScNevpt2Terms &h = got->terms;
  const auto hessian = matrix(want->at("hessian"));
  const auto bar = matrix(want->at("bar"));
  const std::size_t k = hessian.size();
  ASSERT_EQ(h.hessian.size(), k * k);
  const std::string label = std::format("{} K={}", name, state);
  double worst = 0.0;
  double largest = 0.0;
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < k; ++i) {
      const double value = h.hessian[i + (k * j)];
      const double miss = std::abs(value - hessian[i][j]) / bar[i][j];
      worst = std::max(worst, miss);
      largest = std::max(largest, std::abs(hessian[i][j]));
      EXPECT_LE(miss, kBars) << label << ": element (" << i << ", " << j << ") " << value
                             << " against " << hessian[i][j] << ", bar " << bar[i][j];
    }
  }
  std::cout << std::format("[ qd pc    ] {}: worst element {:.3f} bars (allowed {:g}), largest "
                           "|H| {:.3e}; {:.1f} s\n",
                           label, worst, kBars, largest, seconds);

  // Visibility: each part's largest element in bars.
  for (const auto &[part_name, part] :
       {std::pair{"E_SA (E_rho)", &h.e_sa}, std::pair{"z d2g/dRdR", &h.z_lxy},
        std::pair{"z L_Ax x_B", &h.z_lx}, std::pair{"z third", &h.z_third},
        std::pair{"kappa_oo rows", &h.z_koo}, std::pair{"zeta", &h.zeta},
        std::pair{"skeleton", &h.skeleton}, std::pair{"<t, HVP t>", &h.hvp},
        std::pair{"tau-kappa", &h.rotation_tau}, std::pair{"kappa-kappa", &h.rotation_kappa},
        std::pair{"term (ii)", &h.term_ii}}) {
    const double bars = in_bars(*part, bar);
    std::cout << std::format("[ visible  ] {} {:14} {:10.3e} bars (rule {:g})\n", label, part_name,
                             bars, kVisibleBars);
    EXPECT_GT(bars, kVisibleBars) << label << ": " << part_name;
  }
  // PC's conditioning: every class kept above the floor, every t* past the
  // step.
  double kept = std::numeric_limits<double>::infinity();
  for (const std::array<double, 8> &classes : got->kept) {
    for (const double c : classes) {
      if (c > 0.0) {
        kept = std::min(kept, c);
      }
    }
  }
  double lift = std::numeric_limits<double>::infinity();
  for (const auto &coordinate : got->lift_steps) {
    for (const std::array<double, 8> &classes : coordinate) {
      lift = std::min(lift, std::ranges::min(classes));
    }
  }
  std::cout << std::format(
      "[ qd pc    ] {}: E {}, QD gap {:.4f}, root gap {:.4f}, edge {:.4f}, inactive gap {:.3e}, "
      "smallest kept {:.3e}, smallest t* {:.3e} bohr, max |theta| {:.3f}, max |kappa_oo| {:.2f}, "
      "max |zeta| {:.3e}, max |z^c| {:.3e}, term (ii)'s share {:.3e}; CP residual {:.1e}, "
      "Z-vector {:.1e}, asymmetry {:.1e} (bar {:.1e}), sum rule {:.1e}\n",
      label, h.eigenvalues, h.qd_gap, h.root_gap, h.edge_gap, h.min_gap, kept, lift, h.max_theta,
      h.max_kappa_oo, h.max_zeta, h.max_zc, h.term_ii_share, h.cp_residual, h.zvector_residual,
      h.asymmetry, solve_bar(h), h.translation);
  EXPECT_EQ(got->kept.size(), ref.solution->weights.size()) << label;
  EXPECT_EQ(got->lift_steps.size(), k) << label;
  EXPECT_GE(kept, kKept) << label;
  EXPECT_GE(lift, kLiftStep) << label;
  EXPECT_LE(h.cp_residual, kCpResidual) << label;
  EXPECT_LE(h.asymmetry, solve_bar(h)) << label;
  EXPECT_GT(h.term_ii_share, 0.0) << label;
  const bool equal = std::ranges::all_of(
      ref.solution->weights, [&](const double w) { return w == ref.solution->weights.front(); });
  if (equal) {
    expect_theta_visible(ref, h, bar, label);
  }
}

TEST(HessianDfQdPcNevpt2, MatchesTheSkewFixtureStateOne) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_matches("h2co_skew_ccpvdz_sa3", 1);
}

TEST(HessianDfQdPcNevpt2, MatchesTheSkewFixtureStateTwo) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_matches("h2co_skew_ccpvdz_sa3", 2);
}

TEST(HessianDfQdPcNevpt2, MatchesTheUnequalWeightFixture) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  expect_matches("h2co_skew_ccpvdz_sa3_weighted", 1);
}

// The A1/B2 coupling case in its symmetry-adapted coordinates: on both
// states the refusal names exactly #953's PC blocks, each with its coupling
// and a t* under the step, and no Schur lift elsewhere.
TEST(HessianDfQdPcNevpt2, CoupledA1B2RefusesTheExpectedBlocks) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const std::string name = "h2nb_sto3g_sa2";
  const fb::json &doc = fixture(name);
  std::set<std::pair<std::string, std::string>> expected;
  for (const fb::json &block : doc.at("expected_refusals").at("pc")) {
    const auto c = block.at("coordinates").get<std::vector<std::string>>();
    ASSERT_EQ(block.at("states").get<std::vector<std::int32_t>>(),
              (std::vector<std::int32_t>{0, 1}));
    expected.emplace(std::min(c[0], c[1]), std::max(c[0], c[1]));
  }
  ASSERT_FALSE(expected.empty());
  const Reference &ref = reference(start_of(doc, name));
  ASSERT_TRUE(ref.solution.has_value());
  const auto vectors = matrix(doc.at("coordinates").at("vectors"));
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  std::vector<double> q;
  for (const auto &row : vectors) {
    q.insert(q.end(), row.begin(), row.end());
  }
  for (const std::int32_t state : {0, 1}) {
    const auto start = std::chrono::steady_clock::now();
    const auto h = ref.hessian(*ref.solution, state, q);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_FALSE(h.has_value()) << "state " << state << ": the lifted blocks were differentiated";
    const std::string &m = h.error().message;
    EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << m;
    std::cout << std::format("[ qd a1b2  ] state {} refused in {:.1f} s: {}\n", state, seconds, m);
    // Only the blocks: no coordinate-wise branch switch beside them.
    EXPECT_EQ(m.find("switches branch"), std::string::npos) << m;
    const std::string_view lead = "PC-QD-NEVPT2 has no Hessian on the coordinate pairs ";
    const std::size_t at = m.find(lead);
    ASSERT_NE(at, std::string::npos) << m;
    const std::string named = m.substr(at + lead.size(), m.find(':', at) - at - lead.size());
    std::set<std::pair<std::string, std::string>> refused;
    const std::regex pair(R"(\((\d+), (\d+)\))");
    for (auto it = std::sregex_iterator(named.begin(), named.end(), pair);
         it != std::sregex_iterator(); ++it) {
      const std::string &u = labels.at(std::stoul((*it)[1].str()));
      const std::string &v = labels.at(std::stoul((*it)[2].str()));
      refused.emplace(std::min(u, v), std::max(u, v));
    }
    EXPECT_EQ(refused, expected) << m;
    // Each block is the coupling (0, 1), with its t* along both under the
    // step.
    const std::regex steps(
        R"(t\* ([0-9.e+-]+) along the direction, ([0-9.e+-]+) along its partner)");
    std::size_t reasons = 0;
    for (auto it = std::sregex_iterator(m.begin(), m.end(), steps); it != std::sregex_iterator();
         ++it, ++reasons) {
      EXPECT_LT(std::stod((*it)[1].str()), kLiftStep) << m;
      EXPECT_LT(std::stod((*it)[2].str()), kLiftStep) << m;
    }
    EXPECT_EQ(reasons, expected.size()) << m;
    EXPECT_NE(m.find("the coupling (0, 1)"), std::string::npos) << m;
  }
}

// The A1/B2 case's whole state Hessians in the coordinates the reference's
// PC ladders (#1093): its sampler refuses a point along either b2 vibration
// as a rank change of the cut, and every block of the two is an expected
// refusal, so h2nb_sto3g_sa2_hessian carries PC over every other coordinate
// (the totally symmetric block among them), and the analytic Hessian in
// those rows is held to it on both states, every element within kBars.
TEST(HessianDfQdPcNevpt2, CoupledA1B2MatchesOffTheRefusedCoordinates) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &doc = fixture("h2nb_sto3g_sa2_hessian");
  // The coupling case's system: one SA-CASSCF for both A1/B2 cases.
  const Reference &ref = reference(start_of(doc, "h2nb_sto3g_sa2"));
  ASSERT_TRUE(ref.solution.has_value());
  const auto vectors = matrix(doc.at("coordinates").at("vectors"));
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  const auto rows = doc.at("steps").at("pc").at("coordinates").get<std::vector<std::string>>();
  // Every coordinate but those of the coupling record's refused blocks.
  std::set<std::string> refused;
  for (const fb::json &block : fixture("h2nb_sto3g_sa2").at("expected_refusals").at("pc")) {
    for (const fb::json &c : block.at("coordinates")) {
      refused.insert(c.get<std::string>());
    }
  }
  std::vector<std::string> others;
  std::ranges::copy_if(labels, std::back_inserter(others),
                       [&](const std::string &l) { return !refused.contains(l); });
  ASSERT_EQ(rows, others);
  std::vector<double> q;
  for (const std::string &row : rows) {
    const auto at = std::ranges::find(labels, row) - labels.begin();
    q.insert(q.end(), vectors[at].begin(), vectors[at].end());
  }
  const std::size_t k = rows.size();
  for (const std::int32_t state : {0, 1}) {
    const fb::json &want = doc.at("pc").at("states").at(state);
    ASSERT_EQ(want.at("state").get<std::int32_t>(), state);
    const auto start = std::chrono::steady_clock::now();
    const auto got = ref.hessian(*ref.solution, state, q);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_TRUE(got.has_value()) << "state " << state << ": " << got.error().message;
    const fb::hessian::QdScNevpt2Terms &h = got->terms;
    const auto hessian = matrix(want.at("hessian"));
    const auto bar = matrix(want.at("bar"));
    ASSERT_EQ(hessian.size(), k);
    ASSERT_EQ(h.hessian.size(), k * k);
    double worst = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      for (std::size_t i = 0; i < k; ++i) {
        const double value = h.hessian[i + (k * j)];
        const double miss = std::abs(value - hessian[i][j]) / bar[i][j];
        worst = std::max(worst, miss);
        EXPECT_LE(miss, kBars) << "state " << state << ": (" << rows[i] << ", " << rows[j] << ") "
                               << value << " against " << hessian[i][j] << ", bar " << bar[i][j];
      }
    }
    std::cout << std::format("[ qd a1b2  ] state {}: worst element {:.3f} bars (allowed {:g}) over "
                             "{}; asymmetry {:.1e} (bar {:.1e}); {:.1f} s\n",
                             state, worst, kBars, rows, h.asymmetry, solve_bar(h), seconds);
    EXPECT_LE(h.asymmetry, solve_bar(h)) << "state " << state;
  }
}

// The refusals before anything runs, each on a constructed input.
TEST(HessianDfQdPcNevpt2, RefusesBeforeAnythingRuns) {
  FBF_REQUIRE_GPU();
  const Reference &ref = reference(qd_case("h2co_ccpvdz_sa2"));
  ASSERT_TRUE(ref.solution.has_value());
  const auto expect_refused = [](const auto &r, const fb::ErrorKind kind,
                                 const std::string_view what, const std::string_view names) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, kind) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_qd_pc_nevpt2"), std::string::npos)
        << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find(names), std::string::npos)
        << what << ": " << r.error().message;
    std::cout << std::format("[ qd pc    ] refuses {}: {}\n", what, r.error().message);
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
  expect_refused(ref.hessian(*ref.solution, 1, {}, 0.0, 1), fb::ErrorKind::Unsupported,
                 "frozen core", "frozen");
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
    std::cout << std::format("[ qd pc    ] the edge's refusal took {:.2f} s\n", seconds);
  }
  // A QD gap: root 1's energy moved so that PC's H_eff diagonal is equal (its
  // off-diagonal vanishes by C2v symmetry).
  {
    const std::vector<double> totals = ref.totals();
    ASSERT_EQ(totals.size(), 4U);
    std::vector<double> e = energies;
    e[1] = energies[0] + totals[0] - totals[3];
    const auto [r, seconds] = timed([&] { return ref.hessian(ref.with_energies(e), 1); });
    expect_refused(r, fb::ErrorKind::Numerical, "a QD gap", "QD states 0 and 1");
    std::cout << std::format("[ qd pc    ] the QD gap's refusal took {:.2f} s\n", seconds);
  }
  // A state that is not one, and W's checks.
  expect_refused(ref.hessian(*ref.solution, 2), fb::ErrorKind::InvalidConfig, "state 2",
                 "not one of");
  const auto w_form = [&](std::span<const double> w) {
    return fb::hessian::df_qd_pc_nevpt2(ref.res(), ref.molecule(), ref.basis(), ref.classes(),
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

// A class kept under 1e-10 of its metric's largest eigenvalue, at the
// default cut: refused before anything runs, naming the state, the class,
// its kept and the pc_floor.
TEST(HessianDfQdPcNevpt2, RefusesAThinKeptClass) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference &ref = reference(c2v_formaldehyde());
  ASSERT_TRUE(ref.solution.has_value());
  const auto start = std::chrono::steady_clock::now();
  const auto h = ref.hessian(*ref.solution, 1);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_FALSE(h.has_value()) << "a class kept below 1e-10 was differentiated";
  const std::string &m = h.error().message;
  EXPECT_EQ(h.error().kind, fb::ErrorKind::Numerical) << m;
  EXPECT_NE(m.find("hessian::df_qd_pc_nevpt2: state "), std::string::npos) << m;
  EXPECT_NE(m.find("at a pc_floor of 0,"), std::string::npos) << m;
  EXPECT_NE(m.find("a pc_floor of 1e-10 or more cuts it"), std::string::npos) << m;
  // Every value it names is under the bar.
  const std::regex keeps(R"(keeps ([0-9.e+-]+))");
  std::size_t named = 0;
  for (auto it = std::sregex_iterator(m.begin(), m.end(), keeps); it != std::sregex_iterator();
       ++it, ++named) {
    EXPECT_LT(std::stod((*it)[1].str()), kKept) << m;
  }
  EXPECT_GT(named, 0U) << m;
  std::cout << std::format("[ kept     ] refused in {:.2f} s: {}\n", seconds, m);
}

// A degenerate inactive pair a coordinate splits: the C3v ammonia's virtual
// e pairs, which PC's tangents record and df_qd_pc_nevpt2 refuses. At the
// default cut state 0's Si keeps 3.3e-11 and the kept refusal comes first,
// so the case runs at a pc_floor of 1e-10, the floor that refusal names.
constexpr double kAmmoniaFloor = 1e-10;
TEST(HessianDfQdPcNevpt2, RefusesADegenerateInactivePair) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference &ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.solution.has_value());
  const auto r = ref.hessian(*ref.solution, 1, {}, kAmmoniaFloor);
  ASSERT_FALSE(r.has_value()) << "a degenerate inactive pair was taken";
  const std::string &m = r.error().message;
  EXPECT_EQ(r.error().kind, fb::ErrorKind::Numerical) << m;
  EXPECT_NE(m.find("hessian::df_qd_pc_nevpt2: the canonical inactive orbitals are not "
                   "differentiable"),
            std::string::npos)
      << m;
  EXPECT_NE(m.find("(coordinate "), std::string::npos) << m;
  EXPECT_NE(m.find(", orbitals "), std::string::npos) << m;
  std::cout << std::format("[ qd pc    ] nh3 refuses: {}\n", m.substr(0, 400));
}

} // namespace
