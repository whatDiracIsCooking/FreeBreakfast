// mcscf::solve on the device, from each case's recorded start, against the
// converged, polished CASSCF of data/casscf/casscf.json
// (tools/emit/casscf_reference.py): each root's energy and the weighted one;
// the state-averaged natural occupations; ||g|| recomputed from scratch at the
// solution, with its own cold CI; C^T S C = 1; no accepted step that raises
// the energy; at most 20 macro iterations; and the refusals. Each case runs
// twice: integral-direct, and with the Hessian density-fitted
// (DensityFitting::ApproxHessian, over kAuxBasis), which must reach the same
// exact references in at most one macro iteration more. And each case runs
// wholly density-fitted (DensityFitting::Full, over kDfAuxBasis) against
// data/casscf/casscf_df.json's DF-CASSCF to the same bars, its ||g|| recomputed
// from scratch over the same B.
//
// The setup is response_fock_gpu.cpp's: the SCF (run with J unscreened), and the
// start picked by pipeline.active_space (by index, or by AVAS). So this
// executable links pipeline; mcscf itself does not. Runs the device chain, so
// needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOccupationTolerance = 1e-6;
constexpr double kOrthonormalTolerance = 1e-10;
constexpr std::int32_t kMaxMacroIterations = 20;
// mcscf::solve's own: a rise below it is noise, not a rejected step.
constexpr double kEnergyRiseTolerance = 1e-11;
// The test's convergence. Each root's energy is not stationary in the
// orbitals, and the natural occupations neither, so both are first order in
// the orbitals' error, ~||g|| / (H's least eigenvalue): at the default
// conv_tol_grad of 1e-6 that is ~1e-5, past both bars. At 1e-9 it is ~1e-8,
// and a root's energy moves by its own gradient (~0.1) times that.
constexpr double kConvTolGrad = 1e-9;
// ApproxHessian's B. Only the Hessian is fitted, so the set moves the path,
// not the solution: macro iterations within one of direct's (the issue's bar).
constexpr std::string_view kAuxBasis = "aug-cc-pvdz-accd-6-6";
constexpr std::size_t kApproxHessianExtraMacroIterations = 1;
// Full's B: the set casscf_df.json was fitted over (its aux_basis.file), so
// the comparison is of one fitting set, not two.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

// data/casscf/<file>, read once per file.
const fb::json &reference(std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto found = docs.find(file); found != docs.end()) {
    return found->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf" / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{file}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

const fb::json &reference_case(std::string_view name, std::string_view file = "casscf.json") {
  for (const fb::json &entry : reference(file).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf/" << file;
  static const fb::json kNone;
  return kNone;
}

// Copies in res.stream()'s order, waited for.
std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// The reference's molecule through the real registry to the Scf step's inputs, in
// basis if it is not empty and the reference's own otherwise. The Scf step itself
// is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node,
                                               std::string_view basis = {}) {
  const fb::test::ScratchDir dir("mcscf_solve");
  // Bohr in the reference, Angstrom in an XYZ file.
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
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[orthogonalizer]\n[core_hamiltonian]\n[guess]\n",
                  entry.at("charge").get<std::int32_t>(),
                  basis.empty() ? entry.at("basis").get<std::string>() : std::string{basis});
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return std::nullopt;
  }
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return std::nullopt;
  }
  return std::move(*calc);
}

// The reference's start as the Casci step's selection.
fb::CasciStep casci_step(const fb::json &entry) {
  fb::CasciStep step;
  if (!entry.at("avas").is_null()) {
    step.avas = entry.at("avas").at("labels").get<std::vector<std::string>>();
    step.avas_threshold = entry.at("avas").at("threshold").get<double>();
  } else {
    step.ncas = entry.at("ncas").get<std::int32_t>();
    step.nelecas = entry.at("nelecas").get<std::int32_t>();
    if (!entry.at("active").is_null()) {
      step.active = entry.at("active").get<std::vector<std::int32_t>>();
    }
  }
  step.nroots = static_cast<std::int32_t>(entry.at("weights").size());
  return step;
}

// Everything mcscf::solve reads, from one converged SCF, and the start.
struct Setup {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  const fb::Molecule *molecule = nullptr;
  const fb::BasisSet *basis = nullptr;
  const fb::SphericalTransformOutput *transform = nullptr;
  const fb::CoreHamiltonianOutput *h = nullptr;
  const fb::OverlapOutput *overlap = nullptr;
  std::optional<fb::ScfOutput> scf;
  std::vector<fb::CoulombPairClass> j_classes;
  std::vector<fb::OrderedPairClass> k_classes;
  std::optional<fb::ActiveSpace> start;

  [[nodiscard]] fb::DeviceResources &res() { return node.device(0); }

  [[nodiscard]] fb::Result<fb::mcscf::Solution>
  solve(const fb::SphericalMatrixAO &c, std::int32_t nroots, std::span<const double> weights,
        const fb::mcscf::Options &options, const fb::Buffers::device<double> *b = nullptr,
        std::int32_t naux = 0) {
    return fb::mcscf::solve(res(), *basis, *molecule, h->h, transform->u, j_classes, k_classes, c,
                            start->ncore, start->ncas, start->nelecas, nroots, weights, options, b,
                            naux);
  }

  // B over the setup's basis and aux_basis.
  [[nodiscard]] std::optional<fb::DensityFittingOutput> fit(std::string_view aux_basis) {
    const auto &mol = calc->get<fb::MoleculeOutput>();
    const auto &basis_set = calc->get<fb::BasisSetOutput>();
    const auto &pairs = calc->get<fb::PrimitivePairClassesOutput>();
    auto b = fb::DensityFittingStep{.aux_basis = std::string{aux_basis}}.run(
        fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
            *mol, *basis_set, *pairs},
        res());
    if (!b.has_value()) {
      ADD_FAILURE() << b.error().message;
      return std::nullopt;
    }
    return std::move(*b);
  }
};

// The setup over the reference's molecule (in basis if not empty), with the
// start selected when with_start.
std::unique_ptr<Setup> make_setup(const fb::json &entry, bool with_start,
                                  std::string_view basis = {}) {
  auto setup = std::make_unique<Setup>();
  auto calc = run_setup_steps(entry, setup->node, basis);
  if (!calc.has_value()) {
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc)); // a Pipeline moves, but is not move-assigned
  const auto &molecule = setup->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = setup->calc->get<fb::BasisSetOutput>();
  const auto &transform = setup->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = setup->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = setup->calc->get<fb::OverlapOutput>();
  const auto &x = setup->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = setup->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = setup->calc->get<fb::GuessOutput>();
  if (!molecule.has_value() || !basis_set.has_value() || !transform.has_value() ||
      !pairs.has_value() || !overlap.has_value() || !x.has_value() || !h.has_value() ||
      !guess.has_value()) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  setup->molecule = &molecule->molecule;
  setup->basis = &basis_set->basis;
  setup->transform = &*transform;
  setup->h = &*h;
  setup->overlap = &*overlap;
  auto scf = fb::test::kExactScf.run(fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform,
                                                                *pairs, *overlap, *x, *h, *guess},
                                     setup->res());
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return nullptr;
  }
  setup->scf.emplace(std::move(*scf));

  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto k_classes = fb::ordered_pair_classes(res, basis_set->basis, pairs->classes);
  if (!j_classes.has_value() || !k_classes.has_value()) {
    ADD_FAILURE() << "the J or K classes failed";
    return nullptr;
  }
  setup->j_classes = std::move(*j_classes);
  setup->k_classes = std::move(*k_classes);
  if (!with_start) {
    return setup;
  }
  auto space = casci_step(entry).select(res, molecule->molecule, basis_set->basis, *setup->scf);
  if (!space.has_value()) {
    ADD_FAILURE() << space.error().message;
    return nullptr;
  }
  setup->start.emplace(std::move(*space));
  EXPECT_EQ(setup->start->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(setup->start->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(setup->start->nelecas, entry.at("nelecas").get<std::int32_t>());
  return setup;
}

// The weighted energy and ||g|| at orbitals c from scratch: the integrals, a
// cold CI at ci::solve's own tolerance, the RDMs and response::fock, nothing of
// mcscf::solve's. Integral-direct, or over df's B when it is given.
std::optional<std::pair<double, double>>
from_scratch(Setup &setup, const fb::SphericalMatrixAO &c, const std::vector<double> &weights,
             const fb::DensityFittingOutput *df = nullptr) {
  fb::DeviceResources &res = setup.res();
  const std::int32_t ncore = setup.start->ncore;
  const std::int32_t ncas = setup.start->ncas;
  const std::int32_t nelecas = setup.start->nelecas;
  auto ints = df != nullptr ? fb::casscf_integrals(res, *setup.basis, *setup.molecule, setup.h->h,
                                                   df->b, df->naux, c, ncore, ncas)
                            : fb::casscf_integrals(res, *setup.basis, *setup.molecule, setup.h->h,
                                                   setup.transform->u, setup.j_classes,
                                                   setup.k_classes, c, ncore, ncas);
  if (!ints.has_value()) {
    ADD_FAILURE() << ints.error().message;
    return std::nullopt;
  }
  const auto n = static_cast<std::size_t>(ncas);
  const std::vector<double> h_col = download(res, ints->hamiltonian.h_eff.data(), n * n);
  const std::vector<double> eri_col = download(res, ints->hamiltonian.eri.data(), n * n * n * n);
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }
  auto alpha = fb::ci::StringSet::create(ncas, nelecas - (nelecas / 2));
  auto beta = fb::ci::StringSet::create(ncas, nelecas / 2);
  if (!alpha.has_value() || !beta.has_value()) {
    ADD_FAILURE() << "the string sets failed";
    return std::nullopt;
  }
  auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  if (!sigma.has_value() || !rdm.has_value() ||
      !sigma->set_integrals(res, ints->hamiltonian.e_core, h_eff, eri).has_value()) {
    ADD_FAILURE() << "the sigma or the RDMs failed";
    return std::nullopt;
  }
  const auto dim = static_cast<std::size_t>(sigma->size());
  fb::Buffers::device<double> diagonal(dim, res.device());
  EXPECT_TRUE(sigma->diagonal(res, diagonal.data()).has_value());
  auto roots = fb::ci::solve(res, *sigma, *alpha, *beta, diagonal.data(),
                             static_cast<std::int32_t>(weights.size()));
  if (!roots.has_value()) {
    ADD_FAILURE() << roots.error().message;
    return std::nullopt;
  }
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  if (!rdm->rdm12(res, weights, roots->vectors.data(), gamma.data(), gamma2.data()).has_value()) {
    ADD_FAILURE() << "rdm12 failed";
    return std::nullopt;
  }
  const std::vector<double> g1 = download(res, gamma.data(), n * n);
  const std::vector<double> g2 = download(res, gamma2.data(), n * n * n * n);
  auto fock = df != nullptr
                  ? fb::response::fock(res, *setup.molecule, setup.h->h, df->b, df->naux, c, ncore,
                                       ncas, ints->ppaa, ints->papa, g1, g2)
                  : fb::response::fock(res, *setup.basis, *setup.molecule, setup.h->h,
                                       setup.transform->u, setup.j_classes, setup.k_classes, c,
                                       ncore, ncas, ints->ppaa, ints->papa, g1, g2);
  if (!fock.has_value()) {
    ADD_FAILURE() << fock.error().message;
    return std::nullopt;
  }
  const std::vector<double> g = download(res, fock->gradient.data(), fock->gradient.num_elements());
  return std::pair{fock->energy, std::sqrt(std::inner_product(g.begin(), g.end(), g.begin(), 0.0))};
}

// The eigenvalues of the symmetric n x n a, descending, by cyclic Jacobi.
std::vector<double> eigenvalues_descending(std::vector<double> a, std::size_t n) {
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        off += a[p + (n * q)] * a[p + (n * q)];
      }
    }
    if (off < 1e-30) {
      break;
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = a[p + (n * q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double theta = (a[q + (n * q)] - a[p + (n * p)]) / (2.0 * apq);
        const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
        const double cs = 1.0 / std::hypot(t, 1.0);
        const double sn = t * cs;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k + (n * p)];
          const double akq = a[k + (n * q)];
          a[k + (n * p)] = (cs * akp) - (sn * akq);
          a[k + (n * q)] = (sn * akp) + (cs * akq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[p + (n * k)];
          const double aqk = a[q + (n * k)];
          a[p + (n * k)] = (cs * apk) - (sn * aqk);
          a[q + (n * k)] = (sn * apk) + (cs * aqk);
        }
      }
    }
  }
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[i + (n * i)];
  }
  std::ranges::sort(values, std::greater<>{});
  return values;
}

// max |C^T S C - 1| on the host.
double orthonormality_error(fb::DeviceResources &res, const fb::SphericalMatrixAO &c,
                            const fb::SphericalMatrixAO &s) {
  const std::size_t n = c.n();
  const std::vector<double> ch = download(res, c.data(), n * n);
  const std::vector<double> sh = download(res, s.data(), n * n);
  std::vector<double> sc(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t i = 0; i < n; ++i) {
        sc[i + (n * j)] += sh[i + (n * k)] * ch[k + (n * j)];
      }
    }
  }
  double worst = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      double value = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        value += ch[k + (n * i)] * sc[k + (n * j)];
      }
      worst = std::max(worst, std::abs(value - (i == j ? 1.0 : 0.0)));
    }
  }
  return worst;
}

struct Case {
  std::string_view name;
  // Under slow_test.h: every case but H2O cc-pVDZ, the cheapest.
  bool slow;
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class McscfSolve : public ::testing::TestWithParam<Case> {};

// Every check of one solve's solution sol against the case's reference,
// logged (with the solve's wall time, seconds) and traced as mode. ||g|| is
// recomputed over df's B when it is given (the DF-CASSCF's own gradient).
void expect_polished(Setup &setup, const fb::mcscf::Solution &sol, const fb::json &entry,
                     const std::vector<double> &weights, std::string_view mode, double seconds,
                     const fb::DensityFittingOutput *df = nullptr) {
  SCOPED_TRACE(mode);
  const fb::json &want = entry.at("casscf");

  // The macro iterations, logged and within the regression bar.
  const auto macro = static_cast<std::int32_t>(sol.iterations.size());
  const std::int32_t products = std::accumulate(
      sol.iterations.begin(), sol.iterations.end(), 0,
      [](std::int32_t sum, const fb::mcscf::Iteration &it) { return sum + it.hessian_products; });
  std::cout << std::format(
      "[ casscf   ] {} {}: {} macro iterations, {} Hessian products, {:.2f} s\n",
      entry.at("name").get<std::string>(), mode, macro, products, seconds);
  for (std::size_t i = 0; i < sol.iterations.size(); ++i) {
    const fb::mcscf::Iteration &it = sol.iterations[i];
    std::cout << std::format("[ casscf   ]   {:2} E {:.12f} dE {:+.3e} |g| {:.3e} radius {:.3f} "
                             "ci {:2} + {:2} micro {} Hx {:3}{}\n",
                             i + 1, it.energy, it.energy_change, it.gradient_norm, it.trust_radius,
                             it.ci_fp32_iterations, it.ci_iterations, it.micro_iterations,
                             it.hessian_products, it.accepted ? "" : " rejected");
  }
  ::testing::Test::RecordProperty(std::format("{}_macro_iterations", mode), macro);
  EXPECT_LE(macro, kMaxMacroIterations);

  // The energies: each root's and the weighted one.
  ASSERT_EQ(sol.roots.energies.size(), weights.size());
  const auto energies = want.at("energies").get<std::vector<double>>();
  double weighted = 0.0;
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(sol.roots.energies[r], energies[r], kEnergyTolerance) << "root " << r;
    EXPECT_NEAR(sol.roots.spin_squares[r], 0.0, 1e-8) << "root " << r;
    weighted += weights[r] * sol.roots.energies[r];
  }
  EXPECT_NEAR(sol.energy, want.at("energy").get<double>(), kEnergyTolerance);
  EXPECT_NEAR(weighted, sol.energy, 1e-10);

  // The state-averaged natural occupations, and the weighted gamma as the
  // weighted sum of the roots'.
  const auto n = static_cast<std::size_t>(setup.start->ncas);
  ASSERT_EQ(sol.gamma.size(), n * n);
  ASSERT_EQ(sol.gamma2.size(), n * n * n * n);
  ASSERT_EQ(sol.root_gammas.size(), weights.size() * n * n);
  const std::vector<double> occupations = eigenvalues_descending(sol.gamma, n);
  const auto expected = want.at("natural_occupations").get<std::vector<double>>();
  ASSERT_EQ(expected.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(occupations[i], expected[i], kOccupationTolerance) << "occupation " << i;
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    double sum = 0.0;
    for (std::size_t r = 0; r < weights.size(); ++r) {
      sum += weights[r] * sol.root_gammas[(r * n * n) + i];
    }
    EXPECT_NEAR(sum, sol.gamma[i], 1e-10) << "element " << i;
  }

  // ||g|| from scratch, and the energy with it.
  EXPECT_LT(sol.gradient_norm, kConvTolGrad);
  const auto scratch = from_scratch(setup, sol.c, weights, df);
  ASSERT_TRUE(scratch.has_value());
  EXPECT_LT(scratch->second, kConvTolGrad);
  EXPECT_NEAR(scratch->first, sol.energy, 1e-10);

  // C stays orthonormal.
  EXPECT_LT(orthonormality_error(setup.res(), sol.c, setup.overlap->s), kOrthonormalTolerance);

  // No accepted step raises the energy, and a rejected iteration did rise.
  for (std::size_t i = 1; i < sol.iterations.size(); ++i) {
    const fb::mcscf::Iteration &it = sol.iterations[i];
    if (it.accepted) {
      EXPECT_LE(it.energy_change, kEnergyRiseTolerance) << "macro iteration " << i + 1;
    } else {
      EXPECT_GT(it.energy_change, kEnergyRiseTolerance) << "macro iteration " << i + 1;
    }
  }
  EXPECT_TRUE(sol.iterations.back().accepted);
}

TEST_P(McscfSolve, ConvergesToThePolishedCasscf) {
  FBF_REQUIRE_GPU();
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name);
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  auto start = std::chrono::steady_clock::now();
  auto direct = setup->solve(setup->start->c, nroots, weights, options);
  const std::chrono::duration<double> direct_time = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  expect_polished(*setup, *direct, entry, weights, "direct", direct_time.count());

  const auto b = setup->fit(kAuxBasis);
  ASSERT_TRUE(b.has_value());
  options.density_fitting = fb::mcscf::DensityFitting::ApproxHessian;
  start = std::chrono::steady_clock::now();
  auto approx = setup->solve(setup->start->c, nroots, weights, options, &b->b, b->naux);
  const std::chrono::duration<double> approx_time = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(approx.has_value()) << approx.error().message;
  expect_polished(*setup, *approx, entry, weights, "approx_hessian", approx_time.count());
  // The fitted Hessian moves the path only so far.
  EXPECT_LE(approx->iterations.size(),
            direct->iterations.size() + kApproxHessianExtraMacroIterations);
}

// Wholly density-fitted, from the same start (casscf_df.json's cases start
// from the exact RHF and the same pick), against the DF-CASSCF over the same
// fitting set.
TEST_P(McscfSolve, FullDensityFittingConvergesToThePolishedDfCasscf) {
  FBF_REQUIRE_GPU();
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name, "casscf_df.json");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());

  // The reference's set, by name and size.
  const fb::json &aux = entry.at("aux_basis");
  ASSERT_EQ(aux.at("name").get<std::string>(), kDfAuxBasis);
  const auto b = setup->fit(kDfAuxBasis);
  ASSERT_TRUE(b.has_value());
  ASSERT_EQ(b->naux, aux.at("naux").get<std::int32_t>());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  const auto start = std::chrono::steady_clock::now();
  auto full = setup->solve(setup->start->c, nroots, weights, options, &b->b, b->naux);
  const std::chrono::duration<double> full_time = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(full.has_value()) << full.error().message;
  expect_polished(*setup, *full, entry, weights, "full_df", full_time.count(), &*b);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfSolve,
                         ::testing::Values(Case{.name = "h2o_ccpvdz", .slow = false},
                                           Case{.name = "n2_ccpvdz", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2_weighted", .slow = true},
                                           Case{.name = "h2o_ccpvdz_avas", .slow = true}),
                         [](const auto &info) { return std::string(info.param.name); });

// max_cycle running out is MaxIterationsExceeded.
TEST(McscfSolveLimits, OutOfMacroIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  auto setup = make_setup(entry, true, "sto-3g");
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  fb::mcscf::Options options;
  options.max_cycle = 2;
  const auto got = setup->solve(setup->start->c, 1, {}, options);
  ASSERT_FALSE(got.has_value());
  EXPECT_EQ(got.error().kind, fb::ErrorKind::MaxIterationsExceeded) << got.error().message;
}

// The weights and the other refusals, before the loop runs.
TEST(McscfSolveRefusals, AreInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  auto setup = make_setup(entry, true, "sto-3g");
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  const fb::SphericalMatrixAO &c = setup->start->c;
  const auto refused = [](const fb::Result<fb::mcscf::Solution> &r, std::string_view what) {
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(r.error().message.find(what), std::string::npos) << r.error().message;
  };
  const fb::mcscf::Options defaults;
  const std::vector<double> negative{1.2, -0.2};
  const std::vector<double> short_sum{0.5, 0.4};
  const std::vector<double> one{1.0};
  refused(setup->solve(c, 2, one, defaults), "weights");
  refused(setup->solve(c, 2, negative, defaults), "weights");
  refused(setup->solve(c, 2, short_sum, defaults), "weights");
  refused(setup->solve(c, 0, {}, defaults), "nroots");
  fb::mcscf::Options bad = defaults;
  bad.max_cycle = 0;
  refused(setup->solve(c, 1, {}, bad), "max_cycle");
  bad = defaults;
  bad.conv_tol_grad = 0.0;
  refused(setup->solve(c, 1, {}, bad), "conv_tol");
  bad = defaults;
  bad.trust_radius = -1.0;
  refused(setup->solve(c, 1, {}, bad), "trust_radius");
  // A density-fitted mode without B, and a B of the wrong size.
  const fb::Buffers::device<double> short_b(3, setup->res().device());
  for (const auto mode :
       {fb::mcscf::DensityFitting::ApproxHessian, fb::mcscf::DensityFitting::Full}) {
    bad = defaults;
    bad.density_fitting = mode;
    refused(setup->solve(c, 1, {}, bad), "needs B");
    refused(setup->solve(c, 1, {}, bad, &short_b, 1), "naux nmo^2");
  }
  const auto wrong_space =
      fb::mcscf::solve(setup->res(), *setup->basis, *setup->molecule, setup->h->h,
                       setup->transform->u, setup->j_classes, setup->k_classes, c, 6, 2, 4, 1);
  refused(wrong_space, "ncore + ncas");
  const auto too_many =
      fb::mcscf::solve(setup->res(), *setup->basis, *setup->molecule, setup->h->h,
                       setup->transform->u, setup->j_classes, setup->k_classes, c, 3, 2, 5, 1);
  refused(too_many, "nelecas");
  // CAS(4,2) has one determinant.
  const auto past_dim =
      fb::mcscf::solve(setup->res(), *setup->basis, *setup->molecule, setup->h->h,
                       setup->transform->u, setup->j_classes, setup->k_classes, c, 3, 2, 4, 2);
  refused(past_dim, "exceeds the 1 determinants");
}

} // namespace
