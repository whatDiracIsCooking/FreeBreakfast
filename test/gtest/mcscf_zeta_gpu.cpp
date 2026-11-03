// mcscf::root_pair_multipliers and mcscf::zeta_terms, the multipliers that
// hold a state average's roots apart (gradient/README.md "Holding the roots
// apart"), at mcscf_nac_gpu.cpp's converged solutions: H2CO SA-2 and the
// skewed H2CO SA-3, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT, this tree's
// DF-SA-CASSCF from the exact RHF and the Casci step's start, converged to
// ||g|| < 1e-10 and taken as it converges.
//
// McscfZetaTerms, on both cases, at a fixed zeta (one pair of SA-3 at zero,
// to exercise the skip):
//
// - two calls bitwise the same: the measured spread between two identical
//   calls is 0 (transition_orbital_gradient is bitwise, transition/README.md
//   "Reproducibility", and the folds are fixed-order axpys and host sums), so
//   the cross-call bar is the bit;
// - zeta . g bitwise Sum_sr zeta_sr g^(sr), the sum formed here from the
//   per-pair transition_orbital_gradient calls by the same axpys in the same
//   order, and the folded RDMs bitwise Sum_sr zeta_sr times trans_rdm12's;
// - df_transition_gradient at the folded RDMs against Sum_sr zeta_sr times
//   the per-pair calls, every term. df_transition_gradient is not bitwise
//   from call to call (its derivative kernels add atomically,
//   gradient/README.md "Reproducibility"), so this is held at its measured
//   floor, round_off_bar: the larger of 1e3 eps of the total row's largest
//   element and four times the worst spread measured over 60 repeats, per
//   the milestone's cross-call rule, every spread printed beside it.
//
// McscfRootPairMultipliers, on SA-3 (three pairs), at a random CI adjoint
// y_bar from a fixed seed and three sets of weights (equal; 0.4, 0.4, 0.2,
// one equal pair and two unequal; 0.5, 0.3, 0.2, none equal):
//
// - tau against c_s . y_r - c_r . y_s on the host at round_off_bar (1e3 eps
//   of the largest |tau|, or four times the worst measured), and bitwise
//   the same at every weight;
// - each equal pair's zeta bitwise -tau / (E_s - E_r) and t zero, each
//   unequal pair's t bitwise tau / (2 (w_r - w_s)) and zeta zero;
// - refusals: two roots under kRootGapFloor apart as Numerical naming both,
//   weights of the wrong length, and a zeta_terms zeta of the wrong length,
//   as InvalidConfig.
//
// Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.mcscf.zvector; // rotating_pairs: internal, not re-exported
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The CASSCF's convergence, as mcscf_nac_gpu.cpp's.
constexpr double kConvTolGrad = 1e-10;
// The round-off bar of a comparison between two calls (the milestone's
// rule): the larger of kRelativeUlps eps of the largest element of the whole
// quantity compared (the nuclear gradient's total row; tau's every element)
// and kSpreadMargin times the worst spread measured over 60 repeats of each
// case in one process on the RTX 3080, under the card lock.
constexpr double kRelativeUlps = 1e3;
constexpr double kSpreadMargin = 4.0;
// The worst spread of any term, two calls or the fold against the per-pair
// sum, per case, over two sessions of 60 repeats (120 per case), rounded up:
// SA-2 2.36e-15 (two calls 6.7e-16, the fold 2.36e-15) against a total row
// peaking at 1.83e-2, so its bar is 4 x 2.4e-15 = 9.6e-15 over 1e3 eps's
// 4.1e-15; SA-3 1.76e-14 (two calls 1.13e-14, the fold 1.76e-14) against a
// total row peaking at 0.128 to 0.143 (the CI phases differ between solves),
// so 7.2e-14 over 1e3 eps's 2.8e-14 to 3.2e-14. Both about 5e-13 relative.
// The spread rose from 30 repeats to 60 only within the same order (its
// 99th percentile is two thirds of its worst).
constexpr double kWorstSpreadSa2 = 2.4e-15;
constexpr double kWorstSpreadSa3 = 1.8e-14;
// The worst |tau - (c_s . y_r - c_r . y_s)| against the host's dots, 8.88e-16
// over 180 checks; 1e3 eps of the largest |tau| (about 2.2) is the bar.
constexpr double kWorstTauSpread = 8.9e-16;

// The bar at the largest element @p scale of the quantity, over @p worst.
double round_off_bar(double scale, double worst) {
  return std::max(kRelativeUlps * std::numeric_limits<double>::epsilon() * scale,
                  kSpreadMargin * worst);
}
// The zeta of McscfZetaTerms, one per pair: its second pair (SA-3's (0, 2))
// at zero, which zeta_terms skips.
constexpr std::array<double, 3> kZeta{0.7, 0.0, -1.3};

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_nac" / "casscf_nac.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_nac/casscf_nac.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule through the real registry to the Scf step's
// inputs. The Scf step itself is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_zeta");
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
                  entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>());
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

// A reference case's converged DF-SA-CASSCF, with the CI strings' RDM object
// and the fitted integrals over its orbitals, kept together because the
// outputs refer into the calculation.
struct Converged {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::EriOutput> fitted;
  std::optional<fb::ActiveSpace> space;
  std::optional<fb::mcscf::Solution> sol;
  std::optional<fb::ci::DeviceStringSet> alpha;
  std::optional<fb::ci::DeviceStringSet> beta;
  std::optional<fb::ci::Rdm> rdm;
  std::optional<fb::CasscfIntegrals> ints;
  fb::DeviceResources &res() { return node.device(0); }
};

// @p entry converged, or null after a failure is recorded.
std::unique_ptr<Converged> converge(const fb::json &entry) {
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  auto run = std::make_unique<Converged>();
  fb::DeviceResources &res = run->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  std::optional<fb::Calculation> calc = run_setup_steps(entry, run->node);
  if (!calc.has_value()) {
    return nullptr;
  }
  run->calc.emplace(std::move(*calc));
  const auto &molecule = run->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = run->calc->get<fb::BasisSetOutput>();
  const auto &transform = run->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = run->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = run->calc->get<fb::OverlapOutput>();
  const auto &x = run->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = run->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = run->calc->get<fb::GuessOutput>();
  if (!(molecule.has_value() && basis_set.has_value() && transform.has_value() &&
        pairs.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
        guess.has_value())) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  const fb::BasisSet &basis = basis_set->basis;
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs};
  auto eri = fb::EriStep{}.run(view, res);
  if (!eri.has_value()) {
    ADD_FAILURE() << eri.error().message;
    return nullptr;
  }
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return nullptr;
  }
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = entry.at("aux_basis").at("name").get<std::string>()}
                    .run(view, res);
  if (!fitted.has_value()) {
    ADD_FAILURE() << fitted.error().message;
    return nullptr;
  }
  run->fitted.emplace(std::move(*fitted));
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  fb::CasciStep casci;
  casci.ncas = entry.at("ncas").get<std::int32_t>();
  casci.nelecas = entry.at("nelecas").get<std::int32_t>();
  casci.nroots = nroots;
  auto space = casci.select(res, molecule->molecule, basis, *scf);
  if (!space.has_value()) {
    ADD_FAILURE() << space.error().message;
    return nullptr;
  }
  run->space.emplace(std::move(*space));

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {},
                              run->space->c, run->space->ncore, run->space->ncas,
                              run->space->nelecas, nroots, weights, options, &df.b, df.naux);
  if (!sol.has_value()) {
    ADD_FAILURE() << sol.error().message;
    return nullptr;
  }
  run->sol.emplace(std::move(*sol));

  const std::int32_t ncas = run->space->ncas;
  const std::int32_t nelecas = run->space->nelecas;
  const std::int32_t two_s = molecule->molecule.multiplicity() - 1;
  const std::int32_t n_beta = (nelecas - two_s) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta);
  auto beta = fb::ci::DeviceStringSet::create(res, ncas, n_beta);
  if (!alpha.has_value() || !beta.has_value()) {
    ADD_FAILURE() << "the CI strings failed";
    return nullptr;
  }
  run->alpha.emplace(std::move(*alpha));
  run->beta.emplace(std::move(*beta));
  auto rdm = fb::ci::Rdm::create(res, *run->alpha, *run->beta);
  if (!rdm.has_value()) {
    ADD_FAILURE() << rdm.error().message;
    return nullptr;
  }
  run->rdm.emplace(std::move(*rdm));
  auto ints = fb::casscf_integrals(res, basis, molecule->molecule, h->h, df.b, df.naux, run->sol->c,
                                   run->space->ncore, ncas);
  if (!ints.has_value()) {
    ADD_FAILURE() << ints.error().message;
    return nullptr;
  }
  run->ints.emplace(std::move(*ints));
  return run;
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

std::uint64_t bits(double v) {
  return std::bit_cast<std::uint64_t>(v);
}

// The elements of @p a and @p b whose bits differ.
std::size_t differing(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  std::size_t count = 0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    count += bits(a[i]) != bits(b[i]) ? 1U : 0U;
  }
  return count;
}

// max_i |a_i - b_i|.
double spread(std::span<const double> a, std::span<const double> b) {
  double out = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    out = std::max(out, std::abs(a[i] - b[i]));
  }
  return out;
}

double largest(std::span<const double> a) {
  double out = 0.0;
  for (const double v : a) {
    out = std::max(out, std::abs(v));
  }
  return out;
}

class McscfZetaTerms : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfZetaTerms, AreThePairSums) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const auto start_time = std::chrono::steady_clock::now();
  const auto run = converge(reference_case(name));
  ASSERT_NE(run, nullptr);
  fb::DeviceResources &res = run->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &molecule = run->calc->get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = run->calc->get<fb::BasisSetOutput>()->basis;
  const auto &classes = run->calc->get<fb::PrimitivePairClassesOutput>()->classes;
  const fb::SphericalMatrixAO &h = run->calc->get<fb::CoreHamiltonianOutput>()->h;
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  const fb::mcscf::Solution &sol = *run->sol;
  const std::int32_t ncore = run->space->ncore;
  const std::int32_t ncas = run->space->ncas;
  const std::size_t nroots = sol.roots.energies.size();
  const std::size_t n_pairs = nroots * (nroots - 1) / 2;
  ASSERT_LE(n_pairs, kZeta.size());
  const std::span<const double> zeta{kZeta.data(), n_pairs};
  const auto n = static_cast<std::size_t>(ncas);
  const std::size_t m = n * n;
  const auto dim = static_cast<std::size_t>(run->rdm->size());

  const auto terms = [&] {
    return fb::mcscf::zeta_terms(res, molecule, h, df.b, df.naux, sol, ncore, ncas, run->ints->ppaa,
                                 run->ints->papa, *run->rdm, zeta);
  };
  auto first = terms();
  ASSERT_TRUE(first.has_value()) << first.error().message;
  auto second = terms();
  ASSERT_TRUE(second.has_value()) << second.error().message;
  const std::size_t n_orbital = first->orbital.num_elements();
  const std::vector<double> zeta_g = download(res, first->orbital.data(), n_orbital);
  const std::vector<double> zeta_g_again = download(res, second->orbital.data(), n_orbital);
  ASSERT_EQ(first->gamma.size(), m);
  ASSERT_EQ(first->gamma2.size(), m * m);

  // Two calls: the measured spread is 0, so the bar is the bit.
  EXPECT_EQ(differing(zeta_g, zeta_g_again), 0U) << "spread " << spread(zeta_g, zeta_g_again);
  EXPECT_EQ(differing(first->gamma, second->gamma), 0U);
  EXPECT_EQ(differing(first->gamma2, second->gamma2), 0U);

  // The per-pair sums, each pair's g^(sr) from its own call.
  fb::Buffers::device<double> sum(n_orbital, res.device());
  ASSERT_EQ(wwr::wwrMemsetAsync(sum.data(), 0, n_orbital * sizeof(double), res.stream()),
            wwr::wwrSuccess);
  std::vector<double> gamma(m, 0.0);
  std::vector<double> gamma2(m * m, 0.0);
  std::vector<std::vector<double>> pair_gamma;
  std::vector<std::vector<double>> pair_gamma2;
  fb::Buffers::device<double> rdms(m + (m * m), res.device());
  std::size_t pair = 0;
  for (std::size_t r = 0; r < nroots; ++r) {
    for (std::size_t s = r + 1; s < nroots; ++s, ++pair) {
      const double *const ci_s = sol.roots.vectors.data() + (s * dim);
      const double *const ci_r = sol.roots.vectors.data() + (r * dim);
      ASSERT_TRUE(run->rdm->trans_rdm12(res, ci_s, ci_r, rdms.data(), rdms.data() + m).has_value());
      pair_gamma.push_back(download(res, rdms.data(), m));
      pair_gamma2.push_back(download(res, rdms.data() + m, m * m));
      const double z = zeta[pair];
      if (z == 0.0) {
        continue;
      }
      auto g = fb::mcscf::transition_orbital_gradient(res, molecule, h, df.b, df.naux, sol.c, ncore,
                                                      ncas, run->ints->ppaa, run->ints->papa,
                                                      pair_gamma.back(), pair_gamma2.back());
      ASSERT_TRUE(g.has_value()) << g.error().message;
      ASSERT_EQ(g->num_elements(), n_orbital);
      const auto status = wwr::axpy<double, std::int32_t>(
          res.blas().get(), static_cast<std::int32_t>(n_orbital), &z, g->data(), 1, sum.data(), 1);
      ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
      ASSERT_TRUE(res.sync().has_value());
      for (std::size_t i = 0; i < m; ++i) {
        gamma[i] += z * pair_gamma.back()[i];
      }
      for (std::size_t i = 0; i < m * m; ++i) {
        gamma2[i] += z * pair_gamma2.back()[i];
      }
    }
  }
  const std::vector<double> want_g = download(res, sum.data(), n_orbital);
  EXPECT_EQ(differing(zeta_g, want_g), 0U) << "spread " << spread(zeta_g, want_g);
  EXPECT_EQ(differing(first->gamma, gamma), 0U);
  EXPECT_EQ(differing(first->gamma2, gamma2), 0U);
  std::cout << std::format("[ zeta] {}: {} pairs, max |zeta . g| {:.3e}, max |gamma^zeta| "
                           "{:.3e}: two calls and the per-pair sums bitwise\n",
                           name, n_pairs, largest(zeta_g), largest(first->gamma));

  // zeta's nuclear term: one df_transition_gradient at the folded RDMs
  // against Sum zeta_sr times the per-pair calls.
  const auto nuclear = [&](std::span<const double> g1, std::span<const double> g2) {
    return fb::mcscf::df_transition_gradient(res, molecule, basis, classes, df.aux, h, df.b,
                                             df.l.data(), sol, ncore, ncas, g1, g2);
  };
  auto folded = nuclear(first->gamma, first->gamma2);
  ASSERT_TRUE(folded.has_value()) << folded.error().message;
  auto folded_again = nuclear(first->gamma, first->gamma2);
  ASSERT_TRUE(folded_again.has_value()) << folded_again.error().message;
  const auto fields =
      std::array{&fb::mcscf::GradientTerms::kinetic, &fb::mcscf::GradientTerms::potential,
                 &fb::mcscf::GradientTerms::overlap, &fb::mcscf::GradientTerms::two_electron,
                 &fb::mcscf::GradientTerms::total};
  std::vector<std::vector<double>> summed(fields.size(),
                                          std::vector<double>(((*folded).*fields[0]).size(), 0.0));
  for (std::size_t p = 0; p < n_pairs; ++p) {
    if (zeta[p] == 0.0) {
      continue;
    }
    auto per_pair = nuclear(pair_gamma[p], pair_gamma2[p]);
    ASSERT_TRUE(per_pair.has_value()) << per_pair.error().message;
    for (std::size_t f = 0; f < fields.size(); ++f) {
      const std::vector<double> &term = (*per_pair).*fields[f];
      for (std::size_t i = 0; i < term.size(); ++i) {
        summed[f][i] += zeta[p] * term[i];
      }
    }
  }
  constexpr std::array<std::string_view, 5> kNames{"kinetic", "potential", "overlap",
                                                   "two_electron", "total"};
  // Every term at the total row's largest element.
  const double scale = largest(folded->total);
  const double bar =
      round_off_bar(scale, name == "h2co_ccpvdz_sa2" ? kWorstSpreadSa2 : kWorstSpreadSa3);
  for (std::size_t f = 0; f < fields.size(); ++f) {
    const std::vector<double> &ours = (*folded).*fields[f];
    const double self = spread(ours, (*folded_again).*fields[f]);
    const double against = spread(ours, summed[f]);
    EXPECT_LE(self, bar) << kNames[f] << ": two calls";
    EXPECT_LE(against, bar) << kNames[f] << ": against the per-pair sum";
    std::cout << std::format("[ zeta] {} {}: max {:.3e} (scale {:.3e}), two calls {:.2e} "
                             "({:.1f} ulps), fold against the pairs {:.2e} ({:.1f} ulps), bar "
                             "{:.2e}\n",
                             name, kNames[f], largest(ours), scale, self,
                             self / (std::numeric_limits<double>::epsilon() * scale), against,
                             against / (std::numeric_limits<double>::epsilon() * scale), bar);
  }

  // A zeta of the wrong length.
  const std::vector<double> long_zeta(n_pairs + 1, 1.0);
  const auto refused =
      fb::mcscf::zeta_terms(res, molecule, h, df.b, df.naux, sol, ncore, ncas, run->ints->ppaa,
                            run->ints->papa, *run->rdm, long_zeta);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
  EXPECT_NE(refused.error().message.find("mcscf::zeta_terms"), std::string::npos);

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ zeta] {}: {:.2f} s in all\n", name, seconds.count());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfZetaTerms,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

TEST(McscfRootPairMultipliers, SplitsTauByWeight) {
  FBF_REQUIRE_GPU();
  const auto run = converge(reference_case("h2co_skew_ccpvdz_sa3"));
  ASSERT_NE(run, nullptr);
  fb::DeviceResources &res = run->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::ci::Roots &roots = run->sol->roots;
  const std::size_t nroots = roots.energies.size();
  ASSERT_EQ(nroots, 3U);
  const std::size_t dim = roots.vectors.num_elements() / nroots;
  const std::vector<double> c = download(res, roots.vectors.data(), dim * nroots);

  // A random y_bar, dim x nroots column-major.
  std::mt19937_64 engine{700};
  std::normal_distribution<double> normal;
  std::vector<double> y(dim * nroots);
  for (double &v : y) {
    v = normal(engine);
  }
  fb::Buffers::device<double> y_device(y.size(), res.device());
  ASSERT_EQ(wwr::wwrMemcpyAsync(y_device.data(), y.data(), y.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());

  // c_a . y_b on the host.
  const auto dot = [&](std::size_t a, std::size_t b) {
    double value = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      value += c[(a * dim) + i] * y[(b * dim) + i];
    }
    return value;
  };

  std::optional<std::vector<double>> first_tau;
  for (const auto &[weights, want_equal] :
       {std::pair{std::vector<double>{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, std::size_t{3}},
        std::pair{std::vector<double>{0.4, 0.4, 0.2}, std::size_t{1}},
        std::pair{std::vector<double>{0.5, 0.3, 0.2}, std::size_t{0}}}) {
    const std::string label =
        std::format("weights ({}, {}, {})", weights[0], weights[1], weights[2]);
    SCOPED_TRACE(label);
    auto split = fb::mcscf::root_pair_multipliers(res, roots, weights, y_device.data());
    ASSERT_TRUE(split.has_value()) << split.error().message;
    ASSERT_EQ(split->tau.size(), 3U);
    ASSERT_EQ(split->zeta.size(), 3U);
    ASSERT_EQ(split->theta_rhs.size(), 3U);
    const std::vector<bool> rotating = fb::mcscf::rotating_pairs(weights);
    std::size_t pair = 0;
    std::size_t equal = 0;
    double tau_error = 0.0;
    const double tau_bar = round_off_bar(largest(split->tau), kWorstTauSpread);
    for (std::size_t r = 0; r < nroots; ++r) {
      for (std::size_t s = r + 1; s < nroots; ++s, ++pair) {
        const double host = dot(s, r) - dot(r, s);
        const double tau = split->tau[pair];
        EXPECT_NEAR(tau, host, tau_bar) << "tau " << s << r;
        tau_error = std::max(tau_error, std::abs(tau - host));
        if (rotating[pair]) {
          EXPECT_EQ(bits(split->theta_rhs[pair]), bits(tau / (2.0 * (weights[r] - weights[s]))));
          EXPECT_EQ(split->zeta[pair], 0.0);
        } else {
          ++equal;
          EXPECT_EQ(bits(split->zeta[pair]), bits(-tau / (roots.energies[s] - roots.energies[r])));
          EXPECT_EQ(split->theta_rhs[pair], 0.0);
        }
      }
    }
    EXPECT_EQ(equal, want_equal);
    if (first_tau.has_value()) {
      EXPECT_EQ(differing(split->tau, *first_tau), 0U) << "tau changed with the weights";
    } else {
      first_tau = split->tau;
    }
    std::cout << std::format("[ zeta] {}: {} equal pairs, tau ({:.6e}, {:.6e}, {:.6e}) within "
                             "{:.2e} of the host's, bar {:.2e}\n",
                             label, equal, split->tau[0], split->tau[1], split->tau[2], tau_error,
                             tau_bar);
  }
  // Refusals: roots 0 and 1 under the floor apart (checked before the CI is
  // read, so a placeholder vector of one determinant per root stands in),
  // and weights of the wrong length.
  std::vector<double> near_energies = roots.energies;
  near_energies[1] = near_energies[0] + (0.5 * fb::mcscf::kRootGapFloor);
  const fb::ci::Roots near{.energies = near_energies,
                           .spin_squares = roots.spin_squares,
                           .vectors = fb::Buffers::device<double>(nroots, res.device()),
                           .davidson = roots.davidson};
  const std::vector<double> equal_weights(3, 1.0 / 3.0);
  const auto degenerate =
      fb::mcscf::root_pair_multipliers(res, near, equal_weights, y_device.data());
  ASSERT_FALSE(degenerate.has_value());
  EXPECT_EQ(degenerate.error().kind, fb::ErrorKind::Numerical) << degenerate.error().message;
  EXPECT_NE(degenerate.error().message.find("roots 0 and 1"), std::string::npos)
      << degenerate.error().message;
  const std::vector<double> two_weights{0.5, 0.5};
  const auto short_weights =
      fb::mcscf::root_pair_multipliers(res, roots, two_weights, y_device.data());
  ASSERT_FALSE(short_weights.has_value());
  EXPECT_EQ(short_weights.error().kind, fb::ErrorKind::InvalidConfig)
      << short_weights.error().message;
}

} // namespace
