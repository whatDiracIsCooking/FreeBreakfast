// mcscf::df_intersection on the device against the three calls it stands in
// for, on data/casscf_nac/'s cases (H2CO SA-2 and the skewed H2CO SA-3,
// CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT, equal weights): this tree's
// DF-SA-CASSCF converged as mcscf_nac_gpu.cpp converges it, then for every
// pair I < J
//
// - g_I against df_gradient(I)'s total, g_J against df_gradient(J)'s, and
//   h_IJ against df_nac(I, J)'s numerator.
//
// **The bar, and why it is 1e-9.** Every one of the three is a comparison
// between independently converged Z-vector solves: the separate calls each
// solve their own (zvector, to 1e-10 for a gradient and 1e-10 times the gap
// for df_nac), and df_intersection solves the three columns together
// (zvectors' Schur, to 1e-10 per column). So each field is solver-limited,
// and held at 1e-9 relative to the largest element of the two vectors
// compared, the ruling on #767 for such a check; no field here is a
// round-off-only path agreement, so none is at 1e-10.
//
// Then the refusals: a root against itself and a root past the solution's
// (InvalidConfig), and roots I and J of unequal weight (Unsupported, the
// solution's weighted gamma re-mixed to [0.6, 0.4] or [0.5, 0.3, 0.2] so the recovered weights
// differ; nothing is solved before the refusal).
//
// And, in the slow tier only, the timing against the three separate calls on
// the skewed SA-3's (0, 1): one warm-up of each, then the median of three.
//
// The setup is mcscf_nac_gpu.cpp's, so this executable links pipeline; mcscf
// does not. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// Independently converged Z-vector solves against each other (#767's ruling),
// relative to the largest element compared.
constexpr double kSolverTolerance = 1e-9;
// The CASSCF's convergence, as mcscf_nac_gpu.cpp's.
constexpr double kConvTolGrad = 1e-10;
// Repetitions of the timing, after one warm-up.
constexpr std::int32_t kTimingRepeats = 3;

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
  const fb::test::ScratchDir dir("mcscf_intersection");
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

// A reference case's converged DF-SA-CASSCF, kept together because the
// outputs refer into the calculation.
struct Converged {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::EriOutput> fitted;
  std::optional<fb::ActiveSpace> space;
  std::optional<fb::mcscf::Solution> sol;
  fb::DeviceResources &res() { return node.device(0); }
};

// @p entry converged at its own weights, or null after a failure is recorded.
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
  return run;
}

// The three calls df_intersection stands in for, and the one call, bound to
// a converged run.
struct Calls {
  Converged &run;
  const fb::Molecule &molecule() const { return run.calc->get<fb::MoleculeOutput>()->molecule; }
  const fb::BasisSet &basis() const { return run.calc->get<fb::BasisSetOutput>()->basis; }
  const auto &classes() const { return run.calc->get<fb::PrimitivePairClassesOutput>()->classes; }
  const fb::SphericalMatrixAO &h() const { return run.calc->get<fb::CoreHamiltonianOutput>()->h; }
  const fb::DensityFitted &df() const { return std::get<fb::DensityFitted>(run.fitted->eri); }

  fb::Result<fb::mcscf::Intersection> intersection(std::int32_t i, std::int32_t j) const {
    return fb::mcscf::df_intersection(run.res(), molecule(), basis(), classes(), df().aux, h(),
                                      df().b, df().l.data(), *run.sol, run.space->ncore,
                                      run.space->ncas, i, j);
  }
  fb::Result<fb::mcscf::GradientTerms> gradient(std::int32_t root) const {
    return fb::mcscf::df_gradient(run.res(), molecule(), basis(), classes(), df().aux, h(), df().b,
                                  df().l.data(), *run.sol, run.space->ncore, run.space->ncas, root);
  }
  fb::Result<fb::mcscf::Nac> nac(std::int32_t i, std::int32_t j) const {
    return fb::mcscf::df_nac(run.res(), molecule(), basis(), classes(), df().aux, h(), df().b,
                             df().l.data(), *run.sol, run.space->ncore, run.space->ncas, i, j);
  }
};

// max_i |ours_i - want_i| / max(max|ours|, max|want|), each element also
// EXPECTed within @p tolerance of that scale.
double compare(std::span<const double> ours, std::span<const double> want, double tolerance,
               std::string_view what) {
  EXPECT_EQ(ours.size(), want.size()) << what;
  double scale = 0.0;
  for (const std::span<const double> v : {ours, want}) {
    for (const double x : v) {
      scale = std::max(scale, std::abs(x));
    }
  }
  double error = 0.0;
  for (std::size_t i = 0; i < std::min(ours.size(), want.size()); ++i) {
    EXPECT_NEAR(ours[i], want[i], tolerance * scale) << what << " " << i;
    error = std::max(error, std::abs(ours[i] - want[i]));
  }
  return scale > 0.0 ? error / scale : error;
}

class McscfIntersection : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfIntersection, MatchesSeparateCalls) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto nroots = static_cast<std::int32_t>(entry.at("weights").size());
  const auto start_time = std::chrono::steady_clock::now();

  const auto run = converge(entry);
  ASSERT_NE(run, nullptr);
  const fb::ScopedDeviceIndex on{run->res().dev_idx()};
  const Calls calls{*run};

  for (std::int32_t i = 0; i < nroots; ++i) {
    for (std::int32_t j = i + 1; j < nroots; ++j) {
      const std::string label = std::format("{} ({}, {})", name, i, j);
      SCOPED_TRACE(label);
      auto one = calls.intersection(i, j);
      ASSERT_TRUE(one.has_value()) << one.error().message;
      EXPECT_LE(one->residual, 1e-10);
      auto g_i = calls.gradient(i);
      ASSERT_TRUE(g_i.has_value()) << g_i.error().message;
      auto g_j = calls.gradient(j);
      ASSERT_TRUE(g_j.has_value()) << g_j.error().message;
      auto d = calls.nac(i, j);
      ASSERT_TRUE(d.has_value()) << d.error().message;
      const double e_i = compare(one->g_i, g_i->total, kSolverTolerance, "g_I");
      const double e_j = compare(one->g_j, g_j->total, kSolverTolerance, "g_J");
      const double e_h = compare(one->h_ij, d->numerator, kSolverTolerance, "h_IJ");
      std::cout << std::format("[ intersection] {}: g_I {:.2e}, g_J {:.2e}, h_IJ {:.2e} "
                               "(relative), z-vectors {} products to {:.2e}\n",
                               label, e_i, e_j, e_h, one->iterations, one->residual);
    }
  }

  // Refusals: a root against itself and a root past the solution's.
  for (const auto &[i, j] : {std::pair{0, 0}, std::pair{0, nroots}}) {
    const auto refused = calls.intersection(i, j);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
    EXPECT_NE(refused.error().message.find("mcscf::df_intersection"), std::string::npos);
  }
  // Unequal weights on I and J: the weighted gamma re-mixed, the weights
  // being recovered from it, to [0.6, 0.4] (or [0.5, 0.3, 0.2]), then put
  // back.
  {
    fb::mcscf::Solution &sol = *run->sol;
    const std::vector<double> kept = sol.gamma;
    const std::size_t m = kept.size();
    const std::vector<double> mix =
        nroots == 2 ? std::vector<double>{0.6, 0.4} : std::vector<double>{0.5, 0.3, 0.2};
    ASSERT_EQ(mix.size(), static_cast<std::size_t>(nroots));
    std::ranges::fill(sol.gamma, 0.0);
    for (std::size_t r = 0; r < mix.size(); ++r) {
      for (std::size_t k = 0; k < m; ++k) {
        sol.gamma[k] += mix[r] * sol.root_gammas[(r * m) + k];
      }
    }
    const auto refused = calls.intersection(0, 1);
    sol.gamma = kept;
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::Unsupported) << refused.error().message;
    EXPECT_NE(refused.error().message.find("mcscf::df_intersection"), std::string::npos);
  }

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ intersection] {}: {:.2f} s in all\n", name, seconds.count());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfIntersection,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

// The wall clock of df_intersection against df_gradient(I) + df_gradient(J) +
// df_nac(I, J), on each case's (0, 1): one warm-up of each, then the median
// of kTimingRepeats. Measurement, not a gate: nothing is asserted on it.
TEST(McscfIntersectionTiming, AgainstThreeCalls) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  for (const std::string_view name : {"h2co_ccpvdz_sa2", "h2co_skew_ccpvdz_sa3"}) {
    const auto run = converge(reference_case(name));
    ASSERT_NE(run, nullptr);
    const fb::ScopedDeviceIndex on{run->res().dev_idx()};
    const Calls calls{*run};
    const auto one = [&] { return calls.intersection(0, 1).has_value(); };
    const auto three = [&] {
      return calls.gradient(0).has_value() && calls.gradient(1).has_value() &&
             calls.nac(0, 1).has_value();
    };
    const auto median = [](const auto &call) {
      EXPECT_TRUE(call());
      std::vector<double> times;
      for (std::int32_t r = 0; r < kTimingRepeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        EXPECT_TRUE(call());
        times.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      }
      std::ranges::sort(times);
      return times[times.size() / 2];
    };
    const double t_one = median(one);
    const double t_three = median(three);
    std::cout << std::format("[ intersection] {} (0, 1): one call {:.3f} s, three calls {:.3f} s "
                             "({:.2f}x)\n",
                             name, t_one, t_three, t_three / t_one);
  }
}

} // namespace
