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
// differ; nothing is solved before the refusal); and an unconverged Z-vector
// (the block solve capped at one product) is an error, never multipliers.
//
// At the seam, on data/casscf_meci/'s ethylene SA2-CASSCF(2,2) (#786): at the
// MECI against the reference's projected force and in-plane gap slopes
// (McscfIntersectionSeam), and at the MECI and two points about 2e-8 Eh off
// it, a carried-orbital evaluation against a fresh one: the mean gradient
// and the branching plane, which are what a search reads and what nothing
// there divides by the gap (McscfIntersectionNearSeam). Each test's comment
// has its bars.
//
// In ethylene's C3v-ethylidene region at DF-SA3-CASSCF(2,2) (#856), where
// weights recovered from the roots' 1-RDMs lose their conditioning: at an
// exactly C3v geometry every root's gradient against a finite difference
// (McscfC3v), and near the C3v MECI the intersection, which the solution's
// carried weights let through (each test's comment has its bars).
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

// @p carry's orbitals (AO by MO, column-major, from another geometry of the
// same molecule) Loewdin-orthonormalized in @p s, C (C^T S C)^-1/2, the
// orthonormal set nearest them: how [optimize] carries a CASSCF's orbitals
// to the next geometry.
std::optional<fb::SphericalMatrixAO> carried_orbitals(fb::DeviceResources &res,
                                                      const fb::BasisSet &basis,
                                                      const fb::SphericalMatrixAO &s,
                                                      const std::vector<double> &carry) {
  if (carry.size() != s.num_elements()) {
    ADD_FAILURE() << "carried " << carry.size() << " coefficients into " << s.num_elements();
    return std::nullopt;
  }
  const auto n = static_cast<std::int32_t>(s.n());
  const std::size_t n2 = s.num_elements();
  fb::SphericalMatrixAO previous(res, basis);
  fb::SphericalMatrixAO out(res, basis);
  fb::Buffers::device<double> m(n2, res.device());
  fb::Buffers::device<double> m_inv_sqrt(n2, res.device());
  EXPECT_TRUE(fb::gpu_status(wwr::wwrMemcpyAsync(previous.data(), carry.data(), n2 * sizeof(double),
                                                 wwr::wwrMemcpyHostToDevice, res.stream()))
                  .has_value());
  {
    const auto [work] = res.device_regions<double>(n2);
    EXPECT_TRUE(
        fb::gpu_status(fb::gemmwich(res.blas(), n, previous.data(), s.data(), work, m.data()))
            .has_value());
  }
  if (const auto lowdin = fb::orthogonalize<double>(res, n, m.data(), m_inv_sqrt.data());
      !lowdin.has_value()) {
    ADD_FAILURE() << lowdin.error().message;
    return std::nullopt;
  }
  EXPECT_TRUE(fb::gpu_status(fb::gemm(res.blas(), n, &fb::kOne<double>, previous.data(),
                                      m_inv_sqrt.data(), &fb::kZero<double>, out.data()))
                  .has_value());
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

// @p entry converged at its own weights, or null after a failure is recorded:
// from its RHF's frontier orbitals (the RHF converged by @p scf_step), or with
// @p carry, from those orbitals of another geometry (carried_orbitals) over
// @p entry's ncore.
std::unique_ptr<Converged> converge(const fb::json &entry,
                                    const std::vector<double> *carry = nullptr,
                                    const fb::ScfStep &scf_step = fb::test::kExactScf) {
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
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = entry.at("aux_basis").at("name").get<std::string>()}
                    .run(view, res);
  if (!fitted.has_value()) {
    ADD_FAILURE() << fitted.error().message;
    return nullptr;
  }
  run->fitted.emplace(std::move(*fitted));
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  if (carry != nullptr) {
    std::optional<fb::SphericalMatrixAO> c = carried_orbitals(res, basis, overlap->s, *carry);
    if (!c.has_value()) {
      return nullptr;
    }
    run->space.emplace(fb::ActiveSpace{.ncore = entry.at("ncore").get<std::int32_t>(),
                                       .ncas = entry.at("ncas").get<std::int32_t>(),
                                       .nelecas = entry.at("nelecas").get<std::int32_t>(),
                                       .r = {},
                                       .c = std::move(*c)});
  } else {
    auto eri = fb::EriStep{}.run(view, res);
    if (!eri.has_value()) {
      ADD_FAILURE() << eri.error().message;
      return nullptr;
    }
    auto scf = scf_step.run(fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri,
                                                       *overlap, *x, *h, *guess},
                            res);
    if (!scf.has_value()) {
      ADD_FAILURE() << scf.error().message;
      return nullptr;
    }
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
  }

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

  fb::Result<fb::mcscf::Intersection>
  intersection(std::int32_t i, std::int32_t j,
               const fb::mcscf::IntersectionOptions &options = {}) const {
    return fb::mcscf::df_intersection(run.res(), molecule(), basis(), classes(), df().aux, h(),
                                      df().b, df().l.data(), *run.sol, run.space->ncore,
                                      run.space->ncas, i, j, options);
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
  // Under slow_test.h (#977): the skew SA3 formaldehyde, 3.6 s on the RTX 3080.
  if (std::string_view{GetParam()} == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
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

  // An unconverged Z-vector is an error, never multipliers: the block solve
  // capped at one product.
  {
    const auto refused = calls.intersection(0, 1, {.max_iterations = 1});
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::MaxIterationsExceeded)
        << refused.error().message;
  }

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ intersection] {}: {:.2f} s in all\n", name, seconds.count());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfIntersection,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

// IntersectionOptions::tolerance below the default (#858): asked for 1e-12,
// 100x under it, the Z-vector is solved to it and every column's true
// residual (one more product) meets it, the CI response inside H~ at its
// 1e-10 floor (response_tolerance_for), and the three vectors agree with
// the default solve's to that solve's 1e-10: kSolverTolerance, relative.
// H2CO's z is small, so its floor under the old fixed 1e-8 response was
// itself below 1e-12: this is the option's plumbing, not the floor's
// regression. That is PSB3's, in nevpt2_qd_intersection_gpu.cpp
// (Psb3TrueResidual, slow tier), the same mechanism one layer up.
constexpr double kTightZvector = 1e-12;

TEST(McscfIntersectionTight, TrueResidualFollowsTolerance) {
  FBF_REQUIRE_GPU();
  const auto run = converge(reference_case("h2co_skew_ccpvdz_sa3"));
  ASSERT_NE(run, nullptr);
  const fb::ScopedDeviceIndex on{run->res().dev_idx()};
  const Calls calls{*run};
  auto loose = calls.intersection(0, 1);
  ASSERT_TRUE(loose.has_value()) << loose.error().message;
  auto tight = calls.intersection(0, 1, {.tolerance = kTightZvector});
  ASSERT_TRUE(tight.has_value()) << tight.error().message;
  EXPECT_LE(tight->residual, kTightZvector);
  const double e_i = compare(tight->g_i, loose->g_i, kSolverTolerance, "g_I");
  const double e_j = compare(tight->g_j, loose->g_j, kSolverTolerance, "g_J");
  const double e_h = compare(tight->h_ij, loose->h_ij, kSolverTolerance, "h_IJ");
  std::cout << std::format("[ intersection] tight: {} products to {:.2e} (default {} to {:.2e}); "
                           "g_I {:.2e}, g_J {:.2e}, h_IJ {:.2e} relative\n",
                           tight->iterations, tight->residual, loose->iterations, loose->residual,
                           e_i, e_j, e_h);
}

// data/casscf_meci/'s document: ethylene's DF-SA2-CASSCF(2,2) MECI.
const fb::json &meci_reference() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_meci" /
                              "casscf_meci.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

// A flat (n_atom, 3) row-major vector from rows of three.
std::vector<double> flat_rows(const fb::json &rows) {
  std::vector<double> out;
  for (const fb::json &row : rows) {
    for (const double v : row.get<std::vector<double>>()) {
      out.push_back(v);
    }
  }
  return out;
}

double dot(std::span<const double> a, std::span<const double> b) {
  double sum = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

// @p vectors orthonormalized in order by Gram-Schmidt, each pass taken twice
// (enough for orthogonality to rounding).
std::vector<std::vector<double>> orthonormal(std::vector<std::vector<double>> vectors) {
  std::vector<std::vector<double>> basis;
  for (std::vector<double> &v : vectors) {
    for (std::int32_t pass = 0; pass < 2; ++pass) {
      for (const std::vector<double> &u : basis) {
        const double c = dot(u, v);
        for (std::size_t k = 0; k < v.size(); ++k) {
          v[k] -= c * u[k];
        }
      }
    }
    const double norm = std::sqrt(dot(v, v));
    EXPECT_GT(norm, 1e-8);
    for (double &x : v) {
      x /= norm;
    }
    basis.push_back(std::move(v));
  }
  return basis;
}

// The pair's quantities an MECI search reads, from one Intersection: the mean
// gradient, the branching plane's orthonormal basis (from x = g_J - g_I, then
// h_IJ) and the gap's two in-plane slopes, twice the singular values of
// [x/2; h_IJ], greatest first.
struct Seam {
  std::vector<double> gbar;
  std::vector<std::vector<double>> plane;
  std::array<double, 2> slopes;
};

Seam seam_of(std::span<const double> g_i, std::span<const double> g_j,
             std::span<const double> h_ij) {
  Seam out;
  std::vector<double> x(g_i.size());
  std::vector<double> half(g_i.size());
  out.gbar.resize(g_i.size());
  for (std::size_t k = 0; k < g_i.size(); ++k) {
    out.gbar[k] = 0.5 * (g_i[k] + g_j[k]);
    x[k] = g_j[k] - g_i[k];
    half[k] = 0.5 * x[k];
  }
  out.plane = orthonormal({x, std::vector<double>(h_ij.begin(), h_ij.end())});
  const double a = dot(half, half);
  const double b = dot(half, h_ij);
  const double d = dot(h_ij, h_ij);
  const double root = std::sqrt(std::max((((a - d) * (a - d)) / 4.0) + (b * b), 0.0));
  out.slopes = {2.0 * std::sqrt(((a + d) / 2.0) + root),
                2.0 * std::sqrt(std::max(((a + d) / 2.0) - root, 0.0))};
  return out;
}

// The largest principal angle between two planes, each an orthonormal pair:
// asin of the largest singular value of Q2 - Q1 (Q1^T Q2).
double plane_angle(const std::vector<std::vector<double>> &q1,
                   const std::vector<std::vector<double>> &q2) {
  std::array<std::vector<double>, 2> r;
  for (std::size_t c = 0; c < 2; ++c) {
    r[c] = q2[c];
    for (const std::vector<double> &u : q1) {
      const double overlap = dot(u, q2[c]);
      for (std::size_t k = 0; k < r[c].size(); ++k) {
        r[c][k] -= overlap * u[k];
      }
    }
  }
  const double a = dot(r[0], r[0]);
  const double b = dot(r[0], r[1]);
  const double d = dot(r[1], r[1]);
  const double top =
      ((a + d) / 2.0) + std::sqrt(std::max((((a - d) * (a - d)) / 4.0) + (b * b), 0.0));
  return std::asin(std::min(std::sqrt(top), 1.0));
}

// -P gbar: gbar off the three translations, the three rotations about the
// centroid of @p coords (bohr, flat) and the plane, as #771's projector and
// tools/emit/meci_reference.py remove them.
std::vector<double> projected_force(std::span<const double> coords, const Seam &seam) {
  const std::size_t n = coords.size() / 3;
  std::array<double, 3> centroid{};
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      centroid[k] += coords[(3 * a) + k] / static_cast<double>(n);
    }
  }
  std::vector<std::vector<double>> vectors;
  for (std::size_t j = 0; j < 3; ++j) {
    std::vector<double> t(3 * n, 0.0);
    for (std::size_t a = 0; a < n; ++a) {
      t[(3 * a) + j] = 1.0;
    }
    vectors.push_back(std::move(t));
  }
  for (std::size_t j = 0; j < 3; ++j) {
    std::array<double, 3> axis{};
    axis[j] = 1.0;
    std::vector<double> rot(3 * n);
    for (std::size_t a = 0; a < n; ++a) {
      const double r0 = coords[(3 * a) + 0] - centroid[0];
      const double r1 = coords[(3 * a) + 1] - centroid[1];
      const double r2 = coords[(3 * a) + 2] - centroid[2];
      rot[(3 * a) + 0] = (r1 * axis[2]) - (r2 * axis[1]);
      rot[(3 * a) + 1] = (r2 * axis[0]) - (r0 * axis[2]);
      rot[(3 * a) + 2] = (r0 * axis[1]) - (r1 * axis[0]);
    }
    vectors.push_back(std::move(rot));
  }
  vectors.insert(vectors.end(), seam.plane.begin(), seam.plane.end());
  std::vector<double> force(seam.gbar.size());
  std::ranges::transform(seam.gbar, force.begin(), [](double v) { return -v; });
  for (const std::vector<double> &u : orthonormal(std::move(vectors))) {
    const double c = dot(u, seam.gbar);
    for (std::size_t k = 0; k < force.size(); ++k) {
      force[k] += c * u[k];
    }
  }
  return force;
}

// The reference's case at @p coords (bohr, flat), as reference_case's
// entries are shaped.
fb::json meci_entry(std::span<const double> coords) {
  const fb::json &doc = meci_reference();
  fb::json entry;
  for (const char *key : {"name", "basis", "charge", "ncore", "ncas", "nelecas", "weights"}) {
    entry[key] = doc.at(key);
  }
  entry["aux_basis"] = doc.at("aux_basis");
  fb::json atoms = doc.at("meci").at("atoms");
  for (std::size_t a = 0; a < atoms.size(); ++a) {
    atoms[a]["coord"] =
        std::vector<double>{coords[3 * a], coords[(3 * a) + 1], coords[(3 * a) + 2]};
  }
  entry["atoms"] = std::move(atoms);
  return entry;
}

// The MECI's coordinates (bohr, flat) and the reference's quantities there.
struct MeciReference {
  std::vector<double> coords;
  std::vector<double> g0;
  std::vector<double> g1;
  std::vector<double> h;
  Seam seam;
};

const MeciReference &meci_point() {
  static const MeciReference kMeci = [] {
    const fb::json &meci = meci_reference().at("meci");
    MeciReference out;
    for (const fb::json &atom : meci.at("atoms")) {
      for (const double v : atom.at("coord").get<std::vector<double>>()) {
        out.coords.push_back(v);
      }
    }
    out.g0 = flat_rows(meci.at("gradients")[0]);
    out.g1 = flat_rows(meci.at("gradients")[1]);
    out.h = flat_rows(meci.at("h"));
    out.seam = seam_of(out.g0, out.g1, out.h);
    return out;
  }();
  return kMeci;
}

// Each column's true residual after the block solve.
constexpr double kSeamResidual = 1e-10;

// At data/casscf_meci/'s MECI (gap 3e-13 in the reference), one fresh-orbital
// evaluation (from the point's own RHF) against the reference, each check
// once:
// - the Z-vector converges, every column's true residual at most 1e-10;
// - -P gbar within 1e-6 Eh/bohr of its projected force, element by element,
//   P projecting off the translations, the rotations and our plane (the
//   issue's bar);
// - the gap's two in-plane slopes, twice the singular values of
//   [x/2; h_IJ], within 2 sqrt(6 n_atom) 1e-8 times the largest |element| of
//   the reference's g_0, g_1 and h_01 of the slopes recomputed from those
//   vectors (0.2290 and 0.1413 to the README's four places). That is the
//   tree's bar against PySCF's gradients and couplings (mcscf_gradient_gpu
//   and mcscf_nac_gpu, 1e-8 of the largest element) carried through Weyl's
//   inequality: a singular value moves by at most the spectral norm of the
//   matrix's change, at most its Frobenius norm, sqrt(2 x 3 n_atom) times its
//   largest element's; the slopes are twice the singular values, and a turn
//   of (x/2, h) within the plane changes neither.
TEST(McscfIntersectionSeam, MeciAgainstReference) {
  FBF_REQUIRE_GPU();
  constexpr double kForce = 1e-6;
  constexpr double kPyscfElement = 1e-8;
  const MeciReference &ref = meci_point();
  const auto run = converge(meci_entry(ref.coords));
  ASSERT_NE(run, nullptr);
  const fb::ScopedDeviceIndex on{run->res().dev_idx()};
  const Calls calls{*run};
  auto one = calls.intersection(0, 1);
  ASSERT_TRUE(one.has_value()) << one.error().message;
  EXPECT_LE(one->residual, kSeamResidual);
  const Seam ours = seam_of(one->g_i, one->g_j, one->h_ij);

  const std::vector<double> force = projected_force(ref.coords, ours);
  const std::vector<double> want = flat_rows(meci_reference().at("meci").at("projected_force"));
  double worst = 0.0;
  for (std::size_t k = 0; k < force.size(); ++k) {
    EXPECT_NEAR(force[k], want[k], kForce) << "projected force " << k;
    worst = std::max(worst, std::abs(force[k] - want[k]));
  }
  double scale = 0.0;
  for (const std::vector<double> *v : {&ref.g0, &ref.g1, &ref.h}) {
    for (const double x : *v) {
      scale = std::max(scale, std::abs(x));
    }
  }
  const double slope_bar =
      2.0 * std::sqrt(2.0 * static_cast<double>(force.size())) * kPyscfElement * scale;
  for (std::size_t k = 0; k < 2; ++k) {
    EXPECT_NEAR(ours.slopes[k], ref.seam.slopes[k], slope_bar) << "slope " << k;
  }
  const std::vector<double> &e = run->sol->roots.energies;
  std::cout << std::format("[ intersection] meci: gap {:.3e}, z-vectors {} products to {:.2e}; "
                           "against the reference: projected force {:.2e} (bar {:.0e}), slopes "
                           "{:.10f} {:.10f} against {:.10f} {:.10f} (differences {:.1e} {:.1e}, "
                           "bar {:.2e}), plane {:.2e} rad\n",
                           e[1] - e[0], one->iterations, one->residual, worst, kForce,
                           ours.slopes[0], ours.slopes[1], ref.seam.slopes[0], ref.seam.slopes[1],
                           std::abs(ours.slopes[0] - ref.seam.slopes[0]),
                           std::abs(ours.slopes[1] - ref.seam.slopes[1]), slope_bar,
                           plane_angle(ours.plane, ref.seam.plane));
}

// The carried orbitals' source, once per process: the converged orbitals at
// the reference's last iterate but one, 2.4e-3 bohr from the MECI (gap 1e-4
// Eh), on the host. Empty after a failure is recorded.
const std::vector<double> &carried_source() {
  static const std::vector<double> kCarry = [] {
    const fb::json &path = meci_reference().at("path");
    const auto source = converge(meci_entry(flat_rows(path[path.size() - 2].at("coords"))));
    std::vector<double> carry;
    if (source == nullptr) {
      return carry;
    }
    const fb::ScopedDeviceIndex on{source->res().dev_idx()};
    const fb::SphericalMatrixAO &c = source->sol->c;
    carry.resize(c.num_elements());
    EXPECT_TRUE(
        fb::gpu_status(wwr::wwrMemcpyAsync(carry.data(), c.data(), carry.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, source->res().stream()))
            .has_value());
    EXPECT_TRUE(source->res().sync().has_value());
    return carry;
  }();
  return kCarry;
}

// A point at or near the seam: the MECI moved by kSeamStep bohr along a unit
// direction of the reference's plane (x or h_01), or not at all.
enum class SeamPoint : std::uint8_t { Meci, AlongX, AlongH };
constexpr double kSeamStep = 1e-7;

// One fresh-orbital and one carried-orbital evaluation (carried_source's
// orbitals, Loewdin-carried, as [optimize] carries them) at a point at or
// near the seam: the MECI, and the MECI moved 1e-7 bohr along the
// reference's x and h_01, gaps of about 2e-8 Eh, where the mixing of the two
// roots in the CI response would weigh 1/gap. Each check once:
// - the Z-vector converges, every column's true residual at most 1e-10;
// - carried against fresh, two independently converged CASSCFs and Z-vector
//   solves: gbar within 1e-9 of the larger element (#767's ruling), and the
//   planes' largest principal angle below 1e-6 rad (the issue's).
// Along x stands for the three in the fast set: it is where #773's solve
// broke (the MECI's own gap is under the RelaxedHessian's 1e-8 floor).
class McscfIntersectionNearSeam : public ::testing::TestWithParam<SeamPoint> {};

TEST_P(McscfIntersectionNearSeam, CarriedAgreesWithFresh) {
  FBF_REQUIRE_GPU();
  if (GetParam() != SeamPoint::AlongX) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  constexpr double kPlaneAngle = 1e-6;
  const MeciReference &ref = meci_point();
  std::vector<double> coords = ref.coords;
  if (GetParam() != SeamPoint::Meci) {
    const std::vector<double> direction =
        GetParam() == SeamPoint::AlongX ? ref.seam.plane[0] : orthonormal({ref.h})[0];
    for (std::size_t k = 0; k < coords.size(); ++k) {
      coords[k] += kSeamStep * direction[k];
    }
  }
  const std::vector<double> &carry = carried_source();
  ASSERT_FALSE(carry.empty());

  std::array<Seam, 2> seams;
  for (std::size_t which = 0; which < 2; ++which) {
    const auto run = converge(meci_entry(coords), which == 0 ? nullptr : &carry);
    ASSERT_NE(run, nullptr);
    const fb::ScopedDeviceIndex on{run->res().dev_idx()};
    const Calls calls{*run};
    auto one = calls.intersection(0, 1);
    ASSERT_TRUE(one.has_value()) << one.error().message;
    EXPECT_LE(one->residual, kSeamResidual);
    seams[which] = seam_of(one->g_i, one->g_j, one->h_ij);
    const std::vector<double> &e = run->sol->roots.energies;
    std::cout << std::format("[ intersection] {}: gap {:.3e}, z-vectors {} products to {:.2e}\n",
                             which == 0 ? "fresh" : "carried", e[1] - e[0], one->iterations,
                             one->residual);
  }
  const double gbar = compare(seams[1].gbar, seams[0].gbar, kSolverTolerance, "gbar");
  const double angle = plane_angle(seams[0].plane, seams[1].plane);
  EXPECT_LT(angle, kPlaneAngle);
  std::cout << std::format("[ intersection] carried vs fresh: gbar {:.2e} (relative), plane "
                           "{:.2e} rad\n",
                           gbar, angle);
}

INSTANTIATE_TEST_SUITE_P(Ethylene, McscfIntersectionNearSeam,
                         ::testing::Values(SeamPoint::Meci, SeamPoint::AlongX, SeamPoint::AlongH),
                         [](const auto &info) {
                           switch (info.param) {
                           case SeamPoint::Meci:
                             return std::string{"Meci"};
                           case SeamPoint::AlongX:
                             return std::string{"AlongX"};
                           case SeamPoint::AlongH:
                             return std::string{"AlongH"};
                           }
                           return std::string{"Unknown"};
                         });

// #856: ethylene's C3v-ethylidene region at DF-SA3-CASSCF(2,2) / cc-pVDZ /
// cc-pVDZ-JKFIT, equal weights. There the two active orbitals fall into
// different irreducible representations, every root's 1-RDM is nearly
// diag(n, 2 - n), and the three are nearly collinear: weights recovered from
// them by least squares lose their conditioning (an exact C3v start was
// refused as "linearly dependent", near-C3v walks as unequal weights on the
// pair). The solution carries its weights instead; each test prints, and
// does not hold, what the recovery would have given.

// The case at @p angstrom (symbols C, C, H, H, H, H, flat), as
// reference_case's entries are shaped.
fb::json c3v_entry(std::string_view name, std::span<const double> angstrom) {
  fb::json entry;
  entry["name"] = name;
  entry["basis"] = "cc-pvdz";
  entry["aux_basis"]["name"] = "cc-pvdz-jkfit";
  entry["charge"] = 0;
  entry["ncore"] = 7;
  entry["ncas"] = 2;
  entry["nelecas"] = 2;
  entry["weights"] = std::vector<double>(3, 1.0 / 3.0);
  fb::json atoms = fb::json::array();
  for (std::size_t a = 0; a < angstrom.size() / 3; ++a) {
    atoms.push_back(
        {{"symbol", a < 2 ? "C" : "H"},
         {"coord", std::vector<double>{angstrom[3 * a] * fb::kBohrPerAngstrom<double>,
                                       angstrom[(3 * a) + 1] * fb::kBohrPerAngstrom<double>,
                                       angstrom[(3 * a) + 2] * fb::kBohrPerAngstrom<double>}}});
  }
  entry["atoms"] = std::move(atoms);
  return entry;
}

// The RHF that starts the CASSCF here. kExactScf's 1e-10 commutator is out
// of reach at these geometries: its DIIS stalls near |[F,D]| = 1.7e-6 (as the
// pipeline's own RHF did at the walk's MECI, #856's PR). Only its frontier
// orbitals are read, as the CASSCF's start, which is then converged to its
// own 1e-10, so the RHF's residual reaches nothing compared here.
const fb::ScfStep kC3vScf{.energy_tolerance = 1e-8,
                          .commutator_tolerance = 1e-5,
                          .fock_rebuild_interval = 1,
                          .tau_j = 0.0};

// What sa_gradient_weights recovers from @p sol's RDMs with its carried
// weights set aside: the spread of the weights, or the refusal.
std::string recovered_weights(fb::mcscf::Solution &sol) {
  std::vector<double> kept = std::move(sol.weights);
  sol.weights.clear();
  const auto recovered = fb::mcscf::sa_gradient_weights(sol, 2);
  sol.weights = std::move(kept);
  if (!recovered.has_value()) {
    return recovered.error().message;
  }
  const auto [low, high] = std::ranges::minmax(*recovered);
  return std::format("weights {:.17g} to {:.17g}, spread {:.2e} relative", low, high,
                     (high - low) / high);
}

// Exactly C3v: CH3-C-H, C-C 1.45 A, C-H 1.09 A at H-C-C 109.5 degrees on the
// methyl (one H in the xz mirror, the other two its images) and the carbene
// H on the C-C axis at 1.07 A, #846's hand-built start without its tilt.
std::vector<double> exact_c3v(const double cc) {
  constexpr double kCh = 1.09;
  const double theta = (180.0 - 70.5) * std::numbers::pi / 180.0;
  const double rho = kCh * std::sin(theta);
  const double z = kCh * std::cos(theta);
  const double y = rho * std::numbers::sqrt3 / 2.0;
  return {0.0,        0.0, 0.0, 0.0,        0.0, cc, rho, 0.0, z,
          -rho / 2.0, y,   z,   -rho / 2.0, -y,  z,  0.0, 0.0, cc + 1.07};
}

// At the exact C3v geometry the solution carries its weights exactly, and
// every root's gradient is taken: before #856 each was refused, the roots'
// 1-RDMs "linearly dependent" (what the recovery still prints). Roots 0 and
// 1 are degenerate there, an E pair, and stay so along the stretch below.
// Checked once each:
// - sa_gradient_weights returns the carried weights, bitwise;
// - along the C3v-preserving stretch (C2 and the carbene H moved together
//   along z by q), dE_r/dq of every root against the Richardson-extrapolated
//   central difference (4 D(h) - D(2h)) / 3 of re-converged solutions at
//   h = 1e-3 bohr. Bar 1e-8 Eh/bohr, by analysis: the extrapolation leaves
//   O(h^4) E^(5) / 30, about 1e-13; each energy is converged to |g| below
//   1e-10, so off by O(|g|^2), and the CI to its own residual, so the
//   energies carry about 1e-12 of noise, (4 + 1) / 3 x 1e-12 / (2h) about
//   1e-9 in the difference; and the gradient's Z-vector is solved to 1e-10.
TEST(McscfC3v, ExactGradientAgainstFiniteDifference) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr double kCc = 1.45;
  constexpr double kStep = 1e-3;
  constexpr double kFdTolerance = 1e-8;
  const auto run = converge(c3v_entry("ethylene_c3v", exact_c3v(kCc)), nullptr, kC3vScf);
  ASSERT_NE(run, nullptr);
  const fb::ScopedDeviceIndex on{run->res().dev_idx()};
  const Calls calls{*run};
  fb::mcscf::Solution &sol = *run->sol;
  const std::vector<double> weights(3, 1.0 / 3.0);
  EXPECT_EQ(sol.weights, weights);
  const auto taken = fb::mcscf::sa_gradient_weights(sol, 2);
  ASSERT_TRUE(taken.has_value()) << taken.error().message;
  EXPECT_EQ(*taken, weights);
  std::cout << std::format("[ c3v      ] exact: recovered from the RDMs, {}\n",
                           recovered_weights(sol));

  // dE_r/dq of each root: C2's and the carbene H's z.
  std::array<double, 3> analytic{};
  for (std::int32_t root = 0; root < 3; ++root) {
    const auto g = calls.gradient(root);
    ASSERT_TRUE(g.has_value()) << g.error().message;
    analytic[root] = g->total[(3 * 1) + 2] + g->total[(3 * 5) + 2];
  }

  // E_r at q = k h (bohr), k in {-2, -1, 1, 2}.
  std::map<std::int32_t, std::vector<double>> energies;
  for (const std::int32_t k : {-2, -1, 1, 2}) {
    std::vector<double> moved = exact_c3v(kCc);
    const double dz = k * kStep * fb::kAngstromPerBohr<double>;
    moved[(3 * 1) + 2] += dz;
    moved[(3 * 5) + 2] += dz;
    const auto displaced = converge(c3v_entry("ethylene_c3v", moved), nullptr, kC3vScf);
    ASSERT_NE(displaced, nullptr);
    energies[k] = displaced->sol->roots.energies;
  }
  for (std::size_t root = 0; root < 3; ++root) {
    const double d1 = (energies[1][root] - energies[-1][root]) / (2.0 * kStep);
    const double d2 = (energies[2][root] - energies[-2][root]) / (4.0 * kStep);
    const double fd = ((4.0 * d1) - d2) / 3.0;
    EXPECT_NEAR(analytic[root], fd, kFdTolerance) << "root " << root;
    std::cout << std::format("[ c3v      ] exact: root {} E {:.12f} dE/dq {:.12f}, central "
                             "{:.12f} (h) {:.12f} (2h), extrapolated {:.12f}, off {:.2e}\n",
                             root, sol.roots.energies[root], analytic[root], d1, d2, fd,
                             std::abs(analytic[root] - fd));
  }
}

// Near the C3v MECI: evaluation 2 of the SA3 walk from Park's SI C3v geometry
// (acs.jctc.9b00919, SI p. S2, bohr; #846's ethylene_c3v_start.xyz) under
// its #846 config, where the walk was refused before #856 (gap 1.9e-7 Eh,
// "roots 0 and 1 are weighted 0.333333 and 0.333333"); the walk now
// converges there in 12 evaluations. Converged fresh from the point's own
// RHF, each check once:
// - the solution carries equal weights, so the pair is taken;
// - the Z-vector converges, every column's true residual at most 1e-10.
// g_I and g_J against df_gradient(I) and df_gradient(J) are printed, not
// held: the separate calls keep the I-J pair's mixing in the CI response, an
// O(1/gap) term that cancels only to its roundoff at a gap of 1.9e-7 Eh (the
// intersection README's "What the previous version lost"), and two runs on
// the gfx1200 measured from 8e-11 absolute to 1.2e-8 of the largest element.
TEST(McscfC3v, NearMeciIntersection) {
  FBF_REQUIRE_GPU();
  const std::vector<double> evaluation_2{
      -4.018256276418, 1.522677728994,  0.002426277936,  -2.543565336624, 1.420900305347,
      -0.020668261247, -4.357970135620, 2.431084049405,  -0.504820809701, -4.398234881010,
      1.553587929845,  1.028638112537,  -4.484472672178, 0.667810471886,  -0.498412700592,
      -1.490366619750, 1.348638314823,  -0.037187075434};
  const auto run = converge(c3v_entry("ethylene_c3v_near_meci", evaluation_2), nullptr, kC3vScf);
  ASSERT_NE(run, nullptr);
  const fb::ScopedDeviceIndex on{run->res().dev_idx()};
  const Calls calls{*run};
  fb::mcscf::Solution &sol = *run->sol;
  EXPECT_EQ(sol.weights, std::vector<double>(3, 1.0 / 3.0));
  const std::vector<double> &e = sol.roots.energies;
  std::cout << std::format("[ c3v      ] near MECI: gap {:.3e}; recovered from the RDMs, {}\n",
                           e[1] - e[0], recovered_weights(sol));

  auto one = calls.intersection(0, 1);
  ASSERT_TRUE(one.has_value()) << one.error().message;
  EXPECT_LE(one->residual, kSeamResidual);
  auto g_i = calls.gradient(0);
  ASSERT_TRUE(g_i.has_value()) << g_i.error().message;
  auto g_j = calls.gradient(1);
  ASSERT_TRUE(g_j.has_value()) << g_j.error().message;
  double di = 0.0;
  double dj = 0.0;
  for (std::size_t k = 0; k < g_i->total.size(); ++k) {
    di = std::max(di, std::abs(one->g_i[k] - g_i->total[k]));
    dj = std::max(dj, std::abs(one->g_j[k] - g_j->total[k]));
  }
  std::cout << std::format("[ c3v      ] near MECI: z-vectors {} products to {:.2e}; against the "
                           "separate gradients, g_I {:.2e}, g_J {:.2e} (absolute, not held)\n",
                           one->iterations, one->residual, di, dj);
}

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
