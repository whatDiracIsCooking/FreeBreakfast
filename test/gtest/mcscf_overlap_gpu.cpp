// mcscf::state_overlaps on the device, and through it a finite-difference
// check of mcscf::df_nac that needs no PySCF: on data/casscf_nac/'s
// h2co_ccpvdz_sa2 (H2CO SA-2, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT), this tree's
// DF-SA-CASSCF at R0, set up as mcscf_nac_gpu.cpp's (the exact RHF and the
// Casci step's start, converged to ||g|| < 1e-10), then
//
// - S(R0, R0) of the solution against itself, the identity to 1e-12;
// - per coordinate x, the solution re-converged at R0 +- delta and
//   R0 +- 2 delta from R0's orbitals carried over (C0 (C0^T S' C0)^-1/2, S'
//   the displaced AO overlap), each displaced root's phase aligned to R0's
//   by S_JJ(R0, R) > 0; the central difference
//   D(h) = (S_IJ(R0, R0 + h) - S_IJ(R0, R0 - h)) / 2h, the derivative of
//   <I(R0)|J(R)> in the ket's geometry, so d_IJ = <I|dJ>, df_nac's bra and
//   ket; D(delta) against df_nac's total (use_etfs = false, the CSF term
//   in) for both ordered pairs. The tolerance is the issue's 1e-5 of
//   max |d|, and it is not tuned: the central difference's truncation error
//   is (D(2 delta) - D(delta)) / 3 (Richardson), printed beside it and
//   required to be below a tenth of the tolerance, so the bar has room above
//   the FD error it is meant to measure;
// - the overlap's transpose, S(R, R0) = S(R0, R)^T, to 1e-12, at the first
//   displaced point.
//
// The fast case is three coordinates: H1's out-of-plane x, where d is
// largest (0.302 / bohr), and O's z and H1's y, two of the in-plane ones
// that vanish by C2v symmetry; every coordinate is under
// FBF_SKIP_IF_FAST_TESTS(). Measured on the RTX 3080 over two runs: the
// identity to 3.6e-14, the transpose to 2.7e-15; at H1's x (and H2's)
// |D(delta) - d| 5.1e-8 to 6.8e-8, each within 1e-8 of its Richardson
// estimate (the step's error), and the extrapolated value within 1e-9 of d;
// every other component below 2.2e-10; tolerance 3.0e-6. Four displaced
// solves take 0.4 s, so 4 s for the fast case and 6.4 s for every
// coordinate. This executable links
// pipeline for the setup; mcscf does not. Runs the device chain, so needs a
// GPU.

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
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// S(R0, R0) against the identity, and S(R, R0) against S(R0, R)^T: the
// issue's bar, round-off over unit CI vectors and an orthonormal MO overlap.
constexpr double kIdentityTolerance = 1e-12;
// The finite difference against df_nac's d, relative to max |d|: the issue's.
constexpr double kFdRelativeTolerance = 1e-5;
// The Richardson estimate of D(delta)'s own error must sit this far below
// the tolerance, so the comparison measures df_nac and not the step.
constexpr double kFdErrorHeadroom = 0.1;
// The step (bohr). Central differences at 1e-3 leave a truncation error near
// 1e-7 of d here (printed), and a re-converged state's ~1e-10 error, over
// 2 delta, stays under 1e-7 too.
constexpr double kDelta = 1e-3;
// The CASSCF's convergence, as mcscf_nac_gpu.cpp's.
constexpr double kConvTolGrad = 1e-10;
constexpr double kEnergyTolerance = 1e-8;

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

// The reference's molecule at @p coords (bohr, 3 per atom) through the real
// registry to the Scf step's inputs, the Guess step only when @p guess.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry,
                                               std::span<const double> coords, bool guess,
                                               fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_overlap");
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  std::size_t i = 0;
  for (const fb::json &atom : entry.at("atoms")) {
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coords[i] * fb::kAngstromPerBohr<double>,
                       coords[i + 1] * fb::kAngstromPerBohr<double>,
                       coords[i + 2] * fb::kAngstromPerBohr<double>);
    i += 3;
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[orthogonalizer]\n[core_hamiltonian]\n{}",
                  entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(),
                  guess ? "[guess]\n" : "");
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

// A converged DF-SA-CASSCF and what owns its basis and fitted integrals.
// Held by pointer: the references below are into calc, which stays put.
struct Converged {
  std::optional<fb::Calculation> calc;
  std::optional<fb::EriOutput> fitted;
  std::optional<fb::mcscf::Solution> solution;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::int32_t multiplicity = 1;

  [[nodiscard]] const fb::BasisSet &basis() const { return calc->get<fb::BasisSetOutput>()->basis; }
  [[nodiscard]] const fb::DensityFitted &df() const {
    return std::get<fb::DensityFitted>(fitted->eri);
  }
};

fb::mcscf::Options solve_options() {
  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  return options;
}

// The setup steps at @p coords and the fitted integrals. False on failure.
bool set_up(const fb::json &entry, std::span<const double> coords, bool guess,
            fb::NodeResources &node, Converged &out) {
  // Pipeline is move-constructible only, so emplaced rather than assigned.
  std::optional<fb::Calculation> calc = run_setup_steps(entry, coords, guess, node);
  if (!calc.has_value()) {
    return false;
  }
  out.calc.emplace(std::move(*calc));
  const auto &molecule = out.calc->get<fb::MoleculeOutput>();
  const auto &basis_set = out.calc->get<fb::BasisSetOutput>();
  const auto &pairs = out.calc->get<fb::PrimitivePairClassesOutput>();
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs};
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = entry.at("aux_basis").at("name").get<std::string>()}
                    .run(view, node.device(0));
  if (!fitted.has_value()) {
    ADD_FAILURE() << fitted.error().message;
    return false;
  }
  out.fitted = std::move(*fitted);
  out.multiplicity = molecule->molecule.multiplicity();
  return true;
}

// R0's solution, from the exact RHF and the Casci step's start.
std::unique_ptr<Converged> converge_reference(const fb::json &entry, std::span<const double> coords,
                                              fb::NodeResources &node) {
  auto out = std::make_unique<Converged>();
  fb::DeviceResources &res = node.device(0);
  if (!set_up(entry, coords, true, node, *out)) {
    return nullptr;
  }
  const auto &molecule = out->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = out->calc->get<fb::BasisSetOutput>();
  const auto &transform = out->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = out->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = out->calc->get<fb::OverlapOutput>();
  const auto &x = out->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = out->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = out->calc->get<fb::GuessOutput>();
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs};
  auto eri = fb::EriStep{}.run(view, res);
  EXPECT_TRUE(eri.has_value()) << eri.error().message;
  if (!eri.has_value()) {
    return nullptr;
  }
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  EXPECT_TRUE(scf.has_value()) << scf.error().message;
  if (!scf.has_value()) {
    return nullptr;
  }
  fb::CasciStep casci;
  casci.ncas = entry.at("ncas").get<std::int32_t>();
  casci.nelecas = entry.at("nelecas").get<std::int32_t>();
  casci.nroots = static_cast<std::int32_t>(entry.at("weights").size());
  auto space = casci.select(res, molecule->molecule, basis_set->basis, *scf);
  EXPECT_TRUE(space.has_value()) << space.error().message;
  if (!space.has_value()) {
    return nullptr;
  }
  out->ncore = space->ncore;
  out->ncas = space->ncas;
  out->nelecas = space->nelecas;
  const auto weights = entry.at("weights").get<std::vector<double>>();
  auto sol = fb::mcscf::solve(res, basis_set->basis, molecule->molecule, h->h, transform->u, {}, {},
                              space->c, out->ncore, out->ncas, out->nelecas, casci.nroots, weights,
                              solve_options(), &out->df().b, out->df().naux);
  EXPECT_TRUE(sol.has_value()) << sol.error().message;
  if (!sol.has_value()) {
    return nullptr;
  }
  out->solution = std::move(*sol);
  return out;
}

// The solution at @p coords from @p reference's orbitals, carried over by
// C = C0 (C0^T S C0)^-1/2, S the AO overlap at @p coords: orthonormal there
// and as near C0 as an orthonormal set gets.
std::unique_ptr<Converged> converge_displaced(const fb::json &entry, std::span<const double> coords,
                                              const Converged &reference, fb::NodeResources &node) {
  auto out = std::make_unique<Converged>();
  fb::DeviceResources &res = node.device(0);
  if (!set_up(entry, coords, false, node, *out)) {
    return nullptr;
  }
  const auto &molecule = out->calc->get<fb::MoleculeOutput>();
  const auto &transform = out->calc->get<fb::SphericalTransformOutput>();
  const auto &overlap = out->calc->get<fb::OverlapOutput>();
  const auto &h = out->calc->get<fb::CoreHamiltonianOutput>();
  const fb::BasisSet &basis = out->basis();
  const auto nao = static_cast<std::int32_t>(basis.nao_sphe());
  const std::size_t square = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
  const fb::SphericalMatrixAO &c0 = reference.solution->c;
  fb::Buffers::device<double> work(square, res.device());
  fb::Buffers::device<double> metric(square, res.device());
  fb::Buffers::device<double> root(square, res.device());
  EXPECT_TRUE(fb::gpu_status(fb::gemmwich(res.blas(), nao, c0.data(), overlap->s.data(),
                                          work.data(), metric.data()))
                  .has_value());
  if (const fb::Status orth = fb::orthogonalize(res, nao, metric.data(), root.data());
      !orth.has_value()) {
    ADD_FAILURE() << orth.error().message;
    return nullptr;
  }
  fb::SphericalMatrixAO c(res, basis);
  EXPECT_TRUE(fb::gpu_status(fb::gemm(res.blas(), nao, &fb::kOne<double>, c0.data(), root.data(),
                                      &fb::kZero<double>, c.data()))
                  .has_value());
  const auto weights = entry.at("weights").get<std::vector<double>>();
  out->ncore = reference.ncore;
  out->ncas = reference.ncas;
  out->nelecas = reference.nelecas;
  auto sol =
      fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {}, c, out->ncore,
                       out->ncas, out->nelecas, static_cast<std::int32_t>(weights.size()), weights,
                       solve_options(), &out->df().b, out->df().naux);
  EXPECT_TRUE(sol.has_value()) << sol.error().message;
  if (!sol.has_value()) {
    return nullptr;
  }
  out->solution = std::move(*sol);
  return out;
}

std::vector<double> reference_coords(const fb::json &entry) {
  std::vector<double> coords;
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    coords.insert(coords.end(), coord.begin(), coord.end());
  }
  return coords;
}

// S(R0, R) for the solution at R0 displaced by @p step along coordinate
// @p coordinate, each ket root's phase aligned to R0's (S_JJ > 0). With
// @p transpose_error, also max |S(R, R0) - S(R0, R)^T| there.
std::vector<double> displaced_overlaps(const fb::json &entry, const Converged &reference,
                                       std::size_t coordinate, double step, fb::NodeResources &node,
                                       double *transpose_error = nullptr) {
  std::vector<double> coords = reference_coords(entry);
  coords[coordinate] += step;
  const std::unique_ptr<Converged> displaced = converge_displaced(entry, coords, reference, node);
  if (displaced == nullptr) {
    return {};
  }
  fb::DeviceResources &res = node.device(0);
  const std::int32_t multiplicity = reference.multiplicity;
  auto s = fb::mcscf::state_overlaps(res, reference.basis(), *reference.solution,
                                     displaced->basis(), *displaced->solution, reference.ncore,
                                     reference.ncas, reference.nelecas, multiplicity);
  EXPECT_TRUE(s.has_value()) << s.error().message;
  if (!s.has_value()) {
    return {};
  }
  const auto n = displaced->solution->roots.energies.size();
  if (transpose_error != nullptr) {
    auto back = fb::mcscf::state_overlaps(res, displaced->basis(), *displaced->solution,
                                          reference.basis(), *reference.solution, reference.ncore,
                                          reference.ncas, reference.nelecas, multiplicity);
    EXPECT_TRUE(back.has_value()) << back.error().message;
    if (back.has_value()) {
      *transpose_error = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          *transpose_error =
              std::max(*transpose_error, std::abs((*back)[(j * n) + i] - (*s)[(i * n) + j]));
        }
      }
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    const double sign = (*s)[(j * n) + j] < 0.0 ? -1.0 : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      (*s)[(i * n) + j] *= sign;
    }
  }
  return std::move(*s);
}

class McscfOverlap : public ::testing::TestWithParam<bool> {};

TEST_P(McscfOverlap, FiniteDifferenceMatchesNac) {
  FBF_REQUIRE_GPU();
  const bool every_coordinate = GetParam();
  if (every_coordinate) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case("h2co_ccpvdz_sa2");
  const auto energies = entry.at("energies").get<std::vector<double>>();
  const auto start_time = std::chrono::steady_clock::now();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::vector<double> coords = reference_coords(entry);
  const std::unique_ptr<Converged> reference = converge_reference(entry, coords, node);
  ASSERT_NE(reference, nullptr);
  const fb::mcscf::Solution &sol = *reference->solution;
  const std::size_t n = sol.roots.energies.size();
  ASSERT_EQ(n, energies.size());
  for (std::size_t r = 0; r < n; ++r) {
    EXPECT_NEAR(sol.roots.energies[r], energies[r], kEnergyTolerance) << "root " << r;
  }

  // S(R0, R0) is the identity.
  auto identity = fb::mcscf::state_overlaps(res, reference->basis(), sol, reference->basis(), sol,
                                            reference->ncore, reference->ncas, reference->nelecas,
                                            reference->multiplicity);
  ASSERT_TRUE(identity.has_value()) << identity.error().message;
  ASSERT_EQ(identity->size(), n * n);
  double identity_error = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double want = i == j ? 1.0 : 0.0;
      EXPECT_NEAR((*identity)[(i * n) + j], want, kIdentityTolerance) << i << " " << j;
      identity_error = std::max(identity_error, std::abs((*identity)[(i * n) + j] - want));
    }
  }

  // df_nac's d for both ordered pairs, the CSF term in.
  const fb::DensityFitted &df = reference->df();
  const auto &molecule = reference->calc->get<fb::MoleculeOutput>();
  const auto &pairs = reference->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &h = reference->calc->get<fb::CoreHamiltonianOutput>();
  std::vector<std::vector<double>> nac(n * n);
  double scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      auto d =
          fb::mcscf::df_nac(res, molecule->molecule, reference->basis(), pairs->classes, df.aux,
                            h->h, df.b, df.l.data(), sol, reference->ncore, reference->ncas,
                            static_cast<std::int32_t>(i), static_cast<std::int32_t>(j));
      ASSERT_TRUE(d.has_value()) << d.error().message;
      for (const double v : d->total) {
        scale = std::max(scale, std::abs(v));
      }
      nac[(i * n) + j] = std::move(d->total);
    }
  }
  const double tolerance = kFdRelativeTolerance * scale;

  // H1's out-of-plane x, where d is largest, and O's z and H1's y, where it
  // vanishes by symmetry; or every coordinate.
  std::vector<std::size_t> coordinates{6, 5, 7};
  if (every_coordinate) {
    coordinates.resize(coords.size());
    std::iota(coordinates.begin(), coordinates.end(), std::size_t{0});
  }
  double worst = 0.0;
  double worst_richardson = 0.0;
  double transpose_error = 0.0;
  for (const std::size_t x : coordinates) {
    SCOPED_TRACE(std::format("coordinate {}", x));
    const auto coordinate_start = std::chrono::steady_clock::now();
    const std::vector<double> plus = displaced_overlaps(
        entry, *reference, x, kDelta, node, x == coordinates.front() ? &transpose_error : nullptr);
    const std::vector<double> minus = displaced_overlaps(entry, *reference, x, -kDelta, node);
    const std::vector<double> plus2 = displaced_overlaps(entry, *reference, x, 2 * kDelta, node);
    const std::vector<double> minus2 = displaced_overlaps(entry, *reference, x, -2 * kDelta, node);
    ASSERT_EQ(plus.size(), n * n);
    ASSERT_EQ(minus.size(), n * n);
    ASSERT_EQ(plus2.size(), n * n);
    ASSERT_EQ(minus2.size(), n * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        const std::size_t ij = (i * n) + j;
        const double fd = (plus[ij] - minus[ij]) / (2 * kDelta);
        const double fd2 = (plus2[ij] - minus2[ij]) / (4 * kDelta);
        const double richardson = std::abs(fd2 - fd) / 3;
        // The extrapolated (4 D(delta) - D(2 delta)) / 3, printed only: it
        // shows the difference left over is the step's.
        const double extrapolated = ((4 * fd) - fd2) / 3;
        const double want = nac[ij][x];
        EXPECT_NEAR(fd, want, tolerance) << "(" << i << ", " << j << ")";
        EXPECT_LT(richardson, kFdErrorHeadroom * tolerance) << "(" << i << ", " << j << ")";
        worst = std::max(worst, std::abs(fd - want));
        worst_richardson = std::max(worst_richardson, richardson);
        std::cout << std::format("[ overlap] x {} ({}, {}): fd {:.10f}, df_nac {:.10f}, |diff| "
                                 "{:.2e}, Richardson error {:.2e}, extrapolated |diff| {:.2e}\n",
                                 x, i, j, fd, want, std::abs(fd - want), richardson,
                                 std::abs(extrapolated - want));
      }
    }
    const std::chrono::duration<double> seconds =
        std::chrono::steady_clock::now() - coordinate_start;
    std::cout << std::format("[ overlap] x {}: four displaced solves in {:.2f} s\n", x,
                             seconds.count());
  }
  EXPECT_LT(transpose_error, kIdentityTolerance);

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ overlap] identity {:.2e}, transpose {:.2e}, max |fd - d| {:.2e} "
                           "(tolerance {:.2e}), Richardson {:.2e}, {:.2f} s in all\n",
                           identity_error, transpose_error, worst, tolerance, worst_richardson,
                           seconds.count());
}

INSTANTIATE_TEST_SUITE_P(H2coSa2, McscfOverlap, ::testing::Values(false, true),
                         [](const auto &info) {
                           return std::string(info.param ? "EveryCoordinate" : "OneCoordinate");
                         });

} // namespace
