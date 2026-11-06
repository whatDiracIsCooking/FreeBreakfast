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
// coordinate.
//
// The same check at unequal weights, which PySCF refuses, so these finite
// differences are df_nac's only reference there: h2co_ccpvdz_sa2 at
// [0.6, 0.4] (the coupled pair's rotation a parameter of the average) on
// the same three coordinates, and the skewed h2co_skew_ccpvdz_sa3 at
// [0.4, 0.4, 0.2] (partly equal: (0, 1) stays projected, (0, 2) and (1, 2)
// rotate) on the coordinate where |d| is largest, C's z, at delta = 2e-4
// (kSkewDelta says why); every coordinate of each under
// FBF_SKIP_IF_FAST_TESTS(). Measured on the RTX 3080: at [0.6, 0.4]
// |D(delta) - d| at most 6.5e-8 (Richardson 6.9e-8, extrapolated within
// 3.9e-9 of d); on the skewed SA-3 at most 2.2e-7 (Richardson 2.2e-7,
// extrapolated within 1.5e-8), tolerance 4.7e-6. Holding the weights at
// 1 / nroots instead (the Hessian at the wrong weights, no rotations)
// misses by 6.0e-3 and 2.2e-2, so the check sees them. About 3 s and 4 s
// for the fast cases, 5.7 s and 7.5 s for every coordinate.
//
// And df_nac's D^X term (#742), d[D^X] = Sum_xy D^X_xy <phi_x|d phi_y>
// with the orbitals relaxed, by a ladder over the same re-converged
// solutions (McscfOverlapDensity, below, which says what it measured). This
// executable links
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
// 1e-7 of d on h2co_ccpvdz_sa2 (printed), and a re-converged state's ~1e-10
// error, over 2 delta, stays under 1e-7 too.
constexpr double kDelta = 1e-3;
// The skewed SA-3's step. At 1e-3 its (0, 2) coupling along C's z has a
// truncation error of 5.5e-6 by Richardson, above the 4.7e-6 bar itself;
// the error goes as delta^2, so 2e-4 brings it to about 2e-7, under the
// headroom, while the overlaps' round-off (below 1e-12) over 2 delta stays
// near 1e-9.
constexpr double kSkewDelta = 2e-4;
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
  std::vector<double> weights;

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

// R0's solution at @p weights, from the exact RHF and the Casci step's start.
std::unique_ptr<Converged> converge_reference(const fb::json &entry, std::span<const double> coords,
                                              std::span<const double> weights,
                                              fb::NodeResources &node) {
  auto out = std::make_unique<Converged>();
  out->weights.assign(weights.begin(), weights.end());
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
  casci.nroots = static_cast<std::int32_t>(weights.size());
  auto space = casci.select(res, molecule->molecule, basis_set->basis, *scf);
  EXPECT_TRUE(space.has_value()) << space.error().message;
  if (!space.has_value()) {
    return nullptr;
  }
  out->ncore = space->ncore;
  out->ncas = space->ncas;
  out->nelecas = space->nelecas;
  auto sol = fb::mcscf::solve(res, basis_set->basis, molecule->molecule, h->h, transform->u, {}, {},
                              space->c, out->ncore, out->ncas, out->nelecas, casci.nroots,
                              out->weights, solve_options(), &out->df().b, out->df().naux);
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
  const std::vector<double> &weights = reference.weights;
  out->weights = weights;
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

// One finite-difference check: a reference case, the weights it is solved
// at (empty for the case's own), the fast case's coordinates (empty for the
// one where |d| is largest), whether every coordinate is taken instead, and
// the step.
struct FdCase {
  std::string_view name;
  std::vector<double> weights;
  std::vector<std::size_t> coordinates;
  bool every_coordinate;
  double delta;
  std::string_view label;
};

// The label, so gtest's listing (and with it each ctest entry's name) shows
// it instead of the struct's bytes.
void PrintTo(const FdCase &param, std::ostream *os) {
  *os << param.label;
}

class McscfOverlap : public ::testing::TestWithParam<FdCase> {};

TEST_P(McscfOverlap, FiniteDifferenceMatchesNac) {
  FBF_REQUIRE_GPU();
  const FdCase &param = GetParam();
  // Under slow_test.h: every coordinate; and (#977) the unequal and skew
  // one-coordinate cases, 2.7 and 2.8 s on the RTX 3080, Sa2OneCoordinate
  // standing for them.
  if (param.every_coordinate || param.label != "Sa2OneCoordinate") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(param.name);
  const auto energies = entry.at("energies").get<std::vector<double>>();
  const bool own_weights = param.weights.empty();
  const std::vector<double> weights =
      own_weights ? entry.at("weights").get<std::vector<double>>() : param.weights;
  const auto start_time = std::chrono::steady_clock::now();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::vector<double> coords = reference_coords(entry);
  const std::unique_ptr<Converged> reference = converge_reference(entry, coords, weights, node);
  ASSERT_NE(reference, nullptr);
  const fb::mcscf::Solution &sol = *reference->solution;
  const std::size_t n = sol.roots.energies.size();
  ASSERT_EQ(n, energies.size());
  // The reference's energies are at its own weights.
  for (std::size_t r = 0; own_weights && r < n; ++r) {
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

  // The case's coordinates, or the one where |d| is largest; or every one.
  std::vector<std::size_t> coordinates = param.coordinates;
  if (coordinates.empty()) {
    std::size_t largest = 0;
    double most = -1.0;
    for (const std::vector<double> &d : nac) {
      for (std::size_t x = 0; x < d.size(); ++x) {
        if (std::abs(d[x]) > most) {
          most = std::abs(d[x]);
          largest = x;
        }
      }
    }
    coordinates.push_back(largest);
  }
  if (param.every_coordinate) {
    coordinates.resize(coords.size());
    std::iota(coordinates.begin(), coordinates.end(), std::size_t{0});
  }
  double worst = 0.0;
  double worst_richardson = 0.0;
  double transpose_error = 0.0;
  for (const std::size_t x : coordinates) {
    SCOPED_TRACE(std::format("coordinate {}", x));
    const auto coordinate_start = std::chrono::steady_clock::now();
    const double delta = param.delta;
    const std::vector<double> plus = displaced_overlaps(
        entry, *reference, x, delta, node, x == coordinates.front() ? &transpose_error : nullptr);
    const std::vector<double> minus = displaced_overlaps(entry, *reference, x, -delta, node);
    const std::vector<double> plus2 = displaced_overlaps(entry, *reference, x, 2 * delta, node);
    const std::vector<double> minus2 = displaced_overlaps(entry, *reference, x, -2 * delta, node);
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
        const double fd = (plus[ij] - minus[ij]) / (2 * delta);
        const double fd2 = (plus2[ij] - minus2[ij]) / (4 * delta);
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

// h2co_ccpvdz_sa2 at its own equal weights: H1's out-of-plane x, where d is
// largest, and O's z and H1's y, where it vanishes by symmetry. At
// [0.6, 0.4] the same three, the pair's rotation now a parameter of the
// average. The skewed SA-3 at [0.4, 0.4, 0.2], partly equal: its (0, 1)
// rotation stays projected and the other two rotate.
INSTANTIATE_TEST_SUITE_P(
    H2co, McscfOverlap,
    ::testing::Values(
        FdCase{"h2co_ccpvdz_sa2", {}, {6, 5, 7}, false, kDelta, "Sa2OneCoordinate"},
        FdCase{"h2co_ccpvdz_sa2", {}, {}, true, kDelta, "Sa2EveryCoordinate"},
        FdCase{"h2co_ccpvdz_sa2", {0.6, 0.4}, {6, 5, 7}, false, kDelta, "Sa2UnequalOneCoordinate"},
        FdCase{"h2co_ccpvdz_sa2", {0.6, 0.4}, {}, true, kDelta, "Sa2UnequalEveryCoordinate"},
        FdCase{"h2co_skew_ccpvdz_sa3",
               {0.4, 0.4, 0.2},
               {},
               false,
               kSkewDelta,
               "Skew3PartlyEqualOneCoordinate"},
        FdCase{"h2co_skew_ccpvdz_sa3",
               {0.4, 0.4, 0.2},
               {},
               true,
               kSkewDelta,
               "Skew3PartlyEqualEveryCoordinate"}),
    [](const auto &info) { return std::string(info.param.label); });

// -------------------------------------------------------------------------
// df_nac's combination overload at T = 0 and a fixed random D^X on the
// (t, i), (a, t) and (a, i) blocks, d[D^X] = Sum_xy D^X_xy <phi_x|d phi_y>,
// against the derivative of
//
//   f(R) = Sum_xy D^X_xy O_xy(R),   O(R) = C(R0)^T S(R0, R) C(R),
//
// the MO cross overlap <phi_x(R0)|phi_y(R)>, which holds the basis
// functions' motion and the orbitals' response together. The index order is
// pinned to the transition density's (D^X_xy in gamma^MN_xy = <M|E_xy|N>'s
// place, the bra's orbital first), and the sign is the coupling's: the
// derivative in the ket's geometry. C(R) is the solution re-converged at R
// from R0's orbitals carried over, as the couplings' check above, its
// columns phase-aligned to C(R0) (O_yy > 0) before the contraction; a
// rotation within the core, the active or the virtual block moves f only at
// second order, D^X being off those blocks.
//
// Every coordinate of h2co_ccpvdz_sa2, each a ladder of central differences
// at t = kDensityT0 / 2^k, k < kDensityRungs (nevpt2_gradient_gpu.cpp's
// richardson_ladder, its steadiest extrapolant), within 1e-7 of
// max |d[D^X]|, the standing rule's relative bar; a coordinate whose ratios
// never settle at 4 is logged with them and not asserted. Visibility: the
// basis-motion part (NacCombination::basis_motion) and the orbital-response
// part (response, T being zero) each have a max over 10x the bar, so the
// ladder sees both. Under FBF_SKIP_IF_FAST_TESTS(). Measured on the RTX
// 3080: all 12 coordinates settled, the worst extrapolant 1.3e-8 of
// max |d[D^X]| = 1.50 (bar 1.5e-7); motion 1.0e7 and response 8.5e6 times
// the bar; min |O_yy| 0.9975; 15 s.

constexpr double kDensityT0 = 1.6e-2;
constexpr std::int32_t kDensityRungs = 7;
constexpr double kDensityTolerance = 1e-7;
constexpr double kDensityVisibility = 10.0;

// The longest run of ratios (D_k - D_{k+1}) / (D_{k+1} - D_{k+2}) in
// (3.2, 4.8), and over it the finer of the two neighbouring Richardson
// extrapolants (4 D_{k+2} - D_{k+1}) / 3 that agree best (the steadiest);
// at a run of one, its only one. nevpt2_gradient_gpu.cpp's.
struct Ladder {
  std::int32_t run = 0;
  double richardson = 0.0;
  std::string ratios;
};

Ladder richardson_ladder(std::span<const double> d) {
  Ladder l;
  std::int32_t best_end = -1;
  std::int32_t len = 0;
  for (std::size_t k = 0; k + 2 < d.size(); ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    l.ratios += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > l.run) {
      l.run = len;
      best_end = static_cast<std::int32_t>(k);
    }
  }
  if (l.run == 0) {
    return l;
  }
  const auto r = [&](const std::size_t k) { return ((4.0 * d[k + 2]) - d[k + 1]) / 3.0; };
  auto at = static_cast<std::size_t>(best_end);
  if (l.run > 1) {
    double gap = std::numeric_limits<double>::infinity();
    for (auto k = static_cast<std::size_t>(best_end - l.run + 1);
         k < static_cast<std::size_t>(best_end); ++k) {
      if (std::abs(r(k + 1) - r(k)) < gap) {
        gap = std::abs(r(k + 1) - r(k));
        at = k + 1;
      }
    }
  }
  l.richardson = r(at);
  return l;
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// f at R0 displaced by @p step along @p coordinate: Sum_xy D_xy O_xy, O's
// columns phase-aligned. The smallest |O_yy| is kept in @p diagonal.
double density_overlap(const fb::json &entry, const Converged &reference, std::size_t coordinate,
                       double step, std::span<const double> d_x, fb::NodeResources &node,
                       double &diagonal) {
  std::vector<double> coords = reference_coords(entry);
  coords[coordinate] += step;
  const std::unique_ptr<Converged> displaced = converge_displaced(entry, coords, reference, node);
  if (displaced == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  fb::DeviceResources &res = node.device(0);
  const std::size_t nao = reference.basis().nao_sphe();
  fb::Buffers::device<double> s(nao * nao, res.device());
  const fb::Status mixed =
      fb::overlap_mixed_spherical(res, reference.basis(), displaced->basis(), s);
  EXPECT_TRUE(mixed.has_value()) << mixed.error().message;
  // Column-major: S_mn at m + nao n, C_mp at m + nao p.
  const std::vector<double> s_host = download(res, s.data(), nao * nao);
  const std::vector<double> c0 = download(res, reference.solution->c.data(), nao * nao);
  const std::vector<double> c = download(res, displaced->solution->c.data(), nao * nao);
  std::vector<double> sc(nao * nao, 0.0);
  for (std::size_t y = 0; y < nao; ++y) {
    for (std::size_t nu = 0; nu < nao; ++nu) {
      const double cy = c[nu + (nao * y)];
      for (std::size_t mu = 0; mu < nao; ++mu) {
        sc[mu + (nao * y)] += s_host[mu + (nao * nu)] * cy;
      }
    }
  }
  const auto o = [&](std::size_t x, std::size_t y) {
    double v = 0.0;
    for (std::size_t mu = 0; mu < nao; ++mu) {
      v += c0[mu + (nao * x)] * sc[mu + (nao * y)];
    }
    return v;
  };
  double f = 0.0;
  for (std::size_t y = 0; y < nao; ++y) {
    const double oyy = o(y, y);
    diagonal = std::min(diagonal, std::abs(oyy));
    const double sign = oyy < 0.0 ? -1.0 : 1.0;
    for (std::size_t x = 0; x < nao; ++x) {
      const double dxy = d_x[(x * nao) + y];
      if (dxy != 0.0) {
        f += dxy * sign * o(x, y);
      }
    }
  }
  return f;
}

// The ladder over @p coordinates of @p entry, each against df_nac's d[D^X].
void expect_density_ladder(const fb::json &entry, std::span<const std::size_t> coordinates) {
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto start_time = std::chrono::steady_clock::now();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::vector<double> coords = reference_coords(entry);
  const std::unique_ptr<Converged> reference = converge_reference(entry, coords, weights, node);
  ASSERT_NE(reference, nullptr);
  const fb::mcscf::Solution &sol = *reference->solution;
  const std::size_t nroots = sol.roots.energies.size();
  const std::size_t nmo = reference->basis().nao_sphe();
  const auto ncore = static_cast<std::size_t>(reference->ncore);
  const std::size_t nocc = ncore + static_cast<std::size_t>(reference->ncas);

  // D^X on (t, i), (a, t) and (a, i): row in a later space than its column.
  const auto space = [&](std::size_t p) { return p < ncore ? 0 : (p < nocc ? 1 : 2); };
  std::mt19937_64 engine{742};
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> d_x(nmo * nmo, 0.0);
  for (std::size_t x = 0; x < nmo; ++x) {
    for (std::size_t y = 0; y < nmo; ++y) {
      if (space(x) > space(y)) {
        d_x[(x * nmo) + y] = uniform(engine);
      }
    }
  }

  const fb::DensityFitted &df = reference->df();
  const auto &molecule = reference->calc->get<fb::MoleculeOutput>();
  const auto &pairs = reference->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &h = reference->calc->get<fb::CoreHamiltonianOutput>();
  const std::vector<double> t(nroots * nroots, 0.0);
  auto d =
      fb::mcscf::df_nac(res, molecule->molecule, reference->basis(), pairs->classes, df.aux, h->h,
                        df.b, df.l.data(), sol, reference->ncore, reference->ncas, t, d_x);
  ASSERT_TRUE(d.has_value()) << d.error().message;
  double scale = 0.0;
  double motion = 0.0;
  double response = 0.0;
  for (std::size_t i = 0; i < d->total.size(); ++i) {
    scale = std::max(scale, std::abs(d->total[i]));
    motion = std::max(motion, std::abs(d->basis_motion[i]));
    response = std::max(response, std::abs(d->response[i]));
    EXPECT_EQ(d->csf[i], 0.0);
  }
  const double bar = kDensityTolerance * scale;
  EXPECT_GT(motion, kDensityVisibility * bar);
  EXPECT_GT(response, kDensityVisibility * bar);

  double worst = 0.0;
  double diagonal = std::numeric_limits<double>::infinity();
  std::int32_t unsettled = 0;
  for (const std::size_t x : coordinates) {
    SCOPED_TRACE(std::format("coordinate {}", x));
    std::vector<double> ladder;
    for (std::int32_t k = 0; k < kDensityRungs; ++k) {
      const double step = std::ldexp(kDensityT0, -k);
      const double plus = density_overlap(entry, *reference, x, step, d_x, node, diagonal);
      const double minus = density_overlap(entry, *reference, x, -step, d_x, node, diagonal);
      ladder.push_back((plus - minus) / (2.0 * step));
    }
    const Ladder l = richardson_ladder(ladder);
    const double want = d->total[x];
    if (l.run < 3) {
      ++unsettled;
      std::cout << std::format("[ overlap] D^X x {}: not smooth, run {}, ratios{}; analytic "
                               "{:.12e}, finest {:.12e}\n",
                               x, l.run, l.ratios, want, ladder.back());
      continue;
    }
    EXPECT_NEAR(l.richardson, want, bar) << "ratios:" << l.ratios;
    worst = std::max(worst, std::abs(l.richardson - want));
    std::cout << std::format("[ overlap] D^X x {}: analytic {:.12e} (motion {:+.6e}, response "
                             "{:+.6e}), Richardson {:.12e}, relative {:.2e}, ratios{}\n",
                             x, want, d->basis_motion[x], d->response[x], l.richardson,
                             std::abs(l.richardson - want) / scale, l.ratios);
  }
  EXPECT_LT(unsettled, static_cast<std::int32_t>(coordinates.size()));

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format(
      "[ overlap] D^X: max |d| {:.6e}, bar {:.2e}, worst {:.2e} ({:.2e} "
      "relative), {} unsettled, visibility motion {:.3e} and response "
      "{:.3e} times the bar, min |O_yy| {:.6f}, z-vector {} products to {:.2e}, "
      "{:.2f} s in all\n",
      scale, bar, worst, worst / scale, unsettled, motion / bar, response / bar, diagonal,
      d->iterations, d->residual, seconds.count());
}

TEST(McscfOverlapDensity, LadderMatchesNac) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &entry = reference_case("h2co_ccpvdz_sa2");
  std::vector<std::size_t> every(3 * entry.at("atoms").size());
  std::iota(every.begin(), every.end(), std::size_t{0});
  expect_density_ladder(entry, every);
}

// The family's fast case (#1008): the same SA-2 in STO-3G (the same
// cc-pVDZ-JKFIT fit), and the ladder along one coordinate, C's x, where
// every coordinate is fourteen re-converged solutions. The visibility of the
// basis motion and the response is still over every coordinate of d.
TEST(McscfOverlapDensity, LadderMatchesNacAlongOneSto3gCoordinate) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  fb::json entry = reference_case("h2co_ccpvdz_sa2");
  entry["basis"] = "sto-3g";
  const std::array<std::size_t, 1> one{0};
  expect_density_ladder(entry, one);
}

} // namespace
