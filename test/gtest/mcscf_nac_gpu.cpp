// mcscf::df_nac on the device against PySCF's DF SA-CASSCF couplings in
// data/casscf_nac/casscf_nac.json (tools/emit/casscf_nac_reference.py), for
// both cases (H2CO SA-2 and the skewed H2CO SA-3, CAS(4,3), cc-pVDZ /
// cc-pVDZ-JKFIT): this tree's DF-SA-CASSCF (mcscf::solve under
// DensityFitting::Full) from the exact RHF and the Casci step's start,
// converged to ||g|| < 1e-10 and taken as it converges (a coupling does not
// depend on the orbitals' canonicalization or signs, only on the two CI
// phases); then for every ordered pair (I, J),
//
// - the numerator h_IJ against `h`, d^CSF against `csf` and d_IJ against
//   `total`, to 1e-8, up to the one sign s_I s_J the two CI phases carry
//   (read off h's overlap with the reference, and the same for all three);
// - d_IJ = -d_JI, d^CSF_IJ = -d^CSF_JI and h_IJ = h_JI, of this tree's own
//   results (one set of CI phases, so no sign to align);
// - Sum_A h_IJ = 0, the numerator's translational invariance. The CSF term
//   is not invariant (the half-derivative overlap takes only the bra's
//   centre), so neither is d_IJ, and neither is tested for it.
//
// Then on each case's first pair the options: use_etfs against `etf`,
// mult_ediff against h + (E_J - E_I) csf, and both against `h`; and a
// repeated call, whose Z-vector is bitwise the first's (the same products,
// the same residual) and whose results agree to 1e-10 (the derivative
// kernels' atomics, nac/README.md "Reproducibility"). And the
// refusals: equal roots and a root past the solution's. PySCF refuses
// unequal weights, so df_nac is checked there against finite differences
// of mcscf::state_overlaps, in mcscf_overlap_gpu.cpp.
// Measured on the RTX 3080: h_IJ within 3.3e-10, d^CSF 2.2e-10, d_IJ 1.0e-9
// (the skewed (0, 2)); 3.1 s and 4.0 s a case, so neither is a slow test.
//
// Then mcscf::transition_orbital_gradient, the Z-vector's right-hand side
// df_nac takes from it, on the same solutions (its own comment, below); and
// df_nac's combination overload against the pair's (its own comment, below).
//
// The setup is mcscf_transition_gradient_gpu.cpp's, so this executable links
// pipeline; mcscf does not. Runs the device chain, so needs a GPU.

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
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"
#include "slow_test.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar on h_IJ, d^CSF and d_IJ.
constexpr double kNacTolerance = 1e-8;
// This tree's results against each other: the same formula, to round-off.
constexpr double kSelfTolerance = 1e-10;
// Sum_A h_IJ, the reference's measured 2.9e-13 for the etf.
constexpr double kTranslationTolerance = 1e-9;
// The Z-vector's own tolerance is 1e-10; its residual must reach it.
constexpr double kResidualTolerance = 1e-10;
// The CASSCF's convergence, as mcscf_transition_gradient_gpu.cpp's.
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

// The reference's molecule through the real registry to the Scf step's
// inputs. The Scf step itself is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_nac");
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

// A reference n_atom x 3 array, flat as the call's.
std::vector<double> flat(const fb::json &rows) {
  std::vector<double> out;
  for (const auto &row : rows.get<std::vector<std::vector<double>>>()) {
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}

// max_i |ours_i - sign want_i|, each element also EXPECTed to @p tolerance.
double compare(std::span<const double> ours, std::span<const double> want, double sign,
               double tolerance, std::string_view what) {
  EXPECT_EQ(ours.size(), want.size()) << what;
  double error = 0.0;
  for (std::size_t i = 0; i < std::min(ours.size(), want.size()); ++i) {
    EXPECT_NEAR(ours[i], sign * want[i], tolerance) << what << " " << i;
    error = std::max(error, std::abs(ours[i] - (sign * want[i])));
  }
  return error;
}

// A reference case's converged DF-SA-CASSCF: the setup steps, the exact RHF,
// the fitted integrals and mcscf::solve from the Casci step's start, kept
// together because the outputs refer into the calculation.
struct Converged {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::EriOutput> fitted;
  std::optional<fb::ActiveSpace> space;
  std::optional<fb::mcscf::Solution> sol;
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
  EXPECT_EQ(basis.nao_sphe(), entry.at("nao").get<std::size_t>());
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
  const fb::json &aux = entry.at("aux_basis");
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = aux.at("name").get<std::string>()}
                    .run(view, res);
  if (!fitted.has_value()) {
    ADD_FAILURE() << fitted.error().message;
    return nullptr;
  }
  run->fitted.emplace(std::move(*fitted));
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  EXPECT_EQ(df.naux, aux.at("naux").get<std::int32_t>());
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
  EXPECT_EQ(run->space->ncore, entry.at("ncore").get<std::int32_t>());

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

class McscfNac : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfNac, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h (#977): the skew SA3 formaldehyde, 3.0 s on the RTX 3080.
  if (std::string_view{GetParam()} == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  const auto energies = entry.at("energies").get<std::vector<double>>();
  const auto start_time = std::chrono::steady_clock::now();

  const auto run = converge(entry);
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
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(sol.roots.energies[r], energies[r], kEnergyTolerance) << "root " << r;
  }

  const auto nac = [&](std::int32_t bra, std::int32_t ket, fb::mcscf::NacOptions nac_options) {
    return fb::mcscf::df_nac(res, molecule, basis, classes, df.aux, h, df.b, df.l.data(), sol,
                             ncore, ncas, bra, ket, nac_options);
  };

  std::map<std::pair<std::int32_t, std::int32_t>, fb::mcscf::Nac> ours;
  for (const fb::json &coupling : entry.at("couplings")) {
    const auto bra = coupling.at("bra").get<std::int32_t>();
    const auto ket = coupling.at("ket").get<std::int32_t>();
    const std::string label = std::format("{} ({}, {})", name, bra, ket);
    SCOPED_TRACE(label);
    const auto pair_start = std::chrono::steady_clock::now();
    auto d = nac(bra, ket, {});
    ASSERT_TRUE(d.has_value()) << d.error().message;
    const std::chrono::duration<double> pair_seconds =
        std::chrono::steady_clock::now() - pair_start;
    EXPECT_LT(d->residual, kResidualTolerance);
    EXPECT_GT(d->iterations, 0);

    // s_I s_J from h's overlap with the reference.
    const std::vector<double> want_h = flat(coupling.at("h"));
    const double dot =
        std::inner_product(d->numerator.begin(), d->numerator.end(), want_h.begin(), 0.0);
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    const double h_error = compare(d->numerator, want_h, sign, kNacTolerance, "h");
    const double csf_error = compare(d->csf, flat(coupling.at("csf")), sign, kNacTolerance, "csf");
    const double total_error =
        compare(d->total, flat(coupling.at("total")), sign, kNacTolerance, "total");

    // The numerator is translation-invariant.
    double translation = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
      double sum = 0.0;
      for (std::size_t i = k; i < d->numerator.size(); i += 3) {
        sum += d->numerator[i];
      }
      translation = std::max(translation, std::abs(sum));
    }
    EXPECT_LT(translation, kTranslationTolerance);
    std::cout << std::format("[ nac] {}: h {:.2e}, csf {:.2e}, total {:.2e}, sum_A h {:.2e}, "
                             "z-vector {} products to {:.2e} ({:.2f} s)\n",
                             label, h_error, csf_error, total_error, translation, d->iterations,
                             d->residual, pair_seconds.count());
    ours.emplace(std::pair{bra, ket}, std::move(*d));
  }

  // Antisymmetry, of this tree's own couplings.
  double antisymmetry = 0.0;
  for (const auto &[key, d] : ours) {
    const auto other = ours.find(std::pair{key.second, key.first});
    ASSERT_NE(other, ours.end());
    for (std::size_t i = 0; i < d.total.size(); ++i) {
      EXPECT_NEAR(d.total[i], -other->second.total[i], kSelfTolerance) << "total " << i;
      EXPECT_NEAR(d.csf[i], -other->second.csf[i], kSelfTolerance) << "csf " << i;
      EXPECT_NEAR(d.numerator[i], other->second.numerator[i], kSelfTolerance) << "h " << i;
      antisymmetry = std::max({antisymmetry, std::abs(d.total[i] + other->second.total[i]),
                               std::abs(d.numerator[i] - other->second.numerator[i])});
    }
  }

  // The options, on the first pair.
  {
    const fb::json &coupling = entry.at("couplings").at(0);
    const auto bra = coupling.at("bra").get<std::int32_t>();
    const auto ket = coupling.at("ket").get<std::int32_t>();
    const fb::mcscf::Nac &base = ours.at(std::pair{bra, ket});
    const std::vector<double> want_h = flat(coupling.at("h"));
    const std::vector<double> want_csf = flat(coupling.at("csf"));
    const double gap = coupling.at("energy_gap").get<double>();
    const double dot =
        std::inner_product(base.numerator.begin(), base.numerator.end(), want_h.begin(), 0.0);
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    std::vector<double> want_mult(want_h.size());
    for (std::size_t i = 0; i < want_h.size(); ++i) {
      want_mult[i] = want_h[i] + (gap * want_csf[i]);
    }
    for (const auto &[nac_options, want, what] :
         {std::tuple{fb::mcscf::NacOptions{.mult_ediff = false, .use_etfs = true},
                     flat(coupling.at("etf")), "etf"},
          std::tuple{fb::mcscf::NacOptions{.mult_ediff = true, .use_etfs = false}, want_mult,
                     "mult_ediff"},
          std::tuple{fb::mcscf::NacOptions{.mult_ediff = true, .use_etfs = true}, want_h,
                     "mult_ediff and use_etfs"}}) {
      auto d = nac(bra, ket, nac_options);
      ASSERT_TRUE(d.has_value()) << d.error().message;
      compare(d->total, want, sign, kNacTolerance, what);
      compare(d->numerator, base.numerator, 1.0, kSelfTolerance, what);
      compare(d->csf, base.csf, 1.0, kSelfTolerance, what);
    }

    // solve_tolerance takes effect (#866): one above the right-hand side's
    // norm stops the Z-vector before its first product, z = 0, and h_IJ
    // loses the multipliers' part; one not positive is InvalidConfig.
    {
      auto loose = nac(bra, ket, fb::mcscf::NacOptions{.solve_tolerance = 1e6});
      ASSERT_TRUE(loose.has_value()) << loose.error().message;
      EXPECT_EQ(loose->iterations, 0);
      EXPECT_GT(base.iterations, 0);
      double moved = 0.0;
      for (std::size_t i = 0; i < base.numerator.size(); ++i) {
        moved = std::max(moved, std::abs(loose->numerator[i] - base.numerator[i]));
      }
      EXPECT_GT(moved, kNacTolerance);
      const auto refused = nac(bra, ket, fb::mcscf::NacOptions{.solve_tolerance = 0.0});
      ASSERT_FALSE(refused.has_value());
      EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
      EXPECT_NE(refused.error().message.find("solve_tolerance"), std::string::npos);
      std::cout << std::format("[ nac] {}: solve_tolerance 1e6: no products, h moved {:.2e}\n",
                               name, moved);
    }

    // Repeatability: everything before the derivative integrals is bitwise
    // (the Z-vector to its last product), and only their atomics' order is
    // left to vary.
    auto again = nac(bra, ket, {});
    ASSERT_TRUE(again.has_value()) << again.error().message;
    EXPECT_EQ(again->iterations, base.iterations);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(again->residual),
              std::bit_cast<std::uint64_t>(base.residual));
    const double repeat =
        std::max({compare(again->numerator, base.numerator, 1.0, kSelfTolerance, "repeat h"),
                  compare(again->csf, base.csf, 1.0, kSelfTolerance, "repeat csf"),
                  compare(again->total, base.total, 1.0, kSelfTolerance, "repeat total")});
    std::cout << std::format("[ nac] {}: a repeated call within {:.2e}\n", name, repeat);
  }

  // Refusals: a root against itself and a root past the solution's.
  for (const auto &[bra, ket] : {std::pair{0, 0}, std::pair{0, nroots}}) {
    const auto refused = nac(bra, ket, {});
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
    EXPECT_NE(refused.error().message.find("mcscf::df_nac"), std::string::npos);
  }

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ nac] {}: antisymmetry {:.2e}, {:.2f} s in all\n", name, antisymmetry,
                           seconds.count());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfNac,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

// -------------------------------------------------------------------------
// mcscf::transition_orbital_gradient, g^(sr), the orbital gradient of
// <c_s|H|c_r> that df_nac's Z-vector solves against, at the same converged
// solutions, for every pair r < s:
//
// - two calls bitwise the same, and bitwise what df_nac formed inline before
//   it was hoisted (response::fock's gradient at the transition RDMs less its
//   gradient at zero RDMs, the difference taken on the host, the same IEEE
//   subtraction as the axpy's). Everything upstream of the derivative
//   integrals is bitwise from call to call (nac/README.md "Reproducibility");
//   measured spread between two calls 0 on the RTX 3080, so the bar is the
//   bit, not a floor.
// - a directional ladder: <c_s|H|c_r> at C exp(t kappa(x)), the CI vectors
//   held, is sum_tu h_eff,tu gamma_tu + 1/2 sum_tuvw (tu|vw) Gamma_tuvw over
//   casscf_integrals at the rotated orbitals (no response::fock, and so not
//   the function under test); its central differences against <x, g^(sr)>
//   along random unit x, nevpt2_gradient_gpu.cpp's ladder and bar.

// central differences D_k at t = t0 / 2^k, successive changes falling by a
// factor in (3.2, 4.8) over at least three halvings, and the Richardson
// extrapolant at that run's finest pair against the prediction to 1e-7
// relative: nevpt2_gradient_gpu.cpp's.
constexpr double kLadderT0 = 1e-2;
constexpr std::int32_t kRungs = 10;
constexpr double kLadderTolerance = 1e-7;
constexpr std::int32_t kLadderDirections = 2;

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// C exp(t kappa(x)) for the packed x: kappa unpacked on the device, then
// skew_eig and skew_expm (response_fock_gpu.cpp's rotation).
std::optional<fb::SphericalMatrixAO> rotate(fb::DeviceResources &res, const fb::BasisSet &basis,
                                            const fb::SphericalMatrixAO &c, std::int32_t ncore,
                                            std::int32_t ncas, const std::vector<double> &x,
                                            double t) {
  const auto nmo = static_cast<std::int32_t>(c.n());
  const auto nn = static_cast<std::size_t>(nmo) * static_cast<std::size_t>(nmo);
  fb::Buffers::device<double> packed(x.size(), res.device());
  fb::Buffers::device<double> kappa(nn, res.device());
  fb::Buffers::device<wwr::wwrDoubleComplex> u(nn, res.device());
  fb::Buffers::device<double> w(static_cast<std::size_t>(nmo), res.device());
  fb::Buffers::device<double> e(nn, res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(packed.data(), x.data(), sizeof(double) * x.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  if (!fb::response::unpack_orbital_pairs(res, nmo, ncore, ncas, packed.data(), kappa.data())
           .has_value() ||
      !fb::skew_eig<double>(res, nmo, kappa.data(), u.data(), w.data()).has_value() ||
      !fb::skew_expm<double>(res, nmo, u.data(), w.data(), t, e.data()).has_value()) {
    ADD_FAILURE() << "the rotation failed";
    return std::nullopt;
  }
  std::optional<fb::SphericalMatrixAO> rotated;
  rotated.emplace(res, basis);
  constexpr double kOne = 1.0;
  constexpr double kZero = 0.0;
  const auto status = wwr::gemm<double, std::int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, nmo, &kOne, c.data(), nmo,
      e.data(), nmo, &kZero, rotated->data(), nmo);
  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_TRUE(res.sync().has_value());
  return rotated;
}

// Unit random directions in the packed space, from a fixed seed.
std::vector<std::vector<double>> directions(std::size_t count, std::size_t n_pairs) {
  std::mt19937_64 engine{727};
  std::normal_distribution<double> normal;
  std::vector<std::vector<double>> out(count, std::vector<double>(n_pairs));
  for (auto &x : out) {
    double norm = 0.0;
    for (double &value : x) {
      value = normal(engine);
      norm += value * value;
    }
    for (double &value : x) {
      value /= std::sqrt(norm);
    }
  }
  return out;
}

// The longest run of ratios (D_k - D_{k+1}) / (D_{k+1} - D_{k+2}) in
// (3.2, 4.8), and the Richardson extrapolant (4 D_{k+2} - D_{k+1}) / 3 at
// its finest pair.
struct Ladder {
  std::int32_t run = 0;
  double richardson = 0.0;
  std::string ratios;
};

Ladder richardson_ladder(std::span<const double> d) {
  Ladder l;
  std::int32_t len = 0;
  for (std::size_t k = 0; k + 2 < d.size(); ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    l.ratios += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > l.run) {
      l.run = len;
      l.richardson = ((4.0 * d[k + 2]) - d[k + 1]) / 3.0;
    }
  }
  return l;
}

class McscfTransitionOrbitalGradient : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfTransitionOrbitalGradient, IsDfNacsAndTheLadders) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h (#977): the skew SA3 formaldehyde, 2.2 s on the RTX 3080.
  if (std::string_view{GetParam()} == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto run = converge(entry);
  ASSERT_NE(run, nullptr);
  fb::DeviceResources &res = run->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &molecule = run->calc->get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = run->calc->get<fb::BasisSetOutput>()->basis;
  const fb::SphericalMatrixAO &h = run->calc->get<fb::CoreHamiltonianOutput>()->h;
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  const fb::mcscf::Solution &sol = *run->sol;
  const std::int32_t ncore = run->space->ncore;
  const std::int32_t ncas = run->space->ncas;
  const std::int32_t nelecas = run->space->nelecas;
  const auto nroots = static_cast<std::int32_t>(sol.roots.energies.size());
  const auto n = static_cast<std::size_t>(ncas);
  const std::size_t m = n * n;

  const std::int32_t two_s = molecule.multiplicity() - 1;
  const std::int32_t n_beta = (nelecas - two_s) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta);
  auto beta = fb::ci::DeviceStringSet::create(res, ncas, n_beta);
  ASSERT_TRUE(alpha.has_value() && beta.has_value());
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
  const auto dim = static_cast<std::size_t>(rdm->size());
  ASSERT_EQ(sol.roots.vectors.num_elements(), dim * static_cast<std::size_t>(nroots));
  auto ints = fb::casscf_integrals(res, basis, molecule, h, df.b, df.naux, sol.c, ncore, ncas);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;

  // <c_s|H|c_r> at the orbitals @p c, the CI vectors' RDMs held.
  const auto element = [&](const fb::SphericalMatrixAO &c, std::span<const double> gamma,
                           std::span<const double> gamma2) {
    auto at = fb::casscf_integrals(res, basis, molecule, h, df.b, df.naux, c, ncore, ncas);
    EXPECT_TRUE(at.has_value()) << at.error().message;
    if (!at.has_value()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    // h_eff is symmetric and (tu|vw) column-major is (wv|ut) row-major, so
    // both read flat against the row-major RDMs.
    const std::vector<double> h_eff = download(res, at->hamiltonian.h_eff.data(), m);
    const std::vector<double> eri = download(res, at->hamiltonian.eri.data(), m * m);
    return std::inner_product(h_eff.begin(), h_eff.end(), gamma.begin(), 0.0) +
           (0.5 * std::inner_product(eri.begin(), eri.end(), gamma2.begin(), 0.0));
  };

  fb::Buffers::device<double> rdms(m + (m * m), res.device());
  for (std::int32_t s = 1; s < nroots; ++s) {
    for (std::int32_t r = 0; r < s; ++r) {
      SCOPED_TRACE(std::format("{} ({}, {})", name, s, r));
      const double *const ci_s = sol.roots.vectors.data() + (static_cast<std::size_t>(s) * dim);
      const double *const ci_r = sol.roots.vectors.data() + (static_cast<std::size_t>(r) * dim);
      ASSERT_TRUE(rdm->trans_rdm12(res, ci_s, ci_r, rdms.data(), rdms.data() + m).has_value());
      const std::vector<double> gamma = download(res, rdms.data(), m);
      const std::vector<double> gamma2 = download(res, rdms.data() + m, m * m);
      const auto g_sr = [&] {
        return fb::mcscf::transition_orbital_gradient(res, molecule, h, df.b, df.naux, sol.c, ncore,
                                                      ncas, ints->ppaa, ints->papa, gamma, gamma2);
      };
      auto first = g_sr();
      ASSERT_TRUE(first.has_value()) << first.error().message;
      auto second = g_sr();
      ASSERT_TRUE(second.has_value()) << second.error().message;
      const std::size_t pairs = first->num_elements();
      const std::vector<double> g = download(res, first->data(), pairs);
      const std::vector<double> again = download(res, second->data(), pairs);

      // What df_nac formed inline before the hoist.
      auto at_rdms = fb::response::fock(res, molecule, h, df.b, df.naux, sol.c, ncore, ncas,
                                        ints->ppaa, ints->papa, gamma, gamma2);
      ASSERT_TRUE(at_rdms.has_value()) << at_rdms.error().message;
      const std::vector<double> zero_gamma(m, 0.0);
      const std::vector<double> zero_gamma2(m * m, 0.0);
      auto at_zero = fb::response::fock(res, molecule, h, df.b, df.naux, sol.c, ncore, ncas,
                                        ints->ppaa, ints->papa, zero_gamma, zero_gamma2);
      ASSERT_TRUE(at_zero.has_value()) << at_zero.error().message;
      ASSERT_EQ(at_rdms->gradient.num_elements(), pairs);
      std::vector<double> before = download(res, at_rdms->gradient.data(), pairs);
      const std::vector<double> core = download(res, at_zero->gradient.data(), pairs);
      double spread = 0.0;
      double largest = 0.0;
      std::size_t differ_self = 0;
      std::size_t differ_before = 0;
      const auto bits = [](double v) { return std::bit_cast<std::uint64_t>(v); };
      for (std::size_t i = 0; i < pairs; ++i) {
        before[i] -= core[i];
        spread = std::max(spread, std::abs(again[i] - g[i]));
        largest = std::max(largest, std::abs(g[i]));
        differ_self += bits(again[i]) != bits(g[i]) ? 1U : 0U;
        differ_before += bits(before[i]) != bits(g[i]) ? 1U : 0U;
      }
      EXPECT_EQ(differ_self, 0U) << "two calls, spread " << spread;
      EXPECT_EQ(differ_before, 0U) << "against the inline difference";
      std::cout << std::format("[ g^(sr)] {} ({}, {}): {} pairs, max |g| {:.3e}, two calls "
                               "spread {:.1e} ({} elements differ), against the inline "
                               "difference {} differ\n",
                               name, s, r, pairs, largest, spread, differ_self, differ_before);

      // The ladder of <c_s|H|c_r> along random kappa.
      for (const auto &x : directions(kLadderDirections, pairs)) {
        const double predicted = std::inner_product(g.begin(), g.end(), x.begin(), 0.0);
        const auto along = [&](double t) {
          auto c = rotate(res, basis, sol.c, ncore, ncas, x, t);
          return c.has_value() ? element(*c, gamma, gamma2)
                               : std::numeric_limits<double>::quiet_NaN();
        };
        std::vector<double> d;
        for (std::int32_t k = 0; k < kRungs; ++k) {
          const double t = std::ldexp(kLadderT0, -k);
          d.push_back((along(t) - along(-t)) / (2.0 * t));
        }
        const Ladder l = richardson_ladder(d);
        ASSERT_GE(l.run, 3) << "ratios:" << l.ratios;
        EXPECT_NEAR(l.richardson, predicted, kLadderTolerance * std::abs(predicted))
            << "ratios:" << l.ratios;
        std::cout << std::format("[ g^(sr)] {} ({}, {}) ladder: predicted {:.12e}, Richardson "
                                 "{:.12e}, relative {:.2e}, ratios{}\n",
                                 name, s, r, predicted, l.richardson,
                                 std::abs(l.richardson - predicted) / std::abs(predicted),
                                 l.ratios);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfTransitionOrbitalGradient,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

// -------------------------------------------------------------------------
// df_nac over a combination, Sum_MN T_MN d_MN + d[D^X] in one Z-vector,
// against the pair overload at the same converged solutions (#742). Every
// check is a path agreement, so each runs once at 1e-10 of the largest
// element of the quantity compared:
//
// - for every ordered pair (I, J), T = 1/2 (E_IJ - E_JI) gives the pair's
//   d_IJ (total) and d^CSF, no basis-motion term, the pair's gap; and its
//   right-hand side times E_J - E_I is the pair's, the transition orbital
//   gradient at trans_rdm12(I, J) (the combination's is at the RDMs over the
//   gap);
// - on the skewed SA-3, a random antisymmetric T gives
//   Sum_{M<N} 2 T_MN d_MN, and Sum_{M<N} 2 T_MN d^CSF_MN, from the pairs;
// - the refusals: two roots under kRootGapFloor that T couples (Numerical,
//   naming them), a T that is not antisymmetric, and a D^X inside a diagonal
//   block (InvalidConfig).
// D^X's own term is checked by finite differences in mcscf_overlap_gpu.cpp.
// Measured on the RTX 3080: the pairs within 6.6e-12 (total, the skewed
// (2, 0)), 6.8e-16 (csf) and 4.2e-15 (rhs), the random T within 7.8e-12;
// 2.3 s and 4.3 s a case.
// The Z-vector's residual is printed, not asserted: zvector stops on its
// recursive residual, and the one it returns is recomputed from one more
// product, which can sit just above the 1e-10 it stopped at (1.5e-10 once on
// h2co_ccpvdz_sa2, whose right-hand side here is the pair's over its gap).

constexpr double kCombinationTolerance = 1e-10;

// max |a - b| over max |b|, each element EXPECTed within kCombinationTolerance
// of that.
double expect_relative(std::span<const double> a, std::span<const double> b,
                       std::string_view what) {
  EXPECT_EQ(a.size(), b.size()) << what;
  double scale = 0.0;
  for (const double v : b) {
    scale = std::max(scale, std::abs(v));
  }
  double error = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    EXPECT_NEAR(a[i], b[i], kCombinationTolerance * scale) << what << " " << i;
    error = std::max(error, std::abs(a[i] - b[i]));
  }
  return scale > 0.0 ? error / scale : error;
}

class McscfNacCombination : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfNacCombination, IsTheSumOfThePairs) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h (#977): the skew SA3 formaldehyde, 3.3 s on the RTX 3080.
  if (std::string_view{GetParam()} == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto start_time = std::chrono::steady_clock::now();
  const auto run = converge(entry);
  ASSERT_NE(run, nullptr);
  fb::DeviceResources &res = run->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &molecule = run->calc->get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = run->calc->get<fb::BasisSetOutput>()->basis;
  const auto &classes = run->calc->get<fb::PrimitivePairClassesOutput>()->classes;
  const fb::SphericalMatrixAO &h = run->calc->get<fb::CoreHamiltonianOutput>()->h;
  const auto &df = std::get<fb::DensityFitted>(run->fitted->eri);
  fb::mcscf::Solution &sol = *run->sol;
  const std::int32_t ncore = run->space->ncore;
  const std::int32_t ncas = run->space->ncas;
  const std::int32_t nelecas = run->space->nelecas;
  const auto nroots = static_cast<std::size_t>(sol.roots.energies.size());
  const auto n = static_cast<std::size_t>(ncas);
  const std::size_t m = n * n;

  const auto pair = [&](std::size_t bra, std::size_t ket) {
    return fb::mcscf::df_nac(res, molecule, basis, classes, df.aux, h, df.b, df.l.data(), sol,
                             ncore, ncas, static_cast<std::int32_t>(bra),
                             static_cast<std::int32_t>(ket));
  };
  const auto combination = [&](std::span<const double> t, std::span<const double> d_x = {}) {
    return fb::mcscf::df_nac(res, molecule, basis, classes, df.aux, h, df.b, df.l.data(), sol,
                             ncore, ncas, t, d_x);
  };

  // The pair's right-hand side, as df_nac forms it.
  const std::int32_t two_s = molecule.multiplicity() - 1;
  const std::int32_t n_beta = (nelecas - two_s) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta);
  auto beta = fb::ci::DeviceStringSet::create(res, ncas, n_beta);
  ASSERT_TRUE(alpha.has_value() && beta.has_value());
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  ASSERT_TRUE(rdm.has_value()) << rdm.error().message;
  const auto dim = static_cast<std::size_t>(rdm->size());
  auto ints = fb::casscf_integrals(res, basis, molecule, h, df.b, df.naux, sol.c, ncore, ncas);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;
  fb::Buffers::device<double> rdms(m + (m * m), res.device());
  const auto pair_rhs = [&](std::size_t bra, std::size_t ket) {
    EXPECT_TRUE(rdm->trans_rdm12(res, sol.roots.vectors.data() + (bra * dim),
                                 sol.roots.vectors.data() + (ket * dim), rdms.data(),
                                 rdms.data() + m)
                    .has_value());
    const std::vector<double> gamma = download(res, rdms.data(), m);
    const std::vector<double> gamma2 = download(res, rdms.data() + m, m * m);
    auto g = fb::mcscf::transition_orbital_gradient(res, molecule, h, df.b, df.naux, sol.c, ncore,
                                                    ncas, ints->ppaa, ints->papa, gamma, gamma2);
    EXPECT_TRUE(g.has_value()) << g.error().message;
    return g.has_value() ? download(res, g->data(), g->num_elements()) : std::vector<double>{};
  };

  // Every ordered pair as T = 1/2 (E_IJ - E_JI).
  std::map<std::pair<std::size_t, std::size_t>, fb::mcscf::Nac> pairs;
  double worst_total = 0.0;
  double worst_csf = 0.0;
  double worst_rhs = 0.0;
  for (std::size_t bra = 0; bra < nroots; ++bra) {
    for (std::size_t ket = 0; ket < nroots; ++ket) {
      if (bra == ket) {
        continue;
      }
      const std::string label = std::format("{} ({}, {})", name, bra, ket);
      SCOPED_TRACE(label);
      auto want = pair(bra, ket);
      ASSERT_TRUE(want.has_value()) << want.error().message;
      std::vector<double> t(nroots * nroots, 0.0);
      t[(bra * nroots) + ket] = 0.5;
      t[(ket * nroots) + bra] = -0.5;
      auto got = combination(t);
      ASSERT_TRUE(got.has_value()) << got.error().message;
      const double gap = sol.roots.energies[ket] - sol.roots.energies[bra];
      EXPECT_EQ(got->gap, std::abs(gap));
      EXPECT_TRUE(std::ranges::all_of(got->basis_motion, [](double v) { return v == 0.0; }));
      const double total = expect_relative(got->total, want->total, "total");
      const double csf = expect_relative(got->csf, want->csf, "csf");
      std::vector<double> rhs = got->rhs;
      for (double &v : rhs) {
        v *= gap;
      }
      const double rhs_error = expect_relative(rhs, pair_rhs(bra, ket), "rhs");
      worst_total = std::max(worst_total, total);
      worst_csf = std::max(worst_csf, csf);
      worst_rhs = std::max(worst_rhs, rhs_error);
      std::cout << std::format("[ nac T] {}: total {:.2e}, csf {:.2e}, rhs {:.2e} (relative), "
                               "z-vector {} products to {:.2e} against the pair's {}\n",
                               label, total, csf, rhs_error, got->iterations, got->residual,
                               want->iterations);
      pairs.emplace(std::pair{bra, ket}, std::move(*want));
    }
  }

  // A random antisymmetric T against the pairs' sum.
  if (nroots > 2) {
    std::mt19937_64 engine{742};
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<double> t(nroots * nroots, 0.0);
    std::vector<double> want_total(pairs.begin()->second.total.size(), 0.0);
    std::vector<double> want_csf(want_total.size(), 0.0);
    for (std::size_t i = 0; i < nroots; ++i) {
      for (std::size_t j = i + 1; j < nroots; ++j) {
        const double tij = uniform(engine);
        t[(i * nroots) + j] = tij;
        t[(j * nroots) + i] = -tij;
        const fb::mcscf::Nac &d = pairs.at(std::pair{i, j});
        for (std::size_t x = 0; x < want_total.size(); ++x) {
          want_total[x] += 2.0 * tij * d.total[x];
          want_csf[x] += 2.0 * tij * d.csf[x];
        }
      }
    }
    auto got = combination(t);
    ASSERT_TRUE(got.has_value()) << got.error().message;
    const double total = expect_relative(got->total, want_total, "random T total");
    const double csf = expect_relative(got->csf, want_csf, "random T csf");
    std::cout << std::format("[ nac T] {} random T: total {:.2e}, csf {:.2e} (relative), "
                             "smallest gap {:.6f}, z-vector {} products to {:.2e}\n",
                             name, total, csf, got->gap, got->iterations, got->residual);
  }

  // The refusals. Two roots 1e-9 apart that T couples: Numerical, naming
  // them, before anything is built.
  {
    std::vector<double> t(nroots * nroots, 0.0);
    t[1] = 0.5;
    t[nroots] = -0.5;
    const double saved = sol.roots.energies[1];
    sol.roots.energies[1] = sol.roots.energies[0] + 1e-9;
    const auto refused = combination(t);
    sol.roots.energies[1] = saved;
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;
    EXPECT_NE(refused.error().message.find("mcscf::df_nac"), std::string::npos);
    EXPECT_NE(refused.error().message.find("roots 0 and 1"), std::string::npos)
        << refused.error().message;
    std::cout << std::format("[ nac T] {} refuses: {}\n", name, refused.error().message);

    // T not antisymmetric.
    t[nroots] = 0.25;
    const auto skewless = combination(t);
    ASSERT_FALSE(skewless.has_value());
    EXPECT_EQ(skewless.error().kind, fb::ErrorKind::InvalidConfig) << skewless.error().message;

    // D^X inside the active-active block.
    t[nroots] = -0.5;
    const std::size_t nmo = sol.c.n();
    std::vector<double> d_x(nmo * nmo, 0.0);
    const auto t0 = static_cast<std::size_t>(ncore);
    d_x[(t0 * nmo) + t0 + 1] = 1.0;
    const auto in_block = combination(t, d_x);
    ASSERT_FALSE(in_block.has_value());
    EXPECT_EQ(in_block.error().kind, fb::ErrorKind::InvalidConfig) << in_block.error().message;
  }

  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ nac T] {}: worst total {:.2e}, csf {:.2e}, rhs {:.2e} (relative), "
                           "{:.2f} s in all\n",
                           name, worst_total, worst_csf, worst_rhs, seconds.count());
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfNacCombination,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
