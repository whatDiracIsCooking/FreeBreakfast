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
// mult_ediff against h + (E_J - E_I) csf, and both against `h`. And the
// refusals: equal roots and a root past the solution's. PySCF refuses
// unequal weights, so df_nac is checked there against finite differences
// of mcscf::state_overlaps, in mcscf_overlap_gpu.cpp.
// Measured on the RTX 3080: h_IJ within 4e-10, d^CSF 2.2e-10, d_IJ 1.3e-9
// (the skewed (0, 2)); 3.3 s and 4.6 s a case, so neither is a slow test.
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
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

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

class McscfNac : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfNac, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  const auto energies = entry.at("energies").get<std::vector<double>>();
  const auto start_time = std::chrono::steady_clock::now();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  std::optional<fb::Calculation> calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis_set = calc->get<fb::BasisSetOutput>();
  const auto &transform = calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &x = calc->get<fb::OrthogonalizerOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = calc->get<fb::GuessOutput>();
  ASSERT_TRUE(molecule.has_value() && basis_set.has_value() && transform.has_value() &&
              pairs.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
              guess.has_value());
  const fb::BasisSet &basis = basis_set->basis;
  ASSERT_EQ(basis.nao_sphe(), entry.at("nao").get<std::size_t>());
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs};
  auto eri = fb::EriStep{}.run(view, res);
  ASSERT_TRUE(eri.has_value()) << eri.error().message;
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  const fb::json &aux = entry.at("aux_basis");
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = aux.at("name").get<std::string>()}
                    .run(view, res);
  ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
  const auto &df = std::get<fb::DensityFitted>(fitted->eri);
  ASSERT_EQ(df.naux, aux.at("naux").get<std::int32_t>());
  fb::CasciStep casci;
  casci.ncas = entry.at("ncas").get<std::int32_t>();
  casci.nelecas = entry.at("nelecas").get<std::int32_t>();
  casci.nroots = nroots;
  auto space = casci.select(res, molecule->molecule, basis, *scf);
  ASSERT_TRUE(space.has_value()) << space.error().message;
  const std::int32_t ncore = space->ncore;
  const std::int32_t ncas = space->ncas;
  ASSERT_EQ(ncore, entry.at("ncore").get<std::int32_t>());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol =
      fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {}, space->c, ncore,
                       ncas, space->nelecas, nroots, weights, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(sol->roots.energies[r], energies[r], kEnergyTolerance) << "root " << r;
  }

  const auto nac = [&](std::int32_t bra, std::int32_t ket, fb::mcscf::NacOptions nac_options) {
    return fb::mcscf::df_nac(res, molecule->molecule, basis, pairs->classes, df.aux, h->h, df.b,
                             df.l.data(), *sol, ncore, ncas, bra, ket, nac_options);
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

} // namespace
