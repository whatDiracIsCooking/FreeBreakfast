// mcscf::df_gradient on the device: this tree's own DF-CASSCF
// (mcscf::solve under DensityFitting::Full over the case's fitting set, from
// the exact RHF and the case's active orbitals) converged tightly, then its
// nuclear gradient, every term PySCF records and the total, against
// df.grad.casscf in data/casscf_gradient/casscf_gradient.json
// (tools/emit/casscf_gradient_reference.py) to 1e-8 absolute, and the
// total's sum over atoms below 1e-9; for each state-specific case (H2O and the
// skewed H2O, cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4)). And the refusals: a root the
// solution does not have, and a state-averaged solution.
//
// The setup is mcscf_solve_gpu.cpp's: the setup steps through the real
// registry, the SCF run by hand (kExactScf), and the start picked by the Casci
// step's selection. So this executable links pipeline; mcscf does not. Runs the
// device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.mcscf;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars: every term and the total within 1e-8 of PySCF, and the
// total's sum over atoms below 1e-9.
constexpr double kTermTolerance = 1e-8;
constexpr double kSumTolerance = 1e-9;
// The energy against PySCF's, as mcscf_solve_gpu.cpp holds the DF-CASSCF's.
constexpr double kEnergyTolerance = 1e-8;
// The solve's convergence. The gradient formula takes W for the orbitals'
// response, so its error is first order in the orbital gradient: ~||g|| times
// the orbitals' change per bohr, ~1e-9 here at the bar's tenth.
constexpr double kConvTolGrad = 1e-9;

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : doc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_gradient.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule through the real registry to the Scf step's
// inputs, as mcscf_solve_gpu.cpp's run_setup_steps.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_gradient");
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

// @p ours against the reference rows @p ref to kTermTolerance absolute; the
// worst error, for the record a PR reports.
double compare(const std::string &what, const std::vector<double> &ours, const fb::json &ref) {
  const auto rows = ref.get<std::vector<std::vector<double>>>();
  EXPECT_EQ(ours.size(), 3 * rows.size()) << what;
  if (ours.size() != 3 * rows.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(ours[(3 * a) + k], rows[a][k], kTermTolerance)
          << what << " [" << a << "][" << k << "]";
      worst = std::max(worst, std::abs(ours[(3 * a) + k] - rows[a][k]));
    }
  }
  return worst;
}

class McscfGradient : public ::testing::TestWithParam<std::string_view> {};

// The case's DF-CASSCF, its gradient against PySCF's term by term, the sum
// over atoms, and the refusals on the same solution.
TEST_P(McscfGradient, DfCasscfMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  ASSERT_EQ(entry.at("weights").size(), 1U) << "a state-specific case";
  const auto start = std::chrono::steady_clock::now();

  fb::NodeResources node;
  auto calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
  fb::DeviceResources &res = node.device(0);
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  const auto &transform = calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &x = calc->get<fb::OrthogonalizerOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = calc->get<fb::GuessOutput>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && transform.has_value() &&
              pairs.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
              guess.has_value())
      << "a product of the setup steps is missing";
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  ASSERT_EQ(basis->basis.nao_sphe(), entry.at("nao").get<std::size_t>());

  // The exact RHF, which the reference's start is made from too.
  auto direct = fb::EriStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          *molecule, *basis, *pairs},
      res);
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis, *transform, *direct, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto k_classes = fb::ordered_pair_classes(res, basis->basis, pairs->classes);
  ASSERT_TRUE(j_classes.has_value() && k_classes.has_value());

  fb::CasciStep step;
  step.ncas = entry.at("ncas").get<std::int32_t>();
  step.nelecas = entry.at("nelecas").get<std::int32_t>();
  step.active = entry.at("active").get<std::vector<std::int32_t>>();
  auto space = step.select(res, molecule->molecule, basis->basis, *scf);
  ASSERT_TRUE(space.has_value()) << space.error().message;
  ASSERT_EQ(space->ncore, entry.at("ncore").get<std::int32_t>());

  // B, L and the placed aux basis over the reference's set.
  const fb::json &aux = entry.at("aux_basis");
  auto fitted =
      fb::EriStep{.mode = fb::EriMode::DensityFitting,
                  .aux_basis = aux.at("name").get<std::string>()}
          .run(
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
                  *molecule, *basis, *pairs},
              res);
  ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
  const auto &df = std::get<fb::DensityFitted>(fitted->eri);
  ASSERT_EQ(df.naux, aux.at("naux").get<std::int32_t>());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis->basis, molecule->molecule, h->h, transform->u, *j_classes,
                              *k_classes, space->c, space->ncore, space->ncas, space->nelecas, 1,
                              {}, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
  EXPECT_NEAR(sol->energy, entry.at("energy").get<double>(), kEnergyTolerance);

  const auto g =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, 0);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;

  const fb::json &want = entry.at("roots").at(0);
  const fb::json &parts = want.at("parts");
  std::string report = std::format("[ casgrad  ] {} ({} macro iterations, {:.2f} s):", name,
                                   sol->iterations.size(), seconds.count());
  for (const auto &[key, ours] : {std::pair{"kinetic", &g->kinetic},
                                  {"potential", &g->potential},
                                  {"overlap", &g->overlap},
                                  {"two_electron", &g->two_electron},
                                  {"nuclear_repulsion", &g->nuclear_repulsion},
                                  {"total", &g->total}}) {
    const double worst =
        compare(name + " " + key, *ours,
                key == std::string_view{"total"} ? want.at("total") : parts.at(key));
    report += std::format(" {} {:.2e}", key, worst);
  }
  std::array<double, 3> sum{};
  for (std::size_t i = 0; i < g->total.size(); ++i) {
    sum[i % 3] += g->total[i];
  }
  const double worst_sum = std::ranges::max(sum, {}, [](double s) { return std::abs(s); });
  EXPECT_LT(std::abs(worst_sum), kSumTolerance) << name << " translational sum";
  std::cout << report << std::format(" sum {:.2e}\n", std::abs(worst_sum));

  // A root the solution does not have is InvalidConfig; a solution averaging
  // two roots is Unsupported until the state-averaged gradient lifts it.
  const auto past =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, 1);
  ASSERT_FALSE(past.has_value());
  EXPECT_EQ(past.error().kind, fb::ErrorKind::InvalidConfig) << past.error().message;
  EXPECT_NE(past.error().message.find("mcscf::df_gradient"), std::string::npos);
  sol->roots.energies.push_back(sol->roots.energies.front());
  const auto averaged =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, 0);
  ASSERT_FALSE(averaged.has_value());
  EXPECT_EQ(averaged.error().kind, fb::ErrorKind::Unsupported) << averaged.error().message;
  EXPECT_NE(averaged.error().message.find("multipliers"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfGradient,
                         ::testing::Values(std::string_view{"h2o_ccpvdz"},
                                           std::string_view{"h2o_skew_ccpvdz"}),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
