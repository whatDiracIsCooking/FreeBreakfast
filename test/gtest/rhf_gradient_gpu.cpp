// GPU cases for gradient::rhf, the closed-shell integral-direct RHF nuclear
// gradient: this tree's own integral-direct RHF (the Scf step converged tight,
// J unscreened), its gradient total and each term against PySCF's
// scf.grad.rhf (data/gradient/gradient.json's rhf, test/gtest/data/README.md),
// for H2O and a distorted H2O in cc-pVDZ and H2O in cc-pVTZ; the total's sum
// over atoms; and a central finite difference of the RHF energy.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.molecule;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.spherical_transform;
import free_breakfast.pipeline.primitive_pair_classes;
import free_breakfast.pipeline.eri;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.scf;
import free_breakfast.gradient;

namespace {

namespace fb = free_breakfast;

// The issue's bars: every term and the total within 1e-8 of PySCF, the
// total's sum over atoms below 1e-9, and the finite difference within ~1e-6.
constexpr double kTermTolerance = 1e-8;
constexpr double kSumTolerance = 1e-9;
constexpr double kFiniteDifferenceTolerance = 1e-6;
// The step, in bohr. The SCF is converged to 1e-12 Eh, so its noise in the
// difference is ~1e-8, and the step's own O(h^2) error is smaller still.
constexpr double kStep = 1e-4;
// The energy against PySCF's.
constexpr double kEnergyTolerance = 1e-8;

const std::filesystem::path kDataFile =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient" / "gradient.json";

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = [] {
    auto read = fb::read_json(kDataFile);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : doc.at("rhf")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no rhf case " << name;
  static const fb::json kNone;
  return kNone;
}

std::vector<fb::Atom> atoms_of(const fb::json &entry) {
  std::vector<fb::Atom> atoms;
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    atoms.push_back(fb::Atom{{coord[0], coord[1], coord[2]},
                             static_cast<fb::Element>(atom.at("z").get<std::int32_t>())});
  }
  return atoms;
}

// A converged integral-direct RHF and everything its gradient reads: the
// setup steps and the Eri step run by hand, then the Scf step converged well
// past its defaults with J unscreened, since W stands in for the orbitals'
// response only at convergence.
struct Rhf {
  fb::MoleculeOutput molecule;
  fb::BasisSetOutput basis;
  fb::PrimitivePairClassesOutput pairs;
  fb::EriOutput eri;
  fb::ScfOutput scf;

  [[nodiscard]] fb::Result<fb::gradient::RhfTerms> gradient(fb::DeviceResources &res) const {
    return fb::gradient::rhf(res, molecule.molecule, basis.basis, pairs.classes,
                             std::get<fb::IntegralDirect>(eri.eri).k_classes, scf.c,
                             scf.mo_energies, scf.n_alpha);
  }
};

std::optional<Rhf> run_rhf(fb::DeviceResources &res, std::vector<fb::Atom> atoms,
                           const std::string &basis_name, const std::int32_t charge) {
  fb::MoleculeOutput molecule{fb::Molecule(res, std::move(atoms), charge)};
  auto basis = fb::BasisSetStep{.basis = basis_name}.run(fb::View<fb::MoleculeOutput>{molecule});
  if (!basis.has_value()) {
    ADD_FAILURE() << basis.error().message;
    return std::nullopt;
  }
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  if (!transform.has_value() || !pairs.has_value()) {
    ADD_FAILURE() << "spherical transform or pair classes failed";
    return std::nullopt;
  }
  auto eri = fb::EriStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          molecule, *basis, *pairs},
      res);
  if (!eri.has_value()) {
    ADD_FAILURE() << eri.error().message;
    return std::nullopt;
  }
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          *basis, *transform, *pairs},
      res);
  if (!overlap.has_value()) {
    ADD_FAILURE() << overlap.error().message;
    return std::nullopt;
  }
  auto x = fb::OrthogonalizerStep{}.run(
      fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
               fb::PrimitivePairClassesOutput>{molecule, *basis, *transform, *pairs},
      res);
  if (!x.has_value() || !h.has_value()) {
    ADD_FAILURE() << "orthogonalizer or core Hamiltonian failed";
    return std::nullopt;
  }
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::GuessOutput> guess = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{molecule, *basis},
                                subflow);
  });
  executor.run(flow).wait();
  if (!guess.has_value()) {
    ADD_FAILURE() << guess.error().message;
    return std::nullopt;
  }

  const fb::ScfStep step{.energy_tolerance = 1e-12, .commutator_tolerance = 1e-10, .tau_j = 0.0};
  auto scf = step.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput, fb::EriOutput,
               fb::OverlapOutput, fb::OrthogonalizerOutput, fb::CoreHamiltonianOutput,
               fb::GuessOutput>{molecule, *basis, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return std::nullopt;
  }
  return Rhf{std::move(molecule), std::move(*basis), std::move(*pairs), std::move(*eri),
             std::move(*scf)};
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

void check_case(const std::string &name) {
  const fb::json &expected = reference_case(name);
  const auto start = std::chrono::steady_clock::now();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto calc = run_rhf(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(calc.has_value());
  ASSERT_EQ(calc->basis.basis.nao_sphe(), expected.at("nao").get<std::size_t>());
  EXPECT_NEAR(calc->scf.energy, expected.at("energy").get<double>(), kEnergyTolerance);

  const auto g = calc->gradient(res);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const fb::json &terms = expected.at("terms");
  std::string report = std::format("[ rhfgrad ] {}:", name);
  for (const auto &[key, ours] : {std::pair{"kinetic", &g->kinetic},
                                  {"potential", &g->potential},
                                  {"overlap", &g->overlap},
                                  {"coulomb", &g->coulomb},
                                  {"exchange", &g->exchange},
                                  {"nuclear_repulsion", &g->nuclear_repulsion},
                                  {"total", &g->total}}) {
    const double worst =
        compare(name + " " + key, *ours,
                key == std::string_view{"total"} ? expected.at("total") : terms.at(key));
    report += std::format(" {} {:.2e}", key, worst);
  }
  std::array<double, 3> sum{};
  for (std::size_t i = 0; i < g->total.size(); ++i) {
    sum[i % 3] += g->total[i];
  }
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_LE(std::abs(sum[k]), kSumTolerance) << name << ": SUM_A G[A][" << k << "] = " << sum[k];
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << report
            << std::format(", max |SUM_A G| {:.2e}, {} SCF iterations, {:.1f} s\n",
                           std::max({std::abs(sum[0]), std::abs(sum[1]), std::abs(sum[2])}),
                           calc->scf.iterations, elapsed);
}

class RhfGradient : public ::testing::TestWithParam<std::string_view> {};

TEST_P(RhfGradient, MatchesPyscfTermByTerm) {
  FBF_REQUIRE_GPU();
  // cc-pVTZ is about 30 s on the RX 9060 XT, nearly all of it the
  // integral-direct SCF: past slow_test.h's ten, and not the family's
  // cheapest.
  if (GetParam() == "h2o_ccpvtz") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  check_case(std::string(GetParam()));
}

// H2O and the distorted H2O (every component above 1e-3) in cc-pVDZ, about a
// second each, and H2O in cc-pVTZ: f orbitals, the K gradient through (ff|ff)
// unscreened.
INSTANTIATE_TEST_SUITE_P(Rhf, RhfGradient,
                         ::testing::Values("h2o_ccpvdz", "h2o_distorted_ccpvdz", "h2o_ccpvtz"),
                         [](const auto &info) { return std::string(info.param); });

// dE/dy of H2O's first hydrogen in cc-pVDZ against a central difference of
// this tree's own integral-direct RHF energy.
TEST(RhfGradientFiniteDifference, MatchesACentralDifferenceOfTheEnergy) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  constexpr std::size_t kAtom = 1;
  constexpr std::size_t kAxis = 1; // y
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto basis = expected.at("basis").get<std::string>();
  const auto calc = run_rhf(res, atoms_of(expected), basis, 0);
  ASSERT_TRUE(calc.has_value());
  const auto g = calc->gradient(res);
  ASSERT_TRUE(g.has_value()) << g.error().message;

  std::array<double, 2> energies{};
  for (std::size_t side = 0; side < 2; ++side) {
    std::vector<fb::Atom> atoms = atoms_of(expected);
    atoms[kAtom].position.y += side == 0 ? kStep : -kStep;
    const auto shifted = run_rhf(res, std::move(atoms), basis, 0);
    ASSERT_TRUE(shifted.has_value());
    energies[side] = shifted->scf.energy;
  }
  const double difference = (energies[0] - energies[1]) / (2.0 * kStep);
  const double analytic = g->total[(3 * kAtom) + kAxis];
  EXPECT_NEAR(analytic, difference, kFiniteDifferenceTolerance);
  std::cout << std::format("[ rhfgrad ] finite difference: analytic {:.10f}, central "
                           "difference {:.10f}, |diff| {:.2e}\n",
                           analytic, difference, std::abs(analytic - difference));
}

} // namespace
