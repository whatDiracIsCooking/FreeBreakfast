// GPU cases for gradient::df_rhf, the closed-shell DF-RHF nuclear gradient:
// this tree's own DF-RHF (the fit built by hand, so L is kept, then the Scf
// step converged tight), its gradient total and each term against PySCF's
// df.grad.rhf over the same orbital and fitting sets
// (data/gradient/gradient.json's df_rhf, test/gtest/data/README.md), for H2O,
// NH4+ and a distorted H2O in cc-pVDZ / cc-pVDZ-JKFIT, H2O in cc-pVDZ /
// cc-pVDZ-acCD and in cc-pVTZ / cc-pVTZ-JKFIT; the total's sum over atoms; a
// central finite difference of the DF-RHF energy; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.molecule;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.spherical_transform;
import free_breakfast.pipeline.primitive_pair_classes;
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
// The energy against PySCF's, as pipeline_df_scf_gpu.cpp holds it.
constexpr double kEnergyTolerance = 1e-8;

const std::filesystem::path kDataFile =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient" / "gradient.json";

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = [] {
    auto read = fb::read_json(kDataFile);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : doc.at("df_rhf")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no df_rhf case " << name;
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

// A converged DF-RHF and everything its gradient reads: the setup steps run
// by hand, the fit built as the Eri step builds it but with L kept, and the
// Scf step converged well past its defaults, since W stands in for the
// orbitals' response only at convergence.
struct DfRhf {
  fb::MoleculeOutput molecule;
  fb::BasisSetOutput basis;
  fb::PrimitivePairClassesOutput pairs;
  fb::BasisSet aux;
  fb::MatrixAOPair metric; // L in its spherical member's lower triangle
  fb::EriOutput eri;
  fb::ScfOutput scf;

  [[nodiscard]] const double *b() const { return std::get<fb::DensityFitted>(eri.eri).b.data(); }
  [[nodiscard]] const double *l() const { return metric.spherical().data(); }

  [[nodiscard]] fb::Result<fb::gradient::RhfTerms> gradient(fb::DeviceResources &res) const {
    return fb::gradient::df_rhf(res, molecule.molecule, basis.basis, pairs.classes, aux, b(), l(),
                                scf.c, scf.mo_energies, scf.n_alpha);
  }
};

std::optional<DfRhf> run_df_rhf(fb::DeviceResources &res, std::vector<fb::Atom> atoms,
                                const std::string &basis_name, const std::string &aux_name,
                                const std::int32_t charge) {
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

  // pipeline/eri's density_fitted, keeping the metric's factor.
  const auto stamp = fb::resolve_aux_basis(aux_name);
  if (!stamp.has_value()) {
    ADD_FAILURE() << stamp.error().message;
    return std::nullopt;
  }
  auto aux = fb::place(stamp->primitive_normalized(), molecule.molecule);
  auto u = fb::upload_spherical_transform_matrix(res, *aux);
  if (!aux.has_value() || !u.has_value()) {
    ADD_FAILURE() << "aux basis placement or its transform failed";
    return std::nullopt;
  }
  const std::size_t nao = basis->basis.nao_sphe();
  const std::size_t naux = aux->nao_sphe();
  fb::MatrixAOPair metric(res, *aux, std::move(*u));
  fb::Buffers::device<double> b(naux * nao * nao, res.device());
  if (const fb::Status s = fb::metric(res, *aux, metric); !s.has_value()) {
    ADD_FAILURE() << s.error().message;
    return std::nullopt;
  }
  if (const fb::Status s = fb::three_center(res, basis->basis, pairs->classes, *aux, b.data());
      !s.has_value()) {
    ADD_FAILURE() << s.error().message;
    return std::nullopt;
  }
  if (const fb::Status s =
          fb::density_fit(res, static_cast<std::int32_t>(nao), static_cast<std::int32_t>(naux),
                          metric.spherical().data(), b.data());
      !s.has_value()) {
    ADD_FAILURE() << s.error().message;
    return std::nullopt;
  }
  // The product's own copy of L and the aux basis, as the Eri step keeps
  // them; the gradient below reads the metric's.
  fb::Buffers::device<double> l(naux * naux, res.device());
  if (wwr::wwrMemcpyAsync(l.data(), metric.spherical().data(), sizeof(double) * naux * naux,
                          wwr::wwrMemcpyDeviceToDevice, res.stream()) != wwr::wwrSuccess) {
    ADD_FAILURE() << "copying L failed";
    return std::nullopt;
  }
  fb::EriOutput eri{fb::DensityFitted{
      .b = std::move(b), .naux = static_cast<std::int32_t>(naux), .aux = *aux, .l = std::move(l)}};

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

  const fb::ScfStep step{.energy_tolerance = 1e-12, .commutator_tolerance = 1e-10};
  auto scf = step.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput, fb::EriOutput,
               fb::OverlapOutput, fb::OrthogonalizerOutput, fb::CoreHamiltonianOutput,
               fb::GuessOutput>{molecule, *basis, *transform, eri, *overlap, *x, *h, *guess},
      res);
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return std::nullopt;
  }
  return DfRhf{std::move(molecule), std::move(*basis), std::move(*pairs), std::move(*aux),
               std::move(metric),   std::move(eri),    std::move(*scf)};
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
  const auto calc = run_df_rhf(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                               expected.at("aux_basis").get<std::string>(),
                               expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(calc.has_value());
  ASSERT_EQ(calc->basis.basis.nao_sphe(), expected.at("nao").get<std::size_t>());
  ASSERT_EQ(calc->aux.nao_sphe(), expected.at("naux").get<std::size_t>());
  EXPECT_NEAR(calc->scf.energy, expected.at("energy").get<double>(), kEnergyTolerance);

  const auto g = calc->gradient(res);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const fb::json &terms = expected.at("terms");
  std::string report = std::format("[ dfrhfgrad ] {}:", name);
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

class DfRhfGradient : public ::testing::TestWithParam<std::string_view> {};

TEST_P(DfRhfGradient, MatchesPyscfTermByTerm) {
  FBF_REQUIRE_GPU();
  check_case(std::string(GetParam()));
}

// H2O and NH4+ (charged) in cc-pVDZ / cc-pVDZ-JKFIT, the distorted H2O
// (every component above 1e-3), H2O in cc-pVDZ / cc-pVDZ-acCD, and H2O in
// cc-pVTZ / cc-pVTZ-JKFIT: f orbitals, g fitting functions. Each about a
// second or less on an RTX 3080, so none is under FBF_SKIP_IF_FAST_TESTS.
INSTANTIATE_TEST_SUITE_P(DfRhf, DfRhfGradient,
                         ::testing::Values("h2o_ccpvdz", "nh4_ccpvdz", "h2o_distorted_ccpvdz",
                                           "h2o_ccpvdz_accd", "h2o_ccpvtz"),
                         [](const auto &info) { return std::string(info.param); });

// dE/dy of H2O's first hydrogen in cc-pVDZ / cc-pVDZ-JKFIT against a central
// difference of this tree's own DF-RHF energy.
TEST(DfRhfGradientFiniteDifference, MatchesACentralDifferenceOfTheEnergy) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  constexpr std::size_t kAtom = 1;
  constexpr std::size_t kAxis = 1; // y
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto basis = expected.at("basis").get<std::string>();
  const auto aux = expected.at("aux_basis").get<std::string>();
  const auto calc = run_df_rhf(res, atoms_of(expected), basis, aux, 0);
  ASSERT_TRUE(calc.has_value());
  const auto g = calc->gradient(res);
  ASSERT_TRUE(g.has_value()) << g.error().message;

  std::array<double, 2> energies{};
  for (std::size_t side = 0; side < 2; ++side) {
    std::vector<fb::Atom> atoms = atoms_of(expected);
    atoms[kAtom].position.y += side == 0 ? kStep : -kStep;
    const auto shifted = run_df_rhf(res, std::move(atoms), basis, aux, 0);
    ASSERT_TRUE(shifted.has_value());
    energies[side] = shifted->scf.energy;
  }
  const double difference = (energies[0] - energies[1]) / (2.0 * kStep);
  const double analytic = g->total[(3 * kAtom) + kAxis];
  EXPECT_NEAR(analytic, difference, kFiniteDifferenceTolerance);
  std::cout << std::format("[ dfrhfgrad ] finite difference: analytic {:.10f}, central "
                           "difference {:.10f}, |diff| {:.2e}\n",
                           analytic, difference, std::abs(analytic - difference));
}

TEST(DfRhfGradientRefusals, RefusesAnOccupationOutsideTheOrbitalsAndMismatchedEnergies) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto calc = run_df_rhf(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                               expected.at("aux_basis").get<std::string>(), 0);
  ASSERT_TRUE(calc.has_value());
  const auto call = [&](std::span<const double> energies, std::int32_t nocc) {
    return fb::gradient::df_rhf(res, calc->molecule.molecule, calc->basis.basis,
                                calc->pairs.classes, calc->aux, calc->b(), calc->l(), calc->scf.c,
                                energies, nocc);
  };
  for (const std::int32_t nocc : {-1, 25}) {
    const auto g = call(calc->scf.mo_energies, nocc);
    ASSERT_FALSE(g.has_value()) << nocc;
    EXPECT_EQ(g.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(g.error().message.find("occupied orbitals"), std::string::npos) << g.error().message;
  }
  const auto g = call(std::span{calc->scf.mo_energies}.first(3), 5);
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(g.error().message.find("orbital energies"), std::string::npos) << g.error().message;
}

} // namespace
