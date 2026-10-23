// The Scf step against PySCF's RHF (data/scf/rhf.json, tools/emit_scf_reference.py):
// the energy and orbital energies, and the invariants of a closed-shell
// wavefunction, over the setup steps and the guess run by hand; then through
// the real registry from a config naming only [scf]. Runs the device chain, so
// needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

using ScfInput =
    fb::View<fb::Molecule, fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses,
             fb::Overlap, fb::Orthogonalizer, fb::CoreHamiltonian, fb::Guess>;

static_assert(fb::PipelineStep<fb::ScfStep>);
static_assert(fb::kRunShape<fb::ScfStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::ScfStep>, ScfInput>);
static_assert(std::same_as<fb::StepOutput<fb::ScfStep>, fb::Scf>);

// Against PySCF converged to conv_tol 1e-12. What is left is J's and K's
// screening and the step's own tolerances.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOrbitalEnergyTolerance = 1e-6;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "scf" / "rhf.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return doc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : reference().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in scf/rhf.json";
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

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// Column-major n x n products on the host: small enough to check exactly.
std::vector<double> matmul(const std::vector<double> &a, const std::vector<double> &b,
                           std::size_t n, bool transpose_a = false) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      const double bkj = b[k + (n * j)];
      for (std::size_t i = 0; i < n; ++i) {
        c[i + (n * j)] += (transpose_a ? a[k + (n * i)] : a[i + (n * k)]) * bkj;
      }
    }
  }
  return c;
}

// Every product the Scf step reads, made by the steps called by hand in the
// order the planner would choose.
struct Inputs {
  fb::Molecule molecule;
  fb::BasisSet basis;
  fb::SphericalTransform transform;
  fb::PrimitivePairClasses pairs;
  fb::Overlap overlap;
  fb::Orthogonalizer x;
  fb::CoreHamiltonian h;
  fb::Guess guess;

  [[nodiscard]] ScfInput view() const {
    return ScfInput{molecule, basis, transform, pairs, overlap, x, h, guess};
  }
};

std::optional<Inputs> make_inputs(fb::DeviceResources &res, std::vector<fb::Atom> atoms,
                                  std::string_view basis_name, std::int32_t charge,
                                  std::int32_t multiplicity = 1) {
  fb::Molecule molecule(res, std::move(atoms), charge, multiplicity);
  auto basis =
      fb::BasisSetStep{.basis = std::string(basis_name)}.run(fb::View<fb::Molecule>{molecule});
  if (!basis.has_value()) {
    ADD_FAILURE() << basis.error().message;
    return std::nullopt;
  }
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSet>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSet>{*basis}, res);
  if (!transform.has_value() || !pairs.has_value()) {
    ADD_FAILURE() << "spherical transform or pair classes failed";
    return std::nullopt;
  }
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses>{*basis, *transform,
                                                                               *pairs},
      res);
  if (!overlap.has_value()) {
    ADD_FAILURE() << overlap.error().message;
    return std::nullopt;
  }
  auto x = fb::OrthogonalizerStep{}.run(fb::View<fb::BasisSet, fb::Overlap>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::Molecule, fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses>{
          molecule, *basis, *transform, *pairs},
      res);
  if (!x.has_value() || !h.has_value()) {
    ADD_FAILURE() << "orthogonalizer or core Hamiltonian failed";
    return std::nullopt;
  }
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::Guess> guess = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::Molecule, fb::BasisSet>{molecule, *basis}, subflow);
  });
  executor.run(flow).wait();
  if (!guess.has_value()) {
    ADD_FAILURE() << guess.error().message;
    return std::nullopt;
  }
  return Inputs{std::move(molecule), std::move(*basis), std::move(*transform), std::move(*pairs),
                std::move(*overlap), std::move(*x),     std::move(*h),         std::move(*guess)};
}

class PipelineScf : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineScf, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case(GetParam());
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(expected), expected.at("basis").get<std::string>(),
                            expected.at("charge").get<std::int32_t>());
  ASSERT_TRUE(inputs.has_value());
  const std::size_t n = inputs->basis.nao_sphe();
  ASSERT_EQ(n, expected.at("nao").get<std::size_t>());

  auto scf = fb::ScfStep{}.run(inputs->view(), res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  EXPECT_NEAR(fb::nuclear_repulsion_energy(inputs->molecule),
              expected.at("nuclear_repulsion").get<double>(), 1e-10);
  EXPECT_NEAR(scf->energy, expected.at("energy").get<double>(), kEnergyTolerance);
  const auto mo_energies = expected.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << i;
  }
  EXPECT_LT(scf->iterations, fb::ScfStep{}.max_iterations);

  // The invariants of the closed-shell wavefunction it returned, whatever the
  // AO normalisation: C^T S C = I, tr(D S) = N and D S D = 2 D.
  const std::vector<double> s = download(res, inputs->overlap.s);
  const std::vector<double> c = download(res, scf->c);
  const std::vector<double> d = download(res, scf->d);
  const std::vector<double> ctsc = matmul(c, matmul(s, c, n), n, /*transpose_a=*/true);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(ctsc[i + (n * j)], i == j ? 1.0 : 0.0, 1e-10) << i << "," << j;
    }
  }
  double trace = 0.0;
  for (std::size_t k = 0; k < n * n; ++k) {
    trace += d[k] * s[k]; // both symmetric
  }
  EXPECT_NEAR(trace, expected.at("electrons").get<double>(), 1e-10);
  const std::vector<double> dsd = matmul(d, matmul(s, d, n), n);
  for (std::size_t k = 0; k < n * n; ++k) {
    EXPECT_NEAR(dsd[k], 2.0 * d[k], 1e-10) << k;
  }
}

INSTANTIATE_TEST_SUITE_P(Rhf, PipelineScf,
                         ::testing::Values("h2o_sto3g", "h2o_ccpvdz", "nh4_ccpvdz"),
                         [](const auto &info) { return std::string(info.param); });

TEST(PipelineScfErrors, AnOpenShellIsUnsupported) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", 0, 3);
  ASSERT_TRUE(inputs.has_value());
  auto scf = fb::ScfStep{}.run(inputs->view(), res);
  ASSERT_FALSE(scf.has_value());
  EXPECT_EQ(scf.error().kind, fb::ErrorKind::Unsupported);
}

TEST(PipelineScfErrors, TooFewIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", 0);
  ASSERT_TRUE(inputs.has_value());
  auto scf = fb::ScfStep{.max_iterations = 2}.run(inputs->view(), res);
  ASSERT_FALSE(scf.has_value());
  EXPECT_EQ(scf.error().kind, fb::ErrorKind::MaxIterationsExceeded);
}

TEST(PipelineScfErrors, AnOptionOutOfRangeIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto inputs = make_inputs(res, atoms_of(reference_case("h2o_sto3g")), "sto-3g", 0);
  ASSERT_TRUE(inputs.has_value());
  for (const fb::ScfStep step :
       {fb::ScfStep{.max_iterations = 0}, fb::ScfStep{.diis_history = 0},
        fb::ScfStep{.energy_tolerance = -1.0}, fb::ScfStep{.commutator_tolerance = std::nan("")}}) {
    auto scf = step.run(inputs->view(), res);
    ASSERT_FALSE(scf.has_value());
    EXPECT_EQ(scf.error().kind, fb::ErrorKind::InvalidConfig) << scf.error().message;
  }
}

TEST(PipelineScfCalculation, AConfigNamingOnlyScfPlansAndRunsTheWholeChain) {
  FBF_REQUIRE_GPU();
  const fb::json &expected = reference_case("h2o_ccpvdz");
  // The reference geometry is in bohr; XYZ is in Angstrom, read back through
  // the same constant, so the molecule is the reference's to rounding.
  std::string xyz = std::format("{}\nwater\n", expected.at("atoms").size());
  for (const fb::json &atom : expected.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  const fb::test::ScratchDir dir("pipeline_scf");
  ASSERT_TRUE(fb::write_file(dir / "water.xyz", xyz).has_value());
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             "[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n"
                             "[scf]\n")
                  .has_value());

  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_TRUE(calc->get<fb::Scf>().has_value());
  EXPECT_NEAR(calc->get<fb::Scf>()->energy, expected.at("energy").get<double>(), kEnergyTolerance);
}

} // namespace
