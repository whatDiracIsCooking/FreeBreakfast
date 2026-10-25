// The real registry end to end: a config file read, planned and run by
// Calculation, against the same steps called by hand in the order the planner
// should choose. Runs the device chain, with the host guess beside it, so needs
// a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "io_basis_json.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWaterXyz = R"(3
water
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
)";

// Only what main prints past the basis is asked for; the transform, the
// pair classes and the overlap are filled in.
constexpr std::string_view kConfig = R"(
[molecule]
xyz = "water.xyz"

[basis_set]
basis = "basis.json"

[orthogonalizer]

[core_hamiltonian]

[guess]
)";

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

void expect_near(const std::vector<double> &actual, const std::vector<double> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-12) << i;
  }
}

// The inputs, written where the config resolves them.
void write_inputs(const fb::test::ScratchDir &dir, std::string_view config) {
  ASSERT_TRUE(fb::write_file(dir / "water.xyz", kWaterXyz).has_value());
  ASSERT_TRUE(fb::write_file(dir / "basis.json", fb::test::kBasis).has_value());
  ASSERT_TRUE(fb::write_file(dir / "run.toml", config).has_value());
}

TEST(PipelineCalculation, MatchesTheStepsCalledByHand) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, kConfig);
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto molecule = fb::MoleculeStep{.xyz = dir / "water.xyz"}.run(fb::View<>{}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  auto basis = fb::BasisSetStep{.basis = (dir / "basis.json").string()}.run(
      fb::View<fb::MoleculeOutput>{*molecule});
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(transform.has_value());
  ASSERT_TRUE(pairs.has_value());
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          *basis, *transform, *pairs},
      res);
  ASSERT_TRUE(overlap.has_value());
  auto x = fb::OrthogonalizerStep{}.run(
      fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
               fb::PrimitivePairClassesOutput>{*molecule, *basis, *transform, *pairs},
      res);
  ASSERT_TRUE(x.has_value());
  ASSERT_TRUE(h.has_value());
  fb::tf::Taskflow guess_flow;
  fb::Result<fb::GuessOutput> guess = fb::err_config("the guess task did not run");
  guess_flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{*molecule, *basis},
                                subflow);
  });
  node.executor().run(guess_flow).wait();
  ASSERT_TRUE(guess.has_value()) << guess.error().message;

  ASSERT_TRUE(calc->get<fb::BasisSetOutput>().has_value());
  EXPECT_EQ(calc->get<fb::BasisSetOutput>()->basis.nao_sphe(), basis->basis.nao_sphe());
  EXPECT_EQ(calc->get<fb::PrimitivePairClassesOutput>()->classes.size(), pairs->classes.size());
  expect_near(download(res, calc->get<fb::OverlapOutput>()->s), download(res, overlap->s));
  expect_near(download(res, calc->get<fb::OrthogonalizerOutput>()->x), download(res, x->x));
  expect_near(download(res, calc->get<fb::CoreHamiltonianOutput>()->h), download(res, h->h));
  ASSERT_TRUE(calc->get<fb::GuessOutput>().has_value());
  EXPECT_EQ(calc->get<fb::GuessOutput>()->d, guess->d);
}

TEST(PipelineCalculation, TheGuessHoldsEveryElectronAgainstThePipelinesOverlap) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, kConfig);
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  // trace(D S) is Sum_A Z_A exactly: the host guess and the device overlap,
  // made by different code on different sides, must agree on the basis.
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::vector<double> s = download(res, calc->get<fb::OverlapOutput>()->s);
  const std::vector<double> &d = calc->get<fb::GuessOutput>()->d;
  ASSERT_EQ(d.size(), s.size());
  double trace = 0.0;
  for (std::size_t k = 0; k < d.size(); ++k) {
    trace += d[k] * s[k]; // both symmetric, so Sum_ij D_ij S_ji is elementwise
  }
  EXPECT_NEAR(trace, 10.0, 1e-10); // O + 2 H
}

// A guess saved by one run and loaded by the next starts the same SCF: the
// loaded step is planned by its load, beside the device chain, with no SAD.
TEST(PipelineCalculation, AnScfFromALoadedGuessMatchesOneFromItsOwn) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir,
               "[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"basis.json\"\n[scf]\n");
  fb::NodeResources node;
  auto made = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(made.has_value()) << made.error().message;
  const fb::Status made_ran = made->run(node);
  ASSERT_TRUE(made_ran.has_value()) << made_ran.error().message;
  const std::size_t n = made->get<fb::BasisSetOutput>()->basis.nao_sphe();
  ASSERT_TRUE(fb::write_npy(
      dir / "d.npy", fb::NpyArray{.shape = {n, n}, .data = made->get<fb::GuessOutput>()->d}));

  ASSERT_TRUE(fb::write_file(dir / "load.toml", "[molecule]\nxyz = \"water.xyz\"\n[basis_set]\n"
                                                "basis = \"basis.json\"\n[guess]\nfrom = "
                                                "\"d.npy\"\n[scf]\n")
                  .has_value());
  auto loaded = fb::read_config(dir / "load.toml", dir.path(), dir / "out");
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_FALSE(loaded->step<fb::GuessStep>().has_value());
  EXPECT_EQ(loaded->from<fb::GuessStep>(), dir / "d.npy");
  const fb::Status loaded_ran = loaded->run(node);
  ASSERT_TRUE(loaded_ran.has_value()) << loaded_ran.error().message;

  EXPECT_EQ(loaded->get<fb::GuessOutput>()->d, made->get<fb::GuessOutput>()->d);
  const fb::ScfOutput &a = *made->get<fb::ScfOutput>();
  const fb::ScfOutput &b = *loaded->get<fb::ScfOutput>();
  EXPECT_NEAR(b.energy, a.energy, 1e-12);
  EXPECT_EQ(b.iterations, a.iterations);
  expect_near(b.mo_energies, a.mo_energies);
}

// A config naming only the inputs and [casci] plans the chain to the ScfOutput and
// runs the CASCI after it, giving what the step gives run by hand on the same
// products. J adds atomically, so the two Hamiltonians agree to roundoff.
TEST(PipelineCalculation, ACasciConfigRunsEndToEnd) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, "[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
                    "[casci]\nncas = 4\nnelecas = 4\nnroots = 2\n");
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  ASSERT_TRUE(calc->get<fb::ScfOutput>().has_value());
  const auto &casci = calc->get<fb::CasciOutput>();
  ASSERT_TRUE(casci.has_value());
  EXPECT_EQ(casci->ncore, 3);
  ASSERT_EQ(casci->energies.size(), 2U);
  // A CASCI lowers the RHF energy it starts from: the RHF determinant is in
  // the space.
  EXPECT_LT(casci->energies[0], calc->get<fb::ScfOutput>()->energy);
  EXPECT_LT(casci->energies[0], casci->energies[1]);

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto by_hand = calc->step<fb::CasciStep>()->run(
      fb::StepInput<fb::CasciStep>{
          *calc->get<fb::MoleculeOutput>(), *calc->get<fb::BasisSetOutput>(),
          *calc->get<fb::SphericalTransformOutput>(), *calc->get<fb::PrimitivePairClassesOutput>(),
          *calc->get<fb::CoreHamiltonianOutput>(), *calc->get<fb::ScfOutput>()},
      res);
  ASSERT_TRUE(by_hand.has_value()) << by_hand.error().message;
  ASSERT_EQ(by_hand->energies.size(), casci->energies.size());
  for (std::size_t r = 0; r < casci->energies.size(); ++r) {
    EXPECT_NEAR(by_hand->energies[r], casci->energies[r], 1e-10) << r;
    EXPECT_NEAR(by_hand->spin_squares[r], casci->spin_squares[r], 1e-10) << r;
  }
}

// A config naming only the inputs and [casscf] plans the chain to the
// ScfOutput and runs the CASSCF after it, giving what the step gives run by
// hand on the same products, and a lower energy than the CASCI at its start.
TEST(PipelineCalculation, ACasscfConfigRunsEndToEnd) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, "[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
                    "[casci]\nncas = 4\nnelecas = 4\n"
                    "[casscf]\nncas = 4\nnelecas = 4\n");
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casci = calc->get<fb::CasciOutput>();
  const auto &casscf = calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(casci.has_value() && casscf.has_value());
  EXPECT_EQ(casscf->ncore, 3);
  ASSERT_EQ(casscf->energies.size(), 1U);
  EXPECT_EQ(casscf->weights, (std::vector<double>{1.0}));
  // The weighted energy is the RDMs', the root's the CI's: equal to roundoff.
  EXPECT_NEAR(casscf->energy, casscf->energies[0], 1e-10);
  EXPECT_LT(casscf->gradient_norm, calc->step<fb::CasscfStep>()->conv_tol_grad);
  // The CASCI's orbitals are the CASSCF's start, and the optimization only
  // lowers the energy from there.
  EXPECT_LT(casscf->energy, casci->energies[0]);

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto by_hand = calc->step<fb::CasscfStep>()->run(
      fb::StepInput<fb::CasscfStep>{
          *calc->get<fb::MoleculeOutput>(), *calc->get<fb::BasisSetOutput>(),
          *calc->get<fb::SphericalTransformOutput>(), *calc->get<fb::PrimitivePairClassesOutput>(),
          *calc->get<fb::CoreHamiltonianOutput>(), *calc->get<fb::ScfOutput>()},
      res);
  ASSERT_TRUE(by_hand.has_value()) << by_hand.error().message;
  EXPECT_NEAR(by_hand->energy, casscf->energy, 1e-10);
}

TEST(PipelineCalculation, AFailedRootFailsEveryProductAfterIt) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, kConfig);
  std::filesystem::remove(dir / "water.xyz");
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;

  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::IO);
  ASSERT_FALSE(calc->get<fb::CoreHamiltonianOutput>().has_value());
  EXPECT_EQ(calc->get<fb::CoreHamiltonianOutput>().error().kind, fb::ErrorKind::IO);
  EXPECT_EQ(calc->get<fb::OrthogonalizerOutput>().error().message, ran.error().message);
  EXPECT_EQ(calc->get<fb::GuessOutput>().error().message, ran.error().message);
}

} // namespace
