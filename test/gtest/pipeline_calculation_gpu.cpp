// The real registry end to end: a config file read, planned and run by
// Calculation, against the same steps called by hand in the order the planner
// should choose. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
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
  auto calc = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto molecule = fb::MoleculeStep{.xyz = dir / "water.xyz"}.run(fb::View<>{}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  auto basis = fb::BasisSetStep{.basis = dir / "basis.json"}.run(fb::View<fb::Molecule>{*molecule});
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSet>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSet>{*basis}, res);
  ASSERT_TRUE(transform.has_value());
  ASSERT_TRUE(pairs.has_value());
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses>{*basis, *transform,
                                                                               *pairs},
      res);
  ASSERT_TRUE(overlap.has_value());
  auto x = fb::OrthogonalizerStep{}.run(fb::View<fb::BasisSet, fb::Overlap>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::Molecule, fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses>{
          *molecule, *basis, *transform, *pairs},
      res);
  ASSERT_TRUE(x.has_value());
  ASSERT_TRUE(h.has_value());

  ASSERT_TRUE(calc->get<fb::BasisSet>().has_value());
  EXPECT_EQ(calc->get<fb::BasisSet>()->nao_sphe(), basis->nao_sphe());
  EXPECT_EQ(calc->get<fb::PrimitivePairClasses>()->classes.size(), pairs->classes.size());
  expect_near(download(res, calc->get<fb::Overlap>()->s), download(res, overlap->s));
  expect_near(download(res, calc->get<fb::Orthogonalizer>()->x), download(res, x->x));
  expect_near(download(res, calc->get<fb::CoreHamiltonian>()->h), download(res, h->h));
}

TEST(PipelineCalculation, AFailedRootFailsEveryProductAfterIt) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_calculation");
  write_inputs(dir, kConfig);
  std::filesystem::remove(dir / "water.xyz");
  fb::NodeResources node;
  auto calc = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;

  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::IO);
  ASSERT_FALSE(calc->get<fb::CoreHamiltonian>().has_value());
  EXPECT_EQ(calc->get<fb::CoreHamiltonian>().error().kind, fb::ErrorKind::IO);
  EXPECT_EQ(calc->get<fb::Orthogonalizer>().error().message, ran.error().message);
}

} // namespace
