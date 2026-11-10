// The fragmentation headroom pre-flight leaves above a plan's footprint
// (src/pipeline/footprint/README.md, "Fragmentation headroom"): the RHF chain,
// every step from the Molecule to the Scf, run by hand on a fresh
// DeviceResources, its pool's ReservedMemHigh against its UsedMemHigh. What
// the pool took from the driver past what was in use at the peak is what a
// cap must leave room for. Integral-direct and density-fitted, water in
// cc-pVDZ and cc-pVTZ, and benzene in cc-pVTZ density-fitted, whose B and
// three-center scratch are each past a pool chunk.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.scf;

#include "test_workspaces.h"

namespace {

namespace fb = free_breakfast;

// Water, in bohr.
std::vector<fb::Atom> water() {
  return {fb::Atom{{0.0, 0.0, -0.1294}, fb::Element::O},
          fb::Atom{{0.0, -1.4941, 1.0274}, fb::Element::H},
          fb::Atom{{0.0, 1.4941, 1.0274}, fb::Element::H}};
}

// Benzene, in bohr: C-C 1.39 A and C-H 1.09 A, in the xy plane.
std::vector<fb::Atom> benzene() {
  std::vector<fb::Atom> atoms;
  for (const auto &[element, radius] :
       {std::pair{fb::Element::C, 2.6267}, {fb::Element::H, 4.6865}}) {
    for (std::int32_t k = 0; k < 6; ++k) {
      const double phi = std::numbers::pi * k / 3.0;
      atoms.push_back(fb::Atom{{radius * std::cos(phi), radius * std::sin(phi), 0.0}, element});
    }
  }
  return atoms;
}

struct Chain {
  std::string_view name;
  std::vector<fb::Atom> (*atoms)();
  std::string_view basis;
  fb::EriMode mode;
};

// The chain in the planner's order: the setup steps, the ERIs, the
// one-electron steps, the guess and the SCF.
void run_chain(fb::DeviceResources &res, const Chain &chain) {
  const fb::MoleculeOutput molecule{fb::Molecule(res, chain.atoms())};
  auto basis = fb::BasisSetStep{.basis = std::string(chain.basis)}.run(
      fb::View<fb::MoleculeOutput>{molecule});
  ASSERT_TRUE(basis);
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(transform);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(pairs);
  auto eri = fb::EriStep{.mode = chain.mode}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          molecule, *basis, *pairs},
      res);
  ASSERT_TRUE(eri) << eri.error().message;
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          *basis, *transform, *pairs},
      res);
  ASSERT_TRUE(overlap);
  auto x = fb::OrthogonalizerStep{}.run(
      fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, res);
  ASSERT_TRUE(x);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
               fb::PrimitivePairClassesOutput>{molecule, *basis, *transform, *pairs},
      res);
  ASSERT_TRUE(h);
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::GuessOutput> guess = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{molecule, *basis},
                                subflow);
  });
  executor.run(flow).wait();
  ASSERT_TRUE(guess);
  auto scf = fb::ScfStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput, fb::EriOutput,
               fb::OverlapOutput, fb::OrthogonalizerOutput, fb::CoreHamiltonianOutput,
               fb::GuessOutput>{molecule, *basis, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf) << scf.error().message;
}

class RhfChainHeadroom : public ::testing::TestWithParam<Chain> {};

TEST_P(RhfChainHeadroom, ReservedPastUsedIsInsideTheHeadroom) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: water's integral-direct cc-pVTZ SCF and benzene's.
  if (GetParam().name == "WaterTzDirect" || GetParam().name == "BenzeneTzDf") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  fb::Device &dev = *res.device();
  dev.reset_pool_high_water();
  run_chain(res, GetParam());
  ASSERT_EQ(wwr::wwrDeviceSynchronize(), wwr::wwrSuccess);
  // The pool's own high-water marks, not the chain's growth: the pool has
  // reserved its first chunk before the chain starts.
  const fb::PoolUsage usage = dev.pool_usage();
  ASSERT_GE(usage.reserved_high, usage.used_high);
  const std::uint64_t gap = usage.reserved_high - usage.used_high;
  std::cout << std::format("[ headroom ] {}: UsedMemHigh {} ReservedMemHigh {} gap {}\n",
                           GetParam().name, usage.used_high, usage.reserved_high, gap);
  EXPECT_LE(gap, fb::kPoolHeadroomBytes);
}

INSTANTIATE_TEST_SUITE_P(
    Chains, RhfChainHeadroom,
    ::testing::Values(Chain{"WaterDzDirect", water, "cc-pvdz", fb::EriMode::IntegralDirect},
                      Chain{"WaterDzDf", water, "cc-pvdz", fb::EriMode::DensityFitting},
                      Chain{"WaterTzDirect", water, "cc-pvtz", fb::EriMode::IntegralDirect},
                      Chain{"WaterTzDf", water, "cc-pvtz", fb::EriMode::DensityFitting},
                      Chain{"BenzeneTzDf", benzene, "cc-pvtz", fb::EriMode::DensityFitting}),
    [](const ::testing::TestParamInfo<Chain> &info) { return std::string(info.param.name); });

} // namespace
