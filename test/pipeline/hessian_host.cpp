// Host-only cases for the Hessian step: its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- the method and the displacement, the chain copied in from the
// config's own tables, every refusal, and beside an [optimize], the molecule
// it reads loaded from the final.xyz that step writes, so it runs after it.
// Nothing is run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// A Node step, as each displacement runs a Pipeline of its own on the node,
// reading the pipeline's molecule: whichever step makes it, it runs after.
static_assert(fb::PipelineStep<fb::HessianStep>);
static_assert(fb::kRunShape<fb::HessianStep> == fb::RunShape::Node);
static_assert(std::same_as<fb::StepInput<fb::HessianStep>, fb::View<fb::MoleculeOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::HessianStep>, fb::HessianOutput>);
static_assert(!fb::LoadableStep<fb::HessianStep>);

constexpr std::string_view kInputs =
    "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n";

const std::filesystem::path kOut = "/out";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_config_error(std::string_view toml, std::string_view needle) {
  auto calc = fb::parse_config(toml, "", kOut);
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

// [hessian] reads its own keys and copies in the chain it reruns, each step
// as its own table configures it; the molecule is the pipeline's, read
// from the file, and nothing else is filled in for it.
TEST(PipelineHessian, TheStepReadsItsKeysAndCopiesItsChain) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                  "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
                  "[hessian]\nmethod = \"numerical\"\ndisplacement = 5e-4\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  EXPECT_EQ(hessian.method, fb::HessianMethod::Numerical);
  EXPECT_EQ(hessian.displacement, 5e-4);
  EXPECT_EQ(hessian.json_path, kOut / "hessian.json");
  // The chain, read from the tables; a step with no table at its defaults.
  EXPECT_EQ(hessian.basis_set.basis, "/in/b.json");
  EXPECT_EQ(hessian.eri.mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(hessian.eri.aux_basis, "cc-pvdz-jkfit");
  EXPECT_EQ(hessian.scf.energy_tolerance, 1e-12);
  EXPECT_EQ(hessian.scf.commutator_tolerance, 1e-10);
  EXPECT_EQ(hessian.orthogonalizer.eigenvalue_floor, fb::OrthogonalizerStep{}.eigenvalue_floor);
  EXPECT_EQ(hessian.guess.type, fb::GuessType::Sad);
  // The molecule it reads is the config's, run from its file.
  ASSERT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, std::filesystem::path("/in/w.xyz"));
  EXPECT_FALSE(calc->from<fb::MoleculeStep>().has_value());

  // The default step, and the chain's steps at their defaults with no tables.
  auto defaults = fb::parse_config(with_inputs("[hessian]\nmethod = \"numerical\"\n"), "", kOut);
  ASSERT_TRUE(defaults.has_value()) << defaults.error().message;
  const fb::HessianStep &plain = *defaults->step<fb::HessianStep>();
  EXPECT_EQ(plain.displacement, fb::HessianStep{}.displacement);
  EXPECT_EQ(plain.eri.mode, fb::EriMode::IntegralDirect);
  EXPECT_EQ(plain.scf.energy_tolerance, fb::ScfStep{}.energy_tolerance);
  // A table is a request: only the molecule it reads is filled in.
  EXPECT_FALSE(defaults->step<fb::ScfStep>().has_value());
  EXPECT_TRUE(defaults->step<fb::MoleculeStep>().has_value());
}

TEST(PipelineHessian, AMalformedTableIsInvalidConfig) {
  // No default method yet.
  expect_config_error(with_inputs("[hessian]\n"), "[hessian] needs 'method'");
  expect_config_error(with_inputs("[hessian]\nmethod = \"analytic\"\n"),
                      "[hessian] has no method 'analytic' (known: 'numerical')");
  expect_config_error(with_inputs("[hessian]\nmethod = \"numerical\"\nstep = 1e-3\n"),
                      "[hessian] has no option 'step'");
  for (const std::string_view h : {"0", "-1e-3", "nan", "inf"}) {
    expect_config_error(
        with_inputs(std::format("[hessian]\nmethod = \"numerical\"\ndisplacement = {}\n", h)),
        "[hessian] displacement must be finite and positive");
  }
  expect_config_error(with_inputs("[hessian]\nmethod = \"numerical\"\ndisplacement = \"1e-3\"\n"),
                      "[hessian] displacement is not a number");
  // The chain is rerun at every displacement, so none of it is loaded.
  expect_config_error(
      with_inputs("[hessian]\nmethod = \"numerical\"\n[guess]\nfrom = \"d.npy\"\n"),
      "[hessian] reruns [guess] at every geometry, so it cannot be loaded 'from' a file");
  // The molecule it reads comes from somewhere.
  expect_config_error("[basis_set]\nbasis = \"b.json\"\n[hessian]\nmethod = \"numerical\"\n",
                      "[molecule] needs 'xyz'");
}

// Beside an [optimize], the molecule the Hessian reads is the one loaded
// from final.xyz, which reads OptimizeOutput: the Hessian is ordered after
// the optimization and runs at the optimum. Both steps hold the same chain.
TEST(PipelineHessian, BesideAnOptimizeItReadsTheOptimum) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\n[scf]\nenergy_tolerance = 1e-12\n"
                  "[optimize]\n[hessian]\nmethod = \"numerical\"\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_FALSE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_EQ(calc->from<fb::MoleculeStep>(), kOut / "final.xyz");
  static_assert(std::same_as<fb::LoadInput<fb::MoleculeStep>, fb::View<fb::OptimizeOutput>>);
  const fb::HessianStep &hessian = *calc->step<fb::HessianStep>();
  const fb::OptimizeStep &optimize = *calc->step<fb::OptimizeStep>();
  EXPECT_EQ(hessian.basis_set.basis, optimize.basis_set.basis);
  EXPECT_EQ(hessian.eri.mode, optimize.eri.mode);
  EXPECT_EQ(hessian.scf.energy_tolerance, optimize.scf.energy_tolerance);
}

} // namespace
