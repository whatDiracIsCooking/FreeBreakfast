// Host-only cases for the Irc step (#1269): its contract (static_asserts, as
// steps_host.cpp holds every step's), and its table read by the TOML front
// end -- its keys at their defaults and set, the paths it writes, the chain
// copied in from the config's own tables as [optimize]'s is (the RHF's, a
// [casscf_gradient]'s and a [nevpt2_gradient]'s), its start's [hessian]
// computed in the run or loaded 'from' a file, and every refusal, each naming
// its key; and examples/irc.toml parsed. Nothing is run, so nothing touches a
// device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable; // ScratchDir

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// A Node step, as it runs a Pipeline of the chain per point on the node,
// reading the pipeline's molecule, the transition state, and the [hessian]
// there; it writes both sides' paths, and is never loaded.
static_assert(fb::PipelineStep<fb::IrcStep>);
static_assert(fb::kRunShape<fb::IrcStep> == fb::RunShape::Node);
static_assert(
    std::same_as<fb::StepInput<fb::IrcStep>, fb::View<fb::MoleculeOutput, fb::HessianOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::IrcStep>, fb::IrcOutput>);
static_assert(!fb::LoadableStep<fb::IrcStep>);

constexpr std::string_view kInputs = "[molecule]\nxyz = \"ts.xyz\"\n[basis_set]\nbasis = "
                                     "\"b.json\"\n[hessian]\nfrom = \"ts_hessian.json\"\n";

const std::filesystem::path kOut = "/out";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_config_error(std::string_view toml, std::string_view needle,
                         fb::ErrorKind kind = fb::ErrorKind::InvalidConfig) {
  auto calc = fb::parse_config(toml, "/in", kOut);
  ASSERT_FALSE(calc.has_value()) << toml;
  EXPECT_EQ(calc.error().kind, kind) << calc.error().message;
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

// [irc] reads its own keys, writes its three files into the output
// directory, and copies in the RHF chain from the tables, each as its own
// table configures it; its start is the molecule's and the Hessian loaded
// 'from' the file, a step of its own.
TEST(PipelineIrc, TheStepReadsItsKeysAndCopiesItsChain) {
  auto calc = fb::parse_config(
      with_inputs("[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                  "[scf]\nenergy_tolerance = 1e-12\n"
                  "[irc]\nstep_size = 0.2\nmax_points = 7\nmax_force = 1e-5\n"
                  "sphere_gradient = 2e-6\nsphere_displacement = 3e-5\n"
                  "max_sphere_iterations = 9\ndirection = \"backward\"\n"
                  "allow_multiple_imaginary = true\n"),
      "/in", kOut);
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::IrcStep &irc = *calc->step<fb::IrcStep>();
  EXPECT_EQ(irc.options.step_size, 0.2);
  EXPECT_EQ(irc.options.max_points, 7);
  EXPECT_EQ(irc.options.max_force, 1e-5);
  EXPECT_EQ(irc.options.sphere_gradient, 2e-6);
  EXPECT_EQ(irc.options.sphere_displacement, 3e-5);
  EXPECT_EQ(irc.options.max_sphere_iterations, 9);
  EXPECT_TRUE(irc.options.project_rigid_body);
  EXPECT_EQ(irc.direction, fb::IrcDirection::Backward);
  EXPECT_TRUE(irc.allow_multiple_imaginary);
  EXPECT_EQ(irc.history, kOut / "irc.json");
  EXPECT_EQ(irc.forward_trajectory, kOut / "irc_forward.xyz");
  EXPECT_EQ(irc.backward_trajectory, kOut / "irc_backward.xyz");
  // The chain, read from the tables; the RHF's, with no CASSCF gradient.
  EXPECT_EQ(irc.chain.basis_set.basis, "/in/b.json");
  EXPECT_EQ(irc.chain.eri.mode, fb::EriMode::DensityFitting);
  EXPECT_EQ(irc.chain.eri.aux_basis, "cc-pvdz-jkfit");
  EXPECT_EQ(irc.chain.scf.energy_tolerance, 1e-12);
  EXPECT_FALSE(irc.chain.casscf_gradient.has_value());
  EXPECT_FALSE(irc.chain.nevpt2_gradient.has_value());
  EXPECT_FALSE(irc.chain.meci.has_value());
  // The start: the molecule run from its file, the Hessian loaded.
  ASSERT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, std::filesystem::path("/in/ts.xyz"));
  EXPECT_FALSE(calc->step<fb::HessianStep>().has_value());
  EXPECT_EQ(calc->from<fb::HessianStep>(), std::filesystem::path("/in/ts_hessian.json"));

  // The defaults: geometry::irc's, both sides, one imaginary mode only.
  auto defaults = fb::parse_config(with_inputs("[irc]\n"), "/in", kOut);
  ASSERT_TRUE(defaults.has_value()) << defaults.error().message;
  const fb::IrcStep &plain = *defaults->step<fb::IrcStep>();
  const fb::geometry::IrcOptions expected;
  EXPECT_EQ(plain.options.step_size, expected.step_size);
  EXPECT_EQ(plain.options.step_size, 0.1);
  EXPECT_EQ(plain.options.max_points, expected.max_points);
  EXPECT_EQ(plain.options.max_force, expected.max_force);
  EXPECT_EQ(plain.options.sphere_gradient, expected.sphere_gradient);
  EXPECT_EQ(plain.options.sphere_displacement, expected.sphere_displacement);
  EXPECT_EQ(plain.options.max_sphere_iterations, expected.max_sphere_iterations);
  EXPECT_EQ(plain.direction, fb::IrcDirection::Both);
  EXPECT_FALSE(plain.allow_multiple_imaginary);
  EXPECT_EQ(plain.chain.eri.mode, fb::EriMode::IntegralDirect);
  for (const auto &[name, expected_direction] : {std::pair{"forward", fb::IrcDirection::Forward},
                                                 std::pair{"both", fb::IrcDirection::Both}}) {
    auto sided = fb::parse_config(with_inputs(std::format("[irc]\ndirection = \"{}\"\n", name)),
                                  "/in", kOut);
    ASSERT_TRUE(sided.has_value()) << sided.error().message;
    EXPECT_EQ(sided->step<fb::IrcStep>()->direction, expected_direction);
  }
}

// Over a [casscf_gradient] or a [nevpt2_gradient], the chain ends in the
// [casscf] and that gradient, copied in; a [hessian] computed in the same
// run is the step's start too, over the same tables.
TEST(PipelineIrc, TheCasscfChainsAreCopiedIn) {
  constexpr std::string_view kCas =
      "[molecule]\nxyz = \"ts.xyz\"\n[basis_set]\nbasis = \"b.json\"\n"
      "[eri]\nmode = \"density_fitting\"\n[casscf]\nncas = 2\nnelecas = 2\n";
  auto casscf = fb::parse_config(
      std::format("{}[casscf_gradient]\n[hessian]\n[irc]\nstep_size = 0.3\n", kCas), "/in", kOut);
  ASSERT_TRUE(casscf.has_value()) << casscf.error().message;
  const fb::IrcStep &irc = *casscf->step<fb::IrcStep>();
  ASSERT_TRUE(irc.chain.casscf_gradient.has_value());
  EXPECT_FALSE(irc.chain.nevpt2_gradient.has_value());
  EXPECT_EQ(irc.chain.casscf.ncas, 2);
  EXPECT_EQ(irc.chain.casscf.nelecas, 2);
  EXPECT_TRUE(casscf->step<fb::HessianStep>().has_value());
  EXPECT_TRUE(casscf->step<fb::HessianStep>()->casscf_gradient.has_value());

  auto nevpt2 = fb::parse_config(
      std::format("{}[nevpt2_gradient]\nvariant = \"pc\"\n[hessian]\nfrom = \"h.json\"\n[irc]\n",
                  kCas),
      "/in", kOut);
  ASSERT_TRUE(nevpt2.has_value()) << nevpt2.error().message;
  const fb::IrcStep &pt2 = *nevpt2->step<fb::IrcStep>();
  ASSERT_TRUE(pt2.chain.nevpt2_gradient.has_value());
  EXPECT_EQ(pt2.chain.nevpt2_gradient->variant, fb::Nevpt2Variant::Pc);
  EXPECT_FALSE(pt2.chain.casscf_gradient.has_value());
  EXPECT_EQ(pt2.chain.casscf.ncas, 2);
}

// Every malformed [irc] is refused, naming its key.
TEST(PipelineIrc, AMalformedTableIsRefusedNamingTheKey) {
  for (const std::string_view key :
       {"step_size", "max_force", "sphere_gradient", "sphere_displacement"}) {
    for (const std::string_view value : {"0", "-0.1", "nan", "inf"}) {
      expect_config_error(with_inputs(std::format("[irc]\n{} = {}\n", key, value)),
                          std::format("[irc] {} is ", key));
    }
    expect_config_error(with_inputs(std::format("[irc]\n{} = \"0.1\"\n", key)),
                        std::format("[irc] {} is not a number", key));
  }
  for (const std::string_view key : {"max_points", "max_sphere_iterations"}) {
    expect_config_error(with_inputs(std::format("[irc]\n{} = 0\n", key)),
                        std::format("[irc] {} is 0, below 1", key));
    expect_config_error(with_inputs(std::format("[irc]\n{} = 1.5\n", key)),
                        std::format("[irc] {} is not a 32-bit integer", key));
  }
  expect_config_error(with_inputs("[irc]\ndirection = \"sideways\"\n"),
                      "[irc] has no direction 'sideways' (known: 'both', 'forward', 'backward')");
  expect_config_error(with_inputs("[irc]\nallow_multiple_imaginary = 1\n"),
                      "[irc] allow_multiple_imaginary is not a boolean");
  expect_config_error(with_inputs("[irc]\nstep = 0.1\n"), "[irc] has no option 'step'");
  expect_config_error(with_inputs("[irc]\nfrom = \"irc.json\"\n"),
                      "[irc] cannot be loaded 'from' a file");
}

// What the step cannot start from, refused before anything runs.
TEST(PipelineIrc, AStartItCannotTakeIsRefused) {
  // No Hessian at the TS, computed or loaded.
  expect_config_error("[molecule]\nxyz = \"ts.xyz\"\n[basis_set]\nbasis = \"b.json\"\n[irc]\n",
                      "[irc] starts from the transition state's Hessian: add a [hessian]");
  // An [optimize] would move the molecule to a minimum first.
  expect_config_error(with_inputs("[optimize]\n[irc]\n"),
                      "[irc] starts from a transition state at the molecule's geometry, and "
                      "[optimize] moves the molecule to a minimum first");
  // One energy is followed.
  expect_config_error(
      with_inputs("[eri]\nmode = \"density_fitting\"\n[casscf]\nncas = 2\nnelecas = 2\n"
                  "[casscf_gradient]\n[nevpt2_gradient]\n[irc]\n"),
      "[irc] follows the energy of [casscf_gradient] or of [nevpt2_gradient], and the config "
      "names both");
  // The chain is rerun at every point, so none of it is loaded.
  expect_config_error(with_inputs("[irc]\n[guess]\nfrom = \"d.npy\"\n"),
                      "[irc] reruns [guess] at every geometry, so it cannot be loaded 'from' a "
                      "file");
  // A CASSCF chain over integral-direct ERIs, refused as the gradient step
  // itself is.
  expect_config_error(with_inputs("[casscf]\nncas = 2\nnelecas = 2\n[casscf_gradient]\n[irc]\n"),
                      "[casscf_gradient] runs over density fitting only",
                      fb::ErrorKind::Unsupported);
}

// examples/irc.toml plans: the RHF chain over its fit, the Hessian computed
// analytically at the TS, and both sides.
TEST(PipelineIrc, TheExampleConfigPlans) {
  const std::filesystem::path examples{FBF_EXAMPLES_DIR};
  // read_config makes the output directory.
  const fb::test::ScratchDir out("pipeline-irc-example");
  auto calc = fb::read_config(examples / "irc.toml", examples, out.path());
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const auto &irc = calc->step<fb::IrcStep>();
  ASSERT_TRUE(irc.has_value());
  EXPECT_EQ(irc->direction, fb::IrcDirection::Both);
  EXPECT_TRUE(calc->step<fb::HessianStep>().has_value());
}

} // namespace
