// Host-only cases for the TOML front end: a config document -> the pipeline
// it asks for, with path resolution, filled-in producers, and every way a
// config can be refused. Nothing is run, so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWater = R"(
[molecule]
xyz = "water.xyz"
charge = -1
multiplicity = 2

[basis_set]
basis = "/basis/sto-3g.json"

[orthogonalizer]
eigenvalue_floor = 1e-8

[guess]
type = "sad"
)";

// The two tables every config that reaches a basis needs.
constexpr std::string_view kInputs =
    "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n";

std::string with_inputs(std::string_view tables) {
  return std::string(kInputs).append(tables);
}

void expect_config_error(std::string_view toml, std::string_view needle) {
  auto calc = fb::parse_config(toml, "");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
}

TEST(PipelineConfig, EveryOptionIsRead) {
  auto calc = fb::parse_config(kWater, "/inputs");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::MoleculeStep &molecule = *calc->step<fb::MoleculeStep>();
  EXPECT_EQ(molecule.xyz, std::filesystem::path("/inputs/water.xyz"));
  EXPECT_EQ(molecule.charge, -1);
  EXPECT_EQ(molecule.multiplicity, 2);
  // An absolute path is kept as written.
  EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, "/basis/sto-3g.json");
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor, 1e-8);
  EXPECT_EQ(calc->step<fb::GuessStep>()->type, fb::GuessType::Sad);
}

TEST(PipelineConfig, ARelativeBasisPathIsAnchoredToTheConfig) {
  auto calc =
      fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n", "/inputs");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, "/inputs/b.json");
}

TEST(PipelineConfig, ABuiltinBasisNameIsNotAPath) {
  // The name of a compiled-in set must survive the front end verbatim: joined
  // to base_dir it would become /inputs/sto-3g, a file nobody wrote.
  for (const std::string_view spelling : {"sto-3g", "STO-3G", "sto3g", "cc-pVTZ"}) {
    auto calc = fb::parse_config(
        std::format("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"{}\"\n", spelling),
        "/inputs");
    ASSERT_TRUE(calc.has_value()) << spelling << ": " << calc.error().message;
    EXPECT_EQ(calc->step<fb::BasisSetStep>()->basis, spelling);
  }
}

TEST(PipelineConfig, AbsentOptionsKeepTheStepDefaults) {
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::MoleculeStep defaults{};
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->charge, defaults.charge);
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->multiplicity, defaults.multiplicity);
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor,
            fb::OrthogonalizerStep{}.eigenvalue_floor);
  // No base directory: relative to the working directory.
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, std::filesystem::path("w.xyz"));
}

TEST(PipelineConfig, IntegerReadsAsDouble) {
  auto calc = fb::parse_config(with_inputs("[orthogonalizer]\neigenvalue_floor = 0\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::OrthogonalizerStep>()->eigenvalue_floor, 0.0);
}

TEST(PipelineConfig, ATableIsARequestAndItsProducersAreFilledIn) {
  auto calc = fb::parse_config(with_inputs("[core_hamiltonian]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::SphericalTransformStep>().has_value());
  EXPECT_TRUE(calc->step<fb::PrimitivePairClassesStep>().has_value());
  // Not upstream of anything asked for.
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::OrthogonalizerStep>().has_value());
}

TEST(PipelineConfig, TheGuessNeedsNoDeviceStepPastTheMolecule) {
  auto calc = fb::parse_config(with_inputs("[guess]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::GuessStep>()->type, fb::GuessStep{}.type);
  EXPECT_FALSE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_FALSE(calc->step<fb::PrimitivePairClassesStep>().has_value());
}

TEST(PipelineConfig, TheScfFillsInTheWholeChain) {
  auto calc = fb::parse_config(with_inputs("[scf]\n"), "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::OverlapStep>().has_value());
  EXPECT_TRUE(calc->step<fb::OrthogonalizerStep>().has_value());
  EXPECT_TRUE(calc->step<fb::CoreHamiltonianStep>().has_value());
  EXPECT_TRUE(calc->step<fb::GuessStep>().has_value());
  const fb::ScfStep defaults{};
  const fb::ScfStep &scf = *calc->step<fb::ScfStep>();
  EXPECT_EQ(scf.max_iterations, defaults.max_iterations);
  EXPECT_EQ(scf.energy_tolerance, defaults.energy_tolerance);
  EXPECT_EQ(scf.commutator_tolerance, defaults.commutator_tolerance);
  EXPECT_EQ(scf.diis_history, defaults.diis_history);
  EXPECT_EQ(scf.fock_rebuild_interval, defaults.fock_rebuild_interval);
}

TEST(PipelineConfig, EveryScfOptionIsRead) {
  auto calc = fb::parse_config(with_inputs("[scf]\nmax_iterations = 40\nenergy_tolerance = 1e-8\n"
                                           "commutator_tolerance = 1e-5\ndiis_history = 6\n"
                                           "fock_rebuild_interval = 1\n"),
                               "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  const fb::ScfStep &scf = *calc->step<fb::ScfStep>();
  EXPECT_EQ(scf.max_iterations, 40);
  EXPECT_EQ(scf.energy_tolerance, 1e-8);
  EXPECT_EQ(scf.commutator_tolerance, 1e-5);
  EXPECT_EQ(scf.diis_history, 6);
  EXPECT_EQ(scf.fock_rebuild_interval, 1);
}

TEST(PipelineConfig, AMisspeltScfOptionIsInvalidConfig) {
  expect_config_error(with_inputs("[scf]\nmax_iter = 40\n"), "[scf] has no option 'max_iter'");
  expect_config_error(with_inputs("[scf]\nmax_iterations = 1.5\n"), "[scf] max_iterations");
}

TEST(PipelineConfig, OnlyWhatWasAskedForIsPlanned) {
  auto calc = fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n", "");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_TRUE(calc->step<fb::MoleculeStep>().has_value());
  EXPECT_FALSE(calc->step<fb::BasisSetStep>().has_value());
}

TEST(PipelineConfig, AFilledInStepMissingARequiredOptionIsInvalidConfig) {
  // [overlap] needs a basis, and [basis_set] cannot be read from nothing.
  expect_config_error("[molecule]\nxyz = \"w.xyz\"\n[overlap]\n", "[basis_set] needs 'basis'");
  expect_config_error("[basis_set]\nbasis = \"b.json\"\n", "[molecule] needs 'xyz'");
}

TEST(PipelineConfig, MissingRequiredOptionIsInvalidConfig) {
  expect_config_error("[molecule]\ncharge = 0\n[basis_set]\nbasis = \"b.json\"\n", "'xyz'");
}

TEST(PipelineConfig, AConfigNamingNoStepIsInvalidConfig) {
  expect_config_error("", "names no step");
}

TEST(PipelineConfig, UnknownStepIsInvalidConfig) {
  expect_config_error(with_inputs("[mp2]\nfrozen_core = true\n"), "unknown step [mp2]");
}

TEST(PipelineConfig, AStepThatIsNotATableIsInvalidConfig) {
  expect_config_error("molecule = \"w.xyz\"\n", "write it as [molecule]");
}

TEST(PipelineConfig, MisspeltOptionIsInvalidConfig) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\nmultiplicty = 3\n[basis_set]\nbasis = \"b.json\"\n",
      "'multiplicty'");
  expect_config_error(with_inputs("[orthogonalizer]\nfloor = 1e-8\n"), "'floor'");
  // A step with no options has no keys to give.
  expect_config_error(with_inputs("[overlap]\nscreen = 1e-12\n"),
                      "[overlap] has no option 'screen'");
}

TEST(PipelineConfig, AnUnknownGuessTypeIsInvalidConfig) {
  expect_config_error(with_inputs("[guess]\ntype = \"core\"\n"), "[guess] has no type 'core'");
  expect_config_error(with_inputs("[guess]\ntype = \"SAD\"\n"), "[guess] has no type 'SAD'");
}

TEST(PipelineConfig, WrongTypedOptionIsInvalidConfigNamingIt) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 1.5\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
  expect_config_error("[molecule]\nxyz = 3\n[basis_set]\nbasis = \"b.json\"\n", "[molecule] xyz");
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 3000000000\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
  expect_config_error(with_inputs("[orthogonalizer]\neigenvalue_floor = \"small\"\n"),
                      "[orthogonalizer] eigenvalue_floor");
  expect_config_error(with_inputs("[guess]\ntype = 1\n"), "[guess] type");
}

TEST(PipelineConfig, MalformedTomlIsIoError) {
  auto calc = fb::parse_config("[molecule\n", "");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineConfig, ReadResolvesAgainstTheFilesDirectory) {
  const fb::test::ScratchDir dir("pipeline-config");
  std::ofstream(dir / "run.toml") << kWater;
  auto calc = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  EXPECT_EQ(calc->step<fb::MoleculeStep>()->xyz, dir / "water.xyz");
}

TEST(PipelineConfig, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("pipeline-config");
  auto calc = fb::read_config(dir / "absent.toml");
  ASSERT_FALSE(calc.has_value());
  EXPECT_EQ(calc.error().kind, fb::ErrorKind::IO);
}

} // namespace
