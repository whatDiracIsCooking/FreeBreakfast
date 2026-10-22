// Host-only cases for the TOML front end: a config document -> the steps it
// names, with path resolution and every way a config can be refused.

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
)";

void expect_config_error(std::string_view toml, std::string_view needle) {
  auto steps = fb::parse_config(toml, "");
  ASSERT_FALSE(steps.has_value());
  EXPECT_EQ(steps.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(steps.error().message.find(needle), std::string::npos) << steps.error().message;
}

TEST(PipelineConfig, EveryOptionIsRead) {
  auto steps = fb::parse_config(kWater, "/inputs");
  ASSERT_TRUE(steps.has_value()) << steps.error().message;
  const auto &[molecule, basis_set] = *steps;
  EXPECT_EQ(molecule.xyz, std::filesystem::path("/inputs/water.xyz"));
  EXPECT_EQ(molecule.charge, -1);
  EXPECT_EQ(molecule.multiplicity, 2);
  // An absolute path is kept as written.
  EXPECT_EQ(basis_set.basis, std::filesystem::path("/basis/sto-3g.json"));
}

TEST(PipelineConfig, AbsentOptionsKeepTheStepDefaults) {
  auto steps =
      fb::parse_config("[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n", "");
  ASSERT_TRUE(steps.has_value()) << steps.error().message;
  const fb::MoleculeStep defaults{};
  EXPECT_EQ(std::get<fb::MoleculeStep>(*steps).charge, defaults.charge);
  EXPECT_EQ(std::get<fb::MoleculeStep>(*steps).multiplicity, defaults.multiplicity);
  // No base directory: relative to the working directory.
  EXPECT_EQ(std::get<fb::MoleculeStep>(*steps).xyz, std::filesystem::path("w.xyz"));
}

TEST(PipelineConfig, MissingStepTableIsInvalidConfig) {
  expect_config_error("[molecule]\nxyz = \"w.xyz\"\n", "[basis_set]");
}

TEST(PipelineConfig, MissingRequiredOptionIsInvalidConfig) {
  expect_config_error("[molecule]\ncharge = 0\n[basis_set]\nbasis = \"b.json\"\n", "'xyz'");
}

TEST(PipelineConfig, UnknownStepIsInvalidConfig) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\n[basis_set]\nbasis = \"b.json\"\n[scf]\nmax_iterations = 3\n",
      "[scf]");
}

TEST(PipelineConfig, MisspeltOptionIsInvalidConfig) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\nmultiplicty = 3\n[basis_set]\nbasis = \"b.json\"\n",
      "'multiplicty'");
}

TEST(PipelineConfig, WrongTypedOptionIsInvalidConfigNamingIt) {
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 1.5\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
  expect_config_error("[molecule]\nxyz = 3\n[basis_set]\nbasis = \"b.json\"\n", "[molecule] xyz");
  expect_config_error(
      "[molecule]\nxyz = \"w.xyz\"\ncharge = 3000000000\n[basis_set]\nbasis = \"b.json\"\n",
      "[molecule] charge");
}

TEST(PipelineConfig, MalformedTomlIsIoError) {
  auto steps = fb::parse_config("[molecule\n", "");
  ASSERT_FALSE(steps.has_value());
  EXPECT_EQ(steps.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineConfig, ReadResolvesAgainstTheFilesDirectory) {
  const fb::test::ScratchDir dir("pipeline-config");
  std::ofstream(dir / "run.toml") << kWater;
  auto steps = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(steps.has_value()) << steps.error().message;
  EXPECT_EQ(std::get<fb::MoleculeStep>(*steps).xyz, dir / "water.xyz");
}

TEST(PipelineConfig, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("pipeline-config");
  auto steps = fb::read_config(dir / "absent.toml");
  ASSERT_FALSE(steps.has_value());
  EXPECT_EQ(steps.error().kind, fb::ErrorKind::IO);
}

} // namespace
