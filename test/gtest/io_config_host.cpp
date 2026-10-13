// Host-only cases for io/config: TOML -> Config, with defaults and
// classification.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

TEST(IoConfig, EmptyDocumentKeepsEveryDefault) {
  auto config = fb::parse_config("");
  ASSERT_TRUE(config.has_value()) << config.error().message;
  const fb::Config defaults{};
  EXPECT_EQ(config->method, defaults.method);
  EXPECT_EQ(config->basis, defaults.basis);
  EXPECT_EQ(config->aux_basis, defaults.aux_basis);
  EXPECT_EQ(config->charge, 0);
  EXPECT_EQ(config->multiplicity, 1);
  EXPECT_EQ(config->scf_energy_tolerance, 1e-8);
  EXPECT_EQ(config->scf_density_tolerance, 1e-6);
  EXPECT_EQ(config->max_iterations, 100);
}

TEST(IoConfig, EveryKeyIsRead) {
  auto config = fb::parse_config(R"(
method = "RHF"
basis = "cc-pVDZ"
aux_basis = "cc-pVDZ-RI"
charge = -1
multiplicity = 2
scf_energy_tolerance = 1e-10
scf_density_tolerance = 1e-7
max_iterations = 250
)");
  ASSERT_TRUE(config.has_value()) << config.error().message;
  EXPECT_EQ(config->method, "RHF");
  EXPECT_EQ(config->basis, "cc-pVDZ");
  EXPECT_EQ(config->aux_basis, "cc-pVDZ-RI");
  EXPECT_EQ(config->charge, -1);
  EXPECT_EQ(config->multiplicity, 2);
  EXPECT_EQ(config->scf_energy_tolerance, 1e-10);
  EXPECT_EQ(config->scf_density_tolerance, 1e-7);
  EXPECT_EQ(config->max_iterations, 250);
}

TEST(IoConfig, PartialDocumentOverridesOnlyWhatItNames) {
  auto config = fb::parse_config("basis = \"sto-3g\"\ncharge = 1\n");
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->basis, "sto-3g");
  EXPECT_EQ(config->charge, 1);
  EXPECT_EQ(config->multiplicity, 1);
  EXPECT_EQ(config->max_iterations, 100);
}

TEST(IoConfig, UnknownKeysAreIgnored) {
  auto config = fb::parse_config("not_a_key = 3\n[section]\nx = 1\n");
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->max_iterations, 100);
}

TEST(IoConfig, WrongTypedKeyIsInvalidConfigNamingTheKey) {
  struct Case {
    std::string_view text;
    std::string_view key;
  };
  for (const auto &[text, key] :
       {Case{"charge = \"minus one\"", "charge"}, Case{"method = 3", "method"},
        Case{"scf_energy_tolerance = \"tight\"", "scf_energy_tolerance"},
        Case{"max_iterations = [1, 2]", "max_iterations"}}) {
    auto config = fb::parse_config(text);
    ASSERT_FALSE(config.has_value()) << "accepted: " << text;
    EXPECT_EQ(config.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(config.error().message.find(std::format("'{}'", key)), std::string::npos)
        << config.error().message;
  }
}

TEST(IoConfig, MalformedTomlIsIoError) {
  auto config = fb::parse_config("method = \n");
  ASSERT_FALSE(config.has_value());
  EXPECT_EQ(config.error().kind, fb::ErrorKind::IO);
}

TEST(IoConfig, ReadsFromDisk) {
  const fb::test::ScratchDir dir("io-config");
  std::ofstream(dir / "run.toml") << "method = \"RHF\"\nmax_iterations = 7\n";
  auto config = fb::read_config(dir / "run.toml");
  ASSERT_TRUE(config.has_value()) << config.error().message;
  EXPECT_EQ(config->method, "RHF");
  EXPECT_EQ(config->max_iterations, 7);
}

TEST(IoConfig, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("io-config");
  auto config = fb::read_config(dir / "missing.toml");
  ASSERT_FALSE(config.has_value());
  EXPECT_EQ(config.error().kind, fb::ErrorKind::IO);
}

} // namespace
