// Host-only cases for io/toml: classified parsing.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.toml;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

TEST(IoToml, ParsesWellFormedDocument) {
  auto table = fb::parse_toml("method = \"RHF\"\ncharge = -1\n[scf]\ntol = 1e-8\n");
  ASSERT_TRUE(table.has_value()) << table.error().message;
  EXPECT_EQ((*table)["method"].value<std::string>(), "RHF");
  EXPECT_EQ((*table)["charge"].value<std::int64_t>(), -1);
  EXPECT_EQ((*table)["scf"]["tol"].value<double>(), 1e-8);
}

TEST(IoToml, EmptyDocumentIsAnEmptyTable) {
  auto table = fb::parse_toml("");
  ASSERT_TRUE(table.has_value());
  EXPECT_TRUE(table->empty());
}

TEST(IoToml, MalformedDocumentIsIoErrorNamingTheLine) {
  auto table = fb::parse_toml("a = 1\nb = = 2\n");
  ASSERT_FALSE(table.has_value());
  EXPECT_EQ(table.error().kind, fb::ErrorKind::IO);
  EXPECT_TRUE(table.error().message.starts_with("malformed TOML at line 2,"))
      << table.error().message;
}

TEST(IoToml, DuplicateKeyIsIoError) {
  auto table = fb::parse_toml("a = 1\na = 2\n");
  ASSERT_FALSE(table.has_value());
  EXPECT_EQ(table.error().kind, fb::ErrorKind::IO);
}

TEST(IoToml, ReadsFromDisk) {
  const fb::test::ScratchDir dir("io-toml");
  std::ofstream(dir / "c.toml") << "basis = \"sto-3g\"\n";
  auto table = fb::read_toml(dir / "c.toml");
  ASSERT_TRUE(table.has_value()) << table.error().message;
  EXPECT_EQ((*table)["basis"].value<std::string>(), "sto-3g");
}

TEST(IoToml, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("io-toml");
  auto table = fb::read_toml(dir / "missing.toml");
  ASSERT_FALSE(table.has_value());
  EXPECT_EQ(table.error().kind, fb::ErrorKind::IO);
}

} // namespace
