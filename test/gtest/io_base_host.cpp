// Host-only cases for io/base: the whole-file read and write primitives.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

TEST(IoBase, WriteThenReadRoundTripsBytesExactly) {
  const fb::test::ScratchDir dir("io-base");
  // NUL, CRLF and high bytes: binary mode must neither truncate nor translate.
  const std::string bytes{"a\0b\r\nc\xff\x80\n", 9};
  ASSERT_TRUE(fb::write_file(dir / "f.bin", bytes).has_value());

  auto read = fb::read_file(dir / "f.bin");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  EXPECT_EQ(*read, bytes);
}

TEST(IoBase, EmptyFileReadsAsEmptyString) {
  const fb::test::ScratchDir dir("io-base");
  ASSERT_TRUE(fb::write_file(dir / "empty", "").has_value());
  auto read = fb::read_file(dir / "empty");
  ASSERT_TRUE(read.has_value());
  EXPECT_TRUE(read->empty());
}

TEST(IoBase, WriteReplacesExistingContents) {
  const fb::test::ScratchDir dir("io-base");
  ASSERT_TRUE(fb::write_file(dir / "f", "a much longer first version").has_value());
  ASSERT_TRUE(fb::write_file(dir / "f", "short").has_value());
  EXPECT_EQ(fb::read_file(dir / "f").value(), "short");
}

TEST(IoBase, ReadOfMissingFileIsIoErrorNamingThePath) {
  const fb::test::ScratchDir dir("io-base");
  const auto missing = dir / "does-not-exist";
  auto read = fb::read_file(missing);
  ASSERT_FALSE(read.has_value());
  EXPECT_EQ(read.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(read.error().message.find(missing.string()), std::string::npos);
}

TEST(IoBase, WriteIntoMissingDirectoryIsIoErrorNamingThePath) {
  const fb::test::ScratchDir dir("io-base");
  const auto target = dir / "no-such-dir" / "f";
  auto status = fb::write_file(target, "x");
  ASSERT_FALSE(status.has_value());
  EXPECT_EQ(status.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(status.error().message.find(target.string()), std::string::npos);
}

} // namespace
