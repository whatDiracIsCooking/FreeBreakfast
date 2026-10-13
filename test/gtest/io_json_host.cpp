// Host-only cases for io/json: classified parsing and serialization.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.json;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

TEST(IoJson, ParsesWellFormedDocument) {
  auto doc = fb::parse_json(R"({"name": "water", "atoms": [8, 1, 1], "charge": 0})");
  ASSERT_TRUE(doc.has_value()) << doc.error().message;
  EXPECT_EQ((*doc)["name"].get<std::string>(), "water");
  EXPECT_EQ((*doc)["atoms"].size(), 3U);
  EXPECT_EQ((*doc)["charge"].get<int>(), 0);
}

TEST(IoJson, MalformedDocumentIsIoError) {
  for (const std::string_view text : {R"({"a": })", "", "[1, 2", "nul"}) {
    auto doc = fb::parse_json(text);
    ASSERT_FALSE(doc.has_value()) << "accepted: '" << text << "'";
    EXPECT_EQ(doc.error().kind, fb::ErrorKind::IO);
    EXPECT_TRUE(doc.error().message.starts_with("malformed JSON")) << doc.error().message;
  }
}

TEST(IoJson, WrongShapeIsNotAParseError) {
  // What a document means is the domain's job: a well-formed array where an
  // object might be expected still parses.
  auto doc = fb::parse_json("[1, 2, 3]");
  ASSERT_TRUE(doc.has_value());
  EXPECT_TRUE(doc->is_array());
}

TEST(IoJson, ToStringIndentsByTwoByDefaultAndCompactsOnNegative) {
  const fb::json doc = {{"a", 1}};
  EXPECT_EQ(fb::json_to_string(doc), "{\n  \"a\": 1\n}");
  EXPECT_EQ(fb::json_to_string(doc, -1), R"({"a":1})");
}

TEST(IoJson, DoublesRoundTripAtFullPrecision) {
  const std::array values{0.1 + 0.2, 1.0 / 3.0, 1e-300, -2.2250738585072014e-308,
                          std::numeric_limits<double>::max()};
  fb::json doc = fb::json::array();
  for (const double v : values) {
    doc.push_back(v);
  }
  auto back = fb::parse_json(fb::json_to_string(doc));
  ASSERT_TRUE(back.has_value());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ((*back)[i].get<double>(), values[i]) << "index " << i;
  }
}

TEST(IoJson, WriteThenReadRoundTripsOnDisk) {
  const fb::test::ScratchDir dir("io-json");
  const fb::json doc = {{"energy", -74.96}, {"converged", true}, {"tags", {"rhf", "sto-3g"}}};
  ASSERT_TRUE(fb::write_json(dir / "r.json", doc).has_value());
  auto back = fb::read_json(dir / "r.json");
  ASSERT_TRUE(back.has_value()) << back.error().message;
  EXPECT_EQ(*back, doc);
}

TEST(IoJson, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("io-json");
  auto doc = fb::read_json(dir / "missing.json");
  ASSERT_FALSE(doc.has_value());
  EXPECT_EQ(doc.error().kind, fb::ErrorKind::IO);
}

TEST(IoJson, ReadOfMalformedFileIsIoError) {
  const fb::test::ScratchDir dir("io-json");
  std::ofstream(dir / "bad.json") << "{ not json";
  auto doc = fb::read_json(dir / "bad.json");
  ASSERT_FALSE(doc.has_value());
  EXPECT_EQ(doc.error().kind, fb::ErrorKind::IO);
  EXPECT_TRUE(doc.error().message.starts_with("malformed JSON"));
}

} // namespace
