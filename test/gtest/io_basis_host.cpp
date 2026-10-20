// Host-only cases for io/basis: the BSE JSON reader. Its placement onto a
// molecule is io_basis_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.core.nontransferrable;

#include "io_basis_json.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::test::kBasis;

TEST(IoBasis, ParsesElementsIntoShellStamps) {
  auto stamp = fb::parse_basis(kBasis);
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);

  const std::vector<fb::ShellStamp> *h = stamp->shells_for(1);
  ASSERT_NE(h, nullptr);
  ASSERT_EQ(h->size(), 1U);
  EXPECT_EQ((*h)[0].angular_momentum(), 0);
  ASSERT_EQ((*h)[0].nprimitives(), 3U);
  EXPECT_EQ((*h)[0].primitives()[0].exponent, 3.42525091);
  EXPECT_EQ((*h)[0].primitives()[2].coefficient, 0.44463454);
}

TEST(IoBasis, SplitsSpAndGeneralContractionBlocks) {
  auto stamp = fb::parse_basis(kBasis);
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  const std::vector<fb::ShellStamp> &o = *stamp->shells_for(8);
  // s, then sp -> s + p, then the general contraction -> s + s, then d.
  const std::array<std::int32_t, 6> momenta{0, 0, 1, 0, 0, 2};
  ASSERT_EQ(o.size(), momenta.size());
  for (std::size_t i = 0; i < o.size(); ++i) {
    EXPECT_EQ(o[i].angular_momentum(), momenta[i]) << "shell " << i;
  }
  // The sp block's p shell takes the second column over the shared exponents.
  EXPECT_EQ(o[2].primitives()[1].exponent, 1.1695961);
  EXPECT_EQ(o[2].primitives()[1].coefficient, 0.60768372);
  // Each general-contraction column is its own shell.
  EXPECT_EQ(o[3].primitives()[0].coefficient, 1.0);
  EXPECT_EQ(o[4].primitives()[0].coefficient, 0.0);
  EXPECT_EQ(o[4].primitives()[1].coefficient, 1.0);
}

TEST(IoBasis, ReadBasisFromFile) {
  const fb::test::ScratchDir dir("io-basis");
  ASSERT_TRUE(fb::write_file(dir / "basis.json", kBasis).has_value());
  auto stamp = fb::read_basis(dir / "basis.json");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);

  auto missing = fb::read_basis(dir / "absent.json");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().kind, fb::ErrorKind::IO);
}

TEST(IoBasis, MalformedJsonIsIoError) {
  auto stamp = fb::parse_basis(R"({"elements": )");
  ASSERT_FALSE(stamp.has_value());
  EXPECT_EQ(stamp.error().kind, fb::ErrorKind::IO);
}

TEST(IoBasis, WrongContentIsInvalidConfig) {
  const std::array<std::string_view, 10> bad{
      R"([])",
      R"({})",
      R"({"elements": []})",
      R"({"elements": {"H": {"electron_shells": []}}})",
      R"({"elements": {"1": {}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0x"], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": [1.0], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0", "2.0"], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0, 1, 2],
          "exponents": ["1.0"], "coefficients": [["1.0"], ["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0"]}]}}})",
  };
  for (const std::string_view text : bad) {
    auto stamp = fb::parse_basis(text);
    ASSERT_FALSE(stamp.has_value()) << "accepted: " << text;
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig) << text;
  }
}

TEST(IoBasis, OnlySphericalFunctionTypesAreAccepted) {
  auto shell_with = [](std::string_view function_type, std::int32_t l) {
    const std::string tag =
        function_type.empty() ? "" : std::format(R"("function_type": "{}", )", function_type);
    return fb::parse_basis(std::format(
        R"({{"elements": {{"1": {{"electron_shells": [{{{}"angular_momentum": [{}],
            "exponents": ["1.0"], "coefficients": [["1.0"]]}}]}}}}}})",
        tag, l));
  };
  // An explicit spherical tag is accepted at any l.
  for (const std::int32_t l : {0, 2, 3}) {
    auto stamp = shell_with("gto_spherical", l);
    ASSERT_TRUE(stamp.has_value()) << "gto_spherical, l=" << l << ": " << stamp.error().message;
  }
  // Cartesian functions are not supported, at any l.
  for (const std::int32_t l : {0, 1, 2, 3}) {
    auto stamp = shell_with("gto_cartesian", l);
    ASSERT_FALSE(stamp.has_value()) << "gto_cartesian, l=" << l;
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig);
  }
  // Untagged or bare "gto" is exact as spherical for s and p ...
  EXPECT_TRUE(shell_with("", 1).has_value());
  EXPECT_TRUE(shell_with("gto", 0).has_value());
  // ... and ambiguous from d up.
  for (const std::string_view ambiguous : {"", "gto"}) {
    auto stamp = shell_with(ambiguous, 2);
    ASSERT_FALSE(stamp.has_value()) << "'" << ambiguous << "'";
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig);
  }
  // Slater-type orbitals are not Gaussian shells.
  EXPECT_FALSE(shell_with("sto", 0).has_value());
}

} // namespace
