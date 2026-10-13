// Host-only cases for io/basis: the BSE JSON reader and its placement onto a
// molecule.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;
using fb::Harmonic;

// STO-3G hydrogen; oxygen with a plain s, an sp block (one column per
// momentum), a two-column general contraction, and a tagged spherical d.
constexpr std::string_view kBasis = R"({
  "elements": {
    "1": {"electron_shells": [
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["3.42525091", "0.62391373", "0.16885540"],
       "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]}
    ]},
    "8": {"electron_shells": [
      {"angular_momentum": [0],
       "exponents": ["130.7093200"], "coefficients": [["1.0"]]},
      {"function_type": "gto", "angular_momentum": [0, 1],
       "exponents": ["5.0331513", "1.1695961"],
       "coefficients": [["-0.09996723", "0.39951283"], ["0.15591627", "0.60768372"]]},
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["0.38", "0.12"],
       "coefficients": [["1.0", "0.0"], ["0.0", "1.0"]]},
      {"function_type": "gto_spherical", "angular_momentum": [2],
       "exponents": ["1.185"], "coefficients": [["1.0E+00"]]}
    ]}
  }
})";

fb::Molecule water() {
  return fb::Molecule({Atom<double>{0.0, 0.0, 0.2217, Element::O},
                       Atom<double>{0.0, 1.4309, -0.8867, Element::H},
                       Atom<double>{0.0, -1.4309, -0.8867, Element::H}});
}

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
  const std::array<int, 6> momenta{0, 0, 1, 0, 0, 2};
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
  // Only the d shell is spherical.
  EXPECT_EQ(o[5].harmonic(), Harmonic::Spherical);
  EXPECT_EQ(o[1].harmonic(), Harmonic::Cartesian);
}

TEST(IoBasis, ParseAndAssignLaysTheBasisOverTheMolecule) {
  auto basis = fb::parse_and_assign_basis(kBasis, water());
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  // O: 6 shells (s, s, p, s, s, d), then one s per H.
  EXPECT_EQ(basis->nshells(), 8U);
  EXPECT_EQ(basis->nao(), 1U + 1U + 3U + 1U + 1U + 6U + 1U + 1U);
  EXPECT_EQ(basis->spherical_nao(), basis->nao() - 1U); // d: 5, not 6
  EXPECT_EQ(basis->shells()[6].center().y, 1.4309);
  EXPECT_EQ(basis->shells()[7].center().y, -1.4309);
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

TEST(IoBasis, HarmonicTypeResolution) {
  auto shell_with = [](std::string_view function_type, int l) {
    const std::string tag =
        function_type.empty() ? "" : std::format(R"("function_type": "{}", )", function_type);
    return fb::parse_basis(std::format(
        R"({{"elements": {{"1": {{"electron_shells": [{{{}"angular_momentum": [{}],
            "exponents": ["1.0"], "coefficients": [["1.0"]]}}]}}}}}})",
        tag, l));
  };
  auto harmonic_of = [&](std::string_view function_type, int l) {
    return (*shell_with(function_type, l)->shells_for(1))[0].harmonic();
  };
  // Explicit tags are honored at any l.
  EXPECT_EQ(harmonic_of("gto_spherical", 2), Harmonic::Spherical);
  EXPECT_EQ(harmonic_of("gto_cartesian", 2), Harmonic::Cartesian);
  // Untagged or bare "gto" is exact as Cartesian for s and p ...
  EXPECT_EQ(harmonic_of("", 1), Harmonic::Cartesian);
  EXPECT_EQ(harmonic_of("gto", 0), Harmonic::Cartesian);
  // ... and ambiguous from d up.
  for (const std::string_view ambiguous : {"", "gto"}) {
    auto stamp = shell_with(ambiguous, 2);
    ASSERT_FALSE(stamp.has_value()) << "'" << ambiguous << "'";
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig);
  }
  // Slater-type orbitals are not Gaussian shells.
  EXPECT_FALSE(shell_with("sto", 0).has_value());
}

TEST(IoBasis, UncoveredElementIsInvalidConfig) {
  const fb::Molecule co({Atom<double>{0.0, 0.0, 0.0, Element::C},
                         Atom<double>{0.0, 0.0, 2.1, Element::O}});
  auto basis = fb::parse_and_assign_basis(kBasis, co);
  ASSERT_FALSE(basis.has_value());
  EXPECT_EQ(basis.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(basis.error().message.find("Z=6"), std::string::npos) << basis.error().message;
}

TEST(IoBasis, AuxBasisNamesTheFittingRoleOnCoverageFailure) {
  auto ok = fb::parse_and_assign_aux_basis(kBasis, water());
  ASSERT_TRUE(ok.has_value()) << ok.error().message;
  EXPECT_EQ(ok->nshells(), 8U);

  const fb::Molecule ch({Atom<double>{0.0, 0.0, 0.0, Element::C},
                         Atom<double>{0.0, 0.0, 2.0, Element::H}});
  auto bad = fb::parse_and_assign_aux_basis(kBasis, ch);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_TRUE(bad.error().message.starts_with("auxiliary (density-fitting) basis"))
      << bad.error().message;
  EXPECT_NE(bad.error().message.find("Z=6"), std::string::npos);
}

} // namespace
