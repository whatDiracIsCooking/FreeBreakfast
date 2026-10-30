// Host-only cases for atom: the Element enum, the Atom aggregate and the
// symbol table. Nothing in src/atom touches a device, so every case here runs
// on a device-less box.

#include <gtest/gtest.h>

// The header path CUDA / HIP code takes, in the same TU as the module import:
// both must name the one entity, not two lookalikes.
#include <atom.h>

import std;
import free_breakfast.atom;

namespace {

namespace fb = free_breakfast;

static_assert(sizeof(fb::Element) == 1, "Element's underlying type is one byte");
static_assert(std::is_same_v<std::underlying_type_t<fb::Element>, std::uint8_t>);
static_assert(std::is_aggregate_v<fb::Atom>);
static_assert(std::is_same_v<decltype(fb::Atom::position), fb::Vec<double, 3>>);

TEST(Atom, ElementValueIsAtomicNumber) {
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::Sentinel), 0);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::H), 1);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::C), 6);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::O), 8);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::Fe), 26);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::U), 92);
  EXPECT_EQ(static_cast<std::int32_t>(fb::Element::Og), 118);
}

TEST(Atom, DefaultElementIsSentinel) {
  EXPECT_EQ(fb::Element{}, fb::Element::Sentinel);
}

TEST(Atom, DefaultAtomSitsAtOriginWithNoElement) {
  const fb::Atom atom{};
  EXPECT_EQ(atom.position.x, 0.0);
  EXPECT_EQ(atom.position.y, 0.0);
  EXPECT_EQ(atom.position.z, 0.0);
  EXPECT_EQ(atom.element, fb::Element::Sentinel);
}

// Usable in a constant expression, as a header-defined aggregate must be for
// device code to build one.
static_assert(fb::Atom{}.element == fb::Element::Sentinel);
static_assert(fb::Atom{{0.0, 0.0, 1.5}, fb::Element::O}.position.z == 1.5);

TEST(Atom, BraceInitialisesPositionThenElement) {
  const fb::Atom atom{{1.0, -2.5, 3.25}, fb::Element::N};
  EXPECT_EQ(atom.position.x, 1.0);
  EXPECT_EQ(atom.position.y, -2.5);
  EXPECT_EQ(atom.position.z, 3.25);
  EXPECT_EQ(atom.element, fb::Element::N);
}

TEST(Atom, CopyIsIndependentOfTheOriginal) {
  const fb::Atom original{{1.0, 2.0, 3.0}, fb::Element::C};
  fb::Atom copy = original;
  copy.position.x = -1.0;
  copy.element = fb::Element::Si;
  EXPECT_EQ(original.position.x, 1.0);
  EXPECT_EQ(original.element, fb::Element::C);
  EXPECT_EQ(copy.position.x, -1.0);
  EXPECT_EQ(copy.position.y, 2.0);
  EXPECT_EQ(copy.element, fb::Element::Si);
}

static_assert(fb::element_symbol(fb::Element::Og) == "Og");
static_assert(fb::element_from_symbol("Og") == fb::Element::Og);
static_assert(fb::element_symbol(fb::Element::Sentinel).empty());
static_assert(fb::element_from_symbol("") == fb::Element::Sentinel);

TEST(Atom, EverySymbolRoundTrips) {
  for (std::int32_t z = 1; z <= 118; ++z) {
    const auto element = static_cast<fb::Element>(z);
    EXPECT_EQ(fb::element_from_symbol(fb::element_symbol(element)), element) << z;
  }
  EXPECT_EQ(fb::element_symbol(fb::Element::Sentinel), "");
}

TEST(Atom, SymbolsAreACapitalThenAtMostOneLowercase) {
  for (std::int32_t z = 1; z <= 118; ++z) {
    const auto symbol = fb::element_symbol(static_cast<fb::Element>(z));
    ASSERT_TRUE(symbol.size() == 1 || symbol.size() == 2) << z;
    EXPECT_TRUE(symbol[0] >= 'A' && symbol[0] <= 'Z') << z << ' ' << symbol;
    if (symbol.size() == 2) {
      EXPECT_TRUE(symbol[1] >= 'a' && symbol[1] <= 'z') << z << ' ' << symbol;
    }
  }
}

TEST(Atom, SymbolsOfSampledElements) {
  EXPECT_EQ(fb::element_symbol(fb::Element::H), "H");
  EXPECT_EQ(fb::element_symbol(fb::Element::C), "C");
  EXPECT_EQ(fb::element_symbol(fb::Element::Cl), "Cl");
  EXPECT_EQ(fb::element_symbol(fb::Element::Fe), "Fe");
  EXPECT_EQ(fb::element_symbol(fb::Element::U), "U");
}

TEST(Atom, SymbolLookupIgnoresCase) {
  EXPECT_EQ(fb::element_from_symbol("Cl"), fb::Element::Cl);
  EXPECT_EQ(fb::element_from_symbol("CL"), fb::Element::Cl);
  EXPECT_EQ(fb::element_from_symbol("cl"), fb::Element::Cl);
  EXPECT_EQ(fb::element_from_symbol("cL"), fb::Element::Cl);
  EXPECT_EQ(fb::element_from_symbol("h"), fb::Element::H);
}

// A one-letter symbol is not a prefix match for a two-letter one, and folding
// a two-letter symbol's case never splits it into two elements ("CO" is
// cobalt, not carbon).
TEST(Atom, SymbolLookupMatchesTheWholeSymbol) {
  EXPECT_EQ(fb::element_from_symbol("C"), fb::Element::C);
  EXPECT_EQ(fb::element_from_symbol("Co"), fb::Element::Co);
  EXPECT_EQ(fb::element_from_symbol("CO"), fb::Element::Co);
  EXPECT_EQ(fb::element_from_symbol("Cs"), fb::Element::Cs);
  EXPECT_EQ(fb::element_from_symbol("N"), fb::Element::N);
  EXPECT_EQ(fb::element_from_symbol("NA"), fb::Element::Na);
}

TEST(Atom, UnknownSymbolIsSentinel) {
  EXPECT_EQ(fb::element_from_symbol(""), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol("Xx"), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol("Abc"), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol("J"), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol("1"), fb::Element::Sentinel);
}

// No trimming: the readers split on whitespace before they look a symbol up.
TEST(Atom, PaddedSymbolIsSentinel) {
  EXPECT_EQ(fb::element_from_symbol(" H"), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol("H "), fb::Element::Sentinel);
  EXPECT_EQ(fb::element_from_symbol(" Fe"), fb::Element::Sentinel);
}

} // namespace
