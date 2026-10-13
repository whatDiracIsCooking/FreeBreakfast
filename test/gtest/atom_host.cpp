// Host-only cases for atom: the Element enum and the Atom aggregate.

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

TEST(Atom, BraceInitialisesPositionThenElement) {
  const fb::Atom atom{{1.0, -2.5, 3.25}, fb::Element::N};
  EXPECT_EQ(atom.position.x, 1.0);
  EXPECT_EQ(atom.position.y, -2.5);
  EXPECT_EQ(atom.position.z, 3.25);
  EXPECT_EQ(atom.element, fb::Element::N);
}

} // namespace
