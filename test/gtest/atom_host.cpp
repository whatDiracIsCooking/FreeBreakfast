// Host-only cases for atom: the Element enum and the Atom<T> aggregate.

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
static_assert(std::is_aggregate_v<fb::Atom<double>>);

TEST(Atom, ElementValueIsAtomicNumber) {
  EXPECT_EQ(static_cast<int>(fb::Element::kSentinel), 0);
  EXPECT_EQ(static_cast<int>(fb::Element::H), 1);
  EXPECT_EQ(static_cast<int>(fb::Element::C), 6);
  EXPECT_EQ(static_cast<int>(fb::Element::O), 8);
  EXPECT_EQ(static_cast<int>(fb::Element::Fe), 26);
  EXPECT_EQ(static_cast<int>(fb::Element::U), 92);
  EXPECT_EQ(static_cast<int>(fb::Element::Og), 118);
}

TEST(Atom, DefaultElementIsSentinel) {
  EXPECT_EQ(fb::Element{}, fb::Element::kSentinel);
}

TEST(Atom, DefaultAtomSitsAtOriginWithNoElement) {
  const fb::Atom<double> atom{};
  EXPECT_EQ(atom.x, 0.0);
  EXPECT_EQ(atom.y, 0.0);
  EXPECT_EQ(atom.z, 0.0);
  EXPECT_EQ(atom.element, fb::Element::kSentinel);
}

TEST(Atom, BraceInitialisesCoordinatesThenElement) {
  const fb::Atom<float> atom{1.0F, -2.5F, 3.25F, fb::Element::N};
  EXPECT_EQ(atom.x, 1.0F);
  EXPECT_EQ(atom.y, -2.5F);
  EXPECT_EQ(atom.z, 3.25F);
  EXPECT_EQ(atom.element, fb::Element::N);
}

} // namespace
