// Host-only cases for basis's :basis_stamp: the element -> shells table and
// its normalized copy. Placing a stamp onto a Molecule is
// test/gtest/basis_gpu.cpp's, because a Molecule owns its nuclei on the device.

#include <gtest/gtest.h>

import std;
import free_breakfast.basis;

namespace {

namespace fb = free_breakfast;

fb::BasisStamp water_stamp(std::string name = {}) {
  std::map<std::int32_t, std::vector<fb::ShellStamp>> table;
  table.emplace(1, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{1.2, 1.0}})});
  table.emplace(8, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{9.0, 1.0}}),
                                               fb::ShellStamp(1, {{2.0, 1.0}})});
  return fb::BasisStamp(std::move(table), std::move(name));
}

TEST(BasisStamp, EmptyStampCarriesNothing) {
  const fb::BasisStamp stamp;
  EXPECT_TRUE(stamp.element_shells().empty());
  EXPECT_TRUE(stamp.name().empty());
  EXPECT_EQ(stamp.shells_for(1), nullptr);
}

TEST(BasisStamp, ShellsForLooksUpByAtomicNumber) {
  const fb::BasisStamp stamp = water_stamp();
  EXPECT_EQ(stamp.element_shells().size(), 2U);
  EXPECT_EQ(stamp.shells_for(7), nullptr);
  const std::vector<fb::ShellStamp> *oxygen = stamp.shells_for(8);
  ASSERT_NE(oxygen, nullptr);
  ASSERT_EQ(oxygen->size(), 2U);
  EXPECT_EQ((*oxygen)[0].angular_momentum(), 0);
  EXPECT_EQ((*oxygen)[1].angular_momentum(), 1);
  EXPECT_EQ((*oxygen)[1].primitives()[0].exponent, 2.0);
  // The pointer is into the stamp's own table, not a copy.
  EXPECT_EQ(oxygen, &stamp.element_shells().at(8));
}

TEST(BasisStamp, NameIsKeptByPrimitiveNormalized) {
  EXPECT_TRUE(water_stamp().name().empty());
  const fb::BasisStamp named = water_stamp("sto-3g");
  EXPECT_EQ(named.name(), "sto-3g");
  EXPECT_EQ(named.primitive_normalized().name(), "sto-3g");
}

// The stamp-level normalization is the shell-level one, element by element and
// shell by shell, with the table's shape unchanged.
TEST(BasisStamp, PrimitiveNormalizedNormalizesEveryShell) {
  const fb::BasisStamp raw = water_stamp();
  const fb::BasisStamp normalized = raw.primitive_normalized();
  ASSERT_EQ(normalized.element_shells().size(), raw.element_shells().size());
  for (const auto &[z, shells] : raw.element_shells()) {
    const std::vector<fb::ShellStamp> *out = normalized.shells_for(z);
    ASSERT_NE(out, nullptr) << "Z=" << z;
    ASSERT_EQ(out->size(), shells.size()) << "Z=" << z;
    for (std::size_t s = 0; s < shells.size(); ++s) {
      const fb::ShellStamp expected = shells[s].primitive_normalized();
      EXPECT_EQ((*out)[s].angular_momentum(), expected.angular_momentum());
      ASSERT_EQ((*out)[s].nprimitives(), expected.nprimitives());
      for (std::uint32_t p = 0; p < expected.nprimitives(); ++p) {
        EXPECT_EQ((*out)[s].primitives()[p].exponent, expected.primitives()[p].exponent);
        EXPECT_EQ((*out)[s].primitives()[p].coefficient, expected.primitives()[p].coefficient);
      }
    }
  }
  // The source is untouched: a unit coefficient stays one.
  EXPECT_EQ(raw.shells_for(1)->front().primitives()[0].coefficient, 1.0);
  EXPECT_NE(normalized.shells_for(1)->front().primitives()[0].coefficient, 1.0);
}

} // namespace
