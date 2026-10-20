// Host-only cases for molecule: the distance between two points and an atom's
// nuclear charge. Everything that builds a Molecule is molecule_gpu.cpp's: a
// Molecule owns its nuclei on the device.

#include <gtest/gtest.h>

import std;
import free_breakfast.molecule;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

TEST(Molecule, DistanceIsEuclidean) {
  EXPECT_DOUBLE_EQ(
      fb::distance(fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Vec<double, 3>{3.0, 4.0, 12.0}), 13.0);
  EXPECT_DOUBLE_EQ(
      fb::distance(fb::Vec<double, 3>{1.0, 2.0, 3.0}, fb::Vec<double, 3>{1.0, 2.0, 3.0}), 0.0);
}

TEST(Molecule, NuclearChargeReadsTheAtom) {
  const Atom atom{{1.5, -2.0, 0.25}, Element::Cl};
  EXPECT_EQ(fb::nuclear_charge(atom), 17);
}

} // namespace
