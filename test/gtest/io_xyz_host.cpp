// Host-only cases for io/xyz: the XYZ geometry reader.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants;
import free_breakfast.molecule;
import free_breakfast.io.base;
import free_breakfast.io.xyz;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::Element;

constexpr std::string_view kWater = "3\n"
                                    "water, Angstrom\n"
                                    "O   0.000000   0.000000   0.117300\n"
                                    "H   0.000000   0.757200  -0.469200\n"
                                    "H   0.000000  -0.757200  -0.469200\n";

TEST(IoXyz, ParsesAtomsAndConvertsAngstromToBohr) {
  auto molecule = fb::parse_xyz(kWater);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  ASSERT_EQ(molecule->natoms(), 3U);
  EXPECT_EQ(molecule->charge(), 0);
  EXPECT_EQ(molecule->multiplicity(), 1);

  const auto &atoms = molecule->atoms();
  EXPECT_EQ(atoms[0].element, Element::O);
  EXPECT_EQ(atoms[1].element, Element::H);
  EXPECT_EQ(atoms[2].element, Element::H);
  EXPECT_DOUBLE_EQ(atoms[0].z, 0.1173 * fb::kBohrPerAngstrom<double>);
  EXPECT_DOUBLE_EQ(atoms[1].y, 0.7572 * fb::kBohrPerAngstrom<double>);
  EXPECT_DOUBLE_EQ(atoms[2].z, -0.4692 * fb::kBohrPerAngstrom<double>);
  EXPECT_EQ(molecule->electron_count(), 10);
}

TEST(IoXyz, AttachesChargeAndMultiplicity) {
  auto molecule = fb::parse_xyz(kWater, 1, 2);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  EXPECT_EQ(molecule->charge(), 1);
  EXPECT_EQ(molecule->multiplicity(), 2);
  EXPECT_EQ(molecule->electron_count(), 9);
}

TEST(IoXyz, ToleratesCrlfTrailingBlanksAndSymbolCase) {
  auto molecule = fb::parse_xyz("2\r\ncomment\r\nCL 0 0 0\r\nna 0 0 2.5\r\n\r\n   \n");
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  ASSERT_EQ(molecule->natoms(), 2U);
  EXPECT_EQ(molecule->atoms()[0].element, Element::Cl);
  EXPECT_EQ(molecule->atoms()[1].element, Element::Na);
}

TEST(IoXyz, EmptyGeometryIsAnEmptyMolecule) {
  auto molecule = fb::parse_xyz("0\nnothing here\n");
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  EXPECT_EQ(molecule->natoms(), 0U);
}

TEST(IoXyz, MalformedInputIsIoError) {
  for (const std::string_view text : {
           std::string_view{"3\n"},                            // no comment line
           std::string_view{"three\ncomment\n"},               // count not a number
           std::string_view{"-1\ncomment\n"},                  // negative count
           std::string_view{"1\ncomment\nH 0 0\n"},            // missing coordinate
           std::string_view{"1\ncomment\nH 0 zero 0\n"},       // non-numeric coordinate
           std::string_view{"1\ncomment\nXx 0 0 0\n"},         // unknown element
           std::string_view{"1\ncomment\nHel 0 0 0\n"},        // symbol too long
           std::string_view{"2\ncomment\nH 0 0 0\n"},          // count too high
           std::string_view{"1\ncomment\nH 0 0 0\nH 0 0 1\n"}, // count too low
       }) {
    auto molecule = fb::parse_xyz(text);
    ASSERT_FALSE(molecule.has_value()) << text;
    EXPECT_EQ(molecule.error().kind, fb::ErrorKind::IO) << text;
  }
}

TEST(IoXyz, ReadsFromDisk) {
  const fb::test::ScratchDir dir("io_xyz");
  const auto path = dir / "water.xyz";
  ASSERT_TRUE(fb::write_file(path, kWater).has_value());

  auto molecule = fb::read_xyz(path);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  EXPECT_EQ(molecule->natoms(), 3U);
}

TEST(IoXyz, ReadOfMissingFileIsIoError) {
  const fb::test::ScratchDir dir("io_xyz");
  auto molecule = fb::read_xyz(dir / "absent.xyz");
  ASSERT_FALSE(molecule.has_value());
  EXPECT_EQ(molecule.error().kind, fb::ErrorKind::IO);
}

} // namespace
