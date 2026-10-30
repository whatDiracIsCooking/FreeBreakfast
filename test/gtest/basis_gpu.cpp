// GPU cases for basis: placing a stamp onto a molecule. A Molecule owns its
// nuclei on the device, so building one needs a GPU. Everything else in
// src/basis is host-only, in test/basis.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

fb::BasisStamp water_stamp() {
  std::map<std::int32_t, std::vector<fb::ShellStamp>> table;
  table.emplace(1, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{1.2, 1.0}})});
  table.emplace(8, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{9.0, 1.0}}),
                                               fb::ShellStamp(1, {{2.0, 1.0}})});
  table.emplace(6, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{5.0, 1.0}})}); // unused
  return fb::BasisStamp(std::move(table));
}

TEST(Basis, PlaceFollowsAtomOrderThenStampOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule water(res,
                           {Atom{{0.0, 0.0, 0.2}, Element::O}, Atom{{0.0, 1.4, -0.9}, Element::H},
                            Atom{{0.0, -1.4, -0.9}, Element::H}});
  const fb::BasisStamp stamp = water_stamp();
  ASSERT_NE(stamp.shells_for(8), nullptr);
  EXPECT_EQ(stamp.shells_for(7), nullptr);

  auto basis = fb::place(stamp, water);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  ASSERT_EQ(basis->nshells(), 4U);
  EXPECT_EQ(basis->nao_cart(), 6U); // O: s + p, then two H s
  const std::array<std::int32_t, 4> momenta{0, 1, 0, 0};
  const std::array<double, 4> center_y{0.0, 0.0, 1.4, -1.4};
  for (std::size_t s = 0; s < 4; ++s) {
    EXPECT_EQ(basis->shells()[s].angular_momentum(), momenta[s]);
    EXPECT_EQ(basis->shells()[s].center().y, center_y[s]);
  }
  ASSERT_EQ(basis->natoms(), 3U);
  const std::array<std::uint32_t, 3> first_shell{0, 2, 3};
  const std::array<std::uint32_t, 3> shell_count{2, 1, 1};
  for (std::uint32_t a = 0; a < 3; ++a) {
    EXPECT_EQ(basis->atom_shell_offset(a), first_shell[a]);
    EXPECT_EQ(basis->atom_shell_count(a), shell_count[a]);
  }
}

TEST(Basis, PlaceOntoUncoveredElementIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule nh(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::N}, Atom{{0.0, 0.0, 1.9}, Element::H}});
  auto basis = fb::place(water_stamp(), nh);
  ASSERT_FALSE(basis.has_value());
  EXPECT_EQ(basis.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(basis.error().message.find("Z=7"), std::string::npos) << basis.error().message;
}

TEST(Basis, PlacingOneStampAtTwoGeometriesAgreesOnLayout) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisStamp stamp = water_stamp().primitive_normalized();
  const fb::Molecule a(res, {Atom{{0.0, 0.0, 0.0}, Element::O}, Atom{{0.0, 0.0, 1.8}, Element::H}});
  const fb::Molecule b(res, {Atom{{0.5, 0.0, 0.0}, Element::O}, Atom{{0.5, 0.0, 2.1}, Element::H}});
  auto ba = fb::place(stamp, a);
  auto bb = fb::place(stamp, b);
  ASSERT_TRUE(ba.has_value() && bb.has_value());
  ASSERT_EQ(ba->nao_cart(), bb->nao_cart());
  for (std::uint32_t s = 0; s < ba->nshells(); ++s) {
    EXPECT_EQ(ba->ao_offset_cart(s), bb->ao_offset_cart(s));
    EXPECT_EQ(ba->shells()[s].primitives()[0].coefficient,
              bb->shells()[s].primitives()[0].coefficient);
  }
  EXPECT_EQ(bb->shells()[0].center().x, 0.5);
}

} // namespace
