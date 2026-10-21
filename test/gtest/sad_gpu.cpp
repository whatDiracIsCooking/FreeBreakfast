// GPU cases for the SAD initial guess: the guess over a basis placed on a
// Molecule. A Molecule owns its nuclei on the device, so building one needs a
// GPU; the guess itself is sad_host.cpp's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.sad;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

// O: s, s, p, d (the d unoccupied); H: s.
fb::BasisStamp water_stamp() {
  std::map<std::int32_t, std::vector<fb::ShellStamp>> table;
  table.emplace(1, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{1.2, 1.0}})});
  table.emplace(8, std::vector<fb::ShellStamp>{
                       fb::ShellStamp(0, {{90.0, 1.0}}), fb::ShellStamp(0, {{9.0, 1.0}}),
                       fb::ShellStamp(1, {{2.0, 1.0}}), fb::ShellStamp(2, {{0.8, 1.0}})});
  return fb::BasisStamp(std::move(table));
}

fb::Molecule water(fb::DeviceResources &res) {
  return fb::Molecule(res, {Atom{{0.0, 0.0, 0.2}, Element::O}, Atom{{0.0, 1.4, -0.9}, Element::H},
                            Atom{{0.0, -1.4, -0.9}, Element::H}});
}

TEST(Sad, DensityOverPlacedWaterIsBlockDiagonalInTheSphericalLayout) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule = water(res);
  auto basis = fb::place(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  ASSERT_EQ(basis->nao_sphe(), 12U); // O: 1 + 1 + 3 + 5, then two H s
  ASSERT_NE(basis->nao_cart(), basis->nao_sphe());

  auto density = fb::sad_density(*basis, molecule.atoms());
  ASSERT_TRUE(density.has_value()) << density.error().message;
  const std::size_t n = 12;
  ASSERT_EQ(density->size(), n * n);
  const auto atom_of = [](std::size_t ao) -> std::size_t { return ao < 10 ? 0 : ao - 9; };
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      const double d = (*density)[i + (n * j)];
      EXPECT_EQ(d, (*density)[j + (n * i)]) << i << "," << j;
      if (atom_of(i) != atom_of(j)) {
        EXPECT_EQ(d, 0.0) << i << "," << j;
      }
    }
  }
  // The O d shell is unoccupied, and both H share one SCF.
  for (std::size_t ao = 5; ao < 10; ++ao) {
    EXPECT_EQ((*density)[ao + (n * ao)], 0.0) << ao;
  }
  EXPECT_GT((*density)[10 + (n * 10)], 0.0);
  EXPECT_EQ((*density)[10 + (n * 10)], (*density)[11 + (n * 11)]);
}

TEST(Sad, BasisPlacedOverAnotherMoleculeIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule = water(res);
  const fb::Molecule oh(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::O}, Atom{{0.0, 0.0, 1.8}, Element::H}});
  auto basis = fb::place(water_stamp(), oh);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  auto mismatched = fb::sad_density(*basis, molecule.atoms());
  ASSERT_FALSE(mismatched.has_value());
  EXPECT_EQ(mismatched.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
