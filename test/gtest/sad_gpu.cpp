// GPU cases for the SAD initial guess: the assembled guess over a placed
// basis. A Molecule owns its nuclei on the device, so building one needs a GPU.

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

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

fb::Shell shell_at_origin(std::int32_t l) {
  return fb::Shell(l, {{1.0, 1.0}}, kOrigin);
}

double sum(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

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

  auto diagonal = fb::sad_diagonal(*basis, molecule);
  ASSERT_TRUE(diagonal.has_value()) << diagonal.error().message;
  const std::vector<double> expected{2.0, 2.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 0.0,
                                     0.0, 0.0, 0.0,       0.0,       1.0,       1.0};
  ASSERT_EQ(diagonal->size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ((*diagonal)[i], expected[i]) << "AO " << i;
  }
  EXPECT_DOUBLE_EQ(sum(*diagonal), static_cast<double>(molecule.total_nuclear_charge()));

  auto density = fb::sad_density(*basis, molecule);
  ASSERT_TRUE(density.has_value()) << density.error().message;
  const std::size_t n = expected.size();
  ASSERT_EQ(density->size(), n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ((*density)[i + (j * n)], i == j ? expected[i] : 0.0) << i << "," << j;
    }
  }
}

TEST(Sad, BasisWithoutTheMoleculesAtomGroupingIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule = water(res);

  const fb::BasisSet bare({shell_at_origin(0)});
  auto from_bare = fb::sad_density(bare, molecule);
  ASSERT_FALSE(from_bare.has_value());
  EXPECT_EQ(from_bare.error().kind, fb::ErrorKind::InvalidConfig);

  const fb::Molecule oh(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::O}, Atom{{0.0, 0.0, 1.8}, Element::H}});
  auto basis = fb::place(water_stamp(), oh);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  auto mismatched = fb::sad_diagonal(*basis, molecule);
  ASSERT_FALSE(mismatched.has_value());
  EXPECT_EQ(mismatched.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
