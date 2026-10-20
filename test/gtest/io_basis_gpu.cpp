// GPU cases for io/basis: the BSE JSON reader's placement onto a molecule. A
// Molecule owns its nuclei on the device, so building one needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;

#include "io_basis_json.h"

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;
using fb::test::kBasis;

fb::Molecule water(fb::DeviceResources &res) {
  return fb::Molecule(res, {Atom{{0.0, 0.0, 0.2217}, Element::O},
                            Atom{{0.0, 1.4309, -0.8867}, Element::H},
                            Atom{{0.0, -1.4309, -0.8867}, Element::H}});
}

TEST(IoBasis, ParseAndAssignLaysTheBasisOverTheMolecule) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  auto basis = fb::parse_and_assign_basis(kBasis, water(res));
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  // O: 6 shells (s, s, p, s, s, d), then one s per H.
  EXPECT_EQ(basis->nshells(), 8U);
  EXPECT_EQ(basis->nao_cart(), 1U + 1U + 3U + 1U + 1U + 6U + 1U + 1U);
  EXPECT_EQ(basis->nao_sphe(), basis->nao_cart() - 1U); // d: 5, not 6
  // U maps the Cartesian AOs the kernels evaluate onto the spherical layout.
  EXPECT_EQ(fb::spherical_transform_matrix(*basis).size(),
            std::size_t{basis->nao_cart()} * basis->nao_sphe());
  EXPECT_EQ(basis->shells()[6].center().y, 1.4309);
  EXPECT_EQ(basis->shells()[7].center().y, -1.4309);
}

TEST(IoBasis, UncoveredElementIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule co(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::C}, Atom{{0.0, 0.0, 2.1}, Element::O}});
  auto basis = fb::parse_and_assign_basis(kBasis, co);
  ASSERT_FALSE(basis.has_value());
  EXPECT_EQ(basis.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(basis.error().message.find("Z=6"), std::string::npos) << basis.error().message;
}

TEST(IoBasis, AuxBasisNamesTheFittingRoleOnCoverageFailure) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  auto ok = fb::parse_and_assign_aux_basis(kBasis, water(res));
  ASSERT_TRUE(ok.has_value()) << ok.error().message;
  EXPECT_EQ(ok->nshells(), 8U);

  const fb::Molecule ch(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::C}, Atom{{0.0, 0.0, 2.0}, Element::H}});
  auto bad = fb::parse_and_assign_aux_basis(kBasis, ch);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_TRUE(bad.error().message.starts_with("auxiliary (density-fitting) basis"))
      << bad.error().message;
  EXPECT_NE(bad.error().message.find("Z=6"), std::string::npos);
}

} // namespace
