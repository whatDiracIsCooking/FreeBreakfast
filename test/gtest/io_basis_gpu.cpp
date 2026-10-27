// GPU cases for io/basis: the BSE JSON reader's placement onto a molecule, and
// the compiled-in fitting sets' placement. A Molecule owns its nuclei on the
// device, so building one needs a GPU.

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

// The compiled-in fitting sets placed over H2O, as the DF steps will: the
// counts pin data/aux_basis/ (tools/jkfit_from_pyscf.py's conversion of
// PySCF's tables), and match PySCF's naux for H2O, 116 and 139.
TEST(IoBasisAux, AJkfitSetPlacesOverWater) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule h2o = water(res);
  struct Expected {
    std::string_view name;
    std::uint32_t nshells;
    std::uint32_t nao_sphe;
    std::int32_t oxygen_lmax;
  };
  // O: 10s 7p 5d 2f in DZ, one g more in TZ; H: 4s 3p 2d, one f more in TZ.
  for (const auto &[name, nshells, nao_sphe, oxygen_lmax] :
       std::array<Expected, 2>{{{"cc-pVDZ-JKFIT", 24 + 2 * 9, 70 + 2 * 23, 3},
                                {"cc-pVTZ-JKFIT", 25 + 2 * 10, 79 + 2 * 30, 4}}}) {
    SCOPED_TRACE(name);
    auto stamp = fb::resolve_aux_basis(name);
    ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
    auto aux = fb::place(stamp->primitive_normalized(), h2o);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    EXPECT_EQ(aux->nshells(), nshells);
    EXPECT_EQ(aux->nao_sphe(), nao_sphe);
    std::int32_t lmax = 0;
    for (std::uint32_t s = aux->atom_shell_offset(0); s < aux->atom_shell_offset(1); ++s) {
      lmax = std::max(lmax, aux->shells()[s].angular_momentum());
    }
    EXPECT_EQ(lmax, oxygen_lmax);
  }
}

TEST(IoBasisAux, AnElementTheSetLacksIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // Li: the JKFIT sets have H, then B onward.
  const fb::Molecule lih(res,
                         {Atom{{0.0, 0.0, 0.0}, Element::Li}, Atom{{0.0, 0.0, 3.0}, Element::H}});
  auto stamp = fb::resolve_aux_basis("cc-pvdz-jkfit");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto aux = fb::place(stamp->primitive_normalized(), lih);
  ASSERT_FALSE(aux.has_value());
  EXPECT_EQ(aux.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(aux.error().message.find("Z=3"), std::string::npos) << aux.error().message;
}

} // namespace
