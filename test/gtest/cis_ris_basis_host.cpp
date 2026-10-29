// Host cases for cis.ris_basis: H2O's shell and function counts under the three
// shell choices, the exponent θ / R² against the Ghosh radius, gpu4pyscf's own
// documented exponents, and the two refusals.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.core.error_handling;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.cis.ris_basis;

namespace {

namespace fb = free_breakfast;

// Water in bohr, O first.
const std::vector<fb::Atom> kWater{
    {.position = {0.0, 0.0, 0.2217}, .element = fb::Element::O},
    {.position = {0.0, 1.4309, -0.8867}, .element = fb::Element::H},
    {.position = {0.0, -1.4309, -0.8867}, .element = fb::Element::H},
};

// Ghosh radius of O in Angstrom, the table's entry for Z = 8.
constexpr double kOxygenRadius = 0.4652;

TEST(CisRisBasis, WaterCountsUnderEachShellChoice) {
  struct Case {
    fb::cis::RisShells shells;
    std::uint32_t nshells;
    std::uint32_t oxygen_shells;
    std::uint32_t nao_sphe;
    std::uint32_t nao_cart;
  };
  // Spd is 11 spherical (1 + 3 + 5 on O, 1 per H); Cartesian it would be 12.
  for (const Case &c :
       {Case{fb::cis::RisShells::S, 3, 1, 3, 3}, Case{fb::cis::RisShells::Sp, 4, 2, 6, 6},
        Case{fb::cis::RisShells::Spd, 5, 3, 11, 12}}) {
    const auto aux = fb::cis::ris_basis(kWater, 0.2, c.shells);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    EXPECT_EQ(aux->nshells(), c.nshells);
    EXPECT_EQ(aux->nao_sphe(), c.nao_sphe);
    EXPECT_EQ(aux->nao_cart(), c.nao_cart);
    ASSERT_EQ(aux->natoms(), 3U);
    EXPECT_EQ(aux->atom_shell_count(0), c.oxygen_shells);
    EXPECT_EQ(aux->atom_shell_count(1), 1U);
    EXPECT_EQ(aux->atom_shell_count(2), 1U);
    EXPECT_TRUE(aux->name().empty());
    for (std::uint32_t s = 0; s < aux->nshells(); ++s) {
      const fb::Shell &shell = aux->shells()[s];
      EXPECT_EQ(shell.nprimitives(), 1U);
      // Within each atom the shells run s, p, d.
      const std::uint32_t atom = s < c.oxygen_shells ? 0 : s - c.oxygen_shells + 1;
      EXPECT_EQ(shell.angular_momentum(),
                static_cast<std::int32_t>(s - aux->atom_shell_offset(atom)));
      EXPECT_EQ(shell.center().z, kWater[atom].position.z);
    }
  }
  // The default is Sp.
  EXPECT_EQ(fb::cis::ris_basis(kWater)->nao_sphe(), 6U);
}

TEST(CisRisBasis, OxygenExponentIsThetaOverTheGhoshRadiusSquared) {
  constexpr double kTheta = 0.3;
  const auto aux = fb::cis::ris_basis(kWater, kTheta, fb::cis::RisShells::Spd);
  ASSERT_TRUE(aux.has_value()) << aux.error().message;
  const double radius = kOxygenRadius * fb::kBohrPerAngstrom<double>;
  const double alpha = kTheta / (radius * radius);
  for (std::uint32_t s = 0; s < aux->atom_shell_count(0); ++s) {
    const fb::Primitive &p = aux->shells()[s].primitives().front();
    EXPECT_DOUBLE_EQ(p.exponent, alpha) << "l = " << s;
  }
  // The coefficient is the s primitive's normalization (2α/π)^(3/4), as
  // primitive_normalized() folds into a coefficient of 1.
  EXPECT_DOUBLE_EQ(aux->shells()[0].primitives().front().coefficient,
                   std::pow(2.0 * alpha / std::numbers::pi, 0.75));
}

// The exponents gpu4pyscf's get_minimal_auxbasis docstring prints at θ = 0.2.
// Its bohr-per-Angstrom is 1.8897259885789 (CODATA 2014) against this tree's
// CODATA 2018 one, a relative 1.4e-7 in α, hence the tolerance.
TEST(CisRisBasis, MatchesGpu4pyscfDocumentedExponents) {
  const std::vector<fb::Atom> atoms{
      {.position = {0.0, 0.0, 0.0}, .element = fb::Element::C},
      {.position = {0.0, 0.0, 2.0}, .element = fb::Element::H},
      {.position = {0.0, 0.0, -2.7}, .element = fb::Element::O},
  };
  const auto aux = fb::cis::ris_basis(atoms, 0.2, fb::cis::RisShells::S);
  ASSERT_TRUE(aux.has_value()) << aux.error().message;
  const std::array<double, 3> expected{0.1320292535005648, 0.1999828038466018, 0.2587932305664396};
  for (std::uint32_t a = 0; a < 3; ++a) {
    const double alpha = aux->shells()[a].primitives().front().exponent;
    EXPECT_NEAR(alpha / expected[a], 1.0, 1e-6) << "atom " << a;
  }
}

TEST(CisRisBasis, RefusesANonPositiveThetaAndAnElementPastKrypton) {
  const auto invalid = std::optional{fb::ErrorKind::InvalidConfig};
  const auto kind = [](const fb::Result<fb::BasisSet> &r) {
    return r.has_value() ? std::nullopt : std::optional{r.error().kind};
  };
  for (const double theta : {0.0, -0.2, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ(kind(fb::cis::ris_basis(kWater, theta)), invalid) << "theta " << theta;
  }
  const std::vector<fb::Atom> krypton{{.position = {}, .element = fb::Element::Kr}};
  EXPECT_TRUE(fb::cis::ris_basis(krypton).has_value());
  const std::vector<fb::Atom> rubidium{{.position = {}, .element = fb::Element::Rb}};
  EXPECT_EQ(kind(fb::cis::ris_basis(rubidium)), invalid);
}

} // namespace
