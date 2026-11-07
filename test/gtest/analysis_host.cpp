// Host-only cases for src/analysis: partial_charges, orbital_populations and
// bond_orbitals over hand-written populations and weights. Each is host
// arithmetic over a host BasisSet, so nothing touches a device; the
// Localization step's GPU cases (pipeline_localization_gpu.cpp) check the same
// functions over real IAOs against PySCF.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.analysis.population;
import free_breakfast.analysis.bond;

namespace {

namespace fb = free_breakfast;

// partial_charges is host arithmetic, and refuses populations that are not
// whole columns of the atoms.
TEST(Population, PartialChargesAreZLessTheOccupiedPopulations) {
  const std::array atoms{fb::Atom{{0.0, 0.0, 0.0}, fb::Element::O},
                         fb::Atom{{0.0, 0.0, 1.8}, fb::Element::H}};
  // Two orbitals: one all on O, one split 0.75 / 0.25.
  const std::array populations{1.0, 0.0, 0.75, 0.25};
  const auto charges = fb::partial_charges(atoms, populations);
  ASSERT_TRUE(charges.has_value()) << charges.error().message;
  EXPECT_DOUBLE_EQ((*charges)[0], 8.0 - (2.0 * 1.75));
  EXPECT_DOUBLE_EQ((*charges)[1], 1.0 - (2.0 * 0.25));
  const auto one = fb::partial_charges(atoms, populations, 1.0);
  ASSERT_TRUE(one.has_value());
  EXPECT_DOUBLE_EQ((*one)[0], 8.0 - 1.75);

  const std::array ragged{1.0, 0.0, 0.5};
  const auto refused = fb::partial_charges(atoms, ragged);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
}

// orbital_populations and bond_orbitals over hand-written weights, on a
// two-atom minimal basis: an s and a p shell on atom 0, an s on atom 1.
TEST(Bond, CentersAndCharacterFromTheWeights) {
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const auto shell = [&](std::int32_t l) {
    return fb::Shell(l, std::vector<double>{1.0}, std::vector<double>{1.0}, origin);
  };
  const fb::BasisSet minimal({shell(0), shell(1), shell(0)}, {0, 2, 3});
  ASSERT_EQ(minimal.nao_sphe(), 5U);
  // Functions s0 p0x p0y p0z s1. A bond, 60% on atom 0 (a third s) and 40%
  // on atom 1; then a lone pair, 97% on atom 0's s.
  const std::array weights{0.2, 0.1, 0.2, 0.1, 0.4, 0.97, 0.0, 0.0, 0.0, 0.03};

  const auto populations = fb::orbital_populations(minimal, 2, weights);
  ASSERT_TRUE(populations.has_value()) << populations.error().message;
  ASSERT_EQ(populations->size(), 4U);
  EXPECT_DOUBLE_EQ((*populations)[0], 0.6);
  EXPECT_DOUBLE_EQ((*populations)[1], 0.4);
  EXPECT_DOUBLE_EQ((*populations)[2], 0.97);
  EXPECT_DOUBLE_EQ((*populations)[3], 0.03);

  const auto bonds = fb::bond_orbitals(minimal, 2, weights);
  ASSERT_TRUE(bonds.has_value()) << bonds.error().message;
  ASSERT_EQ(bonds->size(), 2U);
  const auto &bond = (*bonds)[0].centers;
  ASSERT_EQ(bond.size(), 2U);
  EXPECT_EQ(bond[0].atom, 0U);
  EXPECT_DOUBLE_EQ(bond[0].population, 0.6);
  EXPECT_NEAR(bond[0].character[0], 1.0 / 3.0, 1e-15);
  EXPECT_NEAR(bond[0].character[1], 2.0 / 3.0, 1e-15);
  EXPECT_EQ(bond[0].character[2], 0.0);
  EXPECT_EQ(bond[1].atom, 1U);
  EXPECT_DOUBLE_EQ(bond[1].character[0], 1.0);
  const auto &lone = (*bonds)[1].centers;
  ASSERT_EQ(lone.size(), 1U) << "3% is under the default threshold";
  EXPECT_EQ(lone[0].atom, 0U);
  EXPECT_DOUBLE_EQ(lone[0].character[0], 1.0);

  // A lower threshold takes the 3% too; the largest atom is a center even
  // above every population.
  EXPECT_EQ(fb::bond_orbitals(minimal, 2, weights, 0.01)->at(1).centers.size(), 2U);
  EXPECT_EQ(fb::bond_orbitals(minimal, 2, weights, 1.0)->at(0).centers.size(), 1U);

  EXPECT_FALSE(fb::bond_orbitals(minimal, 2, weights, 0.0).has_value());
  EXPECT_FALSE(fb::bond_orbitals(minimal, 3, weights).has_value());
  EXPECT_FALSE(fb::orbital_populations(minimal, 3, weights).has_value());
}

// bond_orbitals' sigma / pi kind over hand-written weights, on two atoms
// each with an s and a p shell: pi when every center is at least
// pi_threshold p, one-center orbitals included.
TEST(Bond, PiWhenEveryCenterIsAlmostAllP) {
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const auto shell = [&](std::int32_t l) {
    return fb::Shell(l, std::vector<double>{1.0}, std::vector<double>{1.0}, origin);
  };
  const fb::BasisSet minimal({shell(0), shell(1), shell(0), shell(1)}, {0, 2, 4});
  ASSERT_EQ(minimal.nao_sphe(), 8U);
  // Functions s0 p0x p0y p0z s1 p1x p1y p1z. A pi bond, all p on both
  // atoms; a bond all p on atom 0 but 20% s on atom 1; a one-center pure-p
  // lone pair, with an s tail under the threshold; a lone pair half s.
  const std::array weights{0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5, //
                           0.0, 0.0,  0.0, 0.5, 0.1,  0.0, 0.0, 0.4, //
                           0.0, 0.97, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, //
                           0.5, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.0};
  const auto bonds = fb::bond_orbitals(minimal, 4, weights);
  ASSERT_TRUE(bonds.has_value()) << bonds.error().message;
  ASSERT_EQ(bonds->size(), 4U);
  EXPECT_EQ((*bonds)[0].kind, fb::BondKind::Pi);
  EXPECT_EQ((*bonds)[1].kind, fb::BondKind::Sigma) << "atom 1 is s-bearing";
  ASSERT_EQ((*bonds)[2].centers.size(), 1U);
  EXPECT_EQ((*bonds)[2].kind, fb::BondKind::Pi) << "a pure-p lone pair is pi";
  EXPECT_EQ((*bonds)[3].kind, fb::BondKind::Sigma);

  // The rule at its edge, on the second orbital's s-bearing center: pi just
  // below its p character, sigma just above.
  const double p_share = (*bonds)[1].centers[1].character[1];
  EXPECT_NEAR(p_share, 0.8, 1e-15);
  EXPECT_EQ(fb::bond_orbitals(minimal, 4, weights, 0.1, p_share - 1e-12)->at(1).kind,
            fb::BondKind::Pi);
  EXPECT_EQ(fb::bond_orbitals(minimal, 4, weights, 0.1, p_share + 1e-12)->at(1).kind,
            fb::BondKind::Sigma);
  // A pi_threshold of one keeps only the pure-p orbitals pi.
  const auto pure = fb::bond_orbitals(minimal, 4, weights, 0.1, 1.0);
  ASSERT_TRUE(pure.has_value()) << pure.error().message;
  EXPECT_EQ(pure->at(0).kind, fb::BondKind::Pi);
  EXPECT_EQ(pure->at(2).kind, fb::BondKind::Pi);

  for (const double refused : {0.0, -0.5, 1.0 + 1e-12, std::numeric_limits<double>::quiet_NaN()}) {
    const auto out = fb::bond_orbitals(minimal, 4, weights, 0.1, refused);
    ASSERT_FALSE(out.has_value()) << refused;
    EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
