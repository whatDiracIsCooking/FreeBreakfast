// Host-only cases for molecule: electron counting, the nuclear repulsion energy
// and its analytic gradient.

#include <gtest/gtest.h>

import std;
import free_breakfast.molecule;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

// Water near its equilibrium geometry, in bohr.
fb::Molecule water(std::int32_t charge = 0, std::int32_t multiplicity = 1) {
  return fb::Molecule({Atom{{0.0, 0.0, 0.2217}, Element::O},
                       Atom{{0.0, 1.4309, -0.8867}, Element::H},
                       Atom{{0.0, -1.4309, -0.8867}, Element::H}},
                      charge, multiplicity);
}

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

TEST(Molecule, DefaultIsEmptyNeutralSinglet) {
  const fb::Molecule mol;
  EXPECT_EQ(mol.natoms(), 0U);
  EXPECT_EQ(mol.charge(), 0);
  EXPECT_EQ(mol.multiplicity(), 1);
  EXPECT_EQ(mol.electron_count(), 0);
  EXPECT_EQ(fb::nuclear_repulsion_energy(mol), 0.0);
  EXPECT_TRUE(fb::nuclear_repulsion_gradient(mol).empty());
}

TEST(Molecule, ElectronCountSubtractsCharge) {
  EXPECT_EQ(water().total_nuclear_charge(), 10);
  EXPECT_EQ(water().electron_count(), 10);
  EXPECT_EQ(water(+1, 2).electron_count(), 9); // cation: one fewer electron
  EXPECT_EQ(water(-1, 2).electron_count(), 11);
  EXPECT_EQ(water(+1, 2).multiplicity(), 2);
}

TEST(Molecule, AtomsKeepSuppliedOrder) {
  const fb::Molecule mol = water();
  ASSERT_EQ(mol.natoms(), 3U);
  EXPECT_EQ(mol.atoms()[0].element, Element::O);
  EXPECT_EQ(mol.atoms()[1].position.y, 1.4309);
  EXPECT_EQ(mol.atoms()[2].position.y, -1.4309);
}

TEST(Molecule, SingleAtomHasNoRepulsion) {
  const fb::Molecule he({Atom{{1.0, 2.0, 3.0}, Element::He}});
  EXPECT_EQ(fb::nuclear_repulsion_energy(he), 0.0);
  EXPECT_EQ(fb::nuclear_repulsion_gradient(he), (std::vector<double>{0.0, 0.0, 0.0}));
}

TEST(Molecule, HydrogenMoleculeRepulsion) {
  const fb::Molecule h2({Atom{{0.0, 0.0, 0.0}, Element::H}, Atom{{0.0, 0.0, 1.4}, Element::H}});
  EXPECT_DOUBLE_EQ(fb::nuclear_repulsion_energy(h2), 1.0 / 1.4);
}

TEST(Molecule, WaterRepulsionSumsEveryPairOnce) {
  // O-H twice (Z=8*1) plus H-H once (Z=1*1), each over its own distance.
  const double r_oh = std::hypot(1.4309, 0.2217 + 0.8867);
  const double r_hh = 2.0 * 1.4309;
  const double expected = (2.0 * 8.0 / r_oh) + (1.0 / r_hh);
  EXPECT_NEAR(fb::nuclear_repulsion_energy(water()), expected, 1e-14);
  // Charge and multiplicity are about electrons; the nuclei do not care.
  EXPECT_EQ(fb::nuclear_repulsion_energy(water(+1, 2)), fb::nuclear_repulsion_energy(water()));
}

TEST(Molecule, GradientMatchesCentralFiniteDifference) {
  // A non-symmetric geometry so no component vanishes by symmetry.
  const std::vector<Atom> atoms{
      Atom{{0.1, -0.2, 0.3}, Element::N}, Atom{{1.9, 0.4, -0.1}, Element::C},
      Atom{{-0.7, 1.6, 0.8}, Element::H}, Atom{{0.5, -1.3, 2.2}, Element::F}};
  const std::vector<double> grad = fb::nuclear_repulsion_gradient(fb::Molecule(atoms));
  ASSERT_EQ(grad.size(), atoms.size() * 3);

  const double h = 1e-5;
  for (std::size_t a = 0; a < atoms.size(); ++a) {
    for (std::int32_t axis = 0; axis < 3; ++axis) {
      auto displaced = [&](double step) {
        std::vector<Atom> moved = atoms;
        double &coord = axis == 0 ? moved[a].position.x
                                  : (axis == 1 ? moved[a].position.y : moved[a].position.z);
        coord += step;
        return fb::nuclear_repulsion_energy(fb::Molecule(moved));
      };
      const double numeric = (displaced(+h) - displaced(-h)) / (2.0 * h);
      EXPECT_NEAR(grad[(3 * a) + axis], numeric, 1e-7) << "atom " << a << " axis " << axis;
    }
  }
}

TEST(Molecule, GradientIsTranslationallyInvariant) {
  const std::vector<double> grad = fb::nuclear_repulsion_gradient(water());
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    double sum = 0.0;
    for (std::size_t a = 0; a < 3; ++a) {
      sum += grad[(3 * a) + axis];
    }
    EXPECT_NEAR(sum, 0.0, 1e-14) << "axis " << axis;
  }
}

} // namespace
