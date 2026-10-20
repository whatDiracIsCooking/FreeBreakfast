// GPU cases for molecule: electron counting, the nuclear repulsion energy and
// its analytic gradient, and the nuclei a Molecule owns on the device -- Z and
// the positions in atom order, on the constructing call's device. A Molecule
// owns its nuclei on the device, so building one needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

// Water near its equilibrium geometry, in bohr.
fb::Molecule water(fb::DeviceResources &res, std::int32_t charge = 0,
                   std::int32_t multiplicity = 1) {
  return fb::Molecule(res,
                      {Atom{{0.0, 0.0, 0.2217}, Element::O},
                       Atom{{0.0, 1.4309, -0.8867}, Element::H},
                       Atom{{0.0, -1.4309, -0.8867}, Element::H}},
                      charge, multiplicity);
}

// n doubles of the device array @p data, copied back to the host on the
// nuclei's stream.
std::vector<double> download(const fb::Molecule &molecule, const double *data, std::size_t n) {
  std::vector<double> out(n);
  const wwr::wwrStream_t stream = molecule.device_stream()->stream();
  EXPECT_EQ(
      wwr::wwrMemcpyAsync(out.data(), data, sizeof(double) * n, wwr::wwrMemcpyDeviceToHost, stream),
      wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

TEST(Molecule, ElectronCountSubtractsCharge) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  EXPECT_EQ(water(res).total_nuclear_charge(), 10);
  EXPECT_EQ(water(res).electron_count(), 10);
  EXPECT_EQ(water(res, +1, 2).electron_count(), 9); // cation: one fewer electron
  EXPECT_EQ(water(res, -1, 2).electron_count(), 11);
  EXPECT_EQ(water(res, +1, 2).multiplicity(), 2);
}

TEST(Molecule, AtomsKeepSuppliedOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule mol = water(res);
  ASSERT_EQ(mol.natoms(), 3U);
  EXPECT_EQ(mol.atoms()[0].element, Element::O);
  EXPECT_EQ(mol.atoms()[1].position.y, 1.4309);
  EXPECT_EQ(mol.atoms()[2].position.y, -1.4309);
}

TEST(Molecule, SingleAtomHasNoRepulsion) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule he(res, {Atom{{1.0, 2.0, 3.0}, Element::He}});
  EXPECT_EQ(fb::nuclear_repulsion_energy(he), 0.0);
  EXPECT_EQ(fb::nuclear_repulsion_gradient(he), (std::vector<double>{0.0, 0.0, 0.0}));
}

TEST(Molecule, HydrogenMoleculeRepulsion) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule h2(res,
                        {Atom{{0.0, 0.0, 0.0}, Element::H}, Atom{{0.0, 0.0, 1.4}, Element::H}});
  EXPECT_DOUBLE_EQ(fb::nuclear_repulsion_energy(h2), 1.0 / 1.4);
}

TEST(Molecule, WaterRepulsionSumsEveryPairOnce) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // O-H twice (Z=8*1) plus H-H once (Z=1*1), each over its own distance.
  const double r_oh = std::hypot(1.4309, 0.2217 + 0.8867);
  const double r_hh = 2.0 * 1.4309;
  const double expected = (2.0 * 8.0 / r_oh) + (1.0 / r_hh);
  EXPECT_NEAR(fb::nuclear_repulsion_energy(water(res)), expected, 1e-14);
  // Charge and multiplicity are about electrons; the nuclei do not care.
  EXPECT_EQ(fb::nuclear_repulsion_energy(water(res, +1, 2)),
            fb::nuclear_repulsion_energy(water(res)));
}

TEST(Molecule, GradientMatchesCentralFiniteDifference) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // A non-symmetric geometry so no component vanishes by symmetry.
  const std::vector<Atom> atoms{
      Atom{{0.1, -0.2, 0.3}, Element::N}, Atom{{1.9, 0.4, -0.1}, Element::C},
      Atom{{-0.7, 1.6, 0.8}, Element::H}, Atom{{0.5, -1.3, 2.2}, Element::F}};
  const std::vector<double> grad = fb::nuclear_repulsion_gradient(fb::Molecule(res, atoms));
  ASSERT_EQ(grad.size(), atoms.size() * 3);

  const double h = 1e-5;
  for (std::size_t a = 0; a < atoms.size(); ++a) {
    for (std::int32_t axis = 0; axis < 3; ++axis) {
      auto displaced = [&](double step) {
        std::vector<Atom> moved = atoms;
        double &coord = axis == 0 ? moved[a].position.x
                                  : (axis == 1 ? moved[a].position.y : moved[a].position.z);
        coord += step;
        return fb::nuclear_repulsion_energy(fb::Molecule(res, moved));
      };
      const double numeric = (displaced(+h) - displaced(-h)) / (2.0 * h);
      EXPECT_NEAR(grad[(3 * a) + axis], numeric, 1e-7) << "atom " << a << " axis " << axis;
    }
  }
}

TEST(Molecule, GradientIsTranslationallyInvariant) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const std::vector<double> grad = fb::nuclear_repulsion_gradient(water(res));
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    double sum = 0.0;
    for (std::size_t a = 0; a < 3; ++a) {
      sum += grad[(3 * a) + axis];
    }
    EXPECT_NEAR(sum, 0.0, 1e-14) << "axis " << axis;
  }
}

TEST(Molecule, NucleiOnTheDeviceAreChargesAndPositionsInAtomOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(res, {Atom{fb::Vec<double, 3>{0.0, 0.0, 0.25}, Element::O},
                                    Atom{fb::Vec<double, 3>{1.5, -0.5, 0.0}, Element::H}});
  const fb::NucleiView view = molecule.view();
  ASSERT_EQ(view.n_nuc, 2);
  EXPECT_EQ(download(molecule, view.z, 2), (std::vector<double>{8.0, 1.0}));
  EXPECT_EQ(download(molecule, view.c, 6), (std::vector<double>{0.0, 0.0, 0.25, 1.5, -0.5, 0.0}));
}

TEST(Molecule, NucleiAreOnTheConstructingCallsDevice) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(res, {Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, Element::H}});
  EXPECT_EQ(molecule.device_stream()->dev_idx(), res.dev_idx());
}

TEST(Molecule, MovedToMoleculeKeepsItsNuclei) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Molecule from = water(res);
  const double *z = from.view().z;
  const fb::Molecule to = std::move(from);
  EXPECT_EQ(to.view().z, z);
  EXPECT_EQ(download(to, to.view().z, 3), (std::vector<double>{8.0, 1.0, 1.0}));
}

// The threadsafe style re-executes the binary for the child, so the child
// initialises its own GPU runtime instead of inheriting a forked one.
TEST(MoleculeDeathTest, AbortsWithNoAtoms) {
  FBF_REQUIRE_GPU();
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        fb::DeviceResources res{0};
        const fb::Molecule molecule(res, {});
      },
      "");
}

} // namespace
