// The Molecule step over the Xyz step's geometry, and the Xyz step's load of
// an [optimize]'s final.xyz. The steps' contracts are
// test/pipeline/steps_host.cpp's, and the Xyz step's run is
// test/pipeline/xyz_host.cpp's; a Molecule owns its nuclei on the device, and
// the load reads one, so these need a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.molecule;
import free_breakfast.core.nontransferrable;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWater = "3\n"
                                    "water, Angstrom\n"
                                    "O   0.000000   0.000000   0.117300\n"
                                    "H   0.000000   0.757200  -0.469200\n"
                                    "H   0.000000  -0.757200  -0.469200\n";

void expect_same_atoms(std::span<const fb::Atom> got, std::span<const fb::Atom> want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t a = 0; a < want.size(); ++a) {
    EXPECT_EQ(got[a].element, want[a].element) << a;
    EXPECT_EQ(got[a].position.x, want[a].position.x) << a;
    EXPECT_EQ(got[a].position.y, want[a].position.y) << a;
    EXPECT_EQ(got[a].position.z, want[a].position.z) << a;
  }
}

TEST(PipelineMolecule, HoldsTheGeometryAtItsChargeAndMultiplicity) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  auto atoms = fb::parse_xyz(kWater);
  ASSERT_TRUE(atoms.has_value()) << atoms.error().message;
  const fb::GeometryOutput geometry{*atoms};

  const fb::MoleculeStep step{.charge = 1, .multiplicity = 2};
  auto molecule = step.run(fb::View<fb::GeometryOutput>{geometry}, res);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  EXPECT_EQ(molecule->molecule.charge(), 1);
  EXPECT_EQ(molecule->molecule.multiplicity(), 2);
  EXPECT_EQ(molecule->molecule.electron_count(), 9);
  expect_same_atoms(molecule->molecule.atoms(), *atoms);
}

// The final.xyz an [optimize] wrote, loaded in place of the Xyz step's run:
// the file's atoms when they are the optimized molecule's, element by
// element; InvalidConfig when they are other atoms; IO when it is missing.
TEST(PipelineMolecule, XyzLoadTakesOnlyTheOptimizedAtoms) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::test::ScratchDir dir("pipeline_molecule");
  auto atoms = fb::parse_xyz(kWater);
  ASSERT_TRUE(atoms.has_value()) << atoms.error().message;
  const fb::OptimizeOutput optimized{.molecule = fb::Molecule(res, *atoms, 0, 1), .energy = 0.0};
  const fb::View<fb::OptimizeOutput> in{optimized};

  const auto same = dir / "final.xyz";
  ASSERT_TRUE(fb::write_file(same, kWater).has_value());
  auto loaded = fb::XyzStep::load(same, in);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  expect_same_atoms(loaded->atoms, *atoms);

  const auto other = dir / "other.xyz";
  ASSERT_TRUE(
      fb::write_file(other, "3\nnot water\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nF 0 -0.7572 -0.4692\n")
          .has_value());
  auto refused = fb::XyzStep::load(other, in);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);

  auto missing = fb::XyzStep::load(dir / "absent.xyz", in);
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().kind, fb::ErrorKind::IO);
}

} // namespace
