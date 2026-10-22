// The Molecule step against a direct read_xyz call. The step's contract is
// checked by static_assert; running it builds a Molecule, so the rest needs a
// GPU.

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

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::MoleculeStep>);
static_assert(fb::kRunShape<fb::MoleculeStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::MoleculeStep>, fb::View<>>);
static_assert(std::same_as<fb::StepOutput<fb::MoleculeStep>, fb::Molecule>);

constexpr std::string_view kWater = "3\n"
                                    "water, Angstrom\n"
                                    "O   0.000000   0.000000   0.117300\n"
                                    "H   0.000000   0.757200  -0.469200\n"
                                    "H   0.000000  -0.757200  -0.469200\n";

TEST(PipelineMolecule, MatchesReadXyz) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_molecule");
  const auto path = dir / "water.xyz";
  ASSERT_TRUE(fb::write_file(path, kWater).has_value());

  const fb::MoleculeStep step{.xyz = path, .charge = 1, .multiplicity = 2};
  auto molecule = step.run(fb::View<>{}, res);
  auto direct = fb::read_xyz(res, path, 1, 2);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;

  EXPECT_EQ(molecule->charge(), 1);
  EXPECT_EQ(molecule->multiplicity(), 2);
  ASSERT_EQ(molecule->natoms(), direct->natoms());
  for (std::uint32_t a = 0; a < direct->natoms(); ++a) {
    const fb::Atom &got = molecule->atoms()[a];
    const fb::Atom &want = direct->atoms()[a];
    EXPECT_EQ(got.element, want.element) << a;
    EXPECT_EQ(got.position.x, want.position.x) << a;
    EXPECT_EQ(got.position.y, want.position.y) << a;
    EXPECT_EQ(got.position.z, want.position.z) << a;
  }
}

TEST(PipelineMolecule, MissingFileIsIoError) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_molecule");

  const fb::MoleculeStep step{.xyz = dir / "absent.xyz"};
  auto molecule = step.run(fb::View<>{}, res);
  ASSERT_FALSE(molecule.has_value());
  EXPECT_EQ(molecule.error().kind, fb::ErrorKind::IO);
}

} // namespace
