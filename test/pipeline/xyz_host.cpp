// The Xyz step against a direct read_xyz call: a host step, so its run needs
// no device. Its contract is steps_host.cpp's; its load reads OptimizeOutput,
// which holds a device Molecule, so that runs in
// test/gtest/pipeline_molecule_gpu.cpp.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.base;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.xyz;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWater = "3\n"
                                    "water, Angstrom\n"
                                    "O   0.000000   0.000000   0.117300\n"
                                    "H   0.000000   0.757200  -0.469200\n"
                                    "H   0.000000  -0.757200  -0.469200\n";

TEST(PipelineXyz, MatchesReadXyz) {
  const fb::test::ScratchDir dir("pipeline_xyz");
  const auto path = dir / "water.xyz";
  ASSERT_TRUE(fb::write_file(path, kWater).has_value());

  const fb::XyzStep step{.xyz = path};
  auto geometry = step.run(fb::View<>{});
  auto direct = fb::read_xyz(path);
  ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;

  ASSERT_EQ(geometry->atoms.size(), direct->size());
  for (std::size_t a = 0; a < direct->size(); ++a) {
    const fb::Atom &got = geometry->atoms[a];
    const fb::Atom &want = (*direct)[a];
    EXPECT_EQ(got.element, want.element) << a;
    EXPECT_EQ(got.position.x, want.position.x) << a;
    EXPECT_EQ(got.position.y, want.position.y) << a;
    EXPECT_EQ(got.position.z, want.position.z) << a;
  }
}

TEST(PipelineXyz, MissingFileIsIoError) {
  const fb::test::ScratchDir dir("pipeline_xyz");
  const fb::XyzStep step{.xyz = dir / "absent.xyz"};
  auto geometry = step.run(fb::View<>{});
  ASSERT_FALSE(geometry.has_value());
  EXPECT_EQ(geometry.error().kind, fb::ErrorKind::IO);
}

} // namespace
