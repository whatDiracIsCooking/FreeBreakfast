// The PrimitivePairClasses step against a direct primitive_pair_classes. The
// step's contract is checked by static_assert; running it builds the records
// on the device, so the rest needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.primitive_pair_classes;

#include "mmd_records.h"

namespace {

namespace fb = free_breakfast;
using Class = fb::PrimitivePairClass;

static_assert(fb::PipelineStep<fb::PrimitivePairClassesStep>);
static_assert(fb::kRunShape<fb::PrimitivePairClassesStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::PrimitivePairClassesStep>, fb::View<fb::BasisSet>>);
static_assert(std::same_as<fb::StepOutput<fb::PrimitivePairClassesStep>, fb::PrimitivePairClasses>);

// s and p on two centres, d on a third: classes (0,0) .. (2,2), with a p bra
// before an s ket, so the la <= lb swap is in play.
fb::BasisSet spd_basis() {
  const fb::Vec<double, 3> a{0.0, 0.0, 0.0};
  const fb::Vec<double, 3> b{0.0, 0.0, 1.4};
  const fb::Vec<double, 3> c{1.1, 0.3, -0.5};
  return fb::BasisSet({fb::Shell(1, {{0.62, 1.0}, {2.1, 0.4}}, a),
                       fb::Shell(0, {{3.42, 0.15}, {0.62, 0.53}}, b),
                       fb::Shell(2, {{0.8, 1.0}}, c)});
}

TEST(PipelinePrimitivePairClasses, MatchesPrimitivePairClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = spd_basis();

  auto step = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSet>{basis}, res);
  auto direct = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(step.has_value()) << step.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;

  ASSERT_EQ(step->classes.size(), direct->size());
  EXPECT_EQ(direct->size(), 6U);
  for (std::size_t c = 0; c < direct->size(); ++c) {
    const Class &got = step->classes[c];
    const Class &want = (*direct)[c];
    EXPECT_EQ(got.la, want.la) << c;
    EXPECT_EQ(got.lb, want.lb) << c;
    ASSERT_EQ(got.n_rec, want.n_rec) << c;
    // Every field, bit for bit.
    EXPECT_TRUE(fb::test::copy_back(got) == fb::test::copy_back(want)) << c;
  }
}

TEST(PipelinePrimitivePairClasses, EmptyBasisHasNoClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis{};

  auto step = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSet>{basis}, res);
  ASSERT_TRUE(step.has_value()) << step.error().message;
  EXPECT_TRUE(step->classes.empty());
}

} // namespace
