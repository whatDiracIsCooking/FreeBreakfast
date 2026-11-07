// Host-only cases for nevpt2's density-fitted blocks: the axis each
// external block is batched along, which the demos fix. A table lookup, so
// nothing touches a device; nevpt2_density_fitting_gpu.cpp has the blocks.

#include <gtest/gtest.h>

import std;
import free_breakfast.nevpt2;

namespace {

namespace nevpt2 = free_breakfast::nevpt2;
using nevpt2::ExtBlock;

TEST(Nevpt2DensityFitting, BatchAxesAreTheDemos) {
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::Si), 2);
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::Sij), 2);
  EXPECT_EQ(nevpt2::batch_axis(ExtBlock::SrsiT), 1);
  for (const ExtBlock block : {ExtBlock::Sr, ExtBlock::Sijrs, ExtBlock::Sijr, ExtBlock::Srsi,
                               ExtBlock::Srs, ExtBlock::Sir1, ExtBlock::Sir2}) {
    EXPECT_EQ(nevpt2::batch_axis(block), 0) << static_cast<std::int32_t>(block);
  }
}

} // namespace
