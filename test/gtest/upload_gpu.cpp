// GPU cases for integrals.one_electron.upload's checks: check_nuclei accepts a
// molecule whose nuclei are on the call's device.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

TEST(CheckNuclei, AcceptsNucleiOnTheCallsDevice) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule(res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::H}});
  EXPECT_TRUE(fb::check_nuclei(res, molecule, "test"));
}

} // namespace
