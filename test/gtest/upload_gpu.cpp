// GPU cases for integrals.one_electron.upload's nuclei: Z and the positions in
// atom order on the pinned side, the device side matching it, the nuclei on
// the call's device, and the refusal of a molecule with no atoms.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// A field's device side, copied back to the host on the field's stream.
std::vector<double> download(const fb::MirroredBuffer<double> &field) {
  std::vector<double> out(field.size());
  const wwr::wwrStream_t stream = field.handle()->stream();
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), field.device().data(), sizeof(double) * out.size(),
                                wwr::wwrMemcpyDeviceToHost, stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

// A field's pinned side.
std::vector<double> pinned(const fb::MirroredBuffer<double> &field) {
  const double *p = field.pinned().data();
  return {p, p + field.size()};
}

TEST(UploadNuclei, WritesChargesAndPositionsInAtomOrder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule({fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.25}, fb::Element::O},
                               fb::Atom{fb::Vec<double, 3>{1.5, -0.5, 0.0}, fb::Element::H}});
  const auto nuclei = fb::upload_nuclei(res, molecule);
  ASSERT_TRUE(nuclei);
  EXPECT_EQ(nuclei->n_nuc, 2);
  EXPECT_EQ(pinned(nuclei->z), (std::vector<double>{8.0, 1.0}));
  EXPECT_EQ(pinned(nuclei->c), (std::vector<double>{0.0, 0.0, 0.25, 1.5, -0.5, 0.0}));
}

TEST(UploadNuclei, DeviceSideMatchesThePinnedSide) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule({fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.25}, fb::Element::O},
                               fb::Atom{fb::Vec<double, 3>{1.5, -0.5, 0.0}, fb::Element::H}});
  const auto nuclei = fb::upload_nuclei(res, molecule);
  ASSERT_TRUE(nuclei);
  EXPECT_EQ(download(nuclei->z), pinned(nuclei->z));
  EXPECT_EQ(download(nuclei->c), pinned(nuclei->c));
}

TEST(UploadNuclei, NucleiAreOnTheCallsDevice) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto nuclei = fb::upload_nuclei(
      res, fb::Molecule({fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::H}}));
  ASSERT_TRUE(nuclei);
  EXPECT_EQ(nuclei->device_stream()->dev_idx(), res.dev_idx());
  EXPECT_TRUE(fb::check_nuclei(res, *nuclei, "test"));
}

TEST(UploadNuclei, RefusesAMoleculeWithNoAtoms) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto nuclei = fb::upload_nuclei(res, fb::Molecule{});
  ASSERT_FALSE(nuclei);
  EXPECT_EQ(nuclei.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
