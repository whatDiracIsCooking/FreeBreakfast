// Host-only cases for hessian::infrared_intensities: the intensity of a
// hand-made mode, and the refusals of mismatched shapes. It is host
// arithmetic over host spans, so nothing touches a device; infrared_gpu.cpp
// has the dipole derivative it is fed, against PySCF's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// I = K |Sum_x L_x dmu/dx|^2 for one hand-made mode over two atoms, and a
// mode's sign not mattering; the refusals of mismatched shapes.
TEST(HessianInfraredIntensities, ProjectsOntoTheModes) {
  // dmu/dx = row x: (x + 1, 0, 2 x), x over 6 coordinates.
  std::vector<double> derivative(18);
  for (std::size_t x = 0; x < 6; ++x) {
    derivative[3 * x] = static_cast<double>(x + 1);
    derivative[(3 * x) + 2] = 2.0 * static_cast<double>(x);
  }
  const std::vector<double> modes{0.5, 0.0, 0.0, 0.0, 0.0, -0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.25};
  const auto got = fb::hessian::infrared_intensities(modes, derivative);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->size(), 2U);
  // dmu/dQ = 0.5 (1, 0, 0) - 0.25 (6, 0, 10) = (-1, 0, -2.5).
  constexpr double kKmPerMol = 974.8801111743352;
  EXPECT_NEAR((*got)[0], kKmPerMol * 7.25, 1e-10);
  EXPECT_EQ((*got)[0], (*got)[1]);

  EXPECT_FALSE(
      fb::hessian::infrared_intensities(modes, std::span<const double>{derivative}.first(17)));
  EXPECT_FALSE(
      fb::hessian::infrared_intensities(std::span<const double>{modes}.first(11), derivative));
  EXPECT_FALSE(fb::hessian::infrared_intensities(modes, {}));
}

} // namespace
