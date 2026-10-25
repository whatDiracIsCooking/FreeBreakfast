// The SphericalTransform step against a direct upload_spherical_transform_matrix.
// The step's contract is checked by static_assert; running it uploads U, so the
// rest needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.spherical_transform;

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::SphericalTransformStep>);
static_assert(fb::kRunShape<fb::SphericalTransformStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::SphericalTransformStep>, fb::View<fb::BasisSetOutput>>);
static_assert(
    std::same_as<fb::StepOutput<fb::SphericalTransformStep>, fb::SphericalTransformOutput>);

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &u) {
  std::vector<double> host(u.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), u.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

TEST(PipelineSphericalTransform, MatchesUploadSphericalTransformMatrix) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // s + p + d: nao_cart = 10, nao_sphe = 9.
  const fb::BasisSetOutput basis{
      fb::BasisSet({fb::Shell(0, {{3.42, 0.15}}, kOrigin), fb::Shell(1, {{0.62, 1.0}}, kOrigin),
                    fb::Shell(2, {{0.8, 1.0}}, kOrigin)})};

  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  auto direct = fb::upload_spherical_transform_matrix(res, basis.basis);
  ASSERT_TRUE(transform.has_value()) << transform.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  ASSERT_NE(transform->u, nullptr);
  ASSERT_NE(*direct, nullptr);

  EXPECT_EQ(transform->u->num_elements(), std::size_t{10} * 9);
  EXPECT_EQ(download(res, *transform->u), download(res, **direct));
}

TEST(PipelineSphericalTransform, NoTransformBelowD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSetOutput basis{
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(1, {{1.0, 1.0}}, kOrigin)})};

  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  ASSERT_TRUE(transform.has_value()) << transform.error().message;
  EXPECT_EQ(transform->u, nullptr);
}

} // namespace
