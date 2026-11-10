// The SphericalTransform step against a direct upload_spherical_transform_matrix,
// and its .npy file.
// The step's contract is test/pipeline/steps_host.cpp's; running it uploads U, so the
// rest needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.spherical_transform;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

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
  fb::DeviceResources res{0, fb::test::kWorkspaces};
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
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::BasisSetOutput basis{
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(1, {{1.0, 1.0}}, kOrigin)})};

  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  ASSERT_TRUE(transform.has_value()) << transform.error().message;
  EXPECT_EQ(transform->u, nullptr);
}

TEST(PipelineSphericalTransform, NpyHoldsTheStepsU) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::test::ScratchDir dir("pipeline_spherical_transform");
  // s + p + d + f: nao_cart = 20, nao_sphe = 16.
  const fb::BasisSetOutput basis{
      fb::BasisSet({fb::Shell(0, {{3.42, 0.15}}, kOrigin), fb::Shell(1, {{0.62, 1.0}}, kOrigin),
                    fb::Shell(2, {{0.8, 1.0}}, kOrigin), fb::Shell(3, {{0.5, 1.0}}, kOrigin)})};

  const fb::SphericalTransformStep step{.npy = dir / "spherical_transform.npy"};
  auto transform = step.run(fb::View<fb::BasisSetOutput>{basis}, res);
  ASSERT_TRUE(transform.has_value()) << transform.error().message;
  ASSERT_NE(transform->u, nullptr);
  auto npy = fb::read_npy(step.npy);
  ASSERT_TRUE(npy.has_value()) << npy.error().message;
  EXPECT_EQ(npy->shape, (std::vector<std::size_t>{20, 16}));
  EXPECT_EQ(npy->data, download(res, *transform->u));
}

TEST(PipelineSphericalTransform, NpyIsTheIdentityBelowD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  const fb::test::ScratchDir dir("pipeline_spherical_transform_identity");
  const fb::BasisSetOutput basis{
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, kOrigin), fb::Shell(1, {{1.0, 1.0}}, kOrigin)})};

  // The product is nullptr, but the file is still written, as U = 1.
  const fb::SphericalTransformStep step{.npy = dir / "spherical_transform.npy"};
  auto transform = step.run(fb::View<fb::BasisSetOutput>{basis}, res);
  ASSERT_TRUE(transform.has_value()) << transform.error().message;
  EXPECT_EQ(transform->u, nullptr);
  auto npy = fb::read_npy(step.npy);
  ASSERT_TRUE(npy.has_value()) << npy.error().message;
  ASSERT_EQ(npy->shape, (std::vector<std::size_t>{4, 4}));
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(npy->data[i + (j * 4)], i == j ? 1.0 : 0.0) << i << "," << j;
    }
  }
}

} // namespace
