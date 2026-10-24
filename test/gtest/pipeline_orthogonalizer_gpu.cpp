// The Orthogonalizer step against a direct orthogonalize, and X^T S X = I on
// the overlap the steps before it build. The step's contract is checked by
// static_assert; the rest runs on the device, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.orthogonalizer;

namespace {

namespace fb = free_breakfast;

using OrthogonalizerInput = fb::View<fb::BasisSetOutput, fb::OverlapOutput>;

static_assert(fb::PipelineStep<fb::OrthogonalizerStep>);
static_assert(fb::kRunShape<fb::OrthogonalizerStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::OrthogonalizerStep>, OrthogonalizerInput>);
static_assert(std::same_as<fb::StepOutput<fb::OrthogonalizerStep>, fb::OrthogonalizerOutput>);

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// s + p on one centre, d + s on another: nao_sphe = 10, and S well-conditioned.
fb::BasisSetOutput two_centre_basis() {
  const fb::Vec<double, 3> a{0.0, 0.0, 0.0};
  const fb::Vec<double, 3> b{0.0, 0.3, 1.4};
  return {fb::BasisSet(
      {fb::place(fb::ShellStamp(0, {{3.42, 0.15}, {0.62, 0.53}}).primitive_normalized(), a),
       fb::place(fb::ShellStamp(1, {{0.62, 1.0}}).primitive_normalized(), a),
       fb::place(fb::ShellStamp(2, {{0.8, 1.0}}).primitive_normalized(), b),
       fb::place(fb::ShellStamp(0, {{0.5, 1.0}}).primitive_normalized(), b)})};
}

// S from the steps before the OrthogonalizerOutput.
fb::OverlapOutput overlap_of(fb::DeviceResources &res, const fb::BasisSetOutput &basis) {
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  EXPECT_TRUE(transform.has_value());
  EXPECT_TRUE(pairs.has_value());
  auto s = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          basis, *transform, *pairs},
      res);
  EXPECT_TRUE(s.has_value());
  return std::move(*s);
}

TEST(PipelineOrthogonalizer, MatchesOrthogonalize) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSetOutput basis = two_centre_basis();
  const fb::OverlapOutput s = overlap_of(res, basis);

  auto x = fb::OrthogonalizerStep{}.run(OrthogonalizerInput{basis, s}, res);
  ASSERT_TRUE(x.has_value()) << x.error().message;
  ASSERT_EQ(x->x.n(), 10U);

  fb::SphericalMatrixAO direct(res, basis.basis);
  ASSERT_TRUE(fb::orthogonalize(res, static_cast<std::int32_t>(s.s.n()), s.s.data(), direct.data())
                  .has_value());
  EXPECT_EQ(download(res, x->x), download(res, direct));
}

TEST(PipelineOrthogonalizer, OrthonormalizesTheOverlap) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSetOutput basis = two_centre_basis();
  const fb::OverlapOutput s = overlap_of(res, basis);

  auto x = fb::OrthogonalizerStep{}.run(OrthogonalizerInput{basis, s}, res);
  ASSERT_TRUE(x.has_value()) << x.error().message;
  const std::size_t n = s.s.n();
  const std::vector<double> sh = download(res, s.s);
  const std::vector<double> xh = download(res, x->x);
  // (X^T S X)_ij = sum_kl X_ki S_kl X_lj, column-major.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          sum += xh[k + (i * n)] * sh[k + (l * n)] * xh[l + (j * n)];
        }
      }
      EXPECT_NEAR(sum, i == j ? 1.0 : 0.0, 1e-10) << i << ' ' << j;
    }
  }
}

TEST(PipelineOrthogonalizer, FloorAboveEveryEigenvalueDropsEveryMode) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSetOutput basis = two_centre_basis();
  const fb::OverlapOutput s = overlap_of(res, basis);

  // S's eigenvalues are positive and sum to its trace, about n = 10, so none
  // reaches 100.
  const fb::OrthogonalizerStep step{.eigenvalue_floor = 100.0};
  auto x = step.run(OrthogonalizerInput{basis, s}, res);
  ASSERT_TRUE(x.has_value()) << x.error().message;
  for (const double v : download(res, x->x)) {
    EXPECT_EQ(v, 0.0);
  }
}

} // namespace
