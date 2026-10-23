// The Overlap step against a direct overlap + cartesian_to_spherical, and its
// diagonal on a basis read through the BasisSet step. The step's contract is
// checked by static_assert; the rest builds S on the device, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.base;
import free_breakfast.integrals.one_electron.overlap;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.overlap;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

using OverlapInput = fb::View<fb::BasisSet, fb::SphericalTransform, fb::PrimitivePairClasses>;

static_assert(fb::PipelineStep<fb::OverlapStep>);
static_assert(fb::kRunShape<fb::OverlapStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::OverlapStep>, OverlapInput>);
static_assert(std::same_as<fb::StepOutput<fb::OverlapStep>, fb::Overlap>);

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// S from the Overlap step, its inputs built by the steps before it.
fb::Result<fb::Overlap> step_overlap(fb::DeviceResources &res, const fb::BasisSet &basis) {
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSet>{basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSet>{basis}, res);
  EXPECT_TRUE(transform.has_value());
  EXPECT_TRUE(pairs.has_value());
  if (!transform || !pairs) {
    return fb::err_gpu("an input step failed");
  }
  return fb::OverlapStep{}.run(OverlapInput{basis, *transform, *pairs}, res);
}

TEST(PipelineOverlap, MatchesOverlapThenCartesianToSpherical) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Vec<double, 3> a{0.0, 0.0, 0.0};
  const fb::Vec<double, 3> b{0.0, 0.3, 1.4};
  // s + p on one centre, d + s on another: nao_cart = 11, nao_sphe = 10.
  const fb::BasisSet basis({fb::Shell(0, {{3.42, 0.15}, {0.62, 0.53}}, a),
                            fb::Shell(1, {{0.62, 1.0}}, a), fb::Shell(2, {{0.8, 1.0}}, b),
                            fb::Shell(0, {{0.5, 1.0}}, b)});

  auto s = step_overlap(res, basis);
  ASSERT_TRUE(s.has_value()) << s.error().message;
  EXPECT_EQ(s->s.n(), 10U);

  auto u = fb::upload_spherical_transform_matrix(res, basis);
  auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(u.has_value());
  ASSERT_TRUE(classes.has_value());
  fb::MatrixAOPair direct(res, basis, *u);
  ASSERT_TRUE(fb::overlap(res, *classes, direct.cartesian()).has_value());
  ASSERT_TRUE(direct.cartesian_to_spherical(res).has_value());

  EXPECT_EQ(download(res, s->s), download(res, direct.spherical()));
}

// One primitive per shell, so every normalized function is exactly normalized:
// an s and a p on oxygen, a spherical d, and an s on each hydrogen.
constexpr std::string_view kSinglePrimitiveBasis = R"({
  "elements": {
    "1": {"electron_shells": [
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["0.62"], "coefficients": [["1.0"]]}
    ]},
    "8": {"electron_shells": [
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["5.03"], "coefficients": [["1.0"]]},
      {"function_type": "gto", "angular_momentum": [1],
       "exponents": ["1.17"], "coefficients": [["1.0"]]},
      {"function_type": "gto_spherical", "angular_momentum": [2],
       "exponents": ["1.185"], "coefficients": [["1.0"]]}
    ]}
  }
})";

TEST(PipelineOverlap, NormalizedBasisHasUnitDiagonal) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_overlap");
  const auto path = dir / "basis.json";
  ASSERT_TRUE(fb::write_file(path, kSinglePrimitiveBasis).has_value());
  const fb::Molecule water(res, {Atom{{0.0, 0.0, 0.2217}, Element::O},
                                 Atom{{0.0, 1.4309, -0.8867}, Element::H},
                                 Atom{{0.0, -1.4309, -0.8867}, Element::H}});
  auto basis = fb::BasisSetStep{.basis = path.string()}.run(fb::View<fb::Molecule>{water});
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  auto s = step_overlap(res, *basis);
  ASSERT_TRUE(s.has_value()) << s.error().message;
  const std::uint32_t n = s->s.n();
  ASSERT_EQ(n, 1U + 3U + 5U + 1U + 1U);
  const std::vector<double> host = download(res, s->s);
  for (std::uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(host[i + (i * n)], 1.0, 1e-12) << i;
  }
}

} // namespace
