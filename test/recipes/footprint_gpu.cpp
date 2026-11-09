// The footprints (#1044) of the recipes the RHF chain calls (#1048):
// diagonalize, orthogonalize, Diis, and matrix_ao's U and MatrixAOPair. Each
// is held to the pool peak of one call by test/support/pool_footprint.h's two
// bars, sound (the measured peak is no more than the footprint's) and tight
// (within G x A), so an undercount fails at the routine that owns the
// allocation. recipes.blas's wrappers allocate nothing, so have none.
// test/gtest/footprint_gpu.cpp has the integral routines'.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes;

#include "pool_footprint.h"

namespace {

namespace fb = free_breakfast;
namespace fbt = free_breakfast::test;

// An order past syevd's smallest workspaces, and an SCF's for a small basis.
constexpr std::int32_t kN = 50;

// @p call held to @p fp on @p res, a DeviceResources whose workspaces nothing
// has grown. It runs once first, which grows each workspace to the most the
// call asks of it: the footprint's. Then again, measured, its workspaces
// already grown, as the pipeline grows them before the first step: the pool's
// peak is then the call's persistent and scratch, with @p live allocations
// live at it. Nothing here keeps a CountedPinned or takes the pinned
// workspace.
template <typename Call>
void expect_routine_footprint(fb::DeviceResources &res, const fb::MemoryFootprint &fp,
                              const std::size_t live, Call &&call) {
  call();
  EXPECT_EQ(res.device_workspace(0).num_elements(), fp.device.workspace);
  EXPECT_EQ(res.pinned_bytes(), fp.pinned.peak());
  const fbt::PoolPeak peak = fbt::measure_pool_peak(*res.device(), call);
  fbt::expect_footprint(peak, {.persistent = fp.device.persistent, .scratch = fp.device.scratch},
                        live);
}

// The identity of order kN, on @p res's device: symmetric and well
// conditioned, which is all an allocation needs.
fb::Buffers::device<double> identity(fb::DeviceResources &res) {
  std::vector<double> host(static_cast<std::size_t>(kN) * kN, 0.0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(kN); ++i) {
    host[i * (kN + 1)] = 1.0;
  }
  fb::Buffers::device<double> out(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync());
  return out;
}

TEST(RecipesFootprint, Diagonalize) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto fp = fb::diagonalize_footprint<double>(res, kN);
  ASSERT_TRUE(fp);
  ASSERT_GT(fp->device.workspace, 0U);
  const fb::Buffers::device<double> s = identity(res);
  fb::Buffers::device<double> a(s.num_elements(), res.device());
  fb::Buffers::device<double> w(kN, res.device());
  expect_routine_footprint(res, *fp, 0, [&] {
    EXPECT_EQ(wwr::wwrMemcpyAsync(a.data(), s.data(), sizeof(double) * s.num_elements(),
                                  wwr::wwrMemcpyDeviceToDevice, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(fb::diagonalize<double>(res, kN, a.data(), w.data()));
  });
}

TEST(RecipesFootprint, Orthogonalize) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto fp = fb::orthogonalize_footprint<double>(res, kN);
  ASSERT_TRUE(fp);
  const fb::Buffers::device<double> s = identity(res);
  fb::Buffers::device<double> x(s.num_elements(), res.device());
  // U and the eigenvalues.
  expect_routine_footprint(
      res, *fp, 2, [&] { EXPECT_TRUE(fb::orthogonalize<double>(res, kN, s.data(), x.data())); });
}

// The history allocated, then pushed past its capacity: the pushes allocate
// nothing.
TEST(RecipesFootprint, Diis) {
  FBF_REQUIRE_GPU();
  constexpr std::size_t kLen = static_cast<std::size_t>(kN) * kN;
  constexpr std::size_t kCap = 3;
  const fb::MemoryFootprint fp = fb::diis_footprint(kLen, kCap);
  fb::DeviceResources res{0};
  const fb::Buffers::device<double> f = identity(res);
  fb::Buffers::device<double> e(kLen, res.device());
  fb::Buffers::device<double> out(kLen, res.device());
  // The two histories, the Gram, the coefficients and the flag.
  expect_routine_footprint(res, fp, 5, [&] {
    fb::Diis diis(res, kLen, kCap);
    for (std::size_t i = 0; i < kCap + 1; ++i) {
      EXPECT_TRUE(diis.push_and_extrapolate(res, f.data(), e.data(), out.data()));
    }
  });
}

// An s, p, d basis, so U is not the identity, and an s, p one, where it is.
TEST(RecipesFootprint, MatrixAo) {
  FBF_REQUIRE_GPU();
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const std::array bases{
      fb::BasisSet({fb::Shell(0, {{3.42, 0.15}}, origin), fb::Shell(1, {{0.62, 1.0}}, origin),
                    fb::Shell(2, {{0.8, 1.0}}, origin)}),
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, origin), fb::Shell(1, {{1.0, 1.0}}, origin)})};
  for (const fb::BasisSet &basis : bases) {
    const bool identity_u = basis.nao_cart() == basis.nao_sphe();
    SCOPED_TRACE(identity_u);
    fb::DeviceResources res{0};
    expect_routine_footprint(
        res, fb::upload_spherical_transform_matrix_footprint(basis), identity_u ? 0 : 1,
        [&] { EXPECT_TRUE(fb::upload_spherical_transform_matrix(res, basis)); });
    const auto u = fb::upload_spherical_transform_matrix(res, basis);
    ASSERT_TRUE(u);
    // Both matrices, and the conversion's scratch when U is not the identity.
    expect_routine_footprint(res, fb::matrix_ao_pair_footprint(basis), identity_u ? 2 : 3,
                             [&] { const fb::MatrixAOPair pair(res, basis, *u); });
  }
}

} // namespace
