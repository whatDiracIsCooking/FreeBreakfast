// The footprints (#1044) of the recipes the RHF chain calls (#1048):
// diagonalize, orthogonalize, Diis, and matrix_ao's U and MatrixAOPair; and
// those the CI calls (#1049): orthonormalize and davidson_solve; and those
// CASSCF's orbital step calls (#1151): skew_eig and skew_expm. Each
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
  const auto lwork = fb::syevd_lwork<double>(res, kN);
  ASSERT_TRUE(lwork);
  const fb::MemoryFootprint fp = fb::diagonalize_footprint<double>(*lwork);
  ASSERT_GT(fp.device.workspace, 0U);
  const fb::Buffers::device<double> s = identity(res);
  fb::Buffers::device<double> a(s.num_elements(), res.device());
  fb::Buffers::device<double> w(kN, res.device());
  expect_routine_footprint(res, fp, 0, [&] {
    EXPECT_EQ(wwr::wwrMemcpyAsync(a.data(), s.data(), sizeof(double) * s.num_elements(),
                                  wwr::wwrMemcpyDeviceToDevice, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(fb::diagonalize<double>(res, kN, a.data(), w.data()));
  });
}

TEST(RecipesFootprint, Orthogonalize) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto lwork = fb::syevd_lwork<double>(res, kN);
  ASSERT_TRUE(lwork);
  const fb::MemoryFootprint fp = fb::orthogonalize_footprint<double>(kN, *lwork);
  const fb::Buffers::device<double> s = identity(res);
  fb::Buffers::device<double> x(s.num_elements(), res.device());
  // U and the eigenvalues.
  expect_routine_footprint(
      res, fp, 2, [&] { EXPECT_TRUE(fb::orthogonalize<double>(res, kN, s.data(), x.data())); });
}

// syevd's largest work over orders 1..k: what the orthonormalize and
// Davidson footprints take for the orders they diagonalize at.
std::size_t lwork_through(fb::DeviceResources &res, std::int32_t k) {
  std::size_t most = 0;
  for (std::int32_t order = 1; order <= k; ++order) {
    const auto lwork = fb::syevd_lwork<double>(res, order);
    EXPECT_TRUE(lwork);
    most = std::max(most, lwork.value_or(0));
  }
  return most;
}

// A random kN x kK block, independent columns: both canonical passes keep
// all kK, and the product is kN x kK.
TEST(RecipesFootprint, Orthonormalize) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kK = 6;
  fb::DeviceResources res{0};
  const fb::MemoryFootprint fp =
      fb::orthonormalize_footprint<double>(kN, kK, lwork_through(res, kK));
  std::mt19937 gen{20261109U};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> host(static_cast<std::size_t>(kN) * kK);
  for (double &x : host) {
    x = dist(gen);
  }
  fb::Buffers::device<double> v(host.size(), res.device());
  // G, its eigenvalues and D, one buffer.
  expect_routine_footprint(res, fp, 1, [&] {
    EXPECT_EQ(wwr::wwrMemcpyAsync(v.data(), host.data(), sizeof(double) * host.size(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    const auto kept = fb::orthonormalize<double>(res, kN, kK, v.data(), 1e-12);
    ASSERT_TRUE(kept);
    EXPECT_EQ(*kept, kK);
  });
}

// davidson_solve over A = diag(1..kN) + 1e-2 (R + R^T), one gemm an apply, a
// host-side diagonal preconditioner: neither allocates through the pool. Two
// roots in a subspace of 2 n_roots, which the second iteration fills, so
// syevd runs at every order up to it; with canonical_orthonormalization off
// and on.
TEST(RecipesFootprint, Davidson) {
  FBF_REQUIRE_GPU();
  constexpr std::int32_t kRoots = 2;
  constexpr std::int32_t kSubspace = 2 * kRoots;
  std::mt19937 gen{20261004U};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> a(static_cast<std::size_t>(kN) * kN, 0.0);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = j; i < kN; ++i) {
      a[i + (j * kN)] = a[j + (i * kN)] = i == j ? static_cast<double>(j + 1) : 1e-2 * dist(gen);
    }
  }
  struct Dense {
    const double *a;
    fb::Status apply(fb::DeviceResources &r, std::int32_t k, const double *x, double *y) const {
      const double one = 1.0;
      const double zero = 0.0;
      return fb::gpu_status(wwr::gemm<double, std::int32_t>(r.blas().get(), wwr::WWRBLAS_OP_N,
                                                            wwr::WWRBLAS_OP_N, kN, k, kN, &one, a,
                                                            kN, x, kN, &zero, y, kN));
    }
  };
  for (const bool canonical : {false, true}) {
    SCOPED_TRACE(canonical);
    fb::DeviceResources res{0};
    fb::Buffers::device<double> a_device(a.size(), res.device());
    EXPECT_EQ(wwr::wwrMemcpyAsync(a_device.data(), a.data(), sizeof(double) * a.size(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    // The unit vectors on the two lowest diagonal elements.
    std::vector<double> guess_host(static_cast<std::size_t>(kN) * kRoots, 0.0);
    for (std::int32_t r = 0; r < kRoots; ++r) {
      guess_host[r + (r * kN)] = 1.0;
    }
    fb::Buffers::device<double> guess(guess_host.size(), res.device());
    EXPECT_EQ(wwr::wwrMemcpyAsync(guess.data(), guess_host.data(),
                                  sizeof(double) * guess_host.size(), wwr::wwrMemcpyHostToDevice,
                                  res.stream()),
              wwr::wwrSuccess);
    fb::Buffers::device<double> values(kRoots, res.device());
    fb::Buffers::device<double> vectors(guess_host.size(), res.device());
    EXPECT_TRUE(res.sync());
    fb::DavidsonOptions options;
    options.canonical_orthonormalization = canonical;
    const fb::MemoryFootprint fp = fb::davidson_footprint<double>(kN, kRoots, kSubspace, options,
                                                                  lwork_through(res, kSubspace));
    const auto precondition = [&](fb::DeviceResources &r, std::span<const double> theta,
                                  const double *, const double *residual,
                                  double *correction) -> fb::Status {
      std::vector<double> host(static_cast<std::size_t>(kN) * theta.size());
      EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), residual, sizeof(double) * host.size(),
                                    wwr::wwrMemcpyDeviceToHost, r.stream()),
                wwr::wwrSuccess);
      EXPECT_TRUE(r.sync());
      for (std::size_t c = 0; c < theta.size(); ++c) {
        for (std::int32_t j = 0; j < kN; ++j) {
          double d = theta[c] - a[j + (j * kN)];
          d = std::abs(d) < 1e-3 ? (d < 0.0 ? -1e-3 : 1e-3) : d;
          host[(c * kN) + j] /= d;
        }
      }
      EXPECT_EQ(wwr::wwrMemcpyAsync(correction, host.data(), sizeof(double) * host.size(),
                                    wwr::wwrMemcpyHostToDevice, r.stream()),
                wwr::wwrSuccess);
      return r.sync();
    };
    Dense op{a_device.data()};
    // V, A V, V^T A V, H, the Ritz values, A X, R, the corrections, V^T C, the
    // norms and the overlaps; orthonormalize's one buffer when canonical.
    expect_routine_footprint(res, fp, 11 + (canonical ? 1 : 0), [&] {
      const auto info =
          fb::davidson_solve<double>(res, op, precondition, kN, kRoots, kSubspace, guess.data(),
                                     kRoots, values.data(), vectors.data(), options);
      ASSERT_TRUE(info) << info.error().message;
      EXPECT_GE(info->iterations, 2);
    });
  }
}

// The history allocated, then pushed past its capacity: the pushes allocate
// nothing.
// A real skew-symmetric matrix of order kN, nonzero (an exactly zero one takes
// no heevd): its exponential's eigendecomposition takes heevd's work, and
// each exponential its two complex blocks, both the device workspace only.
TEST(RecipesFootprint, SkewEigAndExpm) {
  FBF_REQUIRE_GPU();
  const auto n = static_cast<std::size_t>(kN);
  std::vector<double> host(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = j + 1; i < n; ++i) {
      host[i + (n * j)] = 1e-2 * static_cast<double>((i + (3 * j)) % 7);
      host[j + (n * i)] = -host[i + (n * j)];
    }
  }
  {
    fb::DeviceResources res{0};
    const auto lwork = fb::skew_eig_lwork<double>(res, kN);
    ASSERT_TRUE(lwork);
    const fb::MemoryFootprint fp = fb::skew_eig_footprint<double>(*lwork);
    ASSERT_GT(fp.device.workspace, 0U);
    fb::Buffers::device<double> h(n * n, res.device());
    fb::Buffers::device<wwr::wwrDoubleComplex> u(n * n, res.device());
    fb::Buffers::device<double> w(n, res.device());
    EXPECT_EQ(wwr::wwrMemcpyAsync(h.data(), host.data(), sizeof(double) * host.size(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    expect_routine_footprint(res, fp, 0, [&] {
      EXPECT_TRUE(fb::skew_eig<double>(res, kN, h.data(), u.data(), w.data()));
    });
  }
  {
    fb::DeviceResources res{0};
    const fb::MemoryFootprint fp = fb::skew_expm_footprint<double>(kN);
    // Two complex blocks, rounded to the workspace's alignment.
    EXPECT_EQ(fp.device.workspace, fb::math::align_up(2 * n * n * sizeof(wwr::wwrDoubleComplex),
                                                      fb::DeviceResources::kWorkspaceAlignment));
    fb::Buffers::device<wwr::wwrDoubleComplex> u(n * n, res.device());
    fb::Buffers::device<double> w(n, res.device());
    fb::Buffers::device<double> e(n * n, res.device());
    EXPECT_EQ(wwr::wwrMemsetAsync(u.data(), 0, sizeof(wwr::wwrDoubleComplex) * n * n, res.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrMemsetAsync(w.data(), 0, sizeof(double) * n, res.stream()), wwr::wwrSuccess);
    expect_routine_footprint(res, fp, 0, [&] {
      EXPECT_TRUE(fb::skew_expm<double>(res, kN, u.data(), w.data(), 0.5, e.data()));
    });
  }
}

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
