// GPU cases for cis.precondition: the orbital-energy diagonal and the
// preconditioner's correction against the host, the guess against a host
// partial sort, and davidson_solve through both on a synthetic diagonally
// dominant operator against diagonalize's full spectrum. Then the clustered
// correction: a 2 × 2 block solved by hand, with θ clamped on either side of
// one of its eigenvalues; a gap of zero leaving every byte of the diagonal
// correction; its refusals; and davidson_solve on a near-degenerate operator
// against diagonalize's spectrum, with and without the clusters.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
import free_breakfast.cis.precondition;

namespace {

namespace fb = free_breakfast;
namespace cis = free_breakfast::cis;

std::vector<double> random_vector(std::size_t n, std::uint32_t seed, double lo, double hi) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{lo, hi};
  std::vector<double> v(n);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

fb::Buffers::device<double> to_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> to_host(fb::DeviceResources &res, const double *d, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// Δε_ia = ε_a − ε_i, row-major (i, a), the occupied energies first.
std::vector<double> host_diagonal(const std::vector<double> &eps, std::int32_t n_occ,
                                  std::int32_t n_virt) {
  std::vector<double> delta(static_cast<std::size_t>(n_occ) * n_virt);
  for (std::int32_t i = 0; i < n_occ; ++i) {
    for (std::int32_t a = 0; a < n_virt; ++a) {
      delta[static_cast<std::size_t>(i * n_virt + a)] =
          eps[static_cast<std::size_t>(n_occ + a)] - eps[static_cast<std::size_t>(i)];
    }
  }
  return delta;
}

// Orbital energies for n_occ occupied below n_virt virtual ones, each set
// ascending, as an SCF would give them.
std::vector<double> orbital_energies(std::int32_t n_occ, std::int32_t n_virt, std::uint32_t seed) {
  auto occ = random_vector(static_cast<std::size_t>(n_occ), seed, -20.0, -0.3);
  auto virt = random_vector(static_cast<std::size_t>(n_virt), seed + 1, 0.1, 5.0);
  std::ranges::sort(occ);
  std::ranges::sort(virt);
  occ.insert(occ.end(), virt.begin(), virt.end());
  return occ;
}

// The host reference correction, the floor and sign rule spelled out.
double host_correction(double r, double theta, double delta) {
  double d = theta - delta;
  if (std::abs(d) < 1e-3) {
    d = d < 0.0 ? -1e-3 : 1e-3;
  }
  return r / d;
}

TEST(CisPrecondition, DiagonalMatchesHost) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kOcc = 7;
  constexpr std::int32_t kVirt = 23;
  const auto eps = orbital_energies(kOcc, kVirt, 20261029U);
  const auto d_eps = to_device(res, eps);
  fb::Buffers::device<double> delta(static_cast<std::size_t>(kOcc) * kVirt, res.device());
  ASSERT_TRUE(cis::diagonal(res, kOcc, kVirt, d_eps.data(), delta.data()));
  const auto got = to_host(res, delta.data(), delta.num_elements());
  const auto want = host_diagonal(eps, kOcc, kVirt);
  for (std::size_t e = 0; e < want.size(); ++e) {
    EXPECT_NEAR(got[e], want[e], 1e-12) << e;
  }
}

TEST(CisPrecondition, DiagonalRejectsEmptySpaces) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> eps(4, res.device());
  fb::Buffers::device<double> delta(4, res.device());
  for (const auto [n_occ, n_virt] : {std::pair{0, 4}, std::pair{4, 0}, std::pair{-1, 2}}) {
    const auto status = cis::diagonal(res, n_occ, n_virt, eps.data(), delta.data());
    ASSERT_FALSE(status) << n_occ << " " << n_virt;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(status.error().message.find("cis::diagonal"), std::string::npos);
  }
}

// The guess against a host partial sort over (Δε, index), with ties built in:
// a pair inside the selected set, and a pair straddling its last slot, where
// only the index decides which is taken.
TEST(CisPrecondition, GuessIsLowestUnitVectorsTiesToLowerIndex) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 97;
  constexpr std::int32_t kCount = 6;
  auto delta = random_vector(kN, 7U, 0.5, 3.0);
  // Two entries tied well inside the lowest six (12 before 40), then the
  // third and fourth lowest.
  delta[40] = 0.10;
  delta[12] = 0.10;
  delta[3] = 0.20;
  delta[70] = 0.30;
  // The fifth and sixth lowest, and two more tied at the sixth: of indices 9,
  // 55 and 88 at 0.40, only 9 (the lowest) makes the cut after 81 at 0.35.
  delta[81] = 0.35;
  delta[88] = 0.40;
  delta[55] = 0.40;
  delta[9] = 0.40;

  std::vector<std::int32_t> order(kN);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + kCount, order.end(),
                    [&delta](std::int32_t l, std::int32_t r) {
                      return std::pair{delta[static_cast<std::size_t>(l)], l} <
                             std::pair{delta[static_cast<std::size_t>(r)], r};
                    });
  const std::vector<std::int32_t> expected{12, 40, 3, 70, 81, 9};
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), order.begin()));

  const auto d_delta = to_device(res, delta);
  // Filled with garbage first, so every element the guess leaves at zero is
  // one it wrote.
  fb::Buffers::device<double> vectors(static_cast<std::size_t>(kN) * kCount, res.device());
  ASSERT_EQ(wwr::wwrMemsetAsync(vectors.data(), 0xFF, vectors.size_bytes(), res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(cis::guess(res, kN, d_delta.data(), kCount, vectors.data()));
  const auto got = to_host(res, vectors.data(), vectors.num_elements());
  for (std::int32_t g = 0; g < kCount; ++g) {
    for (std::int32_t j = 0; j < kN; ++j) {
      const double want = (j == order[static_cast<std::size_t>(g)]) ? 1.0 : 0.0;
      EXPECT_EQ(got[static_cast<std::size_t>(g * kN + j)], want) << g << " " << j;
    }
  }

  // Deterministic: a second call writes the same bytes.
  ASSERT_TRUE(cis::guess(res, kN, d_delta.data(), kCount, vectors.data()));
  EXPECT_EQ(to_host(res, vectors.data(), vectors.num_elements()), got);
}

TEST(CisPrecondition, GuessRejectsCountOutsideTheSpace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> delta(4, res.device());
  fb::Buffers::device<double> vectors(20, res.device());
  for (const std::int32_t count : {0, 5}) {
    const auto status = cis::guess(res, 4, delta.data(), count, vectors.data());
    ASSERT_FALSE(status) << count;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(status.error().message.find("cis::guess"), std::string::npos);
  }
}

// The correction against r / (θ − Δε) over a block, with θ placed exactly on
// one diagonal entry (a zero denominator, clamped to +1e-3) and just above
// another (a small negative one, clamped to −1e-3, its sign kept).
TEST(CisPrecondition, CorrectionMatchesHostAndStaysFinite) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 131;
  constexpr std::int32_t kColumns = 4;
  const auto delta = random_vector(kN, 11U, 0.2, 4.0);
  const auto residual = random_vector(static_cast<std::size_t>(kN) * kColumns, 13U, -1.0, 1.0);
  const std::vector<double> theta{delta[17], delta[60] - 1e-5, 0.9, 2.5};
  const auto d_delta = to_device(res, delta);
  const auto d_residual = to_device(res, residual);
  fb::Buffers::device<double> correction(residual.size(), res.device());

  // Built for one more column than the call uses.
  const cis::Preconditioner precondition{res, kN, d_delta.data(), kColumns + 1};
  ASSERT_TRUE(precondition(res, theta, nullptr, d_residual.data(), correction.data()));
  const auto got = to_host(res, correction.data(), correction.num_elements());
  for (std::int32_t c = 0; c < kColumns; ++c) {
    for (std::int32_t j = 0; j < kN; ++j) {
      const auto e = static_cast<std::size_t>(c * kN + j);
      const double want = host_correction(residual[e], theta[static_cast<std::size_t>(c)],
                                          delta[static_cast<std::size_t>(j)]);
      ASSERT_TRUE(std::isfinite(got[e])) << c << " " << j;
      EXPECT_NEAR(got[e], want, 1e-14 * std::max(1.0, std::abs(want))) << c << " " << j;
    }
  }
  EXPECT_EQ(got[17], residual[17] / 1e-3);
  EXPECT_EQ(got[kN + 60], residual[kN + 60] / -1e-3);

  // In place: correction may be the residual.
  auto d_inplace = to_device(res, residual);
  ASSERT_TRUE(precondition(res, theta, nullptr, d_inplace.data(), d_inplace.data()));
  EXPECT_EQ(to_host(res, d_inplace.data(), d_inplace.num_elements()), got);
}

TEST(CisPrecondition, CorrectionRejectsMoreColumnsThanBuiltFor) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> delta(8, res.device());
  fb::Buffers::device<double> block(24, res.device());
  const cis::Preconditioner precondition{res, 8, delta.data(), 2};
  const std::vector<double> theta{0.1, 0.2, 0.3};
  const auto status = precondition(res, theta, nullptr, block.data(), block.data());
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(status.error().message.find("cis::Preconditioner"), std::string::npos);
}

// A dense device A as a LinearOperator: one gemm per apply.
struct Dense {
  std::int32_t n;
  const double *a;

  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    const double one = 1.0;
    const double zero = 0.0;
    return fb::gpu_status(wwr::gemm<double, std::int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                          wwr::WWRBLAS_OP_N, n, k, n, &one, a, n, x,
                                                          n, &zero, y, n));
  }
};

// davidson_solve through this preconditioner and guess, on a synthetic CIS-like
// A = diag(Δε) + coupling: Δε from orbital energies through cis::diagonal, the
// coupling a small random symmetric matrix, as (ia|jb)-type terms are beside
// the orbital-energy gaps. Against diagonalize's full spectrum of the same A.
TEST(CisPrecondition, DavidsonConvergesToTheLowestRoots) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kOcc = 6;
  constexpr std::int32_t kVirt = 21;
  constexpr std::int32_t kN = kOcc * kVirt;
  constexpr std::int32_t kRoots = 4;
  constexpr std::int32_t kGuess = 2 * kRoots;
  const auto eps = orbital_energies(kOcc, kVirt, 4242U);
  const auto d_eps = to_device(res, eps);
  fb::Buffers::device<double> delta(static_cast<std::size_t>(kN), res.device());
  ASSERT_TRUE(cis::diagonal(res, kOcc, kVirt, d_eps.data(), delta.data()));
  const auto host_delta = to_host(res, delta.data(), delta.num_elements());

  std::mt19937 gen{99U};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> a(static_cast<std::size_t>(kN) * kN, 0.0);
  for (std::int32_t j = 0; j < kN; ++j) {
    for (std::int32_t i = j; i < kN; ++i) {
      const double value =
          (i == j) ? host_delta[static_cast<std::size_t>(j)] + 0.05 * dist(gen) : 0.02 * dist(gen);
      a[static_cast<std::size_t>(i + j * kN)] = a[static_cast<std::size_t>(j + i * kN)] = value;
    }
  }
  const auto d_a = to_device(res, a);

  // The oracle: every eigenvalue of A, ascending.
  auto d_copy = to_device(res, a);
  fb::Buffers::device<double> spectrum(static_cast<std::size_t>(kN), res.device());
  ASSERT_TRUE(fb::diagonalize<double>(res, kN, d_copy.data(), spectrum.data()));
  const auto reference = to_host(res, spectrum.data(), spectrum.num_elements());

  fb::Buffers::device<double> start(static_cast<std::size_t>(kN) * kGuess, res.device());
  ASSERT_TRUE(cis::guess(res, kN, delta.data(), kGuess, start.data()));
  const cis::Preconditioner precondition{res, kN, delta.data(), kRoots};
  Dense op{kN, d_a.data()};
  fb::Buffers::device<double> values(static_cast<std::size_t>(kRoots), res.device());
  fb::Buffers::device<double> vectors(static_cast<std::size_t>(kN) * kRoots, res.device());
  fb::DavidsonOptions options;
  options.residual_tolerance = 1e-10;
  const auto info =
      fb::davidson_solve<double>(res, op, precondition, kN, kRoots, 8 * kRoots, start.data(),
                                 kGuess, values.data(), vectors.data(), options);
  ASSERT_TRUE(info) << info.error().message;
  // The diagonal preconditioner on a diagonally dominant A converges in a
  // handful of steps; the bound only catches a preconditioner gone wrong.
  EXPECT_LT(info->iterations, 30);

  const auto got = to_host(res, values.data(), values.num_elements());
  for (std::int32_t r = 0; r < kRoots; ++r) {
    EXPECT_NEAR(got[static_cast<std::size_t>(r)], reference[static_cast<std::size_t>(r)], 1e-10)
        << r;
  }
}

// A Dense operator over the column-major host matrix a, uploaded.
struct DenseOnDevice {
  fb::Buffers::device<double> a;
  Dense op;

  DenseOnDevice(fb::DeviceResources &res, const std::vector<double> &host, std::int32_t n)
      : a{to_device(res, host)}, op{n, a.data()} {}
};

// diag(delta) with the symmetric coupling c between entries 0 and 3, which
// share Δε = 2: a cluster {0, 3} under any gap below the 1.5 to the others.
constexpr double kPairCoupling = 0.25;
const std::vector<double> kPairDelta{2.0, 3.5, 5.0, 2.0};

std::vector<double> pair_matrix() {
  std::vector<double> a(16, 0.0);
  for (std::size_t j = 0; j < 4; ++j) {
    a[j + (4 * j)] = kPairDelta[j];
  }
  a[0 + (4 * 3)] = a[3 + (4 * 0)] = kPairCoupling;
  return a;
}

// The pair's block B = [[2, c], [c, 2]], eigenvalues 2 ∓ c on (1, ∓1)/√2. Its
// correction solves (θ − B) x = r, so without a clamp
//   x0 = ((θ − 2) r0 + c r3) / ((θ − 2)² − c²), x3 = (c r0 + (θ − 2) r3) / (…);
// and with the denominators d± = clamp(θ − (2 ± c)) in the eigenbasis,
//   x0 = (r0 + r3) / 2d₊ + (r0 − r3) / 2d₋, x3 = (r0 + r3) / 2d₊ − (r0 − r3) / 2d₋.
// The other two entries keep r / (θ − Δε). Column 0 is clear of every pole;
// columns 1 and 2 put θ 1e-5 above and below 2 + c, so d₊ is clamped to +1e-3
// and −1e-3, its sign kept.
TEST(CisPrecondition, ClusteredPairMatchesTheHandSolveAndItsClamp) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 4;
  constexpr std::int32_t kColumns = 3;
  constexpr double kC = kPairCoupling;
  DenseOnDevice dense{res, pair_matrix(), kN};
  const auto d_delta = to_device(res, kPairDelta);
  const auto residual = random_vector(static_cast<std::size_t>(kN) * kColumns, 31U, -1.0, 1.0);
  const auto d_residual = to_device(res, residual);
  fb::Buffers::device<double> correction(residual.size(), res.device());
  const std::vector<double> theta{0.05, 2.0 + kC + 1e-5, 2.0 + kC - 1e-5};

  cis::Preconditioner precondition{res, kN, d_delta.data(), kColumns};
  ASSERT_TRUE(precondition.cluster(res, dense.op, kN, 0.5));
  ASSERT_TRUE(precondition(res, theta, nullptr, d_residual.data(), correction.data()));
  const auto got = to_host(res, correction.data(), correction.num_elements());

  for (std::int32_t c = 0; c < kColumns; ++c) {
    const auto col = static_cast<std::size_t>(c * kN);
    const double t = theta[static_cast<std::size_t>(c)];
    const double r0 = residual[col];
    const double r3 = residual[col + 3];
    double x0 = 0.0;
    double x3 = 0.0;
    if (c == 0) {
      const double e = t - 2.0;
      const double det = (e * e) - (kC * kC);
      x0 = ((e * r0) + (kC * r3)) / det;
      x3 = ((kC * r0) + (e * r3)) / det;
    } else {
      const double d_plus = c == 1 ? 1e-3 : -1e-3;
      const double d_minus = t - (2.0 - kC);
      x0 = ((r0 + r3) / (2.0 * d_plus)) + ((r0 - r3) / (2.0 * d_minus));
      x3 = ((r0 + r3) / (2.0 * d_plus)) - ((r0 - r3) / (2.0 * d_minus));
    }
    EXPECT_NEAR(got[col], x0, 1e-12 * std::max(1.0, std::abs(x0))) << c;
    EXPECT_NEAR(got[col + 3], x3, 1e-12 * std::max(1.0, std::abs(x3))) << c;
    for (const std::size_t j : {std::size_t{1}, std::size_t{2}}) {
      const double want = host_correction(residual[col + j], t, kPairDelta[j]);
      EXPECT_NEAR(got[col + j], want, 1e-14 * std::max(1.0, std::abs(want))) << c << " " << j;
    }
    // The block's coupling moved the covered entries off the diagonal's.
    EXPECT_GT(std::abs(got[col] - host_correction(r0, t, kPairDelta[0])), 1e-3) << c;
  }

  // In place, the covered rows read before the diagonal correction writes.
  auto d_inplace = to_device(res, residual);
  ASSERT_TRUE(precondition(res, theta, nullptr, d_inplace.data(), d_inplace.data()));
  EXPECT_EQ(to_host(res, d_inplace.data(), d_inplace.num_elements()), got);
}

// A gap of zero builds nothing, so the correction is the diagonal one to the
// byte: on its own, and after clusters built at a wider gap, which it
// replaces. The operator has a degenerate pair and a coupled near-degenerate
// run, which any positive gap here would cluster.
TEST(CisPrecondition, ClusterGapZeroLeavesTheCorrectionUnchanged) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 53;
  constexpr std::int32_t kColumns = 3;
  auto delta = random_vector(kN, 41U, 0.3, 4.0);
  delta[7] = delta[30] = 0.25;
  delta[11] = 0.2501;
  delta[40] = 0.2502;
  std::vector<double> a(static_cast<std::size_t>(kN) * kN, 0.0);
  std::mt19937 gen{43U};
  std::uniform_real_distribution<double> dist{-0.01, 0.01};
  for (std::int32_t j = 0; j < kN; ++j) {
    a[static_cast<std::size_t>(j + j * kN)] = delta[static_cast<std::size_t>(j)];
    for (std::int32_t i = j + 1; i < kN; ++i) {
      a[static_cast<std::size_t>(i + j * kN)] = a[static_cast<std::size_t>(j + i * kN)] = dist(gen);
    }
  }
  DenseOnDevice dense{res, a, kN};
  const auto d_delta = to_device(res, delta);
  const auto d_residual =
      to_device(res, random_vector(static_cast<std::size_t>(kN) * kColumns, 47U, -1.0, 1.0));
  const std::vector<double> theta{0.25, 0.2501, 1.7};
  const auto corrected = [&](const cis::Preconditioner &precondition) {
    fb::Buffers::device<double> out(static_cast<std::size_t>(kN) * kColumns, res.device());
    EXPECT_TRUE(precondition(res, theta, nullptr, d_residual.data(), out.data()));
    return to_host(res, out.data(), out.num_elements());
  };

  const cis::Preconditioner plain{res, kN, d_delta.data(), kColumns};
  const auto want = corrected(plain);

  cis::Preconditioner off{res, kN, d_delta.data(), kColumns};
  ASSERT_TRUE(off.cluster(res, dense.op, kN, 0.0));
  EXPECT_EQ(corrected(off), want);

  cis::Preconditioner replaced{res, kN, d_delta.data(), kColumns};
  ASSERT_TRUE(replaced.cluster(res, dense.op, kN, 0.01));
  EXPECT_NE(corrected(replaced), want);
  ASSERT_TRUE(replaced.cluster(res, dense.op, kN, 0.0));
  EXPECT_EQ(corrected(replaced), want);
}

// A sigma that fails: its error is passed up, and no cluster is left built.
struct FailingSigma {
  fb::Status apply(fb::DeviceResources & /*res*/, std::int32_t /*k*/, const double * /*x*/,
                   double * /*y*/) {
    return fb::err_numerical("FailingSigma: refused");
  }
};

// A scope outside [1, n_ov] and a gap negative or NaN are refused before
// anything runs; a failing sigma's error is passed up, leaving no cluster.
TEST(CisPrecondition, ClusterRejectsBadArguments) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  DenseOnDevice dense{res, pair_matrix(), 4};
  const auto d_delta = to_device(res, kPairDelta);
  cis::Preconditioner precondition{res, 4, d_delta.data(), 1};
  for (const auto [scope, gap] : {std::pair{0, 0.5}, std::pair{5, 0.5}, std::pair{4, -0.1},
                                  std::pair{4, std::numeric_limits<double>::quiet_NaN()}}) {
    const auto status = precondition.cluster(res, dense.op, scope, gap);
    ASSERT_FALSE(status) << scope << " " << gap;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(status.error().message.find("cis::Preconditioner::cluster"), std::string::npos)
        << status.error().message;
  }

  FailingSigma failing;
  const auto status = precondition.cluster(res, failing, 4, 0.5);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::Numerical);
  const std::vector<double> theta{2.1};
  const auto residual = random_vector(4, 53U, -1.0, 1.0);
  const auto d_residual = to_device(res, residual);
  fb::Buffers::device<double> correction(4, res.device());
  ASSERT_TRUE(precondition(res, theta, nullptr, d_residual.data(), correction.data()));
  const auto got = to_host(res, correction.data(), correction.num_elements());
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_EQ(got[j], residual[j] / (theta[0] - kPairDelta[j])) << j;
  }
}

// davidson_solve on a near-degenerate operator: an exactly degenerate block
// of eight at the bottom, Δε = 1, split only by a 1e-3 random coupling over
// the whole matrix, the rest at 1 + 0.2 i. The diagonal correction cannot
// tell the block's directions apart; the clusters, over the twelve lowest Δε
// at a gap of 1e-2, take the block exactly. Four roots from a guess of four,
// narrower than the block, so the first subspace is not already the answer.
// Both converge to diagonalize's lowest four.
TEST(CisPrecondition, DavidsonConvergesWithAndWithoutClusters) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 60;
  constexpr std::int32_t kBlock = 8;
  constexpr std::int32_t kRoots = 4;
  constexpr std::int32_t kScope = 12;
  constexpr std::int32_t kMaxSubspace = 40;
  std::vector<double> delta(kN);
  for (std::int32_t i = 0; i < kN; ++i) {
    delta[static_cast<std::size_t>(i)] = i < kBlock ? 1.0 : 1.0 + (0.2 * i);
  }
  std::mt19937 gen{2318U};
  std::uniform_real_distribution<double> dist{-1e-3, 1e-3};
  std::vector<double> a(static_cast<std::size_t>(kN) * kN, 0.0);
  for (std::int32_t j = 0; j < kN; ++j) {
    a[static_cast<std::size_t>(j + j * kN)] = delta[static_cast<std::size_t>(j)];
    for (std::int32_t i = j + 1; i < kN; ++i) {
      a[static_cast<std::size_t>(i + j * kN)] = a[static_cast<std::size_t>(j + i * kN)] = dist(gen);
    }
  }
  DenseOnDevice dense{res, a, kN};
  const auto d_delta = to_device(res, delta);

  auto d_copy = to_device(res, a);
  fb::Buffers::device<double> spectrum(static_cast<std::size_t>(kN), res.device());
  ASSERT_TRUE(fb::diagonalize<double>(res, kN, d_copy.data(), spectrum.data()));
  const auto reference = to_host(res, spectrum.data(), spectrum.num_elements());

  fb::DavidsonOptions options;
  options.residual_tolerance = 1e-10;
  options.max_iterations = 200;
  for (const double gap : {0.0, 1e-2}) {
    cis::Preconditioner precondition{res, kN, d_delta.data(), kRoots};
    ASSERT_TRUE(precondition.cluster(res, dense.op, kScope, gap)) << gap;
    fb::Buffers::device<double> start(static_cast<std::size_t>(kN) * kRoots, res.device());
    ASSERT_TRUE(cis::guess(res, kN, d_delta.data(), kRoots, start.data()));
    fb::Buffers::device<double> values(static_cast<std::size_t>(kRoots), res.device());
    fb::Buffers::device<double> vectors(static_cast<std::size_t>(kN) * kRoots, res.device());
    const auto info =
        fb::davidson_solve<double>(res, dense.op, precondition, kN, kRoots, kMaxSubspace,
                                   start.data(), kRoots, values.data(), vectors.data(), options);
    ASSERT_TRUE(info) << gap << ": " << info.error().message;
    std::cout << "gap " << gap << ": " << info->iterations << " iterations\n";
    const auto got = to_host(res, values.data(), values.num_elements());
    for (std::int32_t r = 0; r < kRoots; ++r) {
      EXPECT_NEAR(got[static_cast<std::size_t>(r)], reference[static_cast<std::size_t>(r)], 1e-10)
          << gap << " " << r;
    }
  }
}

} // namespace
