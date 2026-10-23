// pipek_mezey_localize on the device. The arithmetic is pipek_mezey_host.cpp's
// to check; this checks the driver and kernels reach a right answer with it:
// U orthogonal, C_local = C U and still S-orthonormal, the reported functional
// the one recomputed on the host from the definition, for both exponents and
// both precisions, over a basis whose atoms own several AOs; separated centres
// localizing onto single atoms; the edge counts; and every refusal.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.localization.pm;

namespace {

namespace fb = free_breakfast;

using Matrix = std::vector<double>;

double at(const Matrix &m, std::size_t ld, std::size_t r, std::size_t c) {
  return m[r + (c * ld)];
}
double &at(Matrix &m, std::size_t ld, std::size_t r, std::size_t c) {
  return m[r + (c * ld)];
}

// Column-major (rows x inner) times (inner x cols), optionally A^T.
Matrix multiply(const Matrix &a, const Matrix &b, std::size_t rows, std::size_t inner,
                std::size_t cols, bool transpose_a = false) {
  Matrix c(rows * cols, 0.0);
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t k = 0; k < inner; ++k) {
      for (std::size_t i = 0; i < rows; ++i) {
        c[i + (rows * j)] +=
            (transpose_a ? a[k + (inner * i)] : a[i + (rows * k)]) * b[k + (inner * j)];
      }
    }
  }
  return c;
}

// A basis whose atoms own the given shells' angular momenta, all on one line,
// placed so each atom's AOs are contiguous; S a Kac-Murdock-Szego matrix over
// its spherical AOs, and n_occ delocalized cosines made S-orthonormal.
struct Model {
  fb::BasisSet basis;
  std::size_t n_ao = 0;
  std::int32_t n_occ = 0;
  Matrix s;
  Matrix c;
  std::vector<std::size_t> offsets; // atom A owns AOs [offsets[A], offsets[A + 1])
};

Model make_model(const std::vector<std::vector<std::int32_t>> &atom_shells, std::int32_t n_occ,
                 double decay) {
  std::vector<fb::Shell> shells;
  std::vector<std::uint32_t> shell_offsets{0};
  std::vector<std::size_t> offsets{0};
  for (std::size_t a = 0; a < atom_shells.size(); ++a) {
    const fb::Vec<double, 3> center{0.0, 0.0, 1.5 * static_cast<double>(a)};
    std::size_t aos = 0;
    for (const std::int32_t l : atom_shells[a]) {
      shells.emplace_back(l, std::vector<double>{1.0}, std::vector<double>{1.0}, center);
      aos += static_cast<std::size_t>((2 * l) + 1);
    }
    shell_offsets.push_back(static_cast<std::uint32_t>(shells.size()));
    offsets.push_back(offsets.back() + aos);
  }
  Model m{.basis = fb::BasisSet(std::move(shells), std::move(shell_offsets)),
          .n_ao = offsets.back(),
          .n_occ = n_occ,
          .offsets = std::move(offsets)};
  const std::size_t n = m.n_ao;
  const auto occ = static_cast<std::size_t>(n_occ);
  m.s.resize(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      at(m.s, n, i, j) =
          std::exp(-decay * std::fabs(static_cast<double>(i) - static_cast<double>(j)));
    }
  }
  m.c.resize(n * occ);
  for (std::size_t j = 0; j < occ; ++j) {
    for (std::size_t mu = 0; mu < n; ++mu) {
      at(m.c, n, mu, j) = std::cos(static_cast<double>(j + 1) * (static_cast<double>(mu) + 0.5) *
                                   std::numbers::pi / static_cast<double>(n));
    }
  }
  // Modified Gram-Schmidt in the S metric.
  const auto s_dot = [&](std::size_t a, std::size_t b) {
    double acc = 0.0;
    for (std::size_t mu = 0; mu < n; ++mu) {
      for (std::size_t nu = 0; nu < n; ++nu) {
        acc += at(m.c, n, mu, a) * at(m.s, n, mu, nu) * at(m.c, n, nu, b);
      }
    }
    return acc;
  };
  for (std::size_t j = 0; j < occ; ++j) {
    for (std::size_t i = 0; i < j; ++i) {
      const double ov = s_dot(i, j);
      for (std::size_t mu = 0; mu < n; ++mu) {
        at(m.c, n, mu, j) -= ov * at(m.c, n, mu, i);
      }
    }
    const double norm = std::sqrt(s_dot(j, j));
    for (std::size_t mu = 0; mu < n; ++mu) {
      at(m.c, n, mu, j) /= norm;
    }
  }
  return m;
}

// L_p of the orbitals c, from the Mulliken definition.
double host_functional(const Model &m, const Matrix &c, std::int32_t p) {
  const auto occ = static_cast<std::size_t>(m.n_occ);
  const Matrix w = multiply(m.s, c, m.n_ao, m.n_ao, occ);
  double l = 0.0;
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::size_t a = 0; a + 1 < m.offsets.size(); ++a) {
      double q = 0.0;
      for (std::size_t mu = m.offsets[a]; mu < m.offsets[a + 1]; ++mu) {
        q += at(c, m.n_ao, mu, i) * at(w, m.n_ao, mu, i);
      }
      l += std::pow(q, p);
    }
  }
  return l;
}

template <typename T>
fb::Buffers::device<T> upload(fb::DeviceResources &res, const Matrix &host) {
  const std::vector<T> cast(host.begin(), host.end());
  fb::Buffers::device<T> buffer(std::max<std::size_t>(cast.size(), 1), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(buffer.data(), cast.data(), sizeof(T) * cast.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return buffer;
}

template <typename T>
Matrix download(fb::DeviceResources &res, const fb::Buffers::device<T> &buffer, std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), buffer.data(), sizeof(T) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return {host.begin(), host.end()};
}

struct Localized {
  fb::Result<fb::PipekMezeyInfo> info;
  Matrix c_local;
  Matrix u;
};

template <typename T>
Localized localize(fb::DeviceResources &res, const Model &m,
                   const fb::PipekMezeyOptions &options = {}) {
  const auto occ = static_cast<std::size_t>(m.n_occ);
  auto s = upload<T>(res, m.s);
  auto c = upload<T>(res, m.c);
  fb::Buffers::device<T> c_local(std::max<std::size_t>(m.n_ao * occ, 1), res.device());
  fb::Buffers::device<T> u(std::max<std::size_t>(occ * occ, 1), res.device());
  auto info = fb::pipek_mezey_localize<T>(res, m.basis, m.n_occ, s.data(), c.data(), c_local.data(),
                                          u.data(), options);
  return {std::move(info), download(res, c_local, m.n_ao * occ), download(res, u, occ * occ)};
}

// U^T U = I, C_local = C U and C_local^T S C_local = I, to tolerance.
void expect_rotation_of(const Model &m, const Localized &run, double tolerance) {
  const auto occ = static_cast<std::size_t>(m.n_occ);
  const Matrix utu = multiply(run.u, run.u, occ, occ, occ, /*transpose_a=*/true);
  const Matrix cu = multiply(m.c, run.u, m.n_ao, occ, occ);
  const Matrix ctsc = multiply(run.c_local, multiply(m.s, run.c_local, m.n_ao, m.n_ao, occ), occ,
                               m.n_ao, occ, /*transpose_a=*/true);
  for (std::size_t j = 0; j < occ; ++j) {
    for (std::size_t i = 0; i < occ; ++i) {
      EXPECT_NEAR(at(utu, occ, i, j), i == j ? 1.0 : 0.0, tolerance) << "U^T U " << i << "," << j;
      EXPECT_NEAR(at(ctsc, occ, i, j), i == j ? 1.0 : 0.0, tolerance)
          << "C^T S C " << i << "," << j;
    }
  }
  for (std::size_t k = 0; k < cu.size(); ++k) {
    EXPECT_NEAR(run.c_local[k], cu[k], tolerance) << "C U " << k;
  }
}

// Three atoms owning s+p, s and s+p: nine spherical AOs, ranges [0,4), [4,5),
// [5,9), so an atom's population sums a range, not one AO.
Model mixed_model() {
  return make_model({{0, 1}, {0}, {0, 1}}, 4, 0.6);
}

class PipekMezeyGpu : public ::testing::TestWithParam<fb::PipekMezeyExponent> {};

TEST_P(PipekMezeyGpu, RotatesWithinTheSpanAndReportsTheHostFunctional) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = mixed_model();
  const auto p = static_cast<std::int32_t>(GetParam());
  const Localized run = localize<double>(res, m, {.exponent = GetParam()});
  ASSERT_TRUE(run.info.has_value()) << run.info.error().message;
  expect_rotation_of(m, run, 1e-11);
  EXPECT_NEAR(run.info->initial_functional, host_functional(m, m.c, p), 1e-12);
  EXPECT_NEAR(run.info->functional, host_functional(m, run.c_local, p), 1e-10);
  EXPECT_GT(run.info->functional, run.info->initial_functional);
  EXPECT_GE(run.info->sweeps, 1);
}

TEST_P(PipekMezeyGpu, SinglePrecisionAgreesToItsOwnAccuracy) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = mixed_model();
  const fb::PipekMezeyOptions options{.functional_tolerance = 1e-5, .exponent = GetParam()};
  const Localized single = localize<float>(res, m, options);
  const Localized full = localize<double>(res, m, options);
  ASSERT_TRUE(single.info.has_value()) << single.info.error().message;
  ASSERT_TRUE(full.info.has_value()) << full.info.error().message;
  expect_rotation_of(m, single, 1e-4);
  EXPECT_NEAR(single.info->functional, full.info->functional, 1e-4);
}

INSTANTIATE_TEST_SUITE_P(Exponents, PipekMezeyGpu,
                         ::testing::Values(fb::PipekMezeyExponent::Two,
                                           fb::PipekMezeyExponent::Four),
                         [](const auto &info) {
                           return info.param == fb::PipekMezeyExponent::Two ? "P2" : "P4";
                         });

TEST(PipekMezeyGpuEdges, SeparatedCentresLocalizeOntoSingleAtoms) {
  // S ~ I and as many orbitals as atoms: the maximum puts each orbital wholly
  // on one atom, L = n_occ.
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = make_model({{0}, {0}, {0}, {0}, {0}}, 5, 40.0);
  const Localized run = localize<double>(res, m);
  ASSERT_TRUE(run.info.has_value()) << run.info.error().message;
  EXPECT_LT(run.info->initial_functional, 3.0);
  EXPECT_NEAR(run.info->functional, 5.0, 1e-8);
}

TEST(PipekMezeyGpuEdges, OneOrbitalPassesThrough) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = make_model({{0, 1}, {0}}, 1, 0.6);
  const Localized run = localize<double>(res, m);
  ASSERT_TRUE(run.info.has_value()) << run.info.error().message;
  EXPECT_EQ(run.info->sweeps, 0);
  EXPECT_EQ(run.info->functional, run.info->initial_functional);
  EXPECT_EQ(run.u, Matrix{1.0});
  EXPECT_EQ(run.c_local, m.c);
}

TEST(PipekMezeyGpuEdges, NoOrbitalsIsANoOp) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = make_model({{0}, {0}}, 0, 0.6);
  const Localized run = localize<double>(res, m);
  ASSERT_TRUE(run.info.has_value()) << run.info.error().message;
  EXPECT_EQ(run.info->sweeps, 0);
  EXPECT_EQ(run.info->functional, 0.0);
}

TEST(PipekMezeyGpuEdges, TheSweepCapIsMaxIterationsExceededWithTheLastSweepWritten) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = mixed_model();
  // A tolerance of 0 cannot be met: |change| < 0 is never true.
  const Localized run = localize<double>(res, m, {.max_sweeps = 1, .functional_tolerance = 0.0});
  ASSERT_FALSE(run.info.has_value());
  EXPECT_EQ(run.info.error().kind, fb::ErrorKind::MaxIterationsExceeded);
  expect_rotation_of(m, run, 1e-11);
}

TEST(PipekMezeyGpuEdges, RefusesMalformedRequests) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const Model m = mixed_model();
  const auto occ = static_cast<std::size_t>(m.n_occ);
  auto s = upload<double>(res, m.s);
  auto c = upload<double>(res, m.c);
  fb::Buffers::device<double> out(m.n_ao * occ, res.device());

  const auto expect_refused = [&](const fb::BasisSet &basis, std::int32_t n_occ, const double *sp,
                                  const double *cp, double *outp,
                                  const fb::PipekMezeyOptions &options, std::string_view what) {
    const auto info =
        fb::pipek_mezey_localize<double>(res, basis, n_occ, sp, cp, outp, nullptr, options);
    ASSERT_FALSE(info.has_value()) << what;
    EXPECT_EQ(info.error().kind, fb::ErrorKind::InvalidConfig) << what;
  };
  const fb::BasisSet unplaced(m.basis.shells());
  expect_refused(unplaced, m.n_occ, s.data(), c.data(), out.data(), {}, "no atoms");
  expect_refused(m.basis, m.n_occ, nullptr, c.data(), out.data(), {}, "null s");
  expect_refused(m.basis, m.n_occ, s.data(), nullptr, out.data(), {}, "null c");
  expect_refused(m.basis, m.n_occ, s.data(), c.data(), nullptr, {}, "null c_local");
  expect_refused(m.basis, m.n_occ, s.data(), c.data(), c.data(), {}, "c_local aliases c");
  expect_refused(m.basis, -1, s.data(), c.data(), out.data(), {}, "negative n_occ");
  expect_refused(m.basis, static_cast<std::int32_t>(m.n_ao) + 1, s.data(), c.data(), out.data(), {},
                 "n_occ past n_ao");
  expect_refused(m.basis, m.n_occ, s.data(), c.data(), out.data(), {.max_sweeps = 0},
                 "max_sweeps 0");
}

} // namespace
