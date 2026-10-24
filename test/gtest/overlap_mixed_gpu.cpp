// GPU cases for integrals.one_electron.overlap_mixed, S_{rows,cols} between two
// bases, one case per cell of {same, different} basis set x {same, different}
// geometry:
//
// - same basis, same geometry: H2O in STO-3G and cc-pVDZ and the H10 chain
//   against libcint's int1e_ovlp (test/gtest/data/one_electron/) and against
//   the square S that overlap builds;
// - different basis, same geometry: H2O, rows STO-3G, columns 6-31G or
//   cc-pVDZ;
// - different basis, different geometry: the same rows, columns 6-31G
//   displaced and distorted. Those three against libcint's
//   gto.intor_cross('int1e_ovlp') (test/gtest/data/overlap_mixed/, copied
//   from FelizJueves's fixtures by tools/extract_integral_golden.py), each read
//   in both orientations, so every ordered class through d is reached;
// - same basis, different geometry: no fixture has it, so s and p shells
//   against themselves translated, against the closed form of a primitive
//   pair's overlap.
//
// Plus the refusals. Every build writes into a buffer filled with NaN first,
// so a value the zeroing missed shows.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// FelizJueves holds its one-electron kernels to libcint at 1e-12 * max|S|
// (test/oracle/tolerance.py); the same bar here.
constexpr double kRelTol = 1e-12;

fb::BasisSet read_shells(const fb::json &shells) {
  std::vector<fb::Shell> out;
  for (const auto &sh : shells) {
    const auto c = sh["centre"].get<std::vector<double>>();
    out.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                     sh["coefficients"].get<std::vector<double>>(),
                     fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  return fb::BasisSet(std::move(out));
}

fb::json read_doc(const std::string &dir, const std::string &stem) {
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / dir / (stem + ".json"));
  EXPECT_TRUE(doc) << stem;
  return doc ? *doc : fb::json{};
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &buf) {
  std::vector<double> host(buf.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), buf.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// S_{rows,cols} on the device, column-major rows x cols, built into a buffer
// of NaN (every byte 0xFF).
std::vector<double> device_overlap_mixed(fb::DeviceResources &res, const fb::BasisSet &rows,
                                         const fb::BasisSet &cols) {
  fb::Buffers::device<double> s(static_cast<std::size_t>(rows.nao_cart()) * cols.nao_cart(),
                                res.device());
  EXPECT_EQ(wwr::wwrMemsetAsync(s.data(), 0xFF, sizeof(double) * s.num_elements(), res.stream()),
            wwr::wwrSuccess);
  const fb::Status status = fb::overlap_mixed(res, rows, cols, s);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, s);
}

// got, column-major n_rows x n_cols, against want, row-major n_rows x n_cols,
// within kRelTol of scale; the worst error recorded as name.
void expect_matches(const std::vector<double> &got, const std::vector<double> &want,
                    const std::size_t n_rows, const std::size_t n_cols, const double scale,
                    const std::string &name) {
  ASSERT_EQ(got.size(), n_rows * n_cols);
  ASSERT_EQ(want.size(), n_rows * n_cols);
  double worst = 0.0;
  for (std::size_t m = 0; m < n_rows; ++m) {
    for (std::size_t n = 0; n < n_cols; ++n) {
      const double err = std::abs(got[m + n * n_rows] - want[m * n_cols + n]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale) << name << " S(" << m << ", " << n << ")";
    }
  }
  ::testing::Test::RecordProperty(name, std::format("{:.3e}", worst / scale));
}

// ---------------------------------------------------------------------------
// Different basis set: same geometry, and another.
// ---------------------------------------------------------------------------

class OverlapMixedCross : public ::testing::TestWithParam<const char *> {};

TEST_P(OverlapMixedCross, MatchesLibcintBothWays) {
  FBF_REQUIRE_GPU();
  const fb::json doc = read_doc("overlap_mixed", GetParam());
  ASSERT_FALSE(doc.is_null());
  const fb::BasisSet rows = read_shells(doc["rows"]["shells"]);
  const fb::BasisSet cols = read_shells(doc["cols"]["shells"]);
  const std::size_t nr = rows.nao_cart();
  const std::size_t nc = cols.nao_cart();
  ASSERT_EQ(nr, doc["rows"]["nao"].get<std::size_t>());
  ASSERT_EQ(nc, doc["cols"]["nao"].get<std::size_t>());
  const auto want = doc["int1e_ovlp_cross"]["values"].get<std::vector<double>>();
  // Each basis's own diagonal bounds |S_mn| by Cauchy-Schwarz; libcint's
  // largest element is a fair stand-in at this size.
  const double scale = max_abs(want);

  fb::DeviceResources res{0};
  expect_matches(device_overlap_mixed(res, rows, cols), want, nr, nc, scale, "rows_cols");

  // Swapped: S_{cols,rows} is the transpose, so column-major cols x rows is
  // row-major rows x cols, the golden as it stands.
  const std::vector<double> swapped = device_overlap_mixed(res, cols, rows);
  std::vector<double> back(nr * nc);
  for (std::size_t n = 0; n < nc; ++n) {
    for (std::size_t m = 0; m < nr; ++m) {
      back[m + n * nr] = swapped[n + m * nc];
    }
  }
  expect_matches(back, want, nr, nc, scale, "cols_rows");
}

INSTANTIATE_TEST_SUITE_P(H2O, OverlapMixedCross,
                         ::testing::Values("h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz",
                                           "h2o_sto3g_x_631g_moved"));

// ---------------------------------------------------------------------------
// Same basis set, same geometry: the square S.
// ---------------------------------------------------------------------------

class OverlapMixedSquare : public ::testing::TestWithParam<const char *> {};

TEST_P(OverlapMixedSquare, MatchesLibcintAndOverlap) {
  FBF_REQUIRE_GPU();
  const fb::json doc = read_doc("one_electron", GetParam());
  ASSERT_FALSE(doc.is_null());
  const fb::BasisSet basis = read_shells(doc["shells"]);
  const std::size_t n = basis.nao_cart();
  const auto want = doc["arrays"]["int1e_ovlp"]["values"].get<std::vector<double>>();
  const double scale = max_abs(want);

  fb::DeviceResources res{0};
  const std::vector<double> got = device_overlap_mixed(res, basis, basis);
  expect_matches(got, want, n, n, scale, "libcint");

  // overlap's S, symmetric to rounding, so its column-major storage is also
  // its row-major one to that rounding.
  const auto classes = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(classes);
  fb::CartesianMatrixAO square(res, basis);
  ASSERT_TRUE(fb::overlap(res, *classes, square));
  std::vector<double> sym(square.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(sym.data(), square.data(), sizeof(double) * sym.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  // FelizJueves finds its host rectangular S within 8 eps of scale of the
  // square one (one_electron.md, "The cross-basis overlap"); the atomics'
  // order adds a few more roundings either side.
  const double bar = 32.0 * std::numeric_limits<double>::epsilon() * scale;
  double worst = 0.0;
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t k = 0; k < n; ++k) {
      const double err = std::abs(got[m + k * n] - sym[m + k * n]);
      worst = std::max(worst, err);
      EXPECT_LE(err, bar) << "S(" << m << ", " << k << ")";
    }
  }
  RecordProperty("vs_overlap_eps",
                 std::format("{:.2f}", worst / scale / std::numeric_limits<double>::epsilon()));
}

INSTANTIATE_TEST_SUITE_P(H2O, OverlapMixedSquare, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));
INSTANTIATE_TEST_SUITE_P(Chain, OverlapMixedSquare, ::testing::Values("h10_chain_ccpvdz"));

// ---------------------------------------------------------------------------
// Same basis set, different geometry: the closed form.
// ---------------------------------------------------------------------------

using Vec3 = fb::Vec<double, 3>;

// <a|b> of two unnormalised Cartesian primitives of momentum <= 1, exponents
// alpha on A and beta on B, component ia of a and ib of b (-1 for an s, else
// the axis of a p): with zeta = alpha + beta, P = (alpha A + beta B) / zeta and
// S_ss = (pi / zeta)^{3/2} exp(-alpha beta / zeta |A - B|^2),
//
//   <s|s> = S_ss,  <p_i|s> = (P - A)_i S_ss,  <s|p_j> = (P - B)_j S_ss,
//   <p_i|p_j> = ((P - A)_i (P - B)_j + delta_ij / (2 zeta)) S_ss.
double primitive_overlap(const double alpha, const Vec3 &a, const std::int32_t ia,
                         const double beta, const Vec3 &b, const std::int32_t ib) {
  const double zeta = alpha + beta;
  const std::array<double, 3> pa{(beta * (b.x - a.x)) / zeta, (beta * (b.y - a.y)) / zeta,
                                 (beta * (b.z - a.z)) / zeta};
  const std::array<double, 3> pb{(alpha * (a.x - b.x)) / zeta, (alpha * (a.y - b.y)) / zeta,
                                 (alpha * (a.z - b.z)) / zeta};
  const double r2 =
      ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)) + ((a.z - b.z) * (a.z - b.z));
  const double ss = std::pow(std::numbers::pi / zeta, 1.5) * std::exp(-alpha * beta / zeta * r2);
  if (ia < 0 && ib < 0) {
    return ss;
  }
  if (ib < 0) {
    return pa[ia] * ss;
  }
  if (ia < 0) {
    return pb[ib] * ss;
  }
  return ((pa[ia] * pb[ib]) + (ia == ib ? 0.5 / zeta : 0.0)) * ss;
}

// S_{rows,cols} of two bases of s and p shells, row-major, on the host: each
// shell's Cartesians in x, y, z order.
std::vector<double> host_overlap(const fb::BasisSet &rows, const fb::BasisSet &cols) {
  const std::size_t nc = cols.nao_cart();
  std::vector<double> out(static_cast<std::size_t>(rows.nao_cart()) * nc, 0.0);
  for (std::uint32_t i = 0; i < rows.nshells(); ++i) {
    const fb::Shell &si = rows.shells()[i];
    for (std::uint32_t j = 0; j < cols.nshells(); ++j) {
      const fb::Shell &sj = cols.shells()[j];
      const std::int32_t ni = si.angular_momentum() == 0 ? 1 : 3;
      const std::int32_t nj = sj.angular_momentum() == 0 ? 1 : 3;
      for (std::int32_t ci = 0; ci < ni; ++ci) {
        for (std::int32_t cj = 0; cj < nj; ++cj) {
          double v = 0.0;
          for (const fb::Primitive &p : si.primitives()) {
            for (const fb::Primitive &q : sj.primitives()) {
              v += p.coefficient * q.coefficient *
                   primitive_overlap(p.exponent, si.center(), ni == 1 ? -1 : ci, q.exponent,
                                     sj.center(), nj == 1 ? -1 : cj);
            }
          }
          out[((rows.ao_offset_cart(i) + ci) * nc) + cols.ao_offset_cart(j) + cj] = v;
        }
      }
    }
  }
  return out;
}

// H2O-like s and p shells, the p shell between the s shells (so classes (1, 0)
// and (0, 1) are both reached), every shell moved by shift.
fb::BasisSet sp_basis(const Vec3 &shift) {
  const auto at = [&](const double x, const double y, const double z) {
    return Vec3{x + shift.x, y + shift.y, z + shift.z};
  };
  return fb::BasisSet({
      fb::Shell(0, {{130.7, 0.15}, {23.8, 0.53}, {6.44, 0.44}}, at(0.0, 0.0, 0.0)),
      fb::Shell(1, {{5.03, 0.16}, {1.17, 0.61}, {0.38, 0.39}}, at(0.0, 0.0, 0.0)),
      fb::Shell(0, {{3.43, 0.15}, {0.62, 0.53}, {0.17, 0.44}}, at(1.43, 0.21, -1.07)),
      fb::Shell(0, {{3.43, 0.15}, {0.62, 0.53}, {0.17, 0.44}}, at(-1.38, 0.35, -1.12)),
  });
}

TEST(OverlapMixed, SameBasisAnotherGeometryMatchesTheClosedForm) {
  FBF_REQUIRE_GPU();
  const fb::BasisSet rows = sp_basis({0.0, 0.0, 0.0});
  const fb::BasisSet cols = sp_basis({0.31, -0.22, 0.47});
  const std::vector<double> want = host_overlap(rows, cols);
  fb::DeviceResources res{0};
  expect_matches(device_overlap_mixed(res, rows, cols), want, rows.nao_cart(), cols.nao_cart(),
                 max_abs(want), "closed_form");
}

// ---------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------

TEST(OverlapMixed, RefusesABufferOfTheWrongSize) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = sp_basis({0.0, 0.0, 0.0});
  fb::Buffers::device<double> s(static_cast<std::size_t>(basis.nao_cart()) * basis.nao_cart() - 1,
                                res.device());
  const fb::Status status = fb::overlap_mixed(res, basis, basis, s);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(OverlapMixed, RefusesAShellAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet rows = sp_basis({0.0, 0.0, 0.0});
  const fb::BasisSet cols({fb::Shell(3, {{1.0, 1.0}}, Vec3{0.0, 0.0, 0.0})});
  fb::Buffers::device<double> s(static_cast<std::size_t>(rows.nao_cart()) * cols.nao_cart(),
                                res.device());
  const fb::Status status = fb::overlap_mixed(res, rows, cols, s);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(OverlapMixed, AnEmptyBasisBuildsNothing) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  fb::Buffers::device<double> s(0, res.device());
  EXPECT_TRUE(fb::overlap_mixed(res, sp_basis({0.0, 0.0, 0.0}), fb::BasisSet{}, s));
}

} // namespace
