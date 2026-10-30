// GPU cases for integrals.eri.integral_direct.nonsymmetric_jk: G = J[D] -
// K[D]/2 and K[D] of a random non-symmetric density D on H2O in STO-3G and
// cc-pVDZ, against libcint's vj and vk of the same D, recorded by PySCF
// (test/gtest/data/exchange/*_nonsymmetric.json, written by
// tools/emit/nonsymmetric_jk_golden.py), unscreened (every record, tau = 0)
// and screened (K at its default, J at 3e-12: below), all to FelizJueves's
// 1e-12 bar; K[D^T] = K[D]^T and
// G[D^T] = G[D]^T on the device's own results; that a second build, of each of
// nonsymmetric_jk, coulomb and exchange, allocates nothing; and the refusals.

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

// exchange_gpu.cpp's and coulomb_gpu.cpp's bar, FelizJueves's against
// libcint: 1e-12 of the result's largest element.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<double> density; // nao x nao, row-major; not symmetric
  std::vector<double> vj;
  std::vector<double> vk;
};

Golden read_golden(const std::string &stem) {
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "exchange" /
                                 (stem + "_nonsymmetric.json"));
  EXPECT_TRUE(doc) << stem;
  if (!doc) {
    return {};
  }
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(),
                        fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  const auto &arrays = (*doc)["arrays"];
  Golden g{fb::BasisSet(std::move(shells)), arrays["density"]["values"].get<std::vector<double>>(),
           arrays["vj"]["values"].get<std::vector<double>>(),
           arrays["vk"]["values"].get<std::vector<double>>()};
  EXPECT_EQ(g.basis.nao_cart(), (*doc)["nao"].get<std::uint32_t>()) << stem;
  return g;
}

// m's transpose, n x n.
std::vector<double> transposed(const std::vector<double> &m, const std::size_t n) {
  std::vector<double> t(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      t[j * n + i] = m[i * n + j];
    }
  }
  return t;
}

// A row-major host matrix onto the column-major device matrix: element (i, j)
// is host[i * n + j] on the host and data()[i + j * n] on the device, so the
// copy is of the transpose. The goldens are not symmetric, so this matters.
void upload_rows(fb::DeviceResources &res, const std::vector<double> &host,
                 fb::CartesianMatrixAO &m) {
  ASSERT_EQ(host.size(), m.num_elements());
  const std::vector<double> columns = transposed(host, m.n());
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), columns.data(), sizeof(double) * columns.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // columns is pageable: the copy has not finished reading it until the
  // stream has.
  ASSERT_TRUE(res.sync().has_value());
}

// The device matrix back, row-major.
std::vector<double> download_rows(fb::DeviceResources &res, const fb::CartesianMatrixAO &m) {
  std::vector<double> columns(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(columns.data(), m.data(), sizeof(double) * columns.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return transposed(columns, m.n());
}

double max_abs(const std::vector<double> &m) {
  double scale = 0.0;
  for (const double v : m) {
    scale = std::max(scale, std::abs(v));
  }
  return scale;
}

// J's and K's classes of one basis, as pipeline.eri's IntegralDirect holds
// them: J's pruned at prune, K's every record.
struct Classes {
  std::vector<fb::CoulombPairClass> j;
  std::vector<fb::OrderedPairClass> k;
};

Classes classes_of(fb::DeviceResources &res, const fb::BasisSet &basis, const double prune) {
  const auto mmd = fb::primitive_pair_classes(res, basis);
  EXPECT_TRUE(mmd);
  if (!mmd) {
    return {};
  }
  auto j = fb::coulomb_pair_classes(res, *mmd, prune);
  auto k = fb::ordered_pair_classes(res, basis, *mmd);
  EXPECT_TRUE(j);
  EXPECT_TRUE(k);
  if (!j || !k) {
    return {};
  }
  return {std::move(*j), std::move(*k)};
}

struct GK {
  std::vector<double> g;
  std::vector<double> k;
};

// G and K of the row-major density d over basis, downloaded row-major.
GK run(fb::DeviceResources &res, Classes &classes, const fb::BasisSet &basis,
       const std::vector<double> &d, const double tau_j, const double tau_k) {
  fb::CartesianMatrixAO density(res, basis);
  upload_rows(res, d, density);
  fb::CartesianMatrixAO g(res, basis);
  fb::CartesianMatrixAO k(res, basis);
  const fb::Status status =
      fb::nonsymmetric_jk(res, classes.j, classes.k, density, g, k, tau_j, tau_k);
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return {download_rows(res, g), download_rows(res, k)};
}

class NonsymmetricJkGolden : public ::testing::TestWithParam<const char *> {};

// Unscreened (J's classes pruned at 0, both taus 0) and screened
// (coulomb_pair_classes' default prune, exchange's default 1e-13, and J at
// 3e-12), both to 1e-12 of the reference's largest element: K against vk, G
// against vj - vk/2. J is screened below its default 1e-10 because a dense
// random D is the worst case for its per-term bound: on H2O/cc-pVDZ, G at J's
// 1e-10 misses by 9.0e-12 of max|G| (2.2e-12 at 1e-11, 7.0e-13 at 3e-12;
// RX 9060 XT), all of it J's screen, since K alone at its default is
// 1.7e-13 of max|K|. The README's "The screen" has the ladder. Then D^T, through the same classes: K[D^T] = K[D]^T, by the
// bra-ket swap, and G[D^T] = G[D]^T, J being symmetric, to the same bar; not
// bitwise, since an a == b block of K[S] is computed whole and J sums in the
// atomics' order. Each run's worst errors are recorded.
TEST_P(NonsymmetricJkGolden, MatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden golden = read_golden(GetParam());
  ASSERT_FALSE(golden.vk.empty());
  const std::size_t n = golden.basis.nao_cart();
  std::vector<double> want_g(n * n);
  for (std::size_t e = 0; e < n * n; ++e) {
    want_g[e] = golden.vj[e] - (0.5 * golden.vk[e]);
  }
  const double scale_k = max_abs(golden.vk);
  const double scale_g = max_abs(want_g);
  // Else K[D^T] = K[D]^T could not tell K[D] from its symmetric part's.
  double asymmetry = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      asymmetry = std::max(asymmetry, std::abs(golden.vk[i * n + j] - golden.vk[j * n + i]));
    }
  }
  ASSERT_GT(asymmetry, 1e-3 * scale_k);
  const std::vector<double> d_t = transposed(golden.density, n);

  fb::DeviceResources res{0};
  for (const bool screened : {false, true}) {
    const std::string label = screened ? "screened" : "tau 0";
    Classes classes = classes_of(res, golden.basis, screened ? 1e-13 : 0.0);
    ASSERT_FALSE(classes.k.empty()) << label;
    const double tau_j = screened ? 3e-12 : 0.0;
    const double tau_k = screened ? 1e-13 : 0.0;
    const GK got = run(res, classes, golden.basis, golden.density, tau_j, tau_k);
    const GK swapped = run(res, classes, golden.basis, d_t, tau_j, tau_k);
    ASSERT_EQ(got.g.size(), n * n) << label;
    ASSERT_EQ(swapped.k.size(), n * n) << label;
    double worst_g = 0.0;
    double worst_k = 0.0;
    double worst_swap = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t e = i * n + j;
        const std::size_t t = j * n + i;
        const double err_g = std::abs(got.g[e] - want_g[e]);
        const double err_k = std::abs(got.k[e] - golden.vk[e]);
        worst_g = std::max(worst_g, err_g);
        worst_k = std::max(worst_k, err_k);
        EXPECT_LE(err_g, kRelTol * scale_g) << label << ", G(" << i << ", " << j << ")";
        EXPECT_LE(err_k, kRelTol * scale_k) << label << ", K(" << i << ", " << j << ")";
        const double swap_k = std::abs(swapped.k[e] - got.k[t]);
        const double swap_g = std::abs(swapped.g[e] - got.g[t]);
        worst_swap = std::max({worst_swap, swap_k, swap_g});
        EXPECT_LE(swap_k, kRelTol * scale_k) << label << ", K[D^T](" << i << ", " << j << ")";
        EXPECT_LE(swap_g, kRelTol * scale_g) << label << ", G[D^T](" << i << ", " << j << ")";
      }
    }
    std::cout << "[ libcint  ] " << GetParam() << " " << label << ": G worst " << worst_g << " ("
              << worst_g / scale_g << " of max|G|), K worst " << worst_k << " ("
              << worst_k / scale_k << " of max|K|), D^T against the transpose " << worst_swap
              << "\n";
    const std::string key = screened ? "screened" : "tau_0";
    RecordProperty(key + "_g_worst_rel_error", std::format("{:.3e}", worst_g / scale_g));
    RecordProperty(key + "_k_worst_rel_error", std::format("{:.3e}", worst_k / scale_k));
    RecordProperty(key + "_transpose_worst_abs_error", std::format("{:.3e}", worst_swap));
  }
}

INSTANTIATE_TEST_SUITE_P(H2O, NonsymmetricJkGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

// The pool's high-water mark over fn, above what was in use when it started
// (zero if it never rose past that), with res synchronized on both sides:
// cis_sigma_gpu.cpp's, which has why the clamp (HIP has read the mark below
// the starting use).
std::uint64_t pool_peak(fb::DeviceResources &res, const std::function<void()> &fn) {
  wwr::wwrMemPool_t pool = res.device()->pool().get();
  EXPECT_TRUE(res.sync().has_value());
  std::uint64_t zero = 0;
  EXPECT_EQ(wwr::wwrMemPoolSetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &zero),
            wwr::wwrSuccess);
  std::uint64_t before = 0;
  EXPECT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemCurrent, &before),
            wwr::wwrSuccess);
  fn();
  EXPECT_TRUE(res.sync().has_value());
  std::uint64_t high = 0;
  EXPECT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &high),
            wwr::wwrSuccess);
  return high > before ? high - before : 0;
}

// Every buffer a build writes is its classes' or, for nonsymmetric_jk's K[D],
// the workspace, so once one call has grown the workspace, a second
// nonsymmetric_jk, coulomb or exchange takes nothing from the device's pool
// and leaves the workspace where it was.
TEST(NonsymmetricJk, SecondBuildsAllocateNothing) {
  FBF_REQUIRE_GPU();
  const Golden golden = read_golden("h2o_ccpvdz");
  ASSERT_FALSE(golden.density.empty());
  fb::DeviceResources res{0};
  Classes classes = classes_of(res, golden.basis, 1e-13);
  ASSERT_FALSE(classes.j.empty());
  ASSERT_FALSE(classes.k.empty());
  fb::CartesianMatrixAO density(res, golden.basis);
  upload_rows(res, golden.density, density);
  fb::CartesianMatrixAO g(res, golden.basis);
  fb::CartesianMatrixAO k(res, golden.basis);
  ASSERT_TRUE(fb::nonsymmetric_jk(res, classes.j, classes.k, density, g, k).has_value());

  const std::byte *const workspace = res.device_workspace(0).data();
  const std::size_t workspace_bytes = res.device_workspace(0).num_elements();
  const std::uint64_t jk_peak = pool_peak(res, [&] {
    EXPECT_TRUE(fb::nonsymmetric_jk(res, classes.j, classes.k, density, g, k).has_value());
  });
  const std::uint64_t j_peak =
      pool_peak(res, [&] { EXPECT_TRUE(fb::coulomb(res, classes.j, density, k).has_value()); });
  const std::uint64_t k_peak =
      pool_peak(res, [&] { EXPECT_TRUE(fb::exchange(res, classes.k, density, k).has_value()); });
  std::cout << "pool peak: nonsymmetric_jk " << jk_peak << " B, coulomb " << j_peak
            << " B, exchange " << k_peak << " B\n";
  EXPECT_EQ(jk_peak, 0U);
  EXPECT_EQ(j_peak, 0U);
  EXPECT_EQ(k_peak, 0U);
  EXPECT_EQ(res.device_workspace(0).data(), workspace);
  EXPECT_EQ(res.device_workspace(0).num_elements(), workspace_bytes);
}

// Every refusal comes before anything is queued, so G and K keep what they
// held: here a stale 42.
class NonsymmetricJkRefusals : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    res_.emplace(0);
    classes_ = classes_of(*res_, one_, 0.0);
  }

  fb::Status call(const fb::CartesianMatrixAO &density, fb::CartesianMatrixAO &g,
                  fb::CartesianMatrixAO &k, const double tau_j = 1e-10,
                  const double tau_k = 1e-13) {
    return fb::nonsymmetric_jk(*res_, classes_.j, classes_.k, density, g, k, tau_j, tau_k);
  }

  static void expect_refused(const fb::Status &status, const std::string_view what) {
    ASSERT_FALSE(status) << what;
    EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_TRUE(status.error().message.starts_with("nonsymmetric_jk: "))
        << what << ": " << status.error().message;
  }

  const fb::BasisSet one_ =
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
  const fb::BasisSet two_ =
      fb::BasisSet({fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0}),
                    fb::Shell(0, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})});
  std::optional<fb::DeviceResources> res_;
  Classes classes_;
};

TEST_F(NonsymmetricJkRefusals, ANegativeOrNaNTau) {
  const fb::CartesianMatrixAO density(*res_, one_);
  fb::CartesianMatrixAO g(*res_, one_);
  fb::CartesianMatrixAO k(*res_, one_);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  expect_refused(call(density, g, k, -1e-10, 1e-13), "tau_j negative");
  expect_refused(call(density, g, k, 1e-10, -1e-13), "tau_k negative");
  expect_refused(call(density, g, k, nan, 1e-13), "tau_j NaN");
  expect_refused(call(density, g, k, 1e-10, nan), "tau_k NaN");
}

TEST_F(NonsymmetricJkRefusals, MatricesOfAnotherOrder) {
  const fb::CartesianMatrixAO density(*res_, one_);
  fb::CartesianMatrixAO g(*res_, one_);
  fb::CartesianMatrixAO k(*res_, one_);
  fb::CartesianMatrixAO big(*res_, two_);
  expect_refused(call(density, big, k), "G of another order");
  expect_refused(call(density, g, big), "K of another order");
}

// Each step reads one matrix while it writes another, so the three must be
// three, and a refused call leaves G and K as they were.
TEST_F(NonsymmetricJkRefusals, OneMatrixInTwoRoles) {
  fb::CartesianMatrixAO density(*res_, one_);
  fb::CartesianMatrixAO g(*res_, one_);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(g.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res_->stream()),
            wwr::wwrSuccess);
  expect_refused(call(density, g, g), "G is K");
  expect_refused(call(density, density, g), "G is the density");
  expect_refused(call(density, g, density), "K is the density");
  EXPECT_EQ(download_rows(*res_, g)[0], 42.0);
}

} // namespace
