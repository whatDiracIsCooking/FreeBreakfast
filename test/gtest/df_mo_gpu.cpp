// GPU cases for integrals.eri.density_fitting.mo: B fitted by density_fit from
// PySCF's int2c2e and int3c2e (data/density_fitting/), taken into the MO basis
// of that case's DF-RHF orbitals by b_mo over the occ x virt, occ x occ and
// virt x virt ranges and contracted back to (pq|rs) against PySCF's DF ao2mo
// over the same aux basis; the active and core x active blocks against
// casscf_integrals over the same B; b_half against df_exchange's K and X;
// the documented layouts (the per-p slice's leading dimension, and C = I
// giving back B); and the empty, out-of-range and too-large requests.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.active_space;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.integrals.eri.density_fitting.mo;

namespace {

namespace fb = free_breakfast;

// The issue's bar: (pq|rs) from b_mo against PySCF's DF ao2mo, of max|(pq|rs)|
// over the block. Both contract the same fitted B, in different orders.
constexpr double kRelTol = 1e-10;
// b_mo against casscf_integrals over the same B and C, of the block's max:
// the same products, summed in another order.
constexpr double kSameFitTol = 1e-12;
// b_half's X and the test's syrk against df_exchange's K, of max|K|: a few
// ULP. The same BLAS calls on the same shapes and strides, so bit-identity is
// expected, but not asserted.
constexpr double kUlpTol = 8 * std::numeric_limits<double>::epsilon();

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

// The h2o_ccpvdz case: the metric and (P|mn) as df_jk_gpu.cpp reads them, its
// DF-RHF orbitals C, nao x nao column-major, and PySCF's DF (pq|rs) blocks,
// element (p, q, r, s) at p + np (q + nq (r + nr s)).
struct Golden {
  const fb::json *entry = nullptr;
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::int32_t nocc = 0;
  std::vector<double> metric;
  std::vector<double> b;
  std::vector<double> c;
  std::map<std::string, std::vector<double>> blocks;
};

const fb::json &manifest() {
  static const fb::json doc = [] {
    auto read = fb::read_json(kDataDir / "density_fitting.json");
    EXPECT_TRUE(read) << (read ? "" : read.error().message);
    return read ? std::move(*read) : fb::json{};
  }();
  return doc;
}

std::vector<double> read_array(const std::string &file, const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
  return a->data;
}

Golden read_golden() {
  const std::string name = "h2o_ccpvdz";
  for (const auto &c : manifest()["integrals"]) {
    if (c["name"].get<std::string>() != name) {
      continue;
    }
    Golden g;
    g.entry = &c;
    g.nao = c["nao"].get<std::int32_t>();
    g.naux = c["naux"].get<std::int32_t>();
    const auto nao = static_cast<std::size_t>(g.nao);
    const auto naux = static_cast<std::size_t>(g.naux);
    g.metric = read_array(c["int2c2e"].get<std::string>(), {naux, naux});

    // Each piece is (nao, nao, end - begin), Fortran order: (mn|P) at
    // m + nao (n + nao (P - begin)). Transposed here into [P][n][m].
    g.b.assign(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
    for (const auto &piece : c["int3c2e"]) {
      const auto range = piece["aux"].get<std::vector<std::size_t>>();
      const auto slab =
          read_array(piece["file"].get<std::string>(), {nao, nao, range[1] - range[0]});
      if (slab.empty()) {
        return {};
      }
      for (std::size_t p = range[0]; p < range[1]; ++p) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t m = 0; m < nao; ++m) {
            g.b[p + naux * (n + nao * m)] = slab[m + nao * (n + nao * (p - range[0]))];
          }
        }
      }
    }

    const auto &mo = c["mo"];
    g.nocc = mo["occupied"].get<std::int32_t>();
    g.c = read_array(mo["coefficients"].get<std::string>(), {nao, nao});
    const std::map<char, std::size_t> extent{{'o', static_cast<std::size_t>(g.nocc)},
                                             {'v', nao - static_cast<std::size_t>(g.nocc)}};
    for (const std::string block : {"ovov", "oooo", "vvvv"}) {
      // Split along s, the slowest index, so the pieces concatenate.
      std::vector<std::size_t> shape;
      for (const char x : block) {
        shape.push_back(extent.at(x));
      }
      auto &whole = g.blocks[block];
      for (const auto &piece : mo[block]) {
        const auto s = piece["s"].get<std::vector<std::size_t>>();
        shape.back() = s[1] - s[0];
        const auto part = read_array(piece["file"].get<std::string>(), shape);
        whole.insert(whole.end(), part.begin(), part.end());
      }
    }
    return g;
  }
  ADD_FAILURE() << "no integrals case " << name;
  return {};
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// (pq|rs) = sum_Q B_Q,pq B_Q,rs from b_mo's [Q][p][q] blocks, element
// (p, q, r, s) at p + np (q + nq (r + nr s)).
std::vector<double> contract(const std::vector<double> &left, const std::vector<double> &right,
                             const std::size_t naux, const std::size_t np, const std::size_t nq,
                             const std::size_t nr, const std::size_t ns) {
  std::vector<double> g(np * nq * nr * ns);
  for (std::size_t s = 0; s < ns; ++s) {
    for (std::size_t r = 0; r < nr; ++r) {
      for (std::size_t q = 0; q < nq; ++q) {
        for (std::size_t p = 0; p < np; ++p) {
          double sum = 0.0;
          for (std::size_t k = 0; k < naux; ++k) {
            sum += left[k + naux * (p + np * q)] * right[k + naux * (r + nr * s)];
          }
          g[p + np * (q + nq * (r + nr * s))] = sum;
        }
      }
    }
  }
  return g;
}

// The fitted B and C on the device, from the golden tensors through
// density_fit.
class DfMo : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    g_ = read_golden();
    ASSERT_GT(g_.naux, 0);
    ASSERT_GT(g_.nocc, 0);
    res_.emplace(0);
    metric_.emplace(g_.metric.size(), res_->device());
    b_.emplace(g_.b.size(), res_->device());
    c_.emplace(g_.c.size(), res_->device());
    upload(*res_, g_.metric, metric_->data());
    upload(*res_, g_.b, b_->data());
    upload(*res_, g_.c, c_->data());
    const fb::Status fitted = fb::density_fit(*res_, g_.nao, g_.naux, metric_->data(), b_->data());
    ASSERT_TRUE(fitted) << fitted.error().message;
  }

  // b_mo over rows x cols of C, read back.
  std::vector<double> mo(const fb::OrbitalRange rows, const fb::OrbitalRange cols) {
    const std::size_t count = static_cast<std::size_t>(g_.naux) * rows.count * cols.count;
    fb::Buffers::device<double> out(count, res_->device());
    const fb::Status built =
        fb::b_mo(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), rows, cols, out.data());
    EXPECT_TRUE(built) << built.error().message;
    return download(*res_, out.data(), count);
  }

  [[nodiscard]] fb::OrbitalRange occupied() const { return {.first = 0, .count = g_.nocc}; }
  [[nodiscard]] fb::OrbitalRange virtuals() const {
    return {.first = g_.nocc, .count = g_.nao - g_.nocc};
  }

  Golden g_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Buffers::device<double>> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
};

TEST_F(DfMo, BlocksMatchPyscfDfMoIntegrals) {
  const auto naux = static_cast<std::size_t>(g_.naux);
  const auto o = static_cast<std::size_t>(g_.nocc);
  const auto v = static_cast<std::size_t>(g_.nao - g_.nocc);
  const std::vector<double> ov = mo(occupied(), virtuals());
  const std::vector<double> oo = mo(occupied(), occupied());
  const std::vector<double> vv = mo(virtuals(), virtuals());
  const std::map<std::string, std::vector<double>> got{
      {"ovov", contract(ov, ov, naux, o, v, o, v)},
      {"oooo", contract(oo, oo, naux, o, o, o, o)},
      {"vvvv", contract(vv, vv, naux, v, v, v, v)}};
  for (const auto &[block, want] : g_.blocks) {
    const double scale = max_abs(want);
    const double worst = max_diff(got.at(block), want);
    ASSERT_GT(scale, 0.0) << block;
    EXPECT_LE(worst, kRelTol * scale)
        << block << ": max diff " << worst << ", " << worst / scale << " of max " << scale;
  }
}

// The active space and the core x active block against casscf_integrals over
// the same B and C: (tu|vw) its Hamiltonian's eri, and (iu|jv) its papa's
// core block. h only dresses h_eff, so it is zero here.
TEST_F(DfMo, ActiveBlocksMatchCasscfIntegrals) {
  const fb::json &c = *g_.entry;
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::Molecule molecule(*res_, std::move(atoms));
  const auto stamp = fb::resolve_basis(c["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  ASSERT_EQ(basis->nao_sphe(), static_cast<std::uint32_t>(g_.nao));
  const fb::SphericalMatrixAO h(*res_, *basis);
  fb::SphericalMatrixAO orbitals(*res_, *basis);
  upload(*res_, g_.c, orbitals.data());

  // Two occupied and two virtual orbitals active, the three below them core.
  constexpr std::int32_t kNcas = 4;
  const std::int32_t ncore = g_.nocc - 2;
  auto ints =
      fb::casscf_integrals(*res_, *basis, molecule, h, *b_, g_.naux, orbitals, ncore, kNcas);
  ASSERT_TRUE(ints) << ints.error().message;

  const auto naux = static_cast<std::size_t>(g_.naux);
  const auto n = static_cast<std::size_t>(g_.nao);
  const auto na = static_cast<std::size_t>(kNcas);
  const auto nc = static_cast<std::size_t>(ncore);
  const fb::OrbitalRange active{.first = ncore, .count = kNcas};
  const fb::OrbitalRange core{.first = 0, .count = ncore};

  const std::vector<double> aa = mo(active, active);
  const std::vector<double> tuvw = contract(aa, aa, naux, na, na, na, na);
  const std::vector<double> eri = download(*res_, ints->hamiltonian.eri.data(), na * na * na * na);
  EXPECT_LE(max_diff(tuvw, eri), kSameFitTol * max_abs(eri)) << "(tu|vw)";

  // b_mo over active x core is B_Q,ui; (iu|jv) = sum_Q B_Q,ui B_Q,vj, and
  // papa's (p, q, u, v) at p + n (q + n (u + n_act v)) is (pu|qv).
  const std::vector<double> ac = mo(active, core);
  const std::vector<double> papa = download(*res_, ints->papa.data(), n * n * na * na);
  std::vector<double> got;
  std::vector<double> want;
  for (std::size_t v = 0; v < na; ++v) {
    for (std::size_t u = 0; u < na; ++u) {
      for (std::size_t j = 0; j < nc; ++j) {
        for (std::size_t i = 0; i < nc; ++i) {
          double sum = 0.0;
          for (std::size_t k = 0; k < naux; ++k) {
            sum += ac[k + naux * (u + na * i)] * ac[k + naux * (v + na * j)];
          }
          got.push_back(sum);
          want.push_back(papa[i + n * (j + n * (u + na * v))]);
        }
      }
    }
  }
  EXPECT_LE(max_diff(got, want), kSameFitTol * max_abs(want)) << "(iu|jv)";
}

// b_half over the occupied range is df_exchange's X, element (Q, i, m) at
// Q + naux (i + nocc m); and the test's syrk and symmetrization of it is
// df_exchange's K.
TEST_F(DfMo, HalfOverTheOccupiedIsDfExchangesX) {
  const auto naux = static_cast<std::size_t>(g_.naux);
  const auto nao = static_cast<std::size_t>(g_.nao);
  const auto nocc = static_cast<std::size_t>(g_.nocc);
  const std::size_t rows = naux * nocc;
  fb::Buffers::device<double> x(rows * nao, res_->device());
  const fb::Status built =
      fb::b_half(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), occupied(), x.data());
  ASSERT_TRUE(built) << built.error().message;
  const std::vector<double> got_x = download(*res_, x.data(), x.num_elements());

  // X_Q,im = sum_n B_Q,nm C_n,i on the host.
  const std::vector<double> b = download(*res_, b_->data(), b_->num_elements());
  std::vector<double> want_x(rows * nao);
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t i = 0; i < nocc; ++i) {
      for (std::size_t q = 0; q < naux; ++q) {
        double sum = 0.0;
        for (std::size_t n = 0; n < nao; ++n) {
          sum += b[q + naux * (n + nao * m)] * g_.c[n + nao * i];
        }
        want_x[q + naux * (i + nocc * m)] = sum;
      }
    }
  }
  EXPECT_LE(max_diff(got_x, want_x), kSameFitTol * max_abs(want_x)) << "X";

  // K = L + L^T with the diagonal L's, L = 2 X^T X lower: df_exchange's steps.
  const wwr::wwrblasHandle_t blas = res_->blas().get();
  const auto n32 = g_.nao;
  const auto rows32 = static_cast<std::int32_t>(rows);
  constexpr double kOne = 1.0;
  constexpr double kTwo = 2.0;
  constexpr double kZero = 0.0;
  fb::Buffers::device<double> lower(nao * nao, res_->device());
  fb::Buffers::device<double> k(nao * nao, res_->device());
  ASSERT_EQ(wwr::wwrMemsetAsync(lower.data(), 0, sizeof(double) * nao * nao, res_->stream()),
            wwr::wwrSuccess);
  ASSERT_EQ(
      (wwr::syrk<double, std::int32_t>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T, n32,
                                       rows32, &kTwo, x.data(), rows32, &kZero, lower.data(), n32)),
      wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(
      (wwr::geam<double, std::int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n32, n32, &kOne,
                                       lower.data(), n32, &kOne, lower.data(), n32, k.data(), n32)),
      wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ((wwr::copy<double, std::int32_t>(blas, n32, lower.data(), n32 + 1, k.data(), n32 + 1)),
            wwr::WWRBLAS_STATUS_SUCCESS);
  const std::vector<double> got_k = download(*res_, k.data(), k.num_elements());

  fb::Buffers::device<double> exchange(nao * nao, res_->device());
  const fb::Status reference =
      fb::df_exchange(*res_, g_.nao, g_.naux, g_.nocc, b_->data(), c_->data(), exchange.data());
  ASSERT_TRUE(reference) << reference.error().message;
  const std::vector<double> want_k = download(*res_, exchange.data(), exchange.num_elements());
  const double scale = max_abs(want_k);
  const double worst = max_diff(got_k, want_k);
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kUlpTol * scale) << "max|K - df_exchange| = " << worst << " of max " << scale;
}

// b_mo's slice at fixed p is naux x |cols| with leading dimension naux |rows|,
// consecutive p naux apart: MP2's per-i pair gemm, (ia|ib) = B_i^T B_i, is one
// strided batch over it with no transpose.
TEST_F(DfMo, SliceAtFixedPHasLeadingDimensionNauxRows) {
  const std::int32_t o = g_.nocc;
  const std::int32_t v = g_.nao - g_.nocc;
  const auto naux = static_cast<std::size_t>(g_.naux);
  const auto ld = static_cast<std::int32_t>(naux * static_cast<std::size_t>(o));
  const std::size_t count = naux * static_cast<std::size_t>(o * v);
  fb::Buffers::device<double> ov(count, res_->device());
  const fb::Status built = fb::b_mo(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(),
                                    occupied(), virtuals(), ov.data());
  ASSERT_TRUE(built) << built.error().message;

  const auto vv = static_cast<std::size_t>(v) * static_cast<std::size_t>(v);
  fb::Buffers::device<double> pairs(vv * static_cast<std::size_t>(o), res_->device());
  constexpr double kOne = 1.0;
  constexpr double kZero = 0.0;
  ASSERT_EQ((wwr::gemmStridedBatched<double, std::int32_t>(
                res_->blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, v, v, g_.naux, &kOne,
                ov.data(), ld, g_.naux, ov.data(), ld, g_.naux, &kZero, pairs.data(), v,
                static_cast<std::int64_t>(vv), o)),
            wwr::WWRBLAS_STATUS_SUCCESS);
  const std::vector<double> got = download(*res_, pairs.data(), pairs.num_elements());

  // (ia|ib) out of PySCF's ovov, (i, a, j, b) at i + o (a + v (j + o b)).
  const auto on = static_cast<std::size_t>(o);
  const auto vn = static_cast<std::size_t>(v);
  const std::vector<double> &ovov = g_.blocks.at("ovov");
  std::vector<double> want(got.size());
  for (std::size_t i = 0; i < on; ++i) {
    for (std::size_t b = 0; b < vn; ++b) {
      for (std::size_t a = 0; a < vn; ++a) {
        want[a + vn * (b + vn * i)] = ovov[i + on * (a + vn * (i + on * b))];
      }
    }
  }
  EXPECT_LE(max_diff(got, want), kRelTol * max_abs(ovov));
}

// With C the identity over every column, b_mo is B itself: [Q][p][q] over
// p = n and q = m is B's [Q][n][m], every element a product with 1 and sums
// with exact zeros.
TEST_F(DfMo, IdentityOrbitalsGiveBackB) {
  const auto nao = static_cast<std::size_t>(g_.nao);
  std::vector<double> identity(nao * nao, 0.0);
  for (std::size_t i = 0; i < nao; ++i) {
    identity[i + nao * i] = 1.0;
  }
  upload(*res_, identity, c_->data());
  const fb::OrbitalRange all{.first = 0, .count = g_.nao};
  const std::vector<double> got = mo(all, all);
  const std::vector<double> want = download(*res_, b_->data(), b_->num_elements());
  EXPECT_EQ(max_diff(got, want), 0.0);
}

// An empty range writes nothing; a range past C's columns, or negative, is
// InvalidConfig, for both calls.
TEST_F(DfMo, EmptyRangesWriteNothingAndOutOfRangeIsRefused) {
  constexpr double kSentinel = 42.0;
  const std::vector<double> sentinel(16, kSentinel);
  fb::Buffers::device<double> out(sentinel.size(), res_->device());
  upload(*res_, sentinel, out.data());
  const fb::OrbitalRange none{.first = 3, .count = 0};
  const fb::OrbitalRange full_end{.first = g_.nao, .count = 0};
  for (const auto &[rows, cols] :
       {std::pair{none, occupied()}, std::pair{occupied(), none}, std::pair{full_end, full_end}}) {
    const fb::Status built =
        fb::b_mo(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), rows, cols, out.data());
    EXPECT_TRUE(built) << built.error().message;
  }
  const fb::Status half =
      fb::b_half(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), none, out.data());
  EXPECT_TRUE(half) << half.error().message;
  EXPECT_EQ(download(*res_, out.data(), sentinel.size()), sentinel);

  for (const fb::OrbitalRange bad :
       {fb::OrbitalRange{.first = g_.nao - 2, .count = 3},
        fb::OrbitalRange{.first = -1, .count = 2}, fb::OrbitalRange{.first = 0, .count = -1}}) {
    const fb::Status as_rows = fb::b_mo(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), bad,
                                        occupied(), out.data());
    ASSERT_FALSE(as_rows);
    EXPECT_EQ(as_rows.error().kind, fb::ErrorKind::InvalidConfig) << as_rows.error().message;
    const fb::Status as_cols = fb::b_mo(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(),
                                        occupied(), bad, out.data());
    ASSERT_FALSE(as_cols);
    EXPECT_EQ(as_cols.error().kind, fb::ErrorKind::InvalidConfig) << as_cols.error().message;
    const fb::Status as_half =
        fb::b_half(*res_, g_.naux, g_.nao, g_.nao, b_->data(), c_->data(), bad, out.data());
    ASSERT_FALSE(as_half);
    EXPECT_EQ(as_half.error().kind, fb::ErrorKind::InvalidConfig) << as_half.error().message;
  }
  EXPECT_EQ(download(*res_, out.data(), sentinel.size()), sentinel);
}

// X past the device's free memory is Unsupported, naming its bytes, before
// anything reads B or C: 2^16 fitting functions, 2^14 rows and 2^16 AOs are
// 2^46 doubles, 512 TiB.
TEST(DfMoRefusals, HalfTransformPastFreeMemoryIsUnsupported) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  constexpr std::int32_t kNaux = 1 << 16;
  constexpr std::int32_t kNao = 1 << 16;
  constexpr std::int32_t kRows = 1 << 14;
  const fb::Status built =
      fb::b_mo(res, kNaux, kNao, kNao, nullptr, nullptr, {.first = 0, .count = kRows},
               {.first = 0, .count = 1}, nullptr);
  ASSERT_FALSE(built);
  EXPECT_EQ(built.error().kind, fb::ErrorKind::Unsupported);
  const std::string bytes = std::to_string(sizeof(double) * (std::size_t{1} << 46));
  EXPECT_NE(built.error().message.find(bytes), std::string::npos) << built.error().message;
}

// naux |rows| past int32_t is InvalidConfig, as is a negative count.
TEST(DfMoRefusals, RowsPastInt32AndNegativeCountsAreInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  constexpr std::int32_t kNaux = 1 << 16;
  constexpr std::int32_t kNmo = 1 << 16;
  const fb::Status wide =
      fb::b_half(res, kNaux, 1, kNmo, nullptr, nullptr, {.first = 0, .count = kNmo}, nullptr);
  ASSERT_FALSE(wide);
  EXPECT_EQ(wide.error().kind, fb::ErrorKind::InvalidConfig);
  const fb::Status negative = fb::b_mo(res, -1, 1, 1, nullptr, nullptr, {.first = 0, .count = 1},
                                       {.first = 0, .count = 1}, nullptr);
  ASSERT_FALSE(negative);
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
