// GPU cases for integrals.eri.density_fitting.jk's df_exchange_symmetrized:
// K of the symmetric parts of densities L R_d^T sharing one left factor, at
// rank n_occ, against the block two-factor df_exchange of the symmetrized
// density (L_d = [R_d L], R_d = [L R_d], rank 2 n_occ), which is twice it.
// On random occupied-block factors over data/density_fitting/'s h2o_ccpvdz B,
// and on CP-CASSCF response densities (Wang et al.'s D_CV and D_Gen) built
// from data/casscf_hessian/'s h2o_ccpvdz orbitals, CI vector and lambda^x,
// over that case's own fit; the exact symmetry of each K_d; and the empty,
// negative and too-large counts.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

// The issue's bar: of the largest |K| of the block. The same contractions
// summed in other orders, so rounding.
constexpr double kSymmetrizedTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR};

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
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

std::vector<double> read_array(const std::filesystem::path &file,
                               const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
  return a->data;
}

// count elements uniform on [-1/2, 1/2) from a fixed seed.
std::vector<double> random_values(const std::size_t count, const std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  std::vector<double> a(count);
  std::ranges::generate(a, [&] { return uniform(engine); });
  return a;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_jk_derivative_gpu.cpp).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : shell.primitives()) {
    for (const fb::Primitive &q : shell.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

// K_d by df_exchange_symmetrized of (left, R_d), and twice it by the block
// two-factor df_exchange over [R_d left], [left R_d]: max |2 K_sym - K_595| of
// max |K_595|, and max |K_d - K_d^T| over the block (which must be 0).
void expect_symmetrized_matches(fb::DeviceResources &res, const std::int32_t nao,
                                const std::int32_t naux, const double *b,
                                const std::vector<double> &left, const std::vector<double> &right,
                                const std::int32_t rank, const std::string &what) {
  const auto n = static_cast<std::size_t>(nao);
  const std::size_t factor = n * static_cast<std::size_t>(rank);
  ASSERT_EQ(left.size(), factor);
  const std::size_t count = right.size() / factor;
  const std::size_t nn = n * n;
  // [R_d L] and [L R_d] side by side, rank 2 rank each.
  std::vector<double> doubled_left(2 * right.size());
  std::vector<double> doubled_right(2 * right.size());
  for (std::size_t d = 0; d < count; ++d) {
    const auto rd = right.begin() + static_cast<std::ptrdiff_t>(d * factor);
    const auto ld = doubled_left.begin() + static_cast<std::ptrdiff_t>(2 * d * factor);
    const auto sd = doubled_right.begin() + static_cast<std::ptrdiff_t>(2 * d * factor);
    std::copy_n(rd, factor, ld);
    std::ranges::copy(left, ld + static_cast<std::ptrdiff_t>(factor));
    std::ranges::copy(left, sd);
    std::copy_n(rd, factor, sd + static_cast<std::ptrdiff_t>(factor));
  }
  fb::Buffers::device<double> d_left(left.size(), res.device());
  fb::Buffers::device<double> d_right(right.size(), res.device());
  fb::Buffers::device<double> d_doubled_left(doubled_left.size(), res.device());
  fb::Buffers::device<double> d_doubled_right(doubled_right.size(), res.device());
  fb::Buffers::device<double> k(count * nn, res.device());
  upload(res, left, d_left.data());
  upload(res, right, d_right.data());
  upload(res, doubled_left, d_doubled_left.data());
  upload(res, doubled_right, d_doubled_right.data());

  const auto cols = static_cast<std::int32_t>(count);
  const fb::Status reference = fb::df_exchange(
      res, nao, naux, 2 * rank, cols, b, d_doubled_left.data(), d_doubled_right.data(), k.data());
  ASSERT_TRUE(reference) << reference.error().message;
  const std::vector<double> k595 = download(res, k.data(), k.num_elements());
  const fb::Status symmetrized = fb::df_exchange_symmetrized(
      res, nao, naux, rank, cols, b, d_left.data(), d_right.data(), k.data());
  ASSERT_TRUE(symmetrized) << symmetrized.error().message;
  const std::vector<double> ksym = download(res, k.data(), k.num_elements());

  double worst = 0.0;
  double asymmetry = 0.0;
  for (std::size_t d = 0; d < count; ++d) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < n; ++p) {
        const std::size_t pq = p + (n * q) + (nn * d);
        worst = std::max(worst, std::abs((2.0 * ksym[pq]) - k595[pq]));
        asymmetry = std::max(asymmetry, std::abs(ksym[pq] - ksym[q + (n * p) + (nn * d)]));
      }
    }
  }
  const double scale = max_abs(k595);
  ASSERT_GT(scale, 0.0) << what;
  EXPECT_LE(worst, kSymmetrizedTol * scale)
      << what << ": max|2 K_sym - K[L R^T + R L^T]| = " << worst << ", " << worst / scale
      << " of max " << scale;
  EXPECT_EQ(asymmetry, 0.0) << what;
  std::cout << std::format("[ df jk    ] {}: max|2 K_sym - K_595| / max {:.3e}\n", what,
                           worst / scale);
}

// data/density_fitting/'s h2o_ccpvdz B, fitted, on the device.
class DfJkSymmetrized : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    const std::filesystem::path dir = kDataDir / "density_fitting";
    const auto doc = fb::read_json(dir / "density_fitting.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const auto &c : (*doc)["integrals"]) {
      if (c["name"].get<std::string>() != "h2o_ccpvdz") {
        continue;
      }
      nao_ = c["nao"].get<std::int32_t>();
      naux_ = c["naux"].get<std::int32_t>();
      occupied_ = c["jk"]["occupied"].get<std::int32_t>();
      const auto nao = static_cast<std::size_t>(nao_);
      const auto naux = static_cast<std::size_t>(naux_);
      const auto metric = fb::read_npy(dir / c["int2c2e"].get<std::string>());
      ASSERT_TRUE(metric);
      // Each piece is (nao, nao, end - begin), Fortran order; into [P][n][m].
      std::vector<double> b(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
      for (const auto &piece : c["int3c2e"]) {
        const auto range = piece["aux"].get<std::vector<std::size_t>>();
        const auto slab = fb::read_npy(dir / piece["file"].get<std::string>());
        ASSERT_TRUE(slab);
        for (std::size_t p = range[0]; p < range[1]; ++p) {
          for (std::size_t n = 0; n < nao; ++n) {
            for (std::size_t m = 0; m < nao; ++m) {
              b[p + naux * (n + nao * m)] = slab->data[m + nao * (n + nao * (p - range[0]))];
            }
          }
        }
      }
      res_.emplace(0);
      fb::Buffers::device<double> d_metric(metric->data.size(), res_->device());
      b_.emplace(b.size(), res_->device());
      upload(*res_, metric->data, d_metric.data());
      upload(*res_, b, b_->data());
      const fb::Status fitted = fb::density_fit(*res_, nao_, naux_, d_metric.data(), b_->data());
      ASSERT_TRUE(fitted) << fitted.error().message;
      return;
    }
    FAIL() << "no integrals case h2o_ccpvdz";
  }

  std::int32_t nao_ = 0;
  std::int32_t naux_ = 0;
  std::int32_t occupied_ = 0;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Buffers::device<double>> b_;
};

// Nine random R_d against one random L, rank n_occ: D_d = L R_d^T has nonzero
// rows only in L's span, as an occupied-block density does.
TEST_F(DfJkSymmetrized, RandomOccupiedBlockFactorsMatchTheSymmetrizedDensity) {
  constexpr std::size_t kCount = 9;
  const std::size_t factor = static_cast<std::size_t>(nao_) * static_cast<std::size_t>(occupied_);
  expect_symmetrized_matches(*res_, nao_, naux_, b_->data(), random_values(factor, 612),
                             random_values(kCount * factor, 613), occupied_, "9 random L R_d^T");
}

// One density, and a rank past n_occ, the same identity.
TEST_F(DfJkSymmetrized, OneDensityAndAWiderRank) {
  const std::size_t factor = static_cast<std::size_t>(nao_) * 7;
  expect_symmetrized_matches(*res_, nao_, naux_, b_->data(), random_values(factor, 614),
                             random_values(factor, 615), 7, "1 random L R^T, rank 7");
}

// Wang et al.'s CP-CASSCF densities (their eqs. 14 and 15) at each lambda^x of
// the casscf_hessian h2o_ccpvdz case, p_pq = K_qp for p in the lower space:
// D_CV, the core-virtual block p_cv alone (rank n_core); and D_Gen, rows
// core [0, p_ca (2 - gamma), 2 p_cv] and active [0, 0, gamma p_av] (rank
// n_core + n_act). In the AO basis D = C_o (C D_o.^T)^T: L = C_o, R = C D_o.^T.
TEST(DfJkSymmetrizedCasscf, ResponseDensitiesMatchTheSymmetrizedDensity) {
  FBF_REQUIRE_GPU();
  const std::filesystem::path dir = kDataDir / "casscf_hessian";
  const auto doc = fb::read_json(dir / "casscf_hessian.json");
  ASSERT_TRUE(doc) << doc.error().message;
  fb::json case_;
  for (const fb::json &entry : (*doc)["cases"]) {
    if (entry["name"].get<std::string>() == "h2o_ccpvdz") {
      case_ = entry;
    }
  }
  ASSERT_FALSE(case_.is_null()) << "no case h2o_ccpvdz";
  fb::DeviceResources res{0};
  std::vector<fb::Atom> atoms;
  for (const auto &atom : case_["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
  }
  fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_basis(case_["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis(case_["aux_basis"].get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  auto aux = fb::place(aux_stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;
  const auto nao = case_["nao"].get<std::size_t>();
  const auto naux = case_["naux"].get<std::size_t>();
  ASSERT_EQ(basis->nao_sphe(), nao);
  ASSERT_EQ(case_["nmo"].get<std::size_t>(), nao);
  ASSERT_EQ(aux->nao_sphe(), naux);
  const std::size_t n_x = 3 * basis->natoms();
  const auto ncore = case_["ncore"].get<std::size_t>();
  const auto ncas = case_["ncas"].get<std::size_t>();
  const std::size_t nocc = ncore + ncas;
  const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
  const std::size_t n_det = case_["na"].get<std::size_t>() * case_["nb"].get<std::size_t>();

  // The fit B over this case's bases.
  auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  auto u = fb::upload_spherical_transform_matrix(res, *aux);
  ASSERT_TRUE(u) << u.error().message;
  fb::MatrixAOPair metric(res, *aux, std::move(*u));
  fb::Buffers::device<double> b(naux * nao * nao, res.device());
  ASSERT_TRUE(fb::metric(res, *aux, metric));
  ASSERT_TRUE(fb::three_center(res, *basis, *classes, *aux, b.data()));
  ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nao), static_cast<std::int32_t>(naux),
                              metric.spherical().data(), b.data()));

  // The fixture's C in this tree's AO order and normalization: PySCF orders
  // each atom's shells by l (df_casscf_hessian_gpu.cpp has the reason).
  std::vector<double> norms;
  for (const fb::Shell &shell : basis->shells()) {
    norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  std::vector<std::size_t> order;
  for (std::uint32_t atom = 0; atom < basis->natoms(); ++atom) {
    std::vector<std::uint32_t> shells(basis->atom_shell_count(atom));
    std::iota(shells.begin(), shells.end(), basis->atom_shell_offset(atom));
    std::ranges::stable_sort(
        shells, {}, [&](const std::uint32_t i) { return basis->shells()[i].angular_momentum(); });
    for (const std::uint32_t i : shells) {
      const std::size_t first = basis->ao_offset_sphe(i);
      for (std::size_t f = 0; f < basis->shells()[i].function_count_sphe(); ++f) {
        order.push_back(first + f);
      }
    }
  }
  ASSERT_EQ(order.size(), nao);
  const std::vector<double> c_pyscf =
      read_array(dir / case_["arrays"]["c"].get<std::string>(), {nao, nao});
  std::vector<double> c(nao * nao);
  for (std::size_t j = 0; j < nao; ++j) {
    for (std::size_t k = 0; k < nao; ++k) {
      c[order[k] + (nao * j)] = c_pyscf[k + (nao * j)] / norms[order[k]];
    }
  }
  const std::vector<double> ci =
      read_array(dir / case_["arrays"]["ci"].get<std::string>(), {n_det, 1});
  const std::vector<double> lambda =
      read_array(dir / case_["arrays"]["lambda_orb"].get<std::string>(), {nao, nao, n_x});
  ASSERT_FALSE(::testing::Test::HasFailure());

  // gamma by ci::Rdm, row-major n_act x n_act (symmetric, so either order).
  const auto n_act = static_cast<std::int32_t>(ncas);
  auto alpha = fb::ci::DeviceStringSet::create(res, n_act, nelecas[0]);
  ASSERT_TRUE(alpha) << alpha.error().message;
  auto beta = fb::ci::DeviceStringSet::create(res, n_act, nelecas[1]);
  ASSERT_TRUE(beta) << beta.error().message;
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  ASSERT_TRUE(rdm) << rdm.error().message;
  fb::Buffers::device<double> d_ci(n_det, res.device());
  fb::Buffers::device<double> rdms((ncas * ncas) + (ncas * ncas * ncas * ncas), res.device());
  upload(res, ci, d_ci.data());
  ASSERT_TRUE(rdm->rdm12(res, std::vector<double>{1.0}, d_ci.data(), rdms.data(),
                         rdms.data() + (ncas * ncas))
                  .has_value());
  const std::vector<double> gamma = download(res, rdms.data(), ncas * ncas);

  // D_o. per x, nocc x nao column-major (rows the occupied MOs), then
  // R_x = C D_o.^T, nao x nocc; D_CV's is its first ncore columns' worth.
  std::vector<double> right_gen(n_x * nao * nocc, 0.0);
  std::vector<double> right_cv(n_x * nao * ncore, 0.0);
  for (std::size_t x = 0; x < n_x; ++x) {
    const auto kappa = [&](const std::size_t p, const std::size_t q) {
      return lambda[p + (nao * q) + (nao * nao * x)];
    };
    std::vector<double> gen(nocc * nao, 0.0);
    std::vector<double> cv(ncore * nao, 0.0);
    for (std::size_t i = 0; i < ncore; ++i) {
      for (std::size_t t = 0; t < ncas; ++t) {
        double sum = 0.0; // (p_ca (2 - gamma))_it
        for (std::size_t s = 0; s < ncas; ++s) {
          sum += kappa(ncore + s, i) * (((s == t) ? 2.0 : 0.0) - gamma[(s * ncas) + t]);
        }
        gen[i + (nocc * (ncore + t))] = sum;
      }
      for (std::size_t a = nocc; a < nao; ++a) {
        gen[i + (nocc * a)] = 2.0 * kappa(a, i);
        cv[i + (ncore * a)] = kappa(a, i);
      }
    }
    for (std::size_t t = 0; t < ncas; ++t) {
      for (std::size_t a = nocc; a < nao; ++a) {
        double sum = 0.0; // (gamma p_av)_ta
        for (std::size_t s = 0; s < ncas; ++s) {
          sum += gamma[(t * ncas) + s] * kappa(a, ncore + s);
        }
        gen[ncore + t + (nocc * a)] = sum;
      }
    }
    for (std::size_t i = 0; i < nocc; ++i) {
      for (std::size_t mu = 0; mu < nao; ++mu) {
        double g = 0.0;
        double v = 0.0;
        for (std::size_t p = 0; p < nao; ++p) {
          g += c[mu + (nao * p)] * gen[i + (nocc * p)];
          if (i < ncore) {
            v += c[mu + (nao * p)] * cv[i + (ncore * p)];
          }
        }
        right_gen[mu + (nao * (i + (nocc * x)))] = g;
        if (i < ncore) {
          right_cv[mu + (nao * (i + (ncore * x)))] = v;
        }
      }
    }
  }
  const std::vector<double> c_core(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(nao * ncore));
  const std::vector<double> c_occ(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(nao * nocc));
  const auto n32 = static_cast<std::int32_t>(nao);
  const auto q32 = static_cast<std::int32_t>(naux);
  expect_symmetrized_matches(res, n32, q32, b.data(), c_core, right_cv,
                             static_cast<std::int32_t>(ncore), "D_CV at 9 lambda^x");
  expect_symmetrized_matches(res, n32, q32, b.data(), c_occ, right_gen,
                             static_cast<std::int32_t>(nocc), "D_Gen at 9 lambda^x");
}

// naux = 0 or rank = 0 is K = 0 over every density, written over whatever was
// there; nao = 0 or count = 0 writes nothing; a negative count, or naux count
// rank or nao^2 past int32_t, is InvalidConfig.
TEST(DfJkSymmetrizedEmpty, EmptyCountsAndNegativeCounts) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kNao = 3;
  constexpr std::int32_t kCount = 2;
  const std::vector<double> ones(kCount * kNao * kNao, 1.0);
  const std::vector<double> zeros(ones.size(), 0.0);
  fb::Buffers::device<double> out(ones.size(), res.device());

  upload(res, ones, out.data());
  ASSERT_TRUE(
      fb::df_exchange_symmetrized(res, kNao, 0, 2, kCount, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), zeros);
  upload(res, ones, out.data());
  ASSERT_TRUE(
      fb::df_exchange_symmetrized(res, kNao, 4, 0, kCount, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), zeros);

  upload(res, ones, out.data());
  ASSERT_TRUE(
      fb::df_exchange_symmetrized(res, kNao, 4, 2, 0, nullptr, nullptr, nullptr, out.data()));
  EXPECT_EQ(download(res, out.data(), out.num_elements()), ones);
  EXPECT_TRUE(
      fb::df_exchange_symmetrized(res, 0, 4, 2, kCount, nullptr, nullptr, nullptr, nullptr));

  constexpr std::int32_t kHuge = 1 << 16; // kHuge^2 = 2^32
  for (const fb::Status refused :
       {fb::df_exchange_symmetrized(res, -1, 4, 2, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange_symmetrized(res, 3, -1, 2, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange_symmetrized(res, 3, 4, -1, kCount, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange_symmetrized(res, 3, 4, 2, -1, nullptr, nullptr, nullptr, nullptr),
        fb::df_exchange_symmetrized(res, 3, kHuge, kHuge / 2, kCount, nullptr, nullptr, nullptr,
                                    nullptr),
        fb::df_exchange_symmetrized(res, kHuge, 4, 2, kCount, nullptr, nullptr, nullptr,
                                    nullptr)}) {
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
