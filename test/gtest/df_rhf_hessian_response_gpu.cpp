// GPU cases for hessian::df_rhf_response, the response part of the DF-RHF
// nuclear Hessian: against pyscf.df.hessian.rhf's total less
// partial_hess_elec and hess_nuc over the same fitting set (data/rhf_hessian/,
// tools/emit/rhf_hessian_reference.py), for the skew water and a skew
// formaldehyde in cc-pVDZ / cc-pVDZ-JKFIT; its U^x against PySCF's CPHF
// solutions; its symmetry; the AO form (rhf_hessian.tex's hess-ao, over W^y)
// built here from the same U^x, whose asymmetry is the CPHF residual's, against
// it; and the refusals.

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
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar against PySCF, absolute (max |H| is 0.4 and 1.0). Both CPHFs
// stop near 1e-10 in the residual, which leaves the Hessian good to about 1e-9.
constexpr double kHessianTol = 1e-7;
// U against PySCF's, absolute (as response_cphf_gpu.cpp holds it).
constexpr double kSolutionTol = 1e-8;
// The symmetric form's asymmetry, of max |H|: rounding only, its one
// first-order asymmetric term being symmetrized.
constexpr double kSymmetryRelTol = 1e-14;
// The AO form against the symmetric one, and its own asymmetry: both are the
// CPHF residual's, ||r|| <= 1e-10 over M's smallest eigenvalue (~0.3), times
// |B| ~ 1.
constexpr double kAoFormTol = 1e-8;
constexpr double kCphfTolerance = 1e-10;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "rhf_hessian";

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

// Each spherical function's norm N_i: function i here is N_i times PySCF's
// unit-normalised one (test/gtest/data/README.md, "Normalisation").
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> download(fb::DeviceResources &res, const double *device,
                             const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
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

// max |A - A^T| of an n x n matrix.
double asymmetry(const std::vector<double> &a, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t y = 0; y < n; ++y) {
    for (std::size_t x = 0; x < n; ++x) {
      worst = std::max(worst, std::abs(a[x + (n * y)] - a[y + (n * x)]));
    }
  }
  return worst;
}

// C = A B over column-major m x k A and k x n B, A transposed when asked: the
// host oracle's products, nao at most 38 here.
std::vector<double> product(const std::vector<double> &a, const std::vector<double> &b,
                            const std::size_t m, const std::size_t n, const std::size_t k,
                            const bool transpose_a = false, const bool transpose_b = false) {
  std::vector<double> c(m * n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t p = 0; p < k; ++p) {
      const double bv = transpose_b ? b[col + (n * p)] : b[p + (k * col)];
      for (std::size_t row = 0; row < m; ++row) {
        c[row + (m * col)] += (transpose_a ? a[p + (k * row)] : a[row + (m * p)]) * bv;
      }
    }
  }
  return c;
}

fb::Molecule molecule_of(fb::DeviceResources &res, const fb::json &c) {
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  return fb::Molecule(res, std::move(atoms));
}

// One case set up as the DF steps would: the bases placed, the fit B and its
// L, PySCF's orbitals made ours (rows over N), and the reference.
class HessianDfRhfResponse : public ::testing::Test {
protected:
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "rhf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    const fb::json *c = nullptr;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        c = &entry;
      }
    }
    ASSERT_NE(c, nullptr) << "no case " << name;
    res_.emplace(0);
    molecule_.emplace(molecule_of(*res_, *c));
    const auto stamp = fb::resolve_basis((*c)["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    const auto aux_stamp = fb::resolve_aux_basis((*c)["aux_basis"].get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(aux) << aux.error().message;
    aux_.emplace(std::move(*aux));
    nao_ = (*c)["nao"].get<std::size_t>();
    naux_ = (*c)["naux"].get<std::size_t>();
    nocc_ = (*c)["nocc"].get<std::int32_t>();
    ASSERT_EQ(basis_->nao_sphe(), nao_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    n_x_ = 3 * basis_->natoms();
    energies_ = (*c)["mo_energies"].get<std::vector<double>>();

    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto u = fb::upload_spherical_transform_matrix(*res_, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(*res_, *aux_, std::move(*u));
    b_.emplace(naux_ * nao_ * nao_, res_->device());
    ASSERT_TRUE(fb::metric(*res_, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(*res_, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(*res_, static_cast<std::int32_t>(nao_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));

    const std::vector<double> n_ao = norms(*basis_);
    c_host_ = read_array((*c)["c"].get<std::string>(), {nao_, nao_});
    for (std::size_t j = 0; j < nao_; ++j) {
      for (std::size_t m = 0; m < nao_; ++m) {
        c_host_[m + (nao_ * j)] /= n_ao[m];
      }
    }
    c_.emplace(upload(*res_, c_host_));
    const std::size_t pairs = (nao_ - static_cast<std::size_t>(nocc_)) * nocc_;
    ref_hessian_ = read_array((*c)["response"].get<std::string>(), {n_x_, n_x_});
    ref_u_ = read_array((*c)["u"].get<std::string>(), {pairs, n_x_});
  }

  fb::Result<fb::hessian::RhfResponse> run() {
    return fb::hessian::df_rhf_response(*res_, *molecule_, *basis_, classes_, *aux_, *b_,
                                        metric_->spherical().data(), *c_, energies_, nocc_,
                                        kCphfTolerance);
  }

  // The response part in the AO form, E^xy = tr(D^y F^(x)) - tr(W^y S^x),
  // W^y = 1/2 (D^y F D + D F^y D + D F D^y), F^y = F^(y) + G(D^y),
  // D^y = 2 (X + X^T) - 2 C_o S^y_oo C_o^T with X = C_v U^y C_o^T: on the host
  // over the tree's S^x and F^(x), its overlap and U, with G(D^y) built on the
  // device by the same DF J and K. Not symmetrized.
  std::vector<double> ao_form(const std::vector<double> &u_all) {
    fb::DeviceResources &res = *res_;
    const std::size_t nao = nao_;
    const std::size_t nn = nao * nao;
    const auto occ = static_cast<std::size_t>(nocc_);
    const std::size_t nvir = nao - occ;
    const auto n32 = static_cast<std::int32_t>(nao);

    auto sx_d = fb::overlap_derivative(res, *basis_, classes_);
    auto tx_d = fb::kinetic_derivative(res, *basis_, classes_);
    auto vx_d = fb::potential_derivative(res, *basis_, classes_, *molecule_);
    EXPECT_TRUE(sx_d && tx_d && vx_d);
    if (!sx_d || !tx_d || !vx_d) {
      return {};
    }
    fb::Buffers::device<double> jx(n_x_ * nn, res.device());
    fb::Buffers::device<double> kx(n_x_ * nn, res.device());
    EXPECT_TRUE(fb::df_jk_derivative(res, *basis_, classes_, *aux_, b_->data(),
                                     metric_->spherical().data(), c_->data(), nocc_, jx.data(),
                                     kx.data(), nullptr));
    const std::vector<double> sx = download(res, sx_d->data(), n_x_ * nn);
    const std::vector<double> tx = download(res, tx_d->data(), n_x_ * nn);
    const std::vector<double> vx = download(res, vx_d->data(), n_x_ * nn);
    const std::vector<double> jxh = download(res, jx.data(), n_x_ * nn);
    const std::vector<double> kxh = download(res, kx.data(), n_x_ * nn);
    std::vector<double> fx(n_x_ * nn);
    for (std::size_t i = 0; i < fx.size(); ++i) {
      fx[i] = tx[i] + vx[i] + jxh[i] - (0.5 * kxh[i]);
    }

    // S, then F = S C eps C^T S and D.
    auto su = fb::upload_spherical_transform_matrix(res, *basis_);
    EXPECT_TRUE(su);
    fb::MatrixAOPair s_pair(res, *basis_, *su);
    EXPECT_TRUE(fb::overlap(res, classes_, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(res));
    const std::vector<double> s = download(res, s_pair.spherical().data(), nn);
    std::vector<double> c_eps = c_host_;
    for (std::size_t p = 0; p < nao; ++p) {
      for (std::size_t m = 0; m < nao; ++m) {
        c_eps[m + (nao * p)] *= energies_[p];
      }
    }
    const std::vector<double> sc = product(s, c_host_, nao, nao, nao);
    const std::vector<double> sceps = product(s, c_eps, nao, nao, nao);
    const std::vector<double> fock = product(sceps, sc, nao, nao, nao, false, true);
    const std::vector<double> c_o(c_host_.begin(),
                                  c_host_.begin() + static_cast<std::ptrdiff_t>(nao * occ));
    const std::vector<double> c_v(c_host_.begin() + static_cast<std::ptrdiff_t>(nao * occ),
                                  c_host_.end());
    std::vector<double> d = product(c_o, c_o, nao, nao, occ, false, true);
    std::ranges::for_each(d, [](double &v) { v *= 2.0; });
    const std::vector<double> fd = product(fock, d, nao, nao, nao);

    std::vector<double> h(n_x_ * n_x_, 0.0);
    fb::Buffers::device<double> dens_d(nn, res.device());
    fb::Buffers::device<double> j_d(nn, res.device());
    fb::Buffers::device<double> k_d(nn, res.device());
    for (std::size_t y = 0; y < n_x_; ++y) {
      const std::vector<double> u(u_all.begin() + static_cast<std::ptrdiff_t>(y * nvir * occ),
                                  u_all.begin() +
                                      static_cast<std::ptrdiff_t>((y + 1) * nvir * occ));
      const std::vector<double> sy(sx.begin() + static_cast<std::ptrdiff_t>(y * nn),
                                   sx.begin() + static_cast<std::ptrdiff_t>((y + 1) * nn));
      const std::vector<double> t = product(c_v, u, nao, occ, nvir); // C_v U
      const std::vector<double> s_oo =
          product(c_o, product(sy, c_o, nao, occ, nao), occ, occ, nao, true); // C_o^T S^y C_o
      const std::vector<double> r = product(c_o, s_oo, nao, occ, occ);        // C_o S_oo
      // D^y / 2 = L R^T, L = [T, C_o, C_o], R = [C_o, T, -C_o S_oo].
      std::vector<double> left;
      std::vector<double> right;
      left.insert(left.end(), t.begin(), t.end());
      left.insert(left.end(), c_o.begin(), c_o.end());
      left.insert(left.end(), c_o.begin(), c_o.end());
      right.insert(right.end(), c_o.begin(), c_o.end());
      right.insert(right.end(), t.begin(), t.end());
      for (const double v : r) {
        right.push_back(-v);
      }
      std::vector<double> dy = product(left, right, nao, nao, 3 * occ, false, true);
      std::ranges::for_each(dy, [](double &v) { v *= 2.0; });
      const fb::Buffers::device<double> left_d = upload(res, left);
      const fb::Buffers::device<double> right_d = upload(res, right);
      EXPECT_EQ(wwr::wwrMemcpyAsync(dens_d.data(), dy.data(), sizeof(double) * nn,
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                wwr::wwrSuccess);
      EXPECT_TRUE(fb::df_coulomb(res, n32, static_cast<std::int32_t>(naux_), b_->data(),
                                 dens_d.data(), j_d.data()));
      EXPECT_TRUE(fb::df_exchange(res, n32, static_cast<std::int32_t>(naux_),
                                  static_cast<std::int32_t>(3 * occ), b_->data(), left_d.data(),
                                  right_d.data(), k_d.data()));
      const std::vector<double> jy = download(res, j_d.data(), nn);
      const std::vector<double> ky = download(res, k_d.data(), nn);
      std::vector<double> fy(nn);
      for (std::size_t i = 0; i < nn; ++i) {
        fy[i] = fx[(y * nn) + i] + jy[i] - ky[i];
      }
      // W^y = 1/2 (D^y F D + D F^y D + D F D^y).
      const std::vector<double> a = product(dy, fd, nao, nao, nao);
      const std::vector<double> bm = product(d, product(fy, d, nao, nao, nao), nao, nao, nao);
      const std::vector<double> cm = product(product(d, fock, nao, nao, nao), dy, nao, nao, nao);
      for (std::size_t x = 0; x < n_x_; ++x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < nn; ++i) {
          const double wy = 0.5 * (a[i] + bm[i] + cm[i]);
          sum += (dy[i] * fx[(x * nn) + i]) - (wy * sx[(x * nn) + i]);
        }
        h[x + (n_x_ * y)] = sum;
      }
    }
    return h;
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    const auto start = std::chrono::steady_clock::now();
    auto got = run();
    ASSERT_TRUE(got) << got.error().message;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_EQ(got->hessian.size(), n_x_ * n_x_);
    ASSERT_EQ(got->cphf.residuals.size(), n_x_);
    for (std::size_t x = 0; x < n_x_; ++x) {
      EXPECT_LE(got->cphf.residuals[x], 10 * kCphfTolerance) << "column " << x;
    }
    const double scale = max_abs(ref_hessian_);
    const double worst = max_diff(got->hessian, ref_hessian_);
    EXPECT_LE(worst, kHessianTol) << "max |H - PySCF| " << worst << " of max " << scale;
    const double asym = asymmetry(got->hessian, n_x_);
    EXPECT_LE(asym, kSymmetryRelTol * scale);
    const std::vector<double> u = download(*res_, got->cphf.solution.data(), ref_u_.size());
    const double u_worst = max_diff(u, ref_u_);
    EXPECT_LE(u_worst, kSolutionTol) << "max |U - PySCF| " << u_worst;

    const std::vector<double> ao = ao_form(u);
    ASSERT_EQ(ao.size(), n_x_ * n_x_);
    const double ao_asym = asymmetry(ao, n_x_);
    const double ao_worst = max_diff(ao, got->hessian);
    EXPECT_LE(ao_asym, kAoFormTol);
    EXPECT_LE(ao_worst, kAoFormTol);
    std::cout << std::format(
        "[ hess     ] {}: {} coordinates, {} sweeps, {:.2f} s; max |H| {:.4f}, |H - PySCF| "
        "{:.2e}, asym {:.2e}, |U - PySCF| {:.2e}; AO form |H_ao - H| {:.2e}, asym {:.2e}\n",
        name, n_x_, got->cphf.iterations, elapsed, scale, worst, asym, u_worst, ao_worst, ao_asym);
  }

  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
  std::vector<double> c_host_;
  std::vector<double> energies_;
  std::vector<double> ref_hessian_;
  std::vector<double> ref_u_;
  std::size_t nao_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t nocc_ = 0;
};

TEST_F(HessianDfRhfResponse, WaterMatchesPyscf) {
  check("h2o_ccpvdz");
}

TEST_F(HessianDfRhfResponse, FormaldehydeMatchesPyscf) {
  check("h2co_ccpvdz");
}

TEST_F(HessianDfRhfResponse, Refusals) {
  FBF_REQUIRE_GPU();
  load("h2o_ccpvdz");
  ASSERT_FALSE(HasFatalFailure());
  const auto expect_refused = [](const fb::Result<fb::hessian::RhfResponse> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_rhf_response"), std::string::npos) << what;
  };
  const std::span<const double> energies{energies_};
  const double *l = metric_->spherical().data();
  expect_refused(fb::hessian::df_rhf_response(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l,
                                              *c_, energies, 0),
                 "nocc 0");
  expect_refused(fb::hessian::df_rhf_response(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l,
                                              *c_, energies, static_cast<std::int32_t>(nao_)),
                 "no virtuals");
  expect_refused(fb::hessian::df_rhf_response(*res_, *molecule_, *basis_, classes_, *aux_, *b_, l,
                                              *c_, energies.first(energies.size() - 1), nocc_),
                 "C not nao nmo");
  const fb::Buffers::device<double> short_b(b_->num_elements() - 1, res_->device());
  expect_refused(fb::hessian::df_rhf_response(*res_, *molecule_, *basis_, classes_, *aux_, short_b,
                                              l, *c_, energies, nocc_),
                 "B not naux nao^2");
  const fb::Molecule two(*res_, {molecule_->atoms()[0], molecule_->atoms()[1]});
  expect_refused(fb::hessian::df_rhf_response(*res_, two, *basis_, classes_, *aux_, *b_, l, *c_,
                                              energies, nocc_),
                 "another molecule");
}

} // namespace
