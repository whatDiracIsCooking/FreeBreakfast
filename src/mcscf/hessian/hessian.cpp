/// @file
/// @brief Hessian: gen_g_hop's h_op as gemms over MO intermediates
/// built once and two J and K of transition densities per product, and its
/// h_diag assembled on the host from diagonals read back once.

module free_breakfast.mcscf.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, geam, gemmStridedBatched, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.norm; // norm, Norm
import free_breakfast.recipes.linear_operator;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.mcscf.fock;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

static_assert(LinearOperator<Hessian, double>);

Hessian::Hessian(DeviceResources &res, const BasisSet &basis,
                 const std::span<CoulombPairClass> j_classes,
                 const std::span<OrderedPairClass> k_classes, const int32_t nmo,
                 const int32_t ncore, const int32_t ncas, const double tau_j, const double tau_k)
    : j_classes_(j_classes), k_classes_(k_classes), dev_idx_(res.dev_idx()), nmo_(nmo),
      ncore_(ncore), ncas_(ncas), n_cart_(static_cast<int32_t>(basis.nao_cart())),
      n_pairs_((ncore * (nmo - ncore)) + (ncas * (nmo - ncore - ncas))), tau_j_(tau_j),
      tau_k_(tau_k), c_cart_(static_cast<size_t>(n_cart_) * static_cast<size_t>(nmo), res.device()),
      h_mo_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      fock_sym_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      vhf_c_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      vhf_ca_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      dm1_occ_(static_cast<size_t>(ncore + ncas) * static_cast<size_t>(ncore + ncas), res.device()),
      gamma_(static_cast<size_t>(ncas) * static_cast<size_t>(ncas), res.device()),
      h_act_(static_cast<size_t>(nmo) * static_cast<size_t>(ncas) * static_cast<size_t>(nmo) *
                 static_cast<size_t>(ncas),
             res.device()),
      diagonal_(static_cast<size_t>(n_pairs_), res.device()), density_(res, basis), j_(res, basis),
      k_(res, basis) {}

Result<double> Hessian::capped_density(DeviceResources &res, const double *const from) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t nc = n_cart_;
  // Host pointer mode, so the call has waited for it.
  double top = 0.0;
  FBF_TRY(norm<Norm::inf>(res, nc * nc, from, 1, &top));
  const double s = top > 2.0 ? 2.0 / top : 1.0;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, nc,
                                                &s, from, nc, &kZero<double>, from, nc,
                                                density_.data(), nc)));
  return s;
}

Result<Hessian> Hessian::create(DeviceResources &res, const BasisSet &basis,
                                const SphericalMatrixAO &h,
                                const std::shared_ptr<const Buffers::device<double>> &u,
                                const std::span<CoulombPairClass> j_classes,
                                const std::span<OrderedPairClass> k_classes,
                                const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
                                const Buffers::device<double> &ppaa,
                                const Buffers::device<double> &papa, const Fock &fock,
                                const double tau_j, const double tau_k) {
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t n_orbitals = c.n();
  if (ncore < 0 || ncas < 0) {
    return err_config("mcscf::Hessian::create: ncore ({}) and ncas ({}) must not be negative",
                      ncore, ncas);
  }
  if (ncore + ncas == 0) {
    return err_config("mcscf::Hessian::create: ncore and ncas are both 0, so nothing is occupied");
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, n_orbitals)) {
    return err_config("mcscf::Hessian::create: ncore {} + ncas {} exceeds the {} orbitals", ncore,
                      ncas, n_orbitals);
  }
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("mcscf::Hessian::create: tau_j ({}) and tau_k ({}) must be 0 or above", tau_j,
                      tau_k);
  }
  if (h.n() != n_sphe || n_orbitals != n_sphe) {
    return err_config(
        "mcscf::Hessian::create: h is {} x {} and C {} x {}, but the basis has {} spherical AOs",
        h.n(), h.n(), n_orbitals, n_orbitals, n_sphe);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config(
        "mcscf::Hessian::create: h is on device {} and C on {}, but the call passed device {}",
        h.dev_idx(), c.dev_idx(), res.dev_idx());
  }
  const size_t n_all = n_orbitals;
  const size_t nn = n_all * n_all;
  const auto na = static_cast<size_t>(ncas);
  const auto core = static_cast<size_t>(ncore);
  const size_t occ = core + na;
  if (ppaa.num_elements() != nn * na * na || papa.num_elements() != nn * na * na) {
    return err_config(
        "mcscf::Hessian::create: ppaa holds {} and papa {} doubles, but nmo^2 ncas^2 is {}",
        ppaa.num_elements(), papa.num_elements(), nn * na * na);
  }
  const size_t n_pairs = (core * (n_all - core)) + (na * (n_all - occ));
  if (fock.fock_inactive.num_elements() != nn || fock.fock_active.num_elements() != nn ||
      fock.fock.num_elements() != nn || fock.gradient.num_elements() != n_pairs ||
      fock.gamma.num_elements() != na * na || fock.gamma2.num_elements() != na * na * na * na) {
    return err_config("mcscf::Hessian::create: the Fock is not of {} orbitals, ncore {} and "
                      "ncas {}",
                      n_all, ncore, ncas);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto nmo = static_cast<int32_t>(n_orbitals);
  const int32_t nocc = ncore + ncas;
  Hessian out(res, basis, j_classes, k_classes, nmo, ncore, ncas, tau_j, tau_k);
  const int32_t nc = out.n_cart_;
  const auto n_cart = static_cast<size_t>(nc);
  const auto ns = static_cast<int32_t>(n_sphe);
  constexpr double kMinusOne = -1.0;

  // Scratch: gemmwich's intermediate, one projected nmo x nmo block, and
  // the permuted Gamma of H_act's exchange part. coulomb and exchange, called
  // while it is live, do not take the workspace.
  const auto [work, block, gamma2_exchange] =
      res.device_regions<double>(n_cart * n_all, nn, na * na * na * na);

  // U C, then C^T h C, and the MO matrices the product reads.
  if (u != nullptr) {
    FBF_TRY(gpu_status(gemm(blas, nc, nmo, ns, &kOne<double>, u->data(), c.data(), &kZero<double>,
                            out.c_cart_.data())));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.c_cart_.data(), c.data(), sizeof(double) * nn,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  FBF_TRY(gpu_status(gemmwich(blas, nmo, ns, c.data(), h.data(), work, out.h_mo_.data())));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, &kOne<double>,
      fock.fock_inactive.data(), nmo, &kMinusOne, out.h_mo_.data(), nmo, out.vhf_c_.data(), nmo)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, &kOne<double>, out.vhf_c_.data(), nmo,
      &kOne<double>, fock.fock_active.data(), nmo, out.vhf_ca_.data(), nmo)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo, nmo, &kOne<double>, fock.fock.data(), nmo,
      &kOne<double>, fock.fock.data(), nmo, out.fock_sym_.data(), nmo)));

  // gamma and Gamma back to the host, for the 1-RDM's occupied block, H_act's
  // permuted Gamma and the diagonal; and the core orbitals' columns of U C,
  // for the prune cap of each C_i C_i^T.
  std::vector<double> gamma(na * na);
  std::vector<double> gamma2(na * na * na * na);
  std::vector<double> c_core(n_cart * core);
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.gamma_.data(), fock.gamma.data(),
                                           gamma.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), fock.gamma.data(),
                                           gamma.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2.data(), fock.gamma2.data(),
                                           gamma2.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  }
  if (ncore > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_core.data(), out.c_cart_.data(),
                                           c_core.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  }
  FBF_TRY(res.sync());

  // The 1-RDM over the occupied orbitals: 2 on the core diagonal, gamma on
  // the active block. Live until the final synchronize.
  std::vector<double> dm1(static_cast<size_t>(nocc) * static_cast<size_t>(nocc), 0.0);
  for (size_t i = 0; i < core; ++i) {
    dm1[i * (occ + 1)] = 2.0;
  }
  for (size_t t = 0; t < na; ++t) {
    for (size_t v = 0; v < na; ++v) {
      dm1[core + t + (occ * (core + v))] = gamma[t + (na * v)];
    }
  }
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(out.dm1_occ_.data(), dm1.data(), dm1.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, stream)));

  // H_act[(p, u), (r, v)] = sum_wx (pr|wx) Gamma_wxuv + (pw|rx) Gamma'_wxuv,
  // Gamma'_wxuv = Gamma_uwxv + Gamma_wuxv: gen_g_hop's hdm2. For each v, a
  // gemm per r over ppaa's (or papa's) rows (p, r) of every block (w, x),
  // batched over r: A_r = (p, wx) at nmo r + p + nmo^2 wx, B the Gamma
  // columns (wx, u) of v, and C_r the H_act columns (r, v) over (p, u).
  std::vector<double> gamma2_permuted(gamma2.size());
  for (size_t w = 0; w < na; ++w) {
    for (size_t x = 0; x < na; ++x) {
      for (size_t uu = 0; uu < na; ++uu) {
        for (size_t v = 0; v < na; ++v) {
          const auto at = [na](size_t a, size_t b, size_t cc, size_t d) {
            return a + (na * (b + (na * (cc + (na * d)))));
          };
          gamma2_permuted[at(w, x, uu, v)] = gamma2[at(uu, w, x, v)] + gamma2[at(w, uu, x, v)];
        }
      }
    }
  }
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2_exchange, gamma2_permuted.data(),
                                           gamma2_permuted.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    const auto nmo64 = static_cast<int64_t>(n_all);
    const int32_t pairs = ncas * ncas;
    const auto ld_pairs = static_cast<int32_t>(nn);
    for (size_t v = 0; v < na; ++v) {
      double *const columns = out.h_act_.data() + (n_all * na * n_all * v);
      const std::array<std::pair<const double *, const double *>, 2> terms{
          {{ppaa.data(), fock.gamma2.data()}, {papa.data(), gamma2_exchange}}};
      for (size_t term = 0; term < terms.size(); ++term) {
        const auto [integrals, weights] = terms[term];
        const double *const beta = term == 0 ? &kZero<double> : &kOne<double>;
        FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, ncas, pairs, &kOne<double>, integrals,
            ld_pairs, nmo64, weights + (na * na * na * v), pairs, 0, beta, columns, nmo,
            nmo64 * ncas, nmo)));
      }
    }
  }

  // The diagonal's inputs, read back: diag of C^T h C, F, F^I - C^T h C and
  // F^I + F^A - C^T h C; diag H_act; ppaa's and papa's core rows; and
  // (pp|ii) and (pi|pi) for each core i, of one J and one K of C_i C_i^T.
  const auto diagonal = [&](double *to, const double *matrix, size_t count, size_t ld) {
    return gpu_status(wwr::wwrMemcpy2DAsync(to, sizeof(double), matrix, (ld + 1) * sizeof(double),
                                            sizeof(double), count, wwr::wwrMemcpyDeviceToHost,
                                            stream));
  };
  std::vector<double> h_diag(n_all);
  std::vector<double> f_diag(n_all);
  std::vector<double> vc_diag(n_all);
  std::vector<double> vca_diag(n_all);
  std::vector<double> act_diag(n_all * na);
  std::vector<double> ppaa_core(core * n_all * na * na);
  std::vector<double> papa_core(core * n_all * na * na);
  std::vector<double> j_pc(n_all * core);
  std::vector<double> k_pc(n_all * core);
  FBF_TRY(diagonal(h_diag.data(), out.h_mo_.data(), n_all, n_all));
  FBF_TRY(diagonal(f_diag.data(), fock.fock.data(), n_all, n_all));
  FBF_TRY(diagonal(vc_diag.data(), out.vhf_c_.data(), n_all, n_all));
  FBF_TRY(diagonal(vca_diag.data(), out.vhf_ca_.data(), n_all, n_all));
  if (ncas > 0) {
    FBF_TRY(diagonal(act_diag.data(), out.h_act_.data(), n_all * na, n_all * na));
  }
  if (ncore > 0 && ncas > 0) {
    // Rows [0, ncore) of every column (q, uv) of ppaa and papa.
    for (const auto &[to, from] :
         {std::pair{ppaa_core.data(), ppaa.data()}, std::pair{papa_core.data(), papa.data()}}) {
      FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
          to, core * sizeof(double), from, n_all * sizeof(double), core * sizeof(double),
          n_all * na * na, wwr::wwrMemcpyDeviceToHost, stream)));
    }
  }
  std::vector<double> scale(core, 1.0);
  for (size_t i = 0; i < core; ++i) {
    // |(C_i C_i^T)_cd| <= m_i^2, m_i = max_mu |(U C)_mu i|: the classes prune
    // for |D| <= 2, so the build is of s C_i C_i^T at s tau.
    double m = 0.0;
    for (size_t mu = 0; mu < n_cart; ++mu) {
      m = std::max(m, std::abs(c_core[mu + (n_cart * i)]));
    }
    const double s = m * m > 2.0 ? 2.0 / (m * m) : 1.0;
    scale[i] = s;
    const double *const c_i = out.c_cart_.data() + (n_cart * i);
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nc, nc, 1, &s, c_i, c_i, &kZero<double>, out.density_.data())));
    FBF_TRY(coulomb(res, j_classes, out.density_, out.j_, tau_j * s));
    FBF_TRY(exchange(res, k_classes, out.density_, out.k_, tau_k * s));
    FBF_TRY(gpu_status(gemmwich(blas, nmo, nc, out.c_cart_.data(), out.j_.data(), work, block)));
    FBF_TRY(diagonal(j_pc.data() + (n_all * i), block, n_all, n_all));
    FBF_TRY(gpu_status(gemmwich(blas, nmo, nc, out.c_cart_.data(), out.k_.data(), work, block)));
    FBF_TRY(diagonal(k_pc.data() + (n_all * i), block, n_all, n_all));
  }
  FBF_TRY(res.sync());

  // gen_g_hop's h_diag over the packed pairs (p, q), doubled. With D the
  // 1-RDM's diagonal (2 core, gamma_tt active, 0 virtual):
  //   h_pp D_q + h_qq D_p - F_pp - F_qq, for every pair;
  //   + 2 Vca_pp + 6 (pq|pq) - 2 (pp|qq), q core;
  //   + Vc_qq gamma_uu - sum_v (6 (qu|qv) - 2 (qq|uv)) gamma_uv + H_act[(q, u), (q, u)],
  //     q core and p = ncore + u active;
  //   + Vc_pp gamma_uu + H_act[(p, u), (p, u)], q = ncore + u active.
  const auto occupation = [&](size_t p) {
    if (p < core) {
      return 2.0;
    }
    return p < occ ? gamma[(p - core) * (na + 1)] : 0.0;
  };
  std::vector<double> hd;
  hd.reserve(n_pairs);
  for (size_t q = 0; q < occ; ++q) {
    for (size_t p = q < core ? core : occ; p < n_all; ++p) {
      double value =
          (h_diag[p] * occupation(q)) + (h_diag[q] * occupation(p)) - f_diag[p] - f_diag[q];
      if (q < core) {
        value += (2.0 * vca_diag[p]) +
                 ((6.0 * k_pc[p + (n_all * q)] - 2.0 * j_pc[p + (n_all * q)]) / scale[q]);
        if (p < occ) {
          const size_t uu = p - core;
          double jk = 0.0;
          for (size_t v = 0; v < na; ++v) {
            // (qu|qv) and (qq|uv): row q of column q of block (u, v).
            const size_t at = q + (core * (q + (n_all * (uu + (na * v)))));
            jk += ((6.0 * papa_core[at]) - (2.0 * ppaa_core[at])) * gamma[uu + (na * v)];
          }
          value += (vc_diag[q] * gamma[uu * (na + 1)]) - jk + act_diag[q + (n_all * uu)];
        }
      } else {
        const size_t uu = q - core;
        value += (vc_diag[p] * gamma[uu * (na + 1)]) + act_diag[p + (n_all * uu)];
      }
      hd.push_back(2.0 * value);
    }
  }
  if (n_pairs > 0) {
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(out.diagonal_.data(), hd.data(), n_pairs * sizeof(double),
                                       wwr::wwrMemcpyHostToDevice, stream)));
  }
  FBF_TRY(res.sync());
  return out;
}

Status Hessian::apply(DeviceResources &res, const int32_t k, const double *const x,
                      double *const y) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("mcscf::Hessian::apply: built on device {}, but the call passed device {}",
                      dev_idx_, res.dev_idx());
  }
  if (k < 0) {
    return err_config("mcscf::Hessian::apply: the block has {} columns", k);
  }
  if (k == 0 || n_pairs_ == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t nmo = nmo_;
  const int32_t ncore = ncore_;
  const int32_t ncas = ncas_;
  const int32_t nocc = ncore + ncas;
  const int32_t nc = n_cart_;
  const auto n_all = static_cast<size_t>(nmo);
  const auto n_cart = static_cast<size_t>(nc);
  const size_t nn = n_all * n_all;
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  const int32_t n_rot = nmo - ncore; // the active and virtual columns
  const auto n_act = nmo * ncas;     // the order of H_act
  constexpr double kMinusHalf = -0.5;
  constexpr double kMinusOne = -1.0;
  constexpr double kMinusTwo = -2.0;

  // Scratch, carved once for the block: coulomb and exchange, called while it
  // is live, do not take the workspace.
  const auto [kappa, x2, hx, t_act, t2, w, cart, dm3, jc] = res.device_regions<double>(
      nn, nn, n_all * (core + na), na * n_all, na * n_all, std::max(core, na) * n_cart,
      n_cart * n_cart, n_cart * n_cart, n_cart * n_all);
  const double *const c_cart = c_cart_.data();
  const double *const c_cart_act = c_cart + (n_cart * core);
  // A gemm over a row block of kappa, x2 or V^c + V^a, whose leading dimension
  // is nmo rather than the block's own row count, which recipes/blas's packed
  // gemm cannot express. Every other product here is packed and is gemm.
  const auto block_gemm = [blas](wwr::wwrblasOperation_t op_a, wwr::wwrblasOperation_t op_b,
                                 int32_t m, int32_t n, int32_t kk, double alpha, const double *a,
                                 int32_t lda, const double *b, int32_t ldb, double beta, double *cc,
                                 int32_t ldc) {
    return gpu_status(wwr::gemm<double, int32_t>(blas, op_a, op_b, m, n, kk, &alpha, a, lda, b, ldb,
                                                 &beta, cc, ldc));
  };
  constexpr auto kN = wwr::WWRBLAS_OP_N;
  constexpr auto kT = wwr::WWRBLAS_OP_T;

  for (size_t column = 0; column < static_cast<size_t>(k); ++column) {
    const double *const xi = x + (column * static_cast<size_t>(n_pairs_));
    double *const yi = y + (column * static_cast<size_t>(n_pairs_));
    FBF_TRY(unpack_orbital_pairs(res, nmo, ncore, ncas, xi, kappa));

    // gen_g_hop's x2, then 2 (x2 - x2^T) packed. h D: (C^T h C) kappa over
    // the occupied columns, times the 1-RDM's occupied block.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(x2, 0, nn * sizeof(double), stream)));
    FBF_TRY(gpu_status(
        gemm(blas, nmo, nocc, nmo, &kOne<double>, h_mo_.data(), kappa, &kZero<double>, hx)));
    FBF_TRY(gpu_status(
        gemm(blas, nmo, nocc, nocc, &kOne<double>, hx, dm1_occ_.data(), &kZero<double>, x2)));
    // - (F + F^T) kappa / 2.
    FBF_TRY(gpu_status(gemm(blas, nmo, &kMinusHalf, fock_sym_.data(), kappa, &kOne<double>, x2)));
    // Core rows: 2 kappa[core, ncore:] (V^c + V^a)[ncore:, :].
    if (ncore > 0 && n_rot > 0) {
      FBF_TRY(block_gemm(kN, kN, ncore, nmo, n_rot, 2.0, kappa + (n_all * core), nmo,
                         vhf_ca_.data() + core, nmo, 1.0, x2, nmo));
    }
    // Active rows, gamma (kappa[act, :] V^c + va), va's J and K below.
    if (ncas > 0) {
      FBF_TRY(block_gemm(kN, kN, ncas, nmo, nmo, 1.0, kappa + core, nmo, vhf_c_.data(), nmo, 0.0,
                         t_act, ncas));
    }

    // update_jk_in_ah: D3 = C_core kappa[core, ncore:] C_rot^T + its
    // transpose, D4 = C_act gamma kappa[act, :] C^T + its transpose;
    // va = gamma C_act^T (2 J[D3] - K[D3]) C, and
    // vc = C_core^T (2 J[D'] - K[D']) C_rot, D' = 2 D3 + D4, onto the
    // core rows' active and virtual columns. All over U C.
    if (ncore > 0 && n_rot > 0) {
      FBF_TRY(block_gemm(kN, kT, ncore, nc, n_rot, 1.0, kappa + (n_all * core), nmo,
                         c_cart + (n_cart * core), nc, 0.0, w, ncore));
      FBF_TRY(
          gpu_status(gemm(blas, nc, nc, ncore, &kOne<double>, c_cart, w, &kZero<double>, cart)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kT, nc, nc, &kOne<double>, cart, nc,
                                                    &kOne<double>, cart, nc, dm3, nc)));
      const double s3 = FBF_TRY(capped_density(res, dm3));
      FBF_TRY(coulomb(res, j_classes_, density_, j_, tau_j_ * s3));
      FBF_TRY(exchange(res, k_classes_, density_, k_, tau_k_ * s3));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kN, nc, nc, &kTwo<double>, j_.data(),
                                                    nc, &kMinusOne, k_.data(), nc, j_.data(), nc)));
      if (ncas > 0) {
        FBF_TRY(gpu_status(
            gemm(blas, nc, nmo, nc, &kOne<double>, j_.data(), c_cart, &kZero<double>, jc)));
        // BLAS reads alpha through a pointer, and an expression has no address.
        const double inv_s3 = 1.0 / s3;
        FBF_TRY(gpu_status(
            gemm<kT>(blas, ncas, nmo, nc, &inv_s3, c_cart_act, jc, &kOne<double>, t_act)));
        // D' = 2 D3 + C_act (gamma kappa[act, :] C^T) + its transpose.
        FBF_TRY(block_gemm(kN, kN, ncas, nmo, ncas, 1.0, gamma_.data(), ncas, kappa + core, nmo,
                           0.0, t2, ncas));
        FBF_TRY(gpu_status(
            gemm<kN, kT>(blas, ncas, nc, nmo, &kOne<double>, t2, c_cart, &kZero<double>, w)));
        FBF_TRY(gpu_status(
            gemm(blas, nc, nc, ncas, &kOne<double>, c_cart_act, w, &kZero<double>, cart)));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kN, nc, nc, &kTwo<double>, dm3, nc,
                                                      &kOne<double>, cart, nc, dm3, nc)));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kT, nc, nc, &kOne<double>, dm3, nc,
                                                      &kOne<double>, cart, nc, dm3, nc)));
      } else {
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, nc * nc, &kTwo<double>, dm3, 1)));
      }
      const double s4 = FBF_TRY(capped_density(res, dm3));
      FBF_TRY(coulomb(res, j_classes_, density_, j_, tau_j_ * s4));
      FBF_TRY(exchange(res, k_classes_, density_, k_, tau_k_ * s4));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kN, nc, nc, &kTwo<double>, j_.data(),
                                                    nc, &kMinusOne, k_.data(), nc, j_.data(), nc)));
      FBF_TRY(gpu_status(gemm(blas, nc, n_rot, nc, &kOne<double>, j_.data(),
                              c_cart + (n_cart * core), &kZero<double>, jc)));
      FBF_TRY(block_gemm(kT, kN, ncore, n_rot, nc, 1.0 / s4, c_cart, nc, jc, nc, 1.0,
                         x2 + (n_all * core), nmo));
    }

    if (ncas > 0) {
      FBF_TRY(block_gemm(kN, kN, ncas, nmo, ncas, 1.0, gamma_.data(), ncas, t_act, ncas, 1.0,
                         x2 + core, nmo));
      // Active columns: H_act vec(kappa[:, act]), both nmo ncas contiguous.
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, kN, n_act, n_act, &kOne<double>,
                                                    h_act_.data(), n_act, kappa + (n_all * core), 1,
                                                    &kOne<double>, x2 + (n_all * core), 1)));
    }

    // 2 (x2 - x2^T), the derivative's, over kappa (no longer read).
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, kN, kT, nmo, nmo, &kTwo<double>, x2, nmo,
                                                  &kMinusTwo, x2, nmo, kappa, nmo)));
    FBF_TRY(pack_orbital_pairs(res, nmo, ncore, ncas, kappa, yi));
  }
  return {};
}

} // namespace free_breakfast::mcscf
