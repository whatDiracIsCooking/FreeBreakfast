/// @file
/// @brief df_qd_zeta: the zeta term T = Sum_{r<s} zeta_sr <c_s|H|c_r> of a
/// QD-NEVPT2 Lagrangian's Hessian, its skeleton share (df_casscf_exy at the
/// folded transition RDMs, by the +/- polarization of the roots), its mixed
/// share (the same polarization of mcscf::df_derivatives' right-hand sides,
/// and the transition Hamiltonian's derivative for the rotations among the
/// roots) and its parameter Hessian over a block of columns (the orbital
/// Hessian at the folded RDMs less at zero ones, the orbital gradient at each
/// column's first-order transition RDMs, and transition sigma vectors).

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy, geam, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.active_space;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.hessian;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.gradient;
import free_breakfast.mcscf.transition;
import :state_specific; // state_averaged_electrons

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<QdZetaTerms>
df_qd_zeta(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
           const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
           const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
           const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
           const std::span<const double> zeta, const int32_t k, const double *const lambda) {
  constexpr std::string_view kWho = "hessian::df_qd_zeta";
  const size_t nroots = solution.roots.energies.size();
  // T reads no weights; the check's are any positive ones, and the
  // polarization's (each 1 / n) are its own.
  const std::vector<double> equal(nroots, nroots == 0 ? 0.0 : 1.0 / static_cast<double>(nroots));
  const auto [nalpha, nbeta] = FBF_TRY(
      state_averaged_electrons(kWho, res, molecule, basis, aux, b, solution, equal, ncore, ncas));
  const size_t n_pairs = nroots * (nroots - 1) / 2;
  if (zeta.size() != n_pairs || k < 0 || (k > 0 && lambda == nullptr)) {
    return err_config("{}: needs one zeta per pair of the {} roots ({}), k >= 0 and Lambda for "
                      "k > 0 (got {} zeta, k {})",
                      kWho, nroots, n_pairs, zeta.size(), k);
  }
  const SphericalMatrixAO &c = solution.c;
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const auto naux32 = static_cast<int32_t>(naux);
  const uint32_t n_atom = basis.natoms();
  const ScopedDeviceIndex on{res.dev_idx()};
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const size_t n_ci = dim * nroots;
  const Buffers::device<double> &roots = solution.roots.vectors;
  if (roots.num_elements() != n_ci) {
    return err_config("{}: {} CI doubles for {} roots of {} determinants", kWho,
                      roots.num_elements(), nroots, dim);
  }
  const auto n_act = static_cast<size_t>(ncas);
  const size_t nocc = static_cast<size_t>(ncore) + n_act;
  const size_t pairs = (n_act * static_cast<size_t>(ncore)) + ((nao - nocc) * nocc);
  const size_t size = pairs + n_ci + n_pairs;
  const auto n_x = static_cast<size_t>(3) * n_atom;
  const size_t mm = nao * nao;
  const auto nao32 = static_cast<int32_t>(nao);
  const auto dim32 = static_cast<int32_t>(dim);
  const auto nroots32 = static_cast<int32_t>(nroots);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  const auto upload = [&](std::span<const double> from) -> Result<Buffers::device<double>> {
    Buffers::device<double> to(from.size(), res.device());
    FBF_TRY(copy(to.data(), from.data(), from.size(), wwr::wwrMemcpyHostToDevice));
    FBF_TRY(res.sync());
    return to;
  };
  // y -= C (C^T y) for the dim x count block y: off every root.
  const auto off_roots = [&](double *y, const int32_t count) -> Status {
    Buffers::device<double> overlap(nroots * static_cast<size_t>(count), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nroots32, count, dim32, &kOne<double>,
                                               roots.data(), y, &kZero<double>, overlap.data())));
    FBF_TRY(gpu_status(gemm(blas, dim32, count, nroots32, &kNegOne<double>, roots.data(),
                            overlap.data(), &kOne<double>, y)));
    // overlap is freed on return, while the gemms read it.
    return res.sync();
  };

  // Z, symmetric with a zero diagonal (column-major, (s, r) at s + n r):
  // T = 1/2 Sum_{r != s} Z_rs <c_s|H|c_r>.
  std::vector<double> z_host(nroots * nroots, 0.0);
  {
    size_t pair = 0;
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = r + 1; s < nroots; ++s, ++pair) {
        z_host[s + (nroots * r)] = zeta[pair];
        z_host[r + (nroots * s)] = zeta[pair];
      }
    }
  }
  const auto z_at = [&](const size_t p, const size_t q) { return z_host[p + (nroots * q)]; };

  const auto derivatives = [&](const double *vectors) -> Result<mcscf::Derivatives> {
    return mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                 nalpha, nbeta, vectors, equal, nullptr);
  };
  // df_casscf_exy reads C as a plain buffer.
  Buffers::device<double> c_columns(mm, res.device());
  FBF_TRY(copy(c_columns.data(), c.data(), mm, wwr::wwrMemcpyDeviceToDevice));

  // The rotations' rows of L_Ax: dT/dTheta_sr = (Z M)_rs - (Z M)_sr at
  // M = C^T H C (c_r gaining Theta_sr c_s moves M by A^T M + M A), so at the
  // derivative integrals M^x, df_derivatives' Hamiltonian at the roots. Its
  // core energy is a multiple of the identity, which this difference drops.
  std::vector<double> theta_rows(n_pairs * n_x, 0.0);
  {
    const mcscf::Derivatives at_roots = FBF_TRY(derivatives(roots.data()));
    for (size_t x = 0; x < n_x; ++x) {
      const double *m = at_roots.hamiltonian.data() + (nroots * nroots * x);
      const auto zm = [&](const size_t p, const size_t q) {
        double sum = 0.0;
        for (size_t t = 0; t < nroots; ++t) {
          sum += z_at(p, t) * m[t + (nroots * q)];
        }
        return sum;
      };
      size_t pair = 0;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s, ++pair) {
          theta_rows[pair + (n_pairs * x)] = zm(r, s) - zm(s, r);
        }
      }
    }
  }

  // The polarization: the roots c_r +/- dc_r, dc = C S with S = (n / 4) Z,
  // at the weights 1 / n. Half the difference of any quantity affine in the
  // RDMs at a fixed core is then its value at Sum_r (1 / n) (T(dc_r, c_r) +
  // T(c_r, dc_r)) = Sum_{r<s} zeta_sr T^(sr), the folded transition RDMs,
  // less its value at zero RDMs: E_xy, F^[x] and the orbital right-hand side.
  // The CI one, 2 (1 / n) Q_+- H~^x (c_r +/- dc_r), projected off the roots
  // (Q Q_+- = Q, the vectors lying in the roots' span) and differenced, is
  // Q H~^x (C Z)_r, dT/dd_r at the derivative integrals.
  Buffers::device<double> lxlambda(size * n_x, res.device());
  CasscfExy lxy;
  {
    std::vector<double> s_host = z_host;
    for (double &e : s_host) {
      e *= static_cast<double>(nroots) / 4.0;
    }
    const Buffers::device<double> s_dev = FBF_TRY(upload(s_host));
    Buffers::device<double> plus(n_ci, res.device());
    Buffers::device<double> minus(n_ci, res.device());
    FBF_TRY(copy(plus.data(), roots.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(minus.data(), roots.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kOne<double>, roots.data(),
                            s_dev.data(), &kOne<double>, plus.data())));
    FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kNegOne<double>, roots.data(),
                            s_dev.data(), &kOne<double>, minus.data())));
    FBF_TRY(res.sync());
    const auto rdms =
        [&](const double *vectors) -> Result<std::pair<std::vector<double>, std::vector<double>>> {
      Buffers::device<double> gamma(n_act * n_act, res.device());
      Buffers::device<double> gamma2(n_act * n_act * n_act * n_act, res.device());
      FBF_TRY(rdm.rdm12(res, equal, vectors, gamma.data(), gamma2.data()));
      return std::pair{FBF_TRY(download(gamma.data(), gamma.num_elements())),
                       FBF_TRY(download(gamma2.data(), gamma2.num_elements()))};
    };
    // One side: its E_xy, and its right-hand sides at sign +/- 1/2 (orbital)
    // and +/- 1 (CI) into lxlambda, which the + side writes and the - side
    // adds to.
    Buffers::device<double> packed(pairs, res.device());
    const auto side = [&](const double *vectors, const double sign) -> Result<CasscfExy> {
      const auto [gamma, gamma2] = FBF_TRY(rdms(vectors));
      const mcscf::Derivatives f = FBF_TRY(derivatives(vectors));
      const double half = 0.5 * sign;
      for (size_t x = 0; x < n_x; ++x) {
        double *column = lxlambda.data() + (x * size);
        FBF_TRY(response::pack_orbital_pairs(res, nao32, ncore, ncas,
                                             f.rhs_orbital.data() + (x * mm), packed.data()));
        const double *ci = f.rhs_ci.data() + (x * n_ci);
        if (sign > 0.0) {
          FBF_TRY(gpu_status(wwr::wwrMemsetAsync(column, 0, size * sizeof(double), res.stream())));
        }
        FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(pairs), &half,
                                                      packed.data(), 1, column, 1)));
        FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(n_ci), &sign, ci,
                                                      1, column + pairs, 1)));
      }
      FBF_TRY(res.sync());
      return df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c_columns, ncore, ncas,
                           gamma, gamma2, f.fock, nullptr);
    };
    const CasscfExy at_plus = FBF_TRY(side(plus.data(), 1.0));
    const CasscfExy at_minus = FBF_TRY(side(minus.data(), -1.0));
    const auto half_difference = [](const std::vector<double> &p, const std::vector<double> &m) {
      std::vector<double> out(p.size());
      for (size_t i = 0; i < p.size(); ++i) {
        out[i] = 0.5 * (p[i] - m[i]);
      }
      return out;
    };
    lxy.one_electron = half_difference(at_plus.one_electron, at_minus.one_electron);
    lxy.nuclear_repulsion = half_difference(at_plus.nuclear_repulsion, at_minus.nuclear_repulsion);
    lxy.two_electron.three_center =
        half_difference(at_plus.two_electron.three_center, at_minus.two_electron.three_center);
    lxy.two_electron.metric =
        half_difference(at_plus.two_electron.metric, at_minus.two_electron.metric);
    lxy.two_electron.cross_coulomb =
        half_difference(at_plus.two_electron.cross_coulomb, at_minus.two_electron.cross_coulomb);
    lxy.two_electron.cross_exchange =
        half_difference(at_plus.two_electron.cross_exchange, at_minus.two_electron.cross_exchange);
    lxy.two_electron.cross_active =
        half_difference(at_plus.two_electron.cross_active, at_minus.two_electron.cross_active);
    lxy.two_electron.total =
        half_difference(at_plus.two_electron.total, at_minus.two_electron.total);
    lxy.connection_second = half_difference(at_plus.connection_second, at_minus.connection_second);
    lxy.connection_fock = half_difference(at_plus.connection_fock, at_minus.connection_fock);
    lxy.connection_twice = half_difference(at_plus.connection_twice, at_minus.connection_twice);
    lxy.exy = half_difference(at_plus.exy, at_minus.exy);

    // The CI rows off every root, and the rotations' rows.
    for (size_t x = 0; x < n_x; ++x) {
      double *column = lxlambda.data() + (x * size);
      FBF_TRY(off_roots(column + pairs, nroots32));
      FBF_TRY(copy(column + pairs + n_ci, theta_rows.data() + (x * n_pairs), n_pairs,
                   wwr::wwrMemcpyHostToDevice));
    }
    FBF_TRY(res.sync());
  }

  std::vector<double> lxx;
  if (k > 0) {
    const auto ks = static_cast<size_t>(k);
    const size_t nk = nroots * ks;
    const auto nk32 = static_cast<int32_t>(nk);
    const auto pairs32 = static_cast<int32_t>(pairs);
    const CasscfIntegrals ints =
        FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux32, c, ncore, ncas));

    // The columns' parts: kappa (pairs x k), and per column X_u = K_u C =
    // D_u + C A_u and P_u = K_u H C = D_u (C^T H C) - C (D_u^T H C) +
    // C A_u (C^T H C), D_u its CI part off the roots and A_u its
    // antisymmetric rotations ((s, r) = Theta_sr), each dim x n, the columns
    // side by side. Every buffer is held across sigmas and RDMs, which take
    // the workspace.
    ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
    // The active Hamiltonian without its core energy, which drops from T's
    // CI part (README.md); row-major, as mcscf::solve sets it.
    FBF_TRY(sigma.set_integrals(
        res, 0.0, FBF_TRY(download(ints.hamiltonian.h_eff.data(), n_act * n_act)),
        FBF_TRY(download(ints.hamiltonian.eri.data(), n_act * n_act * n_act * n_act))));
    Buffers::device<double> hc(n_ci, res.device());
    FBF_TRY(sigma.apply(res, nroots32, roots.data(), hc.data()));
    Buffers::device<double> m(nroots * nroots, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nroots32, nroots32, dim32, &kOne<double>,
                                               roots.data(), hc.data(), &kZero<double>, m.data())));
    Buffers::device<double> kappa(pairs * ks, res.device());
    Buffers::device<double> xs(dim * nk, res.device());
    Buffers::device<double> ps(dim * nk, res.device());
    {
      Buffers::device<double> d(n_ci, res.device());
      Buffers::device<double> dhc(nroots * nroots, res.device());
      Buffers::device<double> am(nroots * nroots, res.device());
      std::vector<double> theta(n_pairs);
      std::vector<double> a(nroots * nroots);
      for (size_t i = 0; i < ks; ++i) {
        const double *column = lambda + (i * size);
        double *x = xs.data() + (i * n_ci);
        double *p = ps.data() + (i * n_ci);
        FBF_TRY(copy(kappa.data() + (i * pairs), column, pairs, wwr::wwrMemcpyDeviceToDevice));
        FBF_TRY(copy(d.data(), column + pairs, n_ci, wwr::wwrMemcpyDeviceToDevice));
        FBF_TRY(off_roots(d.data(), nroots32));
        FBF_TRY(copy(theta.data(), column + pairs + n_ci, n_pairs, wwr::wwrMemcpyDeviceToHost));
        FBF_TRY(res.sync());
        std::ranges::fill(a, 0.0);
        size_t pair = 0;
        for (size_t r = 0; r < nroots; ++r) {
          for (size_t s = r + 1; s < nroots; ++s, ++pair) {
            a[s + (nroots * r)] = theta[pair];
            a[r + (nroots * s)] = -theta[pair];
          }
        }
        const Buffers::device<double> a_dev = FBF_TRY(upload(a));
        // X = D + C A.
        FBF_TRY(copy(x, d.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
        FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kOne<double>, roots.data(),
                                a_dev.data(), &kOne<double>, x)));
        // P = D M - C (D^T H C) + C (A M).
        FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kOne<double>, d.data(), m.data(),
                                &kZero<double>, p)));
        FBF_TRY(
            gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nroots32, nroots32, dim32, &kOne<double>,
                                               d.data(), hc.data(), &kZero<double>, dhc.data())));
        FBF_TRY(gpu_status(gemm(blas, nroots32, nroots32, nroots32, &kOne<double>, a_dev.data(),
                                m.data(), &kZero<double>, am.data())));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nroots32, nroots32, &kOne<double>,
            am.data(), nroots32, &kNegOne<double>, dhc.data(), nroots32, am.data(), nroots32)));
        FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kOne<double>, roots.data(),
                                am.data(), &kOne<double>, p)));
        // a_dev is freed at the end of the iteration, while the gemms read it.
        FBF_TRY(res.sync());
      }
    }

    // The CI part: <X_i, H X_j Z> - 1/2 <X_j, P_i Z> - 1/2 <X_i, P_j Z>,
    // from the Gram matrices X^T (H X) and X^T P over every column's roots.
    std::vector<double> ci_part(ks * ks, 0.0);
    {
      Buffers::device<double> hx(dim * nk, res.device());
      FBF_TRY(sigma.apply(res, nk32, xs.data(), hx.data()));
      Buffers::device<double> gram(nk * nk, res.device());
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nk32, nk32, dim32, &kOne<double>, xs.data(),
                                                 hx.data(), &kZero<double>, gram.data())));
      const std::vector<double> xhx = FBF_TRY(download(gram.data(), nk * nk));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nk32, nk32, dim32, &kOne<double>, xs.data(),
                                                 ps.data(), &kZero<double>, gram.data())));
      const std::vector<double> xp = FBF_TRY(download(gram.data(), nk * nk));
      // (column a, root q) at a n + q.
      const auto at = [&](const std::vector<double> &g, const size_t a, const size_t q,
                          const size_t bcol, const size_t p) {
        return g[((a * nroots) + q) + (nk * ((bcol * nroots) + p))];
      };
      for (size_t j = 0; j < ks; ++j) {
        for (size_t i = 0; i < ks; ++i) {
          double sum = 0.0;
          for (size_t q = 0; q < nroots; ++q) {
            for (size_t p = 0; p < nroots; ++p) {
              const double zpq = z_at(p, q);
              sum += zpq * (at(xhx, i, q, j, p) - (0.5 * at(xp, j, q, i, p)) -
                            (0.5 * at(xp, i, q, j, p)));
            }
          }
          ci_part[i + (ks * j)] = sum;
        }
      }
    }

    // The orbital-CI part: g(R_i) . kappa_j + g(R_j) . kappa_i, R_i =
    // Sum_q T(X_i,q, (C Z)_q) symmetrized, the first-order change of the
    // folded transition RDMs along column i.
    Buffers::device<double> grads(pairs * ks, res.device());
    {
      const Buffers::device<double> z_dev = FBF_TRY(upload(z_host));
      Buffers::device<double> cz(n_ci, res.device());
      FBF_TRY(gpu_status(gemm(blas, dim32, nroots32, nroots32, &kOne<double>, roots.data(),
                              z_dev.data(), &kZero<double>, cz.data())));
      const size_t m1 = n_act * n_act;
      const size_t m2 = m1 * m1;
      Buffers::device<double> one(m1 + m2, res.device());
      Buffers::device<double> sum(m1 + m2, res.device());
      for (size_t i = 0; i < ks; ++i) {
        FBF_TRY(gpu_status(
            wwr::wwrMemsetAsync(sum.data(), 0, (m1 + m2) * sizeof(double), res.stream())));
        for (size_t q = 0; q < nroots; ++q) {
          FBF_TRY(rdm.trans_rdm12(res, xs.data() + (i * n_ci) + (q * dim), cz.data() + (q * dim),
                                  one.data(), one.data() + m1));
          FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(
              blas, static_cast<int32_t>(m1 + m2), &kOne<double>, one.data(), 1, sum.data(), 1)));
        }
        const std::vector<double> r1 = FBF_TRY(download(sum.data(), m1));
        const std::vector<double> r2 = FBF_TRY(download(sum.data() + m1, m2));
        const Buffers::device<double> g = FBF_TRY(mcscf::transition_orbital_gradient(
            res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, r1, r2));
        FBF_TRY(copy(grads.data() + (i * pairs), g.data(), pairs, wwr::wwrMemcpyDeviceToDevice));
        // g is freed at the end of the iteration, while the copy reads it.
        FBF_TRY(res.sync());
      }
    }

    // The orbital part: H(gamma^zeta) - H(0) on kappa, H affine in the RDMs
    // at a fixed core, and the exact second derivative along C exp(kappa)
    // off stationarity (README.md).
    Buffers::device<double> y(pairs * ks, res.device());
    {
      const mcscf::ZetaTerms folded = FBF_TRY(mcscf::zeta_terms(
          res, molecule, h, b, naux32, solution, ncore, ncas, ints.ppaa, ints.papa, rdm, zeta));
      const response::Fock at_zeta =
          FBF_TRY(response::fock(res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa,
                                 folded.gamma, folded.gamma2));
      const std::vector<double> zero1(n_act * n_act, 0.0);
      const std::vector<double> zero2(n_act * n_act * n_act * n_act, 0.0);
      const response::Fock at_zero = FBF_TRY(response::fock(
          res, molecule, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, zero1, zero2));
      Buffers::device<double> y0(pairs * ks, res.device());
      {
        response::Hessian hessian = FBF_TRY(response::Hessian::create(
            res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, at_zeta));
        FBF_TRY(hessian.apply(res, k, kappa.data(), y.data()));
      }
      {
        response::Hessian hessian = FBF_TRY(response::Hessian::create(
            res, h, b, naux32, c, ncore, ncas, ints.ppaa, ints.papa, at_zero));
        FBF_TRY(hessian.apply(res, k, kappa.data(), y0.data()));
      }
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, pairs32 * k, &kNegOne<double>, y0.data(),
                                                    1, y.data(), 1)));
      FBF_TRY(res.sync());
    }

    // kappa^T (y + grads) + grads^T kappa: the orbital and orbital-CI parts.
    Buffers::device<double> product(ks * ks, res.device());
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, pairs32 * k, &kOne<double>, grads.data(), 1,
                                                  y.data(), 1)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, k, k, pairs32, &kOne<double>, kappa.data(),
                                               y.data(), &kZero<double>, product.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, k, k, pairs32, &kOne<double>, grads.data(),
                                               kappa.data(), &kOne<double>, product.data())));
    lxx = FBF_TRY(download(product.data(), ks * ks));
    for (size_t i = 0; i < ks * ks; ++i) {
      lxx[i] += ci_part[i];
    }
  }
  return QdZetaTerms{.lxy = std::move(lxy), .lxlambda = std::move(lxlambda), .lxx = std::move(lxx)};
}

MemoryFootprint qd_zeta_footprint(const BasisSet &basis, const BasisSet &aux, const int32_t ncore,
                                  const int32_t ncas, const int32_t nalpha, const int32_t nbeta,
                                  const int32_t nroots, const int32_t k, const size_t syevd_lwork) {
  const size_t nao = basis.nao_sphe();
  const auto nmo = static_cast<int32_t>(nao);
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto n_act = static_cast<size_t>(ncas);
  const auto roots = static_cast<size_t>(nroots);
  const auto ks = static_cast<size_t>(k);
  const size_t na2 = n_act * n_act;
  const size_t n_x = 3 * static_cast<size_t>(basis.natoms());
  const size_t mm = nao * nao;
  const size_t nocc = static_cast<size_t>(ncore) + n_act;
  const size_t pairs = (n_act * static_cast<size_t>(ncore)) + ((nao - nocc) * nocc);
  const auto strings = [ncas](const int32_t electrons) {
    return static_cast<int32_t>(std::round(math::binomial(ncas, electrons)));
  };
  const int32_t alpha_strings = strings(nalpha);
  const int32_t beta_strings = strings(nbeta);
  const size_t dim = static_cast<size_t>(alpha_strings) * static_cast<size_t>(beta_strings);
  const size_t n_ci = dim * roots;
  const size_t size = pairs + n_ci + (roots * (roots - 1) / 2);
  const size_t nk = roots * ks;
  constexpr size_t kD = sizeof(double);
  const auto held = [](const size_t bytes) {
    return MemoryFootprint{.device = {.persistent = bytes}};
  };
  const MemoryFootprint derivatives =
      mcscf::df_derivatives_footprint(basis, aux, ncore, ncas, nalpha, nbeta, nroots, syevd_lwork);
  MemoryFootprint rdms = held(kD * (na2 + (na2 * na2)));
  rdms.device.workspace = ci::rdm12_footprint(ncas, static_cast<int32_t>(dim)).device.workspace;
  const MemoryFootprint square = held(kD * roots * roots);
  Ledger l;
  // The strings; C's copy, held; the roots' derivatives; L_xlambda, returned.
  l.call(ci::device_string_set_footprint(ncas, nalpha));
  l.call(ci::device_string_set_footprint(ncas, nbeta));
  l.take(kD * mm);
  l.run(derivatives);
  l.take(kD * size * n_x);
  // L_xy and the right-hand side: S, c +- C S and the packed column; each
  // side's RDMs, then F^[x] beside E_xy; the columns' rotations off the
  // roots.
  const size_t sides = (kD * ((roots * roots) + (2 * n_ci) + pairs));
  l.take(sides);
  for (int32_t side = 0; side < 2; ++side) {
    l.run(rdms);
    l.call(derivatives);
    l.run(df_casscf_exy_footprint(basis, aux, ncore, ncas));
    l.give(derivatives.device.persistent);
  }
  l.run(square);
  l.give(sides);
  if (k > 0) {
    // The integrals; the sigma's tables, H C and its apply scratch; M,
    // kappa, X and P; per column d, its overlaps and the rotation.
    const MemoryFootprint ints = casscf_integrals_footprint(nmo, naux, ncore, ncas);
    l.call(ints);
    const size_t sigma = ci::sigma_footprint(ncas, alpha_strings, beta_strings).device.persistent +
                         ci::sigma_apply_bytes(ncas, alpha_strings, beta_strings, ci::Layout::full);
    const size_t block = kD * (n_ci + (roots * roots) + (pairs * ks) + (2 * dim * nk));
    l.take(sigma + block);
    l.take(kD * (n_ci + (2 * roots * roots)));
    l.run(square);
    l.run(square);
    l.give(kD * (n_ci + (2 * roots * roots)));
    // H X and the Gram matrices.
    l.run(held(kD * ((dim * nk) + (nk * nk))));
    // The orbital gradients at the transition RDMs: Z, c Z, and the RDMs'
    // two buffers beside one column's gradient.
    l.take(kD * pairs * ks);
    MemoryFootprint transition = held(kD * ((roots * roots) + n_ci + (2 * (na2 + (na2 * na2)))));
    transition.device.workspace =
        ci::trans_rdm12_footprint(ncas, static_cast<int32_t>(dim)).device.workspace;
    l.call(transition);
    l.run(mcscf::transition_orbital_gradient_footprint(nmo, naux, ncore, ncas));
    l.give(transition.device.persistent);
    // y; the folded RDMs and the Focks at them and at none, y0, and each
    // Hessian with its product.
    l.take(kD * pairs * ks);
    const MemoryFootprint zeta =
        mcscf::zeta_terms_footprint(nmo, naux, ncore, ncas, static_cast<int32_t>(dim));
    const MemoryFootprint fock = response::fock_footprint(nmo, naux, ncore, ncas);
    l.call(zeta);
    l.call(fock);
    l.call(fock);
    l.take(kD * pairs * ks);
    const MemoryFootprint hessian = response::hessian_footprint(nmo, naux, ncore, ncas);
    for (int32_t pass = 0; pass < 2; ++pass) {
      l.call(hessian);
      l.run(response::hessian_apply_footprint(nmo, naux, ncore, ncas));
      l.give(hessian.device.persistent);
    }
    l.give(zeta.device.persistent + (2 * fock.device.persistent) + (kD * pairs * ks));
    // The k x k product.
    l.run(held(kD * ks * ks));
    l.give(ints.device.persistent + sigma + block + (2 * kD * pairs * ks));
  }
  return l.footprint(kD * size * n_x);
}

} // namespace free_breakfast::hessian
