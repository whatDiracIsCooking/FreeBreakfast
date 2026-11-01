/// @file
/// @brief I^x z: the state-averaged Hessian along z differentiated along the
/// integrals (ThirdDerivative::apply_integrals), and the MO derivative
/// integrals I^x it is applied at (df_integral_derivatives): the core
/// Hamiltonian's and the MO factor's directions, the connection included.

module free_breakfast.mcscf.third_derivative;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kHalf, kNegHalf, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, geam, dgmm, trsm, scal, axpy, memcpys
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.ci.sigma;
import free_breakfast.response.orbital_pairs;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The spherical functions of a placed @p basis on @p atom, [first, last):
// its shells are contiguous, so their functions are.
std::pair<size_t, size_t> integral_atom_functions(const BasisSet &basis, const uint32_t atom) {
  const auto start_of = [&basis](const uint32_t shell) -> size_t {
    return shell < basis.nshells() ? basis.ao_offset_sphe(shell) : basis.nao_sphe();
  };
  const uint32_t first = basis.atom_shell_offset(atom);
  return {start_of(first), start_of(first + basis.atom_shell_count(atom))};
}

Status df_integral_derivatives(DeviceResources &res, const Molecule &molecule,
                               const BasisSet &basis,
                               const std::span<const PrimitivePairClass> classes,
                               const BasisSet &aux, const SphericalMatrixAO &h,
                               const Buffers::device<double> &b, const double *const l,
                               const SphericalMatrixAO &c, double *const h1, double *const w1) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t n_atom = basis.natoms();
  const size_t n_x = 3 * n_atom;
  if (molecule.natoms() != basis.natoms() || aux.natoms() != basis.natoms() || n_atom == 0) {
    return err_config("mcscf::df_integral_derivatives: the molecule ({} atoms) and both bases ({} "
                      "and {}) must be one placement",
                      molecule.natoms(), basis.natoms(), aux.natoms());
  }
  if (c.n() != nao || h.n() != nao) {
    return err_config("mcscf::df_integral_derivatives: C is {0} x {0} and h {1} x {1}, but the "
                      "basis has {2} functions",
                      c.n(), h.n(), nao);
  }
  if (naux == 0 || b.num_elements() != naux * nao * nao) {
    return err_config(
        "mcscf::df_integral_derivatives: B holds {} doubles, but naux nao^2 is {} x {}^2",
        b.num_elements(), naux, nao);
  }
  if (l == nullptr || h1 == nullptr || w1 == nullptr) {
    return err_config("mcscf::df_integral_derivatives: l, h1 and w1 must be given");
  }
  const size_t nn = nao * nao;
  const size_t stack = n_x * nn;
  const size_t factor = nn * naux;
  size_t max_blk = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [first, last] = integral_atom_functions(aux, a);
    max_blk = std::max(max_blk, last - first);
  }
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (stack > kInt32Max || factor > kInt32Max || 3 * nn * max_blk > kInt32Max) {
    return err_unsupported("mcscf::df_integral_derivatives: {} orbital and {} fitting functions "
                           "over {} coordinates pass the BLAS calls' 32-bit indices",
                           nao, naux, n_x);
  }
  // The call's own buffers at their peak (the MO factor, tau, the metric
  // derivative, one aux atom's block, and the one-electron derivative stacks
  // at their two), past which an allocation would abort (CreatePolicy)
  // rather than return.
  const size_t held = factor + nn + (3 * stack) + (3 * naux * naux) + (3 * nn * max_blk);
  const ScopedDeviceIndex on{res.dev_idx()};
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    if (sizeof(double) * held > free_bytes) {
      return err_unsupported("mcscf::df_integral_derivatives: {} coordinates over {} orbital and "
                             "{} fitting functions need {} bytes of device memory, and device {} "
                             "has {} free",
                             n_x, nao, naux, sizeof(double) * held, res.dev_idx(), free_bytes);
    }
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto nx = static_cast<int32_t>(n_x);
  const auto s_nn = static_cast<int64_t>(nn);
  const double *const cm = c.data();

  // Held across the derivative calls, which take the workspace: the MO
  // factor L[p, q, Q], h_MO and tau^x.
  Buffers::device<double> l_mo(factor, res.device());
  Buffers::device<double> h_mo(nn, res.device());
  Buffers::device<double> tau(stack, res.device());
  {
    // As ThirdDerivative::create: B C over m, then C over n per q, then Q
    // slowest; and C^T h C.
    const auto [half, full] = res.device_regions<double>(factor, factor);
    FBF_TRY(gpu_status(gemm(blas, q * n, n, n, &kOne<double>, b.data(), cm, &kZero<double>, half)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n, n, &kOne<double>, half, q,
        static_cast<int64_t>(naux * nao), cm, n, 0, &kZero<double>, full, q,
        static_cast<int64_t>(naux * nao), n)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n * n,
                                                  q, &kOne<double>, full, q, &kZero<double>, full,
                                                  n * n, l_mo.data(), n * n)));
    FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, h.data(), cm, &kZero<double>, half)));
    FBF_TRY(gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, n, &kOne<double>, cm, half, &kZero<double>, h_mo.data())));
    FBF_TRY(res.sync());
  }
  // out_x = C^T X_x C for every slab of a stack: X_x C, one batched gemm into
  // the workspace, then C^T over all of them side by side.
  const auto to_mo = [&](const double *x, double *out) -> Status {
    auto [y] = res.device_regions<double>(stack);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, x, n, s_nn, cm, n, 0,
        &kZero<double>, y, n, s_nn, nx)));
    return gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, n, n * nx, n, &kOne<double>, cm, y, &kZero<double>, out));
  };
  {
    const Buffers::device<double> sx = FBF_TRY(overlap_derivative(res, basis, classes));
    FBF_TRY(to_mo(sx.data(), tau.data()));
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, static_cast<int32_t>(stack),
                                                  &kNegHalf<double>, tau.data(), 1)));
  }
  {
    Buffers::device<double> hx = FBF_TRY(kinetic_derivative(res, basis, classes));
    {
      const Buffers::device<double> vx =
          FBF_TRY(potential_derivative(res, basis, classes, molecule));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(stack),
                                                    &kOne<double>, vx.data(), 1, hx.data(), 1)));
    }
    FBF_TRY(to_mo(hx.data(), h1));
    FBF_TRY(res.sync());
  }
  // h1^x += tau^x h_MO + h_MO tau^x.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, tau.data(), n, s_nn,
      h_mo.data(), n, 0, &kOne<double>, h1, n, s_nn, nx)));
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, h_mo.data(), n, 0,
      tau.data(), n, s_nn, &kOne<double>, h1, n, s_nn, nx)));

  // The three-center derivative over the MO pairs, one aux atom's block at a
  // time. With X the derivative on m's centre, d(mn|P)/dR_{A,k} = -[m on A]
  // X_mn - [n on A] X_nm + [P on A] (X_mn + X_nm), so with E_A[p][q] =
  // Sum_{m on A, n} C_mp X_mn,P C_nq the MO derivative is S + S^T over the
  // pair, S^{(A, k)} = -E_A + [P on A] Sum_A' E_A'. S goes into w1 as it
  // stands (any order of the pair: only S + S^T is kept).
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(w1, 0, sizeof(double) * factor * n_x, stream)));
  const auto slab = [&](const size_t x) { return w1 + (x * factor); };
  {
    Buffers::device<double> block(3 * nn * max_blk, res.device());
    for (uint32_t aux_atom = 0; aux_atom < n_atom; ++aux_atom) {
      const auto [p0, p1] = integral_atom_functions(aux, aux_atom);
      const size_t nblk = p1 - p0;
      if (nblk == 0) {
        continue;
      }
      FBF_TRY(three_center_derivative(res, basis, classes, aux, aux_atom, block.data()));
      const auto blk = static_cast<int32_t>(nblk);
      const auto rows = static_cast<int32_t>(nblk * nao);
      // H[P', n, p] = Sum_{m on A} X[m][n][P'] C_mp, then G[P', q, p] =
      // Sum_n H[P', n, p] C_nq = E_A[p][q].
      const auto [hh, gg] = res.device_regions<double>(nblk * nn, nblk * nn);
      for (size_t k = 0; k < 3; ++k) {
        const double *const xk = block.data() + (k * nn * nblk);
        for (uint32_t atom = 0; atom < n_atom; ++atom) {
          const auto [a0, a1] = integral_atom_functions(basis, atom);
          if (a1 == a0) {
            continue;
          }
          FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, n, static_cast<int32_t>(a1 - a0),
              &kOne<double>, xk + (a0 * nblk * nao), rows, cm + a0, n, &kZero<double>, hh, rows)));
          FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, n, n, &kOne<double>, hh, blk,
              static_cast<int64_t>(nblk * nao), cm, n, 0, &kZero<double>, gg, blk,
              static_cast<int64_t>(nblk * nao), n)));
          // G is nblk x nmo^2: its transpose is the slab's columns P on the
          // aux atom.
          for (const auto &[x, alpha] : {std::pair{(3 * atom) + k, &kNegOne<double>},
                                         std::pair{(3 * aux_atom) + k, &kOne<double>}}) {
            double *const dst = slab(x) + (p0 * nn);
            FBF_TRY(gpu_status(
                wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n * n, blk,
                                           &kOne<double>, dst, n * n, alpha, gg, blk, dst, n * n)));
          }
        }
      }
    }
  }
  // S + S^T over each pair (I S_Q^T added to S_Q, a batched gemm against a
  // copy), then the fit: U^x = (S + S^T) L_M^-T.
  {
    const auto [copy, ident] = res.device_regions<double>(factor, nn);
    std::vector<double> ident_host(nn, 0.0);
    for (size_t i = 0; i < nao; ++i) {
      ident_host[i * (nao + 1)] = 1.0;
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ident, ident_host.data(), sizeof(double) * nn,
                                           wwr::wwrMemcpyHostToDevice, stream)));
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(copy, slab(x), sizeof(double) * factor,
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n, n, &kOne<double>, ident, n, 0, copy, n,
          s_nn, &kOne<double>, slab(x), n, s_nn, q)));
      FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
          blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
          wwr::WWRBLAS_DIAG_NON_UNIT, n * n, q, &kOne<double>, l, q, slab(x), n * n)));
    }
    FBF_TRY(res.sync());
  }
  // The metric: U^x -= 1/2 L (L_M^-1 M^x L_M^-T), M^x = E + E^T with E
  // metric_derivative's M1[k] (column P = M1[k][P][.]) on A's columns only.
  {
    Buffers::device<double> m1(3 * naux * naux, res.device());
    FBF_TRY(metric_derivative(res, aux, m1.data()));
    const auto [e, g] = res.device_regions<double>(naux * naux, naux * naux);
    for (uint32_t atom = 0; atom < n_atom; ++atom) {
      const auto [q0, q1] = integral_atom_functions(aux, atom);
      for (size_t k = 0; k < 3; ++k) {
        double *const out = slab((3 * atom) + k);
        FBF_TRY(gpu_status(wwr::wwrMemsetAsync(e, 0, sizeof(double) * naux * naux, stream)));
        if (q1 > q0) {
          FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
              e + (q0 * naux), m1.data() + (k * naux * naux) + (q0 * naux),
              sizeof(double) * (q1 - q0) * naux, wwr::wwrMemcpyDeviceToDevice, stream)));
        }
        FBF_TRY(
            gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, q,
                                                  &kOne<double>, e, q, &kOne<double>, e, q, g, q)));
        FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
            blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
            wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, g, q)));
        FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
            blas, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
            wwr::WWRBLAS_DIAG_NON_UNIT, q, q, &kOne<double>, l, q, g, q)));
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                      n * n, q, q, &kNegHalf<double>, l_mo.data(),
                                                      n * n, g, q, &kOne<double>, out, n * n)));
      }
    }
    FBF_TRY(res.sync());
  }
  // The connection: W^x_Q = U^x_Q + tau^x L_Q + L_Q tau^x.
  for (size_t x = 0; x < n_x; ++x) {
    const double *const t = tau.data() + (x * nn);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, t, n, 0, l_mo.data(), n,
        s_nn, &kOne<double>, slab(x), n, s_nn, q)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<double>, l_mo.data(), n, s_nn, t,
        n, 0, &kOne<double>, slab(x), n, s_nn, q)));
  }
  return res.sync();
}

Status ThirdDerivative::apply_integrals(DeviceResources &res, const int32_t k,
                                        const double *const h1, const double *const w1,
                                        double *const y) {
  if (k < 0) {
    return err_config("mcscf::ThirdDerivative::apply_integrals: k must not be negative (got {})",
                      k);
  }
  if (k > 0 && (h1 == nullptr || w1 == nullptr || y == nullptr)) {
    return err_config("mcscf::ThirdDerivative::apply_integrals: h1, w1 and y must be given");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto mo = static_cast<size_t>(nmo_);
  const auto occ = static_cast<size_t>(nocc_);
  const auto nroots = static_cast<size_t>(nroots_);
  const size_t block = static_cast<size_t>(dim_) * nroots;
  const size_t mo2 = mo * mo;
  const size_t lmo = mo * occ * static_cast<size_t>(naux_);
  const size_t pair_len = occ * occ * static_cast<size_t>(naux_);
  const size_t factor = mo2 * static_cast<size_t>(naux_);
  const auto pq = static_cast<int32_t>(occ * occ);
  const int32_t lmo_k = nocc_ * naux_;
  const int32_t nmo = nmo_;
  const int32_t nocc = nocc_;
  const int32_t dim = dim_;
  const int32_t nr = nroots_;
  const auto mo_l = static_cast<int64_t>(nmo_);
  const auto occ_l = static_cast<int64_t>(nocc_);

  // The column's regions, apply's scratch (the two do not run at once).
  double *const m1x = mat_.data(); // G1 at the moved integrals
  double *const mzx = m1x + mo2;   // G2 along z at them
  double *const hkz = mzx + mo2;   // h1 K_z
  double *const t1 = hkz + mo2;
  double *const wo = lmo_.data(); // W's occupied columns
  double *const wkz = wo + lmo;   // W K_z, occupied columns
  double *const woo = pair_.data();
  double *const pwk = woo + pair_len; // K_z^T W + W K_z over the occupied pair
  double *const yw = pwk + pair_len;  // Gamma_0 W_oo
  double *const ywb = yw + pair_len;  // Gamma_0 P_W(K_z) + Gamma_z W_oo
  double *const c1x = ci_.data();     // 2 w H^x c
  double *const czx = c1x + block;    // 2 w (H^x[K_z] c + H^x A_z c)
  double *const w1c = czx + block;
  double *const w2c = w1c + block;
  double *const g_d = w2c + block;
  // small_: 0 unpack's, 1-3 rotate's, 4-5 pull_ci's, 7 X_theta (as apply).
  const size_t nr2 = nroots * nroots;
  double *const x_theta = small_.data() + (7 * nr2);

  const auto mm = [&](const double alpha, const double *a, const double *b, const double beta_v,
                      double *c) {
    return gpu_status(gemm(res.blas(), nmo, &alpha, a, b, &beta_v, c));
  };
  const auto fock_add = [&](const double *lm, const double *ym, double *out) {
    return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), nmo, nocc, lmo_k, &kTwo<double>, lm, ym, &kOne<double>, out));
  };
  const auto h_add = [&](const double *hm, const double *dm, double *out) {
    return gpu_status(gemm(res.blas(), nmo, nocc, nocc, &kTwo<double>, hm, dm, &kOne<double>, out));
  };
  const auto add = [&](const double alpha, const double *a, const double beta_v, const double *b,
                       double *out) {
    return gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                 nr, &alpha, a, dim, &beta_v, b, dim, out, dim));
  };
  // The occupied block of a host nmo x nmo matrix, plus its transpose when
  // @p twice (h1 K_z's symmetric part, times 2) and as it stands otherwise.
  const auto occupied_block = [&](const double *device,
                                  const bool twice) -> Result<std::vector<double>> {
    std::vector<double> host(mo2);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), device, mo2 * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    std::vector<double> out(occ * occ);
    for (size_t j = 0; j < occ; ++j) {
      for (size_t i = 0; i < occ; ++i) {
        out[i + (occ * j)] = twice ? host[i + (mo * j)] + host[j + (mo * i)] : host[i + (mo * j)];
      }
    }
    return out;
  };

  for (int32_t col = 0; col < k; ++col) {
    const double *const h1c = h1 + (static_cast<size_t>(col) * mo2);
    const double *const wc = w1 + (static_cast<size_t>(col) * factor);
    double *const out = y + (static_cast<size_t>(col) * static_cast<size_t>(size_));

    // W's occupied columns and pairs, W K_z, and P_W(K_z) = K_z^T W + W K_z
    // over the occupied pair.
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
        wo, mo * occ * sizeof(double), wc, mo2 * sizeof(double), mo * occ * sizeof(double),
        static_cast<size_t>(naux_), wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(woo, occ * sizeof(double), wo, mo * sizeof(double),
                                             occ * sizeof(double), occ * static_cast<size_t>(naux_),
                                             wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nocc, nmo, &kOne<double>, wc, nmo,
        mo_l * mo_l, k_z_.data(), nmo, 0, &kZero<double>, wkz, nmo, mo_l * occ_l, naux_)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nocc, nocc, nmo, &kOne<double>, k_z_.data(),
        nmo, 0, wo, nmo, mo_l * occ_l, &kZero<double>, pwk, nocc, occ_l * occ_l, naux_)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nocc,
                                                  nocc * naux_, &kOne<double>, wkz, nmo,
                                                  &kOne<double>, pwk, nocc, pwk, nocc)));
    // Y_W = Gamma_0 W_oo; Y_Wb = Gamma_0 P_W(K_z) + Gamma_z W_oo.
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma0_.data(), woo, &kZero<double>, yw)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma0_.data(), pwk, &kZero<double>, ywb)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma_z_.data(), woo, &kOne<double>, ywb)));

    // The orbital covectors, create's M_1 and M_z differentiated along the
    // integrals (occupied columns): G1^x = 2 [W Y_0 + L Y_W + h1 D_0] and
    // G2^x = 2 [W Y_b + L Y_Wb + (W K_z) Y_0 + (L K_z) Y_W + (h1 K_z) D_0 +
    // h1 D_z].
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(m1x, 0, 2 * mo2 * sizeof(double), stream)));
    FBF_TRY(mm(1.0, h1c, k_z_.data(), 0.0, hkz));
    FBF_TRY(fock_add(wo, y0_.data(), m1x));
    FBF_TRY(fock_add(lo_.data(), yw, m1x));
    FBF_TRY(h_add(h1c, d0_.data(), m1x));
    FBF_TRY(fock_add(wo, y_b_.data(), mzx));
    FBF_TRY(fock_add(lo_.data(), ywb, mzx));
    FBF_TRY(fock_add(wkz, y0_.data(), mzx));
    FBF_TRY(fock_add(l_k_z_.data(), yw, mzx));
    FBF_TRY(h_add(hkz, d0_.data(), mzx));
    FBF_TRY(h_add(h1c, d_z_.data(), mzx));

    // The CI covectors: c_1^x = 2 w H^x c and c_z^x = 2 w (H^x[K_z] c +
    // H^x A_z c), H^x over (h1, W L^T + L W^T) and H^x[K_z] its one-index
    // transform by K_z.
    {
      const std::vector<double> h1_oo = FBF_TRY(occupied_block(h1c, false));
      const std::vector<double> hk_oo = FBF_TRY(occupied_block(hkz, true));
      const std::array first{std::pair<const double *, const double *>{woo, p0_.data()},
                             std::pair<const double *, const double *>{p0_.data(), woo}};
      FBF_TRY(set_hamiltonian(res, h1_oo, first, sigma_work_));
      FBF_TRY(sigma_work_.apply(res, nr, roots_.data(), w1c)); // H^x c
      FBF_TRY(sigma_work_.apply(res, nr, rot_z_.data(), w2c)); // H^x A_z c
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, w1c,
                                                    dim, two_w_.data(), 1, c1x, dim)));
      const std::array second{std::pair<const double *, const double *>{pwk, p0_.data()},
                              std::pair<const double *, const double *>{p0_.data(), pwk},
                              std::pair<const double *, const double *>{p_k_z_.data(), woo},
                              std::pair<const double *, const double *>{woo, p_k_z_.data()}};
      FBF_TRY(set_hamiltonian(res, hk_oo, second, sigma_work_));
      FBF_TRY(sigma_work_.apply(res, nr, roots_.data(), w1c)); // H^x[K_z] c
      FBF_TRY(add(1.0, w1c, 1.0, w2c, w1c));
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, w1c,
                                                    dim, two_w_.data(), 1, czx, dim)));
    }

    // The orbital pullback: G2^x along a, and G1^x along (a, z) through
    // (K_a K_z + K_z K_a) / 2, antisymmetrized and packed.
    FBF_TRY(mm(-0.5, m1x, k_z_.data(), 1.0, mzx));
    FBF_TRY(mm(-0.5, k_z_.data(), m1x, 1.0, mzx));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo,
                                                  nmo, &kOne<double>, mzx, nmo, &kNegOne<double>,
                                                  mzx, nmo, t1, nmo)));
    FBF_TRY(response::pack_orbital_pairs(res, nmo, ncore_, ncas_, t1, out));

    // The CI pullback, the pairs (M, V) of M . A_a V: c_z^x - A_z c_1^x / 2
    // with c, c_1^x / 2 with A_z c. The first carries the couplings among
    // the roots at the moved integrals (c_1^x is not along the roots).
    FBF_TRY(rotate(res, d_zp_.data(), theta_z_.data(), nr, c1x, w1c)); // A_z c_1^x
    FBF_TRY(add(1.0, czx, -0.5, w1c, czx));
    FBF_TRY(add(0.5, c1x, 0.0, c1x, w2c));
    const std::array pulls{std::pair<const double *, const double *>{czx, roots_.data()},
                           std::pair<const double *, const double *>{w2c, rot_z_.data()}};
    FBF_TRY(pull_ci(res, pulls, g_d, x_theta));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out + pairs_, g_d, block * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    if (n_theta_ > 0) {
      std::vector<double> x(nr2);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(x.data(), x_theta, nr2 * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
      FBF_TRY(res.sync());
      std::vector<double> packed;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          packed.push_back(x[s + (nroots * r)] - x[r + (nroots * s)]);
        }
      }
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out + pairs_ + block, packed.data(),
                                             packed.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, stream)));
    }
    FBF_TRY(res.sync());
  }
  return {};
}

} // namespace free_breakfast::mcscf
