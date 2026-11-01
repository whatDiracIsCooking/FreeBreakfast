/// @file
/// @brief df_casscf_exy: the fixed-parameter second derivative of a DF-CASSCF
/// energy (casscf_hessian.tex eq. exy) at any active RDMs: the explicit part
/// (the one-electron second derivatives at D^AO, V_nn's and
/// df_casscf_two_electron) and the Lowdin connection's three terms, each a
/// handful of gemms over the MO-basis fit B and the stack of tau^x. With a
/// rotation K, the same at the K-rotated densities (a state average's z_o
/// part of L_xy) by the product rule over those MO-basis quantities.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne, kTwo, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, geam, gemmStridedBatched, wwrMemGetInfo, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.gradient;
import free_breakfast.integrals.one_electron.hessian;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<CasscfExy>
df_casscf_exy(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const Buffers::device<double> &c, const int32_t ncore, const int32_t ncas,
              const std::span<const double> gamma, const std::span<const double> gamma2,
              const Buffers::device<double> &fock_x, const double *const rotation) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const int64_t nocc_wide = int64_t{ncore} + int64_t{ncas};
  const size_t nmo = nao > 0 ? c.num_elements() / nao : 0;
  if (n_atom == 0 || molecule.natoms() != n_atom || aux.natoms() != n_atom || ncore < 0 ||
      ncas < 1 || nao == 0 || c.num_elements() != nao * nmo ||
      nocc_wide > static_cast<int64_t>(nmo) || h.n() != nao ||
      b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_casscf_exy: needs a molecule and two bases on the same atoms, "
                      "ncore >= 0, ncas >= 1, C nao nmo long with ncore + ncas <= nmo, h nao x "
                      "nao and B naux nao^2 long ({}, {} and {} atoms, ncore {}, ncas {}, nao {}, "
                      "naux {}, C {} doubles, h {} x {}, B {} doubles)",
                      molecule.natoms(), n_atom, aux.natoms(), ncore, ncas, nao, naux,
                      c.num_elements(), h.n(), h.n(), b.num_elements());
  }
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t aa = n_act * n_act;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  const size_t mm = nmo * nmo;
  if (gamma.size() != aa || gamma2.size() != aa * aa || fock_x.num_elements() != mm * n_x) {
    return err_config("hessian::df_casscf_exy: RDMs of {} and {} elements and F^[x] of {} for "
                      "ncas {}, nmo {} and {} coordinates",
                      gamma.size(), gamma2.size(), fock_x.num_elements(), ncas, nmo, n_x);
  }
  const auto occ = static_cast<size_t>(nocc_wide);
  const size_t oo = occ * occ;
  // Every matrix below is indexed by int32_t: B over every MO pair, the
  // per-coordinate half transform (naux nocc nmo), the inner dimension of the
  // Gram products over tau (nmo^2) and R (naux nocc^2), and the stacks the
  // batched gemms address one coordinate at a time.
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (naux * mm > kInt32Max || nao * nmo * n_x > kInt32Max || mm * n_x > kInt32Max ||
      oo * oo > kInt32Max || naux * oo > kInt32Max) {
    return err_unsupported("hessian::df_casscf_exy: naux nmo^2 = {}, 3 n_atom nao nmo = {}, 3 "
                           "n_atom nmo^2 = {}, nocc^4 = {} or naux nocc^2 = {} exceeds int32_t",
                           naux * mm, nao * nmo * n_x, mm * n_x, oo * oo, naux * oo);
  }
  const ScopedDeviceIndex on{res.dev_idx()};

  const bool rotated = rotation != nullptr;
  CasscfExy out;
  out.nuclear_repulsion = nuclear_repulsion_hessian(molecule);
  if (rotated) {
    // V_nn does not see the orbitals.
    std::ranges::fill(out.nuclear_repulsion, 0.0);
  }

  // D over the occupied orbitals, 2 on the core diagonal and gamma (as its
  // symmetric part) on the active block, and Gamma over them: the closed-
  // shell core's D (x) D - 1/2 exchange with itself and with gamma, plus the
  // active 2-RDM, 8-fold symmetrized (the 4 + 8 placements of
  // E^(2)'s two-electron part assume it; the energy sees only that part).
  // Column-major, Gamma as the nocc^2 x nocc^2 matrix (p + nocc q, r + nocc s).
  std::vector<double> d_host(oo, 0.0);
  std::vector<double> rdm2(oo * oo, 0.0);
  {
    const auto at = [occ](size_t p, size_t q, size_t r, size_t s) {
      return p + (occ * (q + (occ * (r + (occ * s)))));
    };
    const auto g1 = [&](size_t t, size_t u) {
      return 0.5 * (gamma[(t * n_act) + u] + gamma[(u * n_act) + t]);
    };
    for (size_t i = 0; i < n_core; ++i) {
      d_host[i * (occ + 1)] = 2.0;
    }
    for (size_t t = 0; t < n_act; ++t) {
      for (size_t u = 0; u < n_act; ++u) {
        d_host[(n_core + t) + (occ * (n_core + u))] = g1(t, u);
      }
    }
    std::vector<double> raw(oo * oo, 0.0);
    for (size_t i = 0; i < n_core; ++i) {
      for (size_t j = 0; j < n_core; ++j) {
        raw[at(i, i, j, j)] += 4.0;
        raw[at(i, j, j, i)] -= 2.0;
      }
      for (size_t t = 0; t < n_act; ++t) {
        for (size_t u = 0; u < n_act; ++u) {
          const size_t tt = n_core + t;
          const size_t uu = n_core + u;
          raw[at(i, i, tt, uu)] += 2.0 * g1(t, u);
          raw[at(tt, uu, i, i)] += 2.0 * g1(t, u);
          raw[at(i, tt, uu, i)] -= g1(t, u);
          raw[at(tt, i, i, uu)] -= g1(t, u);
        }
      }
    }
    for (size_t t = 0; t < n_act; ++t) {
      for (size_t u = 0; u < n_act; ++u) {
        for (size_t v = 0; v < n_act; ++v) {
          for (size_t w = 0; w < n_act; ++w) {
            raw[at(n_core + t, n_core + u, n_core + v, n_core + w)] +=
                gamma2[(((((t * n_act) + u) * n_act) + v) * n_act) + w];
          }
        }
      }
    }
    for (size_t s = 0; s < occ; ++s) {
      for (size_t r = 0; r < occ; ++r) {
        for (size_t q = 0; q < occ; ++q) {
          for (size_t p = 0; p < occ; ++p) {
            rdm2[at(p, q, r, s)] =
                0.125 * (raw[at(p, q, r, s)] + raw[at(q, p, r, s)] + raw[at(p, q, s, r)] +
                         raw[at(q, p, s, r)] + raw[at(r, s, p, q)] + raw[at(s, r, p, q)] +
                         raw[at(r, s, q, p)] + raw[at(s, r, q, p)]);
          }
        }
      }
    }
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nao);
  const auto n_mo = static_cast<int32_t>(nmo);
  const auto nocc = static_cast<int32_t>(occ);
  const auto q = static_cast<int32_t>(naux);
  const auto nx = static_cast<int32_t>(n_x);
  const auto n_oo = static_cast<int32_t>(oo);
  const auto n_mm = static_cast<int32_t>(mm);

  // The occupied D, then Gamma, on the device for the whole call. The host
  // vectors outlive their copies: every phase below synchronizes.
  Buffers::device<double> densities(oo + (oo * oo), res.device());
  double *const d_occ = densities.data();
  double *const rdm2_device = d_occ + oo;
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_occ, d_host.data(), sizeof(double) * oo,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rdm2_device, rdm2.data(), sizeof(double) * oo * oo,
                                         wwr::wwrMemcpyHostToDevice, stream)));

  {
    // The explicit one-electron part: h^xy (T's and V's) against
    // D^AO = C_o D C_o^T. S^xy is not here: it is the connection's.
    SphericalMatrixAO d_ao(res, basis);
    if (!rotated) {
      const auto [work] = res.device_regions<double>(oo * nao);
      FBF_TRY(gpu_status(
          gemmwich<wwr::WWRBLAS_OP_T>(blas, n, nocc, c.data(), d_occ, work, d_ao.data())));
    } else {
      // Rotated: D^AO's derivative along C -> C (1 + e K), C~_o D C_o^T +
      // C_o D C~_o^T with C~_o = C K's occupied columns.
      const auto [c_rot, half, a_ao] = res.device_regions<double>(nao * occ, nao * occ, nao * nao);
      FBF_TRY(gpu_status(
          gemm(blas, n, nocc, n_mo, &kOne<double>, c.data(), rotation, &kZero<double>, c_rot)));
      FBF_TRY(
          gpu_status(gemm(blas, n, nocc, nocc, &kOne<double>, c_rot, d_occ, &kZero<double>, half)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n, n, nocc, &kOne<double>, half, c.data(), &kZero<double>, a_ao)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n,
                                                    n, &kOne<double>, a_ao, n, &kOne<double>, a_ao,
                                                    n, d_ao.data(), n)));
    }
    out.one_electron = FBF_TRY(kinetic_hessian(res, basis, classes, d_ao));
    const std::vector<double> potential =
        FBF_TRY(potential_hessian(res, basis, classes, molecule, d_ao));
    for (size_t i = 0; i < out.one_electron.size(); ++i) {
      out.one_electron[i] += potential[i];
    }
  }
  out.two_electron = FBF_TRY(df_casscf_two_electron(res, basis, classes, aux, b, l, c, ncore, ncas,
                                                    gamma, gamma2, rotation));

  // What the connection terms hold across calls that take the workspace
  // (overlap_derivative, b_mo, overlap_hessian), so their own: tau^x for
  // every x, B over every MO pair and over the occupied ones, G = B_oo Gamma,
  // h_MO, F, -sym(F), (-sym F) tau^x for every x, and E^(2)'s stacks: R^x
  // (naux nocc^2) and R^x Gamma, V^x (nmo nocc) for every x, and the four
  // n_x^2 products. Past the device's free memory an allocation would abort
  // (CreatePolicy), so the total is checked first, with S^x's AO stack, which
  // is freed once tau is made, and the larger of its MO transform's and
  // E^(2)'s per-coordinate workspace beside it.
  const size_t held_count = (2 * mm * n_x) + (naux * mm) + (2 * naux * oo) + (3 * mm) +
                            (2 * n_x * naux * oo) + (n_x * nmo * occ) + (4 * n_x * n_x);
  // A rotation adds B's derivative over every pair and the occupied ones,
  // G' and G~, h K and h', F_L, and tau'^x, V'^x and R'^x for every x; and
  // per coordinate Y' and tau^x K.
  const size_t rot_count = rotated ? (naux * mm) + (2 * naux * oo) + (naux * nmo * occ) + (3 * mm) +
                                         (mm * n_x) + (n_x * nmo * occ) + (n_x * naux * oo)
                                   : 0;
  const size_t transient = std::max(2 * n_x * nao * nao + nmo * nao * n_x,
                                    (naux * occ * nmo) + (nmo * occ) + (nao * nmo) +
                                        (rotated ? (naux * occ * nmo) + mm : 0));
  const size_t bytes = sizeof(double) * (held_count + rot_count + transient);
  {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    if (bytes > free_bytes) {
      return err_unsupported("hessian::df_casscf_exy: the connection terms over {} orbitals, {} "
                             "fitting functions, {} occupied orbitals and {} coordinates (B over "
                             "every MO pair, naux nmo^2 = {} doubles, and every coordinate's "
                             "tau^x and R^x at once) need {} bytes of device memory, and device "
                             "{} has {} free",
                             nmo, naux, occ, n_x, naux * mm, bytes, res.dev_idx(), free_bytes);
    }
  }
  Buffers::device<double> held(held_count, res.device());
  double *const tau = held.data();
  double *const f_tau = tau + (mm * n_x);
  double *const b_all = f_tau + (mm * n_x);
  double *const b_oo = b_all + (naux * mm);
  double *const g_fit = b_oo + (naux * oo);
  double *const h_mo = g_fit + (naux * oo);
  double *const fock = h_mo + mm;
  double *const neg_fock_sym = fock + mm;
  double *const r = neg_fock_sym + mm;
  double *const r_gamma = r + (n_x * naux * oo);
  double *const v = r_gamma + (n_x * naux * oo);
  double *const products = v + (n_x * nmo * occ);
  double *const p_square = products;
  double *const p_fock = p_square + (n_x * n_x);
  double *const p_twice = p_fock + (n_x * n_x);
  double *const p_twice_r = p_twice + (n_x * n_x);
  Buffers::device<double> rot_held(std::max<size_t>(rot_count, 1), res.device());
  double *const b_dot = rot_held.data();
  double *const b_dot_oo = b_dot + (naux * mm);
  double *const g_dot = b_dot_oo + (naux * oo);
  double *const g_tilde = g_dot + (naux * oo);
  double *const h_k = g_tilde + (naux * nmo * occ);
  double *const h_dot = h_k + mm;
  double *const fock_rot = h_dot + mm;
  double *const tau_dot = fock_rot + mm;
  double *const v_dot = tau_dot + (mm * n_x);
  double *const r_dot = v_dot + (n_x * nmo * occ);

  {
    // tau^x = -1/2 C^T S^x C for every x: C^T [S^0 S^1 ...] as one gemm,
    // then each slab times C, one strided-batched gemm.
    const Buffers::device<double> s_ao = FBF_TRY(overlap_derivative(res, basis, classes));
    const auto [half] = res.device_regions<double>(nmo * nao * n_x);
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, n * nx, n, &kNegHalf<double>, c.data(),
                                               s_ao.data(), &kZero<double>, half)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_mo, n_mo, n, &kOne<double>, half, n_mo,
        static_cast<int64_t>(nmo * nao), c.data(), n, 0, &kZero<double>, tau, n_mo,
        static_cast<int64_t>(mm), nx)));
    FBF_TRY(res.sync());
  }

  // B in the MO basis, every pair (Q + naux (p + nmo q), symmetric in p and
  // q) and the occupied pairs; G = B_oo Gamma, G_Q,ij = Sum_rs Gamma_ijrs
  // B_Q,rs, at Q + naux (i + nocc j) and symmetric in i and j.
  const OrbitalRange all{.first = 0, .count = n_mo};
  const OrbitalRange occupied{.first = 0, .count = nocc};
  FBF_TRY(b_mo(res, q, n, n_mo, b.data(), c.data(), all, all, b_all));
  FBF_TRY(b_mo(res, q, n, n_mo, b.data(), c.data(), occupied, occupied, b_oo));
  FBF_TRY(gpu_status(
      gemm(blas, q, n_oo, n_oo, &kOne<double>, b_oo, rdm2_device, &kZero<double>, g_fit)));
  {
    const auto [work] = res.device_regions<double>(nao * nmo);
    FBF_TRY(gpu_status(gemmwich(blas, n_mo, n, c.data(), h.data(), work, h_mo)));
  }

  // F, eq. fock, at these RDMs: F_pq = Sum_r h_pr D_rq + Sum_rst (pr|st)
  // Gamma_qrst, zero in the virtual columns. Its two-electron part is
  // Sum_Q,r B_Q,rp G_Q,qr: B's rows (Q, r < nocc) of column p, leading
  // dimension naux nmo, against G.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(fock, 0, sizeof(double) * mm, stream)));
  FBF_TRY(
      gpu_status(gemm(blas, n_mo, nocc, nocc, &kOne<double>, h_mo, d_occ, &kZero<double>, fock)));
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n_mo,
                                                nocc, q * nocc, &kOne<double>, b_all, q * n_mo,
                                                g_fit, q * nocc, &kOne<double>, fock, n_mo)));
  if (rotated) {
    // Everything below at the rotated densities is the product rule over the
    // MO-basis quantities, each X moving as [X, K] = X K - K X along
    // C -> C (1 + e K) with the RDMs held (README.md):
    //   B' = B K - K B (B K one gemm over B's (Q, p) rows; K B_Q = slab_q K^T
    //   per column q), h' = h K + (h K)^T, tau^x' = tau^x K + (tau^x K)^T,
    //   G' = B'_oo Gamma, and F's at the inner indices:
    //   F_L = F K^T + h K D + Sum_Q,a B_Q,ap G~_Q,aq + Sum_Q,r B_Q,rp G'_Q,rq,
    //   G~_Q,aq = Sum_r K_ar G_Q,rq,
    // which replaces F (only F's outer index is held, as mcscf::df_gradient's
    // W takes its z_o).
    FBF_TRY(gpu_status(
        gemm(blas, q * n_mo, n_mo, n_mo, &kOne<double>, b_all, rotation, &kZero<double>, b_dot)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n_mo, n_mo, &kNegOne<double>, b_all, q,
        static_cast<int64_t>(naux * nmo), rotation, n_mo, 0, &kOne<double>, b_dot, q,
        static_cast<int64_t>(naux * nmo), n_mo)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q * nocc, nocc, &kOne<double>, b_dot, q * n_mo,
        &kZero<double>, b_dot, q * n_mo, b_dot_oo, q * nocc)));
    FBF_TRY(gpu_status(
        gemm(blas, q, n_oo, n_oo, &kOne<double>, b_dot_oo, rdm2_device, &kZero<double>, g_dot)));
    FBF_TRY(gpu_status(
        gemm(blas, n_mo, n_mo, n_mo, &kOne<double>, h_mo, rotation, &kZero<double>, h_k)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_mo,
                                                  n_mo, &kOne<double>, h_k, n_mo, &kOne<double>,
                                                  h_k, n_mo, h_dot, n_mo)));
    // G~ per occupied column q: G_q (naux x nocc) times K[:, occ]^T.
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n_mo, nocc, &kOne<double>, g_fit, q,
        static_cast<int64_t>(naux * occ), rotation, n_mo, 0, &kZero<double>, g_tilde, q,
        static_cast<int64_t>(naux * nmo), nocc)));
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_mo,
                                                  n_mo, nocc, &kOne<double>, fock, n_mo, rotation,
                                                  n_mo, &kZero<double>, fock_rot, n_mo)));
    FBF_TRY(gpu_status(
        gemm(blas, n_mo, nocc, nocc, &kOne<double>, h_k, d_occ, &kOne<double>, fock_rot)));
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n_mo, nocc, q * n_mo, &kOne<double>, b_all,
        q * n_mo, g_tilde, q * n_mo, &kOne<double>, fock_rot, n_mo)));
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n_mo,
                                                  nocc, q * nocc, &kOne<double>, b_all, q * n_mo,
                                                  g_dot, q * nocc, &kOne<double>, fock_rot, n_mo)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fock, fock_rot, sizeof(double) * mm,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    const auto [tk] = res.device_regions<double>(mm);
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(gpu_status(gemm(blas, n_mo, n_mo, n_mo, &kOne<double>, tau + (x * mm), rotation,
                              &kZero<double>, tk)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_mo, n_mo, &kOne<double>, tk, n_mo,
          &kOne<double>, tk, n_mo, tau_dot + (x * mm), n_mo)));
    }
  }
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n_mo, n_mo, &kNegHalf<double>, fock, n_mo,
      &kNegHalf<double>, fock, n_mo, neg_fock_sym, n_mo)));

  // 2 Sum F tau^xy, tau^xy = -1/2 S^xy + 3/8 (S^x S^y + S^y S^x). The S^xy
  // part is -Sum (C sym(F) C^T) S^xy, overlap_hessian at -C sym(F) C^T. With
  // tau^x = -1/2 S^x the 3/8 part is 3/2 Sum F (tau^x tau^y + tau^y tau^x) =
  // 3 <sym(F) tau^x + tau^x sym(F), tau^y> = 6 <sym(F) tau^x, tau^y> (tau^x
  // and tau^y symmetric): the Gram matrix of the stacks sym(F) tau^x and
  // tau^y, the first one gemm over all x.
  std::vector<double> overlap_part;
  {
    SphericalMatrixAO m_ao(res, basis);
    {
      const auto [work] = res.device_regions<double>(nmo * nao);
      FBF_TRY(gpu_status(
          gemmwich<wwr::WWRBLAS_OP_T>(blas, n, n_mo, c.data(), neg_fock_sym, work, m_ao.data())));
    }
    overlap_part = FBF_TRY(overlap_hessian(res, basis, classes, m_ao));
  }
  FBF_TRY(gpu_status(
      gemm(blas, n_mo, n_mo * nx, n_mo, &kOne<double>, neg_fock_sym, tau, &kZero<double>, f_tau)));
  const double three_eighths = -6.0;
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, n_mm, &three_eighths, f_tau, tau,
                                             &kZero<double>, p_square)));

  // 2 Sum (F^[x] tau^y + F^[y] tau^x): the Gram matrix <F^[x], tau^y>, and
  // twice it plus its transpose on the host.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, n_mm, &kOne<double>, fock_x.data(), tau,
                                             &kZero<double>, p_fock)));

  // E^(2)[tau^x, tau^y], the energy at the integrals one-index transformed
  // by tau^x on one index and tau^y on another (README.md):
  //
  //   E^(2) = 2 <D, tau^xT h tau^y> + 2 X1 + 4 X2,
  //   X1 = Sum Gamma_pqrs tau^x_ap tau^y_bq (ab|rs) = <tau^y_occ, W^x>,
  //   W^x = Sum_Q B_Q tau^x_occ G_Q,
  //   X2 = Sum Gamma_pqrs tau^x_ap tau^y_cr (aq|cs) = <R^x, R^y Gamma>,
  //   R^x_Q,ip = (B_Q tau^x_occ)_pi over occupied p,
  //
  // Gamma being 8-fold symmetric and nonzero only over the occupied
  // orbitals. Per x, Y_Q = B_Q tau^x_occ (one gemm per orbital p of B's
  // column block, batched), stored (Q, i, p) at Q + naux (i + nocc p), so R^x
  // is its first naux nocc^2 doubles and W^x = Y^T G one gemm; then
  // V^x = 2 h tau^x_occ D + 2 W^x. E^(2) is <V^x, tau^y_occ> + 4 <R^x, R^y
  // Gamma>, two Gram matrices.
  {
    const auto [y_half, h_tau, y_dot] =
        res.device_regions<double>(naux * occ * nmo, nmo * occ, rotated ? naux * occ * nmo : 1);
    for (size_t x = 0; x < n_x; ++x) {
      const double *const tau_x = tau + (x * mm);
      double *const v_x = v + (x * nmo * occ);
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, nocc, n_mo, &kOne<double>, b_all, q,
          static_cast<int64_t>(naux * nmo), tau_x, n_mo, 0, &kZero<double>, y_half, q,
          static_cast<int64_t>(naux * occ), n_mo)));
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(r + (x * naux * oo), y_half, sizeof(double) * naux * oo,
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, nocc, q * nocc, &kTwo<double>, y_half,
                                                 g_fit, &kZero<double>, v_x)));
      FBF_TRY(gpu_status(
          gemm(blas, n_mo, nocc, n_mo, &kOne<double>, h_mo, tau_x, &kZero<double>, h_tau)));
      FBF_TRY(gpu_status(
          gemm(blas, n_mo, nocc, nocc, &kTwo<double>, h_tau, d_occ, &kOne<double>, v_x)));
      if (rotated) {
        // V'^x = 2 Y'^T G + 2 Y^T G' + 2 h tau'^x D + 2 h' tau^x D, with
        // Y' = B' tau^x_occ + B tau'^x_occ, whose first naux nocc^2 are R'^x.
        const double *const tau_dot_x = tau_dot + (x * mm);
        double *const v_dot_x = v_dot + (x * nmo * occ);
        for (const auto &[b_from, t_from, beta] :
             {std::tuple{static_cast<const double *>(b_dot), tau_x, &kZero<double>},
              std::tuple{static_cast<const double *>(b_all), tau_dot_x, &kOne<double>}}) {
          FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
              blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, nocc, n_mo, &kOne<double>, b_from, q,
              static_cast<int64_t>(naux * nmo), t_from, n_mo, 0, beta, y_dot, q,
              static_cast<int64_t>(naux * occ), n_mo)));
        }
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_dot + (x * naux * oo), y_dot,
                                               sizeof(double) * naux * oo,
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, nocc, q * nocc, &kTwo<double>, y_dot,
                                                   g_fit, &kZero<double>, v_dot_x)));
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_mo, nocc, q * nocc, &kTwo<double>,
                                                   y_half, g_dot, &kOne<double>, v_dot_x)));
        for (const auto &[h_from, t_from] :
             {std::pair{static_cast<const double *>(h_mo), tau_dot_x},
              std::pair{static_cast<const double *>(h_dot), tau_x}}) {
          FBF_TRY(gpu_status(
              gemm(blas, n_mo, nocc, n_mo, &kOne<double>, h_from, t_from, &kZero<double>, h_tau)));
          FBF_TRY(gpu_status(
              gemm(blas, n_mo, nocc, nocc, &kTwo<double>, h_tau, d_occ, &kOne<double>, v_dot_x)));
        }
      }
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n_oo, n_oo, &kOne<double>, r, q,
        static_cast<int64_t>(naux * oo), rdm2_device, n_oo, 0, &kZero<double>, r_gamma, q,
        static_cast<int64_t>(naux * oo), nx)));
    // tau^y_occ is the first nmo nocc doubles of slab y: leading dimension
    // nmo^2.
    const double four = 4.0;
    if (!rotated) {
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nx,
                                                    nx, n_mo * nocc, &kOne<double>, v, n_mo * nocc,
                                                    tau, n_mm, &kZero<double>, p_twice, nx)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, q * n_oo, &four, r, r_gamma,
                                                 &kZero<double>, p_twice_r)));
    } else {
      // E^(2)' = <V'^x, tau^y> + <V^x, tau'^y> + 4 (<R'^x, R^y Gamma> + its
      // transpose), the transpose added on the host.
      for (const auto &[v_from, t_from, beta] :
           {std::tuple{static_cast<const double *>(v_dot), static_cast<const double *>(tau),
                       &kZero<double>},
            std::tuple{static_cast<const double *>(v), static_cast<const double *>(tau_dot),
                       &kOne<double>}}) {
        FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
            blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nx, nx, n_mo * nocc, &kOne<double>, v_from,
            n_mo * nocc, t_from, n_mm, beta, p_twice, nx)));
      }
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nx, nx, q * n_oo, &four, r_dot, r_gamma,
                                                 &kZero<double>, p_twice_r)));
    }
  }

  std::vector<double> host(4 * n_x * n_x);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), products, sizeof(double) * host.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  const size_t nn_x = n_x * n_x;
  out.connection_second.resize(nn_x);
  out.connection_fock.resize(nn_x);
  out.connection_twice.resize(nn_x);
  out.exy.resize(nn_x);
  for (size_t y = 0; y < n_x; ++y) {
    for (size_t x = 0; x < n_x; ++x) {
      const size_t xy = x + (n_x * y);
      const size_t yx = y + (n_x * x);
      out.connection_second[xy] = overlap_part[xy] + host[xy];
      out.connection_fock[xy] = 2.0 * (host[nn_x + xy] + host[nn_x + yx]);
      out.connection_twice[xy] =
          host[(2 * nn_x) + xy] + host[(3 * nn_x) + xy] + (rotated ? host[(3 * nn_x) + yx] : 0.0);
    }
  }
  for (size_t i = 0; i < nn_x; ++i) {
    out.exy[i] = out.one_electron[i] + out.nuclear_repulsion[i] + out.two_electron.total[i] +
                 out.connection_second[i] + out.connection_fock[i] + out.connection_twice[i];
  }
  return out;
}

} // namespace free_breakfast::hessian
