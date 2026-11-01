/// @file
/// @brief df_casscf_two_electron: the CASSCF fitted densities' second-
/// derivative contractions (mcscf::df_gradient's Gam and W at one root's
/// RDMs), then the fit's cross term split into its J, K and active pieces,
/// one occupied atom block of L^-1 R^x built per walk over the aux atoms and
/// contracted one pair of atom blocks at a time. With a rotation K, their
/// derivative along C -> C (1 + e K): the product rule on every occupied C,
/// the atom blocks over [C_o | C K_o].

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne, kHalf, kNegHalf, kNegTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, geam, trsm, gemmStridedBatched, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.hessian;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import :atom_block;

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<CasscfTwoElectron>
df_casscf_two_electron(DeviceResources &res, const BasisSet &basis,
                       const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                       const Buffers::device<double> &b, const double *l,
                       const Buffers::device<double> &c, const int32_t ncore, const int32_t ncas,
                       const std::span<const double> gamma, const std::span<const double> gamma2,
                       const double *const rotation) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const int64_t nocc_wide = int64_t{ncore} + int64_t{ncas};
  if (n_atom == 0 || aux.natoms() != n_atom || ncore < 0 || ncas < 1 ||
      nocc_wide > static_cast<int64_t>(nao) ||
      c.num_elements() < nao * static_cast<size_t>(nocc_wide) ||
      b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_casscf_two_electron: needs two bases placed on the same atoms, "
                      "ncore >= 0, ncas >= 1, ncore + ncas <= nao, C at least nao (ncore + ncas) "
                      "and B naux nao^2 long ({} and {} atoms, ncore {}, ncas {}, nao {}, naux {}, "
                      "C {} doubles, B {} doubles)",
                      n_atom, aux.natoms(), ncore, ncas, nao, naux, c.num_elements(),
                      b.num_elements());
  }
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t aa = n_act * n_act;
  if (gamma.size() != aa || gamma2.size() != aa * aa) {
    return err_config("hessian::df_casscf_two_electron: RDMs of {} and {} elements for ncas {}",
                      gamma.size(), gamma2.size(), ncas);
  }
  // With a rotation K (nmo x nmo), C must be every orbital: C~ = C K mixes
  // them all into the occupied columns.
  const size_t nmo = rotation == nullptr ? 0 : c.num_elements() / nao;
  if (rotation != nullptr &&
      (c.num_elements() != nao * nmo || nmo < static_cast<size_t>(nocc_wide))) {
    return err_config("hessian::df_casscf_two_electron: a rotation needs C over every orbital, "
                      "nao x nmo with nmo >= ncore + ncas (C {} doubles, nao {})",
                      c.num_elements(), nao);
  }
  const auto nocc = static_cast<int32_t>(nocc_wide);
  const auto occ = static_cast<size_t>(nocc);
  const size_t oo = occ * occ;
  const size_t nn = nao * nao;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  size_t nblk_max = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_function_range(aux, a);
    nblk_max = std::max(nblk_max, p1 - p0);
  }
  // R^x over the occupied pairs for one atom's three coordinates: one atom
  // block. Its core rows times G are the K piece's half-transform.
  const size_t block = 3 * naux * oo;
  const size_t k_half = 3 * naux * n_core * occ;
  const size_t act_block = 3 * naux * aa;
  const size_t y_count = std::max(naux * std::max(n_core, n_act) * nao, nao * occ);
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  // A rotation's extended atom block is over twice the occupied orbitals.
  const size_t widest_block = (rotation == nullptr ? 1 : 4) * block;
  if (naux * nn > kInt32Max || widest_block > kInt32Max || y_count > kInt32Max ||
      aa * aa > kInt32Max) {
    return err_unsupported(
        "hessian::df_casscf_two_electron: naux nao^2 = {}, an atom block 3 naux "
        "(nocc or 2 nocc)^2 = {}, naux max(ncore, ncas) nao = {} or ncas^4 = {} exceeds "
        "int32_t",
        naux * nn, widest_block, y_count, aa * aa);
  }
  // Held for the whole call: the fitted occupied pairs c, the three occupied
  // weights (G_J = diag(2_core, gamma), G_a = diag(0_core, gamma) and
  // G_K = diag(1_core, gamma)) and Gamma. The explicit part's: Gam, W, Y, V,
  // c's core rows, the active pairs and Z, d, d_a, D and D_a. The cross terms':
  // M1, one aux atom's derivative block X, two atom blocks, the K half-
  // transform, three active blocks, the Coulomb vectors and the products. All
  // are held across calls that take the workspace (b_mo, the Hessian and
  // derivative integrals), so they are the call's own (docs/style.md "Scratch
  // memory"); past the device's memory an allocation would abort
  // (CreatePolicy), not return.
  const size_t held_count = (naux * oo) + (3 * oo) + (aa * aa);
  const size_t explicit_count = (naux * nn) + (naux * naux) + y_count + (2 * naux * n_core * occ) +
                                (2 * naux * aa) + (2 * naux) + (2 * nn);
  const size_t products = (9 * oo) + 36 + 9;
  const size_t cross_count = (3 * naux * naux) + (3 * nn * nblk_max) + (2 * block) + k_half +
                             (3 * act_block) + (2 * 6 * naux) + products;
  // A rotation adds [C_o | C~_o], the fitted pairs over it and their
  // derivative c'; the product rule's dotted explicit intermediates (a second
  // Y, V', c'_core, c'_act, Z', d', d_a', D', D_a'); and per atom block its
  // derivative r', t' and act', beside the extended block it is cut from
  // (four blocks).
  const bool rotated = rotation != nullptr;
  const size_t held_rot = rotated ? (2 * nao * occ) + (5 * naux * oo) : 0;
  const size_t explicit_rot =
      rotated ? y_count + (2 * naux * n_core * occ) + (2 * naux * aa) + (2 * naux) + (2 * nn) : 0;
  const size_t cross_rot = rotated ? (6 * block) + (2 * 6 * naux) + (2 * act_block) : 0;
  const size_t bytes =
      sizeof(double) *
      (held_count + held_rot + std::max(explicit_count + explicit_rot, cross_count + cross_rot));
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (bytes > free_bytes) {
    return err_unsupported("hessian::df_casscf_two_electron: the fitted densities and a pair of "
                           "atom blocks of the cross term (each 3 naux nocc^2 = {} doubles) over "
                           "{} orbital and {} fitting functions and {} occupied orbitals need {} "
                           "bytes of device memory, and device {} has {} free",
                           block, nao, naux, occ, bytes, res.dev_idx(), free_bytes);
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto n_oo = static_cast<int32_t>(oo);
  const auto n_aa = static_cast<int32_t>(aa);
  const double *const c_occ = c.data();
  const double *const c_act = c_occ + (nao * n_core);

  Buffers::device<double> held(held_count, res.device());
  double *const fitted = held.data();
  double *const g_j = fitted + (naux * oo);
  double *const g_a = g_j + oo;
  double *const g_k = g_a + oo;
  double *const gamma2_device = g_k + oo;
  // The weights, nocc x nocc column-major, then Gamma as it stands (row-major
  // (t, u, v, w), which column-major is the ncas^2 square (w v, u t), the
  // same matrix for a real 2-RDM: mcscf::df_gradient reads it so). Read by
  // the copy below, which the stream has finished by the first sync.
  std::vector<double> weights_host((3 * oo) + (aa * aa), 0.0);
  for (size_t i = 0; i < n_core; ++i) {
    weights_host[i * (occ + 1)] = 2.0;
    weights_host[(2 * oo) + (i * (occ + 1))] = 1.0;
  }
  for (size_t t = 0; t < n_act; ++t) {
    for (size_t u = 0; u < n_act; ++u) {
      const size_t at = (n_core + t) + (occ * (n_core + u));
      for (size_t which = 0; which < 3; ++which) {
        weights_host[(which * oo) + at] = gamma[(t * n_act) + u];
      }
    }
  }
  std::ranges::copy(gamma2, weights_host.begin() + static_cast<std::ptrdiff_t>(3 * oo));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(g_j, weights_host.data(), sizeof(double) * weights_host.size(),
                                     wwr::wwrMemcpyHostToDevice, stream)));
  // c_P,ab = (L^-T B_ab)_P over the occupied a, b: the naux x nocc^2 matrix
  // at P + naux (a + nocc b), symmetric in a and b.
  const OrbitalRange occupied{.first = 0, .count = nocc};
  FBF_TRY(b_mo(res, q, n, nocc, b.data(), c_occ, occupied, occupied, fitted));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, n_oo, &kOne<double>, l, q, fitted, q)));
  // With a rotation K: C_ext = [C_o | C~_o], C~_o = C K's occupied columns
  // (the derivative of C_o along C -> C (1 + e K)); the fitted pairs over it,
  // naux x (2 nocc)^2 at P + naux (a + 2 nocc b); and their derivative
  // c' = c_o~o + c_oo~, in fitted's layout.
  Buffers::device<double> rot_held(std::max<size_t>(held_rot, 1), res.device());
  double *const c_ext = rot_held.data();
  double *const fitted_ext = c_ext + (2 * nao * occ);
  double *const fitted_dot = fitted_ext + (4 * naux * oo);
  const double *const c_occ_rot = c_ext + (nao * occ);
  const double *const c_act_rot = c_occ_rot + (nao * n_core);
  const int32_t ext_occ = 2 * nocc;
  const auto ld_ext = static_cast<int32_t>(2 * naux * occ);
  if (rotated) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_ext, c_occ, sizeof(double) * nao * occ,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(gemm(blas, n, nocc, static_cast<int32_t>(nmo), &kOne<double>, c.data(),
                            rotation, &kZero<double>, c_ext + (nao * occ))));
    const OrbitalRange extended{.first = 0, .count = ext_occ};
    FBF_TRY(b_mo(res, q, n, ext_occ, b.data(), c_ext, extended, extended, fitted_ext));
    FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
        blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
        wwr::WWRBLAS_DIAG_NON_UNIT, q, 4 * n_oo, &kOne<double>, l, q, fitted_ext, q)));
    // c_o~o: the rows naux nocc onward of the first nocc columns; c_oo~: the
    // first naux nocc rows of the columns nocc onward.
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q * nocc, nocc, &kOne<double>,
        fitted_ext + (naux * occ), ld_ext, &kOne<double>,
        fitted_ext + (static_cast<size_t>(ld_ext) * occ), ld_ext, fitted_dot, q * nocc)));
  }
  // The active pairs (t, u) of an occupied naux x nocc^2 slab: its rows
  // naux ncore onward of columns ncore onward, leading dimension naux nocc.
  const size_t act_offset = (naux * n_core) + (naux * occ * n_core);
  const auto ld_occ = static_cast<int32_t>(naux * occ);
  const auto act_rows = static_cast<int32_t>(naux * n_act);

  CasscfTwoElectron out;
  if (!rotated) {
    // The explicit part, mcscf::df_gradient's Gam and W at these RDMs. J:
    // d = M^-1 u(D) = c . G_J and d_a = c . G_a, Gam = d vec(D)^T -
    // d_a vec(D_a)^T and W = -1/2 d d^T + 1/2 d_a d_a^T. K: V = c_core G_K
    // (P + naux (i + ncore q)), W += V c_core^T and Gam += -2 Sum_i C_mi
    // Sum_q V_P,iq C_nq. Active: Z = c_act Gamma, W += -1/2 Z c_act^T and
    // Gam += Sum_tu C_mt C_nu Z_P,tu.
    Buffers::device<double> scratch(explicit_count, res.device());
    double *const gam = scratch.data();
    double *const w = gam + (naux * nn);
    double *const y = w + (naux * naux);
    double *const v = y + y_count;
    double *const fitted_core = v + (naux * n_core * occ);
    double *const fitted_act = fitted_core + (naux * n_core * occ);
    double *const z = fitted_act + (naux * aa);
    double *const d = z + (naux * aa);
    double *const d_a = d + naux;
    double *const density = d_a + naux;
    double *const density_a = density + nn;
    for (const auto &[weight, fit, ao] :
         {std::tuple{g_j, d, density}, std::tuple{g_a, d_a, density_a}}) {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, n_oo, &kOne<double>,
                                                    fitted, q, weight, 1, &kZero<double>, fit, 1)));
      // C_o G C_o^T through Y as scratch.
      FBF_TRY(
          gpu_status(gemm(blas, n, nocc, nocc, &kOne<double>, c_occ, weight, &kZero<double>, y)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, n, nocc, &kOne<double>,
                                                                    y, c_occ, &kZero<double>, ao)));
    }
    const auto n_nn = static_cast<int32_t>(nn);
    FBF_TRY(gpu_status(gemm(blas, q, n_nn, 1, &kOne<double>, d, density, &kZero<double>, gam)));
    FBF_TRY(
        gpu_status(gemm(blas, q, n_nn, 1, &kNegOne<double>, d_a, density_a, &kOne<double>, gam)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, &kNegHalf<double>,
                                                                  d, d, &kZero<double>, w)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, &kHalf<double>,
                                                                  d_a, d_a, &kOne<double>, w)));
    if (ncore > 0) {
      const auto rows = static_cast<int32_t>(naux * n_core);
      // c's core rows, packed: (P, i) rows, every occupied column.
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, nocc, &kOne<double>, fitted, ld_occ,
          &kZero<double>, fitted, ld_occ, fitted_core, rows)));
      FBF_TRY(gpu_status(
          gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core, g_k, &kZero<double>, v)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, ncore * nocc, &kOne<double>, v, fitted_core, &kOne<double>, w)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v, c_occ, &kZero<double>, y)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncore, &kNegTwo<double>, y, q,
          int64_t{rows}, c_occ, n, 0, &kOne<double>, gam, q * n, static_cast<int64_t>(naux), n)));
    }
    FBF_TRY(gpu_status(
        wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, act_rows, ncas,
                                   &kOne<double>, fitted + act_offset, ld_occ, &kZero<double>,
                                   fitted + act_offset, ld_occ, fitted_act, act_rows)));
    FBF_TRY(gpu_status(
        gemm(blas, q, n_aa, n_aa, &kOne<double>, fitted_act, gamma2_device, &kZero<double>, z)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, n_aa, &kNegHalf<double>, z, fitted_act, &kOne<double>, w)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, act_rows, n, ncas, &kOne<double>, z, c_act, &kZero<double>, y)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, ncas, &kOne<double>, y, q,
        int64_t{act_rows}, c_act, n, 0, &kOne<double>, gam, q * n, static_cast<int64_t>(naux), n)));
    out.three_center = FBF_TRY(three_center_hessian(res, basis, classes, aux, gam));
    out.metric = FBF_TRY(metric_hessian(res, aux, w));
  } else {
    // The same at the rotated densities: the product rule over every C_o of
    // the explicit part, each in turn C~_o (mcscf::df_gradient's orbital
    // response, README.md), with c' the fitted pairs' derivative:
    //   J: Gam' = d' D^T + d D'^T - d_a' D_a^T - d_a D_a'^T,
    //      W' = -1/2 (d' d^T + d d'^T) + 1/2 (d_a' d_a^T + d_a d_a'^T),
    //      d' = c' . G, D' = C~_o G C_o^T + C_o G C~_o^T;
    //   K: V' = c'_core G_K, W' += V' c_core^T + V c'_core^T,
    //      Gam' += -2 Sum_i [C_mi (V' C_o^T + V C~_o^T)_P,in + C~_mi (V C_o^T)_P,in];
    //   active: Z' = c'_act Gamma, W' += -1/2 (Z' c_act^T + Z c'_act^T),
    //      Gam' += Sum_t [C_mt (Z' C_a^T + Z C~_a^T)_P,tn + C~_mt (Z C_a^T)_P,tn].
    Buffers::device<double> scratch(explicit_count + explicit_rot, res.device());
    double *const gam = scratch.data();
    double *const w = gam + (naux * nn);
    double *const y = w + (naux * naux);
    double *const y2 = y + y_count;
    double *const v = y2 + y_count;
    double *const v_dot = v + (naux * n_core * occ);
    double *const fitted_core = v_dot + (naux * n_core * occ);
    double *const core_dot = fitted_core + (naux * n_core * occ);
    double *const fitted_act = core_dot + (naux * n_core * occ);
    double *const act_dot = fitted_act + (naux * aa);
    double *const z = act_dot + (naux * aa);
    double *const z_dot = z + (naux * aa);
    double *const d = z_dot + (naux * aa);
    double *const d_a = d + naux;
    double *const d_dot = d_a + naux;
    double *const d_a_dot = d_dot + naux;
    double *const density = d_a_dot + naux;
    double *const density_a = density + nn;
    double *const density_dot = density_a + nn;
    double *const density_a_dot = density_dot + nn;
    for (const auto &[weight, fit, fit_dot, ao, ao_dot] :
         {std::tuple{g_j, d, d_dot, density, density_dot},
          std::tuple{g_a, d_a, d_a_dot, density_a, density_a_dot}}) {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, n_oo, &kOne<double>,
                                                    fitted, q, weight, 1, &kZero<double>, fit, 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, n_oo, &kOne<double>,
                                                    fitted_dot, q, weight, 1, &kZero<double>,
                                                    fit_dot, 1)));
      // C_o G and C~_o G, then D = (C_o G) C_o^T and D' = (C_o G) C~_o^T +
      // (C~_o G) C_o^T.
      FBF_TRY(
          gpu_status(gemm(blas, n, nocc, nocc, &kOne<double>, c_occ, weight, &kZero<double>, y)));
      FBF_TRY(gpu_status(
          gemm(blas, n, nocc, nocc, &kOne<double>, c_occ_rot, weight, &kZero<double>, y2)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, n, nocc, &kOne<double>,
                                                                    y, c_occ, &kZero<double>, ao)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n, n, nocc, &kOne<double>, y, c_occ_rot, &kZero<double>, ao_dot)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, n, n, nocc, &kOne<double>, y2, c_occ, &kOne<double>, ao_dot)));
    }
    const auto n_nn = static_cast<int32_t>(nn);
    for (const auto &[alpha, fit, ao, beta] :
         {std::tuple{&kOne<double>, d_dot, density, &kZero<double>},
          std::tuple{&kOne<double>, d, density_dot, &kOne<double>},
          std::tuple{&kNegOne<double>, d_a_dot, density_a, &kOne<double>},
          std::tuple{&kNegOne<double>, d_a, density_a_dot, &kOne<double>}}) {
      FBF_TRY(gpu_status(gemm(blas, q, n_nn, 1, alpha, fit, ao, beta, gam)));
    }
    for (const auto &[alpha, left, right, beta] :
         {std::tuple{&kNegHalf<double>, d_dot, d, &kZero<double>},
          std::tuple{&kNegHalf<double>, d, d_dot, &kOne<double>},
          std::tuple{&kHalf<double>, d_a_dot, d_a, &kOne<double>},
          std::tuple{&kHalf<double>, d_a, d_a_dot, &kOne<double>}}) {
      FBF_TRY(gpu_status(
          gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, alpha, left, right, beta, w)));
    }
    // Sum_i C_mi Y_P,in into Gam (alpha 1) for the rows (P, i) of @p from, and
    // the coefficients @p coeff (C_o or C~_o columns), as the original's
    // strided-batched gemm over n.
    const auto back = [&](const double *alpha, const double *from, const int32_t rows,
                          const double *coeff, const int32_t count) -> Status {
      return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, count, alpha, from, q, int64_t{rows},
          coeff, n, 0, &kOne<double>, gam, q * n, static_cast<int64_t>(naux), n));
    };
    if (ncore > 0) {
      const auto rows = static_cast<int32_t>(naux * n_core);
      for (const auto &[from, to] :
           {std::pair{static_cast<const double *>(fitted), fitted_core},
            std::pair{static_cast<const double *>(fitted_dot), core_dot}}) {
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                      rows, nocc, &kOne<double>, from, ld_occ,
                                                      &kZero<double>, from, ld_occ, to, rows)));
      }
      FBF_TRY(gpu_status(
          gemm(blas, rows, nocc, nocc, &kOne<double>, fitted_core, g_k, &kZero<double>, v)));
      FBF_TRY(gpu_status(
          gemm(blas, rows, nocc, nocc, &kOne<double>, core_dot, g_k, &kZero<double>, v_dot)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, ncore * nocc, &kOne<double>, v_dot, fitted_core, &kOne<double>, w)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, q, q, ncore * nocc, &kOne<double>, v, core_dot, &kOne<double>, w)));
      // Y = V' C_o^T + V C~_o^T against C_o, and Y2 = V C_o^T against C~_o.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v_dot, c_occ, &kZero<double>, y)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v, c_occ_rot, &kOne<double>, y)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, rows, n, nocc, &kOne<double>, v, c_occ, &kZero<double>, y2)));
      FBF_TRY(back(&kNegTwo<double>, y, rows, c_occ, ncore));
      FBF_TRY(back(&kNegTwo<double>, y2, rows, c_occ_rot, ncore));
    }
    for (const auto &[from, to] : {std::pair{static_cast<const double *>(fitted), fitted_act},
                                   std::pair{static_cast<const double *>(fitted_dot), act_dot}}) {
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, act_rows, ncas, &kOne<double>,
          from + act_offset, ld_occ, &kZero<double>, from + act_offset, ld_occ, to, act_rows)));
    }
    FBF_TRY(gpu_status(
        gemm(blas, q, n_aa, n_aa, &kOne<double>, fitted_act, gamma2_device, &kZero<double>, z)));
    FBF_TRY(gpu_status(
        gemm(blas, q, n_aa, n_aa, &kOne<double>, act_dot, gamma2_device, &kZero<double>, z_dot)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, n_aa, &kNegHalf<double>, z_dot, fitted_act, &kOne<double>, w)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, n_aa, &kNegHalf<double>, z, act_dot, &kOne<double>, w)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, act_rows, n, ncas, &kOne<double>, z_dot, c_act, &kZero<double>, y)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, act_rows, n, ncas, &kOne<double>, z, c_act_rot, &kOne<double>, y)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, act_rows, n, ncas, &kOne<double>, z, c_act, &kZero<double>, y2)));
    FBF_TRY(back(&kOne<double>, y, act_rows, c_act, ncas));
    FBF_TRY(back(&kOne<double>, y2, act_rows, c_act_rot, ncas));
    out.three_center = FBF_TRY(three_center_hessian(res, basis, classes, aux, gam));
    out.metric = FBF_TRY(metric_hessian(res, aux, w));
  }

  Buffers::device<double> walk(cross_count + cross_rot, res.device());
  double *const m1 = walk.data();
  double *const x = m1 + (3 * naux * naux);
  double *const r_a = x + (3 * nn * nblk_max);
  double *const r_b = r_a + block;
  double *const half = r_b + block;
  double *const act_a = half + k_half;
  double *const act_b = act_a + act_block;
  double *const z = act_b + act_block;
  double *const t_a = z + act_block;
  double *const t_b = t_a + (6 * naux);
  double *const h_k = t_b + (6 * naux);
  double *const h_j = h_k + (9 * oo);
  double *const h_act = h_j + 36;
  // A rotation's: the extended block, and each atom's derivative block r',
  // its Coulomb vectors t' and active pairs act'.
  double *const r_ext = h_act + 9;
  double *const rd_a = r_ext + (rotated ? 4 * block : 0);
  double *const rd_b = rd_a + (rotated ? block : 0);
  double *const td_a = rd_b + (rotated ? block : 0);
  double *const td_b = td_a + (rotated ? 6 * naux : 0);
  double *const actd_a = td_b + (rotated ? 6 * naux : 0);
  double *const actd_b = actd_a + (rotated ? act_block : 0);
  FBF_TRY(metric_derivative(res, aux, m1));

  // One atom block: r = L^-1 R^x over the occupied pairs (occupied_atom_block),
  // its Coulomb vectors L^-1 g^x = r . G_J and L^-1 g_a^x = r . G_a into
  // @p t (naux x 6, column 2 k + {0, 1}), and its active pairs packed into
  // @p act (slab k naux x ncas^2 at P + naux (t + ncas u)).
  // With a rotation the block is built over C_ext's pairs, and cut into r
  // (C_o with C_o) and r' = r_o~o + r_oo~ (@p rd), with r' 's own Coulomb
  // vectors (@p td) and active pairs (@p actd): R^x is bilinear in its two
  // orbitals, so r' is its derivative along C -> C (1 + e K).
  const auto build = [&](const uint32_t atom, double *r, double *t, double *act, double *rd,
                         double *td, double *actd) -> Status {
    if (!rotated) {
      FBF_TRY(
          occupied_atom_block(res, basis, classes, aux, c_occ, nocc, fitted, m1, l, atom, x, r));
    } else {
      FBF_TRY(occupied_atom_block(res, basis, classes, aux, c_ext, ext_occ, fitted_ext, m1, l, atom,
                                  x, r_ext));
      for (size_t k = 0; k < 3; ++k) {
        const double *const e_k = r_ext + (k * 4 * naux * oo);
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q * nocc, nocc, &kOne<double>, e_k, ld_ext,
            &kZero<double>, e_k, ld_ext, r + (k * naux * oo), ld_occ)));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q * nocc, nocc, &kOne<double>,
            e_k + (naux * occ), ld_ext, &kOne<double>, e_k + (static_cast<size_t>(ld_ext) * occ),
            ld_ext, rd + (k * naux * oo), ld_occ)));
      }
    }
    const std::array<std::tuple<const double *, double *, double *>, 2> sides{
        {{r, t, act}, {rd, td, actd}}};
    for (size_t side = 0; side < (rotated ? 2U : 1U); ++side) {
      const auto &[from, to_t, to_act] = sides[side];
      for (size_t k = 0; k < 3; ++k) {
        const double *const r_k = from + (k * naux * oo);
        FBF_TRY(gpu_status(gemm(blas, q, 2, n_oo, &kOne<double>, r_k, g_j, &kZero<double>,
                                to_t + (2 * k * naux))));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, act_rows, ncas, &kOne<double>,
            r_k + act_offset, ld_occ, &kZero<double>, r_k + act_offset, ld_occ,
            to_act + (k * naux * aa), act_rows)));
      }
    }
    return {};
  };

  // H_AB, both orders, from the blocks of atoms A and B (@p r_other and its
  // vectors and active pairs):
  //   J:      t_A^T t_B - t_a,A^T t_a,B, from the 6 x 6 product of the vectors;
  //   K:      -2 Sum_{i core} Sum_pq G_K,pq r^x_ip r^y_iq: with
  //           s^y = r^y_core G_K (P + naux (i + ncore p)), the traces of the
  //           nocc x nocc blocks of r_A,core^T s_B (3 nocc square);
  //   active: Sum Gamma_tuvw r^x_tu r^y_vw = act_A^T (act_B Gamma).
  std::array<double, 36> host_j{};
  std::vector<double> host_k(9 * oo);
  std::array<double, 9> host_act{};
  out.cross_coulomb.assign(n_x * n_x, 0.0);
  out.cross_exchange.assign(n_x * n_x, 0.0);
  out.cross_active.assign(n_x * n_x, 0.0);
  const auto n_3o = static_cast<int32_t>(3 * occ);
  const auto contract = [&](const uint32_t atom_a, const uint32_t atom_b, const double *r_x,
                            const double *t_x, const double *act_x, const double *r_other,
                            const double *t_other, const double *act_other) -> Status {
    FBF_TRY(gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, 6, 6, q, &kOne<double>, t_x, t_other, &kZero<double>, h_j)));
    if (ncore > 0) {
      const auto rows = static_cast<int32_t>(naux * n_core);
      for (size_t k = 0; k < 3; ++k) {
        FBF_TRY(gpu_status(
            wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, nocc, nocc,
                                       &kOne<double>, r_other + (k * naux * oo), ld_occ, g_k, nocc,
                                       &kZero<double>, half + (k * naux * n_core * occ), rows)));
      }
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                    n_3o, n_3o, rows, &kNegTwo<double>, r_x, ld_occ,
                                                    half, rows, &kZero<double>, h_k, n_3o)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host_k.data(), h_k, sizeof(double) * host_k.size(),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n_aa, n_aa, &kOne<double>, act_other, q,
        static_cast<int64_t>(naux * aa), gamma2_device, n_aa, 0, &kZero<double>, z, q,
        static_cast<int64_t>(naux * aa), 3)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, 3, 3, static_cast<int32_t>(naux * aa),
                                               &kOne<double>, act_x, z, &kZero<double>, h_act)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host_j.data(), h_j, sizeof(double) * host_j.size(),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host_act.data(), h_act, sizeof(double) * host_act.size(),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t k = 0; k < 3; ++k) {
      for (size_t m = 0; m < 3; ++m) {
        // Element (k, m) of each 3 x 3 is coordinate 3 A + k with 3 B + m.
        const double j = host_j[(2 * k) + (6 * 2 * m)] - host_j[(2 * k) + 1 + (6 * ((2 * m) + 1))];
        double ex = 0.0;
        if (ncore > 0) {
          for (size_t p = 0; p < occ; ++p) {
            ex += host_k[((k * occ) + p) + (3 * occ * ((m * occ) + p))];
          }
        }
        const double act = host_act[k + (3 * m)];
        const size_t xa = (3 * size_t{atom_a}) + k;
        const size_t xb = (3 * size_t{atom_b}) + m;
        for (const size_t at : {(xa * n_x) + xb, (xb * n_x) + xa}) {
          out.cross_coulomb[at] += j;
          out.cross_exchange[at] += ex;
          out.cross_active[at] += act;
          if (atom_a == atom_b) {
            break;
          }
        }
      }
    }
    return {};
  };

  // H_AB, or with a rotation its derivative: each block's derivative against
  // the other's own, both ways (every piece is a symmetric bilinear form).
  const auto pair = [&](const uint32_t atom_a, const uint32_t atom_b, const double *r_other,
                        const double *t_other, const double *act_other, const double *rd_other,
                        const double *td_other, const double *actd_other) -> Status {
    if (!rotated) {
      return contract(atom_a, atom_b, r_a, t_a, act_a, r_other, t_other, act_other);
    }
    FBF_TRY(contract(atom_a, atom_b, rd_a, td_a, actd_a, r_other, t_other, act_other));
    return contract(atom_a, atom_b, r_a, t_a, act_a, rd_other, td_other, actd_other);
  };
  for (uint32_t atom_a = 0; atom_a < n_atom; ++atom_a) {
    FBF_TRY(build(atom_a, r_a, t_a, act_a, rd_a, td_a, actd_a));
    FBF_TRY(pair(atom_a, atom_a, r_a, t_a, act_a, rd_a, td_a, actd_a));
    for (uint32_t atom_b = 0; atom_b < atom_a; ++atom_b) {
      FBF_TRY(build(atom_b, r_b, t_b, act_b, rd_b, td_b, actd_b));
      FBF_TRY(pair(atom_a, atom_b, r_b, t_b, act_b, rd_b, td_b, actd_b));
    }
  }

  out.total = out.three_center;
  for (const std::vector<double> *term :
       {&out.metric, &out.cross_coulomb, &out.cross_exchange, &out.cross_active}) {
    std::ranges::transform(out.total, *term, out.total.begin(), std::plus{});
  }
  return out;
}

} // namespace free_breakfast::hessian
