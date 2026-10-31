/// @file
/// @brief df_casscf_two_electron: the CASSCF fitted densities' second-
/// derivative contractions (mcscf::df_gradient's Gam and W at one root's
/// RDMs), then the fit's cross term split into its J, K and active pieces,
/// one occupied atom block of L^-1 R^x built per walk over the aux atoms and
/// contracted one pair of atom blocks at a time.

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
                       const std::span<const double> gamma, const std::span<const double> gamma2) {
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
  if (naux * nn > kInt32Max || block > kInt32Max || y_count > kInt32Max || aa * aa > kInt32Max) {
    return err_unsupported("hessian::df_casscf_two_electron: naux nao^2 = {}, an atom block 3 naux "
                           "nocc^2 = {}, naux max(ncore, ncas) nao = {} or ncas^4 = {} exceeds "
                           "int32_t",
                           naux * nn, block, y_count, aa * aa);
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
  const size_t bytes = sizeof(double) * (held_count + std::max(explicit_count, cross_count));
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
  // The active pairs (t, u) of an occupied naux x nocc^2 slab: its rows
  // naux ncore onward of columns ncore onward, leading dimension naux nocc.
  const size_t act_offset = (naux * n_core) + (naux * occ * n_core);
  const auto ld_occ = static_cast<int32_t>(naux * occ);
  const auto act_rows = static_cast<int32_t>(naux * n_act);

  CasscfTwoElectron out;
  {
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
  }

  Buffers::device<double> walk(cross_count, res.device());
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
  FBF_TRY(metric_derivative(res, aux, m1));

  // One atom block: r = L^-1 R^x over the occupied pairs (occupied_atom_block),
  // its Coulomb vectors L^-1 g^x = r . G_J and L^-1 g_a^x = r . G_a into
  // @p t (naux x 6, column 2 k + {0, 1}), and its active pairs packed into
  // @p act (slab k naux x ncas^2 at P + naux (t + ncas u)).
  const auto build = [&](const uint32_t atom, double *r, double *t, double *act) -> Status {
    FBF_TRY(occupied_atom_block(res, basis, classes, aux, c_occ, nocc, fitted, m1, l, atom, x, r));
    for (size_t k = 0; k < 3; ++k) {
      const double *const r_k = r + (k * naux * oo);
      FBF_TRY(gpu_status(
          gemm(blas, q, 2, n_oo, &kOne<double>, r_k, g_j, &kZero<double>, t + (2 * k * naux))));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                    act_rows, ncas, &kOne<double>, r_k + act_offset,
                                                    ld_occ, &kZero<double>, r_k + act_offset,
                                                    ld_occ, act + (k * naux * aa), act_rows)));
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
  const auto contract = [&](const uint32_t atom_a, const uint32_t atom_b, const double *r_other,
                            const double *t_other, const double *act_other) -> Status {
    FBF_TRY(gpu_status(
        gemm<wwr::WWRBLAS_OP_T>(blas, 6, 6, q, &kOne<double>, t_a, t_other, &kZero<double>, h_j)));
    if (ncore > 0) {
      const auto rows = static_cast<int32_t>(naux * n_core);
      for (size_t k = 0; k < 3; ++k) {
        FBF_TRY(gpu_status(
            wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, nocc, nocc,
                                       &kOne<double>, r_other + (k * naux * oo), ld_occ, g_k, nocc,
                                       &kZero<double>, half + (k * naux * n_core * occ), rows)));
      }
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                    n_3o, n_3o, rows, &kNegTwo<double>, r_a, ld_occ,
                                                    half, rows, &kZero<double>, h_k, n_3o)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host_k.data(), h_k, sizeof(double) * host_k.size(),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
    }
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q, n_aa, n_aa, &kOne<double>, act_other, q,
        static_cast<int64_t>(naux * aa), gamma2_device, n_aa, 0, &kZero<double>, z, q,
        static_cast<int64_t>(naux * aa), 3)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, 3, 3, static_cast<int32_t>(naux * aa),
                                               &kOne<double>, act_a, z, &kZero<double>, h_act)));
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

  for (uint32_t atom_a = 0; atom_a < n_atom; ++atom_a) {
    FBF_TRY(build(atom_a, r_a, t_a, act_a));
    FBF_TRY(contract(atom_a, atom_a, r_a, t_a, act_a));
    for (uint32_t atom_b = 0; atom_b < atom_a; ++atom_b) {
      FBF_TRY(build(atom_b, r_b, t_b, act_b));
      FBF_TRY(contract(atom_a, atom_b, r_b, t_b, act_b));
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
