/// @file
/// @brief df_rhf_two_electron: the fitted densities' second-derivative
/// contractions, then the fit's cross terms as Gram matrices of L^-1 R^x: a
/// batch of atom blocks of R^x, as many atoms' as the memory holds, built per
/// walk over the aux atoms and contracted one pair of batches at a time.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne, kNegTwo, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, trsm, gemmStridedBatched, wwrMemGetInfo
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

// Left free when the cross terms size their batch of atom blocks to the
// device's memory: the BLAS libraries' own workspaces and the pool's.
constexpr size_t kCrossBatchHeadroomBytes = size_t{512} << 20;

Result<RhfTwoElectron> df_rhf_two_electron(DeviceResources &res, const BasisSet &basis,
                                           const std::span<const PrimitivePairClass> classes,
                                           const BasisSet &aux, const Buffers::device<double> &b,
                                           const double *l, const Buffers::device<double> &c,
                                           const int32_t nocc, const size_t max_bytes) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0 || aux.natoms() != n_atom || nocc < 1 || static_cast<size_t>(nocc) > nao ||
      c.num_elements() < nao * static_cast<size_t>(nocc) || b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_rhf_two_electron: needs two bases placed on the same atoms, "
                      "1 <= nocc <= nao, C at least nao nocc and B naux nao^2 long ({} and {} "
                      "atoms, nocc {}, nao {}, naux {}, C {} doubles, B {} doubles)",
                      n_atom, aux.natoms(), nocc, nao, naux, c.num_elements(), b.num_elements());
  }
  const auto occ = static_cast<size_t>(nocc);
  const size_t oo = occ * occ;
  const size_t nn = nao * nao;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  size_t nblk_max = 0;
  size_t nblk_cart_max = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_function_range(aux, a);
    nblk_max = std::max(nblk_max, p1 - p0);
    size_t cart = 0;
    for (uint32_t sh = aux.atom_shell_offset(a);
         sh < aux.atom_shell_offset(a) + aux.atom_shell_count(a); ++sh) {
      cart += aux.shells()[sh].function_count_cart();
    }
    nblk_cart_max = std::max(nblk_cart_max, cart);
  }
  // R^x for one atom's three coordinates, naux x nocc^2 each: one atom block.
  const size_t block = 3 * naux * oo;
  // An atom block and its Coulomb vectors: one slot of the cross terms' batches.
  const size_t slot = block + (3 * naux);
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (naux * nn > kInt32Max || block > kInt32Max || naux * occ * nao > kInt32Max) {
    return err_unsupported("hessian::df_rhf_two_electron: naux nao^2 = {}, an atom block 3 naux "
                           "nocc^2 = {} or naux nocc nao = {} exceeds int32_t",
                           naux * nn, block, naux * occ * nao);
  }
  // Held for the whole call: the fitted pairs c and a vector of twos. The
  // explicit part's: Gam, its half back-transform Y, W, d and D. The cross
  // terms': M1, one aux atom's derivative block X, the two 3 n_atom x 3 n_atom
  // products and at least two slots. All are held across calls that take the
  // workspace (b_mo, the Hessian and derivative integrals), so they are the
  // call's own (docs/style.md "Scratch memory"); past the device's memory an
  // allocation would abort (CreatePolicy), not return.
  const size_t held_count = (naux * oo) + occ;
  const size_t explicit_count = (naux * nn) + (naux * occ * nao) + (naux * naux) + naux + nn;
  const size_t cross_fixed = (3 * naux * naux) + (3 * nn * nblk_max) + (2 * n_x * n_x);
  const size_t cross_least = cross_fixed + (std::min<size_t>(2, n_atom) * slot);
  const size_t bytes = sizeof(double) * (held_count + std::max(explicit_count, cross_least));
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t budget = max_bytes == 0 ? free_bytes : std::min(free_bytes, max_bytes);
  if (bytes > budget) {
    return err_unsupported("hessian::df_rhf_two_electron: the fitted densities and a pair of "
                           "atom blocks of the cross term over {} orbital and {} fitting "
                           "functions and {} occupied orbitals need {} bytes of device memory, "
                           "and device {} has {} to give",
                           nao, naux, occ, bytes, res.dev_idx(), budget);
  }

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto n_oo = static_cast<int32_t>(oo);
  const double *const c_occ = c.data();

  Buffers::device<double> held(held_count, res.device());
  double *const fitted = held.data();
  double *const twos = fitted + (naux * oo);
  // Read by the copy below, which the stream has finished by the first sync.
  const std::vector<double> twos_host(occ, 2.0);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(twos, twos_host.data(), sizeof(double) * occ,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // c_P,ij = (L^-T B_ij)_P, [P][i][j]: the naux x nocc^2 matrix at
  // P + naux (i + nocc j), symmetric in i and j.
  const OrbitalRange occupied{.first = 0, .count = nocc};
  FBF_TRY(b_mo(res, q, n, nocc, b.data(), c_occ, occupied, occupied, fitted));
  FBF_TRY(gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, n_oo, &kOne<double>, l, q, fitted, q)));
  // The pairs' diagonal c_P,ii is the naux x nocc matrix of leading dimension
  // naux (nocc + 1), so 2 Sum_i of it is one gemv against the twos.
  const auto diagonal_ld = static_cast<int32_t>(naux * (occ + 1));

  RhfTwoElectron out;
  {
    // The explicit part, gradient::df_rhf's weights: E_J's Gam = D (x) d and
    // W = -1/2 d d^T, d = M^-1 u = 2 Sum_i c_ii; E_x's Gam_mnP = -2 Sum_ij
    // C_mi C_nj c_P,ij and W = Sum_ij c_ij c_ij^T.
    Buffers::device<double> scratch(explicit_count, res.device());
    double *const gamma = scratch.data();
    double *const y = gamma + (naux * nn);
    double *const w = y + (naux * occ * nao);
    double *const d = w + (naux * naux);
    double *const density = d + naux;
    FBF_TRY(
        gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nocc, &kOne<double>,
                                              fitted, diagonal_ld, twos, 1, &kZero<double>, d, 1)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n, n, nocc, &kTwo<double>, c_occ, c_occ, &kZero<double>, density)));
    // Gam = d vec(D)^T, element (P, n + nao m) = d_P D_nm.
    FBF_TRY(gpu_status(gemm(blas, q, static_cast<int32_t>(nn), 1, &kOne<double>, d, density,
                            &kZero<double>, gamma)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, q, q, 1, &kNegHalf<double>,
                                                                  d, d, &kZero<double>, w)));
    out.coulomb = FBF_TRY(three_center_hessian(res, basis, classes, aux, gamma));
    const std::vector<double> coulomb_metric = FBF_TRY(metric_hessian(res, aux, w));
    std::ranges::transform(out.coulomb, coulomb_metric, out.coulomb.begin(), std::plus{});

    // Y_P,in = Sum_j c_P,ij C_nj at P + naux (i + nocc n); then for each n,
    // Gam_(P, n, m) = -2 Sum_i Y_P,in C_mi (gradient::df_rhf's two products).
    const auto rows = static_cast<int32_t>(naux * occ);
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, rows, n, nocc, &kOne<double>, fitted, c_occ, &kZero<double>, y)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, q, n, nocc, &kNegTwo<double>, y, q,
        int64_t{rows}, c_occ, n, 0, &kZero<double>, gamma, q * n, static_cast<int64_t>(naux), n)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, q, q, n_oo, &kOne<double>, fitted, fitted, &kZero<double>, w)));
    out.exchange = FBF_TRY(three_center_hessian(res, basis, classes, aux, gamma));
    const std::vector<double> exchange_metric = FBF_TRY(metric_hessian(res, aux, w));
    std::ranges::transform(out.exchange, exchange_metric, out.exchange.begin(), std::plus{});
  }

  // The batch: as many slots as the budget holds past the call's other
  // buffers, all n_atom when they fit (one walk), else two batches of half
  // as many, A's and B's, so (n_b + 1) n_b / 2 walks for n_b batches. Sized
  // now, after the explicit part has grown the workspace, and leaving room
  // for three_center_derivative's scratch past what the workspace holds
  // (its Cartesian block, its two half-transforms and staging) and the
  // headroom; never fewer than the two slots the refusal above counted.
  size_t slots = std::min<size_t>(2, n_atom);
  {
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    const size_t n_cart = basis.nao_cart();
    const size_t walk_scratch =
        sizeof(double) * ((3 * n_cart * n_cart * nblk_cart_max) + (nblk_max * n_cart * n_cart) +
                          (nblk_max * nao * n_cart) + (n_cart * nao) + (aux.nao_cart() * naux));
    const size_t held_scratch = res.device_workspace(0).num_elements();
    const size_t reserve =
        kCrossBatchHeadroomBytes + (walk_scratch > held_scratch ? walk_scratch - held_scratch : 0);
    size_t room = free_bytes > reserve ? free_bytes - reserve : 0;
    if (max_bytes != 0) {
      room = std::min(room, max_bytes - (sizeof(double) * held_count));
    }
    const size_t room_count = room / sizeof(double);
    if (room_count > cross_fixed) {
      slots = std::clamp<size_t>((room_count - cross_fixed) / slot, slots, n_atom);
    }
    // A batch's trsm takes its 3 nocc^2 columns per atom as one int32_t (at
    // least one atom's, as block fits it).
    slots = std::min(slots, kInt32Max / (3 * oo));
  }
  const size_t per_batch = slots >= n_atom ? n_atom : std::max<size_t>(slots / 2, 1);
  const size_t batched = per_batch >= n_atom ? n_atom : 2 * per_batch;

  Buffers::device<double> walk(cross_fixed + (batched * slot), res.device());
  double *const m1 = walk.data();
  double *const x = m1 + (3 * naux * naux);
  double *const h_k = x + (3 * nn * nblk_max);
  double *const h_j = h_k + (n_x * n_x);
  double *const r_a = h_j + (n_x * n_x);
  double *const r_b = r_a + (per_batch * block);
  double *const t_a = r_a + (batched * block);
  double *const t_b = t_a + (per_batch * 3 * naux);
  FBF_TRY(metric_derivative(res, aux, m1));

  // One batch of atom blocks (occupied_atom_block), atoms @p first to
  // @p first + count - 1: for x = 3 A + k, r = L^-1 R^x into @p r (slab
  // 3 (A - first) + k, naux x nocc^2 at P + naux (j + nocc i)), and
  // t = L^-1 g^x = 2 Sum_i r_ii into @p t (naux at the same slab's naux).
  const auto build = [&](const uint32_t first, const uint32_t count, double *r,
                         double *t) -> Status {
    FBF_TRY(occupied_atom_block(res, basis, classes, aux, c_occ, nocc, fitted, m1, l, first, x, r,
                                count));
    for (size_t k = 0; k < 3 * size_t{count}; ++k) {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, q, nocc, &kOne<double>,
                                                    r + (k * naux * oo), diagonal_ld, twos, 1,
                                                    &kZero<double>, t + (k * naux), 1)));
    }
    return {};
  };

  // H_AB for every A in batch a and B in batch b, both orders: -2 r_a^T r_b
  // for the exchange and t_a^T t_b for the Coulomb cross term, one gemm each
  // over every pair, added to the explicit part. A batch with itself is its
  // whole square, both orders already.
  std::vector<double> cross_k(n_x * n_x);
  std::vector<double> cross_j(n_x * n_x);
  const auto contract = [&](const uint32_t first_a, const uint32_t count_a, const uint32_t first_b,
                            const uint32_t count_b, const double *r_other,
                            const double *t_other) -> Status {
    const auto rows = static_cast<int32_t>(3 * count_a);
    const auto cols = static_cast<int32_t>(3 * count_b);
    const size_t elements = size_t{3} * count_a * 3 * count_b;
    FBF_TRY(
        gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, rows, cols, static_cast<int32_t>(naux * oo),
                                           &kNegTwo<double>, r_a, r_other, &kZero<double>, h_k)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, rows, cols, q, &kOne<double>, t_a, t_other,
                                               &kZero<double>, h_j)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cross_k.data(), h_k, sizeof(double) * elements,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(cross_j.data(), h_j, sizeof(double) * elements,
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t col = 0; col < size_t{3} * count_b; ++col) {
      for (size_t row = 0; row < size_t{3} * count_a; ++row) {
        // Element (row, col) is coordinate 3 first_a + row with 3 first_b + col.
        const size_t xa = (3 * size_t{first_a}) + row;
        const size_t xb = (3 * size_t{first_b}) + col;
        const size_t from = row + (size_t{3} * count_a * col);
        out.exchange[(xa * n_x) + xb] += cross_k[from];
        out.coulomb[(xa * n_x) + xb] += cross_j[from];
        if (first_a != first_b) {
          out.exchange[(xb * n_x) + xa] += cross_k[from];
          out.coulomb[(xb * n_x) + xa] += cross_j[from];
        }
      }
    }
    return {};
  };

  const auto step = static_cast<uint32_t>(per_batch);
  for (uint32_t first_a = 0; first_a < n_atom; first_a += step) {
    const uint32_t count_a = std::min(step, n_atom - first_a);
    FBF_TRY(build(first_a, count_a, r_a, t_a));
    FBF_TRY(contract(first_a, count_a, first_a, count_a, r_a, t_a));
    for (uint32_t first_b = 0; first_b < first_a; first_b += step) {
      FBF_TRY(build(first_b, step, r_b, t_b));
      FBF_TRY(contract(first_a, count_a, first_b, step, r_b, t_b));
    }
  }
  return out;
}

} // namespace free_breakfast::hessian
