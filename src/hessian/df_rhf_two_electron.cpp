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
import free_breakfast.core.primitives; // align_up
import free_breakfast.core.wwr;        // gemm, gemv, trsm, gemmStridedBatched
import free_breakfast.core.resources;
import free_breakfast.core.footprint; // MemoryFootprint, kPoolHeadroomBytes
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;
import free_breakfast.integrals.eri.density_fitting.hessian;
import free_breakfast.integrals.eri.density_fitting.mo;
import free_breakfast.recipes.blas;
import :atom_block;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// What df_rhf_two_electron and its footprint share: its own buffers' counts,
// in doubles, and the workspace its calls carve, in bytes.
struct RhfTwoElectronSizes {
  // Held for the whole call: the fitted pairs and a vector of twos.
  size_t held = 0;
  // The explicit part's: Gam, its half back-transform Y, W, d and D.
  size_t explicit_part = 0;
  // One aux atom's derivative block X, 3 nao^2 nblk at the largest atom's
  // nblk.
  size_t derivative_block = 0;
  // The cross terms' besides their slots: M1, X and the two 3 n_atom x
  // 3 n_atom products.
  size_t cross_fixed = 0;
  // One slot of the cross terms' batches: an atom block and its Coulomb
  // vectors.
  size_t slot = 0;
  // The fewest slots the call runs with, min(2, n_atom).
  size_t least_slots = 0;
  // The largest workspace carve of b_mo, the second-derivative integrals,
  // metric_derivative, and each aux atom's three_center_derivative and its
  // half-transform W.
  size_t workspace = 0;
};

RhfTwoElectronSizes rhf_two_electron_sizes(const BasisSet &basis, const BasisSet &aux,
                                           const int32_t nocc) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const auto occ = static_cast<size_t>(nocc);
  const size_t oo = occ * occ;
  const size_t n_x = 3 * static_cast<size_t>(n_atom);
  const OrbitalRange occupied{.first = 0, .count = nocc};
  RhfTwoElectronSizes s;
  s.workspace = std::max(
      {b_mo_footprint(static_cast<int32_t>(naux), static_cast<int32_t>(nao), occupied, occupied)
           .device.workspace,
       three_center_hessian_footprint(basis, aux).device.workspace,
       metric_hessian_footprint(aux).device.workspace,
       metric_derivative_footprint(aux).device.workspace});
  size_t nblk_max = 0;
  for (uint32_t a = 0; a < n_atom; ++a) {
    const auto [p0, p1] = atom_function_range(aux, a);
    const size_t nblk = p1 - p0;
    nblk_max = std::max(nblk_max, nblk);
    if (nblk > 0) {
      s.workspace =
          std::max({s.workspace, three_center_derivative_footprint(basis, aux, a).device.workspace,
                    math::align_up(sizeof(double) * 3 * nao * nblk * occ,
                                   DeviceResources::kWorkspaceAlignment)});
    }
  }
  s.held = (naux * oo) + occ;
  s.explicit_part = (naux * nao * nao) + (naux * occ * nao) + (naux * naux) + naux + (nao * nao);
  s.derivative_block = 3 * nao * nao * nblk_max;
  s.cross_fixed = (3 * naux * naux) + s.derivative_block + (2 * n_x * n_x);
  s.slot = (3 * naux * oo) + (3 * naux);
  s.least_slots = std::min<size_t>(2, n_atom);
  return s;
}

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
  // R^x for one atom's three coordinates, naux x nocc^2 each: one atom block.
  const size_t block = 3 * naux * oo;
  constexpr auto kInt32Max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (naux * nn > kInt32Max || block > kInt32Max || naux * occ * nao > kInt32Max) {
    return err_unsupported("hessian::df_rhf_two_electron: naux nao^2 = {}, an atom block 3 naux "
                           "nocc^2 = {} or naux nocc nao = {} exceeds int32_t",
                           naux * nn, block, naux * occ * nao);
  }
  // Held for the whole call: the fitted pairs c and a vector of twos. The
  // explicit part's buffer, then the cross terms' fixed part and at least two
  // slots. All are held across calls that take the workspace (b_mo, the
  // Hessian and derivative integrals), so they are the call's own
  // (docs/style.md "Scratch memory"); past the memory budget an allocation
  // would abort (CreatePolicy), not return.
  const RhfTwoElectronSizes sizes = rhf_two_electron_sizes(basis, aux, nocc);
  const size_t held_count = sizes.held;
  const size_t explicit_count = sizes.explicit_part;
  const size_t cross_fixed = sizes.cross_fixed;
  const size_t slot = sizes.slot;
  const size_t cross_least = cross_fixed + (sizes.least_slots * slot);
  const size_t bytes = sizeof(double) * (held_count + std::max(explicit_count, cross_least));
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t budget = max_bytes == 0 ? memory : std::min(memory, max_bytes);
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

  // The batch: as many slots as the memory budget holds, all n_atom when
  // they fit (one walk), else two batches of half as many, A's and B's, so
  // (n_b + 1) n_b / 2 walks for n_b batches; never fewer than the least the
  // refusal above counted, df_rhf_two_electron_footprint's scratch. Sized
  // now, the explicit part freed, from the budget less what the call has
  // still to allocate besides the batch, its last buffer: the workspace's
  // growth to the walk's carves, if it must grow (the new block is allocated
  // before the old one is freed), and the pool's fragmentation headroom.
  size_t slots = sizes.least_slots;
  {
    const size_t growth =
        sizes.workspace > res.device_workspace(0).num_elements()
            ? math::align_up(sizes.workspace, DeviceResources::kWorkspaceAlignment)
            : 0;
    const size_t reserve = kPoolHeadroomBytes + growth;
    const size_t available = FBF_TRY(res.memory_budget());
    size_t room = available > reserve ? available - reserve : 0;
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
  double *const h_k = x + sizes.derivative_block;
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

MemoryFootprint df_rhf_two_electron_footprint(const BasisSet &basis, const BasisSet &aux,
                                              const int32_t nocc) {
  MemoryFootprint out;
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms() || nocc < 1 ||
      static_cast<size_t>(nocc) > basis.nao_sphe()) {
    return out;
  }
  // The held buffer beside the explicit part's, then beside the batch at its
  // least; the Coulomb and exchange parts are the caller's, on the host.
  const RhfTwoElectronSizes s = rhf_two_electron_sizes(basis, aux, nocc);
  out.device.scratch =
      sizeof(double) *
      (s.held + std::max(s.explicit_part, s.cross_fixed + (s.least_slots * s.slot)));
  out.device.workspace = s.workspace;
  return out;
}

} // namespace free_breakfast::hessian
