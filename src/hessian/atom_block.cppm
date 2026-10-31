/// @file
/// @brief The fit's cross-term atom block that df_rhf_two_electron and
/// df_casscf_two_electron share: L^-1 R^x over the occupied pairs for the
/// three coordinates of one atom, R^x = T^x - M^x c, built by one walk over
/// the aux atoms.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.hessian. See docs/style.md, "Module-private code".

module free_breakfast.hessian:atom_block;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, trsm, gemmStridedBatched, wwrMemsetAsync
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// [begin, end) of the spherical functions on @p atom of a placed basis: its
// shells are contiguous, so their functions are.
std::pair<size_t, size_t> atom_function_range(const BasisSet &basis, const uint32_t atom) {
  const auto start_of = [&basis](const uint32_t shell) -> size_t {
    return shell < basis.nshells() ? basis.ao_offset_sphe(shell) : basis.nao_sphe();
  };
  const uint32_t first = basis.atom_shell_offset(atom);
  return {start_of(first), start_of(first + basis.atom_shell_count(atom))};
}

// One atom block: for x = 3 A + k, A = @p atom, r = L^-1 R^x over the occupied
// pairs into @p r (slab k, naux x nocc^2 at P + naux (j + nocc i), 3 naux
// nocc^2 doubles, symmetric in i and j). @p c_occ is C's first @p nocc
// columns (nao x nocc), @p fitted the fitted pairs c = L^-T B_ij in the same
// layout as a slab, @p m1 @c metric_derivative's M1 and @p l the metric's L;
// @p x holds one aux atom's @c three_center_derivative block (3 nao^2 nblk,
// nblk the largest atom's aux share). With X the derivative on m's centre,
// d(mn|P)/dA_k = -[m on A] X_mn - [n on A] X_nm + [P on A] (X_mn + X_nm), so
// with Y_ij = Sum_{m on A, n} C_mi C_nj X_mn,P and Z the same over every m,
//   T^x_ij = -(Y_ij + Y_ji) + [P on A] (Z_ij + Z_ji),
// each aux atom's rows from its own block of X; and with M1[k] the metric's
// derivative on P's centre, (M^x c)_P = [P on A] Sum_Q M1[k][P][Q] c_Q +
// Sum_{Q on A} M1[k][Q][P] c_Q. Queued on res.stream(); the derivative calls
// synchronize it. The product X C_o is res.device_workspace() scratch.
Status occupied_atom_block(DeviceResources &res, const BasisSet &basis,
                           const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                           const double *const c_occ, const int32_t nocc,
                           const double *const fitted, const double *const m1,
                           const double *const l, const uint32_t atom, double *const x,
                           double *const r) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const auto occ = static_cast<size_t>(nocc);
  const size_t oo = occ * occ;
  const size_t block = 3 * naux * oo;
  const auto n = static_cast<int32_t>(nao);
  const auto q = static_cast<int32_t>(naux);
  const auto n_oo = static_cast<int32_t>(oo);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(r, 0, sizeof(double) * block, res.stream())));
  const auto [a0, a1] = atom_function_range(basis, atom);
  for (uint32_t aux_atom = 0; aux_atom < aux.natoms(); ++aux_atom) {
    const auto [p0, p1] = atom_function_range(aux, aux_atom);
    const size_t nblk = p1 - p0;
    if (nblk == 0) {
      continue;
    }
    FBF_TRY(three_center_derivative(res, basis, classes, aux, aux_atom, x));
    // W[k][m][P][j] = Sum_n X[k][m][n][P] C_nj, at P + nblk (j + nocc (m +
    // nao k)): one gemm per (k, m), each X block nblk x nao.
    const auto blk = static_cast<int32_t>(nblk);
    auto [w] = res.device_regions<double>(3 * nao * nblk * occ);
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, n, &kOne<double>, x, blk,
        static_cast<int64_t>(nblk * nao), c_occ, n, 0, &kZero<double>, w, blk,
        static_cast<int64_t>(nblk * occ), 3 * n)));
    for (size_t k = 0; k < 3; ++k) {
      const double *const w_k = w + (k * nao * nblk * occ);
      double *const r_k = r + (k * naux * oo) + p0;
      // alpha Sum_{m in [m0, m0 + count)} C_m. W[k][m][P][.], into rows P of
      // R^x as both (i, j) and (j, i): one gemm per j, then one per i.
      const auto add = [&](const size_t m0, const size_t count, const double *alpha) -> Status {
        const double *const w_m = w_k + (nblk * occ * m0);
        const auto nm = static_cast<int32_t>(count);
        const auto ld_w = static_cast<int32_t>(nblk * occ);
        FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, nm, alpha, w_m, ld_w,
            static_cast<int64_t>(nblk), c_occ + m0, n, 0, &kOne<double>, r_k,
            static_cast<int32_t>(naux * occ), static_cast<int64_t>(naux), nocc)));
        return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, blk, nocc, nm, alpha, w_m, ld_w,
            static_cast<int64_t>(nblk), c_occ + m0, n, 0, &kOne<double>, r_k, q,
            static_cast<int64_t>(naux * occ), nocc));
      };
      if (a1 > a0) {
        FBF_TRY(add(a0, a1 - a0, &kNegOne<double>));
      }
      if (aux_atom == atom) {
        FBF_TRY(add(0, nao, &kOne<double>));
      }
    }
  }
  const auto [q0, q1] = atom_function_range(aux, atom);
  const auto nq = static_cast<int32_t>(q1 - q0);
  for (size_t k = 0; k < 3; ++k) {
    const double *const m1_k = m1 + (k * naux * naux);
    double *const r_k = r + (k * naux * oo);
    if (nq > 0) {
      // M1[k] is column-major with M1[k][P][Q] at (Q, P): its columns on A,
      // transposed, against c for the rows on A, and the same columns as
      // they stand against c's rows on A for every row.
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nq,
                                                    n_oo, q, &kNegOne<double>, m1_k + (naux * q0),
                                                    q, fitted, q, &kOne<double>, r_k + q0, q)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, q,
                                                    n_oo, nq, &kNegOne<double>, m1_k + (naux * q0),
                                                    q, fitted + q0, q, &kOne<double>, r_k, q)));
    }
  }
  return gpu_status(wwr::trsm<double, int32_t>(
      blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
      wwr::WWRBLAS_DIAG_NON_UNIT, q, 3 * n_oo, &kOne<double>, l, q, r, q));
}

} // namespace free_breakfast::hessian
