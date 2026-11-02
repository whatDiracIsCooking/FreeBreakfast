/// @file
/// @brief mcscf::state_overlaps: the mixed AO overlap and the MO overlap on
/// the device, the core elimination and the string determinants on the host,
/// and the CI contraction as three gemms on the device.

module free_breakfast.mcscf.overlap;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gpu_status, wwrMemcpyAsync
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao; // SphericalMatrixAO, Solution::c
import free_breakfast.integrals.one_electron.overlap_mixed;
import free_breakfast.ci.strings;
import free_breakfast.mcscf.solve;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// Below this |det| of the core blocks' overlap the core is not carried into
// itself, and the Schur complement the string determinants are taken of
// would be ill-conditioned (README.md "The core").
constexpr double kCoreOverlapFloor = 1e-6;

// LU with partial pivoting of the n x n row-major @p a, in place, applied to
// the n x nrhs row-major @p b, which ends as a^-1 b; returns det(a). An exact
// zero pivot returns 0 at once and leaves @p b unsolved.
double overlap_eliminate(std::span<double> a, const size_t n, std::span<double> b,
                         const size_t nrhs) {
  double det = 1.0;
  for (size_t k = 0; k < n; ++k) {
    size_t pivot = k;
    for (size_t i = k + 1; i < n; ++i) {
      if (std::abs(a[(i * n) + k]) > std::abs(a[(pivot * n) + k])) {
        pivot = i;
      }
    }
    if (a[(pivot * n) + k] == 0.0) {
      return 0.0;
    }
    if (pivot != k) {
      std::swap_ranges(a.begin() + static_cast<std::ptrdiff_t>(k * n),
                       a.begin() + static_cast<std::ptrdiff_t>((k + 1) * n),
                       a.begin() + static_cast<std::ptrdiff_t>(pivot * n));
      std::swap_ranges(b.begin() + static_cast<std::ptrdiff_t>(k * nrhs),
                       b.begin() + static_cast<std::ptrdiff_t>((k + 1) * nrhs),
                       b.begin() + static_cast<std::ptrdiff_t>(pivot * nrhs));
      det = -det;
    }
    const double diagonal = a[(k * n) + k];
    det *= diagonal;
    for (size_t i = k + 1; i < n; ++i) {
      const double factor = a[(i * n) + k] / diagonal;
      for (size_t j = k + 1; j < n; ++j) {
        a[(i * n) + j] -= factor * a[(k * n) + j];
      }
      for (size_t r = 0; r < nrhs; ++r) {
        b[(i * nrhs) + r] -= factor * b[(k * nrhs) + r];
      }
    }
  }
  for (size_t k = n; k-- > 0;) {
    for (size_t r = 0; r < nrhs; ++r) {
      double sum = b[(k * nrhs) + r];
      for (size_t j = k + 1; j < n; ++j) {
        sum -= a[(k * n) + j] * b[(j * nrhs) + r];
      }
      b[(k * nrhs) + r] = sum / a[(k * n) + k];
    }
  }
  return det;
}

// The string overlaps of one spin, column-major m x m for the set's m
// strings: element (a, a') = <a|a'> = det of @p active's rows occ(a) and
// columns occ(a'), @p active the ncas x ncas row-major active overlap with
// the core eliminated. Both strings order their creators by ascending
// orbital (ci::StringSet), so the determinant needs no further sign.
std::vector<double> overlap_strings(const ci::StringSet &set, std::span<const double> active,
                                    const size_t ncas) {
  const auto m = static_cast<size_t>(set.size());
  const auto electrons = static_cast<size_t>(set.n_electrons());
  std::vector<size_t> occupied;
  occupied.reserve(m * electrons);
  for (const uint64_t string : set.strings()) {
    for (size_t o = 0; o < ncas; ++o) {
      if (((string >> o) & 1U) != 0U) {
        occupied.push_back(o);
      }
    }
  }
  std::vector<double> out(m * m);
  std::vector<double> block(electrons * electrons);
  for (size_t ket = 0; ket < m; ++ket) {
    for (size_t bra = 0; bra < m; ++bra) {
      for (size_t i = 0; i < electrons; ++i) {
        for (size_t j = 0; j < electrons; ++j) {
          block[(i * electrons) + j] =
              active[(occupied[(bra * electrons) + i] * ncas) + occupied[(ket * electrons) + j]];
        }
      }
      out[bra + (ket * m)] = overlap_eliminate(block, electrons, {}, 0);
    }
  }
  return out;
}

// One solution's shape against @p basis, @p name in the message.
Status overlap_check(const DeviceResources &res, const BasisSet &basis, const Solution &solution,
                     const int32_t ncore, const int32_t ncas, const size_t dim,
                     const std::string_view name) {
  const SphericalMatrixAO &c = solution.c;
  const size_t roots = solution.roots.energies.size();
  if (roots < 1) {
    return err_config("mcscf::state_overlaps: the {} solution has no roots", name);
  }
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::state_overlaps: the {} C is on device {}, but the call passed "
                      "device {}",
                      name, c.dev_idx(), res.dev_idx());
  }
  const int64_t nao = basis.nao_sphe();
  if (c.n() != nao) {
    return err_config("mcscf::state_overlaps: the {} C is {} square, but its basis has {} "
                      "spherical AOs",
                      name, c.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || ncore + ncas > nao) {
    return err_config("mcscf::state_overlaps: ncore {} and ncas {} over the {} solution's {} "
                      "orbitals",
                      ncore, ncas, name, nao);
  }
  if (solution.roots.vectors.num_elements() != dim * roots) {
    return err_config("mcscf::state_overlaps: the {} solution has {} CI doubles for {} roots of "
                      "{} determinants",
                      name, solution.roots.vectors.num_elements(), roots, dim);
  }
  return {};
}

Result<std::vector<double>> state_overlaps(DeviceResources &res, const BasisSet &bra_basis,
                                           const Solution &bra, const BasisSet &ket_basis,
                                           const Solution &ket, const int32_t ncore,
                                           const int32_t ncas, const int32_t nelecas,
                                           const int32_t multiplicity) {
  const int32_t two_s = multiplicity - 1;
  if (two_s < 0 || nelecas < two_s || (nelecas - two_s) % 2 != 0 || ncas < 1 ||
      nelecas - ((nelecas - two_s) / 2) > ncas) {
    return err_config("mcscf::state_overlaps: {} active electrons of multiplicity {} in {} "
                      "orbitals",
                      nelecas, multiplicity, ncas);
  }
  const int32_t n_beta = (nelecas - two_s) / 2;
  const ci::StringSet alpha = FBF_TRY(ci::StringSet::create(ncas, nelecas - n_beta));
  const ci::StringSet beta = FBF_TRY(ci::StringSet::create(ncas, n_beta));
  const auto n_alpha_strings = static_cast<size_t>(alpha.size());
  const auto n_beta_strings = static_cast<size_t>(beta.size());
  const size_t dim = n_alpha_strings * n_beta_strings;
  FBF_TRY(overlap_check(res, bra_basis, bra, ncore, ncas, dim, "bra"));
  FBF_TRY(overlap_check(res, ket_basis, ket, ncore, ncas, dim, "ket"));
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // The MO overlap over the occupied orbitals, column-major nocc square:
  // element (p, q) = <phi_p(bra)|phi_q(ket)> = C_bra^T S_mixed C_ket.
  const auto nao_bra = static_cast<int32_t>(bra_basis.nao_sphe());
  const auto nao_ket = static_cast<int32_t>(ket_basis.nao_sphe());
  const int32_t nocc = ncore + ncas;
  const auto occ = static_cast<size_t>(nocc);
  Buffers::device<double> s_ao(static_cast<size_t>(nao_bra) * static_cast<size_t>(nao_ket),
                               res.device());
  FBF_TRY(overlap_mixed_spherical(res, bra_basis, ket_basis, s_ao));
  std::vector<double> s_mo(occ * occ);
  {
    const auto [half, mo] =
        res.device_regions<double>(static_cast<size_t>(nao_bra) * occ, occ * occ);
    FBF_TRY(gpu_status(gemm(blas, nao_bra, nocc, nao_ket, &kOne<double>, s_ao.data(), ket.c.data(),
                            &kZero<double>, half)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nocc, nocc, nao_bra, &kOne<double>,
                                               bra.c.data(), half, &kZero<double>, mo)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s_mo.data(), mo, occ * occ * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
  }

  // The core eliminated: det(S_cc) and the active block's Schur complement
  // S_aa - S_ac S_cc^-1 S_ca, row-major ncas square (README.md "The core").
  const auto n_core = static_cast<size_t>(ncore);
  const auto n = static_cast<size_t>(ncas);
  const auto at = [&](const size_t p, const size_t q) { return s_mo[p + (q * occ)]; };
  std::vector<double> core(n_core * n_core);
  std::vector<double> core_active(n_core * n);
  for (size_t i = 0; i < n_core; ++i) {
    for (size_t j = 0; j < n_core; ++j) {
      core[(i * n_core) + j] = at(i, j);
    }
    for (size_t u = 0; u < n; ++u) {
      core_active[(i * n) + u] = at(i, n_core + u);
    }
  }
  const double core_det = overlap_eliminate(core, n_core, core_active, n);
  if (std::abs(core_det) < kCoreOverlapFloor) {
    return err_unsupported("mcscf::state_overlaps: the two core blocks overlap with determinant "
                           "{:.3e}, below {:.0e}: the core orbitals are not carried into each "
                           "other",
                           core_det, kCoreOverlapFloor);
  }
  std::vector<double> active(n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      double sum = at(n_core + t, n_core + u);
      for (size_t i = 0; i < n_core; ++i) {
        sum -= at(n_core + t, i) * core_active[(i * n) + u];
      }
      active[(t * n) + u] = sum;
    }
  }
  const std::vector<double> alpha_overlap = overlap_strings(alpha, active, n);
  const std::vector<double> beta_overlap =
      n_beta == nelecas - n_beta ? alpha_overlap : overlap_strings(beta, active, n);

  // The contraction. A ket root column-major is C_J^T (n_beta_strings x
  // n_alpha_strings), so W_J = B C_J^T A^T is (A C_J B^T)^T, and S_IJ the
  // dot product of the bra column I with W_J: Y = B [C_J^T ...] in one gemm,
  // W_J = Y_J A^T per root, then S = C_bra^T W.
  const auto n_bra = static_cast<int32_t>(bra.roots.energies.size());
  const auto n_ket = static_cast<int32_t>(ket.roots.energies.size());
  const auto ket_roots = static_cast<size_t>(n_ket);
  const auto na = static_cast<int32_t>(n_alpha_strings);
  const auto nb = static_cast<int32_t>(n_beta_strings);
  const auto [a_device, b_device, y, w, s_device] = res.device_regions<double>(
      n_alpha_strings * n_alpha_strings, n_beta_strings * n_beta_strings, dim * ket_roots,
      dim * ket_roots, static_cast<size_t>(n_bra) * ket_roots);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_device, alpha_overlap.data(),
                                         alpha_overlap.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(b_device, beta_overlap.data(),
                                         beta_overlap.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(gemm(blas, nb, na * n_ket, nb, &kOne<double>, b_device,
                          ket.roots.vectors.data(), &kZero<double>, y)));
  for (size_t r = 0; r < ket_roots; ++r) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nb, na, na, &kOne<double>, y + (r * dim), a_device, &kZero<double>, w + (r * dim))));
  }
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_bra, n_ket, static_cast<int32_t>(dim),
                                             &kOne<double>, bra.roots.vectors.data(), w,
                                             &kZero<double>, s_device)));
  std::vector<double> s_column(static_cast<size_t>(n_bra) * ket_roots);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(s_column.data(), s_device, s_column.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());

  // Row-major, with the core's determinant once per spin.
  const double core_factor = core_det * core_det;
  const auto bra_roots = static_cast<size_t>(n_bra);
  std::vector<double> s(bra_roots * ket_roots);
  for (size_t i = 0; i < bra_roots; ++i) {
    for (size_t j = 0; j < ket_roots; ++j) {
      s[(i * ket_roots) + j] = core_factor * s_column[i + (j * bra_roots)];
    }
  }
  return s;
}

} // namespace free_breakfast::mcscf
