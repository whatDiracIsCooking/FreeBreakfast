/// @file
/// @brief fock: F^I and F^A from one J and one K each, the generalized
/// Fock from them, gamma and one gemm over (pu|vw), the gradient as its
/// antisymmetric part over the packed pairs, and the energy off their
/// diagonals; and the packed order's two copies each way.

module free_breakfast.mcscf.fock;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // geam, axpy, scal, wwrMemcpy2DAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// What both packing functions refuse; caller names the function.
Status check_orbital_pairs(const std::string_view caller, const int32_t nmo, const int32_t ncore,
                           const int32_t ncas) {
  if (ncore < 0 || ncas < 0) {
    return err_config("{}: ncore ({}) and ncas ({}) must not be negative", caller, ncore, ncas);
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, nmo)) {
    return err_config("{}: ncore {} + ncas {} exceeds the {} orbitals", caller, ncore, ncas, nmo);
  }
  return {};
}

// The packed order's two blocks, as 2D copies between the nmo x nmo matrix and
// the packed vector: from the matrix @p from into the vector @p to when
// to_packed, otherwise from the vector into the matrix. A block with no rows
// or no columns is skipped, since a zero-extent copy is not portable across
// the two runtimes.
Status copy_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                          const int32_t ncas, const double *const from, double *const to,
                          const bool to_packed) {
  const auto n = static_cast<size_t>(nmo);
  const auto core = static_cast<size_t>(ncore);
  const auto occ = core + static_cast<size_t>(ncas);
  const size_t n_rot = n - core; // the active and virtual rows of a core column
  const size_t n_vir = n - occ;  // the virtual rows of an active column
  const size_t pitch = n * sizeof(double);
  // (first row, row count, first column, column count, offset into the vector)
  const std::array<std::array<size_t, 5>, 2> blocks{
      {{core, n_rot, 0, core, 0}, {occ, n_vir, core, occ - core, core * n_rot}}};
  for (const auto &[row, rows, col, cols, at] : blocks) {
    if (rows == 0 || cols == 0) {
      continue;
    }
    const size_t width = rows * sizeof(double);
    const size_t offset = row + (n * col);
    FBF_TRY(gpu_status(
        wwr::wwrMemcpy2DAsync(to_packed ? to + at : to + offset, to_packed ? width : pitch,
                              to_packed ? from + offset : from + at, to_packed ? pitch : width,
                              width, cols, wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  return {};
}

Status pack_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                          const int32_t ncas, const double *const full, double *const packed) {
  FBF_TRY(check_orbital_pairs("pack_orbital_pairs", nmo, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  return copy_orbital_pairs(res, nmo, ncore, ncas, full, packed, true);
}

Status unpack_orbital_pairs(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                            const int32_t ncas, const double *const packed, double *const kappa) {
  FBF_TRY(check_orbital_pairs("unpack_orbital_pairs", nmo, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n = static_cast<size_t>(nmo);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(kappa, 0, n * n * sizeof(double), res.stream())));
  FBF_TRY(copy_orbital_pairs(res, nmo, ncore, ncas, packed, kappa, false));

  // The upper blocks, kappa_qp = -kappa_pq: each the negated transpose of a
  // lower block it shares no element with. geam's B is C itself at beta = 0,
  // its in-place mode, so nothing past kappa is read.
  const wwr::wwrblasHandle_t blas = res.blas().get();
  constexpr double kMinusOne = -1.0;
  const int32_t nocc = ncore + ncas;
  // (rows of the upper block, its columns, the lower block's first row, the
  // shared first column of the lower block and row of the upper one)
  const std::array<std::array<int32_t, 4>, 2> blocks{
      {{ncore, nmo - ncore, ncore, 0}, {ncas, nmo - nocc, nocc, ncore}}};
  for (const auto &[rows, cols, lower_row, first] : blocks) {
    if (rows == 0 || cols == 0) {
      continue;
    }
    const double *const lower = kappa + lower_row + (n * static_cast<size_t>(first));
    double *const upper = kappa + first + (n * static_cast<size_t>(lower_row));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, rows,
                                                  cols, &kMinusOne, lower, nmo, &kZero<double>,
                                                  upper, nmo, upper, nmo)));
  }
  return {};
}

// The call's device scratch, carved from one workspace request. coulomb and
// exchange, the only routines called while it is live, do not take the
// workspace.
struct CasscfFockScratch {
  double *c_cart; // U C, nao_cart x nmo
  double *work;   // gemmwich's intermediate, and C_act gamma: nao_cart x nmo
  double *block;  // one projected nmo x nmo matrix, then 2 (F - F^T)
  double *paaa;   // (pu|vw) at p + nmo (u + ncas (v + ncas w)), nmo ncas^3

  CasscfFockScratch(DeviceResources &res, const size_t n_cart, const size_t nmo,
                    const size_t ncas) {
    std::tie(c_cart, work, block, paaa) =
        res.device_regions<double>(n_cart * nmo, n_cart * nmo, nmo * nmo, nmo * ncas * ncas * ncas);
  }
};

Result<Fock> fock(DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
                  const SphericalMatrixAO &h,
                  const std::shared_ptr<const Buffers::device<double>> &u,
                  const std::span<CoulombPairClass> j_classes,
                  const std::span<OrderedPairClass> k_classes, const SphericalMatrixAO &c,
                  const int32_t ncore, const int32_t ncas, const Buffers::device<double> &ppaa,
                  const std::span<const double> gamma, const std::span<const double> gamma2,
                  const double tau_j, const double tau_k) {
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t n_cart = basis.nao_cart();
  const uint32_t n_orbitals = c.n();
  FBF_TRY(check_orbital_pairs("fock", static_cast<int32_t>(n_orbitals), ncore, ncas));
  if (ncore + ncas == 0) {
    return err_config("mcscf::fock: ncore and ncas are both 0, so nothing is occupied");
  }
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("mcscf::fock: tau_j ({}) and tau_k ({}) must be 0 or above", tau_j, tau_k);
  }
  if (h.n() != n_sphe || n_orbitals != n_sphe) {
    return err_config("mcscf::fock: h is {} x {} and C {} x {}, but the basis has {} spherical AOs",
                      h.n(), h.n(), n_orbitals, n_orbitals, n_sphe);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config("mcscf::fock: h is on device {} and C on {}, but the call passed device {}",
                      h.dev_idx(), c.dev_idx(), res.dev_idx());
  }
  const auto na = static_cast<size_t>(ncas);
  const size_t n_all = n_orbitals;
  if (ppaa.num_elements() != n_all * n_all * na * na) {
    return err_config("mcscf::fock: ppaa holds {} doubles, but nmo^2 ncas^2 is {}",
                      ppaa.num_elements(), n_all * n_all * na * na);
  }
  if (gamma.size() != na * na || gamma2.size() != na * na * na * na) {
    return err_config("mcscf::fock: gamma holds {} and Gamma {} doubles, but ncas is {}",
                      gamma.size(), gamma2.size(), ncas);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto ns = static_cast<int32_t>(n_sphe);
  const auto nc = static_cast<int32_t>(n_cart);
  const auto nmo = static_cast<int32_t>(n_orbitals);
  const int32_t nocc = ncore + ncas;
  const size_t nn = n_all * n_all;
  const auto len_block = static_cast<int32_t>(nn);
  const size_t n_pairs =
      (static_cast<size_t>(ncore) * (n_all - ncore)) + (na * (n_all - static_cast<size_t>(nocc)));

  Fock out{.energy = nuclear_repulsion_energy(molecule),
           .fock_inactive = Buffers::device<double>(nn, res.device()),
           .fock_active = Buffers::device<double>(nn, res.device()),
           .fock = Buffers::device<double>(nn, res.device()),
           .gradient = Buffers::device<double>(n_pairs, res.device()),
           .gamma = Buffers::device<double>(na * na, res.device()),
           .gamma2 = Buffers::device<double>(na * na * na * na, res.device())};

  // gamma and Gamma from ci::Rdm::rdm12's row-major order to the t-fastest one, so
  // (t, k) with k = u + ncas (v + ncas w) is a column-major ncas x ncas^3
  // matrix. Live until the final synchronize, which the uploads precede.
  std::vector<double> gamma_col(na * na);
  std::vector<double> gamma2_col(na * na * na * na);
  for (size_t t = 0; t < na; ++t) {
    for (size_t uu = 0; uu < na; ++uu) {
      gamma_col[t + (na * uu)] = gamma[(t * na) + uu];
      for (size_t v = 0; v < na; ++v) {
        for (size_t w = 0; w < na; ++w) {
          gamma2_col[t + (na * (uu + (na * (v + (na * w)))))] =
              gamma2[(((((t * na) + uu) * na) + v) * na) + w];
        }
      }
    }
  }
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.gamma.data(), gamma_col.data(),
                                           gamma_col.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.gamma2.data(), gamma2_col.data(),
                                           gamma2_col.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }

  // What coulomb and exchange read and write is their own matrix type, so
  // these three are allocated per call: P_c, then P_a; J; and K.
  CartesianMatrixAO density(res, basis);
  CartesianMatrixAO j(res, basis);
  CartesianMatrixAO k(res, basis);
  const CasscfFockScratch scratch(res, n_cart, n_all, na);
  constexpr double kMinusOne = -1.0;
  constexpr double kMinusHalf = -0.5;
  const auto len_cart = static_cast<int32_t>(j.num_elements());

  // U C, the orbitals over the Cartesian AOs the kernels evaluate.
  if (u != nullptr) {
    FBF_TRY(gpu_status(gemm(blas, nc, nmo, ns, &kOne<double>, u->data(), c.data(), &kZero<double>,
                            scratch.c_cart)));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scratch.c_cart, c.data(), sizeof(double) * nn,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  const double *const c_cart_act = scratch.c_cart + (n_cart * static_cast<size_t>(ncore));

  // F^I = C^T h C + (U C)^T (2 J[P_c] - K[P_c]) (U C); diag(C^T h C) is read
  // before G is added, for E_core.
  const auto diagonal = [&](std::vector<double> &to, const double *const matrix) {
    return gpu_status(wwr::wwrMemcpy2DAsync(to.data(), sizeof(double), matrix,
                                            (n_all + 1) * sizeof(double), sizeof(double), n_all,
                                            wwr::wwrMemcpyDeviceToHost, stream));
  };
  std::vector<double> h_diagonal(n_all);
  FBF_TRY(gpu_status(
      gemmwich(blas, nmo, ns, c.data(), h.data(), scratch.work, out.fock_inactive.data())));
  FBF_TRY(diagonal(h_diagonal, out.fock_inactive.data()));
  if (ncore > 0) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nc, nc, ncore, &kOne<double>, scratch.c_cart, scratch.c_cart, &kZero<double>,
        density.data())));
    FBF_TRY(coulomb(res, j_classes, density, j, tau_j));
    FBF_TRY(exchange(res, k_classes, density, k, tau_k));
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, len_cart, &kTwo<double>, j.data(), 1)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, len_cart, &kMinusOne, k.data(), 1, j.data(), 1)));
    FBF_TRY(
        gpu_status(gemmwich(blas, nmo, nc, scratch.c_cart, j.data(), scratch.work, scratch.block)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, len_block, &kOne<double>, scratch.block, 1,
                                                  out.fock_inactive.data(), 1)));
  }

  // F^A = (U C)^T (J[P_a] - K[P_a] / 2) (U C), P_a = (U C)_act gamma
  // (U C)_act^T built as s P_a at s tau and scaled back by 1 / s:
  // |(P_a)_cd| <= sum_tu |gamma_tu| m_t m_u, and the classes prune for
  // |D| <= 2. The m_t are read back once.
  if (ncas > 0) {
    std::vector<double> c_cart_host(n_cart * na);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_cart_host.data(), c_cart_act,
                                           sizeof(double) * c_cart_host.size(),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    std::vector<double> col_max(na, 0.0);
    for (size_t t = 0; t < na; ++t) {
      for (size_t mu = 0; mu < n_cart; ++mu) {
        col_max[t] = std::max(col_max[t], std::abs(c_cart_host[mu + (n_cart * t)]));
      }
    }
    double bound = 0.0;
    for (size_t t = 0; t < na; ++t) {
      for (size_t uu = 0; uu < na; ++uu) {
        bound += std::abs(gamma_col[t + (na * uu)]) * col_max[t] * col_max[uu];
      }
    }
    const double s = bound > 2.0 ? 2.0 / bound : 1.0;
    const double inverse = 1.0 / s;
    FBF_TRY(gpu_status(gemm(blas, nc, ncas, ncas, &kOne<double>, c_cart_act, out.gamma.data(),
                            &kZero<double>, scratch.work)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nc, nc, ncas, &s, scratch.work, c_cart_act, &kZero<double>, density.data())));
    FBF_TRY(coulomb(res, j_classes, density, j, tau_j * s));
    FBF_TRY(exchange(res, k_classes, density, k, tau_k * s));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, len_cart, &kMinusHalf, k.data(), 1, j.data(), 1)));
    FBF_TRY(gpu_status(
        gemmwich(blas, nmo, nc, scratch.c_cart, j.data(), scratch.work, out.fock_active.data())));
    if (s != 1.0) {
      FBF_TRY(gpu_status(
          wwr::scal<double, int32_t>(blas, len_block, &inverse, out.fock_active.data(), 1)));
    }
  }

  // The generalized Fock, column by column of q's space: 2 (F^I + F^A) for
  // the core, F^I gamma + (pu|vw) Gamma for the active space, 0 for the
  // virtuals.
  double *const f = out.fock.data();
  double *const f_act = f + (n_all * static_cast<size_t>(ncore));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(f, 0, nn * sizeof(double), stream)));
  if (ncore > 0) {
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, ncore, &kTwo<double>,
        out.fock_inactive.data(), nmo, &kTwo<double>, out.fock_active.data(), nmo, f, nmo)));
  }
  if (ncas > 0) {
    FBF_TRY(gpu_status(gemm(blas, nmo, ncas, ncas, &kOne<double>,
                            out.fock_inactive.data() + (n_all * static_cast<size_t>(ncore)),
                            out.gamma.data(), &kZero<double>, f_act)));
    // (pu|vw) is ppaa's columns q = ncore + u of each block (v, w): ncas
    // contiguous columns per block, gathered into one nmo x ncas^3 matrix.
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(scratch.paaa, n_all * na * sizeof(double),
                                             ppaa.data() + (n_all * static_cast<size_t>(ncore)),
                                             nn * sizeof(double), n_all * na * sizeof(double),
                                             na * na, wwr::wwrMemcpyDeviceToDevice, stream)));
    // F_pt += sum_k paaa_pk Gamma_tk, k = u + ncas (v + ncas w): Gamma as the
    // column-major ncas x ncas^3 matrix, transposed.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nmo, ncas, ncas * ncas * ncas, &kOne<double>, scratch.paaa, out.gamma2.data(),
        &kOne<double>, f_act)));
  }

  // g = 2 (F - F^T) over the packed pairs.
  constexpr double kMinusTwo = -2.0;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo,
                                                nmo, &kTwo<double>, f, nmo, &kMinusTwo, f, nmo,
                                                scratch.block, nmo)));
  FBF_TRY(copy_orbital_pairs(res, nmo, ncore, ncas, scratch.block, out.gradient.data(), true));

  // E = E_nuc + sum_i (h_ii + F^I_ii) + sum_tu F^I_tu gamma_tu
  //   + 1/2 sum_tuvw (tu|vw) Gamma_tuvw, and sum_t F_tt over the active t is
  // sum_tu F^I_tu gamma_ut + sum_tuvw (tu|vw) Gamma_tuvw: so the active part
  // is half of sum F^I gamma plus half of the active trace of F.
  std::vector<double> fock_inactive_diagonal(n_all);
  std::vector<double> fock_diagonal(n_all);
  std::vector<double> h_eff(na * na);
  FBF_TRY(diagonal(fock_inactive_diagonal, out.fock_inactive.data()));
  FBF_TRY(diagonal(fock_diagonal, f));
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
        h_eff.data(), na * sizeof(double),
        out.fock_inactive.data() + ncore + (n_all * static_cast<size_t>(ncore)),
        n_all * sizeof(double), na * sizeof(double), na, wwr::wwrMemcpyDeviceToHost, stream)));
  }
  FBF_TRY(res.sync());
  for (size_t i = 0; i < static_cast<size_t>(ncore); ++i) {
    out.energy += h_diagonal[i] + fock_inactive_diagonal[i];
  }
  double active = 0.0;
  for (size_t t = 0; t < na; ++t) {
    active += fock_diagonal[static_cast<size_t>(ncore) + t];
    for (size_t uu = 0; uu < na; ++uu) {
      active += h_eff[t + (na * uu)] * gamma_col[uu + (na * t)];
    }
  }
  out.energy += 0.5 * active;
  return out;
}

} // namespace free_breakfast::mcscf
