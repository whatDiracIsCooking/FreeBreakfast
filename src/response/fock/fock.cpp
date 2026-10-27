/// @file
/// @brief fock: F^I from one J and one K, integral-direct or density-fitted,
/// F^A from two gemvs over ppaa and papa, the generalized Fock from them,
/// gamma and one gemm over (pu|vw), the gradient as its antisymmetric part
/// over the packed pairs, and the energy off their diagonals; and the packed
/// order's two copies each way.

module free_breakfast.response.fock;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // geam, gemv, axpy, scal, wwrMemcpy2DAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.eri.density_fitting.jk;

#include "diagnostics.h"

namespace free_breakfast::response {

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

// What both fock overloads refuse past their own arguments: the spaces, and
// ppaa's, papa's, gamma's and Gamma's lengths.
Status check_fock(const uint32_t n_orbitals, const int32_t ncore, const int32_t ncas,
                  const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
                  const std::span<const double> gamma, const std::span<const double> gamma2) {
  FBF_TRY(check_orbital_pairs("fock", static_cast<int32_t>(n_orbitals), ncore, ncas));
  if (ncore + ncas == 0) {
    return err_config("response::fock: ncore and ncas are both 0, so nothing is occupied");
  }
  const auto na = static_cast<size_t>(ncas);
  const size_t n_all = n_orbitals;
  if (ppaa.num_elements() != n_all * n_all * na * na ||
      papa.num_elements() != n_all * n_all * na * na) {
    return err_config("response::fock: ppaa holds {} and papa {} doubles, but nmo^2 ncas^2 is {}",
                      ppaa.num_elements(), papa.num_elements(), n_all * n_all * na * na);
  }
  if (gamma.size() != na * na || gamma2.size() != na * na * na * na) {
    return err_config("response::fock: gamma holds {} and Gamma {} doubles, but ncas is {}",
                      gamma.size(), gamma2.size(), ncas);
  }
  return {};
}

// The n x n diagonal of the column-major matrix into the host vector to,
// queued on res.stream() and not waited for.
Status download_diagonal(DeviceResources &res, std::vector<double> &to,
                         const double *const matrix) {
  const size_t n = to.size();
  return gpu_status(wwr::wwrMemcpy2DAsync(to.data(), sizeof(double), matrix,
                                          (n + 1) * sizeof(double), sizeof(double), n,
                                          wwr::wwrMemcpyDeviceToHost, res.stream()));
}

// What both fock overloads hold for the call: the result, with gamma and
// Gamma uploaded from their t-fastest host copies (which must outlive the
// final synchronize, and which the energy reads), and diag(C^T h C), read back
// before F^I's two-electron part is added, for E_core.
struct FockBuild {
  Fock out;
  std::vector<double> gamma_col;
  std::vector<double> gamma2_col;
  std::vector<double> h_diagonal;
};

// The result's buffers, gamma and Gamma uploaded, F^I = C^T h C (through
// work, nao x nmo), and its diagonal queued for the host.
Result<FockBuild> begin_fock(DeviceResources &res, const Molecule &molecule,
                             const SphericalMatrixAO &h, const SphericalMatrixAO &c,
                             const int32_t ncore, const int32_t ncas,
                             const std::span<const double> gamma,
                             const std::span<const double> gamma2, double *const work) {
  const size_t n_all = c.n();
  const size_t nn = n_all * n_all;
  const auto na = static_cast<size_t>(ncas);
  const size_t n_pairs = (static_cast<size_t>(ncore) * (n_all - static_cast<size_t>(ncore))) +
                         (na * (n_all - static_cast<size_t>(ncore) - na));
  FockBuild build{.out = {.energy = nuclear_repulsion_energy(molecule),
                          .fock_inactive = Buffers::device<double>(nn, res.device()),
                          .fock_active = Buffers::device<double>(nn, res.device()),
                          .fock = Buffers::device<double>(nn, res.device()),
                          .gradient = Buffers::device<double>(n_pairs, res.device()),
                          .gamma = Buffers::device<double>(na * na, res.device()),
                          .gamma2 = Buffers::device<double>(na * na * na * na, res.device())},
                  .gamma_col = std::vector<double>(na * na),
                  .gamma2_col = std::vector<double>(na * na * na * na),
                  .h_diagonal = std::vector<double>(n_all)};

  // gamma and Gamma from ci::Rdm::rdm12's row-major order to the t-fastest
  // one, so (t, k) with k = u + ncas (v + ncas w) is a column-major
  // ncas x ncas^3 matrix, and gamma the ncas^2 vector of (t, u) pairs in
  // ppaa's and papa's block order.
  for (size_t t = 0; t < na; ++t) {
    for (size_t uu = 0; uu < na; ++uu) {
      build.gamma_col[t + (na * uu)] = gamma[(t * na) + uu];
      for (size_t v = 0; v < na; ++v) {
        for (size_t w = 0; w < na; ++w) {
          build.gamma2_col[t + (na * (uu + (na * (v + (na * w)))))] =
              gamma2[(((((t * na) + uu) * na) + v) * na) + w];
        }
      }
    }
  }
  const wwr::wwrStream_t stream = res.stream();
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(build.out.gamma.data(), build.gamma_col.data(),
                                           build.gamma_col.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(build.out.gamma2.data(), build.gamma2_col.data(),
                                           build.gamma2_col.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }
  const auto nmo = static_cast<int32_t>(n_all);
  FBF_TRY(gpu_status(gemmwich(res.blas().get(), nmo, static_cast<int32_t>(h.n()), c.data(),
                              h.data(), work, build.out.fock_inactive.data())));
  FBF_TRY(download_diagonal(res, build.h_diagonal, build.out.fock_inactive.data()));
  return build;
}

// F^A from ppaa and papa, the generalized Fock, the gradient and the energy,
// over the F^I in build. block is nmo x nmo and paaa nmo ncas^3 of scratch.
Result<Fock> finish_fock(DeviceResources &res, FockBuild &build, const int32_t ncore,
                         const int32_t ncas, const Buffers::device<double> &ppaa,
                         const Buffers::device<double> &papa, double *const block,
                         double *const paaa) {
  Fock &out = build.out;
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const size_t n_all = build.h_diagonal.size();
  const size_t nn = n_all * n_all;
  const auto na = static_cast<size_t>(ncas);
  const auto nmo = static_cast<int32_t>(n_all);
  constexpr double kMinusHalf = -0.5;

  // F^A_pq = sum_tu gamma_tu [(pq|tu) - 1/2 (pt|qu)]: ppaa and papa as
  // nmo^2 x ncas^2 matrices, their column t + ncas u block (t, u), against
  // gamma as the vector of the same pairs.
  if (ncas > 0) {
    const auto rows = static_cast<int32_t>(nn);
    const int32_t cols = ncas * ncas;
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, rows, cols, &kOne<double>, ppaa.data(), rows, out.gamma.data(), 1,
        &kZero<double>, out.fock_active.data(), 1)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_N, rows, cols, &kMinusHalf,
                                                  papa.data(), rows, out.gamma.data(), 1,
                                                  &kOne<double>, out.fock_active.data(), 1)));
  } else {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(out.fock_active.data(), 0, nn * sizeof(double), stream)));
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
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(paaa, n_all * na * sizeof(double),
                                             ppaa.data() + (n_all * static_cast<size_t>(ncore)),
                                             nn * sizeof(double), n_all * na * sizeof(double),
                                             na * na, wwr::wwrMemcpyDeviceToDevice, stream)));
    // F_pt += sum_k paaa_pk Gamma_tk, k = u + ncas (v + ncas w): Gamma as the
    // column-major ncas x ncas^3 matrix, transposed.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nmo, ncas, ncas * ncas * ncas, &kOne<double>, paaa, out.gamma2.data(), &kOne<double>,
        f_act)));
  }

  // g = 2 (F - F^T) over the packed pairs.
  constexpr double kMinusTwo = -2.0;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo,
                                                nmo, &kTwo<double>, f, nmo, &kMinusTwo, f, nmo,
                                                block, nmo)));
  FBF_TRY(copy_orbital_pairs(res, nmo, ncore, ncas, block, out.gradient.data(), true));

  // E = E_nuc + sum_i (h_ii + F^I_ii) + sum_tu F^I_tu gamma_tu
  //   + 1/2 sum_tuvw (tu|vw) Gamma_tuvw, and sum_t F_tt over the active t is
  // sum_tu F^I_tu gamma_ut + sum_tuvw (tu|vw) Gamma_tuvw: so the active part
  // is half of sum F^I gamma plus half of the active trace of F.
  std::vector<double> fock_inactive_diagonal(n_all);
  std::vector<double> fock_diagonal(n_all);
  std::vector<double> h_eff(na * na);
  FBF_TRY(download_diagonal(res, fock_inactive_diagonal, out.fock_inactive.data()));
  FBF_TRY(download_diagonal(res, fock_diagonal, f));
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
        h_eff.data(), na * sizeof(double),
        out.fock_inactive.data() + ncore + (n_all * static_cast<size_t>(ncore)),
        n_all * sizeof(double), na * sizeof(double), na, wwr::wwrMemcpyDeviceToHost, stream)));
  }
  FBF_TRY(res.sync());
  for (size_t i = 0; i < static_cast<size_t>(ncore); ++i) {
    out.energy += build.h_diagonal[i] + fock_inactive_diagonal[i];
  }
  double active = 0.0;
  for (size_t t = 0; t < na; ++t) {
    active += fock_diagonal[static_cast<size_t>(ncore) + t];
    for (size_t uu = 0; uu < na; ++uu) {
      active += h_eff[t + (na * uu)] * build.gamma_col[uu + (na * t)];
    }
  }
  out.energy += 0.5 * active;
  return std::move(out);
}

// The integral-direct call's device scratch, carved from one workspace
// request. coulomb and exchange, the only routines called while it is live,
// do not take the workspace.
struct CasscfFockScratch {
  double *c_cart; // U C, nao_cart x nmo
  double *work;   // gemmwich's intermediate, nao_cart x nmo
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
                  const Buffers::device<double> &papa, const std::span<const double> gamma,
                  const std::span<const double> gamma2, const double tau_j, const double tau_k) {
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t n_cart = basis.nao_cart();
  const uint32_t n_orbitals = c.n();
  FBF_TRY(check_fock(n_orbitals, ncore, ncas, ppaa, papa, gamma, gamma2));
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("response::fock: tau_j ({}) and tau_k ({}) must be 0 or above", tau_j, tau_k);
  }
  if (h.n() != n_sphe || n_orbitals != n_sphe) {
    return err_config(
        "response::fock: h is {} x {} and C {} x {}, but the basis has {} spherical AOs", h.n(),
        h.n(), n_orbitals, n_orbitals, n_sphe);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config(
        "response::fock: h is on device {} and C on {}, but the call passed device {}", h.dev_idx(),
        c.dev_idx(), res.dev_idx());
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto ns = static_cast<int32_t>(n_sphe);
  const auto nc = static_cast<int32_t>(n_cart);
  const auto nmo = static_cast<int32_t>(n_orbitals);
  const size_t n_all = n_orbitals;
  const auto len_block = static_cast<int32_t>(n_all * n_all);

  // What coulomb and exchange read and write is their own matrix type, so
  // these three are allocated per call: P_c, J and K.
  CartesianMatrixAO density(res, basis);
  CartesianMatrixAO j(res, basis);
  CartesianMatrixAO k(res, basis);
  const CasscfFockScratch scratch(res, n_cart, n_all, static_cast<size_t>(ncas));
  constexpr double kMinusOne = -1.0;
  const auto len_cart = static_cast<int32_t>(j.num_elements());

  FockBuild build =
      FBF_TRY(begin_fock(res, molecule, h, c, ncore, ncas, gamma, gamma2, scratch.work));

  // U C, the orbitals over the Cartesian AOs the kernels evaluate.
  if (u != nullptr) {
    FBF_TRY(gpu_status(gemm(blas, nc, nmo, ns, &kOne<double>, u->data(), c.data(), &kZero<double>,
                            scratch.c_cart)));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scratch.c_cart, c.data(), sizeof(double) * n_all * n_all,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }

  // F^I += (U C)^T (2 J[P_c] - K[P_c]) (U C).
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
                                                  build.out.fock_inactive.data(), 1)));
  }
  return finish_fock(res, build, ncore, ncas, ppaa, papa, scratch.block, scratch.paaa);
}

Result<Fock> fock(DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
                  const Buffers::device<double> &b, const int32_t naux, const SphericalMatrixAO &c,
                  const int32_t ncore, const int32_t ncas, const Buffers::device<double> &ppaa,
                  const Buffers::device<double> &papa, const std::span<const double> gamma,
                  const std::span<const double> gamma2) {
  const uint32_t n_orbitals = c.n();
  FBF_TRY(check_fock(n_orbitals, ncore, ncas, ppaa, papa, gamma, gamma2));
  if (naux < 0) {
    return err_config("response::fock: naux ({}) must not be negative", naux);
  }
  if (h.n() != n_orbitals) {
    return err_config("response::fock: h is {} x {} but C {} x {}", h.n(), h.n(), n_orbitals,
                      n_orbitals);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config(
        "response::fock: h is on device {} and C on {}, but the call passed device {}", h.dev_idx(),
        c.dev_idx(), res.dev_idx());
  }
  const size_t n_all = n_orbitals;
  const size_t nn = n_all * n_all;
  if (b.num_elements() != static_cast<size_t>(naux) * nn) {
    return err_config("response::fock: B holds {} doubles, but naux nmo^2 is {}", b.num_elements(),
                      static_cast<size_t>(naux) * nn);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto nmo = static_cast<int32_t>(n_orbitals);
  const auto len_block = static_cast<int32_t>(nn);
  constexpr double kMinusHalf = -0.5;

  // 2 P_c, then gemmwich's intermediate; J; and K, then the projected G.
  // Allocated per call: df_coulomb and df_exchange take the workspace.
  Buffers::device<double> density(nn, res.device());
  Buffers::device<double> j(nn, res.device());
  Buffers::device<double> k(nn, res.device());

  FockBuild build =
      FBF_TRY(begin_fock(res, molecule, h, c, ncore, ncas, gamma, gamma2, density.data()));

  // F^I += C^T (J[D] - K[D] / 2) C at D = 2 P_c, df_exchange's own D for the
  // factor C_core.
  if (ncore > 0) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, nmo, nmo, ncore, &kTwo<double>, c.data(), c.data(), &kZero<double>, density.data())));
    FBF_TRY(df_coulomb(res, nmo, naux, b.data(), density.data(), j.data()));
    FBF_TRY(df_exchange(res, nmo, naux, ncore, b.data(), c.data(), k.data()));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, len_block, &kMinusHalf, k.data(), 1, j.data(), 1)));
    FBF_TRY(gpu_status(gemmwich(blas, nmo, nmo, c.data(), j.data(), density.data(), k.data())));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, len_block, &kOne<double>, k.data(), 1,
                                                  build.out.fock_inactive.data(), 1)));
  }
  const auto [block, paaa] =
      res.device_regions<double>(nn, n_all * static_cast<size_t>(ncas) * static_cast<size_t>(ncas) *
                                         static_cast<size_t>(ncas));
  return finish_fock(res, build, ncore, ncas, ppaa, papa, block, paaa);
}

} // namespace free_breakfast::response
