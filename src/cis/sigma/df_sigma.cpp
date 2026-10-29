/// @file
/// @brief DfSigma's checks, its memory check and MO blocks, and the fitted
/// operator's four products over a block.

module free_breakfast.cis.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kOne, kZero, kTwo, kNegOne
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // dgmm, gemmStridedBatched, wwrMemGetInfo, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.linear_operator;
import free_breakfast.integrals.eri.density_fitting.mo;

#include "diagnostics.h"

namespace free_breakfast::cis {

static_assert(LinearOperator<DfSigma, double>);

Result<DfSigma> DfSigma::create(DeviceResources &res, const Buffers::device<double> &b,
                                const int32_t naux, const SphericalMatrixAO &c, const int32_t n_occ,
                                const double *const delta, const Spin spin,
                                const int32_t max_block) {
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("cis::DfSigma::create: the MO coefficients are on device {}, but the call "
                      "passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  const auto nao = static_cast<int64_t>(c.n());
  if (naux < 1 || max_block < 1) {
    return err_config("cis::DfSigma::create: needs naux >= 1 and max_block >= 1 (naux = {}, "
                      "max_block = {})",
                      naux, max_block);
  }
  if (b.num_elements() != static_cast<size_t>(naux) * static_cast<size_t>(nao * nao)) {
    return err_config("cis::DfSigma::create: B holds {} elements, not naux * nao^2 = {} * {}^2",
                      b.num_elements(), naux, nao);
  }
  const int64_t n_virt = nao - n_occ;
  constexpr int64_t kIntMax = std::numeric_limits<int32_t>::max();
  // The int32_t dispatch: n_ov is the operator's order, naux n_virt the
  // exchange's first leading dimension (and b_mo's rows over the virtuals),
  // naux n_occ max_block its second.
  if (n_occ < 1 || n_virt < 1 || n_occ * n_virt > kIntMax || naux * n_virt > kIntMax ||
      naux * int64_t{n_occ} * max_block > kIntMax) {
    return err_config("cis::DfSigma::create: needs 1 <= n_occ < nao, and n_occ * n_virt, "
                      "naux * n_virt and naux * n_occ * max_block to fit an int32_t (n_occ = {}, "
                      "nao = {}, naux = {}, max_block = {})",
                      n_occ, nao, naux, max_block);
  }
  if (delta == nullptr) {
    return err_config("cis::DfSigma::create: delta is null");
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  // Past the device's memory an allocation aborts (CreatePolicy) rather than
  // returns, so the total is refused here, before any: the three MO blocks,
  // and the workspace's growth to the larger of b_mo's half transform (naux
  // n_virt nao, over the virtual rows) and apply's two regions.
  const auto aux = static_cast<size_t>(naux);
  const auto occ = static_cast<size_t>(n_occ);
  const auto virt = static_cast<size_t>(n_virt);
  const auto block = static_cast<size_t>(max_block);
  const size_t blocks_bytes = sizeof(double) * aux * ((occ * virt) + (occ * occ) + (virt * virt));
  const auto aligned = [](const size_t count) {
    return math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
  };
  const size_t scratch_bytes = std::max(aligned(aux * virt * static_cast<size_t>(nao)),
                                        aligned(aux * block) + aligned(aux * occ * virt * block));
  const size_t growth_bytes =
      scratch_bytes > res.device_workspace(0).num_elements() ? scratch_bytes : 0;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (blocks_bytes + growth_bytes > free_bytes) {
    return err_unsupported("cis::DfSigma::create: the MO blocks of B over {} fitting functions, "
                           "{} occupied and {} virtual orbitals ({} bytes) and {} bytes of "
                           "workspace growth for a block of {} need {} bytes of device memory, "
                           "and device {} has {} free",
                           naux, n_occ, n_virt, blocks_bytes, growth_bytes, max_block,
                           blocks_bytes + growth_bytes, res.dev_idx(), free_bytes);
  }

  DfSigma sigma{res, naux, n_occ, static_cast<int32_t>(n_virt), delta, spin, max_block};
  const auto nmo = static_cast<int32_t>(nao);
  const OrbitalRange occupied{.first = 0, .count = n_occ};
  const OrbitalRange virtuals{.first = n_occ, .count = static_cast<int32_t>(n_virt)};
  // B_vo with the virtual index fast: element (Q, a, i) at Q + naux (a + n_virt
  // i), so column a + n_virt i of the naux x n_ov matrix is excitation ia.
  FBF_TRY(b_mo(res, naux, nmo, nmo, b.data(), c.data(), virtuals, occupied, sigma.b_vo_.data()));
  FBF_TRY(b_mo(res, naux, nmo, nmo, b.data(), c.data(), occupied, occupied, sigma.b_oo_.data()));
  FBF_TRY(b_mo(res, naux, nmo, nmo, b.data(), c.data(), virtuals, virtuals, sigma.b_vv_.data()));
  // apply's regions, grown to here so that no apply allocates them. The
  // workspace only grows, so a later, larger user leaves it big enough.
  static_cast<void>(res.device_regions<double>(aux * block, aux * occ * virt * block));
  return sigma;
}

DfSigma::DfSigma(DeviceResources &res, const int32_t naux, const int32_t n_occ,
                 const int32_t n_virt, const double *const delta, const Spin spin,
                 const int32_t max_block)
    : dev_idx_{res.dev_idx()}, naux_{naux}, n_occ_{n_occ}, n_virt_{n_virt}, max_block_{max_block},
      spin_{spin}, delta_{delta},
      b_vo_(static_cast<size_t>(naux) * static_cast<size_t>(n_occ) * static_cast<size_t>(n_virt),
            res.device()),
      b_oo_(static_cast<size_t>(naux) * static_cast<size_t>(n_occ) * static_cast<size_t>(n_occ),
            res.device()),
      b_vv_(static_cast<size_t>(naux) * static_cast<size_t>(n_virt) * static_cast<size_t>(n_virt),
            res.device()) {}

Status DfSigma::apply(DeviceResources &res, const int32_t k, const double *const x,
                      double *const y) {
  if (res.dev_idx() != dev_idx_ || k < 0) {
    return err_config("cis::DfSigma::apply: needs the sigma's device {} and k >= 0 (device {}, "
                      "k = {})",
                      dev_idx_, res.dev_idx(), k);
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t n_ov = size();
  const auto aux = static_cast<size_t>(naux_);
  const auto block = static_cast<size_t>(max_block_);
  // T = B_vo X, naux x k, and the exchange's intermediate W, naux n_ov k: the
  // sizes create grew the workspace to, so this allocates nothing.
  const auto [t, w] =
      res.device_regions<double>(aux * block, aux * static_cast<size_t>(n_ov) * block);

  // σ = Δε ∘ X, over the whole block; every term after accumulates into it.
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_ov, k, x, n_ov,
                                                delta_, 1, y, n_ov)));
  for (int32_t first = 0; first < k; first += max_block_) {
    const int32_t cols = std::min(max_block_, k - first);
    const double *const xs = x + (static_cast<size_t>(first) * static_cast<size_t>(n_ov));
    double *const ys = y + (static_cast<size_t>(first) * static_cast<size_t>(n_ov));

    // The Coulomb term, a singlet's alone: σ += 2 B_voᵀ (B_vo X).
    if (spin_ == Spin::Singlet) {
      FBF_TRY(gpu_status(
          gemm(blas, naux_, cols, n_ov, &kOne<double>, b_vo_.data(), xs, &kZero<double>, t)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_ov, cols, naux_, &kTwo<double>,
                                                 b_vo_.data(), t, &kOne<double>, ys)));
    }

    // The exchange term, σ_ja' −= Σ_ib (ji|a'b) X_ib, in two batched gemms.
    // A column of X read as the n_virt x n_occ matrix Xᵀ, and the slice's
    // columns side by side, is the n_virt x (n_occ cols) matrix at xs.
    //
    // 1. Per a', W_a' = B_vv(:, a', :) X: naux x (n_occ cols) from the
    //    naux x n_virt slice of B_vv at Q + naux a' (leading dimension naux
    //    n_virt). W's element (Q, i, c, a') lands at Q + naux (i + n_occ (c +
    //    cols a')), so at fixed c its (Q, i) rows are adjacent.
    const int32_t w_rows = naux_ * n_occ_; // (Q, i)
    const int32_t w_lead = w_rows * cols;
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux_, n_occ_ * cols, n_virt_, &kOne<double>,
        b_vv_.data(), naux_ * n_virt_, naux_, xs, n_virt_, 0, &kZero<double>, w, naux_, w_lead,
        n_virt_)));
    // 2. Per column c, σ_c −= W_cᵀ B_oo: W_c is the (naux n_occ) x n_virt
    //    matrix at naux n_occ c (leading dimension naux n_occ cols), B_oo the
    //    (naux n_occ) x n_occ one whose row (Q, i) and column j hold B_Q,ij,
    //    and the product, element a' + n_virt j, is σ's column as stored.
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n_virt_, n_occ_, w_rows, &kNegOne<double>, w,
        w_lead, w_rows, b_oo_.data(), w_rows, 0, &kOne<double>, ys, n_virt_, n_ov, cols)));
  }
  return {};
}

} // namespace free_breakfast::cis
