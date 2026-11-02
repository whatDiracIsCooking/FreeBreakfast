/// @file
/// @brief The eight SC-NEVPT2 class energies: the NEVPT2 demo's energy.cpp
/// and energy_sc_classes.cppm (nevpt2-gpu-demo, commit 7c5a674), each
/// deviceEinsum* call an einsum with the same subscripts, the three
/// diagonal slices and the Sijrs reduction launched from nevpt2_energy.cu,
/// and the host finish, PySCF's _norm_to_energy. Each class walks n states
/// at once, for sc_effective_hamiltonian's couplings; class_energies is n = 1.

module;

// The launchers, defined in nevpt2_energy.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "nevpt2_energy_bridge.h"

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import :shared;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// Every array this unit makes is row-major, because einsum writes its output
// packed row-major: that is the demo's layout, so its subscripts and its host
// finish carry over as written. The integrals arrive column-major
// (nevpt2.density_fitting's layout) and enter each einsum through their
// strides.
//
// The RDM-only intermediates (a16, ...) and each class's slabs are held across
// nevpt2::slab, which uses res.device_workspace() for its gemm product, so
// they are buffers of their own (docs/style.md "Scratch memory"). fdm2, which
// lives only between two einsums of one make_*, is the workspace's.

// The einsum operand of a column-major array: strides 1, n0, n0 n1, ...
EinsumOperand column_major(const double *data, std::vector<int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t k = 1; k < dims.size(); ++k) {
    strides[k] = strides[k - 1] * dims[k - 1];
  }
  return {.data = data, .dims = std::move(dims), .strides = std::move(strides)};
}

// A zeroed device buffer of @p count doubles: the demo's deviceEinsumNew is
// this, then an accumulating einsum.
Result<Buffers::device<double>> device_zeros(DeviceResources &res, const int64_t count) {
  Buffers::device<double> out(static_cast<size_t>(count), res.device());
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
        out.data(), 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  }
  return out;
}

// @p count doubles of workspace, zeroed.
Result<double *> workspace_zeros(DeviceResources &res, const int64_t count) {
  double *const out = std::get<0>(res.device_regions<double>(static_cast<size_t>(count)));
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(out, 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  return out;
}

// @p count doubles of @p d, read back once the stream has written them.
Result<std::vector<double>> download(DeviceResources &res, const double *d, const int64_t count) {
  std::vector<double> host(static_cast<size_t>(count));
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * host.size(),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
  }
  FBF_TRY(res.sync());
  return host;
}

// pyscf.mrpt.nevpt2._norm_to_energy with its numerator an argument of its
// own: -Σ_k T_k / (Δ_k + H_k / N_k) over the perturbers whose norm N_k does
// not vanish, and the plain Σ_k T_k. T = N is PySCF's verbatim, a state's own
// energy; a transition norm T_IJ over the ket's N and H is a QD coupling.
std::pair<double, double> norm_to_energy(const std::vector<double> &t,
                                         const std::vector<double> &norm,
                                         const std::vector<double> &h,
                                         const std::vector<double> &diff) {
  double norm_total = 0.0;
  double energy_total = 0.0;
  for (size_t k = 0; k < norm.size(); ++k) {
    norm_total += t[k];
    if (std::abs(norm[k]) > kNumericalZero) {
      energy_total -= t[k] / (diff[k] + (h[k] / norm[k]));
    }
  }
  return {norm_total, energy_total};
}

// The class walk of ns states (class_energies is ns = 1) accumulates each
// slab into ns² + ns tables of one value per perturber: table I ns + J is the
// norm einsum of pair (I, J), so at I = J state J's own N^J, and table
// ns² + J is state J's H^J.
int64_t pair_table_count(const int64_t ns) {
  return (ns * ns) + ns;
}

// One slab's tables, read back, @p count values each.
Result<std::vector<std::vector<double>>> download_pair_tables(DeviceResources &res,
                                                              const double *tables,
                                                              const int64_t ns,
                                                              const int64_t count) {
  const std::vector<double> all = FBF_TRY(download(res, tables, pair_table_count(ns) * count));
  std::vector<std::vector<double>> out;
  for (int64_t t = 0; t < pair_table_count(ns); ++t) {
    out.emplace_back(all.begin() + (t * count), all.begin() + ((t + 1) * count));
  }
  return out;
}

// Adds one slab's share onto @p out[I ns + J] from its tables, each already
// the perturber list @p diff runs over (Sijr and Srsi pack theirs first):
// norm_to_energy over J's N^J and H^J with T = N^J at I = J, and otherwise T
// the mean of the norm einsums of (I, J) and (J, I). PySCF folds two
// Hermitian-conjugate terms of Sr's, Si's and Sir's norms into one (its
// `* 2.0`), which holds only at I = J; the mean is the unfolded sum, and
// leaves the five classes already symmetric in I and J as they are.
void add_pair_shares(const std::vector<std::vector<double>> &tables, const int64_t ns,
                     const std::vector<double> &diff, std::vector<std::pair<double, double>> &out) {
  for (int64_t j = 0; j < ns; ++j) {
    const std::vector<double> &norm = tables[static_cast<size_t>((j * ns) + j)];
    const std::vector<double> &h = tables[static_cast<size_t>((ns * ns) + j)];
    for (int64_t i = 0; i < ns; ++i) {
      std::vector<double> mean;
      if (i != j) {
        const std::vector<double> &ij = tables[static_cast<size_t>((i * ns) + j)];
        const std::vector<double> &ji = tables[static_cast<size_t>((j * ns) + i)];
        mean.resize(ij.size());
        for (size_t k = 0; k < ij.size(); ++k) {
          mean[k] = 0.5 * (ij[k] + ji[k]);
        }
      }
      const auto [n_part, e_part] = norm_to_energy(i == j ? norm : mean, norm, h, diff);
      std::pair<double, double> &sums = out[static_cast<size_t>((i * ns) + j)];
      sums.first += n_part;
      sums.second += e_part;
    }
  }
}

// The walk's step: @p batch, or the whole extent in one piece.
int32_t batch_step(const int32_t extent, const int32_t batch) {
  return batch > 0 ? batch : std::max(extent, 1);
}

// The dimensions of @p block's slab over [b0, b1).
std::vector<int64_t> slab_dims(const MoBlocks &blocks, const ExtBlock block, const int32_t b0,
                               const int32_t b1) {
  std::array<int32_t, 4> shape = block_shape(blocks, block);
  shape[static_cast<size_t>(batch_axis(block))] = b1 - b0;
  return {shape[0], shape[1], shape[2], shape[3]};
}

// A slab buffer for @p block walked in steps of @p step.
Buffers::device<double> slab_buffer(DeviceResources &res, const MoBlocks &blocks,
                                    const ExtBlock block, const int32_t step) {
  const std::array<int32_t, 4> shape = block_shape(blocks, block);
  const int32_t extent = shape[static_cast<size_t>(batch_axis(block))];
  const std::vector<int64_t> dims = slab_dims(blocks, block, 0, std::min(step, extent));
  const int64_t count = std::ranges::fold_left(dims, int64_t{1}, std::multiplies{});
  return Buffers::device<double>(static_cast<size_t>(count), res.device());
}

// The extent @p block's walk covers.
int32_t walk_extent(const MoBlocks &blocks, const ExtBlock block) {
  return block_shape(blocks, block)[static_cast<size_t>(batch_axis(block))];
}

// ---------------------------------------------------------------------------
// The class metrics and Dyall Hamiltonians (the demo's make_*), each n^k
// row-major. h1e and h2e are the active integrals, dm1..dm3, f3ca and f3ac the
// RDMs, delta the n x n identity.
// ---------------------------------------------------------------------------

Result<Buffers::device<double>> make_a16(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm3,
                                         const EinsumOperand &f3ac, const EinsumOperand &f3ca) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm3.dims[0];
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  Buffers::device<double> a16 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
  double *const a = a16.data();
  FBF_TRY(einsum(s, "ib,rpqiac->pqrabc", {h1e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "ia,rpqbic->pqrabc", {h1e, dm3}, a, d6, 1.0));
  FBF_TRY(einsum(s, "ci,rpqbai->pqrabc", {h1e, dm3}, a, d6, -1.0));
  FBF_TRY(transpose(s, f3ca, {1, 4, 0, 2, 5, 3}, a, -1.0));
  FBF_TRY(einsum(s, "kbia,rpqcki->pqrabc", {h2e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "kbaj,rpqjkc->pqrabc", {h2e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "cbij,rpqjai->pqrabc", {h2e, dm3}, a, d6, 1.0));

  double *const fdm2 = FBF_TRY(workspace_zeros(res, n * n * n * n));
  FBF_TRY(einsum(s, "kbij,rpajki->prab", {h2e, dm3}, fdm2, d4, 1.0));
  FBF_TRY(gpu_status(device::diag_a16(s, fdm2, a, static_cast<int32_t>(n))));

  FBF_TRY(transpose(s, f3ac, {1, 2, 0, 4, 3, 5}, a, 1.0));
  FBF_TRY(transpose(s, f3ca, {1, 2, 0, 4, 3, 5}, a, -1.0));
  FBF_TRY(einsum(s, "jbij,rpqiac->pqrabc", {h2e, dm3}, a, d6, 1.0));
  FBF_TRY(einsum(s, "cjka,rpqbjk->pqrabc", {h2e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "jcij,rpqbai->pqrabc", {h2e, dm3}, a, d6, 1.0));
  return a16;
}

Result<Buffers::device<double>> make_a22(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &f3ac,
                                         const EinsumOperand &f3ca) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm3.dims[0];
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  Buffers::device<double> a22 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
  double *const a = a22.data();
  FBF_TRY(einsum(s, "pb,kipjac->ijkabc", {h1e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "pa,kibjpc->ijkabc", {h1e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "cp,kibjap->ijkabc", {h1e, dm3}, a, d6, 1.0));
  FBF_TRY(einsum(s, "cqra,kibjqr->ijkabc", {h2e, dm3}, a, d6, 1.0));
  FBF_TRY(einsum(s, "qcpq,kibjap->ijkabc", {h2e, dm3}, a, d6, -1.0));
  FBF_TRY(transpose(s, f3ac, {1, 5, 0, 2, 4, 3}, a, -1.0));

  // fdm2a is dead once diag_a22a has read it, but the stream has not run that
  // yet when fdm2b is zeroed: two regions, not one.
  const int64_t n4 = n * n * n * n;
  const auto [fdm2a, fdm2b] =
      res.device_regions<double>(static_cast<size_t>(n4), static_cast<size_t>(n4));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(fdm2a, 0, sizeof(double) * static_cast<size_t>(n4), s)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(fdm2b, 0, sizeof(double) * static_cast<size_t>(n4), s)));
  FBF_TRY(einsum(s, "pqrb,kiqcpr->ikbc", {h2e, dm3}, fdm2a, d4, 1.0));
  FBF_TRY(gpu_status(device::diag_a22a(s, fdm2a, a, static_cast<int32_t>(n))));

  FBF_TRY(einsum(s, "pqab,kiqjpc->ijkabc", {h2e, dm3}, a, d6, -1.0));
  FBF_TRY(einsum(s, "pcrb,kiajpr->ijkabc", {h2e, dm3}, a, d6, 1.0));
  FBF_TRY(einsum(s, "cqrb,kiqjar->ijkabc", {h2e, dm3}, a, d6, 1.0));
  FBF_TRY(transpose(s, f3ac, {1, 3, 0, 4, 2, 5}, a, -1.0));
  FBF_TRY(transpose(s, f3ca, {1, 3, 0, 4, 2, 5}, a, 1.0));
  FBF_TRY(einsum(s, "jb,kiac->ijkabc", {h1e, dm2}, a, d6, 2.0));
  FBF_TRY(einsum(s, "pjrb,kiprac->ijkabc", {h2e, dm3}, a, d6, 2.0));

  FBF_TRY(einsum(s, "pa,kipc->ikac", {h1e, dm2}, fdm2b, d4, 1.0));
  FBF_TRY(einsum(s, "cp,kiap->ikac", {h1e, dm2}, fdm2b, d4, -1.0));
  FBF_TRY(einsum(s, "cqra,kiqr->ikac", {h2e, dm2}, fdm2b, d4, -1.0));
  FBF_TRY(einsum(s, "qcpq,kiap->ikac", {h2e, dm2}, fdm2b, d4, 1.0));
  FBF_TRY(einsum(s, "pqra,kiqcpr->ikac", {h2e, dm3}, fdm2b, d4, 1.0));
  FBF_TRY(einsum(s, "rcpq,kiaqrp->ikac", {h2e, dm3}, fdm2b, d4, -1.0));
  FBF_TRY(gpu_status(device::diag_a22b(s, fdm2b, a, static_cast<int32_t>(n))));
  return a22;
}

Result<Buffers::device<double>> make_dm3_h(DeviceResources &res, const EinsumOperand &dm2,
                                           const EinsumOperand &dm3, const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  Buffers::device<double> dm3_h = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
  FBF_TRY(
      einsum(s, "abef,cd->abcdef", {dm2, delta}, dm3_h.data(), std::vector<int64_t>(6, n), 2.0));
  FBF_TRY(transpose(s, dm3, {0, 1, 3, 2, 4, 5}, dm3_h.data(), -1.0));
  return dm3_h;
}

Result<Buffers::device<double>> make_dm2_h(DeviceResources &res, const EinsumOperand &dm1,
                                           const EinsumOperand &dm2, const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  Buffers::device<double> dm2_h = FBF_TRY(device_zeros(res, n * n * n * n));
  FBF_TRY(einsum(s, "ab,cd->abcd", {dm1, delta}, dm2_h.data(), std::vector<int64_t>(4, n), 2.0));
  FBF_TRY(transpose(s, dm2, {0, 1, 3, 2}, dm2_h.data(), -1.0));
  return dm2_h;
}

// 2 delta - dm1^T. Si's dm1_h is the same expression, so it is this.
Result<Buffers::device<double>> make_hdm1(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &delta, const double overlap) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  Buffers::device<double> hdm1 = FBF_TRY(device_zeros(res, n * n));
  if (overlap != 0.0) {
    FBF_TRY(einsum(s, "ij->ij", {delta}, hdm1.data(), {n, n}, 2.0 * overlap));
  }
  FBF_TRY(transpose(s, dm1, {1, 0}, hdm1.data(), -1.0));
  return hdm1;
}

Result<Buffers::device<double>> make_hdm2(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &delta,
                                          const double overlap) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> dm2b = FBF_TRY(device_zeros(res, n * n * n * n));
  FBF_TRY(einsum(s, "ikjl->ijkl", {dm2}, dm2b.data(), d4, 1.0));
  FBF_TRY(einsum(s, "jk,il->ijkl", {delta, dm1}, dm2b.data(), d4, -1.0));
  const EinsumOperand dm2b_op{.data = dm2b.data(), .dims = d4};
  Buffers::device<double> hdm2 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const h = hdm2.data();
  FBF_TRY(einsum(s, "klij->ijkl", {dm2b_op}, h, d4, 1.0));
  FBF_TRY(einsum(s, "il,kj->ijkl", {delta, dm1}, h, d4, 1.0));
  FBF_TRY(einsum(s, "jk,li->ijkl", {delta, dm1}, h, d4, 1.0));
  FBF_TRY(einsum(s, "ik,lj->ijkl", {delta, dm1}, h, d4, -2.0));
  FBF_TRY(einsum(s, "jl,ki->ijkl", {delta, dm1}, h, d4, -2.0));
  if (overlap != 0.0) {
    FBF_TRY(einsum(s, "il,jk->ijkl", {delta, delta}, h, d4, -2.0 * overlap));
    FBF_TRY(einsum(s, "ik,jl->ijkl", {delta, delta}, h, d4, 4.0 * overlap));
  }
  return hdm2;
}

Result<Buffers::device<double>> make_hdm3(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &dm3,
                                          const EinsumOperand &hdm2, const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm3.dims[0];
  const std::vector<int64_t> d6(6, n);
  Buffers::device<double> hdm3 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
  double *const h = hdm3.data();
  FBF_TRY(einsum(s, "pb,qrac->pqrabc", {delta, hdm2}, h, d6, -1.0));
  FBF_TRY(einsum(s, "br,pqac->pqrabc", {delta, hdm2}, h, d6, -1.0));
  FBF_TRY(einsum(s, "bq,prac->pqrabc", {delta, hdm2}, h, d6, 2.0));
  FBF_TRY(einsum(s, "ap,bqcr->pqrabc", {delta, dm2}, h, d6, 2.0));
  FBF_TRY(einsum(s, "ap,cr,bq->pqrabc", {delta, delta, dm1}, h, d6, -4.0));
  FBF_TRY(einsum(s, "cr,bqap->pqrabc", {delta, dm2}, h, d6, 2.0));
  FBF_TRY(einsum(s, "bqapcr->pqrabc", {dm3}, h, d6, -1.0));
  FBF_TRY(einsum(s, "ar,pc,bq->pqrabc", {delta, delta, dm1}, h, d6, 2.0));
  FBF_TRY(einsum(s, "ar,bqcp->pqrabc", {delta, dm2}, h, d6, -1.0));
  return hdm3;
}

Result<Buffers::device<double>> make_a3(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &dm1,
                                        const EinsumOperand &dm2, const EinsumOperand &hdm1,
                                        const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d2{n, n};
  Buffers::device<double> a3 = FBF_TRY(device_zeros(res, n * n));
  FBF_TRY(einsum(s, "ia,ip->pa", {h1e, hdm1}, a3.data(), d2, 1.0));
  FBF_TRY(einsum(s, "ijka,pj,ik->pa", {h2e, delta, dm1}, a3.data(), d2, 2.0));
  FBF_TRY(einsum(s, "ijka,jpik->pa", {h2e, dm2}, a3.data(), d2, -1.0));
  return a3;
}

Result<Buffers::device<double>> make_k27(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  const std::vector<int64_t> d2{n, n};
  Buffers::device<double> k27 = FBF_TRY(device_zeros(res, n * n));
  FBF_TRY(einsum(s, "ai,pi->pa", {h1e, dm1}, k27.data(), d2, -1.0));
  FBF_TRY(einsum(s, "iajk,pkij->pa", {h2e, dm2}, k27.data(), d2, -1.0));
  FBF_TRY(einsum(s, "iaji,pj->pa", {h2e, dm1}, k27.data(), d2, 1.0));
  return k27;
}

Result<Buffers::device<double>> make_rm3(DeviceResources &res, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &rm2,
                                         const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d6(6, n);
  Buffers::device<double> rm3 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
  FBF_TRY(einsum(s, "injmkl->ijklmn", {dm3}, rm3.data(), d6, 1.0));
  FBF_TRY(einsum(s, "jn,imkl->ijklmn", {delta, dm2}, rm3.data(), d6, -1.0));
  FBF_TRY(einsum(s, "km,ijln->ijklmn", {delta, rm2}, rm3.data(), d6, -1.0));
  FBF_TRY(einsum(s, "kn,ijml->ijklmn", {delta, rm2}, rm3.data(), d6, -1.0));
  return rm3;
}

// Srs's metric, alone: what the QD walk reads of a transition.
Result<Buffers::device<double>> make_rm2(DeviceResources &res, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> rm2 = FBF_TRY(device_zeros(res, n * n * n * n));
  FBF_TRY(einsum(s, "iljk->ijkl", {dm2}, rm2.data(), d4, 1.0));
  FBF_TRY(einsum(s, "ik,jl->ijkl", {dm1, delta}, rm2.data(), d4, -1.0));
  return rm2;
}

// {rm2, a7}: Srs's metric and its Dyall Hamiltonian.
Result<std::pair<Buffers::device<double>, Buffers::device<double>>>
make_a7(DeviceResources &res, const EinsumOperand &h1e, const EinsumOperand &h2e,
        const EinsumOperand &dm1, const EinsumOperand &dm2, const EinsumOperand &dm3,
        const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> rm2 = FBF_TRY(make_rm2(res, dm1, dm2, delta));
  const EinsumOperand rm2_op{.data = rm2.data(), .dims = d4};

  const Buffers::device<double> rm3 = FBF_TRY(make_rm3(res, dm2, dm3, rm2_op, delta));
  const EinsumOperand rm3_op{.data = rm3.data(), .dims = std::vector<int64_t>(6, n)};

  Buffers::device<double> a7 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const a = a7.data();
  FBF_TRY(einsum(s, "bi,pqia->pqab", {h1e, rm2_op}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ai,pqbi->pqab", {h1e, rm2_op}, a, d4, -1.0));
  FBF_TRY(einsum(s, "kbij,pqkija->pqab", {h2e, rm3_op}, a, d4, -1.0));
  FBF_TRY(einsum(s, "kaij,pqkibj->pqab", {h2e, rm3_op}, a, d4, -1.0));
  FBF_TRY(einsum(s, "baij,pqij->pqab", {h2e, rm2_op}, a, d4, -1.0));
  return std::pair{std::move(rm2), std::move(a7)};
}

Result<Buffers::device<double>> make_a9(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &hdm2,
                                        const EinsumOperand &hdm3) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = hdm2.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> a9 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const a = a9.data();
  FBF_TRY(einsum(s, "ib,pqai->pqab", {h1e, hdm2}, a, d4, 1.0));
  FBF_TRY(einsum(s, "ijib,pqaj->pqab", {h2e, hdm2}, a, d4, 2.0));
  FBF_TRY(einsum(s, "ijjb,pqai->pqab", {h2e, hdm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ijkb,pkqaij->pqab", {h2e, hdm3}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ia,pqib->pqab", {h1e, hdm2}, a, d4, 1.0));
  FBF_TRY(einsum(s, "ijja,pqib->pqab", {h2e, hdm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ijba,pqji->pqab", {h2e, hdm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ijia,pqjb->pqab", {h2e, hdm2}, a, d4, 2.0));
  FBF_TRY(einsum(s, "ijka,pqkjbi->pqab", {h2e, hdm3}, a, d4, -1.0));
  return a9;
}

Result<Buffers::device<double>> make_a12(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> a12 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const a = a12.data();
  FBF_TRY(einsum(s, "ia,qpib->pqab", {h1e, dm2}, a, d4, 1.0));
  FBF_TRY(einsum(s, "bi,qpai->pqab", {h1e, dm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ijka,qpjbik->pqab", {h2e, dm3}, a, d4, 1.0));
  FBF_TRY(einsum(s, "kbij,qpajki->pqab", {h2e, dm3}, a, d4, -1.0));
  FBF_TRY(einsum(s, "bjka,qpjk->pqab", {h2e, dm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "jbij,qpai->pqab", {h2e, dm2}, a, d4, 1.0));
  return a12;
}

Result<Buffers::device<double>> make_a13(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3,
                                         const EinsumOperand &delta) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm3.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> a13 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const a = a13.data();
  FBF_TRY(einsum(s, "ia,qbip->pqab", {h1e, dm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "pa,qb->pqab", {h1e, dm1}, a, d4, 2.0));
  FBF_TRY(einsum(s, "bi,qiap->pqab", {h1e, dm2}, a, d4, 1.0));
  FBF_TRY(einsum(s, "pa,bi,qi->pqab", {delta, h1e, dm1}, a, d4, -2.0));
  FBF_TRY(einsum(s, "ijka,qbjpik->pqab", {h2e, dm3}, a, d4, -1.0));
  FBF_TRY(einsum(s, "kbij,qjapki->pqab", {h2e, dm3}, a, d4, 1.0));
  FBF_TRY(einsum(s, "blma,qmlp->pqab", {h2e, dm2}, a, d4, 1.0));
  FBF_TRY(einsum(s, "kpma,qbkm->pqab", {h2e, dm2}, a, d4, 2.0));
  FBF_TRY(einsum(s, "bpma,qm->pqab", {h2e, dm1}, a, d4, -2.0));
  FBF_TRY(einsum(s, "lbkl,qkap->pqab", {h2e, dm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "ap,mbkl,qlmk->pqab", {delta, h2e, dm2}, a, d4, -2.0));
  FBF_TRY(einsum(s, "ap,lbkl,qk->pqab", {delta, h2e, dm1}, a, d4, 2.0));
  return a13;
}

// make_a17 / make_a19 are Sr's (with make_a16), make_a23 / make_a25 Si's (with
// make_a22).

Result<Buffers::device<double>> make_a17(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm2.dims[0];
  const std::vector<int64_t> d2{n, n};
  Buffers::device<double> h1e_buf = FBF_TRY(device_zeros(res, n * n));
  FBF_TRY(einsum(s, "mn->mn", {h1e_in}, h1e_buf.data(), d2, 1.0));
  FBF_TRY(einsum(s, "mjjn->mn", {h2e}, h1e_buf.data(), d2, -1.0));
  const EinsumOperand h1e{.data = h1e_buf.data(), .dims = d2};
  Buffers::device<double> a17 = FBF_TRY(device_zeros(res, n * n * n * n));
  const std::vector<int64_t> d4(4, n);
  FBF_TRY(einsum(s, "pi,cabi->abcp", {h1e, dm2}, a17.data(), d4, -1.0));
  FBF_TRY(einsum(s, "kpij,cabjki->abcp", {h2e, dm3}, a17.data(), d4, -1.0));
  return a17;
}

Result<Buffers::device<double>> make_a19(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  const std::vector<int64_t> d2{n, n};
  Buffers::device<double> h1e_buf = FBF_TRY(device_zeros(res, n * n));
  FBF_TRY(einsum(s, "mn->mn", {h1e_in}, h1e_buf.data(), d2, 1.0));
  FBF_TRY(einsum(s, "mjjn->mn", {h2e}, h1e_buf.data(), d2, -1.0));
  const EinsumOperand h1e{.data = h1e_buf.data(), .dims = d2};
  Buffers::device<double> a19 = FBF_TRY(device_zeros(res, n * n));
  FBF_TRY(einsum(s, "pi,ai->ap", {h1e, dm1}, a19.data(), d2, -1.0));
  FBF_TRY(einsum(s, "kpij,ajki->ap", {h2e, dm2}, a19.data(), d2, -1.0));
  return a19;
}

Result<Buffers::device<double>> make_a23(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  Buffers::device<double> a23 = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const a = a23.data();
  FBF_TRY(einsum(s, "ip,caib->abcp", {h1e, dm2}, a, d4, -1.0));
  FBF_TRY(einsum(s, "pijk,cajbik->abcp", {h2e, dm3}, a, d4, -1.0));
  FBF_TRY(einsum(s, "bp,ca->abcp", {h1e, dm1}, a, d4, 2.0));
  FBF_TRY(einsum(s, "pibk,caik->abcp", {h2e, dm2}, a, d4, 2.0));
  return a23;
}

Result<Buffers::device<double>> make_a25(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = dm1.dims[0];
  const std::vector<int64_t> d2{n, n};
  Buffers::device<double> a25 = FBF_TRY(device_zeros(res, n * n));
  double *const a = a25.data();
  FBF_TRY(einsum(s, "pi,ai->ap", {h1e, dm1}, a, d2, -1.0));
  FBF_TRY(einsum(s, "pijk,jaik->ap", {h2e, dm2}, a, d2, -1.0));
  FBF_TRY(einsum(s, "ap->pa", {h1e}, a, d2, 2.0));
  FBF_TRY(einsum(s, "piaj,ij->ap", {h2e, dm1}, a, d2, 2.0));
  return a25;
}

// ---------------------------------------------------------------------------
// The eight classes, over ns states at once. Each walks its batch index
// (batch_axis), building one slab of its external block per step into a plain
// device buffer, contracting it against every state's intermediates and every
// pair's norm intermediates into the slab's pair tables (pair_table_count),
// and finishing it on the host: the per-perturber finish is elementwise, so
// splitting it by slab only reassociates the sum. Srsi alone fills its whole
// (v, v, c) tables first, because its symmetrization pairs (r, s) with (s, r)
// across slabs. @p ops is ns x ns row-major, bra row, ket column: a class's
// Dyall blocks (a16, ...) are built from each diagonal entry, its norm
// intermediates from every entry, a transition's (I != J) without the terms
// that carry ⟨I|J⟩ = 0. At ns = 1 every einsum is the one-state walk's, in
// its order.
// ---------------------------------------------------------------------------

// Sr: over a slab of h2e_v_Sr, (nr, a, a, a), and h1e_v_Sr's rows.
Result<std::vector<std::pair<double, double>>>
energy_sr(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in, const int32_t batch,
          std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  std::vector<Buffers::device<double>> a16;
  std::vector<Buffers::device<double>> a17;
  std::vector<Buffers::device<double>> a19;
  for (int64_t j = 0; j < ns; ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a16.push_back(FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca)));
    a17.push_back(FBF_TRY(make_a17(res, o.h1e, o.h2e, o.dm2, o.dm3)));
    a19.push_back(FBF_TRY(make_a19(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }

  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sr);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sr, step);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};

    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * nb));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const ActiveOperands &o = ops[static_cast<size_t>((i * ns) + j)];
        double *const norm = tables + (((i * ns) + j) * nb);
        if (i == j) {
          const EinsumOperand a16_op{.data = a16[static_cast<size_t>(j)].data(),
                                     .dims = std::vector<int64_t>(6, n)};
          const EinsumOperand a17_op{.data = a17[static_cast<size_t>(j)].data(), .dims = d4};
          const EinsumOperand a19_op{.data = a19[static_cast<size_t>(j)].data(), .dims = d2};
          double *const ener = tables + (((ns * ns) + j) * nb);
          FBF_TRY(einsum(s, "ipqr,pqrabc,iabc->i", {h2e_v, a16_op, h2e_v}, ener, {nb}, 1.0));
          FBF_TRY(einsum(s, "ipqr,pqra,ia->i", {h2e_v, a17_op, h1e_v}, ener, {nb}, 2.0));
          FBF_TRY(einsum(s, "ip,pa,ia->i", {h1e_v, a19_op, h1e_v}, ener, {nb}, 1.0));
        }
        FBF_TRY(einsum(s, "ipqr,rpqbac,iabc->i", {h2e_v, o.dm3, h2e_v}, norm, {nb}, 1.0));
        FBF_TRY(einsum(s, "ipqr,rpqa,ia->i", {h2e_v, o.dm2, h1e_v}, norm, {nb}, 2.0));
        FBF_TRY(einsum(s, "ip,pa,ia->i", {h1e_v, o.dm1, h1e_v}, norm, {nb}, 1.0));
      }
    }

    const std::vector<double> diff(e_virt.begin() + b0, e_virt.begin() + b1);
    add_pair_shares(FBF_TRY(download_pair_tables(res, tables, ns, nb)), ns, diff, out);
  }
  return out;
}

// Si: over a slab of h2e_v_Si, (a, a, ni, a), and h1e_v_Si's columns.
Result<std::vector<std::pair<double, double>>>
energy_si(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in, const int32_t batch,
          std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  std::vector<Buffers::device<double>> a22;
  std::vector<Buffers::device<double>> a23;
  std::vector<Buffers::device<double>> a25;
  for (int64_t j = 0; j < ns; ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a22.push_back(FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca)));
    a23.push_back(FBF_TRY(make_a23(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3)));
    a25.push_back(FBF_TRY(make_a25(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }
  // Per pair: dm1_h's 2δ is ⟨I|J⟩'s.
  std::vector<Buffers::device<double>> dm3_h;
  std::vector<Buffers::device<double>> dm2_h;
  std::vector<Buffers::device<double>> dm1_h;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    const double overlap = p / ns == p % ns ? 1.0 : 0.0;
    dm3_h.push_back(FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta)));
    dm2_h.push_back(FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta)));
    dm1_h.push_back(FBF_TRY(make_hdm1(res, o.dm1, o.delta, overlap)));
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const int32_t extent = walk_extent(blocks, ExtBlock::Si);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Si, step);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};

    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * nb));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const auto p = static_cast<size_t>((i * ns) + j);
        double *const norm = tables + (((i * ns) + j) * nb);
        if (i == j) {
          const EinsumOperand a22_op{.data = a22[static_cast<size_t>(j)].data(), .dims = d6};
          const EinsumOperand a23_op{.data = a23[static_cast<size_t>(j)].data(), .dims = d4};
          const EinsumOperand a25_op{.data = a25[static_cast<size_t>(j)].data(), .dims = d2};
          double *const ener = tables + (((ns * ns) + j) * nb);
          FBF_TRY(einsum(s, "qpir,pqrabc,baic->i", {h2e_v, a22_op, h2e_v}, ener, {nb}, 1.0));
          FBF_TRY(einsum(s, "qpir,pqra,ai->i", {h2e_v, a23_op, h1e_v}, ener, {nb}, 2.0));
          FBF_TRY(einsum(s, "pi,pa,ai->i", {h1e_v, a25_op, h1e_v}, ener, {nb}, 1.0));
        }
        const EinsumOperand dm3_h_op{.data = dm3_h[p].data(), .dims = d6};
        const EinsumOperand dm2_h_op{.data = dm2_h[p].data(), .dims = d4};
        const EinsumOperand dm1_h_op{.data = dm1_h[p].data(), .dims = d2};
        FBF_TRY(einsum(s, "qpir,rpqbac,baic->i", {h2e_v, dm3_h_op, h2e_v}, norm, {nb}, 1.0));
        FBF_TRY(einsum(s, "qpir,rpqa,ai->i", {h2e_v, dm2_h_op, h1e_v}, norm, {nb}, 2.0));
        FBF_TRY(einsum(s, "pi,pa,ai->i", {h1e_v, dm1_h_op, h1e_v}, norm, {nb}, 1.0));
      }
    }

    std::vector<double> diff(e_core.begin() + b0, e_core.begin() + b1);
    for (double &x : diff) {
      x = -x;
    }
    add_pair_shares(FBF_TRY(download_pair_tables(res, tables, ns, nb)), ns, diff, out);
  }
  return out;
}

// Sijrs: no RDM at all, the MP2-like class, reduced on the device one
// core-orbital slab at a time. PySCF applies no vanishing-norm guard here.
Result<std::pair<double, double>> energy_sijrs(DeviceResources &res, const MoBlocks &blocks,
                                               const EnergyInputs &in, const int32_t batch) {
  if (blocks.ncore == 0 || blocks.nvirt == 0) {
    return std::pair{0.0, 0.0};
  }
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 2));
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijrs);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    FBF_TRY(slab(res, blocks, ExtBlock::Sijrs, b0, b1, slab_buf.data()));
    FBF_TRY(gpu_status(device::sijrs(res.stream(), slab_buf.data(), b1 - b0, b0, blocks.ncore,
                                     blocks.nvirt, in.e_core, in.e_virt, acc.data())));
  }
  const std::vector<double> host = FBF_TRY(download(res, acc.data(), 2));
  return std::pair{host[0], host[1]};
}

// Sijr: over a slab of h2e_v_Sijr, (nr, a, c, c). The (r, j, i) tables are
// symmetrized in (i, j) with the diagonal halved, then packed over j <= i with
// Delta = eps_r - eps_j - eps_i.
Result<std::vector<std::pair<double, double>>>
energy_sijr(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
            std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const std::vector<int64_t> d2{n, n};
  // Per pair hdm1, whose 2δ is ⟨I|J⟩'s; per state a3, from its own.
  std::vector<Buffers::device<double>> hdm1;
  std::vector<Buffers::device<double>> a3;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    const bool own = p / ns == p % ns;
    hdm1.push_back(FBF_TRY(make_hdm1(res, o.dm1, o.delta, own ? 1.0 : 0.0)));
    if (own) {
      const EinsumOperand hdm1_op{.data = hdm1.back().data(), .dims = d2};
      a3.push_back(FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta)));
    }
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijr);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijr, step);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sijr, b0, b1));

    const int64_t count = nr * nc * nc;
    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * count));
    const std::vector<int64_t> out_dims{nr, nc, nc};
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const EinsumOperand hdm1_op{.data = hdm1[static_cast<size_t>((i * ns) + j)].data(),
                                    .dims = d2};
        double *const norm = tables + (((i * ns) + j) * count);
        FBF_TRY(einsum(s, "rpji,raji,pa->rji", {h2e_v, h2e_v, hdm1_op}, norm, out_dims, 2.0));
        FBF_TRY(einsum(s, "rpji,raij,pa->rji", {h2e_v, h2e_v, hdm1_op}, norm, out_dims, -1.0));
        if (i == j) {
          const EinsumOperand a3_op{.data = a3[static_cast<size_t>(j)].data(), .dims = d2};
          double *const h = tables + (((ns * ns) + j) * count);
          FBF_TRY(einsum(s, "rpji,raji,pa->rji", {h2e_v, h2e_v, a3_op}, h, out_dims, 2.0));
          FBF_TRY(einsum(s, "rpji,raij,pa->rji", {h2e_v, h2e_v, a3_op}, h, out_dims, -1.0));
        }
      }
    }
    std::vector<std::vector<double>> packed = FBF_TRY(download_pair_tables(res, tables, ns, count));

    const auto at = [&](const int64_t r, const int64_t j, const int64_t i) {
      return static_cast<size_t>((((r * nc) + j) * nc) + i);
    };
    for (std::vector<double> &table : packed) {
      std::vector<double> sym;
      for (int64_t r = 0; r < nr; ++r) {
        for (int64_t j = 0; j < nc; ++j) {
          for (int64_t i = j; i < nc; ++i) {
            const double scale = i == j ? 0.5 : 1.0;
            sym.push_back(scale * (table[at(r, j, i)] + table[at(r, i, j)]));
          }
        }
      }
      table = std::move(sym);
    }
    std::vector<double> diff;
    for (int64_t r = 0; r < nr; ++r) {
      for (int64_t j = 0; j < nc; ++j) {
        for (int64_t i = j; i < nc; ++i) {
          diff.push_back(e_virt[static_cast<size_t>(b0 + r)] - e_core[static_cast<size_t>(j)] -
                         e_core[static_cast<size_t>(i)]);
        }
      }
    }
    add_pair_shares(packed, ns, diff, out);
  }
  return out;
}

// Srsi: over slabs of h2e_v_Srsi batched along its first virtual (hA) and its
// second (hB), filling rows [b0, b1) of the whole (v, v, c) tables; then,
// once, symmetrized in (r, s) with the diagonal halved and packed over
// r <= s with Delta = eps_r + eps_s - eps_i. The tables are v² c doubles
// each: the slabs, not the output, are what grow with the active space.
Result<std::vector<std::pair<double, double>>>
energy_srsi(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
            std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  std::vector<Buffers::device<double>> k27;
  for (int64_t j = 0; j < ns; ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    k27.push_back(FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }

  const int64_t count = nv * nv * nc;
  Buffers::device<double> tables = FBF_TRY(device_zeros(res, pair_table_count(ns) * count));
  const int32_t extent = walk_extent(blocks, ExtBlock::Srsi);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_a = slab_buffer(res, blocks, ExtBlock::Srsi, step);
  Buffers::device<double> slab_b = slab_buffer(res, blocks, ExtBlock::SrsiT, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    // h[r_batch, s, i, a] and h[s, r_batch, i, a].
    const EinsumOperand h_a =
        column_major(slab_a.data(), slab_dims(blocks, ExtBlock::Srsi, b0, b1));
    const EinsumOperand h_b =
        column_major(slab_b.data(), slab_dims(blocks, ExtBlock::SrsiT, b0, b1));
    // Rows [b0, b1) of each row-major (v, v, c) table: contiguous.
    const std::vector<int64_t> rows_dims{nb, nv, nc};
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const ActiveOperands &o = ops[static_cast<size_t>((i * ns) + j)];
        double *const norm_rows = tables.data() + (((i * ns) + j) * count) + (b0 * nv * nc);
        FBF_TRY(einsum(s, "rsip,rsia,pa->rsi", {h_a, h_a, o.dm1}, norm_rows, rows_dims, 2.0));
        FBF_TRY(einsum(s, "rsip,sria,pa->rsi", {h_a, h_b, o.dm1}, norm_rows, rows_dims, -1.0));
        if (i == j) {
          const EinsumOperand k27_op{.data = k27[static_cast<size_t>(j)].data(), .dims = {n, n}};
          double *const h_rows = tables.data() + (((ns * ns) + j) * count) + (b0 * nv * nc);
          FBF_TRY(einsum(s, "rsip,rsia,pa->rsi", {h_a, h_a, k27_op}, h_rows, rows_dims, 2.0));
          FBF_TRY(einsum(s, "rsip,sria,pa->rsi", {h_a, h_b, k27_op}, h_rows, rows_dims, -1.0));
        }
      }
    }
  }

  std::vector<std::vector<double>> packed =
      FBF_TRY(download_pair_tables(res, tables.data(), ns, count));
  const auto at = [&](const int64_t r, const int64_t ss, const int64_t i) {
    return static_cast<size_t>((((r * nv) + ss) * nc) + i);
  };
  for (std::vector<double> &table : packed) {
    std::vector<double> sym;
    for (int64_t r = 0; r < nv; ++r) {
      for (int64_t ss = r; ss < nv; ++ss) {
        for (int64_t i = 0; i < nc; ++i) {
          const double scale = r == ss ? 0.5 : 1.0;
          sym.push_back(scale * (table[at(r, ss, i)] + table[at(ss, r, i)]));
        }
      }
    }
    table = std::move(sym);
  }
  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;
  std::vector<double> diff;
  for (int64_t r = 0; r < nv; ++r) {
    for (int64_t ss = r; ss < nv; ++ss) {
      for (int64_t i = 0; i < nc; ++i) {
        diff.push_back(e_virt[static_cast<size_t>(r)] + e_virt[static_cast<size_t>(ss)] -
                       e_core[static_cast<size_t>(i)]);
      }
    }
  }
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  add_pair_shares(packed, ns, diff, out);
  return out;
}

// Srs: over a slab of h2e_v_Srs, (nr, v, a, a). The metric is read `pqba`,
// the Dyall block `pqab`.
Result<std::vector<std::pair<double, double>>>
energy_srs(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
           std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  if (nv == 0) {
    return out;
  }
  const std::vector<int64_t> d4(4, n);
  // Per pair rm2; per state a7, built beside its own rm2.
  std::vector<Buffers::device<double>> rm2;
  std::vector<Buffers::device<double>> a7;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    if (p / ns == p % ns) {
      auto [own_rm2, own_a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
      rm2.push_back(std::move(own_rm2));
      a7.push_back(std::move(own_a7));
    } else {
      rm2.push_back(FBF_TRY(make_rm2(res, o.dm1, o.dm2, o.delta)));
    }
  }

  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srs);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Srs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Srs, b0, b1));

    const int64_t count = nb * nv;
    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const EinsumOperand rm2_op{.data = rm2[static_cast<size_t>((i * ns) + j)].data(),
                                   .dims = d4};
        double *const norm = tables + (((i * ns) + j) * count);
        FBF_TRY(einsum(s, "rsqp,rsba,pqba->rs", {h2e_v, h2e_v, rm2_op}, norm, {nb, nv}, 0.5));
        if (i == j) {
          const EinsumOperand a7_op{.data = a7[static_cast<size_t>(j)].data(), .dims = d4};
          double *const h = tables + (((ns * ns) + j) * count);
          FBF_TRY(einsum(s, "rsqp,rsba,pqab->rs", {h2e_v, h2e_v, a7_op}, h, {nb, nv}, 0.5));
        }
      }
    }

    std::vector<double> diff;
    for (int64_t r = b0; r < b1; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        diff.push_back(e_virt[static_cast<size_t>(r)] + e_virt[static_cast<size_t>(ss)]);
      }
    }
    add_pair_shares(FBF_TRY(download_pair_tables(res, tables, ns, count)), ns, diff, out);
  }
  return out;
}

// Sij: over a slab of h2e_v_Sij, (a, a, ni, c). One term each side, the same
// subscripts: hdm2 for the norm, a9 for H.
Result<std::vector<std::pair<double, double>>>
energy_sij(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
           std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  if (nc == 0) {
    return out;
  }
  const std::vector<int64_t> d4(4, n);
  // Per pair hdm2, whose δδ terms are ⟨I|J⟩'s; per state hdm3 and a9, from
  // its own. The demo also builds hdm1 here and hands it to make_hdm3 and
  // make_a9, which never read it; it is not built.
  std::vector<Buffers::device<double>> hdm2;
  std::vector<Buffers::device<double>> a9;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    const bool own = p / ns == p % ns;
    hdm2.push_back(FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta, own ? 1.0 : 0.0)));
    if (own) {
      const EinsumOperand hdm2_op{.data = hdm2.back().data(), .dims = d4};
      const Buffers::device<double> hdm3 =
          FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
      const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
      a9.push_back(FBF_TRY(make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op)));
    }
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sij);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sij, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sij, b0, b1));

    const int64_t count = nb * nc;
    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const EinsumOperand hdm2_op{.data = hdm2[static_cast<size_t>((i * ns) + j)].data(),
                                    .dims = d4};
        double *const norm = tables + (((i * ns) + j) * count);
        FBF_TRY(einsum(s, "qpij,baij,pqab->ij", {h2e_v, h2e_v, hdm2_op}, norm, {nb, nc}, 0.5));
        if (i == j) {
          const EinsumOperand a9_op{.data = a9[static_cast<size_t>(j)].data(), .dims = d4};
          double *const h = tables + (((ns * ns) + j) * count);
          FBF_TRY(einsum(s, "qpij,baij,pqab->ij", {h2e_v, h2e_v, a9_op}, h, {nb, nc}, 0.5));
        }
      }
    }

    std::vector<double> diff;
    for (int64_t i = b0; i < b1; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        diff.push_back(-(e_core[static_cast<size_t>(i)] + e_core[static_cast<size_t>(j)]));
      }
    }
    add_pair_shares(FBF_TRY(download_pair_tables(res, tables, ns, count)), ns, diff, out);
  }
  return out;
}

// Sir: over slabs of h2e_v1_Sir, (nr, a, c, a), and h2e_v2_Sir, (nr, a, a, c),
// and h1e_v_Sir's rows. The output is (c, nr). Its h1e_v·h1e_v norm term is
// ⟨I|J⟩'s, so a transition has none.
Result<std::vector<std::pair<double, double>>>
energy_sir(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
           const int32_t batch, std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const std::vector<int64_t> d4(4, n);
  std::vector<Buffers::device<double>> a12;
  std::vector<Buffers::device<double>> a13;
  for (int64_t j = 0; j < ns; ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a12.push_back(FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3)));
    a13.push_back(FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta)));
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sir1);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_1 = slab_buffer(res, blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, blocks, ExtBlock::Sir2, step);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sir1, b0, b1, slab_1.data()));
    FBF_TRY(slab(res, blocks, ExtBlock::Sir2, b0, b1, slab_2.data()));
    const EinsumOperand h2e_v1 =
        column_major(slab_1.data(), slab_dims(blocks, ExtBlock::Sir1, b0, b1));
    const EinsumOperand h2e_v2 =
        column_major(slab_2.data(), slab_dims(blocks, ExtBlock::Sir2, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_sir + b0, .dims = {nb, nc}, .strides = {1, nv}};

    const int64_t count = nc * nb;
    const std::vector<int64_t> out_dims{nc, nb};
    double *const tables = FBF_TRY(workspace_zeros(res, pair_table_count(ns) * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const ActiveOperands &o = ops[static_cast<size_t>((i * ns) + j)];
        double *const norm = tables + (((i * ns) + j) * count);
        FBF_TRY(einsum(s, "rpiq,raib,qpab->ir", {h2e_v1, h2e_v1, o.dm2}, norm, out_dims, 2.0));
        FBF_TRY(einsum(s, "rpiq,rabi,qpab->ir", {h2e_v1, h2e_v2, o.dm2}, norm, out_dims, -1.0));
        FBF_TRY(einsum(s, "rpqi,raib,qpab->ir", {h2e_v2, h2e_v1, o.dm2}, norm, out_dims, -1.0));
        FBF_TRY(einsum(s, "raqi,rabi,qb->ir", {h2e_v2, h2e_v2, o.dm1}, norm, out_dims, 2.0));
        FBF_TRY(einsum(s, "rpqi,rabi,qbap->ir", {h2e_v2, h2e_v2, o.dm2}, norm, out_dims, -1.0));
        FBF_TRY(einsum(s, "rpqi,raai,qp->ir", {h2e_v2, h2e_v2, o.dm1}, norm, out_dims, 1.0));
        FBF_TRY(einsum(s, "rpiq,ri,qp->ir", {h2e_v1, h1e_v, o.dm1}, norm, out_dims, 4.0));
        FBF_TRY(einsum(s, "rpqi,ri,qp->ir", {h2e_v2, h1e_v, o.dm1}, norm, out_dims, -2.0));
        if (i == j) {
          FBF_TRY(einsum(s, "ri,ri->ir", {h1e_v, h1e_v}, norm, out_dims, 2.0));
          const EinsumOperand a12_op{.data = a12[static_cast<size_t>(j)].data(), .dims = d4};
          const EinsumOperand a13_op{.data = a13[static_cast<size_t>(j)].data(), .dims = d4};
          double *const h = tables + (((ns * ns) + j) * count);
          FBF_TRY(einsum(s, "rpiq,raib,pqab->ir", {h2e_v1, h2e_v1, a12_op}, h, out_dims, 2.0));
          FBF_TRY(einsum(s, "rpiq,rabi,pqab->ir", {h2e_v1, h2e_v2, a12_op}, h, out_dims, -1.0));
          FBF_TRY(einsum(s, "rpqi,raib,pqab->ir", {h2e_v2, h2e_v1, a12_op}, h, out_dims, -1.0));
          FBF_TRY(einsum(s, "rpqi,rabi,pqab->ir", {h2e_v2, h2e_v2, a13_op}, h, out_dims, 1.0));
        }
      }
    }

    std::vector<double> diff;
    for (int64_t i = 0; i < nc; ++i) {
      for (int64_t r = b0; r < b1; ++r) {
        diff.push_back(e_virt[static_cast<size_t>(r)] - e_core[static_cast<size_t>(i)]);
      }
    }
    add_pair_shares(FBF_TRY(download_pair_tables(res, tables, ns, count)), ns, diff, out);
  }
  return out;
}

Result<ActiveOperands> active_operands(DeviceResources &res, const MoBlocks &blocks,
                                       const EnergyInputs &in) {
  const int64_t n = blocks.ncas;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);

  // The demo's deviceEye, made once: every make_* that reads delta reads it.
  std::vector<double> identity(static_cast<size_t>(n * n), 0.0);
  for (int64_t p = 0; p < n; ++p) {
    identity[static_cast<size_t>((p * n) + p)] = 1.0;
  }
  Buffers::device<double> delta_buf(identity.size(), res.device());
  if (!identity.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(delta_buf.data(), identity.data(),
                                           sizeof(double) * identity.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
  }
  const double *const delta = delta_buf.data();

  // The host finish's orbital energies. This download also synchronizes, so
  // the pageable identity upload above has been read before it goes away.
  std::vector<double> e_core = FBF_TRY(download(res, in.e_core, blocks.ncore));
  std::vector<double> e_virt = FBF_TRY(download(res, in.e_virt, blocks.nvirt));
  return ActiveOperands{.h1e = column_major(in.h1e, d2),
                        .h2e = column_major(in.h2e, d4),
                        .dm1 = {.data = in.dm1, .dims = d2},
                        .dm2 = {.data = in.dm2, .dims = d4},
                        .dm3 = {.data = in.dm3, .dims = d6},
                        .f3ca = {.data = in.f3ca, .dims = d6},
                        .f3ac = {.data = in.f3ac, .dims = d6},
                        .delta = {.data = delta, .dims = d2},
                        .delta_buf = std::move(delta_buf),
                        .e_core = std::move(e_core),
                        .e_virt = std::move(e_virt)};
}

Result<std::vector<ClassEnergies>> sc_couplings(DeviceResources &res, const MoBlocks &blocks,
                                                std::span<const EnergyInputs> pairs,
                                                const int64_t n, const int32_t batch) {
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<ActiveOperands> ops;
  for (const EnergyInputs &pair : pairs) {
    ops.push_back(FBF_TRY(active_operands(res, blocks, pair)));
  }
  // The fields every pair shares (h1e_v_*, the orbital energies) are read
  // from the first.
  const EnergyInputs &in = pairs.front();

  std::array<std::vector<std::pair<double, double>>, 8> classes;
  classes[0] = FBF_TRY(energy_sr(res, blocks, in, batch, ops, n));
  classes[1] = FBF_TRY(energy_si(res, blocks, in, batch, ops, n));
  // Sijrs reads no RDM: every state's is the one number, and every
  // transition's is ⟨I|J⟩ = 0 times it.
  const std::pair<double, double> sijrs = FBF_TRY(energy_sijrs(res, blocks, in, batch));
  classes[2].assign(static_cast<size_t>(n * n), std::pair{0.0, 0.0});
  for (int64_t j = 0; j < n; ++j) {
    classes[2][static_cast<size_t>((j * n) + j)] = sijrs;
  }
  classes[3] = FBF_TRY(energy_sijr(res, blocks, batch, ops, n));
  classes[4] = FBF_TRY(energy_srsi(res, blocks, batch, ops, n));
  classes[5] = FBF_TRY(energy_srs(res, blocks, batch, ops, n));
  classes[6] = FBF_TRY(energy_sij(res, blocks, batch, ops, n));
  classes[7] = FBF_TRY(energy_sir(res, blocks, in, batch, ops, n));

  std::vector<ClassEnergies> out(static_cast<size_t>(n * n));
  for (size_t p = 0; p < out.size(); ++p) {
    for (size_t k = 0; k < 8; ++k) {
      std::tie(out[p].norms[k], out[p].energies[k]) = classes[k][p];
    }
    out[p].total = std::ranges::fold_left(out[p].energies, 0.0, std::plus{});
  }
  return out;
}

Result<ClassEnergies> class_energies(DeviceResources &res, const MoBlocks &blocks,
                                     const EnergyInputs &in, const int32_t batch) {
  std::vector<ClassEnergies> out = FBF_TRY(sc_couplings(res, blocks, {&in, 1}, 1, batch));
  return out.front();
}

} // namespace free_breakfast::nevpt2
