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
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.footprint;
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
// With @p eta, a one-function class's (single_function_ratio), every
// perturber is kept at -T_k / (Δ_k + η), and N and H are not read.
std::pair<double, double> norm_to_energy(const std::vector<double> &t,
                                         const std::vector<double> &norm,
                                         const std::vector<double> &h,
                                         const std::vector<double> &diff,
                                         const std::optional<double> eta) {
  double norm_total = 0.0;
  double energy_total = 0.0;
  for (size_t k = 0; k < norm.size(); ++k) {
    norm_total += t[k];
    if (eta) {
      energy_total -= t[k] / (diff[k] + *eta);
    } else if (std::abs(norm[k]) > kNumericalZero) {
      energy_total -= t[k] / (diff[k] + (h[k] / norm[k]));
    }
  }
  return {norm_total, energy_total};
}

double norm_rounding(const size_t k, const double norm) {
  constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
  return k == 0 || k == 1 || k == 7 ? 0.5 * kEpsilon / std::abs(norm) : 4.0 * kEpsilon;
}

bool single_function_class(const ActiveOperands &own, const size_t k) {
  const int64_t n = own.dm1.dims.empty() ? 0 : own.dm1.dims[0];
  return (k == 5 && own.nelecas == 2) || (k == 6 && (2 * n) - own.nelecas == 2);
}

Result<std::array<std::vector<double>, 2>>
single_function_blocks(DeviceResources &res, const ActiveOperands &o, const size_t k) {
  const int64_t n = o.dm1.dims[0];
  const int64_t n4 = n * n * n * n;
  std::vector<double> x;
  std::vector<double> y;
  if (k == 5) {
    const auto [rm2, a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
    x = FBF_TRY(download(res, a7.data(), n4));
    const std::vector<double> m = FBF_TRY(download(res, rm2.data(), n4));
    // Srs's norm reads rm2 `pqba` against H's a7 `pqab`.
    y.resize(m.size());
    for (int64_t pq = 0; pq < n * n; ++pq) {
      for (int64_t a = 0; a < n; ++a) {
        for (int64_t b = 0; b < n; ++b) {
          y[static_cast<size_t>((((pq * n) + a) * n) + b)] =
              m[static_cast<size_t>((((pq * n) + b) * n) + a)];
        }
      }
    }
  } else {
    const std::vector<int64_t> d4(4, n);
    const Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta));
    const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
    const Buffers::device<double> hdm3 =
        FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
    const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
    const Buffers::device<double> a9 = FBF_TRY(make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op));
    x = FBF_TRY(download(res, a9.data(), n4));
    y = FBF_TRY(download(res, hdm2.data(), n4));
  }
  return std::array{std::move(x), std::move(y)};
}

std::optional<double> single_function_ratio(const std::array<std::vector<double>, 2> &xy) {
  const double yy = std::inner_product(xy[1].begin(), xy[1].end(), xy[1].begin(), 0.0);
  if (yy <= kNumericalZero) {
    return std::nullopt;
  }
  return std::inner_product(xy[0].begin(), xy[0].end(), xy[1].begin(), 0.0) / yy;
}

// Each state's one-function η of class @p k over @p ops' diagonals, or none
// when the class is not one-function or the diagonals are omitted (the kets
// are then cached amplitudes).
Result<std::vector<std::optional<double>>> single_function_etas(DeviceResources &res,
                                                                std::span<const ActiveOperands> ops,
                                                                const int64_t ns, const size_t k) {
  std::vector<std::optional<double>> eta;
  if (!pair_given(ops, 0) || !single_function_class(ops.front(), k)) {
    return eta;
  }
  for (int64_t j = 0; j < ns; ++j) {
    eta.push_back(single_function_ratio(
        FBF_TRY(single_function_blocks(res, ops[static_cast<size_t>((j * ns) + j)], k))));
  }
  return eta;
}

bool pair_given(std::span<const ActiveOperands> ops, const int64_t p) {
  return ops[static_cast<size_t>(p)].dm1.data != nullptr;
}

// The class walk of ns states (class_energies is ns = 1) accumulates each
// slab into one table of one value per perturber for every given pair, its
// norm einsum, so at I = J state J's own N^J, and one for every given
// state's H^J. An omitted entry has none, so a call over the n diagonals
// alone, or over one pair beside its transpose, holds that many tables and
// not n² + n.
struct PairTableSlots {
  // norm[I ns + J]: pair (I, J)'s table, -1 when the pair is omitted.
  std::vector<int64_t> norm;
  // h[J]: state J's H table, -1 when its diagonal is omitted.
  std::vector<int64_t> h;
  int64_t count = 0;
};

PairTableSlots pair_table_slots(std::span<const ActiveOperands> ops, const int64_t ns) {
  PairTableSlots slots{.norm = std::vector<int64_t>(static_cast<size_t>(ns * ns), -1),
                       .h = std::vector<int64_t>(static_cast<size_t>(ns), -1)};
  for (int64_t p = 0; p < ns * ns; ++p) {
    if (pair_given(ops, p)) {
      slots.norm[static_cast<size_t>(p)] = slots.count++;
    }
  }
  for (int64_t j = 0; j < ns; ++j) {
    if (pair_given(ops, (j * ns) + j)) {
      slots.h[static_cast<size_t>(j)] = slots.count++;
    }
  }
  return slots;
}

// One slab's tables, read back, @p count values each.
Result<std::vector<std::vector<double>>> download_pair_tables(DeviceResources &res,
                                                              const double *tables,
                                                              const PairTableSlots &slots,
                                                              const int64_t count) {
  const std::vector<double> all = FBF_TRY(download(res, tables, slots.count * count));
  std::vector<std::vector<double>> out;
  for (int64_t t = 0; t < slots.count; ++t) {
    out.emplace_back(all.begin() + (t * count), all.begin() + ((t + 1) * count));
  }
  return out;
}

// The kets of a call whose diagonals are omitted: per state J, the strongly
// contracted amplitudes c_t of one slab's perturbers, in its tables' order,
// read from a previous call's ScAmplitudes at @p at onwards (@p kets[J]).
// Null when the call's own diagonals give them.
struct CachedKets {
  const std::vector<std::vector<double>> *kets = nullptr;
  size_t at = 0;
};

// The whole-class amplitudes @p amplitudes of a call over ns states as the
// cached kets of a call whose diagonals are omitted, InvalidConfig when they
// are not ns states' (sc_effective_hamiltonian's caller handed in another
// call's); or none, when @p ops gives the diagonals.
Result<const std::vector<std::vector<double>> *>
sc_cached_kets(const char *name, std::span<const ActiveOperands> ops, const int64_t ns,
               const std::vector<std::vector<double>> *amplitudes) {
  if (pair_given(ops, 0)) {
    return nullptr;
  }
  if (amplitudes == nullptr || std::cmp_not_equal(amplitudes->size(), ns)) {
    return err_config("nevpt2::sc_effective_hamiltonian: the diagonal is omitted and {}'s "
                      "amplitudes are not {} states'",
                      name, ns);
  }
  return amplitudes;
}

// Adds one slab's share onto @p out[I ns + J] from its tables, each already
// the perturber list @p diff runs over (Sijr and Srsi pack theirs first), for
// every given pair: norm_to_energy over J's N^J and H^J with T = N^J at I = J,
// and otherwise T the mean of the norm einsums of (I, J) and (J, I). PySCF
// folds two Hermitian-conjugate terms of Sr's, Si's and Sir's norms into one
// (its `* 2.0`), which holds only at I = J; the mean is the unfolded sum, and
// leaves the five classes already symmetric in I and J as they are. With J's
// diagonal omitted, its c_t come from @p cached instead: -T / (Δ + H / N) is
// T c_t, the same perturbers dropped (c_t = 0). @p eta, when not empty, is
// each ket's one-function η (norm_to_energy).
void add_pair_shares(const std::vector<std::vector<double>> &tables, const PairTableSlots &slots,
                     const int64_t ns, const std::vector<double> &diff, const CachedKets &cached,
                     std::vector<std::pair<double, double>> &out,
                     const std::vector<std::optional<double>> &eta = {}) {
  for (int64_t j = 0; j < ns; ++j) {
    for (int64_t i = 0; i < ns; ++i) {
      const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
      if (slot < 0) {
        continue;
      }
      std::vector<double> mean;
      if (i != j) {
        const std::vector<double> &ij = tables[static_cast<size_t>(slot)];
        const std::vector<double> &ji =
            tables[static_cast<size_t>(slots.norm[static_cast<size_t>((j * ns) + i)])];
        mean.resize(ij.size());
        for (size_t k = 0; k < ij.size(); ++k) {
          mean[k] = 0.5 * (ij[k] + ji[k]);
        }
      }
      std::pair<double, double> &sums = out[static_cast<size_t>((i * ns) + j)];
      const int64_t own = slots.h[static_cast<size_t>(j)];
      if (own < 0) {
        const std::vector<double> &c = (*cached.kets)[static_cast<size_t>(j)];
        for (size_t k = 0; k < mean.size(); ++k) {
          sums.first += mean[k];
          sums.second += mean[k] * c[cached.at + k];
        }
        continue;
      }
      const std::vector<double> &norm =
          tables[static_cast<size_t>(slots.norm[static_cast<size_t>((j * ns) + j)])];
      const std::vector<double> &h = tables[static_cast<size_t>(own)];
      const auto [n_part, e_part] =
          norm_to_energy(i == j ? norm : mean, norm, h, diff,
                         eta.empty() ? std::nullopt : eta[static_cast<size_t>(j)]);
      sums.first += n_part;
      sums.second += e_part;
    }
  }
}

// The strongly contracted amplitudes of one slab's perturbers, in its tables'
// order, appended to @p out[J] for each state J: c_t = -1 / (Δ_t + H^J_t /
// N^J_t), the coefficient of P_t V|J⟩ in Ψ_J⁽¹⁾, and 0 for a perturber
// norm_to_energy drops (ScAmplitudes); with @p eta, -1 / (Δ_t + η^J) for
// every one. Reads the tables add_pair_shares does; every diagonal is given.
void append_sc_amplitudes(const std::vector<std::vector<double>> &tables,
                          const PairTableSlots &slots, const int64_t ns,
                          const std::vector<double> &diff, std::vector<std::vector<double>> &out,
                          const std::vector<std::optional<double>> &eta = {}) {
  out.resize(static_cast<size_t>(ns));
  for (int64_t j = 0; j < ns; ++j) {
    const std::vector<double> &norm =
        tables[static_cast<size_t>(slots.norm[static_cast<size_t>((j * ns) + j)])];
    const std::vector<double> &h = tables[static_cast<size_t>(slots.h[static_cast<size_t>(j)])];
    const std::optional<double> ratio = eta.empty() ? std::nullopt : eta[static_cast<size_t>(j)];
    for (size_t k = 0; k < norm.size(); ++k) {
      double c = 0.0;
      if (ratio) {
        c = -1.0 / (diff[k] + *ratio);
      } else if (std::abs(norm[k]) > kNumericalZero) {
        c = -1.0 / (diff[k] + (h[k] / norm[k]));
      }
      out[static_cast<size_t>(j)].push_back(c);
    }
  }
}

// The walk's step: @p batch, or the whole extent in one piece.
int32_t batch_step(const int32_t extent, const int32_t batch) {
  return batch > 0 ? batch : std::max(extent, 1);
}

size_t walk_bytes(const int32_t ncore, const int32_t ncas, const int32_t nvirt,
                  const std::span<const ExtBlock> slabs, const int32_t step,
                  const int64_t table_per_unit) {
  const auto extent = [&](const ExtBlock block) {
    return block_shape(ncore, ncas, nvirt, block)[static_cast<size_t>(batch_axis(block))];
  };
  const int32_t nb = std::min(step, extent(slabs.front()));
  size_t bytes = sizeof(double) * static_cast<size_t>(table_per_unit * nb);
  size_t product = 0;
  for (const ExtBlock block : slabs) {
    // slab_buffer's count, which is also slab's product's when it permutes.
    std::array<int32_t, 4> shape = block_shape(ncore, ncas, nvirt, block);
    shape[static_cast<size_t>(batch_axis(block))] = std::min(nb, extent(block));
    bytes += sizeof(double) * static_cast<size_t>(std::ranges::fold_left(
                                  shape, int64_t{1}, [](const int64_t a, const int32_t b) {
                                    return a * std::max(b, 0);
                                  }));
    product = std::max(
        product,
        slab_footprint(ncore, ncas, nvirt, block, std::min(nb, extent(block))).device.scratch);
  }
  return bytes + product;
}

Result<int32_t> walk_step(DeviceResources &res, const MoBlocks &blocks,
                          const std::span<const ExtBlock> slabs, const int32_t batch,
                          const int64_t table_per_unit) {
  const int32_t extent = walk_extent(blocks, slabs.front());
  if (batch >= 0 || extent <= 1) {
    return batch_step(extent, batch);
  }
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t budget = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  const auto bytes = [&](const int32_t step) {
    return walk_bytes(blocks.ncore, blocks.ncas, blocks.nvirt, slabs, step, table_per_unit);
  };
  if (bytes(1) > budget) {
    return err_unsupported("nevpt2: a slab batch of one needs {} bytes of device scratch, and "
                           "device {}'s memory budget is {} ({} less {} of headroom)",
                           bytes(1), res.dev_idx(), budget, memory, kPoolHeadroomBytes);
  }
  // The widest step in [1, extent] that fits: walk_bytes grows with it.
  int32_t lo = 1;
  int32_t hi = extent;
  while (lo < hi) {
    const int32_t mid = lo + ((hi - lo + 1) / 2);
    if (bytes(mid) <= budget) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// A walk's per-slab output, the first @p count doubles of @p buffer, zeroed.
Result<double *> zeroed_prefix(DeviceResources &res, Buffers::device<double> &buffer,
                               const int64_t count) {
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
        buffer.data(), 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  }
  return buffer.data();
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
// device buffer, contracting it against every given state's intermediates
// and every given pair's norm intermediates into the slab's pair tables
// (pair_table_slots; an omitted entry, null dm1, has none),
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
          std::span<const ActiveOperands> ops, const int64_t ns,
          std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  std::vector<Buffers::device<double>> a16;
  std::vector<Buffers::device<double>> a17;
  std::vector<Buffers::device<double>> a19;
  for (int64_t j = 0; j < ns && pair_given(ops, (j * ns) + j); ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a16.push_back(FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca)));
    a17.push_back(FBF_TRY(make_a17(res, o.h1e, o.h2e, o.dm2, o.dm3)));
    a19.push_back(FBF_TRY(make_a19(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }

  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sr);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Sr", ops, ns, amplitudes));
  constexpr std::array kSlabs{ExtBlock::Sr};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count));
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sr, step);
  Buffers::device<double> table_buf(static_cast<size_t>(slots.count * std::min(step, extent)),
                                    res.device());
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};

    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * nb));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const ActiveOperands &o = ops[static_cast<size_t>((i * ns) + j)];
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        double *const norm = tables + (slot * nb);
        if (i == j) {
          const EinsumOperand a16_op{.data = a16[static_cast<size_t>(j)].data(),
                                     .dims = std::vector<int64_t>(6, n)};
          const EinsumOperand a17_op{.data = a17[static_cast<size_t>(j)].data(), .dims = d4};
          const EinsumOperand a19_op{.data = a19[static_cast<size_t>(j)].data(), .dims = d2};
          double *const ener = tables + (slots.h[static_cast<size_t>(j)] * nb);
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
    const std::vector<std::vector<double>> read =
        FBF_TRY(download_pair_tables(res, tables, slots, nb));
    add_pair_shares(read, slots, ns, diff, {.kets = cached, .at = static_cast<size_t>(b0)}, out);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(read, slots, ns, diff, *amplitudes);
    }
  }
  return out;
}

// Si: over a slab of h2e_v_Si, (a, a, ni, a), and h1e_v_Si's columns.
Result<std::vector<std::pair<double, double>>>
energy_si(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in, const int32_t batch,
          std::span<const ActiveOperands> ops, const int64_t ns,
          std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  std::vector<Buffers::device<double>> a22;
  std::vector<Buffers::device<double>> a23;
  std::vector<Buffers::device<double>> a25;
  for (int64_t j = 0; j < ns && pair_given(ops, (j * ns) + j); ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a22.push_back(FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca)));
    a23.push_back(FBF_TRY(make_a23(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3)));
    a25.push_back(FBF_TRY(make_a25(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }
  // Per given pair: dm1_h's 2δ is ⟨I|J⟩'s. An omitted pair holds an empty
  // buffer, so the vectors stay indexed by pair.
  std::vector<Buffers::device<double>> dm3_h;
  std::vector<Buffers::device<double>> dm2_h;
  std::vector<Buffers::device<double>> dm1_h;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    if (!pair_given(ops, p)) {
      for (auto *v : {&dm3_h, &dm2_h, &dm1_h}) {
        v->emplace_back(0, res.device());
      }
      continue;
    }
    const double overlap = p / ns == p % ns ? 1.0 : 0.0;
    dm3_h.push_back(FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta)));
    dm2_h.push_back(FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta)));
    dm1_h.push_back(FBF_TRY(make_hdm1(res, o.dm1, o.delta, overlap)));
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const int32_t extent = walk_extent(blocks, ExtBlock::Si);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Si", ops, ns, amplitudes));
  constexpr std::array kSlabs{ExtBlock::Si};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count));
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Si, step);
  Buffers::device<double> table_buf(static_cast<size_t>(slots.count * std::min(step, extent)),
                                    res.device());
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};

    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * nb));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const auto p = static_cast<size_t>((i * ns) + j);
        if (slots.norm[p] < 0) {
          continue;
        }
        double *const norm = tables + (slots.norm[p] * nb);
        if (i == j) {
          const EinsumOperand a22_op{.data = a22[static_cast<size_t>(j)].data(), .dims = d6};
          const EinsumOperand a23_op{.data = a23[static_cast<size_t>(j)].data(), .dims = d4};
          const EinsumOperand a25_op{.data = a25[static_cast<size_t>(j)].data(), .dims = d2};
          double *const ener = tables + (slots.h[static_cast<size_t>(j)] * nb);
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
    const std::vector<std::vector<double>> read =
        FBF_TRY(download_pair_tables(res, tables, slots, nb));
    add_pair_shares(read, slots, ns, diff, {.kets = cached, .at = static_cast<size_t>(b0)}, out);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(read, slots, ns, diff, *amplitudes);
    }
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
  constexpr std::array kSlabs{ExtBlock::Sijrs};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, 0));
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
            std::span<const ActiveOperands> ops, const int64_t ns,
            std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const std::vector<int64_t> d2{n, n};
  // Per given pair hdm1, whose 2δ is ⟨I|J⟩'s (an omitted one empty); per
  // state a3, from its own.
  std::vector<Buffers::device<double>> hdm1;
  std::vector<Buffers::device<double>> a3;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    const bool own = p / ns == p % ns;
    if (!pair_given(ops, p)) {
      hdm1.emplace_back(size_t{0}, res.device());
      continue;
    }
    hdm1.push_back(FBF_TRY(make_hdm1(res, o.dm1, o.delta, own ? 1.0 : 0.0)));
    if (own) {
      const EinsumOperand hdm1_op{.data = hdm1.back().data(), .dims = d2};
      a3.push_back(FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta)));
    }
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijr);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Sijr", ops, ns, amplitudes));
  constexpr std::array kSlabs{ExtBlock::Sijr};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count * nc * nc));
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijr, step);
  Buffers::device<double> table_buf(
      static_cast<size_t>(slots.count * std::min(step, extent) * nc * nc), res.device());
  size_t at_c = 0; // the cached kets' first perturber of this slab
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sijr, b0, b1));

    const int64_t count = nr * nc * nc;
    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * count));
    const std::vector<int64_t> out_dims{nr, nc, nc};
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        const EinsumOperand hdm1_op{.data = hdm1[static_cast<size_t>((i * ns) + j)].data(),
                                    .dims = d2};
        double *const norm = tables + (slot * count);
        FBF_TRY(einsum(s, "rpji,raji,pa->rji", {h2e_v, h2e_v, hdm1_op}, norm, out_dims, 2.0));
        FBF_TRY(einsum(s, "rpji,raij,pa->rji", {h2e_v, h2e_v, hdm1_op}, norm, out_dims, -1.0));
        if (i == j) {
          const EinsumOperand a3_op{.data = a3[static_cast<size_t>(j)].data(), .dims = d2};
          double *const h = tables + (slots.h[static_cast<size_t>(j)] * count);
          FBF_TRY(einsum(s, "rpji,raji,pa->rji", {h2e_v, h2e_v, a3_op}, h, out_dims, 2.0));
          FBF_TRY(einsum(s, "rpji,raij,pa->rji", {h2e_v, h2e_v, a3_op}, h, out_dims, -1.0));
        }
      }
    }
    std::vector<std::vector<double>> packed =
        FBF_TRY(download_pair_tables(res, tables, slots, count));

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
    add_pair_shares(packed, slots, ns, diff, {.kets = cached, .at = at_c}, out);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(packed, slots, ns, diff, *amplitudes);
    }
    at_c += diff.size();
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
            std::span<const ActiveOperands> ops, const int64_t ns,
            std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  std::vector<Buffers::device<double>> k27;
  for (int64_t j = 0; j < ns && pair_given(ops, (j * ns) + j); ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    k27.push_back(FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2)));
  }

  const int64_t count = nv * nv * nc;
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Srsi", ops, ns, amplitudes));
  Buffers::device<double> tables = FBF_TRY(device_zeros(res, slots.count * count));
  const int32_t extent = walk_extent(blocks, ExtBlock::Srsi);
  constexpr std::array kSlabs{ExtBlock::Srsi, ExtBlock::SrsiT};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, 0));
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
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        double *const norm_rows = tables.data() + (slot * count) + (b0 * nv * nc);
        FBF_TRY(einsum(s, "rsip,rsia,pa->rsi", {h_a, h_a, o.dm1}, norm_rows, rows_dims, 2.0));
        FBF_TRY(einsum(s, "rsip,sria,pa->rsi", {h_a, h_b, o.dm1}, norm_rows, rows_dims, -1.0));
        if (i == j) {
          const EinsumOperand k27_op{.data = k27[static_cast<size_t>(j)].data(), .dims = {n, n}};
          double *const h_rows =
              tables.data() + (slots.h[static_cast<size_t>(j)] * count) + (b0 * nv * nc);
          FBF_TRY(einsum(s, "rsip,rsia,pa->rsi", {h_a, h_a, k27_op}, h_rows, rows_dims, 2.0));
          FBF_TRY(einsum(s, "rsip,sria,pa->rsi", {h_a, h_b, k27_op}, h_rows, rows_dims, -1.0));
        }
      }
    }
  }

  std::vector<std::vector<double>> packed =
      FBF_TRY(download_pair_tables(res, tables.data(), slots, count));
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
  add_pair_shares(packed, slots, ns, diff, {.kets = cached}, out);
  if (amplitudes != nullptr && cached == nullptr) {
    append_sc_amplitudes(packed, slots, ns, diff, *amplitudes);
  }
  return out;
}

// Srs: over a slab of h2e_v_Srs, (nr, v, a, a). The metric is read `pqba`,
// the Dyall block `pqab`.
Result<std::vector<std::pair<double, double>>>
energy_srs(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
           std::span<const ActiveOperands> ops, const int64_t ns,
           std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  if (nv == 0) {
    return out;
  }
  const std::vector<int64_t> d4(4, n);
  // Per given pair rm2 (an omitted one empty); per state a7, built beside its
  // own rm2.
  std::vector<Buffers::device<double>> rm2;
  std::vector<Buffers::device<double>> a7;
  for (int64_t p = 0; p < ns * ns; ++p) {
    const ActiveOperands &o = ops[static_cast<size_t>(p)];
    if (!pair_given(ops, p)) {
      rm2.emplace_back(size_t{0}, res.device());
    } else if (p / ns == p % ns) {
      auto [own_rm2, own_a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
      rm2.push_back(std::move(own_rm2));
      a7.push_back(std::move(own_a7));
    } else {
      rm2.push_back(FBF_TRY(make_rm2(res, o.dm1, o.dm2, o.delta)));
    }
  }

  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srs);
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Srs", ops, ns, amplitudes));
  const std::vector<std::optional<double>> eta = FBF_TRY(single_function_etas(res, ops, ns, 5));
  constexpr std::array kSlabs{ExtBlock::Srs};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count * nv));
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Srs, step);
  Buffers::device<double> table_buf(static_cast<size_t>(slots.count * std::min(step, extent) * nv),
                                    res.device());
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Srs, b0, b1));

    const int64_t count = nb * nv;
    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        const EinsumOperand rm2_op{.data = rm2[static_cast<size_t>((i * ns) + j)].data(),
                                   .dims = d4};
        double *const norm = tables + (slot * count);
        FBF_TRY(einsum(s, "rsqp,rsba,pqba->rs", {h2e_v, h2e_v, rm2_op}, norm, {nb, nv}, 0.5));
        if (i == j) {
          const EinsumOperand a7_op{.data = a7[static_cast<size_t>(j)].data(), .dims = d4};
          double *const h = tables + (slots.h[static_cast<size_t>(j)] * count);
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
    const std::vector<std::vector<double>> read =
        FBF_TRY(download_pair_tables(res, tables, slots, count));
    add_pair_shares(read, slots, ns, diff, {.kets = cached, .at = static_cast<size_t>(b0 * nv)},
                    out, eta);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(read, slots, ns, diff, *amplitudes, eta);
    }
  }
  return out;
}

// Sij: over a slab of h2e_v_Sij, (a, a, ni, c). One term each side, the same
// subscripts: hdm2 for the norm, a9 for H.
Result<std::vector<std::pair<double, double>>>
energy_sij(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
           std::span<const ActiveOperands> ops, const int64_t ns,
           std::vector<std::vector<double>> *amplitudes) {
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
    if (!pair_given(ops, p)) {
      hdm2.emplace_back(size_t{0}, res.device());
      continue;
    }
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
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Sij", ops, ns, amplitudes));
  const std::vector<std::optional<double>> eta = FBF_TRY(single_function_etas(res, ops, ns, 6));
  constexpr std::array kSlabs{ExtBlock::Sij};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count * nc));
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sij, step);
  Buffers::device<double> table_buf(static_cast<size_t>(slots.count * std::min(step, extent) * nc),
                                    res.device());
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sij, b0, b1));

    const int64_t count = nb * nc;
    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        const EinsumOperand hdm2_op{.data = hdm2[static_cast<size_t>((i * ns) + j)].data(),
                                    .dims = d4};
        double *const norm = tables + (slot * count);
        FBF_TRY(einsum(s, "qpij,baij,pqab->ij", {h2e_v, h2e_v, hdm2_op}, norm, {nb, nc}, 0.5));
        if (i == j) {
          const EinsumOperand a9_op{.data = a9[static_cast<size_t>(j)].data(), .dims = d4};
          double *const h = tables + (slots.h[static_cast<size_t>(j)] * count);
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
    const std::vector<std::vector<double>> read =
        FBF_TRY(download_pair_tables(res, tables, slots, count));
    add_pair_shares(read, slots, ns, diff, {.kets = cached, .at = static_cast<size_t>(b0 * nc)},
                    out, eta);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(read, slots, ns, diff, *amplitudes, eta);
    }
  }
  return out;
}

// Sir: over slabs of h2e_v1_Sir, (nr, a, c, a), and h2e_v2_Sir, (nr, a, a, c),
// and h1e_v_Sir's rows. The output is (c, nr). Its h1e_v·h1e_v norm term is
// ⟨I|J⟩'s, so a transition has none.
Result<std::vector<std::pair<double, double>>>
energy_sir(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
           const int32_t batch, std::span<const ActiveOperands> ops, const int64_t ns,
           std::vector<std::vector<double>> *amplitudes) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const std::vector<int64_t> d4(4, n);
  std::vector<Buffers::device<double>> a12;
  std::vector<Buffers::device<double>> a13;
  for (int64_t j = 0; j < ns && pair_given(ops, (j * ns) + j); ++j) {
    const ActiveOperands &o = ops[static_cast<size_t>((j * ns) + j)];
    a12.push_back(FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3)));
    a13.push_back(FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta)));
  }

  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sir1);
  std::vector<std::pair<double, double>> out(static_cast<size_t>(ns * ns));
  const PairTableSlots slots = pair_table_slots(ops, ns);
  const auto *const cached = FBF_TRY(sc_cached_kets("Sir", ops, ns, amplitudes));
  constexpr std::array kSlabs{ExtBlock::Sir1, ExtBlock::Sir2};
  const int32_t step = FBF_TRY(walk_step(res, blocks, kSlabs, batch, slots.count * nc));
  Buffers::device<double> slab_1 = slab_buffer(res, blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, blocks, ExtBlock::Sir2, step);
  Buffers::device<double> table_buf(static_cast<size_t>(slots.count * std::min(step, extent) * nc),
                                    res.device());
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
    double *const tables = FBF_TRY(zeroed_prefix(res, table_buf, slots.count * count));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const ActiveOperands &o = ops[static_cast<size_t>((i * ns) + j)];
        const int64_t slot = slots.norm[static_cast<size_t>((i * ns) + j)];
        if (slot < 0) {
          continue;
        }
        double *const norm = tables + (slot * count);
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
          double *const h = tables + (slots.h[static_cast<size_t>(j)] * count);
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
    const std::vector<std::vector<double>> read =
        FBF_TRY(download_pair_tables(res, tables, slots, count));
    // The tables run (i, r); ScAmplitudes' Sir is [r ncore + i] over every r,
    // so cached kets are gathered into the tables' order.
    std::vector<std::vector<double>> slab_c;
    if (cached != nullptr) {
      slab_c.assign(static_cast<size_t>(ns), std::vector<double>(static_cast<size_t>(count)));
      for (int64_t j = 0; j < ns; ++j) {
        for (int64_t i = 0; i < nc; ++i) {
          for (int64_t r = 0; r < nb; ++r) {
            slab_c[static_cast<size_t>(j)][static_cast<size_t>((i * nb) + r)] =
                (*cached)[static_cast<size_t>(j)][static_cast<size_t>(((b0 + r) * nc) + i)];
          }
        }
      }
    }
    add_pair_shares(read, slots, ns, diff, {.kets = cached != nullptr ? &slab_c : nullptr}, out);
    if (amplitudes != nullptr && cached == nullptr) {
      append_sc_amplitudes(read, slots, ns, diff, slab_c);
      amplitudes->resize(static_cast<size_t>(ns),
                         std::vector<double>(static_cast<size_t>(nv * nc), 0.0));
      for (int64_t j = 0; j < ns; ++j) {
        for (int64_t i = 0; i < nc; ++i) {
          for (int64_t r = 0; r < nb; ++r) {
            (*amplitudes)[static_cast<size_t>(j)][static_cast<size_t>(((b0 + r) * nc) + i)] =
                slab_c[static_cast<size_t>(j)][static_cast<size_t>((i * nb) + r)];
          }
        }
      }
    }
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
  int64_t nelecas = 0;
  if (in.dm1 != nullptr) {
    const std::vector<double> dm1 = FBF_TRY(download(res, in.dm1, n * n));
    double trace = 0.0;
    for (int64_t p = 0; p < n; ++p) {
      trace += dm1[static_cast<size_t>((p * n) + p)];
    }
    nelecas = static_cast<int64_t>(std::llround(trace));
  }
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
                        .e_virt = std::move(e_virt),
                        .pc_floor = in.pc_floor,
                        .nelecas = nelecas};
}

Result<std::vector<ClassEnergies>> sc_couplings(DeviceResources &res, const MoBlocks &blocks,
                                                std::span<const EnergyInputs> pairs,
                                                const int64_t n, const int32_t batch,
                                                ScAmplitudes *amplitudes) {
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<ActiveOperands> ops;
  for (const EnergyInputs &pair : pairs) {
    ops.push_back(FBF_TRY(active_operands(res, blocks, pair)));
  }
  // The fields every pair shares (h1e_v_*, the orbital energies) are read
  // from the first.
  const EnergyInputs &in = pairs.front();

  // With the diagonal omitted every class reads its kets from @p amplitudes,
  // which a previous call filled, and Sijrs, diagonal alone, is not walked.
  const bool own = pairs.front().dm1 != nullptr;
  std::array<std::vector<std::pair<double, double>>, 8> classes;
  if (amplitudes != nullptr && own) {
    *amplitudes = ScAmplitudes{};
  }
  classes[0] =
      FBF_TRY(energy_sr(res, blocks, in, batch, ops, n, amplitudes ? &amplitudes->sr : nullptr));
  classes[1] =
      FBF_TRY(energy_si(res, blocks, in, batch, ops, n, amplitudes ? &amplitudes->si : nullptr));
  // Sijrs reads no RDM: every state's is the one number, and every
  // transition's is ⟨I|J⟩ = 0 times it.
  classes[2].assign(static_cast<size_t>(n * n), std::pair{0.0, 0.0});
  if (own) {
    const std::pair<double, double> sijrs = FBF_TRY(energy_sijrs(res, blocks, in, batch));
    for (int64_t j = 0; j < n; ++j) {
      classes[2][static_cast<size_t>((j * n) + j)] = sijrs;
    }
  }
  classes[3] =
      FBF_TRY(energy_sijr(res, blocks, batch, ops, n, amplitudes ? &amplitudes->sijr : nullptr));
  classes[4] =
      FBF_TRY(energy_srsi(res, blocks, batch, ops, n, amplitudes ? &amplitudes->srsi : nullptr));
  classes[5] =
      FBF_TRY(energy_srs(res, blocks, batch, ops, n, amplitudes ? &amplitudes->srs : nullptr));
  classes[6] =
      FBF_TRY(energy_sij(res, blocks, batch, ops, n, amplitudes ? &amplitudes->sij : nullptr));
  classes[7] =
      FBF_TRY(energy_sir(res, blocks, in, batch, ops, n, amplitudes ? &amplitudes->sir : nullptr));

  if (amplitudes != nullptr && own) {
    // A class with no perturber (no slab walked) still holds every state's
    // empty list, so a call reading these kets finds n states'.
    for (auto *c : {&amplitudes->sr, &amplitudes->si, &amplitudes->sijr, &amplitudes->srsi,
                    &amplitudes->srs, &amplitudes->sij, &amplitudes->sir}) {
      c->resize(static_cast<size_t>(n));
    }
  }
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

MemoryFootprint class_energies_footprint(const int32_t ncore, const int32_t ncas,
                                         const int32_t nvirt, const int32_t nelecas,
                                         const int32_t batch) {
  return sc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nelecas, 1, true, 0, batch);
}

MemoryFootprint sc_effective_hamiltonian_footprint(const int32_t ncore, const int32_t ncas,
                                                   const int32_t nvirt, const int32_t nelecas,
                                                   const int32_t nstates, const bool diagonal,
                                                   const int32_t transitions, const int32_t batch) {
  // sc_couplings, every allocation in its order (each make_* and class
  // above): its intermediates n², n⁴ and n⁶ doubles, per state with its own
  // entry given (own) or per given entry (given), and each walk at its step,
  // the least (one) for a batch from the budget, over its tables, one per
  // given entry and one per own state.
  const auto n = static_cast<size_t>(ncas);
  const size_t n2 = sizeof(double) * n * n;
  const size_t n4 = n2 * n * n;
  const size_t n6 = n4 * n * n;
  const auto step = [&](const ExtBlock block) {
    const int32_t extent =
        block_shape(ncore, ncas, nvirt, block)[static_cast<size_t>(batch_axis(block))];
    return batch < 0 ? 1 : batch_step(extent, batch);
  };
  const auto walk = [&](const std::span<const ExtBlock> slabs, const int64_t per_unit) {
    return walk_bytes(ncore, ncas, nvirt, slabs, step(slabs.front()), per_unit);
  };
  const auto nc = static_cast<int64_t>(ncore);
  const auto nv = static_cast<int64_t>(nvirt);
  const auto states = static_cast<size_t>(nstates);
  const size_t own = diagonal ? states : 0;
  const size_t given = own + static_cast<size_t>(transitions);
  const auto tables = static_cast<int64_t>(given + own);
  HeldBytes h;
  h.take(states * states * n2); // every entry's active_operands delta, held through every class

  // Sr: per own state a16; a17 beside make_a17's h1e copy; a19 beside make_a19's.
  for (size_t j = 0; j < own; ++j) {
    h.take(n6);
    h.take(n2);
    h.take(n4);
    h.give(n2);
    h.take(n2);
    h.take(n2);
    h.give(n2);
  }
  h.walk(walk(std::array{ExtBlock::Sr}, tables));
  h.give(own * (n6 + n4 + n2));

  // Si: per own state a22, a23 and a25; then per given entry dm3_h, dm2_h
  // and dm1_h.
  h.take((own + given) * (n6 + n4 + n2));
  h.walk(walk(std::array{ExtBlock::Si}, tables));
  h.give((own + given) * (n6 + n4 + n2));

  // Sijrs, with the diagonal alone: its two accumulators, then the walk, no
  // tables.
  if (diagonal && ncore > 0 && nvirt > 0) {
    h.take(2 * sizeof(double));
    h.walk(walk(std::array{ExtBlock::Sijrs}, 0));
    h.give(2 * sizeof(double));
  }

  // Sijr: per given entry hdm1, per own state a3.
  h.take((given + own) * n2);
  h.walk(walk(std::array{ExtBlock::Sijr}, tables * nc * nc));
  h.give((given + own) * n2);

  // Srsi: per own state k27, and the whole (v, v, c) tables.
  const size_t srsi_tables = sizeof(double) * static_cast<size_t>(tables * nv * nv * nc);
  h.take((own * n2) + srsi_tables);
  h.walk(walk(std::array{ExtBlock::Srsi, ExtBlock::SrsiT}, 0));
  h.give((own * n2) + srsi_tables);

  // Srs: per own state make_a7's rm2, rm3 and a7, rm3 given back; per other
  // given entry make_rm2's rm2; at two active electrons make_a7 again per own
  // state for η, all three given back; then the walk.
  // The entries in pair order: a state's own where the diagonal is given,
  // and the first @p transitions off the diagonal.
  const auto entry = [&](const size_t p) {
    const bool on = p / states == p % states;
    // Off the diagonal, the entries before p less the diagonal's.
    const size_t off = on ? 0 : p - (p / (states + 1)) - 1;
    return std::pair{on && diagonal, !on && std::cmp_less(off, transitions)};
  };
  if (nvirt > 0) {
    for (size_t p = 0; p < states * states; ++p) {
      const auto [mine, transition] = entry(p);
      if (mine) {
        h.take(n4 + n6 + n4);
        h.give(n6);
      } else if (transition) {
        h.take(n4);
      }
    }
    if (diagonal && nelecas == 2) {
      for (size_t j = 0; j < own; ++j) {
        h.take(n4 + n6 + n4);
        h.give(n4 + n6 + n4);
      }
    }
    h.walk(walk(std::array{ExtBlock::Srs}, tables * nv));
    h.give((given + own) * n4);
  }

  // Sij: per given entry make_hdm2's two (its dm2b given back), and per own
  // state hdm3 and a9, hdm3 given back; at two active holes the three again
  // per own state for η, all given back.
  if (ncore > 0) {
    for (size_t p = 0; p < states * states; ++p) {
      const auto [mine, transition] = entry(p);
      if (mine || transition) {
        h.take(2 * n4);
        h.give(n4);
      }
      if (mine) {
        h.take(n6 + n4);
        h.give(n6);
      }
    }
    if (diagonal && (2 * ncas) - nelecas == 2) {
      for (size_t j = 0; j < own; ++j) {
        h.take(2 * n4);
        h.give(n4);
        h.take(n6 + n4);
        h.give(n4 + n6 + n4);
      }
    }
    h.walk(walk(std::array{ExtBlock::Sij}, tables * nc));
    h.give((given + own) * n4);
  }

  // Sir: per own state a12 and a13.
  h.take(own * 2 * n4);
  h.walk(walk(std::array{ExtBlock::Sir1, ExtBlock::Sir2}, tables * nc));
  h.give(own * 2 * n4);

  // make_a16's fdm2 and make_a22's two, the workspace, with the diagonal.
  const size_t region = math::align_up(n4, DeviceResources::kWorkspaceAlignment);
  return {.device = {.scratch = h.peak, .workspace = diagonal ? 2 * region : 0}};
}

} // namespace free_breakfast::nevpt2
