/// @file
/// @brief The QD-NEVPT2 PT2 transition density, SC and PC: the three classes
/// with one external orbital (Sr, Si, Sir), each a contraction of the
/// effective Hamiltonian's amplitudes with the T-weighted transition RDMs.

module;

#include "diagnostics.h"

module free_breakfast.nevpt2.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.energy;

namespace free_breakfast::nevpt2 {

// Every array here is row-major, as einsum writes it, except the slabs and
// h1e_v (column-major, nevpt2.density_fitting's) and the eigenbases U
// (column-major d x m, the solve's), which enter through their strides.
//
// With ρ^M = Σ_N T_MN γ^(MN) the T-weighted transition dm1 and dm2 (bra M),
// D = Σ_M ⟨Ψ_M⁽¹⁾|E_xy|ρ^M⟩ class by class. In a class, E_xy|N⟩ is the
// one-body function χ_y of V's h1e_v term (E_ru, E_ui, E_ri), and Ψ_M⁽¹⁾ a
// combination of the internal functions φ^M_μ (V's slab terms on |M⟩) and,
// for SC, of χ^M as well. So each block needs only ⟨φ^M_μ|χ^N_y⟩ = S₁^(MN),
// the class's PC S₁ over the transition RDMs, and ⟨χ^M_a|χ^N_y⟩ = S₁₁^(MN):
//
//   Sr:  S₁[pqr, a] = dm2[r,p,q,a],                     S₁₁[a, u] = dm1[a,u]
//   Si:  S₁[pqr, a] = dm2_h[r,p,q,a],                   S₁₁[a, u] = -dm1[u,a]
//   Sir: S₁[x, p, q] = [2 dm1[q,p]; -dm1[q,p]],         S₁₁ = 0
//
// dm2_h[a,b,c,d] = 2 dm1[a,b] δ_cd - dm2[a,b,d,c], the energy's make_dm2_h. A
// transition's terms with no RDM (S₁₁'s 2δ for Si, Sir's 2) carry ⟨M|N⟩ = 0,
// and T's diagonal is zero, so none appears. The README has the derivation.

// How far T may be from antisymmetric, relative to its largest element.
constexpr double kPt2AntisymmetryTol = 1e-12;

// The states n of @p pairs, when @p t is n x n and antisymmetric to
// kPt2AntisymmetryTol; InvalidConfig, naming @p who, otherwise. The pairs
// themselves are the effective Hamiltonian's to check.
Result<int64_t> pt2_states(const char *who, std::span<const EnergyInputs> pairs,
                           std::span<const double> t) {
  const auto n = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(pairs.size()))));
  if (pairs.empty() || std::cmp_not_equal(n * n, pairs.size()) || t.size() != pairs.size()) {
    return err_config("{}: {} pair inputs and {} T elements are not n x n for one n >= 1", who,
                      pairs.size(), t.size());
  }
  double largest = 0.0;
  for (const double x : t) {
    if (!std::isfinite(x)) {
      return err_config("{}: T has a non-finite element", who);
    }
    largest = std::max(largest, std::abs(x));
  }
  for (int64_t m = 0; m < n; ++m) {
    for (int64_t k = 0; k <= m; ++k) {
      const double sum = t[static_cast<size_t>((m * n) + k)] + t[static_cast<size_t>((k * n) + m)];
      if (std::abs(sum) > kPt2AntisymmetryTol * largest) {
        return err_config("{}: T is not antisymmetric: T[{}, {}] + T[{}, {}] = {:.3e}", who, m, k,
                          k, m, sum);
      }
    }
  }
  return n;
}

// A zeroed device buffer of @p count doubles.
Result<Buffers::device<double>> pt2_zeros(DeviceResources &res, const int64_t count) {
  Buffers::device<double> out(static_cast<size_t>(count), res.device());
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
        out.data(), 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  }
  return out;
}

// A device copy of @p host. Synchronizes, since @p host is pageable.
Result<Buffers::device<double>> pt2_upload(DeviceResources &res, const std::vector<double> &host) {
  Buffers::device<double> out(host.size(), res.device());
  if (!host.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.data(), host.data(), sizeof(double) * host.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
  }
  FBF_TRY(res.sync());
  return out;
}

// @p count doubles of @p d, read back once the stream has written them.
Result<std::vector<double>> pt2_download(DeviceResources &res, const double *d,
                                         const int64_t count) {
  std::vector<double> host(static_cast<size_t>(count));
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * host.size(),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
  }
  FBF_TRY(res.sync());
  return host;
}

// Each state's T-weighted transition RDMs ρ^M (bra M) and Si's dm2_h over
// them, zero for a state with no T_MN != 0 (not live), which is skipped.
struct WeightedRdms {
  std::vector<bool> live;
  std::vector<Buffers::device<double>> dm1, dm2, dm2_h;
};

// ρ^M = Σ_{N != M} a_MN γ^(MN), a = ½(T - Tᵀ).
Result<WeightedRdms> weighted_rdms(DeviceResources &res, std::span<const EnergyInputs> pairs,
                                   std::span<const double> t, const int64_t ns, const int64_t n) {
  const wwr::wwrStream_t s = res.stream();
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  std::vector<double> identity(static_cast<size_t>(n * n), 0.0);
  for (int64_t p = 0; p < n; ++p) {
    identity[static_cast<size_t>((p * n) + p)] = 1.0;
  }
  const Buffers::device<double> delta = FBF_TRY(pt2_upload(res, identity));
  WeightedRdms w;
  for (int64_t m = 0; m < ns; ++m) {
    w.live.push_back(false);
    w.dm1.push_back(FBF_TRY(pt2_zeros(res, n * n)));
    w.dm2.push_back(FBF_TRY(pt2_zeros(res, n * n * n * n)));
    w.dm2_h.push_back(FBF_TRY(pt2_zeros(res, n * n * n * n)));
    for (int64_t k = 0; k < ns; ++k) {
      const double a =
          0.5 * (t[static_cast<size_t>((m * ns) + k)] - t[static_cast<size_t>((k * ns) + m)]);
      if (k == m || a == 0.0) {
        continue;
      }
      w.live.back() = true;
      const EnergyInputs &pair = pairs[static_cast<size_t>((m * ns) + k)];
      FBF_TRY(einsum(s, "ab->ab", {EinsumOperand{.data = pair.dm1, .dims = d2}},
                     w.dm1.back().data(), d2, a));
      FBF_TRY(einsum(s, "abcd->abcd", {EinsumOperand{.data = pair.dm2, .dims = d4}},
                     w.dm2.back().data(), d4, a));
    }
    const EinsumOperand rho1{.data = w.dm1.back().data(), .dims = d2};
    const EinsumOperand rho2{.data = w.dm2.back().data(), .dims = d4};
    FBF_TRY(einsum(s, "ab,cd->abcd", {rho1, EinsumOperand{.data = delta.data(), .dims = d2}},
                   w.dm2_h.back().data(), d4, 2.0));
    FBF_TRY(transpose(s, rho2, {0, 1, 3, 2}, w.dm2_h.back().data(), -1.0));
  }
  // delta is read by the queued einsums before it goes away.
  FBF_TRY(res.sync());
  return w;
}

// The three blocks, each packed row-major on the device: Sr's (virtual r,
// active u), Si's (core i, active u) and Sir's (virtual r, core i).
struct Pt2Blocks {
  Buffers::device<double> sr, si, sir;
};

Result<Pt2Blocks> pt2_blocks(DeviceResources &res, const MoBlocks &blocks) {
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  return Pt2Blocks{.sr = FBF_TRY(pt2_zeros(res, nv * n)),
                   .si = FBF_TRY(pt2_zeros(res, nc * n)),
                   .sir = FBF_TRY(pt2_zeros(res, nv * nc))};
}

// The blocks in place in the nmo x nmo row-major density, core, active,
// virtual: D[r, u] (a,t), D[u, i] (t,i), D[r, i] (a,i), zero elsewhere.
Result<Buffers::device<double>> assemble(DeviceResources &res, const MoBlocks &blocks,
                                         const Pt2Blocks &b) {
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const int64_t nmo = nc + n + nv;
  const std::vector<double> sr = FBF_TRY(pt2_download(res, b.sr.data(), nv * n));
  const std::vector<double> si = FBF_TRY(pt2_download(res, b.si.data(), nc * n));
  const std::vector<double> sir = FBF_TRY(pt2_download(res, b.sir.data(), nv * nc));
  std::vector<double> d(static_cast<size_t>(nmo * nmo), 0.0);
  const auto at = [&](const int64_t x, const int64_t y) -> double & {
    return d[static_cast<size_t>((x * nmo) + y)];
  };
  for (int64_t r = 0; r < nv; ++r) {
    for (int64_t u = 0; u < n; ++u) {
      at(nc + n + r, nc + u) = sr[static_cast<size_t>((r * n) + u)];
    }
    for (int64_t i = 0; i < nc; ++i) {
      at(nc + n + r, i) = sir[static_cast<size_t>((r * nc) + i)];
    }
  }
  for (int64_t i = 0; i < nc; ++i) {
    for (int64_t u = 0; u < n; ++u) {
      at(nc + u, i) = si[static_cast<size_t>((i * n) + u)];
    }
  }
  return pt2_upload(res, d);
}

// The column-major operand of @p block's slab over [b0, b1).
EinsumOperand slab_operand(const MoBlocks &blocks, const ExtBlock block, const double *data,
                           const int32_t b0, const int32_t b1) {
  std::array<int32_t, 4> shape = block_shape(blocks, block);
  shape[static_cast<size_t>(batch_axis(block))] = b1 - b0;
  std::vector<int64_t> dims{shape[0], shape[1], shape[2], shape[3]};
  std::vector<int64_t> strides(4, 1);
  for (size_t k = 1; k < 4; ++k) {
    strides[k] = strides[k - 1] * dims[k - 1];
  }
  return {.data = data, .dims = std::move(dims), .strides = std::move(strides)};
}

// A buffer for @p block's slabs walked in steps of @p step.
Buffers::device<double> slab_storage(DeviceResources &res, const MoBlocks &blocks,
                                     const ExtBlock block, const int32_t step) {
  const std::array<int32_t, 4> shape = block_shape(blocks, block);
  const int32_t extent = shape[static_cast<size_t>(batch_axis(block))];
  int64_t count = 1;
  for (size_t k = 0; k < 4; ++k) {
    count *= k == static_cast<size_t>(batch_axis(block)) ? std::min(step, extent) : shape[k];
  }
  return Buffers::device<double>(static_cast<size_t>(count), res.device());
}

// SC: per class and perturber t, Ψ_M⁽¹⁾ holds c_t (Σ_μ x_tμ φ_μ + Σ_a h1_ta χ_a),
// x_t the slab row and h1_t h1e_v's, so the block's row t is
// Σ_M c^M_t (x_tᵀ S₁(ρ^M) + h1_tᵀ S₁₁(ρ^M)): one walk of each class's slabs,
// the energy's own subscripts with ρ^M's S₁ in place of the metric.
Status sc_blocks(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                 const int32_t batch, const ScAmplitudes &amps, const WeightedRdms &w,
                 Pt2Blocks &out) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const auto ns = static_cast<int64_t>(w.live.size());
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  // Each state's coefficients, on the device: the energy hands over one set
  // per state for every class that has a perturber.
  const auto coefficients = [&](const std::vector<std::vector<double>> &c)
      -> Result<std::vector<Buffers::device<double>>> {
    std::vector<Buffers::device<double>> dev;
    for (const std::vector<double> &cm : c) {
      dev.push_back(FBF_TRY(pt2_upload(res, cm)));
    }
    return dev;
  };
  const auto step_of = [&](const int32_t extent) {
    return batch > 0 ? batch : std::max(extent, 1);
  };

  // Sr, over r: Σ_pqs x[r,p,q,s] ρ2[s,p,q,u] + Σ_a h1e_v_Sr[r,a] ρ1[a,u].
  if (nv > 0 && n > 0) {
    const std::vector<Buffers::device<double>> c = FBF_TRY(coefficients(amps.sr));
    const int32_t extent = static_cast<int32_t>(nv);
    const int32_t step = step_of(extent);
    Buffers::device<double> buf = slab_storage(res, blocks, ExtBlock::Sr, step);
    for (int32_t b0 = 0; b0 < extent; b0 += step) {
      const int32_t b1 = std::min(extent, b0 + step);
      const int64_t nb = b1 - b0;
      FBF_TRY(slab(res, blocks, ExtBlock::Sr, b0, b1, buf.data()));
      const EinsumOperand h2e_v = slab_operand(blocks, ExtBlock::Sr, buf.data(), b0, b1);
      const EinsumOperand h1e_v{.data = in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};
      for (int64_t m = 0; m < ns; ++m) {
        if (!w.live[static_cast<size_t>(m)]) {
          continue;
        }
        const EinsumOperand cm{.data = c[static_cast<size_t>(m)].data() + b0, .dims = {nb}};
        const EinsumOperand rho1{.data = w.dm1[static_cast<size_t>(m)].data(), .dims = d2};
        const EinsumOperand rho2{.data = w.dm2[static_cast<size_t>(m)].data(), .dims = d4};
        double *const rows = out.sr.data() + (b0 * n);
        FBF_TRY(einsum(s, "i,ipqr,rpqa->ia", {cm, h2e_v, rho2}, rows, {nb, n}, 1.0));
        FBF_TRY(einsum(s, "i,ip,pa->ia", {cm, h1e_v, rho1}, rows, {nb, n}, 1.0));
      }
    }
  }

  // Si, over i: Σ_pqr x[q,p,i,r] dm2_h(ρ)[r,p,q,u] - Σ_a h1e_v_Si[a,i] ρ1[u,a].
  if (nc > 0 && n > 0) {
    const std::vector<Buffers::device<double>> c = FBF_TRY(coefficients(amps.si));
    const int32_t extent = static_cast<int32_t>(nc);
    const int32_t step = step_of(extent);
    Buffers::device<double> buf = slab_storage(res, blocks, ExtBlock::Si, step);
    for (int32_t b0 = 0; b0 < extent; b0 += step) {
      const int32_t b1 = std::min(extent, b0 + step);
      const int64_t nb = b1 - b0;
      FBF_TRY(slab(res, blocks, ExtBlock::Si, b0, b1, buf.data()));
      const EinsumOperand h2e_v = slab_operand(blocks, ExtBlock::Si, buf.data(), b0, b1);
      const EinsumOperand h1e_v{.data = in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};
      for (int64_t m = 0; m < ns; ++m) {
        if (!w.live[static_cast<size_t>(m)]) {
          continue;
        }
        const EinsumOperand cm{.data = c[static_cast<size_t>(m)].data() + b0, .dims = {nb}};
        const EinsumOperand rho1{.data = w.dm1[static_cast<size_t>(m)].data(), .dims = d2};
        const EinsumOperand dm2_h{.data = w.dm2_h[static_cast<size_t>(m)].data(), .dims = d4};
        double *const rows = out.si.data() + (b0 * n);
        FBF_TRY(einsum(s, "i,qpir,rpqa->ia", {cm, h2e_v, dm2_h}, rows, {nb, n}, 1.0));
        FBF_TRY(einsum(s, "i,ai,ua->iu", {cm, h1e_v, rho1}, rows, {nb, n}, -1.0));
      }
    }
  }

  // Sir, over (r, i): Σ_pq (2 x1[r,p,i,q] - x2[r,p,q,i]) ρ1[q,p]; no S₁₁.
  if (nv > 0 && nc > 0 && n > 0) {
    const std::vector<Buffers::device<double>> c = FBF_TRY(coefficients(amps.sir));
    const int32_t extent = static_cast<int32_t>(nv);
    const int32_t step = step_of(extent);
    Buffers::device<double> buf_1 = slab_storage(res, blocks, ExtBlock::Sir1, step);
    Buffers::device<double> buf_2 = slab_storage(res, blocks, ExtBlock::Sir2, step);
    for (int32_t b0 = 0; b0 < extent; b0 += step) {
      const int32_t b1 = std::min(extent, b0 + step);
      const int64_t nb = b1 - b0;
      FBF_TRY(slab(res, blocks, ExtBlock::Sir1, b0, b1, buf_1.data()));
      FBF_TRY(slab(res, blocks, ExtBlock::Sir2, b0, b1, buf_2.data()));
      const EinsumOperand x1 = slab_operand(blocks, ExtBlock::Sir1, buf_1.data(), b0, b1);
      const EinsumOperand x2 = slab_operand(blocks, ExtBlock::Sir2, buf_2.data(), b0, b1);
      for (int64_t m = 0; m < ns; ++m) {
        if (!w.live[static_cast<size_t>(m)]) {
          continue;
        }
        const EinsumOperand cm{.data = c[static_cast<size_t>(m)].data() + (b0 * nc),
                               .dims = {nb, nc}};
        const EinsumOperand rho1{.data = w.dm1[static_cast<size_t>(m)].data(), .dims = d2};
        double *const rows = out.sir.data() + (b0 * nc);
        FBF_TRY(einsum(s, "ri,rpiq,qp->ri", {cm, x1, rho1}, rows, {nb, nc}, 2.0));
        FBF_TRY(einsum(s, "ri,rpqi,qp->ri", {cm, x2, rho1}, rows, {nb, nc}, -1.0));
      }
    }
  }
  // The coefficient buffers die here, after the queued einsums read them.
  return res.sync();
}

// U_M, d x m column-major, as an einsum operand over the row-major basis
// axes @p basis, then k at stride d.
EinsumOperand basis_operand(const double *u, std::vector<int64_t> basis, const int64_t m) {
  std::vector<int64_t> strides(basis.size(), 1);
  for (size_t k = basis.size() - 1; k-- > 0;) {
    strides[k] = strides[k + 1] * basis[k + 1];
  }
  const int64_t d = strides.front() * basis.front();
  basis.push_back(m);
  strides.push_back(d);
  return {.data = u, .dims = std::move(basis), .strides = std::move(strides)};
}

// PC: per class and tuple t, Ψ_M⁽¹⁾ holds -Σ_k g_tk Σ_μ U_M[μ, k] φ_μ, so the
// block's row t is -Σ_k g^M_tk W^M[·, k] with W^M = S₁(ρ^M)ᵀ U_M: no slab.
Status pc_blocks(DeviceResources &res, const MoBlocks &blocks, const PcAmplitudes &amps,
                 const WeightedRdms &w, Pt2Blocks &out) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const auto ns = static_cast<int64_t>(w.live.size());
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d3(3, n);
  const std::vector<int64_t> d4(4, n);
  // S₁'s two coupling blocks' coefficients for Sir, [2, -1].
  const Buffers::device<double> one_body = FBF_TRY(pt2_upload(res, {2.0, -1.0}));

  // State M of @p cls, when it is live and the class has tuples: its rank m
  // and g^M on the device.
  struct Ket {
    int64_t m = 0;
    Buffers::device<double> g;
  };
  const auto ket = [&](const PcAmplitudes::Class &cls, const int64_t m,
                       const int64_t tuples) -> Result<std::optional<Ket>> {
    if (tuples == 0 || !w.live[static_cast<size_t>(m)] || std::cmp_less_equal(cls.g.size(), m)) {
      return std::optional<Ket>{};
    }
    const std::vector<double> &g = cls.g[static_cast<size_t>(m)];
    const int64_t rank = std::ssize(g) / tuples;
    if (rank == 0) {
      return std::optional<Ket>{};
    }
    return std::optional<Ket>{Ket{.m = rank, .g = FBF_TRY(pt2_upload(res, g))}};
  };

  for (int64_t m = 0; m < ns; ++m) {
    const EinsumOperand rho1{.data = w.dm1[static_cast<size_t>(m)].data(), .dims = d2};
    const EinsumOperand rho2{.data = w.dm2[static_cast<size_t>(m)].data(), .dims = d4};
    const EinsumOperand dm2_h{.data = w.dm2_h[static_cast<size_t>(m)].data(), .dims = d4};

    // Sr: W[a, k] = Σ_pqr ρ2[r,p,q,a] U[pqr, k]; block[r, a] -= Σ_k W[a,k] g[r,k].
    if (const std::optional<Ket> k = FBF_TRY(ket(amps.sr, m, nv)); k) {
      const EinsumOperand u = basis_operand(amps.sr.u[static_cast<size_t>(m)].data(), d3, k->m);
      Buffers::device<double> wm = FBF_TRY(pt2_zeros(res, n * k->m));
      FBF_TRY(einsum(s, "rpqa,pqrk->ak", {rho2, u}, wm.data(), {n, k->m}, 1.0));
      FBF_TRY(einsum(s, "ak,rk->ra",
                     {EinsumOperand{.data = wm.data(), .dims = {n, k->m}},
                      EinsumOperand{.data = k->g.data(), .dims = {nv, k->m}}},
                     out.sr.data(), {nv, n}, -1.0));
      FBF_TRY(res.sync());
    }
    // Si: W[a, k] = Σ_pqr dm2_h[r,p,q,a] U[pqr, k]; block[i, a] -= Σ_k W[a,k] g[i,k].
    if (const std::optional<Ket> k = FBF_TRY(ket(amps.si, m, nc)); k) {
      const EinsumOperand u = basis_operand(amps.si.u[static_cast<size_t>(m)].data(), d3, k->m);
      Buffers::device<double> wm = FBF_TRY(pt2_zeros(res, n * k->m));
      FBF_TRY(einsum(s, "rpqa,pqrk->ak", {dm2_h, u}, wm.data(), {n, k->m}, 1.0));
      FBF_TRY(einsum(s, "ak,ik->ia",
                     {EinsumOperand{.data = wm.data(), .dims = {n, k->m}},
                      EinsumOperand{.data = k->g.data(), .dims = {nc, k->m}}},
                     out.si.data(), {nc, n}, -1.0));
      FBF_TRY(res.sync());
    }
    // Sir: W[k] = Σ_xpq [2; -1]_x ρ1[q,p] U[xpq, k]; block[r, i] -= Σ_k W[k] g[ri,k].
    if (const std::optional<Ket> k = FBF_TRY(ket(amps.sir, m, nv * nc)); k) {
      const EinsumOperand u =
          basis_operand(amps.sir.u[static_cast<size_t>(m)].data(), {2, n, n}, k->m);
      Buffers::device<double> wm = FBF_TRY(pt2_zeros(res, k->m));
      FBF_TRY(einsum(s, "x,qp,xpqk->k",
                     {EinsumOperand{.data = one_body.data(), .dims = {2}}, rho1, u}, wm.data(),
                     {k->m}, 1.0));
      FBF_TRY(einsum(s, "k,rik->ri",
                     {EinsumOperand{.data = wm.data(), .dims = {k->m}},
                      EinsumOperand{.data = k->g.data(), .dims = {nv, nc, k->m}}},
                     out.sir.data(), {nv, nc}, -1.0));
      FBF_TRY(res.sync());
    }
  }
  return {};
}

Result<Buffers::device<double>> sc_pt2_transition_density(DeviceResources &res,
                                                          const MoBlocks &blocks,
                                                          std::span<const EnergyInputs> pairs,
                                                          std::span<const double> t,
                                                          const int32_t batch) {
  const int64_t ns = FBF_TRY(pt2_states("nevpt2::sc_pt2_transition_density", pairs, t));
  const ScopedDeviceIndex on{res.dev_idx()};
  ScAmplitudes amps;
  // Only the amplitudes are wanted of it.
  static_cast<void>(FBF_TRY(sc_effective_hamiltonian(res, blocks, pairs, batch, &amps)));
  const WeightedRdms w = FBF_TRY(weighted_rdms(res, pairs, t, ns, blocks.ncas));
  Pt2Blocks b = FBF_TRY(pt2_blocks(res, blocks));
  FBF_TRY(sc_blocks(res, blocks, pairs.front(), batch, amps, w, b));
  return assemble(res, blocks, b);
}

Result<Buffers::device<double>> pc_pt2_transition_density(DeviceResources &res,
                                                          const MoBlocks &blocks,
                                                          std::span<const EnergyInputs> pairs,
                                                          std::span<const double> t,
                                                          const int32_t batch) {
  const int64_t ns = FBF_TRY(pt2_states("nevpt2::pc_pt2_transition_density", pairs, t));
  const ScopedDeviceIndex on{res.dev_idx()};
  PcAmplitudes amps;
  // Only the amplitudes are wanted of it; a refusal refuses the density.
  static_cast<void>(FBF_TRY(pc_effective_hamiltonian(res, blocks, pairs, batch, &amps)));
  const WeightedRdms w = FBF_TRY(weighted_rdms(res, pairs, t, ns, blocks.ncas));
  Pt2Blocks b = FBF_TRY(pt2_blocks(res, blocks));
  FBF_TRY(pc_blocks(res, blocks, amps, w, b));
  return assemble(res, blocks, b);
}

// What weighted_rdms, pt2_blocks and assemble hold for @p ns states beside
// the walk (@p classes, each class's coefficients and slabs or amplitudes,
// one at a time, beside the RDMs and blocks): the density kept.
MemoryFootprint pt2_tail_footprint(const int32_t ncore, const int32_t ncas, const int32_t nvirt,
                                   const int64_t ns, const size_t classes) {
  constexpr size_t kD = sizeof(double);
  const auto n = static_cast<size_t>(ncas);
  const auto nc = static_cast<size_t>(ncore);
  const auto nv = static_cast<size_t>(nvirt);
  const size_t nmo = nc + n + nv;
  const auto states = static_cast<size_t>(ns);
  const size_t rdms = states * kD * ((n * n) + (2 * n * n * n * n));
  const size_t blocks = kD * ((nv * n) + (nc * n) + (nv * nc));
  // The identity beside the RDMs as they are built; then the blocks and a
  // class at a time; then the density.
  const size_t peak = std::max(rdms + (kD * n * n), rdms + blocks + classes);
  const size_t density = kD * nmo * nmo;
  const size_t held = rdms + blocks;
  return {.device = {.persistent = density, .scratch = std::max(peak, held + density) - density}};
}

MemoryFootprint sc_pt2_transition_density_footprint(const int32_t ncore, const int32_t ncas,
                                                    const int32_t nvirt, const int32_t nelecas,
                                                    const int32_t nstates, const int32_t batch) {
  constexpr size_t kD = sizeof(double);
  const auto n = static_cast<size_t>(ncas);
  const auto nc = static_cast<size_t>(ncore);
  const auto nv = static_cast<size_t>(nvirt);
  const auto states = static_cast<size_t>(nstates);
  const auto width = [&](const ExtBlock block) {
    const int32_t extent =
        block_shape(ncore, ncas, nvirt, block)[static_cast<size_t>(batch_axis(block))];
    return std::min(batch > 0 ? batch : std::max(extent, 1), extent);
  };
  const auto slab_bytes = [&](const ExtBlock block) {
    std::array<int32_t, 4> shape = block_shape(ncore, ncas, nvirt, block);
    shape[static_cast<size_t>(batch_axis(block))] = width(block);
    return kD * static_cast<size_t>(
                    std::ranges::fold_left(shape, int64_t{1}, [](const int64_t a, const int32_t b) {
                      return a * std::max(b, 0);
                    }));
  };
  const auto product = [&](const ExtBlock block) {
    return slab_footprint(ncore, ncas, nvirt, block, width(block)).device.scratch;
  };
  // sc_blocks' classes: every state's coefficients, the slabs, a product.
  size_t classes = 0;
  if (nvirt > 0 && ncas > 0) {
    classes =
        std::max(classes, (states * kD * nv) + slab_bytes(ExtBlock::Sr) + product(ExtBlock::Sr));
  }
  if (ncore > 0 && ncas > 0) {
    classes =
        std::max(classes, (states * kD * nc) + slab_bytes(ExtBlock::Si) + product(ExtBlock::Si));
  }
  if (nvirt > 0 && ncore > 0 && ncas > 0) {
    classes = std::max(classes, (states * kD * nv * nc) + slab_bytes(ExtBlock::Sir1) +
                                    slab_bytes(ExtBlock::Sir2) +
                                    std::max(product(ExtBlock::Sir1), product(ExtBlock::Sir2)));
  }
  return then(sc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nelecas, nstates, true,
                                                 nstates * (nstates - 1), batch),
              pt2_tail_footprint(ncore, ncas, nvirt, nstates, classes));
}

MemoryFootprint
pc_pt2_transition_density_footprint(const int32_t ncore, const int32_t ncas, const int32_t nvirt,
                                    const int32_t nstates, const int32_t batch,
                                    const std::span<const size_t> syevd_lwork,
                                    const std::span<const std::array<int64_t, 8>> dropped) {
  constexpr size_t kD = sizeof(double);
  const auto n = static_cast<size_t>(ncas);
  const auto nc = static_cast<size_t>(ncore);
  const auto nv = static_cast<size_t>(nvirt);
  // pc_blocks: one_body beside each state's class's amplitudes g and S₁ᵀ U,
  // one class at a time, at its kept modes.
  const auto m = [&](const int32_t j, const size_t k, const size_t d) {
    const int64_t cut = std::cmp_less(j, dropped.size()) ? dropped[static_cast<size_t>(j)][k] : 0;
    return d - static_cast<size_t>(cut);
  };
  size_t classes = 0;
  const size_t n3 = n * n * n;
  for (int32_t j = 0; j < nstates; ++j) {
    if (nvirt > 0) {
      classes = std::max(classes, kD * ((nv * m(j, 0, n3)) + (n * m(j, 0, n3))));
    }
    if (ncore > 0) {
      classes = std::max(classes, kD * ((nc * m(j, 1, n3)) + (n * m(j, 1, n3))));
    }
    if (nvirt > 0 && ncore > 0) {
      classes = std::max(classes, kD * ((nv * nc * m(j, 7, 2 * n * n)) + m(j, 7, 2 * n * n)));
    }
  }
  const MemoryFootprint kets =
      pc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nstates, true, nstates * (nstates - 1),
                                         batch, syevd_lwork, true, dropped);
  // The kets are held to the end, then freed; the density stays.
  MemoryFootprint out =
      then(kets, pt2_tail_footprint(ncore, ncas, nvirt, nstates, (2 * kD) + classes));
  out.device.scratch += out.device.persistent - kD * (nc + n + nv) * (nc + n + nv);
  out.device.persistent = kD * (nc + n + nv) * (nc + n + nv);
  return out;
}

} // namespace free_breakfast::nevpt2
