/// @file
/// @brief The eight PC-NEVPT2 class energies: the NEVPT2 demo's
/// energy_pc_solve.cppm, energy_pc_classes.cppm and energy_pc.cpp
/// (nevpt2-gpu-demo, commit 7c5a674), over SC's metrics and Dyall
/// Hamiltonians (:shared), and the host finish of its energy_finish.cppm.
/// Each class walks n states at once, for pc_effective_hamiltonian's
/// couplings; pc_class_energies is n = 1.

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas;        // gemm, gemmwich
import free_breakfast.recipes.diagonalize; // diagonalize
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import :shared;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// Every class is SC's two quadratic forms of a slab row x_t, N_t = x_tᵀ S x_t
// and H_t = x_tᵀ K x_t, kept whole: E_t = -b_tᵀ (K + Δ_t S)⁺ b_t with
// b_t = S x_t + S₁ h1_t. S and K are the same for every tuple, so each class
// is solved once, in the eigenbasis U of K over S's kept range:
// K U = S U diag(λ), Uᵀ S U = 1. Then E_t = -Σ_k y_tk² / (λ_k + Δ_t) with
// y_t = Tᵀ x_t + t1ᵀ h1_t, T = S U and t1 = S₁ᵀ U, one slab einsum against T.
//
// S and K are row-major d x d, d over the class's active basis in PySCF's
// order, as every intermediate here is. Symmetrized, a row-major matrix is
// its column-major self, so the solve runs column-major in BLAS and the
// solver, and only T and U, read back by einsum, need their layout spelled.

// S's cut, relative to its largest eigenvalue: two decades above the
// structural null modes' noise (<= 1.2e-15 on every golden), two below the
// first real mode on cc-pVTZ (~2e-11). The demo's PC_TAU.
constexpr double kPcTau = 1e-13;

// The smallest acceptable smallest-kept / largest-dropped ratio: below it the
// rank, and so the energy, depends on where the cut falls. The demo's
// PC_MIN_GAP.
constexpr double kPcMinGap = 1e3;

// A device copy of @p host. Synchronizes, since @p host is pageable.
Result<Buffers::device<double>> upload(DeviceResources &res, const std::vector<double> &host) {
  Buffers::device<double> out(host.size(), res.device());
  if (!host.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.data(), host.data(), sizeof(double) * host.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
  }
  FBF_TRY(res.sync());
  return out;
}

Result<Buffers::device<double>> symmetrized(DeviceResources &res, const double *a,
                                            const int64_t d) {
  const wwr::wwrStream_t s = res.stream();
  const EinsumOperand op{.data = a, .dims = {d, d}};
  Buffers::device<double> out = FBF_TRY(device_zeros(res, d * d));
  FBF_TRY(einsum(s, "ij->ij", {op}, out.data(), {d, d}, 0.5));
  FBF_TRY(transpose(s, op, {1, 0}, out.data(), 0.5));
  return out;
}

// Symmetrize S and K (K is symmetric only to the CI vector's convergence),
// diagonalize S, cut it at kPcTau s_max and refuse a gap below kPcMinGap;
// then K' = Xᵀ K X over X = V_kept s_kept^-1/2, K' = W λ Wᵀ, U = X W.
// @p s1, n1 columns row-major d x n1, is optional.
Result<ClassSolve> solve_class(DeviceResources &res, const char *name, const double *s_rm,
                               const double *k_rm, const int64_t d, const double *s1,
                               const int64_t n1) {
  const wwr::wwrStream_t s = res.stream();
  const auto di = static_cast<int32_t>(d);
  const Buffers::device<double> s_sym = FBF_TRY(symmetrized(res, s_rm, d));
  const Buffers::device<double> k_sym = FBF_TRY(symmetrized(res, k_rm, d));

  // S = V diag(σ) Vᵀ, V column-major (column k is σ_k's).
  Buffers::device<double> v(static_cast<size_t>(d * d), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), s_sym.data(),
                                         sizeof(double) * static_cast<size_t>(d * d),
                                         wwr::wwrMemcpyDeviceToDevice, s)));
  Buffers::device<double> sigma_d(static_cast<size_t>(d), res.device());
  FBF_TRY(diagonalize(res, di, v.data(), sigma_d.data()));
  const std::vector<double> sigma = FBF_TRY(download(res, sigma_d.data(), d));

  // The cut: ascending, so the dropped modes are a prefix.
  const double s_max = sigma.back();
  if (!(s_max > 0.0)) {
    return err_numerical("nevpt2::pc_class_energies: {}'s metric has no positive eigenvalue", name);
  }
  int64_t nd = 0;
  double largest_dropped = 0.0;
  while (nd < d && sigma[static_cast<size_t>(nd)] <= kPcTau * s_max) {
    largest_dropped = std::max(largest_dropped, std::abs(sigma[static_cast<size_t>(nd)]) / s_max);
    ++nd;
  }
  const double smallest_kept = sigma[static_cast<size_t>(nd)] / s_max;
  const double floor = nd > 0 ? largest_dropped : kPcTau;
  const double gap = floor > 0.0 ? smallest_kept / floor : std::numeric_limits<double>::infinity();
  if (gap < kPcMinGap) {
    return err_numerical("nevpt2::pc_class_energies: {}'s metric has no gap at its cut: smallest "
                         "kept / largest dropped = {:.1e} < {:.0e} ({} of {} modes dropped)",
                         name, gap, kPcMinGap, nd, d);
  }
  const int64_t m = d - nd;
  const auto mi = static_cast<int32_t>(m);

  // X = V_kept diag(σ_kept^-1/2), d x m column-major: V_kept is V's last m
  // columns, which read row-major are the m x d rows "ki".
  std::vector<double> inv_sqrt(static_cast<size_t>(m));
  for (int64_t k = 0; k < m; ++k) {
    inv_sqrt[static_cast<size_t>(k)] = 1.0 / std::sqrt(sigma[static_cast<size_t>(nd + k)]);
  }
  const Buffers::device<double> inv_sqrt_d = FBF_TRY(upload(res, inv_sqrt));
  Buffers::device<double> x = FBF_TRY(device_zeros(res, d * m));
  FBF_TRY(einsum(s, "k,ki->ki",
                 {EinsumOperand{.data = inv_sqrt_d.data(), .dims = {m}},
                  EinsumOperand{.data = v.data() + (nd * d), .dims = {m, d}}},
                 x.data(), {m, d}, 1.0));

  // K' = Xᵀ K X, m x m, then K' = W diag(λ) Wᵀ in place.
  Buffers::device<double> work(static_cast<size_t>(d * m), res.device());
  Buffers::device<double> w(static_cast<size_t>(m * m), res.device());
  FBF_TRY(gpu_status(gemmwich(res.blas(), mi, di, x.data(), k_sym.data(), work.data(), w.data())));
  Buffers::device<double> lambda_d(static_cast<size_t>(m), res.device());
  FBF_TRY(diagonalize(res, mi, w.data(), lambda_d.data()));

  // U = X W, then T = S U, both d x m column-major.
  Buffers::device<double> u(static_cast<size_t>(d * m), res.device());
  FBF_TRY(gpu_status(
      gemm(res.blas(), di, mi, mi, &kOne<double>, x.data(), w.data(), &kZero<double>, u.data())));
  Buffers::device<double> t(static_cast<size_t>(d * m), res.device());
  FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, s_sym.data(), u.data(),
                          &kZero<double>, t.data())));
  std::optional<Buffers::device<double>> t1;
  if (s1 != nullptr) {
    // U column-major d x m reads row-major as the m x d "ki".
    t1 = FBF_TRY(device_zeros(res, n1 * m));
    FBF_TRY(einsum(s, "ia,ki->ak",
                   {EinsumOperand{.data = s1, .dims = {d, n1}},
                    EinsumOperand{.data = u.data(), .dims = {m, d}}},
                   t1->data(), {n1, m}, 1.0));
  }
  std::vector<double> lambda = FBF_TRY(download(res, lambda_d.data(), m));
  return ClassSolve{.lambda = std::move(lambda),
                    .t = std::move(t),
                    .t1 = std::move(t1),
                    .dropped = nd,
                    .u = std::move(u)};
}

// T as an einsum operand over @p rank active indices, each n, and k: the
// basis index row-major, I = (p n + q) n + r at rank 3, and k at stride d.
EinsumOperand t_operand(const double *t, const int64_t n, const int64_t rank, const int64_t m,
                        const int64_t d) {
  std::vector<int64_t> dims(static_cast<size_t>(rank), n);
  std::vector<int64_t> strides(static_cast<size_t>(rank), 1);
  for (int64_t k = rank - 1; k-- > 0;) {
    strides[static_cast<size_t>(k)] = strides[static_cast<size_t>(k + 1)] * n;
  }
  dims.push_back(m);
  strides.push_back(d);
  return {.data = t, .dims = std::move(dims), .strides = std::move(strides)};
}

// ---------------------------------------------------------------------------
// The classes walk ns states at once (pc_class_energies is ns = 1), @p ops
// ns x ns row-major, bra row, ket column, as SC's. Per ordered pair (I, J),
// with S_J, K_J and S₁_J state J's own and U_J its solve,
//   ⟨I|V|Ψ_J⁽¹⁾⟩_t = -c Σ_k y_{I←J,tk} y_{J,tk} / (λ_{J,k} + Δ_t),
//   y_{I←J,t} = U_Jᵀ b_{I←J,t},  b_{I←J,t} = S^{(J,I)} x_t + S₁^{(J,I)} h1_t,
// S^{(J,I)} and S₁^{(J,I)} the class's S and S₁ over pair (J, I)'s RDMs (bra
// J, ket I), unsymmetrized, their terms with no RDM dropped (⟨J|I⟩ = 0). So
// y_{I←J,t} = T_{I←J}ᵀ x_t + t1_{I←J}ᵀ h1_t with T_{I←J} = S^{(J,I)ᵀ} U_J and
// t1_{I←J} = S₁^{(J,I)ᵀ} U_J, and at I = J the solve's own T and t1: each
// slab is one einsum per pair, and at ns = 1 every einsum is the one-state
// walk's. test/gtest/data/qd_nevpt2/README.md has what the brute force
// settled.
// ---------------------------------------------------------------------------

// A failure of state @p j's class, at ns > 1 named by its state, with
// pc_class_energies' own prefix replaced; at ns = 1 unchanged, so
// pc_class_energies and its reverse pass refuse with the same message.
std::unexpected<Error> state_error(const Error &e, const int64_t j, const int64_t ns) {
  if (ns == 1) {
    return std::unexpected(e);
  }
  constexpr std::string_view kOwn = "nevpt2::pc_class_energies: ";
  std::string_view what = e.message;
  if (what.starts_with(kOwn)) {
    what.remove_prefix(kOwn.size());
  }
  return std::unexpected(Error{
      e.kind, std::format("nevpt2::pc_effective_hamiltonian: state {}: {}", j, what), e.origin});
}

// One pair's class matrices, row-major: S, K for a state's own RDMs only, and
// S₁ where the class has one.
struct PcMatrices {
  Buffers::device<double> s;
  std::optional<Buffers::device<double>> k;
  std::optional<Buffers::device<double>> s1;
};

// A class's matrices over one pair's operands; @p own is a state's own RDMs
// (K built, the terms with no RDM kept), else a transition's.
using PcBuild = std::function<Result<PcMatrices>(const ActiveOperands &, bool)>;

// One class's solves and transition rows: per ket J its solve, and per ordered
// pair (I, J), I != J, T_{I←J} and t1_{I←J}, d x m_J column-major and
// n1 x m_J row-major as the solve's own.
struct PcKets {
  int64_t ns = 1;
  std::vector<ClassSolve> solves;
  std::vector<std::optional<Buffers::device<double>>> t, t1;

  [[nodiscard]] int64_t m(const int64_t j) const {
    return std::ssize(solves[static_cast<size_t>(j)].lambda);
  }
  [[nodiscard]] const double *t_of(const int64_t i, const int64_t j) const {
    return i == j ? solves[static_cast<size_t>(j)].t.data()
                  : t[static_cast<size_t>((i * ns) + j)]->data();
  }
  [[nodiscard]] const double *t1_of(const int64_t i, const int64_t j) const {
    return i == j ? solves[static_cast<size_t>(j)].t1->data()
                  : t1[static_cast<size_t>((i * ns) + j)]->data();
  }
  // Where pair (I, J)'s block of @p rows tuples starts in one slab's packed y,
  // at [I ns + J], and the total at [ns²].
  [[nodiscard]] std::vector<int64_t> offsets(const int64_t rows) const {
    std::vector<int64_t> at{0};
    for (int64_t p = 0; p < ns * ns; ++p) {
      at.push_back(at.back() + (rows * m(p % ns)));
    }
    return at;
  }
};

// Every state's solve, from its own matrices, then every transition's T and
// t1 over pair (J, I)'s, each pair's matrices freed before the next is built.
Result<PcKets> pc_kets(DeviceResources &res, const char *name, std::span<const ActiveOperands> ops,
                       const int64_t ns, const int64_t d, const int64_t n1, const PcBuild &build) {
  const auto di = static_cast<int32_t>(d);
  PcKets kets{.ns = ns};
  for (int64_t j = 0; j < ns; ++j) {
    Result<ClassSolve> cs = [&]() -> Result<ClassSolve> {
      const PcMatrices mats = FBF_TRY(build(ops[static_cast<size_t>((j * ns) + j)], true));
      return solve_class(res, name, mats.s.data(), mats.k->data(), d,
                         mats.s1 ? mats.s1->data() : nullptr, n1);
    }();
    if (!cs) {
      return state_error(cs.error(), j, ns);
    }
    kets.solves.push_back(std::move(*cs));
  }
  kets.t.resize(static_cast<size_t>(ns * ns));
  kets.t1.resize(static_cast<size_t>(ns * ns));
  for (int64_t i = 0; i < ns; ++i) {
    for (int64_t j = 0; j < ns; ++j) {
      if (i == j) {
        continue;
      }
      const PcMatrices mats = FBF_TRY(build(ops[static_cast<size_t>((j * ns) + i)], false));
      const ClassSolve &cs = kets.solves[static_cast<size_t>(j)];
      const int64_t m = kets.m(j);
      // S^{(J,I)} row-major is S^{(J,I)ᵀ} column-major.
      Buffers::device<double> t(static_cast<size_t>(d * m), res.device());
      FBF_TRY(gpu_status(gemm(res.blas(), di, static_cast<int32_t>(m), di, &kOne<double>,
                              mats.s.data(), cs.u.data(), &kZero<double>, t.data())));
      kets.t[static_cast<size_t>((i * ns) + j)] = std::move(t);
      if (mats.s1) {
        // U column-major d x m reads row-major as the m x d "ki", as in solve_class.
        Buffers::device<double> t1 = FBF_TRY(device_zeros(res, n1 * m));
        FBF_TRY(einsum(res.stream(), "ia,ki->ak",
                       {EinsumOperand{.data = mats.s1->data(), .dims = {d, n1}},
                        EinsumOperand{.data = cs.u.data(), .dims = {m, d}}},
                       t1.data(), {n1, m}, 1.0));
        kets.t1[static_cast<size_t>((i * ns) + j)] = std::move(t1);
      }
    }
  }
  return kets;
}

// A class's running couplings, [I ns + J], and each ket's smallest
// denominator λ_k + Δ_t.
struct Finish {
  int64_t ns;
  std::vector<double> energy;
  std::vector<double> min_denominator;

  explicit Finish(const int64_t states)
      : ns(states), energy(static_cast<size_t>(states * states), 0.0),
        min_denominator(static_cast<size_t>(states), std::numeric_limits<double>::infinity()) {}

  // E_IJ -= c y_I·y_J / (λ_k + Δ) over one tuple's rows: @p yi pair (I, J)'s,
  // @p yj J's own.
  void add(const int64_t i, const int64_t j, const double *yi, const double *yj,
           const std::vector<double> &lambda, const double delta, const double c) {
    double &e = energy[static_cast<size_t>((i * ns) + j)];
    double &floor = min_denominator[static_cast<size_t>(j)];
    for (size_t k = 0; k < lambda.size(); ++k) {
      const double den = lambda[k] + delta;
      floor = std::min(floor, den);
      e -= c * yi[k] * yj[k] / den;
    }
  }

  // Sijr's and Srsi's "2 direct - exchange", bilinear: @p yjx J's own row of
  // the tuple's exchange partner.
  void add_exchange(const int64_t i, const int64_t j, const double *yi, const double *yj,
                    const double *yjx, const std::vector<double> &lambda, const double delta) {
    double &e = energy[static_cast<size_t>((i * ns) + j)];
    double &floor = min_denominator[static_cast<size_t>(j)];
    for (size_t k = 0; k < lambda.size(); ++k) {
      const double den = lambda[k] + delta;
      floor = std::min(floor, den);
      e -= ((2.0 * yi[k] * yj[k]) - (yi[k] * yjx[k])) / den;
    }
  }

  // One slab's packed rows (@p at, PcKets::offsets), tuple t at Δ = @p delta[t].
  void add_rows(const PcKets &kets, const std::vector<double> &y, const std::vector<int64_t> &at,
                const std::vector<double> &delta, const double c) {
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        const double *const yi = y.data() + at[static_cast<size_t>((i * ns) + j)];
        const double *const yj = y.data() + at[static_cast<size_t>((j * ns) + j)];
        for (size_t t = 0; t < delta.size(); ++t) {
          const auto row = static_cast<int64_t>(t) * m;
          add(i, j, yi + row, yj + row, kets.solves[static_cast<size_t>(j)].lambda, delta[t], c);
        }
      }
    }
  }

  // The class's couplings, refused when a ket's denominator is not positive.
  Result<std::vector<double>> done(const char *name) const {
    for (int64_t j = 0; j < ns; ++j) {
      const Status ok = pc_denominators_positive(name, min_denominator[static_cast<size_t>(j)]);
      if (!ok) {
        return state_error(ok.error(), j, ns);
      }
    }
    return energy;
  }
};

// K + Δ S must be positive on the kept range, and a sign change would divide
// through a pole.
Status pc_denominators_positive(const char *name, const double min_denominator) {
  if (!(min_denominator > 0.0)) {
    return err_numerical("nevpt2::pc_class_energies: {} has a denominator λ + Δ = {:.3e} that "
                         "is not positive",
                         name, min_denominator);
  }
  return {};
}

// The output of one slab's einsums, zeroed, from the workspace.
Result<double *> workspace_out(DeviceResources &res, const int64_t count) {
  double *const out = std::get<0>(res.device_regions<double>(static_cast<size_t>(count)));
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(out, 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  return out;
}

// One class over ns states: its couplings [I ns + J] and each ket's dropped
// count.
struct PcClass {
  std::vector<double> energy;
  std::vector<int64_t> dropped;
};

// A class with no perturber: every coupling zero, nothing dropped.
PcClass pc_empty(const int64_t ns) {
  return {.energy = std::vector<double>(static_cast<size_t>(ns * ns), 0.0),
          .dropped = std::vector<int64_t>(static_cast<size_t>(ns), 0)};
}

Result<PcClass> pc_finished(const Finish &f, const char *name, const PcKets &kets) {
  PcClass out{.energy = FBF_TRY(f.done(name)), .dropped = {}};
  for (const ClassSolve &cs : kets.solves) {
    out.dropped.push_back(cs.dropped);
  }
  return out;
}

// Sr: basis (p, q, r), d = n³, S[pqr, abc] = dm3[r,p,q,b,a,c], K = a16,
// S₁[pqr, a] = dm2[r,p,q,a] against h1e_v_Sr, Δ = ε_r.
Result<PcClass> pc_sr(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                      const int32_t batch, std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sr);
  if (extent == 0) {
    return pc_empty(ns);
  }
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    std::optional<Buffers::device<double>> a16;
    if (own) {
      a16 = FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca));
    }
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "rpqbac->pqrabc", {o.dm3}, sm.data(), std::vector<int64_t>(6, n), 1.0));
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    FBF_TRY(einsum(s, "rpqa->pqra", {o.dm2}, s1.data(), std::vector<int64_t>(4, n), 1.0));
    return PcMatrices{.s = std::move(sm), .k = std::move(a16), .s1 = std::move(s1)};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Sr", ops, ns, d, n, build));
  const std::vector<double> &e_virt = ops.front().e_virt;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sr, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};
    const std::vector<int64_t> at = kets.offsets(nb);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 3, m, d);
        const EinsumOperand t1{.data = kets.t1_of(i, j), .dims = {n, m}};
        FBF_TRY(einsum(s, "ipqr,pqrk->ik", {h2e_v, t}, y, {nb, m}, 1.0));
        FBF_TRY(einsum(s, "ia,ak->ik", {h1e_v, t1}, y, {nb, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    f.add_rows(kets, y, at, std::vector<double>(e_virt.begin() + b0, e_virt.begin() + b1), 1.0);
  }
  return pc_finished(f, "Sr", kets);
}

// Si: basis (p, q, r), d = n³, S[pqr, abc] = dm3_h[r,p,q,b,a,c], K = a22,
// S₁[pqr, a] = dm2_h[r,p,q,a] against h1e_v_Si, Δ = -ε_i.
Result<PcClass> pc_si(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                      const int32_t batch, std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Si);
  if (extent == 0) {
    return pc_empty(ns);
  }
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    std::optional<Buffers::device<double>> a22;
    if (own) {
      a22 = FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca));
    }
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    {
      const Buffers::device<double> dm3_h = FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta));
      const EinsumOperand dm3_h_op{.data = dm3_h.data(), .dims = std::vector<int64_t>(6, n)};
      FBF_TRY(einsum(s, "rpqbac->pqrabc", {dm3_h_op}, sm.data(), std::vector<int64_t>(6, n), 1.0));
    }
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    {
      const Buffers::device<double> dm2_h = FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta));
      const EinsumOperand dm2_h_op{.data = dm2_h.data(), .dims = std::vector<int64_t>(4, n)};
      FBF_TRY(einsum(s, "rpqa->pqra", {dm2_h_op}, s1.data(), std::vector<int64_t>(4, n), 1.0));
    }
    return PcMatrices{.s = std::move(sm), .k = std::move(a22), .s1 = std::move(s1)};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Si", ops, ns, d, n, build));
  const std::vector<double> &e_core = ops.front().e_core;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Si, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};
    const std::vector<int64_t> at = kets.offsets(nb);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 3, m, d);
        const EinsumOperand t1{.data = kets.t1_of(i, j), .dims = {n, m}};
        FBF_TRY(einsum(s, "qpir,pqrk->ik", {h2e_v, t}, y, {nb, m}, 1.0));
        FBF_TRY(einsum(s, "ai,ak->ik", {h1e_v, t1}, y, {nb, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    std::vector<double> delta;
    for (int32_t i = b0; i < b1; ++i) {
      delta.push_back(-e_core[static_cast<size_t>(i)]);
    }
    f.add_rows(kets, y, at, delta, 1.0);
  }
  return pc_finished(f, "Si", kets);
}

// Sijr: basis p, d = n, S = hdm1, K = a3, Δ = ε_r - ε_j - ε_i. The two spin
// couplings are SC's "2 direct - exchange", summed over ordered (j, i).
Result<PcClass> pc_sijr(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                        std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijr);
  if (extent == 0 || nc == 0) {
    return pc_empty(ns);
  }
  // hdm1's 2δ is ⟨J|I⟩'s.
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, o.dm1, o.delta, own ? 1.0 : 0.0));
    std::optional<Buffers::device<double>> a3;
    if (own) {
      const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = {n, n}};
      a3 = FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta));
    }
    return PcMatrices{.s = std::move(hdm1), .k = std::move(a3), .s1 = std::nullopt};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Sijr", ops, ns, n, 0, build));
  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijr, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sijr, b0, b1));
    const std::vector<int64_t> at = kets.offsets(nr * nc * nc);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 1, m, n);
        FBF_TRY(einsum(s, "rpji,pk->rjik", {h2e_v, t}, y, {nr, nc, nc, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    for (int64_t bra = 0; bra < ns; ++bra) {
      for (int64_t ket = 0; ket < ns; ++ket) {
        const int64_t m = kets.m(ket);
        const double *const yi = y.data() + at[static_cast<size_t>((bra * ns) + ket)];
        const double *const yj = y.data() + at[static_cast<size_t>((ket * ns) + ket)];
        const auto row = [&](const int64_t r, const int64_t j, const int64_t i) {
          return (((((r * nc) + j) * nc) + i) * m);
        };
        for (int64_t r = 0; r < nr; ++r) {
          for (int64_t j = 0; j < nc; ++j) {
            for (int64_t i = 0; i < nc; ++i) {
              const double delta = e_virt[static_cast<size_t>(b0 + r)] -
                                   e_core[static_cast<size_t>(j)] - e_core[static_cast<size_t>(i)];
              f.add_exchange(bra, ket, yi + row(r, j, i), yj + row(r, j, i), yj + row(r, i, j),
                             kets.solves[static_cast<size_t>(ket)].lambda, delta);
            }
          }
        }
      }
    }
  }
  return pc_finished(f, "Sijr", kets);
}

// Srsi: basis p, d = n, S = dm1, K = k27, Δ = ε_r + ε_s - ε_i, summed over
// ordered (r, s) as Sijr: each batch has its exchange partner in the SrsiT
// slab, so unlike SC no whole (v, v, c) table is held.
Result<PcClass> pc_srsi(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                        std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srsi);
  if (extent == 0 || nc == 0) {
    return pc_empty(ns);
  }
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    std::optional<Buffers::device<double>> k27;
    if (own) {
      k27 = FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2));
    }
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, n * n));
    FBF_TRY(einsum(s, "ij->ij", {o.dm1}, sm.data(), {n, n}, 1.0));
    return PcMatrices{.s = std::move(sm), .k = std::move(k27), .s1 = std::nullopt};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Srsi", ops, ns, n, 0, build));
  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_a = slab_buffer(res, blocks, ExtBlock::Srsi, step);
  Buffers::device<double> slab_b = slab_buffer(res, blocks, ExtBlock::SrsiT, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    // h[r_batch, s, i, p] and h[s, r_batch, i, p].
    const EinsumOperand h_a =
        column_major(slab_a.data(), slab_dims(blocks, ExtBlock::Srsi, b0, b1));
    const EinsumOperand h_b =
        column_major(slab_b.data(), slab_dims(blocks, ExtBlock::SrsiT, b0, b1));
    // y, then the exchange rows yt, each in PcKets::offsets' packing.
    const std::vector<int64_t> at = kets.offsets(nb * nv * nc);
    double *const y_d = FBF_TRY(workspace_out(res, 2 * at.back()));
    double *const yt_d = y_d + at.back();
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        const auto p = static_cast<size_t>((i * ns) + j);
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 1, m, n);
        FBF_TRY(einsum(s, "rsip,pk->rsik", {h_a, t}, y_d + at[p], {nb, nv, nc, m}, 1.0));
        if (i == j) {
          FBF_TRY(einsum(s, "srip,pk->rsik", {h_b, t}, yt_d + at[p], {nb, nv, nc, m}, 1.0));
        }
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, 2 * at.back()));
    const double *const yt = y.data() + at.back();
    for (int64_t bra = 0; bra < ns; ++bra) {
      for (int64_t ket = 0; ket < ns; ++ket) {
        const int64_t m = kets.m(ket);
        const auto own = static_cast<size_t>((ket * ns) + ket);
        const double *const yi = y.data() + at[static_cast<size_t>((bra * ns) + ket)];
        for (int64_t r = 0; r < nb; ++r) {
          for (int64_t ss = 0; ss < nv; ++ss) {
            for (int64_t i = 0; i < nc; ++i) {
              const double delta = e_virt[static_cast<size_t>(b0 + r)] +
                                   e_virt[static_cast<size_t>(ss)] - e_core[static_cast<size_t>(i)];
              const int64_t row = (((r * nv) + ss) * nc + i) * m;
              f.add_exchange(bra, ket, yi + row, y.data() + at[own] + row, yt + at[own] + row,
                             kets.solves[static_cast<size_t>(ket)].lambda, delta);
            }
          }
        }
      }
    }
  }
  return pc_finished(f, "Srsi", kets);
}

// Srs: basis (p, q), d = n², S[pq, ab] = rm2[p,q,b,a], K = a7,
// Δ = ε_r + ε_s, factor 1/2 on every ordered (r, s), as SC walks it.
Result<PcClass> pc_srs(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                       std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srs);
  if (extent == 0) {
    return pc_empty(ns);
  }
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    std::optional<Buffers::device<double>> rm2;
    std::optional<Buffers::device<double>> a7;
    if (own) {
      auto [own_rm2, own_a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
      rm2 = std::move(own_rm2);
      a7 = std::move(own_a7);
    } else {
      rm2 = FBF_TRY(make_rm2(res, o.dm1, o.dm2, o.delta));
    }
    const EinsumOperand rm2_op{.data = rm2->data(), .dims = std::vector<int64_t>(4, n)};
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "pqba->pqab", {rm2_op}, sm.data(), std::vector<int64_t>(4, n), 1.0));
    return PcMatrices{.s = std::move(sm), .k = std::move(a7), .s1 = std::nullopt};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Srs", ops, ns, d, 0, build));
  const std::vector<double> &e_virt = ops.front().e_virt;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Srs, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Srs, b0, b1));
    const std::vector<int64_t> at = kets.offsets(nb * nv);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 2, m, d);
        FBF_TRY(einsum(s, "rsqp,pqk->rsk", {h2e_v, t}, y, {nb, nv, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    std::vector<double> delta;
    for (int32_t r = b0; r < b1; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        delta.push_back(e_virt[static_cast<size_t>(r)] + e_virt[static_cast<size_t>(ss)]);
      }
    }
    f.add_rows(kets, y, at, delta, 0.5);
  }
  return pc_finished(f, "Srs", kets);
}

// Sij: basis (p, q), d = n², S = hdm2, K = a9, Δ = -(ε_i + ε_j), factor 1/2
// on every ordered (i, j), as Srs.
Result<PcClass> pc_sij(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                       std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sij);
  if (extent == 0) {
    return pc_empty(ns);
  }
  // hdm2's δδ terms are ⟨J|I⟩'s.
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    const std::vector<int64_t> d4(4, n);
    Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta, own ? 1.0 : 0.0));
    std::optional<Buffers::device<double>> a9;
    if (own) {
      const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
      const Buffers::device<double> hdm3 =
          FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
      const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
      a9 = FBF_TRY(make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op));
    }
    return PcMatrices{.s = std::move(hdm2), .k = std::move(a9), .s1 = std::nullopt};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Sij", ops, ns, d, 0, build));
  const std::vector<double> &e_core = ops.front().e_core;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sij, step);
  Finish f(ns);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sij, b0, b1));
    const std::vector<int64_t> at = kets.offsets(nb * nc);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        const EinsumOperand t = t_operand(kets.t_of(i, j), n, 2, m, d);
        FBF_TRY(einsum(s, "qpij,pqk->ijk", {h2e_v, t}, y, {nb, nc, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    std::vector<double> delta;
    for (int32_t i = b0; i < b1; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        delta.push_back(-(e_core[static_cast<size_t>(i)] + e_core[static_cast<size_t>(j)]));
      }
    }
    f.add_rows(kets, y, at, delta, 0.5);
  }
  return pc_finished(f, "Sij", kets);
}

// Sir: basis (p, q) ⊕ (p, q), d = 2 n², the two spin couplings as two blocks,
// held (x, p, q, y, a, b) with x, y the block:
//   S11 = 2 dm2[q,p,a,b], S12 = S21 = -dm2[q,p,a,b],
//   S22 = 2 δ_pa dm1[q,b] - dm2[q,b,a,p] + δ_ab dm1[q,p];
//   K11 = 2 a12, K12 = K21 = -a12, K22 = a13;
//   S₁ = [2 dm1[q,p] ; -dm1[q,p]] against h1e_v_Sir. Δ = ε_r - ε_i.
Result<PcClass> pc_sir(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                       const int32_t batch, std::span<const ActiveOperands> ops, const int64_t ns) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const int64_t d = 2 * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sir1);
  if (extent == 0 || nc == 0) {
    return pc_empty(ns);
  }
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> blocked{2, n, n, 2, n, n};
  const Buffers::device<double> off_diag = FBF_TRY(upload(res, {2.0, -1.0, -1.0, 0.0}));
  const Buffers::device<double> block22 = FBF_TRY(upload(res, {0.0, 0.0, 0.0, 1.0}));
  const Buffers::device<double> one_body = FBF_TRY(upload(res, {2.0, -1.0}));
  const EinsumOperand off_diag_op{.data = off_diag.data(), .dims = {2, 2}};
  const EinsumOperand block22_op{.data = block22.data(), .dims = {2, 2}};
  const EinsumOperand one_body_op{.data = one_body.data(), .dims = {2}};
  const PcBuild build = [&](const ActiveOperands &o, const bool own) -> Result<PcMatrices> {
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    {
      Buffers::device<double> s22 = FBF_TRY(device_zeros(res, n * n * n * n));
      FBF_TRY(einsum(s, "pa,qb->pqab", {o.delta, o.dm1}, s22.data(), d4, 2.0));
      FBF_TRY(einsum(s, "qbap->pqab", {o.dm2}, s22.data(), d4, -1.0));
      FBF_TRY(einsum(s, "ab,qp->pqab", {o.delta, o.dm1}, s22.data(), d4, 1.0));
      const EinsumOperand s22_op{.data = s22.data(), .dims = d4};
      FBF_TRY(einsum(s, "xy,qpab->xpqyab", {off_diag_op, o.dm2}, sm.data(), blocked, 1.0));
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, s22_op}, sm.data(), blocked, 1.0));
    }
    std::optional<Buffers::device<double>> km;
    if (own) {
      km = FBF_TRY(device_zeros(res, d * d));
      const Buffers::device<double> a12 = FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3));
      const Buffers::device<double> a13 =
          FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
      const EinsumOperand a12_op{.data = a12.data(), .dims = d4};
      const EinsumOperand a13_op{.data = a13.data(), .dims = d4};
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {off_diag_op, a12_op}, km->data(), blocked, 1.0));
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, a13_op}, km->data(), blocked, 1.0));
    }
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d));
    FBF_TRY(einsum(s, "x,qp->xpq", {one_body_op, o.dm1}, s1.data(), {2, n, n}, 1.0));
    return PcMatrices{.s = std::move(sm), .k = std::move(km), .s1 = std::move(s1)};
  };
  const PcKets kets = FBF_TRY(pc_kets(res, "Sir", ops, ns, d, 1, build));
  const std::vector<double> &e_core = ops.front().e_core;
  const std::vector<double> &e_virt = ops.front().e_virt;

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_1 = slab_buffer(res, blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, blocks, ExtBlock::Sir2, step);
  Finish f(ns);
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
    const std::vector<int64_t> at = kets.offsets(nb * nc);
    double *const y_d = FBF_TRY(workspace_out(res, at.back()));
    for (int64_t i = 0; i < ns; ++i) {
      for (int64_t j = 0; j < ns; ++j) {
        const int64_t m = kets.m(j);
        double *const y = y_d + at[static_cast<size_t>((i * ns) + j)];
        // T's two blocks, rows [0, n²) and [n², 2 n²).
        const EinsumOperand t_1 = t_operand(kets.t_of(i, j), n, 2, m, d);
        const EinsumOperand t_2 = t_operand(kets.t_of(i, j) + (n * n), n, 2, m, d);
        const EinsumOperand t1{.data = kets.t1_of(i, j), .dims = {m}};
        FBF_TRY(einsum(s, "rpiq,pqk->rik", {h2e_v1, t_1}, y, {nb, nc, m}, 1.0));
        FBF_TRY(einsum(s, "rpqi,pqk->rik", {h2e_v2, t_2}, y, {nb, nc, m}, 1.0));
        FBF_TRY(einsum(s, "ri,k->rik", {h1e_v, t1}, y, {nb, nc, m}, 1.0));
      }
    }
    const std::vector<double> y = FBF_TRY(download(res, y_d, at.back()));
    std::vector<double> delta;
    for (int32_t r = b0; r < b1; ++r) {
      for (int64_t i = 0; i < nc; ++i) {
        delta.push_back(e_virt[static_cast<size_t>(r)] - e_core[static_cast<size_t>(i)]);
      }
    }
    f.add_rows(kets, y, at, delta, 1.0);
  }
  return pc_finished(f, "Sir", kets);
}

Result<std::vector<PcClassEnergies>> pc_couplings(DeviceResources &res, const MoBlocks &blocks,
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

  std::array<PcClass, 8> classes;
  classes[0] = FBF_TRY(pc_sr(res, blocks, in, batch, ops, n));
  classes[1] = FBF_TRY(pc_si(res, blocks, in, batch, ops, n));
  // Sijrs has no active index, so its PC energy is SC's: every state's is the
  // one number, and every transition's is ⟨I|J⟩ = 0 times it.
  classes[2] = pc_empty(n);
  const double sijrs = FBF_TRY(energy_sijrs(res, blocks, in, batch)).second;
  for (int64_t j = 0; j < n; ++j) {
    classes[2].energy[static_cast<size_t>((j * n) + j)] = sijrs;
  }
  classes[3] = FBF_TRY(pc_sijr(res, blocks, batch, ops, n));
  classes[4] = FBF_TRY(pc_srsi(res, blocks, batch, ops, n));
  classes[5] = FBF_TRY(pc_srs(res, blocks, batch, ops, n));
  classes[6] = FBF_TRY(pc_sij(res, blocks, batch, ops, n));
  classes[7] = FBF_TRY(pc_sir(res, blocks, in, batch, ops, n));

  std::vector<PcClassEnergies> out(static_cast<size_t>(n * n));
  for (size_t p = 0; p < out.size(); ++p) {
    for (size_t k = 0; k < 8; ++k) {
      out[p].energies[k] = classes[k].energy[p];
      out[p].dropped[k] = classes[k].dropped[p % static_cast<size_t>(n)];
    }
    out[p].total = std::ranges::fold_left(out[p].energies, 0.0, std::plus{});
  }
  return out;
}

Result<PcClassEnergies> pc_class_energies(DeviceResources &res, const MoBlocks &blocks,
                                          const EnergyInputs &in, const int32_t batch) {
  std::vector<PcClassEnergies> out = FBF_TRY(pc_couplings(res, blocks, {&in, 1}, 1, batch));
  return out.front();
}

} // namespace free_breakfast::nevpt2
