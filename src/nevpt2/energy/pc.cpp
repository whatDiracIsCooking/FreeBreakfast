/// @file
/// @brief The eight PC-NEVPT2 class energies: the NEVPT2 demo's
/// energy_pc_solve.cppm, energy_pc_classes.cppm and energy_pc.cpp
/// (nevpt2-gpu-demo, commit 7c5a674), over SC's metrics and Dyall
/// Hamiltonians (:shared), and the host finish of its energy_finish.cppm.

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

// A class's running energy and its smallest denominator λ_k + Δ_t.
struct Finish {
  double energy = 0.0;
  double min_denominator = std::numeric_limits<double>::infinity();

  // E -= c y² / (λ_k + Δ) over one tuple's row @p y.
  void add(const double *y, const std::vector<double> &lambda, const double delta, const double c) {
    for (size_t k = 0; k < lambda.size(); ++k) {
      const double den = lambda[k] + delta;
      min_denominator = std::min(min_denominator, den);
      energy -= c * y[k] * y[k] / den;
    }
  }

  // The class's energy, refused when a denominator is not positive.
  Result<double> done(const char *name) const {
    FBF_TRY(pc_denominators_positive(name, min_denominator));
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

// One class's energy and dropped count.
using PcClass = std::pair<double, int64_t>;

// Sr: basis (p, q, r), d = n³, S[pqr, abc] = dm3[r,p,q,b,a,c], K = a16,
// S₁[pqr, a] = dm2[r,p,q,a] against h1e_v_Sr, Δ = ε_r.
Result<PcClass> pc_sr(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                      const int32_t batch, const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sr);
  if (extent == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const Buffers::device<double> a16 =
        FBF_TRY(make_a16(res, op.h1e, op.h2e, op.dm3, op.f3ac, op.f3ca));
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "rpqbac->pqrabc", {op.dm3}, sm.data(), std::vector<int64_t>(6, n), 1.0));
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    FBF_TRY(einsum(s, "rpqa->pqra", {op.dm2}, s1.data(), std::vector<int64_t>(4, n), 1.0));
    return solve_class(res, "Sr", sm.data(), a16.data(), d, s1.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 3, m, d);
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {n, m}};

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sr, step);
  Finish f;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};
    double *const y_d = FBF_TRY(workspace_out(res, nb * m));
    FBF_TRY(einsum(s, "ipqr,pqrk->ik", {h2e_v, t}, y_d, {nb, m}, 1.0));
    FBF_TRY(einsum(s, "ia,ak->ik", {h1e_v, t1}, y_d, {nb, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, nb * m));
    for (int64_t i = 0; i < nb; ++i) {
      f.add(y.data() + (i * m), cs.lambda, op.e_virt[static_cast<size_t>(b0 + i)], 1.0);
    }
  }
  return PcClass{FBF_TRY(f.done("Sr")), cs.dropped};
}

// Si: basis (p, q, r), d = n³, S[pqr, abc] = dm3_h[r,p,q,b,a,c], K = a22,
// S₁[pqr, a] = dm2_h[r,p,q,a] against h1e_v_Si, Δ = -ε_i.
Result<PcClass> pc_si(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                      const int32_t batch, const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Si);
  if (extent == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const Buffers::device<double> a22 =
        FBF_TRY(make_a22(res, op.h1e, op.h2e, op.dm2, op.dm3, op.f3ac, op.f3ca));
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    {
      const Buffers::device<double> dm3_h = FBF_TRY(make_dm3_h(res, op.dm2, op.dm3, op.delta));
      const EinsumOperand dm3_h_op{.data = dm3_h.data(), .dims = std::vector<int64_t>(6, n)};
      FBF_TRY(einsum(s, "rpqbac->pqrabc", {dm3_h_op}, sm.data(), std::vector<int64_t>(6, n), 1.0));
    }
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    {
      const Buffers::device<double> dm2_h = FBF_TRY(make_dm2_h(res, op.dm1, op.dm2, op.delta));
      const EinsumOperand dm2_h_op{.data = dm2_h.data(), .dims = std::vector<int64_t>(4, n)};
      FBF_TRY(einsum(s, "rpqa->pqra", {dm2_h_op}, s1.data(), std::vector<int64_t>(4, n), 1.0));
    }
    return solve_class(res, "Si", sm.data(), a22.data(), d, s1.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 3, m, d);
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {n, m}};

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Si, step);
  Finish f;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};
    double *const y_d = FBF_TRY(workspace_out(res, nb * m));
    FBF_TRY(einsum(s, "qpir,pqrk->ik", {h2e_v, t}, y_d, {nb, m}, 1.0));
    FBF_TRY(einsum(s, "ai,ak->ik", {h1e_v, t1}, y_d, {nb, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, nb * m));
    for (int64_t i = 0; i < nb; ++i) {
      f.add(y.data() + (i * m), cs.lambda, -op.e_core[static_cast<size_t>(b0 + i)], 1.0);
    }
  }
  return PcClass{FBF_TRY(f.done("Si")), cs.dropped};
}

// Sijr: basis p, d = n, S = hdm1, K = a3, Δ = ε_r - ε_j - ε_i. The two spin
// couplings are SC's "2 direct - exchange", summed over ordered (j, i).
Result<PcClass> pc_sijr(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                        const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijr);
  if (extent == 0 || nc == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, op.dm1, op.delta));
    const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = {n, n}};
    const Buffers::device<double> a3 =
        FBF_TRY(make_a3(res, op.h1e, op.h2e, op.dm1, op.dm2, hdm1_op, op.delta));
    return solve_class(res, "Sijr", hdm1.data(), a3.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 1, m, n);

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijr, step);
  Finish f;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sijr, b0, b1));
    const int64_t count = nr * nc * nc * m;
    double *const y_d = FBF_TRY(workspace_out(res, count));
    FBF_TRY(einsum(s, "rpji,pk->rjik", {h2e_v, t}, y_d, {nr, nc, nc, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, count));
    const auto at = [&](const int64_t r, const int64_t j, const int64_t i, const int64_t k) {
      return static_cast<size_t>((((((r * nc) + j) * nc) + i) * m) + k);
    };
    for (int64_t r = 0; r < nr; ++r) {
      for (int64_t j = 0; j < nc; ++j) {
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = op.e_virt[static_cast<size_t>(b0 + r)] -
                               op.e_core[static_cast<size_t>(j)] -
                               op.e_core[static_cast<size_t>(i)];
          for (int64_t k = 0; k < m; ++k) {
            const double den = cs.lambda[static_cast<size_t>(k)] + delta;
            f.min_denominator = std::min(f.min_denominator, den);
            const double yd = y[at(r, j, i, k)];
            const double yx = y[at(r, i, j, k)];
            f.energy -= ((2.0 * yd * yd) - (yd * yx)) / den;
          }
        }
      }
    }
  }
  return PcClass{FBF_TRY(f.done("Sijr")), cs.dropped};
}

// Srsi: basis p, d = n, S = dm1, K = k27, Δ = ε_r + ε_s - ε_i, summed over
// ordered (r, s) as Sijr: each batch has its exchange partner in the SrsiT
// slab, so unlike SC no whole (v, v, c) table is held.
Result<PcClass> pc_srsi(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                        const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srsi);
  if (extent == 0 || nc == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const Buffers::device<double> k27 = FBF_TRY(make_k27(res, op.h1e, op.h2e, op.dm1, op.dm2));
    return solve_class(res, "Srsi", op.dm1.data, k27.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 1, m, n);

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_a = slab_buffer(res, blocks, ExtBlock::Srsi, step);
  Buffers::device<double> slab_b = slab_buffer(res, blocks, ExtBlock::SrsiT, step);
  Finish f;
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
    const int64_t count = nb * nv * nc * m;
    const auto [y_d, yt_d] =
        res.device_regions<double>(static_cast<size_t>(count), static_cast<size_t>(count));
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(y_d, 0, sizeof(double) * static_cast<size_t>(count), s)));
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(yt_d, 0, sizeof(double) * static_cast<size_t>(count), s)));
    FBF_TRY(einsum(s, "rsip,pk->rsik", {h_a, t}, y_d, {nb, nv, nc, m}, 1.0));
    FBF_TRY(einsum(s, "srip,pk->rsik", {h_b, t}, yt_d, {nb, nv, nc, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, count));
    const std::vector<double> yt = FBF_TRY(download(res, yt_d, count));
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        for (int64_t i = 0; i < nc; ++i) {
          const double delta = op.e_virt[static_cast<size_t>(b0 + r)] +
                               op.e_virt[static_cast<size_t>(ss)] -
                               op.e_core[static_cast<size_t>(i)];
          const auto row = static_cast<size_t>((((r * nv) + ss) * nc + i) * m);
          for (int64_t k = 0; k < m; ++k) {
            const double den = cs.lambda[static_cast<size_t>(k)] + delta;
            f.min_denominator = std::min(f.min_denominator, den);
            const double yd = y[row + static_cast<size_t>(k)];
            const double yx = yt[row + static_cast<size_t>(k)];
            f.energy -= ((2.0 * yd * yd) - (yd * yx)) / den;
          }
        }
      }
    }
  }
  return PcClass{FBF_TRY(f.done("Srsi")), cs.dropped};
}

// Srs: basis (p, q), d = n², S[pq, ab] = rm2[p,q,b,a], K = a7,
// Δ = ε_r + ε_s, factor 1/2 on every ordered (r, s), as SC walks it.
Result<PcClass> pc_srs(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                       const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nv = blocks.nvirt;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Srs);
  if (extent == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const auto [rm2, a7] = FBF_TRY(make_a7(res, op.h1e, op.h2e, op.dm1, op.dm2, op.dm3, op.delta));
    const EinsumOperand rm2_op{.data = rm2.data(), .dims = std::vector<int64_t>(4, n)};
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "pqba->pqab", {rm2_op}, sm.data(), std::vector<int64_t>(4, n), 1.0));
    return solve_class(res, "Srs", sm.data(), a7.data(), d);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 2, m, d);

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Srs, step);
  Finish f;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Srs, b0, b1));
    const int64_t count = nb * nv * m;
    double *const y_d = FBF_TRY(workspace_out(res, count));
    FBF_TRY(einsum(s, "rsqp,pqk->rsk", {h2e_v, t}, y_d, {nb, nv, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, count));
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        f.add(y.data() + (((r * nv) + ss) * m), cs.lambda,
              op.e_virt[static_cast<size_t>(b0 + r)] + op.e_virt[static_cast<size_t>(ss)], 0.5);
      }
    }
  }
  return PcClass{FBF_TRY(f.done("Srs")), cs.dropped};
}

// Sij: basis (p, q), d = n², S = hdm2, K = a9, Δ = -(ε_i + ε_j), factor 1/2
// on every ordered (i, j), as Srs.
Result<PcClass> pc_sij(DeviceResources &res, const MoBlocks &blocks, const int32_t batch,
                       const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sij);
  if (extent == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const std::vector<int64_t> d4(4, n);
    const Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, op.dm1, op.dm2, op.delta));
    const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
    const Buffers::device<double> hdm3 =
        FBF_TRY(make_hdm3(res, op.dm1, op.dm2, op.dm3, hdm2_op, op.delta));
    const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
    const Buffers::device<double> a9 = FBF_TRY(make_a9(res, op.h1e, op.h2e, hdm2_op, hdm3_op));
    return solve_class(res, "Sij", hdm2.data(), a9.data(), d);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  const EinsumOperand t = t_operand(cs.t.data(), n, 2, m, d);

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sij, step);
  Finish f;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(blocks, ExtBlock::Sij, b0, b1));
    const int64_t count = nb * nc * m;
    double *const y_d = FBF_TRY(workspace_out(res, count));
    FBF_TRY(einsum(s, "qpij,pqk->ijk", {h2e_v, t}, y_d, {nb, nc, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, count));
    for (int64_t i = 0; i < nb; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        f.add(y.data() + (((i * nc) + j) * m), cs.lambda,
              -(op.e_core[static_cast<size_t>(b0 + i)] + op.e_core[static_cast<size_t>(j)]), 0.5);
      }
    }
  }
  return PcClass{FBF_TRY(f.done("Sij")), cs.dropped};
}

// Sir: basis (p, q) ⊕ (p, q), d = 2 n², the two spin couplings as two blocks,
// held (x, p, q, y, a, b) with x, y the block:
//   S11 = 2 dm2[q,p,a,b], S12 = S21 = -dm2[q,p,a,b],
//   S22 = 2 δ_pa dm1[q,b] - dm2[q,b,a,p] + δ_ab dm1[q,p];
//   K11 = 2 a12, K12 = K21 = -a12, K22 = a13;
//   S₁ = [2 dm1[q,p] ; -dm1[q,p]] against h1e_v_Sir. Δ = ε_r - ε_i.
Result<PcClass> pc_sir(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                       const int32_t batch, const ActiveOperands &op) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  const int64_t d = 2 * n * n;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sir1);
  if (extent == 0 || nc == 0) {
    return PcClass{0.0, 0};
  }
  // S, K and what builds them are freed before the slab walk.
  const auto solve = [&]() -> Result<ClassSolve> {
    const std::vector<int64_t> d4(4, n);
    const std::vector<int64_t> blocked{2, n, n, 2, n, n};
    const Buffers::device<double> off_diag = FBF_TRY(upload(res, {2.0, -1.0, -1.0, 0.0}));
    const Buffers::device<double> block22 = FBF_TRY(upload(res, {0.0, 0.0, 0.0, 1.0}));
    const Buffers::device<double> one_body = FBF_TRY(upload(res, {2.0, -1.0}));
    const EinsumOperand off_diag_op{.data = off_diag.data(), .dims = {2, 2}};
    const EinsumOperand block22_op{.data = block22.data(), .dims = {2, 2}};
    const EinsumOperand one_body_op{.data = one_body.data(), .dims = {2}};

    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    {
      Buffers::device<double> s22 = FBF_TRY(device_zeros(res, n * n * n * n));
      FBF_TRY(einsum(s, "pa,qb->pqab", {op.delta, op.dm1}, s22.data(), d4, 2.0));
      FBF_TRY(einsum(s, "qbap->pqab", {op.dm2}, s22.data(), d4, -1.0));
      FBF_TRY(einsum(s, "ab,qp->pqab", {op.delta, op.dm1}, s22.data(), d4, 1.0));
      const EinsumOperand s22_op{.data = s22.data(), .dims = d4};
      FBF_TRY(einsum(s, "xy,qpab->xpqyab", {off_diag_op, op.dm2}, sm.data(), blocked, 1.0));
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, s22_op}, sm.data(), blocked, 1.0));
    }
    Buffers::device<double> km = FBF_TRY(device_zeros(res, d * d));
    {
      const Buffers::device<double> a12 = FBF_TRY(make_a12(res, op.h1e, op.h2e, op.dm2, op.dm3));
      const Buffers::device<double> a13 =
          FBF_TRY(make_a13(res, op.h1e, op.h2e, op.dm1, op.dm2, op.dm3, op.delta));
      const EinsumOperand a12_op{.data = a12.data(), .dims = d4};
      const EinsumOperand a13_op{.data = a13.data(), .dims = d4};
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {off_diag_op, a12_op}, km.data(), blocked, 1.0));
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, a13_op}, km.data(), blocked, 1.0));
    }
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d));
    FBF_TRY(einsum(s, "x,qp->xpq", {one_body_op, op.dm1}, s1.data(), {2, n, n}, 1.0));
    return solve_class(res, "Sir", sm.data(), km.data(), d, s1.data(), 1);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const auto m = std::ssize(cs.lambda);
  // T's two blocks, rows [0, n²) and [n², 2 n²).
  const EinsumOperand t_1 = t_operand(cs.t.data(), n, 2, m, d);
  const EinsumOperand t_2 = t_operand(cs.t.data() + (n * n), n, 2, m, d);
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {m}};

  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> slab_1 = slab_buffer(res, blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, blocks, ExtBlock::Sir2, step);
  Finish f;
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
    const int64_t count = nb * nc * m;
    double *const y_d = FBF_TRY(workspace_out(res, count));
    FBF_TRY(einsum(s, "rpiq,pqk->rik", {h2e_v1, t_1}, y_d, {nb, nc, m}, 1.0));
    FBF_TRY(einsum(s, "rpqi,pqk->rik", {h2e_v2, t_2}, y_d, {nb, nc, m}, 1.0));
    FBF_TRY(einsum(s, "ri,k->rik", {h1e_v, t1}, y_d, {nb, nc, m}, 1.0));
    const std::vector<double> y = FBF_TRY(download(res, y_d, count));
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t i = 0; i < nc; ++i) {
        f.add(y.data() + (((r * nc) + i) * m), cs.lambda,
              op.e_virt[static_cast<size_t>(b0 + r)] - op.e_core[static_cast<size_t>(i)], 1.0);
      }
    }
  }
  return PcClass{FBF_TRY(f.done("Sir")), cs.dropped};
}

Result<PcClassEnergies> pc_class_energies(DeviceResources &res, const MoBlocks &blocks,
                                          const EnergyInputs &in, const int32_t batch) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const ActiveOperands op = FBF_TRY(active_operands(res, blocks, in));

  PcClassEnergies out;
  const auto put = [&out](const size_t k, const PcClass c) {
    std::tie(out.energies[k], out.dropped[k]) = c;
  };
  put(0, FBF_TRY(pc_sr(res, blocks, in, batch, op)));
  put(1, FBF_TRY(pc_si(res, blocks, in, batch, op)));
  // Sijrs has no active index, so its PC energy is SC's.
  out.energies[2] = FBF_TRY(energy_sijrs(res, blocks, in, batch)).second;
  put(3, FBF_TRY(pc_sijr(res, blocks, batch, op)));
  put(4, FBF_TRY(pc_srsi(res, blocks, batch, op)));
  put(5, FBF_TRY(pc_srs(res, blocks, batch, op)));
  put(6, FBF_TRY(pc_sij(res, blocks, batch, op)));
  put(7, FBF_TRY(pc_sir(res, blocks, in, batch, op)));
  out.total = std::ranges::fold_left(out.energies, 0.0, std::plus{});
  return out;
}

} // namespace free_breakfast::nevpt2
