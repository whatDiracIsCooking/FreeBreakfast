/// @file
/// @brief The SC and PC reverse passes: from the class energies back to the
/// slabs (and through them onto Λ), h1e_v's rows and the orbital energies,
/// and onto every class block's bar M_G = dE/dG; then the sweep of those bars
/// back through the blocks' builders onto h1e, h2e and the RDMs.
/// einsum_adjoint is the one derivative rule; SC's finish (P, Q, R: Park Eqs.
/// 36-38), PC's (the solve's adjoint) and Sijrs are the only expressions
/// differentiated by hand.

module;

// The Sijrs launchers, defined in nevpt2_energy.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "nevpt2_energy_bridge.h"

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, geam, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import :shared;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// ---------------------------------------------------------------------------
// einsum's adjoint
// ---------------------------------------------------------------------------

// The axes of @p op from slowest to fastest when it is packed, row-major
// (strides empty, or row-major's) or column-major; nothing for any other view.
std::optional<std::vector<int32_t>> packed_axes(const EinsumOperand &op) {
  const auto rank = static_cast<int32_t>(op.dims.size());
  std::vector<int32_t> axes(static_cast<size_t>(rank));
  std::iota(axes.begin(), axes.end(), 0);
  if (op.strides.empty()) {
    return axes;
  }
  std::vector<int64_t> row(op.dims.size(), 1);
  for (int32_t a = rank - 1; a > 0; --a) {
    row[static_cast<size_t>(a - 1)] = row[static_cast<size_t>(a)] * op.dims[static_cast<size_t>(a)];
  }
  if (op.strides == row) {
    return axes;
  }
  if (op.strides == column_major(nullptr, op.dims).strides) {
    std::ranges::reverse(axes);
    return axes;
  }
  return std::nullopt;
}

Status einsum_adjoint(DeviceResources &res, const std::string_view subscripts,
                      const std::vector<EinsumOperand> &operands, const std::vector<double *> &bars,
                      const double *const out_bar, const std::vector<int64_t> &out_dims,
                      const EinsumOperand &delta, const double alpha) {
  const wwr::wwrStream_t s = res.stream();
  const size_t arrow = subscripts.find("->");
  if (arrow == std::string_view::npos) {
    return err_config("nevpt2::einsum_adjoint: '{}' has no '->'", subscripts);
  }
  const std::string_view rhs = subscripts.substr(arrow + 2);
  std::vector<std::string_view> groups;
  for (std::string_view lhs = subscripts.substr(0, arrow);;) {
    const size_t comma = lhs.find(',');
    groups.push_back(lhs.substr(0, comma));
    if (comma == std::string_view::npos) {
      break;
    }
    lhs.remove_prefix(comma + 1);
  }
  if (groups.size() != operands.size() || bars.size() != operands.size()) {
    return err_config(
        "nevpt2::einsum_adjoint: '{}' has {} label groups for {} operands and {} bars", subscripts,
        groups.size(), operands.size(), bars.size());
  }
  // Letters the subscripts do not use: the scatter's second label of each
  // repetition.
  std::string unused;
  for (const char c : std::string_view{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"}) {
    if (subscripts.find(c) == std::string_view::npos) {
      unused.push_back(c);
    }
  }

  for (size_t k = 0; k < operands.size(); ++k) {
    if (bars[k] == nullptr) {
      continue;
    }
    const std::string_view labels = groups[k];
    const EinsumOperand &op = operands[k];
    if (labels.size() != op.dims.size()) {
      return err_config(
          "nevpt2::einsum_adjoint: '{}' gives operand {} {} labels, and it has rank {}", subscripts,
          k, labels.size(), op.dims.size());
    }
    // The forward call with the output and operand k swapping roles: the
    // out bar first, then every other operand.
    std::string others{rhs};
    std::vector<EinsumOperand> ops{{.data = out_bar, .dims = out_dims}};
    for (size_t j = 0; j < operands.size(); ++j) {
      if (j != k) {
        others += ',';
        others += groups[j];
        ops.push_back(operands[j]);
      }
    }
    // Operand k's distinct labels that some other group carries: what its bar
    // depends on. A label only operand k carries is summed by the forward
    // call; repeated, it is a trace, and its bar is the delta scatter's.
    std::string kept;
    std::vector<int64_t> kept_dims;
    bool repeated = false;
    for (size_t a = 0; a < labels.size(); ++a) {
      const char c = labels[a];
      if (labels.find(c) < a) {
        repeated = true;
      } else if (others.find(c) != std::string::npos) {
        kept += c;
        kept_dims.push_back(op.dims[a]);
      } else if (labels.find(c, a + 1) == std::string_view::npos) {
        return err_config("nevpt2::einsum_adjoint: '{}' operand {}'s label '{}' is in no other "
                          "group, once: its bar would be a broadcast",
                          subscripts, k, c);
      }
    }
    const std::optional<std::vector<int32_t>> axes = packed_axes(op);

    if (!repeated && axes) {
      std::string out_labels;
      std::vector<int64_t> dims;
      for (const int32_t a : *axes) {
        out_labels += labels[static_cast<size_t>(a)];
        dims.push_back(op.dims[static_cast<size_t>(a)]);
      }
      FBF_TRY(einsum(s, others + "->" + out_labels, ops, bars[k], dims, alpha));
      continue;
    }

    if (repeated && axes) {
      const int64_t count = std::ranges::fold_left(kept_dims, int64_t{1}, std::multiplies{});
      double *const tmp = std::get<0>(res.device_regions<double>(static_cast<size_t>(count)));
      FBF_TRY(
          gpu_status(wwr::wwrMemsetAsync(tmp, 0, sizeof(double) * static_cast<size_t>(count), s)));
      FBF_TRY(einsum(s, others + "->" + kept, ops, tmp, kept_dims, alpha));
      // bar[.., x, .., x', ..] += tmp[..] delta[x, x'], one delta per
      // repetition, each repeated axis given a label of its own.
      std::string scatter = kept;
      std::vector<EinsumOperand> scatter_ops{{.data = tmp, .dims = kept_dims}};
      std::string target{labels};
      size_t next = 0;
      for (size_t a = 0; a < labels.size(); ++a) {
        if (labels.find(labels[a]) < a) {
          if (next == unused.size()) {
            return err_config("nevpt2::einsum_adjoint: '{}' leaves no label for its scatter",
                              subscripts);
          }
          target[a] = unused[next++];
          scatter += ',';
          scatter += labels[a];
          scatter += target[a];
          scatter_ops.push_back(delta);
        }
      }
      std::string out_labels;
      std::vector<int64_t> dims;
      for (const int32_t a : *axes) {
        out_labels += target[static_cast<size_t>(a)];
        dims.push_back(op.dims[static_cast<size_t>(a)]);
      }
      FBF_TRY(einsum(s, scatter + "->" + out_labels, scatter_ops, bars[k], dims, 1.0));
      continue;
    }

    if (!repeated && op.dims.size() == 2 && op.strides.size() == 2 && op.strides[0] == 1) {
      // A column-major m x n run of a larger array, leading dimension ld:
      // its bar packed column-major (einsum's row-major over the labels
      // reversed), then added in place.
      const int64_t m = op.dims[0];
      const int64_t n = op.dims[1];
      if (m * n == 0) {
        continue;
      }
      double *const tmp = std::get<0>(res.device_regions<double>(static_cast<size_t>(m * n)));
      FBF_TRY(
          gpu_status(wwr::wwrMemsetAsync(tmp, 0, sizeof(double) * static_cast<size_t>(m * n), s)));
      FBF_TRY(einsum(s, others + "->" + labels[1] + labels[0], ops, tmp, {n, m}, alpha));
      const auto ld = static_cast<int32_t>(op.strides[1]);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, static_cast<int32_t>(m),
          static_cast<int32_t>(n), &kOne<double>, tmp, static_cast<int32_t>(m), &kOne<double>,
          bars[k], ld, bars[k], ld)));
      continue;
    }
    return err_config("nevpt2::einsum_adjoint: '{}' operand {} is neither packed nor a "
                      "two-axis column-major view without a repeated label",
                      subscripts, k);
  }
  return {};
}

// ---------------------------------------------------------------------------
// The finish and the term lists
// ---------------------------------------------------------------------------

// One SC perturber: its norm and H are s (x[t1] + x[t2]) over its class's
// tables, or s x[t1] when t2 < 0, and its Δ = ε_v[v1] + ε_v[v2] - ε_c[c1] -
// ε_c[c2] over the indices that are not negative. Sijr and Srsi pair two
// table entries (t1 = t2 and s = ½ on the diagonal); every other class has
// one.
struct Perturber {
  int64_t t1 = 0;
  int64_t t2 = -1;
  double s = 1.0;
  int32_t v1 = -1;
  int32_t v2 = -1;
  int32_t c1 = -1;
  int32_t c2 = -1;
};

// The orbital energies, and the host sums R goes onto.
struct OrbitalEnergies {
  const std::vector<double> &core;
  const std::vector<double> &virt;
  std::vector<double> &core_bar;
  std::vector<double> &virt_bar;
};

// PySCF's _norm_to_energy over @p perturbers, and its derivatives: Park Eqs.
// 36-38, T = -1 / (Δ + H/N), P = dE/dN = 2T + Δ T², Q = dE/dH = T² and
// R = dE/dΔ = N T², each put back onto the tables (@p p_t, @p q_t) and the
// orbital energies the perturber reads. A vanishing norm contributes no
// energy and so no multipliers. Returns {Σ N, Σ E}.
std::pair<double, double> finish_adjoint(const std::vector<Perturber> &perturbers,
                                         const std::vector<double> &norm_t,
                                         const std::vector<double> &h_t, const OrbitalEnergies &e,
                                         std::vector<double> &p_t, std::vector<double> &q_t) {
  double norm_total = 0.0;
  double energy_total = 0.0;
  for (const Perturber &x : perturbers) {
    const auto t1 = static_cast<size_t>(x.t1);
    const auto t2 = static_cast<size_t>(x.t2);
    const double norm = x.s * (norm_t[t1] + (x.t2 < 0 ? 0.0 : norm_t[t2]));
    const double h = x.s * (h_t[t1] + (x.t2 < 0 ? 0.0 : h_t[t2]));
    double diff = 0.0;
    for (const int32_t v : {x.v1, x.v2}) {
      diff += v < 0 ? 0.0 : e.virt[static_cast<size_t>(v)];
    }
    for (const int32_t c : {x.c1, x.c2}) {
      diff -= c < 0 ? 0.0 : e.core[static_cast<size_t>(c)];
    }
    norm_total += norm;
    if (std::abs(norm) <= kNumericalZero) {
      continue;
    }
    energy_total -= norm / (diff + (h / norm));
    const double t = -1.0 / (diff + (h / norm));
    const double p = (2.0 * t) + (diff * t * t);
    const double q = t * t;
    const double r = norm * t * t;
    p_t[t1] += x.s * p;
    q_t[t1] += x.s * q;
    if (x.t2 >= 0) {
      p_t[t2] += x.s * p;
      q_t[t2] += x.s * q;
    }
    for (const int32_t v : {x.v1, x.v2}) {
      if (v >= 0) {
        e.virt_bar[static_cast<size_t>(v)] += r;
      }
    }
    for (const int32_t c : {x.c1, x.c2}) {
      if (c >= 0) {
        e.core_bar[static_cast<size_t>(c)] -= r;
      }
    }
  }
  return {norm_total, energy_total};
}

// One norm (h = false) or H (h = true) term of a class: the forward einsum
// `subscripts` over `ops`, scaled by alpha, and the bar each operand's
// adjoint goes into (null: none). block_free marks the one term that reads
// no class block (Sir's h1e_v-only norm term), whose share of Eq. 40 is
// taken against its multipliers directly.
struct ClassTerm {
  std::string_view subscripts;
  std::vector<EinsumOperand> ops;
  std::vector<double *> bars;
  bool h = false;
  double alpha = 1.0;
  bool block_free = false;
};

// Evaluates @p terms into @p norm and @p h, packed row-major over
// @p out_dims and accumulated into.
Status eval_terms(DeviceResources &res, const std::vector<ClassTerm> &terms, double *const norm,
                  double *const h, const std::vector<int64_t> &out_dims) {
  for (const ClassTerm &t : terms) {
    FBF_TRY(einsum(res.stream(), t.subscripts, t.ops, t.h ? h : norm, out_dims, t.alpha));
  }
  return {};
}

// Every term's adjoint against its multipliers, @p p for the norm terms and
// @p q for H's; a block-free term's <p, term> is added to @p eq40.
Status adjoint_terms(DeviceResources &res, const std::vector<ClassTerm> &terms,
                     const double *const p, const double *const q,
                     const std::vector<int64_t> &out_dims, const EinsumOperand &delta,
                     double *const eq40) {
  for (const ClassTerm &t : terms) {
    const double *const bar = t.h ? q : p;
    FBF_TRY(einsum_adjoint(res, t.subscripts, t.ops, t.bars, bar, out_dims, delta, t.alpha));
    if (t.block_free) {
      const std::string_view sub = t.subscripts;
      const size_t arrow = sub.find("->");
      std::vector<EinsumOperand> ops{{.data = bar, .dims = out_dims}};
      ops.insert(ops.end(), t.ops.begin(), t.ops.end());
      FBF_TRY(einsum(res.stream(),
                     std::string{sub.substr(arrow + 2)} + "," + std::string{sub.substr(0, arrow)} +
                         "->",
                     ops, eq40, {}, t.alpha));
    }
  }
  return {};
}

// Zeroes @p count doubles at @p d.
Status zero(DeviceResources &res, double *const d, const int64_t count) {
  if (count == 0) {
    return {};
  }
  return gpu_status(
      wwr::wwrMemsetAsync(d, 0, sizeof(double) * static_cast<size_t>(count), res.stream()));
}

// One slab's whole reverse step for a class whose perturbers lie in one slab:
// the terms evaluated, read back and finished, the multipliers uploaded, and
// every term's adjoint taken. The norm, H and multiplier arrays are buffers
// of their own, held across einsum_adjoint's workspace. Returns {Σ N, Σ E}.
Result<std::pair<double, double>>
slab_reverse(DeviceResources &res, const std::vector<ClassTerm> &terms,
             const std::vector<int64_t> &out_dims, const std::vector<Perturber> &perturbers,
             const OrbitalEnergies &e, const EinsumOperand &delta, double *const eq40) {
  const int64_t count = std::ranges::fold_left(out_dims, int64_t{1}, std::multiplies{});
  Buffers::device<double> norm_d = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> h_d = FBF_TRY(device_zeros(res, count));
  FBF_TRY(eval_terms(res, terms, norm_d.data(), h_d.data(), out_dims));
  const std::vector<double> norm_t = FBF_TRY(download(res, norm_d.data(), count));
  const std::vector<double> h_t = FBF_TRY(download(res, h_d.data(), count));
  std::vector<double> p_t(static_cast<size_t>(count), 0.0);
  std::vector<double> q_t(static_cast<size_t>(count), 0.0);
  const auto sums = finish_adjoint(perturbers, norm_t, h_t, e, p_t, q_t);
  const Buffers::device<double> p = FBF_TRY(upload(res, p_t));
  const Buffers::device<double> q = FBF_TRY(upload(res, q_t));
  FBF_TRY(adjoint_terms(res, terms, p.data(), q.data(), out_dims, delta, eq40));
  return sums;
}

// <m, g> over @p m's elements, added to the device scalar @p acc.
Status add_dot(DeviceResources &res, const Buffers::device<double> &m, const double *const g,
               double *const acc) {
  const auto count = static_cast<int64_t>(m.num_elements());
  return einsum(res.stream(), "a,a->",
                {{.data = m.data(), .dims = {count}}, {.data = g, .dims = {count}}}, acc, {}, 1.0);
}

// The bars on every class block G, row-major as G is: M_G = dE/dG. A metric a
// class reads straight from an RDM (Sr's dm1..dm3, Srsi's dm1, Sir's dm1 and
// dm2) is a block of that class here, so that each class's Eq. 40 is
// assembled from its own bars. sweep_blocks takes each back through its
// builder (make_*) onto the RDMs, h1e and h2e. class_energies_adjoint
// allocates them all at once, zeroed, and each class adds onto its own.
struct BlockBars {
  Buffers::device<double> a16, a17, a19, sr_dm3, sr_dm2, sr_dm1; // Sr
  Buffers::device<double> a22, a23, a25, dm3_h, dm2_h, dm1_h;    // Si
  Buffers::device<double> hdm1, a3;                              // Sijr
  Buffers::device<double> srsi_dm1, k27;                         // Srsi
  Buffers::device<double> rm2, a7;                               // Srs
  Buffers::device<double> hdm2, a9;                              // Sij
  Buffers::device<double> sir_dm2, sir_dm1, a12, a13;            // Sir
};

// One class's sums: Σ N, Σ E and Eq. 40.
struct ClassSums {
  double norm = 0.0;
  double energy = 0.0;
  double eq40 = 0.0;
};

// What every class's reverse walk reads besides its own blocks.
struct ReverseContext {
  DeviceResources &res;
  const MoBlocks &blocks;
  const EnergyInputs &in;
  const EnergyBars &bars;
  int32_t batch;
  const ActiveOperands &ops;
  OrbitalEnergies e;
};

// A slab bar's buffer, as slab_buffer's, and the slab's element count.
int64_t slab_count(const MoBlocks &blocks, const ExtBlock block, const int32_t b0,
                   const int32_t b1) {
  return std::ranges::fold_left(slab_dims(blocks, block, b0, b1), int64_t{1}, std::multiplies{});
}

// Pulls @p block's slab bar over [b0, b1) back onto Λ, when it is wanted.
Status slab_bar_onto_lambda(const ReverseContext &c, const ExtBlock block, const int32_t b0,
                            const int32_t b1, const double *const slab_bar) {
  if (c.bars.lambda == nullptr) {
    return {};
  }
  return slab_adjoint(c.res, c.blocks, block, b0, b1, slab_bar, c.bars.lambda);
}

// Σ_G <M_G, G> over @p pairs, added to the class's device scalar and read
// back: Eq. 40's block part.
Result<double>
eq40_sum(DeviceResources &res,
         const std::vector<std::pair<const Buffers::device<double> *, const double *>> &pairs,
         Buffers::device<double> &acc) {
  for (const auto &[m, g] : pairs) {
    FBF_TRY(add_dot(res, *m, g, acc.data()));
  }
  const std::vector<double> host = FBF_TRY(download(res, acc.data(), 1));
  return host[0];
}

// ---------------------------------------------------------------------------
// The eight classes, each energy.cpp's walk with its terms listed once and
// both evaluated and differentiated from the list.
// ---------------------------------------------------------------------------

Result<ClassSums> reverse_sr(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const Buffers::device<double> a16 = FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca));
  const Buffers::device<double> a17 = FBF_TRY(make_a17(res, o.h1e, o.h2e, o.dm2, o.dm3));
  const Buffers::device<double> a19 = FBF_TRY(make_a19(res, o.h1e, o.h2e, o.dm1, o.dm2));
  const EinsumOperand a16_op{.data = a16.data(), .dims = d6};
  const EinsumOperand a17_op{.data = a17.data(), .dims = d4};
  const EinsumOperand a19_op{.data = a19.data(), .dims = d2};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sr);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sr, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};
    double *const sb = slab_bar.data();
    double *const hb = c.bars.h1e_v_sr == nullptr ? nullptr : c.bars.h1e_v_sr + b0;
    const std::vector<ClassTerm> terms{
        {"ipqr,pqrabc,iabc->i", {h2e_v, a16_op, h2e_v}, {sb, m.a16.data(), sb}, true, 1.0},
        {"ipqr,pqra,ia->i", {h2e_v, a17_op, h1e_v}, {sb, m.a17.data(), hb}, true, 2.0},
        {"ip,pa,ia->i", {h1e_v, a19_op, h1e_v}, {hb, m.a19.data(), hb}, true, 1.0},
        {"ipqr,rpqbac,iabc->i", {h2e_v, o.dm3, h2e_v}, {sb, m.sr_dm3.data(), sb}, false, 1.0},
        {"ipqr,rpqa,ia->i", {h2e_v, o.dm2, h1e_v}, {sb, m.sr_dm2.data(), hb}, false, 2.0},
        {"ip,pa,ia->i", {h1e_v, o.dm1, h1e_v}, {hb, m.sr_dm1.data(), hb}, false, 1.0}};
    std::vector<Perturber> perturbers;
    for (int64_t r = 0; r < nb; ++r) {
      perturbers.push_back({.t1 = r, .v1 = static_cast<int32_t>(b0 + r)});
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sr, b0, b1, sb));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res,
                               {{&m.a16, a16.data()},
                                {&m.a17, a17.data()},
                                {&m.a19, a19.data()},
                                {&m.sr_dm3, o.dm3.data},
                                {&m.sr_dm2, o.dm2.data},
                                {&m.sr_dm1, o.dm1.data}},
                               eq40));
  return sums;
}

Result<ClassSums> reverse_si(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const Buffers::device<double> a22 =
      FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca));
  const Buffers::device<double> a23 = FBF_TRY(make_a23(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3));
  const Buffers::device<double> a25 = FBF_TRY(make_a25(res, o.h1e, o.h2e, o.dm1, o.dm2));
  const Buffers::device<double> dm3_h = FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta));
  const Buffers::device<double> dm2_h = FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta));
  const Buffers::device<double> dm1_h = FBF_TRY(make_hdm1(res, o.dm1, o.delta));
  const EinsumOperand a22_op{.data = a22.data(), .dims = d6};
  const EinsumOperand a23_op{.data = a23.data(), .dims = d4};
  const EinsumOperand a25_op{.data = a25.data(), .dims = d2};
  const EinsumOperand dm3_h_op{.data = dm3_h.data(), .dims = d6};
  const EinsumOperand dm2_h_op{.data = dm2_h.data(), .dims = d4};
  const EinsumOperand dm1_h_op{.data = dm1_h.data(), .dims = d2};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Si);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Si, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};
    double *const sb = slab_bar.data();
    double *const hb = c.bars.h1e_v_si == nullptr ? nullptr : c.bars.h1e_v_si + (n * b0);
    const std::vector<ClassTerm> terms{
        {"qpir,pqrabc,baic->i", {h2e_v, a22_op, h2e_v}, {sb, m.a22.data(), sb}, true, 1.0},
        {"qpir,pqra,ai->i", {h2e_v, a23_op, h1e_v}, {sb, m.a23.data(), hb}, true, 2.0},
        {"pi,pa,ai->i", {h1e_v, a25_op, h1e_v}, {hb, m.a25.data(), hb}, true, 1.0},
        {"qpir,rpqbac,baic->i", {h2e_v, dm3_h_op, h2e_v}, {sb, m.dm3_h.data(), sb}, false, 1.0},
        {"qpir,rpqa,ai->i", {h2e_v, dm2_h_op, h1e_v}, {sb, m.dm2_h.data(), hb}, false, 2.0},
        {"pi,pa,ai->i", {h1e_v, dm1_h_op, h1e_v}, {hb, m.dm1_h.data(), hb}, false, 1.0}};
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nb; ++i) {
      perturbers.push_back({.t1 = i, .c1 = static_cast<int32_t>(b0 + i)});
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Si, b0, b1, sb));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res,
                               {{&m.a22, a22.data()},
                                {&m.a23, a23.data()},
                                {&m.a25, a25.data()},
                                {&m.dm3_h, dm3_h.data()},
                                {&m.dm2_h, dm2_h.data()},
                                {&m.dm1_h, dm1_h.data()}},
                               eq40));
  return sums;
}

// Sijrs by hand: it reads no block, so its slab's bar and its orbital
// energies' are sijrs_adjoint's, and Eq. 40 is ½ <g-bar, g> (E is quadratic
// in g at fixed orbital energies).
Result<ClassSums> reverse_sijrs(ReverseContext &c) {
  DeviceResources &res = c.res;
  if (c.blocks.ncore == 0 || c.blocks.nvirt == 0) {
    return ClassSums{};
  }
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 3));
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sijrs);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sijrs, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sijrs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t count = slab_count(c.blocks, ExtBlock::Sijrs, b0, b1);
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sijrs, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), count));
    FBF_TRY(gpu_status(device::sijrs(res.stream(), slab_buf.data(), b1 - b0, b0, c.blocks.ncore,
                                     c.blocks.nvirt, c.in.e_core, c.in.e_virt, acc.data())));
    FBF_TRY(gpu_status(device::sijrs_adjoint(
        res.stream(), slab_buf.data(), b1 - b0, b0, c.blocks.ncore, c.blocks.nvirt, c.in.e_core,
        c.in.e_virt, slab_bar.data(), c.bars.e_core, c.bars.e_virt)));
    FBF_TRY(einsum(
        res.stream(), "a,a->",
        {{.data = slab_bar.data(), .dims = {count}}, {.data = slab_buf.data(), .dims = {count}}},
        acc.data() + 2, {}, 0.5));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sijrs, b0, b1, slab_bar.data()));
  }
  const std::vector<double> host = FBF_TRY(download(res, acc.data(), 3));
  return ClassSums{.norm = host[0], .energy = host[1], .eq40 = host[2]};
}

// Sijr: the (r, j, i) tables are symmetrized in (i, j), so each perturber
// over j <= i reads, and its multipliers go back onto, both (r, j, i) and
// (r, i, j).
Result<ClassSums> reverse_sijr(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const std::vector<int64_t> d2{n, n};
  const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, o.dm1, o.delta));
  const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = d2};
  const Buffers::device<double> a3 =
      FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta));
  const EinsumOperand a3_op{.data = a3.data(), .dims = d2};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sijr);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sijr, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sijr, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{
        {"rpji,raji,pa->rji", {h2e_v, h2e_v, hdm1_op}, {sb, sb, m.hdm1.data()}, false, 2.0},
        {"rpji,raij,pa->rji", {h2e_v, h2e_v, hdm1_op}, {sb, sb, m.hdm1.data()}, false, -1.0},
        {"rpji,raji,pa->rji", {h2e_v, h2e_v, a3_op}, {sb, sb, m.a3.data()}, true, 2.0},
        {"rpji,raij,pa->rji", {h2e_v, h2e_v, a3_op}, {sb, sb, m.a3.data()}, true, -1.0}};
    const auto at = [&](const int64_t r, const int64_t j, const int64_t i) {
      return (((r * nc) + j) * nc) + i;
    };
    std::vector<Perturber> perturbers;
    for (int64_t r = 0; r < nr; ++r) {
      for (int64_t j = 0; j < nc; ++j) {
        for (int64_t i = j; i < nc; ++i) {
          perturbers.push_back({.t1 = at(r, j, i),
                                .t2 = at(r, i, j),
                                .s = i == j ? 0.5 : 1.0,
                                .v1 = static_cast<int32_t>(b0 + r),
                                .c1 = static_cast<int32_t>(j),
                                .c2 = static_cast<int32_t>(i)});
        }
      }
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nr, nc, nc}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sijr, b0, b1, sb));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res, {{&m.hdm1, hdm1.data()}, {&m.a3, a3.data()}}, eq40));
  return sums;
}

// Srsi: its (r, s) symmetrization pairs rows of different slabs, so the walk
// runs twice. The first fills the whole (v, v, c) tables, as energy.cpp's
// does; the finish then puts the multipliers onto both (r, s, i) and
// (s, r, i); the second rebuilds each slab pair and takes its adjoint
// against the multiplier tables' rows.
Result<ClassSums> reverse_srsi(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const Buffers::device<double> k27 = FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2));
  const EinsumOperand k27_op{.data = k27.data(), .dims = {n, n}};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srsi);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_a = slab_buffer(res, c.blocks, ExtBlock::Srsi, step);
  Buffers::device<double> slab_b = slab_buffer(res, c.blocks, ExtBlock::SrsiT, step);
  Buffers::device<double> bar_a = slab_buffer(res, c.blocks, ExtBlock::Srsi, step);
  Buffers::device<double> bar_b = slab_buffer(res, c.blocks, ExtBlock::SrsiT, step);
  // h[r_batch, s, i, a] and h[s, r_batch, i, a], with their bars.
  const auto terms = [&](const int32_t b0, const int32_t b1, double *const ba, double *const bb) {
    const EinsumOperand h_a =
        column_major(slab_a.data(), slab_dims(c.blocks, ExtBlock::Srsi, b0, b1));
    const EinsumOperand h_b =
        column_major(slab_b.data(), slab_dims(c.blocks, ExtBlock::SrsiT, b0, b1));
    double *const md = m.srsi_dm1.data();
    double *const mk = m.k27.data();
    return std::vector<ClassTerm>{
        {"rsip,rsia,pa->rsi", {h_a, h_a, o.dm1}, {ba, ba, md}, false, 2.0},
        {"rsip,sria,pa->rsi", {h_a, h_b, o.dm1}, {ba, bb, md}, false, -1.0},
        {"rsip,rsia,pa->rsi", {h_a, h_a, k27_op}, {ba, ba, mk}, true, 2.0},
        {"rsip,sria,pa->rsi", {h_a, h_b, k27_op}, {ba, bb, mk}, true, -1.0}};
  };

  Buffers::device<double> norm_d = FBF_TRY(device_zeros(res, nv * nv * nc));
  Buffers::device<double> h_d = FBF_TRY(device_zeros(res, nv * nv * nc));
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    FBF_TRY(eval_terms(res, terms(b0, b1, nullptr, nullptr), norm_d.data() + (b0 * nv * nc),
                       h_d.data() + (b0 * nv * nc), {b1 - b0, nv, nc}));
  }
  const std::vector<double> norm_t = FBF_TRY(download(res, norm_d.data(), nv * nv * nc));
  const std::vector<double> h_t = FBF_TRY(download(res, h_d.data(), nv * nv * nc));
  const auto at = [&](const int64_t r, const int64_t ss, const int64_t i) {
    return (((r * nv) + ss) * nc) + i;
  };
  std::vector<Perturber> perturbers;
  for (int64_t r = 0; r < nv; ++r) {
    for (int64_t ss = r; ss < nv; ++ss) {
      for (int64_t i = 0; i < nc; ++i) {
        perturbers.push_back({.t1 = at(r, ss, i),
                              .t2 = at(ss, r, i),
                              .s = r == ss ? 0.5 : 1.0,
                              .v1 = static_cast<int32_t>(r),
                              .v2 = static_cast<int32_t>(ss),
                              .c1 = static_cast<int32_t>(i)});
      }
    }
  }
  std::vector<double> p_t(norm_t.size(), 0.0);
  std::vector<double> q_t(norm_t.size(), 0.0);
  const auto [norm_sum, energy_sum] = finish_adjoint(perturbers, norm_t, h_t, c.e, p_t, q_t);
  const Buffers::device<double> p = FBF_TRY(upload(res, p_t));
  const Buffers::device<double> q = FBF_TRY(upload(res, q_t));

  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    FBF_TRY(zero(res, bar_a.data(), slab_count(c.blocks, ExtBlock::Srsi, b0, b1)));
    FBF_TRY(zero(res, bar_b.data(), slab_count(c.blocks, ExtBlock::SrsiT, b0, b1)));
    FBF_TRY(adjoint_terms(res, terms(b0, b1, bar_a.data(), bar_b.data()), p.data() + (b0 * nv * nc),
                          q.data() + (b0 * nv * nc), {b1 - b0, nv, nc}, o.delta, eq40.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srsi, b0, b1, bar_a.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::SrsiT, b0, b1, bar_b.data()));
  }
  const double eq40_value =
      FBF_TRY(eq40_sum(res, {{&m.srsi_dm1, o.dm1.data}, {&m.k27, k27.data()}}, eq40));
  return ClassSums{.norm = norm_sum, .energy = energy_sum, .eq40 = eq40_value};
}

Result<ClassSums> reverse_srs(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  if (nv == 0) {
    return ClassSums{};
  }
  const std::vector<int64_t> d4(4, n);
  const auto [rm2, a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  const EinsumOperand rm2_op{.data = rm2.data(), .dims = d4};
  const EinsumOperand a7_op{.data = a7.data(), .dims = d4};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srs);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Srs, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Srs, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{
        {"rsqp,rsba,pqba->rs", {h2e_v, h2e_v, rm2_op}, {sb, sb, m.rm2.data()}, false, 0.5},
        {"rsqp,rsba,pqab->rs", {h2e_v, h2e_v, a7_op}, {sb, sb, m.a7.data()}, true, 0.5}};
    std::vector<Perturber> perturbers;
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        perturbers.push_back({.t1 = (r * nv) + ss,
                              .v1 = static_cast<int32_t>(b0 + r),
                              .v2 = static_cast<int32_t>(ss)});
      }
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nb, nv}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srs, b0, b1, sb));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res, {{&m.rm2, rm2.data()}, {&m.a7, a7.data()}}, eq40));
  return sums;
}

Result<ClassSums> reverse_sij(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  if (nc == 0) {
    return ClassSums{};
  }
  const std::vector<int64_t> d4(4, n);
  const Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta));
  const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
  const Buffers::device<double> hdm3 =
      FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
  const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
  const Buffers::device<double> a9 = FBF_TRY(make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op));
  const EinsumOperand a9_op{.data = a9.data(), .dims = d4};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sij);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sij, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sij, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{
        {"qpij,baij,pqab->ij", {h2e_v, h2e_v, hdm2_op}, {sb, sb, m.hdm2.data()}, false, 0.5},
        {"qpij,baij,pqab->ij", {h2e_v, h2e_v, a9_op}, {sb, sb, m.a9.data()}, true, 0.5}};
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nb; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        perturbers.push_back({.t1 = (i * nc) + j,
                              .c1 = static_cast<int32_t>(b0 + i),
                              .c2 = static_cast<int32_t>(j)});
      }
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nb, nc}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sij, b0, b1, sb));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res, {{&m.hdm2, hdm2.data()}, {&m.a9, a9.data()}}, eq40));
  return sums;
}

// Sir: two slabs per step and h1e_v_Sir's rows; "raai" reads h2e_v2 on its
// diagonal, whose bar einsum_adjoint scatters against delta.
Result<ClassSums> reverse_sir(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const std::vector<int64_t> d4(4, n);
  const Buffers::device<double> a12 = FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3));
  const Buffers::device<double> a13 =
      FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  const EinsumOperand a12_op{.data = a12.data(), .dims = d4};
  const EinsumOperand a13_op{.data = a13.data(), .dims = d4};
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  ClassSums sums;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sir1);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  Buffers::device<double> bar_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> bar_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sir1, b0, b1, slab_1.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sir2, b0, b1, slab_2.data()));
    FBF_TRY(zero(res, bar_1.data(), slab_count(c.blocks, ExtBlock::Sir1, b0, b1)));
    FBF_TRY(zero(res, bar_2.data(), slab_count(c.blocks, ExtBlock::Sir2, b0, b1)));
    const EinsumOperand v1 =
        column_major(slab_1.data(), slab_dims(c.blocks, ExtBlock::Sir1, b0, b1));
    const EinsumOperand v2 =
        column_major(slab_2.data(), slab_dims(c.blocks, ExtBlock::Sir2, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_sir + b0, .dims = {nb, nc}, .strides = {1, nv}};
    double *const s1 = bar_1.data();
    double *const s2 = bar_2.data();
    double *const hb = c.bars.h1e_v_sir == nullptr ? nullptr : c.bars.h1e_v_sir + b0;
    double *const md2 = m.sir_dm2.data();
    double *const md1 = m.sir_dm1.data();
    const std::vector<ClassTerm> terms{
        {"rpiq,raib,qpab->ir", {v1, v1, o.dm2}, {s1, s1, md2}, false, 2.0},
        {"rpiq,rabi,qpab->ir", {v1, v2, o.dm2}, {s1, s2, md2}, false, -1.0},
        {"rpqi,raib,qpab->ir", {v2, v1, o.dm2}, {s2, s1, md2}, false, -1.0},
        {"raqi,rabi,qb->ir", {v2, v2, o.dm1}, {s2, s2, md1}, false, 2.0},
        {"rpqi,rabi,qbap->ir", {v2, v2, o.dm2}, {s2, s2, md2}, false, -1.0},
        {"rpqi,raai,qp->ir", {v2, v2, o.dm1}, {s2, s2, md1}, false, 1.0},
        {"rpiq,ri,qp->ir", {v1, h1e_v, o.dm1}, {s1, hb, md1}, false, 4.0},
        {"rpqi,ri,qp->ir", {v2, h1e_v, o.dm1}, {s2, hb, md1}, false, -2.0},
        {"ri,ri->ir", {h1e_v, h1e_v}, {hb, hb}, false, 2.0, true},
        {"rpiq,raib,pqab->ir", {v1, v1, a12_op}, {s1, s1, m.a12.data()}, true, 2.0},
        {"rpiq,rabi,pqab->ir", {v1, v2, a12_op}, {s1, s2, m.a12.data()}, true, -1.0},
        {"rpqi,raib,pqab->ir", {v2, v1, a12_op}, {s2, s1, m.a12.data()}, true, -1.0},
        {"rpqi,rabi,pqab->ir", {v2, v2, a13_op}, {s2, s2, m.a13.data()}, true, 1.0}};
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nc; ++i) {
      for (int64_t r = 0; r < nb; ++r) {
        perturbers.push_back({.t1 = (i * nb) + r,
                              .v1 = static_cast<int32_t>(b0 + r),
                              .c1 = static_cast<int32_t>(i)});
      }
    }
    const auto [n_part, e_part] =
        FBF_TRY(slab_reverse(res, terms, {nc, nb}, perturbers, c.e, o.delta, eq40.data()));
    sums.norm += n_part;
    sums.energy += e_part;
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir1, b0, b1, s1));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir2, b0, b1, s2));
  }
  sums.eq40 = FBF_TRY(eq40_sum(res,
                               {{&m.sir_dm2, o.dm2.data},
                                {&m.sir_dm1, o.dm1.data},
                                {&m.a12, a12.data()},
                                {&m.a13, a13.data()}},
                               eq40));
  return sums;
}

// ---------------------------------------------------------------------------
// The sweep: the block bars back through their builders
// ---------------------------------------------------------------------------

// Each block's bar M_G back through its make_* onto @p bars' h1e, h2e and
// RDM fields. Every builder is listed term by term with energy.cpp's
// subscripts, coefficients and operands (a transpose as its one-operand
// einsum, NumPy's axes spelled as labels: axes {1, 4, 0, 2, 5, 3} of
// "abcdef" is "->beacfd"), and einsum_adjoint differentiates each. A term
// whose bars are all null (delta alone) is listed anyway, so a builder still
// reads against its forward line by line.
//
// Reverse topological: an intermediate's bar is complete before it is swept
// onto what built it. Sij's a9 reads hdm3, which reads hdm2 (a9, hdm3,
// hdm2, its dm2b); Srs's a7 reads rm3, which reads rm2 (a7, rm3, rm2);
// Sijr's a3 reads hdm1 (a3, hdm1); and a16's and a22's fdm2, and a17's and
// a19's h1e - Σ (mj|jn), are swept within their builders. A block that is a
// class metric too (hdm1, rm2, hdm2) is @p m's own bar, which the sweep adds
// to before sweeping it, so @p m is spent afterwards.
//
// The intermediates a bar is taken against (hdm1, hdm2, hdm3, rm2, rm3, the
// modified h1e) are rebuilt with the forward's own builders, not kept from
// the walk. An operand whose own bar is the only one its term takes is never
// read (einsum_adjoint reads every operand but k for k's bar), so fdm2 and
// dm2b are not rebuilt: their operand is a shape. Every intermediate bar is
// a buffer of its own, held across einsum_adjoint's workspace.
Status sweep_blocks(DeviceResources &res, const ActiveOperands &o, BlockBars &m,
                    const EnergyBars &bars) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const auto &[h1e, h2e, dm1, dm2, dm3, f3ca, f3ac, delta, delta_buf, e_core, e_virt] = o;
  double *const bh1 = bars.h1e;
  double *const bh2 = bars.h2e;
  double *const b1 = bars.dm1;
  double *const b2 = bars.dm2;
  double *const b3 = bars.dm3;
  double *const bca = bars.f3ca;
  double *const bac = bars.f3ac;
  // Every term of @p terms against @p out_bar, the bar on what they build.
  const auto sweep = [&](const std::vector<ClassTerm> &terms,
                         const Buffers::device<double> &out_bar,
                         const std::vector<int64_t> &out_dims) {
    return adjoint_terms(res, terms, out_bar.data(), out_bar.data(), out_dims, delta, nullptr);
  };
  // A block that is an RDM itself, G = dm: the identity einsum.
  const auto identity = [](const EinsumOperand &dm, double *const bar) {
    const std::string_view subscripts = dm.dims.size() == 6   ? "abcdef->abcdef"
                                        : dm.dims.size() == 4 ? "abcd->abcd"
                                                              : "ab->ab";
    return ClassTerm{.subscripts = subscripts, .ops = {dm}, .bars = {bar}};
  };

  // Sr: a16, its fdm2 (gathered off the diagonal diag_a16 wrote,
  // a16[p, i, r, a, b, i] += fdm2[p, r, a, b]), a17, a19, dm1..dm3.
  FBF_TRY(sweep({{"ib,rpqiac->pqrabc", {h1e, dm3}, {bh1, b3}, false, -1.0},
                 {"ia,rpqbic->pqrabc", {h1e, dm3}, {bh1, b3}, false, 1.0},
                 {"ci,rpqbai->pqrabc", {h1e, dm3}, {bh1, b3}, false, -1.0},
                 {"abcdef->beacfd", {f3ca}, {bca}, false, -1.0},
                 {"kbia,rpqcki->pqrabc", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"kbaj,rpqjkc->pqrabc", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"cbij,rpqjai->pqrabc", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"abcdef->bcaedf", {f3ac}, {bac}, false, 1.0},
                 {"abcdef->bcaedf", {f3ca}, {bca}, false, -1.0},
                 {"jbij,rpqiac->pqrabc", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"cjka,rpqbjk->pqrabc", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"jcij,rpqbai->pqrabc", {h2e, dm3}, {bh2, b3}, false, 1.0}},
                m.a16, d6));
  {
    Buffers::device<double> fdm2 = FBF_TRY(device_zeros(res, n * n * n * n));
    FBF_TRY(einsum(s, "pirabi->prab", {{.data = m.a16.data(), .dims = d6}}, fdm2.data(), d4, 1.0));
    FBF_TRY(sweep({{"kbij,rpajki->prab", {h2e, dm3}, {bh2, b3}, false, 1.0}}, fdm2, d4));
  }
  {
    // make_a17's and make_a19's h1e - Σ_j (mj|jn), row-major.
    Buffers::device<double> h1e_buf = FBF_TRY(device_zeros(res, n * n));
    FBF_TRY(einsum(s, "mn->mn", {h1e}, h1e_buf.data(), d2, 1.0));
    FBF_TRY(einsum(s, "mjjn->mn", {h2e}, h1e_buf.data(), d2, -1.0));
    const EinsumOperand h1e_mod{.data = h1e_buf.data(), .dims = d2};
    Buffers::device<double> h1e_mod_bar = FBF_TRY(device_zeros(res, n * n));
    double *const bm = h1e_mod_bar.data();
    FBF_TRY(sweep({{"pi,cabi->abcp", {h1e_mod, dm2}, {bm, b2}, false, -1.0},
                   {"kpij,cabjki->abcp", {h2e, dm3}, {bh2, b3}, false, -1.0}},
                  m.a17, d4));
    FBF_TRY(sweep({{"pi,ai->ap", {h1e_mod, dm1}, {bm, b1}, false, -1.0},
                   {"kpij,ajki->ap", {h2e, dm2}, {bh2, b2}, false, -1.0}},
                  m.a19, d2));
    FBF_TRY(sweep({{"mn->mn", {h1e}, {bh1}, false, 1.0}, {"mjjn->mn", {h2e}, {bh2}, false, -1.0}},
                  h1e_mod_bar, d2));
  }
  FBF_TRY(sweep({identity(dm3, b3)}, m.sr_dm3, d6));
  FBF_TRY(sweep({identity(dm2, b2)}, m.sr_dm2, d4));
  FBF_TRY(sweep({identity(dm1, b1)}, m.sr_dm1, d2));

  // Si: a22, its fdm2a (diag_a22a: a22[x, i, k, i, b, c] -= fdm2a[x, k, b, c])
  // and fdm2b (diag_a22b: a22[x, i, k, a, i, c] += 2 fdm2b[x, k, a, c]), a23,
  // a25, dm3_h, dm2_h. dm1_h is make_hdm1, so its bar joins Sijr's hdm1's.
  FBF_TRY(sweep({{"pb,kipjac->ijkabc", {h1e, dm3}, {bh1, b3}, false, -1.0},
                 {"pa,kibjpc->ijkabc", {h1e, dm3}, {bh1, b3}, false, -1.0},
                 {"cp,kibjap->ijkabc", {h1e, dm3}, {bh1, b3}, false, 1.0},
                 {"cqra,kibjqr->ijkabc", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"qcpq,kibjap->ijkabc", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"abcdef->bfaced", {f3ac}, {bac}, false, -1.0},
                 {"pqab,kiqjpc->ijkabc", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"pcrb,kiajpr->ijkabc", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"cqrb,kiqjar->ijkabc", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"abcdef->bdaecf", {f3ac}, {bac}, false, -1.0},
                 {"abcdef->bdaecf", {f3ca}, {bca}, false, 1.0},
                 {"jb,kiac->ijkabc", {h1e, dm2}, {bh1, b2}, false, 2.0},
                 {"pjrb,kiprac->ijkabc", {h2e, dm3}, {bh2, b3}, false, 2.0}},
                m.a22, d6));
  {
    Buffers::device<double> fdm2a = FBF_TRY(device_zeros(res, n * n * n * n));
    Buffers::device<double> fdm2b = FBF_TRY(device_zeros(res, n * n * n * n));
    const EinsumOperand a22_bar{.data = m.a22.data(), .dims = d6};
    FBF_TRY(einsum(s, "xikibc->xkbc", {a22_bar}, fdm2a.data(), d4, -1.0));
    FBF_TRY(einsum(s, "xikaic->xkac", {a22_bar}, fdm2b.data(), d4, 2.0));
    FBF_TRY(sweep({{"pqrb,kiqcpr->ikbc", {h2e, dm3}, {bh2, b3}, false, 1.0}}, fdm2a, d4));
    FBF_TRY(sweep({{"pa,kipc->ikac", {h1e, dm2}, {bh1, b2}, false, 1.0},
                   {"cp,kiap->ikac", {h1e, dm2}, {bh1, b2}, false, -1.0},
                   {"cqra,kiqr->ikac", {h2e, dm2}, {bh2, b2}, false, -1.0},
                   {"qcpq,kiap->ikac", {h2e, dm2}, {bh2, b2}, false, 1.0},
                   {"pqra,kiqcpr->ikac", {h2e, dm3}, {bh2, b3}, false, 1.0},
                   {"rcpq,kiaqrp->ikac", {h2e, dm3}, {bh2, b3}, false, -1.0}},
                  fdm2b, d4));
  }
  FBF_TRY(sweep({{"ip,caib->abcp", {h1e, dm2}, {bh1, b2}, false, -1.0},
                 {"pijk,cajbik->abcp", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"bp,ca->abcp", {h1e, dm1}, {bh1, b1}, false, 2.0},
                 {"pibk,caik->abcp", {h2e, dm2}, {bh2, b2}, false, 2.0}},
                m.a23, d4));
  FBF_TRY(sweep({{"pi,ai->ap", {h1e, dm1}, {bh1, b1}, false, -1.0},
                 {"pijk,jaik->ap", {h2e, dm2}, {bh2, b2}, false, -1.0},
                 {"ap->pa", {h1e}, {bh1}, false, 2.0},
                 {"piaj,ij->ap", {h2e, dm1}, {bh2, b1}, false, 2.0}},
                m.a25, d2));
  FBF_TRY(sweep({{"abef,cd->abcdef", {dm2, delta}, {b2, nullptr}, false, 2.0},
                 {"abcdef->abdcef", {dm3}, {b3}, false, -1.0}},
                m.dm3_h, d6));
  FBF_TRY(sweep({{"ab,cd->abcd", {dm1, delta}, {b1, nullptr}, false, 2.0},
                 {"abcd->abdc", {dm2}, {b2}, false, -1.0}},
                m.dm2_h, d4));
  // dm1_h's value is not read: the identity's one operand is the one with a bar.
  FBF_TRY(sweep({identity({.dims = d2}, m.hdm1.data())}, m.dm1_h, d2));

  // Sijr: a3, then hdm1 (Si's dm1_h's bar already in it).
  {
    const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, dm1, delta));
    const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = d2};
    FBF_TRY(sweep({{"ia,ip->pa", {h1e, hdm1_op}, {bh1, m.hdm1.data()}, false, 1.0},
                   {"ijka,pj,ik->pa", {h2e, delta, dm1}, {bh2, nullptr, b1}, false, 2.0},
                   {"ijka,jpik->pa", {h2e, dm2}, {bh2, b2}, false, -1.0}},
                  m.a3, d2));
  }
  FBF_TRY(sweep({{"ij->ij", {delta}, {nullptr}, false, 2.0}, {"ab->ba", {dm1}, {b1}, false, -1.0}},
                m.hdm1, d2));

  // Srsi: k27, dm1.
  FBF_TRY(sweep({{"ai,pi->pa", {h1e, dm1}, {bh1, b1}, false, -1.0},
                 {"iajk,pkij->pa", {h2e, dm2}, {bh2, b2}, false, -1.0},
                 {"iaji,pj->pa", {h2e, dm1}, {bh2, b1}, false, 1.0}},
                m.k27, d2));
  FBF_TRY(sweep({identity(dm1, b1)}, m.srsi_dm1, d2));

  // Srs: a7, rm3, rm2. make_a7 is what builds rm2; its a7 is not read.
  {
    const Buffers::device<double> rm2 = FBF_TRY(make_a7(res, h1e, h2e, dm1, dm2, dm3, delta)).first;
    const EinsumOperand rm2_op{.data = rm2.data(), .dims = d4};
    const Buffers::device<double> rm3 = FBF_TRY(make_rm3(res, dm2, dm3, rm2_op, delta));
    const EinsumOperand rm3_op{.data = rm3.data(), .dims = d6};
    Buffers::device<double> rm3_bar = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
    double *const br2 = m.rm2.data();
    double *const br3 = rm3_bar.data();
    FBF_TRY(sweep({{"bi,pqia->pqab", {h1e, rm2_op}, {bh1, br2}, false, -1.0},
                   {"ai,pqbi->pqab", {h1e, rm2_op}, {bh1, br2}, false, -1.0},
                   {"kbij,pqkija->pqab", {h2e, rm3_op}, {bh2, br3}, false, -1.0},
                   {"kaij,pqkibj->pqab", {h2e, rm3_op}, {bh2, br3}, false, -1.0},
                   {"baij,pqij->pqab", {h2e, rm2_op}, {bh2, br2}, false, -1.0}},
                  m.a7, d4));
    FBF_TRY(sweep({{"injmkl->ijklmn", {dm3}, {b3}, false, 1.0},
                   {"jn,imkl->ijklmn", {delta, dm2}, {nullptr, b2}, false, -1.0},
                   {"km,ijln->ijklmn", {delta, rm2_op}, {nullptr, br2}, false, -1.0},
                   {"kn,ijml->ijklmn", {delta, rm2_op}, {nullptr, br2}, false, -1.0}},
                  rm3_bar, d6));
  }
  FBF_TRY(sweep({{"iljk->ijkl", {dm2}, {b2}, false, 1.0},
                 {"ik,jl->ijkl", {dm1, delta}, {b1, nullptr}, false, -1.0}},
                m.rm2, d4));

  // Sij: a9, hdm3, hdm2, dm2b.
  {
    const Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, dm1, dm2, delta));
    const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
    const Buffers::device<double> hdm3 = FBF_TRY(make_hdm3(res, dm1, dm2, dm3, hdm2_op, delta));
    const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = d6};
    Buffers::device<double> hdm3_bar = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
    double *const bd2 = m.hdm2.data();
    double *const bd3 = hdm3_bar.data();
    FBF_TRY(sweep({{"ib,pqai->pqab", {h1e, hdm2_op}, {bh1, bd2}, false, 1.0},
                   {"ijib,pqaj->pqab", {h2e, hdm2_op}, {bh2, bd2}, false, 2.0},
                   {"ijjb,pqai->pqab", {h2e, hdm2_op}, {bh2, bd2}, false, -1.0},
                   {"ijkb,pkqaij->pqab", {h2e, hdm3_op}, {bh2, bd3}, false, -1.0},
                   {"ia,pqib->pqab", {h1e, hdm2_op}, {bh1, bd2}, false, 1.0},
                   {"ijja,pqib->pqab", {h2e, hdm2_op}, {bh2, bd2}, false, -1.0},
                   {"ijba,pqji->pqab", {h2e, hdm2_op}, {bh2, bd2}, false, -1.0},
                   {"ijia,pqjb->pqab", {h2e, hdm2_op}, {bh2, bd2}, false, 2.0},
                   {"ijka,pqkjbi->pqab", {h2e, hdm3_op}, {bh2, bd3}, false, -1.0}},
                  m.a9, d4));
    FBF_TRY(sweep({{"pb,qrac->pqrabc", {delta, hdm2_op}, {nullptr, bd2}, false, -1.0},
                   {"br,pqac->pqrabc", {delta, hdm2_op}, {nullptr, bd2}, false, -1.0},
                   {"bq,prac->pqrabc", {delta, hdm2_op}, {nullptr, bd2}, false, 2.0},
                   {"ap,bqcr->pqrabc", {delta, dm2}, {nullptr, b2}, false, 2.0},
                   {"ap,cr,bq->pqrabc", {delta, delta, dm1}, {nullptr, nullptr, b1}, false, -4.0},
                   {"cr,bqap->pqrabc", {delta, dm2}, {nullptr, b2}, false, 2.0},
                   {"bqapcr->pqrabc", {dm3}, {b3}, false, -1.0},
                   {"ar,pc,bq->pqrabc", {delta, delta, dm1}, {nullptr, nullptr, b1}, false, 2.0},
                   {"ar,bqcp->pqrabc", {delta, dm2}, {nullptr, b2}, false, -1.0}},
                  hdm3_bar, d6));
  }
  {
    Buffers::device<double> dm2b_bar = FBF_TRY(device_zeros(res, n * n * n * n));
    // dm2b's value is never read: its term's one operand is the one with a bar.
    const EinsumOperand dm2b_op{.data = nullptr, .dims = d4};
    FBF_TRY(sweep({{"klij->ijkl", {dm2b_op}, {dm2b_bar.data()}, false, 1.0},
                   {"il,kj->ijkl", {delta, dm1}, {nullptr, b1}, false, 1.0},
                   {"jk,li->ijkl", {delta, dm1}, {nullptr, b1}, false, 1.0},
                   {"ik,lj->ijkl", {delta, dm1}, {nullptr, b1}, false, -2.0},
                   {"jl,ki->ijkl", {delta, dm1}, {nullptr, b1}, false, -2.0},
                   {"il,jk->ijkl", {delta, delta}, {nullptr, nullptr}, false, -2.0},
                   {"ik,jl->ijkl", {delta, delta}, {nullptr, nullptr}, false, 4.0}},
                  m.hdm2, d4));
    FBF_TRY(sweep({{"ikjl->ijkl", {dm2}, {b2}, false, 1.0},
                   {"jk,il->ijkl", {delta, dm1}, {nullptr, b1}, false, -1.0}},
                  dm2b_bar, d4));
  }

  // Sir: a12, a13, dm2, dm1.
  FBF_TRY(sweep({{"ia,qpib->pqab", {h1e, dm2}, {bh1, b2}, false, 1.0},
                 {"bi,qpai->pqab", {h1e, dm2}, {bh1, b2}, false, -1.0},
                 {"ijka,qpjbik->pqab", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"kbij,qpajki->pqab", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"bjka,qpjk->pqab", {h2e, dm2}, {bh2, b2}, false, -1.0},
                 {"jbij,qpai->pqab", {h2e, dm2}, {bh2, b2}, false, 1.0}},
                m.a12, d4));
  FBF_TRY(sweep({{"ia,qbip->pqab", {h1e, dm2}, {bh1, b2}, false, -1.0},
                 {"pa,qb->pqab", {h1e, dm1}, {bh1, b1}, false, 2.0},
                 {"bi,qiap->pqab", {h1e, dm2}, {bh1, b2}, false, 1.0},
                 {"pa,bi,qi->pqab", {delta, h1e, dm1}, {nullptr, bh1, b1}, false, -2.0},
                 {"ijka,qbjpik->pqab", {h2e, dm3}, {bh2, b3}, false, -1.0},
                 {"kbij,qjapki->pqab", {h2e, dm3}, {bh2, b3}, false, 1.0},
                 {"blma,qmlp->pqab", {h2e, dm2}, {bh2, b2}, false, 1.0},
                 {"kpma,qbkm->pqab", {h2e, dm2}, {bh2, b2}, false, 2.0},
                 {"bpma,qm->pqab", {h2e, dm1}, {bh2, b1}, false, -2.0},
                 {"lbkl,qkap->pqab", {h2e, dm2}, {bh2, b2}, false, -1.0},
                 {"ap,mbkl,qlmk->pqab", {delta, h2e, dm2}, {nullptr, bh2, b2}, false, -2.0},
                 {"ap,lbkl,qk->pqab", {delta, h2e, dm1}, {nullptr, bh2, b1}, false, 2.0}},
                m.a13, d4));
  FBF_TRY(sweep({identity(dm2, b2)}, m.sir_dm2, d4));
  FBF_TRY(sweep({identity(dm1, b1)}, m.sir_dm1, d2));
  return {};
}

Result<ClassEnergiesAdjoint> class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks,
                                                    const EnergyInputs &in, const int32_t batch,
                                                    const EnergyBars &bars) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const ActiveOperands ops = FBF_TRY(active_operands(res, blocks, in));
  std::vector<double> e_core_bar(ops.e_core.size(), 0.0);
  std::vector<double> e_virt_bar(ops.e_virt.size(), 0.0);
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = bars,
      .batch = batch,
      .ops = ops,
      .e = {
          .core = ops.e_core, .virt = ops.e_virt, .core_bar = e_core_bar, .virt_bar = e_virt_bar}};
  const int64_t n = blocks.ncas;
  // A zeroed n^rank block bar.
  const auto zeros = [&](const int32_t rank) {
    int64_t count = 1;
    for (int32_t k = 0; k < rank; ++k) {
      count *= n;
    }
    return device_zeros(res, count);
  };
  BlockBars m{.a16 = FBF_TRY(zeros(6)),
              .a17 = FBF_TRY(zeros(4)),
              .a19 = FBF_TRY(zeros(2)),
              .sr_dm3 = FBF_TRY(zeros(6)),
              .sr_dm2 = FBF_TRY(zeros(4)),
              .sr_dm1 = FBF_TRY(zeros(2)),
              .a22 = FBF_TRY(zeros(6)),
              .a23 = FBF_TRY(zeros(4)),
              .a25 = FBF_TRY(zeros(2)),
              .dm3_h = FBF_TRY(zeros(6)),
              .dm2_h = FBF_TRY(zeros(4)),
              .dm1_h = FBF_TRY(zeros(2)),
              .hdm1 = FBF_TRY(zeros(2)),
              .a3 = FBF_TRY(zeros(2)),
              .srsi_dm1 = FBF_TRY(zeros(2)),
              .k27 = FBF_TRY(zeros(2)),
              .rm2 = FBF_TRY(zeros(4)),
              .a7 = FBF_TRY(zeros(4)),
              .hdm2 = FBF_TRY(zeros(4)),
              .a9 = FBF_TRY(zeros(4)),
              .sir_dm2 = FBF_TRY(zeros(4)),
              .sir_dm1 = FBF_TRY(zeros(2)),
              .a12 = FBF_TRY(zeros(4)),
              .a13 = FBF_TRY(zeros(4))};
  std::array<ClassSums, 8> sums;
  sums[0] = FBF_TRY(reverse_sr(c, m));
  sums[1] = FBF_TRY(reverse_si(c, m));
  sums[2] = FBF_TRY(reverse_sijrs(c));
  sums[3] = FBF_TRY(reverse_sijr(c, m));
  sums[4] = FBF_TRY(reverse_srsi(c, m));
  sums[5] = FBF_TRY(reverse_srs(c, m));
  sums[6] = FBF_TRY(reverse_sij(c, m));
  sums[7] = FBF_TRY(reverse_sir(c, m));

  // R's host sums onto the orbital energies' bars (Sijrs's are already there).
  for (const auto &[host, bar] :
       {std::pair{&e_core_bar, bars.e_core}, std::pair{&e_virt_bar, bars.e_virt}}) {
    if (bar != nullptr && !host->empty()) {
      const Buffers::device<double> d = FBF_TRY(upload(res, *host));
      FBF_TRY(einsum(res.stream(), "i->i", {{.data = d.data(), .dims = {std::ssize(*host)}}}, bar,
                     {std::ssize(*host)}, 1.0));
      FBF_TRY(res.sync());
    }
  }

  ClassEnergiesAdjoint out;
  for (size_t k = 0; k < 8; ++k) {
    out.energies.norms[k] = sums[k].norm;
    out.energies.energies[k] = sums[k].energy;
    out.eq40[k] = sums[k].eq40;
  }
  out.energies.total = std::ranges::fold_left(out.energies.energies, 0.0, std::plus{});

  const std::array wanted{bars.h1e, bars.h2e, bars.dm1, bars.dm2, bars.dm3, bars.f3ca, bars.f3ac};
  if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
    FBF_TRY(sweep_blocks(res, ops, m, bars));
  }
  return out;
}

// ---------------------------------------------------------------------------
// The PC reverse pass: the solve's adjoint
// ---------------------------------------------------------------------------
//
// A PC tuple's energy is -c bᵀ A⁺ b, A = K + Δ S, b = S x + S₁ h1, in the
// eigenbasis U of the class's solve -c Σ_k y_k² / (λ_k + Δ), y = Uᵀ b =
// Tᵀ x + t1ᵀ h1. With g = y / (λ + Δ), the solve the energy ran, dA⁺ =
// -A⁺ dA A⁺ gives dE = <ybar, dy> + <M, Uᵀ dA U> at fixed U, no eigenvector
// or eigenvalue derivative (exact where S's kept rank is locally constant
// and K annihilates S's dropped modes). So, summed over the tuples:
//   dE/dT = Σ x ybarᵀ, dE/dt1 = Σ h1 ybarᵀ  (the slab einsums' adjoints),
//   dE/dK = U M Uᵀ, dE/dS = U (dE/dT)ᵀ + U M_Δ Uᵀ, dE/dS₁ = U (dE/dt1)ᵀ,
// each symmetrized where the solve symmetrized its matrix, and dE/dΔ onto the
// orbital energies. The bars on S, K and S₁ then go back through the einsums
// that built them onto SC's block bars, which sweep_blocks takes to the RDMs.

// One PC class's walk-long accumulators: the tuples' weights in the
// eigenbasis, M = Σ_t g_t w_tᵀ and M_Δ = Σ_t Δ_t g_t w_tᵀ (m x m, row-major),
// and the bars on T (d x m, T's layout) and on t1 (n1 x m; empty when the
// class has no S₁).
struct PcWalkBars {
  Buffers::device<double> mgg, mgg_delta, t, t1;
};

Result<PcWalkBars> pc_walk_bars(DeviceResources &res, const int64_t m, const int64_t d,
                                const int64_t n1) {
  return PcWalkBars{.mgg = FBF_TRY(device_zeros(res, m * m)),
                    .mgg_delta = FBF_TRY(device_zeros(res, m * m)),
                    .t = FBF_TRY(device_zeros(res, d * m)),
                    .t1 = FBF_TRY(device_zeros(res, n1 * m))};
}

// One slab's PC finish and its adjoint, row t of y (m wide) a tuple, its
// Perturber read as: t2 < 0 a single coupling with factor s, t2 >= 0 the
// pair 2 direct - exchange with its exchange row t2 of x (Sijr, Srsi); Δ from
// the orbital energies as finish_adjoint's. With g = y / (λ + Δ) and
// g_x = x / (λ + Δ):
//   single: E -= s g.y,          ybar = -2 s g,           w = s g,
//           R = s g.g;
//   pair:   E -= 2 g.y - g.x,    ybar = -(4 g - g_x), xbar = g,
//           w = 2 g - g_x,       R = 2 g.g - g.g_x;
// R = dE/dΔ onto the orbital energies, as finish_adjoint puts it.
struct PcRows {
  std::vector<double> ybar, xbar, g, w, delta;
  double energy = 0.0;
  double min_denominator = std::numeric_limits<double>::infinity();
};

PcRows pc_finish_adjoint(const std::vector<Perturber> &rows, const std::vector<double> &y,
                         const std::vector<double> &x, const std::vector<double> &lambda,
                         const OrbitalEnergies &e) {
  const size_t m = lambda.size();
  PcRows out;
  out.ybar.assign(y.size(), 0.0);
  out.xbar.assign(x.size(), 0.0);
  out.g.assign(y.size(), 0.0);
  out.w.assign(y.size(), 0.0);
  out.delta.assign(rows.size(), 0.0);
  for (size_t t = 0; t < rows.size(); ++t) {
    const Perturber &p = rows[t];
    double diff = 0.0;
    for (const int32_t v : {p.v1, p.v2}) {
      diff += v < 0 ? 0.0 : e.virt[static_cast<size_t>(v)];
    }
    for (const int32_t c : {p.c1, p.c2}) {
      diff -= c < 0 ? 0.0 : e.core[static_cast<size_t>(c)];
    }
    out.delta[t] = diff;
    const auto row = static_cast<size_t>(p.t1) * m;
    const auto xrow = static_cast<size_t>(p.t2) * m;
    double r = 0.0;
    for (size_t k = 0; k < m; ++k) {
      const double den = lambda[k] + diff;
      out.min_denominator = std::min(out.min_denominator, den);
      const double g = y[row + k] / den;
      out.g[row + k] = g;
      if (p.t2 < 0) {
        out.energy -= p.s * g * y[row + k];
        out.ybar[row + k] += -2.0 * p.s * g;
        out.w[row + k] = p.s * g;
        r += p.s * g * g;
      } else {
        const double gx = x[xrow + k] / den;
        out.energy -= (2.0 * g * y[row + k]) - (g * x[xrow + k]);
        out.ybar[row + k] += -((4.0 * g) - gx);
        out.xbar[xrow + k] += g;
        out.w[row + k] = (2.0 * g) - gx;
        r += (2.0 * g * g) - (g * gx);
      }
    }
    for (const int32_t v : {p.v1, p.v2}) {
      if (v >= 0) {
        e.virt_bar[static_cast<size_t>(v)] += r;
      }
    }
    for (const int32_t c : {p.c1, p.c2}) {
      if (c >= 0) {
        e.core_bar[static_cast<size_t>(c)] -= r;
      }
    }
  }
  return out;
}

// One slab of a PC class: @p terms evaluated over @p out_dims (the tuple
// axes, then m), into y, or into the exchange rows x for a term with h set
// (Sijr's swapped core pair, Srsi's SrsiT slab); read back and finished; the
// weights added to @p w; and every term's adjoint taken against ybar, or
// xbar. Returns {Σ E, the smallest denominator}.
Result<std::pair<double, double>>
pc_slab_reverse(DeviceResources &res, const std::vector<ClassTerm> &terms,
                const std::vector<int64_t> &out_dims, const std::vector<Perturber> &rows,
                const std::vector<double> &lambda, const OrbitalEnergies &e,
                const EinsumOperand &delta, PcWalkBars &w) {
  const wwr::wwrStream_t s = res.stream();
  const int64_t count = std::ranges::fold_left(out_dims, int64_t{1}, std::multiplies{});
  const auto m = std::ssize(lambda);
  const int64_t nt = count / m;
  const bool exchange = std::ranges::any_of(terms, [](const ClassTerm &t) { return t.h; });
  Buffers::device<double> y_d = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> x_d = FBF_TRY(device_zeros(res, exchange ? count : 0));
  FBF_TRY(eval_terms(res, terms, y_d.data(), x_d.data(), out_dims));
  const std::vector<double> y = FBF_TRY(download(res, y_d.data(), count));
  const std::vector<double> x = FBF_TRY(download(res, x_d.data(), exchange ? count : 0));
  const PcRows f = pc_finish_adjoint(rows, y, x, lambda, e);

  const Buffers::device<double> g = FBF_TRY(upload(res, f.g));
  const Buffers::device<double> wt = FBF_TRY(upload(res, f.w));
  const Buffers::device<double> dt = FBF_TRY(upload(res, f.delta));
  const EinsumOperand g_op{.data = g.data(), .dims = {nt, m}};
  const EinsumOperand w_op{.data = wt.data(), .dims = {nt, m}};
  FBF_TRY(einsum(s, "tk,tl->kl", {g_op, w_op}, w.mgg.data(), {m, m}, 1.0));
  FBF_TRY(einsum(s, "t,tk,tl->kl", {{.data = dt.data(), .dims = {nt}}, g_op, w_op},
                 w.mgg_delta.data(), {m, m}, 1.0));
  const Buffers::device<double> ybar = FBF_TRY(upload(res, f.ybar));
  const Buffers::device<double> xbar = FBF_TRY(upload(res, f.xbar));
  FBF_TRY(adjoint_terms(res, terms, ybar.data(), xbar.data(), out_dims, delta, nullptr));
  return std::pair{f.energy, f.min_denominator};
}

// The bars on a class's raw S (d x d), K (d x d) and S₁ (d x n1), each
// row-major as the class built it, from the walk's accumulators; @p w.t is
// spent. Column-major throughout, as the solve: M and M_Δ, row-major, read
// column-major as their transposes, which U Mᵀ's place in U (U Mᵀ)ᵀ =
// U M Uᵀ undoes.
struct PcMatrixBars {
  Buffers::device<double> s, k, s1;
};

Result<PcMatrixBars> pc_matrix_bars(DeviceResources &res, const ClassSolve &cs, PcWalkBars &w,
                                    const int64_t d, const int64_t n1) {
  const int64_t m = std::ssize(cs.lambda);
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  // dE/dS_sym = U (T̄ + U M_Δᵀ)ᵀ, dE/dK_sym = U (U Mᵀ)ᵀ.
  FBF_TRY(gpu_status(gemm(res.blas(), di, mi, mi, &kOne<double>, cs.u.data(), w.mgg_delta.data(),
                          &kOne<double>, w.t.data())));
  Buffers::device<double> z(static_cast<size_t>(d * m), res.device());
  FBF_TRY(gpu_status(gemm(res.blas(), di, mi, mi, &kOne<double>, cs.u.data(), w.mgg.data(),
                          &kZero<double>, z.data())));
  Buffers::device<double> s_sym(static_cast<size_t>(d * d), res.device());
  Buffers::device<double> k_sym(static_cast<size_t>(d * d), res.device());
  FBF_TRY(gpu_status(
      gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(res.blas(), di, di, mi, &kOne<double>, cs.u.data(),
                                                 w.t.data(), &kZero<double>, s_sym.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), di, di, mi, &kOne<double>, cs.u.data(), z.data(), &kZero<double>, k_sym.data())));
  // The solve symmetrized S and K, and the symmetrization is its own adjoint.
  PcMatrixBars out{.s = FBF_TRY(symmetrized(res, s_sym.data(), d)),
                   .k = FBF_TRY(symmetrized(res, k_sym.data(), d)),
                   .s1 = FBF_TRY(device_zeros(res, d * n1))};
  if (n1 > 0) {
    // U column-major d x m reads row-major as the m x d "ki", as in solve_class.
    FBF_TRY(einsum(res.stream(), "ki,ak->ia",
                   {{.data = cs.u.data(), .dims = {m, d}}, {.data = w.t1.data(), .dims = {n1, m}}},
                   out.s1.data(), {d, n1}, 1.0));
  }
  return out;
}

// @p kbar onto the block bar @p block, which is K itself (d x d).
Status add_onto(DeviceResources &res, const Buffers::device<double> &kbar, double *const block,
                const int64_t d) {
  return einsum(res.stream(), "ij->ij", {{.data = kbar.data(), .dims = {d, d}}}, block, {d, d},
                1.0);
}

// One PC class's energy, dropped count and homogeneity sum.
struct PcClassSums {
  double energy = 0.0;
  int64_t dropped = 0;
  double homogeneity = 0.0;
};

// Sr: pc_sr's solve and walk, its bars onto a16, dm3 and dm2 (SC's sr_dm3 and
// sr_dm2: Sr's S and S₁ are permutations of them).
Result<PcClassSums> pc_reverse_sr(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const wwr::wwrStream_t s = res.stream();
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = n * n * n;
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sr);
  if (extent == 0) {
    return PcClassSums{};
  }
  const Buffers::device<double> a16 = FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca));
  // S, S₁ and their builders are freed once solved.
  const auto solve = [&]() -> Result<ClassSolve> {
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "rpqbac->pqrabc", {o.dm3}, sm.data(), d6, 1.0));
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    FBF_TRY(einsum(s, "rpqa->pqra", {o.dm2}, s1.data(), d4, 1.0));
    return solve_class(res, "Sr", sm.data(), a16.data(), d, s1.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, d, n));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n, n, n}};
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {n, mm}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sr, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sr, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sr, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_sr + b0, .dims = {nb, n}, .strides = {1, nv}};
    double *const sb = slab_bar.data();
    double *const hb = c.bars.h1e_v_sr == nullptr ? nullptr : c.bars.h1e_v_sr + b0;
    const std::vector<ClassTerm> terms{{"ipqr,kpqr->ik", {h2e_v, t}, {sb, w.t.data()}},
                                       {"ia,ak->ik", {h1e_v, t1}, {hb, w.t1.data()}}};
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      rows.push_back({.t1 = i, .v1 = static_cast<int32_t>(b0 + i)});
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sr, b0, b1, sb));
  }
  FBF_TRY(pc_denominators_positive("Sr", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, d, n));
  FBF_TRY(adjoint_terms(res, {{"rpqbac->pqrabc", {o.dm3}, {m.sr_dm3.data()}}}, b.s.data(),
                        b.s.data(), d6, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res, {{"rpqa->pqra", {o.dm2}, {m.sr_dm2.data()}}}, b.s1.data(), b.s1.data(),
                        d4, o.delta, nullptr));
  FBF_TRY(add_onto(res, b.k, m.a16.data(), d));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity = FBF_TRY(
      eq40_sum(res, {{&m.sr_dm3, o.dm3.data}, {&m.sr_dm2, o.dm2.data}, {&m.a16, a16.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Si: pc_si's solve and walk, its bars onto a22, dm3_h and dm2_h.
Result<PcClassSums> pc_reverse_si(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const wwr::wwrStream_t s = res.stream();
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t d = n * n * n;
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Si);
  if (extent == 0) {
    return PcClassSums{};
  }
  const Buffers::device<double> a22 =
      FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca));
  const Buffers::device<double> dm3_h = FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta));
  const Buffers::device<double> dm2_h = FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta));
  const EinsumOperand dm3_h_op{.data = dm3_h.data(), .dims = d6};
  const EinsumOperand dm2_h_op{.data = dm2_h.data(), .dims = d4};
  // S, S₁ and their builders are freed once solved.
  const auto solve = [&]() -> Result<ClassSolve> {
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "rpqbac->pqrabc", {dm3_h_op}, sm.data(), d6, 1.0));
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d * n));
    FBF_TRY(einsum(s, "rpqa->pqra", {dm2_h_op}, s1.data(), d4, 1.0));
    return solve_class(res, "Si", sm.data(), a22.data(), d, s1.data(), n);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, d, n));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n, n, n}};
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {n, mm}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Si, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Si, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Si, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_si + (n * b0), .dims = {n, nb}, .strides = {1, n}};
    double *const sb = slab_bar.data();
    double *const hb = c.bars.h1e_v_si == nullptr ? nullptr : c.bars.h1e_v_si + (n * b0);
    const std::vector<ClassTerm> terms{{"qpir,kpqr->ik", {h2e_v, t}, {sb, w.t.data()}},
                                       {"ai,ak->ik", {h1e_v, t1}, {hb, w.t1.data()}}};
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      rows.push_back({.t1 = i, .c1 = static_cast<int32_t>(b0 + i)});
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Si, b0, b1, sb));
  }
  FBF_TRY(pc_denominators_positive("Si", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, d, n));
  FBF_TRY(adjoint_terms(res, {{"rpqbac->pqrabc", {dm3_h_op}, {m.dm3_h.data()}}}, b.s.data(),
                        b.s.data(), d6, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res, {{"rpqa->pqra", {dm2_h_op}, {m.dm2_h.data()}}}, b.s1.data(),
                        b.s1.data(), d4, o.delta, nullptr));
  FBF_TRY(add_onto(res, b.k, m.a22.data(), d));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity = FBF_TRY(eq40_sum(
      res, {{&m.dm3_h, dm3_h.data()}, {&m.dm2_h, dm2_h.data()}, {&m.a22, a22.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Sijr: pc_sijr's solve and walk, the exchange rows y[r, i, j] as x by the
// swapped einsum, its bars onto hdm1 and a3.
Result<PcClassSums> pc_reverse_sijr(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sijr);
  if (extent == 0 || nc == 0) {
    return PcClassSums{};
  }
  const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, o.dm1, o.delta));
  const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = {n, n}};
  const Buffers::device<double> a3 =
      FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta));
  const ClassSolve cs = FBF_TRY(solve_class(res, "Sijr", hdm1.data(), a3.data(), n));
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, n, 0));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sijr, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sijr, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{{"rpji,kp->rjik", {h2e_v, t}, {sb, w.t.data()}, false},
                                       {"rpij,kp->rjik", {h2e_v, t}, {sb, w.t.data()}, true}};
    std::vector<Perturber> rows;
    for (int64_t r = 0; r < nr; ++r) {
      for (int64_t j = 0; j < nc; ++j) {
        for (int64_t i = 0; i < nc; ++i) {
          const int64_t row = (((r * nc) + j) * nc) + i;
          rows.push_back({.t1 = row,
                          .t2 = row,
                          .v1 = static_cast<int32_t>(b0 + r),
                          .c1 = static_cast<int32_t>(j),
                          .c2 = static_cast<int32_t>(i)});
        }
      }
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nr, nc, nc, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sijr, b0, b1, sb));
  }
  FBF_TRY(pc_denominators_positive("Sijr", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, n, 0));
  FBF_TRY(add_onto(res, b.s, m.hdm1.data(), n));
  FBF_TRY(add_onto(res, b.k, m.a3.data(), n));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity =
      FBF_TRY(eq40_sum(res, {{&m.hdm1, hdm1.data()}, {&m.a3, a3.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Srsi: pc_srsi's solve and walk, the exchange rows from the SrsiT slab as x,
// its bars onto dm1 (SC's srsi_dm1) and k27.
Result<PcClassSums> pc_reverse_srsi(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srsi);
  if (extent == 0 || nc == 0) {
    return PcClassSums{};
  }
  const Buffers::device<double> k27 = FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2));
  const ClassSolve cs = FBF_TRY(solve_class(res, "Srsi", o.dm1.data, k27.data(), n));
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, n, 0));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_a = slab_buffer(res, c.blocks, ExtBlock::Srsi, step);
  Buffers::device<double> slab_b = slab_buffer(res, c.blocks, ExtBlock::SrsiT, step);
  Buffers::device<double> bar_a = slab_buffer(res, c.blocks, ExtBlock::Srsi, step);
  Buffers::device<double> bar_b = slab_buffer(res, c.blocks, ExtBlock::SrsiT, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    FBF_TRY(zero(res, bar_a.data(), slab_count(c.blocks, ExtBlock::Srsi, b0, b1)));
    FBF_TRY(zero(res, bar_b.data(), slab_count(c.blocks, ExtBlock::SrsiT, b0, b1)));
    // h[r_batch, s, i, p] and h[s, r_batch, i, p].
    const EinsumOperand h_a =
        column_major(slab_a.data(), slab_dims(c.blocks, ExtBlock::Srsi, b0, b1));
    const EinsumOperand h_b =
        column_major(slab_b.data(), slab_dims(c.blocks, ExtBlock::SrsiT, b0, b1));
    const std::vector<ClassTerm> terms{
        {"rsip,kp->rsik", {h_a, t}, {bar_a.data(), w.t.data()}, false},
        {"srip,kp->rsik", {h_b, t}, {bar_b.data(), w.t.data()}, true}};
    std::vector<Perturber> rows;
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        for (int64_t i = 0; i < nc; ++i) {
          const int64_t row = (((r * nv) + ss) * nc) + i;
          rows.push_back({.t1 = row,
                          .t2 = row,
                          .v1 = static_cast<int32_t>(b0 + r),
                          .v2 = static_cast<int32_t>(ss),
                          .c1 = static_cast<int32_t>(i)});
        }
      }
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, nv, nc, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srsi, b0, b1, bar_a.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::SrsiT, b0, b1, bar_b.data()));
  }
  FBF_TRY(pc_denominators_positive("Srsi", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, n, 0));
  FBF_TRY(add_onto(res, b.s, m.srsi_dm1.data(), n));
  FBF_TRY(add_onto(res, b.k, m.k27.data(), n));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity =
      FBF_TRY(eq40_sum(res, {{&m.srsi_dm1, o.dm1.data}, {&m.k27, k27.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Srs: pc_srs's solve and walk, its bars onto rm2 and a7.
Result<PcClassSums> pc_reverse_srs(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = n * n;
  const std::vector<int64_t> d4(4, n);
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srs);
  if (extent == 0) {
    return PcClassSums{};
  }
  const auto [rm2, a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  const EinsumOperand rm2_op{.data = rm2.data(), .dims = d4};
  // S, S₁ and their builders are freed once solved.
  const auto solve = [&]() -> Result<ClassSolve> {
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(res.stream(), "pqba->pqab", {rm2_op}, sm.data(), d4, 1.0));
    return solve_class(res, "Srs", sm.data(), a7.data(), d);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, d, 0));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n, n}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Srs, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Srs, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{{"rsqp,kpq->rsk", {h2e_v, t}, {sb, w.t.data()}}};
    std::vector<Perturber> rows;
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        rows.push_back({.t1 = (r * nv) + ss,
                        .s = 0.5,
                        .v1 = static_cast<int32_t>(b0 + r),
                        .v2 = static_cast<int32_t>(ss)});
      }
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, nv, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srs, b0, b1, sb));
  }
  FBF_TRY(pc_denominators_positive("Srs", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, d, 0));
  FBF_TRY(adjoint_terms(res, {{"pqba->pqab", {rm2_op}, {m.rm2.data()}}}, b.s.data(), b.s.data(), d4,
                        o.delta, nullptr));
  FBF_TRY(add_onto(res, b.k, m.a7.data(), d));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity =
      FBF_TRY(eq40_sum(res, {{&m.rm2, rm2.data()}, {&m.a7, a7.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Sij: pc_sij's solve and walk, its bars onto hdm2 and a9.
Result<PcClassSums> pc_reverse_sij(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t d = n * n;
  const std::vector<int64_t> d4(4, n);
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sij);
  if (extent == 0) {
    return PcClassSums{};
  }
  const Buffers::device<double> hdm2 = FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta));
  const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
  // hdm3 is freed once a9 is built.
  const auto build_a9 = [&]() -> Result<Buffers::device<double>> {
    const Buffers::device<double> hdm3 =
        FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
    const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
    return make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op);
  };
  const Buffers::device<double> a9 = FBF_TRY(build_a9());
  const ClassSolve cs = FBF_TRY(solve_class(res, "Sij", hdm2.data(), a9.data(), d));
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, d, 0));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, n, n}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sij, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sij, b0, b1));
    double *const sb = slab_bar.data();
    const std::vector<ClassTerm> terms{{"qpij,kpq->ijk", {h2e_v, t}, {sb, w.t.data()}}};
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        rows.push_back({.t1 = (i * nc) + j,
                        .s = 0.5,
                        .c1 = static_cast<int32_t>(b0 + i),
                        .c2 = static_cast<int32_t>(j)});
      }
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, nc, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sij, b0, b1, sb));
  }
  FBF_TRY(pc_denominators_positive("Sij", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, d, 0));
  FBF_TRY(add_onto(res, b.s, m.hdm2.data(), d));
  FBF_TRY(add_onto(res, b.k, m.a9.data(), d));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity =
      FBF_TRY(eq40_sum(res, {{&m.hdm2, hdm2.data()}, {&m.a9, a9.data()}}, acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

// Sir: pc_sir's blocked solve and walk. T's two blocks are picked by a
// selector over T's leading block axis, so T stays one packed operand whose
// bar einsum_adjoint can write. Its bars onto dm1 and dm2 (SC's sir_dm1 and
// sir_dm2), a12 and a13.
Result<PcClassSums> pc_reverse_sir(ReverseContext &c, BlockBars &m) {
  DeviceResources &res = c.res;
  const wwr::wwrStream_t s = res.stream();
  const ActiveOperands &o = c.ops;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = 2 * n * n;
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> blocked{2, n, n, 2, n, n};
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sir1);
  if (extent == 0 || nc == 0) {
    return PcClassSums{};
  }
  // pc_sir's coefficient blocks, and the two selectors.
  const Buffers::device<double> off_diag = FBF_TRY(upload(res, {2.0, -1.0, -1.0, 0.0}));
  const Buffers::device<double> block22 = FBF_TRY(upload(res, {0.0, 0.0, 0.0, 1.0}));
  const Buffers::device<double> one_body = FBF_TRY(upload(res, {2.0, -1.0}));
  const Buffers::device<double> first = FBF_TRY(upload(res, {1.0, 0.0}));
  const Buffers::device<double> second = FBF_TRY(upload(res, {0.0, 1.0}));
  const EinsumOperand off_diag_op{.data = off_diag.data(), .dims = {2, 2}};
  const EinsumOperand block22_op{.data = block22.data(), .dims = {2, 2}};
  const EinsumOperand one_body_op{.data = one_body.data(), .dims = {2}};
  const Buffers::device<double> a12 = FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3));
  const Buffers::device<double> a13 =
      FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  const EinsumOperand a12_op{.data = a12.data(), .dims = d4};
  const EinsumOperand a13_op{.data = a13.data(), .dims = d4};
  // s22's value is never read: its bar's term reads only block22.
  const EinsumOperand s22_op{.dims = d4};
  // S, S₁ and their builders are freed once solved.
  const auto solve = [&]() -> Result<ClassSolve> {
    Buffers::device<double> sm = FBF_TRY(device_zeros(res, d * d));
    {
      Buffers::device<double> s22 = FBF_TRY(device_zeros(res, n * n * n * n));
      FBF_TRY(einsum(s, "pa,qb->pqab", {o.delta, o.dm1}, s22.data(), d4, 2.0));
      FBF_TRY(einsum(s, "qbap->pqab", {o.dm2}, s22.data(), d4, -1.0));
      FBF_TRY(einsum(s, "ab,qp->pqab", {o.delta, o.dm1}, s22.data(), d4, 1.0));
      FBF_TRY(einsum(s, "xy,qpab->xpqyab", {off_diag_op, o.dm2}, sm.data(), blocked, 1.0));
      FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, {.data = s22.data(), .dims = d4}},
                     sm.data(), blocked, 1.0));
    }
    Buffers::device<double> km = FBF_TRY(device_zeros(res, d * d));
    FBF_TRY(einsum(s, "xy,pqab->xpqyab", {off_diag_op, a12_op}, km.data(), blocked, 1.0));
    FBF_TRY(einsum(s, "xy,pqab->xpqyab", {block22_op, a13_op}, km.data(), blocked, 1.0));
    Buffers::device<double> s1 = FBF_TRY(device_zeros(res, d));
    FBF_TRY(einsum(s, "x,qp->xpq", {one_body_op, o.dm1}, s1.data(), {2, n, n}, 1.0));
    return solve_class(res, "Sir", sm.data(), km.data(), d, s1.data(), 1);
  };
  const ClassSolve cs = FBF_TRY(solve());
  const int64_t mm = std::ssize(cs.lambda);
  PcWalkBars w = FBF_TRY(pc_walk_bars(res, mm, d, 1));
  const EinsumOperand t{.data = cs.t.data(), .dims = {mm, 2, n, n}};
  const EinsumOperand t1{.data = cs.t1->data(), .dims = {mm}};
  const EinsumOperand first_op{.data = first.data(), .dims = {2}};
  const EinsumOperand second_op{.data = second.data(), .dims = {2}};

  double energy = 0.0;
  double min_den = std::numeric_limits<double>::infinity();
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  Buffers::device<double> bar_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> bar_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sir1, b0, b1, slab_1.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sir2, b0, b1, slab_2.data()));
    FBF_TRY(zero(res, bar_1.data(), slab_count(c.blocks, ExtBlock::Sir1, b0, b1)));
    FBF_TRY(zero(res, bar_2.data(), slab_count(c.blocks, ExtBlock::Sir2, b0, b1)));
    const EinsumOperand v1 =
        column_major(slab_1.data(), slab_dims(c.blocks, ExtBlock::Sir1, b0, b1));
    const EinsumOperand v2 =
        column_major(slab_2.data(), slab_dims(c.blocks, ExtBlock::Sir2, b0, b1));
    const EinsumOperand h1e_v{.data = c.in.h1e_v_sir + b0, .dims = {nb, nc}, .strides = {1, nv}};
    double *const hb = c.bars.h1e_v_sir == nullptr ? nullptr : c.bars.h1e_v_sir + b0;
    const std::vector<ClassTerm> terms{
        {"rpiq,x,kxpq->rik", {v1, first_op, t}, {bar_1.data(), nullptr, w.t.data()}},
        {"rpqi,x,kxpq->rik", {v2, second_op, t}, {bar_2.data(), nullptr, w.t.data()}},
        {"ri,k->rik", {h1e_v, t1}, {hb, w.t1.data()}}};
    std::vector<Perturber> rows;
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t i = 0; i < nc; ++i) {
        rows.push_back({.t1 = (r * nc) + i,
                        .v1 = static_cast<int32_t>(b0 + r),
                        .c1 = static_cast<int32_t>(i)});
      }
    }
    const auto [e_part, den] =
        FBF_TRY(pc_slab_reverse(res, terms, {nb, nc, mm}, rows, cs.lambda, c.e, o.delta, w));
    energy += e_part;
    min_den = std::min(min_den, den);
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir1, b0, b1, bar_1.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir2, b0, b1, bar_2.data()));
  }
  FBF_TRY(pc_denominators_positive("Sir", min_den));

  const PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, cs, w, d, 1));
  Buffers::device<double> s22_bar = FBF_TRY(device_zeros(res, n * n * n * n));
  double *const md1 = m.sir_dm1.data();
  double *const md2 = m.sir_dm2.data();
  FBF_TRY(adjoint_terms(res,
                        {{"xy,qpab->xpqyab", {off_diag_op, o.dm2}, {nullptr, md2}},
                         {"xy,pqab->xpqyab", {block22_op, s22_op}, {nullptr, s22_bar.data()}}},
                        b.s.data(), b.s.data(), blocked, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res,
                        {{"pa,qb->pqab", {o.delta, o.dm1}, {nullptr, md1}, false, 2.0},
                         {"qbap->pqab", {o.dm2}, {md2}, false, -1.0},
                         {"ab,qp->pqab", {o.delta, o.dm1}, {nullptr, md1}, false, 1.0}},
                        s22_bar.data(), s22_bar.data(), d4, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res,
                        {{"xy,pqab->xpqyab", {off_diag_op, a12_op}, {nullptr, m.a12.data()}},
                         {"xy,pqab->xpqyab", {block22_op, a13_op}, {nullptr, m.a13.data()}}},
                        b.k.data(), b.k.data(), blocked, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res, {{"x,qp->xpq", {one_body_op, o.dm1}, {nullptr, md1}}}, b.s1.data(),
                        b.s1.data(), {2, n, n}, o.delta, nullptr));
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  const double homogeneity = FBF_TRY(eq40_sum(res,
                                              {{&m.sir_dm2, o.dm2.data},
                                               {&m.sir_dm1, o.dm1.data},
                                               {&m.a12, a12.data()},
                                               {&m.a13, a13.data()}},
                                              acc));
  return PcClassSums{.energy = energy, .dropped = cs.dropped, .homogeneity = homogeneity};
}

Result<PcClassEnergiesAdjoint>
pc_class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          const int32_t batch, const EnergyBars &bars) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const ActiveOperands ops = FBF_TRY(active_operands(res, blocks, in));
  std::vector<double> e_core_bar(ops.e_core.size(), 0.0);
  std::vector<double> e_virt_bar(ops.e_virt.size(), 0.0);
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = bars,
      .batch = batch,
      .ops = ops,
      .e = {
          .core = ops.e_core, .virt = ops.e_virt, .core_bar = e_core_bar, .virt_bar = e_virt_bar}};
  // SC's block bars, every one zeroed: PC adds onto those its S, K and S₁ are
  // built from, and the sweep reads them all (a17, a19, a23, a25 and dm1_h
  // stay zero).
  const int64_t n = blocks.ncas;
  const auto zeros = [&](const int32_t rank) {
    int64_t count = 1;
    for (int32_t k = 0; k < rank; ++k) {
      count *= n;
    }
    return device_zeros(res, count);
  };
  BlockBars m{.a16 = FBF_TRY(zeros(6)),
              .a17 = FBF_TRY(zeros(4)),
              .a19 = FBF_TRY(zeros(2)),
              .sr_dm3 = FBF_TRY(zeros(6)),
              .sr_dm2 = FBF_TRY(zeros(4)),
              .sr_dm1 = FBF_TRY(zeros(2)),
              .a22 = FBF_TRY(zeros(6)),
              .a23 = FBF_TRY(zeros(4)),
              .a25 = FBF_TRY(zeros(2)),
              .dm3_h = FBF_TRY(zeros(6)),
              .dm2_h = FBF_TRY(zeros(4)),
              .dm1_h = FBF_TRY(zeros(2)),
              .hdm1 = FBF_TRY(zeros(2)),
              .a3 = FBF_TRY(zeros(2)),
              .srsi_dm1 = FBF_TRY(zeros(2)),
              .k27 = FBF_TRY(zeros(2)),
              .rm2 = FBF_TRY(zeros(4)),
              .a7 = FBF_TRY(zeros(4)),
              .hdm2 = FBF_TRY(zeros(4)),
              .a9 = FBF_TRY(zeros(4)),
              .sir_dm2 = FBF_TRY(zeros(4)),
              .sir_dm1 = FBF_TRY(zeros(2)),
              .a12 = FBF_TRY(zeros(4)),
              .a13 = FBF_TRY(zeros(4))};

  PcClassEnergiesAdjoint out;
  const auto put = [&out](const size_t k, const PcClassSums &x) {
    out.energies.energies[k] = x.energy;
    out.energies.dropped[k] = x.dropped;
    out.homogeneity[k] = x.homogeneity;
  };
  put(0, FBF_TRY(pc_reverse_sr(c, m)));
  put(1, FBF_TRY(pc_reverse_si(c, m)));
  // Sijrs has no active index, so its PC energy, and its reverse pass, are SC's.
  const ClassSums sijrs = FBF_TRY(reverse_sijrs(c));
  put(2, PcClassSums{.energy = sijrs.energy, .homogeneity = sijrs.eq40});
  put(3, FBF_TRY(pc_reverse_sijr(c, m)));
  put(4, FBF_TRY(pc_reverse_srsi(c, m)));
  put(5, FBF_TRY(pc_reverse_srs(c, m)));
  put(6, FBF_TRY(pc_reverse_sij(c, m)));
  put(7, FBF_TRY(pc_reverse_sir(c, m)));
  out.energies.total = std::ranges::fold_left(out.energies.energies, 0.0, std::plus{});

  // dE/dΔ's host sums onto the orbital energies' bars (Sijrs's are already
  // there).
  for (const auto &[host, bar] :
       {std::pair{&e_core_bar, bars.e_core}, std::pair{&e_virt_bar, bars.e_virt}}) {
    if (bar != nullptr && !host->empty()) {
      const Buffers::device<double> d = FBF_TRY(upload(res, *host));
      FBF_TRY(einsum(res.stream(), "i->i", {{.data = d.data(), .dims = {std::ssize(*host)}}}, bar,
                     {std::ssize(*host)}, 1.0));
      FBF_TRY(res.sync());
    }
  }

  const std::array wanted{bars.h1e, bars.h2e, bars.dm1, bars.dm2, bars.dm3, bars.f3ca, bars.f3ac};
  if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
    FBF_TRY(sweep_blocks(res, ops, m, bars));
  }
  return out;
}

} // namespace free_breakfast::nevpt2
