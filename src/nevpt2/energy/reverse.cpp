/// @file
/// @brief The SC reverse pass, part A: from the class energies back to the
/// slabs (and through them onto Λ), h1e_v's rows and the orbital energies,
/// and onto every class block's bar M_G = dE/dG. einsum_adjoint is the one
/// derivative rule; the finish's P, Q, R (Park Eqs. 36-38) and Sijrs are the
/// only expressions differentiated by hand.

module;

// The Sijrs launchers, defined in nevpt2_energy.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "nevpt2_energy_bridge.h"

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, geam, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
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
// assembled from its own bars. The RDM sweep takes each back through its
// builder (make_*) onto the RDMs, h1e and h2e. class_bars allocates them all
// at once, zeroed, and each class adds onto its own.
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

// The whole walk: every class's sums, the orbital energies' bars added onto
// @p bars, and the block bars, which the RDM sweep will take from here.
Result<std::pair<ClassEnergiesAdjoint, BlockBars>>
class_bars(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
           const int32_t batch, const EnergyBars &bars) {
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
  return std::pair{std::move(out), std::move(m)};
}

Result<ClassEnergiesAdjoint> class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks,
                                                    const EnergyInputs &in, const int32_t batch,
                                                    const EnergyBars &bars) {
  const ScopedDeviceIndex on{res.dev_idx()};
  // The block bars are dropped here until the RDM sweep reads them.
  std::pair<ClassEnergiesAdjoint, BlockBars> walked =
      FBF_TRY(class_bars(res, blocks, in, batch, bars));
  return std::move(walked.first);
}

} // namespace free_breakfast::nevpt2
