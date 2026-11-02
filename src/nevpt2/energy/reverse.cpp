/// @file
/// @brief The SC and PC reverse passes, of one state and of n: from the class
/// energies back to the slabs (and through them onto Λ), h1e_v's rows and
/// the orbital energies, and onto every class block's bar M_G = dE/dG; then
/// the sweep of those bars back through the blocks' builders onto h1e, h2e and the RDMs.
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

// @p x's Δ over @p e.
double perturber_delta(const Perturber &x, const OrbitalEnergies &e) {
  double diff = 0.0;
  for (const int32_t v : {x.v1, x.v2}) {
    diff += v < 0 ? 0.0 : e.virt[static_cast<size_t>(v)];
  }
  for (const int32_t c : {x.c1, x.c2}) {
    diff -= c < 0 ? 0.0 : e.core[static_cast<size_t>(c)];
  }
  return diff;
}

// @p x's value in the table @p a: s (a[t1] + a[t2]), or s a[t1].
double combined(const Perturber &x, const std::vector<double> &a) {
  return x.s * (a[static_cast<size_t>(x.t1)] + (x.t2 < 0 ? 0.0 : a[static_cast<size_t>(x.t2)]));
}

// The bar @p m on @p x's value, back onto its entries of @p bar.
void spread(const Perturber &x, const double m, std::vector<double> &bar) {
  bar[static_cast<size_t>(x.t1)] += x.s * m;
  if (x.t2 >= 0) {
    bar[static_cast<size_t>(x.t2)] += x.s * m;
  }
}

// R = dE/dΔ of @p x onto the orbital energies its Δ reads.
void add_delta_bar(const Perturber &x, const double r, const OrbitalEnergies &e) {
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

// One norm (h = false) or H (h = true) term of a class: the forward einsum
// `subscripts` over `ops`, scaled by alpha, and the bar each operand's
// adjoint goes into (null: none). h picks the second of a walk's two tables,
// which is H in a state's own walk, the norm einsum of (N, M) in a pair's
// walk (ScWalk), and in PC's the exchange rows. block_free marks the one
// term that reads no class block (Sir's h1e_v-only norm term), whose share
// of Eq. 40 is taken against its multipliers directly.
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
// builder (make_*) onto the RDMs, h1e and h2e. block_bars allocates them all
// at once, zeroed, and each class adds onto its own.
//
// The Dyall blocks (a16, a17, a19, a22, a23, a25, a3, k27, a7, a9, a12, a13)
// are read for H alone, so only a state's own walk has them; the metrics
// (the rest) are what a pair's norm terms read too.
struct BlockBars {
  Buffers::device<double> a16, a17, a19, sr_dm3, sr_dm2, sr_dm1; // Sr
  Buffers::device<double> a22, a23, a25, dm3_h, dm2_h, dm1_h;    // Si
  Buffers::device<double> hdm1, a3;                              // Sijr
  Buffers::device<double> srsi_dm1, k27;                         // Srsi
  Buffers::device<double> rm2, a7;                               // Srs
  Buffers::device<double> hdm2, a9;                              // Sij
  Buffers::device<double> sir_dm2, sir_dm1, a12, a13;            // Sir
};

// The block bars over @p n active orbitals, zeroed: the Dyall blocks' when
// @p dyall, the metrics' when @p metrics, and every other one empty.
Result<BlockBars> block_bars(DeviceResources &res, const int64_t n, const bool dyall,
                             const bool metrics) {
  // A zeroed n^rank block bar, or an empty one when it is not @p wanted.
  const auto zeros = [&](const int32_t rank, const bool wanted) {
    int64_t count = wanted ? 1 : 0;
    for (int32_t k = 0; k < rank; ++k) {
      count *= n;
    }
    return device_zeros(res, count);
  };
  return BlockBars{.a16 = FBF_TRY(zeros(6, dyall)),
                   .a17 = FBF_TRY(zeros(4, dyall)),
                   .a19 = FBF_TRY(zeros(2, dyall)),
                   .sr_dm3 = FBF_TRY(zeros(6, metrics)),
                   .sr_dm2 = FBF_TRY(zeros(4, metrics)),
                   .sr_dm1 = FBF_TRY(zeros(2, metrics)),
                   .a22 = FBF_TRY(zeros(6, dyall)),
                   .a23 = FBF_TRY(zeros(4, dyall)),
                   .a25 = FBF_TRY(zeros(2, dyall)),
                   .dm3_h = FBF_TRY(zeros(6, metrics)),
                   .dm2_h = FBF_TRY(zeros(4, metrics)),
                   .dm1_h = FBF_TRY(zeros(2, metrics)),
                   .hdm1 = FBF_TRY(zeros(2, metrics)),
                   .a3 = FBF_TRY(zeros(2, dyall)),
                   .srsi_dm1 = FBF_TRY(zeros(2, metrics)),
                   .k27 = FBF_TRY(zeros(2, dyall)),
                   .rm2 = FBF_TRY(zeros(4, metrics)),
                   .a7 = FBF_TRY(zeros(4, dyall)),
                   .hdm2 = FBF_TRY(zeros(4, metrics)),
                   .a9 = FBF_TRY(zeros(4, dyall)),
                   .sir_dm2 = FBF_TRY(zeros(4, metrics)),
                   .sir_dm1 = FBF_TRY(zeros(2, metrics)),
                   .a12 = FBF_TRY(zeros(4, dyall)),
                   .a13 = FBF_TRY(zeros(4, dyall))};
}

// One class's sums: Σ N, Σ E and Eq. 40.
struct ClassSums {
  double norm = 0.0;
  double energy = 0.0;
  double eq40 = 0.0;
};

// What every class's reverse walk reads besides its own blocks: the fields
// every pair shares (@p in's h1e_v_* and the orbital energies) and the bars
// on them (@p bars' h1e_v_* and Λ). @p ops is PC's one state's operands; SC's
// walks read theirs from their ScWalk.
struct ReverseContext {
  DeviceResources &res;
  const MoBlocks &blocks;
  const EnergyInputs &in;
  const EnergyBars &bars;
  int32_t batch;
  const ActiveOperands &ops;
  OrbitalEnergies e;
};

// One walk of an SC class over its slabs. A state's own walk reads @p a, the
// state's operands: its norm terms into the first table, its H terms into
// the second, and their bars into @p m_a, Dyall blocks and metrics. A pair's
// walk reads the norm terms alone, of the transition operands (M, N) as @p a
// into the first table and (N, M) as @p b into the second, each with its
// metric bars. @p finish takes each slab's two tables, read back and packed
// over the class's table entries, and adds the multipliers onto their bars
// (@p a_bar, @p b_bar, zeroed); @p first is the walk-order index of the
// slab's first perturber (Srsi finishes once, at 0). With @p adjoint false
// the walk only evaluates and finishes.
struct ScWalk {
  const ActiveOperands *a = nullptr;
  BlockBars *m_a = nullptr;
  const ActiveOperands *b = nullptr;
  BlockBars *m_b = nullptr;
  bool adjoint = true;
  std::function<void(int64_t first, const std::vector<Perturber> &perturbers,
                     const std::vector<double> &a, const std::vector<double> &b,
                     std::vector<double> &a_bar, std::vector<double> &b_bar)>
      finish;
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

// One slab's whole step of @p w, for a class whose perturbers lie in one
// slab: the terms evaluated, read back and finished, and when the walk takes
// its adjoint the multipliers uploaded and every term's adjoint taken. The
// tables and multipliers are buffers of their own, held across
// einsum_adjoint's workspace.
Status slab_reverse(DeviceResources &res, const std::vector<ClassTerm> &terms,
                    const std::vector<int64_t> &out_dims, const std::vector<Perturber> &perturbers,
                    const ScWalk &w, const int64_t first, double *const eq40) {
  const int64_t count = std::ranges::fold_left(out_dims, int64_t{1}, std::multiplies{});
  Buffers::device<double> norm_d = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> h_d = FBF_TRY(device_zeros(res, count));
  FBF_TRY(eval_terms(res, terms, norm_d.data(), h_d.data(), out_dims));
  const std::vector<double> norm_t = FBF_TRY(download(res, norm_d.data(), count));
  const std::vector<double> h_t = FBF_TRY(download(res, h_d.data(), count));
  std::vector<double> p_t(static_cast<size_t>(count), 0.0);
  std::vector<double> q_t(static_cast<size_t>(count), 0.0);
  w.finish(first, perturbers, norm_t, h_t, p_t, q_t);
  if (!w.adjoint) {
    return {};
  }
  const Buffers::device<double> p = FBF_TRY(upload(res, p_t));
  const Buffers::device<double> q = FBF_TRY(upload(res, q_t));
  return adjoint_terms(res, terms, p.data(), q.data(), out_dims, w.a->delta, eq40);
}

// The device scalar @p acc, read back.
Result<double> scalar(DeviceResources &res, const Buffers::device<double> &acc) {
  const std::vector<double> host = FBF_TRY(download(res, acc.data(), 1));
  return host[0];
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
// The SC classes, each energy.cpp's walk with its terms listed once and both
// evaluated and differentiated from the list. Each takes one ScWalk, a
// state's own or a pair's, and returns its Eq. 40 share: its sums are the
// walk's finish's.
// ---------------------------------------------------------------------------

Result<double> reverse_sr(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  // The Dyall blocks, an own walk's alone.
  std::optional<Buffers::device<double>> a16;
  std::optional<Buffers::device<double>> a17;
  std::optional<Buffers::device<double>> a19;
  if (own) {
    a16 = FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca));
    a17 = FBF_TRY(make_a17(res, o.h1e, o.h2e, o.dm2, o.dm3));
    a19 = FBF_TRY(make_a19(res, o.h1e, o.h2e, o.dm1, o.dm2));
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sr);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sr, step);
  int64_t first = 0;
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
    std::vector<ClassTerm> terms;
    // One side's norm terms, into the second table when @p second.
    const auto norm_terms = [&](const ActiveOperands &x, BlockBars &m, const bool second) {
      terms.push_back(
          {"ipqr,rpqbac,iabc->i", {h2e_v, x.dm3, h2e_v}, {sb, m.sr_dm3.data(), sb}, second, 1.0});
      terms.push_back(
          {"ipqr,rpqa,ia->i", {h2e_v, x.dm2, h1e_v}, {sb, m.sr_dm2.data(), hb}, second, 2.0});
      terms.push_back(
          {"ip,pa,ia->i", {h1e_v, x.dm1, h1e_v}, {hb, m.sr_dm1.data(), hb}, second, 1.0});
    };
    if (own) {
      const EinsumOperand a16_op{.data = a16->data(), .dims = d6};
      const EinsumOperand a17_op{.data = a17->data(), .dims = d4};
      const EinsumOperand a19_op{.data = a19->data(), .dims = d2};
      BlockBars &m = *w.m_a;
      terms = {{"ipqr,pqrabc,iabc->i", {h2e_v, a16_op, h2e_v}, {sb, m.a16.data(), sb}, true, 1.0},
               {"ipqr,pqra,ia->i", {h2e_v, a17_op, h1e_v}, {sb, m.a17.data(), hb}, true, 2.0},
               {"ip,pa,ia->i", {h1e_v, a19_op, h1e_v}, {hb, m.a19.data(), hb}, true, 1.0}};
      norm_terms(o, m, false);
    } else {
      norm_terms(*w.a, *w.m_a, false);
      norm_terms(*w.b, *w.m_b, true);
    }
    std::vector<Perturber> perturbers;
    for (int64_t r = 0; r < nb; ++r) {
      perturbers.push_back({.t1 = r, .v1 = static_cast<int32_t>(b0 + r)});
    }
    FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sr, b0, b1, sb));
    }
  }
  if (w.adjoint) {
    // One side's metrics against their bars.
    const auto metrics = [&](const ActiveOperands &x, const BlockBars &m) -> Status {
      FBF_TRY(add_dot(res, m.sr_dm3, x.dm3.data, eq40.data()));
      FBF_TRY(add_dot(res, m.sr_dm2, x.dm2.data, eq40.data()));
      return add_dot(res, m.sr_dm1, x.dm1.data, eq40.data());
    };
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a16, a16->data(), eq40.data()));
      FBF_TRY(add_dot(res, w.m_a->a17, a17->data(), eq40.data()));
      FBF_TRY(add_dot(res, w.m_a->a19, a19->data(), eq40.data()));
    } else {
      FBF_TRY(metrics(*w.b, *w.m_b));
    }
    FBF_TRY(metrics(*w.a, *w.m_a));
  }
  return scalar(res, eq40);
}

// Si: its norm intermediates are built from each side's RDMs, dm1_h's 2δ
// only from a state's own (⟨I|J⟩ = 0 off it).
Result<double> reverse_si(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const double overlap = own ? 1.0 : 0.0;
  const int64_t n = c.blocks.ncas;
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  std::optional<Buffers::device<double>> a22;
  std::optional<Buffers::device<double>> a23;
  std::optional<Buffers::device<double>> a25;
  if (own) {
    a22 = FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca));
    a23 = FBF_TRY(make_a23(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3));
    a25 = FBF_TRY(make_a25(res, o.h1e, o.h2e, o.dm1, o.dm2));
  }
  // {dm3_h, dm2_h, dm1_h} of each side, a then b.
  std::vector<Buffers::device<double>> metric;
  for (const ActiveOperands *x : {w.a, w.b}) {
    if (x != nullptr) {
      metric.push_back(FBF_TRY(make_dm3_h(res, x->dm2, x->dm3, x->delta)));
      metric.push_back(FBF_TRY(make_dm2_h(res, x->dm1, x->dm2, x->delta)));
      metric.push_back(FBF_TRY(make_hdm1(res, x->dm1, x->delta, overlap)));
    }
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Si);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Si, step);
  int64_t first = 0;
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
    std::vector<ClassTerm> terms;
    if (own) {
      const EinsumOperand a22_op{.data = a22->data(), .dims = d6};
      const EinsumOperand a23_op{.data = a23->data(), .dims = d4};
      const EinsumOperand a25_op{.data = a25->data(), .dims = d2};
      BlockBars &m = *w.m_a;
      terms = {{"qpir,pqrabc,baic->i", {h2e_v, a22_op, h2e_v}, {sb, m.a22.data(), sb}, true, 1.0},
               {"qpir,pqra,ai->i", {h2e_v, a23_op, h1e_v}, {sb, m.a23.data(), hb}, true, 2.0},
               {"pi,pa,ai->i", {h1e_v, a25_op, h1e_v}, {hb, m.a25.data(), hb}, true, 1.0}};
    }
    // Side @p side's norm terms (0 a, 1 b), into the second table for b.
    for (const int32_t side : {0, 1}) {
      if (side == 1 && own) {
        break;
      }
      BlockBars &m = side == 0 ? *w.m_a : *w.m_b;
      const auto at = static_cast<size_t>(3 * side);
      const EinsumOperand dm3_h_op{.data = metric[at].data(), .dims = d6};
      const EinsumOperand dm2_h_op{.data = metric[at + 1].data(), .dims = d4};
      const EinsumOperand dm1_h_op{.data = metric[at + 2].data(), .dims = d2};
      const bool second = side == 1;
      terms.push_back(
          {"qpir,rpqbac,baic->i", {h2e_v, dm3_h_op, h2e_v}, {sb, m.dm3_h.data(), sb}, second, 1.0});
      terms.push_back(
          {"qpir,rpqa,ai->i", {h2e_v, dm2_h_op, h1e_v}, {sb, m.dm2_h.data(), hb}, second, 2.0});
      terms.push_back(
          {"pi,pa,ai->i", {h1e_v, dm1_h_op, h1e_v}, {hb, m.dm1_h.data(), hb}, second, 1.0});
    }
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nb; ++i) {
      perturbers.push_back({.t1 = i, .c1 = static_cast<int32_t>(b0 + i)});
    }
    FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Si, b0, b1, sb));
    }
  }
  if (w.adjoint) {
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a22, a22->data(), eq40.data()));
      FBF_TRY(add_dot(res, w.m_a->a23, a23->data(), eq40.data()));
      FBF_TRY(add_dot(res, w.m_a->a25, a25->data(), eq40.data()));
    }
    for (size_t side = 0; side < metric.size() / 3; ++side) {
      const BlockBars &m = side == 0 ? *w.m_a : *w.m_b;
      FBF_TRY(add_dot(res, m.dm3_h, metric[3 * side].data(), eq40.data()));
      FBF_TRY(add_dot(res, m.dm2_h, metric[(3 * side) + 1].data(), eq40.data()));
      FBF_TRY(add_dot(res, m.dm1_h, metric[(3 * side) + 2].data(), eq40.data()));
    }
  }
  return scalar(res, eq40);
}

// Sijrs by hand: it reads no block, so its slab's bar and its orbital
// energies' are sijrs_adjoint's, of @p weight E, and Eq. 40 is ½ <g-bar, g>
// (E is quadratic in g at fixed orbital energies), weight E.
Result<ClassSums> reverse_sijrs(ReverseContext &c, const double weight) {
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
        c.in.e_virt, slab_bar.data(), c.bars.e_core, c.bars.e_virt, weight)));
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
// (r, i, j). hdm1's 2δ is a state's own alone.
Result<double> reverse_sijr(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const std::vector<int64_t> d2{n, n};
  // hdm1 of each side, a then b.
  std::vector<Buffers::device<double>> hdm1;
  for (const ActiveOperands *x : {w.a, w.b}) {
    if (x != nullptr) {
      hdm1.push_back(FBF_TRY(make_hdm1(res, x->dm1, x->delta, own ? 1.0 : 0.0)));
    }
  }
  std::optional<Buffers::device<double>> a3;
  if (own) {
    const EinsumOperand hdm1_op{.data = hdm1.front().data(), .dims = d2};
    a3 = FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta));
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sijr);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sijr, step);
  int64_t first = 0;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nr = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sijr, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sijr, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sijr, b0, b1));
    double *const sb = slab_bar.data();
    std::vector<ClassTerm> terms;
    for (size_t side = 0; side < hdm1.size(); ++side) {
      const EinsumOperand hdm1_op{.data = hdm1[side].data(), .dims = d2};
      double *const mb = (side == 0 ? *w.m_a : *w.m_b).hdm1.data();
      terms.push_back({"rpji,raji,pa->rji", {h2e_v, h2e_v, hdm1_op}, {sb, sb, mb}, side == 1, 2.0});
      terms.push_back(
          {"rpji,raij,pa->rji", {h2e_v, h2e_v, hdm1_op}, {sb, sb, mb}, side == 1, -1.0});
    }
    if (own) {
      const EinsumOperand a3_op{.data = a3->data(), .dims = d2};
      double *const mb = w.m_a->a3.data();
      terms.push_back({"rpji,raji,pa->rji", {h2e_v, h2e_v, a3_op}, {sb, sb, mb}, true, 2.0});
      terms.push_back({"rpji,raij,pa->rji", {h2e_v, h2e_v, a3_op}, {sb, sb, mb}, true, -1.0});
    }
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
    FBF_TRY(slab_reverse(res, terms, {nr, nc, nc}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sijr, b0, b1, sb));
    }
  }
  if (w.adjoint) {
    for (size_t side = 0; side < hdm1.size(); ++side) {
      FBF_TRY(add_dot(res, (side == 0 ? *w.m_a : *w.m_b).hdm1, hdm1[side].data(), eq40.data()));
    }
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a3, a3->data(), eq40.data()));
    }
  }
  return scalar(res, eq40);
}

// Srsi: its (r, s) symmetrization pairs rows of different slabs, so the walk
// runs twice. The first fills the whole (v, v, c) tables, as energy.cpp's
// does, and the finish runs once over them; the second, when the walk takes
// its adjoint, rebuilds each slab pair and takes its adjoint against the
// multiplier tables' rows.
Result<double> reverse_srsi(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  std::optional<Buffers::device<double>> k27;
  if (own) {
    k27 = FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2));
  }
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
    std::vector<ClassTerm> out;
    const auto norm_terms = [&](const ActiveOperands &x, BlockBars &m, const bool second) {
      double *const md = m.srsi_dm1.data();
      out.push_back({"rsip,rsia,pa->rsi", {h_a, h_a, x.dm1}, {ba, ba, md}, second, 2.0});
      out.push_back({"rsip,sria,pa->rsi", {h_a, h_b, x.dm1}, {ba, bb, md}, second, -1.0});
    };
    norm_terms(o, *w.m_a, false);
    if (own) {
      const EinsumOperand k27_op{.data = k27->data(), .dims = {n, n}};
      double *const mk = w.m_a->k27.data();
      out.push_back({"rsip,rsia,pa->rsi", {h_a, h_a, k27_op}, {ba, ba, mk}, true, 2.0});
      out.push_back({"rsip,sria,pa->rsi", {h_a, h_b, k27_op}, {ba, bb, mk}, true, -1.0});
    } else {
      norm_terms(*w.b, *w.m_b, true);
    }
    return out;
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
  w.finish(0, perturbers, norm_t, h_t, p_t, q_t);
  if (!w.adjoint) {
    return 0.0;
  }
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
  FBF_TRY(add_dot(res, w.m_a->srsi_dm1, o.dm1.data, eq40.data()));
  if (own) {
    FBF_TRY(add_dot(res, w.m_a->k27, k27->data(), eq40.data()));
  } else {
    FBF_TRY(add_dot(res, w.m_b->srsi_dm1, w.b->dm1.data, eq40.data()));
  }
  return scalar(res, eq40);
}

// Srs: rm2 of each side; an own walk builds it with its a7, as energy.cpp's.
Result<double> reverse_srs(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  if (nv == 0) {
    return 0.0;
  }
  const std::vector<int64_t> d4(4, n);
  // rm2 of each side, a then b.
  std::vector<Buffers::device<double>> rm2;
  std::optional<Buffers::device<double>> a7;
  if (own) {
    auto [own_rm2, own_a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
    rm2.push_back(std::move(own_rm2));
    a7 = std::move(own_a7);
  } else {
    for (const ActiveOperands *x : {w.a, w.b}) {
      rm2.push_back(FBF_TRY(make_rm2(res, x->dm1, x->dm2, x->delta)));
    }
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srs);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Srs, step);
  int64_t first = 0;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srs, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Srs, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Srs, b0, b1));
    double *const sb = slab_bar.data();
    std::vector<ClassTerm> terms;
    for (size_t side = 0; side < rm2.size(); ++side) {
      const EinsumOperand rm2_op{.data = rm2[side].data(), .dims = d4};
      double *const mb = (side == 0 ? *w.m_a : *w.m_b).rm2.data();
      terms.push_back({"rsqp,rsba,pqba->rs", {h2e_v, h2e_v, rm2_op}, {sb, sb, mb}, side == 1, 0.5});
    }
    if (own) {
      const EinsumOperand a7_op{.data = a7->data(), .dims = d4};
      terms.push_back(
          {"rsqp,rsba,pqab->rs", {h2e_v, h2e_v, a7_op}, {sb, sb, w.m_a->a7.data()}, true, 0.5});
    }
    std::vector<Perturber> perturbers;
    for (int64_t r = 0; r < nb; ++r) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        perturbers.push_back({.t1 = (r * nv) + ss,
                              .v1 = static_cast<int32_t>(b0 + r),
                              .v2 = static_cast<int32_t>(ss)});
      }
    }
    FBF_TRY(slab_reverse(res, terms, {nb, nv}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srs, b0, b1, sb));
    }
  }
  if (w.adjoint) {
    for (size_t side = 0; side < rm2.size(); ++side) {
      FBF_TRY(add_dot(res, (side == 0 ? *w.m_a : *w.m_b).rm2, rm2[side].data(), eq40.data()));
    }
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a7, a7->data(), eq40.data()));
    }
  }
  return scalar(res, eq40);
}

// Sij: hdm2 of each side, its δδ terms a state's own alone.
Result<double> reverse_sij(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  if (nc == 0) {
    return 0.0;
  }
  const std::vector<int64_t> d4(4, n);
  // hdm2 of each side, a then b.
  std::vector<Buffers::device<double>> hdm2;
  for (const ActiveOperands *x : {w.a, w.b}) {
    if (x != nullptr) {
      hdm2.push_back(FBF_TRY(make_hdm2(res, x->dm1, x->dm2, x->delta, own ? 1.0 : 0.0)));
    }
  }
  std::optional<Buffers::device<double>> a9;
  if (own) {
    const EinsumOperand hdm2_op{.data = hdm2.front().data(), .dims = d4};
    const Buffers::device<double> hdm3 =
        FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
    const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
    a9 = FBF_TRY(make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op));
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sij);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  Buffers::device<double> slab_bar = slab_buffer(res, c.blocks, ExtBlock::Sij, step);
  int64_t first = 0;
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t nb = b1 - b0;
    FBF_TRY(slab(res, c.blocks, ExtBlock::Sij, b0, b1, slab_buf.data()));
    FBF_TRY(zero(res, slab_bar.data(), slab_count(c.blocks, ExtBlock::Sij, b0, b1)));
    const EinsumOperand h2e_v =
        column_major(slab_buf.data(), slab_dims(c.blocks, ExtBlock::Sij, b0, b1));
    double *const sb = slab_bar.data();
    std::vector<ClassTerm> terms;
    for (size_t side = 0; side < hdm2.size(); ++side) {
      const EinsumOperand hdm2_op{.data = hdm2[side].data(), .dims = d4};
      double *const mb = (side == 0 ? *w.m_a : *w.m_b).hdm2.data();
      terms.push_back(
          {"qpij,baij,pqab->ij", {h2e_v, h2e_v, hdm2_op}, {sb, sb, mb}, side == 1, 0.5});
    }
    if (own) {
      const EinsumOperand a9_op{.data = a9->data(), .dims = d4};
      terms.push_back(
          {"qpij,baij,pqab->ij", {h2e_v, h2e_v, a9_op}, {sb, sb, w.m_a->a9.data()}, true, 0.5});
    }
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nb; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        perturbers.push_back({.t1 = (i * nc) + j,
                              .c1 = static_cast<int32_t>(b0 + i),
                              .c2 = static_cast<int32_t>(j)});
      }
    }
    FBF_TRY(slab_reverse(res, terms, {nb, nc}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sij, b0, b1, sb));
    }
  }
  if (w.adjoint) {
    for (size_t side = 0; side < hdm2.size(); ++side) {
      FBF_TRY(add_dot(res, (side == 0 ? *w.m_a : *w.m_b).hdm2, hdm2[side].data(), eq40.data()));
    }
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a9, a9->data(), eq40.data()));
    }
  }
  return scalar(res, eq40);
}

// Sir: two slabs per step and h1e_v_Sir's rows; "raai" reads h2e_v2 on its
// diagonal, whose bar einsum_adjoint scatters against delta. Its
// h1e_v·h1e_v norm term is ⟨I|J⟩'s, so a pair's walk has none.
Result<double> reverse_sir(ReverseContext &c, const ScWalk &w) {
  DeviceResources &res = c.res;
  const ActiveOperands &o = *w.a;
  const bool own = w.b == nullptr;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const std::vector<int64_t> d4(4, n);
  std::optional<Buffers::device<double>> a12;
  std::optional<Buffers::device<double>> a13;
  if (own) {
    a12 = FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3));
    a13 = FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  }
  Buffers::device<double> eq40 = FBF_TRY(device_zeros(res, 1));

  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sir1);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> slab_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  Buffers::device<double> bar_1 = slab_buffer(res, c.blocks, ExtBlock::Sir1, step);
  Buffers::device<double> bar_2 = slab_buffer(res, c.blocks, ExtBlock::Sir2, step);
  int64_t first = 0;
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
    std::vector<ClassTerm> terms;
    const auto norm_terms = [&](const ActiveOperands &x, BlockBars &m, const bool second) {
      double *const md2 = m.sir_dm2.data();
      double *const md1 = m.sir_dm1.data();
      terms.push_back({"rpiq,raib,qpab->ir", {v1, v1, x.dm2}, {s1, s1, md2}, second, 2.0});
      terms.push_back({"rpiq,rabi,qpab->ir", {v1, v2, x.dm2}, {s1, s2, md2}, second, -1.0});
      terms.push_back({"rpqi,raib,qpab->ir", {v2, v1, x.dm2}, {s2, s1, md2}, second, -1.0});
      terms.push_back({"raqi,rabi,qb->ir", {v2, v2, x.dm1}, {s2, s2, md1}, second, 2.0});
      terms.push_back({"rpqi,rabi,qbap->ir", {v2, v2, x.dm2}, {s2, s2, md2}, second, -1.0});
      terms.push_back({"rpqi,raai,qp->ir", {v2, v2, x.dm1}, {s2, s2, md1}, second, 1.0});
      terms.push_back({"rpiq,ri,qp->ir", {v1, h1e_v, x.dm1}, {s1, hb, md1}, second, 4.0});
      terms.push_back({"rpqi,ri,qp->ir", {v2, h1e_v, x.dm1}, {s2, hb, md1}, second, -2.0});
    };
    norm_terms(o, *w.m_a, false);
    if (own) {
      BlockBars &m = *w.m_a;
      const EinsumOperand a12_op{.data = a12->data(), .dims = d4};
      const EinsumOperand a13_op{.data = a13->data(), .dims = d4};
      terms.push_back({"ri,ri->ir", {h1e_v, h1e_v}, {hb, hb}, false, 2.0, true});
      terms.push_back({"rpiq,raib,pqab->ir", {v1, v1, a12_op}, {s1, s1, m.a12.data()}, true, 2.0});
      terms.push_back({"rpiq,rabi,pqab->ir", {v1, v2, a12_op}, {s1, s2, m.a12.data()}, true, -1.0});
      terms.push_back({"rpqi,raib,pqab->ir", {v2, v1, a12_op}, {s2, s1, m.a12.data()}, true, -1.0});
      terms.push_back({"rpqi,rabi,pqab->ir", {v2, v2, a13_op}, {s2, s2, m.a13.data()}, true, 1.0});
    } else {
      norm_terms(*w.b, *w.m_b, true);
    }
    std::vector<Perturber> perturbers;
    for (int64_t i = 0; i < nc; ++i) {
      for (int64_t r = 0; r < nb; ++r) {
        perturbers.push_back({.t1 = (i * nb) + r,
                              .v1 = static_cast<int32_t>(b0 + r),
                              .c1 = static_cast<int32_t>(i)});
      }
    }
    FBF_TRY(slab_reverse(res, terms, {nc, nb}, perturbers, w, first, eq40.data()));
    first += std::ssize(perturbers);
    if (w.adjoint) {
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir1, b0, b1, s1));
      FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir2, b0, b1, s2));
    }
  }
  if (w.adjoint) {
    FBF_TRY(add_dot(res, w.m_a->sir_dm2, o.dm2.data, eq40.data()));
    FBF_TRY(add_dot(res, w.m_a->sir_dm1, o.dm1.data, eq40.data()));
    if (own) {
      FBF_TRY(add_dot(res, w.m_a->a12, a12->data(), eq40.data()));
      FBF_TRY(add_dot(res, w.m_a->a13, a13->data(), eq40.data()));
    } else {
      FBF_TRY(add_dot(res, w.m_b->sir_dm2, w.b->dm2.data, eq40.data()));
      FBF_TRY(add_dot(res, w.m_b->sir_dm1, w.b->dm1.data, eq40.data()));
    }
  }
  return scalar(res, eq40);
}

// ---------------------------------------------------------------------------
// The sweep: the block bars back through their builders
// ---------------------------------------------------------------------------

// The Dyall blocks' bars (a16, a17, a19, a22, a23, a25, a3, k27, a7, a9,
// a12, a13) back through their make_* onto @p bars' h1e, h2e and RDM fields.
// Every builder is listed term by term with energy.cpp's subscripts,
// coefficients and operands (a transpose as its one-operand einsum, NumPy's
// axes spelled as labels: axes {1, 4, 0, 2, 5, 3} of "abcdef" is
// "->beacfd"), and einsum_adjoint differentiates each. A term whose bars are
// all null (delta alone) is listed anyway, so a builder still reads against
// its forward line by line.
//
// Reverse topological: an intermediate's bar is complete before it is swept
// onto what built it. Sij's a9 reads hdm3, which reads hdm2 (a9, hdm3, then
// sweep_metrics' hdm2 and its dm2b); Srs's a7 reads rm3, which reads rm2
// (a7, rm3, then rm2); Sijr's a3 reads hdm1 (a3, then hdm1); and a16's and
// a22's fdm2, and a17's and a19's h1e - Σ (mj|jn), are swept within their
// builders. A block that is a class metric too (hdm1, rm2, hdm2) is @p m's
// own bar, which this sweep adds to, so sweep_metrics runs after it.
//
// The intermediates a bar is taken against (hdm1, hdm2, hdm3, rm2, rm3, the
// modified h1e) are rebuilt with the forward's own builders, not kept from
// the walk, from a state's own RDMs: only a state's own walk has Dyall
// blocks. An operand whose own bar is the only one its term takes is never
// read (einsum_adjoint reads every operand but k for k's bar), so fdm2 is
// not rebuilt: its operand is a shape. Every intermediate bar is a buffer of
// its own, held across einsum_adjoint's workspace.
Status sweep_dyall(DeviceResources &res, const ActiveOperands &o, BlockBars &m,
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
  // Sr: a16, its fdm2 (gathered off the diagonal diag_a16 wrote,
  // a16[p, i, r, a, b, i] += fdm2[p, r, a, b]), a17, a19.
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

  // Si: a22, its fdm2a (diag_a22a: a22[x, i, k, i, b, c] -= fdm2a[x, k, b, c])
  // and fdm2b (diag_a22b: a22[x, i, k, a, i, c] += 2 fdm2b[x, k, a, c]), a23,
  // a25.
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

  // Sijr: a3, onto hdm1's bar among others.
  {
    const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, dm1, delta));
    const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = d2};
    FBF_TRY(sweep({{"ia,ip->pa", {h1e, hdm1_op}, {bh1, m.hdm1.data()}, false, 1.0},
                   {"ijka,pj,ik->pa", {h2e, delta, dm1}, {bh2, nullptr, b1}, false, 2.0},
                   {"ijka,jpik->pa", {h2e, dm2}, {bh2, b2}, false, -1.0}},
                  m.a3, d2));
  }

  // Srsi: k27.
  FBF_TRY(sweep({{"ai,pi->pa", {h1e, dm1}, {bh1, b1}, false, -1.0},
                 {"iajk,pkij->pa", {h2e, dm2}, {bh2, b2}, false, -1.0},
                 {"iaji,pj->pa", {h2e, dm1}, {bh2, b1}, false, 1.0}},
                m.k27, d2));

  // Srs: a7, rm3. make_a7 is what builds rm2; its a7 is not read.
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

  // Sij: a9, hdm3.
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

  // Sir: a12, a13.
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
  return {};
}

// The metrics' bars (Sr's dm1..dm3, dm3_h, dm2_h, dm1_h, hdm1, Srsi's dm1,
// rm2, hdm2, Sir's dm1 and dm2) back through their make_* onto @p bars' RDM
// fields, as sweep_dyall does the Dyall blocks', and after it: its hdm1,
// rm2 and hdm2 bars are complete only then. They read no integral, so a
// pair's transition RDMs take theirs here alone, @p o then the pair's
// operands. dm2b is not rebuilt: its term's one operand is the one with a
// bar. @p m is spent afterwards.
Status sweep_metrics(DeviceResources &res, const ActiveOperands &o, BlockBars &m,
                     const EnergyBars &bars) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d2{n, n};
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const EinsumOperand &dm1 = o.dm1;
  const EinsumOperand &dm2 = o.dm2;
  const EinsumOperand &dm3 = o.dm3;
  const EinsumOperand &delta = o.delta;
  double *const b1 = bars.dm1;
  double *const b2 = bars.dm2;
  double *const b3 = bars.dm3;
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

  // Sr: dm1..dm3.
  FBF_TRY(sweep({identity(dm3, b3)}, m.sr_dm3, d6));
  FBF_TRY(sweep({identity(dm2, b2)}, m.sr_dm2, d4));
  FBF_TRY(sweep({identity(dm1, b1)}, m.sr_dm1, d2));

  // Si: dm3_h, dm2_h. dm1_h is make_hdm1, so its bar joins Sijr's hdm1's.
  FBF_TRY(sweep({{"abef,cd->abcdef", {dm2, delta}, {b2, nullptr}, false, 2.0},
                 {"abcdef->abdcef", {dm3}, {b3}, false, -1.0}},
                m.dm3_h, d6));
  FBF_TRY(sweep({{"ab,cd->abcd", {dm1, delta}, {b1, nullptr}, false, 2.0},
                 {"abcd->abdc", {dm2}, {b2}, false, -1.0}},
                m.dm2_h, d4));
  // dm1_h's value is not read: the identity's one operand is the one with a bar.
  FBF_TRY(sweep({identity({.dims = d2}, m.hdm1.data())}, m.dm1_h, d2));

  // Sijr: hdm1 (Si's dm1_h's bar, and sweep_dyall's a3's, already in it).
  FBF_TRY(sweep({{"ij->ij", {delta}, {nullptr}, false, 2.0}, {"ab->ba", {dm1}, {b1}, false, -1.0}},
                m.hdm1, d2));

  // Srsi: dm1.
  FBF_TRY(sweep({identity(dm1, b1)}, m.srsi_dm1, d2));

  // Srs: rm2 (sweep_dyall's a7's and rm3's bars already in it).
  FBF_TRY(sweep({{"iljk->ijkl", {dm2}, {b2}, false, 1.0},
                 {"ik,jl->ijkl", {dm1, delta}, {b1, nullptr}, false, -1.0}},
                m.rm2, d4));

  // Sij: hdm2 (sweep_dyall's a9's and hdm3's bars already in it), dm2b.
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

  // Sir: dm2, dm1.
  FBF_TRY(sweep({identity(dm2, b2)}, m.sir_dm2, d4));
  FBF_TRY(sweep({identity(dm1, b1)}, m.sir_dm1, d2));
  return {};
}

// Every block's bar M_G back through its builder: sweep_dyall, then
// sweep_metrics. @p m is spent afterwards.
Status sweep_blocks(DeviceResources &res, const ActiveOperands &o, BlockBars &m,
                    const EnergyBars &bars) {
  FBF_TRY(sweep_dyall(res, o, m, bars));
  return sweep_metrics(res, o, m, bars);
}

// ---------------------------------------------------------------------------
// The SC-QD reverse pass, and SC's at n = 1
// ---------------------------------------------------------------------------

// One state's own tables over one class's walk, perturber by perturber in
// walk order: its norm N^J and H^J, and A^J = Σ_{I != J} H̄_IJ T_IJ / D_J²,
// the bar on its denominator D_J = Δ + H^J / N^J that the pairs with ket J
// put there.
struct KetTables {
  std::vector<double> norm, h, a;
};

Result<ScEffectiveHamiltonianAdjoint> sc_couplings_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    const int64_t ns, std::span<const double> hbar, const int32_t batch,
    std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition) {
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<ActiveOperands> ops;
  for (const EnergyInputs &pair : pairs) {
    ops.push_back(FBF_TRY(active_operands(res, blocks, pair)));
  }
  const auto at = [ns](const int64_t i, const int64_t j) {
    return static_cast<size_t>((i * ns) + j);
  };
  std::vector<double> e_core_bar(ops.front().e_core.size(), 0.0);
  std::vector<double> e_virt_bar(ops.front().e_virt.size(), 0.0);
  // The fields every pair shares, and their bars, are the first's.
  ReverseContext c{.res = res,
                   .blocks = blocks,
                   .in = pairs.front(),
                   .bars = bars.front(),
                   .batch = batch,
                   .ops = ops.front(),
                   .e = {.core = ops.front().e_core,
                         .virt = ops.front().e_virt,
                         .core_bar = e_core_bar,
                         .virt_bar = e_virt_bar}};
  // Every class but Sijrs, in PySCF's order.
  const std::array<std::pair<size_t, Result<double> (*)(ReverseContext &, const ScWalk &)>, 7>
      walks{{{0, reverse_sr},
             {1, reverse_si},
             {3, reverse_sijr},
             {4, reverse_srsi},
             {5, reverse_srs},
             {6, reverse_sij},
             {7, reverse_sir}}};
  ScEffectiveHamiltonianAdjoint out;
  out.elements.resize(static_cast<size_t>(ns * ns));
  std::array<std::vector<KetTables>, 8> kets;
  for (std::vector<KetTables> &k : kets) {
    k.resize(static_cast<size_t>(ns));
  }

  // 1. Each state's own N and H, perturber by perturber: the denominators
  //    every pair with that ket reads. One state has no pair, so none.
  if (ns > 1) {
    BlockBars none = FBF_TRY(block_bars(res, blocks.ncas, false, false));
    for (int64_t j = 0; j < ns; ++j) {
      for (const auto &[k, walk] : walks) {
        KetTables &kt = kets[k][static_cast<size_t>(j)];
        const ScWalk w{.a = &ops[at(j, j)],
                       .m_a = &none,
                       .adjoint = false,
                       .finish = [&](int64_t, const std::vector<Perturber> &perturbers,
                                     const std::vector<double> &a, const std::vector<double> &b,
                                     std::vector<double> &, std::vector<double> &) {
                         for (const Perturber &x : perturbers) {
                           kt.norm.push_back(combined(x, a));
                           kt.h.push_back(combined(x, b));
                         }
                       }};
        FBF_TRY(walk(c, w));
        kt.a.assign(kt.norm.size(), 0.0);
      }
    }
  }

  // 2. Each unordered pair M < N: the 2 x 2 block's couplings (M, N) and
  //    (N, M), the bars on their shared T (onto the slabs, h1e_v's rows and
  //    the pair's metrics), and A on both kets. Its metrics' bars are then
  //    swept onto its transition RDMs, handed over, and freed.
  for (int64_t mi = 0; mi < ns; ++mi) {
    for (int64_t ni = mi + 1; ni < ns; ++ni) {
      BlockBars m_mn = FBF_TRY(block_bars(res, blocks.ncas, false, true));
      BlockBars m_nm = FBF_TRY(block_bars(res, blocks.ncas, false, true));
      for (const auto &[k, walk] : walks) {
        const ScWalk w{.a = &ops[at(mi, ni)],
                       .m_a = &m_mn,
                       .b = &ops[at(ni, mi)],
                       .m_b = &m_nm,
                       .finish = [&](const int64_t first, const std::vector<Perturber> &perturbers,
                                     const std::vector<double> &a, const std::vector<double> &b,
                                     std::vector<double> &a_bar, std::vector<double> &b_bar) {
                         for (size_t x = 0; x < perturbers.size(); ++x) {
                           const Perturber &p = perturbers[x];
                           const auto g = static_cast<size_t>(first) + x;
                           // T, the mean of the two norm einsums.
                           const double t = 0.5 * (combined(p, a) + combined(p, b));
                           double coefficient = 0.0;
                           for (const auto &[bra, ket] : {std::pair{mi, ni}, std::pair{ni, mi}}) {
                             ClassEnergies &element = out.elements[at(bra, ket)];
                             element.norms[k] += t;
                             KetTables &kt = kets[k][static_cast<size_t>(ket)];
                             if (std::abs(kt.norm[g]) <= kNumericalZero) {
                               continue;
                             }
                             const double d = perturber_delta(p, c.e) + (kt.h[g] / kt.norm[g]);
                             element.energies[k] -= t / d;
                             const double weight = hbar[at(bra, ket)];
                             kt.a[g] += weight * t / (d * d);
                             coefficient -= weight / d;
                           }
                           spread(p, 0.5 * coefficient, a_bar);
                           spread(p, 0.5 * coefficient, b_bar);
                         }
                       }};
        out.eq40[k] += FBF_TRY(walk(c, w));
      }
      if (transition) {
        const int64_t n = blocks.ncas;
        for (const auto &[bra, ket, m] : {std::tuple{mi, ni, &m_mn}, std::tuple{ni, mi, &m_nm}}) {
          Buffers::device<double> dm1 = FBF_TRY(device_zeros(res, n * n));
          Buffers::device<double> dm2 = FBF_TRY(device_zeros(res, n * n * n * n));
          Buffers::device<double> dm3 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
          const EnergyBars t{.dm1 = dm1.data(), .dm2 = dm2.data(), .dm3 = dm3.data()};
          FBF_TRY(sweep_metrics(res, ops[at(bra, ket)], *m, t));
          FBF_TRY(transition(bra, ket, t));
        }
      }
    }
  }

  // 3. Each state's own walk: its norm and H, the finish's multipliers with
  //    the pairs' A^J added, every term's adjoint, and its block bars swept
  //    onto its own RDMs' bars and the shared h1e and h2e.
  for (int64_t j = 0; j < ns; ++j) {
    BlockBars m = FBF_TRY(block_bars(res, blocks.ncas, true, true));
    const double weight = hbar[at(j, j)];
    ClassEnergies &element = out.elements[at(j, j)];
    for (const auto &[k, walk] : walks) {
      const KetTables &kt = kets[k][static_cast<size_t>(j)];
      // PySCF's _norm_to_energy over the walk's own tables, and its
      // derivatives, Park Eqs. 36-38 for H̄_JJ E_J: T = -1 / (Δ + H/N),
      // P = dE/dN = 2T + Δ T², Q = dE/dH = T² and R = dE/dΔ = N T², each put
      // back onto the tables and the orbital energies the perturber reads.
      // A^J adds the pairs' share through D_J: -A H / N² on N, A / N on H,
      // A on Δ. A vanishing norm contributes no energy and no multipliers.
      const ScWalk w{
          .a = &ops[at(j, j)],
          .m_a = &m,
          .finish = [&](const int64_t first, const std::vector<Perturber> &perturbers,
                        const std::vector<double> &a, const std::vector<double> &b,
                        std::vector<double> &a_bar, std::vector<double> &b_bar) {
            for (size_t x = 0; x < perturbers.size(); ++x) {
              const Perturber &p = perturbers[x];
              const double norm = combined(p, a);
              const double h = combined(p, b);
              element.norms[k] += norm;
              if (std::abs(norm) <= kNumericalZero) {
                continue;
              }
              const double diff = perturber_delta(p, c.e);
              element.energies[k] -= norm / (diff + (h / norm));
              const double t = -1.0 / (diff + (h / norm));
              const double ap = kt.a.empty() ? 0.0 : kt.a[static_cast<size_t>(first) + x];
              spread(p, (weight * ((2.0 * t) + (diff * t * t))) + (ap * (-h / (norm * norm))),
                     a_bar);
              spread(p, (weight * (t * t)) + (ap / norm), b_bar);
              add_delta_bar(p, (weight * (norm * t * t)) + ap, c.e);
            }
          }};
      out.eq40[k] += FBF_TRY(walk(c, w));
    }
    const EnergyBars &own = bars[static_cast<size_t>(j)];
    const std::array wanted{own.h1e, own.h2e, own.dm1, own.dm2, own.dm3, own.f3ca, own.f3ac};
    if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
      FBF_TRY(sweep_blocks(res, ops[at(j, j)], m, own));
    }
  }

  // Sijrs reads no RDM: every state's own is the one E, and every
  // transition's ⟨I|J⟩ = 0 times it, so its bars are Σ_J H̄_JJ times E's.
  double trace = 0.0;
  for (int64_t j = 0; j < ns; ++j) {
    trace += hbar[at(j, j)];
  }
  const ClassSums sijrs = FBF_TRY(reverse_sijrs(c, trace));
  for (int64_t j = 0; j < ns; ++j) {
    out.elements[at(j, j)].norms[2] = sijrs.norm;
    out.elements[at(j, j)].energies[2] = sijrs.energy;
  }
  out.eq40[2] = sijrs.eq40;

  // R's host sums onto the orbital energies' bars (Sijrs's are already there).
  for (const auto &[host, bar] :
       {std::pair{&e_core_bar, c.bars.e_core}, std::pair{&e_virt_bar, c.bars.e_virt}}) {
    if (bar != nullptr && !host->empty()) {
      const Buffers::device<double> d = FBF_TRY(upload(res, *host));
      FBF_TRY(einsum(res.stream(), "i->i", {{.data = d.data(), .dims = {std::ssize(*host)}}}, bar,
                     {std::ssize(*host)}, 1.0));
      FBF_TRY(res.sync());
    }
  }
  for (ClassEnergies &element : out.elements) {
    element.total = std::ranges::fold_left(element.energies, 0.0, std::plus{});
  }
  return out;
}

Result<ClassEnergiesAdjoint> class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks,
                                                    const EnergyInputs &in, const int32_t batch,
                                                    const EnergyBars &bars) {
  const double one = 1.0;
  ScEffectiveHamiltonianAdjoint got =
      FBF_TRY(sc_couplings_adjoint(res, blocks, {&in, 1}, 1, {&one, 1}, batch, {&bars, 1}, {}));
  return ClassEnergiesAdjoint{.energies = got.elements.front(), .eq40 = got.eq40};
}

// ---------------------------------------------------------------------------
// The PC reverse pass: the solve's adjoint, over n states
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
//
// Over n states (pc_effective_hamiltonian's couplings) the coupling of pair
// (I, J) is -c b_{I←J}ᵀ A_J⁺ b_J, bilinear in the two rows, with
// b_{I←J} = S^{(J,I)} x + S₁^{(J,I)} h1 and y_{I←J} = U_Jᵀ b_{I←J}. It puts
// -c H̄_IJ g_J on y_{I←J}, -c H̄_IJ g_{I←J} on y_J and c H̄_IJ g_{I←J} g_Jᵀ on
// U_Jᵀ dA_J U_J: the same rule, still no eigen-derivative, b_{I←J} lying in
// S_J's kept range (a mode that annihilates |J⟩ annihilates ⟨J| S^{(J,I)}).
// T_{I←J} = S^{(J,I)ᵀ} U_J, so its bar goes onto S^{(J,I)} unsymmetrized,
// U_J held, and so onto pair (J, I)'s metric blocks.

// One PC class's walk-long accumulators for one pair (I, J): the bars on
// T_{I←J} (d x m_J, T's layout) and on t1_{I←J} (n1 x m_J; empty when the
// class has no S₁), and on the diagonal ket J's weights in its eigenbasis,
// M = Σ_t g_t w_tᵀ and M_Δ = Σ_t Δ_t g_t w_tᵀ (m x m, row-major).
struct PcWalkBars {
  Buffers::device<double> mgg, mgg_delta, t, t1;
};

Result<PcWalkBars> pc_walk_bars(DeviceResources &res, const int64_t m, const int64_t d,
                                const int64_t n1, const bool own) {
  return PcWalkBars{.mgg = FBF_TRY(device_zeros(res, own ? m * m : 0)),
                    .mgg_delta = FBF_TRY(device_zeros(res, own ? m * m : 0)),
                    .t = FBF_TRY(device_zeros(res, d * m)),
                    .t1 = FBF_TRY(device_zeros(res, n1 * m))};
}

// What every PC class's reverse walk reads of the n states besides
// ReverseContext: their operands and H̄, n x n row-major as the pairs, and the
// block bars each pair's matrices are built from, [I ns + J]: a state's own
// (every block) on the diagonal, pair (I, J)'s metrics off it.
struct PcStates {
  int64_t ns = 1;
  std::span<const ActiveOperands> ops;
  std::span<const double> hbar;
  std::span<BlockBars> m;
};

// One class's reverse walk under way: its solves and transition rows, the
// walk bars of every pair (I, J) at [I ns + J], the couplings there, and each
// ket's smallest denominator.
struct PcReverse {
  const char *name;
  PcKets kets;
  std::vector<PcWalkBars> w;
  std::vector<double> energy;
  std::vector<double> min_denominator;
};

Result<PcReverse> pc_reverse_kets(ReverseContext &c, const PcStates &q, const char *name,
                                  const int64_t d, const int64_t n1, const PcBuild build) {
  const int64_t ns = q.ns;
  PcReverse r{.name = name,
              .kets = FBF_TRY(pc_kets(c.res, name, q.ops, ns, d, n1, build)),
              .w = {},
              .energy = std::vector<double>(static_cast<size_t>(ns * ns), 0.0),
              .min_denominator = std::vector<double>(static_cast<size_t>(ns),
                                                     std::numeric_limits<double>::infinity())};
  for (int64_t p = 0; p < ns * ns; ++p) {
    r.w.push_back(FBF_TRY(pc_walk_bars(c.res, r.kets.m(p % ns), d, n1, p / ns == p % ns)));
  }
  return r;
}

// One slab of a PC class over every pair: @p terms (I, J) evaluated over the
// tuple axes @p tuple_dims and m_J into pair (I, J)'s block of y, packed at
// PcKets::offsets, or for a term with h set into the same block of the
// exchange rows x (Sijr's swapped core pair, Srsi's SrsiT slab: a state's own
// pair alone); read back and finished on the host; the weights added to each
// ket's M and M_Δ; and every term's adjoint taken against ybar, or xbar.
//
// Per ket J and tuple t (row t1 of @p rows, Perturber read as: t2 < 0 a
// single coupling with factor s, t2 >= 0 the pair 2 direct - exchange with its
// exchange row t2 of x; Δ as SC's finish takes it), with g = y_J / (λ + Δ),
// g_x = x_J / (λ + Δ) and, for each bra I, g_I = y_{I←J} / (λ + Δ):
//   single: E_IJ -= s g_I.y_J,  ybar_{I←J} += -s H̄ g,  ybar_J += -s H̄ g_I,
//           R += s H̄ g_I.g,     w_J = s g;
//   pair:   E_IJ -= 2 g_I.y_J - g_I.x_J,  ybar_{I←J} += -H̄ (2 g - g_x),
//           ybar_J += -2 H̄ g_I,  xbar_J += H̄ g_I,
//           R += H̄ (2 g_I.g - g_I.g_x),   w_J = 2 g - g_x;
// H̄ = H̄_IJ, M_J += Σ_t (Σ_I H̄_IJ g_I) w_Jᵀ and M_Δ the same with Δ_t, and
// R = dL/dΔ onto the orbital energies, as SC's finish puts it. The own pair
// I = J is taken first, its two ybar terms as one (-2 s g, -(4 g - g_x)), so
// that at n = 1, H̄ = 1, every multiplier is the one-state pass's to the bit.
Status pc_slab_reverse(ReverseContext &c, const PcStates &q, PcReverse &r,
                       const std::function<std::vector<ClassTerm>(int64_t, int64_t)> &terms,
                       const std::vector<int64_t> &tuple_dims, const std::vector<Perturber> &rows) {
  DeviceResources &res = c.res;
  const wwr::wwrStream_t s = res.stream();
  const int64_t ns = q.ns;
  const int64_t nt = std::ranges::fold_left(tuple_dims, int64_t{1}, std::multiplies{});
  const std::vector<int64_t> at = r.kets.offsets(nt);
  const int64_t count = at.back();
  const auto dims = [&](const int64_t j) {
    std::vector<int64_t> out = tuple_dims;
    out.push_back(r.kets.m(j));
    return out;
  };
  std::vector<std::vector<ClassTerm>> pair_terms;
  bool exchange = false;
  for (int64_t p = 0; p < ns * ns; ++p) {
    pair_terms.push_back(terms(p / ns, p % ns));
    exchange = exchange || std::ranges::any_of(pair_terms.back(), &ClassTerm::h);
  }
  Buffers::device<double> y_d = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> x_d = FBF_TRY(device_zeros(res, exchange ? count : 0));
  for (int64_t p = 0; p < ns * ns; ++p) {
    const auto pp = static_cast<size_t>(p);
    FBF_TRY(eval_terms(res, pair_terms[pp], y_d.data() + at[pp],
                       exchange ? x_d.data() + at[pp] : nullptr, dims(p % ns)));
  }
  const std::vector<double> y = FBF_TRY(download(res, y_d.data(), count));
  const std::vector<double> x = FBF_TRY(download(res, x_d.data(), exchange ? count : 0));

  std::vector<double> ybar(y.size(), 0.0);
  std::vector<double> xbar(x.size(), 0.0);
  std::vector<double> gsum(y.size(), 0.0);
  std::vector<double> w(y.size(), 0.0);
  std::vector<double> delta(rows.size(), 0.0);
  for (int64_t j = 0; j < ns; ++j) {
    const auto jj = static_cast<size_t>((j * ns) + j);
    const std::vector<double> &lambda = r.kets.solves[static_cast<size_t>(j)].lambda;
    const size_t m = lambda.size();
    const double hj = q.hbar[jj];
    double &floor = r.min_denominator[static_cast<size_t>(j)];
    for (size_t t = 0; t < rows.size(); ++t) {
      const Perturber &p = rows[t];
      const double diff = perturber_delta(p, c.e);
      delta[t] = diff;
      const size_t row = static_cast<size_t>(at[jj]) + (static_cast<size_t>(p.t1) * m);
      const size_t xrow =
          p.t2 < 0 ? row : static_cast<size_t>(at[jj]) + (static_cast<size_t>(p.t2) * m);
      double rbar = 0.0;
      for (size_t k = 0; k < m; ++k) {
        const double den = lambda[k] + diff;
        floor = std::min(floor, den);
        const double g = y[row + k] / den;
        const double gx = p.t2 < 0 ? 0.0 : x[xrow + k] / den;
        gsum[row + k] = hj * g;
        if (p.t2 < 0) {
          r.energy[jj] -= p.s * g * y[row + k];
          ybar[row + k] += hj * (-2.0 * p.s * g);
          w[row + k] = p.s * g;
          rbar += hj * (p.s * g * g);
        } else {
          r.energy[jj] -= (2.0 * g * y[row + k]) - (g * x[xrow + k]);
          ybar[row + k] += hj * (-((4.0 * g) - gx));
          xbar[xrow + k] += hj * g;
          w[row + k] = (2.0 * g) - gx;
          rbar += hj * ((2.0 * g * g) - (g * gx));
        }
        for (int64_t i = 0; i < ns; ++i) {
          if (i == j) {
            continue;
          }
          const auto ij = static_cast<size_t>((i * ns) + j);
          const size_t irow = static_cast<size_t>(at[ij]) + (static_cast<size_t>(p.t1) * m);
          const double hij = q.hbar[ij];
          const double gi = y[irow + k] / den;
          gsum[row + k] += hij * gi;
          if (p.t2 < 0) {
            r.energy[ij] -= p.s * gi * y[row + k];
            ybar[irow + k] += -p.s * hij * g;
            ybar[row + k] += -p.s * hij * gi;
            rbar += hij * p.s * gi * g;
          } else {
            r.energy[ij] -= (2.0 * gi * y[row + k]) - (gi * x[xrow + k]);
            ybar[irow + k] += -hij * ((2.0 * g) - gx);
            ybar[row + k] += -2.0 * hij * gi;
            xbar[xrow + k] += hij * gi;
            rbar += hij * ((2.0 * gi * g) - (gi * gx));
          }
        }
      }
      add_delta_bar(p, rbar, c.e);
    }
  }

  const Buffers::device<double> gsum_d = FBF_TRY(upload(res, gsum));
  const Buffers::device<double> w_d = FBF_TRY(upload(res, w));
  const Buffers::device<double> dt = FBF_TRY(upload(res, delta));
  for (int64_t j = 0; j < ns; ++j) {
    const auto jj = static_cast<size_t>((j * ns) + j);
    const int64_t m = r.kets.m(j);
    const EinsumOperand g_op{.data = gsum_d.data() + at[jj], .dims = {nt, m}};
    const EinsumOperand w_op{.data = w_d.data() + at[jj], .dims = {nt, m}};
    FBF_TRY(einsum(s, "tk,tl->kl", {g_op, w_op}, r.w[jj].mgg.data(), {m, m}, 1.0));
    FBF_TRY(einsum(s, "t,tk,tl->kl", {{.data = dt.data(), .dims = {nt}}, g_op, w_op},
                   r.w[jj].mgg_delta.data(), {m, m}, 1.0));
  }
  const Buffers::device<double> ybar_d = FBF_TRY(upload(res, ybar));
  const Buffers::device<double> xbar_d = FBF_TRY(upload(res, xbar));
  for (int64_t p = 0; p < ns * ns; ++p) {
    const auto pp = static_cast<size_t>(p);
    FBF_TRY(adjoint_terms(res, pair_terms[pp], ybar_d.data() + at[pp],
                          exchange ? xbar_d.data() + at[pp] : nullptr, dims(p % ns), c.ops.delta,
                          nullptr));
  }
  return {};
}

// The bars on the raw S (d x d), K (d x d) and S₁ (d x n1) pair (I, J)'s rows
// read, each row-major as the class built it, from the walk's accumulators.
// Column-major throughout, as the solve. @p own (I = J): ket J's own, @p w.t
// spent; M and M_Δ, row-major, read column-major as their transposes, which
// U Mᵀ's place in U (U Mᵀ)ᵀ = U M Uᵀ undoes, and S and K symmetrized as the
// solve symmetrized them. Else pair (J, I)'s transition S and S₁, which
// T_{I←J} = A U_J reads unsymmetrized (A the row-major S read column-major):
// A's bar, the buffer's own, is T̄ Uᵀ; K is not formed.
struct PcMatrixBars {
  Buffers::device<double> s, k, s1;
};

Result<PcMatrixBars> pc_matrix_bars(DeviceResources &res, const ClassSolve &cs, PcWalkBars &w,
                                    const int64_t d, const int64_t n1, const bool own) {
  const int64_t m = std::ssize(cs.lambda);
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  PcMatrixBars out{.s = FBF_TRY(device_zeros(res, 0)),
                   .k = FBF_TRY(device_zeros(res, 0)),
                   .s1 = FBF_TRY(device_zeros(res, d * n1))};
  if (own) {
    // dE/dS_sym = U (T̄ + U M_Δᵀ)ᵀ, dE/dK_sym = U (U Mᵀ)ᵀ.
    FBF_TRY(gpu_status(gemm(res.blas(), di, mi, mi, &kOne<double>, cs.u.data(), w.mgg_delta.data(),
                            &kOne<double>, w.t.data())));
    Buffers::device<double> z(static_cast<size_t>(d * m), res.device());
    FBF_TRY(gpu_status(gemm(res.blas(), di, mi, mi, &kOne<double>, cs.u.data(), w.mgg.data(),
                            &kZero<double>, z.data())));
    Buffers::device<double> s_sym(static_cast<size_t>(d * d), res.device());
    Buffers::device<double> k_sym(static_cast<size_t>(d * d), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), di, di, mi, &kOne<double>, cs.u.data(), w.t.data(), &kZero<double>,
        s_sym.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), di, di, mi, &kOne<double>, cs.u.data(), z.data(), &kZero<double>,
        k_sym.data())));
    // The solve symmetrized S and K, and the symmetrization is its own adjoint.
    out.s = FBF_TRY(symmetrized(res, s_sym.data(), d));
    out.k = FBF_TRY(symmetrized(res, k_sym.data(), d));
  } else {
    out.s = Buffers::device<double>(static_cast<size_t>(d * d), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), di, di, mi, &kOne<double>, w.t.data(), cs.u.data(), &kZero<double>,
        out.s.data())));
  }
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

// One class's bars on one pair's raw matrices @p b back onto the blocks they
// were built from over @p o (@p own: a state's own S, K and S₁; else a
// transition's S and S₁), and Σ_G <M_G, G> over those blocks added to the
// device scalar @p acc. Each rebuilds the block values its sum reads.
using PcBack = Status (*)(DeviceResources &, const ActiveOperands &o, bool own,
                          const PcMatrixBars &b, BlockBars &m, double *acc);

// One PC class over the n states: its couplings [I ns + J], each ket's
// dropped count, and its homogeneity sum.
struct PcClassSums {
  std::vector<double> energy;
  std::vector<int64_t> dropped;
  double homogeneity = 0.0;
};

// A class with no tuple: nothing coupled, nothing dropped.
PcClassSums pc_no_tuples(const int64_t ns) {
  return {.energy = std::vector<double>(static_cast<size_t>(ns * ns), 0.0),
          .dropped = std::vector<int64_t>(static_cast<size_t>(ns), 0),
          .homogeneity = 0.0};
}

// After a class's walk: each ket's denominators checked as the energy's, then
// every pair's matrix bars back through @p back, pair (I, J)'s onto ket J's
// own blocks at I = J and onto pair (J, I)'s, whose matrices its rows read,
// off it.
Result<PcClassSums> pc_reverse_finish(ReverseContext &c, const PcStates &q, PcReverse &r,
                                      const int64_t d, const int64_t n1, const PcBack back) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  for (int64_t j = 0; j < ns; ++j) {
    const Status ok = pc_denominators_positive(r.name, r.min_denominator[static_cast<size_t>(j)]);
    if (!ok) {
      return state_error(ok.error(), j, ns);
    }
  }
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  for (int64_t i = 0; i < ns; ++i) {
    for (int64_t j = 0; j < ns; ++j) {
      const auto built = static_cast<size_t>((j * ns) + i);
      const PcMatrixBars b =
          FBF_TRY(pc_matrix_bars(res, r.kets.solves[static_cast<size_t>(j)],
                                 r.w[static_cast<size_t>((i * ns) + j)], d, n1, i == j));
      FBF_TRY(back(res, q.ops[built], i == j, b, q.m[built], acc.data()));
    }
  }
  PcClassSums out{
      .energy = std::move(r.energy), .dropped = {}, .homogeneity = FBF_TRY(scalar(res, acc))};
  for (const ClassSolve &cs : r.kets.solves) {
    out.dropped.push_back(cs.dropped);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The classes: each pc.cpp's walk, its slab einsums listed once as ClassTerms
// per pair (T's basis axes row-major after k, so T is packed and
// einsum_adjoint writes its bar), and its blocks' way back.
// ---------------------------------------------------------------------------

// Sr's bars onto sr_dm3 and sr_dm2 (its S and S₁ are permutations of them)
// and a16.
Status pc_back_sr(DeviceResources &res, const ActiveOperands &o, const bool own,
                  const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  FBF_TRY(adjoint_terms(res, {{"rpqbac->pqrabc", {o.dm3}, {m.sr_dm3.data()}}}, b.s.data(),
                        b.s.data(), d6, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res, {{"rpqa->pqra", {o.dm2}, {m.sr_dm2.data()}}}, b.s1.data(), b.s1.data(),
                        d4, o.delta, nullptr));
  FBF_TRY(add_dot(res, m.sr_dm3, o.dm3.data, acc));
  FBF_TRY(add_dot(res, m.sr_dm2, o.dm2.data, acc));
  if (own) {
    FBF_TRY(add_onto(res, b.k, m.a16.data(), n * n * n));
    const Buffers::device<double> a16 = FBF_TRY(make_a16(res, o.h1e, o.h2e, o.dm3, o.f3ac, o.f3ca));
    FBF_TRY(add_dot(res, m.a16, a16.data(), acc));
  }
  return {};
}

Result<PcClassSums> pc_reverse_sr(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sr);
  if (extent == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Sr", d, n, pc_sr_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      const int64_t m = r.kets.m(j);
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {m, n, n, n}};
      const EinsumOperand t1{.data = r.kets.t1_of(i, j), .dims = {n, m}};
      return {{"ipqr,kpqr->ik", {h2e_v, t}, {sb, w.t.data()}},
              {"ia,ak->ik", {h1e_v, t1}, {hb, w.t1.data()}}};
    };
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      rows.push_back({.t1 = i, .v1 = static_cast<int32_t>(b0 + i)});
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sr, b0, b1, sb));
  }
  return pc_reverse_finish(c, q, r, d, n, pc_back_sr);
}

// Si's bars onto dm3_h, dm2_h and a22.
Status pc_back_si(DeviceResources &res, const ActiveOperands &o, const bool own,
                  const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> d6(6, n);
  const Buffers::device<double> dm3_h = FBF_TRY(make_dm3_h(res, o.dm2, o.dm3, o.delta));
  const Buffers::device<double> dm2_h = FBF_TRY(make_dm2_h(res, o.dm1, o.dm2, o.delta));
  const EinsumOperand dm3_h_op{.data = dm3_h.data(), .dims = d6};
  const EinsumOperand dm2_h_op{.data = dm2_h.data(), .dims = d4};
  FBF_TRY(adjoint_terms(res, {{"rpqbac->pqrabc", {dm3_h_op}, {m.dm3_h.data()}}}, b.s.data(),
                        b.s.data(), d6, o.delta, nullptr));
  FBF_TRY(adjoint_terms(res, {{"rpqa->pqra", {dm2_h_op}, {m.dm2_h.data()}}}, b.s1.data(),
                        b.s1.data(), d4, o.delta, nullptr));
  FBF_TRY(add_dot(res, m.dm3_h, dm3_h.data(), acc));
  FBF_TRY(add_dot(res, m.dm2_h, dm2_h.data(), acc));
  if (own) {
    FBF_TRY(add_onto(res, b.k, m.a22.data(), n * n * n));
    const Buffers::device<double> a22 =
        FBF_TRY(make_a22(res, o.h1e, o.h2e, o.dm2, o.dm3, o.f3ac, o.f3ca));
    FBF_TRY(add_dot(res, m.a22, a22.data(), acc));
  }
  return {};
}

Result<PcClassSums> pc_reverse_si(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t d = n * n * n;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Si);
  if (extent == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Si", d, n, pc_si_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      const int64_t m = r.kets.m(j);
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {m, n, n, n}};
      const EinsumOperand t1{.data = r.kets.t1_of(i, j), .dims = {n, m}};
      return {{"qpir,kpqr->ik", {h2e_v, t}, {sb, w.t.data()}},
              {"ai,ak->ik", {h1e_v, t1}, {hb, w.t1.data()}}};
    };
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      rows.push_back({.t1 = i, .c1 = static_cast<int32_t>(b0 + i)});
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Si, b0, b1, sb));
  }
  return pc_reverse_finish(c, q, r, d, n, pc_back_si);
}

// Sijr's bars onto hdm1 (its S) and a3.
Status pc_back_sijr(DeviceResources &res, const ActiveOperands &o, const bool own,
                    const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const Buffers::device<double> hdm1 = FBF_TRY(make_hdm1(res, o.dm1, o.delta, own ? 1.0 : 0.0));
  FBF_TRY(add_onto(res, b.s, m.hdm1.data(), n));
  FBF_TRY(add_dot(res, m.hdm1, hdm1.data(), acc));
  if (own) {
    FBF_TRY(add_onto(res, b.k, m.a3.data(), n));
    const EinsumOperand hdm1_op{.data = hdm1.data(), .dims = {n, n}};
    const Buffers::device<double> a3 =
        FBF_TRY(make_a3(res, o.h1e, o.h2e, o.dm1, o.dm2, hdm1_op, o.delta));
    FBF_TRY(add_dot(res, m.a3, a3.data(), acc));
  }
  return {};
}

// Sijr: the exchange rows y[r, i, j] of a state's own pair as x, by the
// swapped einsum.
Result<PcClassSums> pc_reverse_sijr(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sijr);
  if (extent == 0 || nc == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Sijr", n, 0, pc_sijr_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {r.kets.m(j), n}};
      std::vector<ClassTerm> out{{"rpji,kp->rjik", {h2e_v, t}, {sb, w.t.data()}, false}};
      if (i == j) {
        out.push_back({"rpij,kp->rjik", {h2e_v, t}, {sb, w.t.data()}, true});
      }
      return out;
    };
    std::vector<Perturber> rows;
    for (int64_t rr = 0; rr < nr; ++rr) {
      for (int64_t j = 0; j < nc; ++j) {
        for (int64_t i = 0; i < nc; ++i) {
          const int64_t row = (((rr * nc) + j) * nc) + i;
          rows.push_back({.t1 = row,
                          .t2 = row,
                          .v1 = static_cast<int32_t>(b0 + rr),
                          .c1 = static_cast<int32_t>(j),
                          .c2 = static_cast<int32_t>(i)});
        }
      }
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nr, nc, nc}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sijr, b0, b1, sb));
  }
  return pc_reverse_finish(c, q, r, n, 0, pc_back_sijr);
}

// Srsi's bars onto dm1 (SC's srsi_dm1, its S) and k27.
Status pc_back_srsi(DeviceResources &res, const ActiveOperands &o, const bool own,
                    const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  FBF_TRY(add_onto(res, b.s, m.srsi_dm1.data(), n));
  FBF_TRY(add_dot(res, m.srsi_dm1, o.dm1.data, acc));
  if (own) {
    FBF_TRY(add_onto(res, b.k, m.k27.data(), n));
    const Buffers::device<double> k27 = FBF_TRY(make_k27(res, o.h1e, o.h2e, o.dm1, o.dm2));
    FBF_TRY(add_dot(res, m.k27, k27.data(), acc));
  }
  return {};
}

// Srsi: the exchange rows of a state's own pair from the SrsiT slab as x.
Result<PcClassSums> pc_reverse_srsi(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srsi);
  if (extent == 0 || nc == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Srsi", n, 0, pc_srsi_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {r.kets.m(j), n}};
      std::vector<ClassTerm> out{{"rsip,kp->rsik", {h_a, t}, {bar_a.data(), w.t.data()}, false}};
      if (i == j) {
        out.push_back({"srip,kp->rsik", {h_b, t}, {bar_b.data(), w.t.data()}, true});
      }
      return out;
    };
    std::vector<Perturber> rows;
    for (int64_t rr = 0; rr < nb; ++rr) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        for (int64_t i = 0; i < nc; ++i) {
          const int64_t row = (((rr * nv) + ss) * nc) + i;
          rows.push_back({.t1 = row,
                          .t2 = row,
                          .v1 = static_cast<int32_t>(b0 + rr),
                          .v2 = static_cast<int32_t>(ss),
                          .c1 = static_cast<int32_t>(i)});
        }
      }
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nv, nc}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srsi, b0, b1, bar_a.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::SrsiT, b0, b1, bar_b.data()));
  }
  return pc_reverse_finish(c, q, r, n, 0, pc_back_srsi);
}

// Srs's bars onto rm2 and a7; a state's own rm2 is make_a7's, as the walk's.
Status pc_back_srs(DeviceResources &res, const ActiveOperands &o, const bool own,
                   const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  const auto onto_rm2 = [&](const Buffers::device<double> &rm2) -> Status {
    FBF_TRY(adjoint_terms(res, {{"pqba->pqab", {{.data = rm2.data(), .dims = d4}}, {m.rm2.data()}}},
                          b.s.data(), b.s.data(), d4, o.delta, nullptr));
    return add_dot(res, m.rm2, rm2.data(), acc);
  };
  if (!own) {
    return onto_rm2(FBF_TRY(make_rm2(res, o.dm1, o.dm2, o.delta)));
  }
  const auto [rm2, a7] = FBF_TRY(make_a7(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
  FBF_TRY(onto_rm2(rm2));
  FBF_TRY(add_onto(res, b.k, m.a7.data(), n * n));
  return add_dot(res, m.a7, a7.data(), acc);
}

Result<PcClassSums> pc_reverse_srs(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Srs);
  if (extent == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Srs", d, 0, pc_srs_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {r.kets.m(j), n, n}};
      return {{"rsqp,kpq->rsk", {h2e_v, t}, {sb, w.t.data()}}};
    };
    std::vector<Perturber> rows;
    for (int64_t rr = 0; rr < nb; ++rr) {
      for (int64_t ss = 0; ss < nv; ++ss) {
        rows.push_back({.t1 = (rr * nv) + ss,
                        .s = 0.5,
                        .v1 = static_cast<int32_t>(b0 + rr),
                        .v2 = static_cast<int32_t>(ss)});
      }
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nv}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Srs, b0, b1, sb));
  }
  return pc_reverse_finish(c, q, r, d, 0, pc_back_srs);
}

// Sij's bars onto hdm2 (its S) and a9.
Status pc_back_sij(DeviceResources &res, const ActiveOperands &o, const bool own,
                   const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  const Buffers::device<double> hdm2 =
      FBF_TRY(make_hdm2(res, o.dm1, o.dm2, o.delta, own ? 1.0 : 0.0));
  FBF_TRY(add_onto(res, b.s, m.hdm2.data(), n * n));
  FBF_TRY(add_dot(res, m.hdm2, hdm2.data(), acc));
  if (own) {
    FBF_TRY(add_onto(res, b.k, m.a9.data(), n * n));
    const EinsumOperand hdm2_op{.data = hdm2.data(), .dims = d4};
    // hdm3 is freed once a9 is built.
    const auto build_a9 = [&]() -> Result<Buffers::device<double>> {
      const Buffers::device<double> hdm3 =
          FBF_TRY(make_hdm3(res, o.dm1, o.dm2, o.dm3, hdm2_op, o.delta));
      const EinsumOperand hdm3_op{.data = hdm3.data(), .dims = std::vector<int64_t>(6, n)};
      return make_a9(res, o.h1e, o.h2e, hdm2_op, hdm3_op);
    };
    const Buffers::device<double> a9 = FBF_TRY(build_a9());
    FBF_TRY(add_dot(res, m.a9, a9.data(), acc));
  }
  return {};
}

Result<PcClassSums> pc_reverse_sij(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t d = n * n;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sij);
  if (extent == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Sij", d, 0, pc_sij_matrices));
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {r.kets.m(j), n, n}};
      return {{"qpij,kpq->ijk", {h2e_v, t}, {sb, w.t.data()}}};
    };
    std::vector<Perturber> rows;
    for (int64_t i = 0; i < nb; ++i) {
      for (int64_t j = 0; j < nc; ++j) {
        rows.push_back({.t1 = (i * nc) + j,
                        .s = 0.5,
                        .c1 = static_cast<int32_t>(b0 + i),
                        .c2 = static_cast<int32_t>(j)});
      }
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nc}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sij, b0, b1, sb));
  }
  return pc_reverse_finish(c, q, r, d, 0, pc_back_sij);
}

// Sir's bars onto dm1 and dm2 (SC's sir_dm1 and sir_dm2: its S and S₁ are
// built from them, through pc_sir_matrices' blocks), a12 and a13.
Status pc_back_sir(DeviceResources &res, const ActiveOperands &o, const bool own,
                   const PcMatrixBars &b, BlockBars &m, double *const acc) {
  const int64_t n = o.dm1.dims[0];
  const std::vector<int64_t> d4(4, n);
  const std::vector<int64_t> blocked{2, n, n, 2, n, n};
  // pc_sir_matrices' coefficient blocks.
  const Buffers::device<double> off_diag = FBF_TRY(upload(res, {2.0, -1.0, -1.0, 0.0}));
  const Buffers::device<double> block22 = FBF_TRY(upload(res, {0.0, 0.0, 0.0, 1.0}));
  const Buffers::device<double> one_body = FBF_TRY(upload(res, {2.0, -1.0}));
  const EinsumOperand off_diag_op{.data = off_diag.data(), .dims = {2, 2}};
  const EinsumOperand block22_op{.data = block22.data(), .dims = {2, 2}};
  const EinsumOperand one_body_op{.data = one_body.data(), .dims = {2}};
  // s22's value is never read: its bar's term reads only block22.
  const EinsumOperand s22_op{.dims = d4};
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
  FBF_TRY(adjoint_terms(res, {{"x,qp->xpq", {one_body_op, o.dm1}, {nullptr, md1}}}, b.s1.data(),
                        b.s1.data(), {2, n, n}, o.delta, nullptr));
  FBF_TRY(add_dot(res, m.sir_dm2, o.dm2.data, acc));
  FBF_TRY(add_dot(res, m.sir_dm1, o.dm1.data, acc));
  if (own) {
    const Buffers::device<double> a12 = FBF_TRY(make_a12(res, o.h1e, o.h2e, o.dm2, o.dm3));
    const Buffers::device<double> a13 =
        FBF_TRY(make_a13(res, o.h1e, o.h2e, o.dm1, o.dm2, o.dm3, o.delta));
    const EinsumOperand a12_op{.data = a12.data(), .dims = d4};
    const EinsumOperand a13_op{.data = a13.data(), .dims = d4};
    FBF_TRY(adjoint_terms(res,
                          {{"xy,pqab->xpqyab", {off_diag_op, a12_op}, {nullptr, m.a12.data()}},
                           {"xy,pqab->xpqyab", {block22_op, a13_op}, {nullptr, m.a13.data()}}},
                          b.k.data(), b.k.data(), blocked, o.delta, nullptr));
    FBF_TRY(add_dot(res, m.a12, a12.data(), acc));
    FBF_TRY(add_dot(res, m.a13, a13.data(), acc));
  }
  return {};
}

// Sir: T's two blocks are picked by a selector over T's leading block axis,
// so T stays one packed operand whose bar einsum_adjoint can write.
Result<PcClassSums> pc_reverse_sir(ReverseContext &c, const PcStates &q) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  const int64_t n = c.blocks.ncas;
  const int64_t nc = c.blocks.ncore;
  const int64_t nv = c.blocks.nvirt;
  const int64_t d = 2 * n * n;
  const int32_t extent = walk_extent(c.blocks, ExtBlock::Sir1);
  if (extent == 0 || nc == 0) {
    return pc_no_tuples(ns);
  }
  PcReverse r = FBF_TRY(pc_reverse_kets(c, q, "Sir", d, 1, pc_sir_matrices));
  const Buffers::device<double> first = FBF_TRY(upload(res, {1.0, 0.0}));
  const Buffers::device<double> second = FBF_TRY(upload(res, {0.0, 1.0}));
  const EinsumOperand first_op{.data = first.data(), .dims = {2}};
  const EinsumOperand second_op{.data = second.data(), .dims = {2}};
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
    const auto terms = [&](const int64_t i, const int64_t j) -> std::vector<ClassTerm> {
      const int64_t m = r.kets.m(j);
      PcWalkBars &w = r.w[static_cast<size_t>((i * ns) + j)];
      const EinsumOperand t{.data = r.kets.t_of(i, j), .dims = {m, 2, n, n}};
      const EinsumOperand t1{.data = r.kets.t1_of(i, j), .dims = {m}};
      return {{"rpiq,x,kxpq->rik", {v1, first_op, t}, {bar_1.data(), nullptr, w.t.data()}},
              {"rpqi,x,kxpq->rik", {v2, second_op, t}, {bar_2.data(), nullptr, w.t.data()}},
              {"ri,k->rik", {h1e_v, t1}, {hb, w.t1.data()}}};
    };
    std::vector<Perturber> rows;
    for (int64_t rr = 0; rr < nb; ++rr) {
      for (int64_t i = 0; i < nc; ++i) {
        rows.push_back({.t1 = (rr * nc) + i,
                        .v1 = static_cast<int32_t>(b0 + rr),
                        .c1 = static_cast<int32_t>(i)});
      }
    }
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nc}, rows));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir1, b0, b1, bar_1.data()));
    FBF_TRY(slab_bar_onto_lambda(c, ExtBlock::Sir2, b0, b1, bar_2.data()));
  }
  return pc_reverse_finish(c, q, r, d, 1, pc_back_sir);
}

// ---------------------------------------------------------------------------
// The PC-QD reverse pass, and PC's at n = 1
// ---------------------------------------------------------------------------

Result<PcEffectiveHamiltonianAdjoint> pc_couplings_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    const int64_t ns, std::span<const double> hbar, const int32_t batch,
    std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition) {
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<ActiveOperands> ops;
  for (const EnergyInputs &pair : pairs) {
    ops.push_back(FBF_TRY(active_operands(res, blocks, pair)));
  }
  std::vector<double> e_core_bar(ops.front().e_core.size(), 0.0);
  std::vector<double> e_virt_bar(ops.front().e_virt.size(), 0.0);
  // The fields every pair shares, and their bars, are the first's.
  ReverseContext c{.res = res,
                   .blocks = blocks,
                   .in = pairs.front(),
                   .bars = bars.front(),
                   .batch = batch,
                   .ops = ops.front(),
                   .e = {.core = ops.front().e_core,
                         .virt = ops.front().e_virt,
                         .core_bar = e_core_bar,
                         .virt_bar = e_virt_bar}};
  // SC's block bars, zeroed: every state's own whole set, and each
  // transition's metrics, held until every class has walked. PC adds onto
  // those its S, K and S₁ are built from, and the sweeps read them all (a17,
  // a19, a23, a25 and dm1_h stay zero).
  std::vector<BlockBars> m;
  for (int64_t p = 0; p < ns * ns; ++p) {
    m.push_back(FBF_TRY(block_bars(res, blocks.ncas, p / ns == p % ns, true)));
  }
  const PcStates q{.ns = ns, .ops = ops, .hbar = hbar, .m = m};

  PcEffectiveHamiltonianAdjoint out;
  out.elements.resize(static_cast<size_t>(ns * ns));
  const auto put = [&](const size_t k, const PcClassSums &x) {
    for (size_t p = 0; p < out.elements.size(); ++p) {
      out.elements[p].energies[k] = x.energy[p];
      out.elements[p].dropped[k] = x.dropped[p % static_cast<size_t>(ns)];
    }
    out.homogeneity[k] = x.homogeneity;
  };
  put(0, FBF_TRY(pc_reverse_sr(c, q)));
  put(1, FBF_TRY(pc_reverse_si(c, q)));
  // Sijrs has no active index, so its PC energy and its reverse pass are SC's:
  // every state's own is the one E, every transition's ⟨I|J⟩ = 0 times it,
  // so its bars are Σ_J H̄_JJ times E's.
  double trace = 0.0;
  for (int64_t j = 0; j < ns; ++j) {
    trace += hbar[static_cast<size_t>((j * ns) + j)];
  }
  const ClassSums sijrs = FBF_TRY(reverse_sijrs(c, trace));
  for (int64_t j = 0; j < ns; ++j) {
    out.elements[static_cast<size_t>((j * ns) + j)].energies[2] = sijrs.energy;
  }
  out.homogeneity[2] = sijrs.eq40;
  put(3, FBF_TRY(pc_reverse_sijr(c, q)));
  put(4, FBF_TRY(pc_reverse_srsi(c, q)));
  put(5, FBF_TRY(pc_reverse_srs(c, q)));
  put(6, FBF_TRY(pc_reverse_sij(c, q)));
  put(7, FBF_TRY(pc_reverse_sir(c, q)));
  for (PcClassEnergies &element : out.elements) {
    element.total = std::ranges::fold_left(element.energies, 0.0, std::plus{});
  }

  // dL/dΔ's host sums onto the orbital energies' bars (Sijrs's are already
  // there).
  for (const auto &[host, bar] :
       {std::pair{&e_core_bar, c.bars.e_core}, std::pair{&e_virt_bar, c.bars.e_virt}}) {
    if (bar != nullptr && !host->empty()) {
      const Buffers::device<double> d = FBF_TRY(upload(res, *host));
      FBF_TRY(einsum(res.stream(), "i->i", {{.data = d.data(), .dims = {std::ssize(*host)}}}, bar,
                     {std::ssize(*host)}, 1.0));
      FBF_TRY(res.sync());
    }
  }

  // Each state's own block bars onto bars[J] and the shared h1e and h2e; then
  // each transition's metrics onto its RDMs, handed over and freed.
  for (int64_t j = 0; j < ns; ++j) {
    const auto jj = static_cast<size_t>((j * ns) + j);
    const EnergyBars &own = bars[static_cast<size_t>(j)];
    const std::array wanted{own.h1e, own.h2e, own.dm1, own.dm2, own.dm3, own.f3ca, own.f3ac};
    if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
      FBF_TRY(sweep_blocks(res, ops[jj], m[jj], own));
    }
  }
  if (transition) {
    const int64_t n = blocks.ncas;
    for (int64_t p = 0; p < ns * ns; ++p) {
      if (p / ns == p % ns) {
        continue;
      }
      Buffers::device<double> dm1 = FBF_TRY(device_zeros(res, n * n));
      Buffers::device<double> dm2 = FBF_TRY(device_zeros(res, n * n * n * n));
      Buffers::device<double> dm3 = FBF_TRY(device_zeros(res, n * n * n * n * n * n));
      const EnergyBars t{.dm1 = dm1.data(), .dm2 = dm2.data(), .dm3 = dm3.data()};
      FBF_TRY(sweep_metrics(res, ops[static_cast<size_t>(p)], m[static_cast<size_t>(p)], t));
      FBF_TRY(transition(p / ns, p % ns, t));
    }
  }
  return out;
}

Result<PcClassEnergiesAdjoint>
pc_class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          const int32_t batch, const EnergyBars &bars) {
  const double one = 1.0;
  PcEffectiveHamiltonianAdjoint got =
      FBF_TRY(pc_couplings_adjoint(res, blocks, {&in, 1}, 1, {&one, 1}, batch, {&bars, 1}, {}));
  return PcClassEnergiesAdjoint{.energies = got.elements.front(), .homogeneity = got.homogeneity};
}

} // namespace free_breakfast::nevpt2
