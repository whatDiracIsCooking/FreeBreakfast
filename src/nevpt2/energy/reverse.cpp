/// @file
/// @brief The SC and PC reverse passes, of one state and of n: from the class
/// energies back to the slabs (and through them onto Λ), h1e_v's rows and
/// the orbital energies, and onto every class block's bar M_G = dE/dG; then
/// the sweep of those bars back through the blocks' builders onto h1e, h2e and the RDMs.
/// einsum_adjoint is the one derivative rule; SC's finish (P, Q, R: Park Eqs.
/// 36-38), PC's (the solve's adjoint) and Sijrs are the only expressions
/// differentiated by hand. And SC's and PC's forward tangents, over the same
/// walks: each term at its operands' tangents, the blocks' and slabs'
/// tangents their builders' exact polarizations, and PC's solve at its own
/// factorization with the kept span's rotation into the dropped modes.

module;

// The Sijrs launchers, defined in nevpt2_energy.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "nevpt2_energy_bridge.h"

module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
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

// ---------------------------------------------------------------------------
// The forward tangent's state, which an own walk carries beside its bars
// ---------------------------------------------------------------------------

// Where an array a class term reads has its tangent: the whole of the
// @p count elements at @p base, so that a view into it (a run of h1e_v's
// rows, a slab in its buffer) finds its own at the same offset. dots[u] is
// direction u's, null when it is zero. A slab's region has one buffer,
// @p slab, refilled per direction.
struct TangentRegion {
  const double *base = nullptr;
  int64_t count = 0;
  std::vector<const double *> dots;
  double *slab = nullptr;
};

// One direction's polarization steps (class_energies_tangent): s for the
// blocks, 0 when the direction moves no integral and no RDM, and for the
// slabs, 0 when it moves no B.
struct TangentSteps {
  double blocks = 0.0;
  double slabs = 0.0;
};

// What a walk's tangent reads and holds. Every class term's operand is
// looked up in @p regions: the inputs' first ([0, inputs), the call's), then
// the class's blocks and slabs, whose tangents' buffers @p held keeps, both
// dropped as the class's walk returns. @p directions and @p steps are the
// walk's first operands' (ScWalk::a); a pair's walk also has its second's,
// @p b_directions and @p b_steps, the (N, M) entry's, and their RDMs among
// the inputs. @p shifted is B + s Ḃ of direction @p shifted_for (Ḃ is every
// pair's), and @p plus and @p dd a slab's two scratch builds. @p finish
// takes each slab's tables as ScWalk::finish does, and per direction their
// tangents.
struct ScTangent {
  std::span<const EnergyDirection> directions;
  const MoBlocks *b = nullptr;
  std::vector<TangentSteps> steps;
  std::span<const EnergyDirection> b_directions;
  std::vector<TangentSteps> b_steps;
  std::vector<TangentRegion> regions;
  size_t inputs = 0;
  std::vector<Buffers::device<double>> held;
  std::optional<MoBlocks> shifted;
  int64_t shifted_for = -1;
  std::optional<Buffers::device<double>> plus, dd;
  std::function<void(int64_t first, const std::vector<Perturber> &perturbers,
                     const std::vector<double> &a, const std::vector<double> &b,
                     const std::vector<std::vector<double>> &a_dots,
                     const std::vector<std::vector<double>> &b_dots)>
      finish;
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
// the walk only evaluates and finishes. A walk with @p tangent also forms
// every block's and slab's tangent and the two tables' (an own walk's Ṅ and
// Ḣ, a pair's the two norm einsums'), and hands them to the tangent's
// finish.
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
  ScTangent *tangent = nullptr;
};

// A slab a walk's terms read: what its tangent rebuilds along Ḃ.
struct SlabRead {
  ExtBlock block = ExtBlock::Sr;
  int32_t b0 = 0;
  int32_t b1 = 0;
  const double *base = nullptr;
};

// A slab bar's buffer, as slab_buffer's, and the slab's element count.
int64_t slab_count(const MoBlocks &blocks, const ExtBlock block, const int32_t b0,
                   const int32_t b1) {
  return std::ranges::fold_left(slab_dims(blocks, block, b0, b1), int64_t{1}, std::multiplies{});
}

// ---------------------------------------------------------------------------
// The forward tangent's rules: a term at its operands' tangents, a block's
// and a slab's polarization
// ---------------------------------------------------------------------------

// Direction @p u's tangent of the operand whose first element is @p p: at
// the same offset in the region holding @p p; null when none does or the
// tangent is zero.
const double *tangent_at(const ScTangent &t, const double *const p, const size_t u) {
  const std::less<const double *> before;
  for (const TangentRegion &r : t.regions) {
    if (r.base != nullptr && !before(p, r.base) && before(p, r.base + r.count)) {
      return r.dots[u] == nullptr ? nullptr : r.dots[u] + (p - r.base);
    }
  }
  return nullptr;
}

// Every term of @p terms differentiated along direction @p u: the term at
// each operand's tangent in turn, the rest held (a slab read twice taken
// twice), into @p norm_dot and @p h_dot as eval_terms evaluates them.
Status tangent_terms(DeviceResources &res, const ScTangent &t, const size_t u,
                     const std::vector<ClassTerm> &terms, double *const norm_dot,
                     double *const h_dot, const std::vector<int64_t> &out_dims) {
  for (const ClassTerm &term : terms) {
    for (size_t k = 0; k < term.ops.size(); ++k) {
      const double *const dot =
          term.ops[k].data == nullptr ? nullptr : tangent_at(t, term.ops[k].data, u);
      if (dot == nullptr) {
        continue;
      }
      std::vector<EinsumOperand> ops = term.ops;
      ops[k].data = dot;
      FBF_TRY(einsum(res.stream(), term.subscripts, ops, term.h ? h_dot : norm_dot, out_dims,
                     term.alpha));
    }
  }
  return {};
}

// @p out += @p alpha @p x over @p count packed elements.
Status add_scaled(DeviceResources &res, const double *const x, const int64_t count,
                  double *const out, const double alpha) {
  return einsum(res.stream(), "a->a", {{.data = x, .dims = {count}}}, out, {count}, alpha);
}

// The bars a one-function class's η = <X, Y> / <Y, Y> (single_function_ratio)
// puts on its blocks @p xy for a bar @p eta_bar on η: [0] on X, η̄ Y / S,
// and [1] on Y, η̄ (X - 2η Y) / S, S = <Y, Y>; in single_function_blocks'
// pairing. Their Eq. 40 share, <M_X, X> + <M_Y, Y> = η̄ (η + η - 2η), is zero.
std::array<std::vector<double>, 2>
single_function_eta_bars(const std::array<std::vector<double>, 2> &xy, const double eta,
                         const double eta_bar) {
  const auto &[x, y] = xy;
  const double s = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
  std::array<std::vector<double>, 2> out{std::vector<double>(x.size()),
                                         std::vector<double>(y.size())};
  for (size_t e = 0; e < x.size(); ++e) {
    out[0][e] = eta_bar * y[e] / s;
    out[1][e] = eta_bar * (x[e] - (2.0 * eta * y[e])) / s;
  }
  return out;
}

// Adds host bars @p mxy on one-function class @p k's X and Y
// (single_function_blocks' pairing) onto @p m's: a7 and rm2 (Y read back
// `pqba`) for Srs, a9 and hdm2 for Sij.
Status add_single_function_bars(DeviceResources &res, const size_t k, const int64_t n,
                                const std::array<std::vector<double>, 2> &mxy, BlockBars &m) {
  const std::vector<int64_t> d4(4, n);
  const int64_t n4 = n * n * n * n;
  const Buffers::device<double> mx = FBF_TRY(upload(res, mxy[0]));
  const Buffers::device<double> my = FBF_TRY(upload(res, mxy[1]));
  FBF_TRY(add_scaled(res, mx.data(), n4, k == 5 ? m.a7.data() : m.a9.data(), 1.0));
  if (k == 5) {
    FBF_TRY(einsum(res.stream(), "pqab->pqba", {{.data = my.data(), .dims = d4}}, m.rm2.data(), d4,
                   1.0));
  } else {
    FBF_TRY(add_scaled(res, my.data(), n4, m.hdm2.data(), 1.0));
  }
  return res.sync();
}

// @p o with every array a block builder reads (h1e, h2e and the RDMs) moved
// by @p step times @p dot's, in buffers of its own; an array whose tangent is
// null stays @p o's. delta is @p o's, and the orbital energies are not read.
struct ShiftedOperands {
  std::vector<Buffers::device<double>> arrays;
  ActiveOperands ops;
};

Result<ShiftedOperands> shifted_operands(DeviceResources &res, const ActiveOperands &o,
                                         const EnergyInputs &dot, const double step) {
  ShiftedOperands out{.ops = {.h1e = o.h1e,
                              .h2e = o.h2e,
                              .dm1 = o.dm1,
                              .dm2 = o.dm2,
                              .dm3 = o.dm3,
                              .f3ca = o.f3ca,
                              .f3ac = o.f3ac,
                              .delta = o.delta,
                              .delta_buf = Buffers::device<double>(0, res.device())}};
  for (const auto &[op, d] : {std::pair{&out.ops.h1e, dot.h1e}, std::pair{&out.ops.h2e, dot.h2e},
                              std::pair{&out.ops.dm1, dot.dm1}, std::pair{&out.ops.dm2, dot.dm2},
                              std::pair{&out.ops.dm3, dot.dm3}, std::pair{&out.ops.f3ca, dot.f3ca},
                              std::pair{&out.ops.f3ac, dot.f3ac}}) {
    if (d == nullptr) {
      continue;
    }
    const int64_t count = std::ranges::fold_left(op->dims, int64_t{1}, std::multiplies{});
    Buffers::device<double> moved = FBF_TRY(device_zeros(res, count));
    FBF_TRY(add_scaled(res, op->data, count, moved.data(), 1.0));
    FBF_TRY(add_scaled(res, d, count, moved.data(), step));
    op->data = moved.data();
    out.arrays.push_back(std::move(moved));
  }
  return out;
}

// single_function_blocks' tangents along each of @p directions: the blocks'
// central polarization at each direction's block step (@p steps), exact as
// block_tangent's (a block is of degree at most two); zero where the step is.
Result<std::vector<std::array<std::vector<double>, 2>>>
single_function_block_dots(DeviceResources &res, const ActiveOperands &o, const size_t k,
                           std::span<const EnergyDirection> directions,
                           const std::vector<TangentSteps> &steps) {
  const int64_t n = o.dm1.dims[0];
  const auto n4 = static_cast<size_t>(n * n * n * n);
  std::vector<std::array<std::vector<double>, 2>> out;
  for (size_t u = 0; u < directions.size(); ++u) {
    const double s = steps[u].blocks;
    std::array<std::vector<double>, 2> dot{std::vector<double>(n4, 0.0),
                                           std::vector<double>(n4, 0.0)};
    if (s != 0.0) {
      for (const double sign : {1.0, -1.0}) {
        const ShiftedOperands x = FBF_TRY(shifted_operands(res, o, directions[u].in, sign * s));
        const std::array<std::vector<double>, 2> xy =
            FBF_TRY(single_function_blocks(res, x.ops, k));
        for (size_t b = 0; b < 2; ++b) {
          for (size_t e = 0; e < n4; ++e) {
            dot[b][e] += sign * xy[b][e] / (2.0 * s);
          }
        }
      }
    }
    out.push_back(std::move(dot));
  }
  return out;
}

// η̇ of a one-function class at its blocks @p xy and η along their tangents
// @p dot: (<Ẋ, Y> + <X, Ẏ> - 2η <Y, Ẏ>) / <Y, Y>.
double single_function_ratio_dot(const std::array<std::vector<double>, 2> &xy, const double eta,
                                 const std::array<std::vector<double>, 2> &dot) {
  const auto &[x, y] = xy;
  const double s = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
  const double xdy = std::inner_product(dot[0].begin(), dot[0].end(), y.begin(), 0.0);
  const double ydx = std::inner_product(x.begin(), x.end(), dot[1].begin(), 0.0);
  const double ydy = std::inner_product(y.begin(), y.end(), dot[1].begin(), 0.0);
  return (xdy + ydx - (2.0 * eta * ydy)) / s;
}

// A block built from a walk's own operands by its make_*.
using BlockBuild = std::function<Result<Buffers::device<double>>(const ActiveOperands &)>;

// When @p w carries a tangent, registers the tangent of the block @p base,
// which @p make builds from @p w's operands (its second's, a pair's (N, M),
// when @p second), along every direction: its central polarization
// (G(x + s ẋ) - G(x - s ẋ)) / 2s. A block is one integral times one RDM,
// plus terms linear in one or the other, so of degree at most two in (h1e,
// h2e, RDMs), and the difference is exact, a transition's constant (⟨I|J⟩)
// terms cancelling with the rest; s (TangentSteps) sizes s ẋ to x, so it
// cancels nothing past round-off.
Status block_tangent(DeviceResources &res, const ScWalk &w, const Buffers::device<double> &base,
                     const BlockBuild &make, const bool second = false) {
  ScTangent *const t = w.tangent;
  if (t == nullptr) {
    return {};
  }
  const ActiveOperands &o = second ? *w.b : *w.a;
  const std::span<const EnergyDirection> directions = second ? t->b_directions : t->directions;
  const std::vector<TangentSteps> &steps = second ? t->b_steps : t->steps;
  const auto count = static_cast<int64_t>(base.num_elements());
  TangentRegion region{.base = base.data(), .count = count};
  for (size_t u = 0; u < directions.size(); ++u) {
    const double s = steps[u].blocks;
    if (s == 0.0) {
      region.dots.push_back(nullptr);
      continue;
    }
    Buffers::device<double> dot = FBF_TRY(device_zeros(res, count));
    for (const double sign : {1.0, -1.0}) {
      const ShiftedOperands x = FBF_TRY(shifted_operands(res, o, directions[u].in, sign * s));
      const Buffers::device<double> g = FBF_TRY(make(x.ops));
      FBF_TRY(add_scaled(res, g.data(), count, dot.data(), sign / (2.0 * s)));
    }
    region.dots.push_back(dot.data());
    t->held.push_back(std::move(dot));
  }
  t->regions.push_back(std::move(region));
  return {};
}

// @p x + @p step @p dot over every MO block, into @p out (shaped as @p x).
Status shift_blocks(DeviceResources &res, const MoBlocks &x, const MoBlocks &dot, const double step,
                    MoBlocks &out) {
  for (const auto &[from, d, to] :
       {std::tuple{&x.aa, &dot.aa, &out.aa}, std::tuple{&x.ca, &dot.ca, &out.ca},
        std::tuple{&x.va, &dot.va, &out.va}, std::tuple{&x.cv, &dot.cv, &out.cv},
        std::tuple{&x.vc, &dot.vc, &out.vc}}) {
    const auto count = static_cast<int64_t>(from->num_elements());
    FBF_TRY(zero(res, to->data(), count));
    FBF_TRY(add_scaled(res, from->data(), count, to->data(), 1.0));
    FBF_TRY(add_scaled(res, d->data(), count, to->data(), step));
  }
  return {};
}

// A buffer of at least @p count doubles in @p buf, (re)allocated when short.
double *at_least(DeviceResources &res, std::optional<Buffers::device<double>> &buf,
                 const int64_t count) {
  if (!buf || std::cmp_less(buf->num_elements(), count)) {
    buf.emplace(static_cast<size_t>(count), res.device());
  }
  return buf->data();
}

// Direction @p u's tangent of the slab @p r reads, whose value r.base holds,
// into its region's buffer (registered, and sized by this slab, at the
// buffer's first slab, the walk's largest): a slab is bilinear in B, so
// (slab(B + s Ḃ) - slab(B) - s² slab(Ḃ)) / s is exact, s sizing s Ḃ to B.
// Zero when the direction moves no B.
Result<const double *> slab_tangent(DeviceResources &res, ScTangent &t, const size_t u,
                                    const SlabRead &r) {
  const int64_t count = slab_count(*t.b, r.block, r.b0, r.b1);
  auto found = std::ranges::find(t.regions, r.base, &TangentRegion::base);
  if (found == t.regions.end()) {
    Buffers::device<double> buf(static_cast<size_t>(count), res.device());
    t.regions.push_back({.base = r.base,
                         .count = count,
                         .dots = std::vector<const double *>(t.directions.size(), buf.data()),
                         .slab = buf.data()});
    t.held.push_back(std::move(buf));
    found = std::prev(t.regions.end());
  }
  if (count > found->count) {
    return err_config("nevpt2::class_energies_tangent: a slab of {} past its buffer's {}", count,
                      found->count);
  }
  double *const dot = found->slab;
  FBF_TRY(zero(res, dot, count));
  const double s = t.steps[u].slabs;
  if (s == 0.0) {
    return dot;
  }
  const MoBlocks &db = *t.directions[u].blocks;
  if (t.shifted_for != static_cast<int64_t>(u)) {
    FBF_TRY(shift_blocks(res, *t.b, db, s, *t.shifted));
    t.shifted_for = static_cast<int64_t>(u);
  }
  double *const plus = at_least(res, t.plus, count);
  double *const dd = at_least(res, t.dd, count);
  FBF_TRY(slab(res, *t.shifted, r.block, r.b0, r.b1, plus));
  FBF_TRY(slab(res, db, r.block, r.b0, r.b1, dd));
  FBF_TRY(add_scaled(res, plus, count, dot, 1.0 / s));
  FBF_TRY(add_scaled(res, r.base, count, dot, -1.0 / s));
  FBF_TRY(add_scaled(res, dd, count, dot, -s));
  return dot;
}

// Per direction, every slab of @p slabs at its tangent, then @p terms'
// tables' tangents along it, read back: [u] the norm table's, and H's.
Result<std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>>
tangent_tables(DeviceResources &res, ScTangent &t, const std::vector<ClassTerm> &terms,
               const std::vector<SlabRead> &slabs, const std::vector<int64_t> &out_dims) {
  const int64_t count = std::ranges::fold_left(out_dims, int64_t{1}, std::multiplies{});
  Buffers::device<double> norm_dot = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> h_dot = FBF_TRY(device_zeros(res, count));
  std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> out;
  for (size_t u = 0; u < t.directions.size(); ++u) {
    for (const SlabRead &r : slabs) {
      FBF_TRY(slab_tangent(res, t, u, r));
    }
    FBF_TRY(zero(res, norm_dot.data(), count));
    FBF_TRY(zero(res, h_dot.data(), count));
    FBF_TRY(tangent_terms(res, t, u, terms, norm_dot.data(), h_dot.data(), out_dims));
    out.first.push_back(FBF_TRY(download(res, norm_dot.data(), count)));
    out.second.push_back(FBF_TRY(download(res, h_dot.data(), count)));
  }
  return out;
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
                    const ScWalk &w, const int64_t first, double *const eq40,
                    const std::vector<SlabRead> &slabs) {
  const int64_t count = std::ranges::fold_left(out_dims, int64_t{1}, std::multiplies{});
  Buffers::device<double> norm_d = FBF_TRY(device_zeros(res, count));
  Buffers::device<double> h_d = FBF_TRY(device_zeros(res, count));
  FBF_TRY(eval_terms(res, terms, norm_d.data(), h_d.data(), out_dims));
  const std::vector<double> norm_t = FBF_TRY(download(res, norm_d.data(), count));
  const std::vector<double> h_t = FBF_TRY(download(res, h_d.data(), count));
  std::vector<double> p_t(static_cast<size_t>(count), 0.0);
  std::vector<double> q_t(static_cast<size_t>(count), 0.0);
  w.finish(first, perturbers, norm_t, h_t, p_t, q_t);
  if (w.tangent != nullptr) {
    const auto [norm_dots, h_dots] =
        FBF_TRY(tangent_tables(res, *w.tangent, terms, slabs, out_dims));
    w.tangent->finish(first, perturbers, norm_t, h_t, norm_dots, h_dots);
  }
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
    FBF_TRY(block_tangent(res, w, *a16, [&](const ActiveOperands &x) {
      return make_a16(res, x.h1e, x.h2e, x.dm3, x.f3ac, x.f3ca);
    }));
    FBF_TRY(block_tangent(res, w, *a17, [&](const ActiveOperands &x) {
      return make_a17(res, x.h1e, x.h2e, x.dm2, x.dm3);
    }));
    FBF_TRY(block_tangent(res, w, *a19, [&](const ActiveOperands &x) {
      return make_a19(res, x.h1e, x.h2e, x.dm1, x.dm2);
    }));
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
    FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, w, first, eq40.data(),
                         {{ExtBlock::Sr, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(block_tangent(res, w, *a22, [&](const ActiveOperands &x) {
      return make_a22(res, x.h1e, x.h2e, x.dm2, x.dm3, x.f3ac, x.f3ca);
    }));
    FBF_TRY(block_tangent(res, w, *a23, [&](const ActiveOperands &x) {
      return make_a23(res, x.h1e, x.h2e, x.dm1, x.dm2, x.dm3);
    }));
    FBF_TRY(block_tangent(res, w, *a25, [&](const ActiveOperands &x) {
      return make_a25(res, x.h1e, x.h2e, x.dm1, x.dm2);
    }));
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
  // Each side's, a pair's (N, M) its second.
  for (size_t side = 0; side < metric.size() / 3; ++side) {
    const bool second = side == 1;
    FBF_TRY(block_tangent(
        res, w, metric[3 * side],
        [&](const ActiveOperands &x) { return make_dm3_h(res, x.dm2, x.dm3, x.delta); }, second));
    FBF_TRY(block_tangent(
        res, w, metric[(3 * side) + 1],
        [&](const ActiveOperands &x) { return make_dm2_h(res, x.dm1, x.dm2, x.delta); }, second));
    FBF_TRY(block_tangent(
        res, w, metric[(3 * side) + 2],
        [&](const ActiveOperands &x) { return make_hdm1(res, x.dm1, x.delta); }, second));
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
    FBF_TRY(slab_reverse(res, terms, {nb}, perturbers, w, first, eq40.data(),
                         {{ExtBlock::Si, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(
        block_tangent(res, w, *a3, [&](const ActiveOperands &x) -> Result<Buffers::device<double>> {
          const Buffers::device<double> x_hdm1 = FBF_TRY(make_hdm1(res, x.dm1, x.delta));
          const EinsumOperand x_hdm1_op{.data = x_hdm1.data(), .dims = d2};
          return make_a3(res, x.h1e, x.h2e, x.dm1, x.dm2, x_hdm1_op, x.delta);
        }));
  }
  for (size_t side = 0; side < hdm1.size(); ++side) {
    FBF_TRY(block_tangent(
        res, w, hdm1[side], [&](const ActiveOperands &x) { return make_hdm1(res, x.dm1, x.delta); },
        side == 1));
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
    FBF_TRY(slab_reverse(res, terms, {nr, nc, nc}, perturbers, w, first, eq40.data(),
                         {{ExtBlock::Sijr, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(block_tangent(res, w, *k27, [&](const ActiveOperands &x) {
      return make_k27(res, x.h1e, x.h2e, x.dm1, x.dm2);
    }));
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

  const int64_t table = nv * nv * nc;
  Buffers::device<double> norm_d = FBF_TRY(device_zeros(res, table));
  Buffers::device<double> h_d = FBF_TRY(device_zeros(res, table));
  // With a tangent, each direction's whole tables' tangents too.
  const size_t ndir = w.tangent == nullptr ? 0 : w.tangent->directions.size();
  std::vector<Buffers::device<double>> norm_dots;
  std::vector<Buffers::device<double>> h_dots;
  for (size_t u = 0; u < ndir; ++u) {
    norm_dots.push_back(FBF_TRY(device_zeros(res, table)));
    h_dots.push_back(FBF_TRY(device_zeros(res, table)));
  }
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    FBF_TRY(slab(res, c.blocks, ExtBlock::Srsi, b0, b1, slab_a.data()));
    FBF_TRY(slab(res, c.blocks, ExtBlock::SrsiT, b0, b1, slab_b.data()));
    const std::vector<ClassTerm> slab_terms = terms(b0, b1, nullptr, nullptr);
    FBF_TRY(eval_terms(res, slab_terms, norm_d.data() + (b0 * nv * nc), h_d.data() + (b0 * nv * nc),
                       {b1 - b0, nv, nc}));
    for (size_t u = 0; u < ndir; ++u) {
      for (const SlabRead &r : {SlabRead{ExtBlock::Srsi, b0, b1, slab_a.data()},
                                SlabRead{ExtBlock::SrsiT, b0, b1, slab_b.data()}}) {
        FBF_TRY(slab_tangent(res, *w.tangent, u, r));
      }
      FBF_TRY(tangent_terms(res, *w.tangent, u, slab_terms, norm_dots[u].data() + (b0 * nv * nc),
                            h_dots[u].data() + (b0 * nv * nc), {b1 - b0, nv, nc}));
    }
  }
  const std::vector<double> norm_t = FBF_TRY(download(res, norm_d.data(), table));
  const std::vector<double> h_t = FBF_TRY(download(res, h_d.data(), table));
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
  if (ndir > 0) {
    std::vector<std::vector<double>> a_dots;
    std::vector<std::vector<double>> b_dots;
    for (size_t u = 0; u < ndir; ++u) {
      a_dots.push_back(FBF_TRY(download(res, norm_dots[u].data(), table)));
      b_dots.push_back(FBF_TRY(download(res, h_dots[u].data(), table)));
    }
    w.tangent->finish(0, perturbers, norm_t, h_t, a_dots, b_dots);
  }
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
    // make_a7 builds both; each tangent takes its own half.
    for (const bool first_half : {true, false}) {
      FBF_TRY(block_tangent(res, w, first_half ? rm2.front() : *a7,
                            [&](const ActiveOperands &x) -> Result<Buffers::device<double>> {
                              auto built =
                                  FBF_TRY(make_a7(res, x.h1e, x.h2e, x.dm1, x.dm2, x.dm3, x.delta));
                              return first_half ? std::move(built.first) : std::move(built.second);
                            }));
    }
  } else {
    for (const ActiveOperands *x : {w.a, w.b}) {
      rm2.push_back(FBF_TRY(make_rm2(res, x->dm1, x->dm2, x->delta)));
    }
    for (size_t side = 0; side < rm2.size(); ++side) {
      FBF_TRY(block_tangent(
          res, w, rm2[side],
          [&](const ActiveOperands &x) { return make_rm2(res, x.dm1, x.dm2, x.delta); },
          side == 1));
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
    FBF_TRY(slab_reverse(res, terms, {nb, nv}, perturbers, w, first, eq40.data(),
                         {{ExtBlock::Srs, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(
        block_tangent(res, w, *a9, [&](const ActiveOperands &x) -> Result<Buffers::device<double>> {
          const Buffers::device<double> x_hdm2 = FBF_TRY(make_hdm2(res, x.dm1, x.dm2, x.delta));
          const EinsumOperand x_hdm2_op{.data = x_hdm2.data(), .dims = d4};
          const Buffers::device<double> x_hdm3 =
              FBF_TRY(make_hdm3(res, x.dm1, x.dm2, x.dm3, x_hdm2_op, x.delta));
          const EinsumOperand x_hdm3_op{.data = x_hdm3.data(), .dims = std::vector<int64_t>(6, n)};
          return make_a9(res, x.h1e, x.h2e, x_hdm2_op, x_hdm3_op);
        }));
  }
  for (size_t side = 0; side < hdm2.size(); ++side) {
    FBF_TRY(block_tangent(
        res, w, hdm2[side],
        [&](const ActiveOperands &x) { return make_hdm2(res, x.dm1, x.dm2, x.delta); }, side == 1));
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
    FBF_TRY(slab_reverse(res, terms, {nb, nc}, perturbers, w, first, eq40.data(),
                         {{ExtBlock::Sij, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(block_tangent(res, w, *a12, [&](const ActiveOperands &x) {
      return make_a12(res, x.h1e, x.h2e, x.dm2, x.dm3);
    }));
    FBF_TRY(block_tangent(res, w, *a13, [&](const ActiveOperands &x) {
      return make_a13(res, x.h1e, x.h2e, x.dm1, x.dm2, x.dm3, x.delta);
    }));
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
    FBF_TRY(slab_reverse(
        res, terms, {nc, nb}, perturbers, w, first, eq40.data(),
        {{ExtBlock::Sir1, b0, b1, slab_1.data()}, {ExtBlock::Sir2, b0, b1, slab_2.data()}}));
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
  const auto &[h1e, h2e, dm1, dm2, dm3, f3ca, f3ac, delta, delta_buf, e_core, e_virt, pc_floor,
               nelecas] = o;
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
// put there. A one-function class (single_function_class) has @p eta, its
// η^J from its blocks @p xy, and D_J = Δ + η^J for every perturber instead.
struct KetTables {
  std::vector<double> norm, h, a;
  std::array<std::vector<double>, 2> xy;
  std::optional<double> eta;
};

// R's host sums @p core and @p virt onto @p bars' orbital energies', when
// wanted.
Status add_orbital_energy_bars(DeviceResources &res, const std::vector<double> &core,
                               const std::vector<double> &virt, const EnergyBars &bars) {
  for (const auto &[host, bar] : {std::pair{&core, bars.e_core}, std::pair{&virt, bars.e_virt}}) {
    if (bar != nullptr && !host->empty()) {
      const Buffers::device<double> d = FBF_TRY(upload(res, *host));
      FBF_TRY(einsum(res.stream(), "i->i", {{.data = d.data(), .dims = {std::ssize(*host)}}}, bar,
                     {std::ssize(*host)}, 1.0));
      FBF_TRY(res.sync());
    }
  }
  return {};
}

// Every SC class's walk but Sijrs's, in PySCF's order, with its class index.
const std::array<std::pair<size_t, Result<double> (*)(ReverseContext &, const ScWalk &)>, 7>
    kScWalks{{{0, reverse_sr},
              {1, reverse_si},
              {3, reverse_sijr},
              {4, reverse_srsi},
              {5, reverse_srs},
              {6, reverse_sij},
              {7, reverse_sir}}};

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
  const auto &walks = kScWalks;
  ScEffectiveHamiltonianAdjoint out;
  out.elements.resize(static_cast<size_t>(ns * ns));
  std::array<std::vector<KetTables>, 8> kets;
  for (std::vector<KetTables> &k : kets) {
    k.resize(static_cast<size_t>(ns));
  }
  // A one-function class's η, per state, from its own blocks.
  for (int64_t j = 0; j < ns; ++j) {
    for (const auto &[k, walk] : walks) {
      if (single_function_class(ops[at(j, j)], k)) {
        KetTables &kt = kets[k][static_cast<size_t>(j)];
        kt.xy = FBF_TRY(single_function_blocks(res, ops[at(j, j)], k));
        kt.eta = single_function_ratio(kt.xy);
      }
    }
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
                             if (!kt.eta && std::abs(kt.norm[g]) <= kNumericalZero) {
                               continue;
                             }
                             const double d = perturber_delta(p, c.e) +
                                              (kt.eta ? *kt.eta : (kt.h[g] / kt.norm[g]));
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
      // A one-function class: E_J = Σ_t N_t c_t, c_t = -1 / (Δ_t + η). The
      // bar on each D_t = Δ_t + η is H̄_JJ N_t c_t² + A_t, onto Δ_t and,
      // summed into η̄, onto η, which goes to the class's blocks after the
      // walk; beside it H̄_JJ c_t on N_t, and nothing on H_t.
      double eta_bar = 0.0;
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
              if (kt.eta) {
                const double cg = -1.0 / (perturber_delta(p, c.e) + *kt.eta);
                const double ap = kt.a.empty() ? 0.0 : kt.a[static_cast<size_t>(first) + x];
                const double d_bar = (weight * norm * cg * cg) + ap;
                element.energies[k] += norm * cg;
                spread(p, weight * cg, a_bar);
                add_delta_bar(p, d_bar, c.e);
                eta_bar += d_bar;
                continue;
              }
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
      if (kt.eta) {
        FBF_TRY(add_single_function_bars(res, k, blocks.ncas,
                                         single_function_eta_bars(kt.xy, *kt.eta, eta_bar), m));
      }
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
  FBF_TRY(add_orbital_energy_bars(res, e_core_bar, e_virt_bar, c.bars));
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
// The SC forward tangent: each own walk with a tangent, and Sijrs's kernel
// ---------------------------------------------------------------------------

// Perturber @p x of a class-@p k slab whose first is @p first, in
// ClassEnergiesTangent's order: Sir's at r ncore + i (its tables run (i, r)
// per slab), every other class's in walk order, whatever the batch.
size_t perturber_index(const size_t k, const int64_t first, const size_t x, const Perturber &p,
                       const int64_t ncore) {
  return k == 7 ? static_cast<size_t>((int64_t{p.v1} * ncore) + p.c1)
                : static_cast<size_t>(first) + x;
}

// Σ x² over @p count elements at @p x, added to the device scalar @p acc;
// nothing for a null @p x.
Status add_squares(DeviceResources &res, const double *const x, const int64_t count,
                   double *const acc) {
  if (x == nullptr || count == 0) {
    return {};
  }
  return einsum(res.stream(), "a,a->", {{.data = x, .dims = {count}}, {.data = x, .dims = {count}}},
                acc, {}, 1.0);
}

// The power of two nearest @p x / @p dx, the step that sizes s dx to x; 1
// when x is zero, and 0 (no tangent) when dx is.
double polarization_step(const double x, const double dx) {
  if (dx == 0.0) {
    return 0.0;
  }
  return x == 0.0 ? 1.0 : std::exp2(std::round(std::log2(x / dx)));
}

// Each direction's polarization steps over @p o's arrays and @p blocks'.
// The blocks' s is the RDMs' ratio ‖r‖ / ‖ṙ‖ when the RDMs move, else the
// integrals': with a block one integral h times one RDM r, the difference's
// round-off is ε (‖h‖ + s ‖ḣ‖)(‖r‖ + s ‖ṙ‖) / s, and at s = ‖r‖ / ‖ṙ‖ that is
// 2ε (‖h‖ ‖ṙ‖ + ‖ḣ‖ ‖r‖), twice the tangent's own scale. A slab's is
// ‖B‖ / ‖Ḃ‖ the same way.
Result<std::vector<TangentSteps>> tangent_steps(DeviceResources &res, const MoBlocks &blocks,
                                                const EnergyInputs &in,
                                                std::span<const EnergyDirection> directions) {
  const int64_t n = blocks.ncas;
  const int64_t n2 = n * n;
  const int64_t n4 = n2 * n2;
  const int64_t n6 = n4 * n2;
  // [3 v + 0] the RDMs' Σ x², [3 v + 1] the integrals', [3 v + 2] B's, for
  // v = 0 the inputs and v = 1 + u direction u.
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 3 * std::ssize(directions) + 3));
  const auto squares = [&](const EnergyInputs &x, const MoBlocks *b, double *const at) -> Status {
    for (const auto &[a, count] : {std::pair{x.dm1, n2}, std::pair{x.dm2, n4}, std::pair{x.dm3, n6},
                                   std::pair{x.f3ca, n6}, std::pair{x.f3ac, n6}}) {
      FBF_TRY(add_squares(res, a, count, at));
    }
    FBF_TRY(add_squares(res, x.h1e, n2, at + 1));
    FBF_TRY(add_squares(res, x.h2e, n4, at + 1));
    if (b != nullptr) {
      for (const Buffers::device<double> *g : {&b->aa, &b->ca, &b->va, &b->cv}) {
        FBF_TRY(add_squares(res, g->data(), static_cast<int64_t>(g->num_elements()), at + 2));
      }
    }
    return {};
  };
  FBF_TRY(squares(in, &blocks, acc.data()));
  for (size_t u = 0; u < directions.size(); ++u) {
    FBF_TRY(squares(directions[u].in, directions[u].blocks, acc.data() + (3 * (u + 1))));
  }
  const std::vector<double> sq = FBF_TRY(download(res, acc.data(), 3 * std::ssize(directions) + 3));
  std::vector<TangentSteps> out;
  for (size_t u = 0; u < directions.size(); ++u) {
    const auto norm = [&](const size_t v, const size_t k) { return std::sqrt(sq[(3 * v) + k]); };
    const double rdm = polarization_step(norm(0, 0), norm(u + 1, 0));
    out.push_back({.blocks = rdm != 0.0 ? rdm : polarization_step(norm(0, 1), norm(u + 1, 1)),
                   .slabs = polarization_step(norm(0, 2), norm(u + 1, 2))});
  }
  return out;
}

// Sijrs's norm, energy and Ė per direction, slab by slab: device::sijrs and
// device::sijrs_tangent over each slab and its tangent along each direction.
Status sijrs_tangent_walk(ReverseContext &c, ScTangent &t, ClassEnergiesTangent &out) {
  DeviceResources &res = c.res;
  const MoBlocks &blocks = c.blocks;
  if (blocks.ncore == 0 || blocks.nvirt == 0) {
    return {};
  }
  const auto m = std::ssize(t.directions);
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 2 + m));
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijrs);
  const int32_t step = batch_step(extent, c.batch);
  Buffers::device<double> slab_buf = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    FBF_TRY(slab(res, blocks, ExtBlock::Sijrs, b0, b1, slab_buf.data()));
    FBF_TRY(gpu_status(device::sijrs(res.stream(), slab_buf.data(), b1 - b0, b0, blocks.ncore,
                                     blocks.nvirt, c.in.e_core, c.in.e_virt, acc.data())));
    for (int64_t u = 0; u < m; ++u) {
      const EnergyInputs &d = t.directions[static_cast<size_t>(u)].in;
      const double *const dot = FBF_TRY(
          slab_tangent(res, t, static_cast<size_t>(u), {ExtBlock::Sijrs, b0, b1, slab_buf.data()}));
      FBF_TRY(gpu_status(device::sijrs_tangent(res.stream(), slab_buf.data(), dot, b1 - b0, b0,
                                               blocks.ncore, blocks.nvirt, c.in.e_core, c.in.e_virt,
                                               d.e_core, d.e_virt, acc.data() + 2 + u)));
    }
  }
  const std::vector<double> host = FBF_TRY(download(res, acc.data(), 2 + m));
  out.energies.norms[2] = host[0];
  out.energies.energies[2] = host[1];
  for (int64_t u = 0; u < m; ++u) {
    out.energy_dot[static_cast<size_t>(u)][2] = host[static_cast<size_t>(2 + u)];
  }
  return {};
}

// InvalidConfig unless every direction's blocks, when given, are shaped as
// @p blocks; @p who names the call.
Status check_directions(const MoBlocks &blocks, std::span<const EnergyDirection> directions,
                        const std::string_view who = "class_energies_tangent") {
  for (size_t u = 0; u < directions.size(); ++u) {
    const MoBlocks *const d = directions[u].blocks;
    if (d == nullptr) {
      continue;
    }
    const bool shaped = d->naux == blocks.naux && d->ncore == blocks.ncore &&
                        d->ncas == blocks.ncas && d->nvirt == blocks.nvirt &&
                        d->aa.num_elements() == blocks.aa.num_elements() &&
                        d->ca.num_elements() == blocks.ca.num_elements() &&
                        d->va.num_elements() == blocks.va.num_elements() &&
                        d->cv.num_elements() == blocks.cv.num_elements() &&
                        d->vc.num_elements() == blocks.vc.num_elements();
    if (!shaped) {
      return err_config("nevpt2::{}: direction {}'s blocks are not shaped as "
                        "the blocks' (naux {}, ncore {}, ncas {}, nvirt {})",
                        who, u, blocks.naux, blocks.ncore, blocks.ncas, blocks.nvirt);
    }
  }
  return {};
}

// A tangent over @p in along @p directions: each direction's steps, the
// inputs the class terms read directly as its first regions, and with any
// direction moving B, the B + s Ḃ buffer. The classes' own regions follow.
Result<ScTangent> input_tangent(DeviceResources &res, const MoBlocks &blocks,
                                const EnergyInputs &in,
                                std::span<const EnergyDirection> directions) {
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;
  ScTangent t{.directions = directions,
              .b = &blocks,
              .steps = FBF_TRY(tangent_steps(res, blocks, in, directions))};
  const auto input = [&](const double *const base, const int64_t count,
                         const double *EnergyInputs::*field) {
    TangentRegion r{.base = base, .count = count};
    for (const EnergyDirection &d : directions) {
      r.dots.push_back(d.in.*field);
    }
    t.regions.push_back(std::move(r));
  };
  input(in.h1e, n * n, &EnergyInputs::h1e);
  input(in.h2e, n * n * n * n, &EnergyInputs::h2e);
  input(in.dm1, n * n, &EnergyInputs::dm1);
  input(in.dm2, n * n * n * n, &EnergyInputs::dm2);
  input(in.dm3, n * n * n * n * n * n, &EnergyInputs::dm3);
  input(in.f3ca, n * n * n * n * n * n, &EnergyInputs::f3ca);
  input(in.f3ac, n * n * n * n * n * n, &EnergyInputs::f3ac);
  input(in.h1e_v_sr, nv * n, &EnergyInputs::h1e_v_sr);
  input(in.h1e_v_si, n * nc, &EnergyInputs::h1e_v_si);
  input(in.h1e_v_sir, nv * nc, &EnergyInputs::h1e_v_sir);
  t.inputs = t.regions.size();
  if (std::ranges::any_of(directions,
                          [](const EnergyDirection &d) { return d.blocks != nullptr; })) {
    t.shifted.emplace(
        MoBlocks{.naux = blocks.naux,
                 .ncore = blocks.ncore,
                 .ncas = blocks.ncas,
                 .nvirt = blocks.nvirt,
                 .aa = Buffers::device<double>(blocks.aa.num_elements(), res.device()),
                 .ca = Buffers::device<double>(blocks.ca.num_elements(), res.device()),
                 .va = Buffers::device<double>(blocks.va.num_elements(), res.device()),
                 .cv = Buffers::device<double>(blocks.cv.num_elements(), res.device()),
                 .vc = Buffers::device<double>(blocks.vc.num_elements(), res.device())});
  }
  return t;
}

// Each direction's tangent of the orbital energies @p field (@p count of
// them), on the host, for Δ̇: zero where the direction's is null.
Result<std::vector<std::vector<double>>>
orbital_energy_dots(DeviceResources &res, std::span<const EnergyDirection> directions,
                    const double *EnergyInputs::*field, const int64_t count) {
  std::vector<std::vector<double>> out;
  for (const EnergyDirection &d : directions) {
    out.push_back(d.in.*field == nullptr ? std::vector<double>(static_cast<size_t>(count), 0.0)
                                         : FBF_TRY(download(res, d.in.*field, count)));
  }
  return out;
}

Result<ClassEnergiesTangent> class_energies_tangent(DeviceResources &res, const MoBlocks &blocks,
                                                    const EnergyInputs &in, const int32_t batch,
                                                    std::span<const EnergyDirection> directions) {
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(check_directions(blocks, directions));
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  const size_t m = directions.size();
  const int64_t n = blocks.ncas;
  const int64_t nc = blocks.ncore;
  const int64_t nv = blocks.nvirt;

  ScTangent t = FBF_TRY(input_tangent(res, blocks, in, directions));
  const std::vector<std::vector<double>> e_core_dot =
      FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_core, nc));
  const std::vector<std::vector<double>> e_virt_dot =
      FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_virt, nv));

  ClassEnergiesTangent out;
  out.energy_dot.resize(m);
  out.total_dot.resize(m);
  std::vector<double> unused_core;
  std::vector<double> unused_virt;
  const EnergyBars no_bars{};
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = no_bars,
      .batch = batch,
      .ops = o,
      .e = {.core = o.e_core, .virt = o.e_virt, .core_bar = unused_core, .virt_bar = unused_virt}};
  BlockBars none = FBF_TRY(block_bars(res, n, false, false));
  for (const auto &[k, walk] : kScWalks) {
    // A one-function class: E = Σ_t N_t c_t, c_t = -1 / (Δ_t + η), so
    // Ė = Σ_t c_t Ṅ_t + N_t c_t² (Δ̇_t + η̇), η̇ from its blocks' tangents.
    std::optional<double> eta;
    std::vector<double> eta_dot;
    if (single_function_class(o, k)) {
      const std::array<std::vector<double>, 2> xy = FBF_TRY(single_function_blocks(res, o, k));
      eta = single_function_ratio(xy);
      if (eta) {
        for (const auto &dot :
             FBF_TRY(single_function_block_dots(res, o, k, directions, t.steps))) {
          eta_dot.push_back(single_function_ratio_dot(xy, *eta, dot));
        }
      }
    }
    // Each perturber's N, H and Δ, their tangents, and the finish's in the
    // (N, d) form, d = N Δ + H, E = -N² / d: no 1/N, which the guard's
    // neighbourhood would make fatal. Sir's tables run (i, r) per slab;
    // its perturbers land at r ncore + i, whatever the batch.
    t.finish = [&, k](const int64_t first, const std::vector<Perturber> &perturbers,
                      const std::vector<double> &a, const std::vector<double> &b,
                      const std::vector<std::vector<double>> &a_dots,
                      const std::vector<std::vector<double>> &b_dots) {
      for (size_t x = 0; x < perturbers.size(); ++x) {
        const Perturber &p = perturbers[x];
        const size_t g = perturber_index(k, first, x, p, nc);
        if (out.norm[k].size() <= g) {
          for (std::vector<double> *table : {&out.norm[k], &out.h[k], &out.delta[k]}) {
            table->resize(g + 1, 0.0);
          }
          for (std::vector<double> *table : {&out.norm_dot[k], &out.h_dot[k], &out.delta_dot[k]}) {
            table->resize((g + 1) * m, 0.0);
          }
        }
        const double norm = combined(p, a);
        const double h = combined(p, b);
        const double diff = perturber_delta(p, c.e);
        out.norm[k][g] = norm;
        out.h[k][g] = h;
        out.delta[k][g] = diff;
        out.energies.norms[k] += norm;
        const bool kept = !eta && std::abs(norm) > kNumericalZero;
        if (kept) {
          out.energies.energies[k] -= norm / (diff + (h / norm));
        }
        const double cg = eta ? -1.0 / (diff + *eta) : 0.0;
        if (eta) {
          out.energies.energies[k] += norm * cg;
        }
        const double d = (norm * diff) + h;
        for (size_t u = 0; u < m; ++u) {
          const double norm_dot = combined(p, a_dots[u]);
          const double h_dot = combined(p, b_dots[u]);
          const double diff_dot = perturber_delta(p, {.core = e_core_dot[u],
                                                      .virt = e_virt_dot[u],
                                                      .core_bar = unused_core,
                                                      .virt_bar = unused_virt});
          out.norm_dot[k][(g * m) + u] = norm_dot;
          out.h_dot[k][(g * m) + u] = h_dot;
          out.delta_dot[k][(g * m) + u] = diff_dot;
          if (eta) {
            out.energy_dot[u][k] += (cg * norm_dot) + (norm * cg * cg * (diff_dot + eta_dot[u]));
          }
          if (kept) {
            const double e_n = (-2.0 * norm / d) + (norm * norm * diff / (d * d));
            const double e_h = norm * norm / (d * d);
            const double e_delta = norm * norm * norm / (d * d);
            out.energy_dot[u][k] += (e_n * norm_dot) + (e_h * h_dot) + (e_delta * diff_dot);
          }
        }
      }
    };
    const ScWalk w{.a = &o,
                   .m_a = &none,
                   .adjoint = false,
                   .finish = [](int64_t, const std::vector<Perturber> &,
                                const std::vector<double> &, const std::vector<double> &,
                                std::vector<double> &, std::vector<double> &) {},
                   .tangent = &t};
    FBF_TRY(walk(c, w));
    // The class's blocks and slab buffers are freed: drop their regions.
    t.regions.resize(t.inputs);
    t.held.clear();
  }
  FBF_TRY(sijrs_tangent_walk(c, t, out));
  out.energies.total = std::ranges::fold_left(out.energies.energies, 0.0, std::plus{});
  for (size_t u = 0; u < m; ++u) {
    out.total_dot[u] = std::ranges::fold_left(out.energy_dot[u], 0.0, std::plus{});
  }
  return out;
}

// ---------------------------------------------------------------------------
// The SC-QD forward tangent: each state's own walk, then each pair's
// ---------------------------------------------------------------------------

// One unordered pair's transition tables and their tangents, [k][t] and
// [k][t m + u] over m = da.size(): T and Ṫ, the means of the norm einsums of
// @p mn, (M, N), along @p da and of @p nm, (N, M), along @p db.
struct PairTables {
  std::array<std::vector<double>, 8> t, t_dot;
};

// Every class's slabs walked once for all the directions, the norm einsums
// of (M, N) and (N, M) at their operands' tangents, the (N, M) side's blocks
// along its own directions: sc_couplings_tangent's pair walk.
Result<PairTables> pair_tables(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &mn,
                               const EnergyInputs &nm, const int32_t batch,
                               std::span<const EnergyDirection> da,
                               std::span<const EnergyDirection> db) {
  const int64_t n = blocks.ncas;
  const size_t m = da.size();
  const ActiveOperands a = FBF_TRY(active_operands(res, blocks, mn));
  const ActiveOperands b = FBF_TRY(active_operands(res, blocks, nm));
  std::vector<double> unused_core;
  std::vector<double> unused_virt;
  const EnergyBars no_bars{};
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = mn,
      .bars = no_bars,
      .batch = batch,
      .ops = a,
      .e = {.core = a.e_core, .virt = a.e_virt, .core_bar = unused_core, .virt_bar = unused_virt}};
  BlockBars none = FBF_TRY(block_bars(res, n, false, false));
  ScTangent t = FBF_TRY(input_tangent(res, blocks, mn, da));
  t.b_directions = db;
  t.b_steps = FBF_TRY(tangent_steps(res, blocks, nm, db));
  // (N, M)'s RDMs, which its side's terms read directly (Sr's, Srsi's and
  // Sir's metrics).
  for (const auto &[field, count] :
       {std::pair{&EnergyInputs::dm1, n * n}, std::pair{&EnergyInputs::dm2, n * n * n * n},
        std::pair{&EnergyInputs::dm3, n * n * n * n * n * n}}) {
    TangentRegion r{.base = nm.*field, .count = count};
    for (const EnergyDirection &d : db) {
      r.dots.push_back(d.in.*field);
    }
    t.regions.push_back(std::move(r));
  }
  t.inputs = t.regions.size();
  PairTables out;
  for (const auto &[k, walk] : kScWalks) {
    std::vector<double> &table = out.t[k];
    std::vector<double> &table_dot = out.t_dot[k];
    t.finish = [&, k](const int64_t first, const std::vector<Perturber> &perturbers,
                      const std::vector<double> &ta, const std::vector<double> &tb,
                      const std::vector<std::vector<double>> &a_dots,
                      const std::vector<std::vector<double>> &b_dots) {
      for (size_t x = 0; x < perturbers.size(); ++x) {
        const Perturber &p = perturbers[x];
        const size_t g = perturber_index(k, first, x, p, blocks.ncore);
        if (table.size() <= g) {
          table.resize(g + 1, 0.0);
          table_dot.resize((g + 1) * m, 0.0);
        }
        table[g] = 0.5 * (combined(p, ta) + combined(p, tb));
        for (size_t u = 0; u < m; ++u) {
          table_dot[(g * m) + u] = 0.5 * (combined(p, a_dots[u]) + combined(p, b_dots[u]));
        }
      }
    };
    const ScWalk w{.a = &a,
                   .m_a = &none,
                   .b = &b,
                   .m_b = &none,
                   .adjoint = false,
                   .finish = [](int64_t, const std::vector<Perturber> &,
                                const std::vector<double> &, const std::vector<double> &,
                                std::vector<double> &, std::vector<double> &) {},
                   .tangent = &t};
    FBF_TRY(walk(c, w));
    // The class's blocks and slab buffers are freed: drop their regions.
    t.regions.resize(t.inputs);
    t.held.clear();
  }
  return out;
}

Result<ScEffectiveHamiltonianTangent>
sc_couplings_tangent(DeviceResources &res, const MoBlocks &blocks,
                     std::span<const EnergyInputs> pairs, const int64_t ns, const int32_t batch,
                     std::span<const EnergyDirection> directions) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n2 = static_cast<size_t>(ns * ns);
  const size_t m = directions.size() / n2;
  const int64_t n = blocks.ncas;
  const auto at = [ns](const int64_t i, const int64_t j) {
    return static_cast<size_t>((i * ns) + j);
  };
  // Pair @p p's directions, one per u.
  const auto along = [&](const size_t p) {
    std::vector<EnergyDirection> d;
    for (size_t u = 0; u < m; ++u) {
      d.push_back(directions[(u * n2) + p]);
    }
    return d;
  };
  std::vector<ActiveOperands> ops;
  for (const EnergyInputs &pair : pairs) {
    ops.push_back(FBF_TRY(active_operands(res, blocks, pair)));
  }
  ScEffectiveHamiltonianTangent out;
  out.elements.resize(n2);
  out.transition.resize(n2);
  out.transition_dot.resize(n2);
  out.empty.resize(static_cast<size_t>(ns));
  out.element_dot.assign(m, std::vector<std::array<double, 8>>(n2));
  out.total_dot.assign(m, std::vector<double>(n2, 0.0));

  // 1. Each state's own walk, the diagonal element and the tables every
  //    coupling with that ket reads; a one-function class's η^J and η̇^J,
  //    and the tuples the guard drops.
  std::array<std::vector<std::optional<double>>, 8> eta;
  std::array<std::vector<std::vector<double>>, 8> eta_dot;
  for (int64_t j = 0; j < ns; ++j) {
    const size_t jj = at(j, j);
    const std::vector<EnergyDirection> d = along(jj);
    ClassEnergiesTangent kt = FBF_TRY(class_energies_tangent(res, blocks, pairs[jj], batch, d));
    out.elements[jj] = kt.energies;
    for (size_t u = 0; u < m; ++u) {
      out.element_dot[u][jj] = kt.energy_dot[u];
      out.total_dot[u][jj] = kt.total_dot[u];
    }
    out.transition[jj] = kt.norm;
    out.transition_dot[jj] = kt.norm_dot;
    const std::vector<TangentSteps> steps = FBF_TRY(tangent_steps(res, blocks, pairs[jj], d));
    for (const auto &[k, walk] : kScWalks) {
      std::optional<double> &e = eta[k].emplace_back();
      std::vector<double> &e_dot = eta_dot[k].emplace_back();
      if (single_function_class(ops[jj], k)) {
        const std::array<std::vector<double>, 2> xy =
            FBF_TRY(single_function_blocks(res, ops[jj], k));
        e = single_function_ratio(xy);
        if (e) {
          for (const auto &dot : FBF_TRY(single_function_block_dots(res, ops[jj], k, d, steps))) {
            e_dot.push_back(single_function_ratio_dot(xy, *e, dot));
          }
        }
      }
      if (!e) {
        for (size_t g = 0; g < kt.norm[k].size(); ++g) {
          if (std::abs(kt.norm[k][g]) <= kNumericalZero) {
            out.empty[static_cast<size_t>(j)][k].push_back(static_cast<int64_t>(g));
          }
        }
      }
    }
    out.kets.push_back(std::move(kt));
  }

  // 2. Each unordered pair M < N: every class's slabs walked once, its two
  //    norm einsums' tables and their tangents, the (N, M) side's blocks
  //    along its own directions, T and Ṫ their means.
  for (int64_t mi = 0; mi < ns; ++mi) {
    for (int64_t ni = mi + 1; ni < ns; ++ni) {
      const size_t mn = at(mi, ni);
      const size_t nm = at(ni, mi);
      PairTables t =
          FBF_TRY(pair_tables(res, blocks, pairs[mn], pairs[nm], batch, along(mn), along(nm)));
      out.transition[nm] = t.t;
      out.transition_dot[nm] = t.t_dot;
      out.transition[mn] = std::move(t.t);
      out.transition_dot[mn] = std::move(t.t_dot);
    }
  }

  // 3. Each coupling (I, J), I != J, over ket J's tables: C = -T N / d,
  //    d = N Δ + H, and Ċ in that (N, d) form; a one-function class at η^J.
  for (int64_t bra = 0; bra < ns; ++bra) {
    for (int64_t ket = 0; ket < ns; ++ket) {
      if (bra == ket) {
        continue;
      }
      const size_t p = at(bra, ket);
      const auto j = static_cast<size_t>(ket);
      const ClassEnergiesTangent &kt = out.kets[j];
      ClassEnergies &element = out.elements[p];
      for (const auto &[k, walk] : kScWalks) {
        const std::vector<double> &table = out.transition[p][k];
        const std::vector<double> &table_dot = out.transition_dot[p][k];
        const std::optional<double> &e = eta[k][j];
        for (size_t g = 0; g < table.size(); ++g) {
          const double tt = table[g];
          const double norm = kt.norm[k][g];
          const double h = kt.h[k][g];
          const double diff = kt.delta[k][g];
          element.norms[k] += tt;
          if (e) {
            const double cg = -1.0 / (diff + *e);
            element.energies[k] += tt * cg;
            for (size_t u = 0; u < m; ++u) {
              out.element_dot[u][p][k] +=
                  (cg * table_dot[(g * m) + u]) +
                  (tt * cg * cg * (kt.delta_dot[k][(g * m) + u] + eta_dot[k][j][u]));
            }
            continue;
          }
          if (std::abs(norm) <= kNumericalZero) {
            continue;
          }
          element.energies[k] -= tt / (diff + (h / norm));
          const double d = (norm * diff) + h;
          for (size_t u = 0; u < m; ++u) {
            const size_t gu = (g * m) + u;
            out.element_dot[u][p][k] += (-(norm / d) * table_dot[gu]) -
                                        (tt * h / (d * d) * kt.norm_dot[k][gu]) +
                                        (tt * norm / (d * d) * kt.h_dot[k][gu]) +
                                        (tt * norm * norm / (d * d) * kt.delta_dot[k][gu]);
          }
        }
      }
      element.total = std::ranges::fold_left(element.energies, 0.0, std::plus{});
      for (size_t u = 0; u < m; ++u) {
        out.total_dot[u][p] = std::ranges::fold_left(out.element_dot[u][p], 0.0, std::plus{});
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The SC Hessian-vector product: forward over the reverse pass
// ---------------------------------------------------------------------------

// Fixed multipliers per class and perturber (perturber_index's order): P on
// N, Q on H and R on Δ. An empty r puts nothing on the orbital energies.
// A one-function class's bars on its blocks through η are @p blocks[k]
// (single_function_eta_bars' layout), empty for any other class.
struct ScMultipliers {
  std::array<std::vector<double>, 8> p, q, r;
  std::array<std::array<std::vector<double>, 2>, 8> blocks;
};

// The SC reverse pass at @p mult in place of the finish's own multipliers:
// @p bars += Σ_t (P_t ∂N_t + Q_t ∂H_t + R_t ∂Δ_t) over every class but
// Sijrs, through the same walks, sweep and host sums as
// class_energies_adjoint's own walk.
Status fixed_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                     const int32_t batch, const EnergyBars &bars, const ScMultipliers &mult) {
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  std::vector<double> e_core_bar(o.e_core.size(), 0.0);
  std::vector<double> e_virt_bar(o.e_virt.size(), 0.0);
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = bars,
      .batch = batch,
      .ops = o,
      .e = {.core = o.e_core, .virt = o.e_virt, .core_bar = e_core_bar, .virt_bar = e_virt_bar}};
  BlockBars m = FBF_TRY(block_bars(res, blocks.ncas, true, true));
  const int64_t nc = blocks.ncore;
  for (const auto &[k, walk] : kScWalks) {
    const ScWalk w{.a = &o,
                   .m_a = &m,
                   .finish = [&, k](const int64_t first, const std::vector<Perturber> &perturbers,
                                    const std::vector<double> &, const std::vector<double> &,
                                    std::vector<double> &a_bar, std::vector<double> &b_bar) {
                     for (size_t x = 0; x < perturbers.size(); ++x) {
                       const Perturber &p = perturbers[x];
                       const size_t g = perturber_index(k, first, x, p, nc);
                       spread(p, mult.p[k][g], a_bar);
                       spread(p, mult.q[k][g], b_bar);
                       if (!mult.r[k].empty()) {
                         add_delta_bar(p, mult.r[k][g], c.e);
                       }
                     }
                   }};
    FBF_TRY(walk(c, w));
    if (!mult.blocks[k][0].empty()) {
      FBF_TRY(add_single_function_bars(res, k, blocks.ncas, mult.blocks[k], m));
    }
  }
  const std::array wanted{bars.h1e, bars.h2e, bars.dm1, bars.dm2, bars.dm3, bars.f3ca, bars.f3ac};
  if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
    FBF_TRY(sweep_blocks(res, o, m, bars));
  }
  return add_orbital_energy_bars(res, e_core_bar, e_virt_bar, bars);
}

// Every EnergyInputs field with its element count over @p b's sizes.
std::array<std::pair<const double * EnergyInputs::*, int64_t>, 12> input_fields(const MoBlocks &b) {
  const int64_t n = b.ncas;
  const int64_t n2 = n * n;
  const int64_t n4 = n2 * n2;
  const int64_t n6 = n4 * n2;
  return {{{&EnergyInputs::h1e, n2},
           {&EnergyInputs::h2e, n4},
           {&EnergyInputs::e_core, b.ncore},
           {&EnergyInputs::e_virt, b.nvirt},
           {&EnergyInputs::h1e_v_sr, int64_t{b.nvirt} * n},
           {&EnergyInputs::h1e_v_si, n * b.ncore},
           {&EnergyInputs::h1e_v_sir, int64_t{b.nvirt} * b.ncore},
           {&EnergyInputs::dm1, n2},
           {&EnergyInputs::dm2, n4},
           {&EnergyInputs::dm3, n6},
           {&EnergyInputs::f3ca, n6},
           {&EnergyInputs::f3ac, n6}}};
}

// Every EnergyBars field with its element count over @p b's sizes, Λ's
// naux x nmo².
std::array<std::pair<double * EnergyBars::*, int64_t>, 13> bar_fields(const MoBlocks &b) {
  const int64_t nmo = int64_t{b.ncore} + b.ncas + b.nvirt;
  const auto in = input_fields(b);
  return {{{&EnergyBars::h1e, in[0].second},
           {&EnergyBars::h2e, in[1].second},
           {&EnergyBars::e_core, in[2].second},
           {&EnergyBars::e_virt, in[3].second},
           {&EnergyBars::h1e_v_sr, in[4].second},
           {&EnergyBars::h1e_v_si, in[5].second},
           {&EnergyBars::h1e_v_sir, in[6].second},
           {&EnergyBars::dm1, in[7].second},
           {&EnergyBars::dm2, in[8].second},
           {&EnergyBars::dm3, in[9].second},
           {&EnergyBars::f3ca, in[10].second},
           {&EnergyBars::f3ac, in[11].second},
           {&EnergyBars::lambda, int64_t{b.naux} * nmo * nmo}}};
}

// @p in and @p blocks moved by @p step along @p d, in buffers of their own;
// a field whose tangent is null stays @p in's, and the blocks stay
// @p blocks when @p d moves no B.
struct ShiftedInputs {
  std::vector<Buffers::device<double>> arrays;
  std::optional<MoBlocks> own;
  const MoBlocks *base = nullptr;
  EnergyInputs in;
  [[nodiscard]] const MoBlocks &blocks() const { return own ? *own : *base; }
};

Result<ShiftedInputs> shifted_inputs(DeviceResources &res, const MoBlocks &blocks,
                                     const EnergyInputs &in, const EnergyDirection &d,
                                     const double step) {
  ShiftedInputs out{.base = &blocks, .in = in};
  for (const auto &[field, count] : input_fields(blocks)) {
    if (in.*field == nullptr || d.in.*field == nullptr) {
      continue;
    }
    Buffers::device<double> moved = FBF_TRY(device_zeros(res, count));
    FBF_TRY(add_scaled(res, in.*field, count, moved.data(), 1.0));
    FBF_TRY(add_scaled(res, d.in.*field, count, moved.data(), step));
    out.in.*field = moved.data();
    out.arrays.push_back(std::move(moved));
  }
  if (d.blocks != nullptr) {
    out.own.emplace(
        MoBlocks{.naux = blocks.naux,
                 .ncore = blocks.ncore,
                 .ncas = blocks.ncas,
                 .nvirt = blocks.nvirt,
                 .aa = Buffers::device<double>(blocks.aa.num_elements(), res.device()),
                 .ca = Buffers::device<double>(blocks.ca.num_elements(), res.device()),
                 .va = Buffers::device<double>(blocks.va.num_elements(), res.device()),
                 .cv = Buffers::device<double>(blocks.cv.num_elements(), res.device()),
                 .vc = Buffers::device<double>(blocks.vc.num_elements(), res.device())});
    FBF_TRY(shift_blocks(res, blocks, *d.blocks, step, *out.own));
  }
  return out;
}

// The step a line along @p d is differenced at: the power of two nearest an
// eighth of the smallest ‖x‖ / ‖ẋ‖ over the groups @p d moves (B; h1e, h2e
// and h1e_v's rows; the RDMs), so no group moves past its own size on the
// widest rung; 0 when it moves none of them, since the orbital energies
// alone move neither the tables nor the bars at fixed multipliers.
Result<double> line_step(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                         const EnergyDirection &d) {
  // [3 v + g]: group g's Σ x², v = 0 the inputs and v = 1 the direction.
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 6));
  const auto squares = [&](const EnergyInputs &x, const MoBlocks *b, double *const at) -> Status {
    const auto fields = input_fields(blocks);
    for (const size_t f : {7U, 8U, 9U, 10U, 11U}) {
      FBF_TRY(add_squares(res, x.*fields[f].first, fields[f].second, at));
    }
    for (const size_t f : {0U, 1U, 4U, 5U, 6U}) {
      FBF_TRY(add_squares(res, x.*fields[f].first, fields[f].second, at + 1));
    }
    if (b != nullptr) {
      for (const Buffers::device<double> *g : {&b->aa, &b->ca, &b->va, &b->cv}) {
        FBF_TRY(add_squares(res, g->data(), static_cast<int64_t>(g->num_elements()), at + 2));
      }
    }
    return {};
  };
  FBF_TRY(squares(in, &blocks, acc.data()));
  FBF_TRY(squares(d.in, d.blocks, acc.data() + 3));
  const std::vector<double> sq = FBF_TRY(download(res, acc.data(), 6));
  double step = 0.0;
  for (size_t g = 0; g < 3; ++g) {
    const double s = polarization_step(std::sqrt(sq[g]), 8.0 * std::sqrt(sq[3 + g]));
    if (s != 0.0) {
      step = step == 0.0 ? s : std::min(step, s);
    }
  }
  return step;
}

// The derivative along a line from symmetric differences at k s, k = 1, 2,
// 3: Σ_k w_k (f(k s) - f(-k s)) / 2ks is exact for f a polynomial of degree
// at most six, the weights cancelling the s² and s⁴ terms.
constexpr std::array<double, 3> kLineWeights{1.5, -0.6, 0.1};

// Σ_k w_k (A(x + k s ẋ) - A(x - k s ẋ)) / 2ks of fixed_adjoint at @p mult,
// R left out, onto @p bars: the derivative of the bars at fixed multipliers
// along @p d. Every table is a polynomial in (B, h1e, h2e, h1e_v, RDMs) of
// degree at most six (two slabs, each quadratic in B, around one block of
// degree at most two), so the bars are of degree at most five along a line
// and kLineWeights' difference is exact. The orbital energies are read by
// R alone, which is left out.
//
// line_derivative is the difference itself: Σ_k w_k (A(k s) - A(-k s)) / 2ks
// onto each of @p bars' wanted fields but the orbital energies', A(t) the
// bars @p pass puts on its scratch (one EnergyBars per entry of @p bars) at
// the signed step t; over @p levels rungs, 3 (kLineWeights') or 1 (one
// symmetric difference, weight 1: exact for A quadratic).
Status line_derivative(DeviceResources &res, const MoBlocks &blocks,
                       std::span<const EnergyBars> bars, const double step, const size_t levels,
                       const std::function<Status(double, std::span<const EnergyBars>)> &pass) {
  // Scratch bars for the wanted fields but the orbital energies'.
  std::vector<Buffers::device<double>> held;
  std::vector<EnergyBars> scratch(bars.size());
  std::vector<std::tuple<double *, double *, int64_t>> pairs;
  for (size_t b = 0; b < bars.size(); ++b) {
    for (const auto &[field, count] : bar_fields(blocks)) {
      if (bars[b].*field == nullptr || field == &EnergyBars::e_core ||
          field == &EnergyBars::e_virt) {
        continue;
      }
      held.push_back(FBF_TRY(device_zeros(res, count)));
      scratch[b].*field = held.back().data();
      pairs.emplace_back(held.back().data(), bars[b].*field, count);
    }
  }
  for (size_t level = 0; level < levels; ++level) {
    const auto k = static_cast<double>(level + 1);
    const double weight = levels == 1 ? 1.0 : kLineWeights[level];
    for (const double sign : {1.0, -1.0}) {
      for (const auto &[from, to, count] : pairs) {
        FBF_TRY(zero(res, from, count));
      }
      FBF_TRY(pass(sign * k * step, scratch));
      for (const auto &[from, to, count] : pairs) {
        FBF_TRY(add_scaled(res, from, count, to, sign * weight / (2.0 * k * step)));
      }
    }
  }
  return {};
}

Status fixed_adjoint_tangent(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                             const int32_t batch, const EnergyDirection &d, const double step,
                             const ScMultipliers &mult, const EnergyBars &bars) {
  const ScMultipliers no_r{.p = mult.p, .q = mult.q, .r = {}, .blocks = mult.blocks};
  return line_derivative(res, blocks, {&bars, 1}, step, kLineWeights.size(),
                         [&](const double t, std::span<const EnergyBars> scratch) -> Status {
                           const ShiftedInputs x = FBF_TRY(shifted_inputs(res, blocks, in, d, t));
                           return fixed_adjoint(res, x.blocks(), x.in, batch, scratch.front(),
                                                no_r);
                         });
}

// Every perturber's N and H second derivatives along (@p line, along[j]),
// [k][t m + j] over m = along.size(): kLineWeights' difference of
// class_energies_tangent's (Ṅ, Ḣ) along each of @p along, at the inputs
// moved along @p line by ±k @p step. N and H are polynomials of degree at
// most six, so this is exact.
struct SecondTables {
  std::array<std::vector<double>, 8> norm, h;
};

Result<SecondTables> second_tables(DeviceResources &res, const MoBlocks &blocks,
                                   const EnergyInputs &in, const int32_t batch,
                                   const EnergyDirection &line,
                                   std::span<const EnergyDirection> along, const double step) {
  SecondTables out;
  for (size_t level = 0; level < kLineWeights.size(); ++level) {
    const auto k = static_cast<double>(level + 1);
    for (const double sign : {1.0, -1.0}) {
      const ShiftedInputs x = FBF_TRY(shifted_inputs(res, blocks, in, line, sign * k * step));
      const ClassEnergiesTangent t =
          FBF_TRY(class_energies_tangent(res, x.blocks(), x.in, batch, along));
      const double weight = sign * kLineWeights[level] / (2.0 * k * step);
      for (size_t c = 0; c < 8; ++c) {
        for (const auto &[to, from] :
             {std::pair{&out.norm[c], &t.norm_dot[c]}, std::pair{&out.h[c], &t.h_dot[c]}}) {
          to->resize(from->size(), 0.0);
          for (size_t e = 0; e < from->size(); ++e) {
            (*to)[e] += weight * (*from)[e];
          }
        }
      }
    }
  }
  return out;
}

// Sijrs's share of the Hessian-vector product along @p d, slab by slab: its
// slab's tangent ġ (bilinear in B, (slab(B + s Ḃ) - slab(B) - s² slab(Ḃ)) / s
// as slab_tangent's), the tangent of sijrs_adjoint's bars along (ġ, ε̇) by
// device::sijrs_adjoint_tangent (ε̇'s onto @p bars' orbital energies), pulled
// back onto Λ through B, and the bar itself through Ḃ: Λ is linear in B at a
// fixed slab bar. Everything is scaled by @p weight, Σ_J H̄_JJ in SC-QD.
Status sijrs_hvp(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                 const int32_t batch, const EnergyDirection &d, const EnergyBars &bars,
                 const double weight = 1.0) {
  if (blocks.ncore == 0 || blocks.nvirt == 0 || weight == 0.0) {
    return {};
  }
  // The kernel adds its orbital-energy bars at weight 1: away from it they
  // go through buffers of their own, scaled once at the end.
  const bool scaled = weight != 1.0;
  Buffers::device<double> e_core_bar = FBF_TRY(device_zeros(res, scaled ? blocks.ncore : 0));
  Buffers::device<double> e_virt_bar = FBF_TRY(device_zeros(res, scaled ? blocks.nvirt : 0));
  double *const core_bar = scaled && bars.e_core != nullptr ? e_core_bar.data() : bars.e_core;
  double *const virt_bar = scaled && bars.e_virt != nullptr ? e_virt_bar.data() : bars.e_virt;
  const int32_t extent = walk_extent(blocks, ExtBlock::Sijrs);
  const int32_t step = batch_step(extent, batch);
  Buffers::device<double> g = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  Buffers::device<double> g_dot = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  Buffers::device<double> scratch = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  Buffers::device<double> g_bar = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  Buffers::device<double> g_bar_dot = slab_buffer(res, blocks, ExtBlock::Sijrs, step);
  // B + s Ḃ, s sizing s Ḃ to B, when B moves.
  double s = 0.0;
  std::optional<ShiftedInputs> shifted;
  if (d.blocks != nullptr) {
    const EnergyDirection b_only{.blocks = d.blocks};
    Buffers::device<double> acc = FBF_TRY(device_zeros(res, 2));
    for (const auto &[x, at] :
         {std::pair{&blocks, acc.data()}, std::pair{d.blocks, acc.data() + 1}}) {
      for (const Buffers::device<double> *b : {&x->aa, &x->ca, &x->va, &x->cv}) {
        FBF_TRY(add_squares(res, b->data(), static_cast<int64_t>(b->num_elements()), at));
      }
    }
    const std::vector<double> sq = FBF_TRY(download(res, acc.data(), 2));
    s = polarization_step(std::sqrt(sq[0]), std::sqrt(sq[1]));
    if (s != 0.0) {
      shifted = FBF_TRY(shifted_inputs(res, blocks, in, b_only, s));
    }
  }
  for (int32_t b0 = 0; b0 < extent; b0 += step) {
    const int32_t b1 = std::min(extent, b0 + step);
    const int64_t count = slab_count(blocks, ExtBlock::Sijrs, b0, b1);
    FBF_TRY(slab(res, blocks, ExtBlock::Sijrs, b0, b1, g.data()));
    FBF_TRY(zero(res, g_dot.data(), count));
    if (shifted) {
      FBF_TRY(slab(res, shifted->blocks(), ExtBlock::Sijrs, b0, b1, scratch.data()));
      FBF_TRY(add_scaled(res, scratch.data(), count, g_dot.data(), 1.0 / s));
      FBF_TRY(add_scaled(res, g.data(), count, g_dot.data(), -1.0 / s));
      FBF_TRY(slab(res, *d.blocks, ExtBlock::Sijrs, b0, b1, scratch.data()));
      FBF_TRY(add_scaled(res, scratch.data(), count, g_dot.data(), -s));
    }
    FBF_TRY(zero(res, g_bar_dot.data(), count));
    FBF_TRY(gpu_status(device::sijrs_adjoint_tangent(
        res.stream(), g.data(), g_dot.data(), b1 - b0, b0, blocks.ncore, blocks.nvirt, in.e_core,
        in.e_virt, d.in.e_core, d.in.e_virt, g_bar_dot.data(), core_bar, virt_bar)));
    if (bars.lambda == nullptr) {
      continue;
    }
    if (scaled) {
      // g_dot is spent: it takes the weighted slab bar.
      FBF_TRY(zero(res, g_dot.data(), count));
      FBF_TRY(add_scaled(res, g_bar_dot.data(), count, g_dot.data(), weight));
    }
    FBF_TRY(slab_adjoint(res, blocks, ExtBlock::Sijrs, b0, b1,
                         scaled ? g_dot.data() : g_bar_dot.data(), bars.lambda));
    if (d.blocks != nullptr) {
      FBF_TRY(zero(res, g_bar.data(), count));
      FBF_TRY(gpu_status(device::sijrs_adjoint(res.stream(), g.data(), b1 - b0, b0, blocks.ncore,
                                               blocks.nvirt, in.e_core, in.e_virt, g_bar.data(),
                                               nullptr, nullptr, weight)));
      FBF_TRY(slab_adjoint(res, *d.blocks, ExtBlock::Sijrs, b0, b1, g_bar.data(), bars.lambda));
    }
  }
  if (scaled) {
    for (const auto &[from, to, count] :
         {std::tuple{&e_core_bar, bars.e_core, int64_t{blocks.ncore}},
          std::tuple{&e_virt_bar, bars.e_virt, int64_t{blocks.nvirt}}}) {
      if (to != nullptr) {
        FBF_TRY(add_scaled(res, from->data(), count, to, weight));
      }
    }
  }
  return res.sync();
}

// The largest weighted defect a pair may carry before it is refused: over
// the empty perturbers both directions populate, max ρ_τ w, w = √(N^{uu}
// N^{vv}) / |Δ + η| (the perturber's share of the pair's Hessian), against
// this fraction of S, the largest such w over every perturber. ρ_τ alone has
// no gap: on PSB3 its round-off grows as 1/population and reaches 2e-5 on a
// proportional pair, past real defects of 5e-9 (README, #914).
constexpr double kEmptyDefectTolerance = 1e-6;

constexpr std::array<std::string_view, 8> kScClassNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                        "Srsi", "Srs", "Sij",   "Sir"};

// One guarded perturber's second-order tables along a pair of directions:
// N^{ww} and H^{ww} for w = u, v, and the cross N^{uv} and H^{uv}.
struct EmptyPair {
  double nuu, huu, nvv, hvv, nuv, huv;
};

// The ρ_τ defect of an empty perturber both directions populate (the demo's
// hessian-theory §5.6): with unit populations, c = N^{uv} / √(N^{uu} N^{vv}),
// κ = H^{uv} / √(N^{uu} N^{vv}), η_w = H^{ww} / N^{ww} and η their mean,
//
//     ρ = max(|η_u - η_v|, 2 |κ - c η|) / |Δ + η|,
//
// zero exactly when H is proportional to N on the pair, and invariant under
// rescaling either direction.
double empty_defect(const EmptyPair &e, const double delta) {
  const double root = std::sqrt(e.nuu * e.nvv);
  const double eta_u = e.huu / e.nuu;
  const double eta_v = e.hvv / e.nvv;
  const double eta = 0.5 * (eta_u + eta_v);
  return std::max(std::abs(eta_u - eta_v), 2.0 * std::abs((e.huv / root) - (e.nuv / root * eta))) /
         std::abs(delta + eta);
}

Result<ClassEnergiesHvp> class_energies_hvp(DeviceResources &res, const MoBlocks &blocks,
                                            const EnergyInputs &in, const int32_t batch,
                                            std::span<const EnergyDirection> directions,
                                            std::span<const EnergyBars> bars,
                                            std::span<const EnergyDirection> partners) {
  const ScopedDeviceIndex on{res.dev_idx()};
  if (bars.size() != directions.size()) {
    return err_config("nevpt2::class_energies_hvp: {} bars for {} directions", bars.size(),
                      directions.size());
  }
  if (!partners.empty() && partners.size() != directions.size()) {
    return err_config("nevpt2::class_energies_hvp: {} partners for {} directions (none, or one "
                      "per direction)",
                      partners.size(), directions.size());
  }
  FBF_TRY(check_directions(blocks, partners));
  const ClassEnergiesTangent t =
      FBF_TRY(class_energies_tangent(res, blocks, in, batch, directions));
  const size_t m = directions.size();

  // A one-function class (single_function_class): its blocks, η, and their
  // tangents along each direction. It has no guarded perturber.
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  const std::vector<TangentSteps> steps = FBF_TRY(tangent_steps(res, blocks, in, directions));
  std::array<std::optional<double>, 8> eta_of;
  std::array<std::array<std::vector<double>, 2>, 8> xy;
  std::array<std::vector<std::array<std::vector<double>, 2>>, 8> xy_dot;
  for (size_t k = 0; k < 8; ++k) {
    if (single_function_class(o, k)) {
      xy[k] = FBF_TRY(single_function_blocks(res, o, k));
      eta_of[k] = single_function_ratio(xy[k]);
      xy_dot[k] = FBF_TRY(single_function_block_dots(res, o, k, directions, steps));
    }
  }

  // The finish's multipliers at the base point, P = E_N, Q = E_H and
  // R = E_Δ in the (N, d) form, d = N Δ + H: zero on a guarded perturber.
  // A one-function class's, E = Σ_t N_t c_t with c_t = -1 / (Δ_t + η):
  // P_t = c_t, Q_t = 0 and R_t = N_t c_t², and η̄ = Σ_t R_t on its blocks.
  ScMultipliers base;
  std::array<double, 8> eta_bar{};
  bool any_guarded = false;
  for (size_t k = 0; k < 8; ++k) {
    const size_t count = t.norm[k].size();
    for (std::vector<double> *v : {&base.p[k], &base.q[k], &base.r[k]}) {
      v->assign(count, 0.0);
    }
    if (eta_of[k]) {
      for (size_t g = 0; g < count; ++g) {
        const double cg = -1.0 / (t.delta[k][g] + *eta_of[k]);
        base.p[k][g] = cg;
        base.r[k][g] = t.norm[k][g] * cg * cg;
        eta_bar[k] += base.r[k][g];
      }
      base.blocks[k] = single_function_eta_bars(xy[k], *eta_of[k], eta_bar[k]);
      continue;
    }
    for (size_t g = 0; g < count; ++g) {
      const double norm = t.norm[k][g];
      if (std::abs(norm) <= kNumericalZero) {
        any_guarded = true;
        continue;
      }
      const double d = (norm * t.delta[k][g]) + t.h[k][g];
      base.p[k][g] = (-2.0 * norm / d) + (norm * norm * t.delta[k][g] / (d * d));
      base.q[k][g] = norm * norm / (d * d);
      base.r[k][g] = norm * norm * norm / (d * d);
    }
  }

  ClassEnergiesHvp out{.energies = t.energies,
                       .populated = std::vector<std::array<int64_t, 8>>(m),
                       .population = std::vector<double>(m, 0.0),
                       .scale = std::vector<double>(m, 0.0),
                       .defect = std::vector<double>(m, 0.0)};
  for (size_t u = 0; u < m; ++u) {
    const EnergyDirection &du = directions[u];
    const EnergyDirection *const dv = partners.empty() ? nullptr : &partners[u];
    const double step = FBF_TRY(line_step(res, blocks, in, du));

    // The multipliers the fixed-multiplier bars are differenced at: the
    // finish's, and on each empty perturber u populates, K6's limit along the
    // ray, P̄ = 2T̄ + Δ T̄² and Q̄ = T̄², T̄ = -1 / (Δ + η).
    ScMultipliers fixed{.p = base.p, .q = base.q, .r = {}, .blocks = base.blocks};
    if (any_guarded && step != 0.0) {
      std::vector<EnergyDirection> along{du};
      if (dv != nullptr) {
        along.push_back(*dv);
      }
      const SecondTables su = FBF_TRY(second_tables(res, blocks, in, batch, du, along, step));
      std::optional<SecondTables> sv;
      double step_v = 0.0;
      if (dv != nullptr) {
        step_v = FBF_TRY(line_step(res, blocks, in, *dv));
        if (step_v != 0.0) {
          sv = FBF_TRY(second_tables(res, blocks, in, batch, *dv, {dv, 1}, step_v));
        }
      }
      const size_t mu = along.size();
      // A population is read against its direction's scale: the larger of
      // its largest |N^{ww}| over every perturber and the largest |N| over
      // (8 s)², s ≈ ‖x‖ / 8 ‖ẋ‖, the size N^{ww} takes when the direction
      // moves N at all (a direction of the RDMs alone moves it linearly, so
      // every N^{ww} is round-off).
      double largest = 0.0;
      for (size_t k = 0; k < 8; ++k) {
        for (const double x : t.norm[k]) {
          largest = std::max(largest, std::abs(x));
        }
      }
      double scale_u = largest / (64.0 * step * step);
      double scale_v = 0.0;
      if (sv) {
        scale_v = largest / (64.0 * step_v * step_v);
      }
      for (size_t k = 0; k < 8; ++k) {
        for (size_t g = 0; g < t.norm[k].size(); ++g) {
          scale_u = std::max(scale_u, std::abs(su.norm[k][g * mu]));
          if (sv) {
            scale_v = std::max(scale_v, std::abs(sv->norm[k][g]));
          }
        }
      }
      out.scale[u] = scale_u;
      // S, the pair's Hessian scale: the largest √|N^{uu} N^{vv}| / |Δ + η|
      // over every perturber, η its own H / N, or an empty one's mean of
      // η_u and η_v. An empty perturber's defect is weighed against it.
      double pair_scale = 0.0;
      if (sv) {
        for (size_t k = 0; k < 8; ++k) {
          for (size_t g = 0; g < t.norm[k].size(); ++g) {
            const double n = t.norm[k][g];
            const double nuu = su.norm[k][g * mu];
            const double nvv = sv->norm[k][g];
            double eta = 0.0;
            if (eta_of[k]) {
              eta = *eta_of[k];
            } else if (std::abs(n) > kNumericalZero) {
              eta = t.h[k][g] / n;
            } else if (nuu > kEmptyPopulationFloor * scale_u &&
                       nvv > kEmptyPopulationFloor * scale_v) {
              eta = 0.5 * ((su.h[k][g * mu] / nuu) + (sv->h[k][g] / nvv));
            } else {
              continue;
            }
            pair_scale = std::max(pair_scale,
                                  std::sqrt(std::abs(nuu * nvv)) / std::abs(t.delta[k][g] + eta));
          }
        }
      }
      double worst = 0.0;
      size_t worst_k = 0;
      size_t worst_g = 0;
      double worst_rho = 0.0;
      for (size_t k = 0; k < 8; ++k) {
        for (size_t g = 0; g < t.norm[k].size(); ++g) {
          if (eta_of[k] || std::abs(t.norm[k][g]) > kNumericalZero) {
            continue;
          }
          const EmptyPair e{.nuu = su.norm[k][g * mu],
                            .huu = su.h[k][g * mu],
                            .nvv = sv ? sv->norm[k][g] : 0.0,
                            .hvv = sv ? sv->h[k][g] : 0.0,
                            .nuv = mu > 1 ? su.norm[k][(g * mu) + 1] : 0.0,
                            .huv = mu > 1 ? su.h[k][(g * mu) + 1] : 0.0};
          if (e.nuu <= kEmptyPopulationFloor * scale_u) {
            continue;
          }
          const double delta = t.delta[k][g];
          double eta = e.huu / e.nuu;
          if (sv && e.nvv > kEmptyPopulationFloor * scale_v) {
            eta = 0.5 * (eta + (e.hvv / e.nvv));
            const double rho = empty_defect(e, delta);
            const double weighted =
                rho * std::sqrt(e.nuu * e.nvv) / std::abs(delta + eta) / pair_scale;
            if (weighted > worst) {
              worst = weighted;
              worst_k = k;
              worst_g = g;
              worst_rho = rho;
            }
          }
          const double tbar = -1.0 / (delta + eta);
          fixed.p[k][g] = (2.0 * tbar) + (delta * tbar * tbar);
          fixed.q[k][g] = tbar * tbar;
          ++out.populated[u][k];
          out.population[u] = std::max(out.population[u], e.nuu);
        }
      }
      out.defect[u] = worst;
      if (worst > kEmptyDefectTolerance) {
        return err_numerical(
            "nevpt2::class_energies_hvp: class {} perturber {} is empty (|N| <= 1e-14) and "
            "populated by direction {} and its partner, but its H is not proportional to its "
            "N on the pair: rho_tau {:.3e}, weighted by its share of the pair's Hessian scale "
            "{:.3e}, past {:.0e}. The energy has no Hessian on that pair",
            kScClassNames[worst_k], worst_g, u, worst_rho, worst, kEmptyDefectTolerance);
      }
    }

    // The multipliers' tangent: the finish's 3 x 3 in (N, H, Δ) times
    // (Ṅ, Ḣ, Δ̇), zero on a guarded perturber, whose (Ṅ, Ḣ) vanish with N.
    // A one-function class's through η too: ċ_t = c_t² (Δ̇_t + η̇),
    // Ṗ_t = ċ_t, Ṙ_t = Ṅ_t c_t² + 2 N_t c_t ċ_t, and its block bars'
    // tangent at η̄̇ = Σ Ṙ and the blocks' (Ẋ, Ẏ).
    ScMultipliers dot;
    for (size_t k = 0; k < 8; ++k) {
      const size_t count = t.norm[k].size();
      for (std::vector<double> *v : {&dot.p[k], &dot.q[k], &dot.r[k]}) {
        v->assign(count, 0.0);
      }
      if (eta_of[k]) {
        const double e = *eta_of[k];
        const auto &[x, y] = xy[k];
        const auto &[x_dot, y_dot] = xy_dot[k][u];
        const double e_dot = single_function_ratio_dot(xy[k], e, xy_dot[k][u]);
        double eta_bar_dot = 0.0;
        for (size_t g = 0; g < count; ++g) {
          const double cg = -1.0 / (t.delta[k][g] + e);
          const double c_dot = cg * cg * (t.delta_dot[k][(g * m) + u] + e_dot);
          dot.p[k][g] = c_dot;
          dot.r[k][g] = (t.norm_dot[k][(g * m) + u] * cg * cg) + (2.0 * t.norm[k][g] * cg * c_dot);
          eta_bar_dot += dot.r[k][g];
        }
        // single_function_eta_bars' M_X = η̄ Y / S and M_Y = η̄ (X - 2η Y) / S
        // differentiated, Ṡ = 2 <Y, Ẏ>.
        const double sy = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
        const double sy_dot = 2.0 * std::inner_product(y.begin(), y.end(), y_dot.begin(), 0.0);
        const double eb = eta_bar[k];
        dot.blocks[k] = {std::vector<double>(x.size()), std::vector<double>(y.size())};
        for (size_t i = 0; i < x.size(); ++i) {
          const double r = x[i] - (2.0 * e * y[i]);
          const double r_dot = x_dot[i] - (2.0 * e_dot * y[i]) - (2.0 * e * y_dot[i]);
          dot.blocks[k][0][i] =
              ((eta_bar_dot * y[i]) + (eb * y_dot[i])) / sy - (eb * y[i] * sy_dot / (sy * sy));
          dot.blocks[k][1][i] =
              ((eta_bar_dot * r) + (eb * r_dot)) / sy - (eb * r * sy_dot / (sy * sy));
        }
        continue;
      }
      for (size_t g = 0; g < count; ++g) {
        const double n = t.norm[k][g];
        if (std::abs(n) <= kNumericalZero) {
          continue;
        }
        const double delta = t.delta[k][g];
        const double d = (n * delta) + t.h[k][g];
        const double d2 = d * d;
        const double d3 = d2 * d;
        const double e_nn =
            (-2.0 / d) + (4.0 * n * delta / d2) - (2.0 * n * n * delta * delta / d3);
        const double e_nh = (2.0 * n / d2) - (2.0 * n * n * delta / d3);
        const double e_nd = (3.0 * n * n / d2) - (2.0 * n * n * n * delta / d3);
        const double e_hh = -2.0 * n * n / d3;
        const double e_hd = -2.0 * n * n * n / d3;
        const double e_dd = -2.0 * n * n * n * n / d3;
        const double n_dot = t.norm_dot[k][(g * m) + u];
        const double h_dot = t.h_dot[k][(g * m) + u];
        const double d_dot = t.delta_dot[k][(g * m) + u];
        dot.p[k][g] = (e_nn * n_dot) + (e_nh * h_dot) + (e_nd * d_dot);
        dot.q[k][g] = (e_nh * n_dot) + (e_hh * h_dot) + (e_hd * d_dot);
        dot.r[k][g] = (e_nd * n_dot) + (e_hd * h_dot) + (e_dd * d_dot);
      }
    }

    FBF_TRY(fixed_adjoint(res, blocks, in, batch, bars[u], dot));
    if (step != 0.0) {
      FBF_TRY(fixed_adjoint_tangent(res, blocks, in, batch, du, step, fixed, bars[u]));
    }
    FBF_TRY(sijrs_hvp(res, blocks, in, batch, du, bars[u]));
  }
  return out;
}

// ---------------------------------------------------------------------------
// The SC-QD Hessian-vector product: forward over the pair walk
// ---------------------------------------------------------------------------

// One tuple's coupling g = -T N / d, d = N Δ + H over its ket's own tables,
// differentiated (docs/nevpt2/nevpt2.tex Eq. qd-scfinish): the first
// derivatives in (T, N, H, Δ), and the second, g_TT = 0. At T = N it is the
// diagonal energy -N² / d, whose total derivatives are E_N = g_T + g_N,
// E_NN = 2 g_TN + g_NN, E_NH = g_TH + g_NH, E_NΔ = g_TΔ + g_NΔ and the rest
// g's own (Eq. finish): so a state's own tables take the ket's share of
// every coupling, its own included, through the one form.
struct CouplingFinish {
  double t, n, h, d;
  double tn, th, td, nn, nh, nd, hh, hd, dd;
};

CouplingFinish coupling_finish(const double t, const double n, const double h, const double delta) {
  const double d = (n * delta) + h;
  const double d2 = d * d;
  const double d3 = d2 * d;
  return {.t = -n / d,
          .n = -t * h / d2,
          .h = t * n / d2,
          .d = t * n * n / d2,
          .tn = -h / d2,
          .th = n / d2,
          .td = n * n / d2,
          .nn = 2.0 * t * h * delta / d3,
          .nh = t * (h - (n * delta)) / d3,
          .nd = 2.0 * t * n * h / d3,
          .hh = -2.0 * t * n / d3,
          .hd = -2.0 * t * n * n / d3,
          .dd = -2.0 * t * n * n * n / d3};
}

// The pair walk of (M, N) at fixed multipliers: @p coef[k][t] on T^t_MN,
// half onto each of the norm einsums of @p mn, (M, N), and @p nm, (N, M),
// through the slabs (onto @p shared's Λ) and h1e_v's rows (its h1e_v_*), and
// each side's metric bars swept onto @p t_mn's and @p t_nm's dm1..dm3 (when
// its dm1 is set): sc_couplings_adjoint's part 2 with a table in place of
// its finish.
Status pair_fixed_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &mn,
                          const EnergyInputs &nm, const int32_t batch, const EnergyBars &shared,
                          const std::array<std::vector<double>, 8> &coef, const EnergyBars &t_mn,
                          const EnergyBars &t_nm) {
  const ActiveOperands a = FBF_TRY(active_operands(res, blocks, mn));
  const ActiveOperands b = FBF_TRY(active_operands(res, blocks, nm));
  std::vector<double> unused_core(a.e_core.size(), 0.0);
  std::vector<double> unused_virt(a.e_virt.size(), 0.0);
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = mn,
      .bars = shared,
      .batch = batch,
      .ops = a,
      .e = {.core = a.e_core, .virt = a.e_virt, .core_bar = unused_core, .virt_bar = unused_virt}};
  BlockBars m_a = FBF_TRY(block_bars(res, blocks.ncas, false, true));
  BlockBars m_b = FBF_TRY(block_bars(res, blocks.ncas, false, true));
  const int64_t nc = blocks.ncore;
  for (const auto &[k, walk] : kScWalks) {
    const ScWalk w{.a = &a,
                   .m_a = &m_a,
                   .b = &b,
                   .m_b = &m_b,
                   .finish = [&, k](const int64_t first, const std::vector<Perturber> &perturbers,
                                    const std::vector<double> &, const std::vector<double> &,
                                    std::vector<double> &a_bar, std::vector<double> &b_bar) {
                     for (size_t x = 0; x < perturbers.size(); ++x) {
                       const Perturber &p = perturbers[x];
                       const double half = 0.5 * coef[k][perturber_index(k, first, x, p, nc)];
                       spread(p, half, a_bar);
                       spread(p, half, b_bar);
                     }
                   }};
    FBF_TRY(walk(c, w));
  }
  for (const auto &[o, m, t] : {std::tuple{&a, &m_a, &t_mn}, std::tuple{&b, &m_b, &t_nm}}) {
    if (t->dm1 != nullptr) {
      FBF_TRY(sweep_metrics(res, *o, *m, *t));
    }
  }
  return res.sync();
}

// The pair walk's shared bars alone: h1e_v's rows and Λ, all it adds to.
EnergyBars pair_shared_bars(const EnergyBars &b) {
  return {
      .h1e_v_sr = b.h1e_v_sr, .h1e_v_si = b.h1e_v_si, .h1e_v_sir = b.h1e_v_sir, .lambda = b.lambda};
}

// The step a pair's line is differenced at: the smaller of its two sides'
// line_step, each moving its own RDMs; 0 when neither moves anything.
Result<double> pair_line_step(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &mn,
                              const EnergyInputs &nm, const EnergyDirection &da,
                              const EnergyDirection &db) {
  double step = 0.0;
  for (const auto &[in, d] : {std::pair{&mn, &da}, std::pair{&nm, &db}}) {
    const double s = FBF_TRY(line_step(res, blocks, *in, *d));
    if (s != 0.0) {
      step = step == 0.0 ? s : std::min(step, s);
    }
  }
  return step;
}

// T's second derivatives along one direction, [k][t]: kLineWeights'
// difference of pair_tables' Ṫ along (@p da, @p db) at the pair moved along
// them by ±k @p step, exact as second_tables' (T is of degree at most six).
Result<std::array<std::vector<double>, 8>>
pair_second_tables(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &mn,
                   const EnergyInputs &nm, const int32_t batch, const EnergyDirection &da,
                   const EnergyDirection &db, const double step) {
  std::array<std::vector<double>, 8> out;
  for (size_t level = 0; level < kLineWeights.size(); ++level) {
    const auto k = static_cast<double>(level + 1);
    for (const double sign : {1.0, -1.0}) {
      const ShiftedInputs xa = FBF_TRY(shifted_inputs(res, blocks, mn, da, sign * k * step));
      const ShiftedInputs xb = FBF_TRY(shifted_inputs(res, blocks, nm, db, sign * k * step));
      const PairTables t =
          FBF_TRY(pair_tables(res, xa.blocks(), xa.in, xb.in, batch, {&da, 1}, {&db, 1}));
      const double weight = sign * kLineWeights[level] / (2.0 * k * step);
      for (size_t c = 0; c < 8; ++c) {
        out[c].resize(t.t_dot[c].size(), 0.0);
        for (size_t e = 0; e < t.t_dot[c].size(); ++e) {
          out[c][e] += weight * t.t_dot[c][e];
        }
      }
    }
  }
  return out;
}

Result<ScEffectiveHamiltonianHvp> sc_couplings_hvp(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    const int64_t ns, std::span<const double> hbar, const int32_t batch,
    std::span<const EnergyDirection> directions, std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, int64_t, const EnergyBars &)> &transition,
    std::span<const EnergyDirection> partners) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n2 = static_cast<size_t>(ns * ns);
  const size_t m = directions.size() / n2;
  const int64_t n = blocks.ncas;
  const auto at = [ns](const int64_t i, const int64_t j) {
    return static_cast<size_t>((i * ns) + j);
  };
  const auto pair_of = [ns](const int64_t i, const int64_t j) {
    return static_cast<size_t>((std::min(i, j) * ns) + std::max(i, j));
  };
  // Direction u's (or, past m, partner u - m's) entry for pair p.
  const auto dir = [&](const size_t u, const size_t p) -> const EnergyDirection & {
    return u < m ? directions[(u * n2) + p] : partners[((u - m) * n2) + p];
  };

  // Every pair's tables and their tangents along the directions, and along
  // the partners after them: mt columns, [k][t mt + u].
  std::vector<EnergyDirection> all(directions.begin(), directions.end());
  all.insert(all.end(), partners.begin(), partners.end());
  const size_t mt = all.size() / n2;
  ScEffectiveHamiltonianTangent tan =
      FBF_TRY(sc_couplings_tangent(res, blocks, pairs, ns, batch, all));
  const auto table = [&](const int64_t i, const int64_t j) -> const auto & {
    return tan.transition[at(i, j)];
  };
  const auto table_dot = [&](const int64_t i, const int64_t j) -> const auto & {
    return tan.transition_dot[at(i, j)];
  };

  // A one-function class per ket: its blocks, η, and their tangents along
  // each direction. It has no guarded tuple.
  std::vector<std::array<std::optional<double>, 8>> eta_of(static_cast<size_t>(ns));
  std::vector<std::array<std::array<std::vector<double>, 2>, 8>> xy(static_cast<size_t>(ns));
  std::vector<std::array<std::vector<std::array<std::vector<double>, 2>>, 8>> xy_dot(
      static_cast<size_t>(ns));
  for (int64_t j = 0; j < ns; ++j) {
    const auto jj = static_cast<size_t>(j);
    const ActiveOperands o = FBF_TRY(active_operands(res, blocks, pairs[at(j, j)]));
    std::vector<EnergyDirection> own;
    for (size_t u = 0; u < m; ++u) {
      own.push_back(dir(u, at(j, j)));
    }
    const std::vector<TangentSteps> steps =
        FBF_TRY(tangent_steps(res, blocks, pairs[at(j, j)], own));
    for (size_t k = 0; k < 8; ++k) {
      if (single_function_class(o, k)) {
        xy[jj][k] = FBF_TRY(single_function_blocks(res, o, k));
        eta_of[jj][k] = single_function_ratio(xy[jj][k]);
        xy_dot[jj][k] = FBF_TRY(single_function_block_dots(res, o, k, own, steps));
      }
    }
  }
  const auto guarded = [&](const int64_t j, const size_t k, const size_t g) {
    return !eta_of[static_cast<size_t>(j)][k] &&
           std::abs(tan.kets[static_cast<size_t>(j)].norm[k][g]) <= kNumericalZero;
  };

  // The multipliers at the base point. Ket J's own tables take every
  // coupling with ket J, its own included: P = Σ_I H̄_IJ g_N, Q = Σ_I H̄_IJ
  // g_H, R = Σ_I H̄_IJ g_Δ, and the I = J coupling's T slot on N as well (its
  // T is N). Pair {I, J}'s T takes H̄_IJ g_T of ket J and H̄_JI g_T of ket I.
  // A one-function class's coupling is T c, c = -1 / (Δ + η^J): c on T, and
  // D̄ = Σ_I H̄_IJ T c² on Δ and, summed, on η^J, which goes to its blocks.
  // Zero on a guarded tuple.
  std::vector<ScMultipliers> own(static_cast<size_t>(ns));
  std::vector<std::array<double, 8>> eta_bar(static_cast<size_t>(ns));
  std::map<size_t, std::array<std::vector<double>, 8>> coef;
  for (int64_t mi = 0; mi < ns; ++mi) {
    for (int64_t ni = mi + 1; ni < ns; ++ni) {
      std::array<std::vector<double>, 8> &c = coef[at(mi, ni)];
      for (size_t k = 0; k < 8; ++k) {
        c[k].assign(table(mi, ni)[k].size(), 0.0);
      }
    }
  }
  std::vector<bool> any_guarded(static_cast<size_t>(ns), false);
  for (int64_t j = 0; j < ns; ++j) {
    const auto jj = static_cast<size_t>(j);
    const ClassEnergiesTangent &kt = tan.kets[jj];
    ScMultipliers &mult = own[jj];
    for (size_t k = 0; k < 8; ++k) {
      const size_t count = kt.norm[k].size();
      for (std::vector<double> *v : {&mult.p[k], &mult.q[k], &mult.r[k]}) {
        v->assign(count, 0.0);
      }
      for (size_t g = 0; g < count; ++g) {
        const double delta = kt.delta[k][g];
        if (eta_of[jj][k]) {
          const double cg = -1.0 / (delta + *eta_of[jj][k]);
          for (int64_t i = 0; i < ns; ++i) {
            const double w = hbar[at(i, j)];
            (i == j ? mult.p[k][g] : coef[pair_of(i, j)][k][g]) += w * cg;
            mult.r[k][g] += w * table(i, j)[k][g] * cg * cg;
          }
          eta_bar[jj][k] += mult.r[k][g];
          continue;
        }
        if (guarded(j, k, g)) {
          any_guarded[jj] = true;
          continue;
        }
        for (int64_t i = 0; i < ns; ++i) {
          const double w = hbar[at(i, j)];
          const CouplingFinish f =
              coupling_finish(table(i, j)[k][g], kt.norm[k][g], kt.h[k][g], delta);
          (i == j ? mult.p[k][g] : coef[pair_of(i, j)][k][g]) += w * f.t;
          mult.p[k][g] += w * f.n;
          mult.q[k][g] += w * f.h;
          mult.r[k][g] += w * f.d;
        }
      }
      if (eta_of[jj][k]) {
        mult.blocks[k] = single_function_eta_bars(xy[jj][k], *eta_of[jj][k], eta_bar[jj][k]);
      }
    }
  }

  ScEffectiveHamiltonianHvp out{
      .elements = tan.elements,
      .populated = std::vector<std::vector<std::array<int64_t, 8>>>(
          m, std::vector<std::array<int64_t, 8>>(static_cast<size_t>(ns))),
      .defect = std::vector<double>(m, 0.0),
      .coupling_defect = std::vector<double>(m, 0.0),
      .coupling_k6 = std::vector<double>(m, 0.0)};
  double trace = 0.0;
  for (int64_t j = 0; j < ns; ++j) {
    trace += hbar[at(j, j)];
  }

  for (size_t u = 0; u < m; ++u) {
    // The multipliers' tangent along u: Eq. qd-scfinish's second derivatives
    // times (Ṫ, Ṅ, Ḣ, Δ̇), the I = J coupling's Ṫ its Ṅ; a one-function
    // class's through ċ = c² (Δ̇ + η̇^J) and D̄'s tangent, and its block bars'
    // at η̄̇ and the blocks' (Ẋ, Ẏ). Zero on a guarded tuple.
    std::vector<ScMultipliers> dot(static_cast<size_t>(ns));
    std::map<size_t, std::array<std::vector<double>, 8>> coef_dot;
    for (const auto &[p, c] : coef) {
      for (size_t k = 0; k < 8; ++k) {
        coef_dot[p][k].assign(c[k].size(), 0.0);
      }
    }
    for (int64_t j = 0; j < ns; ++j) {
      const auto jj = static_cast<size_t>(j);
      const ClassEnergiesTangent &kt = tan.kets[jj];
      ScMultipliers &d = dot[jj];
      for (size_t k = 0; k < 8; ++k) {
        const size_t count = kt.norm[k].size();
        for (std::vector<double> *v : {&d.p[k], &d.q[k], &d.r[k]}) {
          v->assign(count, 0.0);
        }
        if (eta_of[jj][k]) {
          const double e = *eta_of[jj][k];
          const double e_dot = single_function_ratio_dot(xy[jj][k], e, xy_dot[jj][k][u]);
          double eta_bar_dot = 0.0;
          for (size_t g = 0; g < count; ++g) {
            const size_t gu = (g * mt) + u;
            const double cg = -1.0 / (kt.delta[k][g] + e);
            const double c_dot = cg * cg * (kt.delta_dot[k][gu] + e_dot);
            for (int64_t i = 0; i < ns; ++i) {
              const double w = hbar[at(i, j)];
              (i == j ? d.p[k][g] : coef_dot[pair_of(i, j)][k][g]) += w * c_dot;
              d.r[k][g] +=
                  w * ((table_dot(i, j)[k][gu] * cg * cg) + (2.0 * table(i, j)[k][g] * cg * c_dot));
            }
            eta_bar_dot += d.r[k][g];
          }
          // single_function_eta_bars' M_X = η̄ Y / S and M_Y = η̄ (X - 2η Y) / S
          // differentiated, Ṡ = 2 <Y, Ẏ>.
          const auto &[x, y] = xy[jj][k];
          const auto &[x_dot, y_dot] = xy_dot[jj][k][u];
          const double sy = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
          const double sy_dot = 2.0 * std::inner_product(y.begin(), y.end(), y_dot.begin(), 0.0);
          const double eb = eta_bar[jj][k];
          d.blocks[k] = {std::vector<double>(x.size()), std::vector<double>(y.size())};
          for (size_t i = 0; i < x.size(); ++i) {
            const double r = x[i] - (2.0 * e * y[i]);
            const double r_dot = x_dot[i] - (2.0 * e_dot * y[i]) - (2.0 * e * y_dot[i]);
            d.blocks[k][0][i] =
                ((eta_bar_dot * y[i]) + (eb * y_dot[i])) / sy - (eb * y[i] * sy_dot / (sy * sy));
            d.blocks[k][1][i] =
                ((eta_bar_dot * r) + (eb * r_dot)) / sy - (eb * r * sy_dot / (sy * sy));
          }
          continue;
        }
        for (size_t g = 0; g < count; ++g) {
          if (guarded(j, k, g)) {
            continue;
          }
          const size_t gu = (g * mt) + u;
          const double n_dot = kt.norm_dot[k][gu];
          const double h_dot = kt.h_dot[k][gu];
          const double d_dot = kt.delta_dot[k][gu];
          for (int64_t i = 0; i < ns; ++i) {
            const double w = hbar[at(i, j)];
            const CouplingFinish f =
                coupling_finish(table(i, j)[k][g], kt.norm[k][g], kt.h[k][g], kt.delta[k][g]);
            const double t_dot = table_dot(i, j)[k][gu];
            (i == j ? d.p[k][g] : coef_dot[pair_of(i, j)][k][g]) +=
                w * ((f.tn * n_dot) + (f.th * h_dot) + (f.td * d_dot));
            d.p[k][g] += w * ((f.tn * t_dot) + (f.nn * n_dot) + (f.nh * h_dot) + (f.nd * d_dot));
            d.q[k][g] += w * ((f.th * t_dot) + (f.nh * n_dot) + (f.hh * h_dot) + (f.hd * d_dot));
            d.r[k][g] += w * ((f.td * t_dot) + (f.nd * n_dot) + (f.hd * h_dot) + (f.dd * d_dot));
          }
        }
      }
    }

    // The fixed multipliers the bars are differenced at: the base's, and on
    // each empty tuple of a ket's that u populates, K6, every coupling's
    // limit along the ray (Part I's, n = 1, on the I = J coupling).
    std::vector<ScMultipliers> fixed;
    for (const ScMultipliers &x : own) {
      fixed.push_back({.p = x.p, .q = x.q, .r = {}, .blocks = x.blocks});
    }
    std::map<size_t, std::array<std::vector<double>, 8>> coef_fixed = coef;
    double worst = 0.0;
    std::tuple<int64_t, size_t, size_t, double> worst_at{};
    double worst_coupling = 0.0;
    std::tuple<int64_t, int64_t, size_t, size_t, double> worst_coupling_at{};
    std::map<size_t, std::array<std::vector<double>, 8>> second_t;
    for (int64_t j = 0; j < ns; ++j) {
      const auto jj = static_cast<size_t>(j);
      if (!any_guarded[jj]) {
        continue;
      }
      const ClassEnergiesTangent &kt = tan.kets[jj];
      const EnergyDirection &du = dir(u, at(j, j));
      const EnergyDirection *const dv = partners.empty() ? nullptr : &dir(m + u, at(j, j));
      const double step = FBF_TRY(line_step(res, blocks, pairs[at(j, j)], du));
      if (step == 0.0) {
        continue;
      }
      std::vector<EnergyDirection> along{du};
      if (dv != nullptr) {
        along.push_back(*dv);
      }
      const size_t mu = along.size();
      const SecondTables su =
          FBF_TRY(second_tables(res, blocks, pairs[at(j, j)], batch, du, along, step));
      std::optional<SecondTables> sv;
      double step_v = 0.0;
      if (dv != nullptr) {
        step_v = FBF_TRY(line_step(res, blocks, pairs[at(j, j)], *dv));
        if (step_v != 0.0) {
          sv = FBF_TRY(second_tables(res, blocks, pairs[at(j, j)], batch, *dv, {dv, 1}, step_v));
        }
      }
      // Part I's population scales over ket J's tables.
      double largest = 0.0;
      for (size_t k = 0; k < 8; ++k) {
        for (const double x : kt.norm[k]) {
          largest = std::max(largest, std::abs(x));
        }
      }
      double scale_u = largest / (64.0 * step * step);
      double scale_v = sv ? largest / (64.0 * step_v * step_v) : 0.0;
      for (size_t k = 0; k < 8; ++k) {
        for (size_t g = 0; g < kt.norm[k].size(); ++g) {
          scale_u = std::max(scale_u, std::abs(su.norm[k][g * mu]));
          if (sv) {
            scale_v = std::max(scale_v, std::abs(sv->norm[k][g]));
          }
        }
      }
      const auto populated_both = [&](const size_t k, const size_t g) {
        return sv && su.norm[k][g * mu] > kEmptyPopulationFloor * scale_u &&
               sv->norm[k][g] > kEmptyPopulationFloor * scale_v;
      };
      // A tuple's η: its own H / N, a one-function class's η^J, or an empty
      // one both populate, the mean of η_u and η_v; none otherwise.
      const auto eta_at = [&](const size_t k, const size_t g) -> std::optional<double> {
        if (eta_of[jj][k]) {
          return *eta_of[jj][k];
        }
        if (!guarded(j, k, g)) {
          return kt.h[k][g] / kt.norm[k][g];
        }
        if (populated_both(k, g)) {
          return 0.5 * ((su.h[k][g * mu] / su.norm[k][g * mu]) + (sv->h[k][g] / sv->norm[k][g]));
        }
        return std::nullopt;
      };
      // With a partner, S: ket J's Hessian scale on the pair, Part I's, and
      // per bra I the coupling (I, J)'s first-order scale, the largest
      // max(|Ṫ_u|, |Ṫ_v|) / |Δ + η| over ket J's tuples.
      double pair_scale = 0.0;
      std::vector<double> coupling_scale(static_cast<size_t>(ns), 0.0);
      if (sv) {
        for (size_t k = 0; k < 8; ++k) {
          for (size_t g = 0; g < kt.norm[k].size(); ++g) {
            const std::optional<double> eta = eta_at(k, g);
            if (!eta) {
              continue;
            }
            const double gap = std::abs(kt.delta[k][g] + *eta);
            pair_scale = std::max(pair_scale,
                                  std::sqrt(std::abs(su.norm[k][g * mu] * sv->norm[k][g])) / gap);
            for (int64_t i = 0; i < ns; ++i) {
              if (i != j) {
                const std::vector<double> &td = table_dot(i, j)[k];
                coupling_scale[static_cast<size_t>(i)] = std::max(
                    coupling_scale[static_cast<size_t>(i)],
                    std::max(std::abs(td[(g * mt) + u]), std::abs(td[(g * mt) + m + u])) / gap);
              }
            }
          }
        }
      }
      for (size_t k = 0; k < 8; ++k) {
        for (size_t g = 0; g < kt.norm[k].size(); ++g) {
          if (!guarded(j, k, g)) {
            continue;
          }
          const EmptyPair e{.nuu = su.norm[k][g * mu],
                            .huu = su.h[k][g * mu],
                            .nvv = sv ? sv->norm[k][g] : 0.0,
                            .hvv = sv ? sv->h[k][g] : 0.0,
                            .nuv = mu > 1 ? su.norm[k][(g * mu) + 1] : 0.0,
                            .huv = mu > 1 ? su.h[k][(g * mu) + 1] : 0.0};
          if (e.nuu <= kEmptyPopulationFloor * scale_u) {
            continue;
          }
          const double delta = kt.delta[k][g];
          double eta = e.huu / e.nuu;
          double rho = 0.0;
          if (populated_both(k, g)) {
            eta = *eta_at(k, g);
            rho = empty_defect(e, delta);
            const double weighted =
                rho * std::sqrt(e.nuu * e.nvv) / std::abs(delta + eta) / pair_scale;
            if (weighted > worst) {
              worst = weighted;
              worst_at = {j, k, g, rho};
            }
          }
          const double tbar = -1.0 / (delta + eta);
          ++out.populated[u][jj][k];
          for (int64_t i = 0; i < ns; ++i) {
            const double w = hbar[at(i, j)];
            if (i == j) {
              fixed[jj].p[k][g] += w * ((2.0 * tbar) + (delta * tbar * tbar));
              fixed[jj].q[k][g] += w * tbar * tbar;
              continue;
            }
            // The coupling's T slot at the ray's limit, T̄.
            coef_fixed[pair_of(i, j)][k][g] += w * tbar;
            const std::vector<double> &td = table_dot(i, j)[k];
            if (rho > 0.0 && coupling_scale[static_cast<size_t>(i)] > 0.0) {
              // A first-order transition tangent to bra I: the coupling is
              // Eq. qd-scfail's, of degree one, and has a Hessian on the pair
              // only where H is proportional to N there.
              const double share =
                  std::max(std::abs(td[(g * mt) + u]), std::abs(td[(g * mt) + m + u])) /
                  std::abs(delta + eta) / coupling_scale[static_cast<size_t>(i)];
              if (rho * share > worst_coupling) {
                worst_coupling = rho * share;
                worst_coupling_at = {i, j, k, g, rho};
              }
            }
            if (!guarded(i, k, g)) {
              // Populated for bra I: T is of first order along u and T / N
              // grows as 1/s, so N's and H's multipliers have no limit; only
              // T's is formed (README, "The SC-QD Hessian-vector product").
              continue;
            }
            // Empty for bra I too: T, N and H are all of second order, and
            // the coupling a quadratic form along u, T̄ T^{uu}. N's and H's
            // multipliers at the ray's limit, with ρ = T^{uu} / N^{uu}:
            // -ρ η T̄² and ρ T̄².
            const size_t p = pair_of(i, j);
            auto found = second_t.find(p);
            if (found == second_t.end()) {
              const size_t mn = p;
              const size_t nm = at(static_cast<int64_t>(p) % ns, static_cast<int64_t>(p) / ns);
              const double s = FBF_TRY(
                  pair_line_step(res, blocks, pairs[mn], pairs[nm], dir(u, mn), dir(u, nm)));
              std::array<std::vector<double>, 8> tuu{};
              if (s != 0.0) {
                tuu = FBF_TRY(pair_second_tables(res, blocks, pairs[mn], pairs[nm], batch,
                                                 dir(u, mn), dir(u, nm), s));
              }
              found = second_t.emplace(p, std::move(tuu)).first;
            }
            if (found->second[k].empty()) {
              continue;
            }
            const double ratio = found->second[k][g] / e.nuu;
            fixed[jj].p[k][g] += w * (-ratio * eta * tbar * tbar);
            fixed[jj].q[k][g] += w * ratio * tbar * tbar;
            out.coupling_k6[u] += w * ((tbar * found->second[k][g]) +
                                       (ratio * tbar * tbar * (e.huu - (eta * e.nuu))));
          }
        }
      }
    }
    out.defect[u] = worst;
    out.coupling_defect[u] = worst_coupling;
    if (worst_coupling > kEmptyDefectTolerance) {
      const auto [bra, ket, k, g, rho] = worst_coupling_at;
      return err_numerical(
          "nevpt2::sc_effective_hamiltonian_hvp: the coupling ({}, {}) has no Hessian on "
          "directions ({}, partner {}): ket {}'s class {} tuple {} is empty (|N| <= 1e-14) and "
          "populated by both, with a first-order transition tangent to bra {}, but its H is not "
          "proportional to its N on the pair: rho_tau {:.3e}, weighted by its share of the "
          "coupling's first-order scale {:.3e}, past {:.0e}",
          bra, ket, u, u, ket, kScClassNames[k], g, bra, rho, worst_coupling,
          kEmptyDefectTolerance);
    }
    if (worst > kEmptyDefectTolerance) {
      const auto [j, k, g, rho] = worst_at;
      return err_numerical(
          "nevpt2::sc_effective_hamiltonian_hvp: state {}'s class {} perturber {} is empty "
          "(|N| <= 1e-14) and populated by direction {} and its partner, but its H is not "
          "proportional to its N on the pair: rho_tau {:.3e}, weighted by its share of the "
          "pair's Hessian scale {:.3e}, past {:.0e}. Its energy has no Hessian on that pair",
          j, kScClassNames[k], g, u, rho, worst, kEmptyDefectTolerance);
    }

    // Each state's own walk: the multipliers' tangent through one fixed
    // walk (R's onto the orbital energies), and the bars at fixed
    // multipliers differenced along the direction.
    const EnergyBars &shared = bars[u * static_cast<size_t>(ns)];
    for (int64_t j = 0; j < ns; ++j) {
      const auto jj = static_cast<size_t>(j);
      const EnergyBars &b = bars[(u * static_cast<size_t>(ns)) + jj];
      const EnergyDirection &du = dir(u, at(j, j));
      FBF_TRY(fixed_adjoint(res, blocks, pairs[at(j, j)], batch, b, dot[jj]));
      const double step = FBF_TRY(line_step(res, blocks, pairs[at(j, j)], du));
      if (step != 0.0) {
        FBF_TRY(fixed_adjoint_tangent(res, blocks, pairs[at(j, j)], batch, du, step, fixed[jj], b));
      }
    }
    // Each pair's walk the same way, its transition bars handed over.
    const EnergyBars pair_shared = pair_shared_bars(shared);
    for (int64_t mi = 0; mi < ns; ++mi) {
      for (int64_t ni = mi + 1; ni < ns; ++ni) {
        const size_t mn = at(mi, ni);
        const size_t nm = at(ni, mi);
        std::vector<Buffers::device<double>> held;
        std::array<EnergyBars, 2> t{};
        if (transition) {
          for (EnergyBars &x : t) {
            for (const auto &[field, count] :
                 {std::pair{&EnergyBars::dm1, n * n}, std::pair{&EnergyBars::dm2, n * n * n * n},
                  std::pair{&EnergyBars::dm3, n * n * n * n * n * n}}) {
              held.push_back(FBF_TRY(device_zeros(res, count)));
              x.*field = held.back().data();
            }
          }
        }
        FBF_TRY(pair_fixed_adjoint(res, blocks, pairs[mn], pairs[nm], batch, pair_shared,
                                   coef_dot[mn], t[0], t[1]));
        const double step =
            FBF_TRY(pair_line_step(res, blocks, pairs[mn], pairs[nm], dir(u, mn), dir(u, nm)));
        if (step != 0.0) {
          const std::array<EnergyBars, 3> targets{pair_shared, t[0], t[1]};
          FBF_TRY(line_derivative(
              res, blocks, targets, step, kLineWeights.size(),
              [&](const double s, std::span<const EnergyBars> scratch) -> Status {
                const ShiftedInputs xa =
                    FBF_TRY(shifted_inputs(res, blocks, pairs[mn], dir(u, mn), s));
                const ShiftedInputs xb =
                    FBF_TRY(shifted_inputs(res, blocks, pairs[nm], dir(u, nm), s));
                return pair_fixed_adjoint(res, xa.blocks(), xa.in, xb.in, batch, scratch[0],
                                          coef_fixed[mn], scratch[1], scratch[2]);
              }));
        }
        if (transition) {
          FBF_TRY(transition(static_cast<int64_t>(u), mi, ni, t[0]));
          FBF_TRY(transition(static_cast<int64_t>(u), ni, mi, t[1]));
        }
      }
    }
    // Sijrs reads no RDM: Σ_J H̄_JJ times its one state's product.
    FBF_TRY(sijrs_hvp(res, blocks, pairs.front(), batch, dir(u, 0), shared, trace));
  }
  return out;
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
// and K annihilates S's dropped modes; pc_span_bar adds the rest, the kept
// span's rotation, where real modes are dropped). So, summed over the tuples:
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
// U_Jᵀ dA_J U_J: the same rule, still no eigenvalue derivative. b_{I←J} has
// no part along a null mode of S_J (a mode that annihilates |J⟩ annihilates
// ⟨J| S^{(J,I)}), but one of order √σ along a real dropped mode, which is
// pc_span_bar's.
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

// The PC forward tangent's state (pc_class_energies_tangent), which a walk
// of one state carries in place of bars: SC's ScTangent, whose regions find
// each slab term's operands' tangents (the inputs', then the class's T, t1
// and slabs'), and the orbital energies' tangents on the host, for Δ̇.
// @p span false leaves the kept span's rotation out
// (pc_class_energies_tangent's).
struct PcTangent {
  ScTangent t;
  std::vector<std::vector<double>> e_core, e_virt;
  std::vector<double> unused;
  bool span = true;

  // R Δ̇ of @p p along every direction, onto @p dot.
  void add_delta(const Perturber &p, const double r, std::vector<double> &dot) {
    for (size_t u = 0; u < dot.size(); ++u) {
      dot[u] +=
          r *
          perturber_delta(
              p, {.core = e_core[u], .virt = e_virt[u], .core_bar = unused, .virt_bar = unused});
    }
  }
};

// What every PC class's reverse walk reads of the n states besides
// ReverseContext: their operands and H̄, n x n row-major as the pairs, and the
// block bars each pair's matrices are built from, [I ns + J]: a state's own
// (every block) on the diagonal, pair (I, J)'s metrics off it. With
// @p tangent (one state), the walk forms Ė instead of the bars.
// One class of the PC Hessian-vector product (pc_class_energies_hvp), as
// its base walk keeps it for the fixed walks: the solve's U and S's modes
// (V, σ and the pair floor; their leaks a fixed walk forms anew over its own
// matrices), the multipliers, M and M_Δ (m x m, row-major as the walk
// accumulates them), the slab rows' ȳ and x̄ per slab, and per direction
// their tangents, U̇ = V_n Θ' (empty where nothing rotates) and the span
// bar's derivative in V and σ (d x d column-major, unsymmetrized; none
// where nothing rotates).
struct PcHvpClass {
  bool walked = false;
  int64_t m = 0;
  std::optional<Buffers::device<double>> u;
  std::optional<DroppedModes> modes;
  std::optional<Buffers::device<double>> mgg, mgg_delta;
  std::vector<Buffers::device<double>> mgg_dot, mgg_delta_dot;
  std::vector<std::vector<double>> ybar, xbar;
  std::vector<std::vector<std::vector<double>>> ybar_dot, xbar_dot;
  std::vector<Buffers::device<double>> u_dot;
  std::vector<std::optional<Buffers::device<double>>> span_dot;
  /// @brief A fixed walk's slab, counted as the base walk's.
  size_t slab = 0;
};

// The PC Hessian-vector product's walks. The base walk (@p fixed false, one
// state, with a tangent) keeps every class's PcHvpClass and the orbital
// energies' bar tangents [u]; a fixed walk then runs the reverse pass at the
// kept multipliers, their tangent along direction @p dot when set, over its
// own inputs' matrices with no solve, U moved by @p u_step U̇ of direction
// @p line. @p span false leaves the kept span's rotation out.
struct PcHvp {
  bool fixed = false;
  bool span = true;
  std::array<PcHvpClass, 8> classes;
  std::optional<size_t> dot;
  size_t line = 0;
  double u_step = 0.0;
  std::vector<std::vector<double>> core_dot, virt_dot;
};

struct PcStates {
  int64_t ns = 1;
  std::span<const ActiveOperands> ops;
  std::span<const double> hbar;
  std::span<BlockBars> m;
  PcTangent *tangent = nullptr;
  /// @brief The Hessian-vector product's walks (pc_class_energies_hvp): with
  /// a tangent, the base walk that keeps the multipliers; without, a fixed
  /// walk at them.
  PcHvp *hvp = nullptr;
};

// One class's reverse walk under way: its solves and transition rows, the
// walk bars of every pair (I, J) at [I ns + J], the couplings there, and each
// ket's denominators and amplitudes (PcDenominators). A tangent walk's Ė per
// direction, and per direction the projected pencil's tangents K̇' and Ṡ'
// (pc_solve_tangent).
struct PcReverse {
  const char *name;
  PcKets kets;
  std::vector<PcWalkBars> w;
  std::vector<double> energy;
  std::vector<PcDenominators> modes;
  std::vector<double> dot;
  std::vector<Buffers::device<double>> kdot, sdot;
  /// @brief The HVP's base walk's, per direction where the span rotates
  /// (empty elsewhere): Θ (nd x m, its pairs divided), Θ' = Θ V_kᵀ U, and
  /// Vᵀ Ṡ V's dropped and kept blocks, nd x nd and m x m.
  std::vector<Buffers::device<double>> theta, theta_u, edd, ekk;
  /// @brief The class's builder, which the HVP's walks rebuild from.
  PcBuild build = nullptr;
};

// Per direction, the tangent of a one-state walk's solve at its own
// factorization: S's, K's and S₁'s tangents, their builder's central
// polarization (exact, as block_tangent's), S and K symmetrized as the solve
// symmetrized them. The kept span's first-order rotation into the dropped
// modes, the forward side of pc_span_bar: S = V diag(σ) Vᵀ moves V_k by
// V_n Θ, Θ_nl = (V_nᵀ Ṡ V_k)_nl / (σ_l - σ_n), the pairs within
// DroppedModes::floor skipped as there, so U = V_k σ^-½ W moves by
// U̇ = V_n Θ' with Θ' = Θ V_kᵀ U (the in-span part of U̇ changes no energy).
// Then T = S U and t1 = S₁ᵀ U move by
//   Ṫ = Ṡ U + S U̇ = Ṡ U + leakᵀ Θ',  ṫ1 = Ṡ₁ᵀ U + leak1ᵀ Θ',
// registered as their regions' tangents, and the projected pencil Uᵀ A U by
//   K̇' = Uᵀ K̇ U + X + Xᵀ, X = Uᵀ K U̇ = kvuᵀ Θ',   Ṡ' = Uᵀ Ṡ U,
// Uᵀ S U̇ being σ_n Uᵀ V_n Θ' = 0, so A = K + Δ S moves by K̇' + Δ Ṡ' + Δ̇ 1.
Status pc_solve_tangent(ReverseContext &c, const PcStates &q, PcReverse &r, const int64_t d,
                        const int64_t n1, const PcBuild build) {
  DeviceResources &res = c.res;
  ScTangent &t = q.tangent->t;
  const ClassSolve &cs = r.kets.solves.front();
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nd = d - m;
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  const auto ndi = static_cast<int32_t>(nd);
  const auto n1i = static_cast<int32_t>(n1);
  const DroppedModes *const dm = cs.modes ? &*cs.modes : nullptr;
  const double *const vk = dm == nullptr ? nullptr : dm->v.data() + (nd * d);
  // V_kᵀ U, m x m.
  Buffers::device<double> vku = FBF_TRY(device_zeros(res, dm == nullptr ? 0 : m * m));
  if (dm != nullptr) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, vk,
                                               cs.u.data(), &kZero<double>, vku.data())));
  }
  TangentRegion t_region{.base = cs.t.data(), .count = d * m};
  TangentRegion t1_region{.base = cs.t1 ? cs.t1->data() : nullptr, .count = n1 * m};
  r.dot.assign(t.directions.size(), 0.0);
  for (size_t u = 0; u < t.directions.size(); ++u) {
    Buffers::device<double> tdot = FBF_TRY(device_zeros(res, d * m));
    Buffers::device<double> t1dot = FBF_TRY(device_zeros(res, n1 * m));
    Buffers::device<double> kdot = FBF_TRY(device_zeros(res, m * m));
    Buffers::device<double> sdot = FBF_TRY(device_zeros(res, m * m));
    const double step = t.steps[u].blocks;
    if (step != 0.0) {
      Buffers::device<double> s_raw = FBF_TRY(device_zeros(res, d * d));
      Buffers::device<double> k_raw = FBF_TRY(device_zeros(res, d * d));
      Buffers::device<double> s1_raw = FBF_TRY(device_zeros(res, d * n1));
      for (const double sign : {1.0, -1.0}) {
        const ShiftedOperands x =
            FBF_TRY(shifted_operands(res, q.ops.front(), t.directions[u].in, sign * step));
        const PcMatrices mats = FBF_TRY(build(res, x.ops, true));
        const double w = sign / (2.0 * step);
        FBF_TRY(add_scaled(res, mats.s.data(), d * d, s_raw.data(), w));
        FBF_TRY(add_scaled(res, mats.k->data(), d * d, k_raw.data(), w));
        if (n1 > 0) {
          FBF_TRY(add_scaled(res, mats.s1->data(), d * n1, s1_raw.data(), w));
        }
      }
      const Buffers::device<double> s_sym = FBF_TRY(symmetrized(res, s_raw.data(), d));
      const Buffers::device<double> k_sym = FBF_TRY(symmetrized(res, k_raw.data(), d));
      // Ṡ U, Ṡ₁ᵀ U (U column-major d x m read row-major as "ki", as solve_class).
      FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, s_sym.data(), cs.u.data(),
                              &kZero<double>, tdot.data())));
      if (n1 > 0) {
        FBF_TRY(einsum(res.stream(), "ia,ki->ak",
                       {EinsumOperand{.data = s1_raw.data(), .dims = {d, n1}},
                        EinsumOperand{.data = cs.u.data(), .dims = {m, d}}},
                       t1dot.data(), {n1, m}, 1.0));
      }
      Buffers::device<double> work(static_cast<size_t>(d * m), res.device());
      FBF_TRY(gpu_status(
          gemmwich(res.blas(), mi, di, cs.u.data(), k_sym.data(), work.data(), kdot.data())));
      FBF_TRY(gpu_status(
          gemmwich(res.blas(), mi, di, cs.u.data(), s_sym.data(), work.data(), sdot.data())));
      if (dm != nullptr && q.tangent->span) {
        // Θ: V_nᵀ (Ṡ V_k), divided pair by pair on the host (nd x m).
        FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, s_sym.data(), vk,
                                &kZero<double>, work.data())));
        Buffers::device<double> a(static_cast<size_t>(nd * m), res.device());
        FBF_TRY(
            gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), ndi, mi, di, &kOne<double>, dm->v.data(),
                                               work.data(), &kZero<double>, a.data())));
        if (q.hvp != nullptr) {
          // The HVP's span bar derivative reads Vᵀ Ṡ V's dropped and kept
          // blocks: V_kᵀ (Ṡ V_k) from work, and V_nᵀ (Ṡ V_n).
          Buffers::device<double> ekk(static_cast<size_t>(m * m), res.device());
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, vk,
                                                     work.data(), &kZero<double>, ekk.data())));
          Buffers::device<double> sv(static_cast<size_t>(d * nd), res.device());
          FBF_TRY(gpu_status(gemm(res.blas(), di, ndi, di, &kOne<double>, s_sym.data(),
                                  dm->v.data(), &kZero<double>, sv.data())));
          Buffers::device<double> edd(static_cast<size_t>(nd * nd), res.device());
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), ndi, ndi, di, &kOne<double>,
                                                     dm->v.data(), sv.data(), &kZero<double>,
                                                     edd.data())));
          r.edd.push_back(std::move(edd));
          r.ekk.push_back(std::move(ekk));
        }
        std::vector<double> theta = FBF_TRY(download(res, a.data(), nd * m));
        for (int64_t l = 0; l < m; ++l) {
          for (int64_t n = 0; n < nd; ++n) {
            const double gap =
                dm->sigma[static_cast<size_t>(nd + l)] - dm->sigma[static_cast<size_t>(n)];
            double &e = theta[static_cast<size_t>(n + (nd * l))];
            e = gap > dm->floor ? e / gap : 0.0;
          }
        }
        Buffers::device<double> theta_d = FBF_TRY(upload(res, theta));
        // Θ' = Θ V_kᵀ U.
        FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, mi, &kOne<double>, theta_d.data(), vku.data(),
                                &kZero<double>, a.data())));
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), di, mi, ndi, &kOne<double>,
                                                   dm->leak.data(), a.data(), &kOne<double>,
                                                   tdot.data())));
        if (n1 > 0) {
          // ṫ1 row-major n1 x m is column-major m x n1: += Θ'ᵀ leak1.
          FBF_TRY(
              gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, n1i, ndi, &kOne<double>, a.data(),
                                                 dm->leak1.data(), &kOne<double>, t1dot.data())));
        }
        Buffers::device<double> x(static_cast<size_t>(m * m), res.device());
        FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, ndi, &kOne<double>,
                                                   dm->kvu.data(), a.data(), &kZero<double>,
                                                   x.data())));
        const EinsumOperand x_op{.data = x.data(), .dims = {m, m}};
        FBF_TRY(einsum(res.stream(), "ij->ij", {x_op}, kdot.data(), {m, m}, 1.0));
        FBF_TRY(transpose(res.stream(), x_op, {1, 0}, kdot.data(), 1.0));
        if (q.hvp != nullptr) {
          FBF_TRY(res.sync());
          r.theta.push_back(std::move(theta_d));
          r.theta_u.push_back(std::move(a));
        }
      }
    }
    if (q.hvp != nullptr) {
      // Where nothing rotates along this direction, empty.
      for (auto *v : {&r.theta, &r.theta_u, &r.edd, &r.ekk}) {
        if (v->size() < u + 1) {
          v->push_back(FBF_TRY(device_zeros(res, 0)));
        }
      }
    }
    t_region.dots.push_back(tdot.data());
    t1_region.dots.push_back(t1dot.data());
    t.held.push_back(std::move(tdot));
    t.held.push_back(std::move(t1dot));
    r.kdot.push_back(std::move(kdot));
    r.sdot.push_back(std::move(sdot));
  }
  t.regions.push_back(std::move(t_region));
  if (n1 > 0) {
    t.regions.push_back(std::move(t1_region));
  }
  return {};
}

// Class @p name's index in the eight (kScClassNames').
size_t pc_class_index(const std::string_view name) {
  return static_cast<size_t>(std::ranges::find(kScClassNames, name) - kScClassNames.begin());
}

// @p count doubles at @p x, into a buffer of their own.
Result<Buffers::device<double>> duplicate(DeviceResources &res, const double *const x,
                                          const int64_t count) {
  Buffers::device<double> out = FBF_TRY(device_zeros(res, count));
  if (count > 0) {
    FBF_TRY(add_scaled(res, x, count, out.data(), 1.0));
  }
  return out;
}

// A fixed walk's kets (one state): the class's matrices over this walk's
// own operands, no solve. U is the base walk's, moved by u_step U̇ of the
// line's direction; T = S U and t1 = S₁ᵀ U are formed over the matrices, and
// with the span the dropped modes' leaks V_nᵀ S, V_nᵀ S₁ and V_nᵀ K U over
// the base walk's V and σ.
Result<PcReverse> pc_fixed_kets(ReverseContext &c, const PcStates &q, const char *name,
                                const int64_t d, const int64_t n1, const PcBuild build) {
  DeviceResources &res = c.res;
  const PcHvp &hvp = *q.hvp;
  PcHvpClass &h = q.hvp->classes[pc_class_index(name)];
  h.slab = 0;
  const int64_t m = h.m;
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  const PcMatrices mats = FBF_TRY(build(res, q.ops.front(), true));
  const Buffers::device<double> s_sym = FBF_TRY(symmetrized(res, mats.s.data(), d));
  Buffers::device<double> u = FBF_TRY(duplicate(res, h.u->data(), d * m));
  if (hvp.u_step != 0.0 && hvp.line < h.u_dot.size() && h.u_dot[hvp.line].num_elements() > 0) {
    FBF_TRY(add_scaled(res, h.u_dot[hvp.line].data(), d * m, u.data(), hvp.u_step));
  }
  Buffers::device<double> t(static_cast<size_t>(d * m), res.device());
  FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, s_sym.data(), u.data(),
                          &kZero<double>, t.data())));
  std::optional<Buffers::device<double>> t1;
  if (n1 > 0) {
    // U column-major d x m reads row-major as the m x d "ki", as solve_class.
    t1 = FBF_TRY(device_zeros(res, n1 * m));
    FBF_TRY(einsum(res.stream(), "ia,ki->ak",
                   {EinsumOperand{.data = mats.s1->data(), .dims = {d, n1}},
                    EinsumOperand{.data = u.data(), .dims = {m, d}}},
                   t1->data(), {n1, m}, 1.0));
  }
  std::optional<DroppedModes> modes;
  if (hvp.span && h.modes) {
    const Buffers::device<double> k_sym = FBF_TRY(symmetrized(res, mats.k->data(), d));
    modes = FBF_TRY(dropped_modes(res, FBF_TRY(duplicate(res, h.modes->v.data(), d * d)),
                                  h.modes->sigma, h.modes->floor, k_sym.data(), s_sym.data(),
                                  u.data(), d, m, n1 > 0 ? mats.s1->data() : nullptr, n1));
  }
  FBF_TRY(res.sync());
  PcReverse r{.name = name,
              .kets = PcKets{.ns = 1},
              .w = {},
              .energy = {0.0},
              .modes = std::vector<PcDenominators>(1),
              .dot = {},
              .kdot = {},
              .sdot = {},
              .build = build};
  r.kets.solves.push_back(ClassSolve{.lambda = std::vector<double>(static_cast<size_t>(m), 0.0),
                                     .t = std::move(t),
                                     .t1 = std::move(t1),
                                     .dropped = d - m,
                                     .u = std::move(u),
                                     .modes = std::move(modes)});
  r.kets.ms = {m};
  r.kets.given = {true};
  r.kets.t.resize(1);
  r.kets.t1.resize(1);
  r.kets.leak.resize(1);
  r.kets.leak1.resize(1);
  r.w.push_back(FBF_TRY(pc_walk_bars(res, m, d, n1, true)));
  return r;
}

Result<PcReverse> pc_reverse_kets(ReverseContext &c, const PcStates &q, const char *name,
                                  const int64_t d, const int64_t n1, const PcBuild build) {
  if (q.hvp != nullptr && q.hvp->fixed) {
    return pc_fixed_kets(c, q, name, d, n1, build);
  }
  const int64_t ns = q.ns;
  PcReverse r{.name = name,
              .kets = FBF_TRY(pc_kets(c.res, name, q.ops, ns, d, n1, build, nullptr, true)),
              .w = {},
              .energy = std::vector<double>(static_cast<size_t>(ns * ns), 0.0),
              .modes = std::vector<PcDenominators>(static_cast<size_t>(ns)),
              .dot = {},
              .kdot = {},
              .sdot = {},
              .build = build};
  for (int64_t p = 0; p < ns * ns; ++p) {
    r.w.push_back(FBF_TRY(pc_walk_bars(c.res, r.kets.m(p % ns), d, n1, p / ns == p % ns)));
  }
  if (q.tangent != nullptr) {
    FBF_TRY(pc_solve_tangent(c, q, r, d, n1, build));
  }
  if (q.hvp != nullptr) {
    // The base walk: the class's record, its multipliers' tangents zeroed.
    PcHvpClass &h = q.hvp->classes[pc_class_index(name)];
    h = PcHvpClass{};
    h.walked = true;
    h.m = r.kets.m(0);
    const size_t dirs = q.tangent->t.directions.size();
    for (size_t u = 0; u < dirs; ++u) {
      h.mgg_dot.push_back(FBF_TRY(device_zeros(c.res, h.m * h.m)));
      h.mgg_delta_dot.push_back(FBF_TRY(device_zeros(c.res, h.m * h.m)));
    }
    h.ybar_dot.resize(dirs);
    h.xbar_dot.resize(dirs);
  }
  return r;
}

// One slab's rows as the HVP's base walk reads them (one state): y and the
// exchange rows x, their tangents per direction, and the reverse pass's ȳ,
// x̄, weights w and Δ per tuple.
struct PcSlabRows {
  const std::vector<double> &y, &x;
  const std::vector<std::vector<double>> &ydot, &xdot;
  const std::vector<double> &ybar, &xbar, &w, &delta;
};

// The HVP's base walk over one slab (one state): ȳ and x̄ kept, and per
// direction their tangents, M's and M_Δ's, and R's onto the orbital
// energies' bar tangents. With g = y / (λ + Δ) (g_x the exchange row's), the
// solve differentiated through its own factorization:
//   ġ = (ẏ - Ḋ g) / (λ + Δ),  Ḋ = K̇' + Δ Ṡ' + Δ̇ 1,
// ẏ and Ḋ the tangent's, the span's rotation in them. Then, as the reverse
// pass's multipliers are linear in g at a single coupling (ȳ = -2 s g,
// w = s g) and in (g, g_x) at a pair (ȳ = -(4 g - g_x) on the direct row,
// x̄ = g on the exchange row, w = 2 g - g_x), their tangents are the same in
// (ġ, ġ_x); M = Σ g wᵀ and M_Δ = Σ Δ g wᵀ move by Σ (ġ wᵀ + g ẇᵀ) and
// Σ Δ̇ g wᵀ + Δ (ġ wᵀ + g ẇᵀ); R = gᵀ (Uᵀ S U) w by ġ·w + g·ẇ + gᵀ Ṡ' w.
Status pc_hvp_slab(ReverseContext &c, const PcStates &q, PcReverse &r,
                   const std::vector<Perturber> &rows, const PcSlabRows &s) {
  DeviceResources &res = c.res;
  const wwr::wwrStream_t stream = res.stream();
  PcHvpClass &h = q.hvp->classes[pc_class_index(r.name)];
  const std::vector<double> &lambda = r.kets.solves.front().lambda;
  const auto m = static_cast<size_t>(std::ssize(lambda));
  const auto mi = static_cast<int32_t>(m);
  const bool exchange = !s.x.empty();
  const auto nt = static_cast<int64_t>(s.y.size() / std::max<size_t>(m, 1));
  const auto nti = static_cast<int32_t>(nt);
  const auto row_of = [&](const Perturber &p) { return static_cast<size_t>(p.t1) * m; };
  const auto xrow_of = [&](const Perturber &p) {
    return p.t2 < 0 ? row_of(p) : static_cast<size_t>(p.t2) * m;
  };
  // g and g_x, each at its tuple's row (as w is).
  std::vector<double> g(s.y.size(), 0.0);
  std::vector<double> gx(exchange ? s.y.size() : 0, 0.0);
  for (size_t t = 0; t < rows.size(); ++t) {
    const Perturber &p = rows[t];
    for (size_t k = 0; k < m; ++k) {
      const double den = lambda[k] + s.delta[t];
      g[row_of(p) + k] = s.y[row_of(p) + k] / den;
      if (p.t2 >= 0) {
        gx[row_of(p) + k] = s.x[xrow_of(p) + k] / den;
      }
    }
  }
  h.ybar.push_back(s.ybar);
  h.xbar.push_back(s.xbar);
  const Buffers::device<double> g_d = FBF_TRY(upload(res, g));
  const Buffers::device<double> gx_d = FBF_TRY(upload(res, gx));
  const Buffers::device<double> w_d = FBF_TRY(upload(res, s.w));
  const Buffers::device<double> delta_d = FBF_TRY(upload(res, s.delta));
  Buffers::device<double> work(static_cast<size_t>(std::max<int64_t>(nt, 1) * mi), res.device());
  // Ḋ's matrices times the rows' amplitudes, m x nt column-major (each
  // matrix symmetric), read back.
  const auto times = [&](const Buffers::device<double> &a,
                         const Buffers::device<double> &v) -> Result<std::vector<double>> {
    if (v.num_elements() == 0) {
      return std::vector<double>{};
    }
    FBF_TRY(gpu_status(gemm(res.blas(), mi, nti, mi, &kOne<double>, a.data(), v.data(),
                            &kZero<double>, work.data())));
    return download(res, work.data(), nt * static_cast<int64_t>(m));
  };
  PcTangent &tg = *q.tangent;
  for (size_t u = 0; u < tg.t.directions.size(); ++u) {
    const std::vector<double> kg = FBF_TRY(times(r.kdot[u], g_d));
    const std::vector<double> sg = FBF_TRY(times(r.sdot[u], g_d));
    const std::vector<double> kgx = FBF_TRY(times(r.kdot[u], gx_d));
    const std::vector<double> sgx = FBF_TRY(times(r.sdot[u], gx_d));
    std::vector<double> ybar_dot(s.ybar.size(), 0.0);
    std::vector<double> xbar_dot(s.xbar.size(), 0.0);
    std::vector<double> g_dot(g.size(), 0.0);
    std::vector<double> w_dot(g.size(), 0.0);
    std::vector<double> delta_dot(rows.size(), 0.0);
    std::vector<double> unused;
    const OrbitalEnergies moved{
        .core = tg.e_core[u], .virt = tg.e_virt[u], .core_bar = unused, .virt_bar = unused};
    const OrbitalEnergies onto{.core = c.e.core,
                               .virt = c.e.virt,
                               .core_bar = q.hvp->core_dot[u],
                               .virt_bar = q.hvp->virt_dot[u]};
    for (size_t t = 0; t < rows.size(); ++t) {
      const Perturber &p = rows[t];
      const double dd = perturber_delta(p, moved);
      delta_dot[t] = dd;
      const size_t row = row_of(p);
      double r_dot = 0.0;
      for (size_t k = 0; k < m; ++k) {
        const double den = lambda[k] + s.delta[t];
        const size_t e = row + k;
        const double gd = (s.ydot[u][e] - kg[e] - (s.delta[t] * sg[e]) - (dd * g[e])) / den;
        g_dot[e] = gd;
        if (p.t2 < 0) {
          ybar_dot[e] += -2.0 * p.s * gd;
          w_dot[e] = p.s * gd;
          r_dot += (s.w[e] * gd) + (g[e] * w_dot[e]) + (g[e] * p.s * sg[e]);
        } else {
          const size_t xe = xrow_of(p) + k;
          const double gxd = (s.xdot[u][xe] - kgx[e] - (s.delta[t] * sgx[e]) - (dd * gx[e])) / den;
          ybar_dot[e] += -((4.0 * gd) - gxd);
          xbar_dot[xe] += gd;
          w_dot[e] = (2.0 * gd) - gxd;
          r_dot += (s.w[e] * gd) + (g[e] * w_dot[e]) + (g[e] * ((2.0 * sg[e]) - sgx[e]));
        }
      }
      add_delta_bar(p, r_dot, onto);
    }
    h.ybar_dot[u].push_back(std::move(ybar_dot));
    h.xbar_dot[u].push_back(std::move(xbar_dot));
    const Buffers::device<double> g_dot_d = FBF_TRY(upload(res, g_dot));
    const Buffers::device<double> w_dot_d = FBF_TRY(upload(res, w_dot));
    const Buffers::device<double> delta_dot_d = FBF_TRY(upload(res, delta_dot));
    const auto op = [&](const Buffers::device<double> &b) {
      return EinsumOperand{.data = b.data(), .dims = {nt, static_cast<int64_t>(m)}};
    };
    const EinsumOperand dt{.data = delta_d.data(), .dims = {nt}};
    const EinsumOperand ddt{.data = delta_dot_d.data(), .dims = {nt}};
    const std::vector<int64_t> mm{static_cast<int64_t>(m), static_cast<int64_t>(m)};
    double *const mdot = h.mgg_dot[u].data();
    double *const mddot = h.mgg_delta_dot[u].data();
    FBF_TRY(einsum(stream, "tk,tl->kl", {op(g_dot_d), op(w_d)}, mdot, mm, 1.0));
    FBF_TRY(einsum(stream, "tk,tl->kl", {op(g_d), op(w_dot_d)}, mdot, mm, 1.0));
    FBF_TRY(einsum(stream, "t,tk,tl->kl", {dt, op(g_dot_d), op(w_d)}, mddot, mm, 1.0));
    FBF_TRY(einsum(stream, "t,tk,tl->kl", {dt, op(g_d), op(w_dot_d)}, mddot, mm, 1.0));
    FBF_TRY(einsum(stream, "t,tk,tl->kl", {ddt, op(g_d), op(w_d)}, mddot, mm, 1.0));
    FBF_TRY(res.sync());
  }
  return {};
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
//
// A tangent walk (one state) takes no adjoint: per direction it forms ẏ and
// ẋ, every term at its operands' tangents (@p slabs, the slabs it reads,
// rebuilt along Ḃ), and adds ȳ·ẏ + x̄·ẋ + R Δ̇ to Ė; the rest of Ė,
// <M, K̇'> + <M_Δ, Ṡ'>, is pc_reverse_finish's.
Status pc_slab_reverse(ReverseContext &c, const PcStates &q, PcReverse &r,
                       const std::function<std::vector<ClassTerm>(int64_t, int64_t)> &terms,
                       const std::vector<int64_t> &tuple_dims, const std::vector<Perturber> &rows,
                       const std::vector<SlabRead> &slabs) {
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
  if (q.hvp != nullptr && q.hvp->fixed) {
    // A fixed walk: the base walk's ȳ and x̄ for this slab (or their tangent
    // along hvp.dot), each term's adjoint taken against them, nothing
    // evaluated.
    PcHvpClass &h = q.hvp->classes[pc_class_index(r.name)];
    const size_t at_slab = h.slab++;
    const auto &ybar = q.hvp->dot ? h.ybar_dot[*q.hvp->dot] : h.ybar;
    const auto &xbar = q.hvp->dot ? h.xbar_dot[*q.hvp->dot] : h.xbar;
    if (at_slab >= ybar.size()) {
      return err_config("nevpt2::pc_class_energies_hvp: {}'s fixed walk has more slabs than its "
                        "base walk ({})",
                        r.name, ybar.size());
    }
    const Buffers::device<double> ybar_d = FBF_TRY(upload(res, ybar[at_slab]));
    const Buffers::device<double> xbar_d = FBF_TRY(upload(res, xbar[at_slab]));
    return adjoint_terms(res, pair_terms.front(), ybar_d.data(), exchange ? xbar_d.data() : nullptr,
                         dims(0), c.ops.delta, nullptr);
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
    PcDenominators &own = r.modes[static_cast<size_t>(j)];
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
        own.add(den, y[row + k]);
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
      if (q.tangent != nullptr) {
        q.tangent->add_delta(p, rbar, r.dot);
      }
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
  if (q.tangent != nullptr) {
    const auto [ydot, xdot] =
        FBF_TRY(tangent_tables(res, q.tangent->t, pair_terms.front(), slabs, dims(0)));
    for (size_t u = 0; u < r.dot.size(); ++u) {
      r.dot[u] += std::inner_product(ybar.begin(), ybar.end(), ydot[u].begin(), 0.0) +
                  std::inner_product(xbar.begin(), xbar.end(), xdot[u].begin(), 0.0);
    }
    if (q.hvp == nullptr) {
      return {};
    }
    // The HVP's base walk also takes the adjoint below, for T̄ and t̄1.
    FBF_TRY(pc_hvp_slab(c, q, r, rows,
                        {.y = y,
                         .x = x,
                         .ydot = ydot,
                         .xdot = xdot,
                         .ybar = ybar,
                         .xbar = xbar,
                         .w = w,
                         .delta = delta}));
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

// @p g (nd x m column-major, over @p dm's dropped and kept modes) divided
// pair by pair by σ_l - σ_n, a pair within σ's round-off (DroppedModes::floor)
// zeroed, into a buffer of its own.
Result<Buffers::device<double>> by_gaps(DeviceResources &res, const Buffers::device<double> &g,
                                        const DroppedModes &dm, const int64_t m) {
  const auto nd = static_cast<int64_t>(dm.sigma.size()) - m;
  std::vector<double> gh = FBF_TRY(download(res, g.data(), nd * m));
  for (int64_t l = 0; l < m; ++l) {
    for (int64_t n = 0; n < nd; ++n) {
      const double gap = dm.sigma[static_cast<size_t>(nd + l)] - dm.sigma[static_cast<size_t>(n)];
      double &e = gh[static_cast<size_t>(n + (nd * l))];
      e = gap > dm.floor ? e / gap : 0.0;
    }
  }
  return upload(res, gh);
}

// The bar on ket J's S from its kept span's rotation into its dropped modes,
// d x d column-major, before symmetrization; none when nothing was dropped.
// The fixed-U pass above holds S's kept span still, which is exact only when
// every row the solve reads lies in it. A transition row does not: its leak
// V_nᵀ b_{I←J} = V_nᵀ S^{(J,I)} x is of order √σ_n against the own row's σ_n,
// since ⟨J| O_nᵀ is only √σ_n small (README "The PC reverse pass", #862). So
// with δV_k = V_n Θ, Θ_nl = (V_nᵀ δS V_k)_nl / (σ_l - σ_n), and Ū = dL/dU at
// fixed S, K and S₁,
//   Ū = Σ_I (S^{(J,I)} T̄_{I←J} + S₁^{(J,I)} t̄1_{I←J}) + K U (M + Mᵀ)
// (the own pair's S symmetrized; S U (M_Δ + M_Δᵀ) has no dropped part), and
// U = V_k σ^-½ W gives dL/dS = V_n [(V_nᵀ Ū Uᵀ V_k) ⊘ (σ_l - σ_n)] V_kᵀ. A
// pair within σ's round-off of each other (DroppedModes::floor) is skipped:
// at the cut's own noise multiple instead, the pairs that straddle the cut
// were all skipped, and PSB3's state 0 missed by 8.6e-9 still (#862).
// Its coefficients, Ψ = (V_nᵀ Ū Uᵀ V_k) ⊘ (σ_l - σ_n), nd x m column-major
// (pairs within σ's round-off zeroed), are pc_span_coefficients'; none when
// nothing was dropped.
Result<std::optional<Buffers::device<double>>>
pc_span_coefficients(DeviceResources &res, const PcStates &q, PcReverse &r, const int64_t j,
                     const int64_t d, const int64_t n1) {
  const int64_t ns = q.ns;
  const ClassSolve &cs = r.kets.solves[static_cast<size_t>(j)];
  if (!cs.modes) {
    return std::nullopt;
  }
  const DroppedModes &dm = *cs.modes;
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nd = d - m;
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  const auto ndi = static_cast<int32_t>(nd);
  // Q = V_nᵀ Ū, nd x m.
  Buffers::device<double> qbar = FBF_TRY(device_zeros(res, nd * m));
  for (int64_t i = 0; i < ns; ++i) {
    const auto p = static_cast<size_t>((i * ns) + j);
    if (!r.kets.given[p]) {
      continue;
    }
    const double *const leak = i == j ? dm.leak.data() : r.kets.leak[p]->data();
    FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, di, &kOne<double>, leak, r.w[p].t.data(),
                            &kOne<double>, qbar.data())));
    if (n1 > 0) {
      // t̄1 row-major n1 x m reads column-major as its transpose.
      const double *const leak1 = i == j ? dm.leak1.data() : r.kets.leak1[p]->data();
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), ndi, mi, static_cast<int32_t>(n1), &kOne<double>, leak1, r.w[p].t1.data(),
          &kOne<double>, qbar.data())));
    }
  }
  const double *const mgg = r.w[static_cast<size_t>((j * ns) + j)].mgg.data();
  FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, mi, &kOne<double>, dm.kvu.data(), mgg, &kOne<double>,
                          qbar.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), ndi, mi, mi, &kOne<double>, dm.kvu.data(), mgg, &kOne<double>, qbar.data())));
  // G = Q Uᵀ V_k, then divided pair by pair on the host (nd x m).
  const double *const vk = dm.v.data() + (nd * d);
  Buffers::device<double> uv(static_cast<size_t>(m * m), res.device());
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, cs.u.data(), vk,
                                             &kZero<double>, uv.data())));
  Buffers::device<double> g(static_cast<size_t>(nd * m), res.device());
  FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, mi, &kOne<double>, qbar.data(), uv.data(),
                          &kZero<double>, g.data())));
  return std::optional<Buffers::device<double>>{FBF_TRY(by_gaps(res, g, dm, m))};
}

Result<std::optional<Buffers::device<double>>> pc_span_bar(DeviceResources &res, const PcStates &q,
                                                           PcReverse &r, const int64_t j,
                                                           const int64_t d, const int64_t n1) {
  const std::optional<Buffers::device<double>> gq =
      FBF_TRY(pc_span_coefficients(res, q, r, j, d, n1));
  if (!gq) {
    return std::nullopt;
  }
  const ClassSolve &cs = r.kets.solves[static_cast<size_t>(j)];
  const DroppedModes &dm = *cs.modes;
  const int64_t m = std::ssize(cs.lambda);
  const int64_t nd = d - m;
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  const auto ndi = static_cast<int32_t>(nd);
  const double *const vk = dm.v.data() + (nd * d);
  // V_n Ψ V_kᵀ.
  Buffers::device<double> y(static_cast<size_t>(d * m), res.device());
  FBF_TRY(gpu_status(gemm(res.blas(), di, mi, ndi, &kOne<double>, dm.v.data(), gq->data(),
                          &kZero<double>, y.data())));
  Buffers::device<double> x(static_cast<size_t>(d * d), res.device());
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), di, di, mi, &kOne<double>, y.data(), vk, &kZero<double>, x.data())));
  return std::optional<Buffers::device<double>>{std::move(x)};
}

// One class's bars on one pair's raw matrices @p b back onto the blocks they
// were built from over @p o (@p own: a state's own S, K and S₁; else a
// transition's S and S₁), and Σ_G <M_G, G> over those blocks added to the
// device scalar @p acc. Each rebuilds the block values its sum reads.
using PcBack = Status (*)(DeviceResources &, const ActiveOperands &o, bool own,
                          const PcMatrixBars &b, BlockBars &m, double *acc);

// One PC class over the n states: its couplings [I ns + J], each ket's
// dropped count and modes, and its homogeneity sum.
struct PcClassSums {
  std::vector<double> energy;
  std::vector<int64_t> dropped;
  double homogeneity = 0.0;
  std::vector<PcDenominators> modes;
  /// @brief A tangent walk's Ė per direction.
  std::vector<double> dot;
  /// @brief Each ket's largest dropped σ and smallest kept one, relative to
  /// its largest (PcClassEnergies::cut, kept); empty reads as 0.
  std::vector<double> cut, kept;
};

// A class with no tuple: nothing coupled, nothing dropped.
PcClassSums pc_no_tuples(const int64_t ns) {
  return {.energy = std::vector<double>(static_cast<size_t>(ns * ns), 0.0),
          .dropped = std::vector<int64_t>(static_cast<size_t>(ns), 0),
          .homogeneity = 0.0};
}

// @p x (d x d column-major) symmetrized onto the raw S bar @p s, as the
// solve symmetrized S.
Status add_symmetrized(DeviceResources &res, const Buffers::device<double> &x, double *const s,
                       const int64_t d) {
  const EinsumOperand op{.data = x.data(), .dims = {d, d}};
  FBF_TRY(einsum(res.stream(), "ij->ij", {op}, s, {d, d}, 0.5));
  return transpose(res.stream(), op, {1, 0}, s, 0.5);
}

// A fixed walk's finish (one state): M and M_Δ (or their tangent) as the
// base walk kept them, the matrix bars over this walk's T̄ and t̄1, the span
// bar over this walk's leaks, and along a direction the span bar's
// derivative in V and σ (pc_hvp_keep's), back onto the blocks. Nothing is
// checked: the base walk has.
Result<PcClassSums> pc_fixed_finish(ReverseContext &c, const PcStates &q, PcReverse &r,
                                    const int64_t d, const int64_t n1, const PcBack back) {
  DeviceResources &res = c.res;
  const PcHvp &hvp = *q.hvp;
  const PcHvpClass &h = hvp.classes[pc_class_index(r.name)];
  PcWalkBars &w = r.w.front();
  const int64_t mm = h.m * h.m;
  FBF_TRY(
      add_scaled(res, hvp.dot ? h.mgg_dot[*hvp.dot].data() : h.mgg->data(), mm, w.mgg.data(), 1.0));
  FBF_TRY(add_scaled(res, hvp.dot ? h.mgg_delta_dot[*hvp.dot].data() : h.mgg_delta->data(), mm,
                     w.mgg_delta.data(), 1.0));
  std::optional<Buffers::device<double>> span;
  if (hvp.span) {
    span = FBF_TRY(pc_span_bar(res, q, r, 0, d, n1));
  }
  PcMatrixBars b = FBF_TRY(pc_matrix_bars(res, r.kets.solves.front(), w, d, n1, true));
  if (span) {
    FBF_TRY(add_symmetrized(res, *span, b.s.data(), d));
  }
  if (hvp.dot && *hvp.dot < h.span_dot.size() && h.span_dot[*hvp.dot]) {
    FBF_TRY(add_symmetrized(res, *h.span_dot[*hvp.dot], b.s.data(), d));
  }
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  FBF_TRY(back(res, q.ops.front(), true, b, q.m.front(), acc.data()));
  FBF_TRY(res.sync());
  return pc_no_tuples(1);
}

// The HVP's base walk, once a class has walked (one state): its record
// kept for the fixed walks (U, S's modes, M, M_Δ), and per direction where
// the span rotates, U̇ = V_n Θ' and the span bar's derivative in S's
// eigenvectors and eigenvalues at fixed matrices (Daleckii–Krein). With the
// span bar Y = V_n Ψ V_kᵀ, Ψ = (V_nᵀ R V_k) ⊘ Γ, R = Ū Uᵀ, Γ_nl = σ_l - σ_n,
// and E = Vᵀ Ṡ V, Θ = E_nk ⊘ Γ:
//   Ẏ = V_n [(E_nn Ψ - Ψ E_kk - Θ R_kk) ⊘ Γ] V_kᵀ + V_n Ψ Θᵀ V_nᵀ - V_k Θᵀ Ψ V_kᵀ,
// R_kk = V_kᵀ R V_k (R's other blocks vanish, U ⊥ V_n); every divided
// difference of 1 / (σ_l - σ_n) is a product of two first-order ones, so no
// pair within one side of the cut divides. A pair within σ's round-off
// rotates nothing, as at first order (Ψ and Θ zero there).
Status pc_hvp_keep(ReverseContext &c, const PcStates &q, PcReverse &r, const int64_t d,
                   const int64_t n1) {
  DeviceResources &res = c.res;
  PcHvpClass &h = q.hvp->classes[pc_class_index(r.name)];
  ClassSolve &cs = r.kets.solves.front();
  PcWalkBars &w = r.w.front();
  const int64_t m = h.m;
  const int64_t nd = d - m;
  const auto di = static_cast<int32_t>(d);
  const auto mi = static_cast<int32_t>(m);
  const auto ndi = static_cast<int32_t>(nd);
  const size_t dirs = q.tangent->t.directions.size();
  h.span_dot.resize(dirs);
  h.u_dot.clear();
  const bool rotates =
      q.hvp->span && cs.modes && std::ranges::any_of(r.theta, [](const Buffers::device<double> &x) {
        return x.num_elements() > 0;
      });
  if (rotates) {
    const DroppedModes &dm = *cs.modes;
    const double *const vn = dm.v.data();
    const double *const vk = dm.v.data() + (nd * d);
    const Buffers::device<double> psi = *FBF_TRY(pc_span_coefficients(res, q, r, 0, d, n1));
    // R_kk = (V_kᵀ Ū)(Uᵀ V_k), V_kᵀ Ū = (S V_k)ᵀ T̄ + (V_kᵀ S₁) t̄1ᵀ + (K V_k)ᵀ U (M + Mᵀ),
    // over the matrices rebuilt.
    const PcMatrices mats = FBF_TRY(r.build(res, q.ops.front(), true));
    const Buffers::device<double> s_sym = FBF_TRY(symmetrized(res, mats.s.data(), d));
    const Buffers::device<double> k_sym = FBF_TRY(symmetrized(res, mats.k->data(), d));
    Buffers::device<double> work(static_cast<size_t>(d * m), res.device());
    Buffers::device<double> vkbar(static_cast<size_t>(m * m), res.device());
    FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, s_sym.data(), vk, &kZero<double>,
                            work.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, work.data(),
                                               w.t.data(), &kZero<double>, vkbar.data())));
    if (n1 > 0) {
      const Buffers::device<double> vks1 = FBF_TRY(leak_of(res, vk, mats.s1->data(), d, m, n1));
      // t̄1 row-major n1 x m reads column-major as its transpose.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), mi, mi, static_cast<int32_t>(n1), &kOne<double>, vks1.data(), w.t1.data(),
          &kOne<double>, vkbar.data())));
      FBF_TRY(res.sync());
    }
    FBF_TRY(gpu_status(gemm(res.blas(), di, mi, di, &kOne<double>, k_sym.data(), vk, &kZero<double>,
                            work.data())));
    Buffers::device<double> kvu(static_cast<size_t>(m * m), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, work.data(),
                                               cs.u.data(), &kZero<double>, kvu.data())));
    FBF_TRY(gpu_status(gemm(res.blas(), mi, mi, mi, &kOne<double>, kvu.data(), w.mgg.data(),
                            &kOne<double>, vkbar.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), mi, mi, mi, &kOne<double>, kvu.data(), w.mgg.data(), &kOne<double>,
        vkbar.data())));
    Buffers::device<double> uv(static_cast<size_t>(m * m), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, di, &kOne<double>, cs.u.data(),
                                               vk, &kZero<double>, uv.data())));
    Buffers::device<double> rkk(static_cast<size_t>(m * m), res.device());
    FBF_TRY(gpu_status(gemm(res.blas(), mi, mi, mi, &kOne<double>, vkbar.data(), uv.data(),
                            &kZero<double>, rkk.data())));
    Buffers::device<double> dk(static_cast<size_t>(nd * m), res.device());
    Buffers::device<double> dd(static_cast<size_t>(nd * nd), res.device());
    Buffers::device<double> kk(static_cast<size_t>(m * m), res.device());
    Buffers::device<double> part(static_cast<size_t>(d * std::max(m, nd)), res.device());
    for (size_t u = 0; u < dirs; ++u) {
      const Buffers::device<double> &theta = r.theta[u];
      if (theta.num_elements() == 0) {
        h.u_dot.push_back(FBF_TRY(device_zeros(res, 0)));
        continue;
      }
      // U̇ = V_n Θ'.
      Buffers::device<double> u_dot(static_cast<size_t>(d * m), res.device());
      FBF_TRY(gpu_status(gemm(res.blas(), di, mi, ndi, &kOne<double>, vn, r.theta_u[u].data(),
                              &kZero<double>, u_dot.data())));
      h.u_dot.push_back(std::move(u_dot));
      // (E_nn Ψ - Ψ E_kk - Θ R_kk) ⊘ Γ.
      FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, ndi, &kOne<double>, r.edd[u].data(), psi.data(),
                              &kZero<double>, dk.data())));
      FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, mi, &kNegOne<double>, psi.data(),
                              r.ekk[u].data(), &kOne<double>, dk.data())));
      FBF_TRY(gpu_status(gemm(res.blas(), ndi, mi, mi, &kNegOne<double>, theta.data(), rkk.data(),
                              &kOne<double>, dk.data())));
      const Buffers::device<double> dk_g = FBF_TRY(by_gaps(res, dk, dm, m));
      // Ψ Θᵀ and -Θᵀ Ψ.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), ndi, ndi, mi, &kOne<double>, psi.data(), theta.data(), &kZero<double>,
          dd.data())));
      FBF_TRY(
          gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), mi, mi, ndi, &kNegOne<double>,
                                             theta.data(), psi.data(), &kZero<double>, kk.data())));
      Buffers::device<double> y(static_cast<size_t>(d * d), res.device());
      // V_n DK V_kᵀ + V_n DD V_nᵀ + V_k KK V_kᵀ.
      FBF_TRY(gpu_status(gemm(res.blas(), di, mi, ndi, &kOne<double>, vn, dk_g.data(),
                              &kZero<double>, part.data())));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), di, di, mi, &kOne<double>, part.data(), vk, &kZero<double>, y.data())));
      FBF_TRY(gpu_status(gemm(res.blas(), di, ndi, ndi, &kOne<double>, vn, dd.data(),
                              &kZero<double>, part.data())));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), di, di, ndi, &kOne<double>, part.data(), vn, &kOne<double>, y.data())));
      FBF_TRY(gpu_status(
          gemm(res.blas(), di, mi, mi, &kOne<double>, vk, kk.data(), &kZero<double>, part.data())));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), di, di, mi, &kOne<double>, part.data(), vk, &kOne<double>, y.data())));
      FBF_TRY(res.sync());
      h.span_dot[u] = std::move(y);
    }
  }
  h.u = std::move(cs.u);
  h.modes = std::move(cs.modes);
  h.mgg = std::move(w.mgg);
  h.mgg_delta = std::move(w.mgg_delta);
  return res.sync();
}

// After a class's walk: each ket's denominators checked as the energy's, then
// every pair's matrix bars back through @p back, pair (I, J)'s onto ket J's
// own blocks at I = J and onto pair (J, I)'s, whose matrices its rows read,
// off it.
Result<PcClassSums> pc_reverse_finish(ReverseContext &c, const PcStates &q, PcReverse &r,
                                      const int64_t d, const int64_t n1, const PcBack back) {
  DeviceResources &res = c.res;
  const int64_t ns = q.ns;
  if (q.hvp != nullptr && q.hvp->fixed) {
    return pc_fixed_finish(c, q, r, d, n1, back);
  }
  for (int64_t j = 0; j < ns; ++j) {
    const Status ok = pc_intruder_check(r.name, r.modes[static_cast<size_t>(j)], d);
    if (!ok) {
      return state_error(ok.error(), j, ns);
    }
  }
  if (q.tangent != nullptr) {
    // Ė's last part, <M, K̇'> + <M_Δ, Ṡ'>, ket 0's.
    const auto dirs = std::ssize(r.dot);
    Buffers::device<double> acc = FBF_TRY(device_zeros(res, dirs));
    for (int64_t u = 0; u < dirs; ++u) {
      const auto uu = static_cast<size_t>(u);
      FBF_TRY(add_dot(res, r.w.front().mgg, r.kdot[uu].data(), acc.data() + u));
      FBF_TRY(add_dot(res, r.w.front().mgg_delta, r.sdot[uu].data(), acc.data() + u));
    }
    const std::vector<double> host = FBF_TRY(download(res, acc.data(), dirs));
    const ClassSolve &cs = r.kets.solves.front();
    PcClassSums out{.energy = std::move(r.energy),
                    .dropped = {cs.dropped},
                    .homogeneity = 0.0,
                    .modes = std::move(r.modes),
                    .dot = std::move(r.dot),
                    .cut = {cs.cut},
                    .kept = {cs.kept}};
    for (size_t u = 0; u < out.dot.size(); ++u) {
      out.dot[u] += host[u];
    }
    if (q.hvp != nullptr) {
      FBF_TRY(pc_hvp_keep(c, q, r, d, n1));
    }
    return out;
  }
  // Before pc_matrix_bars, which spends the own T̄.
  std::vector<std::optional<Buffers::device<double>>> span;
  for (int64_t j = 0; j < ns; ++j) {
    span.push_back(FBF_TRY(pc_span_bar(res, q, r, j, d, n1)));
  }
  Buffers::device<double> acc = FBF_TRY(device_zeros(res, 1));
  for (int64_t i = 0; i < ns; ++i) {
    for (int64_t j = 0; j < ns; ++j) {
      const auto built = static_cast<size_t>((j * ns) + i);
      PcMatrixBars b =
          FBF_TRY(pc_matrix_bars(res, r.kets.solves[static_cast<size_t>(j)],
                                 r.w[static_cast<size_t>((i * ns) + j)], d, n1, i == j));
      if (const auto &x = span[static_cast<size_t>(j)]; i == j && x) {
        // Onto the raw S, symmetrized as the solve symmetrized it.
        const EinsumOperand op{.data = x->data(), .dims = {d, d}};
        FBF_TRY(einsum(res.stream(), "ij->ij", {op}, b.s.data(), {d, d}, 0.5));
        FBF_TRY(transpose(res.stream(), op, {1, 0}, b.s.data(), 0.5));
      }
      FBF_TRY(back(res, q.ops[built], i == j, b, q.m[built], acc.data()));
    }
  }
  PcClassSums out{.energy = std::move(r.energy),
                  .dropped = {},
                  .homogeneity = FBF_TRY(scalar(res, acc)),
                  .modes = std::move(r.modes)};
  for (const ClassSolve &cs : r.kets.solves) {
    out.dropped.push_back(cs.dropped);
    out.cut.push_back(cs.cut);
    out.kept.push_back(cs.kept);
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
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb}, rows, {{ExtBlock::Sr, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb}, rows, {{ExtBlock::Si, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nr, nc, nc}, rows,
                            {{ExtBlock::Sijr, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(pc_slab_reverse(
        c, q, r, terms, {nb, nv, nc}, rows,
        {{ExtBlock::Srsi, b0, b1, slab_a.data()}, {ExtBlock::SrsiT, b0, b1, slab_b.data()}}));
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
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nv}, rows,
                            {{ExtBlock::Srs, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(pc_slab_reverse(c, q, r, terms, {nb, nc}, rows,
                            {{ExtBlock::Sij, b0, b1, slab_buf.data()}}));
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
    FBF_TRY(pc_slab_reverse(
        c, q, r, terms, {nb, nc}, rows,
        {{ExtBlock::Sir1, b0, b1, slab_1.data()}, {ExtBlock::Sir2, b0, b1, slab_2.data()}}));
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
      const size_t j = p % static_cast<size_t>(ns);
      out.elements[p].dropped[k] = x.dropped[j];
      out.elements[p].cut[k] = j < x.cut.size() ? x.cut[j] : 0.0;
      out.elements[p].kept[k] = j < x.kept.size() ? x.kept[j] : 0.0;
      pc_report_modes(x.modes, j, k, out.elements[p]);
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

// ---------------------------------------------------------------------------
// The PC forward tangent: each class's one-state walk with a tangent, and
// Sijrs's kernel (SC's)
// ---------------------------------------------------------------------------

Result<PcClassEnergiesTangent>
pc_class_energies_tangent(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          const int32_t batch, std::span<const EnergyDirection> directions,
                          const bool span) {
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(check_directions(blocks, directions, "pc_class_energies_tangent"));
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  const size_t dirs = directions.size();
  PcTangent tg{
      .t = FBF_TRY(input_tangent(res, blocks, in, directions)),
      .e_core = FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_core, blocks.ncore)),
      .e_virt = FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_virt, blocks.nvirt)),
      .unused = {},
      .span = span};
  std::vector<double> core_bar(o.e_core.size(), 0.0);
  std::vector<double> virt_bar(o.e_virt.size(), 0.0);
  const EnergyBars no_bars{};
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = no_bars,
      .batch = batch,
      .ops = o,
      .e = {.core = o.e_core, .virt = o.e_virt, .core_bar = core_bar, .virt_bar = virt_bar}};
  // A tangent walk sweeps no bars: the block bars are empty.
  std::vector<BlockBars> m;
  m.push_back(FBF_TRY(block_bars(res, blocks.ncas, false, false)));
  const double one = 1.0;
  const PcStates q{.ns = 1, .ops = {&o, 1}, .hbar = {&one, 1}, .m = m, .tangent = &tg};

  PcClassEnergiesTangent out;
  out.energy_dot.resize(dirs);
  out.total_dot.resize(dirs);
  const auto put = [&](const size_t k, const PcClassSums &x) {
    out.energies.energies[k] = x.energy.front();
    out.energies.dropped[k] = x.dropped.front();
    pc_report_modes(x.modes, 0, k, out.energies);
    out.energies.cut[k] = x.cut.empty() ? 0.0 : x.cut.front();
    out.energies.kept[k] = x.kept.empty() ? 0.0 : x.kept.front();
    for (size_t u = 0; u < dirs && !x.dot.empty(); ++u) {
      out.energy_dot[u][k] = x.dot[u];
    }
    // The class's T, t1 and slab buffers are freed: drop their regions.
    tg.t.regions.resize(tg.t.inputs);
    tg.t.held.clear();
  };
  put(0, FBF_TRY(pc_reverse_sr(c, q)));
  put(1, FBF_TRY(pc_reverse_si(c, q)));
  put(3, FBF_TRY(pc_reverse_sijr(c, q)));
  put(4, FBF_TRY(pc_reverse_srsi(c, q)));
  put(5, FBF_TRY(pc_reverse_srs(c, q)));
  put(6, FBF_TRY(pc_reverse_sij(c, q)));
  put(7, FBF_TRY(pc_reverse_sir(c, q)));
  // Sijrs has no active index: its energy and tangent are SC's.
  ClassEnergiesTangent sijrs;
  sijrs.energy_dot.resize(dirs);
  FBF_TRY(sijrs_tangent_walk(c, tg.t, sijrs));
  out.energies.energies[2] = sijrs.energies.energies[2];
  out.energies.total = std::ranges::fold_left(out.energies.energies, 0.0, std::plus{});
  for (size_t u = 0; u < dirs; ++u) {
    out.energy_dot[u][2] = sijrs.energy_dot[u][2];
    out.total_dot[u] = std::ranges::fold_left(out.energy_dot[u], 0.0, std::plus{});
  }
  return out;
}

// ---------------------------------------------------------------------------
// The PC Hessian-vector product: forward over the reverse pass
// ---------------------------------------------------------------------------

// Below this smallest kept metric eigenvalue, over its largest, a class's
// second derivative is not attainable (#917, README "The conditioning
// floor"): the HVP refuses rather than differentiate the solve twice there.
constexpr double kPcHessianKept = 1e-10;

// The PC classes with a solve, by index, and their walks.
using PcWalk = Result<PcClassSums> (*)(ReverseContext &, const PcStates &);
const std::array<std::pair<size_t, PcWalk>, 7> kPcWalks{{{0, pc_reverse_sr},
                                                         {1, pc_reverse_si},
                                                         {3, pc_reverse_sijr},
                                                         {4, pc_reverse_srsi},
                                                         {5, pc_reverse_srs},
                                                         {6, pc_reverse_sij},
                                                         {7, pc_reverse_sir}}};

// The PC reverse pass at @p hvp's kept multipliers (a fixed walk) over
// @p in and @p blocks, onto @p bars but the orbital energies': every class
// the base walk walked, then the sweep. Sijrs is not walked.
Status pc_fixed_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                        const int32_t batch, const EnergyBars &bars, PcHvp &hvp) {
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  std::vector<double> core_bar(o.e_core.size(), 0.0);
  std::vector<double> virt_bar(o.e_virt.size(), 0.0);
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = bars,
      .batch = batch,
      .ops = o,
      .e = {.core = o.e_core, .virt = o.e_virt, .core_bar = core_bar, .virt_bar = virt_bar}};
  std::vector<BlockBars> m;
  m.push_back(FBF_TRY(block_bars(res, blocks.ncas, true, true)));
  const double one = 1.0;
  const PcStates q{.ns = 1, .ops = {&o, 1}, .hbar = {&one, 1}, .m = m, .hvp = &hvp};
  for (const auto &[k, walk] : kPcWalks) {
    if (hvp.classes[k].walked) {
      FBF_TRY(walk(c, q));
    }
  }
  const std::array wanted{bars.h1e, bars.h2e, bars.dm1, bars.dm2, bars.dm3, bars.f3ca, bars.f3ac};
  if (std::ranges::any_of(wanted, [](const double *b) { return b != nullptr; })) {
    FBF_TRY(sweep_blocks(res, o, m.front(), bars));
  }
  return res.sync();
}

// The step U's symmetric difference takes along direction @p u: the power of
// two nearest ‖U‖ / ‖U̇‖ over the classes whose U moves (the difference is
// exact, so the step only keeps it clear of round-off); 0 when none does.
Result<double> pc_u_step(DeviceResources &res, const PcHvp &hvp, const size_t u) {
  double step = 0.0;
  for (const PcHvpClass &h : hvp.classes) {
    if (!h.walked || u >= h.u_dot.size() || h.u_dot[u].num_elements() == 0) {
      continue;
    }
    Buffers::device<double> acc = FBF_TRY(device_zeros(res, 2));
    const auto count = static_cast<int64_t>(h.u->num_elements());
    FBF_TRY(add_squares(res, h.u->data(), count, acc.data()));
    FBF_TRY(add_squares(res, h.u_dot[u].data(), count, acc.data() + 1));
    const std::vector<double> sq = FBF_TRY(download(res, acc.data(), 2));
    const double s = polarization_step(std::sqrt(sq[0]), std::sqrt(sq[1]));
    if (s != 0.0) {
      step = step == 0.0 ? s : std::min(step, s);
    }
  }
  return step;
}

Result<PcClassEnergiesHvp> pc_class_energies_hvp(DeviceResources &res, const MoBlocks &blocks,
                                                 const EnergyInputs &in, const int32_t batch,
                                                 std::span<const EnergyDirection> directions,
                                                 std::span<const EnergyBars> bars,
                                                 const bool span) {
  const ScopedDeviceIndex on{res.dev_idx()};
  if (bars.size() != directions.size()) {
    return err_config("nevpt2::pc_class_energies_hvp: {} bars for {} directions", bars.size(),
                      directions.size());
  }
  FBF_TRY(check_directions(blocks, directions, "pc_class_energies_hvp"));
  const ActiveOperands o = FBF_TRY(active_operands(res, blocks, in));
  const size_t dirs = directions.size();

  // The base walk: every class's solve, its tangent along every direction,
  // and the multipliers and their tangents kept.
  PcTangent tg{
      .t = FBF_TRY(input_tangent(res, blocks, in, directions)),
      .e_core = FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_core, blocks.ncore)),
      .e_virt = FBF_TRY(orbital_energy_dots(res, directions, &EnergyInputs::e_virt, blocks.nvirt)),
      .unused = {},
      .span = span};
  PcHvp hvp{
      .fixed = false,
      .span = span,
      .classes = {},
      .dot = std::nullopt,
      .line = 0,
      .u_step = 0.0,
      .core_dot = std::vector<std::vector<double>>(dirs, std::vector<double>(o.e_core.size())),
      .virt_dot = std::vector<std::vector<double>>(dirs, std::vector<double>(o.e_virt.size()))};
  std::vector<double> core_bar(o.e_core.size(), 0.0);
  std::vector<double> virt_bar(o.e_virt.size(), 0.0);
  const EnergyBars no_bars{};
  ReverseContext c{
      .res = res,
      .blocks = blocks,
      .in = in,
      .bars = no_bars,
      .batch = batch,
      .ops = o,
      .e = {.core = o.e_core, .virt = o.e_virt, .core_bar = core_bar, .virt_bar = virt_bar}};
  std::vector<BlockBars> m;
  m.push_back(FBF_TRY(block_bars(res, blocks.ncas, false, false)));
  const double one = 1.0;
  const PcStates q{.ns = 1, .ops = {&o, 1}, .hbar = {&one, 1}, .m = m, .tangent = &tg, .hvp = &hvp};
  PcClassEnergiesHvp out;
  for (const auto &[k, walk] : kPcWalks) {
    const PcClassSums x = FBF_TRY(walk(c, q));
    out.energies.energies[k] = x.energy.front();
    out.energies.dropped[k] = x.dropped.front();
    pc_report_modes(x.modes, 0, k, out.energies);
    out.energies.cut[k] = x.cut.empty() ? 0.0 : x.cut.front();
    out.energies.kept[k] = x.kept.empty() ? 0.0 : x.kept.front();
    // The class's T, t1 and slab buffers are freed: drop their regions.
    tg.t.regions.resize(tg.t.inputs);
    tg.t.held.clear();
  }
  out.energies.energies[2] = FBF_TRY(energy_sijrs(res, blocks, in, batch)).second;
  out.energies.total = std::ranges::fold_left(out.energies.energies, 0.0, std::plus{});
  for (const auto &[k, walk] : kPcWalks) {
    if (hvp.classes[k].walked && out.energies.kept[k] < kPcHessianKept) {
      return err_numerical("nevpt2::pc_class_energies_hvp: {} keeps a metric mode at {:.3e} of "
                           "its largest, below {:.0e}: its solve is not differentiable twice "
                           "there (#917). Raise EnergyInputs::pc_floor to {:.0e} or more",
                           kScClassNames[k], out.energies.kept[k], kPcHessianKept, kPcHessianKept);
    }
  }

  hvp.fixed = true;
  for (size_t u = 0; u < dirs; ++u) {
    // The multipliers' tangent, through one fixed walk, and R's onto the
    // orbital energies.
    hvp.dot = u;
    FBF_TRY(pc_fixed_adjoint(res, blocks, in, batch, bars[u], hvp));
    FBF_TRY(add_orbital_energy_bars(res, hvp.core_dot[u], hvp.virt_dot[u], bars[u]));
    hvp.dot.reset();
    // U's tangent: the bars are quadratic in U, so one symmetric difference.
    if (span) {
      const double u_step = FBF_TRY(pc_u_step(res, hvp, u));
      if (u_step != 0.0) {
        hvp.line = u;
        FBF_TRY(line_derivative(res, blocks, {&bars[u], 1}, u_step, 1,
                                [&](const double t, std::span<const EnergyBars> scratch) -> Status {
                                  hvp.u_step = t;
                                  return pc_fixed_adjoint(res, blocks, in, batch, scratch.front(),
                                                          hvp);
                                }));
        hvp.u_step = 0.0;
      }
    }
    // The inputs' tangent at fixed multipliers. The orbital energies are
    // read by the multipliers alone, so they need no difference.
    EnergyDirection du = directions[u];
    du.in.e_core = nullptr;
    du.in.e_virt = nullptr;
    const double step = FBF_TRY(line_step(res, blocks, in, du));
    if (step != 0.0) {
      FBF_TRY(line_derivative(res, blocks, {&bars[u], 1}, step, kLineWeights.size(),
                              [&](const double t, std::span<const EnergyBars> scratch) -> Status {
                                const ShiftedInputs x =
                                    FBF_TRY(shifted_inputs(res, blocks, in, du, t));
                                return pc_fixed_adjoint(res, x.blocks(), x.in, batch,
                                                        scratch.front(), hvp);
                              }));
    }
    FBF_TRY(sijrs_hvp(res, blocks, in, batch, directions[u], bars[u]));
  }
  return out;
}

} // namespace free_breakfast::nevpt2
