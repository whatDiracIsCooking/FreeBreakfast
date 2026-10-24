/// @file
/// @brief primitive_pair_classes and mixed_primitive_pair_classes: the
/// shell-pair sweeps (the la <= lb swap in the first), the ranks, the tables the
/// device forms the records from, the order there and the record bounds.

module;

// The record, record-order and record-bound launchers, defined in mmd_launch.cu: declared in the
// GMF so they keep external linkage (see the header).
#include "mmd_bridge.h"

module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;
import free_breakfast.basis;

#include "diagnostics.h"

namespace free_breakfast {

// A shell-pair class, (la, lb): la <= lb for primitive_pair_classes, any
// order for mixed_primitive_pair_classes.
using ClassKey = std::pair<int32_t, int32_t>;

// Shell pair (i, j) as (bra, ket) shell indices, swapped when i has the higher
// angular momentum so the pair lands in its la <= lb class.
std::pair<uint32_t, uint32_t> bra_ket_order(const std::vector<Shell> &shells, const uint32_t i,
                                            const uint32_t j) {
  if (shells[i].angular_momentum() > shells[j].angular_momentum()) {
    return {j, i};
  }
  return {i, j};
}

// One class's shell pairs, in sweep order, and each one's first record: the
// prefix of their primitive-pair counts, in int64_t so a class past INT32_MAX
// is caught before it is sized.
struct ClassPairs {
  std::vector<uint32_t> bra;
  std::vector<uint32_t> ket;
  std::vector<int64_t> rec_begin{0};
};

// The shells and primitives of the bases swept, staged, one basis after the
// other: what every class's records are formed from. A MirroredBuffer's destructor waits for its stream, so the
// pinned side outlives the upload.
struct ShellTable {
  MirroredBuffer<int32_t> prim_begin;
  MirroredBuffer<double> exponent;
  MirroredBuffer<double> coefficient;
  MirroredBuffer<double> c_x;
  MirroredBuffer<double> c_y;
  MirroredBuffer<double> c_z;
  MirroredBuffer<int32_t> ao_offset;
  MirroredBuffer<int32_t> rank;
};

// One class's ClassPairs, staged.
struct PairTable {
  MirroredBuffer<uint32_t> bra;
  MirroredBuffer<uint32_t> ket;
  MirroredBuffer<int32_t> rec_begin;
};

// values in a new MirroredBuffer on ds, on its pinned side only.
template <typename T>
MirroredBuffer<T> staged(const std::vector<T> &values, const std::shared_ptr<DeviceStream> &ds) {
  MirroredBuffer<T> out(values.size(), ds);
  std::ranges::copy(values, out.pinned().data());
  return out;
}

// Each shell's rank: its index among basis's shells of its momentum, in basis
// order.
std::vector<int32_t> shell_ranks(const BasisSet &basis) {
  std::vector<int32_t> rank;
  std::map<int32_t, int32_t> next_rank;
  for (const Shell &shell : basis.shells()) {
    rank.push_back(next_rank[shell.angular_momentum()]++);
  }
  return rank;
}

// The shells and primitives of bases staged on ds, one basis after the other,
// so a shell's index in the table is its index in its basis plus the shell
// counts of the bases before it. Each shell keeps its own basis's AO offset and
// rank. BasisSet guarantees nao_cart() fits an int32_t, so the offset casts
// are exact; the primitive count, across every basis, is checked here.
ShellTable stage_shells(const std::span<const BasisSet *const> bases,
                        const std::shared_ptr<DeviceStream> &ds, const std::string_view what) {
  std::vector<int32_t> prim_begin{0};
  std::vector<double> exponent;
  std::vector<double> coefficient;
  std::vector<double> c_x;
  std::vector<double> c_y;
  std::vector<double> c_z;
  std::vector<int32_t> ao_offset;
  std::vector<int32_t> rank;
  int64_t n_prim = 0;
  for (const BasisSet *basis : bases) {
    const std::vector<Shell> &shells = basis->shells();
    for (uint32_t i = 0; i < basis->nshells(); ++i) {
      n_prim += shells[i].nprimitives();
      if (n_prim > std::numeric_limits<int32_t>::max()) {
        check_or_abort(Status{err_config("{}: the bases have more than {} primitives", what,
                                         std::numeric_limits<int32_t>::max())});
      }
      prim_begin.push_back(static_cast<int32_t>(n_prim));
      for (const Primitive &p : shells[i].primitives()) {
        exponent.push_back(p.exponent);
        coefficient.push_back(p.coefficient);
      }
      c_x.push_back(shells[i].center().x);
      c_y.push_back(shells[i].center().y);
      c_z.push_back(shells[i].center().z);
      ao_offset.push_back(static_cast<int32_t>(basis->ao_offset_cart(i)));
    }
    std::ranges::copy(shell_ranks(*basis), std::back_inserter(rank));
  }
  return {
      .prim_begin = staged(prim_begin, ds),
      .exponent = staged(exponent, ds),
      .coefficient = staged(coefficient, ds),
      .c_x = staged(c_x, ds),
      .c_y = staged(c_y, ds),
      .c_z = staged(c_z, ds),
      .ao_offset = staged(ao_offset, ds),
      .rank = staged(rank, ds),
  };
}

// One class's shell pairs staged on ds, their record prefix narrowed to
// int32_t: its last entry, the class's record count, is checked already.
PairTable stage_pairs(const ClassPairs &pairs, const std::shared_ptr<DeviceStream> &ds) {
  std::vector<int32_t> rec_begin(pairs.rec_begin.size());
  std::ranges::transform(pairs.rec_begin, rec_begin.begin(),
                         [](const int64_t r) { return static_cast<int32_t>(r); });
  return {.bra = staged(pairs.bra, ds),
          .ket = staged(pairs.ket, ds),
          .rec_begin = staged(rec_begin, ds)};
}

// Queue every staged field's pinned-to-device copy on its stream.
Status upload(ShellTable &table) {
  for (MirroredBuffer<double> *field :
       {&table.exponent, &table.coefficient, &table.c_x, &table.c_y, &table.c_z}) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  for (MirroredBuffer<int32_t> *field : {&table.prim_begin, &table.ao_offset, &table.rank}) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  return {};
}

// Queue each staged shell-pair field's pinned-to-device copy on its stream.
Status upload(PairTable &pairs) {
  FBF_TRY(gpu_status(pairs.bra.to_device()));
  FBF_TRY(gpu_status(pairs.ket.to_device()));
  FBF_TRY(gpu_status(pairs.rec_begin.to_device()));
  return {};
}

// A class of n records on a new stream of res's Device, every field zeroed.
PrimitivePairClass make_class(DeviceResources &res, const ClassKey key, const int32_t n) {
  const auto ds = std::make_shared<DeviceStream>(res.device());
  const auto len = static_cast<size_t>(n);
  return {
      .la = key.first,
      .lb = key.second,
      .n_rec = n,
      .handle = ds,
      .zeta = StreamBuffers::device<double>(len, ds),
      .p_x = StreamBuffers::device<double>(len, ds),
      .p_y = StreamBuffers::device<double>(len, ds),
      .p_z = StreamBuffers::device<double>(len, ds),
      .pa_x = StreamBuffers::device<double>(len, ds),
      .pa_y = StreamBuffers::device<double>(len, ds),
      .pa_z = StreamBuffers::device<double>(len, ds),
      .pb_x = StreamBuffers::device<double>(len, ds),
      .pb_y = StreamBuffers::device<double>(len, ds),
      .pb_z = StreamBuffers::device<double>(len, ds),
      .s = StreamBuffers::device<double>(len, ds),
      .ea = StreamBuffers::device<double>(len, ds),
      .eb = StreamBuffers::device<double>(len, ds),
      .bound = StreamBuffers::device<double>(len, ds),
      .bound_ab = StreamBuffers::device<double>(len, ds),
      .bound_ba = StreamBuffers::device<double>(len, ds),
      .off_a = StreamBuffers::device<int32_t>(len, ds),
      .off_b = StreamBuffers::device<int32_t>(len, ds),
      .rank_a = StreamBuffers::device<int32_t>(len, ds),
      .rank_b = StreamBuffers::device<int32_t>(len, ds),
  };
}

// cls's records formed from src on its stream, in descending zeta order: the
// order first, sorted from zeta alone, then every field written once, already
// in it, so nothing is permuted after. Within a class the record bound
// (form_bounds) rises with zeta, so this is descending Ip: a J build's bras
// with the longest ket loops launch first. Every kernel accumulates, so none
// depends on the order. The sort's scratch is drawn on cls's stream and
// returned in its order.
Status form_records(PrimitivePairClass &cls, const device::PrimitivePairSource &src) {
  const std::shared_ptr<DeviceStream> &ds = cls.handle;
  const wwr::wwrStream_t stream = ds->stream();
  const auto n = static_cast<size_t>(cls.n_rec);
  StreamBuffers::device<double> key(n, ds);
  StreamBuffers::device<double> key_alt(n, ds);
  StreamBuffers::device<uint32_t> perm(n, ds);
  StreamBuffers::device<uint32_t> perm_alt(n, ds);
  FBF_TRY(gpu_status(device::mmd_order(stream, cls.n_rec, src, key.data(), key_alt.data(),
                                       perm.data(), perm_alt.data())));
  const device::PrimitivePairRecords out{
      .zeta = cls.zeta.data(),
      .p_x = cls.p_x.data(),
      .p_y = cls.p_y.data(),
      .p_z = cls.p_z.data(),
      .pa_x = cls.pa_x.data(),
      .pa_y = cls.pa_y.data(),
      .pa_z = cls.pa_z.data(),
      .pb_x = cls.pb_x.data(),
      .pb_y = cls.pb_y.data(),
      .pb_z = cls.pb_z.data(),
      .s = cls.s.data(),
      .ea = cls.ea.data(),
      .eb = cls.eb.data(),
      .off_a = cls.off_a.data(),
      .off_b = cls.off_b.data(),
      .rank_a = cls.rank_a.data(),
      .rank_b = cls.rank_b.data(),
  };
  return gpu_status(device::mmd_records(stream, cls.n_rec, src, perm.data(), out));
}

// Each record's three bounds, on cls's stream after form_records, so they are
// in record order, on the device only: the Schwarz bound
// I = pair_bound(zeta, la + lb) from its zeta, which rises with zeta and so is
// non-increasing in record order, and the dressed bound (ordered_pair_bound) of
// the record as stored and swapped, the swap a view with la and lb, and P - A
// and P - B, exchanged. A class above kMaxAngularMomentum has no kernel and
// keeps its zeroed bounds: every build refuses such a class, so nothing reads
// them. pair_bound depends on la + lb alone, so a mixed class with la > lb
// takes the kernel of (lb, la), the one emitted.
Status form_bounds(PrimitivePairClass &cls) {
  const auto [lo, hi] = std::minmax(cls.la, cls.lb);
  if (hi > kMaxAngularMomentum) {
    return {};
  }
  const wwr::wwrStream_t stream = cls.handle->stream();
  FBF_TRY(
      gpu_status(device::mmd_bound(stream, lo, hi, cls.n_rec, cls.zeta.data(), cls.bound.data())));
  const PrimitivePairClassView v = cls.view();
  FBF_TRY(gpu_status(device::ordered_pair_bound(stream, v, cls.bound_ab.data())));
  PrimitivePairClassView swapped = v;
  std::swap(swapped.la, swapped.lb);
  std::swap(swapped.pa_x, swapped.pb_x);
  std::swap(swapped.pa_y, swapped.pb_y);
  std::swap(swapped.pa_z, swapped.pb_z);
  FBF_TRY(gpu_status(device::ordered_pair_bound(stream, swapped, cls.bound_ba.data())));
  return {};
}

// Every class formed on its own stream: its shell pairs uploaded there, the
// stream forked from table_ds's (where the shell table was uploaded), then its
// records and bounds.
Status form_classes(std::vector<PrimitivePairClass> &classes, std::vector<PairTable> &pairs,
                    ShellTable &table, DeviceStream &table_ds) {
  FBF_TRY(upload(table));
  for (size_t c = 0; c < classes.size(); ++c) {
    PrimitivePairClass &cls = classes[c];
    PairTable &pt = pairs[c];
    FBF_TRY(upload(pt));
    FBF_TRY(gpu_status(cls.handle->fork_stream().fork_from(table_ds.stream())));
    const device::PrimitivePairSource src{
        .n_pairs = static_cast<int32_t>(pt.bra.size()),
        .bra = pt.bra.device().data(),
        .ket = pt.ket.device().data(),
        .rec_begin = pt.rec_begin.device().data(),
        .prim_begin = table.prim_begin.device().data(),
        .exponent = table.exponent.device().data(),
        .coefficient = table.coefficient.device().data(),
        .c_x = table.c_x.device().data(),
        .c_y = table.c_y.device().data(),
        .c_z = table.c_z.device().data(),
        .ao_offset = table.ao_offset.device().data(),
        .rank = table.rank.device().data(),
    };
    FBF_TRY(form_records(cls, src));
    FBF_TRY(form_bounds(cls));
  }
  return {};
}

// Append shell pair (bra, ket), whose shells are bra_shell and ket_shell, to
// its class in sweep, with its first record.
void add_pair(std::map<ClassKey, ClassPairs> &sweep, const Shell &bra_shell, const Shell &ket_shell,
              const uint32_t bra, const uint32_t ket) {
  ClassPairs &cp = sweep[{bra_shell.angular_momentum(), ket_shell.angular_momentum()}];
  cp.bra.push_back(bra);
  cp.ket.push_back(ket);
  cp.rec_begin.push_back(cp.rec_begin.back() + (static_cast<int64_t>(bra_shell.nprimitives()) *
                                                static_cast<int64_t>(ket_shell.nprimitives())));
}

// The classes of sweep, whose shell indices name bases' shells one basis after
// the other (stage_shells), formed on the device: each class sized once, at its
// final record count, then its records and bounds formed on its own stream.
// what names the caller in the abort messages.
Result<std::vector<PrimitivePairClass>> form_sweep(DeviceResources &res,
                                                   const std::map<ClassKey, ClassPairs> &sweep,
                                                   const std::span<const BasisSet *const> bases,
                                                   const std::string_view what) {
  // The shell table, on a stream of its own that every class's stream forks
  // from. Its device side is freed on that stream when the call returns, so
  // the classes' streams, which read it, are waited for first (below).
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto table_ds = std::make_shared<DeviceStream>(res.device());
  ShellTable table = stage_shells(bases, table_ds, what);

  std::vector<PrimitivePairClass> out;
  std::vector<PairTable> pairs;
  out.reserve(sweep.size());
  pairs.reserve(sweep.size());
  for (const auto &[key, cp] : sweep) {
    const int64_t n = cp.rec_begin.back();
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("{}: class ({}, {}) has {} records, past {}", what, key.first,
                            key.second, n, std::numeric_limits<int32_t>::max())});
    }
    out.push_back(make_class(res, key, static_cast<int32_t>(n)));
    pairs.push_back(stage_pairs(cp, out.back().handle));
  }

  // Wait for every class's stream however the forming went: the tables are
  // this call's, and are freed when it returns.
  const Status formed = form_classes(out, pairs, table, *table_ds);
  Status waited;
  for (PrimitivePairClass &cls : out) {
    if (const Status s = gpu_status(cls.handle->fork_stream().sync()); !s && waited) {
      waited = s;
    }
  }
  FBF_TRY(formed);
  FBF_TRY(waited);
  return out;
}

Result<std::vector<PrimitivePairClass>> primitive_pair_classes(DeviceResources &res,
                                                               const BasisSet &basis) {
  const std::vector<Shell> &shells = basis.shells();
  const uint32_t nsh = basis.nshells();
  if (nsh == 0) {
    return std::vector<PrimitivePairClass>{};
  }

  // The sweep: every shell pair i <= j, in la <= lb order, listed in its
  // class with its first record. Keyed by (la, lb), so the result comes out in
  // ascending class order. No record is formed here: a pair is O(1) however
  // many primitives it has.
  std::map<ClassKey, ClassPairs> sweep;
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      add_pair(sweep, shells[bra], shells[ket], bra, ket);
    }
  }
  const std::array<const BasisSet *, 1> bases{&basis};
  return form_sweep(res, sweep, bases, "primitive_pair_classes");
}

Result<std::vector<PrimitivePairClass>>
mixed_primitive_pair_classes(DeviceResources &res, const BasisSet &rows, const BasisSet &cols) {
  const std::vector<Shell> &row_shells = rows.shells();
  const std::vector<Shell> &col_shells = cols.shells();
  const uint32_t n_rows = rows.nshells();
  if (n_rows == 0 || cols.nshells() == 0) {
    return std::vector<PrimitivePairClass>{};
  }

  // The sweep: every ordered shell pair (i of rows, j of cols), never swapped,
  // listed in its class (la of i, lb of j). In the shell table cols' shells
  // follow rows', so j is n_rows + j there.
  std::map<ClassKey, ClassPairs> sweep;
  for (uint32_t i = 0; i < n_rows; ++i) {
    for (uint32_t j = 0; j < cols.nshells(); ++j) {
      add_pair(sweep, row_shells[i], col_shells[j], i, n_rows + j);
    }
  }
  const std::array<const BasisSet *, 2> bases{&rows, &cols};
  return form_sweep(res, sweep, bases, "mixed_primitive_pair_classes");
}

} // namespace free_breakfast
