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

// The shells and primitives of the bases swept, one basis after the other, on
// the host: what every class's records are formed from.
struct ShellTable {
  std::vector<int32_t> prim_begin{0};
  std::vector<double> exponent;
  std::vector<double> coefficient;
  std::vector<double> c_x;
  std::vector<double> c_y;
  std::vector<double> c_z;
  std::vector<int32_t> ao_offset;
  std::vector<int32_t> rank;
};

// Every class's shell pairs on the host, one class after the other, in class
// order: class c's start at pair_begin[c] in bra and ket, and its n_pairs + 1
// record prefix at pair_begin[c] + c in rec_begin.
struct PairTable {
  std::vector<uint32_t> bra;
  std::vector<uint32_t> ket;
  std::vector<int32_t> rec_begin;
  std::vector<size_t> pair_begin{0};
};

// Both tables on the device, a buffer per field, for the call only.
struct DeviceTables {
  Buffers::device<int32_t> prim_begin;
  Buffers::device<double> exponent;
  Buffers::device<double> coefficient;
  Buffers::device<double> c_x;
  Buffers::device<double> c_y;
  Buffers::device<double> c_z;
  Buffers::device<int32_t> ao_offset;
  Buffers::device<int32_t> rank;
  Buffers::device<uint32_t> bra;
  Buffers::device<uint32_t> ket;
  Buffers::device<int32_t> rec_begin;
};

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

// The shells and primitives of bases, one basis after the other, so a shell's
// index in the table is its index in its basis plus the shell counts of the
// bases before it. Each shell keeps its own basis's AO offset and rank.
// BasisSet guarantees nao_cart() fits an int32_t, so the offset casts are
// exact; the primitive count, across every basis, is checked here.
ShellTable shell_table(const std::span<const BasisSet *const> bases, const std::string_view what) {
  ShellTable t;
  int64_t n_prim = 0;
  for (const BasisSet *basis : bases) {
    const std::vector<Shell> &shells = basis->shells();
    for (uint32_t i = 0; i < basis->nshells(); ++i) {
      n_prim += shells[i].nprimitives();
      if (n_prim > std::numeric_limits<int32_t>::max()) {
        check_or_abort(Status{err_config("{}: the bases have more than {} primitives", what,
                                         std::numeric_limits<int32_t>::max())});
      }
      t.prim_begin.push_back(static_cast<int32_t>(n_prim));
      for (const Primitive &p : shells[i].primitives()) {
        t.exponent.push_back(p.exponent);
        t.coefficient.push_back(p.coefficient);
      }
      t.c_x.push_back(shells[i].center().x);
      t.c_y.push_back(shells[i].center().y);
      t.c_z.push_back(shells[i].center().z);
      t.ao_offset.push_back(static_cast<int32_t>(basis->ao_offset_cart(i)));
    }
    std::ranges::copy(shell_ranks(*basis), std::back_inserter(t.rank));
  }
  return t;
}

// sweep's shell pairs, class after class, their record prefixes narrowed to
// int32_t: each one's last entry, its class's record count, is checked
// already.
PairTable pair_table(const std::map<ClassKey, ClassPairs> &sweep) {
  PairTable t;
  for (const ClassPairs &cp : sweep | std::views::values) {
    std::ranges::copy(cp.bra, std::back_inserter(t.bra));
    std::ranges::copy(cp.ket, std::back_inserter(t.ket));
    std::ranges::transform(cp.rec_begin, std::back_inserter(t.rec_begin),
                           [](const int64_t r) { return static_cast<int32_t>(r); });
    t.pair_begin.push_back(t.bra.size());
  }
  return t;
}

// values written to their pinned region, then copied from it to `to` on
// res.stream().
template <typename T>
Status stage(DeviceResources &res, const std::vector<T> &values, T *const pinned,
             Buffers::device<T> &to) {
  std::ranges::copy(values, pinned);
  return gpu_status(wwr::wwrMemcpyAsync(to.data(), pinned, sizeof(T) * values.size(),
                                        wwr::wwrMemcpyHostToDevice, res.stream()));
}

// Both tables staged from one carve of res's pinned workspace and copied to
// the device on res.stream(). The carve is res's until res.stream() is
// synchronized, which the caller does before it returns.
Result<DeviceTables> upload(DeviceResources &res, const ShellTable &s, const PairTable &p,
                            const std::string_view what) {
  const auto [prim_begin, exponent, coefficient, c_x, c_y, c_z, ao_offset, rank, bra, ket,
              rec_begin] =
      FBF_TRY((res.pinned_regions<int32_t, double, double, double, double, double, int32_t, int32_t,
                                  uint32_t, uint32_t, int32_t>(
          what, s.prim_begin.size(), s.exponent.size(), s.coefficient.size(), s.c_x.size(),
          s.c_y.size(), s.c_z.size(), s.ao_offset.size(), s.rank.size(), p.bra.size(), p.ket.size(),
          p.rec_begin.size())));
  const std::shared_ptr<Device> &dev = res.device();
  DeviceTables out{
      .prim_begin = Buffers::device<int32_t>(s.prim_begin.size(), dev),
      .exponent = Buffers::device<double>(s.exponent.size(), dev),
      .coefficient = Buffers::device<double>(s.coefficient.size(), dev),
      .c_x = Buffers::device<double>(s.c_x.size(), dev),
      .c_y = Buffers::device<double>(s.c_y.size(), dev),
      .c_z = Buffers::device<double>(s.c_z.size(), dev),
      .ao_offset = Buffers::device<int32_t>(s.ao_offset.size(), dev),
      .rank = Buffers::device<int32_t>(s.rank.size(), dev),
      .bra = Buffers::device<uint32_t>(p.bra.size(), dev),
      .ket = Buffers::device<uint32_t>(p.ket.size(), dev),
      .rec_begin = Buffers::device<int32_t>(p.rec_begin.size(), dev),
  };
  FBF_TRY(stage(res, s.prim_begin, prim_begin, out.prim_begin));
  FBF_TRY(stage(res, s.exponent, exponent, out.exponent));
  FBF_TRY(stage(res, s.coefficient, coefficient, out.coefficient));
  FBF_TRY(stage(res, s.c_x, c_x, out.c_x));
  FBF_TRY(stage(res, s.c_y, c_y, out.c_y));
  FBF_TRY(stage(res, s.c_z, c_z, out.c_z));
  FBF_TRY(stage(res, s.ao_offset, ao_offset, out.ao_offset));
  FBF_TRY(stage(res, s.rank, rank, out.rank));
  FBF_TRY(stage(res, p.bra, bra, out.bra));
  FBF_TRY(stage(res, p.ket, ket, out.ket));
  FBF_TRY(stage(res, p.rec_begin, rec_begin, out.rec_begin));
  return out;
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
// and P - B, exchanged. Each family has its own limit, because two emitters
// write them: pair_bound (tools/emit/one_electron_kernels.py, J's screen)
// through device::kPairBoundMaxMomentum, and the dressed bounds
// (tools/emit/exchange_kernels.py, K's screen) through
// device::kRecordBoundMaxMomentum, both f today. A class above a family's
// limit keeps that family's zeroed bounds, and every build that reads them
// refuses such a class.
// pair_bound depends on la + lb alone, so a mixed class with la > lb takes the
// kernel of (lb, la), the one emitted.
static_assert(device::kPairBoundMaxMomentum == kMaxAngularMomentum &&
                  device::kRecordBoundMaxMomentum == kMaxAngularMomentum,
              "mmd's bound limits and the tree-wide kMaxAngularMomentum disagree");
Status form_bounds(PrimitivePairClass &cls) {
  const auto [lo, hi] = std::minmax(cls.la, cls.lb);
  const wwr::wwrStream_t stream = cls.handle->stream();
  if (hi <= device::kPairBoundMaxMomentum) {
    FBF_TRY(gpu_status(
        device::mmd_bound(stream, lo, hi, cls.n_rec, cls.zeta.data(), cls.bound.data())));
  }
  if (hi > device::kRecordBoundMaxMomentum) {
    return {};
  }
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

// Every class formed on its own stream, forked from res.stream(), where the
// tables were uploaded: its records, then its bounds.
Status form_classes(DeviceResources &res, std::vector<PrimitivePairClass> &classes,
                    const PairTable &pairs, const DeviceTables &t) {
  for (size_t c = 0; c < classes.size(); ++c) {
    PrimitivePairClass &cls = classes[c];
    FBF_TRY(gpu_status(cls.handle->fork_stream().fork_from(res.stream())));
    const size_t first = pairs.pair_begin[c];
    const device::PrimitivePairSource src{
        .n_pairs = static_cast<int32_t>(pairs.pair_begin[c + 1] - first),
        .bra = t.bra.data() + first,
        .ket = t.ket.data() + first,
        .rec_begin = t.rec_begin.data() + first + c,
        .prim_begin = t.prim_begin.data(),
        .exponent = t.exponent.data(),
        .coefficient = t.coefficient.data(),
        .c_x = t.c_x.data(),
        .c_y = t.c_y.data(),
        .c_z = t.c_z.data(),
        .ao_offset = t.ao_offset.data(),
        .rank = t.rank.data(),
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
  const ScopedDeviceIndex on{res.dev_idx()};
  const ShellTable shells = shell_table(bases, what);

  std::vector<PrimitivePairClass> out;
  out.reserve(sweep.size());
  for (const auto &[key, cp] : sweep) {
    const int64_t n = cp.rec_begin.back();
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("{}: class ({}, {}) has {} records, past {}", what, key.first,
                            key.second, n, std::numeric_limits<int32_t>::max())});
    }
    out.push_back(make_class(res, key, static_cast<int32_t>(n)));
  }
  const PairTable pairs = pair_table(sweep);

  // The tables are this call's: their device side is freed on res.stream()
  // when it returns, and the pinned carve they were staged from is released.
  // Every class's stream reads them, so it and res.stream() are waited for
  // first, however the upload and the forming went.
  const Result<DeviceTables> tables = upload(res, shells, pairs, what);
  Status formed;
  if (tables) {
    formed = form_classes(res, out, pairs, *tables);
  } else {
    formed = std::unexpected(tables.error());
  }
  Status waited = res.sync();
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
