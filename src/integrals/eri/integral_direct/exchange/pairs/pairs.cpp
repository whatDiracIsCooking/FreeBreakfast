/// @file
/// @brief ordered_pair_classes: mmd's records in both orientations, one class
/// per ordered (la, lb), its groups by first shell and its tables' sizes from
/// the basis on the host, and on the device each group sorted by its dressed
/// bound, the records gathered from mmd's in that order, the rolled kernel's
/// E tables when the basis reaches past kb, and the K build's Dcd rows and
/// block tables.

module;

// The order, gather and Dcd-row launchers, defined in pairs_launch.cu, and the
// block table's, in the exchange helpers: declared in the GMF so they keep
// external linkage (see the headers).
#include "exchange_helpers_bridge.h"
#include "pairs_bridge.h"

module free_breakfast.integrals.eri.integral_direct.exchange.pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

#include "diagnostics.h"

namespace free_breakfast {

int32_t ordered_cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

// The blocks of a bra's table against a ket's: one per (a, b) of the bra's
// first shells @p a and the ket's @p b, both in basis order, with a <= b. The
// device cuts on AO offsets, which rise with the shells' indices.
int64_t ordered_block_count(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b) {
  int64_t n = 0;
  for (const uint32_t sa : a) {
    n += std::ranges::end(b) - std::ranges::lower_bound(b, sa);
  }
  return n;
}

// An ordered class of n records in n_groups groups, of a basis of n_ao
// Cartesian AOs and n_shells shells per momentum, its tables n_blk blocks, its
// E tables sized when rolled and the class within K's limit (else empty), and
// the set's own scratch when it is the set's first, on a new stream of res's
// Device.
OrderedPairClass make_ordered_class(DeviceResources &res, const int32_t la, const int32_t lb,
                                    const int32_t n, const int32_t n_groups, const int32_t n_ao,
                                    const std::vector<int32_t> &n_shells,
                                    std::vector<int32_t> n_blk, const bool rolled,
                                    const bool first) {
  const auto ds = std::make_shared<DeviceStream>(res.device());
  const auto len = static_cast<size_t>(n);
  size_t n_blocks = 0;
  for (const int32_t count : n_blk) {
    n_blocks += static_cast<size_t>(count);
  }
  const size_t n_dcd = static_cast<size_t>(n_shells[static_cast<size_t>(la)]) *
                       static_cast<size_t>(n_shells[static_cast<size_t>(lb)]) *
                       static_cast<size_t>(ordered_cartesian_count(la)) *
                       static_cast<size_t>(ordered_cartesian_count(lb));
  const auto fields = static_cast<size_t>(device::kExchangeBlockFields);
  const auto groups = static_cast<size_t>(n_groups);
  // A class above K's limit has no kernel, and the K build refuses it: no
  // table.
  const bool tables = rolled && std::max(la, lb) <= device::kExchangeMaxMomentum;
  const size_t n_e = tables ? len * static_cast<size_t>(device::exchange_e_table_width(la, lb)) : 0;
  const size_t n_rank = la == lb ? static_cast<size_t>(n_shells[static_cast<size_t>(la)]) : 0;
  const size_t set_ao = first ? static_cast<size_t>(n_ao) : 0;
  return {
      .la = la,
      .lb = lb,
      .n_rec = n,
      .n_groups = n_groups,
      .n_ao = n_ao,
      .n_shells = n_shells,
      .n_blk = std::move(n_blk),
      .work_stream = ds,
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
      .record_bound = StreamBuffers::device<double>(len, ds),
      .off_a = StreamBuffers::device<int32_t>(len, ds),
      .off_b = StreamBuffers::device<int32_t>(len, ds),
      .rank_b = StreamBuffers::device<int32_t>(len, ds),
      .group_start = StreamBuffers::device<int32_t>(groups + 1, ds),
      .dc = StreamBuffers::device<int32_t>(n_shells.size() * len, ds),
      .blocks = StreamBuffers::device<int32_t>(std::max(fields * n_blocks, size_t{1}), ds),
      .e_bra = StreamBuffers::device<double>(n_e, ds),
      .e_ket = StreamBuffers::device<double>(n_e, ds),
      .dcd = StreamBuffers::device<double>(n_dcd, ds),
      .geom = StreamBuffers::device<double>(
          static_cast<size_t>(device::kExchangeGroupFields) * groups, ds),
      .factor = StreamBuffers::device<double>(std::max(n_blocks, size_t{1}), ds),
      .rank_offset = StreamBuffers::device<int32_t>(n_rank, ds),
      .centre = StreamBuffers::device<double>(size_t{3} * set_ao, ds),
      .shell_start = StreamBuffers::device<int32_t>(set_ao, ds),
      .decay = StreamBuffers::device<double>(
          first ? static_cast<size_t>(device::kExchangeDecayFields) : 0, ds),
      .g_max = StreamBuffers::device<double>(first ? 1 : 0, ds),
  };
}

// The class's order and its records from the mmd class from, on its stream:
// the order and group_start (ordered_pair_order), then every field in that
// order (ordered_pair_gather). The scratch is drawn on the class's stream and
// returned in its order.
Status gather_ordered(OrderedPairClass &cls, const PrimitivePairClass &from) {
  const std::shared_ptr<DeviceStream> &ds = cls.work_stream;
  const wwr::wwrStream_t stream = ds->stream();
  const PrimitivePairClassView v = from.view();
  // One slot per orientation the class takes of each mmd record
  // (pairs_bridge.h).
  const size_t n_slots = static_cast<size_t>(from.n_rec) * (from.la == from.lb ? 2 : 1);
  StreamBuffers::device<double> key(n_slots, ds);
  StreamBuffers::device<double> key_alt(n_slots, ds);
  StreamBuffers::device<uint32_t> perm(n_slots, ds);
  StreamBuffers::device<uint32_t> perm_alt(n_slots, ds);
  FBF_TRY(gpu_status(device::ordered_pair_order(stream, v, cls.la, cls.n_groups, key.data(),
                                                key_alt.data(), perm.data(), perm_alt.data(),
                                                cls.group_start.data())));
  double *const doubles[] = {
      cls.zeta.data(), cls.p_x.data(),          cls.p_y.data(),  cls.p_z.data(),  cls.pa_x.data(),
      cls.pa_y.data(), cls.pa_z.data(),         cls.pb_x.data(), cls.pb_y.data(), cls.pb_z.data(),
      cls.s.data(),    cls.record_bound.data(), cls.ea.data(),   cls.eb.data()};
  int32_t *const ints[] = {cls.off_a.data(), cls.off_b.data(), cls.rank_b.data()};
  return gpu_status(
      device::ordered_pair_gather(stream, v, cls.la, cls.n_rec, perm.data(), doubles, ints));
}

// The class's two E tables, kbr's Eac and Ebd, from its gathered records, on
// its stream.
Status write_e_tables(OrderedPairClass &cls) {
  const wwr::wwrStream_t stream = cls.work_stream->stream();
  const std::array pa{cls.pa_x.data(), cls.pa_y.data(), cls.pa_z.data()};
  const std::array pb{cls.pb_x.data(), cls.pb_y.data(), cls.pb_z.data()};
  for (const bool ket : {false, true}) {
    FBF_TRY(gpu_status(device::exchange_e_table(stream, cls.la, cls.lb, cls.n_rec, cls.zeta.data(),
                                                pa.data(), pb.data(), ket,
                                                (ket ? cls.e_ket : cls.e_bra).data())));
  }
  return {};
}

// The bra's Dcd rows, one per momentum the basis reaches, and its block table
// against every class of @p set as a ket, in set order, on its stream. Every
// ket's group_start and off_a must be gathered before it.
Status write_bra(OrderedPairClass &bra, const std::span<const OrderedPairClass> set) {
  const wwr::wwrStream_t stream = bra.work_stream->stream();
  const auto n_rec = static_cast<size_t>(bra.n_rec);
  for (size_t ld = 0; ld < bra.n_shells.size(); ++ld) {
    if (bra.n_shells[ld] > 0) {
      FBF_TRY(gpu_status(device::ordered_pair_bra_rows(
          stream, bra.n_rec, bra.rank_b.data(), bra.n_shells[ld], bra.dc.data() + ld * n_rec)));
    }
  }
  int32_t *table = bra.blocks.data();
  for (size_t k = 0; k < set.size(); ++k) {
    const int32_t count = bra.n_blk[k];
    if (count == 0) {
      continue;
    }
    const OrderedPairClass &ket = set[k];
    FBF_TRY(gpu_status(device::exchange_blocks(
        stream, bra.n_groups, bra.group_start.data(), bra.off_a.data(), ket.n_groups,
        ket.group_start.data(), ket.off_a.data(), count, table)));
    table += static_cast<size_t>(count) * device::kExchangeBlockFields;
  }
  return {};
}

// Every class's stream joined into @p hub, then forked back from it: each is
// then after all of them.
Status rejoin(std::vector<OrderedPairClass> &classes, const wwr::wwrStream_t hub) {
  for (OrderedPairClass &cls : classes) {
    ForkStream &fork = cls.work_stream->fork_stream();
    FBF_TRY(gpu_status(fork.record_join()));
    FBF_TRY(gpu_status(fork.join_into(hub)));
  }
  for (OrderedPairClass &cls : classes) {
    FBF_TRY(gpu_status(cls.work_stream->fork_stream().fork_from(hub)));
  }
  return {};
}

Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes) {
  // Per momentum, its shells in basis order (a shell's index there is its
  // rank), their primitives in all, and the sum of their squares: a class's
  // record count, mmd's and ordered alike, is made of these, and its block
  // tables' sizes of the shells.
  const std::vector<Shell> &shells = basis.shells();
  std::vector<std::vector<uint32_t>> of_l;
  std::vector<int64_t> prims;
  std::vector<int64_t> prims_sq;
  for (uint32_t i = 0; i < basis.nshells(); ++i) {
    const auto l = static_cast<size_t>(shells[i].angular_momentum());
    if (l >= of_l.size()) {
      of_l.resize(l + 1);
      prims.resize(l + 1);
      prims_sq.resize(l + 1);
    }
    const auto n = static_cast<int64_t>(shells[i].primitives().size());
    of_l[l].push_back(i);
    prims[l] += n;
    prims_sq[l] += n * n;
  }

  // The classes must be the basis's mmd classes, each once: mmd stores every
  // shell pair i <= j once, so (la, lb) has P(la) P(lb) records, and (l, l)
  // (P(l)^2 + the diagonal pairs) / 2. Nothing reads their records on the
  // host, so a count is what tells another basis's classes apart.
  size_t n_reached = 0;
  for (size_t la = 0; la < of_l.size(); ++la) {
    for (size_t lb = la; lb < of_l.size(); ++lb) {
      n_reached += !of_l[la].empty() && !of_l[lb].empty() ? 1 : 0;
    }
  }
  std::set<std::pair<int32_t, int32_t>> seen;
  for (const PrimitivePairClass &cls : classes) {
    const auto la = static_cast<size_t>(cls.la);
    const auto lb = static_cast<size_t>(cls.lb);
    const bool reached = cls.la >= 0 && cls.la <= cls.lb && lb < of_l.size() && !of_l[la].empty() &&
                         !of_l[lb].empty();
    const int64_t want = !reached  ? -1
                         : la < lb ? prims[la] * prims[lb]
                                   : (prims[la] * prims[la] + prims_sq[la]) / 2;
    if (want != cls.n_rec || !seen.insert({cls.la, cls.lb}).second) {
      return err_config("ordered_pair_classes: class ({}, {}) of {} records is not one of the "
                        "basis's mmd classes",
                        cls.la, cls.lb, cls.n_rec);
    }
  }
  if (seen.size() != n_reached) {
    return err_config("ordered_pair_classes: {} mmd classes, but the basis reaches {}", seen.size(),
                      n_reached);
  }

  // Every ordered class, ascending (la, lb), with the mmd class it is drawn
  // from: (la, lb) as stored, (lb, la) swapped.
  std::vector<std::pair<std::pair<int32_t, int32_t>, const PrimitivePairClass *>> plan;
  for (const PrimitivePairClass &cls : classes) {
    plan.push_back({{cls.la, cls.lb}, &cls});
    if (cls.la != cls.lb) {
      plan.push_back({{cls.lb, cls.la}, &cls});
    }
  }
  std::ranges::sort(plan, {}, [](const auto &p) { return p.first; });

  // The shell counts by momentum, through K's limit at least, and every bra's
  // table sizes against every ket, in plan order: kb and the table index a
  // block in 32 bits.
  std::vector<int32_t> n_shells(
      std::max(of_l.size(), static_cast<size_t>(device::kExchangeMaxMomentum + 1)));
  for (size_t l = 0; l < of_l.size(); ++l) {
    n_shells[l] = static_cast<int32_t>(of_l[l].size());
  }
  std::vector<std::vector<int32_t>> n_blk(plan.size());
  for (size_t a = 0; a < plan.size(); ++a) {
    const auto [la, lc] = plan[a].first;
    for (size_t b = 0; b < plan.size(); ++b) {
      const auto [lb, ld] = plan[b].first;
      const int64_t count =
          ordered_block_count(of_l[static_cast<size_t>(la)], of_l[static_cast<size_t>(lb)]);
      if (count * device::kExchangeBlockFields > std::numeric_limits<int32_t>::max()) {
        return err_config("ordered_pair_classes: classes ({}, {}) and ({}, {}) make {} blocks, "
                          "too many for the kernels' 32-bit index",
                          la, lc, lb, ld, count);
      }
      n_blk[a].push_back(static_cast<int32_t>(count));
    }
  }

  // A basis with a shell past kb's momenta launches kbr, and every class of
  // its set is a side of some kbr launch, so every class gets E tables.
  const bool rolled = of_l.size() > static_cast<size_t>(device::kExchangeKbMaxMomentum + 1);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t hub = res.stream();
  std::vector<OrderedPairClass> out;
  out.reserve(plan.size());
  for (size_t c = 0; c < plan.size(); ++c) {
    const auto [la, lb] = plan[c].first;
    const PrimitivePairClass *from = plan[c].second;
    const int64_t n = prims[static_cast<size_t>(la)] * prims[static_cast<size_t>(lb)];
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("ordered_pair_classes: class ({}, {}) has {} records, past {}", la, lb,
                            n, std::numeric_limits<int32_t>::max())});
    }
    OrderedPairClass &cls = out.emplace_back(make_ordered_class(
        res, la, lb, static_cast<int32_t>(n),
        static_cast<int32_t>(of_l[static_cast<size_t>(la)].size()),
        static_cast<int32_t>(basis.nao_cart()), n_shells, std::move(n_blk[c]), rolled, c == 0));
    // Fork from mmd's stream, so the class reads mmd's final records, without
    // a host synchronization.
    FBF_TRY(gpu_status(cls.work_stream->fork_stream().fork_from(from->device_stream()->stream())));
    FBF_TRY(gather_ordered(cls, *from));
    if (cls.e_bra.num_elements() > 0) {
      FBF_TRY(write_e_tables(cls));
    }
  }
  // A bra's tables read every class's groups, so every gather is in before
  // any table is made, and every table before any class's frees can run.
  FBF_TRY(rejoin(out, hub));
  for (OrderedPairClass &cls : out) {
    FBF_TRY(write_bra(cls, out));
  }
  FBF_TRY(rejoin(out, hub));
  // mmd's streams wait on hub, which has joined every class: freeing an mmd
  // class queues its frees on its stream, so they wait for every read of it.
  for (const PrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(cls.device_stream()->fork_stream().fork_from(hub)));
  }
  return out;
}

} // namespace free_breakfast
