/// @file
/// @brief ordered_pair_classes: mmd's records in both orientations, one class
/// per ordered (la, lb), its groups by first shell from the basis on the host,
/// and on the device each group sorted by its dressed bound and the records
/// gathered from mmd's in that order.

module;

// The order and gather launchers, defined in pairs_launch.cu: declared in the
// GMF so they keep external linkage (see the header).
#include "pairs_bridge.h"

module free_breakfast.integrals.exchange.pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

#include "diagnostics.h"

namespace free_breakfast {

// An ordered class of n records in n_groups groups on a new stream of res's
// Device.
OrderedPairClass make_ordered_class(DeviceResources &res, const int32_t la, const int32_t lb,
                                    const int32_t n, const size_t n_groups) {
  const auto ds = std::make_shared<DeviceStream>(res.device());
  const auto len = static_cast<size_t>(n);
  return {
      .la = la,
      .lb = lb,
      .n_rec = n,
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
      .record_bound = StreamBuffers::device<double>(len, ds),
      .off_a = StreamBuffers::device<int32_t>(len, ds),
      .off_b = StreamBuffers::device<int32_t>(len, ds),
      .rank_b = StreamBuffers::device<int32_t>(len, ds),
      .groups = {},
      .group_start = StreamBuffers::device<int32_t>(n_groups + 1, ds),
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
  FBF_TRY(gpu_status(device::ordered_pair_order(
      stream, v, cls.la, static_cast<int32_t>(cls.groups.size()), key.data(), key_alt.data(),
      perm.data(), perm_alt.data(), cls.group_start.data())));
  double *const doubles[] = {cls.zeta.data(), cls.p_x.data(),  cls.p_y.data(),
                             cls.p_z.data(),  cls.pa_x.data(), cls.pa_y.data(),
                             cls.pa_z.data(), cls.pb_x.data(), cls.pb_y.data(),
                             cls.pb_z.data(), cls.s.data(),    cls.record_bound.data()};
  int32_t *const ints[] = {cls.off_a.data(), cls.off_b.data(), cls.rank_b.data()};
  return gpu_status(
      device::ordered_pair_gather(stream, v, cls.la, cls.n_rec, perm.data(), doubles, ints));
}

Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes) {
  // Per momentum, its shells in basis order (a shell's index there is its
  // rank), their primitives in all, and the sum of their squares: a class's
  // record count, mmd's and ordered alike, is made of these.
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

  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<OrderedPairClass> out;
  out.reserve(plan.size());
  for (const auto &[key, from] : plan) {
    const auto [la, lb] = key;
    const std::vector<uint32_t> &firsts = of_l[static_cast<size_t>(la)];
    const int64_t n = prims[static_cast<size_t>(la)] * prims[static_cast<size_t>(lb)];
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("ordered_pair_classes: class ({}, {}) has {} records, past {}", la, lb,
                            n, std::numeric_limits<int32_t>::max())});
    }
    OrderedPairClass &cls =
        out.emplace_back(make_ordered_class(res, la, lb, static_cast<int32_t>(n), firsts.size()));
    int32_t start = 0;
    for (const uint32_t a : firsts) {
      const auto count = static_cast<int32_t>(static_cast<int64_t>(shells[a].primitives().size()) *
                                              prims[static_cast<size_t>(lb)]);
      cls.groups.push_back({.shell = static_cast<int32_t>(a),
                            .ao_offset = static_cast<int32_t>(basis.ao_offset_cart(a)),
                            .start = start,
                            .count = count});
      start += count;
    }
    // Fork from mmd's stream, so the class reads mmd's final records, without
    // a host synchronization. Every fork precedes every join below: a join
    // into mmd's stream ahead of a sibling's fork would order the sibling
    // after this class's whole gather.
    FBF_TRY(gpu_status(cls.work_stream->fork_stream().fork_from(from->device_stream()->stream())));
    FBF_TRY(gather_ordered(cls, *from));
  }
  // mmd's stream waits on every class drawn from it: freeing an mmd class
  // queues its frees there, so they wait for every read of it.
  for (size_t c = 0; c < out.size(); ++c) {
    ForkStream &fork = out[c].work_stream->fork_stream();
    FBF_TRY(gpu_status(fork.record_join()));
    FBF_TRY(gpu_status(fork.join_into(plan[c].second->device_stream()->stream())));
  }
  return out;
}

} // namespace free_breakfast
