/// @file
/// @brief ordered_pair_classes: mmd's records in both orientations, gathered
/// on the host into one class per ordered (la, lb), grouped by first shell;
/// then on the device each record's dressed bound, and each group sorted by
/// it.

module;

// The record-bound and record-order launchers, defined in pairs_launch.cu:
// declared in the GMF so they keep external linkage (see the header).
#include "pairs_bridge.h"

module free_breakfast.integrals.exchange.pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

#include "diagnostics.h"

namespace free_breakfast {

// One ordered record: record rec of mmd class cls, as stored or swapped.
struct OrderedPairSource {
  const PrimitivePairClass *cls;
  int32_t rec;
  bool swapped;
};

// An ordered class's records by first shell (a basis shell index), each
// group's in mmd's record order. A std::map, so the groups come out in basis
// order.
using OrderedPairPlan = std::map<int32_t, std::vector<OrderedPairSource>>;

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
      .zeta = MirroredBuffer<double>(len, ds),
      .p_x = MirroredBuffer<double>(len, ds),
      .p_y = MirroredBuffer<double>(len, ds),
      .p_z = MirroredBuffer<double>(len, ds),
      .pa_x = MirroredBuffer<double>(len, ds),
      .pa_y = MirroredBuffer<double>(len, ds),
      .pa_z = MirroredBuffer<double>(len, ds),
      .pb_x = MirroredBuffer<double>(len, ds),
      .pb_y = MirroredBuffer<double>(len, ds),
      .pb_z = MirroredBuffer<double>(len, ds),
      .s = MirroredBuffer<double>(len, ds),
      .record_bound = MirroredBuffer<double>(len, ds),
      .off_a = MirroredBuffer<int32_t>(len, ds),
      .off_b = MirroredBuffer<int32_t>(len, ds),
      .rank_b = MirroredBuffer<int32_t>(len, ds),
      .groups = {},
      .group_start = MirroredBuffer<int32_t>(n_groups + 1, ds),
  };
}

// Write record r of out's pinned side from src; rank_of names a shell's rank
// by its AO offset.
void write_ordered_record(OrderedPairClass &out, const size_t r, const OrderedPairSource &src,
                          const std::vector<int32_t> &rank_of) {
  const PrimitivePairClass &c = *src.cls;
  const auto k = static_cast<size_t>(src.rec);
  const auto from = [k](const auto &field) { return field.pinned().data()[k]; };
  const auto to = [r](auto &field) -> auto & { return field.pinned().data()[r]; };
  // The first shell's P - A and offset are mmd's ket's when swapped.
  const auto &[pa_x, pb_x] = src.swapped ? std::tie(c.pb_x, c.pa_x) : std::tie(c.pa_x, c.pb_x);
  const auto &[pa_y, pb_y] = src.swapped ? std::tie(c.pb_y, c.pa_y) : std::tie(c.pa_y, c.pb_y);
  const auto &[pa_z, pb_z] = src.swapped ? std::tie(c.pb_z, c.pa_z) : std::tie(c.pa_z, c.pb_z);
  const auto &[off_a, off_b] =
      src.swapped ? std::tie(c.off_b, c.off_a) : std::tie(c.off_a, c.off_b);
  to(out.zeta) = from(c.zeta);
  to(out.p_x) = from(c.p_x);
  to(out.p_y) = from(c.p_y);
  to(out.p_z) = from(c.p_z);
  to(out.pa_x) = from(pa_x);
  to(out.pa_y) = from(pa_y);
  to(out.pa_z) = from(pa_z);
  to(out.pb_x) = from(pb_x);
  to(out.pb_y) = from(pb_y);
  to(out.pb_z) = from(pb_z);
  to(out.s) = from(c.s);
  // Formed on the device (form_ordered_bounds); zero until then, and in a
  // class above d, which has no kernel.
  to(out.record_bound) = 0.0;
  to(out.off_a) = from(off_a);
  to(out.off_b) = from(off_b);
  to(out.rank_b) = rank_of[static_cast<size_t>(from(off_b))];
}

// Every per-record field, the ones the order permutes.
std::array<MirroredBuffer<double> *, 12> ordered_doubles(OrderedPairClass &cls) {
  return {&cls.zeta, &cls.p_x,  &cls.p_y,  &cls.p_z,  &cls.pa_x, &cls.pa_y,
          &cls.pa_z, &cls.pb_x, &cls.pb_y, &cls.pb_z, &cls.s,    &cls.record_bound};
}

std::array<MirroredBuffer<int32_t> *, 3> ordered_ints(OrderedPairClass &cls) {
  return {&cls.off_a, &cls.off_b, &cls.rank_b};
}

// Queue every field's pinned-to-device copy on cls's stream.
Status copy_ordered_to_device(OrderedPairClass &cls) {
  for (MirroredBuffer<double> *field : ordered_doubles(cls)) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  for (MirroredBuffer<int32_t> *field : ordered_ints(cls)) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  return gpu_status(cls.group_start.to_device());
}

// Each record's dressed bound, then each group sorted by it, non-increasing,
// on cls's stream after its copy to the device: the order, every field
// permuted to it in place, and the pinned side refreshed from the device. The
// groups do not move, so neither do group_start, the host's groups or any
// record's first shell. A class above d has no kernel and keeps its zero
// bounds and gather order: every build refuses such a class. The scratch is
// drawn on cls's stream and returned in its order.
Status form_ordered_bounds(OrderedPairClass &cls) {
  if (cls.la > kMaxAngularMomentum || cls.lb > kMaxAngularMomentum) {
    return {};
  }
  const std::shared_ptr<DeviceStream> &ds = cls.device_stream();
  const wwr::wwrStream_t stream = ds->stream();
  const OrderedPairClassView v = cls.view();
  FBF_TRY(gpu_status(device::ordered_pair_bound(stream, v, cls.record_bound.device().data())));
  const auto n = static_cast<size_t>(cls.n_rec);
  StreamBuffers::device<double> key(n, ds);
  StreamBuffers::device<double> key_alt(n, ds);
  StreamBuffers::device<uint32_t> perm(n, ds);
  StreamBuffers::device<uint32_t> perm_alt(n, ds);
  StreamBuffers::device<double> tmp(n, ds);
  StreamBuffers::device<int32_t> tmp_int(n, ds);
  FBF_TRY(gpu_status(device::ordered_pair_order(stream, cls.n_rec, v.n_groups, v.group_start,
                                                v.record_bound, key.data(), key_alt.data(),
                                                perm.data(), perm_alt.data())));
  for (MirroredBuffer<double> *field : ordered_doubles(cls)) {
    FBF_TRY(gpu_status(device::ordered_pair_permute(stream, cls.n_rec, perm.data(),
                                                    field->device().data(), tmp.data())));
    FBF_TRY(gpu_status(field->to_pinned()));
  }
  for (MirroredBuffer<int32_t> *field : ordered_ints(cls)) {
    FBF_TRY(gpu_status(device::ordered_pair_permute(stream, cls.n_rec, perm.data(),
                                                    field->device().data(), tmp_int.data())));
    FBF_TRY(gpu_status(field->to_pinned()));
  }
  return {};
}

Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes) {
  // Each shell's index and rank by its AO offset (-1: no shell starts there).
  // BasisSet guarantees nao_cart() fits an int32_t.
  const std::vector<Shell> &shells = basis.shells();
  std::vector<int32_t> shell_of(basis.nao_cart(), -1);
  std::vector<int32_t> rank_of(basis.nao_cart(), -1);
  std::map<int32_t, int32_t> next_rank;
  for (uint32_t i = 0; i < basis.nshells(); ++i) {
    const size_t off = basis.ao_offset_cart(i);
    shell_of[off] = static_cast<int32_t>(i);
    rank_of[off] = next_rank[shells[i].angular_momentum()]++;
  }
  const auto shell_at = [&shell_of](const int32_t off) {
    return off >= 0 && static_cast<size_t>(off) < shell_of.size()
               ? shell_of[static_cast<size_t>(off)]
               : -1;
  };

  const ScopedDeviceIndex on{res.dev_idx()};

  // Every ordered record's source, by ordered class, then first shell, each
  // group in mmd's record order.
  std::map<std::pair<int32_t, int32_t>, OrderedPairPlan> plan;
  for (const PrimitivePairClass &cls : classes) {
    // The pinned side holds the zeta-ordered records only once mmd's queued
    // order and refresh have run.
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(cls.device_stream()->stream())));
    for (int32_t r = 0; r < cls.n_rec; ++r) {
      const int32_t off_a = cls.off_a.pinned().data()[r];
      const int32_t off_b = cls.off_b.pinned().data()[r];
      const int32_t a = shell_at(off_a);
      const int32_t b = shell_at(off_b);
      if (a < 0 || b < 0) {
        return err_config("ordered_pair_classes: class ({}, {}) record {} has AO offsets ({}, "
                          "{}), not both the start of a shell of the basis",
                          cls.la, cls.lb, r, off_a, off_b);
      }
      plan[{cls.la, cls.lb}][a].push_back({&cls, r, false});
      if (off_a != off_b) {
        plan[{cls.lb, cls.la}][b].push_back({&cls, r, true});
      }
    }
  }

  std::vector<OrderedPairClass> out;
  out.reserve(plan.size());
  for (const auto &[key, groups] : plan) {
    int64_t n = 0;
    for (const auto &[shell, recs] : groups) {
      n += static_cast<int64_t>(recs.size());
    }
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("ordered_pair_classes: class ({}, {}) has {} records, "
                            "past {}",
                            key.first, key.second, n, std::numeric_limits<int32_t>::max())});
    }
    OrderedPairClass &cls = out.emplace_back(
        make_ordered_class(res, key.first, key.second, static_cast<int32_t>(n), groups.size()));
    size_t r = 0;
    for (const auto &[shell, recs] : groups) {
      cls.group_start.pinned().data()[cls.groups.size()] = static_cast<int32_t>(r);
      cls.groups.push_back({.shell = shell,
                            .ao_offset = static_cast<int32_t>(basis.ao_offset_cart(shell)),
                            .start = static_cast<int32_t>(r),
                            .count = static_cast<int32_t>(recs.size())});
      for (const OrderedPairSource &src : recs) {
        write_ordered_record(cls, r++, src, rank_of);
      }
    }
    cls.group_start.pinned().data()[cls.groups.size()] = static_cast<int32_t>(r);
    FBF_TRY(copy_ordered_to_device(cls));
    FBF_TRY(form_ordered_bounds(cls));
  }
  // The pinned side is final only once every class's refresh has run.
  for (const OrderedPairClass &cls : out) {
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(cls.device_stream()->stream())));
  }
  return out;
}

} // namespace free_breakfast
