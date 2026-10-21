/// @file
/// @brief primitive_pair_classes: the shell-pair sweep, the la <= lb swap, the
/// per-record arithmetic, the copy to the device and the order there.

module;

// The record-order launchers, defined in mmd_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "mmd_bridge.h"

module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;
import free_breakfast.basis;

#include "diagnostics.h"

namespace free_breakfast {

// A shell-pair class, (la, lb) with la <= lb.
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

// A class of n records on a new stream of res's Device, every field zeroed.
PrimitivePairClass make_class(DeviceResources &res, const ClassKey key, const int32_t n) {
  const auto ds = std::make_shared<DeviceStream>(res.device());
  const auto len = static_cast<size_t>(n);
  return {
      .la = key.first,
      .lb = key.second,
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
      .eb = MirroredBuffer<double>(len, ds),
      .off_a = MirroredBuffer<int32_t>(len, ds),
      .off_b = MirroredBuffer<int32_t>(len, ds),
  };
}

// Write the records of shell pair (a, b), already in la <= lb order, into
// cls's pinned side from record index rec; return the index one past the last
// record written.
int32_t write_shell_pair_records(PrimitivePairClass &cls, const int32_t rec, const Shell &a,
                                 const int32_t off_a, const Shell &b, const int32_t off_b) {
  const Vec<double, 3> &ca = a.center();
  const Vec<double, 3> &cb = b.center();
  auto r = static_cast<size_t>(rec);
  const auto at = [&r](auto &field) -> auto & { return field.pinned().data()[r]; };
  for (const Primitive &pa : a.primitives()) {
    for (const Primitive &pb : b.primitives()) {
      const double zeta = pa.exponent + pb.exponent;
      const double px = ((pa.exponent * ca.x) + (pb.exponent * cb.x)) / zeta;
      const double py = ((pa.exponent * ca.y) + (pb.exponent * cb.y)) / zeta;
      const double pz = ((pa.exponent * ca.z) + (pb.exponent * cb.z)) / zeta;
      at(cls.zeta) = zeta;
      at(cls.p_x) = px;
      at(cls.p_y) = py;
      at(cls.p_z) = pz;
      at(cls.pa_x) = px - ca.x;
      at(cls.pa_y) = py - ca.y;
      at(cls.pa_z) = pz - ca.z;
      at(cls.pb_x) = px - cb.x;
      at(cls.pb_y) = py - cb.y;
      at(cls.pb_z) = pz - cb.z;
      at(cls.s) = pa.coefficient * pb.coefficient * std::pow(kPi<double> / zeta, 1.5);
      at(cls.eb) = pb.exponent;
      at(cls.off_a) = off_a;
      at(cls.off_b) = off_b;
      ++r;
    }
  }
  return static_cast<int32_t>(r);
}

// Queue every field's pinned-to-device copy on cls's stream.
Status copy_to_device(PrimitivePairClass &cls) {
  for (MirroredBuffer<double> *field :
       {&cls.zeta, &cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z, &cls.pb_x,
        &cls.pb_y, &cls.pb_z, &cls.s, &cls.eb}) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  for (MirroredBuffer<int32_t> *field : {&cls.off_a, &cls.off_b}) {
    FBF_TRY(gpu_status(field->to_device()));
  }
  return {};
}

// Put cls's records in descending zeta order on its stream, after their copy
// to the device: the order, then every field permuted to it in place, then the
// pinned side refreshed from the device. Within a class the record bound Ip =
// pair_bound(zeta, la + lb) rises with zeta, so this is descending Ip: a J
// build's bras with the longest ket loops launch first. Every kernel
// accumulates, so none depends on the order. The scratch is drawn on cls's
// stream and returned in its order.
Status order_by_zeta(PrimitivePairClass &cls) {
  const std::shared_ptr<DeviceStream> &ds = cls.device_stream();
  const wwr::wwrStream_t stream = ds->stream();
  const auto n = static_cast<size_t>(cls.n_rec);
  StreamBuffers::device<double> key(n, ds);
  StreamBuffers::device<double> key_alt(n, ds);
  StreamBuffers::device<uint32_t> perm(n, ds);
  StreamBuffers::device<uint32_t> perm_alt(n, ds);
  StreamBuffers::device<double> tmp(n, ds);
  StreamBuffers::device<int32_t> tmp_offsets(n, ds);
  FBF_TRY(gpu_status(device::mmd_order(stream, cls.n_rec, cls.zeta.device().data(), key.data(),
                                       key_alt.data(), perm.data(), perm_alt.data())));
  for (MirroredBuffer<double> *field :
       {&cls.zeta, &cls.p_x, &cls.p_y, &cls.p_z, &cls.pa_x, &cls.pa_y, &cls.pa_z, &cls.pb_x,
        &cls.pb_y, &cls.pb_z, &cls.s, &cls.eb}) {
    FBF_TRY(gpu_status(
        device::mmd_permute(stream, cls.n_rec, perm.data(), field->device().data(), tmp.data())));
    FBF_TRY(gpu_status(field->to_pinned()));
  }
  for (MirroredBuffer<int32_t> *field : {&cls.off_a, &cls.off_b}) {
    FBF_TRY(gpu_status(device::mmd_permute(stream, cls.n_rec, perm.data(), field->device().data(),
                                           tmp_offsets.data())));
    FBF_TRY(gpu_status(field->to_pinned()));
  }
  return {};
}

Result<std::vector<PrimitivePairClass>> primitive_pair_classes(DeviceResources &res,
                                                               const BasisSet &basis) {
  // BasisSet guarantees nao_cart() fits an int32_t, so the offset casts below are exact.
  const std::vector<Shell> &shells = basis.shells();
  const uint32_t nsh = basis.nshells();

  // Pass 1: every reached class's record count, in int64_t so a class past
  // INT32_MAX is caught before it is sized. Keyed by (la, lb), so the result
  // comes out in ascending class order.
  std::map<ClassKey, int64_t> counts;
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      counts[{shells[bra].angular_momentum(), shells[ket].angular_momentum()}] +=
          static_cast<int64_t>(shells[bra].primitives().size()) *
          static_cast<int64_t>(shells[ket].primitives().size());
    }
  }

  // Size each class once, at its final record count.
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<PrimitivePairClass> out;
  out.reserve(counts.size());
  std::map<ClassKey, size_t> slot;
  for (const auto &[key, n] : counts) {
    if (n > std::numeric_limits<int32_t>::max()) {
      check_or_abort(
          Status{err_config("primitive_pair_classes: class ({}, {}) has {} records, "
                            "past {}",
                            key.first, key.second, n, std::numeric_limits<int32_t>::max())});
    }
    slot.emplace(key, out.size());
    out.push_back(make_class(res, key, static_cast<int32_t>(n)));
  }

  // Pass 2: the same sweep, each shell pair's records written at its class's
  // cursor, in shell-pair, then bra-primitive-major, order: the order the host
  // writes them in, before order_by_zeta reorders them on the device.
  std::vector<int32_t> cursor(out.size(), 0);
  for (uint32_t i = 0; i < nsh; ++i) {
    for (uint32_t j = i; j < nsh; ++j) {
      const auto [bra, ket] = bra_ket_order(shells, i, j);
      const size_t c = slot.at({shells[bra].angular_momentum(), shells[ket].angular_momentum()});
      cursor[c] = write_shell_pair_records(
          out[c], cursor[c], shells[bra], static_cast<int32_t>(basis.ao_offset_cart(bra)),
          shells[ket], static_cast<int32_t>(basis.ao_offset_cart(ket)));
    }
  }

  for (PrimitivePairClass &cls : out) {
    FBF_TRY(copy_to_device(cls));
    FBF_TRY(order_by_zeta(cls));
  }
  return out;
}

} // namespace free_breakfast
