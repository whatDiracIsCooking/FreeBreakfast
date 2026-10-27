/// @file
/// @brief The prune that makes J's classes.

module;

// The kernel launchers, defined in pair_class_launch.cu: declared in the GMF
// so they keep external linkage (see the header).
#include "pair_class_bridge.h"

module free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;

#include "diagnostics.h"

namespace free_breakfast {

// The Hermite functions of total order up to l, |t| <= l: one row of a
// record's D_q or J_p.
size_t coulomb_hermite_count(const int32_t l) {
  const auto n = static_cast<size_t>(l);
  return (n + 1) * (n + 2) * (n + 3) / 6;
}

// J's classes from mmd's, once every class's stream has been joined into
// res.stream(): the maxima over every class, each class's keep flags and
// their scan, the kept counts read back, then each kept class's scatter on its
// own new stream, joined back into res.stream() before its scratch is freed.
Result<std::vector<CoulombPairClass>>
coulomb_prune(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
              const double tau) {
  const wwr::wwrStream_t stream = res.stream();
  // Zeroed on stream at allocation: the maxima start from 0, and every bound
  // is 0 or above.
  Buffers::device<double> maxima(2, res.device());
  Buffers::device<int32_t> counts(classes.size(), res.device());
  std::vector<Buffers::device<int32_t>> dest;
  dest.reserve(classes.size());
  for (const PrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(device::coulomb_prune_maxima(stream, cls.view(), maxima.data())));
    dest.emplace_back(static_cast<size_t>(cls.n_rec), res.device());
  }
  for (size_t c = 0; c < classes.size(); ++c) {
    FBF_TRY(gpu_status(device::coulomb_prune_keep(stream, classes[c].view(), maxima.data(), tau,
                                                  dest[c].data(), counts.data() + c)));
  }
  // The one readback: J's classes are sized to their kept counts.
  Buffers::pinned<int32_t> kept(classes.size());
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(kept.data(), counts.data(), sizeof(int32_t) * classes.size(),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));

  std::vector<CoulombPairClass> out;
  // out[k] is classes[from[k]] pruned.
  std::vector<size_t> from;
  for (size_t c = 0; c < classes.size(); ++c) {
    const int32_t n_rec = kept.data()[c];
    if (n_rec == 0) {
      continue;
    }
    const auto n = static_cast<size_t>(n_rec);
    const size_t n_hermite = coulomb_hermite_count(classes[c].la + classes[c].lb);
    const auto ds = std::make_shared<DeviceStream>(res.device());
    out.push_back(CoulombPairClass{
        .la = classes[c].la,
        .lb = classes[c].lb,
        .n_rec = n_rec,
        .n_hermite = static_cast<int32_t>(n_hermite),
        .handle = ds,
        .zeta = StreamBuffers::device<double>(n, ds),
        .p_x = StreamBuffers::device<double>(n, ds),
        .p_y = StreamBuffers::device<double>(n, ds),
        .p_z = StreamBuffers::device<double>(n, ds),
        .pa_x = StreamBuffers::device<double>(n, ds),
        .pa_y = StreamBuffers::device<double>(n, ds),
        .pa_z = StreamBuffers::device<double>(n, ds),
        .pb_x = StreamBuffers::device<double>(n, ds),
        .pb_y = StreamBuffers::device<double>(n, ds),
        .pb_z = StreamBuffers::device<double>(n, ds),
        .s = StreamBuffers::device<double>(n, ds),
        .bound = StreamBuffers::device<double>(n, ds),
        .off_a = StreamBuffers::device<int32_t>(n, ds),
        .off_b = StreamBuffers::device<int32_t>(n, ds),
        .w = StreamBuffers::device<double>(n, ds),
        .dq = StreamBuffers::device<double>(n * n_hermite, ds),
        .dmax = StreamBuffers::device<double>(n, ds),
        .jp = StreamBuffers::device<double>(n * n_hermite, ds),
        .ket_key = StreamBuffers::device<double>(n, ds),
        .ket_key_alt = StreamBuffers::device<double>(n, ds),
        .ket_perm = StreamBuffers::device<uint32_t>(n, ds),
        .ket_perm_alt = StreamBuffers::device<uint32_t>(n, ds),
        .ket_zeta = StreamBuffers::device<double>(n, ds),
        .ket_p_x = StreamBuffers::device<double>(n, ds),
        .ket_p_y = StreamBuffers::device<double>(n, ds),
        .ket_p_z = StreamBuffers::device<double>(n, ds),
        .ket_bound = StreamBuffers::device<double>(n, ds),
        .ket_dmax = StreamBuffers::device<double>(n, ds),
        .ket_dq = StreamBuffers::device<double>(n * n_hermite, ds),
    });
    from.push_back(c);
  }

  // Each class's scatter on its own stream, after its allocation there and
  // after the scan on stream (the fork), and stream after every scatter (the
  // join), so dest is freed on stream after its last read.
  FBF_TRY(gpu_status(launch_each(out, stream, [&](const CoulombPairClass &pruned) {
    const auto k = static_cast<size_t>(&pruned - out.data());
    CoulombPairClass &to = out[k];
    const PrimitivePairClassView v = classes[from[k]].view();
    const int32_t *d = dest[from[k]].data();
    const wwr::wwrStream_t s = to.handle->stream();
    const std::array<std::pair<const double *, double *>, 12> fields{{
        {v.zeta, to.zeta.data()},
        {v.p_x, to.p_x.data()},
        {v.p_y, to.p_y.data()},
        {v.p_z, to.p_z.data()},
        {v.pa_x, to.pa_x.data()},
        {v.pa_y, to.pa_y.data()},
        {v.pa_z, to.pa_z.data()},
        {v.pb_x, to.pb_x.data()},
        {v.pb_y, to.pb_y.data()},
        {v.pb_z, to.pb_z.data()},
        {v.s, to.s.data()},
        {v.bound, to.bound.data()},
    }};
    for (const auto &[f, t] : fields) {
      if (const wwr::wwrError_t code = device::coulomb_prune_scatter(s, v.n_rec, d, f, t);
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    if (const wwr::wwrError_t code =
            device::coulomb_prune_scatter(s, v.n_rec, d, v.off_a, to.off_a.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    return device::coulomb_prune_scatter(s, v.n_rec, d, v.off_b, to.off_b.data());
  })));
  return out;
}

static_assert(device::kCoulombMaxMomentum == kMaxAngularMomentum,
              "J's limit and the tree-wide kMaxAngularMomentum disagree");

Result<std::vector<CoulombPairClass>>
coulomb_pair_classes(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
                     const double tau) {
  // Written so that NaN fails too.
  if (!(tau >= 0.0)) {
    return err_config("coulomb_pair_classes: the threshold tau is {}, but must be 0 or above", tau);
  }
  for (const PrimitivePairClass &cls : classes) {
    if (cls.la > device::kCoulombMaxMomentum || cls.lb > device::kCoulombMaxMomentum) {
      return err_config("coulomb_pair_classes: class ({}, {}) is above {}, the largest momentum "
                        "the kernels support",
                        cls.la, cls.lb, device::kCoulombMaxMomentum);
    }
  }
  if (classes.empty()) {
    return std::vector<CoulombPairClass>{};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // stream waits on every class's stream, where mmd's order and bounds may
  // still be queued: an event, not a host synchronization.
  for (const PrimitivePairClass &cls : classes) {
    ForkStream &fork = cls.device_stream()->fork_stream();
    FBF_TRY(gpu_status(fork.record_join()));
    FBF_TRY(gpu_status(fork.join_into(stream)));
  }
  Result<std::vector<CoulombPairClass>> out = coulomb_prune(res, classes, tau);
  // And every class's stream on stream, on success or failure, so nothing
  // queued there later (the records' own frees) runs ahead of a read of them.
  for (const PrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(cls.device_stream()->fork_stream().fork_from(stream)));
  }
  return out;
}

} // namespace free_breakfast
