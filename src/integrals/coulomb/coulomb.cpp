/// @file
/// @brief The J build, and the prune that makes its classes.

module;

// The kernel launchers, defined in coulomb_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "coulomb_bridge.h"

module free_breakfast.integrals.coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.coulomb.sort_bounds;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// The Hermite functions of total order up to l, |t| <= l: one row of a
// record's D_q or J_p.
size_t coulomb_hermite_count(const int32_t l) {
  const auto n = static_cast<size_t>(l);
  return (n + 1) * (n + 2) * (n + 3) / 6;
}

// One class's scratch: its records' weights and, as a ket, D_q and Dmax, in
// record order, and those with zeta, P and the Schwarz bound Iq (the class's
// own bound field) sorted by Iq * Dmax; as a bra, J_p. D_q and J_p share the
// class's Hermite order, la + lb.
struct CoulombScratch {
  Buffers::device<double> w;
  Buffers::device<double> dq;
  Buffers::device<double> dmax;
  Buffers::device<double> jp;
  SortedKets kets;
};

Status coulomb(DeviceResources &res, const std::span<const CoulombPairClass> classes,
               const CartesianMatrixAO &density, CartesianMatrixAO &j, const double tau) {
  // The matrix's checks are the one-electron builds'; J's classes are not
  // mmd's, so theirs are here, with the same messages.
  FBF_TRY(check_one_electron_build(res, {}, j, "coulomb"));
  for (const CoulombPairClass &cls : classes) {
    if (const int32_t dev = cls.device_stream()->dev_idx(); dev != res.dev_idx()) {
      return err_config("coulomb: class ({}, {}) is on device {}, but the call passed device {}",
                        cls.la, cls.lb, dev, res.dev_idx());
    }
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum) {
      return err_config("coulomb: no kernel for class ({}, {}); the kernels cover la <= lb <= {}",
                        cls.la, cls.lb, kMaxAngularMomentum);
    }
  }
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau >= 0.0)) {
    return err_config("coulomb: the screening threshold tau is {}, but must be 0 or above", tau);
  }
  if (density.dev_idx() != res.dev_idx()) {
    return err_config("coulomb: the density is on device {}, but the call passed device {}",
                      density.dev_idx(), res.dev_idx());
  }
  if (density.n() != j.n()) {
    return err_config("coulomb: the density is {0} x {0}, but J is {1} x {1}", density.n(), j.n());
  }
  if (j.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // jab only adds, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(j.data(), 0, sizeof(double) * j.num_elements(), stream)));
  if (classes.empty()) {
    return {};
  }

  // Drawn from res.device()'s pool in stream's order, so every class's fork
  // from stream sees them allocated, and returned in its order at the end,
  // after both phases' joins.
  int32_t max_rec = 0;
  for (const CoulombPairClass &cls : classes) {
    max_rec = std::max(max_rec, cls.n_rec);
  }
  Buffers::device<int32_t> rows(static_cast<size_t>(max_rec), res.device());
  FBF_TRY(gpu_status(device::coulomb_rows(stream, max_rec, rows.data())));
  std::vector<CoulombScratch> scratch;
  scratch.reserve(classes.size());
  for (const CoulombPairClass &cls : classes) {
    const auto n_rec = static_cast<size_t>(cls.n_rec);
    const size_t n_hermite = coulomb_hermite_count(cls.la + cls.lb);
    const size_t n = n_rec * n_hermite;
    CoulombScratch &sc = scratch.emplace_back(CoulombScratch{
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device()),
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device()),
        SortedKets(res.device(), cls.view(), static_cast<int32_t>(n_hermite))});
    // e_hermite adds every ket class into J_p.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(sc.jp.data(), 0, sizeof(double) * n, stream)));
  }
  const auto of = [&](const CoulombPairClass &cls) -> CoulombScratch & {
    return scratch[static_cast<size_t>(&cls - classes.data())];
  };
  const auto n_ao = static_cast<int32_t>(j.n());

  // Every class's inputs: its weights and, as a ket, its D_q and Dmax, then
  // the class sorted by Iq * Dmax as a ket (Iq its records' bound, mmd's),
  // which every bra class reads, so the join closes the phase before any of
  // them starts.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const CoulombPairClass &cls) {
    const wwr::wwrStream_t s = cls.device_stream()->stream();
    CoulombScratch &k = of(cls);
    const PrimitivePairClassView v = cls.view();
    if (const wwr::wwrError_t code = device::coulomb_prepare(s, v, n_ao, density.data(), k.w.data(),
                                                             k.dq.data(), k.dmax.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    return k.kets.sort(s, v.bound, k.dmax.data(), k.dq.data());
  })));

  // Every class as a bra: J_p over every ket class, sorted, then its scatter
  // into J, in that order on the bra's own stream. The bra side reads the
  // class in record order, its Ip the class's own bound, so jab's rows stay
  // the identity.
  return gpu_status(launch_each(classes, stream, [&](const CoulombPairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    CoulombScratch &b = of(bra);
    const PrimitivePairClassView v = bra.view();
    for (const CoulombPairClass &ket : classes) {
      const SortedKets &k = of(ket).kets;
      if (const wwr::wwrError_t code = device::coulomb_e_hermite(
              s, v, k.view(), tau, v.bound, k.bound(), k.dq(), k.dmax(), b.jp.data());
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return device::coulomb_jab(s, v, n_ao, rows.data(), b.jp.data(), j.data());
  }));
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
    const auto ds = std::make_shared<DeviceStream>(res.device());
    out.push_back(CoulombPairClass{
        .la = classes[c].la,
        .lb = classes[c].lb,
        .n_rec = n_rec,
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

Result<std::vector<CoulombPairClass>>
coulomb_pair_classes(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
                     const double tau) {
  // Written so that NaN fails too.
  if (!(tau >= 0.0)) {
    return err_config("coulomb_pair_classes: the threshold tau is {}, but must be 0 or above", tau);
  }
  for (const PrimitivePairClass &cls : classes) {
    if (cls.la > kMaxAngularMomentum || cls.lb > kMaxAngularMomentum) {
      return err_config("coulomb_pair_classes: class ({}, {}) is above d, the largest momentum "
                        "the kernels support",
                        cls.la, cls.lb);
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
