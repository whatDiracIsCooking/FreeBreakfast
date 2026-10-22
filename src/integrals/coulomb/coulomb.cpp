/// @file
/// @brief The J build.

module;

// The kernel launchers, defined in coulomb_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "coulomb_bridge.h"
// The dressed record bound's launcher (mmd's device library), which
// coulomb_pair_classes runs over mmd's records: the same arrangement.
#include "mmd_bridge.h"

module free_breakfast.integrals.coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
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

Status coulomb(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
               const CartesianMatrixAO &density, CartesianMatrixAO &j, const double tau) {
  FBF_TRY(check_one_electron_build(res, classes, j, "coulomb"));
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
  for (const PrimitivePairClass &cls : classes) {
    max_rec = std::max(max_rec, cls.n_rec);
  }
  Buffers::device<int32_t> rows(static_cast<size_t>(max_rec), res.device());
  FBF_TRY(gpu_status(device::coulomb_rows(stream, max_rec, rows.data())));
  std::vector<CoulombScratch> scratch;
  scratch.reserve(classes.size());
  for (const PrimitivePairClass &cls : classes) {
    const auto n_rec = static_cast<size_t>(cls.n_rec);
    const size_t n_hermite = coulomb_hermite_count(cls.la + cls.lb);
    const size_t n = n_rec * n_hermite;
    CoulombScratch &sc = scratch.emplace_back(CoulombScratch{
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device()),
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device()),
        SortedKets(res.device(), cls, static_cast<int32_t>(n_hermite))});
    // e_hermite adds every ket class into J_p.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(sc.jp.data(), 0, sizeof(double) * n, stream)));
  }
  const auto of = [&](const PrimitivePairClass &cls) -> CoulombScratch & {
    return scratch[static_cast<size_t>(&cls - classes.data())];
  };
  const auto n_ao = static_cast<int32_t>(j.n());

  // Every class's inputs: its weights and, as a ket, its D_q and Dmax, then
  // the class sorted by Iq * Dmax as a ket (Iq its records' bound, mmd's),
  // which every bra class reads, so the join closes the phase before any of
  // them starts.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
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
  return gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    CoulombScratch &b = of(bra);
    const PrimitivePairClassView v = bra.view();
    for (const PrimitivePairClass &ket : classes) {
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

// The density cap coulomb_pair_classes assumes: |D_cd| <= this for every
// density its classes are used with (its doc says why 2, and that it is an
// assumption).
constexpr double kCoulombPruneDensityCap = 2.0;

// Each record's dressed bound B (ordered_pair_bound over the class as mmd
// stores it), read back to the host.
Result<std::vector<double>> coulomb_record_bounds(const PrimitivePairClass &cls) {
  const std::shared_ptr<DeviceStream> &ds = cls.device_stream();
  MirroredBuffer<double> bound(static_cast<size_t>(cls.n_rec), ds);
  FBF_TRY(gpu_status(device::ordered_pair_bound(ds->stream(), cls.view(), bound.device().data())));
  FBF_TRY(gpu_status(bound.to_pinned_sync()));
  const double *b = bound.pinned().data();
  return std::vector<double>(b, b + cls.n_rec);
}

// A copy of cls holding only the records keep names, in that order, on a new
// stream of res's Device, its copy to the device queued there.
Result<PrimitivePairClass> coulomb_pruned_class(DeviceResources &res, const PrimitivePairClass &cls,
                                                const std::vector<int32_t> &keep) {
  const auto ds = std::make_shared<DeviceStream>(res.device());
  const size_t n = keep.size();
  PrimitivePairClass out{
      .la = cls.la,
      .lb = cls.lb,
      .n_rec = static_cast<int32_t>(n),
      .zeta = MirroredBuffer<double>(n, ds),
      .p_x = MirroredBuffer<double>(n, ds),
      .p_y = MirroredBuffer<double>(n, ds),
      .p_z = MirroredBuffer<double>(n, ds),
      .pa_x = MirroredBuffer<double>(n, ds),
      .pa_y = MirroredBuffer<double>(n, ds),
      .pa_z = MirroredBuffer<double>(n, ds),
      .pb_x = MirroredBuffer<double>(n, ds),
      .pb_y = MirroredBuffer<double>(n, ds),
      .pb_z = MirroredBuffer<double>(n, ds),
      .s = MirroredBuffer<double>(n, ds),
      .ea = MirroredBuffer<double>(n, ds),
      .eb = MirroredBuffer<double>(n, ds),
      .bound = MirroredBuffer<double>(n, ds),
      .bound_ab = MirroredBuffer<double>(n, ds),
      .bound_ba = MirroredBuffer<double>(n, ds),
      .off_a = MirroredBuffer<int32_t>(n, ds),
      .off_b = MirroredBuffer<int32_t>(n, ds),
      .rank_a = MirroredBuffer<int32_t>(n, ds),
      .rank_b = MirroredBuffer<int32_t>(n, ds),
  };
  const auto copy = [&keep, n](auto &to, const auto &from) {
    for (size_t r = 0; r < n; ++r) {
      to.pinned().data()[r] = from.pinned().data()[static_cast<size_t>(keep[r])];
    }
    return gpu_status(to.to_device());
  };
  FBF_TRY(copy(out.zeta, cls.zeta));
  FBF_TRY(copy(out.p_x, cls.p_x));
  FBF_TRY(copy(out.p_y, cls.p_y));
  FBF_TRY(copy(out.p_z, cls.p_z));
  FBF_TRY(copy(out.pa_x, cls.pa_x));
  FBF_TRY(copy(out.pa_y, cls.pa_y));
  FBF_TRY(copy(out.pa_z, cls.pa_z));
  FBF_TRY(copy(out.pb_x, cls.pb_x));
  FBF_TRY(copy(out.pb_y, cls.pb_y));
  FBF_TRY(copy(out.pb_z, cls.pb_z));
  FBF_TRY(copy(out.s, cls.s));
  FBF_TRY(copy(out.ea, cls.ea));
  FBF_TRY(copy(out.eb, cls.eb));
  FBF_TRY(copy(out.bound, cls.bound));
  FBF_TRY(copy(out.bound_ab, cls.bound_ab));
  FBF_TRY(copy(out.bound_ba, cls.bound_ba));
  FBF_TRY(copy(out.off_a, cls.off_a));
  FBF_TRY(copy(out.off_b, cls.off_b));
  FBF_TRY(copy(out.rank_a, cls.rank_a));
  FBF_TRY(copy(out.rank_b, cls.rank_b));
  return out;
}

Result<std::vector<PrimitivePairClass>>
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
  const ScopedDeviceIndex on{res.dev_idx()};

  // Each record's B as a bra and K = w ncart(la) B as a ket, and their maxima.
  std::vector<std::vector<double>> bra(classes.size());
  std::vector<std::vector<double>> ket(classes.size());
  double max_bra = 0.0;
  double max_ket = 0.0;
  for (size_t c = 0; c < classes.size(); ++c) {
    const PrimitivePairClass &cls = classes[c];
    // mmd's pinned side is final only once its queued order and refresh have
    // run; the copies below read it.
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(cls.device_stream()->stream())));
    Result<std::vector<double>> bounds = coulomb_record_bounds(cls);
    if (!bounds) {
      return std::unexpected(bounds.error());
    }
    bra[c] = std::move(*bounds);
    const auto ncart = static_cast<double>(function_count_cart(cls.la));
    ket[c].resize(bra[c].size());
    for (size_t r = 0; r < bra[c].size(); ++r) {
      const double w = cls.off_a.pinned().data()[r] == cls.off_b.pinned().data()[r] ? 1.0 : 2.0;
      ket[c][r] = w * ncart * bra[c][r];
      max_bra = std::max(max_bra, bra[c][r]);
      max_ket = std::max(max_ket, ket[c][r]);
    }
  }

  std::vector<PrimitivePairClass> out;
  for (size_t c = 0; c < classes.size(); ++c) {
    std::vector<int32_t> keep;
    for (size_t r = 0; r < bra[c].size(); ++r) {
      // Pruned only when under tau both as a bra and as a ket; tau = 0 keeps
      // every record.
      if (bra[c][r] * kCoulombPruneDensityCap * max_ket >= tau ||
          ket[c][r] * kCoulombPruneDensityCap * max_bra >= tau) {
        keep.push_back(static_cast<int32_t>(r));
      }
    }
    if (keep.empty()) {
      continue;
    }
    Result<PrimitivePairClass> pruned = coulomb_pruned_class(res, classes[c], keep);
    if (!pruned) {
      return std::unexpected(pruned.error());
    }
    out.push_back(std::move(*pruned));
  }
  return out;
}

} // namespace free_breakfast
