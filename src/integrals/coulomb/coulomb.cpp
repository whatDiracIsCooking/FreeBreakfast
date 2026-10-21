/// @file
/// @brief The J build.

module;

// The kernel launchers, defined in coulomb_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "coulomb_bridge.h"

module free_breakfast.integrals.coulomb;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
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

// One class's scratch: its records' weights and Schwarz bounds (Ip as a bra,
// Iq as a ket) and, as a ket, D_q and Dmax, in record order and sorted by
// Iq * Dmax; as a bra, J_p. D_q and J_p share the class's Hermite order,
// la + lb.
struct CoulombScratch {
  Buffers::device<double> w;
  Buffers::device<double> bound;
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
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n_rec, res.device()),
        Buffers::device<double>(n, res.device()), Buffers::device<double>(n_rec, res.device()),
        Buffers::device<double>(n, res.device()),
        SortedKets(res.device(), cls, static_cast<int32_t>(n_hermite))});
    // e_hermite adds every ket class into J_p.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(sc.jp.data(), 0, sizeof(double) * n, stream)));
  }
  const auto of = [&](const PrimitivePairClass &cls) -> CoulombScratch & {
    return scratch[static_cast<size_t>(&cls - classes.data())];
  };
  const auto n_ao = static_cast<int32_t>(j.n());

  // Every class's inputs: its weights, its bounds and, as a ket, its D_q and
  // Dmax, then the class sorted by Iq * Dmax as a ket, which every bra class
  // reads, so the join closes the phase before any of them starts.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    const wwr::wwrStream_t s = cls.device_stream()->stream();
    CoulombScratch &k = of(cls);
    if (const wwr::wwrError_t code =
            device::coulomb_prepare(s, cls.view(), n_ao, density.data(), k.w.data(), k.bound.data(),
                                    k.dq.data(), k.dmax.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    return k.kets.sort(s, k.bound.data(), k.dmax.data(), k.dq.data());
  })));

  // Every class as a bra: J_p over every ket class, sorted, then its scatter
  // into J, in that order on the bra's own stream. The bra side reads the
  // class in record order, so jab's rows stay the identity.
  return gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    CoulombScratch &b = of(bra);
    for (const PrimitivePairClass &ket : classes) {
      const SortedKets &k = of(ket).kets;
      if (const wwr::wwrError_t code =
              device::coulomb_e_hermite(s, bra.view(), k.view(), tau, b.bound.data(), k.bound(),
                                        k.dq(), k.dmax(), b.jp.data());
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return device::coulomb_jab(s, bra.view(), n_ao, rows.data(), b.jp.data(), j.data());
  }));
}

} // namespace free_breakfast
