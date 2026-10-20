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

// One class's scratch: its records' weights and, as a ket, D_q and Dmax; as a
// bra, J_p. D_q and J_p share the class's Hermite order, la + lb.
struct CoulombScratch {
  Buffers::device<double> w;
  Buffers::device<double> dq;
  Buffers::device<double> dmax;
  Buffers::device<double> jp;
};

Status coulomb(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
               const CartesianMatrixAO &density, CartesianMatrixAO &j) {
  FBF_TRY(check_one_electron_build(res, classes, j, "coulomb"));
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
  Buffers::device<double> ones(static_cast<size_t>(max_rec), res.device());
  Buffers::device<int32_t> rows(static_cast<size_t>(max_rec), res.device());
  FBF_TRY(gpu_status(device::coulomb_constants(stream, max_rec, ones.data(), rows.data())));
  std::vector<CoulombScratch> scratch;
  scratch.reserve(classes.size());
  for (const PrimitivePairClass &cls : classes) {
    const auto n_rec = static_cast<size_t>(cls.n_rec);
    const size_t n = n_rec * coulomb_hermite_count(cls.la + cls.lb);
    CoulombScratch &sc = scratch.emplace_back(CoulombScratch{
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device()),
        Buffers::device<double>(n_rec, res.device()), Buffers::device<double>(n, res.device())});
    // e_hermite adds every ket class into J_p.
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(sc.jp.data(), 0, sizeof(double) * n, stream)));
  }
  const auto of = [&](const PrimitivePairClass &cls) -> CoulombScratch & {
    return scratch[static_cast<size_t>(&cls - classes.data())];
  };
  const auto n_ao = static_cast<int32_t>(j.n());

  // Every class as a ket: its D_q and Dmax, which every bra class reads, so
  // the join closes the phase before any of them starts.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    CoulombScratch &k = of(cls);
    return device::coulomb_dq(cls.device_stream()->stream(), cls.view(), n_ao, density.data(),
                              k.w.data(), k.dq.data(), k.dmax.data());
  })));

  // Every class as a bra: J_p over every ket class, then its scatter into J,
  // in that order on the bra's own stream.
  return gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    CoulombScratch &b = of(bra);
    for (const PrimitivePairClass &ket : classes) {
      const CoulombScratch &k = of(ket);
      if (const wwr::wwrError_t code = device::coulomb_e_hermite(
              s, bra.view(), ket.view(), ones.data(), k.dq.data(), k.dmax.data(), b.jp.data());
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return device::coulomb_jab(s, bra.view(), n_ao, rows.data(), b.jp.data(), j.data());
  }));
}

} // namespace free_breakfast
