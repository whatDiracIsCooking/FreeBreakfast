/// @file
/// @brief The J build.

module;

// The ket sort's launcher, defined in sort_bounds/sort_bounds_launch.cu:
// declared in the GMF so it keeps external linkage (see the header).
#include "sort_bounds_bridge.h"

module free_breakfast.integrals.eri.integral_direct.coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;
import free_breakfast.integrals.eri.integral_direct.coulomb.dq;
import free_breakfast.integrals.eri.integral_direct.coulomb.e_hermite;
import free_breakfast.integrals.eri.integral_direct.coulomb.j_coulomb;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// The build proper, once coulomb has checked its inputs and zeroed j: J_p
// zeroed, then the ket phase and the bra phase, each forked from and joined
// back into stream.
Status coulomb_build(DeviceResources &res, const std::span<CoulombPairClass> classes,
                     const CartesianMatrixAO &density, CartesianMatrixAO &j, const double tau) {
  const wwr::wwrStream_t stream = res.stream();
  // Drawn from res.device()'s pool in stream's order, so every class's fork
  // from stream sees it allocated, and returned in its order at the end,
  // after both phases' joins.
  int32_t max_rec = 0;
  for (const CoulombPairClass &cls : classes) {
    max_rec = std::max(max_rec, cls.n_rec);
  }
  Buffers::device<int32_t> rows(static_cast<size_t>(max_rec), res.device());
  const std::span<int32_t> row_map{rows.data(), static_cast<size_t>(max_rec)};
  FBF_TRY(gpu_status(j_coulomb_rows(stream, row_map)));
  // e_hermite adds every ket class into J_p, from every class's stream, so it
  // is zeroed on stream, before the fork; the last build's last touch of it
  // (its jab, on the class's stream) was joined into stream.
  for (CoulombPairClass &cls : classes) {
    const size_t n = static_cast<size_t>(cls.n_rec) * static_cast<size_t>(cls.n_hermite);
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(cls.jp.data(), 0, sizeof(double) * n, stream)));
  }
  // launch_each hands each class over const; the build writes its scratch.
  const auto own = [&](const CoulombPairClass &cls) -> CoulombPairClass & {
    return classes[static_cast<size_t>(&cls - classes.data())];
  };

  // Every class as a ket, on its own stream: its weights, D_q and Dmax, the
  // class sorted by Iq * Dmax (Iq its records' bound, mmd's), then every bra
  // class contracted against it into the bra's J_p. Its sorted copies are
  // written and read on its stream alone; the bras' records are only read,
  // and their J_p only added to, atomically.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const CoulombPairClass &cls) {
    CoulombPairClass &ket = own(cls);
    if (const wwr::wwrError_t code = dq(ket, density); code != wwr::wwrSuccess) {
      return code;
    }
    const PrimitivePairClassView v = ket.view();
    if (const wwr::wwrError_t code = device::sort_bounds_kets(
            ket.device_stream()->stream(), v, ket.n_hermite, v.bound, ket.dmax.data(),
            ket.dq.data(), ket.ket_key.data(), ket.ket_key_alt.data(), ket.ket_perm.data(),
            ket.ket_perm_alt.data(), ket.ket_zeta.data(), ket.ket_p_x.data(), ket.ket_p_y.data(),
            ket.ket_p_z.data(), ket.ket_bound.data(), ket.ket_dmax.data(), ket.ket_dq.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    return e_hermite(ket, classes, tau);
  })));

  // Every class as a bra, its J_p complete once the ket phase's join closed:
  // its scatter into J, on its own stream, reading the class in record order.
  return gpu_status(launch_each(
      classes, stream, [&](const CoulombPairClass &bra) { return j_coulomb(bra, row_map, j); }));
}

Status coulomb(DeviceResources &res, const std::span<CoulombPairClass> classes,
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
  const Status built = coulomb_build(res, classes, density, j, tau);
  // A ket's stream reads every bra's records and adds into its J_p, and a
  // failed phase leaves some streams unforked by the next, so every class's
  // stream waits on stream, which has joined them all, before anything else
  // is queued there: a class's frees then follow every read of it.
  for (const CoulombPairClass &cls : classes) {
    FBF_TRY(gpu_status(cls.device_stream()->fork_stream().fork_from(stream)));
  }
  return built;
}

} // namespace free_breakfast
