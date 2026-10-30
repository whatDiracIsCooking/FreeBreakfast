/// @file
/// @brief The J build, and its nuclear gradient: the same ket phase, the bra
/// raised by one, then gjab.

module;

// The launchers, defined in sort_bounds/sort_bounds_launch.cu (the ket sort),
// e_hermite/e_hermite_launch.cu (the gradient's raised [p|q]) and
// gradient/coulomb_gradient_launch.cu (gjab): declared in the GMF so they keep
// external linkage (see the headers). J's L limit, kCoulombMaxMomentum, is
// pair_class's bridge's.
#include "coulomb_gradient_bridge.h"
#include "e_hermite_bridge.h"
#include "pair_class_bridge.h"
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

// One class made ready as a ket, on its own stream, by the build and the
// gradient alike: its weights, D_q and Dmax of density, then the class sorted
// by Iq * Dmax (Iq its records' bound, mmd's) into its ket_ fields, which
// e_hermite reads of a ket. The sorted copies are written and read on that
// stream alone.
wwr::wwrError_t coulomb_ket(CoulombPairClass &ket, const CartesianMatrixAO &density) {
  if (const wwr::wwrError_t code = dq(ket, density); code != wwr::wwrSuccess) {
    return code;
  }
  const PrimitivePairClassView v = ket.view();
  return device::sort_bounds_kets(
      ket.device_stream()->stream(), v, ket.n_hermite, v.bound, ket.dmax.data(), ket.dq.data(),
      ket.ket_key.data(), ket.ket_key_alt.data(), ket.ket_perm.data(), ket.ket_perm_alt.data(),
      ket.ket_zeta.data(), ket.ket_p_x.data(), ket.ket_p_y.data(), ket.ket_p_z.data(),
      ket.ket_bound.data(), ket.ket_dmax.data(), ket.ket_dq.data());
}

// The build proper, once coulomb has checked its inputs and zeroed j: J_p
// zeroed, then the ket phase and the bra phase, each forked from and joined
// back into stream.
Status coulomb_build(DeviceResources &res, const std::span<CoulombPairClass> classes,
                     const CartesianMatrixAO &density, CartesianMatrixAO &j, const double tau) {
  const wwr::wwrStream_t stream = res.stream();
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

  // Every class as a ket, on its own stream (coulomb_ket), then every bra
  // class contracted against it into the bra's J_p. Its sorted copies are
  // written and read on its stream alone; the bras' records are only read,
  // and their J_p only added to, atomically.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const CoulombPairClass &cls) {
    CoulombPairClass &ket = own(cls);
    if (const wwr::wwrError_t code = coulomb_ket(ket, density); code != wwr::wwrSuccess) {
      return code;
    }
    return e_hermite(ket, classes, tau);
  })));

  // Every class as a bra, its J_p complete once the ket phase's join closed:
  // its scatter into J, on its own stream, reading the class in record order.
  return gpu_status(
      launch_each(classes, stream, [&](const CoulombPairClass &bra) { return j_coulomb(bra, j); }));
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
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > device::kCoulombMaxMomentum) {
      return err_config("coulomb: no kernel for class ({}, {}); the kernels cover la <= lb <= {}",
                        cls.la, cls.lb, device::kCoulombMaxMomentum);
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

// Where each of J's classes starts in the gradient's scratch: its raised J_p
// (n_rec x n_hermite(la + lb + 1)) and its records' two atoms (2 n_rec), each
// list one entry longer than the classes, the last the total.
struct CoulombGradientSlices {
  std::vector<size_t> jp{0};
  std::vector<size_t> atoms{0};
};

// The two phases over J's classes, once every input is checked and uploaded:
// every class as a ket (its D_q, sorted, then the raised [p|q] of every bra
// class against it into that bra's slice of jp), then every class as a bra
// (gjab of its raised J_p into g). jp and g are zeroed; mmd's classes[k] is
// j_classes[k] record for record, and gives gjab the exponents ea and eb that
// J's classes do not carry.
Status coulomb_gradient_build(DeviceResources &res,
                              const std::span<const PrimitivePairClass> classes,
                              const std::span<CoulombPairClass> j_classes,
                              const CartesianMatrixAO &density, const double tau,
                              const int32_t n_atom, const CoulombGradientSlices &at, double *jp,
                              const int32_t *ao_atom, int32_t *atoms, double *g) {
  const wwr::wwrStream_t stream = res.stream();
  const auto index = [&](const CoulombPairClass &cls) {
    return static_cast<size_t>(&cls - j_classes.data());
  };
  // Every class as a ket, on its own stream, as the J build runs it; the bras
  // are only read, and their raised J_p only added to, atomically.
  FBF_TRY(gpu_status(launch_each(j_classes, stream, [&](const CoulombPairClass &cls) {
    CoulombPairClass &ket = j_classes[index(cls)];
    if (const wwr::wwrError_t code = coulomb_ket(ket, density); code != wwr::wwrSuccess) {
      return code;
    }
    const PrimitivePairClassView k = ket.ket_view();
    for (const CoulombPairClass &bra : j_classes) {
      const PrimitivePairClassView b = bra.view();
      if (const wwr::wwrError_t code = device::e_hermite_class(
              ket.device_stream()->stream(), b, k, tau, b.bound, k.bound, ket.ket_dq.data(),
              ket.ket_dmax.data(), jp + at.jp[index(bra)], 1);
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return wwr::wwrSuccess;
  })));

  // Every class as a bra, its raised J_p complete once the ket phase's join
  // closed, read where it lies.
  const auto n_ao = static_cast<int32_t>(density.n());
  return gpu_status(launch_each(j_classes, stream, [&](const CoulombPairClass &bra) {
    const size_t c = index(bra);
    PrimitivePairClassView v = bra.view();
    const PrimitivePairClassView exponents = classes[c].view();
    v.ea = exponents.ea;
    v.eb = exponents.eb;
    return device::coulomb_gradient_class(bra.device_stream()->stream(), v, bra.row.data(), ao_atom,
                                          atoms + at.atoms[c], n_ao, n_atom, jp + at.jp[c],
                                          density.data(), g);
  }));
}

Result<std::vector<double>> coulomb_gradient(DeviceResources &res, const BasisSet &basis,
                                             const std::span<const PrimitivePairClass> classes,
                                             const SphericalMatrixAO &density, const double tau) {
  if (res.dev_idx() != density.dev_idx()) {
    return err_config("coulomb_gradient: the density is on device {}, but the call passed "
                      "device {}",
                      density.dev_idx(), res.dev_idx());
  }
  if (density.n() != basis.nao_sphe()) {
    return err_config("coulomb_gradient: the density is {} square, but the basis has {} "
                      "spherical AOs",
                      density.n(), basis.nao_sphe());
  }
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("coulomb_gradient: the basis carries no per-atom shell map (make it with "
                      "place)");
  }
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau >= 0.0)) {
    return err_config("coulomb_gradient: the screening threshold tau is {}, but must be 0 or "
                      "above",
                      tau);
  }
  for (const PrimitivePairClass &cls : classes) {
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > device::kCoulombMaxMomentum) {
      return err_config("coulomb_gradient: no kernel for class ({}, {}); the kernels cover "
                        "la <= lb <= {}",
                        cls.la, cls.lb, device::kCoulombMaxMomentum);
    }
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  FBF_TRY(check_one_electron_build(res, classes, pair.cartesian(), "coulomb_gradient"));
  std::vector<double> g(static_cast<size_t>(n_atom) * 3);
  if (density.n() == 0 || classes.empty()) {
    return g;
  }
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(pair.spherical().data(), density.data(),
                                         sizeof(double) * density.num_elements(),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(pair.spherical_to_cartesian(res));

  // J's classes over every record: the prune bounds the energy's terms, not
  // their derivatives, so nothing is pruned (tau = 0 keeps every record, in
  // mmd's order, so classes[k] and j_classes[k] are one class).
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, classes, 0.0));
  if (j_classes.size() != classes.size()) {
    return err_config("coulomb_gradient: {} of mmd's {} classes kept no record",
                      classes.size() - j_classes.size(), classes.size());
  }
  CoulombGradientSlices at;
  for (size_t c = 0; c < classes.size(); ++c) {
    const auto n_rec = static_cast<size_t>(j_classes[c].n_rec);
    if (j_classes[c].n_rec != classes[c].n_rec) {
      return err_config("coulomb_gradient: class ({}, {}) kept {} of its {} records", classes[c].la,
                        classes[c].lb, j_classes[c].n_rec, classes[c].n_rec);
    }
    at.jp.push_back(at.jp.back() +
                    n_rec * coulomb_hermite_count(classes[c].la + classes[c].lb + 1));
    at.atoms.push_back(at.atoms.back() + 2 * n_rec);
  }

  // Each Cartesian AO's atom, from the basis's per-atom shell map.
  std::vector<int32_t> ao_atom(basis.nao_cart());
  for (uint32_t atom = 0; atom < n_atom; ++atom) {
    const uint32_t first = basis.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + basis.atom_shell_count(atom); ++shell) {
      std::fill_n(ao_atom.begin() + basis.ao_offset_cart(shell),
                  basis.shells()[shell].function_count_cart(), static_cast<int32_t>(atom));
    }
  }
  // The call's scratch, the workspace's: nothing below takes it again.
  const auto [jp, g_d, atoms, ao_atom_d] = res.device_regions<double, double, int32_t, int32_t>(
      at.jp.back(), g.size(), at.atoms.back(), ao_atom.size());
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(ao_atom_d, ao_atom.data(), sizeof(int32_t) * ao_atom.size(),
                                     wwr::wwrMemcpyHostToDevice, stream)));
  // e_hermite and gjab only add, from every class's stream: zeroed on stream,
  // before the fork.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(jp, 0, sizeof(double) * at.jp.back(), stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g_d, 0, sizeof(double) * g.size(), stream)));

  const Status built =
      coulomb_gradient_build(res, classes, j_classes, pair.cartesian(), tau,
                             static_cast<int32_t>(n_atom), at, jp, ao_atom_d, atoms, g_d);
  // As coulomb's closing fork: a ket's stream read every bra's records, so a
  // class's frees, queued on its own stream when j_classes goes, must follow
  // every class's stream, which stream has joined.
  for (const CoulombPairClass &cls : j_classes) {
    FBF_TRY(gpu_status(cls.device_stream()->fork_stream().fork_from(stream)));
  }
  FBF_TRY(built);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), g_d, sizeof(double) * g.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what keeps ao_atom, pageable, alive until its upload has read it.
  FBF_TRY(res.sync());
  return g;
}

} // namespace free_breakfast
