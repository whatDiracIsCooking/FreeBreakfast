/// @file
/// @brief The K build.

module;

// The launchers, defined in exchange_launch.cu and the helpers' .cu: declared
// in the GMF so they keep external linkage (see the headers).
#include "exchange_bridge.h"
#include "exchange_helpers_bridge.h"

module free_breakfast.integrals.eri.integral_direct.exchange;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.eri.integral_direct.exchange.pairs;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// One extent per angular momentum the kernels cover: K's own limit, not the
// tree-wide kMaxAngularMomentum, though the two are kept equal.
constexpr auto kExchangeMomenta = static_cast<size_t>(device::kExchangeMaxMomentum + 1);
static_assert(device::kExchangeMaxMomentum == kMaxAngularMomentum,
              "K's limit and the tree-wide kMaxAngularMomentum disagree");

int32_t exchange_cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

// What exchange() and exchange_block_factors() check before they queue
// anything: the threshold, the devices, the orders, and that @p classes are
// one whole set of ordered_pair_classes, in its order, of a basis of the
// density's order. @p k may be null.
Status exchange_check(const DeviceResources &res, const std::span<OrderedPairClass> classes,
                      const CartesianMatrixAO &density, const CartesianMatrixAO *k,
                      const double tau) {
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau >= 0.0)) {
    return err_config("exchange: the screening threshold tau is {}, but must be 0 or above", tau);
  }
  if (k != nullptr && k->dev_idx() != res.dev_idx()) {
    return err_config("exchange: K is on device {}, but the call passed device {}", k->dev_idx(),
                      res.dev_idx());
  }
  if (density.dev_idx() != res.dev_idx()) {
    return err_config("exchange: the density is on device {}, but the call passed device {}",
                      density.dev_idx(), res.dev_idx());
  }
  if (k != nullptr && density.n() != k->n()) {
    return err_config("exchange: the density is {0} x {0}, but K is {1} x {1}", density.n(),
                      k->n());
  }
  const uint32_t n = density.n();
  // The kernels and the helpers index the matrices as a 32-bit m * n_ao + n.
  if (static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("exchange: {} Cartesian AOs is too many for the kernels' 32-bit index", n);
  }
  for (size_t c = 0; c < classes.size(); ++c) {
    const OrderedPairClass &cls = classes[c];
    if (const int32_t dev = cls.device_stream()->dev_idx(); dev != res.dev_idx()) {
      return err_config("exchange: class ({}, {}) is on device {}, but the call passed device {}",
                        cls.la, cls.lb, dev, res.dev_idx());
    }
    // K's limit, the kernels' tables' (exchange_launch.cu checks the two
    // agree), and what kExchangeMomenta sizes the build's storage by.
    if (cls.la < 0 || cls.la > device::kExchangeMaxMomentum || cls.lb < 0 ||
        cls.lb > device::kExchangeMaxMomentum) {
      return err_config("exchange: no kernel for class ({}, {}); the kernels cover la, lb <= {}",
                        cls.la, cls.lb, device::kExchangeMaxMomentum);
    }
    if (static_cast<uint32_t>(cls.n_ao) != n) {
      return err_config("exchange: class ({}, {}) is of a basis of {} Cartesian AOs, but the "
                        "density is {} x {}",
                        cls.la, cls.lb, cls.n_ao, n, n);
    }
    // A bra's tables are sized against every class of its set, in its order:
    // the whole set, ascending (la, lb), is the only span they fit.
    if (cls.n_blk.size() != classes.size() ||
        (c > 0 && std::pair{classes[c - 1].la, classes[c - 1].lb} >= std::pair{cls.la, cls.lb})) {
      return err_config("exchange: class ({}, {}) is one of a set of {}, but the call passed {} "
                        "classes, or not in the set's order",
                        cls.la, cls.lb, cls.n_blk.size(), classes.size());
    }
  }
  return {};
}

// The build of exchange(), or with @p k null everything but K itself (the
// zeroing, the kb launches and the mirror) and, into @p factors, every block's
// density factor, bra class major, then ket class, then each table's order
// (exchange_block_factors). Reads only what exchange() reads, so the factors
// are the ones its kb launches are given. @p sign is the mirror's.
Status exchange_build(DeviceResources &res, const std::span<OrderedPairClass> classes,
                      const CartesianMatrixAO &density, CartesianMatrixAO *k, const double tau,
                      const double sign, std::vector<double> *factors) {
  const auto n_ao = static_cast<int32_t>(density.n());
  const wwr::wwrStream_t stream = res.stream();
  // The basis's shells by momentum, which every class of the set carries, and
  // the classes by momenta. Every shell of momentum l pairs with itself, so
  // the class (l, l) names every shell of l, and every (lc, ld) of two reached
  // momenta is reached.
  const std::vector<int32_t> &n_shells = classes.front().n_shells;
  std::array<std::array<const OrderedPairClass *, kExchangeMomenta>, kExchangeMomenta> by_momenta{};
  for (const OrderedPairClass &cls : classes) {
    by_momenta[static_cast<size_t>(cls.la)][static_cast<size_t>(cls.lb)] = &cls;
  }
  // The rank table of every momentum l the basis reaches, the class (l, l)'s,
  // which that class writes and the classes of l read.
  std::array<int32_t *, kExchangeMomenta> offsets{};
  for (OrderedPairClass &cls : classes) {
    if (cls.la == cls.lb) {
      offsets[static_cast<size_t>(cls.la)] = cls.rank_offset.data();
    }
  }

  // The set's own scratch, its first class's, allocated with the set: written
  // on stream and every class's stream between the forks and the joins, as
  // the classes' own is.
  OrderedPairClass &set = classes.front();
  double *const decay = set.decay.data();
  double *const g_max = set.g_max.data();
  double *const centre = set.centre.data();
  int32_t *const starts = set.shell_start.data();
  // launch_each hands each class over const; the build writes its scratch.
  const auto index_of = [&](const OrderedPairClass &cls) {
    return static_cast<size_t>(&cls - classes.data());
  };
  const auto own = [&](const OrderedPairClass &cls) -> OrderedPairClass & {
    return classes[index_of(cls)];
  };
  // The decay's max|D|, and the largest record bound's zero, which every class
  // folds into.
  FBF_TRY(gpu_status(device::exchange_density_max(stream, n_ao, density.data(), decay)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g_max, 0, sizeof(double), stream)));

  // Every class's largest bound into g_max; and from every class (l, l), its
  // momentum's rank-to-AO table and its AOs' centres. The gathers of every
  // (lc, ld) read two tables, the decay every centre and every geometry g_max,
  // so the join closes the phase.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    const wwr::wwrStream_t s = cls.device_stream()->stream();
    const OrderedPairClassView v = cls.view();
    if (const wwr::wwrError_t code = device::exchange_bound_max(s, v.n_rec, v.record_bound, g_max);
        code != wwr::wwrSuccess || cls.la != cls.lb) {
      return code;
    }
    if (const wwr::wwrError_t code =
            device::exchange_rank_offsets(s, v, offsets[static_cast<size_t>(cls.la)]);
        code != wwr::wwrSuccess) {
      return code;
    }
    const std::array p{v.p_x, v.p_y, v.p_z};
    const std::array pa{v.pa_x, v.pa_y, v.pa_z};
    return device::exchange_ao_centres(s, v.n_groups, v.group_start,
                                       exchange_cartesian_count(cls.la), v.off_a, p.data(),
                                       pa.data(), n_ao, centre);
  })));
  // The density's decay with distance, from the centres: with max|D|, what
  // every block's factor is made from.
  FBF_TRY(gpu_status(device::exchange_density_decay(stream, n_ao, density.data(), centre, decay)));
  // The first AO of each AO's shell, from the tables, for the mirror at the
  // end.
  if (k != nullptr) {
    for (size_t l = 0; l < kExchangeMomenta; ++l) {
      if (n_shells[l] > 0) {
        FBF_TRY(gpu_status(device::exchange_shell_starts(stream, static_cast<int32_t>(l),
                                                         n_shells[l], offsets[l], starts)));
      }
    }
  }

  // Every class's density block, Dcd of (lc, ld) = (la, lb), which every bra
  // class of second momentum lc reads, and its group geometry, which every
  // factor of a table it is a side of reads: the join closes this phase too.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    OrderedPairClass &c = own(cls);
    const wwr::wwrStream_t s = c.device_stream()->stream();
    const auto lc = static_cast<size_t>(c.la);
    const auto ld = static_cast<size_t>(c.lb);
    if (const wwr::wwrError_t code =
            device::exchange_gather_dcd(s, c.la, c.lb, n_ao, density.data(), n_shells[lc],
                                        offsets[lc], n_shells[ld], offsets[ld], c.dcd.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    const OrderedPairClassView v = c.view();
    const std::array p{v.p_x, v.p_y, v.p_z};
    const std::array pa{v.pa_x, v.pa_y, v.pa_z};
    const std::array pb{v.pb_x, v.pb_y, v.pb_z};
    return device::exchange_group_geometry(s, v.n_groups, v.group_start, v.record_bound, p.data(),
                                           pa.data(), pb.data(), tau, g_max, decay, c.geom.data());
  })));

  // Every class as a bra (a, c): for every ket class (b, d), its table's
  // factors and one kb launch over the table, in that order on the bra's own
  // stream. The tables, made once per basis, hold the a <= b blocks.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    OrderedPairClass &bra = own(cls);
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    const OrderedPairClassView v = bra.view();
    const auto n_rec = static_cast<size_t>(bra.n_rec);
    const auto lc = static_cast<size_t>(bra.lb);
    const int32_t *table = bra.blocks.data();
    double *factor = bra.factor.data();
    for (size_t kk = 0; kk < classes.size(); ++kk) {
      const int32_t count = bra.n_blk[kk];
      if (count == 0) {
        continue;
      }
      const OrderedPairClass &ket = classes[kk];
      const auto ld = static_cast<size_t>(ket.lb);
      const OrderedPairClass *density_class = by_momenta[lc][ld];
      if (density_class == nullptr) {
        return wwr::wwrErrorInvalidValue;
      }
      const OrderedPairClassView w = ket.view();
      if (const wwr::wwrError_t code = device::exchange_factors(
              s, v.n_groups, v.group_start, v.off_a, bra.geom.data(), w.n_groups, w.group_start,
              w.off_a, ket.geom.data(), decay, count, factor);
          code != wwr::wwrSuccess) {
        return code;
      }
      if (k != nullptr) {
        if (const wwr::wwrError_t code = device::exchange_k(
                s, v, bra.dc.data() + ld * n_rec, w, n_shells[lc] * n_shells[ld], tau,
                density_class->dcd.data(), count, table, factor, n_ao, k->data());
            code != wwr::wwrSuccess) {
          return code;
        }
      }
      table += static_cast<size_t>(count) * device::kExchangeBlockFields;
      factor += count;
    }
    return wwr::wwrSuccess;
  })));

  // The sweep computed the a <= b blocks: after the join, every launch is in,
  // and the a > b blocks are their transposes, times the sign.
  if (k != nullptr) {
    FBF_TRY(gpu_status(device::exchange_mirror(stream, n_ao, starts, sign, k->data())));
  }
  // Every bra class's factors, in class order, read back after the join.
  if (factors != nullptr) {
    factors->clear();
    for (const OrderedPairClass &bra : classes) {
      size_t count = 0;
      for (const int32_t c : bra.n_blk) {
        count += static_cast<size_t>(c);
      }
      const size_t at = factors->size();
      factors->resize(at + count);
      if (count > 0) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(factors->data() + at, bra.factor.data(),
                                               sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                               stream)));
      }
    }
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));
  }
  return {};
}

// exchange() and exchange_block_factors(): the checks, K's zeroing, the build,
// and the classes' streams forked from res.stream() at the end.
Status exchange_run(DeviceResources &res, const std::span<OrderedPairClass> classes,
                    const CartesianMatrixAO &density, CartesianMatrixAO *k, const double tau,
                    const double sign, std::vector<double> *factors) {
  FBF_TRY(exchange_check(res, classes, density, k, tau));
  if (factors != nullptr) {
    factors->clear();
  }
  if (density.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // kb only adds, so the build starts from zero.
  if (k != nullptr) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(k->data(), 0, sizeof(double) * k->num_elements(), stream)));
  }
  if (classes.empty()) {
    return {};
  }
  const Status built = exchange_build(res, classes, density, k, tau, sign, factors);
  // A bra's stream reads every ket's records, geometry and Dcd, and a failed
  // phase leaves some streams unforked by the next, so every class's stream
  // waits on stream, which has joined them all, before anything else is
  // queued there: a class's frees, and the next build's scratch writes, then
  // follow every read of it.
  for (const OrderedPairClass &cls : classes) {
    FBF_TRY(gpu_status(cls.device_stream()->fork_stream().fork_from(stream)));
  }
  return built;
}

Status exchange(DeviceResources &res, const std::span<OrderedPairClass> classes,
                const CartesianMatrixAO &density, CartesianMatrixAO &k, const double tau,
                const DensitySymmetry symmetry) {
  const double sign = symmetry == DensitySymmetry::Antisymmetric ? -1.0 : 1.0;
  return exchange_run(res, classes, density, &k, tau, sign, nullptr);
}

Result<std::vector<double>> exchange_block_factors(DeviceResources &res,
                                                   const std::span<OrderedPairClass> classes,
                                                   const CartesianMatrixAO &density,
                                                   const double tau) {
  std::vector<double> factors;
  FBF_TRY(exchange_run(res, classes, density, nullptr, tau, 1.0, &factors));
  return factors;
}

Result<std::vector<double>> exchange_gradient(DeviceResources &res, const BasisSet &basis,
                                              const std::span<OrderedPairClass> classes,
                                              const SphericalMatrixAO &density) {
  if (res.dev_idx() != density.dev_idx()) {
    return err_config("exchange_gradient: the density is on device {}, but the call passed "
                      "device {}",
                      density.dev_idx(), res.dev_idx());
  }
  if (density.n() != basis.nao_sphe()) {
    return err_config("exchange_gradient: the density is {} square, but the basis has {} "
                      "spherical AOs",
                      density.n(), basis.nao_sphe());
  }
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("exchange_gradient: the basis carries no per-atom shell map (make it with "
                      "place)");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  // exchange's own checks, on the Cartesian density the kernels read; no K
  // and no screen.
  FBF_TRY(exchange_check(res, classes, pair.cartesian(), nullptr, 0.0));
  std::vector<double> g(static_cast<size_t>(n_atom) * 3);
  if (density.n() == 0 || classes.empty()) {
    return g;
  }
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(pair.spherical().data(), density.data(),
                                         sizeof(double) * density.num_elements(),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(pair.spherical_to_cartesian(res));

  // Each Cartesian AO's atom, from the basis's per-atom shell map, and from
  // it each bra record's two atoms, which the classes do not carry: a class's
  // slice of the scratch is its first shells' atoms, then its second's.
  std::vector<int32_t> ao_atom(basis.nao_cart());
  for (uint32_t atom = 0; atom < n_atom; ++atom) {
    const uint32_t first = basis.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + basis.atom_shell_count(atom); ++shell) {
      std::fill_n(ao_atom.begin() + basis.ao_offset_cart(shell),
                  basis.shells()[shell].function_count_cart(), static_cast<int32_t>(atom));
    }
  }
  std::vector<size_t> atoms_begin{0};
  for (const OrderedPairClass &cls : classes) {
    atoms_begin.push_back(atoms_begin.back() + (2 * static_cast<size_t>(cls.n_rec)));
  }
  Buffers::device<int32_t> ao_atom_d(ao_atom.size(), res.device());
  Buffers::device<int32_t> atoms(atoms_begin.back(), res.device());
  Buffers::device<double> g_d(g.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ao_atom_d.data(), ao_atom.data(),
                                         sizeof(int32_t) * ao_atom.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g_d.data(), 0, sizeof(double) * g.size(), stream)));
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const double *m = pair.cartesian().data();
  // Every class as a bra (a, c) on its own stream: its record atoms, then one
  // gk launch against every class of the set as a ket (b, d).
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    const OrderedPairClassView v = bra.view();
    int32_t *const atom_a = atoms.data() + atoms_begin[static_cast<size_t>(&bra - classes.data())];
    int32_t *const atom_c = atom_a + bra.n_rec;
    if (const wwr::wwrError_t code =
            device::exchange_record_atoms(s, v.n_rec, v.off_a, ao_atom_d.data(), atom_a);
        code != wwr::wwrSuccess) {
      return code;
    }
    if (const wwr::wwrError_t code =
            device::exchange_record_atoms(s, v.n_rec, v.off_b, ao_atom_d.data(), atom_c);
        code != wwr::wwrSuccess) {
      return code;
    }
    for (const OrderedPairClass &ket : classes) {
      if (const wwr::wwrError_t code = device::exchange_gradient_class(
              s, v, atom_a, atom_c, ket.view(), n_ao, static_cast<int32_t>(n_atom), m, g_d.data());
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return wwr::wwrSuccess;
  })));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), g_d.data(), sizeof(double) * g.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what keeps ao_atom, pageable, alive until its upload has read it.
  FBF_TRY(res.sync());
  return g;
}

} // namespace free_breakfast
