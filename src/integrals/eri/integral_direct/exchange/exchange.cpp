/// @file
/// @brief The K build.

module;

// The launchers, defined in exchange_launch.cu and the helpers' .cu: declared
// in the GMF so they keep external linkage (see the headers).
#include "exchange_bridge.h"
#include "exchange_helpers_bridge.h"

module free_breakfast.integrals.exchange;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.exchange.pairs;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// One extent per angular momentum the kernels cover.
constexpr auto kExchangeMomenta = static_cast<size_t>(kMaxAngularMomentum + 1);

int32_t exchange_cartesian_count(const int32_t l) {
  return (l + 1) * (l + 2) / 2;
}

// A shell the classes reach, as their groups name it: its momentum and first
// Cartesian AO. momentum < 0 for an index no group names.
struct ExchangeShell {
  int32_t momentum = -1;
  int32_t offset = 0;
};

// One class's scratch: as a density class (lc, ld), its Dcd gather; as a bra
// or a ket, its group geometry (kExchangeGroupFields x n_groups); as a bra, its
// Dcd rows per ld (kExchangeMomenta x n_rec), and its kb block tables, one per
// ket class in class order, each kExchangeBlockFields x n_blk, with their
// blocks' density factors beside them, n_blk each.
struct ExchangeScratch {
  Buffers::device<double> dcd;
  Buffers::device<double> geom;
  Buffers::device<int32_t> dc;
  Buffers::device<int32_t> blocks;
  Buffers::device<double> factor;
};

// The blocks of one kb launch of @p bra against @p ket: one per (a group, b
// group), or with @p symmetric only those of a <= b. The groups are in basis
// order, so a shell's AO offset, which exchange_blocks compares on the device,
// rises with its index, which this compares.
int64_t exchange_block_count(const OrderedPairClass &bra, const OrderedPairClass &ket,
                             const bool symmetric) {
  if (!symmetric) {
    return static_cast<int64_t>(bra.groups.size()) * static_cast<int64_t>(ket.groups.size());
  }
  int64_t n = 0;
  for (const OrderedPairGroup &a : bra.groups) {
    n += std::ranges::end(ket.groups) -
         std::ranges::partition_point(
             ket.groups, [&a](const OrderedPairGroup &b) { return b.shell < a.shell; });
  }
  return n;
}

// The build of exchange(), or with @p k null everything but K itself (the
// zeroing, the kb launches and the mirror) and, into @p factors, every block's
// density factor, bra class major, then ket class, then each table's order
// (exchange_block_factors). Reads only what exchange() reads, so the factors
// are the ones its kb launches are given.
Status exchange_build(DeviceResources &res, const std::span<const OrderedPairClass> classes,
                      const CartesianMatrixAO &density, CartesianMatrixAO *k, const double tau,
                      const bool symmetric, std::vector<double> *factors) {
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
  for (const OrderedPairClass &cls : classes) {
    if (const int32_t dev = cls.device_stream()->dev_idx(); dev != res.dev_idx()) {
      return err_config("exchange: class ({}, {}) is on device {}, but the call passed device {}",
                        cls.la, cls.lb, dev, res.dev_idx());
    }
    if (cls.la < 0 || cls.la > kMaxAngularMomentum || cls.lb < 0 || cls.lb > kMaxAngularMomentum) {
      return err_config("exchange: no kernel for class ({}, {}); the kernels cover la, lb <= {}",
                        cls.la, cls.lb, kMaxAngularMomentum);
    }
  }
  if (n == 0) {
    return {};
  }
  const auto n_ao = static_cast<int32_t>(n);

  // The shells, read off the host's groups, and the classes by momenta.
  // Every shell of momentum l pairs with itself, so the class (l, l) names
  // every shell of l, and every (lc, ld) of two reached momenta is reached.
  std::vector<ExchangeShell> shells;
  std::array<int32_t, kExchangeMomenta> n_shells{};
  std::array<std::array<const OrderedPairClass *, kExchangeMomenta>, kExchangeMomenta> by_momenta{};
  for (const OrderedPairClass &cls : classes) {
    by_momenta[static_cast<size_t>(cls.la)][static_cast<size_t>(cls.lb)] = &cls;
    for (const OrderedPairGroup &g : cls.groups) {
      const auto at = static_cast<size_t>(g.shell);
      if (at >= shells.size()) {
        shells.resize(at + 1);
      }
      ExchangeShell &sh = shells[at];
      if (sh.momentum < 0) {
        sh = {cls.la, g.ao_offset};
        ++n_shells[static_cast<size_t>(cls.la)];
        if (sh.offset < 0 || sh.offset + exchange_cartesian_count(sh.momentum) > n_ao) {
          return err_config("exchange: shell {} reaches AO {}, but K is {} x {}", g.shell,
                            sh.offset + exchange_cartesian_count(sh.momentum), n, n);
        }
      }
    }
  }

  // Every launch's block count, bra-major; kb and the table index a block in
  // 32 bits.
  const size_t n_classes = classes.size();
  std::vector<int32_t> n_blk(n_classes * n_classes);
  for (size_t a = 0; a < n_classes; ++a) {
    for (size_t b = 0; b < n_classes; ++b) {
      const int64_t count = exchange_block_count(classes[a], classes[b], symmetric);
      if (count * device::kExchangeBlockFields > std::numeric_limits<int32_t>::max()) {
        return err_config("exchange: classes ({}, {}) and ({}, {}) make {} blocks, too many for "
                          "the kernels' 32-bit index",
                          classes[a].la, classes[a].lb, classes[b].la, classes[b].lb, count);
      }
      n_blk[a * n_classes + b] = static_cast<int32_t>(count);
    }
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

  // Drawn from res.device()'s pool in stream's order, so every class's fork
  // from stream sees them allocated, and returned in its order at the end,
  // after every phase's join.
  Buffers::device<double> decay(device::kExchangeDecayFields, res.device());
  Buffers::device<double> g_max(1, res.device());
  Buffers::device<double> centre(size_t{3} * n, res.device());
  Buffers::device<int32_t> starts(symmetric ? n : 1, res.device());
  std::vector<Buffers::device<int32_t>> offsets;
  offsets.reserve(kExchangeMomenta);
  for (const int32_t count : n_shells) {
    offsets.emplace_back(static_cast<size_t>(std::max(count, 1)), res.device());
  }
  std::vector<ExchangeScratch> scratch;
  scratch.reserve(n_classes);
  for (size_t a = 0; a < n_classes; ++a) {
    const OrderedPairClass &cls = classes[a];
    const auto n_rec = static_cast<size_t>(cls.n_rec);
    const auto n_dcd = static_cast<size_t>(n_shells[static_cast<size_t>(cls.la)]) *
                       static_cast<size_t>(n_shells[static_cast<size_t>(cls.lb)]) *
                       static_cast<size_t>(exchange_cartesian_count(cls.la)) *
                       static_cast<size_t>(exchange_cartesian_count(cls.lb));
    size_t n_blocks = 0;
    for (size_t b = 0; b < n_classes; ++b) {
      n_blocks += static_cast<size_t>(n_blk[a * n_classes + b]);
    }
    const size_t n_geom = static_cast<size_t>(device::kExchangeGroupFields) * cls.groups.size();
    scratch.push_back(ExchangeScratch{
        Buffers::device<double>(n_dcd, res.device()), Buffers::device<double>(n_geom, res.device()),
        Buffers::device<int32_t>(kExchangeMomenta * n_rec, res.device()),
        Buffers::device<int32_t>(std::max(n_blocks * device::kExchangeBlockFields, size_t{1}),
                                 res.device()),
        Buffers::device<double>(std::max(n_blocks, size_t{1}), res.device())});
  }
  const auto index_of = [&](const OrderedPairClass &cls) {
    return static_cast<size_t>(&cls - classes.data());
  };
  // The decay's max|D|, and the largest record bound's zero, which every class
  // folds into.
  FBF_TRY(gpu_status(device::exchange_density_max(stream, n_ao, density.data(), decay.data())));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g_max.data(), 0, sizeof(double), stream)));

  // Every class's largest bound into g_max; and from every class (l, l), its
  // momentum's rank-to-AO table and its AOs' centres. The gathers of every
  // (lc, ld) read two tables, the decay every centre and every geometry g_max,
  // so the join closes the phase.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    const wwr::wwrStream_t s = cls.device_stream()->stream();
    const OrderedPairClassView v = cls.view();
    if (const wwr::wwrError_t code =
            device::exchange_bound_max(s, v.n_rec, v.record_bound, g_max.data());
        code != wwr::wwrSuccess || cls.la != cls.lb) {
      return code;
    }
    if (const wwr::wwrError_t code =
            device::exchange_rank_offsets(s, v, offsets[static_cast<size_t>(cls.la)].data());
        code != wwr::wwrSuccess) {
      return code;
    }
    const std::array p{v.p_x, v.p_y, v.p_z};
    const std::array pa{v.pa_x, v.pa_y, v.pa_z};
    return device::exchange_ao_centres(s, v.n_groups, v.group_start,
                                       exchange_cartesian_count(cls.la), v.off_a, p.data(),
                                       pa.data(), n_ao, centre.data());
  })));
  // The density's decay with distance, from the centres: with max|D|, what
  // every block's factor is made from.
  FBF_TRY(gpu_status(
      device::exchange_density_decay(stream, n_ao, density.data(), centre.data(), decay.data())));
  // The first AO of each AO's shell, from the tables, for the mirror at the
  // end.
  if (symmetric) {
    for (size_t l = 0; l < kExchangeMomenta; ++l) {
      if (n_shells[l] > 0) {
        FBF_TRY(gpu_status(device::exchange_shell_starts(
            stream, static_cast<int32_t>(l), n_shells[l], offsets[l].data(), starts.data())));
      }
    }
  }

  // Every class's density block, Dcd of (lc, ld) = (la, lb), which every bra
  // class of second momentum lc reads, and its group geometry, which every
  // block table it is a side of reads: the join closes this phase too.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    const wwr::wwrStream_t s = cls.device_stream()->stream();
    const auto lc = static_cast<size_t>(cls.la);
    const auto ld = static_cast<size_t>(cls.lb);
    ExchangeScratch &c = scratch[index_of(cls)];
    if (const wwr::wwrError_t code = device::exchange_gather_dcd(
            s, cls.la, cls.lb, n_ao, density.data(), n_shells[lc], offsets[lc].data(), n_shells[ld],
            offsets[ld].data(), c.dcd.data());
        code != wwr::wwrSuccess) {
      return code;
    }
    const OrderedPairClassView v = cls.view();
    const std::array p{v.p_x, v.p_y, v.p_z};
    const std::array pa{v.pa_x, v.pa_y, v.pa_z};
    const std::array pb{v.pb_x, v.pb_y, v.pb_z};
    return device::exchange_group_geometry(s, v.n_groups, v.group_start, v.record_bound, p.data(),
                                           pa.data(), pb.data(), tau, g_max.data(), decay.data(),
                                           c.geom.data());
  })));

  // Every class as a bra (a, c): its Dcd rows, then for every ket class (b, d)
  // its block table and factors and one kb launch over them, in that order on
  // the bra's own stream. The symmetric sweep's tables hold the a <= b blocks
  // only.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    const size_t a = index_of(bra);
    ExchangeScratch &b = scratch[a];
    const OrderedPairClassView v = bra.view();
    const auto n_rec = static_cast<size_t>(bra.n_rec);
    for (size_t ld = 0; ld < kExchangeMomenta; ++ld) {
      if (n_shells[ld] == 0) {
        continue;
      }
      if (const wwr::wwrError_t code =
              device::exchange_bra_rows(s, v, n_shells[ld], b.dc.data() + ld * n_rec);
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    const auto lc = static_cast<size_t>(bra.lb);
    int32_t *table = b.blocks.data();
    double *factor = b.factor.data();
    for (const OrderedPairClass &ket : classes) {
      const int32_t count = n_blk[a * n_classes + index_of(ket)];
      if (count == 0) {
        continue;
      }
      const auto ld = static_cast<size_t>(ket.lb);
      const OrderedPairClass *density_class = by_momenta[lc][ld];
      if (density_class == nullptr) {
        return wwr::wwrErrorInvalidValue;
      }
      const OrderedPairClassView w = ket.view();
      if (const wwr::wwrError_t code = device::exchange_blocks(
              s, v.n_groups, v.group_start, v.off_a, b.geom.data(), w.n_groups, w.group_start,
              w.off_a, scratch[index_of(ket)].geom.data(), decay.data(), symmetric, count, table,
              factor);
          code != wwr::wwrSuccess) {
        return code;
      }
      if (k != nullptr) {
        if (const wwr::wwrError_t code =
                device::exchange_k(s, v, b.dc.data() + ld * n_rec, w, n_shells[lc] * n_shells[ld],
                                   tau, scratch[index_of(*density_class)].dcd.data(), count, table,
                                   factor, n_ao, k->data());
            code != wwr::wwrSuccess) {
          return code;
        }
      }
      table += static_cast<size_t>(count) * device::kExchangeBlockFields;
      factor += count;
    }
    return wwr::wwrSuccess;
  })));

  // The symmetric sweep computed the a <= b blocks: after the join, every
  // launch is in, and the a > b blocks are their transposes.
  if (symmetric && k != nullptr) {
    FBF_TRY(gpu_status(device::exchange_mirror(stream, n_ao, starts.data(), k->data())));
  }
  // Every bra class's factors, in class order, read back after the join.
  if (factors != nullptr) {
    factors->clear();
    for (size_t a = 0; a < n_classes; ++a) {
      size_t count = 0;
      for (size_t b = 0; b < n_classes; ++b) {
        count += static_cast<size_t>(n_blk[a * n_classes + b]);
      }
      const size_t at = factors->size();
      factors->resize(at + count);
      if (count > 0) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(factors->data() + at, scratch[a].factor.data(),
                                               sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                               stream)));
      }
    }
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));
  }
  return {};
}

Status exchange(DeviceResources &res, const std::span<const OrderedPairClass> classes,
                const CartesianMatrixAO &density, CartesianMatrixAO &k, const double tau,
                const bool symmetric) {
  return exchange_build(res, classes, density, &k, tau, symmetric, nullptr);
}

Result<std::vector<double>> exchange_block_factors(DeviceResources &res,
                                                   const std::span<const OrderedPairClass> classes,
                                                   const CartesianMatrixAO &density,
                                                   const double tau, const bool symmetric) {
  std::vector<double> factors;
  FBF_TRY(exchange_build(res, classes, density, nullptr, tau, symmetric, &factors));
  return factors;
}

} // namespace free_breakfast
