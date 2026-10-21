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

// One class's scratch: as a density class (lc, ld), its Dcd gather; as a bra,
// its Ip with max|D| folded in, its Dcd rows per ld (kExchangeMomenta x n_rec),
// and its (bra class, b) blocks for every shell b, n_rec x n_x(la) x n_ao in
// all, shell b's block at n_rec n_x(la) times b's AO offset (the symmetric
// sweep writes only the rows of a <= b, about half).
struct ExchangeScratch {
  Buffers::device<double> dcd;
  Buffers::device<double> ip;
  Buffers::device<int32_t> dc;
  Buffers::device<double> blocks;
};

// The records of @p bra whose first shell is at most @p shell: a prefix, the
// groups being in basis order and covering the records in order.
int32_t exchange_prefix(const OrderedPairClass &bra, const int32_t shell) {
  const auto past = std::ranges::partition_point(
      bra.groups, [shell](const OrderedPairGroup &g) { return g.shell <= shell; });
  return past == bra.groups.begin() ? 0 : std::prev(past)->start + std::prev(past)->count;
}

Status exchange(DeviceResources &res, const std::span<const OrderedPairClass> classes,
                const CartesianMatrixAO &density, CartesianMatrixAO &k, const double tau,
                const bool symmetric) {
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau >= 0.0)) {
    return err_config("exchange: the screening threshold tau is {}, but must be 0 or above", tau);
  }
  if (k.dev_idx() != res.dev_idx()) {
    return err_config("exchange: K is on device {}, but the call passed device {}", k.dev_idx(),
                      res.dev_idx());
  }
  if (density.dev_idx() != res.dev_idx()) {
    return err_config("exchange: the density is on device {}, but the call passed device {}",
                      density.dev_idx(), res.dev_idx());
  }
  if (density.n() != k.n()) {
    return err_config("exchange: the density is {0} x {0}, but K is {1} x {1}", density.n(), k.n());
  }
  const uint32_t n = k.n();
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

  // The shells, read off the groups' first records on the host (the pinned
  // side holds what was copied to the device), and the classes by momenta.
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
        sh = {cls.la, cls.off_a.pinned().data()[g.start]};
        ++n_shells[static_cast<size_t>(cls.la)];
        if (sh.offset < 0 || sh.offset + exchange_cartesian_count(sh.momentum) > n_ao) {
          return err_config("exchange: shell {} reaches AO {}, but K is {} x {}", g.shell,
                            sh.offset + exchange_cartesian_count(sh.momentum), n, n);
        }
      }
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The scatter only adds, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(k.data(), 0, sizeof(double) * k.num_elements(), stream)));
  if (classes.empty()) {
    return {};
  }

  // Drawn from res.device()'s pool in stream's order, so every class's fork
  // from stream sees them allocated, and returned in its order at the end,
  // after every phase's join.
  Buffers::device<double> dmax(1, res.device());
  std::vector<Buffers::device<int32_t>> offsets;
  offsets.reserve(kExchangeMomenta);
  for (const int32_t count : n_shells) {
    offsets.emplace_back(static_cast<size_t>(std::max(count, 1)), res.device());
  }
  std::vector<ExchangeScratch> scratch;
  scratch.reserve(classes.size());
  for (const OrderedPairClass &cls : classes) {
    const auto n_rec = static_cast<size_t>(cls.n_rec);
    const auto n_dcd = static_cast<size_t>(n_shells[static_cast<size_t>(cls.la)]) *
                       static_cast<size_t>(n_shells[static_cast<size_t>(cls.lb)]) *
                       static_cast<size_t>(exchange_cartesian_count(cls.la)) *
                       static_cast<size_t>(exchange_cartesian_count(cls.lb));
    const size_t n_blocks = n_rec * static_cast<size_t>(exchange_cartesian_count(cls.la)) * n;
    ExchangeScratch &sc = scratch.emplace_back(ExchangeScratch{
        Buffers::device<double>(n_dcd, res.device()), Buffers::device<double>(n_rec, res.device()),
        Buffers::device<int32_t>(kExchangeMomenta * n_rec, res.device()),
        Buffers::device<double>(n_blocks, res.device())});
    // k adds every ket class and group of a b shell into its block.
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(sc.blocks.data(), 0, sizeof(double) * n_blocks, stream)));
  }
  const auto of = [&](const OrderedPairClass &cls) -> ExchangeScratch & {
    return scratch[static_cast<size_t>(&cls - classes.data())];
  };
  FBF_TRY(gpu_status(device::exchange_density_max(stream, n_ao, density.data(), dmax.data())));

  // Every momentum's rank-to-AO table, from its class (l, l): the gathers of
  // every (lc, ld) read two of them, so the join closes the phase.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    if (cls.la != cls.lb) {
      return wwr::wwrSuccess;
    }
    return device::exchange_rank_offsets(cls.device_stream()->stream(), cls.view(),
                                         offsets[static_cast<size_t>(cls.la)].data());
  })));

  // Every class's density block, Dcd of (lc, ld) = (la, lb), which every bra
  // class of second momentum lc reads: the join closes this phase too.
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &cls) {
    const auto lc = static_cast<size_t>(cls.la);
    const auto ld = static_cast<size_t>(cls.lb);
    return device::exchange_gather_dcd(cls.device_stream()->stream(), cls.la, cls.lb, n_ao,
                                       density.data(), n_shells[lc], offsets[lc].data(),
                                       n_shells[ld], offsets[ld].data(), of(cls).dcd.data());
  })));

  // Every class as a bra (a, c): its inputs, then k against every ket class
  // (b, d) and every b group of it, then its blocks scattered into K, in that
  // order on the bra's own stream. The symmetric sweep cuts each launch and
  // scatter to the bra's records with a <= b, and the scatter mirrors a < b.
  return gpu_status(launch_each(classes, stream, [&](const OrderedPairClass &bra) {
    const wwr::wwrStream_t s = bra.device_stream()->stream();
    ExchangeScratch &b = of(bra);
    const OrderedPairClassView v = bra.view();
    const auto n_rec = static_cast<size_t>(bra.n_rec);
    const size_t row = n_rec * static_cast<size_t>(exchange_cartesian_count(bra.la));
    const auto n_bra = [&](const int32_t shell) {
      return symmetric ? exchange_prefix(bra, shell) : bra.n_rec;
    };
    if (const wwr::wwrError_t code = device::exchange_bra_bound(s, v, dmax.data(), b.ip.data());
        code != wwr::wwrSuccess) {
      return code;
    }
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
    for (const OrderedPairClass &ket : classes) {
      const auto ld = static_cast<size_t>(ket.lb);
      const OrderedPairClass *density_class = by_momenta[lc][ld];
      if (density_class == nullptr) {
        return wwr::wwrErrorInvalidValue;
      }
      const double *dcd = of(*density_class).dcd.data();
      const int32_t n_dc = n_shells[lc] * n_shells[ld];
      const OrderedPairClassView w = ket.view();
      for (const OrderedPairGroup &g : ket.groups) {
        const int32_t cut = n_bra(g.shell);
        if (cut == 0) {
          continue;
        }
        const auto off_b = static_cast<size_t>(shells[static_cast<size_t>(g.shell)].offset);
        if (const wwr::wwrError_t code =
                device::exchange_k(s, v, cut, b.ip.data(), b.dc.data() + ld * n_rec, w, g.start,
                                   g.count, n_dc, tau, dcd, b.blocks.data() + row * off_b);
            code != wwr::wwrSuccess) {
          return code;
        }
      }
    }
    for (size_t shell = 0; shell < shells.size(); ++shell) {
      const ExchangeShell &sh = shells[shell];
      const int32_t cut = n_bra(static_cast<int32_t>(shell));
      if (sh.momentum < 0 || cut == 0) {
        continue;
      }
      const auto off_b = static_cast<size_t>(sh.offset);
      if (const wwr::wwrError_t code =
              device::exchange_scatter_block(s, bra.la, sh.momentum, n_ao, cut, v.off_a, sh.offset,
                                             symmetric, b.blocks.data() + row * off_b, k.data());
          code != wwr::wwrSuccess) {
        return code;
      }
    }
    return wwr::wwrSuccess;
  }));
}

} // namespace free_breakfast
