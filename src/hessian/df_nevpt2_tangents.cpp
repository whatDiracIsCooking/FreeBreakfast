/// @file
/// @brief df_nevpt2_tangents: the NEVPT2 parameter response x^x (lambda^x's
/// kappa_nr and c^x, and the closed-form kappa_oo^x) and the total tangent
/// t_x of (h_MO, B_MO, c), per chunk of mcscf::df_integral_derivatives'
/// derivative integrals: the one-index transforms by kappa as gemms over the
/// MO factor, f's inactive blocks' change by nevpt2::builders_tangent, and
/// the degenerate-pair rule on its division.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, geam, gemmStridedBatched, wwrMemcpyAsync,
                                // ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, OrbitalRange
import free_breakfast.recipes.blas;                     // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.response.orbital_pairs; // unpack_orbital_pairs
import free_breakfast.mcscf.solve;
import free_breakfast.response.zvector;
import free_breakfast.mcscf.third_derivative; // df_integral_derivatives
import free_breakfast.nevpt2.gradient;        // builders_tangent
import :state_specific;                       // state_specific_electrons, Ledger
import :atom_block;                           // atom_function_range

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The NEVPT2 gradient's degenerate-pair rule (nevpt2/gradient/zc.cpp's
// kZcGapFloor and kZcZeroSource, the demo's PC_GAP_FLOOR and PC_ZERO_TOL):
// below this gap (Eh) a core or virtual pair's division is not one the state
// determines, ...
constexpr double kTangentGapFloor = 1e-6;
// ... and zero stands for it only while the numerator is round-off, as it is
// for a degenerate pair an abelian subgroup splits across irreps.
constexpr double kTangentZeroSource = 1e-8;
// The largest off-diagonal |f_pq| over the core or the virtuals of orbitals
// still canonical: kappa_oo's division is the stationarity of f_pq = 0 at
// first order only where f is diagonal. A canonicalized CASSCF's sits at
// round-off (1e-14).
constexpr double kTangentCanonicalTolerance = 1e-8;

Result<std::vector<DegeneratePair>> df_nevpt2_tangents(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
    const response::ZVectors &cp, const bool pc,
    const std::function<Status(int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
                               const double *c_dot, const double *kappa)> &consume,
    const size_t max_bytes) {
  constexpr std::string_view kWho = "hessian::df_nevpt2_tangents";
  const auto [nalpha, nbeta] =
      FBF_TRY(state_specific_electrons(kWho, res, molecule, basis, aux, b, solution, ncore, ncas));
  if (!consume) {
    return err_config("{}: no consumer", kWho);
  }
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nn = nmo * nmo;
  const size_t n_x = static_cast<size_t>(3) * basis.natoms();
  if (naux * nmo > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("{}: naux {} times nmo {} is past int32_t", kWho, naux, nmo);
  }
  const auto nmo32 = static_cast<int32_t>(nmo);
  const auto naux32 = static_cast<int32_t>(naux);
  const auto nn32 = static_cast<int32_t>(nn);
  const auto nc = static_cast<size_t>(ncore);
  const size_t nocc = nc + static_cast<size_t>(ncas);
  const size_t nv = nmo - nocc;
  const size_t pairs = (nc * (nmo - nc)) + (static_cast<size_t>(ncas) * nv);
  const ScopedDeviceIndex on{res.dev_idx()};

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  const auto ndet = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  if (solution.roots.vectors.num_elements() != ndet) {
    return err_config("{}: a root of {} doubles over {} determinants", kWho,
                      solution.roots.vectors.num_elements(), ndet);
  }
  if (cp.orbital.num_elements() != pairs * n_x || cp.ci.num_elements() != ndet * n_x) {
    return err_config(
        "{}: lambda^x of {} orbital and {} CI doubles, not {} columns of {} pairs and "
        "{} determinants",
        kWho, cp.orbital.num_elements(), cp.ci.num_elements(), n_x, pairs, ndet);
  }
  const wwr::wwrblasHandle_t blas = res.blas().get();

  // h_MO = C^T h C and the MO factor B_Q,pq = (C^T B_Q C)_pq at
  // Q + naux (p + nmo q), held across the calls below that take the
  // workspace, as is every buffer here.
  Buffers::device<double> h_mo(nn, res.device());
  Buffers::device<double> l_mo(naux * nn, res.device());
  {
    Buffers::device<double> work(nn, res.device());
    FBF_TRY(gpu_status(
        gemmwich(blas, nmo32, nmo32, solution.c.data(), h.data(), work.data(), h_mo.data())));
  }
  const OrbitalRange all{.first = 0, .count = nmo32};
  FBF_TRY(b_mo(res, naux32, nmo32, nmo32, b.data(), solution.c.data(), all, all, l_mo.data()));

  // The one-index transform by an antisymmetric @p kappa, onto one
  // coordinate's slabs: h_MO kappa + kappa^T h_MO, and per Q B_Q kappa (B as
  // the (naux nmo) x nmo matrix of rows (Q, p)) + kappa^T B_Q (per q, the
  // naux x nmo slab of rows Q and columns r times kappa_rp).
  const auto one_index = [&](const double *const kappa, double *const h_dot,
                             double *const b_dot) -> Status {
    FBF_TRY(gpu_status(gemm(blas, nmo32, &kOne<double>, h_mo.data(), kappa, &kOne<double>, h_dot)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, &kOne<double>, kappa, h_mo.data(),
                                               &kOne<double>, h_dot)));
    FBF_TRY(gpu_status(gemm(blas, naux32 * nmo32, nmo32, nmo32, &kOne<double>, l_mo.data(), kappa,
                            &kOne<double>, b_dot)));
    return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux32, nmo32, nmo32, &kOne<double>,
        l_mo.data(), naux32, static_cast<int64_t>(naux * nmo), kappa, nmo32, 0, &kOne<double>,
        b_dot, naux32, static_cast<int64_t>(naux * nmo), nmo32));
  };
  const auto download = [&res](const double *const from,
                               const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    if (count > 0) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to.data(), from, count * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, res.stream())));
      FBF_TRY(res.sync());
    }
    return to;
  };

  std::vector<DegeneratePair> recorded;
  std::vector<double> eps;
  // PC's pairs taken by their off-diagonal f: disjoint, smallest gap first.
  std::vector<std::array<size_t, 2>> taken;
  // One chunk of @p count coordinates from x0, as wide as the budget holds.
  const auto part = [&](const int32_t x0, const int32_t count, const double *const h1,
                        const double *const w1) -> Status {
    const auto k = static_cast<size_t>(count);
    Buffers::device<double> h_dot(k * nn, res.device());
    Buffers::device<double> b_dot(k * naux * nn, res.device());
    Buffers::device<double> kappa(k * nn, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_dot.data(), h1, k * nn * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    const double *const c_dot = cp.ci.data() + (static_cast<size_t>(x0) * ndet);

    // Everything but kappa_oo: h1^x, W^x to B's layout (Q fastest), and the
    // one-index transform by kappa_nr^x.
    for (size_t j = 0; j < k; ++j) {
      const double *const w = w1 + (j * nn * naux);
      double *const bj = b_dot.data() + (j * naux * nn);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                    naux32, nn32, &kOne<double>, w, nn32,
                                                    &kZero<double>, w, naux32, bj, naux32)));
      FBF_TRY(response::unpack_orbital_pairs(
          res, nmo32, ncore, ncas, cp.orbital.data() + ((static_cast<size_t>(x0) + j) * pairs),
          kappa.data() + (j * nn)));
      FBF_TRY(one_index(kappa.data() + (j * nn), h_dot.data() + (j * nn), bj));
    }

    // f's inactive blocks at the point (eps their diagonals) and their change
    // along that.
    nevpt2::BuildersTangent tangent = FBF_TRY(nevpt2::builders_tangent(
        res, alpha, beta, naux32, nmo32, nmo32, b.data(), solution.c.data(), h.data(), ncore, ncas,
        solution.roots.vectors.data(), count, h_dot.data(), b_dot.data(), c_dot));
    if (eps.empty()) {
      eps.assign(nmo, 0.0);
      for (const auto &[f_dev, first, width] :
           {std::tuple{tangent.value.f_core.data(), size_t{0}, nc},
            std::tuple{tangent.value.f_virt.data(), nocc, nv}}) {
        const std::vector<double> f = FBF_TRY(download(f_dev, width * width));
        for (size_t q = 0; q < width; ++q) {
          eps[first + q] = f[q * (width + 1)];
          for (size_t p = 0; p < width; ++p) {
            if (p != q && std::abs(f[p + (width * q)]) > kTangentCanonicalTolerance) {
              return err_config("{}: f_{},{} is {:.2e} (above {:g}): the core and virtuals are "
                                "not canonical",
                                kWho, first + p, first + q, f[p + (width * q)],
                                kTangentCanonicalTolerance);
            }
          }
        }
      }
      if (pc) {
        taken = pc_taken_pairs(eps, nc, nocc, kPcPairReach);
      }
    }

    // kappa_oo^x_pq = -f^(x)_pq / (eps_p - eps_q) over each block, under the
    // degenerate-pair rule; then its one-index transform, and kappa the sum.
    std::vector<double> kappa_oo(k * nn, 0.0);
    for (size_t j = 0; j < k; ++j) {
      const auto x = static_cast<int32_t>(static_cast<size_t>(x0) + j);
      for (const auto &[f_dev, first, width] :
           {std::tuple{tangent.tangents[j].f_core.data(), size_t{0}, nc},
            std::tuple{tangent.tangents[j].f_virt.data(), nocc, nv}}) {
        const std::vector<double> f = FBF_TRY(download(f_dev, width * width));
        for (size_t p = 1; p < width; ++p) {
          for (size_t q = 0; q < p; ++q) {
            const double f_pq = 0.5 * (f[p + (width * q)] + f[q + (width * p)]);
            const double gap = eps[first + p] - eps[first + q];
            const bool pair =
                std::ranges::find(taken, std::array{first + p, first + q}) != taken.end();
            if (pair || std::abs(gap) < kTangentGapFloor) {
              if (!pc && std::abs(f_pq) <= kTangentZeroSource) {
                continue;
              }
              if (!pc) {
                return err_numerical(
                    "{}: orbitals {} and {} are {:.2e} Eh apart (floor {:g}) and f^(x)_pq is "
                    "{:.2e} along coordinate {} (round-off {:g}): their kappa_oo^x is not "
                    "determined by the state, and SC is not invariant to it",
                    kWho, first + p, first + q, std::abs(gap), kTangentGapFloor, f_pq, x,
                    kTangentZeroSource);
              }
              recorded.push_back({.coordinate = x,
                                  .p = static_cast<int32_t>(first + p),
                                  .q = static_cast<int32_t>(first + q),
                                  .gap = gap,
                                  .f = f_pq,
                                  .split = 0.5 * (f[p * (width + 1)] - f[q * (width + 1)])});
              continue;
            }
            const double value = -f_pq / gap;
            kappa_oo[(j * nn) + (first + p) + (nmo * (first + q))] = value;
            kappa_oo[(j * nn) + (first + q) + (nmo * (first + p))] = -value;
          }
        }
      }
    }
    Buffers::device<double> kappa_oo_dev(k * nn, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(kappa_oo_dev.data(), kappa_oo.data(),
                                           k * nn * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                           res.stream())));
    FBF_TRY(res.sync());
    for (size_t j = 0; j < k; ++j) {
      FBF_TRY(one_index(kappa_oo_dev.data() + (j * nn), h_dot.data() + (j * nn),
                        b_dot.data() + (j * naux * nn)));
    }
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, count * nn32, &kOne<double>,
                                                  kappa_oo_dev.data(), 1, kappa.data(), 1)));
    FBF_TRY(consume(x0, count, h_dot.data(), b_dot.data(), c_dot, kappa.data()));
    // The launches above read this chunk's buffers, which go with this frame.
    return res.sync();
  };
  // Each chunk of the derivative integrals in parts as wide as the memory
  // budget holds at its call, at least one coordinate.
  const auto chunk_bytes = [&](const size_t width) {
    return nevpt2_tangent_chunk_footprint(naux32, nmo32, ncore, ncas, static_cast<int32_t>(width))
        .device.scratch;
  };
  const auto chunk = [&](const int32_t x0, const int32_t count, const double *const h1,
                         const double * /*tau*/, const double *const w1) -> Status {
    const size_t width =
        FBF_TRY(widest_fitting(res, kWho, static_cast<size_t>(count), chunk_bytes));
    for (size_t j0 = 0; j0 < static_cast<size_t>(count); j0 += width) {
      const size_t part_count = std::min(width, static_cast<size_t>(count) - j0);
      FBF_TRY(part(x0 + static_cast<int32_t>(j0), static_cast<int32_t>(part_count), h1 + (j0 * nn),
                   w1 + (j0 * nn * naux)));
    }
    return {};
  };

  // W^x's chunk takes a third of the budget, never what one coordinate's
  // part needs; the parts take the rest.
  const size_t w_bytes =
      FBF_TRY(integral_derivatives_bytes(res, basis, aux, chunk_bytes(1), max_bytes));
  FBF_TRY(mcscf::df_integral_derivatives(res, molecule, basis, classes, aux, h, b, l, solution.c,
                                         chunk, w_bytes));
  return recorded;
}

std::vector<std::array<size_t, 2>> pc_taken_pairs(const std::span<const double> eps,
                                                  const size_t nc, const size_t nocc,
                                                  const double reach) {
  std::vector<std::tuple<double, size_t, size_t>> close;
  for (const auto &[first, last] : {std::pair{size_t{0}, nc}, std::pair{nocc, eps.size()}}) {
    for (size_t p = first + 1; p < last; ++p) {
      for (size_t q = first; q < p; ++q) {
        if (std::abs(eps[p] - eps[q]) < reach) {
          close.emplace_back(std::abs(eps[p] - eps[q]), p, q);
        }
      }
    }
  }
  std::ranges::sort(close);
  std::vector<bool> used(eps.size(), false);
  std::vector<std::array<size_t, 2>> taken;
  for (const auto &[gap, p, q] : close) {
    if (!used[p] && !used[q]) {
      used[p] = true;
      used[q] = true;
      taken.push_back({p, q});
    }
  }
  return taken;
}

MemoryFootprint nevpt2_tangent_chunk_footprint(const int32_t naux, const int32_t nmo,
                                               const int32_t ncore, const int32_t ncas,
                                               const int32_t count) {
  const auto nn = static_cast<size_t>(nmo) * static_cast<size_t>(nmo);
  const auto k = static_cast<size_t>(count);
  Ledger l;
  // h_dot, B_dot and kappa; the builders' tangent beside them; kappa_oo.
  l.take(sizeof(double) * k * (nn + (static_cast<size_t>(naux) * nn) + nn));
  l.call(nevpt2::builders_tangent_footprint(naux, nmo, nmo, ncore, ncas, count));
  l.take(sizeof(double) * k * nn);
  return drop(l.footprint(l.live));
}

size_t integral_derivatives_held_bytes(const BasisSet &basis, const BasisSet &aux,
                                       const size_t count) {
  const size_t nn = basis.nao_sphe() * basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t factor = nn * naux;
  const size_t stack = 3 * static_cast<size_t>(basis.natoms()) * nn;
  size_t max_blk = 0;
  for (uint32_t a = 0; a < aux.natoms(); ++a) {
    const auto [first, last] = atom_function_range(aux, a);
    max_blk = std::max(max_blk, last - first);
  }
  return sizeof(double) *
         (factor + nn + (2 * stack) + (3 * naux * naux) + (count * factor) + (3 * nn * max_blk));
}

Result<size_t> integral_derivatives_bytes(const DeviceResources &res, const BasisSet &basis,
                                          const BasisSet &aux, const size_t reserve,
                                          const size_t max_bytes) {
  const size_t memory = FBF_TRY(res.memory_budget());
  if (max_bytes > 0) {
    return std::max<size_t>(std::min(max_bytes, memory) / 3, 1);
  }
  const size_t usable = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  const size_t spare = usable > reserve ? usable - reserve : 0;
  return std::max(mcscf::df_integral_derivatives_footprint(basis, aux).device.scratch,
                  std::min(usable / 3, spare));
}

Result<size_t> widest_fitting(const DeviceResources &res, const std::string_view who,
                              const size_t count, const std::function<size_t(size_t)> &bytes) {
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t room = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  if (bytes(1) > room) {
    return err_unsupported("{}: one coordinate's part needs {} bytes of device memory, and device "
                           "{} has {} to give",
                           who, bytes(1), res.dev_idx(), room);
  }
  // The widest in [1, count] that fits: the bytes grow with it.
  size_t lo = 1;
  size_t hi = std::max<size_t>(count, 1);
  while (lo < hi) {
    const size_t mid = lo + ((hi - lo + 1) / 2);
    if (bytes(mid) <= room) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

MemoryFootprint df_nevpt2_tangents_footprint(const BasisSet &basis, const BasisSet &aux,
                                             const int32_t ncore, const int32_t ncas,
                                             const int32_t nalpha, const int32_t nbeta) {
  const size_t nao = basis.nao_sphe();
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms() || aux.nao_sphe() == 0 || ncore < 0 ||
      ncas < 1 || static_cast<size_t>(ncore) + static_cast<size_t>(ncas) > nao) {
    return {};
  }
  const auto nmo = static_cast<int32_t>(nao);
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const size_t nn = nao * nao;
  const OrbitalRange all{.first = 0, .count = nmo};
  Ledger l;
  // The strings; h_MO and the MO factor, held, beside the gemmwich's work.
  l.call(ci::device_string_set_footprint(ncas, nalpha));
  l.call(ci::device_string_set_footprint(ncas, nbeta));
  l.take(sizeof(double) * (nn + (aux.nao_sphe() * nn)));
  l.run(MemoryFootprint{.device = {.persistent = sizeof(double) * nn}});
  l.borrow(b_mo_footprint(naux, nmo, all, all));
  // The derivative integrals at their least, and beside what they hold at
  // their first chunk, one coordinate's part.
  l.run(mcscf::df_integral_derivatives_footprint(basis, aux));
  const size_t held = integral_derivatives_held_bytes(basis, aux, 1);
  l.take(held);
  l.run(nevpt2_tangent_chunk_footprint(naux, nmo, ncore, ncas, 1));
  l.give(held);
  return l.footprint(0);
}

} // namespace free_breakfast::hessian
