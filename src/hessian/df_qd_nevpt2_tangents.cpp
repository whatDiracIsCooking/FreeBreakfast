/// @file
/// @brief df_qd_nevpt2_tangents: the QD-NEVPT2 parameter response over a
/// state average (lambda^x's kappa_nr, xi_r and unequal-weight rotations, the
/// closed-form equal-weight rotations of Eq. qd-theta, and kappa_oo at f^SA)
/// and the total tangent t_x of (h_MO, B_MO, {c_r}), per chunk of
/// mcscf::df_integral_derivatives' derivative integrals: the one-index
/// transforms by kappa as gemms over the MO factor, the active Hamiltonian's
/// and f^SA's change by nevpt2::qd_builders_tangent, and the degenerate-pair
/// rule on kappa_oo's division (df_nevpt2_tangents', the single-state one).

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, geam, gemmStridedBatched, wwrMemcpyAsync,
                                // ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, OrbitalRange
import free_breakfast.recipes.blas;                     // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.rdm;
import free_breakfast.response.orbital_pairs; // unpack_orbital_pairs
import free_breakfast.mcscf.solve;
import free_breakfast.response.zvector;       // ZVectors, rotating_pairs
import free_breakfast.mcscf.third_derivative; // df_integral_derivatives
import free_breakfast.nevpt2.gradient;        // qd_builders_tangent
import :state_specific;                       // state_averaged_electrons, kSaCasscfMaxRotation

#include "diagnostics.h"

namespace free_breakfast::hessian {

// df_nevpt2_tangents' degenerate-pair rule (the NEVPT2 gradient's): below
// this gap (Eh) a core or virtual pair's division is not one the state
// determines, ...
constexpr double kQdTangentGapFloor = 1e-6;
// ... and zero stands for it only while the numerator is round-off.
constexpr double kQdTangentZeroSource = 1e-8;
// The largest off-diagonal |f^SA_pq| over the core or the virtuals of
// orbitals still canonical (df_nevpt2_tangents' bar, at f^SA).
constexpr double kQdTangentCanonicalTolerance = 1e-8;

Result<QdNevpt2Tangents> df_qd_nevpt2_tangents(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, const std::span<const double> weights, const int32_t ncore,
    const int32_t ncas, const response::ZVectors &cp, const bool pc,
    const std::function<Status(int32_t x0, int32_t count, const double *h_dot, const double *b_dot,
                               const double *c_dot, const double *kappa)> &consume,
    const size_t max_bytes) {
  constexpr std::string_view kWho = "hessian::df_qd_nevpt2_tangents";
  const auto [nalpha, nbeta] = FBF_TRY(
      state_averaged_electrons(kWho, res, molecule, basis, aux, b, solution, weights, ncore, ncas));
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
  const auto n_act = static_cast<size_t>(ncas);
  const size_t nocc = nc + n_act;
  const size_t nv = nmo - nocc;
  const size_t pairs = (nc * (nmo - nc)) + (n_act * nv);
  const std::span<const double> energies = solution.roots.energies;
  const size_t nroots = energies.size();
  const size_t root_pairs = nroots * (nroots - 1) / 2;
  const ScopedDeviceIndex on{res.dev_idx()};

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  const auto ndet = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const size_t n_ci = ndet * nroots;
  if (solution.roots.vectors.num_elements() != n_ci) {
    return err_config("{}: {} roots of {} doubles over {} determinants", kWho, nroots,
                      solution.roots.vectors.num_elements(), ndet);
  }
  if (cp.orbital.num_elements() != pairs * n_x || cp.ci.num_elements() != n_ci * n_x ||
      cp.theta.size() != root_pairs * n_x) {
    return err_config(
        "{}: lambda^x of {} orbital, {} CI and {} rotation doubles, not {} columns of "
        "{} pairs, {} roots' {} determinants and {} pairs of roots",
        kWho, cp.orbital.num_elements(), cp.ci.num_elements(), cp.theta.size(), n_x, pairs, nroots,
        ndet, root_pairs);
  }
  // The pairs of unequal weights rotate in the CP solve; the rest are
  // Eq. qd-theta's.
  const std::vector<bool> rotates = response::rotating_pairs(weights);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto ndet32 = static_cast<int32_t>(ndet);
  const double *const vectors = solution.roots.vectors.data();

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
  // coordinate's slabs (df_nevpt2_tangents').
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
  // A rotation theta of roots r < s onto one coordinate's c_dot block:
  // c_dot_r += theta c_s, c_dot_s -= theta c_r.
  const auto rotate = [&](double *const c_dot, const size_t r, const size_t s,
                          const double theta) -> Status {
    const double minus = -theta;
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, ndet32, &theta, vectors + (s * ndet), 1,
                                                  c_dot + (r * ndet), 1)));
    return gpu_status(wwr::axpy<double, int32_t>(blas, ndet32, &minus, vectors + (r * ndet), 1,
                                                 c_dot + (s * ndet), 1));
  };

  QdNevpt2Tangents out{.theta = std::vector<double>(root_pairs * n_x, 0.0)};
  std::vector<double> eps;
  // The transition RDMs of each pair r < s, gamma then Gamma.
  std::vector<std::pair<std::vector<double>, std::vector<double>>> transition;
  // One chunk of @p count coordinates from x0, as wide as the budget holds.
  const auto part = [&](const int32_t x0, const int32_t count, const double *const h1,
                        const double *const w1) -> Status {
    const auto k = static_cast<size_t>(count);
    Buffers::device<double> h_dot(k * nn, res.device());
    Buffers::device<double> b_dot(k * naux * nn, res.device());
    Buffers::device<double> kappa(k * nn, res.device());
    Buffers::device<double> c_dot(k * n_ci, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_dot.data(), h1, k * nn * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
        c_dot.data(), cp.ci.data() + (static_cast<size_t>(x0) * n_ci), k * n_ci * sizeof(double),
        wwr::wwrMemcpyDeviceToDevice, res.stream())));

    // Step 1: h1^x, W^x to B's layout (Q fastest), the one-index transform by
    // kappa_nr^x, and c_dot = xi + the CP solve's rotations.
    for (size_t j = 0; j < k; ++j) {
      const size_t x = static_cast<size_t>(x0) + j;
      const double *const w = w1 + (j * nn * naux);
      double *const bj = b_dot.data() + (j * naux * nn);
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                                    naux32, nn32, &kOne<double>, w, nn32,
                                                    &kZero<double>, w, naux32, bj, naux32)));
      FBF_TRY(response::unpack_orbital_pairs(
          res, nmo32, ncore, ncas, cp.orbital.data() + (x * pairs), kappa.data() + (j * nn)));
      FBF_TRY(one_index(kappa.data() + (j * nn), h_dot.data() + (j * nn), bj));
      size_t at = 0;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s, ++at) {
          if (rotates[at]) {
            out.theta[(x * root_pairs) + at] = cp.theta[(x * root_pairs) + at];
            FBF_TRY(rotate(c_dot.data() + (j * n_ci), r, s, cp.theta[(x * root_pairs) + at]));
          }
        }
      }
    }

    // The QD builders at the point (f^SA's blocks) and their change along
    // that: the active Hamiltonian's and f^SA's.
    nevpt2::QdBuildersTangent tangent = FBF_TRY(nevpt2::qd_builders_tangent(
        res, alpha, beta, naux32, nmo32, nmo32, b.data(), solution.c.data(), h.data(), ncore, ncas,
        weights, vectors, count, h_dot.data(), b_dot.data(), c_dot.data()));
    if (eps.empty()) {
      eps.assign(nmo, 0.0);
      for (const auto &[f_dev, first, width] :
           {std::tuple{tangent.value.shared.f_core.data(), size_t{0}, nc},
            std::tuple{tangent.value.shared.f_virt.data(), nocc, nv}}) {
        const std::vector<double> f = FBF_TRY(download(f_dev, width * width));
        for (size_t q = 0; q < width; ++q) {
          eps[first + q] = f[q * (width + 1)];
          for (size_t p = 0; p < width; ++p) {
            if (p != q && std::abs(f[p + (width * q)]) > kQdTangentCanonicalTolerance) {
              return err_config("{}: f^SA_{},{} is {:.2e} (above {:g}): the core and virtuals "
                                "are not canonical",
                                kWho, first + p, first + q, f[p + (width * q)],
                                kQdTangentCanonicalTolerance);
            }
          }
        }
      }
      // The roots' symmetrized transition RDMs, rdm12's convention
      // (<H> = h1eff.gamma + 1/2 (tu|vw).Gamma).
      ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
      Buffers::device<double> gamma(n_act * n_act, res.device());
      Buffers::device<double> gamma2(n_act * n_act * n_act * n_act, res.device());
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          FBF_TRY(rdm.trans_rdm12(res, vectors + (s * ndet), vectors + (r * ndet), gamma.data(),
                                  gamma2.data()));
          transition.emplace_back(FBF_TRY(download(gamma.data(), gamma.num_elements())),
                                  FBF_TRY(download(gamma2.data(), gamma2.num_elements())));
        }
      }
    }

    // Step 2: theta^x_sr = H^x_sr / (E_r - E_s) for the equal pairs, H^x_sr =
    // sum h1eff^(x) gamma^(sr) + 1/2 sum (tu|vw)^(x) Gamma^(sr) (the
    // integrals column-major, the RDMs row-major, which their symmetry makes
    // one flat dot); every rotation then held to the bound.
    for (size_t j = 0; j < k; ++j) {
      const size_t x = static_cast<size_t>(x0) + j;
      const std::vector<double> h1e =
          FBF_TRY(download(tangent.tangents[j].shared.h1e.data(), n_act * n_act));
      const std::vector<double> eri =
          FBF_TRY(download(tangent.tangents[j].shared.eri.data(), n_act * n_act * n_act * n_act));
      size_t at = 0;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s, ++at) {
          double &theta = out.theta[(x * root_pairs) + at];
          if (!rotates[at]) {
            const auto &[gamma, gamma2] = transition[at];
            const double coupling =
                std::inner_product(h1e.begin(), h1e.end(), gamma.begin(), 0.0) +
                (0.5 * std::inner_product(eri.begin(), eri.end(), gamma2.begin(), 0.0));
            theta = coupling / (energies[r] - energies[s]);
            FBF_TRY(rotate(c_dot.data() + (j * n_ci), r, s, theta));
          }
          if (!(std::abs(theta) <= kSaCasscfMaxRotation)) {
            return err_numerical("{}: roots {} and {} rotate into each other at {:.3e} per bohr "
                                 "along coordinate {} (above {:.0f}): too near a degeneracy for "
                                 "the QD states to have a Hessian",
                                 kWho, r, s, std::abs(theta), x, kSaCasscfMaxRotation);
          }
          out.max_theta = std::max(out.max_theta, std::abs(theta));
        }
      }
    }

    // Step 3: kappa_oo^x_pq = -f^(x)_pq / (eps_p - eps_q) over each block of
    // f^SA, under the degenerate-pair rule; then its one-index transform, and
    // kappa the sum.
    std::vector<double> kappa_oo(k * nn, 0.0);
    for (size_t j = 0; j < k; ++j) {
      const auto x = static_cast<int32_t>(static_cast<size_t>(x0) + j);
      for (const auto &[f_dev, first, width] :
           {std::tuple{tangent.tangents[j].shared.f_core.data(), size_t{0}, nc},
            std::tuple{tangent.tangents[j].shared.f_virt.data(), nocc, nv}}) {
        const std::vector<double> f = FBF_TRY(download(f_dev, width * width));
        for (size_t p = 1; p < width; ++p) {
          for (size_t q = 0; q < p; ++q) {
            const double f_pq = 0.5 * (f[p + (width * q)] + f[q + (width * p)]);
            const double gap = eps[first + p] - eps[first + q];
            if (std::abs(gap) < kQdTangentGapFloor) {
              if (!pc && std::abs(f_pq) <= kQdTangentZeroSource) {
                continue;
              }
              if (!pc) {
                return err_numerical(
                    "{}: orbitals {} and {} are {:.2e} Eh apart (floor {:g}) and f^SA(x)_pq is "
                    "{:.2e} along coordinate {} (round-off {:g}): their kappa_oo^x is not "
                    "determined by the state, and SC is not invariant to it",
                    kWho, first + p, first + q, std::abs(gap), kQdTangentGapFloor, f_pq, x,
                    kQdTangentZeroSource);
              }
              out.recorded.push_back({.coordinate = x,
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
            out.max_kappa_oo = std::max(out.max_kappa_oo, std::abs(value));
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
    FBF_TRY(consume(x0, count, h_dot.data(), b_dot.data(), c_dot.data(), kappa.data()));
    // The launches above read this chunk's buffers, which go with this frame.
    return res.sync();
  };
  // Each chunk of the derivative integrals in parts as wide as the memory
  // budget holds at its call, at least one coordinate.
  const auto chunk_bytes = [&](const size_t width) {
    return qd_nevpt2_tangent_chunk_footprint(naux32, nmo32, ncore, ncas, nalpha, nbeta,
                                             static_cast<int32_t>(nroots),
                                             static_cast<int32_t>(width))
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
  return out;
}

MemoryFootprint qd_nevpt2_tangent_chunk_footprint(const int32_t naux, const int32_t nmo,
                                                  const int32_t ncore, const int32_t ncas,
                                                  const int32_t nalpha, const int32_t nbeta,
                                                  const int32_t nroots, const int32_t count) {
  const auto nn = static_cast<size_t>(nmo) * static_cast<size_t>(nmo);
  const auto k = static_cast<size_t>(count);
  const auto n_act = static_cast<size_t>(ncas);
  const auto strings = [ncas](const int32_t electrons) {
    return static_cast<size_t>(std::round(math::binomial(ncas, electrons)));
  };
  const size_t ndet = strings(nalpha) * strings(nbeta);
  const size_t n_ci = ndet * static_cast<size_t>(nroots);
  Ledger l;
  // h_dot, B_dot, kappa and c_dot; the QD builders' tangent beside them; the
  // first chunk's transition RDMs; kappa_oo.
  l.take(sizeof(double) * k * (nn + (static_cast<size_t>(naux) * nn) + nn + n_ci));
  l.call(nevpt2::qd_builders_tangent_footprint(naux, nmo, nmo, ncore, ncas, nroots, count));
  MemoryFootprint transition{
      .device = {.persistent =
                     sizeof(double) * ((n_act * n_act) + (n_act * n_act * n_act * n_act))}};
  transition.device.workspace =
      ci::trans_rdm12_footprint(ncas, static_cast<int32_t>(ndet)).device.workspace;
  l.run(transition);
  l.take(sizeof(double) * k * nn);
  return drop(l.footprint(l.live));
}

MemoryFootprint df_qd_nevpt2_tangents_footprint(const BasisSet &basis, const BasisSet &aux,
                                                const int32_t ncore, const int32_t ncas,
                                                const int32_t nalpha, const int32_t nbeta,
                                                const int32_t nroots) {
  const size_t nao = basis.nao_sphe();
  if (basis.natoms() == 0 || aux.natoms() != basis.natoms() || aux.nao_sphe() == 0 || ncore < 0 ||
      ncas < 1 || static_cast<size_t>(ncore) + static_cast<size_t>(ncas) > nao || nroots < 1) {
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
  l.run(qd_nevpt2_tangent_chunk_footprint(naux, nmo, ncore, ncas, nalpha, nbeta, nroots, 1));
  l.give(held);
  return l.footprint(0);
}

} // namespace free_breakfast::hessian
