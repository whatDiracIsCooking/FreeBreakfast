/// @file
/// @brief df_sc_nevpt2: the SC-NEVPT2 analytic nuclear Hessian on a
/// single-state DF-CASSCF (nuclear_hessian.tex eq. hess), assembled from
/// merged parts: E_cas by df_casscf; the Z terms by df_casscf_z at the
/// gradient's multipliers, and their kappa_oo rows in closed form; E_corr +
/// Phi as df_adjoint_xy at its adjoint, the Hessian-vector product between
/// the tangents t_x of df_nevpt2_tangents, and the rotation cross terms, all
/// in the caller's coordinates. Pairs whose empty perturbers are not a
/// quadratic form are refused together.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne, kTwo, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, axpy, gemm, gemv, gemmStridedBatched, wwrMemcpyAsync,
                                // wwrMemsetAsync, wwrMemGetInfo, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, OrbitalRange
import free_breakfast.recipes.blas;                     // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.response.orbital_pairs; // unpack_orbital_pairs
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.zvector;
import free_breakfast.nevpt2.energy;   // kEmptyPopulationFloor
import free_breakfast.nevpt2.gradient; // sc_adjoint, zc_multipliers, zc_adjoint,
                                       // lagrangian_gradient, sc_hessian_product
import :state_specific;                // state_specific_electrons

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The coordinates' rows are orthonormal to this (Q Q^T = 1), so that
// Q H Q^T is the Hessian in them.
constexpr double kScNevpt2OrthonormalTolerance = 1e-10;
// The displacement, in bohr along a coordinate, within which a canonical
// inactive pair may not mix (t_mix = 1 / 2 |kappa_oo|): #817's convention,
// df_pc_nevpt2's kPcNevpt2LiftStep. Closer, the SC surface switches branch
// inside a vibrational or finite-difference step, and one analytic Hessian
// does not represent it.
constexpr double kScNevpt2MixStep = 1e-3;

Result<ScNevpt2Terms>
df_sc_nevpt2(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
             const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
             const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
             const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
             const int32_t batch, const std::span<const double> coordinates) {
  constexpr std::string_view kWho = "hessian::df_sc_nevpt2";
  const auto [nalpha, nbeta] =
      FBF_TRY(state_specific_electrons(kWho, res, molecule, basis, aux, b, solution, ncore, ncas));
  const size_t n_x = static_cast<size_t>(3) * basis.natoms();
  const size_t nmo = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nn = nmo * nmo;
  const size_t nc = static_cast<size_t>(ncore);
  const size_t n_act = static_cast<size_t>(ncas);
  const size_t pairs = (nc * (nmo - nc)) + (n_act * (nmo - nc - n_act));

  // The coordinates: k orthonormal rows of n_x, the identity when none.
  if (coordinates.size() % n_x != 0 || (!coordinates.empty() && coordinates.size() < n_x)) {
    return err_config("{}: {} coordinate doubles are not whole rows of {}", kWho,
                      coordinates.size(), n_x);
  }
  const size_t k = coordinates.empty() ? n_x : coordinates.size() / n_x;
  std::vector<double> q(k * n_x, 0.0); // row i at i n_x, as given
  if (coordinates.empty()) {
    for (size_t i = 0; i < n_x; ++i) {
      q[i * (n_x + 1)] = 1.0;
    }
  } else {
    std::ranges::copy(coordinates, q.begin());
  }
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (size_t x = 0; x < n_x; ++x) {
        s += q[(i * n_x) + x] * q[(j * n_x) + x];
      }
      if (!(std::abs(s - (i == j ? 1.0 : 0.0)) <= kScNevpt2OrthonormalTolerance)) {
        return err_config("{}: coordinates {} and {} have overlap {:.3e}: not orthonormal rows",
                          kWho, i, j, s);
      }
    }
  }
  if (naux * nn > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("{}: naux {} times nmo^2 {} is past int32_t", kWho, naux, nn);
  }
  const auto k32 = static_cast<int32_t>(k);
  const auto n_x32 = static_cast<int32_t>(n_x);
  const auto nmo32 = static_cast<int32_t>(nmo);
  const auto naux32 = static_cast<int32_t>(naux);
  const auto nn32 = static_cast<int32_t>(nn);
  const auto bn32 = static_cast<int32_t>(naux * nn);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();

  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  const auto ndet = static_cast<size_t>(alpha.size()) * static_cast<size_t>(beta.size());
  const auto ndet32 = static_cast<int32_t>(ndet);
  if (solution.roots.vectors.num_elements() != ndet) {
    return err_config("{}: a root of {} doubles over {} determinants", kWho,
                      solution.roots.vectors.num_elements(), ndet);
  }

  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  const auto zeroed = [&res](const size_t count) -> Result<Buffers::device<double>> {
    Buffers::device<double> d(count, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d.data(), 0, count * sizeof(double), res.stream())));
    return d;
  };
  // (i, j) = <a_i, b_j> over two stacks of k columns of @p rows, on the host.
  const auto gram = [&](const int32_t rows, const double *a,
                        const double *bb) -> Result<std::vector<double>> {
    Buffers::device<double> out(k * k, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, k32, k32, rows, &kOne<double>, a, bb,
                                               &kZero<double>, out.data())));
    return download(out.data(), k * k);
  };
  // Q P Q^T of a Cartesian n_x x n_x matrix (column-major), k x k.
  const auto to_coordinates = [&](std::span<const double> p) {
    std::vector<double> qp(k * n_x, 0.0); // (i, y) at i + k y
    for (size_t y = 0; y < n_x; ++y) {
      for (size_t x = 0; x < n_x; ++x) {
        const double pxy = p[x + (n_x * y)];
        for (size_t i = 0; i < k; ++i) {
          qp[i + (k * y)] += q[(i * n_x) + x] * pxy;
        }
      }
    }
    std::vector<double> out(k * k, 0.0);
    for (size_t j = 0; j < k; ++j) {
      for (size_t y = 0; y < n_x; ++y) {
        const double qjy = q[(j * n_x) + y];
        for (size_t i = 0; i < k; ++i) {
          out[i + (k * j)] += qp[i + (k * y)] * qjy;
        }
      }
    }
    return out;
  };

  // 1. E_cas, and lambda^x (kappa_nr^x, c^x) with it.
  CasscfTerms casscf =
      FBF_TRY(df_casscf(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas));

  // 2. t_x in the coordinates: t_i = sum_x Q_ix t_x, accumulated a chunk of
  // Cartesian coordinates at a time, with kappa (the total) and kappa_nr.
  Buffers::device<double> qt(n_x * k, res.device()); // Q^T, n_x x k column-major
  FBF_TRY(copy(qt.data(), q.data(), n_x * k, wwr::wwrMemcpyHostToDevice));
  Buffers::device<double> t_h = FBF_TRY(zeroed(nn * k));
  Buffers::device<double> t_b = FBF_TRY(zeroed(naux * nn * k));
  Buffers::device<double> t_c = FBF_TRY(zeroed(ndet * k));
  Buffers::device<double> t_kappa = FBF_TRY(zeroed(nn * k));
  // S_coords += S_chunk Q^T[x0 .. x0 + count), each stack of @p rows.
  const auto accumulate = [&](const int32_t rows, const int32_t x0, const int32_t count,
                              const double *chunk, double *stack) -> Status {
    return gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, rows, k32, count, &kOne<double>, chunk, rows,
        qt.data() + x0, n_x32, &kOne<double>, stack, rows));
  };
  FBF_TRY(df_nevpt2_tangents(
      res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, casscf.cp, false,
      [&](const int32_t x0, const int32_t count, const double *h_dot, const double *b_dot,
          const double *c_dot, const double *kappa) -> Status {
        FBF_TRY(accumulate(nn32, x0, count, h_dot, t_h.data()));
        FBF_TRY(accumulate(bn32, x0, count, b_dot, t_b.data()));
        FBF_TRY(accumulate(ndet32, x0, count, c_dot, t_c.data()));
        return accumulate(nn32, x0, count, kappa, t_kappa.data());
      }));
  Buffers::device<double> t_nr = FBF_TRY(zeroed(nn * k));
  {
    Buffers::device<double> kappa_nr(nn * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(response::unpack_orbital_pairs(res, nmo32, ncore, ncas,
                                             casscf.cp.orbital.data() + (x * pairs),
                                             kappa_nr.data() + (x * nn)));
    }
    FBF_TRY(accumulate(nn32, 0, n_x32, kappa_nr.data(), t_nr.data()));
    FBF_TRY(res.sync());
  }

  // 3. E_corr + Phi's adjoint, kept (h_bar, Lambda), then the gradient's
  // Lagrangian: its multipliers (Z, z) and z^c.
  const double *const c = solution.c.data();
  const double *const vector = solution.roots.vectors.data();
  Buffers::device<double> h_bar(nn, res.device());
  Buffers::device<double> lambda(naux * nn, res.device());
  std::optional<nevpt2::ScGradient> lagrangian;
  double mix_step = std::numeric_limits<double>::infinity();
  {
    nevpt2::ScAdjoint corr = FBF_TRY(nevpt2::sc_adjoint(
        res, alpha, beta, naux32, nmo32, nmo32, b.data(), c, h.data(), ncore, ncas, vector, batch));
    // The inactive pairs' mixing: kappa_oo = kappa - kappa_nr along each
    // coordinate is F^i_pq / gap on a core-core or virtual-virtual pair. The
    // pair's 2 x 2 block [[e_p, F t], [F t, e_q]] has gap sqrt(g^2 + 4 F^2
    // t^2) and mixing angle tan 2 theta = 2 F t / g, so t_mix = g / 2 |F| =
    // 1 / 2 |kappa_oo_pq| is its avoided crossing's half-width: 2 theta at 45
    // degrees, the gap sqrt(2) g. SC is not invariant to that rotation, so
    // below kScNevpt2MixStep its second derivative is the curvature of a
    // narrower region than a vibration (README, "Mixing inactive pairs").
    {
      const std::vector<double> epsilon = FBF_TRY(download(corr.epsilon.data(), nmo));
      const std::vector<double> kappa = FBF_TRY(download(t_kappa.data(), nn * k));
      const std::vector<double> kappa_nr = FBF_TRY(download(t_nr.data(), nn * k));
      // 0 core, 1 active, 2 virtual.
      const auto space = [&](const size_t p) -> int32_t {
        return p < nc ? 0 : (p < nc + n_act ? 1 : 2);
      };
      std::string named;
      std::string why;
      for (size_t i = 0; i < k; ++i) {
        double worst = 0.0;
        size_t wp = 0;
        size_t wq = 0;
        for (size_t q = 0; q < nmo; ++q) {
          for (size_t p = 0; p < q; ++p) {
            if (space(p) != space(q) || space(p) == 1) {
              continue;
            }
            const size_t at = (i * nn) + p + (nmo * q);
            const double oo = std::abs(kappa[at] - kappa_nr[at]);
            if (oo > worst) {
              worst = oo;
              wp = p;
              wq = q;
            }
          }
        }
        const double t_mix = 0.5 / worst;
        mix_step = std::min(mix_step, t_mix);
        if (t_mix < kScNevpt2MixStep) {
          named += std::format("{}{}", named.empty() ? "" : ", ", i);
          why += std::format("; coordinate {}: orbitals ({}, {}), gap {:.3e} Eh, t_mix {:.3e} bohr",
                             i, wp, wq, std::abs(epsilon[wp] - epsilon[wq]), t_mix);
        }
      }
      if (!named.empty()) {
        return err_numerical(
            "{}: SC-NEVPT2 has no Hessian along coordinates {}: each mixes a canonical inactive "
            "pair within {:g} bohr (t_mix = gap / 2 |F^i_pq|), and SC is not invariant to that "
            "rotation, so its surface switches branch inside a vibrational or finite-difference "
            "step and one analytic Hessian does not represent it; use PC-NEVPT2{}",
            kWho, named, kScNevpt2MixStep, why);
      }
    }
    nevpt2::ZcMultipliers zc = FBF_TRY(
        nevpt2::zc_multipliers(res, nmo32, ncore, ncas, corr.a.data(), corr.epsilon.data()));
    const nevpt2::ZcAdjoint phi =
        FBF_TRY(nevpt2::zc_adjoint(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c, h.data(),
                                   ncore, ncas, std::array{1.0}, vector, zc.z.data()));
    FBF_TRY(copy(h_bar.data(), corr.h_bar.data(), nn, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(lambda.data(), corr.lambda.data(), naux * nn, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, nn32, &kOne<double>, phi.h_bar.data(), 1,
                                                  h_bar.data(), 1)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, bn32, &kOne<double>, phi.lambda.data(), 1,
                                                  lambda.data(), 1)));
    FBF_TRY(res.sync());
    lagrangian.emplace(
        FBF_TRY(nevpt2::lagrangian_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                            ncore, ncas, std::move(corr), std::move(zc), phi)));
  }
  const nevpt2::ScGradient &gradient = *lagrangian;

  // 4. Sum <t_i, HVP t_j>: the product along every coordinate, then each
  // pair of coordinates that both populate an empty perturber again with
  // the other as its partner (K6 at the pair's mean eta, and its rho_tau
  // check). The directions are chunked by an estimate of the product's
  // per-direction memory (its builders' tangent, seven reverse passes'
  // copies of the moved fields and the tangent bars).
  std::vector<double> hvp(k * k, 0.0);
  double defect = 0.0;
  std::vector<std::string> refused;
  {
    Buffers::device<double> paired(k * k, res.device());
    // Column j of @p into (k long) = <t_i, P> over every i.
    const auto pair_with = [&](const nevpt2::AdjointTangent &p, double *into) -> Status {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, nn32, k32,
                                                    &kOne<double>, t_h.data(), nn32, p.h_bar.data(),
                                                    1, &kZero<double>, into, 1)));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, bn32, k32,
                                                    &kOne<double>, t_b.data(), bn32,
                                                    p.lambda.data(), 1, &kOne<double>, into, 1)));
      return gpu_status(wwr::gemv<double, int32_t>(blas, wwr::WWRBLAS_OP_T, ndet32, k32,
                                                   &kOne<double>, t_c.data(), ndet32, p.y.data(), 1,
                                                   &kOne<double>, into, 1));
    };
    const auto product = [&](const int32_t count, const double *dh, const double *db,
                             const double *dc, const nevpt2::MoDirections &partners) {
      return nevpt2::sc_hessian_product(res, alpha, beta, naux32, nmo32, nmo32, b.data(), c,
                                        h.data(), ncore, ncas, vector, gradient.zc.z.data(), batch,
                                        {.count = count, .h = dh, .b = db, .c = dc}, partners);
    };
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    const size_t act6 = n_act * n_act * n_act * n_act * n_act * n_act;
    const size_t per_direction =
        sizeof(double) * ((16 * naux * nn) + (8 * act6) + (8 * ndet) + (16 * nn));
    const size_t chunk = std::clamp<size_t>(free_bytes / 2 / per_direction, 1, k);
    std::vector<double> population(k, 0.0);
    double scale = 0.0;
    for (size_t j0 = 0; j0 < k; j0 += chunk) {
      const size_t count = std::min(chunk, k - j0);
      const nevpt2::ScHessianProduct p =
          FBF_TRY(product(static_cast<int32_t>(count), t_h.data() + (j0 * nn),
                          t_b.data() + (j0 * naux * nn), t_c.data() + (j0 * ndet), {}));
      for (size_t u = 0; u < count; ++u) {
        FBF_TRY(pair_with(p.products[u], paired.data() + ((j0 + u) * k)));
        population[j0 + u] = p.population[u];
        scale = std::max(scale, p.scale[u]);
      }
      FBF_TRY(res.sync());
    }
    hvp = FBF_TRY(download(paired.data(), k * k));
    // A coordinate breaks the symmetry when it populates an empty perturber
    // against the largest population scale over the coordinates (one size,
    // orthonormal rows): a rigid translation's own scale is round-off.
    std::vector<bool> breaking(k, false);
    for (size_t i = 0; i < k; ++i) {
      breaking[i] = population[i] > nevpt2::kEmptyPopulationFloor * scale;
    }

    // The pairs: directions (t_j, t_i), partners (t_i, t_j).
    Buffers::device<double> two_h(2 * nn, res.device());
    Buffers::device<double> two_b(2 * naux * nn, res.device());
    Buffers::device<double> two_c(2 * ndet, res.device());
    Buffers::device<double> swap_h(2 * nn, res.device());
    Buffers::device<double> swap_b(2 * naux * nn, res.device());
    Buffers::device<double> swap_c(2 * ndet, res.device());
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < j; ++i) {
        if (!breaking[i] || !breaking[j]) {
          continue;
        }
        for (const auto &[stack, rows, two, swap] :
             {std::tuple{t_h.data(), nn, two_h.data(), swap_h.data()},
              std::tuple{t_b.data(), naux * nn, two_b.data(), swap_b.data()},
              std::tuple{t_c.data(), ndet, two_c.data(), swap_c.data()}}) {
          FBF_TRY(copy(two, stack + (j * rows), rows, wwr::wwrMemcpyDeviceToDevice));
          FBF_TRY(copy(two + rows, stack + (i * rows), rows, wwr::wwrMemcpyDeviceToDevice));
          FBF_TRY(copy(swap, stack + (i * rows), rows, wwr::wwrMemcpyDeviceToDevice));
          FBF_TRY(copy(swap + rows, stack + (j * rows), rows, wwr::wwrMemcpyDeviceToDevice));
        }
        auto p = product(2, two_h.data(), two_b.data(), two_c.data(),
                         {.count = 2, .h = swap_h.data(), .b = swap_b.data(), .c = swap_c.data()});
        if (!p.has_value()) {
          if (p.error().kind != ErrorKind::Numerical) {
            return std::unexpected(std::move(p.error()));
          }
          refused.push_back(std::format("({}, {}): {}", i, j, p.error().message));
          continue;
        }
        defect = std::max({defect, p->defect[0], p->defect[1]});
        FBF_TRY(pair_with(p->products[0], paired.data()));
        FBF_TRY(pair_with(p->products[1], paired.data() + k));
        const std::vector<double> two_columns = FBF_TRY(download(paired.data(), 2 * k));
        hvp[i + (k * j)] = two_columns[i];
        hvp[j + (k * i)] = two_columns[k + j];
      }
    }
  }
  if (!refused.empty()) {
    std::string named;
    std::string why;
    for (const std::string &r : refused) {
      named += std::format("{}{}", named.empty() ? "" : ", ", r.substr(0, r.find(':')));
      why += std::format("; {}", r);
    }
    return err_numerical("{}: SC-NEVPT2 has no Hessian on the coordinate pairs {}: their empty "
                         "perturbers' energy is not a quadratic form on the pair{}",
                         kWho, named, why);
  }

  // 5. The Z terms at (Z, z): Z . d2g/dRdR, the L_Ax pair and the third
  // derivative over Lambda = lambda^x in ThirdDerivative's layout.
  const size_t size = pairs + ndet;
  const auto size32 = static_cast<int32_t>(size);
  std::optional<CasscfZTerms> z_terms;
  std::vector<double> lx_cart;
  {
    Buffers::device<double> z(size, res.device());
    FBF_TRY(
        copy(z.data(), gradient.multipliers.orbital.data(), pairs, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(
        copy(z.data() + pairs, gradient.multipliers.ci.data(), ndet, wwr::wwrMemcpyDeviceToDevice));
    Buffers::device<double> lambda_x(size * n_x, res.device());
    for (size_t x = 0; x < n_x; ++x) {
      FBF_TRY(copy(lambda_x.data() + (x * size), casscf.cp.orbital.data() + (x * pairs), pairs,
                   wwr::wwrMemcpyDeviceToDevice));
      FBF_TRY(copy(lambda_x.data() + (x * size) + pairs, casscf.cp.ci.data() + (x * ndet), ndet,
                   wwr::wwrMemcpyDeviceToDevice));
    }
    FBF_TRY(res.sync());
    z_terms.emplace(FBF_TRY(df_casscf_z(res, molecule, basis, classes, aux, h, b, l, solution,
                                        ncore, ncas, z, n_x32, lambda_x.data())));
    Buffers::device<double> out(n_x * n_x, res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_x32, n_x32, size32, &kOne<double>,
                                               z_terms->lxlambda.data(), lambda_x.data(),
                                               &kZero<double>, out.data())));
    lx_cart = FBF_TRY(download(out.data(), n_x * n_x));
  }

  // 6. The skeleton at E_corr + Phi's adjoint.
  const std::vector<double> skeleton_cart = FBF_TRY(df_adjoint_xy(
      res, molecule, basis, classes, aux, h, b, l, solution.c, h_bar.data(), lambda.data()));

  // 7. The rotation cross terms and the Z terms' kappa_oo rows, from
  // per-coordinate nmo x nmo stacks: X(t_i), Y_i = X(O(I, kappa_i)) and
  // V_i = [X_N, kappa_nr_i], X(M) = 2 (M_h h_bar + sum_Q M_Q Lambda_Q) and
  // O(M, kappa) = kappa^T M + M kappa (README "The SC-NEVPT2 Hessian").
  std::vector<double> rotation_tau(k * k);
  std::vector<double> rotation_kappa(k * k);
  std::vector<double> z_koo(k * k);
  {
    Buffers::device<double> h_mo(nn, res.device());
    Buffers::device<double> l_mo(naux * nn, res.device());
    {
      Buffers::device<double> work(nn, res.device());
      FBF_TRY(gpu_status(gemmwich(blas, nmo32, nmo32, c, h.data(), work.data(), h_mo.data())));
      FBF_TRY(res.sync());
    }
    const OrbitalRange all{.first = 0, .count = nmo32};
    FBF_TRY(b_mo(res, naux32, nmo32, nmo32, b.data(), c, all, all, l_mo.data()));
    // out = 2 (m_h h_bar + sum_Q m_Q Lambda_Q), m_b read as the
    // (naux nmo) x nmo matrix of rows (Q, r).
    const auto x_of = [&](const double *m_h, const double *m_b, double *out) -> Status {
      FBF_TRY(gpu_status(gemm(blas, nmo32, &kTwo<double>, m_h, h_bar.data(), &kZero<double>, out)));
      return gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, nmo32, naux32 * nmo32, &kTwo<double>,
                                                m_b, lambda.data(), &kOne<double>, out));
    };
    Buffers::device<double> x_n(nn, res.device());
    FBF_TRY(x_of(h_mo.data(), l_mo.data(), x_n.data()));
    Buffers::device<double> x_t(nn * k, res.device());
    Buffers::device<double> y(nn * k, res.device());
    Buffers::device<double> v(nn * k, res.device());
    Buffers::device<double> work(nn, res.device());
    Buffers::device<double> kl(naux * nn, res.device());
    for (size_t i = 0; i < k; ++i) {
      const double *const kappa = t_kappa.data() + (i * nn);
      const double *const kappa_nr = t_nr.data() + (i * nn);
      double *const yi = y.data() + (i * nn);
      double *const vi = v.data() + (i * nn);
      FBF_TRY(x_of(t_h.data() + (i * nn), t_b.data() + (i * naux * nn), x_t.data() + (i * nn)));
      // Y_i = kappa^T X_N + 2 h kappa h_bar + 2 sum_Q B_Q kappa Lambda_Q,
      // (kappa Lambda_Q)_rs per slab s as Lambda's [Q, t] slab times kappa^T.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, &kOne<double>, kappa, x_n.data(),
                                                 &kZero<double>, yi)));
      FBF_TRY(gpu_status(
          gemm(blas, nmo32, &kOne<double>, h_mo.data(), kappa, &kZero<double>, work.data())));
      FBF_TRY(gpu_status(
          gemm(blas, nmo32, &kTwo<double>, work.data(), h_bar.data(), &kOne<double>, yi)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, naux32, nmo32, nmo32, &kOne<double>,
          lambda.data(), naux32, static_cast<int64_t>(naux * nmo), kappa, nmo32, 0, &kZero<double>,
          kl.data(), naux32, static_cast<int64_t>(naux * nmo), nmo32)));
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, nmo32, nmo32, naux32 * nmo32, &kTwo<double>,
                                                 l_mo.data(), kl.data(), &kOne<double>, yi)));
      // V_i = X_N kappa_nr - kappa_nr X_N.
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kOne<double>, x_n.data(), kappa_nr, &kZero<double>, vi)));
      FBF_TRY(
          gpu_status(gemm(blas, nmo32, &kNegOne<double>, kappa_nr, x_n.data(), &kOne<double>, vi)));
    }
    // X(I^i) = X(t_i) - Y_i, and kappa_oo = kappa - kappa_nr.
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, k32 * nn32, &kNegOne<double>, y.data(), 1,
                                                  x_t.data(), 1)));
    Buffers::device<double> t_oo(nn * k, res.device());
    FBF_TRY(copy(t_oo.data(), t_kappa.data(), nn * k, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, k32 * nn32, &kNegOne<double>, t_nr.data(),
                                                  1, t_oo.data(), 1)));
    const std::vector<double> g_tau = FBF_TRY(gram(nn32, t_kappa.data(), x_t.data()));
    const std::vector<double> g_kappa = FBF_TRY(gram(nn32, t_kappa.data(), y.data()));
    const std::vector<double> g_oo = FBF_TRY(gram(nn32, t_oo.data(), v.data()));
    for (size_t j = 0; j < k; ++j) {
      for (size_t i = 0; i < k; ++i) {
        const size_t at = i + (k * j);
        const size_t ta = j + (k * i);
        rotation_tau[at] = g_tau[at] + g_tau[ta];
        rotation_kappa[at] = 0.5 * (g_kappa[at] + g_kappa[ta]);
        z_koo[at] = 0.5 * (g_oo[at] + g_oo[ta]);
      }
    }
  }

  // 8. The parts in the coordinates, summed and symmetrized.
  std::vector<double> e_cas = to_coordinates(casscf.hessian);
  std::vector<double> z_lxy = to_coordinates(z_terms->lxy.exy);
  std::vector<double> z_lx = to_coordinates(lx_cart);
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < j; ++i) {
      const double s = z_lx[i + (k * j)] + z_lx[j + (k * i)];
      z_lx[i + (k * j)] = s;
      z_lx[j + (k * i)] = s;
    }
    z_lx[j * (k + 1)] *= 2.0;
  }
  std::vector<double> z_third = to_coordinates(z_terms->third);
  std::vector<double> skeleton = to_coordinates(skeleton_cart);
  std::vector<double> total(k * k);
  double asymmetry = 0.0;
  const auto element = [&](const size_t at) {
    return e_cas[at] + z_lxy[at] + z_lx[at] + z_third[at] + z_koo[at] + skeleton[at] + hvp[at] +
           rotation_tau[at] + rotation_kappa[at];
  };
  for (size_t j = 0; j < k; ++j) {
    for (size_t i = 0; i < k; ++i) {
      const double upper = element(i + (k * j));
      const double lower = element(j + (k * i));
      asymmetry = std::max(asymmetry, std::abs(upper - lower));
      total[i + (k * j)] = 0.5 * (upper + lower);
    }
  }
  // The translational sum rule, on a full set of coordinates only: the
  // Cartesian Hessian Q^T H Q against each rigid translation.
  double translation = std::numeric_limits<double>::quiet_NaN();
  if (k == n_x) {
    translation = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
      std::vector<double> qt_axis(k, 0.0); // Q T_axis
      for (size_t i = 0; i < k; ++i) {
        for (size_t a = 0; a < n_x / 3; ++a) {
          qt_axis[i] += q[(i * n_x) + (3 * a) + axis];
        }
      }
      std::vector<double> hq(k, 0.0); // H_q Q T_axis
      for (size_t j = 0; j < k; ++j) {
        for (size_t i = 0; i < k; ++i) {
          hq[i] += total[i + (k * j)] * qt_axis[j];
        }
      }
      for (size_t x = 0; x < n_x; ++x) { // Q^T of it
        double s = 0.0;
        for (size_t i = 0; i < k; ++i) {
          s += q[(i * n_x) + x] * hq[i];
        }
        translation = std::max(translation, std::abs(s));
      }
    }
  }
  return ScNevpt2Terms{.hessian = std::move(total),
                       .asymmetry = asymmetry,
                       .translation = translation,
                       .e_cas = std::move(e_cas),
                       .z_lxy = std::move(z_lxy),
                       .z_lx = std::move(z_lx),
                       .z_third = std::move(z_third),
                       .z_koo = std::move(z_koo),
                       .skeleton = std::move(skeleton),
                       .hvp = std::move(hvp),
                       .rotation_tau = std::move(rotation_tau),
                       .rotation_kappa = std::move(rotation_kappa),
                       .casscf = std::move(casscf),
                       .e_corr = gradient.energies.total,
                       .max_zc = gradient.zc.max_z,
                       .min_gap = gradient.zc.min_gap,
                       .zvector_residual = gradient.multipliers.residual,
                       .defect = defect,
                       .mix_step = mix_step};
}

} // namespace free_breakfast::hessian
