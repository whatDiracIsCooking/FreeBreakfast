/// @file
/// @brief zc_multipliers and zc_adjoint: the inactive-rotation multipliers
/// z^c of a correlation energy's Lagrangian, one division per pair, and the
/// term Phi = <z, f> they add with its adjoints, over B whole in the MO basis.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kZero, kNegHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, gemmStridedBatched, geam, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.einsum;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import :operands; // column_operand, contract, zeroed, add_onto, to_host, to_device, check_counts

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// Below this gap (Eh) a core or virtual pair is degenerate and its z^c is not
// a division the state determines. An abelian subgroup splits a degenerate
// set across irreps, the two partners keep one orbital energy, and the energy
// does not move along a rotation mixing them, so A vanishes too and zero is
// the right representative. The demo's PC_GAP_FLOOR (reference_data/
// nevpt2_grad.py; docs/nuclear_gradients.tex, "Degenerate (O2) pairs").
constexpr double kZcGapFloor = 1e-6;
// What a degenerate pair's |A_pq| must be under for that zero to stand; past
// it the pair is refused. The demo's PC_ZERO_TOL.
constexpr double kZcZeroSource = 1e-8;

Result<ZcMultipliers> zc_multipliers(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                                     const int32_t ncas, const double *a, const double *epsilon) {
  if (nmo <= 0 || ncore < 0 || ncas < 0 || ncore + ncas > nmo) {
    return err_config("nevpt2::zc_multipliers: nmo {}, ncore {} and ncas {} do not fit: nmo "
                      "positive, the others not negative, and ncore + ncas <= nmo",
                      nmo, ncore, ncas);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n = static_cast<size_t>(nmo);
  const std::vector<double> source = FBF_TRY(to_host(res, a, n * n));
  const std::vector<double> eps = FBF_TRY(to_host(res, epsilon, n));
  std::vector<double> z(n * n, 0.0);
  double max_z = 0.0;
  double min_gap = std::numeric_limits<double>::infinity();
  const auto nc = static_cast<size_t>(ncore);
  const auto nocc = static_cast<size_t>(ncore + ncas);
  // Along kappa_pq = -kappa_qp = t, p > q, f_pq and f_qp each move by
  // t (eps_p - eps_q) where f is diagonal, and nothing else z reads moves at
  // first order, so d/dt (E + <z, f>) = A_pq + 2 z_pq (eps_p - eps_q).
  for (const auto &[first, last] : {std::pair{size_t{0}, nc}, std::pair{nocc, n}}) {
    for (size_t p = first; p < last; ++p) {
      for (size_t q = first; q < p; ++q) {
        const double gap = eps[p] - eps[q];
        const double a_pq = source[p + (n * q)];
        min_gap = std::min(min_gap, std::abs(gap));
        if (std::abs(gap) < kZcGapFloor) {
          if (std::abs(a_pq) > kZcZeroSource) {
            return err_numerical(
                "nevpt2::zc_multipliers: orbitals {} and {} are {:.2e} Eh apart (floor {:g}) "
                "but |A_pq| is {:.2e} (round-off {:g}): their z^c is not determined by the "
                "state, and dividing through would manufacture one",
                p, q, std::abs(gap), kZcGapFloor, std::abs(a_pq), kZcZeroSource);
          }
          continue;
        }
        const double value = -a_pq / (2.0 * gap);
        z[p + (n * q)] = value;
        z[q + (n * p)] = value;
        max_z = std::max(max_z, std::abs(value));
      }
    }
  }
  return ZcMultipliers{.z = FBF_TRY(to_device(res, z)), .max_z = max_z, .min_gap = min_gap};
}

Result<ZcAdjoint> zc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
                             const int32_t nmo, const double *b, const double *c, const double *h,
                             const int32_t ncore, const int32_t ncas, const double *vector,
                             const double *z) {
  FBF_TRY(check_counts("nevpt2::zc_adjoint", alpha, beta, naux, nao, nmo, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  // B in the MO basis, (L, p, q) at L + naux (p + nmo q), read as the
  // (naux nmo) x nmo matrix of rows (L, p), leading dimension lb.
  const int64_t lb = int64_t{naux} * nmo;

  // gamma first: rdm1's scratch is the workspace, which the regions below
  // take after it.
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  Buffers::device<double> gamma_d(static_cast<size_t>(n * n), res.device());
  FBF_TRY(sigma.rdm1(res, vector, gamma_d.data()));
  const std::vector<double> gamma =
      FBF_TRY(to_host(res, gamma_d.data(), static_cast<size_t>(n * n)));
  // D: 2 on the core's diagonal, gamma on the active block.
  std::vector<double> dens_host(static_cast<size_t>(nmo2), 0.0);
  for (int64_t i = 0; i < nc; ++i) {
    dens_host[static_cast<size_t>(i * (nmo + 1))] = 2.0;
  }
  for (int64_t t = 0; t < n; ++t) {
    for (int64_t u = 0; u < n; ++u) {
      dens_host[static_cast<size_t>((nc + t) + (nmo * (nc + u)))] =
          gamma[static_cast<size_t>((t * n) + u)];
    }
  }
  const Buffers::device<double> dens = FBF_TRY(to_device(res, dens_host));

  Buffers::device<double> h_mo = FBF_TRY(zeroed(res, nmo2));
  {
    double *const work = std::get<0>(res.device_regions<double>(static_cast<size_t>(nao) * nmo));
    FBF_TRY(gpu_status(gemmwich(res.blas(), nmo, nao, c, h, work, h_mo.data())));
  }
  Buffers::device<double> bmo(static_cast<size_t>(lb * nmo), res.device());
  if (naux > 0) {
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, {.first = 0, .count = nmo}, {.first = 0, .count = nmo},
                 bmo.data()));
  }

  const EinsumOperand b3 = column_operand(bmo.data(), {naux, nmo, nmo});
  const EinsumOperand z2 = column_operand(z, {nmo, nmo});
  const EinsumOperand d2 = column_operand(dens.data(), {nmo, nmo});
  // g_L = sum_rs B_L,rs D_rs and w_L = sum_pq B_L,pq z_pq: the Coulomb
  // contractions of the two densities.
  Buffers::device<double> g = FBF_TRY(zeroed(res, naux));
  Buffers::device<double> w = FBF_TRY(zeroed(res, naux));
  FBF_TRY(contract(res, "Lrs,rs->L", {b3, d2}, g.data(), {naux}, 1.0));
  FBF_TRY(contract(res, "Lrs,rs->L", {b3, z2}, w.data(), {naux}, 1.0));
  const EinsumOperand g1 = column_operand(g.data(), {naux});
  const EinsumOperand w1 = column_operand(w.data(), {naux});

  // f = h + sum_L g_L B_L - 1/2 sum_L B_L D B_L.
  Buffers::device<double> f = FBF_TRY(zeroed(res, nmo2));
  FBF_TRY(add_onto(res, nmo, nmo, h_mo.data(), f.data(), nmo));
  FBF_TRY(contract(res, "Lpq,L->pq", {b3, g1}, f.data(), {nmo, nmo}, 1.0));

  // Lambda = z g_L + D w_L - 1/2 (z B_L D + D B_L z), and the active block of
  // W = dPhi/dD = sum_L w_L B_L - 1/2 sum_L B_L z B_L.
  Buffers::device<double> lambda = FBF_TRY(zeroed(res, lb * nmo));
  Buffers::device<double> w_act = FBF_TRY(zeroed(res, n * n));
  const EinsumOperand b_act{.data = bmo.data() + (naux * (nc + (nmo * nc))),
                            .dims = {naux, n, n},
                            .strides = {1, naux, lb}};
  FBF_TRY(contract(res, "L,Ltu->tu", {w1, b_act}, w_act.data(), {n, n}, 1.0));
  FBF_TRY(contract(res, "pq,L->Lpq", {z2, g1}, lambda.data(), {naux, nmo, nmo}, 1.0));
  FBF_TRY(contract(res, "pq,L->Lpq", {d2, w1}, lambda.data(), {naux, nmo, nmo}, 1.0));
  if (naux > 0) {
    // Two naux nmo^2 regions, held across nothing that takes the workspace:
    // gemm, gemmStridedBatched and einsum take none.
    const auto [left, prod] =
        res.device_regions<double>(static_cast<size_t>(lb * nmo), static_cast<size_t>(lb * nmo));
    // M (L, r, q) = sum_s M_s B_L,sq for a symmetric M_s: per q, the naux x
    // nmo block B(., ., q) times M, so a left product per L is one
    // strided-batched gemm over q.
    const auto left_product = [&](const double *m, double *out) {
      return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, nmo, nmo, &kOne<double>,
          bmo.data(), naux, lb, m, nmo, 0, &kZero<double>, out, naux, lb, nmo));
    };
    // (D B_L) into left; f -= 1/2 sum_{L,r} B_L,rp (D B_L)_rq, B read as
    // (L, r) x p by its symmetry in the pair.
    FBF_TRY(left_product(dens.data(), left));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, nmo, static_cast<int32_t>(lb),
                                               &kNegHalf<double>, bmo.data(), left, &kOne<double>,
                                               f.data())));
    // (D B_L z) = (z B_L D)^T per L: Lambda's exchange half.
    FBF_TRY(gpu_status(gemm(res.blas(), static_cast<int32_t>(lb), nmo, nmo, &kOne<double>, left, z,
                            &kZero<double>, prod)));
    const EinsumOperand m3 = column_operand(prod, {naux, nmo, nmo});
    FBF_TRY(contract(res, "Lpq->Lpq", {m3}, lambda.data(), {naux, nmo, nmo}, -0.5));
    FBF_TRY(contract(res, "Lqp->Lpq", {m3}, lambda.data(), {naux, nmo, nmo}, -0.5));
    // (z B_L) into left; W_tu -= 1/2 sum_{L,r} B_L,rt (z B_L)_ru, the active
    // columns of both.
    FBF_TRY(left_product(z, left));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(
        res.blas(), static_cast<int32_t>(n), static_cast<int32_t>(n), static_cast<int32_t>(lb),
        &kNegHalf<double>, bmo.data() + (lb * nc), left + (lb * nc), &kOne<double>, w_act.data())));
  }

  // Phi = <z, f>.
  Buffers::device<double> phi_d = FBF_TRY(zeroed(res, 1));
  FBF_TRY(
      contract(res, "pq,pq->", {z2, column_operand(f.data(), {nmo, nmo})}, phi_d.data(), {}, 1.0));
  const double phi = FBF_TRY(to_host(res, phi_d.data(), 1))[0];

  // X = 2 h z + 2 sum_{L,q} B_L,qx Lambda_L,qp, and A = X - X^T.
  Buffers::device<double> x = FBF_TRY(zeroed(res, nmo2));
  FBF_TRY(
      gpu_status(gemm(res.blas(), nmo, &kTwo<double>, h_mo.data(), z, &kOne<double>, x.data())));
  if (naux > 0) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, nmo, static_cast<int32_t>(lb),
                                               &kTwo<double>, bmo.data(), lambda.data(),
                                               &kOne<double>, x.data())));
  }
  Buffers::device<double> a(static_cast<size_t>(nmo2), res.device());
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo, nmo, &kOne<double>, x.data(),
      nmo, &kNegOne<double>, x.data(), nmo, a.data(), nmo)));
  Buffers::device<double> h_bar = FBF_TRY(zeroed(res, nmo2));
  FBF_TRY(add_onto(res, nmo, nmo, z, h_bar.data(), nmo));

  // y^c = sum_tu (W + W^T)_tu (E_tu c): dgamma_tu/dc = (E_tu + E_ut) c. One
  // sigma over that one-body operator, no two-electron part; its scratch is
  // the workspace, whose regions above are dead by now.
  const std::vector<double> w_host =
      FBF_TRY(to_host(res, w_act.data(), static_cast<size_t>(n * n)));
  std::vector<double> one_body(static_cast<size_t>(n * n));
  for (int64_t t = 0; t < n; ++t) {
    for (int64_t u = 0; u < n; ++u) {
      // Row-major (t, u), from the column-major W.
      one_body[static_cast<size_t>((t * n) + u)] =
          w_host[static_cast<size_t>(t + (n * u))] + w_host[static_cast<size_t>(u + (n * t))];
    }
  }
  const std::vector<double> no_eri(static_cast<size_t>(n * n * n * n), 0.0);
  FBF_TRY(sigma.set_integrals(res, 0.0, one_body, no_eri));
  Buffers::device<double> y(static_cast<size_t>(sigma.size()), res.device());
  FBF_TRY(sigma.apply(res, 1, vector, y.data()));
  return ZcAdjoint{.phi = phi,
                   .h_bar = std::move(h_bar),
                   .lambda = std::move(lambda),
                   .y = std::move(y),
                   .x = std::move(x),
                   .a = std::move(a)};
}

} // namespace free_breakfast::nevpt2
