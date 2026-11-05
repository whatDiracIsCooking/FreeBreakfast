/// @file
/// @brief qd_sc_intersection and qd_pc_intersection: H_eff and R once, three
/// W-form gradients at R's pair, each Z-vector's true residual checked, and
/// the gap sensitivities from their multipliers.

module;

#include "diagnostics.h"

module free_breakfast.nevpt2.intersection;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.root_rotation;    // kRootGapFloor
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;
import free_breakfast.nevpt2.gradient; // qd_inputs, the W forms

namespace free_breakfast::nevpt2 {

// What the intersection reads off one W-form call, SC's or PC's: the
// gradient, the multipliers on the host, the SA weights, and the CASSCF
// roots' smallest gap.
struct WFormPart {
  std::vector<double> total;
  std::vector<double> orbital;
  std::vector<double> ci;
  std::vector<double> weights;
  double root_gap = 0.0;
};

// H_eff = diag(E_r) + 1/2 (C + C^T) from the n x n elements' @p totals
// ([M n + N] bra M, ket N), diagonalized: eigenvalues ascending and R,
// column K at [K n], each column signed so that its largest-magnitude
// component is positive (qd_sc_coupling's convention).
Result<std::pair<std::vector<double>, std::vector<double>>>
heff_eigen(DeviceResources &res, const std::span<const double> energies,
           const std::span<const double> totals) {
  const size_t n = energies.size();
  std::vector<double> heff(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      heff[i + (n * j)] =
          (i == j ? energies[i] : 0.0) + (0.5 * (totals[(i * n) + j] + totals[(j * n) + i]));
    }
  }
  std::vector<double> values(n);
  {
    Buffers::device<double> heff_d(n * n, res.device());
    Buffers::device<double> values_d(n, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(heff_d.data(), heff.data(), n * n * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(n), heff_d.data(), values_d.data()));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(heff.data(), heff_d.data(), n * n * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(values.data(), values_d.data(), n * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
  }
  for (size_t col = 0; col < n; ++col) {
    const std::span<double> v{heff.data() + (col * n), n};
    if (*std::ranges::max_element(v, {}, [](const double x) { return std::abs(x); }) < 0.0) {
      std::ranges::transform(v, v.begin(), std::negate<>{});
    }
  }
  return std::pair{std::move(values), std::move(heff)};
}

// 1/2 (a b^T + b a^T), n x n row-major: at a = b exactly a a^T, and
// symmetric bit for bit (the sum commutes).
std::vector<double> pair_w(const std::span<const double> a, const std::span<const double> b) {
  const size_t n = a.size();
  std::vector<double> w(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      w[(i * n) + j] = 0.5 * ((a[i] * b[j]) + (b[i] * a[j]));
    }
  }
  return w;
}

// @p g's gradient and multipliers on the host, its Z-vector's true residual
// checked against @p tolerance (Numerical naming @p who and @p which W): the
// tolerance the W-form call solved to, and checked against already, so this
// guards the one parameter's path rather than a second bar (the conjugate
// gradients stop on their recurrence, which can report convergence while
// the true residual has not, #786). @p pairs is the packed orbital pairs'
// count.
template <typename G>
Result<WFormPart> w_form_part(DeviceResources &res, const std::string_view who,
                              const std::string_view which, G g, const size_t pairs,
                              const double tolerance) {
  const auto &z = g.lagrangian.multipliers;
  if (!(z.residual <= tolerance)) {
    return err_numerical("{}: the Z-vector of G({}) has true residual {:.3e} after {} products, "
                         "above {:g}",
                         who, which, z.residual, z.iterations, tolerance);
  }
  WFormPart part{.total = std::move(g.total),
                 .orbital = std::vector<double>(pairs),
                 .ci = std::vector<double>(z.ci.num_elements()),
                 .weights = g.lagrangian.weights,
                 .root_gap = g.root_gap};
  if (pairs > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(part.orbital.data(), z.orbital.data(),
                                           pairs * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                           res.stream())));
  }
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(part.ci.data(), z.ci.data(), part.ci.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return part;
}

// sqrt(||b - a||^2 + 4 ||c||^2) over (a, b, c) = (z^P, z^Q, z^PQ), each
// element of block k scaled by scale(k): the gap's sensitivity.
template <typename Scale>
double gap_sensitivity(const std::span<const double> a, const std::span<const double> b,
                       const std::span<const double> c, const Scale &scale) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double s = scale(i);
    const double d = s * (b[i] - a[i]);
    const double e = s * c[i];
    sum += (d * d) + (4.0 * e * e);
  }
  return std::sqrt(sum);
}

// One code path for SC (@p pc false) and PC: H_eff over qd_inputs and the
// method's effective Hamiltonian, R, then the W form at the pair's three W's.
Result<QdIntersection> qd_intersection(
    DeviceResources &res, const std::string_view who, const bool pc, const Molecule &molecule,
    const BasisSet &basis, const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas, const int32_t p,
    const int32_t q, const int32_t batch, const QdIntersectionOptions &options) {
  const size_t nroots = solution.roots.energies.size();
  for (const int32_t state : {p, q}) {
    if (state < 0 || static_cast<size_t>(state) >= nroots) {
      return err_config("{}: state {} of {} roots", who, state, nroots);
    }
  }
  if (p == q) {
    return err_config("{}: states {} and {}: an intersection is between two states", who, p, q);
  }
  const double tolerance = options.zvector_tolerance;
  if (!(tolerance > 0.0)) {
    return err_config("{}: a Z-vector tolerance of {:g}", who, tolerance);
  }

  // H_eff's correlation elements; a refused PC class is the walk's error.
  std::vector<double> totals;
  FBF_TRY(qd_inputs(
      res, molecule, basis, aux, h, b, solution, ncore, ncas,
      [&](const MoBlocks &blocks, const std::span<const EnergyInputs> pairs) -> Status {
        if (pc) {
          const std::vector<PcClassEnergies> e =
              FBF_TRY(pc_effective_hamiltonian(res, blocks, pairs, batch));
          std::ranges::transform(e, std::back_inserter(totals), &PcClassEnergies::total);
        } else {
          const std::vector<ClassEnergies> e =
              FBF_TRY(sc_effective_hamiltonian(res, blocks, pairs, batch));
          std::ranges::transform(e, std::back_inserter(totals), &ClassEnergies::total);
        }
        return {};
      },
      options.nfrozen));
  auto [eigenvalues, r] = FBF_TRY(heff_eigen(res, solution.roots.energies, totals));
  const auto kp = static_cast<size_t>(p);
  const auto kq = static_cast<size_t>(q);
  const std::span<const double> u_p{r.data() + (kp * nroots), nroots};
  const std::span<const double> u_q{r.data() + (kq * nroots), nroots};

  const auto nmo = static_cast<size_t>(solution.c.n());
  const auto core = static_cast<size_t>(ncore);
  const size_t nocc = core + static_cast<size_t>(ncas);
  const size_t pairs = (core * (nmo - core)) + (static_cast<size_t>(ncas) * (nmo - nocc));
  const double heff_gap = std::abs(eigenvalues[kq] - eigenvalues[kp]);

  // Inside a degenerate CASSCF pair (README.md "A MECI inside a degenerate
  // CASSCF pair"): PC, the closest pair of roots (a, b) under the floor, and
  // that pair carrying most of u_P and of u_Q. Then only the mean gradient,
  // one W form at the QD pair's spectral projector over 2, with (a, b)'s
  // zeta at its limit 0 and its tau gated. Anything else under the floor
  // (SC, a pair that is not the QD pair's) takes the path below and is
  // refused there, by root_pair_multipliers.
  const std::vector<double> &energies = solution.roots.energies;
  std::array<size_t, 2> closest{0, 1};
  double root_gap = std::numeric_limits<double>::infinity();
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots; ++s) {
      if (std::abs(energies[s] - energies[r]) < root_gap) {
        root_gap = std::abs(energies[s] - energies[r]);
        closest = {r, s};
      }
    }
  }
  const auto carries = [&](const std::span<const double> u) {
    return (u[closest[0]] * u[closest[0]]) + (u[closest[1]] * u[closest[1]]) > 0.5;
  };
  if (pc && root_gap < mcscf::kRootGapFloor && carries(u_p) && carries(u_q)) {
    std::vector<double> w = pair_w(u_p, u_p);
    const std::vector<double> w_q = pair_w(u_q, u_q);
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = 0.5 * (w[i] + w_q[i]);
    }
    const auto [lo, hi] = closest;
    QdPcGradient g = FBF_TRY(qd_pc_gradient(
        res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, w, batch, tolerance,
        options.nfrozen, std::array{static_cast<int32_t>(lo), static_cast<int32_t>(hi)}));
    // (lo, hi)'s place among the pairs r < s in lexicographic order.
    const double tau = g.lagrangian.pairs.tau[(lo * nroots) - (lo * (lo + 1) / 2) + (hi - lo - 1)];
    const WFormPart mean = FBF_TRY(w_form_part(res, who, "W^mean", std::move(g), pairs, tolerance));
    constexpr double kNone = std::numeric_limits<double>::quiet_NaN();
    return QdIntersection{.g_mean = mean.total,
                          .eigenvalues = std::move(eigenvalues),
                          .eigenvectors = std::move(r),
                          .heff_gap = heff_gap,
                          .root_gap = root_gap,
                          .gap_sensitivity_orbital = kNone,
                          .gap_sensitivity_ci = kNone,
                          .reference_pair_tau = tau};
  }
  const auto part = [&](const std::string_view which,
                        const std::vector<double> &w) -> Result<WFormPart> {
    if (pc) {
      return w_form_part(
          res, who, which,
          FBF_TRY(qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                                 w, batch, tolerance, options.nfrozen)),
          pairs, tolerance);
    }
    return w_form_part(res, who, which,
                       FBF_TRY(qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                              ncore, ncas, w, batch, tolerance, options.nfrozen)),
                       pairs, tolerance);
  };
  constexpr std::array<std::string_view, 3> kWhich{"u_P u_P^T", "u_Q u_Q^T", "W^PQ"};
  const std::array<std::vector<double>, 3> ws{pair_w(u_p, u_p), pair_w(u_q, u_q), pair_w(u_p, u_q)};
  std::vector<WFormPart> parts;
  if (options.shared) {
    // One several-W call: its Z-vector's columns are checked there too.
    const auto take = [&](auto gs) -> Status {
      for (size_t k = 0; k < gs.size(); ++k) {
        parts.push_back(
            FBF_TRY(w_form_part(res, who, kWhich[k], std::move(gs[k]), pairs, tolerance)));
      }
      return {};
    };
    if (pc) {
      FBF_TRY(take(FBF_TRY(qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, ws, batch, tolerance, options.nfrozen))));
    } else {
      FBF_TRY(take(FBF_TRY(qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, ws, batch, tolerance, options.nfrozen))));
    }
  } else {
    for (size_t k = 0; k < ws.size(); ++k) {
      parts.push_back(FBF_TRY(part(kWhich[k], ws[k])));
    }
  }
  WFormPart &gp = parts[0];
  WFormPart &gq = parts[1];
  WFormPart &hpq = parts[2];

  // The CI rows of the Z-vector system are dL/dc_r over 2 w_r, so z^c_r
  // pairs with root r's Davidson residual at 2 w_r (README.md).
  const std::vector<double> &weights = gp.weights;
  const size_t dim = gp.ci.size() / std::max<size_t>(nroots, 1);
  const double s_o =
      gap_sensitivity(gp.orbital, gq.orbital, hpq.orbital, [](size_t) { return 1.0; });
  const double s_c = gap_sensitivity(gp.ci, gq.ci, hpq.ci, [&](const size_t i) {
    return 2.0 * weights[std::min(i / dim, nroots - 1)];
  });
  std::vector<double> g_mean(gp.total.size());
  for (size_t i = 0; i < g_mean.size(); ++i) {
    g_mean[i] = 0.5 * (gp.total[i] + gq.total[i]);
  }
  return QdIntersection{.g_p = std::move(gp.total),
                        .g_q = std::move(gq.total),
                        .h_pq = std::move(hpq.total),
                        .g_mean = std::move(g_mean),
                        .eigenvalues = std::move(eigenvalues),
                        .eigenvectors = std::move(r),
                        .heff_gap = heff_gap,
                        .root_gap = gp.root_gap,
                        .gap_sensitivity_orbital = s_o,
                        .gap_sensitivity_ci = s_c};
}

Result<QdIntersection>
qd_sc_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                   const int32_t p, const int32_t q, const int32_t batch,
                   const QdIntersectionOptions &options) {
  return qd_intersection(res, "nevpt2::qd_sc_intersection", false, molecule, basis, classes, aux, h,
                         b, l, solution, ncore, ncas, p, q, batch, options);
}

Result<QdIntersection>
qd_pc_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                   const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                   const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                   const int32_t p, const int32_t q, const int32_t batch,
                   const QdIntersectionOptions &options) {
  return qd_intersection(res, "nevpt2::qd_pc_intersection", true, molecule, basis, classes, aux, h,
                         b, l, solution, ncore, ncas, p, q, batch, options);
}

} // namespace free_breakfast::nevpt2
