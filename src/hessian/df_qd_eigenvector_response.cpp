/// @file
/// @brief df_qd_eigenvector_response: term (ii) of a QD-NEVPT2 state's
/// Hessian, the eigenvector response, from the interstate couplings h_PQ
/// alone. H_eff and R once over nevpt2::qd_inputs, every pair's gap checked
/// before any adjoint is built, then one several-W gradient over every
/// trace-zero W^PQ (one Z-vector solve), and the closed forms on the host;
/// PC's at its pc_floor throughout.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.root_rotation;    // kRootGapFloor
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;
import free_breakfast.nevpt2.gradient; // qd_inputs, the several-W forms
import :state_specific;                // qd_response_eigen

#include "diagnostics.h"

namespace free_breakfast::hessian {

// The partition's: h_PQ's sign is R's.
Result<std::pair<std::vector<double>, std::vector<double>>>
qd_response_eigen(DeviceResources &res, const std::span<const double> energies,
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

Result<QdEigenvectorResponse>
df_qd_eigenvector_response(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                           const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                           const SphericalMatrixAO &h, const Buffers::device<double> &b,
                           const double *l, const mcscf::Solution &solution, const int32_t ncore,
                           const int32_t ncas, const bool pc, const int32_t batch,
                           const double zvector_tolerance, const double pc_floor,
                           const std::optional<std::array<int32_t, 2>> pair) {
  constexpr std::string_view kWho = "hessian::df_qd_eigenvector_response";
  if (!(zvector_tolerance > 0.0)) {
    return err_config("{}: a Z-vector tolerance of {:g}", kWho, zvector_tolerance);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t n = solution.roots.energies.size();
  if (pair && ((*pair)[0] < 0 || (*pair)[1] < 0 || std::cmp_greater_equal((*pair)[0], n) ||
               std::cmp_greater_equal((*pair)[1], n) || (*pair)[0] == (*pair)[1])) {
    return err_config("{}: ({}, {}) is not a pair of two of the {} QD states", kWho, (*pair)[0],
                      (*pair)[1], n);
  }
  // The crossing pair (P, Q) in either order: its gap is never divided by.
  const auto excluded = [&pair](const size_t a, const size_t c) {
    return pair && ((std::cmp_equal(a, (*pair)[0]) && std::cmp_equal(c, (*pair)[1])) ||
                    (std::cmp_equal(a, (*pair)[1]) && std::cmp_equal(c, (*pair)[0])));
  };

  // H_eff's correlation elements; a refused PC class is the walk's error.
  std::vector<double> totals;
  FBF_TRY(nevpt2::qd_inputs(
      res, molecule, basis, aux, h, b, solution, ncore, ncas,
      [&](const nevpt2::MoBlocks &blocks,
          const std::span<const nevpt2::EnergyInputs> pairs) -> Status {
        if (pc) {
          // The pairs at the floor, which the kets' solves read.
          std::vector<nevpt2::EnergyInputs> floored(pairs.begin(), pairs.end());
          for (nevpt2::EnergyInputs &in : floored) {
            in.pc_floor = pc_floor;
          }
          const std::vector<nevpt2::PcClassEnergies> e =
              FBF_TRY(nevpt2::pc_effective_hamiltonian(res, blocks, floored, batch));
          std::ranges::transform(e, std::back_inserter(totals), &nevpt2::PcClassEnergies::total);
        } else {
          const std::vector<nevpt2::ClassEnergies> e =
              FBF_TRY(nevpt2::sc_effective_hamiltonian(res, blocks, pairs, batch));
          std::ranges::transform(e, std::back_inserter(totals), &nevpt2::ClassEnergies::total);
        }
        return {};
      }));
  auto [eigenvalues, r] = FBF_TRY(qd_response_eigen(res, solution.roots.energies, totals));

  // Every pair's gap but the crossing pair's, before any adjoint: term (ii)
  // divides by each.
  double qd_gap = std::numeric_limits<double>::infinity();
  for (size_t p = 0; p < n; ++p) {
    for (size_t q = p + 1; q < n; ++q) {
      if (excluded(p, q)) {
        continue;
      }
      const double gap = std::abs(eigenvalues[q] - eigenvalues[p]);
      if (gap < mcscf::kRootGapFloor) {
        return err_numerical("{}: QD states {} and {} are {:.3e} Eh apart, under {:g}: term (ii) "
                             "divides by their gap, and a degenerate state has no Hessian of "
                             "this form",
                             kWho, p, q, gap, mcscf::kRootGapFloor);
      }
      qd_gap = std::min(qd_gap, gap);
    }
  }

  const size_t nx = 3 * static_cast<size_t>(basis.natoms());
  const size_t n_pairs = n * (n - 1) / 2;
  QdEigenvectorResponse out{.eigenvalues = std::move(eigenvalues),
                            .eigenvectors = std::move(r),
                            .couplings = std::vector<double>(n_pairs * nx, 0.0),
                            .term = std::vector<double>(n * nx * nx, 0.0),
                            .dr = std::vector<double>(n * n * nx, 0.0),
                            .largest = std::vector<double>(n, 0.0),
                            .qd_gap = qd_gap};
  if (n_pairs == 0) {
    return out;
  }

  // W^PQ = 1/2 (R_P R_Q^T + R_Q R_P^T), n x n row-major, for every P < Q,
  // all of them through one several-W gradient (one Z-vector solve).
  std::vector<std::vector<double>> ws;
  ws.reserve(n_pairs);
  for (size_t p = 0; p < n; ++p) {
    for (size_t q = p + 1; q < n; ++q) {
      const double *const u_p = out.eigenvectors.data() + (p * n);
      const double *const u_q = out.eigenvectors.data() + (q * n);
      std::vector<double> w(n * n);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          w[(i * n) + j] = 0.5 * ((u_p[i] * u_q[j]) + (u_q[i] * u_p[j]));
        }
      }
      ws.push_back(std::move(w));
    }
  }
  const auto take = [&](auto gradients) -> Status {
    for (size_t pair = 0; pair < n_pairs; ++pair) {
      const auto &g = gradients[pair];
      if (g.total.size() != nx) {
        return err_config("{}: coupling {} has {} components, not {}", kWho, pair, g.total.size(),
                          nx);
      }
      std::ranges::copy(g.total, out.couplings.begin() + static_cast<std::ptrdiff_t>(pair * nx));
      out.max_residual = std::max(out.max_residual, g.lagrangian.multipliers.residual);
    }
    return {};
  };
  if (pc) {
    FBF_TRY(take(
        FBF_TRY(nevpt2::qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore,
                                       ncas, ws, batch, zvector_tolerance, 0, pc_floor))));
  } else {
    FBF_TRY(
        take(FBF_TRY(nevpt2::qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
                                            ncore, ncas, ws, batch, zvector_tolerance))));
  }

  // Term (ii) and dR_P/dR_x (Eqs. qd-hess and qd-dR) per state, over its
  // pairs: h_PQ = h_QP, and pair (min, max)'s place in lexicographic order.
  const auto pair_of = [n](const size_t a, const size_t c) {
    const size_t lo = std::min(a, c);
    const size_t hi = std::max(a, c);
    return (lo * n) - (lo * (lo + 1) / 2) + (hi - lo - 1);
  };
  std::vector<double> sum(nx * nx, 0.0);
  for (size_t p = 0; p < n; ++p) {
    double *const term = out.term.data() + (p * nx * nx);
    double *const dr = out.dr.data() + (p * n * nx);
    for (size_t q = 0; q < n; ++q) {
      if (q == p || excluded(p, q)) {
        continue;
      }
      const double *const hpq = out.couplings.data() + (pair_of(p, q) * nx);
      const double inverse = 1.0 / (out.eigenvalues[p] - out.eigenvalues[q]);
      const double *const u_q = out.eigenvectors.data() + (q * n);
      for (size_t y = 0; y < nx; ++y) {
        for (size_t x = 0; x < nx; ++x) {
          term[x + (nx * y)] += 2.0 * hpq[x] * hpq[y] * inverse;
        }
      }
      for (size_t x = 0; x < nx; ++x) {
        for (size_t i = 0; i < n; ++i) {
          dr[i + (n * x)] += u_q[i] * hpq[x] * inverse;
        }
      }
    }
    for (size_t e = 0; e < nx * nx; ++e) {
      out.largest[p] = std::max(out.largest[p], std::abs(term[e]));
      sum[e] += term[e];
    }
  }
  for (const double e : sum) {
    out.sum_rule = std::max(out.sum_rule, std::abs(e));
  }
  return out;
}

MemoryFootprint qd_response_eigen_footprint(const int32_t nroots,
                                            const std::span<const size_t> syevd_lwork) {
  const auto n = static_cast<size_t>(nroots);
  const MemoryFootprint held{.device = {.persistent = sizeof(double) * ((n * n) + n)}};
  return drop(n < syevd_lwork.size() ? then(held, diagonalize_footprint<double>(syevd_lwork[n]))
                                     : held);
}

MemoryFootprint qd_eigenvector_response_footprint(
    const BasisSet &basis, const BasisSet &aux, const int32_t ncore, const int32_t ncas,
    const int32_t nalpha, const int32_t nbeta, const int32_t nroots, const bool rotating,
    const int32_t batch, const std::span<const size_t> syevd_lwork, const bool pc,
    const int32_t n_w, const std::span<const std::array<int64_t, 8>> dropped) {
  const auto nmo = static_cast<int32_t>(basis.nao_sphe());
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const int32_t nelecas = nalpha + nbeta;
  const int32_t multiplicity = nalpha - nbeta + 1;
  const int32_t nvirt = nmo - ncore - ncas;
  const int32_t transitions = nroots * (nroots - 1);
  Ledger l;
  // H_eff's elements over qd_inputs, and its eigenpairs.
  const MemoryFootprint energy =
      pc ? nevpt2::pc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nroots, true, transitions,
                                                      batch, syevd_lwork, false, dropped)
         : nevpt2::sc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nelecas, nroots, true,
                                                      transitions, batch);
  l.run(nevpt2::qd_inputs_footprint(naux, nmo, ncore, ncas, nelecas, multiplicity, nroots, energy,
                                    0));
  l.run(qd_response_eigen_footprint(nroots, syevd_lwork));
  // The couplings' several-W gradient: at most n_w times one trace-zero W's.
  if (n_w > 0) {
    MemoryFootprint one =
        pc ? nevpt2::qd_pc_w_gradient_footprint(basis, aux, ncore, ncas, nelecas, multiplicity,
                                                nroots, rotating, batch, syevd_lwork, true, true,
                                                dropped)
           : nevpt2::qd_sc_w_gradient_footprint(basis, aux, ncore, ncas, nelecas, multiplicity,
                                                nroots, rotating, batch, syevd_lwork, true, true);
    const auto w = static_cast<size_t>(n_w);
    one.device.persistent *= w;
    one.device.scratch *= w;
    l.run(one);
  }
  return l.footprint(0);
}

} // namespace free_breakfast::hessian
