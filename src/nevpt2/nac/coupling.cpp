/// @file
/// @brief qd_sc_nac and qd_pc_nac: the QD-NEVPT2 derivative coupling
/// d_PQ = h_PQ / (E_Q - E_P) + d^CAS + d^PT2, over the interstate coupling,
/// the PT2 transition density at T^PQ and mcscf::df_nac's combination.

module;

#include "diagnostics.h"

module free_breakfast.nevpt2.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.nac;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.energy;
import free_breakfast.nevpt2.gradient; // qd_sc_coupling, qd_pc_coupling, qd_inputs

namespace free_breakfast::nevpt2 {

// What qd_nac takes from the interstate coupling, SC's or PC's.
struct QdPair {
  std::vector<double> h;
  std::vector<double> eigenvalues;
  std::vector<double> eigenvectors;
  double heff_gap = 0.0;
  double max_zeta = 0.0;
  int32_t iterations = 0;
  double residual = 0.0;
};

template <typename G>
QdPair qd_pair(G &&g) {
  return QdPair{.h = std::move(g.total),
                .eigenvalues = std::move(g.lagrangian.eigenvalues),
                .eigenvectors = std::move(g.lagrangian.eigenvectors),
                .heff_gap = g.heff_gap,
                .max_zeta = g.max_zeta,
                .iterations = g.lagrangian.multipliers.iterations,
                .residual = g.lagrangian.multipliers.residual};
}

// T^PQ_MN = 1/2 (R_MP R_NQ - R_MQ R_NP) from @p r, R column P at
// [P n, (P + 1) n): exactly antisymmetric, its diagonal exactly zero.
std::vector<double> pair_t(const std::span<const double> r, const size_t n, const size_t up,
                           const size_t uq) {
  std::vector<double> t(n * n);
  for (size_t m = 0; m < n; ++m) {
    for (size_t k = 0; k < n; ++k) {
      t[(m * n) + k] =
          0.5 * ((r[(up * n) + m] * r[(uq * n) + k]) - (r[(uq * n) + m] * r[(up * n) + k]));
    }
  }
  return t;
}

// D^PT2 at @p t, nmo x nmo row-major on the host as df_nac takes it.
Result<std::vector<double>>
pt2_density(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
            const BasisSet &aux, const SphericalMatrixAO &h, const Buffers::device<double> &b,
            const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
            const std::span<const double> t, const int32_t batch, const bool pc) {
  const auto nmo = static_cast<size_t>(solution.c.n());
  std::vector<double> d_x(nmo * nmo);
  FBF_TRY(qd_inputs(
      res, molecule, basis, aux, h, b, solution, ncore, ncas,
      [&](const MoBlocks &blocks, const std::span<const EnergyInputs> pairs) -> Status {
        Result<Buffers::device<double>> density =
            pc ? pc_pt2_transition_density(res, blocks, pairs, t, batch)
               : sc_pt2_transition_density(res, blocks, pairs, t, batch);
        const Buffers::device<double> d = FBF_TRY(std::move(density));
        const ScopedDeviceIndex on{res.dev_idx()};
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_x.data(), d.data(), d_x.size() * sizeof(double),
                                               wwr::wwrMemcpyDeviceToHost, res.stream())));
        return res.sync();
      }));
  return d_x;
}

// total as mcscf::NacOptions asks, from the folded numerator h_PQ + ΔE r
// (@p folded), r the responses, and the CSF parts @p csf: h_PQ / ΔE + r +
// csf by default.
std::vector<double> folded_total(const std::span<const double> folded,
                                 const std::span<const double> csf, const double ediff,
                                 const mcscf::NacOptions &options) {
  std::vector<double> total(folded.size());
  for (size_t i = 0; i < total.size(); ++i) {
    const double rest = options.use_etfs ? 0.0 : csf[i];
    total[i] = options.mult_ediff ? folded[i] + (ediff * rest) : (folded[i] / ediff) + rest;
  }
  return total;
}

// The coupling with its three Z-vectors folded into one (Park 2020 Eqs.
// 51-52): h_PQ's Lagrangian takes ΔE times the combination's RDMs at
// (T^PQ, D^PT2) and its D^X source, so that one solve and its passes give
// h_PQ + ΔE (d^CAS + d^PT2 less their CSF parts). README "The fold".
Result<QdNac> qd_nac_folded(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                            const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                            const SphericalMatrixAO &h, const Buffers::device<double> &b,
                            const double *l, const mcscf::Solution &solution, const int32_t ncore,
                            const int32_t ncas, const int32_t p, const int32_t q,
                            const int32_t batch, const mcscf::NacOptions &options, const bool pc) {
  const size_t n = solution.roots.energies.size();
  const auto up = static_cast<size_t>(p);
  const auto uq = static_cast<size_t>(q);
  const auto nmo = static_cast<size_t>(solution.c.n());
  // What the fold computes beside the Lagrangian's terms.
  mcscf::NacCombinationTerms terms{};
  double ediff = 0.0;
  const auto fold = [&](const std::span<const double> eigenvalues,
                        const std::span<const double> r) -> Result<QdFold> {
    ediff = eigenvalues[uq] - eigenvalues[up];
    const std::vector<double> t = pair_t(r, n, up, uq);
    const std::vector<double> d_x =
        FBF_TRY(pt2_density(res, molecule, basis, aux, h, b, solution, ncore, ncas, t, batch, pc));
    terms = FBF_TRY(mcscf::nac_combination_terms(res, molecule, basis, classes, aux, b, solution,
                                                 ncore, ncas, t, d_x));
    QdFold f{.gamma = terms.gamma, .gamma2 = terms.gamma2, .a = std::vector<double>(nmo * nmo)};
    for (double &v : f.gamma) {
      v *= ediff;
    }
    for (double &v : f.gamma2) {
      v *= ediff;
    }
    // A (column-major) at (x, y) is D^X_xy - D^X_yx, D^X row-major: the
    // combination's D^X_pq - D^X_qp on the packed pairs.
    for (size_t y = 0; y < nmo; ++y) {
      for (size_t x = 0; x < nmo; ++x) {
        f.a[x + (nmo * y)] = ediff * (d_x[(x * nmo) + y] - d_x[(y * nmo) + x]);
      }
    }
    return f;
  };
  QdPair pair;
  if (pc) {
    pair = qd_pair(FBF_TRY(qd_pc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, p, q, batch, fold)));
  } else {
    pair = qd_pair(FBF_TRY(qd_sc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, p, q, batch, fold)));
  }
  std::vector<double> csf(pair.h.size());
  for (size_t i = 0; i < csf.size(); ++i) {
    csf[i] = terms.csf[i] + terms.basis_motion[i];
  }
  std::vector<double> total = folded_total(pair.h, csf, ediff, options);
  return QdNac{.numerator = {},
               .cas = {},
               .pt2 = {},
               .csf = std::move(csf),
               .total = std::move(total),
               .heff_gap = pair.heff_gap,
               .root_gap = terms.gap,
               .max_zeta = pair.max_zeta,
               .iterations = {pair.iterations, 0, 0},
               .residual = {pair.residual, 0.0, 0.0}};
}

// The coupling, @p pc choosing PC's h_PQ and D^PT2 over SC's, @p fold
// qd_nac_folded over the three solves.
Result<QdNac> qd_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                     const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                     const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                     const mcscf::Solution &solution, const int32_t ncore, const int32_t ncas,
                     const int32_t p, const int32_t q, const int32_t batch,
                     const mcscf::NacOptions &options, const bool pc, const bool fold) {
  if (fold) {
    return qd_nac_folded(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q,
                         batch, options, pc);
  }
  // h_PQ, H_eff's eigenvalues and R; every refusal of the pair is its.
  QdPair pair;
  if (pc) {
    pair = qd_pair(FBF_TRY(qd_pc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, p, q, batch)));
  } else {
    pair = qd_pair(FBF_TRY(qd_sc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
                                          ncore, ncas, p, q, batch)));
  }
  const size_t n = solution.roots.energies.size();
  const auto up = static_cast<size_t>(p);
  const auto uq = static_cast<size_t>(q);
  const std::vector<double> t = pair_t(pair.eigenvectors, n, up, uq);
  const std::vector<double> d_x =
      FBF_TRY(pt2_density(res, molecule, basis, aux, h, b, solution, ncore, ncas, t, batch, pc));

  // d^CAS and d^PT2 apart: one Z-vector cannot split its response between
  // T and D^X, so df_nac's combination runs once for each.
  const mcscf::NacCombination cas =
      FBF_TRY(mcscf::df_nac(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, t));
  const std::vector<double> no_t(n * n, 0.0);
  const mcscf::NacCombination pt2 = FBF_TRY(
      mcscf::df_nac(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, no_t, d_x));

  const double ediff = pair.eigenvalues[uq] - pair.eigenvalues[up];
  const size_t count = pair.h.size();
  std::vector<double> csf(count);
  std::vector<double> total(count);
  for (size_t i = 0; i < count; ++i) {
    csf[i] = cas.csf[i] + pt2.basis_motion[i];
    const double rest = cas.total[i] + pt2.total[i] - (options.use_etfs ? csf[i] : 0.0);
    total[i] = options.mult_ediff ? pair.h[i] + (ediff * rest) : (pair.h[i] / ediff) + rest;
  }
  return QdNac{.numerator = pair.h,
               .cas = cas.total,
               .pt2 = pt2.total,
               .csf = std::move(csf),
               .total = std::move(total),
               .heff_gap = pair.heff_gap,
               .root_gap = cas.gap,
               .max_zeta = pair.max_zeta,
               .iterations = {pair.iterations, cas.iterations, pt2.iterations},
               .residual = {pair.residual, cas.residual, pt2.residual}};
}

Result<QdNac> qd_sc_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                        const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                        const SphericalMatrixAO &h, const Buffers::device<double> &b,
                        const double *l, const mcscf::Solution &solution, const int32_t ncore,
                        const int32_t ncas, const int32_t p, const int32_t q, const int32_t batch,
                        const mcscf::NacOptions &options, const bool fold) {
  return qd_nac(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q, batch,
                options, false, fold);
}

Result<QdNac> qd_pc_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                        const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                        const SphericalMatrixAO &h, const Buffers::device<double> &b,
                        const double *l, const mcscf::Solution &solution, const int32_t ncore,
                        const int32_t ncas, const int32_t p, const int32_t q, const int32_t batch,
                        const mcscf::NacOptions &options, const bool fold) {
  return qd_nac(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q, batch,
                options, true, fold);
}

} // namespace free_breakfast::nevpt2
