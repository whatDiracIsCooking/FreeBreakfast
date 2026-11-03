/// @file
/// @brief qd_sc_energy, qd_pc_energy, qd_sc_adjoint and qd_pc_adjoint: the
/// SC- and PC-QD-NEVPT2 effective Hamiltonians' correlation elements over
/// (h, B, C) and the roots, one set of shared inputs over the state-averaged
/// Fock and every ordered pair's transition RDMs, and the reverse pass of
/// E[W] = Σ_MN W_MN H_MN through sc_ or pc_effective_hamiltonian_adjoint,
/// the RDMs' adjoints and the shared builders'.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.nevpt2.rdm;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;
import :operands; // column_operand, contract, zeroed, add_onto, check_weights
import :builders; // ScArrays, the builders, their runs and builders_adjoint

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// One ordered pair's RDMs, row-major in nevpt2::rdms' layout: dm1, dm2 and
// dm3, and f3ca and f3ac on the diagonal alone (empty off it). Or the bars on
// a root's own.
struct QdRdms {
  Buffers::device<double> dm1;
  Buffers::device<double> dm2;
  Buffers::device<double> dm3;
  Buffers::device<double> f3ca;
  Buffers::device<double> f3ac;
};

// What both entry points build: the shared arrays, their dm1 gamma_SA, and
// the n^2 pairs' RDMs, [M n + N] that of bra M and ket N.
struct QdArrays {
  ScArrays x;
  std::vector<QdRdms> pairs;
};

// A zeroed QdRdms over @p ncas active orbitals, with f3ca and f3ac if @p f3.
Result<QdRdms> zeroed_rdms(DeviceResources &res, const int64_t ncas, const bool f3) {
  const int64_t n4 = ncas * ncas * ncas * ncas;
  const int64_t n6 = f3 ? n4 * ncas * ncas : 0;
  return QdRdms{.dm1 = FBF_TRY(zeroed(res, ncas * ncas)),
                .dm2 = FBF_TRY(zeroed(res, n4)),
                .dm3 = FBF_TRY(zeroed(res, n4 * ncas * ncas)),
                .f3ca = FBF_TRY(zeroed(res, n6)),
                .f3ac = FBF_TRY(zeroed(res, n6))};
}

// shared_arrays; every ordered pair's dm3 by nevpt2::rdms (one vector on the
// diagonal, with its f3s; bra/ket off it), and its dm2 and dm1 as traces;
// gamma_SA = Σ_r w_r dm1_rr into x.dm1; then the builders that read it.
Result<QdArrays> qd_arrays(DeviceResources &res, const std::string_view who,
                           const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                           const int32_t naux, const int32_t nao, const int32_t nmo,
                           const double *b, const double *c, const double *h, const int32_t ncore,
                           const int32_t ncas, const std::span<const double> weights,
                           const double *vectors) {
  FBF_TRY(check_weights(who, weights));
  ScArrays x = FBF_TRY(shared_arrays(res, who, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas));
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto nroots = static_cast<int64_t>(weights.size());
  const int64_t n = ncas;
  const int64_t n6 = n * n * n * n * n * n;
  const int64_t ndet = int64_t{alpha.size()} * beta.size();
  // The transition f3s, built and dropped: QD reads f3 on the diagonal alone,
  // and the bra/ket rdms cannot skip them yet (#719).
  Buffers::device<double> f3_scratch(static_cast<size_t>(2 * n6), res.device());

  std::vector<QdRdms> pairs;
  pairs.reserve(static_cast<size_t>(nroots * nroots));
  for (int64_t bra = 0; bra < nroots; ++bra) {
    for (int64_t ket = 0; ket < nroots; ++ket) {
      QdRdms p = FBF_TRY(zeroed_rdms(res, n, bra == ket));
      if (bra == ket) {
        FBF_TRY(rdms(res, alpha, beta, vectors + (ket * ndet), x.eri.data(), p.dm3.data(),
                     p.f3ca.data(), p.f3ac.data(), FBF_TRY(rdm_budget())));
      } else {
        FBF_TRY(rdms(res, alpha, beta, vectors + (bra * ndet), vectors + (ket * ndet), x.eri.data(),
                     nullptr, nullptr, p.dm3.data(), f3_scratch.data(), f3_scratch.data() + n6,
                     FBF_TRY(rdm_budget())));
      }
      FBF_TRY(run_builds(res, trace_builds(x, p.dm1.data(), p.dm2.data(), p.dm3.data(), nullptr,
                                           nullptr, nullptr)));
      pairs.push_back(std::move(p));
    }
  }
  for (int64_t r = 0; r < nroots; ++r) {
    FBF_TRY(
        contract(res, "tu->tu",
                 {column_operand(pairs[static_cast<size_t>(r * (nroots + 1))].dm1.data(), {n, n})},
                 x.dm1.data(), {n, n}, weights[static_cast<size_t>(r)]));
  }
  const std::vector<FockBuild> builds = fock_builds(x, nullptr);
  FBF_TRY(run_builds(res, std::span(builds).subspan(1)));
  return QdArrays{.x = std::move(x), .pairs = std::move(pairs)};
}

// sc_effective_hamiltonian's pairs: the shared inputs, each pair's RDMs, and
// f3ca and f3ac on the diagonal alone.
std::vector<EnergyInputs> pair_inputs(const QdArrays &q, const size_t nroots) {
  std::vector<EnergyInputs> out;
  for (size_t k = 0; k < q.pairs.size(); ++k) {
    const QdRdms &p = q.pairs[k];
    const bool diagonal = k % (nroots + 1) == 0;
    EnergyInputs in = q.x.inputs();
    in.dm1 = p.dm1.data();
    in.dm2 = p.dm2.data();
    in.dm3 = p.dm3.data();
    in.f3ca = diagonal ? p.f3ca.data() : nullptr;
    in.f3ac = diagonal ? p.f3ac.data() : nullptr;
    out.push_back(in);
  }
  return out;
}

Status
with_qd_inputs(DeviceResources &res, const std::string_view who, const ci::DeviceStringSet &alpha,
               const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
               const int32_t nmo, const double *b, const double *c, const double *h,
               const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
               const double *vectors,
               const std::function<Status(const MoBlocks &, std::span<const EnergyInputs>)> &use) {
  const QdArrays q = FBF_TRY(
      qd_arrays(res, who, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors));
  return use(q.x.blocks, pair_inputs(q, weights.size()));
}

Result<std::vector<ClassEnergies>>
qd_sc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
             const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
             const int32_t nmo, const double *b, const double *c, const double *h,
             const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
             const double *vectors, const int32_t batch) {
  const QdArrays q = FBF_TRY(qd_arrays(res, "nevpt2::qd_sc_energy", alpha, beta, naux, nao, nmo, b,
                                       c, h, ncore, ncas, weights, vectors));
  return sc_effective_hamiltonian(res, q.x.blocks, pair_inputs(q, weights.size()), batch);
}

Result<std::vector<PcClassEnergies>>
qd_pc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
             const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
             const int32_t nmo, const double *b, const double *c, const double *h,
             const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
             const double *vectors, const int32_t batch) {
  const QdArrays q = FBF_TRY(qd_arrays(res, "nevpt2::qd_pc_energy", alpha, beta, naux, nao, nmo, b,
                                       c, h, ncore, ncas, weights, vectors));
  return pc_effective_hamiltonian(res, q.x.blocks, pair_inputs(q, weights.size()), batch);
}

// The reverse pass of E[W] around @p walk, the effective Hamiltonian's
// adjoint at H̄ = W (SC's or PC's), which adds onto the bars it is handed and
// hands each transition pair to the callback it is given: the RDMs' bars
// onto every root's y, then builders_adjoint. The energies are @p walk's to
// keep.
Result<ScAdjoint>
qd_adjoint(DeviceResources &res, const std::string_view who, const ci::DeviceStringSet &alpha,
           const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
           const int32_t nmo, const double *b, const double *c, const double *h,
           const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
           const double *vectors, const std::span<const double> w,
           const std::function<
               Status(const MoBlocks &, std::span<const EnergyInputs>, std::span<const EnergyBars>,
                      const std::function<Status(int64_t, int64_t, const EnergyBars &)> &)> &walk) {
  FBF_TRY(check_weights(who, weights));
  const size_t nroots = weights.size();
  if (w.size() != nroots * nroots) {
    return err_config("{}: W holds {} elements for {} roots: n^2", who, w.size(), nroots);
  }
  QdArrays q = FBF_TRY(
      qd_arrays(res, who, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors));
  ScArrays &x = q.x;
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t n = ncas;
  const int64_t n4 = n * n * n * n;
  const int64_t ndet = int64_t{alpha.size()} * beta.size();

  // The shared bars, zeroed, Λ as sc_adjoint's; bar.dm1 is gamma_SA's. Each
  // root's own RDMs' bars apart, and y over every root.
  ScArrays bar = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nmo - ncore - ncas));
  Buffers::device<double> lambda_raw = FBF_TRY(zeroed(res, naux * int64_t{nmo} * nmo));
  std::vector<QdRdms> own;
  std::vector<EnergyBars> bars;
  for (size_t r = 0; r < nroots; ++r) {
    own.push_back(FBF_TRY(zeroed_rdms(res, n, true)));
    EnergyBars e = energy_bars(bar, lambda_raw.data());
    e.dm1 = own.back().dm1.data();
    e.dm2 = own.back().dm2.data();
    e.dm3 = own.back().dm3.data();
    e.f3ca = own.back().f3ca.data();
    e.f3ac = own.back().f3ac.data();
    bars.push_back(e);
  }
  Buffers::device<double> y = FBF_TRY(zeroed(res, static_cast<int64_t>(nroots) * ndet));
  // rdms_adjoint overwrites its outputs, so each call writes these and they
  // are added on.
  Buffers::device<double> bra_bar(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> ket_bar(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> eri_bar(static_cast<size_t>(n4), res.device());

  // Each transition's bars as the walk hands them over: dm1's and dm2's
  // folded into dm3's, then the bra/ket rdms_adjoint with no f3 bars onto both
  // roots' y. (tu|vw) takes none: dm3 does not read it.
  const auto transition = [&](const int64_t bra, const int64_t ket, const EnergyBars &t) -> Status {
    QdRdms &p = q.pairs[static_cast<size_t>((bra * static_cast<int64_t>(nroots)) + ket)];
    FBF_TRY(run_builds_adjoint(
        res, trace_builds(x, p.dm1.data(), p.dm2.data(), p.dm3.data(), t.dm1, t.dm2, t.dm3)));
    FBF_TRY(rdms_adjoint(res, alpha, beta, vectors + (bra * ndet), vectors + (ket * ndet),
                         x.eri.data(), t.dm3, nullptr, nullptr, bra_bar.data(), ket_bar.data(),
                         eri_bar.data(), FBF_TRY(rdm_budget())));
    FBF_TRY(add_onto(res, ndet, 1, bra_bar.data(), y.data() + (bra * ndet), ndet));
    FBF_TRY(add_onto(res, ndet, 1, ket_bar.data(), y.data() + (ket * ndet), ndet));
    return {};
  };
  // A refused class is the walk's error, returned before any bar is read.
  FBF_TRY(walk(x.blocks, pair_inputs(q, nroots), bars, transition));

  // Each root's own: gamma_SA's bar at its weight onto dm1's, the traces'
  // adjoint, then the one-vector rdms_adjoint onto its y and (tu|vw)'s bar.
  const auto own_rdms_adjoint = [&]() -> Result<Buffers::device<double>> {
    for (size_t r = 0; r < nroots; ++r) {
      QdRdms &p = q.pairs[r * (nroots + 1)];
      QdRdms &o = own[r];
      const auto root = static_cast<int64_t>(r);
      FBF_TRY(contract(res, "tu->tu", {column_operand(bar.dm1.data(), {n, n})}, o.dm1.data(),
                       {n, n}, weights[r]));
      FBF_TRY(run_builds_adjoint(res, trace_builds(x, p.dm1.data(), p.dm2.data(), p.dm3.data(),
                                                   o.dm1.data(), o.dm2.data(), o.dm3.data())));
      FBF_TRY(rdms_adjoint(res, alpha, beta, vectors + (root * ndet), x.eri.data(), o.dm3.data(),
                           o.f3ca.data(), o.f3ac.data(), bra_bar.data(), eri_bar.data(),
                           FBF_TRY(rdm_budget())));
      FBF_TRY(add_onto(res, ndet, 1, bra_bar.data(), y.data() + (root * ndet), ndet));
      FBF_TRY(add_onto(res, n4, 1, eri_bar.data(), bar.eri.data(), n4));
    }
    return std::move(y);
  };
  return builders_adjoint(res, naux, nao, nmo, b, c, ncore, ncas, x, bar, std::move(lambda_raw),
                          own_rdms_adjoint);
}

Result<QdScAdjoint> qd_sc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                  const ci::DeviceStringSet &beta, const int32_t naux,
                                  const int32_t nao, const int32_t nmo, const double *b,
                                  const double *c, const double *h, const int32_t ncore,
                                  const int32_t ncas, const std::span<const double> weights,
                                  const double *vectors, const std::span<const double> w,
                                  const int32_t batch) {
  std::vector<ClassEnergies> elements;
  ScAdjoint adj = FBF_TRY(
      qd_adjoint(res, "nevpt2::qd_sc_adjoint", alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                 weights, vectors, w,
                 [&](const MoBlocks &blocks, const std::span<const EnergyInputs> pairs,
                     const std::span<const EnergyBars> bars, const auto &transition) -> Status {
                   auto walked = FBF_TRY(sc_effective_hamiltonian_adjoint(res, blocks, pairs, w,
                                                                          batch, bars, transition));
                   elements = std::move(walked.elements);
                   return {};
                 }));
  return QdScAdjoint{.elements = std::move(elements),
                     .h_bar = std::move(adj.h_bar),
                     .lambda = std::move(adj.lambda),
                     .y = std::move(adj.y),
                     .x = std::move(adj.x),
                     .a = std::move(adj.a),
                     .epsilon = std::move(adj.epsilon)};
}

Result<QdPcAdjoint> qd_pc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                  const ci::DeviceStringSet &beta, const int32_t naux,
                                  const int32_t nao, const int32_t nmo, const double *b,
                                  const double *c, const double *h, const int32_t ncore,
                                  const int32_t ncas, const std::span<const double> weights,
                                  const double *vectors, const std::span<const double> w,
                                  const int32_t batch) {
  std::vector<PcClassEnergies> elements;
  ScAdjoint adj = FBF_TRY(
      qd_adjoint(res, "nevpt2::qd_pc_adjoint", alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                 weights, vectors, w,
                 [&](const MoBlocks &blocks, const std::span<const EnergyInputs> pairs,
                     const std::span<const EnergyBars> bars, const auto &transition) -> Status {
                   auto walked = FBF_TRY(pc_effective_hamiltonian_adjoint(res, blocks, pairs, w,
                                                                          batch, bars, transition));
                   elements = std::move(walked.elements);
                   return {};
                 }));
  return QdPcAdjoint{.elements = std::move(elements),
                     .h_bar = std::move(adj.h_bar),
                     .lambda = std::move(adj.lambda),
                     .y = std::move(adj.y),
                     .x = std::move(adj.x),
                     .a = std::move(adj.a),
                     .epsilon = std::move(adj.epsilon)};
}

} // namespace free_breakfast::nevpt2
