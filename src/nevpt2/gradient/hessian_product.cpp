/// @file
/// @brief sc_hessian_product: the tangent of sc_adjoint + zc_adjoint at
/// fixed z^c along directions (h_dot, B_dot, c_dot) in the MO basis, forward
/// over the reverse pass. The builders' forward tangent feeds
/// class_energies_hvp; its tangent bars then run back through the builders'
/// adjoints, the RDM step and the folds, beside the one-slot-dotted terms
/// those adjoints take at the fixed bars; Phi's share is closed form; and
/// the CI direction is on the unit sphere through the Rayleigh-normalized
/// functional. qd_sc_hessian_product: the same over qd_sc_adjoint at a
/// fixed W, sc_effective_hamiltonian_hvp's tangent bars and every pair's
/// RDM step, with the root set moved by c_r(K) = e^K c_r and its curvature.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, geam, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.core.footprint;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.einsum;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.nevpt2.density_fitting; // MoBlocks, h2e_adjoint
import free_breakfast.nevpt2.rdm;
import free_breakfast.nevpt2.energy;
import :operands; // column_operand, contract, zeroed, add_onto, to_host, to_device, check_counts
import :builders; // ScArrays, the builders, their adjoints and tangents

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// How far |c|^2 may sit from 1, and over a root set C^T C from 1: the
// Rayleigh normalization's and the root set's formulas are taken at unit
// (orthonormal) vectors, and converged CI roots are so to round-off.
constexpr double kUnitVectorTolerance = 1e-10;

// The one-slot-dotted terms of one build's adjoint at fixed bars: per term,
// per operand k that takes a bar, the adjoint's contraction of @p fixed's
// output bar with every other operand, one at a time swapped for its
// tangent in @p along, onto k's tangent bar in @p target. @p target and
// @p fixed are the same build over x's operands (the tangent bars and the
// fixed ones); the δ operands are x's constants and vary with nothing.
Status one_slot_adjoint(DeviceResources &res, const ScArrays &x, const FockBuild &fixed,
                        const FockBuild &target, const FockBuild &along) {
  const auto constant = [&](const EinsumOperand &op) {
    return op.data == x.delta_core.data() || op.data == x.delta_act.data();
  };
  for (size_t t = 0; t < target.terms.size(); ++t) {
    const FockTerm &term = target.terms[t];
    for (size_t k = 0; k < term.operands.size(); ++k) {
      if (term.bars[k] == nullptr) {
        continue;
      }
      for (size_t j = 0; j < term.operands.size(); ++j) {
        if (j == k || constant(term.operands[j])) {
          continue;
        }
        std::vector<EinsumOperand> operands = term.operands;
        operands[j] = along.terms[t].operands[j];
        FBF_TRY(adjoint_slot(res, term.subscripts, operands, k, fixed.out_bar, fixed.dims,
                             term.bars[k], term.alpha));
      }
    }
  }
  return {};
}

// bars_tangent's result: the raw tangents of h_bar, Λ (both symmetrized)
// and y, before Phi's and the normalization's.
struct BarsTangent {
  Buffers::device<double> h_bar;
  Buffers::device<double> lambda;
  Buffers::device<double> y;
};

// The tangent of builders_adjoint's h_bar, Λ and y along one direction:
// @p bar_dot holds the tangent bars the class walk handed over (zero for
// the one-slot part alone) and @p lambda_raw its Λ, both added onto and
// consumed; @p dot the builders' tangent along the direction. Every build
// runs backwards at the tangent bars, the last first, as builders_adjoint
// runs it at the bars; with @p one_slot, each also adds its terms at the
// fixed bars @p bar with one operand at its tangent, and so does h2e's
// adjoint (linear in B). @p rdm_step is the RDM step's tangent, run where
// builders_adjoint runs its own: it reads bar_dot's RDM and γ bars, adds
// (tu|vw)'s tangent bar onto bar_dot.eri, and returns y's.
Result<BarsTangent> bars_tangent(DeviceResources &res, const int32_t nmo, ScArrays &x,
                                 ScArrays &bar, ScArrays &dot, ScArrays &bar_dot,
                                 Buffers::device<double> &lambda_raw, const bool one_slot,
                                 const std::function<Result<Buffers::device<double>>()> &rdm_step) {
  const std::vector<FockBuild> fixed = fock_builds(x, &bar);
  const std::vector<FockBuild> tangent = fock_builds(x, &bar_dot);
  const std::vector<FockBuild> along = fock_builds(dot, nullptr);
  const auto build_back = [&](const size_t i) -> Status {
    FBF_TRY(run_builds_adjoint(res, std::span(tangent).subspan(i, 1)));
    if (one_slot) {
      FBF_TRY(one_slot_adjoint(res, x, fixed[i], tangent[i], along[i]));
    }
    return {};
  };
  for (size_t i = tangent.size(); i-- > 1;) {
    FBF_TRY(build_back(i));
  }
  Buffers::device<double> y = FBF_TRY(rdm_step());

  // (tu|vw)'s builder, then the folds: h2e's adjoint at B's tangent first,
  // at the fixed bar.
  FBF_TRY(build_back(0));
  if (one_slot) {
    FBF_TRY(h2e_adjoint(res, dot.blocks, bar.h2e.data(), lambda_raw.data()));
  }
  FoldedBars folded = FBF_TRY(fold_bars(res, nmo, x.blocks, bar_dot, lambda_raw));
  // The launches above read this frame's buffers.
  FBF_TRY(res.sync());
  return BarsTangent{
      .h_bar = std::move(folded.h_bar), .lambda = std::move(folded.lambda), .y = std::move(y)};
}

// One root's RDM step of bars_tangent, at the root @p vector and its
// direction's CI part @p u: @p bar_dot's dm1 and dm2 bars folded into its
// dm3's by the traces' adjoint, then the one-vector rdms_adjoint onto y and
// (tu|vw)'s bar; with @p one_slot, its share at the fixed bars @p bar too
// (rdms_adjoint is bilinear in its bars and c, and linear in (tu|vw)
// through f3).
Result<Buffers::device<double>> root_rdms_tangent(DeviceResources &res,
                                                  const ci::DeviceStringSet &alpha,
                                                  const ci::DeviceStringSet &beta,
                                                  const double *vector, const double *u,
                                                  ScArrays &x, ScArrays &bar, ScArrays &dot,
                                                  ScArrays &bar_dot, const bool one_slot) {
  const int64_t n = x.blocks.ncas;
  const int64_t n4 = n * n * n * n;
  const int64_t n6 = n4 * n * n;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  // Its outputs land in buffers of their own, added on: each rdms_adjoint
  // overwrites its outputs and takes the workspace.
  FBF_TRY(run_builds_adjoint(res, trace_builds(x, x.dm1.data(), x.dm2.data(), x.dm3.data(),
                                               bar_dot.dm1.data(), bar_dot.dm2.data(),
                                               bar_dot.dm3.data())));
  Buffers::device<double> y = FBF_TRY(zeroed(res, ndet));
  Buffers::device<double> y_part(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> spare(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> eri_part(static_cast<size_t>(n4), res.device());
  const auto add_y = [&](const Buffers::device<double> &from) {
    return add_onto(res, ndet, 1, from.data(), y.data(), ndet);
  };
  const auto add_eri = [&] {
    return add_onto(res, n4, 1, eri_part.data(), bar_dot.eri.data(), n4);
  };
  FBF_TRY(rdms_adjoint(res, alpha, beta, vector, x.eri.data(), bar_dot.dm3.data(),
                       bar_dot.f3ca.data(), bar_dot.f3ac.data(), y_part.data(), eri_part.data()));
  FBF_TRY(add_y(y_part));
  FBF_TRY(add_eri());
  Buffers::device<double> no_dm3(0, res.device());
  if (one_slot) {
    // c's tangent at the fixed bars. With R(a, b) the bra/ket build, y is
    // ∂/∂a <R̄, R(a, c)> + ∂/∂b <R̄, R(c, b)>, so its tangent along u is the
    // bra's bar at (c, u) plus the ket's at (u, c); (tu|vw)'s bar takes both
    // builds' bars.
    FBF_TRY(rdms_adjoint(res, alpha, beta, vector, u, x.eri.data(), bar.dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), y_part.data(), spare.data(), eri_part.data()));
    FBF_TRY(add_y(y_part));
    FBF_TRY(add_eri());
    FBF_TRY(rdms_adjoint(res, alpha, beta, u, vector, x.eri.data(), bar.dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), spare.data(), y_part.data(), eri_part.data()));
    FBF_TRY(add_y(y_part));
    FBF_TRY(add_eri());
    // (tu|vw)'s tangent: f3 is linear in it and dm3 does not read it, so y's
    // share is the f3 bars' adjoint at (tu|vw)'s tangent, dm3's bar zero. Its
    // (tu|vw) bar does not move with (tu|vw), and is dropped.
    no_dm3 = FBF_TRY(zeroed(res, n6));
    FBF_TRY(rdms_adjoint(res, alpha, beta, vector, dot.eri.data(), no_dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), y_part.data(), eri_part.data()));
    FBF_TRY(add_y(y_part));
  }
  // The launches above read this frame's buffers.
  FBF_TRY(res.sync());
  return y;
}

// Phi's Λ at (B, D) onto @p lambda (naux x nmo^2): z g_L + D w_L
// - 1/2 (z B_L D + D B_L z), g_L = <B_L, D> and w_L = <B_L, z>, zc_adjoint's,
// bilinear in (B, D) at fixed z. All MO, column-major, B symmetric in its
// pair.
Status phi_lambda(DeviceResources &res, const int64_t naux, const int64_t nmo, const double *b,
                  const double *dens, const double *z, double *lambda) {
  const EinsumOperand b3 = column_operand(b, {naux, nmo, nmo});
  const EinsumOperand d2 = column_operand(dens, {nmo, nmo});
  const EinsumOperand z2 = column_operand(z, {nmo, nmo});
  const std::vector<int64_t> dims{naux, nmo, nmo};
  Buffers::device<double> g = FBF_TRY(zeroed(res, naux));
  Buffers::device<double> w = FBF_TRY(zeroed(res, naux));
  FBF_TRY(contract(res, "Lrs,rs->L", {b3, d2}, g.data(), {naux}, 1.0));
  FBF_TRY(contract(res, "Lrs,rs->L", {b3, z2}, w.data(), {naux}, 1.0));
  FBF_TRY(contract(res, "pq,L->Lpq", {z2, column_operand(g.data(), {naux})}, lambda, dims, 1.0));
  FBF_TRY(contract(res, "pq,L->Lpq", {d2, column_operand(w.data(), {naux})}, lambda, dims, 1.0));
  Buffers::device<double> bd = FBF_TRY(zeroed(res, naux * nmo * nmo));
  Buffers::device<double> bz = FBF_TRY(zeroed(res, naux * nmo * nmo));
  FBF_TRY(contract(res, "Lrs,sq->Lrq", {b3, d2}, bd.data(), dims, 1.0));
  FBF_TRY(contract(res, "Lrs,sq->Lrq", {b3, z2}, bz.data(), dims, 1.0));
  FBF_TRY(contract(res, "pr,Lrq->Lpq", {z2, column_operand(bd.data(), dims)}, lambda, dims, -0.5));
  FBF_TRY(contract(res, "pr,Lrq->Lpq", {d2, column_operand(bz.data(), dims)}, lambda, dims, -0.5));
  // The launches above read this frame's buffers.
  return res.sync();
}

// W = dPhi/dγ's active block at (B1, B2) onto @p w_act (ncas^2, column-major):
// Σ_L <B1_L, z> B2_L,tu - 1/2 Σ_L (B1_L z B2_L)_tu, zc_adjoint's W at
// B1 = B2 = B, bilinear in (B1, B2).
Status phi_w(DeviceResources &res, const int64_t naux, const int64_t nmo, const int64_t ncore,
             const int64_t ncas, const double *b1, const double *b2, const double *z,
             double *w_act) {
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const EinsumOperand z2 = column_operand(z, {nmo, nmo});
  Buffers::device<double> w = FBF_TRY(zeroed(res, naux));
  FBF_TRY(contract(res, "Lrs,rs->L", {column_operand(b1, {naux, nmo, nmo}), z2}, w.data(), {naux},
                   1.0));
  FBF_TRY(contract(res, "L,Ltu->tu",
                   {column_operand(w.data(), {naux}), b_block(b2, naux, nmo, nc, n, nc, n)}, w_act,
                   {n, n}, 1.0));
  // (B1_L z) over the active rows, naux x ncas x nmo.
  Buffers::device<double> bz = FBF_TRY(zeroed(res, naux * n * nmo));
  FBF_TRY(contract(res, "Ltr,rs->Lts", {b_block(b1, naux, nmo, nc, n, 0, nmo), z2}, bz.data(),
                   {naux, n, nmo}, 1.0));
  FBF_TRY(
      contract(res, "Lts,Lsu->tu",
               {column_operand(bz.data(), {naux, n, nmo}), b_block(b2, naux, nmo, 0, nmo, nc, n)},
               w_act, {n, n}, -0.5));
  return res.sync();
}

// y_r += w_r sigma(W + W^T) c_r over the weights.size() vectors @p vectors
// (root r's at r ndet), W column-major ncas^2 on the device: zc_adjoint's
// y^c at those weights, onto @p y.
Status phi_sigma(DeviceResources &res, ci::Sigma &sigma, const int64_t ncas, const double *w_act,
                 const std::span<const double> weights, const double *vectors, double *y) {
  const int64_t n = ncas;
  const std::vector<double> w = FBF_TRY(to_host(res, w_act, static_cast<size_t>(n * n)));
  std::vector<double> one_body(static_cast<size_t>(n * n));
  for (int64_t t = 0; t < n; ++t) {
    for (int64_t s = 0; s < n; ++s) {
      one_body[static_cast<size_t>((t * n) + s)] =
          w[static_cast<size_t>(t + (n * s))] + w[static_cast<size_t>(s + (n * t))];
    }
  }
  const std::vector<double> no_eri(static_cast<size_t>(n * n * n * n), 0.0);
  FBF_TRY(sigma.set_integrals(res, 0.0, one_body, no_eri));
  const int64_t ndet = sigma.size();
  const auto count = static_cast<int64_t>(weights.size());
  Buffers::device<double> out(static_cast<size_t>(ndet * count), res.device());
  FBF_TRY(sigma.apply(res, static_cast<int32_t>(count), vectors, out.data()));
  for (int64_t r = 0; r < count; ++r) {
    FBF_TRY(contract(res, "a->a", {column_operand(out.data() + (r * ndet), {ndet})}, y + (r * ndet),
                     {ndet}, weights[static_cast<size_t>(r)]));
  }
  return res.sync();
}

// X += 2 h h_bar + 2 Σ_L B_L Λ_L over MO arrays (B read as (naux nmo) x nmo,
// which its symmetry in its pair allows, as zc_adjoint reads it); a null
// h_bar or Λ is zero.
Status add_source(DeviceResources &res, const int32_t naux, const int32_t nmo, const double *h,
                  const double *b, const double *h_bar, const double *lambda, double *x) {
  if (h_bar != nullptr) {
    FBF_TRY(gpu_status(gemm(res.blas(), nmo, &kTwo<double>, h, h_bar, &kOne<double>, x)));
  }
  if (naux > 0 && lambda != nullptr) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, nmo, naux * nmo, &kTwo<double>, b,
                                               lambda, &kOne<double>, x)));
  }
  return {};
}

// A = X - X^T, a new buffer.
Result<Buffers::device<double>> antisymmetric_part(DeviceResources &res, const int32_t nmo,
                                                   const double *x) {
  Buffers::device<double> a(static_cast<size_t>(nmo) * nmo, res.device());
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                wwr::WWRBLAS_OP_T, nmo, nmo, &kOne<double>, x, nmo,
                                                &kNegOne<double>, x, nmo, a.data(), nmo)));
  return a;
}

double host_dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// A product's class walk: its reverse pass at the point onto the bars, and
// its Hessian-vector product along directions onto bars of their own, one
// per direction (@p both: the call whose diagnostics the product reports).
// @p check, when set, reads every direction's builders' tangent (and its CI
// part u, on the sphere's tangent) before the class walk does.
struct ProductWalk {
  std::function<Status(const MoBlocks &, const EnergyInputs &, const EnergyBars &)> adjoint;
  std::function<Status(const MoBlocks &, const EnergyInputs &, std::span<const EnergyDirection>,
                       std::span<const EnergyBars>, std::span<const EnergyDirection>, bool both)>
      hvp;
  std::function<Status(ScArrays &x, std::span<ScArrays> dots, const double *u)> check;
};

// The products and their parts, of either class walk.
struct Products {
  std::vector<AdjointTangent> products;
  std::vector<std::array<AdjointTangent, 5>> parts;
};

// sc_hessian_product's and pc_hessian_product's one body: everything but
// the class walk (README "The SC Hessian-vector product in (h, B, c)").
Result<Products> hessian_product(DeviceResources &res, const std::string_view kWho,
                                 const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                                 const int32_t naux, const int32_t nao, const int32_t nmo,
                                 const double *b, const double *c, const double *h,
                                 const int32_t ncore, const int32_t ncas, const double *vector,
                                 const double *z, const MoDirections &directions,
                                 const MoDirections &partners, const bool parts,
                                 const ProductWalk &walk_of) {
  const auto complete = [](const MoDirections &d) {
    return d.count >= 0 && (d.count == 0 || (d.h != nullptr && d.b != nullptr && d.c != nullptr));
  };
  if (!complete(directions) || !complete(partners) || z == nullptr) {
    return err_config("{}: {} directions and {} partners, each of h, b and c given when any are, "
                      "and z given",
                      kWho, directions.count, partners.count);
  }
  if (partners.count != 0 && partners.count != directions.count) {
    return err_config("{}: {} partners for {} directions (none, or one per direction)", kWho,
                      partners.count, directions.count);
  }
  // All-electron (nfrozen 0): the Hessian this feeds refuses a frozen core.
  ScArrays x =
      FBF_TRY(sc_arrays(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, 0));
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  const int64_t lambda_size = int64_t{naux} * nmo2;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  const auto nvirt = static_cast<int32_t>(nmo - ncore - ncas);
  const auto count = static_cast<size_t>(directions.count);

  const std::vector<double> c_host = FBF_TRY(to_host(res, vector, static_cast<size_t>(ndet)));
  if (const double norm2 = host_dot(c_host, c_host); std::abs(norm2 - 1.0) > kUnitVectorTolerance) {
    return err_config("{}: |c|^2 is {:.17g}: the vector must be a unit vector within {:g}", kWho,
                      norm2, kUnitVectorTolerance);
  }

  // sc_adjoint's reverse pass, its bars kept in bar for the one-slot terms.
  ScArrays bar = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
  LambdaBars lambda = FBF_TRY(lambda_bars(res, x, nmo));
  FBF_TRY(walk_of.adjoint(x.walk_blocks(), x.inputs(), energy_bars(bar, lambda)));
  const ScAdjoint corr = FBF_TRY(
      builders_adjoint(res, naux, nao, nmo, b, c, ncore, ncas, x, bar, std::move(lambda),
                       [&] { return root_rdms_adjoint(res, alpha, beta, vector, x, bar); }));
  // Phi's at the root, weight 1.
  const std::array<double, 1> one_root{1.0};
  const ZcAdjoint phi = FBF_TRY(
      zc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, one_root, vector, z));

  // B whole in the MO basis; D (2 on the core's diagonal, γ on the active
  // block, zc_adjoint's density); W at the point; and the summed adjoint's
  // h_bar, Λ and y, which the source term's one-slot share and the
  // normalization read.
  Buffers::device<double> bmo = FBF_TRY(zeroed(res, lambda_size));
  if (naux > 0) {
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, {.first = 0, .count = nmo}, {.first = 0, .count = nmo},
                 bmo.data()));
  }
  std::vector<double> core_density(static_cast<size_t>(nmo2), 0.0);
  for (int64_t i = 0; i < nc; ++i) {
    core_density[static_cast<size_t>(i * (nmo + 1))] = 2.0;
  }
  Buffers::device<double> dens = FBF_TRY(to_device(res, core_density));
  FBF_TRY(add_onto(res, n, n, x.dm1.data(), dens.data() + nc + (nmo * nc), nmo));
  Buffers::device<double> w_act = FBF_TRY(zeroed(res, n * n));
  FBF_TRY(phi_w(res, naux, nmo, nc, n, bmo.data(), bmo.data(), z, w_act.data()));
  Buffers::device<double> h_total = FBF_TRY(zeroed(res, nmo2));
  FBF_TRY(add_onto(res, nmo2, 1, corr.h_bar.data(), h_total.data(), nmo2));
  FBF_TRY(add_onto(res, nmo2, 1, phi.h_bar.data(), h_total.data(), nmo2));
  Buffers::device<double> lambda_total = FBF_TRY(zeroed(res, lambda_size));
  FBF_TRY(add_onto(res, lambda_size, 1, corr.lambda.data(), lambda_total.data(), lambda_size));
  FBF_TRY(add_onto(res, lambda_size, 1, phi.lambda.data(), lambda_total.data(), lambda_size));
  std::vector<double> y_total = FBF_TRY(to_host(res, corr.y.data(), static_cast<size_t>(ndet)));
  {
    const std::vector<double> yc = FBF_TRY(to_host(res, phi.y.data(), static_cast<size_t>(ndet)));
    for (size_t k = 0; k < y_total.size(); ++k) {
      y_total[k] += yc[k];
    }
  }
  const double cy = host_dot(c_host, y_total);
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));

  // Each direction's CI part on the sphere's tangent, u = ċ - (c . ċ) c,
  // and (c . ċ) for the normalization's last term.
  const auto projected = [&](const MoDirections &d)
      -> Result<std::pair<Buffers::device<double>, std::vector<double>>> {
    const auto m = static_cast<size_t>(d.count);
    std::vector<double> all = FBF_TRY(to_host(res, d.c, m * static_cast<size_t>(ndet)));
    std::vector<double> along(m);
    for (size_t k = 0; k < m; ++k) {
      const std::span<double> ck = std::span(all).subspan(k * c_host.size(), c_host.size());
      along[k] = host_dot(c_host, ck);
      for (size_t e = 0; e < ck.size(); ++e) {
        ck[e] -= along[k] * c_host[e];
      }
    }
    return std::pair{FBF_TRY(to_device(res, all)), std::move(along)};
  };
  const auto [u_all, along] = FBF_TRY(projected(directions));
  const auto [u_partners, partner_along] = FBF_TRY(projected(partners));

  // The builders' tangent along each direction, and each partner.
  const auto tangents = [&](const MoDirections &d,
                            const Buffers::device<double> &u) -> Result<std::vector<ScArrays>> {
    std::vector<ScArrays> out;
    for (int64_t k = 0; k < d.count; ++k) {
      out.push_back(
          FBF_TRY(direction_arrays(res, alpha, beta, x, vector, d.h + (k * nmo2),
                                   d.b + (k * lambda_size), u.data() + (k * ndet), nullptr)));
    }
    return out;
  };
  std::vector<ScArrays> dots = FBF_TRY(tangents(directions, u_all));
  std::vector<ScArrays> partner_dots = FBF_TRY(tangents(partners, u_partners));
  if (walk_of.check) {
    FBF_TRY(walk_of.check(x, dots, u_all.data()));
  }

  // class_energies_hvp along @p chosen of each direction, onto fresh bars.
  struct WalkBars {
    std::vector<ScArrays> bars;
    std::vector<LambdaBars> lambdas;
  };
  enum class Along { Both, Slabs, Builders };
  const auto walk = [&](const Along chosen) -> Result<WalkBars> {
    WalkBars out;
    std::vector<EnergyDirection> along_dirs;
    for (ScArrays &dot : dots) {
      out.bars.push_back(FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt)));
      out.lambdas.push_back(FBF_TRY(lambda_bars(res, x, nmo)));
      along_dirs.push_back({.in = chosen == Along::Slabs ? EnergyInputs{} : dot.inputs(),
                            .blocks = chosen == Along::Builders ? nullptr : &dot.blocks});
    }
    std::vector<EnergyBars> bars;
    for (size_t k = 0; k < count; ++k) {
      bars.push_back(energy_bars(out.bars[k], out.lambdas[k]));
    }
    std::vector<EnergyDirection> partner_dirs;
    if (chosen == Along::Both) {
      for (ScArrays &dot : partner_dots) {
        partner_dirs.push_back({.in = dot.inputs(), .blocks = &dot.blocks});
      }
    }
    FBF_TRY(walk_of.hvp(x.walk_blocks(), x.inputs(), along_dirs, bars, partner_dirs,
                        chosen == Along::Both));
    return out;
  };

  Products out;
  WalkBars both = FBF_TRY(walk(Along::Both));
  std::optional<WalkBars> slabs;
  std::optional<WalkBars> builders;
  if (parts) {
    slabs.emplace(FBF_TRY(walk(Along::Slabs)));
    builders.emplace(FBF_TRY(walk(Along::Builders)));
  }

  for (size_t k = 0; k < count; ++k) {
    const double *const h_dot = directions.h + (static_cast<int64_t>(k) * nmo2);
    const double *const b_dot = directions.b + (static_cast<int64_t>(k) * lambda_size);
    const double *const uk = u_all.data() + (static_cast<int64_t>(k) * ndet);
    ScArrays &dot = dots[k];

    // Phi's tangent: Λ at (Ḃ, D) + (B, Ḋ), Ḋ the active block's γ̇; y^c's,
    // sigma(W + W^T) at u plus sigma(Ẇ + Ẇ^T) at c, Ẇ = W(Ḃ, B) + W(B, Ḃ).
    Buffers::device<double> dens_dot = FBF_TRY(zeroed(res, nmo2));
    FBF_TRY(add_onto(res, n, n, dot.dm1.data(), dens_dot.data() + nc + (nmo * nc), nmo));
    Buffers::device<double> phi_lambda_dot = FBF_TRY(zeroed(res, lambda_size));
    FBF_TRY(phi_lambda(res, naux, nmo, b_dot, dens.data(), z, phi_lambda_dot.data()));
    FBF_TRY(phi_lambda(res, naux, nmo, bmo.data(), dens_dot.data(), z, phi_lambda_dot.data()));
    Buffers::device<double> w_dot = FBF_TRY(zeroed(res, n * n));
    FBF_TRY(phi_w(res, naux, nmo, nc, n, b_dot, bmo.data(), z, w_dot.data()));
    FBF_TRY(phi_w(res, naux, nmo, nc, n, bmo.data(), b_dot, z, w_dot.data()));
    Buffers::device<double> phi_y_dot = FBF_TRY(zeroed(res, ndet));
    FBF_TRY(phi_sigma(res, sigma, n, w_act.data(), one_root, uk, phi_y_dot.data()));
    FBF_TRY(phi_sigma(res, sigma, n, w_dot.data(), one_root, vector, phi_y_dot.data()));

    // The product: the reverse pass's tangent at the class walk's tangent
    // bars with the one-slot terms, plus Phi's; X's own one-slot share is
    // 2 ḣ h̄ + 2 Ḃ Λ at the summed adjoint.
    const auto root_step = [&](ScArrays &bar_dot, const bool one_slot) {
      return [&, one_slot] {
        return root_rdms_tangent(res, alpha, beta, vector, uk, x, bar, dot, bar_dot, one_slot);
      };
    };
    BarsTangent t = FBF_TRY(bars_tangent(res, nmo, x, bar, dot, both.bars[k], both.lambdas[k].whole,
                                         true, root_step(both.bars[k], true)));
    FBF_TRY(add_onto(res, lambda_size, 1, phi_lambda_dot.data(), t.lambda.data(), lambda_size));
    FBF_TRY(add_onto(res, ndet, 1, phi_y_dot.data(), t.y.data(), ndet));
    AdjointTangent product{.h_bar = std::move(t.h_bar),
                           .lambda = std::move(t.lambda),
                           .y = FBF_TRY(zeroed(res, ndet)),
                           .x = FBF_TRY(zeroed(res, nmo2)),
                           .a = Buffers::device<double>(0, res.device())};
    FBF_TRY(add_source(res, naux, nmo, x.h_mo.data(), bmo.data(), product.h_bar.data(),
                       product.lambda.data(), product.x.data()));
    FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, h_total.data(), lambda_total.data(),
                       product.x.data()));
    product.a = FBF_TRY(antisymmetric_part(res, nmo, product.x.data()));

    // y: the raw tangent, projected, and the normalization's terms.
    const std::vector<double> y_raw = FBF_TRY(to_host(res, t.y.data(), static_cast<size_t>(ndet)));
    const std::vector<double> u_host = FBF_TRY(to_host(res, uk, static_cast<size_t>(ndet)));
    const double cy_dot = host_dot(c_host, y_raw);
    const double uy = host_dot(u_host, y_total);
    std::vector<double> y_norm(y_raw.size());
    for (size_t e = 0; e < y_norm.size(); ++e) {
      y_norm[e] = -(c_host[e] * cy_dot) - (u_host[e] * cy) - (c_host[e] * uy) -
                  (along[k] * (y_total[e] - (cy * c_host[e])));
    }
    {
      std::vector<double> y_product = y_raw;
      for (size_t e = 0; e < y_product.size(); ++e) {
        y_product[e] += y_norm[e];
      }
      product.y = FBF_TRY(to_device(res, y_product));
    }
    out.products.push_back(std::move(product));

    if (!parts) {
      continue;
    }
    // The parts, each a whole AdjointTangent, zero where it has no share.
    const auto finish = [&](BarsTangent part,
                            const bool one_slot_source) -> Result<AdjointTangent> {
      AdjointTangent p{.h_bar = std::move(part.h_bar),
                       .lambda = std::move(part.lambda),
                       .y = std::move(part.y),
                       .x = FBF_TRY(zeroed(res, nmo2)),
                       .a = Buffers::device<double>(0, res.device())};
      FBF_TRY(add_source(res, naux, nmo, x.h_mo.data(), bmo.data(), p.h_bar.data(), p.lambda.data(),
                         p.x.data()));
      if (one_slot_source) {
        FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, corr.h_bar.data(), corr.lambda.data(),
                           p.x.data()));
      }
      p.a = FBF_TRY(antisymmetric_part(res, nmo, p.x.data()));
      return p;
    };
    AdjointTangent slab_part = FBF_TRY(
        finish(FBF_TRY(bars_tangent(res, nmo, x, bar, dot, slabs->bars[k], slabs->lambdas[k].whole,
                                    false, root_step(slabs->bars[k], false))),
               false));
    AdjointTangent builder_part = FBF_TRY(finish(
        FBF_TRY(bars_tangent(res, nmo, x, bar, dot, builders->bars[k], builders->lambdas[k].whole,
                             false, root_step(builders->bars[k], false))),
        false));
    ScArrays none = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
    Buffers::device<double> no_lambda = FBF_TRY(zeroed(res, lambda_size));
    AdjointTangent one_slot_part = FBF_TRY(finish(
        FBF_TRY(bars_tangent(res, nmo, x, bar, dot, none, no_lambda, true, root_step(none, true))),
        true));
    const auto phi_share = [&]() -> Result<AdjointTangent> {
      AdjointTangent p{.h_bar = FBF_TRY(zeroed(res, nmo2)),
                       .lambda = std::move(phi_lambda_dot),
                       .y = std::move(phi_y_dot),
                       .x = FBF_TRY(zeroed(res, nmo2)),
                       .a = Buffers::device<double>(0, res.device())};
      FBF_TRY(add_source(res, naux, nmo, x.h_mo.data(), bmo.data(), nullptr, p.lambda.data(),
                         p.x.data()));
      FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, z, phi.lambda.data(), p.x.data()));
      p.a = FBF_TRY(antisymmetric_part(res, nmo, p.x.data()));
      return p;
    };
    AdjointTangent phi_part = FBF_TRY(phi_share());
    AdjointTangent normalization_part{.h_bar = FBF_TRY(zeroed(res, nmo2)),
                                      .lambda = FBF_TRY(zeroed(res, lambda_size)),
                                      .y = FBF_TRY(to_device(res, y_norm)),
                                      .x = FBF_TRY(zeroed(res, nmo2)),
                                      .a = FBF_TRY(zeroed(res, nmo2))};
    out.parts.push_back({std::move(slab_part), std::move(builder_part), std::move(one_slot_part),
                         std::move(phi_part), std::move(normalization_part)});
  }
  FBF_TRY(res.sync());
  return out;
}

Result<ScHessianProduct> sc_hessian_product(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                            const ci::DeviceStringSet &beta, const int32_t naux,
                                            const int32_t nao, const int32_t nmo, const double *b,
                                            const double *c, const double *h, const int32_t ncore,
                                            const int32_t ncas, const double *vector,
                                            const double *z, const int32_t batch,
                                            const MoDirections &directions,
                                            const MoDirections &partners, const bool parts) {
  ScHessianProduct out;
  const ProductWalk walk{
      .adjoint = [&](const MoBlocks &blocks, const EnergyInputs &in,
                     const EnergyBars &bars) -> Status {
        out.energies = FBF_TRY(class_energies_adjoint(res, blocks, in, batch, bars)).energies;
        return {};
      },
      .hvp = [&](const MoBlocks &blocks, const EnergyInputs &in,
                 std::span<const EnergyDirection> dirs, std::span<const EnergyBars> bars,
                 std::span<const EnergyDirection> partner_dirs, const bool both) -> Status {
        ClassEnergiesHvp got =
            FBF_TRY(class_energies_hvp(res, blocks, in, batch, dirs, bars, partner_dirs));
        if (got.rounding > out.rounding) {
          out.rounding = got.rounding;
          out.rounding_at = got.rounding_at;
          out.rounding_norm = got.rounding_norm;
        }
        if (both) {
          out.populated = std::move(got.populated);
          out.population = std::move(got.population);
          out.scale = std::move(got.scale);
          out.defect = std::move(got.defect);
        }
        return {};
      },
      .check = {}};
  Products p =
      FBF_TRY(hessian_product(res, "nevpt2::sc_hessian_product", alpha, beta, naux, nao, nmo, b, c,
                              h, ncore, ncas, vector, z, directions, partners, parts, walk));
  out.products = std::move(p.products);
  out.parts = std::move(p.parts);
  return out;
}

Result<PcHessianProduct> pc_hessian_product(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                            const ci::DeviceStringSet &beta, const int32_t naux,
                                            const int32_t nao, const int32_t nmo, const double *b,
                                            const double *c, const double *h, const int32_t ncore,
                                            const int32_t ncas, const double *vector,
                                            const double *z, const int32_t batch,
                                            const MoDirections &directions, const bool parts,
                                            const double pc_floor, const bool span) {
  constexpr std::string_view kWho = "nevpt2::pc_hessian_product";
  PcHessianProduct out;
  const auto floored = [&](const EnergyInputs &in) {
    EnergyInputs x = in;
    x.pc_floor = pc_floor;
    return x;
  };
  const ProductWalk walk{
      .adjoint = [&](const MoBlocks &blocks, const EnergyInputs &in,
                     const EnergyBars &bars) -> Status {
        out.energies =
            FBF_TRY(pc_class_energies_adjoint(res, blocks, floored(in), batch, bars)).energies;
        return {};
      },
      .hvp = [&](const MoBlocks &blocks, const EnergyInputs &in,
                 std::span<const EnergyDirection> dirs, std::span<const EnergyBars> bars,
                 std::span<const EnergyDirection>, const bool) -> Status {
        FBF_TRY(pc_class_energies_hvp(res, blocks, floored(in), batch, dirs, bars, span));
        return {};
      },
      // The Schur lift along each direction: the RDMs' second derivative on
      // the sphere, 2 γ(u, u) - 2 |u|² γ(c, c) (u ⊥ c), dm3's from
      // nevpt2::rdms of u and dm2 and dm1 its traces.
      .check = [&](ScArrays &x, std::span<ScArrays> dots, const double *u) -> Status {
        const int64_t n = ncas;
        const int64_t n6 = n * n * n * n * n * n;
        const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
        for (size_t k = 0; k < dots.size(); ++k) {
          const double *const uk = u + (static_cast<int64_t>(k) * ndet);
          const std::vector<double> uh = FBF_TRY(to_host(res, uk, static_cast<size_t>(ndet)));
          const double uu = host_dot(uh, uh);
          PcMetricLifts lifts{};
          if (uu > 0.0) {
            Buffers::device<double> dm3 = FBF_TRY(zeroed(res, n6));
            FBF_TRY(rdms(res, alpha, beta, uk, x.eri.data(), dm3.data(), nullptr, nullptr));
            Buffers::device<double> second3 = FBF_TRY(zeroed(res, n6));
            FBF_TRY(contract(res, "a->a", {column_operand(dm3.data(), {n6})}, second3.data(), {n6},
                             2.0));
            FBF_TRY(contract(res, "a->a", {column_operand(x.dm3.data(), {n6})}, second3.data(),
                             {n6}, -2.0 * uu));
            Buffers::device<double> second2 = FBF_TRY(zeroed(res, n * n * n * n));
            Buffers::device<double> second1 = FBF_TRY(zeroed(res, n * n));
            FBF_TRY(run_builds(res, trace_builds(x, second1.data(), second2.data(), second3.data(),
                                                 nullptr, nullptr, nullptr)));
            const EnergyInputs second{
                .dm1 = second1.data(), .dm2 = second2.data(), .dm3 = second3.data()};
            lifts = FBF_TRY(pc_metric_lifts(res, x.walk_blocks(), floored(x.inputs()),
                                            dots[k].inputs(), second));
            for (double &l : lifts.lift) {
              l /= uu;
            }
            // The launches above read this branch's buffers.
            FBF_TRY(res.sync());
          }
          // t* in the direction's own units: the null mode's ½ ℓ |u|² t² reaches
          // the cut there.
          std::array<double, 8> steps{};
          for (size_t cls = 0; cls < steps.size(); ++cls) {
            steps[cls] = lifts.lift[cls] > 0.0
                             ? std::sqrt(2.0 * lifts.cut[cls] / (lifts.lift[cls] * uu))
                             : std::numeric_limits<double>::infinity();
          }
          out.lifts.push_back(lifts.lift);
          out.lift_steps.push_back(steps);
        }
        return {};
      }};
  Products p = FBF_TRY(hessian_product(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                                       vector, z, directions, {}, parts, walk));
  out.products = std::move(p.products);
  out.parts = std::move(p.parts);
  return out;
}

// ---- The footprints: the helpers' mirrors, shared with the QD products ----

// Every EnergyInputs field given, a direction's along the builders, as the
// footprints read one: only whether a field is null.
const double kGiven = 0.0;
EnergyInputs given_inputs() {
  return {.h1e = &kGiven,
          .h2e = &kGiven,
          .e_core = &kGiven,
          .e_virt = &kGiven,
          .h1e_v_sr = &kGiven,
          .h1e_v_si = &kGiven,
          .h1e_v_sir = &kGiven,
          .dm1 = &kGiven,
          .dm2 = &kGiven,
          .dm3 = &kGiven,
          .f3ca = &kGiven,
          .f3ac = &kGiven};
}
// A B tangent given: the footprints read its nullness alone, never what it
// points to.
const MoBlocks *given_blocks() {
  return reinterpret_cast<const MoBlocks *>(&kGiven);
}

// The sizes the products' mirrors read, in bytes: nmo², Λ's naux nmo², and
// one CI vector.
struct ProductSizes {
  size_t nn = 0;
  size_t lambda = 0;
  size_t ndet = 0;
};

ProductSizes product_sizes(const int32_t naux, const int32_t nmo, const int32_t ncas,
                           const int32_t n_alpha, const int32_t n_beta) {
  const auto strings = [ncas](const int32_t electrons) {
    return static_cast<size_t>(std::round(math::binomial(ncas, electrons)));
  };
  const size_t nn = sizeof(double) * static_cast<size_t>(nmo) * static_cast<size_t>(nmo);
  return {.nn = nn,
          .lambda = static_cast<size_t>(naux) * nn,
          .ndet = sizeof(double) * strings(n_alpha) * strings(n_beta)};
}

// One AdjointTangent over @p y bytes of y: h_bar, Λ, y, X and A.
size_t adjoint_tangent_bytes(const ProductSizes &z, const size_t y) {
  return (3 * z.nn) + z.lambda + y;
}

// phi_lambda's scratch: g and w, then B D and B z.
size_t phi_lambda_bytes(const int32_t naux, const ProductSizes &z) {
  return (2 * sizeof(double) * static_cast<size_t>(naux)) + (2 * z.lambda);
}

// phi_w's scratch: w, then B z over the active rows.
size_t phi_w_bytes(const int32_t naux, const int32_t nmo, const int32_t ncas) {
  const auto q = static_cast<size_t>(naux);
  return sizeof(double) * (q + (q * static_cast<size_t>(ncas) * static_cast<size_t>(nmo)));
}

// The products' ci::Sigma: its tables at create, and its apply scratch, kept
// from its first apply (phi_sigma's, beside that call's out).
struct SigmaBytes {
  size_t tables = 0;
  size_t apply = 0;
  bool applied = false;
  void create(Held &h) const { h.take(tables); }
  // One phi_sigma over @p out bytes of output.
  void call(Held &h, const size_t out) {
    h.take(out);
    if (!applied) {
      h.take(apply);
      applied = true;
    }
    h.give(out);
  }
  // The Sigma goes: its tables and the scratch.
  void release(Held &h) const { h.give(tables + (applied ? apply : 0)); }
};

SigmaBytes sigma_bytes(const int32_t ncas, const int32_t n_alpha, const int32_t n_beta) {
  const auto strings = [ncas](const int32_t electrons) {
    return static_cast<int32_t>(std::round(math::binomial(ncas, electrons)));
  };
  const int32_t alpha = strings(n_alpha);
  const int32_t beta = strings(n_beta);
  return {.tables = ci::sigma_footprint(ncas, alpha, beta).device.persistent,
          .apply = ci::sigma_apply_bytes(ncas, alpha, beta, ci::Layout::full)};
}

// bars_tangent onto @p h, its RDM step @p rdm_step's (which leaves y held):
// h2e's adjoint at B's tangent (with the one-slot terms) and fold_bars'
// workspaces, and the folded h_bar and Λ. Leaves the BarsTangent held:
// returns its bytes, y's @p y among them.
size_t bars_tangent_bytes(Held &h, const int32_t naux, const int32_t ncore, const int32_t ncas,
                          const int32_t nvirt, const ProductSizes &z, const size_t y,
                          const std::function<void(Held &)> &rdm_step) {
  rdm_step(h);
  h.borrow(h2e_adjoint_footprint(ncas));
  h.borrow(mo_blocks_adjoint_footprint(naux, ncore, nvirt));
  h.take(z.nn + z.lambda);
  return y + z.nn + z.lambda;
}

// root_rdms_tangent onto @p h: y, held; y's two parts and (tu|vw)'s bar's;
// rdms_adjoint at the root, and with @p one_slot at (c, u) and (u, c) and,
// beside dm3's zero bar, at (tu|vw)'s tangent; all at one determinant.
void root_rdms_tangent_bytes(Held &h, const int32_t ncas, const ProductSizes &z,
                             const bool one_slot) {
  const auto n = static_cast<size_t>(ncas);
  const size_t n4 = sizeof(double) * n * n * n * n;
  const size_t n6 = n4 * n * n;
  h.take(z.ndet);
  const size_t scratch = (2 * z.ndet) + n4;
  h.take(scratch);
  h.call(rdms_adjoint_footprint(ncas));
  if (one_slot) {
    h.call(rdms_adjoint_footprint(ncas, 1, true));
    h.call(rdms_adjoint_footprint(ncas, 1, true));
    h.take(n6);
    h.call(rdms_adjoint_footprint(ncas));
    h.give(n6);
  }
  h.give(scratch);
}

// What a single-root product's class walk asks of the footprint: its
// reverse pass's, its Hessian-vector product's along the directions (with
// partners, empty for none), and before the walk, per direction, its check.
struct ProductWalkBytes {
  MemoryFootprint adjoint;
  std::function<MemoryFootprint(std::span<const EnergyDirection>, std::span<const EnergyDirection>)>
      hvp;
  std::function<void(Held &)> check;
};

// hessian_product, every allocation in its order onto a Held, in bytes:
// sc_product_footprint's and pc_product_footprint's one body.
MemoryFootprint hessian_product_footprint(const int32_t naux, const int32_t nao, const int32_t nmo,
                                          const int32_t ncore, const int32_t ncas,
                                          const int32_t n_alpha, const int32_t n_beta,
                                          const int32_t count, const int32_t partners,
                                          const bool parts, const ProductWalkBytes &walk) {
  constexpr size_t kD = sizeof(double);
  const int32_t nvirt = nmo - ncore - ncas;
  const auto n = static_cast<size_t>(ncas);
  const size_t n2 = kD * n * n;
  const ProductSizes z = product_sizes(naux, nmo, ncas, n_alpha, n_beta);
  const size_t arrays = zeroed_arrays_bytes(naux, ncore, ncas, nvirt);
  const auto m = static_cast<size_t>(count);
  const size_t product = adjoint_tangent_bytes(z, z.ndet);
  Held h;
  // sc_arrays; the bars and Λ's raw bar; the class walk's reverse pass;
  // builders_adjoint (y, rdms_adjoint, the tail: the ScAdjoint kept).
  const MemoryFootprint shared = shared_arrays_footprint(naux, nao, nmo, ncore, ncas, 0);
  h.call(shared);
  h.call(rdms_footprint(ncas));
  h.take(arrays + z.lambda);
  h.call(walk.adjoint);
  h.take(z.ndet);
  h.call(rdms_adjoint_footprint(ncas));
  builders_adjoint_tail(h, naux, nao, nmo, ncore, ncas);
  // Phi's adjoint at the root; B whole; D, W, and the summed h_bar and Λ.
  const MemoryFootprint phi = zc_adjoint_footprint(naux, nao, nmo, ncas, n_alpha, n_beta, 1);
  h.call(phi);
  h.take(z.lambda);
  if (naux > 0) {
    h.borrow(b_mo_footprint(naux, nao, {.first = 0, .count = nmo}, {.first = 0, .count = nmo}));
  }
  h.take(z.nn + n2);
  h.take(phi_w_bytes(naux, nmo, ncas));
  h.give(phi_w_bytes(naux, nmo, ncas));
  h.take(z.nn + z.lambda);
  SigmaBytes sigma = sigma_bytes(ncas, n_alpha, n_beta);
  sigma.create(h);
  // u over the directions and the partners; the builders' tangent along
  // each, kept; the check.
  h.take((m + static_cast<size_t>(partners)) * z.ndet);
  const MemoryFootprint dot = direction_arrays_footprint(naux, ncore, ncas, nvirt, false);
  for (int32_t k = 0; k < count + partners; ++k) {
    h.call(dot);
  }
  if (walk.check) {
    walk.check(h);
  }
  // The class walk along every direction onto fresh bars, kept: along both,
  // and with the parts along the slabs alone and the builders alone.
  const auto walked = [&](const EnergyInputs &in, const MoBlocks *blocks, const bool with) {
    h.take(m * (arrays + z.lambda));
    const std::vector<EnergyDirection> dirs(m, EnergyDirection{.in = in, .blocks = blocks});
    const std::vector<EnergyDirection> others(
        with ? static_cast<size_t>(partners) : 0,
        EnergyDirection{.in = given_inputs(), .blocks = given_blocks()});
    h.call(walk.hvp(dirs, others));
  };
  walked(given_inputs(), given_blocks(), true);
  if (parts) {
    walked({}, given_blocks(), false);
    walked(given_inputs(), nullptr, false);
  }

  size_t kept = 0;
  for (size_t k = 0; k < m; ++k) {
    // Phi's tangent: Ḋ, Λ̇ (two phi_lambda), Ẇ (two phi_w), ẏ (two
    // phi_sigma).
    h.take(z.nn + z.lambda);
    for (int32_t side = 0; side < 2; ++side) {
      h.take(phi_lambda_bytes(naux, z));
      h.give(phi_lambda_bytes(naux, z));
    }
    h.take(n2);
    for (int32_t side = 0; side < 2; ++side) {
      h.take(phi_w_bytes(naux, nmo, ncas));
      h.give(phi_w_bytes(naux, nmo, ncas));
    }
    h.take(z.ndet);
    for (int32_t side = 0; side < 2; ++side) {
      sigma.call(h, z.ndet);
    }
    // The product: the reverse pass's tangent (its h_bar and Λ moved in, y
    // left behind), y, X and A, y replaced by the projected one.
    const auto step = [&](const bool one_slot) {
      return [&, one_slot](Held &held) { root_rdms_tangent_bytes(held, ncas, z, one_slot); };
    };
    bars_tangent_bytes(h, naux, ncore, ncas, nvirt, z, z.ndet, step(true));
    h.take(z.ndet + z.nn);
    h.take(z.nn);
    h.take(z.ndet);
    h.give(z.ndet);
    kept += product;
    size_t local = z.ndet; // the tangent's y
    if (parts) {
      // The slabs' and the builders' parts, the one-slot part over zeroed
      // bars and Λ, Phi's (its Λ̇ and ẏ moved in) and the normalization's.
      for (int32_t part = 0; part < 2; ++part) {
        bars_tangent_bytes(h, naux, ncore, ncas, nvirt, z, z.ndet, step(false));
        h.take(z.nn);
        h.take(z.nn);
      }
      h.take(arrays + z.lambda);
      local += arrays + z.lambda;
      bars_tangent_bytes(h, naux, ncore, ncas, nvirt, z, z.ndet, step(true));
      h.take(z.nn);
      h.take(z.nn);
      h.take(3 * z.nn);
      h.take(product);
      kept += 5 * product;
    } else {
      local += z.lambda + z.ndet; // Phi's Λ̇ and ẏ
    }
    // The iteration's own: Ḋ, Ẇ, the tangent's y, and without the parts
    // Phi's Λ̇ and ẏ; with them, the one-slot part's bars and Λ.
    h.give(z.nn + n2 + local);
  }
  // Everything but the products goes.
  h.give(h.live - kept);
  return h.footprint(kept);
}

MemoryFootprint sc_hessian_product_footprint(const int32_t naux, const int32_t nao,
                                             const int32_t nmo, const int32_t ncore,
                                             const int32_t ncas, const int32_t n_alpha,
                                             const int32_t n_beta, const int32_t batch,
                                             const int32_t count, const int32_t partners,
                                             const bool parts) {
  const int32_t nvirt = nmo - ncore - ncas;
  const int32_t nelecas = n_alpha + n_beta;
  const ProductWalkBytes walk{
      .adjoint = class_energies_adjoint_footprint(naux, ncore, ncas, nvirt, nelecas, batch),
      .hvp =
          [&](std::span<const EnergyDirection> dirs, std::span<const EnergyDirection> with) {
            return class_energies_hvp_footprint(naux, ncore, ncas, nvirt, nelecas, batch, dirs,
                                                with);
          },
      .check = {}};
  return hessian_product_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, count, partners,
                                   parts, walk);
}

MemoryFootprint pc_hessian_product_footprint(
    const int32_t naux, const int32_t nao, const int32_t nmo, const int32_t ncore,
    const int32_t ncas, const int32_t n_alpha, const int32_t n_beta, const int32_t batch,
    const std::span<const size_t> syevd_lwork, const int32_t count, const bool parts,
    const std::span<const std::array<int64_t, 8>> dropped) {
  const int32_t nvirt = nmo - ncore - ncas;
  const auto n = static_cast<size_t>(ncas);
  const size_t n2 = sizeof(double) * n * n;
  const size_t n4 = n2 * n * n;
  const size_t n6 = n4 * n * n;
  const ProductWalkBytes walk{
      .adjoint = pc_effective_hamiltonian_adjoint_footprint(naux, ncore, ncas, nvirt, 1, batch,
                                                            syevd_lwork, dropped),
      .hvp =
          [&](std::span<const EnergyDirection> dirs, std::span<const EnergyDirection>) {
            return pc_effective_hamiltonian_hvp_footprint(naux, ncore, ncas, nvirt, 1, batch,
                                                          syevd_lwork, dirs, dropped);
          },
      // The Schur lift per direction: the RDMs' second derivative (dm3 by
      // nevpt2::rdms of u at one determinant, f3 off; dm3, dm2 and dm1 of
      // the path) beside pc_metric_lifts.
      .check =
          [&](Held &h) {
            for (int32_t k = 0; k < count; ++k) {
              h.take(n6);
              h.call(rdms_footprint(ncas, 1, false, false));
              h.take(n6 + n4 + n2);
              h.call(pc_metric_lifts_footprint(ncore, ncas, nvirt, 1, syevd_lwork, dropped));
              h.give((2 * n6) + n4 + n2);
            }
          }};
  return hessian_product_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, count, 0, parts,
                                   walk);
}

// ---- SC-QD: qd_sc_hessian_product ----------------------------------------

// One ndet slice of the n stacked vectors @p v, root r's.
std::span<const double> root_of(std::span<const double> v, const size_t ndet, const size_t r) {
  return v.subspan(r * ndet, ndet);
}

// Pi(@p dot) over the roots @p c (n stacked, root r's at r ndet): the part
// of each direction along the root set's generators (Eq. qd-K),
// u_r = dot_r - 1/2 sum_s c_s (c_s . dot_r + c_r . dot_s), so each root's
// out-of-span move and the in-span move's antisymmetric part; at n = 1,
// dot - (c . dot) c. An orthogonal projection on the stacked vectors.
std::vector<double> root_set_projection(std::span<const double> c, const size_t n,
                                        std::span<const double> dot) {
  const size_t ndet = c.size() / n;
  std::vector<double> out(dot.begin(), dot.end());
  for (size_t r = 0; r < n; ++r) {
    for (size_t s = 0; s < n; ++s) {
      const double coef = 0.5 * (host_dot(root_of(c, ndet, s), root_of(dot, ndet, r)) +
                                 host_dot(root_of(c, ndet, r), root_of(dot, ndet, s)));
      for (size_t e = 0; e < ndet; ++e) {
        out[(r * ndet) + e] -= coef * c[(s * ndet) + e];
      }
    }
  }
  return out;
}

// K @p y, K the generator of the projected direction @p u (Pi's image) over
// the roots @p c: K = sum_r (xi_r c_r^T - c_r xi_r^T) + C A C^T with
// xi_r = u_r - sum_s c_s (c_s . u_r) and A = C^T U, so
// K y = sum_r [u_r (c_r . y) - c_r (xi_r . y)].
std::vector<double> generator_apply(std::span<const double> c, const size_t n,
                                    std::span<const double> u, std::span<const double> y) {
  const size_t ndet = y.size();
  std::vector<double> out(ndet, 0.0);
  for (size_t r = 0; r < n; ++r) {
    const std::span<const double> ur = root_of(u, ndet, r);
    double xi_y = host_dot(ur, y);
    for (size_t s = 0; s < n; ++s) {
      xi_y -= host_dot(root_of(c, ndet, s), ur) * host_dot(root_of(c, ndet, s), y);
    }
    const double cy = host_dot(root_of(c, ndet, r), y);
    for (size_t e = 0; e < ndet; ++e) {
      out[e] += (cy * ur[e]) - (xi_y * c[(r * ndet) + e]);
    }
  }
  return out;
}

// K^*(M) for M = sum_i a_i b_i^T over the m stacked pairs (@p a, @p b): the
// gradient over raw CI directions t (n stacked) of <M, K(Pi t)>_F, n stacked
// vectors in the tangent space:
//   K^*(M)_r = (1 - P) (M - M^T) c_r + 1/2 sum_s c_s (Q_sr - Q_rs),
// Q = C^T M C and P the roots' projector. K^*(sum_r y_r c_r^T) is Pi(y).
std::vector<double> generator_adjoint(std::span<const double> c, const size_t n,
                                      std::span<const double> a, std::span<const double> b) {
  const size_t ndet = c.size() / n;
  const size_t m = a.size() / ndet;
  // c_s . a_i and c_s . b_i.
  std::vector<double> ca(n * m);
  std::vector<double> cb(n * m);
  for (size_t s = 0; s < n; ++s) {
    for (size_t i = 0; i < m; ++i) {
      ca[(s * m) + i] = host_dot(root_of(c, ndet, s), root_of(a, ndet, i));
      cb[(s * m) + i] = host_dot(root_of(c, ndet, s), root_of(b, ndet, i));
    }
  }
  const auto q = [&](const size_t s, const size_t r) {
    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
      sum += ca[(s * m) + i] * cb[(r * m) + i];
    }
    return sum;
  };
  std::vector<double> out(n * ndet, 0.0);
  for (size_t r = 0; r < n; ++r) {
    const std::span<double> v = std::span(out).subspan(r * ndet, ndet);
    for (size_t i = 0; i < m; ++i) {
      for (size_t e = 0; e < ndet; ++e) {
        v[e] += (a[(i * ndet) + e] * cb[(r * m) + i]) - (b[(i * ndet) + e] * ca[(r * m) + i]);
      }
    }
    for (size_t s = 0; s < n; ++s) {
      const double coef = host_dot(root_of(c, ndet, s), v) - (0.5 * (q(s, r) - q(r, s)));
      for (size_t e = 0; e < ndet; ++e) {
        v[e] -= coef * c[(s * ndet) + e];
      }
    }
  }
  return out;
}

// The QD RDM step of bars_tangent along one direction, at the roots
// @p vectors of @p weights and the direction's CI part @p u (n stacked):
// each root r's own bars' tangent @p own_dot[r] takes w_r times bar_dot's
// gamma_SA bar, folds by the traces' adjoint and goes through the
// one-vector rdms_adjoint at c_r onto y_r and (tu|vw)'s bar; with
// @p one_slot, root r's share at its fixed own bars as sc_hessian_product's
// root's (three calls), and each transition's at its fixed dm3 bar, the
// bra/ket rdms_adjoint at (u_M, u_N): bilinear in (bra, ket), its bra's bar
// is linear in the ket alone and its ket's in the bra, so the one call is
// both roots' tangent. y starts at @p y_walk, the transitions' share the
// walk handed over, when given.
Result<Buffers::device<double>>
qd_rdms_tangent(DeviceResources &res, const ci::DeviceStringSet &alpha,
                const ci::DeviceStringSet &beta, const std::span<const double> weights,
                const double *vectors, const double *u, QdFixedBars &fixed, ScArrays &dot,
                ScArrays &bar_dot, std::span<PairRdms> own_dot,
                const Buffers::device<double> *y_walk, const bool one_slot) {
  ScArrays &x = fixed.arrays.x;
  const int64_t n = x.blocks.ncas;
  const int64_t n4 = n * n * n * n;
  const int64_t n6 = n4 * n * n;
  const auto nroots = static_cast<int64_t>(weights.size());
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  Buffers::device<double> y = FBF_TRY(zeroed(res, nroots * ndet));
  if (y_walk != nullptr) {
    FBF_TRY(add_onto(res, nroots * ndet, 1, y_walk->data(), y.data(), nroots * ndet));
  }
  Buffers::device<double> y_part(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> spare(static_cast<size_t>(ndet), res.device());
  Buffers::device<double> eri_part(static_cast<size_t>(n4), res.device());
  const auto add_y = [&](const Buffers::device<double> &from, const int64_t root) {
    return add_onto(res, ndet, 1, from.data(), y.data() + (root * ndet), ndet);
  };
  const auto add_eri = [&] {
    return add_onto(res, n4, 1, eri_part.data(), bar_dot.eri.data(), n4);
  };
  Buffers::device<double> no_dm3 = FBF_TRY(zeroed(res, one_slot ? n6 : 0));
  for (int64_t r = 0; r < nroots; ++r) {
    const double *const cr = vectors + (r * ndet);
    const double *const ur = u + (r * ndet);
    PairRdms &p = fixed.arrays.pairs[static_cast<size_t>(r * (nroots + 1))];
    PairRdms &o = own_dot[static_cast<size_t>(r)];
    FBF_TRY(contract(res, "tu->tu", {column_operand(bar_dot.dm1.data(), {n, n})}, o.dm1.data(),
                     {n, n}, weights[static_cast<size_t>(r)]));
    FBF_TRY(run_builds_adjoint(res, trace_builds(x, p.dm1.data(), p.dm2.data(), p.dm3.data(),
                                                 o.dm1.data(), o.dm2.data(), o.dm3.data())));
    FBF_TRY(rdms_adjoint(res, alpha, beta, cr, x.eri.data(), o.dm3.data(), o.f3ca.data(),
                         o.f3ac.data(), y_part.data(), eri_part.data()));
    FBF_TRY(add_y(y_part, r));
    FBF_TRY(add_eri());
    if (!one_slot) {
      continue;
    }
    const PairRdms &f = fixed.own[static_cast<size_t>(r)];
    FBF_TRY(rdms_adjoint(res, alpha, beta, cr, ur, x.eri.data(), f.dm3.data(), f.f3ca.data(),
                         f.f3ac.data(), y_part.data(), spare.data(), eri_part.data()));
    FBF_TRY(add_y(y_part, r));
    FBF_TRY(add_eri());
    FBF_TRY(rdms_adjoint(res, alpha, beta, ur, cr, x.eri.data(), f.dm3.data(), f.f3ca.data(),
                         f.f3ac.data(), spare.data(), y_part.data(), eri_part.data()));
    FBF_TRY(add_y(y_part, r));
    FBF_TRY(add_eri());
    FBF_TRY(rdms_adjoint(res, alpha, beta, cr, dot.eri.data(), no_dm3.data(), f.f3ca.data(),
                         f.f3ac.data(), y_part.data(), eri_part.data()));
    FBF_TRY(add_y(y_part, r));
  }
  if (one_slot) {
    // A transition dm3 does not read (tu|vw): its bar is written zero.
    for (int64_t bra = 0; bra < nroots; ++bra) {
      for (int64_t ket = 0; ket < nroots; ++ket) {
        if (bra == ket) {
          continue;
        }
        FBF_TRY(rdms_adjoint(res, alpha, beta, u + (bra * ndet), u + (ket * ndet), x.eri.data(),
                             fixed.transitions[static_cast<size_t>((bra * nroots) + ket)].data(),
                             nullptr, nullptr, y_part.data(), spare.data(), eri_part.data()));
        FBF_TRY(add_y(y_part, bra));
        FBF_TRY(add_y(spare, ket));
      }
    }
  }
  // The launches above read this frame's buffers.
  FBF_TRY(res.sync());
  return y;
}

// qd_rdms_tangent_footprint (declared in :builders).
MemoryFootprint qd_rdms_tangent_footprint(const int32_t ncas, const int32_t n_alpha,
                                          const int32_t n_beta, const int32_t nroots,
                                          const bool one_slot) {
  const auto n = static_cast<size_t>(ncas);
  const size_t n4 = sizeof(double) * n * n * n * n;
  const size_t n6 = n4 * n * n;
  const size_t ndet = product_sizes(0, 0, ncas, n_alpha, n_beta).ndet;
  const size_t y = static_cast<size_t>(nroots) * ndet;
  Held h;
  // y; y_part, spare and eri_part; with one_slot, the zero dm3 bar.
  h.take(y);
  const size_t scratch = (2 * ndet) + n4 + (one_slot ? n6 : 0);
  h.take(scratch);
  for (int32_t r = 0; r < nroots; ++r) {
    // The root's own bars' tangent at c_r; with one_slot, its fixed bars at
    // (c_r, u_r) and (u_r, c_r), and its f3 bars at (tu|vw)'s tangent.
    h.call(rdms_adjoint_footprint(ncas, 1, false, true));
    if (one_slot) {
      h.call(rdms_adjoint_footprint(ncas, 1, true, true));
      h.call(rdms_adjoint_footprint(ncas, 1, true, true));
      h.call(rdms_adjoint_footprint(ncas, 1, false, true));
    }
  }
  if (one_slot && nroots > 1) {
    // Each transition's fixed dm3 bar at (u_M, u_N).
    h.call(rdms_adjoint_footprint(ncas, 1, true, false));
  }
  h.give(scratch);
  return h.footprint(y);
}

// What qd_hessian_product's body asks of its pair walk: @p adjoint runs
// qd_sc_adjoint's or qd_pc_adjoint's reverse pass at W, its arrays and
// folded bars kept into the given QdFixedBars; @p check reads every
// direction's QD builders' tangent (and its CI part u along the generators)
// and its partner's before the walk; @p hvp is the pair walk's
// Hessian-vector product along them onto the bars, each transition's
// tangent bars handed to the transition callback.
struct QdProductWalk {
  std::function<Result<QdScAdjoint>(std::optional<QdFixedBars> &)> adjoint;
  std::function<Status(QdFixedBars &, std::span<const QdDirectionArrays>, const double *u,
                       std::span<const QdDirectionArrays>, const double *u_partners)>
      check;
  std::function<Status(const MoBlocks &, std::span<const EnergyInputs>,
                       std::span<const EnergyDirection>, std::span<const EnergyBars>,
                       const std::function<Status(int64_t, int64_t, int64_t, const EnergyBars &)> &,
                       std::span<const EnergyDirection>)>
      hvp;
};

// The QD products and their parts, of either pair walk.
struct QdProducts {
  std::vector<AdjointTangent> products;
  std::vector<std::array<AdjointTangent, 4>> parts;
};

// qd_sc_hessian_product's and qd_pc_hessian_product's one body: everything
// but the pair walk (README "The SC-QD Hessian-vector product in
// (h, B, {c_r})").
Result<QdProducts>
qd_hessian_product(DeviceResources &res, const std::string_view kWho,
                   const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                   const int32_t naux, const int32_t nao, const int32_t nmo, const double *b,
                   const double *c, const double *h, const int32_t ncore, const int32_t ncas,
                   const std::span<const double> weights, const double *vectors,
                   const std::span<const double> w, const double *z, const MoDirections &directions,
                   const MoDirections &partners, const bool parts, const QdProductWalk &walk_of) {
  const auto complete = [](const MoDirections &d) {
    return d.count >= 0 && (d.count == 0 || (d.h != nullptr && d.b != nullptr && d.c != nullptr));
  };
  if (!complete(directions) || !complete(partners) || z == nullptr) {
    return err_config("{}: {} directions and {} partners, each of h, b and c given when any are, "
                      "and z given",
                      kWho, directions.count, partners.count);
  }
  if (partners.count != 0 && partners.count != directions.count) {
    return err_config("{}: {} partners for {} directions (none, or one per direction)", kWho,
                      partners.count, directions.count);
  }
  FBF_TRY(check_weights(kWho, weights));
  const size_t nroots = weights.size();
  if (w.size() != nroots * nroots) {
    return err_config("{}: W holds {} elements for {} roots: n^2", kWho, w.size(), nroots);
  }
  // The pair walk's reverse pass, its arrays and folded bars kept (one W,
  // all-electron: the Hessian this feeds refuses a frozen core).
  QdProducts out;
  std::optional<QdFixedBars> kept;
  QdScAdjoint corr = FBF_TRY(walk_of.adjoint(kept));
  QdFixedBars &fixed = *kept;
  ScArrays &x = fixed.arrays.x;
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  const int64_t lambda_size = int64_t{naux} * nmo2;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  const auto nn = static_cast<int64_t>(nroots);
  const int64_t ny = nn * ndet;
  const auto nvirt = static_cast<int32_t>(nmo - ncore - ncas);
  const auto count = static_cast<size_t>(directions.count);

  const std::vector<double> c_host = FBF_TRY(to_host(res, vectors, static_cast<size_t>(ny)));
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < nroots; ++s) {
      const double overlap = host_dot(root_of(c_host, static_cast<size_t>(ndet), r),
                                      root_of(c_host, static_cast<size_t>(ndet), s));
      if (std::abs(overlap - (r == s ? 1.0 : 0.0)) > kUnitVectorTolerance) {
        return err_config("{}: roots {} and {} overlap by {:.17g}: the vectors must be "
                          "orthonormal within {:g}",
                          kWho, r, s, overlap, kUnitVectorTolerance);
      }
    }
  }

  // Phi^SA's adjoint at the weights; B whole in the MO basis; D (2 on the
  // core's diagonal, gamma_SA on the active block); W at the point; and the
  // summed adjoint's h_bar, Λ and y.
  const ZcAdjoint phi = FBF_TRY(
      zc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, z));
  Buffers::device<double> bmo = FBF_TRY(zeroed(res, lambda_size));
  if (naux > 0) {
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, {.first = 0, .count = nmo}, {.first = 0, .count = nmo},
                 bmo.data()));
  }
  std::vector<double> core_density(static_cast<size_t>(nmo2), 0.0);
  for (int64_t i = 0; i < nc; ++i) {
    core_density[static_cast<size_t>(i * (nmo + 1))] = 2.0;
  }
  Buffers::device<double> dens = FBF_TRY(to_device(res, core_density));
  FBF_TRY(add_onto(res, n, n, x.dm1.data(), dens.data() + nc + (nmo * nc), nmo));
  Buffers::device<double> w_act = FBF_TRY(zeroed(res, n * n));
  FBF_TRY(phi_w(res, naux, nmo, nc, n, bmo.data(), bmo.data(), z, w_act.data()));
  Buffers::device<double> h_total = FBF_TRY(zeroed(res, nmo2));
  FBF_TRY(add_onto(res, nmo2, 1, corr.h_bar.data(), h_total.data(), nmo2));
  FBF_TRY(add_onto(res, nmo2, 1, phi.h_bar.data(), h_total.data(), nmo2));
  Buffers::device<double> lambda_total = FBF_TRY(zeroed(res, lambda_size));
  FBF_TRY(add_onto(res, lambda_size, 1, corr.lambda.data(), lambda_total.data(), lambda_size));
  FBF_TRY(add_onto(res, lambda_size, 1, phi.lambda.data(), lambda_total.data(), lambda_size));
  std::vector<double> y_total = FBF_TRY(to_host(res, corr.y.data(), static_cast<size_t>(ny)));
  {
    const std::vector<double> yc = FBF_TRY(to_host(res, phi.y.data(), static_cast<size_t>(ny)));
    for (size_t k = 0; k < y_total.size(); ++k) {
      y_total[k] += yc[k];
    }
  }
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));

  // Each direction's CI part along the generators, u = Pi(c_dot), and the
  // QD builders' tangent along it.
  const auto projected = [&](const MoDirections &d) -> Result<Buffers::device<double>> {
    const auto m = static_cast<size_t>(d.count);
    const size_t block = static_cast<size_t>(ny);
    const std::vector<double> all = FBF_TRY(to_host(res, d.c, m * block));
    std::vector<double> u(all.size());
    for (size_t k = 0; k < m; ++k) {
      const std::vector<double> uk =
          root_set_projection(c_host, nroots, std::span(all).subspan(k * block, block));
      std::ranges::copy(uk, u.begin() + static_cast<std::ptrdiff_t>(k * block));
    }
    return to_device(res, u);
  };
  const Buffers::device<double> u_all = FBF_TRY(projected(directions));
  const Buffers::device<double> u_partners = FBF_TRY(projected(partners));
  const auto tangents =
      [&](const MoDirections &d,
          const Buffers::device<double> &u) -> Result<std::vector<QdDirectionArrays>> {
    std::vector<QdDirectionArrays> dots;
    for (int64_t k = 0; k < d.count; ++k) {
      dots.push_back(FBF_TRY(qd_direction_arrays(res, alpha, beta, fixed.arrays, weights, vectors,
                                                 d.h + (k * nmo2), d.b + (k * lambda_size),
                                                 u.data() + (k * ny), nullptr)));
    }
    return dots;
  };
  std::vector<QdDirectionArrays> dots = FBF_TRY(tangents(directions, u_all));
  std::vector<QdDirectionArrays> partner_dots = FBF_TRY(tangents(partners, u_partners));
  if (walk_of.check) {
    FBF_TRY(walk_of.check(fixed, dots, u_all.data(), partner_dots, u_partners.data()));
  }

  // sc_effective_hamiltonian_hvp along every direction at once, onto fresh
  // bars: per direction the shared ones, Λ and each root's own; each
  // transition's tangent bars, as they are handed over, folded and through
  // the bra/ket rdms_adjoint at the roots onto that direction's y_walk.
  std::vector<ScArrays> bar_dots;
  std::vector<LambdaBars> lambda_dots;
  std::vector<std::vector<PairRdms>> own_dots;
  std::vector<Buffers::device<double>> y_walk;
  std::vector<EnergyBars> hvp_bars;
  std::vector<EnergyDirection> hvp_dirs;
  std::vector<EnergyDirection> hvp_partners;
  for (size_t k = 0; k < count; ++k) {
    bar_dots.push_back(FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt)));
    lambda_dots.push_back(FBF_TRY(lambda_bars(res, x, nmo)));
    own_dots.emplace_back();
    for (size_t r = 0; r < nroots; ++r) {
      own_dots.back().push_back(FBF_TRY(zeroed_rdms(res, n, true)));
    }
    y_walk.push_back(FBF_TRY(zeroed(res, ny)));
  }
  for (size_t k = 0; k < count; ++k) {
    for (size_t r = 0; r < nroots; ++r) {
      EnergyBars e = energy_bars(bar_dots[k], lambda_dots[k]);
      PairRdms &o = own_dots[k][r];
      e.dm1 = o.dm1.data();
      e.dm2 = o.dm2.data();
      e.dm3 = o.dm3.data();
      e.f3ca = o.f3ca.data();
      e.f3ac = o.f3ac.data();
      hvp_bars.push_back(e);
    }
    std::ranges::copy(dots[k].directions(), std::back_inserter(hvp_dirs));
  }
  for (const QdDirectionArrays &d : partner_dots) {
    std::ranges::copy(d.directions(), std::back_inserter(hvp_partners));
  }
  if (count > 0) {
    Buffers::device<double> bra_bar(static_cast<size_t>(ndet), res.device());
    Buffers::device<double> ket_bar(static_cast<size_t>(ndet), res.device());
    Buffers::device<double> eri_bar(static_cast<size_t>(n * n * n * n), res.device());
    const auto transition = [&](const int64_t u, const int64_t bra, const int64_t ket,
                                const EnergyBars &t) -> Status {
      PairRdms &p = fixed.arrays.pairs[static_cast<size_t>((bra * nn) + ket)];
      FBF_TRY(run_builds_adjoint(
          res, trace_builds(x, p.dm1.data(), p.dm2.data(), p.dm3.data(), t.dm1, t.dm2, t.dm3)));
      FBF_TRY(rdms_adjoint(res, alpha, beta, vectors + (bra * ndet), vectors + (ket * ndet),
                           x.eri.data(), t.dm3, nullptr, nullptr, bra_bar.data(), ket_bar.data(),
                           eri_bar.data()));
      double *const yu = y_walk[static_cast<size_t>(u)].data();
      FBF_TRY(add_onto(res, ndet, 1, bra_bar.data(), yu + (bra * ndet), ndet));
      FBF_TRY(add_onto(res, ndet, 1, ket_bar.data(), yu + (ket * ndet), ndet));
      return {};
    };
    FBF_TRY(walk_of.hvp(x.walk_blocks(), pair_inputs(fixed.arrays, nroots), hvp_dirs, hvp_bars,
                        transition, hvp_partners));
    // The launches above read the scratch, which goes with this branch.
    FBF_TRY(res.sync());
  }

  // An AdjointTangent from a reverse-pass tangent's (h_bar, Λ, raw y), y
  // taken to the generators' (Pi), X = 2 h h_bar + 2 B Λ and A.
  const auto finish = [&](BarsTangent t) -> Result<AdjointTangent> {
    const std::vector<double> y_raw = FBF_TRY(to_host(res, t.y.data(), static_cast<size_t>(ny)));
    AdjointTangent p{.h_bar = std::move(t.h_bar),
                     .lambda = std::move(t.lambda),
                     .y = FBF_TRY(to_device(res, generator_adjoint(c_host, nroots, y_raw, c_host))),
                     .x = FBF_TRY(zeroed(res, nmo2)),
                     .a = Buffers::device<double>(0, res.device())};
    FBF_TRY(add_source(res, naux, nmo, x.h_mo.data(), bmo.data(), p.h_bar.data(), p.lambda.data(),
                       p.x.data()));
    return p;
  };
  // @p from's h_bar, Λ, y and X onto @p to's; A is formed from the sum.
  const auto add_into = [&](const AdjointTangent &from, AdjointTangent &to) -> Status {
    FBF_TRY(add_onto(res, nmo2, 1, from.h_bar.data(), to.h_bar.data(), nmo2));
    FBF_TRY(add_onto(res, lambda_size, 1, from.lambda.data(), to.lambda.data(), lambda_size));
    FBF_TRY(add_onto(res, ny, 1, from.y.data(), to.y.data(), ny));
    return add_onto(res, nmo2, 1, from.x.data(), to.x.data(), nmo2);
  };

  for (size_t k = 0; k < count; ++k) {
    const double *const h_dot = directions.h + (static_cast<int64_t>(k) * nmo2);
    const double *const b_dot = directions.b + (static_cast<int64_t>(k) * lambda_size);
    const double *const uk = u_all.data() + (static_cast<int64_t>(k) * ny);
    ScArrays &dot = dots[k].shared;
    const std::vector<double> u_host = FBF_TRY(to_host(res, uk, static_cast<size_t>(ny)));

    // Phi^SA's tangent: Λ at (Ḃ, D) + (B, Ḋ), Ḋ gamma_SA's tangent; y^c_r's,
    // w_r sigma(W + W^T) at u_r plus w_r sigma(Ẇ + Ẇ^T) at c_r.
    Buffers::device<double> dens_dot = FBF_TRY(zeroed(res, nmo2));
    FBF_TRY(add_onto(res, n, n, dot.dm1.data(), dens_dot.data() + nc + (nmo * nc), nmo));
    Buffers::device<double> phi_lambda_dot = FBF_TRY(zeroed(res, lambda_size));
    FBF_TRY(phi_lambda(res, naux, nmo, b_dot, dens.data(), z, phi_lambda_dot.data()));
    FBF_TRY(phi_lambda(res, naux, nmo, bmo.data(), dens_dot.data(), z, phi_lambda_dot.data()));
    Buffers::device<double> w_dot = FBF_TRY(zeroed(res, n * n));
    FBF_TRY(phi_w(res, naux, nmo, nc, n, b_dot, bmo.data(), z, w_dot.data()));
    FBF_TRY(phi_w(res, naux, nmo, nc, n, bmo.data(), b_dot, z, w_dot.data()));
    Buffers::device<double> phi_y_dot = FBF_TRY(zeroed(res, ny));
    FBF_TRY(phi_sigma(res, sigma, n, w_act.data(), weights, uk, phi_y_dot.data()));
    FBF_TRY(phi_sigma(res, sigma, n, w_dot.data(), weights, vectors, phi_y_dot.data()));

    // The curvature, 1/2 {K_t, K_u} c_r against every root's raw y_r:
    // K^*(M), M = 1/2 sum_r (y_r u_r^T - (K_u y_r) c_r^T).
    std::vector<double> left(static_cast<size_t>(2 * ny));
    std::vector<double> right(static_cast<size_t>(2 * ny));
    for (size_t r = 0; r < nroots; ++r) {
      const auto at = static_cast<std::ptrdiff_t>(r * static_cast<size_t>(ndet));
      const auto second = at + static_cast<std::ptrdiff_t>(ny);
      const std::vector<double> ky =
          generator_apply(c_host, nroots, u_host, root_of(y_total, static_cast<size_t>(ndet), r));
      for (int64_t e = 0; e < ndet; ++e) {
        left[static_cast<size_t>(at + e)] = -0.5 * ky[static_cast<size_t>(e)];
        right[static_cast<size_t>(at + e)] = c_host[static_cast<size_t>(at + e)];
        left[static_cast<size_t>(second + e)] = 0.5 * y_total[static_cast<size_t>(at + e)];
        right[static_cast<size_t>(second + e)] = u_host[static_cast<size_t>(at + e)];
      }
    }
    AdjointTangent curvature{
        .h_bar = FBF_TRY(zeroed(res, nmo2)),
        .lambda = FBF_TRY(zeroed(res, lambda_size)),
        .y = FBF_TRY(to_device(res, generator_adjoint(c_host, nroots, left, right))),
        .x = FBF_TRY(zeroed(res, nmo2)),
        .a = Buffers::device<double>(0, res.device())};

    // Phi^SA's share: its Λ and y tangents, X's 2 B Λ̇^c, and its one-slot
    // X terms 2 ḣ z + 2 Ḃ Λ^c.
    const auto phi_share = [&]() -> Result<AdjointTangent> {
      AdjointTangent p = FBF_TRY(finish(BarsTangent{.h_bar = FBF_TRY(zeroed(res, nmo2)),
                                                    .lambda = std::move(phi_lambda_dot),
                                                    .y = std::move(phi_y_dot)}));
      FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, z, phi.lambda.data(), p.x.data()));
      return p;
    };
    const auto step = [&](std::vector<PairRdms> &own_dot, ScArrays &bar_dot,
                          const Buffers::device<double> *walk_y, const bool one_slot) {
      return [&, walk_y, one_slot] {
        return qd_rdms_tangent(res, alpha, beta, weights, vectors, uk, fixed, dot, bar_dot, own_dot,
                               walk_y, one_slot);
      };
    };

    if (!parts) {
      // One sweep: the walk's tangent bars and the one-slot terms together,
      // X's own one-slot share 2 ḣ h̄ + 2 Ḃ Λ at the summed adjoint.
      AdjointTangent product = FBF_TRY(finish(
          FBF_TRY(bars_tangent(res, nmo, x, fixed.bar, dot, bar_dots[k], lambda_dots[k].whole, true,
                               step(own_dots[k], bar_dots[k], &y_walk[k], true)))));
      FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, corr.h_bar.data(), corr.lambda.data(),
                         product.x.data()));
      FBF_TRY(add_into(FBF_TRY(phi_share()), product));
      FBF_TRY(add_into(curvature, product));
      product.a = FBF_TRY(antisymmetric_part(res, nmo, product.x.data()));
      out.products.push_back(std::move(product));
      continue;
    }
    // The parts, each a whole AdjointTangent, and the product their sum.
    AdjointTangent walk_part = FBF_TRY(
        finish(FBF_TRY(bars_tangent(res, nmo, x, fixed.bar, dot, bar_dots[k], lambda_dots[k].whole,
                                    false, step(own_dots[k], bar_dots[k], &y_walk[k], false)))));
    ScArrays none = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
    Buffers::device<double> no_lambda = FBF_TRY(zeroed(res, lambda_size));
    std::vector<PairRdms> no_own;
    for (size_t r = 0; r < nroots; ++r) {
      no_own.push_back(FBF_TRY(zeroed_rdms(res, n, true)));
    }
    AdjointTangent one_slot_part = FBF_TRY(finish(FBF_TRY(bars_tangent(
        res, nmo, x, fixed.bar, dot, none, no_lambda, true, step(no_own, none, nullptr, true)))));
    FBF_TRY(add_source(res, naux, nmo, h_dot, b_dot, corr.h_bar.data(), corr.lambda.data(),
                       one_slot_part.x.data()));
    AdjointTangent phi_part = FBF_TRY(phi_share());
    std::array<AdjointTangent, 4> split{std::move(walk_part), std::move(one_slot_part),
                                        std::move(phi_part), std::move(curvature)};
    AdjointTangent product{.h_bar = FBF_TRY(zeroed(res, nmo2)),
                           .lambda = FBF_TRY(zeroed(res, lambda_size)),
                           .y = FBF_TRY(zeroed(res, ny)),
                           .x = FBF_TRY(zeroed(res, nmo2)),
                           .a = Buffers::device<double>(0, res.device())};
    for (AdjointTangent &p : split) {
      p.a = FBF_TRY(antisymmetric_part(res, nmo, p.x.data()));
      FBF_TRY(add_into(p, product));
    }
    product.a = FBF_TRY(antisymmetric_part(res, nmo, product.x.data()));
    out.products.push_back(std::move(product));
    out.parts.push_back(std::move(split));
  }
  FBF_TRY(res.sync());
  return out;
}

Result<QdScHessianProduct> qd_sc_hessian_product(
    DeviceResources &res, const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
    const int32_t naux, const int32_t nao, const int32_t nmo, const double *b, const double *c,
    const double *h, const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
    const double *vectors, const std::span<const double> w, const double *z, const int32_t batch,
    const MoDirections &directions, const MoDirections &partners, const bool parts) {
  constexpr std::string_view kWho = "nevpt2::qd_sc_hessian_product";
  QdScHessianProduct out;
  const QdProductWalk walk{
      .adjoint = [&](std::optional<QdFixedBars> &kept) -> Result<QdScAdjoint> {
        std::vector<QdScAdjoint> adj =
            FBF_TRY(qd_adjoints(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                                weights, vectors, w, batch, nullptr, 0, 0.0, nullptr, &kept));
        out.elements = std::move(adj.front().elements);
        return std::move(adj.front());
      },
      .check = {},
      .hvp = [&](const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
                 std::span<const EnergyDirection> dirs, std::span<const EnergyBars> bars,
                 const auto &transition, std::span<const EnergyDirection> partner_dirs) -> Status {
        ScEffectiveHamiltonianHvp walked = FBF_TRY(sc_effective_hamiltonian_hvp(
            res, blocks, pairs, w, batch, dirs, bars, transition, partner_dirs));
        out.populated = std::move(walked.populated);
        out.defect = std::move(walked.defect);
        out.coupling_defect = std::move(walked.coupling_defect);
        out.coupling_k6 = std::move(walked.coupling_k6);
        out.rounding = walked.rounding;
        out.rounding_at = walked.rounding_at;
        out.rounding_norm = walked.rounding_norm;
        return {};
      }};
  QdProducts p =
      FBF_TRY(qd_hessian_product(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                                 weights, vectors, w, z, directions, partners, parts, walk));
  out.products = std::move(p.products);
  out.parts = std::move(p.parts);
  return out;
}

// A Schur-lift overlap past this many units of its round-off (PcMetricLifts::
// overlap) is a first-order overlap of the lifted functions with the bra:
// the cut's own multiple of σ's round-off (100 ε √d), which every structural
// null's has measured below (README "The Schur lift on every model state").
constexpr double kPcLiftOverlap = 100.0;
// A null mode lifts when it reaches the cut within this step along the
// direction, in its own units (t* below it): the step #817's single-state
// assembly refuses on (hessian's kPcNevpt2LiftStep, 1e-3 bohr for its
// nuclear directions). A mode that needs more is no lift within any step a
// Hessian takes, and a direction that the response leaves at round-off by
// symmetry lifts nothing (README "The Schur lift on every model state").
constexpr double kPcLiftStep = 1e-3;

// The eight classes, in the energies' order, for the refusal's message.
constexpr std::array<std::string_view, 8> kPcClassNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                        "Srsi", "Srs", "Sij",   "Sir"};

// qd_pc_lifts_footprint (declared in :builders).
MemoryFootprint qd_pc_lifts_footprint(const int32_t ncore, const int32_t ncas, const int32_t nvirt,
                                      const int32_t n_alpha, const int32_t n_beta,
                                      const int32_t nroots,
                                      const std::span<const size_t> syevd_lwork,
                                      const std::span<const std::array<int64_t, 8>> dropped) {
  const auto n = static_cast<size_t>(ncas);
  const size_t n2 = sizeof(double) * n * n;
  const size_t n4 = n2 * n * n;
  const size_t n6 = n4 * n * n;
  const size_t ndet = product_sizes(0, 0, ncas, n_alpha, n_beta).ndet;
  Held h;
  h.take(n6); // part
  for (int32_t r = 0; r < nroots; ++r) {
    // K u_r and the second derivative's dm3; the two bra/ket rdms and the
    // one-vector one, no f3; its traces; the lifts.
    const size_t held = ndet + n6;
    h.take(held);
    h.call(rdms_footprint(ncas, 1, true, false));
    h.call(rdms_footprint(ncas, 1, true, false));
    h.call(rdms_footprint(ncas, 1, false, false));
    h.take(n4 + n2);
    h.call(pc_metric_lifts_footprint(ncore, ncas, nvirt, nroots, syevd_lwork,
                                     dropped.empty() ? dropped
                                                     : dropped.subspan(static_cast<size_t>(r), 1)));
    h.give(held + n4 + n2);
  }
  h.give(n6);
  return h.footprint(0);
}

Result<QdPcHessianProduct> qd_pc_hessian_product(
    DeviceResources &res, const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
    const int32_t naux, const int32_t nao, const int32_t nmo, const double *b, const double *c,
    const double *h, const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
    const double *vectors, const std::span<const double> w, const double *z, const int32_t batch,
    const MoDirections &directions, const MoDirections &partners, const bool parts,
    const double pc_floor, const bool span) {
  constexpr std::string_view kWho = "nevpt2::qd_pc_hessian_product";
  QdPcHessianProduct out;
  const size_t nroots = weights.size();
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  // Ket N's lifts and overlaps along one direction's QD builders' tangent
  // @p dot, its CI part @p u along the generators: the path of N's own RDMs
  // is c_N(t) = e^{tK} c_N, so their second derivative is
  // Γ(K u_N, c_N) + Γ(c_N, K u_N) + 2 Γ(u_N, u_N) (dm3's from nevpt2::rdms,
  // dm2 and dm1 its traces; at n = 1, 2 γ(u, u) - 2 |u|² γ(c, c)), and each
  // transition (N, M)'s first derivative the tangent's own.
  //
  // The lifts are the direction's in the point's sign gauge, the roots' and
  // the active MOs' signs, which the SA-CASSCF sets anew each run (#1203):
  // a fixed raw u is another direction once a sign comes back flipped. On
  // the footprint test's SA-2 water, six processes gave ket 0's Sr lift
  // 0.33842 (three times), 0.33585, 0.35837 and 0.36020, each value a sign
  // pattern of the roots and the active MOs; within one process they repeat
  // to 1e-10 and the kept ranks and null modes do not move (σ ≤ 1e-16 of
  // the largest, the kept ≥ 5e-8). Negating root 0 alone moves 0.33842 to
  // 0.33585; negating its u and W's row and column with it gives the same
  // lifts, and the product but for root 0's y negated, to 3e-12.
  const auto lifts_along = [&](QdFixedBars &fixed, const QdDirectionArrays &dot, const double *u,
                               std::vector<PcMetricLifts> &into,
                               std::vector<double> &uu) -> Status {
    ScArrays &x = fixed.arrays.x;
    const int64_t n = ncas;
    const int64_t n6 = n * n * n * n * n * n;
    const std::vector<double> c_host =
        FBF_TRY(to_host(res, vectors, static_cast<size_t>(nroots) * static_cast<size_t>(ndet)));
    const std::vector<double> u_host =
        FBF_TRY(to_host(res, u, static_cast<size_t>(nroots) * static_cast<size_t>(ndet)));
    const std::vector<EnergyInputs> pairs = pair_inputs(fixed.arrays, nroots, pc_floor);
    std::vector<EnergyInputs> first;
    for (const EnergyDirection &d : dot.directions()) {
      first.push_back(d.in);
    }
    Buffers::device<double> part = FBF_TRY(zeroed(res, n6));
    for (size_t r = 0; r < nroots; ++r) {
      const std::span<const double> ur = root_of(u_host, static_cast<size_t>(ndet), r);
      const double norm = host_dot(ur, ur);
      uu.push_back(norm);
      PcMetricLifts lifts{};
      lifts.overlap.assign(nroots, {});
      if (norm > 0.0) {
        const double *const cr = vectors + (static_cast<int64_t>(r) * ndet);
        const double *const urd = u + (static_cast<int64_t>(r) * ndet);
        const Buffers::device<double> ku =
            FBF_TRY(to_device(res, generator_apply(c_host, nroots, u_host, ur)));
        Buffers::device<double> second3 = FBF_TRY(zeroed(res, n6));
        for (const auto &[bra, ket] : {std::pair{ku.data(), cr}, std::pair{cr, ku.data()}}) {
          FBF_TRY(rdms(res, alpha, beta, bra, ket, x.eri.data(), nullptr, nullptr, part.data(),
                       nullptr, nullptr));
          FBF_TRY(contract(res, "a->a", {column_operand(part.data(), {n6})}, second3.data(), {n6},
                           1.0));
        }
        FBF_TRY(rdms(res, alpha, beta, urd, x.eri.data(), part.data(), nullptr, nullptr));
        FBF_TRY(
            contract(res, "a->a", {column_operand(part.data(), {n6})}, second3.data(), {n6}, 2.0));
        Buffers::device<double> second2 = FBF_TRY(zeroed(res, n * n * n * n));
        Buffers::device<double> second1 = FBF_TRY(zeroed(res, n * n));
        FBF_TRY(run_builds(res, trace_builds(x, second1.data(), second2.data(), second3.data(),
                                             nullptr, nullptr, nullptr)));
        const EnergyInputs second{
            .dm1 = second1.data(), .dm2 = second2.data(), .dm3 = second3.data()};
        lifts = FBF_TRY(pc_metric_lifts(res, x.walk_blocks(), pairs, first, second,
                                        static_cast<int64_t>(r), kPcLiftStep));
        for (double &l : lifts.lift) {
          l /= norm;
        }
        // The launches above read this branch's buffers.
        FBF_TRY(res.sync());
      }
      into.push_back(std::move(lifts));
    }
    return {};
  };
  // t* in the direction's own units: the null mode's ½ ℓ |u_N|² t² reaches
  // the cut there.
  const auto steps_of = [](const PcMetricLifts &l, const double norm) {
    std::array<double, 8> steps{};
    for (size_t k = 0; k < steps.size(); ++k) {
      steps[k] = l.lift[k] > 0.0 && norm > 0.0 ? std::sqrt(2.0 * l.cut[k] / (l.lift[k] * norm))
                                               : std::numeric_limits<double>::infinity();
    }
    return steps;
  };
  const QdProductWalk walk{
      .adjoint = [&](std::optional<QdFixedBars> &kept) -> Result<QdScAdjoint> {
        std::vector<QdScAdjoint> adj = FBF_TRY(
            qd_adjoints(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights,
                        vectors, w, batch, &out.elements, 0, pc_floor, nullptr, &kept));
        return std::move(adj.front());
      },
      // The Schur lift on every model state, before the walk: per direction
      // and every ket N, its lifts, t* and each bra's overlap; a ket whose
      // class's null modes reach a bra M at first order along the direction
      // and along its partner (itself without partners) refuses the block
      // (Eq. qd-pcfail).
      .check = [&](QdFixedBars &fixed, std::span<const QdDirectionArrays> dots, const double *u,
                   std::span<const QdDirectionArrays> partner_dots,
                   const double *u_partners) -> Status {
        const int64_t ny = static_cast<int64_t>(nroots) * ndet;
        for (size_t k = 0; k < dots.size(); ++k) {
          std::vector<PcMetricLifts> own;
          std::vector<double> uu;
          FBF_TRY(lifts_along(fixed, dots[k], u + (static_cast<int64_t>(k) * ny), own, uu));
          std::vector<PcMetricLifts> other;
          std::vector<double> vv;
          if (!partner_dots.empty()) {
            FBF_TRY(lifts_along(fixed, partner_dots[k], u_partners + (static_cast<int64_t>(k) * ny),
                                other, vv));
          }
          const std::vector<PcMetricLifts> &with = partner_dots.empty() ? own : other;
          const std::vector<double> &with_uu = partner_dots.empty() ? uu : vv;
          out.lifts.emplace_back();
          out.lift_steps.emplace_back();
          out.overlaps.emplace_back();
          for (size_t ket = 0; ket < nroots; ++ket) {
            out.lifts.back().push_back(own[ket].lift);
            out.lift_steps.back().push_back(steps_of(own[ket], uu[ket]));
            out.overlaps.back().push_back(own[ket].overlap);
            for (size_t bra = 0; bra < nroots; ++bra) {
              for (size_t cls = 0; cls < 8; ++cls) {
                const double a = own[ket].overlap[bra][cls];
                const double e = with[ket].overlap[bra][cls];
                if (bra == ket || a <= kPcLiftOverlap || e <= kPcLiftOverlap) {
                  continue;
                }
                return err_numerical(
                    "{}: the coupling ({}, {}) has no Hessian on directions ({}, partner {}): "
                    "ket {}'s class {} null modes lift (t* {:.3e} along the direction, {:.3e} "
                    "along its partner) with a first-order overlap to bra {} ({:.3e} and "
                    "{:.3e} of their round-off, past {:.0f}): the cut keeps them dropped, and "
                    "the coupling misses a first-order piece (Eq. qd-pcfail)",
                    kWho, bra, ket, k, k, ket, kPcClassNames[cls], steps_of(own[ket], uu[ket])[cls],
                    steps_of(with[ket], with_uu[ket])[cls], bra, a, e, kPcLiftOverlap);
              }
            }
          }
        }
        return {};
      },
      .hvp = [&](const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
                 std::span<const EnergyDirection> dirs, std::span<const EnergyBars> bars,
                 const auto &transition, std::span<const EnergyDirection>) -> Status {
        // The pairs at the floor, which the kets' solves read.
        std::vector<EnergyInputs> floored(pairs.begin(), pairs.end());
        for (EnergyInputs &in : floored) {
          in.pc_floor = pc_floor;
        }
        FBF_TRY(pc_effective_hamiltonian_hvp(res, blocks, floored, w, batch, dirs, bars, transition,
                                             span));
        return {};
      }};
  QdProducts p =
      FBF_TRY(qd_hessian_product(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas,
                                 weights, vectors, w, z, directions, partners, parts, walk));
  out.products = std::move(p.products);
  out.parts = std::move(p.parts);
  return out;
}

// ---- The QD products' footprints --------------------------------------------

// The n² EnergyDirections of one QD direction, as QdDirectionArrays::
// directions() gives them and the footprints read them: every field given,
// f3ca and f3ac on the diagonal alone.
std::vector<EnergyDirection> qd_given_directions(const size_t nroots) {
  std::vector<EnergyDirection> out;
  for (size_t p = 0; p < nroots * nroots; ++p) {
    EnergyInputs in = given_inputs();
    if (p % (nroots + 1) != 0) {
      in.f3ca = nullptr;
      in.f3ac = nullptr;
    }
    out.push_back({.in = in, .blocks = given_blocks()});
  }
  return out;
}

// What a QD product's pair walk asks of the footprint: its reverse pass's
// with the QdFixedBars kept (qd_adjoint_footprint's, fixed), its
// Hessian-vector product's along the directions and the partners, its
// transition handed every pair, and before the walk its check.
struct QdProductWalkBytes {
  MemoryFootprint adjoint;
  std::function<MemoryFootprint(std::span<const EnergyDirection>, std::span<const EnergyDirection>,
                                const MemoryFootprint &)>
      hvp;
  std::function<void(Held &)> check;
};

// qd_hessian_product, every allocation in its order onto a Held, in bytes:
// qd_sc_hessian_product_footprint's and qd_pc_hessian_product_footprint's
// one body.
MemoryFootprint qd_hessian_product_footprint(const int32_t naux, const int32_t nao,
                                             const int32_t nmo, const int32_t ncore,
                                             const int32_t ncas, const int32_t n_alpha,
                                             const int32_t n_beta, const int32_t nroots,
                                             const int32_t count, const int32_t partners,
                                             const bool parts, const QdProductWalkBytes &walk) {
  constexpr size_t kD = sizeof(double);
  const int32_t nvirt = nmo - ncore - ncas;
  const auto n = static_cast<size_t>(ncas);
  const size_t n2 = kD * n * n;
  const size_t n4 = n2 * n * n;
  const size_t n6 = n4 * n * n;
  const ProductSizes z = product_sizes(naux, nmo, ncas, n_alpha, n_beta);
  const auto roots = static_cast<size_t>(nroots);
  const size_t ny = roots * z.ndet;
  const size_t arrays = zeroed_arrays_bytes(naux, ncore, ncas, nvirt);
  const size_t own = roots * (n2 + n4 + (3 * n6));
  const auto m = static_cast<size_t>(count);
  const size_t product = adjoint_tangent_bytes(z, ny);
  Held h;
  // The pair walk's reverse pass: the QdScAdjoint and the QdFixedBars kept.
  h.call(walk.adjoint);
  // Phi^SA's adjoint at the weights; B whole; D, W, and the summed h_bar
  // and Λ; the Sigma.
  h.call(zc_adjoint_footprint(naux, nao, nmo, ncas, n_alpha, n_beta, nroots));
  h.take(z.lambda);
  if (naux > 0) {
    h.borrow(b_mo_footprint(naux, nao, {.first = 0, .count = nmo}, {.first = 0, .count = nmo}));
  }
  h.take(z.nn + n2);
  h.take(phi_w_bytes(naux, nmo, ncas));
  h.give(phi_w_bytes(naux, nmo, ncas));
  h.take(z.nn + z.lambda);
  SigmaBytes sigma = sigma_bytes(ncas, n_alpha, n_beta);
  sigma.create(h);
  // u over the directions and the partners; the QD builders' tangent along
  // each, kept; the check.
  h.take((m + static_cast<size_t>(partners)) * ny);
  const MemoryFootprint dot =
      qd_direction_arrays_footprint(naux, ncore, ncas, nvirt, nroots, false);
  for (int32_t k = 0; k < count + partners; ++k) {
    h.call(dot);
  }
  if (walk.check) {
    walk.check(h);
  }
  // Per direction its fresh bars: the shared ones, Λ, each root's own and
  // y_walk, kept; the transition's outputs beside the pair walk.
  h.take(m * (arrays + z.lambda + own + ny));
  if (count > 0) {
    h.take((2 * z.ndet) + n4);
    std::vector<EnergyDirection> dirs;
    std::vector<EnergyDirection> others;
    for (size_t k = 0; k < m; ++k) {
      std::ranges::copy(qd_given_directions(roots), std::back_inserter(dirs));
    }
    for (int32_t k = 0; k < partners; ++k) {
      std::ranges::copy(qd_given_directions(roots), std::back_inserter(others));
    }
    h.call(walk.hvp(dirs, others, qd_adjoint_transition_footprint(ncas, false)));
    h.give((2 * z.ndet) + n4);
  }

  // The reverse pass's tangent along one direction: qd_rdms_tangent's y
  // (kept) and the folds' h_bar and Λ; then finish: y projected and X, the
  // raw y given back.
  const auto finished = [&](const bool one_slot) {
    bars_tangent_bytes(h, naux, ncore, ncas, nvirt, z, ny, [&](Held &held) {
      held.call(qd_rdms_tangent_footprint(ncas, n_alpha, n_beta, nroots, one_slot));
    });
    h.take(ny + z.nn);
    h.give(ny);
  };
  // Phi^SA's share: a zero h_bar, then finish over its Λ̇ and ẏ.
  const auto phi_share = [&] {
    h.take(z.nn);
    h.take(ny + z.nn);
    h.give(ny);
  };
  size_t kept = 0;
  for (size_t k = 0; k < m; ++k) {
    // Phi^SA's tangent: Ḋ, Λ̇ (two phi_lambda), Ẇ (two phi_w), ẏ (two
    // phi_sigma over the roots).
    h.take(z.nn + z.lambda);
    for (int32_t side = 0; side < 2; ++side) {
      h.take(phi_lambda_bytes(naux, z));
      h.give(phi_lambda_bytes(naux, z));
    }
    h.take(n2);
    for (int32_t side = 0; side < 2; ++side) {
      h.take(phi_w_bytes(naux, nmo, ncas));
      h.give(phi_w_bytes(naux, nmo, ncas));
    }
    h.take(ny);
    for (int32_t side = 0; side < 2; ++side) {
      sigma.call(h, ny);
    }
    // The curvature: zero h_bar, Λ and X beside its y.
    const size_t curvature = (2 * z.nn) + z.lambda + ny;
    h.take(curvature);
    if (!parts) {
      // The product in one sweep, Phi^SA's share added and given back, then A.
      finished(true);
      phi_share();
      h.give(adjoint_tangent_bytes(z, ny) - z.nn); // Phi's share: no A
      h.take(z.nn);
      kept += product;
      // The iteration's own: Ḋ, Ẇ and the curvature.
      h.give(z.nn + n2 + curvature);
      continue;
    }
    // The walk's part, the one-slot part over zeroed bars, Λ and own bars,
    // Phi's; the product, zeroed; every part's A and the product's.
    finished(false);
    const size_t none = arrays + z.lambda + own;
    h.take(none);
    finished(true);
    phi_share();
    h.take((2 * z.nn) + z.lambda + ny);
    h.take(5 * z.nn);
    kept += 5 * product;
    // The iteration's own: Ḋ, Ẇ and the zeroed bars.
    h.give(z.nn + n2 + none);
  }
  // Everything but the products and their parts goes.
  h.give(h.live - kept);
  return h.footprint(kept);
}

MemoryFootprint
qd_sc_hessian_product_footprint(const int32_t naux, const int32_t nao, const int32_t nmo,
                                const int32_t ncore, const int32_t ncas, const int32_t n_alpha,
                                const int32_t n_beta, const int32_t nroots, const int32_t batch,
                                const std::span<const double> w, const int32_t count,
                                const int32_t partners, const bool parts) {
  const int32_t nvirt = nmo - ncore - ncas;
  const int32_t nelecas = n_alpha + n_beta;
  const QdProductWalkBytes walk{
      .adjoint = qd_adjoint_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, nroots, 0,
                                      sc_effective_hamiltonian_adjoint_footprint(
                                          naux, ncore, ncas, nvirt, nelecas, nroots, batch,
                                          qd_adjoint_transition_footprint(ncas, true)),
                                      true),
      .hvp =
          [&](std::span<const EnergyDirection> dirs, std::span<const EnergyDirection> with,
              const MemoryFootprint &transition) {
            return sc_effective_hamiltonian_hvp_footprint(naux, ncore, ncas, nvirt, nelecas, nroots,
                                                          batch, w, dirs, with, transition);
          },
      .check = {}};
  return qd_hessian_product_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, nroots, count,
                                      partners, parts, walk);
}

MemoryFootprint
qd_pc_hessian_product_footprint(const int32_t naux, const int32_t nao, const int32_t nmo,
                                const int32_t ncore, const int32_t ncas, const int32_t n_alpha,
                                const int32_t n_beta, const int32_t nroots, const int32_t batch,
                                const std::span<const size_t> syevd_lwork, const int32_t count,
                                const int32_t partners, const bool parts,
                                const std::span<const std::array<int64_t, 8>> dropped) {
  const int32_t nvirt = nmo - ncore - ncas;
  const QdProductWalkBytes walk{
      .adjoint = qd_adjoint_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, nroots, 0,
                                      pc_effective_hamiltonian_adjoint_footprint(
                                          naux, ncore, ncas, nvirt, nroots, batch, syevd_lwork,
                                          dropped, qd_adjoint_transition_footprint(ncas, true)),
                                      true),
      .hvp =
          [&](std::span<const EnergyDirection> dirs, std::span<const EnergyDirection>,
              const MemoryFootprint &transition) {
            return pc_effective_hamiltonian_hvp_footprint(naux, ncore, ncas, nvirt, nroots, batch,
                                                          syevd_lwork, dirs, dropped, transition);
          },
      // The Schur lifts along every direction, and its partner's.
      .check =
          [&](Held &h) {
            const MemoryFootprint lifts = qd_pc_lifts_footprint(ncore, ncas, nvirt, n_alpha, n_beta,
                                                                nroots, syevd_lwork, dropped);
            for (int32_t k = 0; k < count; ++k) {
              h.call(lifts);
              if (partners > 0) {
                h.call(lifts);
              }
            }
          }};
  return qd_hessian_product_footprint(naux, nao, nmo, ncore, ncas, n_alpha, n_beta, nroots, count,
                                      partners, parts, walk);
}

} // namespace free_breakfast::nevpt2
