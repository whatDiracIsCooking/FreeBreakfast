/// @file
/// @brief sc_hessian_product: the tangent of sc_adjoint + zc_adjoint at
/// fixed z^c along directions (h_dot, B_dot, c_dot) in the MO basis, forward
/// over the reverse pass. The builders' forward tangent feeds
/// class_energies_hvp; its tangent bars then run back through the builders'
/// adjoints, the RDM step and the folds, beside the one-slot-dotted terms
/// those adjoints take at the fixed bars; Phi's share is closed form; and
/// the CI direction is on the unit sphere through the Rayleigh-normalized
/// functional.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, geam, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
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

// How far |c|^2 may sit from 1: the Rayleigh normalization's formulas are
// taken at a unit vector, and a converged CI root is one to round-off.
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
// consumed; @p dot the builders' tangent along the direction, @p u its CI
// part. Every build runs backwards at the tangent bars, the last first, as
// builders_adjoint runs it at the bars; with @p one_slot, each also adds
// its terms at the fixed bars @p bar with one operand at its tangent, and
// so do the RDM step (rdms_adjoint is bilinear in its bars and c, and
// linear in (tu|vw) through f3) and h2e's adjoint (linear in B).
Result<BarsTangent> bars_tangent(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                 const ci::DeviceStringSet &beta, const int32_t nmo,
                                 const double *vector, const double *u, ScArrays &x, ScArrays &bar,
                                 ScArrays &dot, ScArrays &bar_dot,
                                 Buffers::device<double> &lambda_raw, const bool one_slot) {
  const int64_t n = x.blocks.ncas;
  const int64_t n4 = n * n * n * n;
  const int64_t n6 = n4 * n * n;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
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

  // The RDM step. Its outputs land in buffers of their own, added on: each
  // rdms_adjoint overwrites its outputs and takes the workspace.
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
                       bar_dot.f3ca.data(), bar_dot.f3ac.data(), y_part.data(), eri_part.data(),
                       FBF_TRY(rdm_budget())));
  FBF_TRY(add_y(y_part));
  FBF_TRY(add_eri());
  Buffers::device<double> no_dm3(0, res.device());
  if (one_slot) {
    // c's tangent at the fixed bars. With R(a, b) the bra/ket build, y is
    // ∂/∂a <R̄, R(a, c)> + ∂/∂b <R̄, R(c, b)>, so its tangent along u is the
    // bra's bar at (c, u) plus the ket's at (u, c); (tu|vw)'s bar takes both
    // builds' bars.
    FBF_TRY(rdms_adjoint(res, alpha, beta, vector, u, x.eri.data(), bar.dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), y_part.data(), spare.data(), eri_part.data(),
                         FBF_TRY(rdm_budget())));
    FBF_TRY(add_y(y_part));
    FBF_TRY(add_eri());
    FBF_TRY(rdms_adjoint(res, alpha, beta, u, vector, x.eri.data(), bar.dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), spare.data(), y_part.data(), eri_part.data(),
                         FBF_TRY(rdm_budget())));
    FBF_TRY(add_y(y_part));
    FBF_TRY(add_eri());
    // (tu|vw)'s tangent: f3 is linear in it and dm3 does not read it, so y's
    // share is the f3 bars' adjoint at (tu|vw)'s tangent, dm3's bar zero. Its
    // (tu|vw) bar does not move with (tu|vw), and is dropped.
    no_dm3 = FBF_TRY(zeroed(res, n6));
    FBF_TRY(rdms_adjoint(res, alpha, beta, vector, dot.eri.data(), no_dm3.data(), bar.f3ca.data(),
                         bar.f3ac.data(), y_part.data(), eri_part.data(), FBF_TRY(rdm_budget())));
    FBF_TRY(add_y(y_part));
  }

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

// y = sigma(W + W^T) applied to @p vector, W column-major ncas^2 on the
// device: zc_adjoint's y^c at weight 1, onto @p y.
Status phi_sigma(DeviceResources &res, ci::Sigma &sigma, const int64_t ncas, const double *w_act,
                 const double *vector, double *y) {
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
  Buffers::device<double> out(static_cast<size_t>(ndet), res.device());
  FBF_TRY(sigma.apply(res, 1, vector, out.data()));
  FBF_TRY(add_onto(res, ndet, 1, out.data(), y, ndet));
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
    FBF_TRY(phi_sigma(res, sigma, n, w_act.data(), uk, phi_y_dot.data()));
    FBF_TRY(phi_sigma(res, sigma, n, w_dot.data(), vector, phi_y_dot.data()));

    // The product: the reverse pass's tangent at the class walk's tangent
    // bars with the one-slot terms, plus Phi's; X's own one-slot share is
    // 2 ḣ h̄ + 2 Ḃ Λ at the summed adjoint.
    BarsTangent t = FBF_TRY(bars_tangent(res, alpha, beta, nmo, vector, uk, x, bar, dot,
                                         both.bars[k], both.lambdas[k].whole, true));
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
    AdjointTangent slab_part =
        FBF_TRY(finish(FBF_TRY(bars_tangent(res, alpha, beta, nmo, vector, uk, x, bar, dot,
                                            slabs->bars[k], slabs->lambdas[k].whole, false)),
                       false));
    AdjointTangent builder_part =
        FBF_TRY(finish(FBF_TRY(bars_tangent(res, alpha, beta, nmo, vector, uk, x, bar, dot,
                                            builders->bars[k], builders->lambdas[k].whole, false)),
                       false));
    ScArrays none = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
    Buffers::device<double> no_lambda = FBF_TRY(zeroed(res, lambda_size));
    AdjointTangent one_slot_part =
        FBF_TRY(finish(FBF_TRY(bars_tangent(res, alpha, beta, nmo, vector, uk, x, bar, dot, none,
                                            no_lambda, true)),
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
        if (both) {
          out.populated = std::move(got.populated);
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
            FBF_TRY(rdms(res, alpha, beta, uk, x.eri.data(), dm3.data(), nullptr, nullptr,
                         FBF_TRY(rdm_budget())));
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

} // namespace free_breakfast::nevpt2
