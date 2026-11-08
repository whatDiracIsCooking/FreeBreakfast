/// @file
/// @brief builders_tangent: the forward tangent of the per-root input
/// builders along directions (h_dot, B_dot, c_dot) in the MO basis, over the
/// same einsum terms the forward and the reverse pass run (fock_builds), each
/// term by its product rule; the RDMs' through the bra/ket nevpt2::rdms; and
/// the generalized Fock's core-core and virtual-virtual blocks, which the
/// Hessian's kappa_oo response divides. qd_builders_tangent: the same over a
/// state average, at gamma_SA's tangent, with every ordered pair's RDMs'.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo
import free_breakfast.ci.strings;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.rdm;
import :operands; // column_operand, contract, zeroed, add_onto
import :builders; // ScArrays, FockBuild, the builders, sc_arrays, qd_arrays

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// f's core-core and virtual-virtual blocks over x's arrays, B_vv whole at
// @p vv (naux x nvirt^2), into @p f_core and @p f_virt:
//   f_pq = h_pq + Σ_L B_L,pq (2 d_L + g_L) − Σ_j (pj|jq) − ½ Σ_tu γ_tu (pt|uq),
// zc_adjoint's f, whose diagonals are fock_builds' e_core and e_virt term for
// term. Labels and layouts as fock_builds'. No bars: nothing runs these
// backwards.
std::vector<FockBuild> f_oo_builds(const ScArrays &x, const double *vv, double *f_core,
                                   double *f_virt) {
  const int64_t naux = x.blocks.naux;
  const int64_t nc = x.blocks.ncore;
  const int64_t n = x.blocks.ncas;
  const int64_t nv = x.blocks.nvirt;
  const int64_t nmo = nc + n + nv;
  const int64_t nocc = nc + n;
  const EinsumOperand ca = column_operand(x.blocks.ca.data(), {naux, nc, n});
  const EinsumOperand va = column_operand(x.blocks.va.data(), {naux, nv, n});
  const EinsumOperand vc = column_operand(x.blocks.vc.data(), {naux, nv, nc});
  const EinsumOperand cc = column_operand(x.cc.data(), {naux, nc, nc});
  const EinsumOperand vvo = column_operand(vv, {naux, nv, nv});
  const EinsumOperand d = column_operand(x.d.data(), {naux});
  const EinsumOperand g = column_operand(x.g.data(), {naux});
  const EinsumOperand dm1 = column_operand(x.dm1.data(), {n, n});
  const auto h_view = [&](const int64_t p0, const int64_t rows) {
    return EinsumOperand{
        .data = x.h_mo.data() + p0 + (nmo * p0), .dims = {rows, rows}, .strides = {1, nmo}};
  };
  const std::vector<double *> none2{nullptr, nullptr};
  const std::vector<double *> none3{nullptr, nullptr, nullptr};
  return {
      {.out = f_core,
       .out_bar = nullptr,
       .dims = {nc, nc},
       .terms = {{"ij->ij", {h_view(0, nc)}, {nullptr}, 1.0},
                 {"Lij,L->ij", {cc, d}, none2, 2.0},
                 {"Lij,L->ij", {cc, g}, none2, 1.0},
                 {"Lik,Lkj->ij", {cc, cc}, none2, -1.0},
                 {"ut,Lit,Lju->ij", {dm1, ca, ca}, none3, -0.5}}},
      {.out = f_virt,
       .out_bar = nullptr,
       .dims = {nv, nv},
       .terms = {{"ab->ab", {h_view(nocc, nv)}, {nullptr}, 1.0},
                 {"Lab,L->ab", {vvo, d}, none2, 2.0},
                 {"Lab,L->ab", {vvo, g}, none2, 1.0},
                 {"Laj,Lbj->ab", {vc, vc}, none2, -1.0},
                 {"ut,Lat,Lbu->ab", {dm1, va, va}, none3, -0.5}}},
  };
}

// The forward tangent of the builds @p at, over x's arrays: each term's
// product rule, one contraction per operand that varies with that operand
// swapped for its own in @p along (the same builds over the tangent arrays),
// onto along's outputs. The δ operands are x's constants and vary with
// nothing; every other operand, the h views included, does. Builds run in
// order, so a build reading an earlier one's output (d, g) reads its tangent.
Status run_builds_tangent(DeviceResources &res, const ScArrays &x,
                          const std::span<const FockBuild> at,
                          const std::span<const FockBuild> along) {
  const auto constant = [&](const EinsumOperand &op) {
    return op.data == x.delta_core.data() || op.data == x.delta_act.data();
  };
  for (size_t b = 0; b < at.size(); ++b) {
    for (size_t t = 0; t < at[b].terms.size(); ++t) {
      const FockTerm &term = at[b].terms[t];
      for (size_t k = 0; k < term.operands.size(); ++k) {
        if (constant(term.operands[k])) {
          continue;
        }
        std::vector<EinsumOperand> operands = term.operands;
        operands[k] = along[b].terms[t].operands[k];
        FBF_TRY(contract(res, term.subscripts, operands, along[b].out, along[b].dims, term.alpha));
      }
    }
  }
  return {};
}

// b_block (declared in :builders).
EinsumOperand b_block(const double *b, const int64_t naux, const int64_t nmo, const int64_t p0,
                      const int64_t rows, const int64_t q0, const int64_t cols) {
  return {.data = b + (naux * (p0 + (nmo * q0))),
          .dims = {naux, rows, cols},
          .strides = {1, naux, naux * nmo}};
}

// The arrays of @p x a BuilderOutputs holds, moved out of it, with f's blocks.
BuilderOutputs builder_outputs(ScArrays &x, Buffers::device<double> f_core,
                               Buffers::device<double> f_virt) {
  return {.blocks = std::move(x.blocks),
          .eri = std::move(x.eri),
          .h1e = std::move(x.h1e),
          .h2e = std::move(x.h2e),
          .e_core = std::move(x.e_core),
          .e_virt = std::move(x.e_virt),
          .h1e_v_sr = std::move(x.h1e_v_sr),
          .h1e_v_si = std::move(x.h1e_v_si),
          .h1e_v_sir = std::move(x.h1e_v_sir),
          .dm1 = std::move(x.dm1),
          .dm2 = std::move(x.dm2),
          .dm3 = std::move(x.dm3),
          .f3ca = std::move(x.f3ca),
          .f3ac = std::move(x.f3ac),
          .f_core = std::move(f_core),
          .f_virt = std::move(f_virt)};
}

// The direction's tangent of every ScArrays field no RDM reaches, at the
// point @p x: h's and the B blocks' (slices of @p b_dot), B_vv diagonal's,
// (tu|vw)'s by its builder and h2e's, its permute; the RDMs, gamma and what
// reads it zero, the δ operands the identities. With @p vv_dot, B_vv's
// tangent (naux x nvirt^2) is allocated into it, for f's virtual block.
Result<ScArrays> integrals_tangent(DeviceResources &res, ScArrays &x, const double *h_dot,
                                   const double *b_dot, Buffers::device<double> *vv_dot) {
  const int64_t naux = x.blocks.naux;
  const int64_t nc = x.blocks.ncore;
  const int64_t n = x.blocks.ncas;
  const int64_t nv = x.blocks.nvirt;
  const int64_t nocc = nc + n;
  const int64_t nmo = nocc + nv;
  const int64_t nmo2 = nmo * nmo;
  ScArrays dot =
      FBF_TRY(zeroed_arrays(res, x.blocks.naux, x.blocks.ncore, x.blocks.ncas, x.blocks.nvirt));
  dot.electrons = x.electrons;

  // The direction's own: h, and every block of B the builders read.
  if (nmo2 > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dot.h_mo.data(), h_dot,
                                           sizeof(double) * static_cast<size_t>(nmo2),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  const auto slice = [&](const int64_t p0, const int64_t rows, const int64_t q0, const int64_t cols,
                         double *to) {
    return contract(res, "Lpq->Lpq", {b_block(b_dot, naux, nmo, p0, rows, q0, cols)}, to,
                    {naux, rows, cols}, 1.0);
  };
  FBF_TRY(slice(nc, n, nc, n, dot.blocks.aa.data()));
  FBF_TRY(slice(0, nc, nc, n, dot.blocks.ca.data()));
  FBF_TRY(slice(nocc, nv, nc, n, dot.blocks.va.data()));
  FBF_TRY(slice(0, nc, nocc, nv, dot.blocks.cv.data()));
  FBF_TRY(slice(nocc, nv, 0, nc, dot.blocks.vc.data()));
  FBF_TRY(slice(0, nc, 0, nc, dot.cc.data()));
  if (vv_dot != nullptr) {
    *vv_dot = FBF_TRY(zeroed(res, naux * nv * nv));
    FBF_TRY(slice(nocc, nv, nocc, nv, vv_dot->data()));
  }
  FBF_TRY(contract(res, "Laa->La", {b_block(b_dot, naux, nmo, nocc, nv, nocc, nv)}, dot.vvd.data(),
                   {naux, nv}, 1.0));

  // (tu|vw), then h2e[t, u, v, w] = (tv|uw), its permute.
  const std::vector<FockBuild> at = fock_builds(x, nullptr);
  const std::vector<FockBuild> along = fock_builds(dot, nullptr);
  FBF_TRY(run_builds_tangent(res, x, std::span(at).first(1), std::span(along).first(1)));
  FBF_TRY(contract(res, "tvuw->tuvw", {column_operand(dot.eri.data(), {n, n, n, n})},
                   dot.h2e.data(), {n, n, n, n}, 1.0));
  return dot;
}

// The tangent of the pair (bra, ket)'s RDMs along (@p bra_dot, @p ket_dot),
// at x's (tu|vw) and its tangent @p eri_dot, into the zeroed @p dm1 ... f3ac:
// dm3 (and with @p f3ca, the diagonal's f3ca and f3ac) by the bra/ket
// nevpt2::rdms at (ḃ, k) and at (b, k̇), and on the diagonal f3 of the one
// vector at (tu|vw)'s tangent too; then dm2 and dm1, dm3's linear traces.
// The second and third builds' outputs, added onto the first's, are buffers
// of their own, since nevpt2::rdms overwrites its outputs and takes the
// workspace. f3 needs bra == ket.
Status pair_rdms_tangent(DeviceResources &res, const ci::DeviceStringSet &alpha,
                         const ci::DeviceStringSet &beta, const ScArrays &x, const double *eri_dot,
                         const double *bra, const double *ket, const double *bra_dot,
                         const double *ket_dot, double *dm1, double *dm2, double *dm3, double *f3ca,
                         double *f3ac) {
  const int64_t n = x.blocks.ncas;
  const int64_t n6 = n * n * n * n * n * n;
  const bool f3 = f3ca != nullptr;
  const auto more = [&](const bool wanted) {
    return Buffers::device<double>(wanted ? static_cast<size_t>(n6) : 0, res.device());
  };
  Buffers::device<double> dm3_more = more(true);
  Buffers::device<double> f3ca_more = more(f3);
  Buffers::device<double> f3ac_more = more(f3);
  const auto add_n6 = [&](const Buffers::device<double> &from, double *to) {
    return add_onto(res, n6, 1, from.data(), to, n6);
  };
  FBF_TRY(rdms(res, alpha, beta, bra_dot, ket, x.eri.data(), nullptr, nullptr, dm3, f3ca, f3ac,
               FBF_TRY(rdm_budget())));
  FBF_TRY(rdms(res, alpha, beta, bra, ket_dot, x.eri.data(), nullptr, nullptr, dm3_more.data(),
               f3 ? f3ca_more.data() : nullptr, f3 ? f3ac_more.data() : nullptr,
               FBF_TRY(rdm_budget())));
  FBF_TRY(add_n6(dm3_more, dm3));
  if (f3) {
    FBF_TRY(add_n6(f3ca_more, f3ca));
    FBF_TRY(add_n6(f3ac_more, f3ac));
    FBF_TRY(rdms(res, alpha, beta, bra, eri_dot, dm3_more.data(), f3ca_more.data(),
                 f3ac_more.data(), FBF_TRY(rdm_budget())));
    FBF_TRY(add_n6(f3ca_more, f3ca));
    FBF_TRY(add_n6(f3ac_more, f3ac));
  }
  FBF_TRY(run_builds(res, trace_builds(x, dm1, dm2, dm3, nullptr, nullptr, nullptr)));
  // The launches above read the scratch buffers, which go with this frame.
  return res.sync();
}

Result<ScArrays> direction_arrays(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                  const ci::DeviceStringSet &beta, ScArrays &x,
                                  const double *vector, const double *h_dot, const double *b_dot,
                                  const double *c_dot, Buffers::device<double> *vv_dot) {
  const ScopedDeviceIndex on{res.dev_idx()};
  ScArrays dot = FBF_TRY(integrals_tangent(res, x, h_dot, b_dot, vv_dot));
  // The root's RDMs, the diagonal pair's at (ċ, ċ).
  FBF_TRY(pair_rdms_tangent(res, alpha, beta, x, dot.eri.data(), vector, vector, c_dot, c_dot,
                            dot.dm1.data(), dot.dm2.data(), dot.dm3.data(), dot.f3ca.data(),
                            dot.f3ac.data()));
  // The builders that read γ, d and g.
  const std::vector<FockBuild> at = fock_builds(x, nullptr);
  const std::vector<FockBuild> along = fock_builds(dot, nullptr);
  FBF_TRY(run_builds_tangent(res, x, std::span(at).subspan(1), std::span(along).subspan(1)));
  return dot;
}

Result<BuildersTangent> builders_tangent(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                         const ci::DeviceStringSet &beta, const int32_t naux,
                                         const int32_t nao, const int32_t nmo, const double *b,
                                         const double *c, const double *h, const int32_t ncore,
                                         const int32_t ncas, const double *vector,
                                         const int32_t count, const double *h_dot,
                                         const double *b_dot, const double *c_dot) {
  constexpr std::string_view kWho = "nevpt2::builders_tangent";
  if (count < 0 || (count > 0 && (h_dot == nullptr || b_dot == nullptr || c_dot == nullptr))) {
    return err_config("{}: {} directions, each of h_dot, b_dot and c_dot given when any are", kWho,
                      count);
  }
  // All-electron (nfrozen 0): the Hessian this feeds refuses a frozen core.
  ScArrays x =
      FBF_TRY(sc_arrays(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, 0));
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t nocc = nc + ncas;
  const int64_t nv = nmo - nocc;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  const auto nvirt = static_cast<int32_t>(nv);

  // B_vv whole, which f_virt reads off its diagonal: its own buffer, held
  // across the calls below that take the workspace.
  Buffers::device<double> vv = FBF_TRY(zeroed(res, naux * nv * nv));
  if (naux > 0 && nv > 0) {
    const OrbitalRange virt{.first = static_cast<int32_t>(nocc), .count = nvirt};
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, virt, virt, vv.data()));
  }
  Buffers::device<double> f_core = FBF_TRY(zeroed(res, nc * nc));
  Buffers::device<double> f_virt = FBF_TRY(zeroed(res, nv * nv));
  const std::vector<FockBuild> f_at = f_oo_builds(x, vv.data(), f_core.data(), f_virt.data());
  FBF_TRY(run_builds(res, f_at));

  std::vector<BuilderOutputs> tangents;
  tangents.reserve(static_cast<size_t>(count));
  for (int64_t k = 0; k < count; ++k) {
    Buffers::device<double> vv_dot(0, res.device());
    ScArrays dot =
        FBF_TRY(direction_arrays(res, alpha, beta, x, vector, h_dot + (k * nmo2),
                                 b_dot + (k * naux * nmo2), c_dot + (k * ndet), &vv_dot));
    // f's two blocks.
    Buffers::device<double> f_core_dot = FBF_TRY(zeroed(res, nc * nc));
    Buffers::device<double> f_virt_dot = FBF_TRY(zeroed(res, nv * nv));
    FBF_TRY(run_builds_tangent(
        res, x, f_at, f_oo_builds(dot, vv_dot.data(), f_core_dot.data(), f_virt_dot.data())));
    // The launches above read vv_dot, which goes with this iteration.
    FBF_TRY(res.sync());
    tangents.push_back(builder_outputs(dot, std::move(f_core_dot), std::move(f_virt_dot)));
  }
  // The launches above read vv, which goes with this frame.
  FBF_TRY(res.sync());
  return BuildersTangent{.value = builder_outputs(x, std::move(f_core), std::move(f_virt)),
                         .tangents = std::move(tangents)};
}

// The shared outputs of the QD arrays @p x (dm1 gamma_SA), with f's blocks:
// builder_outputs, its one root's RDMs emptied (x's are not read: each
// pair's are its own).
BuilderOutputs qd_shared_outputs(DeviceResources &res, ScArrays &x, Buffers::device<double> f_core,
                                 Buffers::device<double> f_virt) {
  BuilderOutputs out = builder_outputs(x, std::move(f_core), std::move(f_virt));
  for (Buffers::device<double> *field : {&out.dm2, &out.dm3, &out.f3ca, &out.f3ac}) {
    *field = Buffers::device<double>(0, res.device());
  }
  return out;
}

// qd_direction_arrays (declared in :builders).
Result<QdDirectionArrays> qd_direction_arrays(
    DeviceResources &res, const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
    QdArrays &q, const std::span<const double> weights, const double *vectors, const double *h_dot,
    const double *b_dot, const double *c_dot, Buffers::device<double> *vv_dot) {
  ScArrays &x = q.x;
  const auto nroots = static_cast<int64_t>(weights.size());
  const int64_t n = x.blocks.ncas;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  ScArrays dot = FBF_TRY(integrals_tangent(res, x, h_dot, b_dot, vv_dot));
  // Every pair's RDMs, Eq. qd-rdm2's first order; gamma_SA's tangent the
  // weights' sum of the diagonal's dm1.
  std::vector<PairRdms> pairs;
  for (int64_t bra = 0; bra < nroots; ++bra) {
    for (int64_t ket = 0; ket < nroots; ++ket) {
      const bool diagonal = bra == ket;
      PairRdms p = FBF_TRY(zeroed_rdms(res, n, diagonal));
      FBF_TRY(pair_rdms_tangent(
          res, alpha, beta, x, dot.eri.data(), vectors + (bra * ndet), vectors + (ket * ndet),
          c_dot + (bra * ndet), c_dot + (ket * ndet), p.dm1.data(), p.dm2.data(), p.dm3.data(),
          diagonal ? p.f3ca.data() : nullptr, diagonal ? p.f3ac.data() : nullptr));
      if (diagonal) {
        FBF_TRY(contract(res, "tu->tu", {column_operand(p.dm1.data(), {n, n})}, dot.dm1.data(),
                         {n, n}, weights[static_cast<size_t>(bra)]));
      }
      pairs.push_back(std::move(p));
    }
  }
  // The builders that read gamma_SA, d and g.
  const std::vector<FockBuild> at = fock_builds(x, nullptr);
  const std::vector<FockBuild> along = fock_builds(dot, nullptr);
  FBF_TRY(run_builds_tangent(res, x, std::span(at).subspan(1), std::span(along).subspan(1)));
  return QdDirectionArrays{.shared = std::move(dot), .pairs = std::move(pairs)};
}

Result<QdBuildersTangent>
qd_builders_tangent(DeviceResources &res, const ci::DeviceStringSet &alpha,
                    const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
                    const int32_t nmo, const double *b, const double *c, const double *h,
                    const int32_t ncore, const int32_t ncas, const std::span<const double> weights,
                    const double *vectors, const int32_t count, const double *h_dot,
                    const double *b_dot, const double *c_dot) {
  constexpr std::string_view kWho = "nevpt2::qd_builders_tangent";
  if (count < 0 || (count > 0 && (h_dot == nullptr || b_dot == nullptr || c_dot == nullptr))) {
    return err_config("{}: {} directions, each of h_dot, b_dot and c_dot given when any are", kWho,
                      count);
  }
  // All-electron, as builders_tangent.
  QdArrays q = FBF_TRY(
      qd_arrays(res, kWho, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, 0));
  ScArrays &x = q.x;
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto nroots = static_cast<int64_t>(weights.size());
  const int64_t nc = ncore;
  const int64_t nocc = nc + ncas;
  const int64_t nv = nmo - nocc;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  const auto ndet = static_cast<int64_t>(alpha.size()) * static_cast<int64_t>(beta.size());
  const auto nvirt = static_cast<int32_t>(nv);

  // B_vv whole and f's two blocks at gamma_SA, as builders_tangent's.
  Buffers::device<double> vv = FBF_TRY(zeroed(res, naux * nv * nv));
  if (naux > 0 && nv > 0) {
    const OrbitalRange virt{.first = static_cast<int32_t>(nocc), .count = nvirt};
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, virt, virt, vv.data()));
  }
  Buffers::device<double> f_core = FBF_TRY(zeroed(res, nc * nc));
  Buffers::device<double> f_virt = FBF_TRY(zeroed(res, nv * nv));
  const std::vector<FockBuild> f_at = f_oo_builds(x, vv.data(), f_core.data(), f_virt.data());
  FBF_TRY(run_builds(res, f_at));

  std::vector<QdBuilderOutputs> tangents;
  tangents.reserve(static_cast<size_t>(count));
  for (int64_t k = 0; k < count; ++k) {
    Buffers::device<double> vv_dot(0, res.device());
    QdDirectionArrays d = FBF_TRY(qd_direction_arrays(res, alpha, beta, q, weights, vectors,
                                                      h_dot + (k * nmo2), b_dot + (k * naux * nmo2),
                                                      c_dot + (k * nroots * ndet), &vv_dot));
    ScArrays &dot = d.shared;
    // f's two blocks.
    Buffers::device<double> f_core_dot = FBF_TRY(zeroed(res, nc * nc));
    Buffers::device<double> f_virt_dot = FBF_TRY(zeroed(res, nv * nv));
    FBF_TRY(run_builds_tangent(
        res, x, f_at, f_oo_builds(dot, vv_dot.data(), f_core_dot.data(), f_virt_dot.data())));
    // The launches above read vv_dot, which goes with this iteration.
    FBF_TRY(res.sync());
    tangents.push_back(
        {.shared = qd_shared_outputs(res, dot, std::move(f_core_dot), std::move(f_virt_dot)),
         .pairs = std::move(d.pairs)});
  }
  // The launches above read vv, which goes with this frame.
  FBF_TRY(res.sync());
  return QdBuildersTangent{
      .value = {.shared = qd_shared_outputs(res, x, std::move(f_core), std::move(f_virt)),
                .pairs = std::move(q.pairs)},
      .tangents = std::move(tangents)};
}

} // namespace free_breakfast::nevpt2
