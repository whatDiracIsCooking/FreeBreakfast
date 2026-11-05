/// @file
/// @brief sc_energy, sc_adjoint, pc_energy and pc_adjoint: the per-root input
/// builders, each listed once as einsum terms and run forward for the energy
/// and backwards for the adjoint, around the SC or PC class energies (and
/// their adjoints), rdms(_adjoint) and the density-fitted adjoints; then the
/// symmetrization and the AO half-transform of the orbital source term.

module free_breakfast.nevpt2.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kTwo, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, gemmStridedBatched, geam, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.einsum;
import free_breakfast.recipes.symmetrize;
import free_breakfast.integrals.eri.density_fitting.mo; // b_mo, b_half
import free_breakfast.ci.strings;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.rdm;
import free_breakfast.nevpt2.energy;
import :operands; // column_operand, contract, zeroed, add_onto, check_counts
import :builders; // ScArrays, FockBuild, the builders and their reverse pass

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

Result<size_t> rdm_budget() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  return free_bytes / 2;
}

// The adjoint of one term: each operand that takes a bar gets the forward
// contraction with it and the output swapping roles.
Status contract_adjoint(DeviceResources &res, const FockTerm &term, const double *out_bar,
                        const std::vector<int64_t> &out_dims) {
  const size_t arrow = term.subscripts.find("->");
  const std::string_view rhs = term.subscripts.substr(arrow + 2);
  std::vector<std::string_view> groups;
  for (std::string_view lhs = term.subscripts.substr(0, arrow);;) {
    const size_t comma = lhs.find(',');
    groups.push_back(lhs.substr(0, comma));
    if (comma == std::string_view::npos) {
      break;
    }
    lhs.remove_prefix(comma + 1);
  }
  for (size_t k = 0; k < term.operands.size(); ++k) {
    if (term.bars[k] == nullptr) {
      continue;
    }
    std::string subscripts{rhs};
    std::vector<EinsumOperand> operands{column_operand(out_bar, out_dims)};
    for (size_t j = 0; j < term.operands.size(); ++j) {
      if (j != k) {
        subscripts += ',';
        subscripts += groups[j];
        operands.push_back(term.operands[j]);
      }
    }
    subscripts += "->";
    subscripts += groups[k];
    FBF_TRY(contract(res, subscripts, operands, term.bars[k], term.operands[k].dims, term.alpha));
  }
  return {};
}

// fock_builds' (declared in :builders) labels:
// Labels: L fitting, i j core, t u active, a virtual. Each MO block is
// [L][p][q] ("Lat" is B_va), h a view of C^T h C (strides 1, nmo), and the
// RDMs row-major, labels reversed.
//   F^I_pq = h_pq + Σ_j [2 (pq|jj) − (pj|jq)],
//   ε_p = F^I_pp + Σ_tu γ_tu [(pp|tu) − ½ (pt|up)],
// over (pq|rs) = Σ_L B_L,pq B_L,rs; h1e_v_sr's −Σ_u (au|ut) is Sr's own.
std::vector<FockBuild> fock_builds(ScArrays &x, ScArrays *bar) {
  const int64_t naux = x.blocks.naux;
  const int64_t nc = x.blocks.ncore;
  const int64_t n = x.blocks.ncas;
  const int64_t nv = x.blocks.nvirt;
  const int64_t nmo = nc + n + nv;
  const int64_t nocc = nc + n;
  const auto bar_of = [bar](Buffers::device<double> ScArrays::*field) -> double * {
    return bar == nullptr ? nullptr : (bar->*field).data();
  };
  double *const aa_bar = bar == nullptr ? nullptr : bar->blocks.aa.data();
  double *const ca_bar = bar == nullptr ? nullptr : bar->blocks.ca.data();
  double *const va_bar = bar == nullptr ? nullptr : bar->blocks.va.data();
  double *const vc_bar = bar == nullptr ? nullptr : bar->blocks.vc.data();
  double *const cc_bar = bar_of(&ScArrays::cc);
  double *const vvd_bar = bar_of(&ScArrays::vvd);
  double *const d_bar = bar_of(&ScArrays::d);
  double *const g_bar = bar_of(&ScArrays::g);
  double *const dm1_bar = bar_of(&ScArrays::dm1);

  const EinsumOperand aa = column_operand(x.blocks.aa.data(), {naux, n, n});
  const EinsumOperand ca = column_operand(x.blocks.ca.data(), {naux, nc, n});
  const EinsumOperand va = column_operand(x.blocks.va.data(), {naux, nv, n});
  const EinsumOperand vc = column_operand(x.blocks.vc.data(), {naux, nv, nc});
  const EinsumOperand cc = column_operand(x.cc.data(), {naux, nc, nc});
  const EinsumOperand vvd = column_operand(x.vvd.data(), {naux, nv});
  const EinsumOperand d = column_operand(x.d.data(), {naux});
  const EinsumOperand g = column_operand(x.g.data(), {naux});
  const EinsumOperand dm1 = column_operand(x.dm1.data(), {n, n});
  const EinsumOperand delta_c = column_operand(x.delta_core.data(), {nc, nc});
  // The rows x cols block of C^T h C at (r0, c0).
  const auto h_view = [&](const int64_t r0, const int64_t rows, const int64_t c0,
                          const int64_t cols) {
    return EinsumOperand{
        .data = x.h_mo.data() + r0 + (nmo * c0), .dims = {rows, cols}, .strides = {1, nmo}};
  };
  const auto out_bar = [&](Buffers::device<double> ScArrays::*field) -> const double * {
    return bar_of(field);
  };
  return {
      {.out = x.eri.data(),
       .out_bar = out_bar(&ScArrays::eri),
       .dims = {n, n, n, n},
       .terms = {{"Ltu,Lvw->tuvw", {aa, aa}, {aa_bar, aa_bar}, 1.0}}},
      {.out = x.d.data(),
       .out_bar = d_bar,
       .dims = {naux},
       .terms = {{"Lij,ij->L", {cc, delta_c}, {cc_bar, nullptr}, 1.0}}},
      {.out = x.g.data(),
       .out_bar = g_bar,
       .dims = {naux},
       .terms = {{"ut,Ltu->L", {dm1, aa}, {dm1_bar, aa_bar}, 1.0}}},
      {.out = x.h1e.data(),
       .out_bar = out_bar(&ScArrays::h1e),
       .dims = {n, n},
       .terms = {{"tu->tu", {h_view(nc, n, nc, n)}, {nullptr}, 1.0},
                 {"Ltu,L->tu", {aa, d}, {aa_bar, d_bar}, 2.0},
                 {"Ljt,Lju->tu", {ca, ca}, {ca_bar, ca_bar}, -1.0}}},
      {.out = x.h1e_v_sr.data(),
       .out_bar = out_bar(&ScArrays::h1e_v_sr),
       .dims = {nv, n},
       .terms = {{"at->at", {h_view(nocc, nv, nc, n)}, {nullptr}, 1.0},
                 {"Lat,L->at", {va, d}, {va_bar, d_bar}, 2.0},
                 {"Laj,Ljt->at", {vc, ca}, {vc_bar, ca_bar}, -1.0},
                 {"Lau,Lut->at", {va, aa}, {va_bar, aa_bar}, -1.0}}},
      {.out = x.h1e_v_si.data(),
       .out_bar = out_bar(&ScArrays::h1e_v_si),
       .dims = {n, nc},
       .terms = {{"ti->ti", {h_view(nc, n, 0, nc)}, {nullptr}, 1.0},
                 {"Lit,L->ti", {ca, d}, {ca_bar, d_bar}, 2.0},
                 {"Ljt,Lji->ti", {ca, cc}, {ca_bar, cc_bar}, -1.0}}},
      {.out = x.h1e_v_sir.data(),
       .out_bar = out_bar(&ScArrays::h1e_v_sir),
       .dims = {nv, nc},
       .terms = {{"ai->ai", {h_view(nocc, nv, 0, nc)}, {nullptr}, 1.0},
                 {"Lai,L->ai", {vc, d}, {vc_bar, d_bar}, 2.0},
                 {"Laj,Lji->ai", {vc, cc}, {vc_bar, cc_bar}, -1.0}}},
      // (ij|ji) reads B_cc off its diagonal; (ii|jj) and (ii|tu) on it.
      {.out = x.e_core.data(),
       .out_bar = out_bar(&ScArrays::e_core),
       .dims = {nc},
       .terms = {{"ii->i", {h_view(0, nc, 0, nc)}, {nullptr}, 1.0},
                 {"Lij,L,ij->i", {cc, d, delta_c}, {cc_bar, d_bar, nullptr}, 2.0},
                 {"Lij,Lji->i", {cc, cc}, {cc_bar, cc_bar}, -1.0},
                 {"Lij,L,ij->i", {cc, g, delta_c}, {cc_bar, g_bar, nullptr}, 1.0},
                 {"ut,Lit,Liu->i", {dm1, ca, ca}, {dm1_bar, ca_bar, ca_bar}, -0.5}}},
      {.out = x.e_virt.data(),
       .out_bar = out_bar(&ScArrays::e_virt),
       .dims = {nv},
       .terms = {{"aa->a", {h_view(nocc, nv, nocc, nv)}, {nullptr}, 1.0},
                 {"La,L->a", {vvd, d}, {vvd_bar, d_bar}, 2.0},
                 {"Laj,Laj->a", {vc, vc}, {vc_bar, vc_bar}, -1.0},
                 {"La,L->a", {vvd, g}, {vvd_bar, g_bar}, 1.0},
                 {"ut,Lat,Lau->a", {dm1, va, va}, {dm1_bar, va_bar, va_bar}, -0.5}}},
  };
}

// dm2[p,q,r,s] = Σ_t dm3[p,q,r,s,t,t] / N, and dm1 the same of dm2: E_tt
// summed is the number operator, exact on an N-electron vector, and on the
// ket of a transition.
std::vector<FockBuild> trace_builds(const ScArrays &x, double *dm1, double *dm2, const double *dm3,
                                    double *dm1_bar, double *dm2_bar, double *dm3_bar) {
  const int64_t n = x.blocks.ncas;
  const EinsumOperand delta_a = column_operand(x.delta_act.data(), {n, n});
  const double inv_n = 1.0 / x.electrons;
  return {{.out = dm2,
           .out_bar = dm2_bar,
           .dims = {n, n, n, n},
           .terms = {{"utsrqp,ut->srqp",
                      {column_operand(dm3, {n, n, n, n, n, n}), delta_a},
                      {dm3_bar, nullptr},
                      inv_n}}},
          {.out = dm1,
           .out_bar = dm1_bar,
           .dims = {n, n},
           .terms = {{"srqp,sr->qp",
                      {column_operand(dm2, {n, n, n, n}), delta_a},
                      {dm2_bar, nullptr},
                      inv_n}}}};
}

Status run_builds(DeviceResources &res, const std::span<const FockBuild> builds) {
  for (const FockBuild &build : builds) {
    for (const FockTerm &term : build.terms) {
      FBF_TRY(contract(res, term.subscripts, term.operands, build.out, build.dims, term.alpha));
    }
  }
  return {};
}

Status run_builds_adjoint(DeviceResources &res, const std::span<const FockBuild> builds) {
  for (size_t k = builds.size(); k-- > 0;) {
    for (const FockTerm &term : builds[k].terms) {
      FBF_TRY(contract_adjoint(res, term, builds[k].out_bar, builds[k].dims));
    }
  }
  return {};
}

// The n x n identity, packed.
Result<Buffers::device<double>> identity_matrix(DeviceResources &res, const int64_t n) {
  std::vector<double> host(static_cast<size_t>(n * n), 0.0);
  for (int64_t k = 0; k < n; ++k) {
    host[static_cast<size_t>(k * (n + 1))] = 1.0;
  }
  Buffers::device<double> out(host.size(), res.device());
  if (!host.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    // The upload reads pageable host memory, which goes with this frame.
    FBF_TRY(res.sync());
  }
  return out;
}

// Every ScArrays buffer, zeroed, at the sizes of an ncore, ncas, nvirt
// split over naux fitting functions: what the builders accumulate into, and
// the bars.
Result<ScArrays> zeroed_arrays(DeviceResources &res, const int32_t naux, const int32_t ncore,
                               const int32_t ncas, const int32_t nvirt) {
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nv = nvirt;
  const int64_t nmo = nc + n + nv;
  const int64_t n4 = n * n * n * n;
  const int64_t n6 = n4 * n * n;
  return ScArrays{.blocks = {.naux = naux,
                             .ncore = ncore,
                             .ncas = ncas,
                             .nvirt = nvirt,
                             .aa = FBF_TRY(zeroed(res, naux * n * n)),
                             .ca = FBF_TRY(zeroed(res, naux * nc * n)),
                             .va = FBF_TRY(zeroed(res, naux * nv * n)),
                             .cv = FBF_TRY(zeroed(res, naux * nc * nv)),
                             .vc = FBF_TRY(zeroed(res, naux * nv * nc))},
                  .nfrozen = 0,
                  .correlated = std::nullopt,
                  .h_mo = FBF_TRY(zeroed(res, nmo * nmo)),
                  .cc = FBF_TRY(zeroed(res, naux * nc * nc)),
                  .vvd = FBF_TRY(zeroed(res, naux * nv)),
                  .d = FBF_TRY(zeroed(res, naux)),
                  .g = FBF_TRY(zeroed(res, naux)),
                  .delta_core = FBF_TRY(identity_matrix(res, nc)),
                  .delta_act = FBF_TRY(identity_matrix(res, n)),
                  .eri = FBF_TRY(zeroed(res, n4)),
                  .h2e = FBF_TRY(zeroed(res, n4)),
                  .h1e = FBF_TRY(zeroed(res, n * n)),
                  .h1e_v_sr = FBF_TRY(zeroed(res, nv * n)),
                  .h1e_v_si = FBF_TRY(zeroed(res, n * nc)),
                  .h1e_v_sir = FBF_TRY(zeroed(res, nv * nc)),
                  .e_core = FBF_TRY(zeroed(res, nc)),
                  .e_virt = FBF_TRY(zeroed(res, nv)),
                  .dm1 = FBF_TRY(zeroed(res, n * n)),
                  .dm2 = FBF_TRY(zeroed(res, n4)),
                  .dm3 = FBF_TRY(zeroed(res, n6)),
                  .f3ca = FBF_TRY(zeroed(res, n6)),
                  .f3ac = FBF_TRY(zeroed(res, n6)),
                  .electrons = 0.0};
}

Result<ScArrays> shared_arrays(DeviceResources &res, const std::string_view who,
                               const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                               const int32_t naux, const int32_t nao, const int32_t nmo,
                               const double *b, const double *c, const double *h,
                               const int32_t ncore, const int32_t ncas, const int32_t nfrozen) {
  FBF_TRY(check_counts(who, alpha, beta, naux, nao, nmo, ncore, ncas));
  if (nfrozen < 0 || nfrozen > ncore) {
    return err_config("{}: {} frozen core orbitals of {}: 0 to the core", who, nfrozen, ncore);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nocc = nc + n;
  const int64_t nv = nmo - nocc;

  ScArrays x = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, static_cast<int32_t>(nv)));
  x.blocks = FBF_TRY(mo_blocks(res, naux, nao, nmo, b, c, ncore, ncas));
  // The class walk's: the MO blocks over C's columns from nfrozen on, so
  // that their core is the correlated one, as the Nevpt2 step builds them.
  x.nfrozen = nfrozen;
  if (nfrozen > 0) {
    x.correlated.emplace(FBF_TRY(mo_blocks(res, naux, nao, nmo - nfrozen, b,
                                           c + (int64_t{nao} * nfrozen), ncore - nfrozen, ncas)));
  }
  {
    double *const work = std::get<0>(res.device_regions<double>(static_cast<size_t>(nao) * nmo));
    FBF_TRY(gpu_status(gemmwich(res.blas(), nmo, nao, c, h, work, x.h_mo.data())));
  }
  if (naux > 0 && nc > 0) {
    FBF_TRY(b_mo(res, naux, nao, nmo, b, c, {.first = 0, .count = ncore},
                 {.first = 0, .count = ncore}, x.cc.data()));
  }
  // B_Q,aa = Σ_m X_Q,am C_m,a over b_half's X, the virtual rows' half
  // transform, which is scratch: einsum takes no workspace.
  if (naux > 0 && nv > 0) {
    double *const half =
        std::get<0>(res.device_regions<double>(static_cast<size_t>(naux * nv) * nao));
    FBF_TRY(b_half(res, naux, nao, nmo, b, c,
                   {.first = static_cast<int32_t>(nocc), .count = static_cast<int32_t>(nv)}, half));
    FBF_TRY(contract(res, "Lam,ma->La",
                     {column_operand(half, {naux, nv, nao}),
                      column_operand(c + (int64_t{nao} * nocc), {nao, nv})},
                     x.vvd.data(), {naux, nv}, 1.0));
  }
  x.electrons = static_cast<double>(alpha.n_electrons() + beta.n_electrons());
  if (x.electrons == 0.0) {
    return err_config("{}: no active electrons", who);
  }

  FBF_TRY(h2e(res, x.blocks, x.h2e.data()));
  const std::vector<FockBuild> builds = fock_builds(x, nullptr);
  FBF_TRY(run_builds(res, std::span(builds).first(1)));
  return x;
}

// sc_arrays (declared in :builders): @p who names the caller in its errors.
Result<ScArrays> sc_arrays(DeviceResources &res, const std::string_view who,
                           const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                           const int32_t naux, const int32_t nao, const int32_t nmo,
                           const double *b, const double *c, const double *h, const int32_t ncore,
                           const int32_t ncas, const double *vector, const int32_t nfrozen) {
  ScArrays x =
      FBF_TRY(shared_arrays(res, who, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, nfrozen));
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(rdms(res, alpha, beta, vector, x.eri.data(), x.dm3.data(), x.f3ca.data(), x.f3ac.data(),
               FBF_TRY(rdm_budget())));
  FBF_TRY(run_builds(
      res, trace_builds(x, x.dm1.data(), x.dm2.data(), x.dm3.data(), nullptr, nullptr, nullptr)));
  const std::vector<FockBuild> builds = fock_builds(x, nullptr);
  FBF_TRY(run_builds(res, std::span(builds).subspan(1)));
  return x;
}

Result<ClassEnergies> sc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                const ci::DeviceStringSet &beta, const int32_t naux,
                                const int32_t nao, const int32_t nmo, const double *b,
                                const double *c, const double *h, const int32_t ncore,
                                const int32_t ncas, const double *vector, const int32_t batch,
                                const int32_t nfrozen) {
  const ScArrays x = FBF_TRY(sc_arrays(res, "nevpt2::sc_energy", alpha, beta, naux, nao, nmo, b, c,
                                       h, ncore, ncas, vector, nfrozen));
  return class_energies(res, x.walk_blocks(), x.inputs(), batch);
}

Result<PcClassEnergies> pc_energy(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                  const ci::DeviceStringSet &beta, const int32_t naux,
                                  const int32_t nao, const int32_t nmo, const double *b,
                                  const double *c, const double *h, const int32_t ncore,
                                  const int32_t ncas, const double *vector, const int32_t batch,
                                  const int32_t nfrozen) {
  const ScArrays x = FBF_TRY(sc_arrays(res, "nevpt2::pc_energy", alpha, beta, naux, nao, nmo, b, c,
                                       h, ncore, ncas, vector, nfrozen));
  return pc_class_energies(res, x.walk_blocks(), x.inputs(), batch);
}

Result<LambdaBars> lambda_bars(DeviceResources &res, const ScArrays &x, const int32_t nmo) {
  const int64_t naux = x.blocks.naux;
  const int64_t walk = int64_t{nmo} - x.nfrozen;
  return LambdaBars{.whole = FBF_TRY(zeroed(res, naux * nmo * nmo)),
                    .correlated = FBF_TRY(zeroed(res, x.nfrozen > 0 ? naux * walk * walk : 0)),
                    .nfrozen = x.nfrozen};
}

EnergyBars energy_bars(ScArrays &bar, LambdaBars &lambda) {
  const int64_t nf = lambda.nfrozen;
  return {.h1e = bar.h1e.data(),
          .h2e = bar.h2e.data(),
          .dm1 = bar.dm1.data(),
          .dm2 = bar.dm2.data(),
          .dm3 = bar.dm3.data(),
          .f3ca = bar.f3ca.data(),
          .f3ac = bar.f3ac.data(),
          .h1e_v_sr = bar.h1e_v_sr.data(),
          .h1e_v_si = bar.h1e_v_si.data() + (bar.blocks.ncas * nf),
          .h1e_v_sir = bar.h1e_v_sir.data() + (bar.blocks.nvirt * nf),
          .e_core = bar.e_core.data() + nf,
          .e_virt = bar.e_virt.data(),
          .lambda = lambda.walk()};
}

Result<ScAdjoint>
builders_adjoint(DeviceResources &res, const int32_t naux, const int32_t nao, const int32_t nmo,
                 const double *b, const double *c, const int32_t ncore, const int32_t ncas,
                 ScArrays &x, ScArrays &bar, LambdaBars lambdas,
                 const std::function<Result<Buffers::device<double>>()> &rdm_adjoint) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t nc = ncore;
  const int64_t n = ncas;
  const int64_t nocc = nc + n;
  const int64_t nv = nmo - nocc;
  const int64_t nmo2 = int64_t{nmo} * nmo;
  const int64_t ld_lambda = naux * int64_t{nmo};
  Buffers::device<double> lambda_raw = std::move(lambdas.whole);

  // The walk's Λ over the orbitals from nf on, (L, p, q) at
  // L + naux (p + (nmo - nf) q), onto the whole frame's at (L, p + nf, q + nf):
  // one column of (L, p) rows per q.
  if (const int64_t nf = lambdas.nfrozen; nf > 0) {
    const int64_t walk = nmo - nf;
    FBF_TRY(add_onto(res, naux * walk, walk, lambdas.correlated.data(),
                     lambda_raw.data() + (naux * nf) + (ld_lambda * nf), ld_lambda));
    lambdas.correlated = Buffers::device<double>(0, res.device());
  }

  // The builders backwards, the last first: the orbital energies and F^I's
  // blocks, g and d; the RDMs' step; (tu|vw).
  const std::vector<FockBuild> builds = fock_builds(x, &bar);
  FBF_TRY(run_builds_adjoint(res, std::span(builds).subspan(1)));
  Buffers::device<double> y = FBF_TRY(rdm_adjoint());
  FBF_TRY(run_builds_adjoint(res, std::span(builds).first(1)));

  // Onto Λ: h2e's bar, the MO blocks' and B_vv's diagonal's, and B_cc's at
  // its corner, every element of it (off the diagonal for (ij|ji)).
  FBF_TRY(h2e_adjoint(res, x.blocks, bar.h2e.data(), lambda_raw.data()));
  FBF_TRY(mo_blocks_adjoint(res, bar.blocks, nullptr, bar.vvd.data(), lambda_raw.data()));
  FBF_TRY(add_onto(res, naux * nc, nc, bar.cc.data(), lambda_raw.data(), ld_lambda));

  // h_bar: each F^I block's bar at its corner of C^T h C, and the orbital
  // energies' on its diagonal.
  Buffers::device<double> h_bar = FBF_TRY(zeroed(res, nmo2));
  double *const hb = h_bar.data();
  FBF_TRY(add_onto(res, n, n, bar.h1e.data(), hb + nc + (nmo * nc), nmo));
  FBF_TRY(add_onto(res, nv, n, bar.h1e_v_sr.data(), hb + nocc + (nmo * nc), nmo));
  FBF_TRY(add_onto(res, n, nc, bar.h1e_v_si.data(), hb + nc, nmo));
  FBF_TRY(add_onto(res, nv, nc, bar.h1e_v_sir.data(), hb + nocc, nmo));
  FBF_TRY(add_onto(res, 1, nc, bar.e_core.data(), hb, nmo + 1));
  FBF_TRY(add_onto(res, 1, nv, bar.e_virt.data(), hb + (nocc * (nmo + 1)), nmo + 1));
  FBF_TRY(symmetrize(res, nmo, hb, nmo));

  // Λ = (Λ_raw + Λ_rawᵀ) / 2 per L.
  Buffers::device<double> lambda = FBF_TRY(zeroed(res, naux * nmo2));
  const EinsumOperand raw = column_operand(lambda_raw.data(), {naux, nmo, nmo});
  FBF_TRY(contract(res, "Lpq->Lpq", {raw}, lambda.data(), {naux, nmo, nmo}, 0.5));
  FBF_TRY(contract(res, "Lqp->Lpq", {raw}, lambda.data(), {naux, nmo, nmo}, 0.5));
  lambda_raw = Buffers::device<double>(0, res.device());

  // X = 2 h h_bar + 2 Σ_L C^T B_AO^L (C Λ^L): per q, Λ̃_q = Λ_q C^T
  // (naux x nao, one strided-batched gemm over q); then D = B_AOᵀ Λ̃ with B
  // read as (naux nao) x nao, which its symmetry in its AO pair allows, the
  // DF exchange's shape; then X = 2 C^T D.
  Buffers::device<double> x_source = FBF_TRY(zeroed(res, nmo2));
  if (naux > 0) {
    Buffers::device<double> half(static_cast<size_t>(naux) * nao * nmo, res.device());
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, naux, nao, nmo, &kOne<double>,
        lambda.data(), naux, ld_lambda, c, nao, 0, &kZero<double>, half.data(), naux,
        int64_t{naux} * nao, nmo)));
    double *const dmat = std::get<0>(res.device_regions<double>(static_cast<size_t>(nao) * nmo));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nao, nmo, naux * nao, &kOne<double>, b,
                                               half.data(), &kZero<double>, dmat)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, nmo, nao, &kTwo<double>, c, dmat,
                                               &kOne<double>, x_source.data())));
  }
  FBF_TRY(gpu_status(
      gemm(res.blas(), nmo, &kTwo<double>, x.h_mo.data(), hb, &kOne<double>, x_source.data())));
  Buffers::device<double> a_source(static_cast<size_t>(nmo2), res.device());
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo, nmo, &kOne<double>,
      x_source.data(), nmo, &kNegOne<double>, x_source.data(), nmo, a_source.data(), nmo)));
  // ε as class_energies read it, at the orbitals' own indices: what
  // zc_multipliers divides by.
  Buffers::device<double> epsilon = FBF_TRY(zeroed(res, nmo));
  FBF_TRY(add_onto(res, 1, nc, x.e_core.data(), epsilon.data(), 1));
  FBF_TRY(add_onto(res, 1, nv, x.e_virt.data(), epsilon.data() + nocc, 1));
  return ScAdjoint{.energies = {},
                   .h_bar = std::move(h_bar),
                   .lambda = std::move(lambda),
                   .y = std::move(y),
                   .x = std::move(x_source),
                   .a = std::move(a_source),
                   .epsilon = std::move(epsilon)};
}

// The one-root RDM step of the reverse pass: γ̄ and dm2̄ folded into dm3̄
// by the traces' adjoint, then nevpt2::rdms_adjoint onto y and (tu|vw)'s bar.
Result<Buffers::device<double>> root_rdms_adjoint(DeviceResources &res,
                                                  const ci::DeviceStringSet &alpha,
                                                  const ci::DeviceStringSet &beta,
                                                  const double *vector, ScArrays &x,
                                                  ScArrays &bar) {
  FBF_TRY(run_builds_adjoint(res, trace_builds(x, x.dm1.data(), x.dm2.data(), x.dm3.data(),
                                               bar.dm1.data(), bar.dm2.data(), bar.dm3.data())));
  Buffers::device<double> y(static_cast<size_t>(alpha.size()) * beta.size(), res.device());
  FBF_TRY(rdms_adjoint(res, alpha, beta, vector, x.eri.data(), bar.dm3.data(), bar.f3ca.data(),
                       bar.f3ac.data(), y.data(), bar.eri.data(), FBF_TRY(rdm_budget())));
  return y;
}

Result<ScAdjoint> sc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
                             const int32_t nmo, const double *b, const double *c, const double *h,
                             const int32_t ncore, const int32_t ncas, const double *vector,
                             const int32_t batch, const int32_t nfrozen) {
  ScArrays x = FBF_TRY(sc_arrays(res, "nevpt2::sc_energy", alpha, beta, naux, nao, nmo, b, c, h,
                                 ncore, ncas, vector, nfrozen));
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t nvirt = nmo - ncore - ncas;
  // The bars, zeroed: every builder adds onto them. Λ as the maps are
  // written, onto the corners they read; symmetrized last.
  ScArrays bar = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
  LambdaBars lambda = FBF_TRY(lambda_bars(res, x, nmo));
  const ClassEnergiesAdjoint walked = FBF_TRY(
      class_energies_adjoint(res, x.walk_blocks(), x.inputs(), batch, energy_bars(bar, lambda)));
  ScAdjoint out = FBF_TRY(
      builders_adjoint(res, naux, nao, nmo, b, c, ncore, ncas, x, bar, std::move(lambda),
                       [&] { return root_rdms_adjoint(res, alpha, beta, vector, x, bar); }));
  out.energies = walked.energies;
  return out;
}

Result<PcAdjoint> pc_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
                             const int32_t nmo, const double *b, const double *c, const double *h,
                             const int32_t ncore, const int32_t ncas, const double *vector,
                             const int32_t batch, const int32_t nfrozen) {
  ScArrays x = FBF_TRY(sc_arrays(res, "nevpt2::pc_energy", alpha, beta, naux, nao, nmo, b, c, h,
                                 ncore, ncas, vector, nfrozen));
  const ScopedDeviceIndex on{res.dev_idx()};
  const int32_t nvirt = nmo - ncore - ncas;
  ScArrays bar = FBF_TRY(zeroed_arrays(res, naux, ncore, ncas, nvirt));
  LambdaBars lambda = FBF_TRY(lambda_bars(res, x, nmo));
  // A refused class is the energy's error, returned before any bar is read.
  const PcClassEnergiesAdjoint walked = FBF_TRY(
      pc_class_energies_adjoint(res, x.walk_blocks(), x.inputs(), batch, energy_bars(bar, lambda)));
  ScAdjoint out = FBF_TRY(
      builders_adjoint(res, naux, nao, nmo, b, c, ncore, ncas, x, bar, std::move(lambda),
                       [&] { return root_rdms_adjoint(res, alpha, beta, vector, x, bar); }));
  return PcAdjoint{.energies = walked.energies,
                   .h_bar = std::move(out.h_bar),
                   .lambda = std::move(out.lambda),
                   .y = std::move(out.y),
                   .x = std::move(out.x),
                   .a = std::move(out.a),
                   .epsilon = std::move(out.epsilon)};
}

} // namespace free_breakfast::nevpt2
