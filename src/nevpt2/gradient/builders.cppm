/// @file
/// @brief What the one-root functionals (gradient.cpp) and the SC-QD one
/// (qd.cpp) share: the arrays the builders write and read, the builders as
/// einsum terms, their forward and reverse runs, and the reverse pass from
/// the class walk's bars to h_bar, Lambda, X, A and epsilon around an RDM
/// step each caller supplies.
///
/// An internal partition (`module`, not `export module`): gradient.cpp and
/// qd.cpp import it, nothing reaches an importer of
/// free_breakfast.nevpt2.gradient. It only declares; every definition is
/// gradient.cpp's. See docs/style.md, "Module-private code".

module free_breakfast.nevpt2.gradient:builders;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.ci.strings;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;          // EnergyInputs, EnergyBars
import free_breakfast.nevpt2.gradient;        // ScAdjoint

namespace free_breakfast::nevpt2 {

/// @brief Half the device's free memory: nevpt2::rdms' budget, as the Nevpt2
/// step gives it, so that class_energies' own buffers still find room after
/// the workspace has grown to it (the workspace never shrinks).
Result<size_t> rdm_budget();

/// @brief Every array the builders write or read beyond (h, B, c):
/// class_energies' EnergyInputs and the intermediates, each packed
/// column-major (the RDMs row-major). The adjoint holds a second one for the
/// bars, the same field by field, of which only what takes a bar is
/// allocated. One root's own RDMs in dm1 ... f3ac; in the SC-QD functional,
/// dm1 is gamma_SA, which the shared builders read, and dm2 ... f3ac are not
/// used (each pair's RDMs are qd.cpp's own).
struct ScArrays {
  MoBlocks blocks;
  // C^T h C, nmo x nmo.
  Buffers::device<double> h_mo;
  // B_cc, naux x ncore x ncore, as b_mo leaves it; and B_Q,aa over the
  // virtuals a, naux x nvirt: what the core and virtual orbital energies and
  // the core blocks of F^I read beyond the MoBlocks.
  Buffers::device<double> cc;
  Buffers::device<double> vvd;
  // d_Q = Σ_j B_Q,jj and g_Q = Σ_tu γ_tu B_Q,tu: the Coulomb densities of the
  // core and of the active space, naux each.
  Buffers::device<double> d;
  Buffers::device<double> g;
  // The ncore and ncas identities, the δ operands of the traces.
  Buffers::device<double> delta_core;
  Buffers::device<double> delta_act;
  // (tu|vw) at t + n (u + n (v + n w)), nevpt2::rdms' eri.
  Buffers::device<double> eri;
  Buffers::device<double> h2e;
  Buffers::device<double> h1e;
  Buffers::device<double> h1e_v_sr;
  Buffers::device<double> h1e_v_si;
  Buffers::device<double> h1e_v_sir;
  Buffers::device<double> e_core;
  Buffers::device<double> e_virt;
  Buffers::device<double> dm1;
  Buffers::device<double> dm2;
  Buffers::device<double> dm3;
  Buffers::device<double> f3ca;
  Buffers::device<double> f3ac;
  // The active electrons, N: dm2 = Σ_t dm3[.., t, t] / N.
  double electrons = 0.0;

  [[nodiscard]] EnergyInputs inputs() const {
    return {.h1e = h1e.data(),
            .h2e = h2e.data(),
            .e_core = e_core.data(),
            .e_virt = e_virt.data(),
            .h1e_v_sr = h1e_v_sr.data(),
            .h1e_v_si = h1e_v_si.data(),
            .h1e_v_sir = h1e_v_sir.data(),
            .dm1 = dm1.data(),
            .dm2 = dm2.data(),
            .dm3 = dm3.data(),
            .f3ca = f3ca.data(),
            .f3ac = f3ac.data()};
  }
};

/// @brief One term of a builder, out += alpha einsum(subscripts, operands)
/// with the output column-major, and the bar each operand's adjoint adds
/// onto: packed column-major in that operand's extents, or null (a constant,
/// an h view, or the forward). No operand that takes a bar repeats a label:
/// the traces go through the δ operands, which take none.
struct FockTerm {
  std::string_view subscripts;
  std::vector<EinsumOperand> operands;
  std::vector<double *> bars;
  double alpha = 1.0;
};

/// @brief A builder: its output, the output's bar (null in the forward), the
/// output's extents, and its terms.
struct FockBuild {
  double *out = nullptr;
  const double *out_bar = nullptr;
  std::vector<int64_t> dims;
  std::vector<FockTerm> terms;
};

/// @brief The builders of the shared inputs, in the order the forward runs
/// them: (tu|vw) first, at index 0, which nevpt2::rdms reads; then, once
/// x.dm1 is set, the Coulomb densities d and g, the four F^I blocks and the
/// two orbital energies. With @p bar, each term carries its operands' bars
/// and each build its output's bar; without, the forward's.
std::vector<FockBuild> fock_builds(ScArrays &x, ScArrays *bar);

/// @brief The two traces of one set of RDMs, dm2 of @p dm3 and then @p dm1
/// of @p dm2, over x's identity and electron count: dm2[p,q,r,s] =
/// Σ_t dm3[p,q,r,s,t,t] / N and dm1 the same of dm2. The bars are null in the
/// forward.
std::vector<FockBuild> trace_builds(const ScArrays &x, double *dm1, double *dm2, const double *dm3,
                                    double *dm1_bar, double *dm2_bar, double *dm3_bar);

/// @brief Each build of @p builds forward, in order.
Status run_builds(DeviceResources &res, std::span<const FockBuild> builds);

/// @brief Each build of @p builds backwards, the last first: every operand
/// that takes a bar gets the forward contraction with it and the output
/// swapping roles.
Status run_builds_adjoint(DeviceResources &res, std::span<const FockBuild> builds);

/// @brief Every ScArrays buffer, zeroed, at the sizes of an ncore, ncas,
/// nvirt split over naux fitting functions: what the builders accumulate
/// into, and the bars.
Result<ScArrays> zeroed_arrays(DeviceResources &res, int32_t naux, int32_t ncore, int32_t ncas,
                               int32_t nvirt);

/// @brief The inputs that no RDM reaches, built from (h, B, c): the counts
/// checked (InvalidConfig naming @p who), the MO blocks, B_cc and B_vv's
/// diagonal, C^T h C, h2e and (tu|vw); the RDMs, gamma and what reads it are
/// the caller's.
Result<ScArrays> shared_arrays(DeviceResources &res, std::string_view who,
                               const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                               int32_t naux, int32_t nao, int32_t nmo, const double *b,
                               const double *c, const double *h, int32_t ncore, int32_t ncas);

/// @brief Where a class walk adds its bars: @p bar's fields, and Λ at
/// @p lambda.
EnergyBars energy_bars(ScArrays &bar, double *lambda);

/// @brief The reverse pass after a class walk (SC's, PC's or the SC-QD
/// matrix's) has added its bars onto @p bar and @p lambda_raw: the shared
/// builders backwards onto the MO blocks, B_cc, B_vv's diagonal and bar.dm1;
/// then @p rdm_adjoint, which takes the RDMs' bars (bar.dm1's among them)
/// onto the CI vectors, returned as y, and onto bar.eri; then (tu|vw)'s
/// builder backwards, and h_bar, Λ, X, A and ε. The energies are left to the
/// caller.
Result<ScAdjoint>
builders_adjoint(DeviceResources &res, int32_t naux, int32_t nao, int32_t nmo, const double *b,
                 const double *c, int32_t ncore, int32_t ncas, ScArrays &x, ScArrays &bar,
                 Buffers::device<double> lambda_raw,
                 const std::function<Result<Buffers::device<double>>()> &rdm_adjoint);

} // namespace free_breakfast::nevpt2
