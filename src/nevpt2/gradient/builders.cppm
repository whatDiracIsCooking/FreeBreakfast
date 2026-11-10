/// @file
/// @brief What the one-root functionals (gradient.cpp) and the QD ones
/// (qd.cpp) share: the arrays the builders write and read, the builders as
/// einsum terms, their forward and reverse runs, and the reverse pass from
/// the class walk's bars to h_bar, Lambda, X, A and epsilon around an RDM
/// step each caller supplies.
///
/// An internal partition (`module`, not `export module`): gradient.cpp and
/// qd.cpp import it, nothing reaches an importer of
/// free_breakfast.nevpt2.gradient. It only declares; every definition is
/// gradient.cpp's but with_qd_inputs, qd_arrays, zeroed_rdms and
/// qd_adjoints, qd.cpp's, and tangent.cpp's named so. See docs/style.md,
/// "Module-private code".

module free_breakfast.nevpt2.gradient:builders;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.recipes.einsum;
import free_breakfast.ci.strings;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;          // EnergyInputs, EnergyBars
import free_breakfast.nevpt2.gradient;        // ScAdjoint, QdScAdjoint
import :operands;                             // Held

namespace free_breakfast::nevpt2 {

/// @brief Every array the builders write or read beyond (h, B, c):
/// class_energies' EnergyInputs and the intermediates, each packed
/// column-major (the RDMs row-major). The adjoint holds a second one for the
/// bars, the same field by field, of which only what takes a bar is
/// allocated. One root's own RDMs in dm1 ... f3ac; in the QD functionals,
/// dm1 is gamma_SA, which the shared builders read, and dm2 ... f3ac are not
/// used (each pair's RDMs are qd.cpp's own).
struct ScArrays {
  // Over the whole core: what F^I, the orbital energies and the builders
  // read.
  MoBlocks blocks;
  // The core orbitals NEVPT2 leaves uncorrelated, the lowest nfrozen; and,
  // when there are any, the MO blocks over the orbitals from nfrozen on, the
  // correlated core's, which the class walk reads (walk_blocks). None at 0.
  int32_t nfrozen = 0;
  std::optional<MoBlocks> correlated;
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

  // The blocks the class walk reads: the correlated core's, or the whole
  // core's without a frozen one.
  [[nodiscard]] const MoBlocks &walk_blocks() const { return correlated ? *correlated : blocks; }

  // The class walk's inputs: the core-indexed ones (e_core, h1e_v_si,
  // h1e_v_sir, each with the core as its slow index) past the frozen core,
  // so over the correlated one; F^I itself is the whole core's.
  [[nodiscard]] EnergyInputs inputs() const {
    const int64_t nf = nfrozen;
    return {.h1e = h1e.data(),
            .h2e = h2e.data(),
            .e_core = e_core.data() + nf,
            .e_virt = e_virt.data(),
            .h1e_v_sr = h1e_v_sr.data(),
            .h1e_v_si = h1e_v_si.data() + (blocks.ncas * nf),
            .h1e_v_sir = h1e_v_sir.data() + (blocks.nvirt * nf),
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

/// @brief The adjoint of one einsum term in one operand: @p target (packed
/// column-major in operand @p k's extents) += alpha times the contraction of
/// @p out_bar (the output's bar, over @p out_dims) with every operand but
/// the k-th, onto the k-th's labels. contract_adjoint's per operand; the
/// Hessian product's one-slot terms call it with another operand at its
/// tangent.
Status adjoint_slot(DeviceResources &res, std::string_view subscripts,
                    const std::vector<EinsumOperand> &operands, size_t k, const double *out_bar,
                    const std::vector<int64_t> &out_dims, double *target, double alpha);

/// @brief Each build of @p builds forward, in order.
Status run_builds(DeviceResources &res, std::span<const FockBuild> builds);

/// @brief Each build of @p builds backwards, the last first: every operand
/// that takes a bar gets the forward contraction with it and the output
/// swapping roles.
Status run_builds_adjoint(DeviceResources &res, std::span<const FockBuild> builds);

/// @brief Λ's bars: @ref whole over every orbital, which the builders'
/// adjoints and X read; and with a frozen core @ref correlated, over the
/// orbitals from x.nfrozen on (naux x (nmo - nfrozen)^2, the class walk's
/// MO frame), which builders_adjoint adds onto whole's corner. Empty without
/// one, when the walk adds onto whole itself.
struct LambdaBars {
  Buffers::device<double> whole;
  Buffers::device<double> correlated;
  int32_t nfrozen = 0;

  // Where the class walk adds its Λ.
  [[nodiscard]] double *walk() { return nfrozen > 0 ? correlated.data() : whole.data(); }
};

/// @brief LambdaBars for @p x, zeroed, over @p nmo orbitals.
Result<LambdaBars> lambda_bars(DeviceResources &res, const ScArrays &x, int32_t nmo);

/// @brief Every ScArrays buffer, zeroed, at the sizes of an ncore, ncas,
/// nvirt split over naux fitting functions: what the builders accumulate
/// into, and the bars.
Result<ScArrays> zeroed_arrays(DeviceResources &res, int32_t naux, int32_t ncore, int32_t ncas,
                               int32_t nvirt);

/// @brief The bytes zeroed_arrays allocates at those sizes, every buffer held
/// until the ScArrays goes: the footprints' count of a forward's arrays or
/// their bars; and of shared_arrays (@p nfrozen past it, the correlated
/// core's MO blocks), with what its builds borrow, the MO blocks' replacing
/// the zeroed ones, h2e's product and the workspace, beside it.
size_t zeroed_arrays_bytes(int32_t naux, int32_t ncore, int32_t ncas, int32_t nvirt);
MemoryFootprint shared_arrays_footprint(int32_t naux, int32_t nao, int32_t nmo, int32_t ncore,
                                        int32_t ncas, int32_t nfrozen);

/// @brief The inputs that no RDM reaches, built from (h, B, c): the counts
/// checked (InvalidConfig naming @p who, @p nfrozen past the core among
/// them), the MO blocks (and the correlated core's, with @p nfrozen), B_cc
/// and B_vv's diagonal, C^T h C, h2e and (tu|vw); the RDMs, gamma and what
/// reads it are the caller's.
Result<ScArrays> shared_arrays(DeviceResources &res, std::string_view who,
                               const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                               int32_t naux, int32_t nao, int32_t nmo, const double *b,
                               const double *c, const double *h, int32_t ncore, int32_t ncas,
                               int32_t nfrozen);

/// @brief Every input of class_energies (and pc_class_energies), built from
/// (h, B, c) and the root @p vector: shared_arrays, then nevpt2::rdms, dm2
/// and dm1 as its traces, and the builders that read dm1. gradient.cpp's,
/// for tangent.cpp's forward point.
Result<ScArrays> sc_arrays(DeviceResources &res, std::string_view who,
                           const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                           int32_t naux, int32_t nao, int32_t nmo, const double *b, const double *c,
                           const double *h, int32_t ncore, int32_t ncas, const double *vector,
                           int32_t nfrozen);

/// @brief The QD functionals' inputs, built as qd_sc_energy builds them from
/// (h, B, c), the roots @p vectors and their @p weights, handed to @p use:
/// the class walk's MO blocks (the correlated core's, with @p nfrozen) and
/// the n x n pairs [M n + N]. qd.cpp's, for qd_inputs in lagrangian.cpp.
Status
with_qd_inputs(DeviceResources &res, std::string_view who, const ci::DeviceStringSet &alpha,
               const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
               const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
               std::span<const double> weights, const double *vectors, int32_t nfrozen,
               const std::function<Status(const MoBlocks &, std::span<const EnergyInputs>)> &use);

/// @brief The QD functionals' arrays: the shared ones, their dm1 gamma_SA
/// and what reads it built, and the n^2 pairs' RDMs, [M n + N] that of bra M
/// and ket N. qd.cpp's, for tangent.cpp's QD point.
struct QdArrays {
  ScArrays x;
  std::vector<PairRdms> pairs;
};

/// @brief qd_sc_energy's arrays from (h, B, c), the roots @p vectors and
/// their @p weights: shared_arrays; every ordered pair's dm3 by
/// nevpt2::rdms (one vector on the diagonal, with its f3s; bra/ket off it)
/// and its dm2 and dm1 as traces; gamma_SA into x.dm1, and the builders that
/// read it. qd.cpp's.
Result<QdArrays> qd_arrays(DeviceResources &res, std::string_view who,
                           const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta,
                           int32_t naux, int32_t nao, int32_t nmo, const double *b, const double *c,
                           const double *h, int32_t ncore, int32_t ncas,
                           std::span<const double> weights, const double *vectors, int32_t nfrozen);

/// @brief What qd_arrays holds over @p nroots roots: shared_arrays', every
/// ordered pair's RDMs (f3 on the diagonal), and nevpt2::rdms at one
/// determinant, a root's own or a pair's bra/ket into the transition f3s'
/// scratch. Persistent: the QdArrays.
MemoryFootprint qd_arrays_footprint(int32_t naux, int32_t nao, int32_t nmo, int32_t ncore,
                                    int32_t ncas, int32_t nroots, int32_t nfrozen);

/// @brief What qd_adjoint holds at one W, @p walk the effective
/// Hamiltonian's reverse pass's (its transition
/// qd_adjoint_transition_footprint's at the same @p fixed). With @p fixed
/// (qd_adjoints' QdFixedBars, all-electron), the arrays, the shared bars
/// and each root's own stay, persistent beside the result and the walk's
/// kept copies.
MemoryFootprint qd_adjoint_footprint(int32_t naux, int32_t nao, int32_t nmo, int32_t ncore,
                                     int32_t ncas, int32_t n_alpha, int32_t n_beta, int32_t nroots,
                                     int32_t nfrozen, const MemoryFootprint &walk,
                                     bool fixed = false);

/// @brief One call of qd_adjoint's transition callback: the traces' adjoint
/// and the bra/ket rdms_adjoint without f3 bars at one determinant; with
/// @p fixed, dm3's folded bar copied into QdFixedBars::transitions, kept
/// (persistent). Without, qd_hessian_product's transition callback too (its
/// bra/ket and (tu|vw) outputs are the caller's, held around the walk).
MemoryFootprint qd_adjoint_transition_footprint(int32_t ncas, bool fixed);

/// @brief What builders_adjoint holds once the RDMs' adjoint has run, onto
/// @p h: the folded bars, Λ's raw bar given back, X, the half transform, A
/// and ε, and the workspace of h2e's, the MO blocks' and D's (gradient.cpp's).
void builders_adjoint_tail(Held &h, int32_t naux, int32_t nao, int32_t nmo, int32_t ncore,
                           int32_t ncas);

/// @brief A zeroed PairRdms over @p ncas active orbitals, with f3ca and f3ac
/// if @p f3. qd.cpp's.
Result<PairRdms> zeroed_rdms(DeviceResources &res, int64_t ncas, bool f3);

/// @brief sc_effective_hamiltonian's n^2 pairs over @p q: the shared inputs,
/// each pair's RDMs, and f3ca and f3ac on the diagonal alone; PC's
/// conditioning floor @p pc_floor on every pair. qd.cpp's.
std::vector<EnergyInputs> pair_inputs(const QdArrays &q, size_t nroots, double pc_floor = 0.0);

/// @brief What qd_sc_adjoint's reverse pass leaves at the point, for a QD
/// Hessian-vector product's one-slot-dotted terms (hessian_product.cpp):
/// its forward arrays, the shared bars once the builders have run backwards,
/// each root's own RDMs' bars and each transition's dm3 bar ([M n + N],
/// empty on the diagonal), both after the traces' adjoint has folded dm1's
/// and dm2's into dm3's. qd.cpp's.
struct QdFixedBars {
  QdArrays arrays;
  ScArrays bar;
  std::vector<PairRdms> own;
  std::vector<Buffers::device<double>> transitions;
};

/// @brief qd_sc_adjoint (@p pc null) or qd_pc_adjoint at each of the k
/// matrices @p ws holds (k n^2 doubles, W_k at [k n^2, (k + 1) n^2)), over
/// one set of forward arrays and RDMs: one QdScAdjoint per W, in order. PC's
/// elements, the same at every W, go to *@p pc and each result's elements
/// is empty; @p pc_floor is set on every pair's inputs (PC reads it, SC
/// does not). With @p builder_bars (one W, all-electron), the walk's bars
/// kept in it, qd_sc_adjoint's; with @p fixed (the same), the pass's arrays
/// and folded bars moved into it. qd.cpp's, for the stacked QD Lagrangian
/// in lagrangian.cpp and qd_sc_hessian_product.
Result<std::vector<QdScAdjoint>>
qd_adjoints(DeviceResources &res, std::string_view who, const ci::DeviceStringSet &alpha,
            const ci::DeviceStringSet &beta, int32_t naux, int32_t nao, int32_t nmo,
            const double *b, const double *c, const double *h, int32_t ncore, int32_t ncas,
            std::span<const double> weights, const double *vectors, std::span<const double> ws,
            int32_t batch, std::vector<PcClassEnergies> *pc, int32_t nfrozen, double pc_floor,
            std::optional<QdBuilderBars> *builder_bars = nullptr,
            std::optional<QdFixedBars> *fixed = nullptr);

/// @brief fold_bars' result: h_bar and Λ, symmetrized.
struct FoldedBars {
  Buffers::device<double> h_bar;
  Buffers::device<double> lambda;
};

/// @brief The last steps of the reverse pass, once the builders have run
/// backwards onto @p bar: h2e's bar onto @p lambda_raw (at @p blocks' B),
/// the MO blocks', B_vv diagonal's and B_cc's bars onto it, the F^I blocks'
/// and orbital energies' bars placed into h_bar; then h_bar and Λ
/// symmetrized. @p lambda_raw (naux x nmo^2, every orbital's frame) is added
/// onto and read, and is the caller's to free.
Result<FoldedBars> fold_bars(DeviceResources &res, int32_t nmo, const MoBlocks &blocks,
                             ScArrays &bar, Buffers::device<double> &lambda_raw);

/// @brief The one-root RDM step of the reverse pass: @p bar's dm1 and dm2
/// bars folded into dm3's by the traces' adjoint, then nevpt2::rdms_adjoint
/// at @p vector onto the returned y and (tu|vw)'s bar, bar.eri.
/// gradient.cpp's.
Result<Buffers::device<double>> root_rdms_adjoint(DeviceResources &res,
                                                  const ci::DeviceStringSet &alpha,
                                                  const ci::DeviceStringSet &beta,
                                                  const double *vector, ScArrays &x, ScArrays &bar);

/// @brief The rows [p0, p0 + rows) x columns [q0, q0 + cols) block of an MO
/// B (naux x nmo^2, (L, p, q) at L + naux (p + nmo q)), as a strided
/// operand. tangent.cpp's.
EinsumOperand b_block(const double *b, int64_t naux, int64_t nmo, int64_t p0, int64_t rows,
                      int64_t q0, int64_t cols);

/// @brief The builders' forward tangent along one direction (@p h_dot,
/// @p b_dot, @p c_dot), MO basis, laid out as builders_tangent reads one,
/// at the point @p x (sc_arrays', all-electron) and its root @p vector:
/// every ScArrays field's tangent, the δ operands the identities. With
/// @p vv_dot, B_vv's tangent (naux x nvirt^2) is allocated into it, for f's
/// virtual block. tangent.cpp's.
Result<ScArrays> direction_arrays(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                  const ci::DeviceStringSet &beta, ScArrays &x,
                                  const double *vector, const double *h_dot, const double *b_dot,
                                  const double *c_dot, Buffers::device<double> *vv_dot);

/// @brief The QD builders' forward tangent along one direction: every shared
/// ScArrays field's (dm1 gamma_SA's, the weights' sum of the diagonal pairs'
/// dm1 tangents) and every ordered pair's RDMs', [M n + N] (f3ca and f3ac on
/// the diagonal alone).
struct QdDirectionArrays {
  ScArrays shared;
  std::vector<PairRdms> pairs;

  // The n^2 EnergyDirections sc_effective_hamiltonian_tangent and _hvp read,
  // [M n + N], each over @ref shared's blocks.
  [[nodiscard]] std::vector<EnergyDirection> directions() const {
    std::vector<EnergyDirection> out;
    for (const PairRdms &p : pairs) {
      EnergyInputs in = shared.inputs();
      in.dm1 = p.dm1.data();
      in.dm2 = p.dm2.data();
      in.dm3 = p.dm3.data();
      in.f3ca = p.f3ca.num_elements() > 0 ? p.f3ca.data() : nullptr;
      in.f3ac = p.f3ac.num_elements() > 0 ? p.f3ac.data() : nullptr;
      out.push_back({.in = in, .blocks = &shared.blocks});
    }
    return out;
  }
};

/// @brief qd_builders_tangent's along one direction (@p h_dot, @p b_dot, and
/// every root's @p c_dot, root r's at + r ndet), at the QD point @p q of the
/// roots @p vectors at @p weights; with @p vv_dot, B_vv's tangent allocated
/// into it, as direction_arrays'. tangent.cpp's.
Result<QdDirectionArrays>
qd_direction_arrays(DeviceResources &res, const ci::DeviceStringSet &alpha,
                    const ci::DeviceStringSet &beta, QdArrays &q, std::span<const double> weights,
                    const double *vectors, const double *h_dot, const double *b_dot,
                    const double *c_dot, Buffers::device<double> *vv_dot);

/// @brief What direction_arrays holds over an ncore, ncas, nvirt split over
/// @p naux fitting functions, with @p vv_dot B_vv's tangent: the tangent
/// arrays (zeroed_arrays') and B_vv's, kept (persistent); the RDMs'
/// tangent's three scratch n⁶ beside each nevpt2::rdms at one determinant
/// (two bra/ket, one at (tu|vw)'s tangent). tangent.cpp's, and the Hessian
/// products'.
MemoryFootprint direction_arrays_footprint(int32_t naux, int32_t ncore, int32_t ncas, int32_t nvirt,
                                           bool vv_dot);

/// @brief What qd_direction_arrays holds over @p nroots roots: the shared
/// tangent arrays and B_vv's (with @p vv_dot), and every ordered pair's RDM
/// tangent, kept; each pair's scratch beside its nevpt2::rdms at one
/// determinant (the diagonal's as direction_arrays', the transitions' two
/// bra/ket without f3). tangent.cpp's, and the QD Hessian products'.
MemoryFootprint qd_direction_arrays_footprint(int32_t naux, int32_t ncore, int32_t ncas,
                                              int32_t nvirt, int32_t nroots, bool vv_dot);

/// @brief Where a class walk adds its bars: @p bar's fields, the
/// core-indexed ones past the frozen core as ScArrays::inputs reads them, and
/// Λ at @p lambda.walk().
EnergyBars energy_bars(ScArrays &bar, LambdaBars &lambda);

/// @brief The reverse pass after a class walk (SC's, PC's or a QD
/// matrix's) has added its bars onto @p bar and @p lambda: the correlated
/// frame's Λ onto the whole one's corner, then the shared builders backwards onto the MO blocks, B_cc, B_vv's diagonal and bar.dm1;
/// then @p rdm_adjoint, which takes the RDMs' bars (bar.dm1's among them)
/// onto the CI vectors, returned as y, and onto bar.eri; then (tu|vw)'s
/// builder backwards, and h_bar, Λ, X, A and ε. The energies are left to the
/// caller.
Result<ScAdjoint>
builders_adjoint(DeviceResources &res, int32_t naux, int32_t nao, int32_t nmo, const double *b,
                 const double *c, int32_t ncore, int32_t ncas, ScArrays &x, ScArrays &bar,
                 LambdaBars lambda,
                 const std::function<Result<Buffers::device<double>>()> &rdm_adjoint);

/// @brief What qd_hessian_product's QD RDM step (qd_rdms_tangent,
/// hessian_product.cpp's) holds over @p nroots roots of @p n_alpha and
/// @p n_beta electrons in @p ncas orbitals: y, kept (persistent), beside its
/// three scratch outputs and each rdms_adjoint at one determinant; with
/// @p one_slot, the zero dm3 bar and the fixed bars' calls too.
MemoryFootprint qd_rdms_tangent_footprint(int32_t ncas, int32_t n_alpha, int32_t n_beta,
                                          int32_t nroots, bool one_slot);

/// @brief What qd_pc_hessian_product's Schur lifts along one direction hold
/// (hessian_product.cpp's lifts_along): per ket, K u, its RDMs' second
/// derivative (two bra/ket nevpt2::rdms and one of u, no f3, and the
/// traces) and pc_metric_lifts_footprint over the @p nroots states, ket r's
/// modes dropped as @p dropped[r] says (empty: every class at the rank that
/// holds the most). Nothing persists.
MemoryFootprint qd_pc_lifts_footprint(int32_t ncore, int32_t ncas, int32_t nvirt, int32_t n_alpha,
                                      int32_t n_beta, int32_t nroots,
                                      std::span<const size_t> syevd_lwork,
                                      std::span<const std::array<int64_t, 8>> dropped);

} // namespace free_breakfast::nevpt2
