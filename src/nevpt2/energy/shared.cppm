/// @file
/// @brief What the SC and PC class energies and their reverse passes share: the
/// slab walk's helpers, the class metrics and Dyall Hamiltonians (make_*),
/// Sijrs, the active operands every class reads, einsum's adjoint, and PC's
/// solve.
///
/// An internal partition (`module`, not `export module`): energy.cpp, pc.cpp
/// and reverse.cpp import it, nothing reaches an importer of
/// free_breakfast.nevpt2.energy. It only declares; every definition is
/// energy.cpp's, except upload and the PC solve's (pc.cpp's) and
/// einsum_adjoint (reverse.cpp's), so no bridge header reaches a partition's
/// global module fragment. PcKets' accessors are its one inline code. See
/// docs/style.md, "Module-private code".

module free_breakfast.nevpt2.energy:shared;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.energy; // EnergyInputs, EnergyBars, the result and amplitude types

namespace free_breakfast::nevpt2 {

/// @brief Everything a class reads of the active space, as einsum operands:
/// the integrals column-major, the RDMs row-major, delta the n x n identity
/// (owned here), and the orbital energies the host finish reads.
struct ActiveOperands {
  EinsumOperand h1e, h2e, dm1, dm2, dm3, f3ca, f3ac, delta;
  Buffers::device<double> delta_buf;
  std::vector<double> e_core, e_virt;
  /// @brief EnergyInputs::pc_floor, which a ket's solve reads.
  double pc_floor = 0.0;
  /// @brief The active electrons, tr dm1 rounded: a state's own count, 0 for
  /// a transition's and for an omitted entry. single_function_class reads it.
  int64_t nelecas = 0;
};

/// @brief Whether entry @p p of a walk's @p ops is given: an entry
/// sc_effective_hamiltonian's caller omits has a null dm1, and a walk builds,
/// contracts and reads nothing of it.
bool pair_given(std::span<const ActiveOperands> ops, int64_t p);

/// @brief The operands of @p in over @p blocks' sizes. Synchronizes once,
/// reading the orbital energies back.
Result<ActiveOperands> active_operands(DeviceResources &res, const MoBlocks &blocks,
                                       const EnergyInputs &in);

// Norms at or below this are an empty perturber space and contribute no
// energy: pyscf.mrpt.nevpt2's NUMERICAL_ZERO. The reverse pass gives such a
// perturber no multipliers, for the same reason. A one-function class
// (single_function_class) has no guard.
constexpr double kNumericalZero = 1e-14;

// Whether SC class @p k (PySCF's order) of the state whose own operands are
// @p own spans one function per perturber: Srs at two active electrons (both
// removed leave the active vacuum) and Sij at two active holes (both filled
// leave the closed shell). Each perturber's norm is then one amplitude
// squared, N_t = A_t², and its H is η N_t, η the one function's own Dyall
// energy, the same for every perturber of the state's class. So c_t =
// -1 / (Δ_t + η) does not depend on A_t: such a class keeps every perturber,
// at η from its blocks (single_function_ratio), and has no guard (README
// "One-function classes", #949).
bool single_function_class(const ActiveOperands &own, size_t k);

// One-function class @p k's blocks over the state operands @p o, on the
// host, in the pairing its norm and H contract them with the external
// amplitude W_pq W_ab: [0] X, the Dyall block (a7, a9), and [1] Y, the
// metric (rm2 read `pqba`, hdm2), each n⁴ row-major over (p, q, a, b). On a
// state's own RDMs X = η Y to round-off.
Result<std::array<std::vector<double>, 2>>
single_function_blocks(DeviceResources &res, const ActiveOperands &o, size_t k);

// η = <X, Y> / <Y, Y> over single_function_blocks' @p xy: the one function's
// Dyall energy, read without any amplitude, so defined where every A_t
// vanishes. None when Y is zero.
std::optional<double> single_function_ratio(const std::array<std::vector<double>, 2> &xy);

EinsumOperand column_major(const double *data, std::vector<int64_t> dims);
Result<Buffers::device<double>> device_zeros(DeviceResources &res, int64_t count);
Result<std::vector<double>> download(DeviceResources &res, const double *d, int64_t count);
Result<Buffers::device<double>> upload(DeviceResources &res, const std::vector<double> &host);
int32_t batch_step(int32_t extent, int32_t batch);
std::vector<int64_t> slab_dims(const MoBlocks &blocks, ExtBlock block, int32_t b0, int32_t b1);
Buffers::device<double> slab_buffer(DeviceResources &res, const MoBlocks &blocks, ExtBlock block,
                                    int32_t step);
int32_t walk_extent(const MoBlocks &blocks, ExtBlock block);

Result<Buffers::device<double>> make_a16(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm3,
                                         const EinsumOperand &f3ac, const EinsumOperand &f3ca);
Result<Buffers::device<double>> make_a22(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &f3ac,
                                         const EinsumOperand &f3ca);
Result<Buffers::device<double>> make_dm3_h(DeviceResources &res, const EinsumOperand &dm2,
                                           const EinsumOperand &dm3, const EinsumOperand &delta);
Result<Buffers::device<double>> make_dm2_h(DeviceResources &res, const EinsumOperand &dm1,
                                           const EinsumOperand &dm2, const EinsumOperand &delta);
// hdm1 and hdm2 carry terms with no RDM (hdm1's 2δ, hdm2's δδ), which are
// ⟨I|J⟩ times a number: @p overlap is that ⟨I|J⟩, 1 for a state's own RDMs and
// 0 for a transition's, whose such terms are then not added.
Result<Buffers::device<double>> make_hdm1(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &delta, double overlap = 1.0);
Result<Buffers::device<double>> make_hdm2(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &delta,
                                          double overlap = 1.0);
Result<Buffers::device<double>> make_hdm3(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &dm3,
                                          const EinsumOperand &hdm2, const EinsumOperand &delta);
Result<Buffers::device<double>> make_a3(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &dm1,
                                        const EinsumOperand &dm2, const EinsumOperand &hdm1,
                                        const EinsumOperand &delta);
Result<Buffers::device<double>> make_k27(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);
// Srs's metric, alone: what the QD walks read of a transition.
Result<Buffers::device<double>> make_rm2(DeviceResources &res, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &delta);
// Srs's rm3, from its rm2; make_a7 builds it internally. Declared for the
// reverse sweep, which re-reads it.
Result<Buffers::device<double>> make_rm3(DeviceResources &res, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &rm2,
                                         const EinsumOperand &delta);
Result<std::pair<Buffers::device<double>, Buffers::device<double>>>
make_a7(DeviceResources &res, const EinsumOperand &h1e, const EinsumOperand &h2e,
        const EinsumOperand &dm1, const EinsumOperand &dm2, const EinsumOperand &dm3,
        const EinsumOperand &delta);
Result<Buffers::device<double>> make_a9(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &hdm2,
                                        const EinsumOperand &hdm3);
Result<Buffers::device<double>> make_a12(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a13(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3,
                                         const EinsumOperand &delta);
// Sr's and Si's one-body Dyall blocks: SC's alone (PC reads S₁ h1 instead),
// declared for the reverse pass.
Result<Buffers::device<double>> make_a17(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a19(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);
Result<Buffers::device<double>> make_a23(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a25(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);

/// @brief The adjoint of `einsum(subscripts, operands) -> out` (out_dims,
/// alpha): for every operand k whose @p bars[k] is not null,
/// bars[k] += alpha d<out_bar, einsum(...)>/d(operand k), queued on
/// res.stream().
///
/// @p subscripts, @p operands, @p out_dims and @p alpha are the forward
/// call's, unchanged: the bar on operand k is the same einsum with the output
/// and operand k swapping roles, so no derivative is written by hand. Each
/// bar has its operand's layout (its dims and strides, row-major when the
/// strides are empty), and a bar may be shared by several operands (a slab
/// read twice), each adding its own term. @p out_bar is packed row-major over
/// @p out_dims, as einsum's output is.
///
/// - A row-major or column-major operand's bar is written in place, the
///   output labels ordered so that einsum's packed row-major output is the
///   operand's layout.
/// - An operand that repeats a label (`jbij`, `mjjn`, `raai`) takes its bar
///   over its distinct labels into workspace, then scatters it onto the
///   diagonal as an einsum against @p delta, the identity over the repeated
///   label's extent (one delta per repetition, at most two). @p delta is read
///   only then.
/// - A two-axis view of a larger column-major array (strides {1, ld}: a run
///   of h1e_v's rows) takes its bar packed into workspace, then one in-place
///   geam adds it at the view.
///
/// Scratch is res.device_workspace()'s, so no argument may point into it,
/// and a caller holds no workspace region across the call.
///
/// @return Success; InvalidConfig for subscripts that do not parse into one
///         group per operand and bar, a label only operand k carries and only
///         once (its bar would be a broadcast), a layout outside the three
///         above, or einsum's own refusals; or the launch status.
Status einsum_adjoint(DeviceResources &res, std::string_view subscripts,
                      const std::vector<EinsumOperand> &operands, const std::vector<double *> &bars,
                      const double *out_bar, const std::vector<int64_t> &out_dims,
                      const EinsumOperand &delta, double alpha);

/// @brief The SC class walk of n states over @p pairs, n x n EnergyInputs
/// row-major (bra I row, ket J column, I = J a state's own): per ordered pair
/// each class's Σ_t T^t_IJ and -Σ_t T^t_IJ / (Δ_t + H^J_t / N^J_t), as
/// sc_effective_hamiltonian documents, at [I n + J], and @p amplitudes when
/// not null. class_energies is n = 1. Nothing is validated: the caller has.
Result<std::vector<ClassEnergies>> sc_couplings(DeviceResources &res, const MoBlocks &blocks,
                                                std::span<const EnergyInputs> pairs, int64_t n,
                                                int32_t batch, ScAmplitudes *amplitudes = nullptr);

/// @brief The SC reverse pass of n states over @p pairs, as
/// sc_effective_hamiltonian_adjoint documents (reverse.cpp's).
/// class_energies_adjoint is n = 1. Nothing is validated: the caller has.
Result<ScEffectiveHamiltonianAdjoint>
sc_couplings_adjoint(DeviceResources &res, const MoBlocks &blocks,
                     std::span<const EnergyInputs> pairs, int64_t n, std::span<const double> hbar,
                     int32_t batch, std::span<const EnergyBars> bars,
                     const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition);

/// @brief The SC forward tangent of n states over @p pairs along the m n²
/// @p directions, as sc_effective_hamiltonian_tangent documents
/// (reverse.cpp's). Nothing is validated: the caller has.
Result<ScEffectiveHamiltonianTangent>
sc_couplings_tangent(DeviceResources &res, const MoBlocks &blocks,
                     std::span<const EnergyInputs> pairs, int64_t n, int32_t batch,
                     std::span<const EnergyDirection> directions);

/// @brief The SC-QD Hessian-vector product of n states, as
/// sc_effective_hamiltonian_hvp documents (reverse.cpp's). Nothing is
/// validated: the caller has.
Result<ScEffectiveHamiltonianHvp> sc_couplings_hvp(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs, int64_t n,
    std::span<const double> hbar, int32_t batch, std::span<const EnergyDirection> directions,
    std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, int64_t, const EnergyBars &)> &transition,
    std::span<const EnergyDirection> partners);

/// @brief The PC class walk of n states over @p pairs, as sc_couplings is
/// SC's: per ordered pair each class's ⟨I|V|Ψ_J⁽¹⁾⟩ and ket J's dropped
/// count, as pc_effective_hamiltonian documents, at [I n + J], and
/// @p amplitudes when not null. pc_class_energies is n = 1. Nothing is
/// validated: the caller has.
Result<std::vector<PcClassEnergies>> pc_couplings(DeviceResources &res, const MoBlocks &blocks,
                                                  std::span<const EnergyInputs> pairs, int64_t n,
                                                  int32_t batch,
                                                  PcAmplitudes *amplitudes = nullptr);

/// @brief Sijrs: {norm, energy}. It reads no RDM, so it is PC's too.
Result<std::pair<double, double>> energy_sijrs(DeviceResources &res, const MoBlocks &blocks,
                                               const EnergyInputs &in, int32_t batch);

/// @brief A solve's dropped modes, for the reverse pass's rotation of the kept
/// span into them (README "The PC reverse pass", #862): S = V diag(σ) Vᵀ, the
/// first nd columns of V dropped, the other m kept.
struct DroppedModes {
  /// @brief V, d x d column-major, σ ascending.
  Buffers::device<double> v;
  /// @brief σ, ascending, d.
  std::vector<double> sigma;
  /// @brief A pair σ_k - σ_n at or below this, ε √d of S's largest
  /// eigenvalue (σ's own round-off), is undetermined, and rotates nothing.
  double floor = 0.0;
  /// @brief V_nᵀ K U, nd x m column-major (K symmetrized).
  Buffers::device<double> kvu;
  /// @brief The own pair's leak V_nᵀ S and V_nᵀ S₁ (S symmetrized), nd x d
  /// and nd x n1 column-major, as PcKets::leak's.
  Buffers::device<double> leak, leak1;
};

/// @brief One PC class's solve, pc.cpp's solve_class: K U = S U diag(λ),
/// Uᵀ S U = 1 over S's kept range.
struct ClassSolve {
  /// @brief λ, ascending: the eigenvalues of K over S's kept range.
  std::vector<double> lambda;
  /// @brief T = S U, d x m column-major: element (I, k) at I + d k.
  Buffers::device<double> t;
  /// @brief t1 = S₁ᵀ U, n1 x m row-major, when an S₁ was given.
  std::optional<Buffers::device<double>> t1;
  /// @brief S's modes cut as null.
  int64_t dropped = 0;
  /// @brief U, d x m column-major, as T: what the reverse pass maps the
  /// eigenbasis bars back through.
  Buffers::device<double> u;
  /// @brief The dropped modes, when asked for and at least one was dropped.
  std::optional<DroppedModes> modes;
  /// @brief The largest dropped σ and the smallest kept one, over S's
  /// largest (PcClassEnergies::cut, kept).
  double cut = 0.0;
  double kept = 0.0;
};

/// @brief (A + Aᵀ) / 2 of the row-major d x d @p a, into a buffer of its own.
Result<Buffers::device<double>> symmetrized(DeviceResources &res, const double *a, int64_t d);

/// @brief The solve of the class @p name over its row-major d x d metric
/// @p s_rm and Dyall Hamiltonian @p k_rm, and its optional S₁ (@p s1, d x n1
/// row-major): both symmetrized, S cut at max(1e-13, 100 ε √d, @p floor) of
/// its largest eigenvalue, the pencil diagonalized. A Numerical error naming
/// the class when S has no positive eigenvalue. @p keep_modes keeps the
/// dropped modes (ClassSolve::modes) for the reverse pass.
Result<ClassSolve> solve_class(DeviceResources &res, const char *name, const double *s_rm,
                               const double *k_rm, int64_t d, const double *s1, int64_t n1,
                               bool keep_modes, double floor);
/// @brief V_nᵀ A, nd x c column-major, for the first nd columns of @p v (d x d
/// column-major) and the row-major d x c @p a_rm.
Result<Buffers::device<double>> leak_of(DeviceResources &res, const double *v, const double *a_rm,
                                        int64_t d, int64_t nd, int64_t c);
/// @brief A solve's DroppedModes from S's eigenvectors @p v and eigenvalues
/// @p sigma (the first d - m dropped), the pair floor @p floor, the
/// symmetrized @p k_sym and @p s_sym, @p u (d x m) and the optional S₁
/// (@p s1, d x n1 row-major): V_nᵀ K U, V_nᵀ S and V_nᵀ S₁ formed here.
Result<DroppedModes> dropped_modes(DeviceResources &res, Buffers::device<double> v,
                                   const std::vector<double> &sigma, double floor,
                                   const double *k_sym, const double *s_sym, const double *u,
                                   int64_t d, int64_t m, const double *s1, int64_t n1);

/// @brief One pair's class matrices, row-major: S, K for a state's own RDMs
/// alone, and S₁ where the class has one.
struct PcMatrices {
  Buffers::device<double> s;
  std::optional<Buffers::device<double>> k;
  std::optional<Buffers::device<double>> s1;
};

/// @brief A class's matrices over one pair's operands; own is a state's own
/// RDMs (K built, the terms with no RDM kept), else a transition's. One per
/// class, pc.cpp's pc_*_matrices, which the walk and its reverse pass share.
using PcBuild = Result<PcMatrices> (*)(DeviceResources &, const ActiveOperands &, bool own);

Result<PcMatrices> pc_sr_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_si_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_sijr_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_srsi_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_srs_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_sij_matrices(DeviceResources &res, const ActiveOperands &o, bool own);
Result<PcMatrices> pc_sir_matrices(DeviceResources &res, const ActiveOperands &o, bool own);

/// @brief One class's solves and transition rows over ns states: per ket J
/// its solve (none when the diagonal is omitted and the kets are a previous
/// call's), its m_J, and per given ordered pair (I, J), I != J,
/// T_{I←J} = S^{(J,I)ᵀ} U_J and t1_{I←J} = S₁^{(J,I)ᵀ} U_J, d x m_J
/// column-major and n1 x m_J row-major as the solve's own T and t1.
struct PcKets {
  int64_t ns = 1;
  std::vector<ClassSolve> solves;
  std::vector<int64_t> ms;
  /// @brief Each ket's ClassSolve::cut and kept; 0 for cached kets.
  std::vector<double> cut, kept;
  /// @brief [I ns + J]: whether the pair is given (pair_given).
  std::vector<bool> given;
  std::vector<std::optional<Buffers::device<double>>> t, t1;
  /// @brief [I ns + J], I != J, when the kets keep their dropped modes: the
  /// transition rows' leak into ket J's dropped modes, V_nᵀ S^{(J,I)} and
  /// V_nᵀ S₁^{(J,I)} (nd x d and nd x n1 column-major; S₁'s empty without
  /// one). README "The PC reverse pass".
  std::vector<std::optional<Buffers::device<double>>> leak, leak1;

  [[nodiscard]] int64_t m(const int64_t j) const { return ms[static_cast<size_t>(j)]; }
  [[nodiscard]] const double *t_of(const int64_t i, const int64_t j) const {
    return i == j ? solves[static_cast<size_t>(j)].t.data()
                  : t[static_cast<size_t>((i * ns) + j)]->data();
  }
  [[nodiscard]] const double *t1_of(const int64_t i, const int64_t j) const {
    return i == j ? solves[static_cast<size_t>(j)].t1->data()
                  : t1[static_cast<size_t>((i * ns) + j)]->data();
  }
  /// @brief Where pair (I, J)'s block of @p rows tuples starts in one slab's
  /// packed rows, at [I ns + J], and the total at [ns²]; an omitted pair's
  /// block is empty.
  [[nodiscard]] std::vector<int64_t> offsets(const int64_t rows) const {
    std::vector<int64_t> at{0};
    for (int64_t p = 0; p < ns * ns; ++p) {
      at.push_back(at.back() + (given[static_cast<size_t>(p)] ? rows * m(p % ns) : 0));
    }
    return at;
  }
};

/// @brief Every state's solve over @p ops (ns x ns, [I ns + J]), from its own
/// matrices, or with the diagonal omitted each ket's U_J from @p cached (a
/// previous call's amplitudes; InvalidConfig when they are not ns states');
/// then every given transition's T and t1 over pair (J, I)'s, each pair's
/// matrices freed before the next is built (with @p keep_modes, its leak too,
/// and every solve's modes). A refusal is state_error's.
Result<PcKets> pc_kets(DeviceResources &res, const char *name, std::span<const ActiveOperands> ops,
                       int64_t ns, int64_t d, int64_t n1, const PcBuild &build,
                       const PcAmplitudes::Class *cached = nullptr, bool keep_modes = false);

/// @brief A failure of state @p j's class: at @p ns > 1 named by its state,
/// pc_class_energies' prefix replaced by pc_effective_hamiltonian's; at ns = 1
/// unchanged. The forward walks' and the reverse passes' one rule, so both
/// refuse with the same message.
std::unexpected<Error> state_error(const Error &e, int64_t j, int64_t ns);

/// @brief The PC reverse pass of n states over @p pairs, as
/// pc_effective_hamiltonian_adjoint documents (reverse.cpp's).
/// pc_class_energies_adjoint is n = 1. Nothing is validated: the caller has.
Result<PcEffectiveHamiltonianAdjoint>
pc_couplings_adjoint(DeviceResources &res, const MoBlocks &blocks,
                     std::span<const EnergyInputs> pairs, int64_t n, std::span<const double> hbar,
                     int32_t batch, std::span<const EnergyBars> bars,
                     const std::function<Status(int64_t, int64_t, const EnergyBars &)> &transition);

/// @brief The PC forward tangent of n states over @p pairs along the m n²
/// @p directions, as pc_effective_hamiltonian_tangent documents
/// (reverse.cpp's); pc_class_energies_tangent is n = 1. Only the directions'
/// blocks are checked (InvalidConfig naming @p who): the caller has checked
/// the rest.
Result<PcEffectiveHamiltonianTangent>
pc_couplings_tangent(DeviceResources &res, const MoBlocks &blocks,
                     std::span<const EnergyInputs> pairs, int64_t n, int32_t batch,
                     std::span<const EnergyDirection> directions, bool span, std::string_view who);

/// @brief The PC Hessian-vector product of n states over @p pairs along the
/// m n² @p directions onto the m n @p bars, as pc_effective_hamiltonian_hvp
/// documents (reverse.cpp's); pc_class_energies_hvp is n = 1, H̄ = 1, no
/// transition. Only the directions' blocks are checked (InvalidConfig naming
/// @p who): the caller has checked the rest.
Result<std::vector<PcClassEnergies>> pc_couplings_hvp(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs, int64_t n,
    std::span<const double> hbar, int32_t batch, std::span<const EnergyDirection> directions,
    std::span<const EnergyBars> bars,
    const std::function<Status(int64_t, int64_t, int64_t, const EnergyBars &)> &transition,
    bool span, std::string_view who);

/// @brief One ket's perturber functions of a class, as the walk meets them:
/// the mode nearest a pole, the mode of largest amplitude, and the largest
/// |λ + Δ|, what pc_intruder_check reads.
struct PcDenominators {
  PcMode nearest{.denominator = std::numeric_limits<double>::infinity(), .amplitude = 0.0};
  PcMode strongest;
  double largest = 0.0;

  /// @brief Mode @p denominator λ_k + Δ_t, its ket's own row @p y.
  void add(double denominator, double y);
};

/// @brief A class's refusal for one ket (README "The cut, and refusals"): a
/// denominator at round-off, |λ + Δ| at most 100 ε √d of the class's largest
/// (@p d its basis size), or an intruder, a mode whose amplitude
/// |y / (λ + Δ)| exceeds 1. A Numerical error naming the class, the mode's
/// denominator and amplitude; success otherwise. The energy's and the
/// reverse pass's one check, so both refuse with the same error. No sign is
/// checked: a perturber below its reference is a negative denominator.
Status pc_intruder_check(const char *name, const PcDenominators &modes, int64_t d);

/// @brief Ket @p j's modes of class @p k into @p out's strongest and nearest,
/// when @p modes holds the ket's (a solved one); left zero otherwise.
void pc_report_modes(std::span<const PcDenominators> modes, size_t j, size_t k,
                     PcClassEnergies &out);

} // namespace free_breakfast::nevpt2
